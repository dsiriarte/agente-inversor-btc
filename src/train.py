"""Entrenamiento del agente DQN.

Entrena solo con el periodo de ajuste (80 % inicial del entrenamiento). El
periodo de prueba no se usa en este script.

Cada experimento de configs/experimentos.json combina una recompensa y la
decisión de escalar o no el estado. Si se escala, la media y la desviación se
calculan con el periodo de ajuste y se guardan junto al modelo.

Criterio de parada: número fijo de pasos (configs/base.json -> agente.pasos).
Cada 20.000 pasos se guarda un punto de control para revisar la estabilidad en
validación; el modelo que se usa siempre es el del último paso.

Uso:
    python -m src.train --experimento R1 --semilla 42
    python -m src.train --todos                     # todos los experimentos y semillas
    python -m src.train --todos --experimentos R1 R2
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.agent import PoliticaDQN, cargar_agente, crear_agente
from src.config import MODELOS, RESULTADOS, cargar_config, guardar_json
from src.data import cargar_conjuntos
from src.environment import crear_entorno, escalado_desde
from src.metrics import recorrer, resumir

CADA_PUNTO_CONTROL = 20_000


def carpeta_modelo(experimento: str, semilla: int):
    return MODELOS / experimento / f"semilla_{semilla}"


def entorno_para(experimento: str, semilla: int, datos, cfg):
    """Entorno con la recompensa y el escalado que usó ese modelo al entrenar."""
    exp = cargar_config("experimentos.json")[experimento]
    ruta = carpeta_modelo(experimento, semilla) / "escalado.json"
    escalado = json.loads(ruta.read_text()) if ruta.exists() else None
    return crear_entorno(datos, cfg, exp["recompensa"], escalado)


class Bitacora(BaseCallback):
    """Guarda epsilon y puntos de control durante el entrenamiento."""

    def __init__(self, carpeta):
        super().__init__()
        self.carpeta, self.filas = carpeta, []

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            self.filas.append({"paso": self.num_timesteps, "epsilon": self.model.exploration_rate})
        if (self.num_timesteps % CADA_PUNTO_CONTROL == 0
                and self.num_timesteps < self.model._total_timesteps):
            self.model.save(self.carpeta / "puntos_control" / f"paso_{self.num_timesteps}")
        return True


def entrenar(experimento: str, semilla: int) -> dict:
    cfg = cargar_config()
    exp = cargar_config("experimentos.json")[experimento]
    conjuntos = cargar_conjuntos(cfg)
    carpeta = carpeta_modelo(experimento, semilla)
    (carpeta / "puntos_control").mkdir(parents=True, exist_ok=True)

    escalado = escalado_desde(conjuntos["ajuste"]) if exp["escalar_estado"] else None
    if escalado:
        guardar_json(escalado, carpeta / "escalado.json")
    env = Monitor(crear_entorno(conjuntos["ajuste"], cfg, exp["recompensa"], escalado),
                  str(carpeta / "monitor"))
    modelo = crear_agente(env, cfg["agente"], semilla)
    bitacora = Bitacora(carpeta)
    inicio = time.time()
    modelo.learn(total_timesteps=cfg["agente"]["pasos"], callback=bitacora, progress_bar=False)
    duracion = time.time() - inicio
    modelo.save(carpeta / "modelo_final")
    pd.DataFrame(bitacora.filas).to_csv(carpeta / "epsilon.csv", index=False)

    resumen = validar(experimento, semilla, cfg, conjuntos)
    guardar_json({"experimento": experimento, "semilla": semilla, "pasos": cfg["agente"]["pasos"],
                  "segundos": round(duracion, 1), "validacion": resumen}, carpeta / "resumen.json")
    print(f"{experimento} semilla {semilla}: {duracion:.0f} s | validación "
          f"{resumen['retorno_pct']:.2f} % | DD {resumen['max_drawdown_pct']:.2f} % | "
          f"exposición media {resumen['exposicion_media_pct']:.0f} %", flush=True)
    return resumen


def validar(experimento, semilla, cfg, conjuntos) -> dict:
    """Evalúa el modelo final (y los puntos de control) en validación.

    El desempeño siempre se mide con el capital real del portafolio, no con la
    recompensa, para poder comparar recompensas distintas.
    """
    carpeta = carpeta_modelo(experimento, semilla)
    salida = RESULTADOS / "validacion" / experimento / f"semilla_{semilla}"
    salida.mkdir(parents=True, exist_ok=True)
    capital = cfg["entorno"]["capital_inicial"]
    politica = PoliticaDQN(cargar_agente(carpeta / "modelo_final.zip"))

    hist = recorrer(politica, entorno_para(experimento, semilla, conjuntos["validacion"], cfg))
    hist.to_csv(salida / "historial.csv", index=False)
    resumen = resumir(hist, f"DQN {experimento} (semilla {semilla})", capital)

    # El mismo modelo dentro de muestra (periodo de ajuste), para comparar.
    h_ajuste = recorrer(politica, entorno_para(experimento, semilla, conjuntos["ajuste"], cfg))
    h_ajuste.to_csv(salida / "historial_ajuste.csv", index=False)
    r_ajuste = resumir(h_ajuste, "ajuste", capital)
    for k in ["retorno_pct", "max_drawdown_pct", "exposicion_media_pct", "cambios_exposicion"]:
        resumen[f"ajuste_{k}"] = r_ajuste[k]

    filas = []
    for ruta in sorted((carpeta / "puntos_control").glob("paso_*.zip"),
                       key=lambda p: int(p.stem.split("_")[1])):
        h = recorrer(PoliticaDQN(cargar_agente(ruta)),
                     entorno_para(experimento, semilla, conjuntos["validacion"], cfg))
        fila = resumir(h, ruta.stem, capital)
        fila["paso"] = int(ruta.stem.split("_")[1])
        filas.append(fila)
    pd.DataFrame(filas).to_csv(salida / "puntos_control.csv", index=False)
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experimento", default="R1")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--todos", action="store_true", help="todos los experimentos y semillas")
    parser.add_argument("--experimentos", nargs="*", help="limitar --todos a estos experimentos")
    parser.add_argument("--procesos", type=int, default=5)
    args = parser.parse_args()

    if not args.todos:
        entrenar(args.experimento, args.semilla)
        return
    cfg = cargar_config()
    nombres = args.experimentos or list(cargar_config("experimentos.json"))
    trabajos = [(e, s) for e in nombres for s in cfg["semillas"]]
    with ProcessPoolExecutor(max_workers=args.procesos) as ex:
        list(ex.map(entrenar, *zip(*trabajos)))


if __name__ == "__main__":
    main()
