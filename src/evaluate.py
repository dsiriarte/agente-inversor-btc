"""Selección en validación y evaluación final en datos no vistos.

Protocolo:
1. `--validacion`: resume todos los experimentos en validación, aplica la regla
   de configs/base.json -> seleccion y guarda la decisión en configs/seleccion.json.
   Si ese archivo ya existe no se sobrescribe (la decisión queda congelada).
2. `--prueba`: evalúa una sola vez, en el periodo de prueba, las 5 semillas del
   experimento elegido y los baselines con las mismas condiciones.
   No se ejecuta si no existe configs/seleccion.json.
3. `--posterior` (opcional): evalúa en prueba los experimentos NO elegidos, solo
   para el análisis. No cambia la selección.

Uso:
    python -m src.evaluate --validacion
    python -m src.evaluate --prueba
    python -m src.evaluate --posterior
"""
import argparse
import hashlib
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.agent import PoliticaDQN, cargar_agente
from src.baselines import Aleatorio, ComprarYMantener, Constante50
from src.config import CONFIGS, RESULTADOS, cargar_config, guardar_json
from src.data import cargar_conjuntos
from src.environment import crear_entorno
from src.metrics import recorrer, resumir
from src.train import carpeta_modelo, entorno_para

RUTA_SELECCION = CONFIGS / "seleccion.json"


def evaluar_baselines(datos, cfg, carpeta) -> pd.DataFrame:
    """Comprar y mantener, constante 50 % y 100 semillas del agente aleatorio."""
    carpeta.mkdir(parents=True, exist_ok=True)
    capital = cfg["entorno"]["capital_inicial"]
    filas = []
    for pol, archivo in [(ComprarYMantener(), "comprar_y_mantener"), (Constante50(), "constante_50")]:
        h = recorrer(pol, crear_entorno(datos, cfg))
        h.to_csv(carpeta / f"historial_{archivo}.csv", index=False)
        filas.append(resumir(h, pol.nombre, capital))
    a = cfg["aleatorio"]
    aleatorios = []
    for s in range(a["primera_semilla"], a["primera_semilla"] + a["semillas"]):
        h = recorrer(Aleatorio(s), crear_entorno(datos, cfg))
        fila = resumir(h, "Aleatorio", capital)
        fila["semilla"] = s
        aleatorios.append(fila)
    aleatorios = pd.DataFrame(aleatorios)
    aleatorios.to_csv(carpeta / "aleatorio_100_semillas.csv", index=False)
    # Curva del aleatorio "típico": la semilla con el retorno mediano.
    s_med = aleatorios.iloc[(aleatorios.retorno_pct - aleatorios.retorno_pct.median()).abs().argmin()].semilla
    recorrer(Aleatorio(int(s_med)), crear_entorno(datos, cfg)).to_csv(
        carpeta / "historial_aleatorio_mediano.csv", index=False)
    med = aleatorios.drop(columns=["estrategia", "semilla"]).median()
    filas.append({"estrategia": f"Aleatorio (mediana de {len(aleatorios)} semillas)", **med.to_dict()})
    for q, nombre in [(0.05, "percentil 5"), (0.95, "percentil 95")]:
        filas.append({"estrategia": f"Aleatorio ({nombre})",
                      **aleatorios.drop(columns=["estrategia", "semilla"]).quantile(q).to_dict()})
    return pd.DataFrame(filas)


def evaluar_experimento(experimento, datos, cfg, carpeta) -> pd.DataFrame:
    filas = []
    for s in cfg["semillas"]:
        pol = PoliticaDQN(cargar_agente(carpeta_modelo(experimento, s) / "modelo_final.zip"))
        h = recorrer(pol, entorno_para(experimento, s, datos, cfg))
        (carpeta / experimento).mkdir(parents=True, exist_ok=True)
        h.to_csv(carpeta / experimento / f"historial_semilla_{s}.csv", index=False)
        fila = resumir(h, f"DQN {experimento}", cfg["entorno"]["capital_inicial"])
        fila["semilla"] = s
        filas.append(fila)
    return pd.DataFrame(filas)


def huella_modelo(experimento, semilla) -> str:
    """Huella del archivo .zip completo (incluye fechas, cambia en cada reentrenamiento)."""
    ruta = carpeta_modelo(experimento, semilla) / "modelo_final.zip"
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def huella_pesos(experimento, semilla) -> str:
    """Huella solo de los pesos de la red (policy.pth): igual si el entrenamiento se repite exacto."""
    ruta = carpeta_modelo(experimento, semilla) / "modelo_final.zip"
    with zipfile.ZipFile(ruta) as z:
        return hashlib.sha256(z.read("policy.pth")).hexdigest()


def validacion(cfg, conjuntos) -> None:
    carpeta = RESULTADOS / "validacion"
    base = evaluar_baselines(conjuntos["validacion"], cfg, carpeta / "baselines")
    base.to_csv(carpeta / "baselines.csv", index=False)
    experimentos = cargar_config("experimentos.json")
    por_semilla = pd.concat([evaluar_experimento(e, conjuntos["validacion"], cfg, carpeta / "recorridos")
                             .assign(experimento=e) for e in experimentos])
    por_semilla.to_csv(carpeta / "experimentos_por_semilla.csv", index=False)
    cols = ["retorno_pct", "max_drawdown_pct", "costos_usd", "cambios_exposicion", "exposicion_media_pct"]
    resumen = por_semilla.groupby("experimento", sort=False)[cols].median()
    resumen["retorno_min"] = por_semilla.groupby("experimento", sort=False).retorno_pct.min()
    resumen["retorno_max"] = por_semilla.groupby("experimento", sort=False).retorno_pct.max()

    dd_ref = float(base.loc[base.estrategia == "Comprar y mantener", "max_drawdown_pct"].iloc[0])
    resumen["pasa_filtro_drawdown"] = resumen.max_drawdown_pct >= dd_ref
    resumen.to_csv(carpeta / "resumen_experimentos.csv")
    print(f"Drawdown de comprar y mantener en validación: {dd_ref:.2f} %")
    print(resumen.round(2).to_string())

    if RUTA_SELECCION.exists():
        print(f"\n{RUTA_SELECCION.name} ya existe: la selección está congelada y no se cambia.")
        return
    candidatos = resumen[resumen.pasa_filtro_drawdown]
    elegido = candidatos.retorno_pct.idxmax()
    filas = por_semilla[por_semilla.experimento == elegido]
    tipica = int(filas.iloc[(filas.retorno_pct - filas.retorno_pct.median()).abs().argmin()].semilla)
    guardar_json({
        "experimento": elegido,
        "descripcion": experimentos[elegido]["descripcion"],
        "semillas": cfg["semillas"],
        "semilla_tipica": tipica,
        "regla": cfg["seleccion"]["regla"],
        "drawdown_referencia_validacion_pct": dd_ref,
        "mediana_validacion": resumen.loc[elegido, cols].round(4).to_dict(),
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256_modelos": {str(s): huella_modelo(elegido, s) for s in cfg["semillas"]},
        "sha256_pesos": {str(s): huella_pesos(elegido, s) for s in cfg["semillas"]},
    }, RUTA_SELECCION)
    print(f"\nSelección congelada en {RUTA_SELECCION}: {elegido} (semilla típica {tipica})")


def prueba(cfg, conjuntos) -> None:
    if not RUTA_SELECCION.exists():
        raise SystemExit("Primero hay que fijar la selección con --validacion.")
    sel = cargar_config("seleccion.json")
    for s in sel["semillas"]:
        if huella_pesos(sel["experimento"], s) != sel["sha256_pesos"][str(s)]:
            # Pasa si se reentrena en otra máquina: el modelo ya no es el que se congeló.
            print(f"AVISO: el modelo de la semilla {s} no es el que se congeló en seleccion.json "
                  "(¿se reentrenó en otro computador?). Los resultados pueden diferir del informe.")
    carpeta = RESULTADOS / "prueba"
    base = evaluar_baselines(conjuntos["prueba"], cfg, carpeta / "baselines")
    agente = evaluar_experimento(sel["experimento"], conjuntos["prueba"], cfg, carpeta / "recorridos")
    agente.to_csv(carpeta / "agente_por_semilla.csv", index=False)
    tipica = agente[agente.semilla == sel["semilla_tipica"]].assign(
        estrategia=f"Agente DQN final ({sel['experimento']}, semilla {sel['semilla_tipica']})")
    mediana = {"estrategia": f"Agente DQN ({sel['experimento']}, mediana de {len(agente)} semillas)",
               **agente.drop(columns=["estrategia", "semilla"]).median().to_dict()}
    tabla = pd.concat([tipica.drop(columns="semilla"), pd.DataFrame([mediana]), base], ignore_index=True)
    tabla.to_csv(carpeta / "tabla_final.csv", index=False)
    cols = ["estrategia", "retorno_pct", "max_drawdown_pct", "costos_usd", "cambios_exposicion",
            "exposicion_media_pct"]
    print(tabla[cols].round(2).to_string(index=False))
    print("\nPor semilla:")
    print(agente[["semilla", *cols[1:]]].round(2).to_string(index=False))


def posterior(cfg, conjuntos) -> None:
    sel = cargar_config("seleccion.json")
    carpeta = RESULTADOS / "prueba_posterior"
    otros = [e for e in cargar_config("experimentos.json") if e != sel["experimento"]]
    tabla = pd.concat([evaluar_experimento(e, conjuntos["prueba"], cfg, carpeta / "recorridos")
                       .assign(experimento=e) for e in otros])
    tabla.to_csv(carpeta / "experimentos_por_semilla.csv", index=False)
    cols = ["retorno_pct", "max_drawdown_pct", "costos_usd", "cambios_exposicion", "exposicion_media_pct"]
    print("ANÁLISIS POSTERIOR (no cambia la selección)")
    print(tabla.groupby("experimento", sort=False)[cols].median().round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--validacion", action="store_true")
    grupo.add_argument("--prueba", action="store_true")
    grupo.add_argument("--posterior", action="store_true")
    args = parser.parse_args()
    cfg = cargar_config()
    conjuntos = cargar_conjuntos(cfg)
    if args.validacion:
        validacion(cfg, conjuntos)
    elif args.prueba:
        prueba(cfg, conjuntos)
    else:
        posterior(cfg, conjuntos)


if __name__ == "__main__":
    main()
