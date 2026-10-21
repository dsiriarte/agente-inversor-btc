"""Auditoría de información futura (lookahead) y de la contabilidad del entorno.

1. Indicadores: el valor de la hora t no cambia si se borran las horas t+1 en adelante.
2. Agente: si se reemplaza todo el futuro (precios e indicadores después de t)
   por datos al azar, las acciones del agente hasta t son las mismas.
3. Contabilidad: capital = efectivo + BTC x precio en cada paso, y la suma de
   recompensas R1 es igual a ln(capital final / capital inicial).

Uso:  python -m src.auditoria
"""
import numpy as np

from src.agent import PoliticaDQN, cargar_agente
from src.baselines import Aleatorio
from src.config import RESULTADOS, cargar_config, guardar_json
from src.data import COLUMNAS_ESTADO, auditar_lookahead, cargar_csv
from src.data import cargar_conjuntos
from src.environment import crear_entorno
from src.metrics import recorrer
from src.train import carpeta_modelo, entorno_para


def acciones_hasta(politica, env, t):
    politica.reiniciar()
    obs, _ = env.reset()
    acciones = []
    for _ in range(t + 1):
        a = politica.act(obs)
        acciones.append(a)
        obs, *_ = env.step(a)
    return acciones


def auditar_agente(experimento, semilla, datos, cfg, n_cortes=20, rng=None) -> int:
    rng = rng or np.random.default_rng(0)
    politica = PoliticaDQN(cargar_agente(carpeta_modelo(experimento, semilla) / "modelo_final.zip"))
    for t in rng.choice(np.arange(50, len(datos) - 2), size=n_cortes, replace=False):
        alterado = datos.copy()
        futuro = alterado.index > t
        alterado.loc[futuro, "close"] = rng.uniform(1_000, 100_000, futuro.sum())
        for c in COLUMNAS_ESTADO:
            alterado.loc[futuro, c] = rng.normal(0, 1, futuro.sum())
        a = acciones_hasta(politica, entorno_para(experimento, semilla, datos, cfg), t)
        b = acciones_hasta(politica, entorno_para(experimento, semilla, alterado, cfg), t)
        if a != b:
            raise AssertionError(f"La acción en t={t} depende del futuro")
    return n_cortes


def auditar_contabilidad(datos, cfg) -> None:
    env = crear_entorno(datos, cfg)
    pol = Aleatorio(7)
    obs, info = env.reset()
    total = 0.0
    terminado = False
    while not terminado:
        obs, r, terminado, _, info = env.step(pol.act(obs))
        total += r
        assert np.isclose(info["capital"], env.cash + env.btc * env.p[env.i])
    assert np.isclose(total, np.log(info["capital"] / cfg["entorno"]["capital_inicial"]))


def main() -> None:
    cfg = cargar_config()
    conjuntos = cargar_conjuntos(cfg)
    n_ind = auditar_lookahead(cargar_csv(cfg["datos"]), cfg["datos"]["ventana"])
    sel = cargar_config("seleccion.json")
    n_ag = sum(auditar_agente(sel["experimento"], s, conjuntos["validacion"], cfg) for s in sel["semillas"])
    auditar_contabilidad(conjuntos["validacion"], cfg)
    resultado = {"indicadores_revisados": n_ind, "cortes_agente_revisados": n_ag,
                 "contabilidad": "correcta", "conclusion": "ninguna decisión usa información futura"}
    guardar_json(resultado, RESULTADOS / "auditoria.json")
    print(resultado)


if __name__ == "__main__":
    main()
