"""Métricas de evaluación y recorrido de una política por un periodo."""
import numpy as np
import pandas as pd


def recorrer(politica, env) -> pd.DataFrame:
    """Recorre el periodo completo sin aprender y devuelve el historial hora a hora."""
    politica.reiniciar()
    obs, info = env.reset(seed=0)
    filas = [{"fecha": info["fecha"], "precio": info["precio"], "accion": np.nan,
              "objetivo": np.nan, "posicion": info["posicion"], "capital": info["capital"],
              "costos": 0.0, "recompensa": np.nan}]
    terminado = False
    while not terminado:
        accion = politica.act(obs)
        obs, r, terminado, truncado, info = env.step(accion)
        terminado = terminado or truncado
        filas.append({"fecha": info["fecha"], "precio": info["precio"], "accion": accion,
                      "objetivo": info["objetivo"], "posicion": info["posicion"],
                      "capital": info["capital"], "costos": info["costos_acumulados"],
                      "recompensa": r})
    hist = pd.DataFrame(filas)
    hist.attrs["cambios"] = info["cambios_exposicion"]
    hist.attrs["rebalanceos"] = info["rebalanceos"]
    return hist


def max_drawdown(capital) -> float:
    capital = np.asarray(capital, dtype=float)
    return float((capital / np.maximum.accumulate(capital) - 1).min())


def resumir(hist: pd.DataFrame, nombre: str, capital_inicial: float) -> dict:
    """Retorno acumulado, máximo drawdown, costos y uso de cada nivel de exposición."""
    cap = hist["capital"].to_numpy()
    obj = hist["objetivo"].dropna()
    return {
        "estrategia": nombre,
        "capital_final": cap[-1],
        "retorno_pct": (cap[-1] / capital_inicial - 1) * 100,
        "max_drawdown_pct": max_drawdown(cap) * 100,
        "costos_usd": hist["costos"].iloc[-1],
        "costos_pct_capital_inicial": hist["costos"].iloc[-1] / capital_inicial * 100,
        "cambios_exposicion": hist.attrs.get("cambios", np.nan),
        "rebalanceos": hist.attrs.get("rebalanceos", np.nan),
        "exposicion_media_pct": obj.mean() * 100,
        "horas_0_pct": (obj == 0.0).mean() * 100,
        "horas_50_pct": (obj == 0.5).mean() * 100,
        "horas_100_pct": (obj == 1.0).mean() * 100,
        "recompensa_total": hist["recompensa"].sum(),
    }
