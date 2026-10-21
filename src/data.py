"""Procesamiento de datos.

Pasos:
1. Ordena la serie horaria por fecha (UTC).
2. Calcula los indicadores con ventanas que solo miran hacia atrás.
3. Elimina las primeras horas sin historia suficiente.
4. Divide en orden cronológico: 70 % entrenamiento y 30 % prueba.
5. Dentro del entrenamiento separa el 20 % final como validación.

Uso:  python -m src.data      (prepara, audita y guarda los CSV en data/procesados/)
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAIZ, cargar_config

COLUMNAS_ESTADO = ["return_24h", "price_vs_sma_24h", "volatility_24h"]
COLUMNAS_FINALES = ["date", "open", "high", "low", "close", *COLUMNAS_ESTADO]


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def cargar_csv(cfg_datos: dict) -> pd.DataFrame:
    """Lee el CSV original y comprueba que sea exactamente el mismo archivo."""
    ruta = RAIZ / cfg_datos["archivo"]
    huella = sha256(ruta)
    if huella != cfg_datos["sha256"]:
        raise ValueError(f"El archivo {ruta} no coincide con el dataset del proyecto ({huella}).")
    return pd.read_csv(ruta)


def calcular_indicadores(df: pd.DataFrame, ventana: int = 24) -> pd.DataFrame:
    """Indicadores del estado. Todos usan solo precios hasta la hora t (incluida)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    df["return"] = df["close"].pct_change()
    df["return_24h"] = df["close"].pct_change(ventana)
    df["sma_24h"] = df["close"].rolling(ventana, min_periods=ventana).mean()
    df["price_vs_sma_24h"] = df["close"] / df["sma_24h"] - 1
    df["volatility_24h"] = df["return"].rolling(ventana, min_periods=ventana).std()
    return df


def preparar_datos(df: pd.DataFrame, ventana: int = 24) -> pd.DataFrame:
    """Base analítica final (33.235 filas con el dataset del proyecto)."""
    df = calcular_indicadores(df, ventana)
    final = df[COLUMNAS_FINALES].dropna(subset=COLUMNAS_ESTADO).reset_index(drop=True)
    return final


def dividir(final: pd.DataFrame, frac_entrenamiento: float, frac_validacion: float) -> dict:
    """División cronológica sin mezclar: ajuste | validación | prueba."""
    corte = int(len(final) * frac_entrenamiento)
    entrenamiento = final.iloc[:corte].reset_index(drop=True)
    prueba = final.iloc[corte:].reset_index(drop=True)
    corte_val = int(len(entrenamiento) * (1 - frac_validacion))
    ajuste = entrenamiento.iloc[:corte_val].reset_index(drop=True)
    validacion = entrenamiento.iloc[corte_val:].reset_index(drop=True)
    return {"entrenamiento": entrenamiento, "ajuste": ajuste,
            "validacion": validacion, "prueba": prueba}


def cargar_conjuntos(cfg: dict | None = None) -> dict:
    """Atajo usado por el resto del código."""
    cfg = cfg or cargar_config()
    d = cfg["datos"]
    final = preparar_datos(cargar_csv(d), d["ventana"])
    return dividir(final, d["fraccion_entrenamiento"], d["fraccion_validacion"])


def auditar_lookahead(df_original: pd.DataFrame, ventana: int = 24,
                      n_muestras: int = 300, semilla: int = 0) -> int:
    """Comprueba que el indicador de la hora t no cambia si se borra el futuro.

    Para horas t elegidas al azar se recalculan los indicadores usando solo
    las filas 0..t. Si algún valor difiere del calculado con la serie completa,
    el indicador estaría usando información futura. Devuelve las horas revisadas.
    """
    completo = calcular_indicadores(df_original, ventana)
    rng = np.random.default_rng(semilla)
    horas = rng.choice(np.arange(ventana + 1, len(completo)), size=n_muestras, replace=False)
    ordenado = df_original.assign(_f=pd.to_datetime(df_original["date"], utc=True))
    ordenado = ordenado.sort_values("_f").drop(columns="_f").reset_index(drop=True)
    for t in horas:
        parcial = calcular_indicadores(ordenado.iloc[: t + 1], ventana)
        a = parcial.loc[t, COLUMNAS_ESTADO].to_numpy(dtype=float)
        b = completo.loc[t, COLUMNAS_ESTADO].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=0, atol=1e-12):
            raise AssertionError(f"Indicador con información futura en la hora {t}")
    return len(horas)


def main() -> None:
    cfg = cargar_config()
    d = cfg["datos"]
    original = cargar_csv(d)
    final = preparar_datos(original, d["ventana"])
    conjuntos = dividir(final, d["fraccion_entrenamiento"], d["fraccion_validacion"])
    revisadas = auditar_lookahead(original, d["ventana"])
    salida = RAIZ / "data" / "procesados"
    salida.mkdir(parents=True, exist_ok=True)
    final.to_csv(salida / "base_analitica.csv", index=False)
    print(f"Filas originales: {len(original):,} | base analítica: {len(final):,}")
    for nombre, tabla in conjuntos.items():
        tabla.to_csv(salida / f"{nombre}.csv", index=False)
        print(f"{nombre:>13}: {len(tabla):>6,} filas | {tabla.date.min()} -> {tabla.date.max()}")
    print(f"Auditoría de lookahead: {revisadas} horas revisadas, ningún indicador usa el futuro.")


if __name__ == "__main__":
    main()
