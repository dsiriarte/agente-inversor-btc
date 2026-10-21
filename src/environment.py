"""Entorno de inversión BTC/USD compatible con Gymnasium.

Rebalancea hasta la exposición pedida cobrando el costo sobre el monto operado.
La recompensa es configurable y `info` registra costos, cambios de exposición,
posición y fecha de cada paso.

Momento de la decisión (lookahead-safe):
- En la hora t el agente ve los indicadores calculados con precios hasta el
  cierre de t y la posición que tiene en ese momento.
- La orden se ejecuta al precio de cierre de t y paga el costo sobre el monto
  operado.
- El resultado se mide con el cierre de t+1, que el agente todavía no conoce.

Acciones (exposición objetivo): 0 -> 0 % BTC, 1 -> 50 % BTC, 2 -> 100 % BTC.
No hay apalancamiento, ventas en corto ni intereses sobre el efectivo.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.data import COLUMNAS_ESTADO
from src.rewards import crear_recompensa


class EntornoInversionBTC(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, datos: pd.DataFrame, capital_inicial: float = 10_000.0,
                 costo_transaccion: float = 0.001, fracciones=(0.0, 0.5, 1.0),
                 recompensa: dict | None = None, escalado: dict | None = None):
        super().__init__()
        datos = datos.reset_index(drop=True)
        self.x = datos[COLUMNAS_ESTADO].to_numpy(dtype=np.float64)
        if escalado is not None:
            # Media y desviación calculadas SOLO con el periodo de ajuste (ver escalado_desde).
            self.x = (self.x - np.asarray(escalado["media"])) / np.asarray(escalado["desv"])
        self.x = self.x.astype(np.float32)
        self.p = datos["close"].to_numpy(dtype=np.float64)
        self.fechas = datos["date"].to_numpy() if "date" in datos else np.arange(len(datos))
        if len(self.p) < 2 or not np.all(np.isfinite(self.p)) or not np.all(self.p > 0):
            raise ValueError("Los precios deben ser positivos y finitos (mínimo dos horas).")
        if not np.all(np.isfinite(self.x)):
            raise ValueError("Hay indicadores no finitos.")
        self.capital_inicial = float(capital_inicial)
        self.costo = float(costo_transaccion)
        self.fracciones = np.asarray(fracciones, dtype=np.float64)
        self.recompensa_cfg = recompensa or {"tipo": "log"}
        self.action_space = spaces.Discrete(len(self.fracciones))
        # Tres indicadores sin escalar + fracción del capital que está en BTC.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(COLUMNAS_ESTADO) + 1,),
                                            dtype=np.float32)

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        return np.r_[self.x[self.i], np.float32(self.pos)].astype(np.float32)

    def _info(self) -> dict:
        return {"capital": self.capital, "costos_acumulados": self.costos,
                "cambios_exposicion": self.cambios, "rebalanceos": self.rebalanceos,
                "posicion": self.pos, "objetivo": self.objetivo,
                "precio": float(self.p[self.i]), "fecha": self.fechas[self.i]}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0
        self.cash, self.btc = self.capital_inicial, 0.0
        self.capital = self.capital_inicial
        self.pos = 0.0                      # fracción real en BTC
        self.objetivo = None                # última exposición pedida
        self.costos, self.cambios, self.rebalanceos = 0.0, 0, 0
        self.recompensa = crear_recompensa(self.recompensa_cfg)
        self.recompensa.reiniciar(self.capital_inicial)
        return self._obs(), self._info()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")
        action = int(action)
        precio = self.p[self.i]
        anterior = self.cash + self.btc * precio
        objetivo = float(self.fracciones[action])
        actual = self.btc * precio

        # Rebalanceo exacto hasta la fracción objetivo, pagando el costo sobre el monto operado.
        if objetivo > actual / anterior + 1e-12:
            monto = (objetivo * anterior - actual) / (1 + objetivo * self.costo)
            monto = max(0.0, min(monto, self.cash / (1 + self.costo)))
            self.cash -= monto * (1 + self.costo)
            self.btc += monto / precio
        else:
            monto = (actual - objetivo * anterior) / (1 - objetivo * self.costo)
            monto = max(0.0, min(monto, actual))
            self.cash += monto * (1 - self.costo)
            self.btc -= monto / precio
        costo_paso = monto * self.costo
        self.costos += costo_paso
        if monto > 1e-8:
            if self.objetivo is None or objetivo != self.objetivo:
                self.cambios += 1          # el agente cambió de nivel de exposición
            else:
                self.rebalanceos += 1      # mismo nivel: solo corrige lo que movió el precio
        self.objetivo = objetivo

        # Avanza una hora: el resultado se mide con el cierre de t+1.
        self.i += 1
        self.capital = self.cash + self.btc * self.p[self.i]
        self.pos = (self.btc * self.p[self.i]) / self.capital
        r = self.recompensa(valor_antes=anterior, valor_despues=self.capital,
                            exposicion=objetivo, costo=costo_paso,
                            precio_antes=precio, precio_despues=self.p[self.i])
        terminado = self.i >= len(self.p) - 1
        return self._obs(), float(r), terminado, False, self._info()


def escalado_desde(datos_ajuste: pd.DataFrame) -> dict:
    """Media y desviación de cada indicador en el periodo de ajuste.

    Se calculan una sola vez con datos de entrenamiento y se reutilizan sin
    cambios en validación y prueba, para no filtrar información del futuro.
    """
    return {"columnas": COLUMNAS_ESTADO,
            "media": datos_ajuste[COLUMNAS_ESTADO].mean().tolist(),
            "desv": datos_ajuste[COLUMNAS_ESTADO].std().tolist()}


def crear_entorno(datos: pd.DataFrame, cfg: dict, recompensa: dict | None = None,
                  escalado: dict | None = None) -> EntornoInversionBTC:
    e = cfg["entorno"]
    return EntornoInversionBTC(datos, e["capital_inicial"], e["costo_transaccion"],
                               e["fracciones"], recompensa, escalado)
