"""Funciones de recompensa.

Cada recompensa recibe lo que pasó en un paso y devuelve un número.
`valor_antes` es el capital antes de operar en la hora t y `valor_despues` es
el capital al cierre de t+1, ya descontado el costo de la operación. Por eso
todas las versiones incluyen los costos de transacción.
"""
import numpy as np


class RecompensaLog:
    """R1 — retorno logarítmico del portafolio neto de costos.

    r = ln(V_t+1 / V_t). Es la recompensa de la primera versión del entorno.
    """
    nombre = "R1"

    def reiniciar(self, capital_inicial: float) -> None:
        pass

    def __call__(self, valor_antes, valor_despues, **_) -> float:
        return float(np.log(max(valor_despues, 1e-12) / valor_antes))


class RecompensaLogRiesgo:
    """R2 — retorno logarítmico menos una penalización por riesgo.

    r = ln(V_t+1 / V_t) - lambda * ln(V_t+1 / V_t)^2

    Para un movimiento pequeño, ln(1 + w*x) ≈ w*x - (w*x)^2 / 2: el retorno
    logarítmico ya castiga un poco la varianza. Con lambda = 0,5 ese castigo
    se duplica. El término cuadrático depende de la volatilidad, que sí es
    predecible a partir del estado (volatility_24h), mientras que la dirección
    del precio no lo es.
    """
    nombre = "R2"

    def __init__(self, lam: float = 0.5):
        self.lam = float(lam)

    def reiniciar(self, capital_inicial: float) -> None:
        pass

    def __call__(self, valor_antes, valor_despues, **_) -> float:
        r = np.log(max(valor_despues, 1e-12) / valor_antes)
        return float(r - self.lam * r * r)


class RecompensaRiesgoRotacion:
    """R3 — R2 más una penalización por cambiar de exposición.

    r = ln(V_t+1 / V_t) - lambda * ln(V_t+1 / V_t)^2 - kappa * |w_t - w_t-1|

    w es la exposición objetivo (0; 0,5; 1). El costo real (0,1 %) ya está en
    V_t+1; kappa es un castigo adicional que solo existe para el aprendizaje:
    hace que cambiar de nivel tenga que "valer la pena" para el agente. Si
    lam = 0 queda R1 con penalización por rotación (versión de control).
    """
    nombre = "R3"

    def __init__(self, lam: float = 0.5, kappa: float = 0.005):
        self.lam, self.kappa = float(lam), float(kappa)

    def reiniciar(self, capital_inicial: float) -> None:
        self.previa = 0.0              # el agente empieza 100 % en efectivo

    def __call__(self, valor_antes, valor_despues, exposicion, **_) -> float:
        r = np.log(max(valor_despues, 1e-12) / valor_antes)
        rotacion = abs(exposicion - self.previa)
        self.previa = exposicion
        return float(r - self.lam * r * r - self.kappa * rotacion)


TIPOS = {"log": RecompensaLog, "log_riesgo": RecompensaLogRiesgo,
         "log_riesgo_rotacion": RecompensaRiesgoRotacion}


def crear_recompensa(cfg: dict):
    cfg = dict(cfg)
    tipo = cfg.pop("tipo")
    cfg.pop("descripcion", None)
    return TIPOS[tipo](**cfg)
