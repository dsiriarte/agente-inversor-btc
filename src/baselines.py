"""Estrategias de referencia.

Todas usan el mismo entorno, el mismo capital y el mismo costo que el agente.
"""
import numpy as np


class ComprarYMantener:
    """Pide 100 % en BTC desde la primera hora y no vuelve a operar."""
    nombre = "Comprar y mantener"

    def reiniciar(self) -> None:
        pass

    def act(self, obs) -> int:
        return 2


class Constante50:
    """Referencia extra: mantiene siempre 50 % en BTC (rebalancea cada hora)."""
    nombre = "Constante 50 %"

    def reiniciar(self) -> None:
        pass

    def act(self, obs) -> int:
        return 1


class Aleatorio:
    """Elige 0 %, 50 % o 100 % al azar en cada hora, con una semilla fija."""

    def __init__(self, semilla: int):
        self.semilla = semilla
        self.nombre = f"Aleatorio (semilla {semilla})"
        self.reiniciar()

    def reiniciar(self) -> None:
        self.rng = np.random.default_rng(self.semilla)

    def act(self, obs) -> int:
        return int(self.rng.integers(0, 3))


def main() -> None:
    """python -m src.baselines --conjunto validacion   (o ajuste / prueba)"""
    import argparse

    from src.config import RESULTADOS, cargar_config
    from src.data import cargar_conjuntos
    from src.evaluate import evaluar_baselines

    parser = argparse.ArgumentParser(description="Evalúa solo las estrategias de referencia.")
    parser.add_argument("--conjunto", default="validacion", choices=["ajuste", "validacion", "prueba"])
    args = parser.parse_args()
    cfg = cargar_config()
    tabla = evaluar_baselines(cargar_conjuntos(cfg)[args.conjunto], cfg,
                              RESULTADOS / args.conjunto / "baselines")
    cols = ["estrategia", "retorno_pct", "max_drawdown_pct", "costos_usd", "cambios_exposicion"]
    print(tabla[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
