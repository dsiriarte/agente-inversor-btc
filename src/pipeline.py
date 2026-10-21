"""Ejecuta todo el experimento en orden.

Uso:
    python -m src.pipeline                 # usa los modelos incluidos en modelos/
    python -m src.pipeline --entrenar      # reentrena los 30 modelos (unos 3 minutos con 8 núcleos)

La selección ya está congelada en configs/seleccion.json, así que el paso de
validación no la cambia. Para repetir la selección desde cero hay que borrar
ese archivo antes de correr el pipeline.
"""
import argparse
import subprocess
import sys


def correr(*args) -> None:
    print("\n>>> python -m", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrenar", action="store_true", help="reentrenar todos los modelos")
    parser.add_argument("--procesos", default="8")
    args = parser.parse_args()
    correr("src.data")
    if args.entrenar:
        correr("src.train", "--todos", "--procesos", args.procesos)
    correr("src.evaluate", "--validacion")
    correr("src.evaluate", "--prueba")
    correr("src.evaluate", "--posterior")
    correr("src.auditoria")
    correr("src.figuras")


if __name__ == "__main__":
    main()
