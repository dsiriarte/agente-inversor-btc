"""Lectura de la configuración del proyecto y rutas comunes."""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CONFIGS = RAIZ / "configs"
MODELOS = RAIZ / "modelos"
RESULTADOS = RAIZ / "resultados"
FIGURAS = RESULTADOS / "figuras"


def cargar_config(nombre: str = "base.json") -> dict:
    """Lee un archivo JSON de la carpeta configs/."""
    return json.loads((CONFIGS / nombre).read_text(encoding="utf-8"))


def guardar_json(datos: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
