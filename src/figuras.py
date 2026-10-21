"""Figuras del informe a partir de los CSV guardados en resultados/.

Uso:  python -m src.figuras
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURAS, MODELOS, RESULTADOS, cargar_config
from src.data import cargar_conjuntos

# Paleta categórica validada (orden fijo): agente, comprar y mantener, constante 50 %.
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRIS, GRIS_CLARO, TINTA, TINTA_2 = "#8a8986", "#d7d6d2", "#0b0b0b", "#52514e"
AZUL_CLARO = "#9ec5f4"
NOMBRES = {"R1_sin_escalar": "R1\nsin esc.", "R1": "R1", "R2": "R2",
           "R3_k002": "R3\n0,002", "R3_k005": "R3\n0,005", "R3_k010": "R3\n0,010"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9.5, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": GRIS_CLARO,
    "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.grid": True, "grid.color": "#ecebe8", "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.8, "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
})


def _fechas(serie):
    return pd.to_datetime(serie, utc=True)


def _guardar(fig, nombre):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / nombre, bbox_inches="tight")
    plt.close(fig)


MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def _formato_fechas(ax, meses=2):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=meses))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{MESES[mdates.num2date(v).month - 1]} {mdates.num2date(v).year}"))


def _etiqueta_final(ax, x, y, texto, color):
    ax.annotate(texto, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=TINTA, fontweight="bold")
    ax.plot([x], [y], "o", color=color, ms=4)


def fig_division(conjuntos):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    colores = {"ajuste": GRIS_CLARO, "validacion": "#cde2fb", "prueba": "#fbd9c9"}
    textos = {"ajuste": "Ajuste (entrenamiento)", "validacion": "Validación", "prueba": "Prueba (held-out)"}
    for nombre in ["ajuste", "validacion", "prueba"]:
        d = conjuntos[nombre]
        f = _fechas(d.date)
        ax.axvspan(f.iloc[0], f.iloc[-1], color=colores[nombre], alpha=0.55, lw=0)
        ax.text(f.iloc[len(f) // 2], 79_000, f"{textos[nombre]}\n{len(d):,} h".replace(",", "."),
                ha="center", va="top", fontsize=8.5, color=TINTA)
        ax.plot(f, d.close, color=TINTA, lw=0.9)
    ax.set_ylabel("Precio de cierre (USD)")
    ax.set_ylim(0, 82_000)
    ax.set_title("BTC/USD horario y división cronológica de los datos")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f} mil"))
    _formato_fechas(ax, 6)
    _guardar(fig, "fig01_division_datos.png")


def _historiales_prueba(sel):
    base = RESULTADOS / "prueba"
    agente = {s: pd.read_csv(base / "recorridos" / sel["experimento"] / f"historial_semilla_{s}.csv")
              for s in sel["semillas"]}
    ref = {n: pd.read_csv(base / "baselines" / f"historial_{n}.csv")
           for n in ["comprar_y_mantener", "constante_50", "aleatorio_mediano"]}
    return agente, ref


def fig_capital(sel):
    agente, ref = _historiales_prueba(sel)
    t = sel["semilla_tipica"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for s, h in agente.items():
        if s != t:
            ax.plot(_fechas(h.fecha), h.capital, color=AZUL_CLARO, lw=0.9, alpha=0.9)
    ax.plot([], [], color=AZUL_CLARO, lw=1, label="Agente DQN, otras 4 semillas")
    series = [(ref["aleatorio_mediano"], GRIS, "Aleatorio (semilla mediana)"),
              (ref["constante_50"], AQUA, "Constante 50 %"),
              (ref["comprar_y_mantener"], NARANJA, "Comprar y mantener"),
              (agente[t], AZUL, f"Agente DQN final (semilla {t})")]
    for h, color, nombre in series:
        f = _fechas(h.fecha)
        ax.plot(f, h.capital, color=color, label=nombre, lw=2 if color == AZUL else 1.6)
        _etiqueta_final(ax, f.iloc[-1], h.capital.iloc[-1], f"{h.capital.iloc[-1]:,.0f}".replace(",", "."), color)
    ax.axhline(10_000, color=GRIS, lw=0.8)
    ax.set_ylabel("Capital (USD)")
    ax.set_title("Curva de capital en el periodo de prueba (9 ene 2021 – 1 mar 2022)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8.5)
    _formato_fechas(ax)
    ax.set_xlim(right=ax.get_xlim()[1] + 25)
    _guardar(fig, "fig02_curva_capital_prueba.png")


def fig_drawdown(sel):
    agente, ref = _historiales_prueba(sel)
    t = sel["semilla_tipica"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for h, color, nombre in [(ref["constante_50"], AQUA, "Constante 50 %"),
                             (ref["comprar_y_mantener"], NARANJA, "Comprar y mantener"),
                             (agente[t], AZUL, f"Agente DQN final (semilla {t})")]:
        dd = (h.capital / h.capital.cummax() - 1) * 100
        ax.plot(_fechas(h.fecha), dd, color=color, label=f"{nombre}: mínimo {dd.min():.1f} %")
    ax.set_ylabel("Caída desde el máximo (%)")
    ax.set_title("Drawdown en el periodo de prueba")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8.5)
    _formato_fechas(ax)
    _guardar(fig, "fig03_drawdown_prueba.png")


def fig_posiciones(sel):
    agente, ref = _historiales_prueba(sel)
    t = sel["semilla_tipica"]
    h = agente[t]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True,
                                 gridspec_kw={"height_ratios": [1.6, 1]})
    f = _fechas(h.fecha)
    a1.plot(f, h.precio, color=TINTA, lw=0.9)
    a1.set_ylabel("Precio (USD)")
    a1.set_title(f"Exposición elegida por el agente final (semilla {t}) en la prueba")
    a2.step(f, h.objetivo * 100, where="post", color=AZUL, lw=1.2)
    a2.set_yticks([0, 50, 100])
    a2.set_ylabel("Exposición (%)")
    a2.set_ylim(-8, 108)
    _formato_fechas(a2)
    _guardar(fig, "fig04_posiciones_prueba.png")


def fig_costos(sel):
    tabla = pd.read_csv(RESULTADOS / "prueba" / "tabla_final.csv")
    por_semilla = pd.read_csv(RESULTADOS / "prueba" / "agente_por_semilla.csv")
    filas = [("Comprar y mantener", tabla.iloc[2].costos_usd, NARANJA),
             ("Constante 50 %", tabla.iloc[3].costos_usd, AQUA),
             (f"Agente DQN final (semilla {sel['semilla_tipica']})", tabla.iloc[0].costos_usd, AZUL),
             ("Agente DQN, mediana 5 semillas", por_semilla.costos_usd.median(), AZUL_CLARO),
             ("Aleatorio, mediana 100 semillas", tabla.iloc[4].costos_usd, GRIS)]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    y = np.arange(len(filas))
    ax.barh(y, [f[1] for f in filas], color=[f[2] for f in filas], height=0.6)
    for i, (_, v, _) in enumerate(filas):
        ax.text(v * 1.08, i, f"{v:,.0f} USD".replace(",", "."), va="center", fontsize=8.5, color=TINTA)
    ax.set_yticks(y, [f[0] for f in filas])
    ax.set_xscale("log")
    ax.set_xlim(5, 60_000)
    ax.invert_yaxis()
    ax.set_xlabel("Costos de transacción pagados (USD, escala logarítmica)")
    ax.set_title("Costos de transacción en el periodo de prueba")
    ax.grid(axis="y", visible=False)
    _guardar(fig, "fig05_costos_prueba.png")


def fig_recompensas():
    v = pd.read_csv(RESULTADOS / "validacion" / "experimentos_por_semilla.csv")
    base = pd.read_csv(RESULTADOS / "validacion" / "baselines.csv").set_index("estrategia")
    orden = list(NOMBRES)
    paneles = [("retorno_pct", "Retorno acumulado (%)"), ("max_drawdown_pct", "Máximo drawdown (%)"),
               ("cambios_exposicion", "Cambios de exposición")]
    fig, ejes = plt.subplots(1, 3, figsize=(9.5, 4.2))
    for ax, (col, titulo) in zip(ejes, paneles):
        for i, e in enumerate(orden):
            vals = v.loc[v.experimento == e, col]
            ax.scatter(np.full(len(vals), i), vals, s=22, color=AZUL, alpha=0.55, lw=0)
            ax.plot([i - 0.28, i + 0.28], [vals.median()] * 2, color=TINTA, lw=2)
        if col != "cambios_exposicion":
            for nombre, color in [("Comprar y mantener", NARANJA), ("Constante 50 %", AQUA)]:
                ax.axhline(base.loc[nombre, col], color=color, lw=1.4, label=nombre)
        ax.set_xticks(range(len(orden)), [NOMBRES[e] for e in orden], fontsize=8.5)
        ax.set_title(titulo, fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.legend(*ejes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    ejes[2].set_yscale("log")
    fig.suptitle("Versiones de la recompensa en validación (κ en las etiquetas de R3)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    _guardar(fig, "fig06_recompensas_validacion.png")


def fig_puntos_control(experimento="R1_sin_escalar"):
    cfg = cargar_config()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, s in enumerate(cfg["semillas"]):
        pc = pd.read_csv(RESULTADOS / "validacion" / experimento / f"semilla_{s}" / "puntos_control.csv")
        final = json.loads((MODELOS / experimento / f"semilla_{s}" / "resumen.json").read_text())
        pasos = list(pc.paso / 1000) + [cfg["agente"]["pasos"] / 1000]
        expo = list(pc.exposicion_media_pct) + [final["validacion"]["exposicion_media_pct"]]
        ax.plot(pasos, expo, marker="o", ms=4, lw=1.3, label=f"semilla {s}",
                color=[AZUL, NARANJA, AQUA, "#eda100", "#e87ba4"][i])
    ax.set_xlabel("Pasos de entrenamiento (miles)")
    ax.set_ylabel("Exposición media en validación (%)")
    ax.set_title("R1 sin escalar: la política cambia de nivel durante el entrenamiento")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    _guardar(fig, "fig07_puntos_control_R1.png")


def fig_anomalia(sel, semilla=456):
    base = RESULTADOS / "prueba"
    h = pd.read_csv(base / "recorridos" / sel["experimento"] / f"historial_semilla_{semilla}.csv")
    bh = pd.read_csv(base / "baselines" / "historial_comprar_y_mantener.csv")
    f = _fechas(h.fecha)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    a1.plot(f, h.precio, color=TINTA, lw=0.9)
    entra = h.objetivo > 0
    a1.scatter(f[entra], h.precio[entra], s=14, color=AZUL, zorder=3, label="Horas invertido")
    a1.set_ylabel("Precio (USD)")
    a1.legend(loc="upper left", fontsize=8.5)
    a1.set_title(f"Anomalía: semilla {semilla} invertida solo {int(entra.sum())} de {len(h) - 1} horas")
    a2.plot(f, h.capital, color=AZUL, label=f"Agente semilla {semilla}")
    a2.plot(_fechas(bh.fecha), bh.capital, color=NARANJA, label="Comprar y mantener")
    a2.set_ylabel("Capital (USD)")
    a2.legend(loc="upper left", fontsize=8.5)
    _formato_fechas(a2)
    _guardar(fig, "fig08_anomalia_semilla_456.png")


def fig_sobreoperacion(semilla=42):
    """Una semana de validación: el agente con estado escalado cambia de nivel sin parar."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for exp, color, nombre in [("R1", AZUL, "R1 (estado escalado)"),
                               ("R1_sin_escalar", NARANJA, "R1 sin escalar")]:
        h = pd.read_csv(RESULTADOS / "validacion" / exp / f"semilla_{semilla}" / "historial.csv")
        f = _fechas(h.fecha)
        ventana = (f >= "2020-11-02") & (f < "2020-11-09")
        cambios = int((h.objetivo[ventana].diff().fillna(0) != 0).sum())
        ax.step(f[ventana], h.objetivo[ventana] * 100, where="post", color=color, lw=1.3,
                label=f"{nombre}: {cambios} cambios en la semana")
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Exposición (%)")
    ax.set_ylim(-8, 108)
    ax.set_title(f"Sobreoperación: una semana de validación (2 al 8 nov 2020), semilla {semilla}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{mdates.num2date(v).day} {MESES[mdates.num2date(v).month - 1]}"))
    _guardar(fig, "fig09_sobreoperacion.png")


def fig_entrenamiento(experimento="R1_sin_escalar"):
    cfg = cargar_config()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, s in enumerate(cfg["semillas"]):
        carpeta = MODELOS / experimento / f"semilla_{s}"
        m = pd.read_csv(carpeta / "monitor.monitor.csv", skiprows=1)
        capital = 10_000 * np.exp(m.r)      # con R1, la suma de recompensas es ln(V_final / V_0)
        a1.plot(np.arange(1, len(m) + 1), capital, marker="o", ms=4, lw=1.3, label=f"semilla {s}",
                color=[AZUL, NARANJA, AQUA, "#eda100", "#e87ba4"][i])
        if i == 0:
            e = pd.read_csv(carpeta / "epsilon.csv")
            a2.plot(e.paso / 1000, e.epsilon, color=AZUL)
    a1.axhline(10_000, color=GRIS, lw=0.8)
    a1.set_xlabel("Episodio (recorrido completo del ajuste)")
    a1.set_ylabel("Capital al final del episodio (USD)")
    a1.set_title("Capital al final de cada episodio")
    a1.legend(fontsize=8, ncol=2, loc="center left")
    a2.set_xlabel("Pasos de entrenamiento (miles)")
    a2.set_ylabel("ε")
    a2.set_title("Exploración ε-greedy")
    fig.tight_layout()
    _guardar(fig, "fig10_entrenamiento.png")


def main() -> None:
    sel = cargar_config("seleccion.json")
    fig_division(cargar_conjuntos())
    fig_capital(sel)
    fig_drawdown(sel)
    fig_posiciones(sel)
    fig_costos(sel)
    fig_recompensas()
    fig_puntos_control()
    fig_anomalia(sel)
    fig_sobreoperacion()
    fig_entrenamiento()
    print("Figuras guardadas en", FIGURAS)


if __name__ == "__main__":
    main()
