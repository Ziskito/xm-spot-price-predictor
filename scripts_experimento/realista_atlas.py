# -*- coding: utf-8 -*-
"""
Atlas mensual del escenario realista: todos los dias de 2026 con el precio real, el ensamble 7 realista (combinador
anterior) y el nuevo combinado predespacho + CatBoost (objetivo MAE). Un PNG por mes.
Uso: realista_atlas.py [archivo de predicciones en mejoras/] [columna nueva] [solo]
     (solo: dibuja unicamente el precio real y el modelo nuevo, en mejoras/atlas_solo/)
Salida: resultados/escenario_realista/mejoras/atlas/atlas_realista_nuevo_2026-MM.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402

MESES = {1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio", 7: "julio", 8: "agosto"}

if __name__ == "__main__":
    arch = sys.argv[1] if len(sys.argv) > 1 else "candidato_predicciones_42.csv"
    nueva = sys.argv[2] if len(sys.argv) > 2 else "Pre+Candidato · mae"
    solo = "solo" in sys.argv          # solo el precio real y el modelo nuevo
    out = rc.MEJ / ("atlas_solo" if solo else "atlas")
    out.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(rc.MEJ / arch, parse_dates=[0], index_col=0)
    p["dia"] = pd.to_datetime(p["dia"])
    p = p[p["dia"] >= "2026-01-15"].dropna(subset=["real", "ens7_realista", nueva, "Predespacho"])
    p["hora"] = (p.index.hour - 1) % 24 + 1          # ventana 01:00 a 00:00 (la 00:00 se dibuja como 24)
    for (anio, mes), dm in p.groupby([p["dia"].dt.year, p["dia"].dt.month]):
        lista = sorted(dm["dia"].unique())
        ncol, nfil = 6, int(np.ceil(len(lista) / 6))
        fig, axes = plt.subplots(nfil, ncol, figsize=(ncol * 3.1, nfil * 2.3), sharex=True)
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.set_visible(False)
        for ax, dia in zip(axes.ravel(), lista):
            g = dm[dm["dia"] == dia].sort_values("hora")
            ax.set_visible(True)
            ax.plot(g["hora"], g["real"], color="#0B1F3A", lw=1.7, label="real")
            if not solo:
                ax.plot(g["hora"], g["Predespacho"], color="#94A3B8", lw=1.0, ls="--", label="predespacho XM")
                ax.plot(g["hora"], g["ens7_realista"], color="#7C3AED", lw=1.1, alpha=.8, label="ensamble 7 realista")
            ax.plot(g["hora"], g[nueva], color="#D97706", lw=1.6, label="modelo nuevo (pre + 3 CatBoost)")
            e_n, e_v = np.abs(g[nueva] - g["real"]).mean(), np.abs(g["ens7_realista"] - g["real"]).mean()
            ax.set_title(f"{pd.Timestamp(dia):%a %d/%m}  MAE {e_n:.0f}" + ("" if solo else f" (antes {e_v:.0f})"), fontsize=7.5)
            ax.tick_params(labelsize=6.5)
            ax.set_xticks([1, 6, 12, 18, 24])
            ax.grid(alpha=.25)
        axes.ravel()[0].legend(fontsize=6, loc="upper left")
        mae_n = np.abs(dm[nueva] - dm["real"]).mean()
        mae_v = np.abs(dm["ens7_realista"] - dm["real"]).mean()
        if solo:
            fig.suptitle(f"Escenario realista · {MESES[mes]} {anio} · precio real y modelo nuevo · MAE del mes {mae_n:.1f} · "
                         f"sesgo {(dm[nueva] - dm['real']).mean():+.1f}", fontsize=10)
        else:
            fig.suptitle(f"Escenario realista · {MESES[mes]} {anio} · MAE del mes {mae_n:.1f} (ensamble 7 realista {mae_v:.1f}) · "
                         f"sesgo {(dm[nueva] - dm['real']).mean():+.1f} (antes {(dm['ens7_realista'] - dm['real']).mean():+.1f})", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(out / f"atlas_realista_nuevo_{anio}-{mes:02d}.png", dpi=110)
        plt.close(fig)
        print(f"  {anio}-{mes:02d}: MAE {mae_n:.1f} vs {mae_v:.1f}")
