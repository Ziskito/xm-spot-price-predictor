# -*- coding: utf-8 -*-
"""
Diagnostico de "acierta la direccion pero no la magnitud" en el pronostico de 24 h (ensamble + bandas, 2026).

1. Atlas: una grafica por dia (real frente a mediana q50, con la banda [q10, q90]), una imagen por mes.
2. Descomposicion del error cuadratico de cada dia (Murphy, 1988):
       MSE = (media_p - media_r)^2  +  (sd_p - sd_r)^2  +  2 sd_p sd_r (1 - rho)
             \\_____ nivel _____/     \\__ amplitud __/    \\____ forma y tiempo ____/
   donde sd es la desviacion estandar dentro del dia y rho la correlacion horaria real-pronostico.
3. Cambios hora a hora: si el signo del cambio coincide (direccion) y la pendiente de
   delta_pronostico frente a delta_real (1 = magnitud correcta; < 1 = amortiguado).
4. Rampas: el 10 % de los cambios de 3 h mas grandes del precio real; cuanto de cada rampa anticipa el pronostico.
Salidas: data/processed/resultados/diagnostico_amplitud/ (atlas_AAAA-MM.png, por_dia.csv, resumen.txt)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "data" / "processed" / "resultados"
OUT = RES / "diagnostico_amplitud"
MESES = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]


def cargar():
    d = pd.read_csv(RES / "pronostico_ensamble_bandas_24h_2026.csv", parse_dates=["fecha_hora"])
    d["dia"] = d["fecha_hora"].dt.normalize()
    d["hora"] = d["fecha_hora"].dt.hour
    completos = d.groupby("dia")["hora"].transform("size") == 24
    return d[completos].reset_index(drop=True)


def por_dia(d):
    filas = []
    for dia, g in d.groupby("dia"):
        r, p = g["real"].to_numpy(), g["q50"].to_numpy()
        sr, sp = r.std(), p.std()
        rho = np.corrcoef(r, p)[0, 1] if sr > 0 and sp > 0 else np.nan
        filas.append({"dia": dia, "MAE": np.abs(r - p).mean(), "MSE": ((r - p) ** 2).mean(),
                      "nivel": (p.mean() - r.mean()) ** 2, "amplitud": (sp - sr) ** 2,
                      "forma": 2 * sp * sr * (1 - rho) if not np.isnan(rho) else np.nan,
                      "sesgo": p.mean() - r.mean(), "sd_real": sr, "sd_pron": sp,
                      "ratio_amplitud": sp / sr if sr > 0 else np.nan,
                      "rango_real": r.max() - r.min(), "rango_pron": p.max() - p.min(), "rho": rho})
    return pd.DataFrame(filas)


def atlas(d, dias):
    OUT.mkdir(parents=True, exist_ok=True)
    info = dias.set_index("dia")
    for (anio, mes), dm in d.groupby([d["dia"].dt.year, d["dia"].dt.month]):
        lista = sorted(dm["dia"].unique())
        cols, filas = 6, int(np.ceil(len(lista) / 6))
        fig, axes = plt.subplots(filas, cols, figsize=(cols * 3.1, filas * 2.25), sharex=True)
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.set_visible(False)
        for ax, dia in zip(axes.ravel(), lista):
            g = dm[dm["dia"] == dia]
            ax.set_visible(True)
            ax.fill_between(g["hora"], g["q10"], g["q90"], color="#1E3A8A", alpha=.12, lw=0)
            ax.plot(g["hora"], g["real"], color="#0B1F3A", lw=1.6, label="real")
            ax.plot(g["hora"], g["q50"], color="#D97706", lw=1.6, label="pronóstico")
            i = info.loc[dia]
            ax.set_title(f"{pd.Timestamp(dia):%a %d/%m}  MAE {i.MAE:.0f} · amp {i.ratio_amplitud:.2f} · ρ {i.rho:.2f}",
                         fontsize=7.5)
            ax.tick_params(labelsize=6.5)
            ax.set_xticks([0, 6, 12, 18, 23])
            ax.grid(alpha=.25)
        axes.ravel()[0].legend(fontsize=6.5, loc="upper left")
        fig.suptitle(f"Pronóstico de 24 h frente a precio real · {MESES[mes]} {anio}   "
                     "(amp = desviación del pronóstico / desviación real dentro del día; ρ = correlación horaria)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(OUT / f"atlas_{anio}-{mes:02d}.png", dpi=110)
        plt.close(fig)


def cambios(d):
    """Direccion y magnitud de los cambios hora a hora y de las rampas de 3 h (dentro de cada dia)."""
    res = {}
    for k in (1, 3):
        dr = d.groupby("dia")["real"].diff(k)
        dp = d.groupby("dia")["q50"].diff(k)
        m = dr.notna() & dp.notna()
        dr, dp = dr[m].to_numpy(), dp[m].to_numpy()
        grandes = np.abs(dr) >= np.quantile(np.abs(dr), .5)          # ignora cambios minusculos
        res[f"dir_{k}h"] = float((np.sign(dr[grandes]) == np.sign(dp[grandes])).mean() * 100)
        res[f"pendiente_{k}h"] = float((dr @ dp) / (dr @ dr))           # dp ~ b * dr
        top = np.abs(dr) >= np.quantile(np.abs(dr), .9)                  # rampas: 10 % mas grandes
        res[f"rampas_{k}h_dir"] = float((np.sign(dr[top]) == np.sign(dp[top])).mean() * 100)
        res[f"rampas_{k}h_fraccion"] = float(np.median(dp[top] / dr[top]))   # cuanto de la rampa anticipa
    return res


if __name__ == "__main__":
    d = cargar()
    dias = por_dia(d)
    OUT.mkdir(parents=True, exist_ok=True)
    dias.to_csv(OUT / "por_dia.csv", index=False)
    atlas(d, dias)
    tot = dias[["nivel", "amplitud", "forma"]].sum()
    pct = tot / tot.sum() * 100
    c = cambios(d)
    lineas = [
        f"Dias completos: {len(dias)} ({dias.dia.min():%d/%m} a {dias.dia.max():%d/%m})",
        f"MAE medio por dia: {dias.MAE.mean():.1f} COP/kWh",
        "Descomposicion del error cuadratico (suma de todos los dias):",
        f"  nivel (sesgo del dia)     {pct['nivel']:5.1f} %",
        f"  amplitud (sd distinta)    {pct['amplitud']:5.1f} %",
        f"  forma y tiempo (1 - rho)  {pct['forma']:5.1f} %",
        f"Amplitud: el pronostico se mueve {dias.ratio_amplitud.median():.2f} veces lo que se mueve el real (mediana por dia); "
        f"amortiguado (<0.9) en {(dias.ratio_amplitud < .9).mean() * 100:.0f} % de los dias, exagerado (>1.1) en "
        f"{(dias.ratio_amplitud > 1.1).mean() * 100:.0f} %",
        f"Rango del dia: pronostico/real = {(dias.rango_pron / dias.rango_real).median():.2f} (mediana)",
        f"Forma: correlacion horaria mediana {dias.rho.median():.2f}",
        f"Cambios de 1 h (mitad mas grande): direccion correcta {c['dir_1h']:.0f} %, pendiente {c['pendiente_1h']:.2f}",
        f"Cambios de 3 h (mitad mas grande): direccion correcta {c['dir_3h']:.0f} %, pendiente {c['pendiente_3h']:.2f}",
        f"Rampas (10 % de cambios de 3 h mas grandes): direccion {c['rampas_3h_dir']:.0f} %, "
        f"el pronostico anticipa {c['rampas_3h_fraccion'] * 100:.0f} % de la magnitud (mediana)",
    ]
    (OUT / "resumen.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))
