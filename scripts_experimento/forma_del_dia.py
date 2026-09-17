# -*- coding: utf-8 -*-
"""
Problema 2 -- la FORMA del dia: cuando son las horas baratas y las caras.

El MAE mide que tan cerca queda cada hora de su valor real. Una regla de compra/venta necesita otra
cosa: ubicar bien en el dia las horas baratas y las caras. Tres metricas de Maciejowska, Lipiecki &
Uniejewski (2026) lo miden, y las tres dependen SOLO de la forma del dia (sumarle una constante a
todo un dia no las cambia):
  Corr-f  correlacion de Spearman entre el perfil pronosticado y el real (1 = mismo orden de horas)
  MHD     horas de error al ubicar el minimo y el maximo del dia (suma de ambos)
  MPD     COP/kWh que se pierden por operar en la hora senalada en vez de la optima

Diagnostico (metricas_forma_2026.py): el ensamble v4 es el mejor en MPD, pero la persistencia ordena
mejor las horas (Corr-f 0.835 vs 0.799) y ubica mejor el pico y el valle (MHD 4.0 vs 5.0 h), y v1
ordenaba mejor que v4 (los pesos por franja crean saltos en las fronteras de las franjas).

Solucion probada: separar NIVEL y FORMA. Nivel = media de la ventana de pronostico (01:00->00:00)
del ensamble v4, el que gana en MAE. Forma = desviacion de cada hora respecto a esa media, tomada de
otro pronostico o de un ensamble de formas ajustado solo sobre las desviaciones. Como las metricas
de forma solo dependen de las desviaciones, se optimizan sin tocar el nivel ganador.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import BASE5, pesos_lad
from qra_bandas import ajustar_cuantil
from metricas_forma_2026 import cargar_24h

RES = Path(r"C:\Users\mgdbj\xm-spot-price-predictor") / "data/processed/resultados"


def por_dia(fecha_hora, real, pred):
    """Series diarias (dia calendario) de Corr-f, MHD, MPD y amplitud -- formulas de metricas_decision.py."""
    d = pd.DataFrame({"f": pd.to_datetime(fecha_hora), "r": np.asarray(real, float), "p": np.asarray(pred, float)})
    d["dia"] = d["f"].dt.normalize()
    d["h"] = d["f"].dt.hour
    filas = []
    for dia, g in d.groupby("dia"):
        if len(g) < 20:
            continue
        rho = spearmanr(g["r"], g["p"]).correlation if g["r"].std() > 0 and g["p"].std() > 0 else np.nan
        rmin, rmax, pmin, pmax = g["r"].idxmin(), g["r"].idxmax(), g["p"].idxmin(), g["p"].idxmax()
        mhd = abs(g.at[rmin, "h"] - g.at[pmin, "h"]) + abs(g.at[rmax, "h"] - g.at[pmax, "h"])
        mpd = abs(g.at[rmin, "r"] - g.at[pmin, "r"]) + abs(g.at[rmax, "r"] - g.at[pmax, "r"])
        rango = g["r"].max() - g["r"].min()
        amp = (g["p"].max() - g["p"].min()) / rango if rango > 1 else np.nan
        filas.append((dia, rho, mhd, mpd, amp))
    return pd.DataFrame(filas, columns=["dia", "corr_f", "mhd", "mpd", "amp"]).set_index("dia")


def hac(d, maxlags):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def main():
    df = cargar_24h().rename(columns={"Ensamble v1 (NNLS)": "v1", "Ensamble v4 (QRA franjas)": "v4"}).sort_index()
    df["ventana"] = (df.index - pd.Timedelta(hours=1)).normalize()
    df["dia"] = df.index.normalize()
    cand = BASE5 + ["v1", "v4"]
    medias = df.groupby("ventana")[cand + ["real"]].transform("mean")
    dev = df[cand + ["real"]] - medias
    nivel_v4 = medias["v4"].to_numpy()

    dias = np.sort(df["dia"].unique())
    pliegues = np.array_split(np.random.default_rng(42).permutation(dias), 5)
    X, y = dev[cand].to_numpy(), dev["real"].to_numpy()
    forma_suma1 = np.full(len(df), np.nan)
    forma_libre = np.full(len(df), np.nan)
    w1s, w2s = [], []
    for dt in pliegues:
        te = df["dia"].isin(dt).to_numpy()
        tr = ~te
        w1 = pesos_lad(X[tr], y[tr])
        forma_suma1[te] = X[te] @ w1
        w2, b2 = ajustar_cuantil(X[tr], y[tr], 0.5)
        forma_libre[te] = X[te] @ w2 + b2
        w1s.append(w1)
        w2s.append(w2)

    variantes = {
        "v4 (referencia)": df["v4"].to_numpy(),
        "v1 (NNLS global)": df["v1"].to_numpy(),
        "Persistencia": df["Persistencia"].to_numpy(),
        "nivel v4 + forma persistencia": nivel_v4 + dev["Persistencia"].to_numpy(),
        "nivel v4 + forma v1": nivel_v4 + dev["v1"].to_numpy(),
        "nivel v4 + forma ajustada (pesos suman 1)": nivel_v4 + forma_suma1,
        "nivel v4 + forma ajustada (escala libre)": nivel_v4 + forma_libre,
    }
    real = df["real"].to_numpy()
    ref = variantes["v4 (referencia)"]
    ref_dia = por_dia(df.index, real, ref)
    filas = []
    print(f"{'variante':44s} {'MAE':>6s} {'MAPE':>6s} {'Corr-f':>7s} {'MHD':>5s} {'MPD':>6s} {'amp':>5s}   p(MAE) p(Corr-f) p(MHD) p(MPD)")
    for nom, p in variantes.items():
        e = np.abs(real - p)
        pdia = por_dia(df.index, real, p)
        fila = {"variante": nom, "mae": e.mean(), "mape": (e / real).mean() * 100, "corr_f": pdia["corr_f"].mean(),
                "mhd": pdia["mhd"].mean(), "mpd": pdia["mpd"].mean(), "amp": pdia["amp"].median()}
        ps = ""
        if nom != "v4 (referencia)":
            _, fila["p_mae"] = hac(np.abs(real - ref) - e, 23)
            j = ref_dia.join(pdia, rsuffix="_n", how="inner")
            _, fila["p_corr_f"] = hac(j["corr_f_n"] - j["corr_f"], 7)
            _, fila["p_mhd"] = hac(j["mhd"] - j["mhd_n"], 7)
            _, fila["p_mpd"] = hac(j["mpd"] - j["mpd_n"], 7)
            ps = f"  {fila['p_mae']:.4f} {fila['p_corr_f']:.4f}  {fila['p_mhd']:.4f} {fila['p_mpd']:.4f}"
        print(f"{nom:44s} {fila['mae']:6.2f} {fila['mape']:5.2f}% {fila['corr_f']:7.3f} {fila['mhd']:5.2f} "
              f"{fila['mpd']:6.2f} {fila['amp']:5.2f}{ps}")
        filas.append(fila)

    print("\nPesos de la forma ajustada (suma 1):  " + " | ".join(f"{m}={w:.2f}" for m, w in zip(cand, np.mean(w1s, 0))))
    print("Pesos de la forma ajustada (libre):   " + " | ".join(f"{m}={w:.2f}" for m, w in zip(cand, np.mean(w2s, 0))))
    pd.DataFrame(filas).to_csv(RES / "forma_del_dia_metricas.csv", index=False)
    out = pd.DataFrame({k: v for k, v in variantes.items()}, index=df.index)
    out.insert(0, "real", real)
    out.reset_index().to_csv(RES / "forma_del_dia_2026.csv", index=False)
    print("\nGuardado: forma_del_dia_metricas.csv, forma_del_dia_2026.csv")


if __name__ == "__main__":
    main()
