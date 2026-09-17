# -*- coding: utf-8 -*-
"""
Bandas de incertidumbre para los ensambles: QRA + calibracion conforme adaptativa CAUSAL.

1) Correccion de una fuga leve en la calibracion adaptativa de 72h. La version actual
   (calibrar_bandas_72h_adaptativo.py y ensamble_72h_v2.py) calcula el margen de cada DIA OBJETIVO D
   con los scores de los dias objetivo D-30..D-1. En el tramo 49-72h el corte fue en D-3, asi que
   los scores de D-2 y D-1 todavia no existian al momento de pronosticar. Version corregida: el
   margen se calcula POR CORTE, usando solo filas cuyo instante objetivo es <= al corte.
   (En 24h no hay fuga: el corte es a las 23:00 de D-1 y todos los dias < D ya se observaron.)

2) QRA -- Quantile Regression Averaging (Nowotarski & Weron, 2015, Computational Statistics 30).
   El meta-modelo LAD que gano es exactamente QRA en el cuantil 0.5. La extension natural es
   ajustar tambien los cuantiles 0.1 y 0.9 sobre las mismas predicciones puntuales de los votantes,
   lo que le da BANDAS PROPIAS al ensamble de 24h (hoy solo N-BEATSx solo tiene bandas a 24h) y
   bandas coherentes con el punto en 72h. Variante con pesos no negativos e intercepto libre, por
   estabilidad ante votantes muy correlacionados.

Se evalua con cobertura, ancho y el Interval Score (Gneiting & Raftery 2007), que penaliza a la vez
el ancho y las observaciones que caen fuera: IS = (u-l) + (2/a)(l-y)1[y<l] + (2/a)(y-u)1[y>u], a=0.2.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog
from scipy import sparse
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import cargar as cargar24, BASE5
from ensamble_72h_v2 import cargar as cargar72, TRAMOS, RES

ALFA, NIVEL, VENTANA, CALENT = 0.2, 0.80, 30, 30
TAUS = [0.1, 0.5, 0.9]


def ajustar_cuantil(X, y, tau):
    """min sum rho_tau(y - Xw - b)  s.a. w >= 0, b libre."""
    n, k = X.shape
    c = np.concatenate([np.zeros(k + 2), np.full(n, tau), np.full(n, 1 - tau)])
    A = sparse.hstack([sparse.csr_matrix(X), sparse.csr_matrix(np.ones((n, 1))),
                       sparse.csr_matrix(-np.ones((n, 1))), sparse.eye(n), -sparse.eye(n)]).tocsr()
    r = linprog(c, A_eq=A, b_eq=y, bounds=(0, None), method="highs")
    return r.x[:k], r.x[k] - r.x[k + 1]


def qra_cv(df, cols, grupo=None):
    dias = np.sort(df["dia"].unique())
    pliegues = np.array_split(np.random.default_rng(42).permutation(dias), 5)
    X, y = df[cols].to_numpy(), df["real"].to_numpy()
    g = np.zeros(len(df), int) if grupo is None else df[grupo].to_numpy()
    Q = np.full((len(df), 3), np.nan)
    for dt in pliegues:
        mte = df["dia"].isin(dt).to_numpy()
        for gg in np.unique(g):
            tr, te = (~mte) & (g == gg), mte & (g == gg)
            for j, tau in enumerate(TAUS):
                w, b = ajustar_cuantil(X[tr], y[tr], tau)
                Q[te, j] = X[te] @ w + b
    Q[:, 0] = np.minimum(Q[:, 0], Q[:, 1])
    Q[:, 2] = np.maximum(Q[:, 2], Q[:, 1])
    return Q


def margen_por_corte(objetivo, corte, score):
    """Margen conforme causal: por corte, solo filas con instante objetivo <= corte (ventana 30d)."""
    m = np.zeros(len(score))
    for c in np.unique(corte):
        hist = (objetivo <= c) & (objetivo > c - np.timedelta64(VENTANA, "D"))
        n = hist.sum()
        if n >= 120:
            q = min(np.ceil((n + 1) * NIVEL) / n, 1.0)
            val = np.quantile(score[hist], q)
        else:
            prev = objetivo <= c
            val = np.quantile(score[prev], 0.90) if prev.sum() > 24 else 0.0
        m[corte == c] = val
    return m


def margen_por_dia_objetivo(fecha_obj, score):
    """Metodo anterior (con fuga a 72h): por dia objetivo, scores de los 30 dias objetivo previos."""
    dia = fecha_obj.astype("datetime64[D]")
    m = np.zeros(len(score))
    for dd in np.unique(dia):
        hist = (dia < dd) & (dia >= dd - np.timedelta64(VENTANA, "D"))
        n = hist.sum()
        if n >= 120:
            val = np.quantile(score[hist], min(np.ceil((n + 1) * NIVEL) / n, 1.0))
        else:
            prev = dia < dd
            val = np.quantile(score[prev], 0.90) if prev.sum() > 24 else 0.0
        m[dia == dd] = val
    return m


def metricas_banda(y, l, u):
    cob = ((y >= l) & (y <= u)).mean() * 100
    ancho = (u - l).mean()
    isc = (u - l) + (2 / ALFA) * (l - y) * (y < l) + (2 / ALFA) * (y - u) * (y > u)
    return cob, ancho, isc


def comparar(etiqueta, y, bandas, ref, filas, maxlags):
    print(f"  {'variante':44s} {'cobertura':>9s} {'ancho':>7s} {'IS':>7s}")
    is_ref = metricas_banda(y, *bandas[ref])[2]
    for nom, (l, u) in bandas.items():
        cob, ancho, isc = metricas_banda(y, l, u)
        extra = ""
        if nom != ref:
            d = is_ref - isc
            r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
            extra = f"  IS vs ref p={float(r.pvalues[0]):.4f} {'MEJOR' if r.pvalues[0]<0.05 and r.tvalues[0]>0 else ('PEOR' if r.pvalues[0]<0.05 else '')}"
        print(f"  {nom:44s} {cob:8.1f}% {ancho:7.1f} {isc.mean():7.1f}{extra}")
        filas.append({"caso": etiqueta, "variante": nom, "cobertura": cob, "ancho": ancho, "interval_score": isc.mean()})


def main():
    filas = []
    # ------------------------------ 72h ------------------------------
    print("=" * 90 + "\n72h -- 9 votantes, por tramo\n" + "=" * 90)
    d72, modelos, _ = cargar72()
    d72["dia"] = d72["fecha_hora"].dt.normalize()
    v3 = pd.read_csv(RES / "pronostico_ensamble_72h_v3_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v3_contrato = pd.read_csv(RES / "pronostico_ensamble_72h_v3_adaptativo.csv", parse_dates=["fecha_hora", "cutoff"])
    salida72 = []
    for nombre, ini, fin in TRAMOS:
        sub = (d72[(d72.paso_horas >= ini) & (d72.paso_horas <= fin)]
               .sort_values(["cutoff", "paso_horas"]).reset_index(drop=True))
        clave = ["fecha_hora", "cutoff", "paso_horas"]
        sub = sub.merge(v3[clave + ["q10", "q50", "q90"]].rename(columns={"q10": "v3_q10", "q50": "v3_q50", "q90": "v3_q90"}),
                        on=clave).merge(v3_contrato[clave + ["q10", "q90"]].rename(columns={"q10": "ct_q10", "q90": "ct_q90"}), on=clave)
        y = sub["real"].to_numpy()
        obj = sub["fecha_hora"].to_numpy()
        cor = sub["cutoff"].to_numpy()

        Q = qra_cv(sub, modelos)
        s_v3 = np.maximum(sub["v3_q10"] - y, y - sub["v3_q90"]).to_numpy()
        m_v3 = margen_por_corte(obj, cor, s_v3)
        s_q = np.maximum(Q[:, 0] - y, y - Q[:, 2])
        m_q = margen_por_corte(obj, cor, s_q)
        m_v3_fuga = margen_por_dia_objetivo(obj, s_v3)

        ev = obj >= obj.min() + np.timedelta64(CALENT, "D")
        print(f"\n--- Tramo {nombre} ({ev.sum()} filas evaluadas) --- MAE punto: LAD v3={np.abs(y-sub['v3_q50']).mean():.2f} | QRA q50={np.abs(y-Q[:,1]).mean():.2f}")
        bandas = {
            "contrato v3 actual (calibracion con fuga)": (sub["ct_q10"].to_numpy()[ev], sub["ct_q90"].to_numpy()[ev]),
            "v3 recalculado, calibracion con fuga": ((sub["v3_q10"] - m_v3_fuga).to_numpy()[ev], (sub["v3_q90"] + m_v3_fuga).to_numpy()[ev]),
            "v3 + calibracion CAUSAL (corregida)": ((sub["v3_q10"] - m_v3).to_numpy()[ev], (sub["v3_q90"] + m_v3).to_numpy()[ev]),
            "QRA cruda": (Q[ev, 0], Q[ev, 2]),
            "QRA + calibracion CAUSAL": ((Q[:, 0] - m_q)[ev], (Q[:, 2] + m_q)[ev]),
        }
        comparar(f"72h {nombre}", y[ev], bandas, "v3 + calibracion CAUSAL (corregida)", filas, 71)
        salida72.append(pd.DataFrame({"fecha_hora": obj, "cutoff": cor, "paso_horas": sub["paso_horas"], "real": y,
                                      "q50_lad": sub["v3_q50"], "q10_v3_causal": sub["v3_q10"] - m_v3,
                                      "q90_v3_causal": sub["v3_q90"] + m_v3, "q10_qra": Q[:, 0] - m_q,
                                      "q50_qra": Q[:, 1], "q90_qra": Q[:, 2] + m_q}))
    pd.concat(salida72).sort_values(["cutoff", "paso_horas"]).to_csv(RES / "bandas_72h_qra_causal_2026.csv", index=False)

    # ------------------------------ 24h ------------------------------
    print("\n" + "=" * 90 + "\n24h -- 5 modelos base, QRA por franja de 6h\n" + "=" * 90)
    d24 = cargar24().sort_index()
    d24["franja"] = d24.index.hour // 6
    Q = qra_cv(d24, BASE5, "franja")
    y = d24["real"].to_numpy()
    obj = d24.index.to_numpy()
    cor = (d24.index.normalize() - pd.Timedelta(hours=1)).to_numpy()
    m_q = margen_por_corte(obj, cor, np.maximum(Q[:, 0] - y, y - Q[:, 2]))
    ev = obj >= obj.min() + np.timedelta64(CALENT, "D")
    print(f"MAE punto QRA q50 (por franja): {np.abs(y-Q[:,1]).mean():.2f}  MAPE={(np.abs(y-Q[:,1])/y).mean()*100:.2f}%")
    bandas = {"QRA cruda": (Q[ev, 0], Q[ev, 2]),
              "QRA + calibracion CAUSAL": ((Q[:, 0] - m_q)[ev], (Q[:, 2] + m_q)[ev])}
    ruta_nb = RES / "pronostico_con_bandas_2026_adaptativo.csv"
    if ruta_nb.exists():
        nb = pd.read_csv(ruta_nb, parse_dates=["fecha_hora"])
        print(f"(columnas del contrato N-BEATSx 24h: {list(nb.columns)})")
        if {"q10", "q90"}.issubset(nb.columns):
            nb = nb.drop_duplicates("fecha_hora").set_index("fecha_hora").reindex(d24.index)
            okb = ev & nb["q10"].notna().to_numpy()
            if okb.sum() == ev.sum():
                bandas = {"N-BEATSx solo, adaptativa (contrato 24h actual)": (nb["q10"].to_numpy()[ev], nb["q90"].to_numpy()[ev]), **bandas}
    ref = "N-BEATSx solo, adaptativa (contrato 24h actual)" if len(bandas) == 3 else "QRA cruda"
    comparar("24h", y[ev], bandas, ref, filas, 23)
    # un margen conforme negativo (banda sobrecubierta) puede cruzar los cuantiles: se reordenan
    q10f = np.minimum(Q[:, 0] - m_q, Q[:, 1])
    q90f = np.maximum(Q[:, 2] + m_q, Q[:, 1])
    pd.DataFrame({"fecha_hora": obj, "real": y, "q10": q10f, "q50": Q[:, 1], "q90": q90f}).to_csv(
        RES / "bandas_24h_qra_causal_2026.csv", index=False)
    pd.DataFrame(filas).to_csv(RES / "qra_bandas_metricas.csv", index=False)
    print("\nGuardado: bandas_72h_qra_causal_2026.csv, bandas_24h_qra_causal_2026.csv, qra_bandas_metricas.csv")


if __name__ == "__main__":
    main()
