# -*- coding: utf-8 -*-
"""
Prueba de las tres metricas de estres del sistema recien descargadas de XM.

Tamiz de fuga previo (mismo metodo que uso la sesion del 2026-09-09, correlacion parcial
descontando precio_lag24h), sobre las 65.833 horas del proyecto:

  restricciones_sin_alivios       contemporanea -0.328   parcial -0.125   PASA
  generacion_programada_despacho  contemporanea +0.342   parcial +0.117   PASA
  generacion_fuera_merito         contemporanea -0.048   parcial -0.031   no pasa (< 0.05)

Ninguna muestra fuga (la fuga se veria como correlacion contemporanea ~0.99, que es lo que delato
a max_precio_oferta). Las dos que pasan tienen correlacion parcial del mismo orden que
costo_marginal_despacho (0.178), que era el mejor candidato de aquella sesion.

Punto importante sobre GeneProgDesp: es el PROGRAMA DE DESPACHO, que XM publica el dia anterior.
Para un objetivo t del dia D con corte a las 00:00 del dia D, el programa de TODAS las horas de D
ya esta publicado. Por eso se prueba en dos versiones:
  (a) valor EN EL CORTE -- indiscutiblemente seguro;
  (b) valor EN LA HORA OBJETIVO -- legitimo si la publicacion es dia-adelante, y es donde estaria
      su valor real, porque seria informacion genuinamente anticipativa. Se marca como tal.
Se prueban por separado justamente para que la version (b) quede senalada y se pueda descartar si
alguien objeta el supuesto de publicacion.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES
from lear_24h import construir_para_paso
from stacking_24h_v2 import cargar as cargar24, BASE5
from qra_bandas import ajustar_cuantil

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
CORTE = pd.Timestamp("2026-01-01")
H, PICO = 24, [18, 19, 20]
NUEVAS = ["restricciones_sin_alivios", "generacion_programada_despacho", "generacion_fuera_merito"]


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def qra_cv(d, cols, grupo="g4", semilla=42):
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d[grupo].to_numpy(), d["dia"].to_numpy()
    P = np.full(len(d), np.nan)
    for dt in np.array_split(np.random.default_rng(semilla).permutation(np.sort(np.unique(dia))), 5):
        te_m = np.isin(dia, dt)
        for gg in np.unique(g[te_m]):
            tr, te = (~te_m) & (g == gg), te_m & (g == gg)
            if tr.sum() >= len(cols) + 5:
                w, b = ajustar_cuantil(X[tr], y[tr], 0.5)
                P[te] = X[te] @ w + b
    return P


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler

    df = cargar_completo().reset_index(drop=True)
    idx = pd.DatetimeIndex(df["fecha_hora"])
    nuevas = {}
    for n in NUEVAS:
        s = pd.read_csv(RAIZ / f"data/{n}_2019_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
        nuevas[n] = s.iloc[:, 0].reindex(idx).to_numpy(np.float64)

    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - H)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]
    pos = pd.Series(np.arange(len(df)), index=fechas)

    # ---------- LEAR24 en tres variantes ----------
    print("Construyendo LEAR24 en 3 variantes (un LASSO por hora)...")
    salidas = {}
    for etiqueta in ["base", "estres en el corte", "estres + programa en la hora objetivo"]:
        partes, alphas = [], []
        for h in range(1, H + 1):
            Xtr, ytr, f_tr, c_tr = construir_para_paso(df, h, cortes_tr)
            Xte, yte, f_te, c_te = construir_para_paso(df, h, cortes_te)
            ictr = pos.reindex(pd.DatetimeIndex(c_tr)).to_numpy()
            icte = pos.reindex(pd.DatetimeIndex(c_te)).to_numpy()
            itr = pos.reindex(pd.DatetimeIndex(f_tr)).to_numpy()
            ite = pos.reindex(pd.DatetimeIndex(f_te)).to_numpy()

            def bloque(ic, it):
                b = {}
                if etiqueta == "base":
                    return pd.DataFrame(index=range(len(ic)))
                for n in NUEVAS:                       # siempre: valor en el corte (seguro)
                    b[f"{n}_corte"] = nuevas[n][ic]
                    b[f"{n}_corte24"] = nuevas[n][np.maximum(ic - 24, 0)]
                if etiqueta.endswith("hora objetivo"):  # ademas: programa en la hora objetivo
                    b["programa_objetivo"] = nuevas["generacion_programada_despacho"][it]
                return pd.DataFrame(b)

            A = pd.concat([Xtr.reset_index(drop=True), bloque(ictr, itr)], axis=1)
            B = pd.concat([Xte.reset_index(drop=True), bloque(icte, ite)], axis=1)
            A, B = A.fillna(A.median()), B.fillna(A.median())
            sc = StandardScaler().fit(A)
            if h <= 4:
                m = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(sc.transform(A), ytr)
                alphas.append(m.alpha_)
            else:
                m = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(
                    sc.transform(A), ytr)
            partes.append(pd.DataFrame({"fecha_hora": f_te, "real": yte, "pred": m.predict(sc.transform(B))}))
        r = pd.concat(partes, ignore_index=True)
        salidas[etiqueta] = r[r["fecha_hora"] >= CORTE]

    print(f"\n{'='*96}\nLEAR24 SUELTO CON LAS VARIABLES DE ESTRES\n{'='*96}")
    print(f"{'variante':42s} {'MAE':>8s} {'MAPE %':>8s} {'MAPE pico':>10s}   DM vs base")
    ref = salidas["base"]
    yb = ref["real"].to_numpy()
    pb = ref["pred"].to_numpy()
    hb = pd.DatetimeIndex(ref["fecha_hora"]).hour
    pkb = np.isin(hb, PICO)
    for etiqueta, r in salidas.items():
        p = r["pred"].to_numpy()
        linea = f"{etiqueta:42s} {np.abs(yb-p).mean():8.2f} {mape(yb, p):8.2f} {mape(yb[pkb], p[pkb]):10.2f}"
        if etiqueta != "base":
            t, pv = dm(yb, pb, p)
            marca = ("MEJORA" if t > 0 else "empeora") if pv < 0.05 else "n.s."
            linea += f"   p={pv:.4f} {marca}"
        print(linea)

    # ---------- en el ensamble ----------
    d = cargar24().sort_index()
    for etiqueta, r in salidas.items():
        if etiqueta == "base":
            continue
        nom = "LEAR_corte" if "corte" in etiqueta and "objetivo" not in etiqueta else "LEAR_prog"
        d = d.join(r.set_index("fecha_hora")["pred"].rename(nom), how="left")
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d.index.hour // 6
    d = d.dropna(subset=BASE5 + ["LEAR_corte", "LEAR_prog", "real"])
    y = d["real"].to_numpy()
    pk = d["hora"].isin(PICO).to_numpy()
    base_p = qra_cv(d, BASE5)
    print(f"\n{'='*96}\nEN EL ENSAMBLE DE 24h (QRA por franja, CV por dias)\n{'='*96}")
    print(f"{'conjunto':44s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s} {'pico %':>8s}   DM vs v4")
    for etiqueta, cols in {
        "v4 vigente": BASE5,
        "v4 + LEAR(estres en corte)": BASE5 + ["LEAR_corte"],
        "v4 + LEAR(estres + programa objetivo)": BASE5 + ["LEAR_prog"],
        "v4 + ambos": BASE5 + ["LEAR_corte", "LEAR_prog"],
    }.items():
        p = qra_cv(d, cols)
        ok = ~np.isnan(p)
        sm_ = float(np.mean(np.abs(y[ok]-p[ok]) / ((np.abs(y[ok])+np.abs(p[ok]))/2)) * 100)
        linea = (f"{etiqueta:44s} {np.abs(y[ok]-p[ok]).mean():8.2f} {mape(y[ok], p[ok]):8.2f} "
                 f"{sm_:8.2f} {mape(y[ok&pk], p[ok&pk]):8.2f}")
        if etiqueta != "v4 vigente":
            t, pv = dm(y[ok], base_p[ok], p[ok])
            marca = ("MEJORA" if t > 0 else "peor") if pv < 0.05 else "n.s."
            linea += f"   p={pv:.4f} {marca}"
        print(linea)

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
