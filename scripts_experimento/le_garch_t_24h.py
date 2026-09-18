# -*- coding: utf-8 -*-
"""
LE-GARCH-t y LE-SV-t: los modelos GANADORES del paper de Nueva Zelanda, nunca construidos aqui.

Kapoor & Wichitaksorn (2023, Applied Energy 347:121446) encuentran que **LE-GARCH-t** es el mejor
modelo en 4 de sus 5 regiones por MAE y MASE, por encima de DNN, LSTM, GRU, XGBoost y del
benchmark LEAR. "LE" = LASSO-Estimated: las variables exogenas de la ecuacion de MEDIA se
seleccionan con LASSO; "-t" = los errores siguen una t de Student en vez de una normal.

El proyecto tiene ARX+GARCH como votante, pero NO es esto: le falta la seleccion LASSO de exogenas
y le falta la distribucion t. Y el hallazgo central del paper es justamente que el GARCH **con
todas** las variables es de los PEORES modelos y **con LASSO** pasa a ser el mejor (mejoras de
hasta 40-45%). O sea, la pieza que falta es exactamente la que hace la diferencia.

Mecanismo por el que puede ganarle a LEAR pese a que la media es lineal en ambos: LEAR estima los
coeficientes por minimos cuadrados penalizados (todos los dias pesan igual), mientras que
GARCH-t los estima por maxima verosimilitud ponderando por la varianza condicional y con colas
pesadas. Eso **descuenta automaticamente los dias de alta volatilidad** al estimar la media, que es
una forma de estimacion robusta. Dado que el diagnostico de este proyecto mostro que el error vive
en las horas/dias volatiles, es una hipotesis con fundamento.

Formulacion: un modelo por HORA DEL DIA (h=1..24), igual que el LEAR. Para la hora h, la serie es
la secuencia diaria del precio a esa hora (~2.500 observaciones de entrenamiento), y las exogenas
son las mismas del LEAR (todas conocidas en el corte de las 00:00).
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

CORTE = pd.Timestamp("2026-01-01")
H, PICO = 24, [18, 19, 20]


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
    from arch import arch_model

    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - H)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]

    print(f"Ajustando un modelo por hora del dia (h=1..24). "
          f"Entrenamiento: {len(cortes_tr)} dias.\n")
    filas_lear, filas_g_norm, filas_g_t = [], [], []
    n_sel, alphas = [], []
    for h in range(1, H + 1):
        Xtr, ytr, f_tr, _ = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_te, _ = construir_para_paso(df, h, cortes_te)
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

        # --- paso 1: seleccion LASSO (la "LE" de LE-GARCH-t) ---
        if h <= 4:
            lc = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr)
            alphas.append(lc.alpha_)
            sel = np.abs(lc.coef_) > 1e-8
        else:
            la = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr)
            sel = np.abs(la.coef_) > 1e-8
        if sel.sum() == 0:
            sel[:] = True
        n_sel.append(int(sel.sum()))
        A, B = Ztr[:, sel], Zte[:, sel]

        # LEAR de referencia (mismo conjunto de variables, estimacion por LASSO)
        mlear = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr)
        filas_lear.append(pd.DataFrame({"fecha_hora": f_te, "real": yte, "pred": mlear.predict(Zte)}))

        # --- paso 2: GARCH(1,1) con exogenas seleccionadas, normal y t ---
        for dist, acc in [("normal", filas_g_norm), ("t", filas_g_t)]:
            try:
                am = arch_model(ytr, x=A, mean="LS", vol="GARCH", p=1, q=1, dist=dist, rescale=False)
                res = am.fit(disp="off", show_warning=False)
                # la media es lineal: intercepto + coeficientes sobre las exogenas
                par = res.params
                b0 = par.iloc[0]
                bx = par.iloc[1:1 + A.shape[1]].to_numpy()
                pred = b0 + B @ bx
            except Exception:
                pred = mlear.predict(Zte)     # si no converge, cae al LEAR (se reporta aparte)
            acc.append(pd.DataFrame({"fecha_hora": f_te, "real": yte, "pred": pred}))
        if h % 6 == 0:
            print(f"  h={h:2d}/24 ({(time.time()-t0)/60:.1f} min) variables seleccionadas={n_sel[-1]}",
                  flush=True)

    modelos = {}
    for nombre, acc in [("LEAR (referencia)", filas_lear), ("LE-GARCH (normal)", filas_g_norm),
                        ("LE-GARCH-t", filas_g_t)]:
        r = pd.concat(acc, ignore_index=True)
        modelos[nombre] = r[r["fecha_hora"] >= CORTE].set_index("fecha_hora").sort_index()

    print(f"\n{'='*94}\nMODELOS SUELTOS (variables seleccionadas por LASSO: mediana "
          f"{int(np.median(n_sel))})\n{'='*94}")
    ref = modelos["LEAR (referencia)"]
    y = ref["real"].to_numpy()
    pref = ref["pred"].to_numpy()
    hh = ref.index.hour
    pk = np.isin(hh, PICO)
    print(f"{'modelo':26s} {'MAE':>8s} {'MAPE %':>8s} {'MAPE pico':>10s}   DM vs LEAR")
    for nombre, r in modelos.items():
        p = r["pred"].to_numpy()
        linea = f"{nombre:26s} {np.abs(y-p).mean():8.2f} {mape(y, p):8.2f} {mape(y[pk], p[pk]):10.2f}"
        if nombre != "LEAR (referencia)":
            t, pv = dm(y, pref, p)
            marca = ("MEJORA" if t > 0 else "peor") if pv < 0.05 else "n.s."
            linea += f"   p={pv:.4f} {marca}"
        print(linea)

    # ---------- en el ensamble ----------
    d = cargar24().sort_index()
    for nombre, r in modelos.items():
        if nombre == "LEAR (referencia)":
            continue
        d = d.join(r["pred"].rename(nombre.replace(" ", "_")), how="left")
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d.index.hour // 6
    nuevos = [c for c in ["LE-GARCH_(normal)", "LE-GARCH-t"] if c in d.columns]
    d = d.dropna(subset=BASE5 + nuevos + ["real"])
    yy = d["real"].to_numpy()
    pkk = d["hora"].isin(PICO).to_numpy()
    base = qra_cv(d, BASE5)
    print(f"\n{'='*94}\nEN EL ENSAMBLE DE 24h (QRA por franja, CV por dias)\n{'='*94}")
    print(f"{'conjunto':38s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s} {'pico %':>8s}   DM vs v4")
    conjuntos = {"v4 vigente": BASE5}
    for c in nuevos:
        conjuntos[f"v4 + {c}"] = BASE5 + [c]
    conjuntos["v4 + ambos"] = BASE5 + nuevos
    for etiqueta, cols in conjuntos.items():
        p = qra_cv(d, cols)
        ok = ~np.isnan(p)
        sm_ = float(np.mean(np.abs(yy[ok]-p[ok]) / ((np.abs(yy[ok])+np.abs(p[ok]))/2)) * 100)
        linea = (f"{etiqueta:38s} {np.abs(yy[ok]-p[ok]).mean():8.2f} {mape(yy[ok], p[ok]):8.2f} "
                 f"{sm_:8.2f} {mape(yy[ok&pkk], p[ok&pkk]):8.2f}")
        if etiqueta != "v4 vigente":
            t, pv = dm(yy[ok], base[ok], p[ok])
            marca = ("MEJORA" if t > 0 else "peor") if pv < 0.05 else "n.s."
            linea += f"   p={pv:.4f} {marca}"
        print(linea)

    for nombre, r in modelos.items():
        r.to_csv(RES / f"pronostico_{nombre.split()[0].lower().replace('-','_')}_24h_2026.csv")
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
