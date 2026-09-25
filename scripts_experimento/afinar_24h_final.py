# -*- coding: utf-8 -*-
"""
Dos ideas mas para el ensamble de 24h, ambas derivadas de lo unico que ha funcionado.

Contexto: LE-GARCH-t bajo el ensamble a MAPE 10.89% / sMAPE 10.09%, y el barrido de la familia
GARCH (familia_garch_24h.py) mostro que ahi ya no hay mas: GED empata con t, skew-t empeora, GJR y
EGARCH no aportan, la aproximacion SV-t tampoco. El mejor punto alcanzado es MAPE 10.87% /
sMAPE 10.06%. El mecanismo que produjo la ganancia fue ESTIMAR LA MEDIA PONDERANDO POR LA VARIANZA
CONDICIONAL, es decir descontar los dias volatiles al ajustar los coeficientes.

  IDEA A -- APLICAR ESE MISMO MECANISMO AL COMBINADOR. Los pesos del ensamble (QRA por franja) se
  ajustan hoy dando el mismo peso a todos los dias. Pero ya existe un predictor de riesgo por hora
  (riesgo_24h.csv, AUC 0.818 para anticipar el 20% peor). Nunca se ha usado para PONDERAR EL AJUSTE
  DE LOS PESOS -- solo se intento enrutar (fallo) y modular bandas (fallo). Ponderar el ajuste es
  distinto y es el analogo exacto de lo que hace GARCH-t: estimar los pesos descontando las horas
  que se sabe ruidosas, para que no distorsionen la mezcla que se aplicara a las horas normales.

  IDEA B -- PRECIOS DE CONTRATOS EN LA ECUACION DE MEDIA. Recien descargados de XM
  (precio_contratos_regulado / no_regulado, horarios 2019-2026). Son el equivalente colombiano de
  los FORWARD PRICES que en el paper de Nueva Zelanda eran de las variables mas usadas por los
  modelos ganadores (60 de sus 423 features). Reflejan la expectativa del mercado sobre el precio
  futuro: informacion que el precio de bolsa rezagado no contiene por construccion. Se agregan a la
  ecuacion de media del LE-GARCH-t, medidas EN EL CORTE.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy.optimize import linprog
from scipy import sparse

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES
from lear_24h import construir_para_paso
from stacking_24h_v2 import cargar as cargar24, BASE5

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
CORTE = pd.Timestamp("2026-01-01")
H, PICO = 24, [18, 19, 20]


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def smape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / ((np.abs(real) + np.abs(pred)) / 2)) * 100)


def qra_pesos(X, y, tau=0.5, w=None):
    """QRA (cuantil tau) con intercepto y pesos >= 0, opcionalmente PONDERANDO cada observacion."""
    n, k = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    c = np.concatenate([np.zeros(k + 2), w * tau, w * (1 - tau)])
    A = sparse.hstack([sparse.csr_matrix(X), sparse.csr_matrix(np.ones((n, 1))),
                       sparse.csr_matrix(-np.ones((n, 1))), sparse.eye(n), -sparse.eye(n)]).tocsr()
    r = linprog(c, A_eq=A, b_eq=y, bounds=(0, None), method="highs")
    if not r.success:
        return np.full(k, 1 / k), 0.0
    return r.x[:k], r.x[k] - r.x[k + 1]


def qra_cv(d, cols, peso=None, semilla=42):
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d["g4"].to_numpy(), d["dia"].to_numpy()
    P = np.full(len(d), np.nan)
    for dt in np.array_split(np.random.default_rng(semilla).permutation(np.sort(np.unique(dia))), 5):
        te_m = np.isin(dia, dt)
        for gg in np.unique(g[te_m]):
            tr, te = (~te_m) & (g == gg), te_m & (g == gg)
            if tr.sum() >= len(cols) + 5:
                w = None if peso is None else peso[tr]
                ww, b = qra_pesos(X[tr], y[tr], 0.5, w)
                P[te] = X[te] @ ww + b
    return P


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    from arch import arch_model

    # ---------------- IDEA B: LE-GARCH-t con precios de contratos ----------------
    df = cargar_completo().reset_index(drop=True)
    idx = pd.DatetimeIndex(df["fecha_hora"])
    contratos = {}
    for n in ["precio_contratos_regulado", "precio_contratos_no_regulado"]:
        s = pd.read_csv(RAIZ / f"data/{n}_2019_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
        contratos[n] = s.iloc[:, 0].reindex(idx).to_numpy(np.float64)
    precio = df["precio_bolsa"].to_numpy(np.float64)
    # prima de bolsa sobre contratos: cuanto se aparta el spot de la expectativa contractual
    contratos["prima_sobre_contrato"] = precio / np.maximum(contratos["precio_contratos_regulado"], 1)
    print("Correlacion parcial de los contratos con el precio (descontando precio_lag24h):")
    plag = pd.Series(precio).shift(24).to_numpy()
    for n, v in contratos.items():
        ok = ~(np.isnan(v) | np.isnan(precio) | np.isnan(plag))
        A = np.c_[np.ones(ok.sum()), plag[ok]]
        rp = precio[ok] - A @ np.linalg.lstsq(A, precio[ok], rcond=None)[0]
        rv = v[ok] - A @ np.linalg.lstsq(A, v[ok], rcond=None)[0]
        print(f"  {n:28s} {np.corrcoef(rp, rv)[0,1]:+.4f}")

    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - H)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]
    pos = pd.Series(np.arange(len(df)), index=fechas)

    acc = {"LG-t base": [], "LG-t + contratos": []}
    alphas = []
    for h in range(1, H + 1):
        Xtr, ytr, f_tr, c_tr = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_te, c_te = construir_para_paso(df, h, cortes_te)
        ictr = pos.reindex(pd.DatetimeIndex(c_tr)).to_numpy()
        icte = pos.reindex(pd.DatetimeIndex(c_te)).to_numpy()
        extra_tr = pd.DataFrame({f"{n}_corte": v[ictr] for n, v in contratos.items()})
        extra_te = pd.DataFrame({f"{n}_corte": v[icte] for n, v in contratos.items()})

        for etiqueta, (P, Q) in {
            "LG-t base": (Xtr, Xte),
            "LG-t + contratos": (pd.concat([Xtr.reset_index(drop=True), extra_tr], axis=1),
                                 pd.concat([Xte.reset_index(drop=True), extra_te], axis=1)),
        }.items():
            P2, Q2 = P.fillna(P.median()), Q.fillna(P.median())
            sc = StandardScaler().fit(P2)
            Ztr, Zte = sc.transform(P2), sc.transform(Q2)
            if h <= 4 and etiqueta == "LG-t base":
                lc = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr)
                alphas.append(lc.alpha_)
            la = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr)
            sel = np.abs(la.coef_) > 1e-8
            if sel.sum() == 0:
                sel[:] = True
            A, B = Ztr[:, sel], Zte[:, sel]
            try:
                res = arch_model(ytr, x=A, mean="LS", vol="GARCH", p=1, q=1, dist="t",
                                 rescale=False).fit(disp="off", show_warning=False)
                par = res.params
                pred = par.iloc[0] + B @ par.iloc[1:1 + A.shape[1]].to_numpy()
                if not np.isfinite(pred).all():
                    raise ValueError
            except Exception:
                pred = la.predict(Zte)
            acc[etiqueta].append(pd.DataFrame({"fecha_hora": f_te, "real": yte, "pred": pred}))

    modelos = {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}
    modelos = {k: v[v["fecha_hora"] >= CORTE].set_index("fecha_hora").sort_index()
               for k, v in modelos.items()}
    print(f"\n{'='*92}\nIDEA B: LE-GARCH-t con precios de contratos (modelo suelto)\n{'='*92}")
    y0 = modelos["LG-t base"]["real"].to_numpy()
    for k, r in modelos.items():
        p = r["pred"].to_numpy()
        print(f"  {k:20s} MAE={np.abs(y0-p).mean():7.2f}  MAPE={mape(y0, p):6.2f}%  sMAPE={smape(y0, p):6.2f}%")

    # ---------------- ensamble con ambas ideas ----------------
    d = cargar24().sort_index()
    for k, r in modelos.items():
        d = d.join(r["pred"].rename(k), how="left")
    ged = RES / "pronostico_GARCH-ged_24h_2026.csv"
    if ged.exists():
        d = d.join(pd.read_csv(ged, parse_dates=["fecha_hora"]).set_index("fecha_hora")["pred"]
                   .rename("GARCH-ged"), how="left")
    riesgo = pd.read_csv(RES / "riesgo_24h.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(riesgo["riesgo_predicho"], how="left")
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d.index.hour // 6
    votantes_extra = [c for c in ["LG-t base", "LG-t + contratos", "GARCH-ged"] if c in d.columns]
    d = d.dropna(subset=BASE5 + votantes_extra + ["real"])
    y = d["real"].to_numpy()
    pk = d["hora"].isin(PICO).to_numpy()
    base = qra_cv(d, BASE5)
    mejor_actual = qra_cv(d, BASE5 + ["LG-t base"])
    print(f"\n{'='*92}\nENSAMBLE (QRA por franja, CV)\n{'='*92}")
    print(f"{'conjunto':44s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s}   DM vs v4 | vs v4+LG-t")

    def linea(et, p):
        _, p1 = dm(y, base, p)
        _, p2 = dm(y, mejor_actual, p)
        print(f"{et:44s} {np.abs(y-p).mean():8.2f} {mape(y, p):8.2f} {smape(y, p):8.2f}"
              f"   p={p1:.4f} | p={p2:.4f}")

    print(f"{'v4 vigente':44s} {np.abs(y-base).mean():8.2f} {mape(y, base):8.2f} {smape(y, base):8.2f}")
    linea("v4 + LG-t (mejor actual)", mejor_actual)
    for c in votantes_extra:
        if c != "LG-t base":
            linea(f"v4 + {c}", qra_cv(d, BASE5 + [c]))
    linea("v4 + LG-t + contratos", qra_cv(d, BASE5 + ["LG-t base", "LG-t + contratos"]))
    if "GARCH-ged" in d.columns:
        linea("v4 + LG-t + GED", qra_cv(d, BASE5 + ["LG-t base", "GARCH-ged"]))
        linea("v4 + LG-t + GED + contratos",
              qra_cv(d, BASE5 + ["LG-t base", "GARCH-ged", "LG-t + contratos"]))

    # ---------------- IDEA A: combinador ponderado por riesgo ----------------
    print(f"\n{'='*92}\nIDEA A: pesos del ensamble ajustados PONDERANDO por el riesgo predicho\n{'='*92}")
    r_ = d["riesgo_predicho"].to_numpy()
    r_ = np.where(np.isnan(r_), np.nanmedian(r_), r_)
    cols_mejor = BASE5 + ["LG-t base"]
    for expo in [0.5, 1.0, 2.0]:
        w = (np.median(r_) / np.maximum(r_, 1e-6)) ** expo    # menos peso a las horas ruidosas
        w = np.clip(w / w.mean(), 0.05, 20.0)
        p = qra_cv(d, cols_mejor, peso=w)
        linea(f"v4+LG-t, ponderado por riesgo^{expo}", p)

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
