# -*- coding: utf-8 -*-
"""
Llevar al modelo de 72h las dos mejoras que funcionaron en 24h.

Las dos ideas que ganaron esta noche en 24h, ambas verificadas con 10/10 semillas:
  (A) COLAS PESADAS: un votante GARCH-t/GED (la media estimada por maxima verosimilitud ponderando
      por la varianza condicional, lo que descuenta los dias volatiles). En 24h bajo el MAPE de
      11.27% a 10.86%.
  (B) COMBINADOR ALINEADO CON LA METRICA: ajustar los pesos minimizando la perdida reponderada
      (sMAPE iterativo) en vez de la desviacion absoluta. En 24h bajo el MAPE a 10.74% y ademas el
      MAE, por un efecto de regularizacion del meta-modelo que se comprobo (pesos mas estables).

Ninguna de las dos se ha probado en 72h, cuyo ensamble vigente es LEAR + 2 redes + naive con
combinador LAD por tramo. Hay motivos para esperar que ayuden mas aun: a 48-72h el error es mayor y
mas disperso, que es justo donde las colas pesadas y la reponderacion importan.

Para (A) se construye un GARCH-t por paso de horizonte (72 modelos) sobre la formulacion directa,
igual que el LEAR de 72h. Para (B) se reemplaza el combinador LAD por su version reponderada.
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
from lear_72h import construir_para_paso as construir_72
from ensamble_72h_diario import TRAMOS, hac_p

CORTE = pd.Timestamp("2026-01-01")
PRIMER_CORTE = pd.Timestamp("2026-01-01 00:00")
N_WIN, H = 214, 72
NUEVO_72 = ["LEAR", "NBX", "NBX_exog", "NaiveEstac"]     # el mejor conjunto hallado hoy


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def smape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / ((np.abs(real) + np.abs(pred)) / 2)) * 100)


def pesos_lad_pond(X, y, w=None):
    """min sum w_i |y_i - Xw|  s.a. pesos >= 0 y suman 1 (mismo LAD del proyecto, ponderado)."""
    n, k = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    c = np.concatenate([np.zeros(k), w, w])
    A_eq = sparse.vstack([
        sparse.hstack([sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)]),
        sparse.hstack([sparse.csr_matrix(np.ones((1, k))), sparse.csr_matrix((1, 2 * n))]),
    ]).tocsr()
    r = linprog(c, A_eq=A_eq, b_eq=np.concatenate([y, [1.0]]), bounds=(0, None), method="highs")
    return r.x[:k] if r.success else np.full(k, 1 / k)


def lad_cv(sub, cols, modo="lad", semilla=42, iteraciones=3):
    X, y = sub[cols].to_numpy(), sub["real"].to_numpy()
    P = np.full(len(sub), np.nan)
    dias = np.sort(sub["dia"].unique())
    for dt in np.array_split(np.random.default_rng(semilla).permutation(dias), 5):
        te = sub["dia"].isin(dt).to_numpy()
        tr = ~te
        w = None if modo == "lad" else 1.0 / np.maximum(y[tr], 1e-6)
        ww = pesos_lad_pond(X[tr], y[tr], w)
        if modo == "smape":
            for _ in range(iteraciones):
                pr = X[tr] @ ww
                w = 2.0 / np.maximum(np.abs(y[tr]) + np.abs(pr), 1e-6)
                ww = pesos_lad_pond(X[tr], y[tr], w)
        P[te] = X[te] @ ww
    return P


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    from arch import arch_model

    # ---------- (A) GARCH-t por paso de horizonte ----------
    df = cargar_completo().reset_index(drop=True)
    pos = pd.Series(df.index.values, index=df["fecha_hora"])
    cortes_te = pos.reindex(pd.date_range(PRIMER_CORTE, periods=N_WIN, freq="24h")).astype(int).to_numpy()
    idx_fin = int(pos[CORTE])
    cortes_tr = np.arange(200, idx_fin - H)

    print(f"(A) Construyendo GARCH-t por paso de horizonte (72 modelos)...")
    partes, alphas, fallos = [], [], 0
    for h in range(1, H + 1):
        Xtr, ytr, _, _ = construir_72(df, h, cortes_tr)
        Xte, yte, f_obj, f_cor = construir_72(df, h, cortes_te)
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        if h <= 6 or h % 12 == 0:
            lc = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=8000).fit(Ztr, ytr)
            alphas.append(lc.alpha_)
        al = float(np.median(alphas))
        la = Lasso(alpha=al, max_iter=8000, random_state=42).fit(Ztr, ytr)
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
            pred, fallos = la.predict(Zte), fallos + 1
        partes.append(pd.DataFrame({"fecha_hora": f_obj, "cutoff": f_cor, "paso_horas": h,
                                    "GARCHt72": pred}))
        if h % 18 == 0:
            print(f"  paso {h:2d}/72 ({(time.time()-t0)/60:.1f} min)", flush=True)
    gar = pd.concat(partes, ignore_index=True)
    gar.to_csv(RES / "pronostico_garcht_72h_2026.csv", index=False)
    print(f"  no convergieron: {fallos}/72")

    # ---------- unir con los votantes ----------
    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    lear = pd.read_csv(RES / "pronostico_lear_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v = v.merge(lear[["fecha_hora", "cutoff", "paso_horas", "LEAR"]],
                on=["fecha_hora", "cutoff", "paso_horas"], how="inner")
    v = v.merge(gar, on=["fecha_hora", "cutoff", "paso_horas"], how="left")
    v["dia"] = v["fecha_hora"].dt.normalize()
    v = v.dropna(subset=NUEVO_72 + ["GARCHt72", "real"])
    print(f"\nFilas: {len(v):,}")

    print(f"\n{'='*104}\nGARCH-t SUELTO POR TRAMO (contra LEAR, el mejor individual a 72h)\n{'='*104}")
    print(f"{'tramo':10s} {'LEAR MAE':>10s} {'GARCHt MAE':>12s} {'LEAR MAPE':>11s} {'GARCHt MAPE':>13s}")
    for nom, ini, fin in TRAMOS:
        s = v[v.paso_horas.between(ini, fin)]
        print(f"{nom:10s} {np.abs(s['real']-s['LEAR']).mean():10.2f} "
              f"{np.abs(s['real']-s['GARCHt72']).mean():12.2f} "
              f"{mape(s['real'], s['LEAR']):11.2f} {mape(s['real'], s['GARCHt72']):13.2f}")

    # ---------- (A)+(B) en el ensamble ----------
    print(f"\n{'='*104}\nENSAMBLE DE 72h: (A) votante GARCH-t y (B) combinador alineado con la metrica\n{'='*104}")
    print(f"{'configuracion':44s} {'tramo':9s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s}   DM vs base")
    filas = []
    for nom, ini, fin in TRAMOS:
        sub = v[v.paso_horas.between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
        y = sub["real"].to_numpy()
        base = lad_cv(sub, NUEVO_72, "lad")
        for etiqueta, cols, modo in [
            ("nuevo 72h (4 votantes, LAD)", NUEVO_72, "lad"),
            ("+ GARCH-t (5 votantes, LAD)", NUEVO_72 + ["GARCHt72"], "lad"),
            ("4 votantes, combinador sMAPE", NUEVO_72, "smape"),
            ("+ GARCH-t y combinador sMAPE", NUEVO_72 + ["GARCHt72"], "smape"),
        ]:
            p = lad_cv(sub, cols, modo)
            ok = ~np.isnan(p)
            linea = (f"{etiqueta:44s} {nom:9s} {np.abs(y[ok]-p[ok]).mean():8.2f} "
                     f"{mape(y[ok], p[ok]):8.2f} {smape(y[ok], p[ok]):8.2f}")
            if etiqueta != "nuevo 72h (4 votantes, LAD)":
                _, pv = hac_p(np.abs(y[ok]-base[ok]) - np.abs(y[ok]-p[ok]))
                marca = "MEJORA" if pv < 0.05 and np.abs(y[ok]-p[ok]).mean() < np.abs(y[ok]-base[ok]).mean() else \
                        ("peor" if pv < 0.05 else "n.s.")
                linea += f"   p={pv:.4f} {marca}"
            print(linea)
            filas.append({"tramo": nom, "config": etiqueta, "MAE": np.abs(y[ok]-p[ok]).mean(),
                          "MAPE": mape(y[ok], p[ok]), "sMAPE": smape(y[ok], p[ok])})
        print()

    pd.DataFrame(filas).to_csv(RES / "mejoras_a_72h.csv", index=False)
    print(f"Guardado: mejoras_a_72h.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
