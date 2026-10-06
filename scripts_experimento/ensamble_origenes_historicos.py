# -*- coding: utf-8 -*-
"""
Ensamble completo de 24 h en los origenes historicos 1 a 5 del walk-forward (lo que faltaba para cerrar OE2).

Hasta ahora el ensamble de 6 votantes solo se habia evaluado en 2026 (origen 6). En los origenes 1-5 ya
existen las predicciones de 5 votantes (walkforward_predicciones_crudas.csv: persistencia, XGBoost,
ARX+GARCH, N-BEATSx, N-HiTS), con ventanas de 00:00 a 23:00 y corte a las 23:00 del dia anterior.
Este script:
  1. genera el sexto votante, GARCH-ged, en cada origen con el mismo procedimiento de
     familia_garch_24h.py (LASSO por paso + GARCH(1,1) con errores GED sobre las variables elegidas),
     entrenado solo con datos anteriores al inicio del origen y con cortes a las 23:00 para quedar
     alineado hora a hora con los otros cinco;
  2. combina los 6 votantes con el combinador CAUSAL de la version desplegable (QRA por franja de 6 h,
     objetivo sMAPE, pesos solo con dias anteriores, 14 dias de calentamiento);
  3. compara con la persistencia y con el mejor votante individual (Diebold-Mariano, varianza HAC);
  4. guarda las predicciones con una banda empirica [q10, q90] (cuantiles de los residuos de los 30 dias
     previos, por hora del dia) para el backtest historico del motor.
Salidas: data/processed/resultados/ensamble_origenes_historicos.csv (metricas) y
         ensamble_origenes_historicos_predicciones.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES  # noqa: E402
from lear_24h import construir_para_paso  # noqa: E402
from franja_medianoche_24h import causal  # noqa: E402

H = 24
VOTANTES = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS", "GARCH-ged"]
ORIGENES = ["Origen 1", "Origen 2", "Origen 3", "Origen 4", "Origen 5"]


def garch_ged(df, inicio, fin):
    """Sexto votante para la ventana [inicio, fin], entrenado solo con lo anterior a inicio."""
    from arch import arch_model
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler
    fechas = df["fecha_hora"]
    idx_23 = np.where(fechas.dt.hour.to_numpy() == 23)[0]
    i_ini = int(np.searchsorted(fechas.to_numpy(), np.datetime64(inicio)))
    i_fin = int(np.searchsorted(fechas.to_numpy(), np.datetime64(fin), side="right")) - 1
    cortes_tr = idx_23[(idx_23 >= 200) & (idx_23 + H < i_ini)]
    cortes_te = idx_23[(idx_23 + 1 >= i_ini) & (idx_23 + H <= i_fin)]
    salida, alphas, fallos = [], [], 0
    for h in range(1, H + 1):
        Xtr, ytr, _, _ = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_te, _ = construir_para_paso(df, h, cortes_te)
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        if h <= 4:
            lc = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr)
            alphas.append(lc.alpha_)
            sel = np.abs(lc.coef_) > 1e-8
        else:
            sel = np.abs(Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42)
                         .fit(Ztr, ytr).coef_) > 1e-8
        if sel.sum() == 0:
            sel[:] = True
        A, B = Ztr[:, sel], Zte[:, sel]
        try:
            res = arch_model(ytr, x=A, mean="LS", rescale=False, vol="GARCH", p=1, q=1,
                             dist="ged").fit(disp="off", show_warning=False)
            par = res.params
            pred = par.iloc[0] + B @ par.iloc[1:1 + A.shape[1]].to_numpy()
            if not np.isfinite(pred).all():
                raise ValueError
        except Exception:
            pred = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr).predict(Zte)
            fallos += 1
        salida.append(pd.DataFrame({"fecha_hora": f_te, "GARCH-ged": pred}))
    return pd.concat(salida).set_index("fecha_hora").sort_index()["GARCH-ged"], fallos


def dm(y, p_ref, p_mod, maxlags=24):
    d = np.abs(y - p_ref) - np.abs(y - p_mod)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    t = float(r.tvalues[0])
    return t, float(r.pvalues[0]), float(1 - __import__("scipy").stats.norm.cdf(t))


def banda_empirica(d, ventana_dias=30, nivel=(0.10, 0.90)):
    """q10/q90 = prediccion + cuantiles de los residuos de los 30 dias previos en la misma hora (causal)."""
    r = d["real"] - d["ensamble"]
    q10, q90 = np.full(len(d), np.nan), np.full(len(d), np.nan)
    dias = d["dia"].to_numpy()
    horas = d.index.hour.to_numpy()
    for dia in np.unique(dias):
        hist = (dias < dia) & (dias >= dia - np.timedelta64(ventana_dias, "D"))
        hoy = dias == dia
        for hh in np.unique(horas[hoy]):
            rr = r.to_numpy()[hist & (horas == hh)]
            rr = rr[~np.isnan(rr)]
            if len(rr) >= 10:
                m = hoy & (horas == hh)
                q10[m] = d["ensamble"].to_numpy()[m] + np.quantile(rr, nivel[0])
                q90[m] = d["ensamble"].to_numpy()[m] + np.quantile(rr, nivel[1])
    return q10, q90


if __name__ == "__main__":
    t0 = time.time()
    df = cargar_completo().reset_index(drop=True)
    crudas = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    filas, preds = [], []
    for o in ORIGENES:
        c = crudas[crudas["origen"] == o]
        d = c.pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
        d = d.join(c.groupby("fecha_hora")["real"].first())
        g, fallos = garch_ged(df, d.index.min(), d.index.max())
        d = d.join(g, how="left").dropna(subset=VOTANTES + ["real"])
        d["dia"] = d.index.normalize()
        d["g4"] = d.index.hour // 6
        d["ensamble"] = causal(d, VOTANTES, "g4", "smape")
        d = d.dropna(subset=["ensamble"])
        y = d["real"].to_numpy()
        per = d["Persistencia"].to_numpy()
        mae = {m: float(np.abs(y - d[m].to_numpy()).mean()) for m in VOTANTES + ["ensamble"]}
        mejor_ind = min(VOTANTES[1:], key=lambda m: mae[m])
        t_p, p2_p, p1_p = dm(y, per, d["ensamble"].to_numpy())
        t_b, p2_b, _ = dm(y, d[mejor_ind].to_numpy(), d["ensamble"].to_numpy())
        q10, q90 = banda_empirica(d)
        ok = ~np.isnan(q10)
        cob = float(((y[ok] >= q10[ok]) & (y[ok] <= q90[ok])).mean() * 100)
        fila = {"origen": o, "desde": d.index.min(), "hasta": d.index.max(), "horas": len(d),
                "MAE_ensamble": mae["ensamble"], "MAE_persistencia": mae["Persistencia"],
                "rMAE": mae["ensamble"] / mae["Persistencia"],
                "MAPE_ensamble": float(np.mean(np.abs(y - d["ensamble"]) / y) * 100),
                "MAPE_persistencia": float(np.mean(np.abs(y - per) / y) * 100),
                "DM_t_vs_persistencia": t_p, "p_unilateral": p1_p, "p_bilateral": p2_p,
                "mejor_individual": mejor_ind, "MAE_mejor_individual": mae[mejor_ind],
                "DM_t_vs_mejor_individual": t_b, "p_bilateral_vs_mejor": p2_b,
                "MAE_GARCH-ged": mae["GARCH-ged"], "garch_fallos_de_24": fallos,
                "cobertura_banda_empirica": cob}
        filas.append(fila)
        preds.append(pd.DataFrame({"origen": o, "fecha_hora": d.index, "real": y,
                                   "q50": d["ensamble"].to_numpy(), "q10": q10, "q90": q90}))
        print(f"{o}: ensamble MAE {mae['ensamble']:.2f} vs persistencia {mae['Persistencia']:.2f} "
              f"(rMAE {fila['rMAE']:.3f}, DM t={t_p:.2f}, p1={p1_p:.4f}) | mejor individual {mejor_ind} "
              f"{mae[mejor_ind]:.2f} (DM t={t_b:.2f}) | GARCH-ged {mae['GARCH-ged']:.2f} | "
              f"cobertura banda {cob:.1f}% | {(time.time() - t0) / 60:.1f} min", flush=True)
    pd.DataFrame(filas).to_csv(RES / "ensamble_origenes_historicos.csv", index=False)
    pd.concat(preds).to_csv(RES / "ensamble_origenes_historicos_predicciones.csv", index=False)
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
