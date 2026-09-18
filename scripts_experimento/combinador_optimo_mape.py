# -*- coding: utf-8 -*-
"""
Alinear el COMBINADOR con la metrica que de verdad se reporta: MAPE (y sMAPE).

El desajuste, que lleva todo el proyecto sin corregirse: el ensamble ajusta sus pesos minimizando
la DESVIACION ABSOLUTA (LAD/QRA al cuantil 0.5), que es optima para el MAE. Pero el proyecto se
evalua y se compara con la literatura en MAPE y sMAPE, y el minimizador de esas metricas NO es el
mismo. Formalmente:

    MAE  optimo -> min sum |y - Xw|                       (mediana condicional)
    MAPE optimo -> min sum |y - Xw| / y                   (mediana PONDERADA por 1/y)
    sMAPE       -> min sum |y - Xw| / ((|y|+|Xw|)/2)      (no lineal; se aproxima iterando)

O sea: basta ponderar cada observacion por 1/y al ajustar los pesos para obtener la combinacion
optima en MAPE en vez de en MAE. Es un cambio de una linea en el programa lineal, exacto (no
aproximado), y nunca se ha probado. Consecuencia esperada: al ponderar por 1/y, las horas de precio
BAJO pesan mas en el ajuste -- que es justo donde el MAPE castiga mas y donde el proyecto tiene
documentado que el modelo sobreestima.

Para sMAPE no hay forma cerrada porque el denominador depende de la propia prediccion, asi que se
usa el esquema iterativo estandar: ajustar con pesos 2/(|y|+|pred_anterior|) y repetir hasta que
los pesos se estabilicen.

Se prueba sobre el mejor ensamble actual (v4 + GARCH-ged, MAPE 10.86% / sMAPE 10.08%) y se mide el
compromiso: cuanto MAE se sacrifica por cuanto MAPE se gana.
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
from stacking_24h_v2 import cargar as cargar24, BASE5
from o6_comun import RES

PICO = [18, 19, 20]
SEMILLAS = list(range(10))


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def smape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / ((np.abs(real) + np.abs(pred)) / 2)) * 100)


def dm_metrica(real, p_ref, p_nuevo, metrica, maxlags=24):
    """DM sobre la perdida de la metrica indicada, no sobre el error absoluto."""
    real = np.asarray(real, float)
    if metrica == "mape":
        l1 = np.abs(real - p_ref) / real
        l2 = np.abs(real - p_nuevo) / real
    elif metrica == "smape":
        l1 = np.abs(real - p_ref) / ((np.abs(real) + np.abs(p_ref)) / 2)
        l2 = np.abs(real - p_nuevo) / ((np.abs(real) + np.abs(p_nuevo)) / 2)
    else:
        l1, l2 = np.abs(real - p_ref), np.abs(real - p_nuevo)
    d = l1 - l2
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def qra_pesos(X, y, w=None, tau=0.5):
    """QRA con intercepto, pesos de modelo >= 0, y PONDERACION por observacion (w)."""
    n, k = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    c = np.concatenate([np.zeros(k + 2), w * tau, w * (1 - tau)])
    A = sparse.hstack([sparse.csr_matrix(X), sparse.csr_matrix(np.ones((n, 1))),
                       sparse.csr_matrix(-np.ones((n, 1))), sparse.eye(n), -sparse.eye(n)]).tocsr()
    r = linprog(c, A_eq=A, b_eq=y, bounds=(0, None), method="highs")
    if not r.success:
        return np.full(k, 1 / k), 0.0
    return r.x[:k], r.x[k] - r.x[k + 1]


def qra_cv(d, cols, modo="mae", semilla=42, iteraciones=3):
    """modo: 'mae' (pesos iguales), 'mape' (w=1/y), 'smape' (iterativo)."""
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d["g4"].to_numpy(), d["dia"].to_numpy()
    P = np.full(len(d), np.nan)
    pliegues = np.array_split(np.random.default_rng(semilla).permutation(np.sort(np.unique(dia))), 5)
    for dt in pliegues:
        te_m = np.isin(dia, dt)
        for gg in np.unique(g[te_m]):
            tr, te = (~te_m) & (g == gg), te_m & (g == gg)
            if tr.sum() < len(cols) + 5:
                continue
            if modo == "mae":
                w = None
            elif modo == "mape":
                w = 1.0 / np.maximum(y[tr], 1e-6)
            else:                                    # smape: iterar
                w = 1.0 / np.maximum(y[tr], 1e-6)
            ww, b = qra_pesos(X[tr], y[tr], w)
            if modo == "smape":
                for _ in range(iteraciones):
                    pred_tr = X[tr] @ ww + b
                    w = 2.0 / np.maximum(np.abs(y[tr]) + np.abs(pred_tr), 1e-6)
                    ww, b = qra_pesos(X[tr], y[tr], w)
            P[te] = X[te] @ ww + b
    return P


def main():
    t0 = time.time()
    d = cargar24().sort_index()
    for arch, nom in [("pronostico_GARCH-ged_24h_2026.csv", "GARCH-ged"),
                      ("pronostico_GARCH-t_24h_2026.csv", "GARCH-t")]:
        ruta = RES / arch
        if ruta.exists():
            s = pd.read_csv(ruta, parse_dates=["fecha_hora"]).set_index("fecha_hora")
            col = "pred" if "pred" in s.columns else s.columns[-1]
            d = d.join(s[col].rename(nom), how="left")
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d.index.hour // 6
    extra = [c for c in ["GARCH-ged", "GARCH-t"] if c in d.columns]
    d = d.dropna(subset=BASE5 + extra + ["real"])
    y = d["real"].to_numpy()
    pk = d["hora"].isin(PICO).to_numpy()
    MEJOR = BASE5 + (["GARCH-ged"] if "GARCH-ged" in d.columns else [])
    print(f"Filas: {len(d):,} | votantes: {MEJOR}\n")

    print(f"{'='*100}\nCOMBINADOR ALINEADO CON LA METRICA (mismo conjunto de votantes)\n{'='*100}")
    print(f"{'objetivo del combinador':28s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s} {'pico %':>8s}"
          f"   DM(MAPE) | DM(sMAPE)")
    preds = {}
    for modo, etiqueta in [("mae", "MAE (actual)"), ("mape", "MAPE (pesos 1/y)"),
                           ("smape", "sMAPE (iterativo)")]:
        p = qra_cv(d, MEJOR, modo)
        preds[modo] = p
        linea = (f"{etiqueta:28s} {np.abs(y-p).mean():8.2f} {mape(y,p):8.2f} {smape(y,p):8.2f} "
                 f"{mape(y[pk],p[pk]):8.2f}")
        if modo != "mae":
            _, p1 = dm_metrica(y, preds["mae"], p, "mape")
            _, p2 = dm_metrica(y, preds["mae"], p, "smape")
            linea += f"   p={p1:.4f} | p={p2:.4f}"
        print(linea)

    # compromiso explicito
    print(f"\n  Compromiso al pasar de MAE a MAPE como objetivo: "
          f"MAE {np.abs(y-preds['mae']).mean():.2f} -> {np.abs(y-preds['mape']).mean():.2f} "
          f"({np.abs(y-preds['mape']).mean()-np.abs(y-preds['mae']).mean():+.2f}), "
          f"MAPE {mape(y,preds['mae']):.2f}% -> {mape(y,preds['mape']):.2f}% "
          f"({mape(y,preds['mape'])-mape(y,preds['mae']):+.2f})")

    # ---------- verificacion multi-semilla del mejor en MAPE ----------
    mejor_modo = min(["mape", "smape"], key=lambda m: mape(y, preds[m]))
    print(f"\n{'='*100}\nVERIFICACION MULTI-SEMILLA (objetivo '{mejor_modo}' vs 'mae')\n{'='*100}")
    m_mae = np.array([mape(y, qra_cv(d, MEJOR, "mae", s)) for s in SEMILLAS])
    m_new = np.array([mape(y, qra_cv(d, MEJOR, mejor_modo, s)) for s in SEMILLAS])
    delta = m_new - m_mae
    print(f"  MAPE con objetivo MAE : {m_mae.mean():.3f} +/- {m_mae.std():.3f}")
    print(f"  MAPE con objetivo {mejor_modo:5s}: {m_new.mean():.3f} +/- {m_new.std():.3f}")
    print(f"  delta: {delta.mean():+.3f} +/- {delta.std():.3f}, negativo en "
          f"{int((delta<0).sum())}/{len(SEMILLAS)} semillas")
    solido = (delta < 0).all() and abs(delta.mean()) > 2 * m_mae.std()
    print(f"  VEREDICTO: {'SOLIDO' if solido else 'NO concluyente'}")

    # ---------- tambien sobre el v4 original, para separar el efecto ----------
    print(f"\n{'='*100}\nEL MISMO CAMBIO SOBRE EL v4 ORIGINAL (sin GARCH), para aislar el efecto\n{'='*100}")
    for modo, etiqueta in [("mae", "v4, objetivo MAE"), ("mape", "v4, objetivo MAPE")]:
        p = qra_cv(d, BASE5, modo)
        print(f"  {etiqueta:26s} MAE={np.abs(y-p).mean():7.2f}  MAPE={mape(y,p):6.2f}%  "
              f"sMAPE={smape(y,p):6.2f}%")

    pd.DataFrame({"fecha_hora": d.index, "real": y,
                  **{f"pred_{m}": p for m, p in preds.items()}}).to_csv(
        RES / "combinador_optimo_mape.csv", index=False)
    print(f"\nGuardado: combinador_optimo_mape.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
