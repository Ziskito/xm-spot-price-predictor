# -*- coding: utf-8 -*-
"""
Punto 1 de la lista de alternativas: ensamble ENTRE familias de modelo (stacking), no solo
entre semillas/ventanas de la misma familia (ya probado con N-BEATSx).

Se usan las predicciones ya guardadas de los 5 modelos (walkforward_predicciones_crudas.csv,
con festivos, 6 origenes) como inputs de un meta-modelo. Dos variantes de meta-modelo:
  - Ridge: pesos libres (puede ser negativo), regularizado para evitar sobreajuste.
  - NNLS: combinacion convexa no-negativa (mas interpretable, evita pesos absurdos).

Split ALEATORIO ESTRATIFICADO por dia dentro de cada origen (calibracion/evaluacion), mismo
criterio que el resto de la sesion, para no dejar ningun regimen fuera de la calibracion.
Se compara contra el mejor modelo individual (N-BEATSx) con prueba de Diebold-Mariano.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
import statsmodels.api as sm

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
MODELOS = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]

df = pd.read_csv(RAIZ / "data/processed/resultados/walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])

def calcular_metricas(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    err = y_real - y_pred
    mae = np.abs(err).mean()
    mape = (np.abs(err) / y_real).mean() * 100
    return mae, mape

def prueba_dm(e1, e2, horizonte=24):
    d = np.asarray(e1) - np.asarray(e2)
    m = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": horizonte - 1})
    return m.tvalues[0], m.pvalues[0]

resultados = []
rng = np.random.default_rng(42)

for origen in sorted(df["origen"].unique()):
    datos_o = df[df["origen"] == origen]
    # --- Pivotear a formato ancho: una fila por hora, una columna por modelo ---
    pivot = datos_o.pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
    real = datos_o.groupby("fecha_hora")["real"].first()
    pivot = pivot.join(real).dropna()
    if len(pivot) < 200 or not all(m in pivot.columns for m in MODELOS):
        continue

    pivot["dia"] = pivot.index.normalize()
    dias = pivot["dia"].unique()
    dias_calib = set(rng.choice(dias, size=len(dias) // 2, replace=False))
    calib = pivot[pivot["dia"].isin(dias_calib)]
    evalua = pivot[~pivot["dia"].isin(dias_calib)]

    X_calib, y_calib = calib[MODELOS].values, calib["real"].values
    X_eval, y_eval = evalua[MODELOS].values, evalua["real"].values

    # --- Meta-modelo 1: Ridge (pesos libres, regularizado) ---
    ridge = Ridge(alpha=10.0, fit_intercept=True)
    ridge.fit(X_calib, y_calib)
    pred_ridge = ridge.predict(X_eval)

    # --- Meta-modelo 2: NNLS (combinacion convexa, sin intercepto, pesos >= 0) ---
    pesos_nnls, _ = nnls(X_calib, y_calib)
    pesos_nnls_norm = pesos_nnls / pesos_nnls.sum() if pesos_nnls.sum() > 0 else pesos_nnls
    pred_nnls = X_eval @ pesos_nnls_norm

    # --- Mejor individual (por MAE en evaluacion, para saber contra quien comparar) ---
    mae_individuales = {m: calcular_metricas(y_eval, evalua[m].values)[0] for m in MODELOS}
    mejor_individual = min(mae_individuales, key=mae_individuales.get)
    pred_mejor_individual = evalua[mejor_individual].values

    mae_ridge, mape_ridge = calcular_metricas(y_eval, pred_ridge)
    mae_nnls, mape_nnls = calcular_metricas(y_eval, pred_nnls)
    mae_mejor, mape_mejor = calcular_metricas(y_eval, pred_mejor_individual)

    dm_ridge, p_ridge = prueba_dm(np.abs(y_eval - pred_ridge), np.abs(y_eval - pred_mejor_individual))
    dm_nnls, p_nnls = prueba_dm(np.abs(y_eval - pred_nnls), np.abs(y_eval - pred_mejor_individual))

    concl_ridge = "ENSAMBLE GANA (sig.)" if (p_ridge < 0.05 and dm_ridge < 0) else ("ENSAMBLE PIERDE (sig.)" if (p_ridge < 0.05 and dm_ridge > 0) else "sin diferencia")
    concl_nnls = "ENSAMBLE GANA (sig.)" if (p_nnls < 0.05 and dm_nnls < 0) else ("ENSAMBLE PIERDE (sig.)" if (p_nnls < 0.05 and dm_nnls > 0) else "sin diferencia")

    print(f"\n--- {origen} --- (mejor individual: {mejor_individual}, MAE={mae_mejor:.2f})")
    print(f"  Pesos NNLS: " + ", ".join(f"{m}={w:.3f}" for m, w in zip(MODELOS, pesos_nnls_norm)))
    print(f"  Pesos Ridge (coef): " + ", ".join(f"{m}={w:.3f}" for m, w in zip(MODELOS, ridge.coef_)) + f"  intercepto={ridge.intercept_:.2f}")
    print(f"  Ridge -> MAE {mae_ridge:.2f} MAPE {mape_ridge:.2f}%  DM t={dm_ridge:.3f} p={p_ridge:.4f}  {concl_ridge}")
    print(f"  NNLS  -> MAE {mae_nnls:.2f} MAPE {mape_nnls:.2f}%  DM t={dm_nnls:.3f} p={p_nnls:.4f}  {concl_nnls}")

    resultados.append({
        "origen": origen, "mejor_individual": mejor_individual, "mae_mejor_individual": mae_mejor,
        "mape_mejor_individual": mape_mejor, "mae_ridge": mae_ridge, "mape_ridge": mape_ridge,
        "p_ridge": p_ridge, "concl_ridge": concl_ridge,
        "mae_nnls": mae_nnls, "mape_nnls": mape_nnls, "p_nnls": p_nnls, "concl_nnls": concl_nnls,
    })

df_res = pd.DataFrame(resultados)
df_res.to_csv(RAIZ / "data" / "processed" / "resultados" / "stacking_ensamble.csv", index=False)

print("\n" + "=" * 100)
print("RESUMEN")
print("=" * 100)
print(df_res[["origen", "mejor_individual", "mae_mejor_individual", "mae_ridge", "concl_ridge", "mae_nnls", "concl_nnls"]].to_string(index=False))
print("\nGuardado: stacking_ensamble.csv")
