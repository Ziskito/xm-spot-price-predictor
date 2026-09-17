# -*- coding: utf-8 -*-
"""
Punto 4 de la lista original de alternativas: CatBoost, variante barata de probar. Ordered
boosting (en teoria reduce sobreajuste en series temporales mejor que XGBoost) y mejor manejo
nativo de variables categoricas/calendario -- aunque aqui las variables ya vienen numericas
(seno/coseno, dummies 0/1), asi que la ventaja principal a testear es el ordered boosting.

Mismas features y mismo holdout 2026 que el resto del proyecto, para comparacion directa.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

df_train = pd.read_csv(RAIZ / "data" / "processed" / "dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
df_test = pd.read_csv(RAIZ / "data" / "processed" / "dataset_features_2026.csv", parse_dates=["fecha_hora"])

columnas_excluir = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora", "dia_semana", "dia_anio"]
columnas_features = [c for c in df_train.columns if c not in columnas_excluir]

X_train, y_train = df_train[columnas_features], np.log(df_train["precio_bolsa"])
X_test = df_test[columnas_features]
y_test_real = df_test["precio_bolsa"].values

modelo = CatBoostRegressor(
    n_estimators=1500, max_depth=6, learning_rate=0.03,
    loss_function="MAE", random_seed=42, verbose=False,
    boosting_type="Ordered",
)
modelo.fit(X_train, y_train)
pred = np.exp(modelo.predict(X_test))

error = y_test_real - pred
mae = np.abs(error).mean()
rmse = np.sqrt((error ** 2).mean())
mape = (np.abs(error) / y_test_real).mean() * 100

print(f"{'='*70}\nCatBoost (ordered boosting) -- holdout 2026\n{'='*70}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"\nReferencia -- XGBoost: MAE 61.15 | MAPE 15.89%")
print(f"Referencia -- ARX+GARCH: MAE 55.76 | MAPE 15.43%")
print(f"Referencia -- Ensamble stacking (NNLS, CV 5 pliegues): MAE 44.51 | MAPE 12.05%")

# --- DM vs XGBoost (referencia guardada en walkforward_predicciones_crudas.csv, Origen 6) ---
crudas = pd.read_csv(RAIZ / "data/processed/resultados/walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
xgb_o6 = crudas[(crudas["origen"] == "Origen 6") & (crudas["modelo"] == "XGBoost")].set_index("fecha_hora")["prediccion"]
df_cb = pd.DataFrame({"fecha_hora": df_test["fecha_hora"], "real": y_test_real, "pred_catboost": pred}).set_index("fecha_hora")
merged = df_cb.join(xgb_o6.rename("pred_xgb"), how="inner")

e_cb = (merged["real"] - merged["pred_catboost"]).abs()
e_xgb = (merged["real"] - merged["pred_xgb"]).abs()
d = e_xgb.values - e_cb.values
r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
print(f"\nDM (CatBoost vs XGBoost): t={r.tvalues[0]:.3f}  p={r.pvalues[0]:.5f}  "
      f"{'CATBOOST GANA (sig.)' if r.pvalues[0]<0.05 and r.tvalues[0]>0 else 'sin diferencia / pierde'}")

pd.DataFrame({"fecha_hora": df_test["fecha_hora"], "real": y_test_real, "prediccion": pred}).to_csv(
    RAIZ / "data" / "processed" / "resultados" / "catboost_2026.csv", index=False
)
print("\nGuardado: catboost_2026.csv")
