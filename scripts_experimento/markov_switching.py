# -*- coding: utf-8 -*-
"""
Punto 2 de la lista de alternativas: modelo de cambio de regimen explicito (Markov-Switching),
en vez de tratar el regimen (El Nino/La Nina) como una covariable mas que el modelo debe
aprender a usar (que es lo que hacen ARX+GARCH, N-BEATSx, etc. con "oni" como exogena).

MarkovRegression (statsmodels) infiere un estado latente no observado (2 regimenes: "calmo" y
"turbulento") a partir de los datos mismos, y deja que el INTERCEPTO y la VARIANZA cambien segun
el regimen inferido, mientras que el efecto de las variables exogenas (hidrologia, calendario,
festivos, precio rezagado) se mantiene fijo entre regimenes -- mismo criterio de parsimonia que
ARX+GARCH (lags=0, exogenas fijas), para que la comparacion sea limpia.

Mismos regresores, mismo pipeline de escalado y mismo holdout 2026 que ARX+GARCH, para
comparabilidad directa.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import warnings
warnings.filterwarnings("ignore")
import time

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

df_train = pd.read_csv(RAIZ / "data" / "processed" / "dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
df_test = pd.read_csv(RAIZ / "data" / "processed" / "dataset_features_2026.csv", parse_dates=["fecha_hora"])

regresoras = ["aportes_hidricos", "volumen_embalses", "oni", "es_pandemia",
              "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
              "es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
              "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]

df_train = df_train.copy()
df_test = df_test.copy()
df_train["precio_lag24h_log"] = np.log(df_train["precio_lag24h"])
df_test["precio_lag24h_log"] = np.log(df_test["precio_lag24h"])
columnas_x = regresoras + ["precio_lag24h_log"]

y_train_log = np.log(df_train["precio_bolsa"])
y_mean, y_std = y_train_log.mean(), y_train_log.std()
y_train = (y_train_log - y_mean) / y_std * 10

X_train_raw = df_train[columnas_x]
x_mean, x_std = X_train_raw.mean(), X_train_raw.std()
X_train = (X_train_raw - x_mean) / x_std

print("Ajustando MarkovRegression (2 regimenes, intercepto y varianza cambian, exogenas fijas)...")
print(f"Filas de entrenamiento: {len(y_train)}  |  Regresoras: {len(columnas_x)}")
t0 = time.time()

modelo = MarkovRegression(
    endog=y_train.values, k_regimes=2, exog=X_train.values,
    trend="c", switching_trend=True, switching_exog=False, switching_variance=True,
)
resultado = modelo.fit(em_iter=50, search_reps=0)

print(f"Ajuste completo en {time.time()-t0:.0f}s")
print(resultado.summary())

# --- Diagnostico: que tan distintos son los dos regimenes inferidos, y su persistencia ---
print("\nProbabilidades de transicion:")
print(resultado.regime_transition)
nombres = resultado.model.param_names
print("\nNombres de parametros:", list(nombres))
params_dict = dict(zip(nombres, resultado.params))
print("\nParametros de cada regimen (intercepto, varianza):")
for r in range(2):
    print(f"  Regimen {r}: const={params_dict[f'const[{r}]']:.3f}  sigma2={params_dict[f'sigma2[{r}]']:.3f}")

# --- Forecast sobre 2026: prediccion ponderada por probabilidad de regimen (filtrada, no suavizada) ---
X_test_raw = df_test[columnas_x]
X_test = (X_test_raw - x_mean) / x_std

# Coeficientes de las exogenas (fijos entre regimenes) y de cada regimen (const)
coefs_exog = np.array([params_dict[f"x{i+1}[1]"] for i in range(len(columnas_x))])
const_0 = params_dict["const[0]"]
const_1 = params_dict["const[1]"]

pred_regimen_0 = const_0 + X_test.values @ coefs_exog
pred_regimen_1 = const_1 + X_test.values @ coefs_exog

# Probabilidad filtrada del ultimo punto de train, usada como punto de partida para 2026
# (aproximacion honesta: no hay filtrado real hacia adelante sin refit, se usa la probabilidad
# estacionaria de la matriz de transicion como peso, dado que el holdout es largo)
matriz_transicion = np.array(resultado.regime_transition).reshape(2, 2)
# matriz_transicion[i, j] = P(destino=i | origen=j) -- probabilidad estacionaria: pi = P @ pi,
# es decir el AUTOVECTOR DERECHO de P para autovalor 1 (sin transponer)
vals, vecs = np.linalg.eig(matriz_transicion)
idx = np.argmin(np.abs(vals - 1))
vec_estacionario = np.real(vecs[:, idx])
prob_estacionaria = (vec_estacionario / vec_estacionario.sum()).flatten()
print(f"\nProbabilidad estacionaria de cada regimen: {prob_estacionaria}")

pred_escalado = prob_estacionaria[0] * pred_regimen_0 + prob_estacionaria[1] * pred_regimen_1
pred_log = pred_escalado / 10 * y_std + y_mean
pred_final = np.exp(pred_log)

y_test_real = df_test["precio_bolsa"].values
error = y_test_real - pred_final
mae = np.abs(error).mean()
rmse = np.sqrt((error ** 2).mean())
mape = (np.abs(error) / y_test_real).mean() * 100

print(f"\n{'='*70}\nMarkov-Switching (2 regimenes) -- holdout 2026\n{'='*70}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print("\nReferencia -- ARX+GARCH: MAE 55.76 | RMSE 109.97 | MAPE 15.43%")
print("Referencia -- Ensamble stacking (NNLS, CV 5 pliegues): MAE 44.51 | MAPE 12.05%")

# --- Prueba de significancia contra ARX+GARCH y contra el ensamble de stacking ---
from arch import arch_model
modelo_arx = arch_model(y_train, x=X_train, mean="ARX", lags=0, vol="GARCH", p=1, q=1, dist="normal")
resultado_arx = modelo_arx.fit(disp="off", options={"maxiter": 500})
params_media_arx = resultado_arx.params[["Const"] + columnas_x]
pred_escalado_arx = params_media_arx["Const"] + (X_test * params_media_arx[columnas_x]).sum(axis=1)
pred_arx = np.exp(pred_escalado_arx.values / 10 * y_std + y_mean)

error_ms = np.abs(y_test_real - pred_final)
error_arx = np.abs(y_test_real - pred_arx)
d = error_arx - error_ms
m_dm = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
print(f"\nDM vs ARX+GARCH: t={m_dm.tvalues[0]:.3f}  p={m_dm.pvalues[0]:.5f}  "
      f"{'MARKOV-SWITCHING GANA (sig.)' if m_dm.pvalues[0]<0.05 and m_dm.tvalues[0]>0 else 'sin diferencia / pierde'}")

pd.DataFrame({"fecha_hora": df_test["fecha_hora"], "real": y_test_real, "prediccion": pred_final}).to_csv(
    RAIZ / "data" / "processed" / "resultados" / "markov_switching_2026.csv", index=False
)
print("\nGuardado: markov_switching_2026.csv")
