# -*- coding: utf-8 -*-
"""
Idea nueva #3: coeficientes que cambian en el tiempo, en vez de fijos. RecursiveLS (statsmodels)
ajusta un modelo lineal donde los coeficientes se actualizan con cada observacion nueva que
llega (filtro recursivo tipo Kalman sobre una regresion lineal) -- la herramienta correcta
cuando la relacion misma esta cambiando (regimen), no solo el nivel.

Mismos regresores que ARX+GARCH (sin la parte GARCH, RecursiveLS no modela varianza condicional)
para poder comparar directo. Como el filtro se actualiza observacion a observacion, se puede
evaluar en 2026 de dos formas:
  (a) "one-step": usando los coeficientes ya estimados con TODO 2019-2025 (sin ver 2026) --
      equivalente a lo que hacen los demas modelos del proyecto.
  (b) "recursivo real": dejando que el filtro siga actualizando coeficientes CON 2026 a medida
      que ese periodo se recorre -- la version que aprovecha la adaptabilidad real del metodo.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.regression.recursive_ls import RecursiveLS
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

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
X_train = sm.add_constant((X_train_raw - x_mean) / x_std)

print(f"Ajustando RecursiveLS sobre {len(y_train)} observaciones, {X_train.shape[1]} regresoras (con constante)...")
import time
t0 = time.time()
# Inicializacion aproximadamente difusa (varianza inicial FINITA) -- la inicializacion difusa
# EXACTA por defecto (varianza infinita) es numericamente inestable en series tan largas
# (coeficientes que explotan a ~1e6/1e7). Con varianza inicial finita (1e4) el filtro converge
# a valores razonables sin perder el espiritu "sin informacion previa fuerte" de un prior difuso.
modelo = RecursiveLS(y_train.values, X_train.values, loglikelihood_burn=0)
modelo.ssm.initialize_approximate_diffuse(variance=1e4)
resultado = modelo.fit()
print(f"Ajuste completo en {time.time()-t0:.0f}s")

y_test_log = np.log(df_test["precio_bolsa"])
y_test_escalado = (y_test_log - y_mean) / y_std * 10
X_test_raw = df_test[columnas_x]
X_test = sm.add_constant((X_test_raw - x_mean) / x_std, has_constant="add")

def calcular_metricas(y_real, y_pred, nombre):
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    err = y_real - y_pred
    mae = np.abs(err).mean()
    mape = (np.abs(err) / y_real).mean() * 100
    print(f"{nombre:40s} MAE {mae:7.2f}  MAPE {mape:5.2f}%")
    return mae, mape

# --- (a) "one-step": coeficientes finales de 2019-2025, aplicados fijos a todo 2026 ---
coefs_finales = resultado.params
pred_escalado_fijo = X_test.values @ coefs_finales
pred_fijo = np.exp(pred_escalado_fijo / 10 * y_std + y_mean)
y_test_real = df_test["precio_bolsa"].values

print(f"\n{'='*70}\nRecursiveLS -- holdout 2026\n{'='*70}")
mae_fijo, mape_fijo = calcular_metricas(y_test_real, pred_fijo, "(a) Coeficientes fijos (fin de 2025)")

# --- (b) recursivo real: extender el filtro con 2026, actualizando coef. observacion a obs. ---
modelo_extendido = RecursiveLS(np.concatenate([y_train.values, y_test_escalado.values]),
                                 np.vstack([X_train.values, X_test.values]), loglikelihood_burn=0)
modelo_extendido.ssm.initialize_approximate_diffuse(variance=1e4)
resultado_extendido = modelo_extendido.fit()

# recursive_coefficients.filtered da los coeficientes DESPUES de ver cada observacion -- para
# predecir la observacion t de forma honesta (sin ver su propio precio), se usan los coeficientes
# FILTRADOS HASTA t-1 (un paso atras), no los que ya incorporaron t.
coefs_recursivos = resultado_extendido.recursive_coefficients.filtered.T  # (n_obs, n_params)
n_train = len(y_train)
coefs_test_prev = coefs_recursivos[n_train - 1: n_train - 1 + len(y_test_real)]  # coef. en t-1, para predecir t

pred_escalado_recursivo = np.sum(X_test.values * coefs_test_prev, axis=1)
pred_recursivo = np.exp(pred_escalado_recursivo / 10 * y_std + y_mean)
mae_rec, mape_rec = calcular_metricas(y_test_real, pred_recursivo, "(b) Coeficientes recursivos (actualizados con 2026)")

print(f"\nReferencia -- ARX+GARCH: MAE 55.76 | MAPE 15.43%")
print(f"Referencia -- Ensamble stacking (NNLS, CV 5 pliegues): MAE 44.51 | MAPE 12.05%")

# --- Prueba de significancia: recursivo vs fijo ---
d = np.abs(y_test_real - pred_fijo) - np.abs(y_test_real - pred_recursivo)
m_dm = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
print(f"\nDM (recursivo vs fijo): t={m_dm.tvalues[0]:.3f}  p={m_dm.pvalues[0]:.5f}  "
      f"{'RECURSIVO GANA (sig.)' if m_dm.pvalues[0]<0.05 and m_dm.tvalues[0]>0 else 'sin diferencia / pierde'}")

# --- Zoom a junio: es ahi donde mas se necesita la adaptabilidad ---
mask_junio = df_test["fecha_hora"].dt.month == 6
print(f"\n{'='*70}\nZoom a junio 2026\n{'='*70}")
calcular_metricas(y_test_real[mask_junio], pred_fijo[mask_junio], "(a) Coeficientes fijos, junio")
calcular_metricas(y_test_real[mask_junio], pred_recursivo[mask_junio], "(b) Coeficientes recursivos, junio")

pd.DataFrame({"fecha_hora": df_test["fecha_hora"], "real": y_test_real,
              "pred_fijo": pred_fijo, "pred_recursivo": pred_recursivo}).to_csv(
    RAIZ / "data" / "processed" / "resultados" / "recursive_ls_2026.csv", index=False
)
print("\nGuardado: recursive_ls_2026.csv")
