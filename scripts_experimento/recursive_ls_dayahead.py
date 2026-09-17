# -*- coding: utf-8 -*-
"""
Correccion de una fuga de informacion en recursive_ls.py (variante "recursiva").

La version original predecia la hora t con los coeficientes filtrados hasta t-1. Eso es honesto
para un pronostico a UNA hora, pero el protocolo del proyecto es day-ahead: el corte es a las
23:00 del dia anterior y desde ahi se pronostican las 24 horas del dia siguiente. Con la version
original, los coeficientes usados para la hora 15 del dia D ya habian incorporado los precios
reales de las horas 0-14 del mismo dia D -- informacion que no existe en el momento del corte.

Correccion: todas las horas del dia D usan los coeficientes filtrados hasta las 23:00 de D-1.
Se guardan las tres variantes (fija, recursiva con fuga, recursiva day-ahead) para documentar
cuanto de la ganancia reportada anoche (DM p=0.00000) sobrevive sin la fuga.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.regression.recursive_ls import RecursiveLS
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

df_train = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
df_test = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
assert df_test["fecha_hora"].iloc[0] - df_train["fecha_hora"].iloc[-1] == pd.Timedelta(hours=1)

regresoras = ["aportes_hidricos", "volumen_embalses", "oni", "es_pandemia",
              "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
              "es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
              "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]
for d in (df_train, df_test):
    d["precio_lag24h_log"] = np.log(d["precio_lag24h"])
columnas_x = regresoras + ["precio_lag24h_log"]

y_train_log = np.log(df_train["precio_bolsa"])
y_mean, y_std = y_train_log.mean(), y_train_log.std()
y_train = (y_train_log - y_mean) / y_std * 10
x_mean, x_std = df_train[columnas_x].mean(), df_train[columnas_x].std()
X_train = sm.add_constant((df_train[columnas_x] - x_mean) / x_std)
X_test = sm.add_constant((df_test[columnas_x] - x_mean) / x_std, has_constant="add")
y_test_esc = (np.log(df_test["precio_bolsa"]) - y_mean) / y_std * 10

modelo = RecursiveLS(np.concatenate([y_train.values, y_test_esc.values]),
                     np.vstack([X_train.values, X_test.values]), loglikelihood_burn=0)
modelo.ssm.initialize_approximate_diffuse(variance=1e4)
res = modelo.fit()
coefs = res.recursive_coefficients.filtered.T

n_train = len(y_train)
j = np.arange(len(df_test))
hora = df_test["fecha_hora"].dt.hour.values
idx_1paso = n_train + j - 1             # original: coeficientes hasta t-1 (fuga intradia)
idx_dayahead = n_train + j - hora - 1   # corregido: coeficientes hasta las 23:00 del dia anterior
idx_fijo = np.full(len(j), n_train - 1)  # coeficientes congelados a fin de 2025


def predecir(idx):
    return np.exp(np.sum(X_test.values * coefs[idx], axis=1) / 10 * y_std + y_mean)


real = df_test["precio_bolsa"].values
preds = {"fijo": predecir(idx_fijo), "recursivo_1paso": predecir(idx_1paso),
         "recursivo_dayahead": predecir(idx_dayahead)}

print(f"{'variante':22s} {'MAE':>8s} {'MAPE':>8s}")
for k, p in preds.items():
    print(f"{k:22s} {np.abs(real-p).mean():8.2f} {(np.abs(real-p)/real).mean()*100:7.2f}%")


def dm(p_ref, p_nuevo):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
    return r.tvalues[0], r.pvalues[0]


t, p = dm(preds["fijo"], preds["recursivo_dayahead"])
print(f"\nDM day-ahead vs fijo: t={t:.3f} p={p:.5f}  "
      f"{'DAY-AHEAD GANA (sig.)' if p < 0.05 and t > 0 else 'sin diferencia / pierde'}")
t, p = dm(preds["recursivo_dayahead"], preds["recursivo_1paso"])
print(f"DM 1-paso (con fuga) vs day-ahead: t={t:.3f} p={p:.5f}  -> cuanto 'regalaba' la fuga")

pd.DataFrame({"fecha_hora": df_test["fecha_hora"], "real": real,
              "pred_fijo": preds["fijo"], "pred_recursivo_1paso": preds["recursivo_1paso"],
              "pred_recursivo_dayahead": preds["recursivo_dayahead"]}).to_csv(
    RAIZ / "data/processed/resultados/recursive_ls_dayahead_2026.csv", index=False)
print("Guardado: recursive_ls_dayahead_2026.csv")
