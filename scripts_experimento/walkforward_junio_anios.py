# -*- coding: utf-8 -*-
"""
Confirma si el patron de racha de dias malos visto en junio 2026 (Origen 6) se repite en junio de
otros anios, o si es especifico de 2026. Los Origenes 1-5 del walk-forward original (notebook 10)
NUNCA cubren junio (jul-sep, jul-sep, oct-dic, oct-dic, ene-mar), asi que no sirven para esto --
se necesitan origenes nuevos con ventana de prueba = solo junio, un origen por anio 2020-2025
(2026 ya esta cubierto por el Origen 6 original).

Mismos 4 modelos, mismos hiperparametros e igual metodologia que notebook 10 (para que sea
comparable): Persistencia, XGBoost, ARX+GARCH(1,1), N-BEATSx, N-HiTS. Guarda predicciones
hora por hora igual que walkforward_predicciones_crudas.csv, en un archivo separado.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")


def calcular_metricas(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real, dtype=float), np.asarray(y_pred, dtype=float)
    error = y_real - y_pred
    mae = np.abs(error).mean()
    rmse = np.sqrt((error ** 2).mean())
    mape = (np.abs(error) / y_real).mean() * 100
    return mae, rmse, mape


df_train = pd.read_csv(RAIZ / "data" / "processed" / "dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
df_test = pd.read_csv(RAIZ / "data" / "processed" / "dataset_features_2026.csv", parse_dates=["fecha_hora"])
df_completo = pd.concat([df_train, df_test], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
print("Filas:", len(df_completo))

origenes = [
    {"nombre": "Junio 2020", "corte_train": "2020-06-01", "test_inicio": "2020-06-01", "test_fin": "2020-06-30"},
    {"nombre": "Junio 2021", "corte_train": "2021-06-01", "test_inicio": "2021-06-01", "test_fin": "2021-06-30"},
    {"nombre": "Junio 2022", "corte_train": "2022-06-01", "test_inicio": "2022-06-01", "test_fin": "2022-06-30"},
    {"nombre": "Junio 2023", "corte_train": "2023-06-01", "test_inicio": "2023-06-01", "test_fin": "2023-06-30"},
    {"nombre": "Junio 2024", "corte_train": "2024-06-01", "test_inicio": "2024-06-01", "test_fin": "2024-06-30"},
    {"nombre": "Junio 2025", "corte_train": "2025-06-01", "test_inicio": "2025-06-01", "test_fin": "2025-06-30"},
]

columnas_excluir = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora", "dia_semana", "dia_anio"]
columnas_features = [c for c in df_completo.columns if c not in columnas_excluir]
print(len(columnas_features), "features para XGBoost")

import xgboost as xgb
from arch import arch_model

predicciones_crudas = []

for o in origenes:
    train = df_completo[df_completo["fecha_hora"] < o["corte_train"]].dropna(subset=columnas_features).copy()
    test = df_completo[(df_completo["fecha_hora"] >= o["test_inicio"]) & (df_completo["fecha_hora"] <= o["test_fin"])].copy()

    for _, fila in test.iterrows():
        predicciones_crudas.append({"origen": o["nombre"], "fecha_hora": fila["fecha_hora"],
                                     "real": fila["precio_bolsa"], "modelo": "Persistencia",
                                     "prediccion": fila["precio_lag24h"]})

    X_train, y_train = train[columnas_features], np.log(train["precio_bolsa"])
    X_test = test[columnas_features]
    m_xgb = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
    m_xgb.fit(X_train, y_train)
    pred_xgb = np.exp(m_xgb.predict(X_test))
    for fh, real, pred in zip(test["fecha_hora"], test["precio_bolsa"], pred_xgb):
        predicciones_crudas.append({"origen": o["nombre"], "fecha_hora": fh, "real": real,
                                     "modelo": "XGBoost", "prediccion": pred})

    regresoras_arx = ["aportes_hidricos", "volumen_embalses", "oni", "es_pandemia",
                       "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
                       "es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
                       "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]
    train["precio_lag24h_log"] = np.log(train["precio_lag24h"])
    test["precio_lag24h_log"] = np.log(test["precio_lag24h"])
    columnas_x_arx = regresoras_arx + ["precio_lag24h_log"]

    y_train_log = np.log(train["precio_bolsa"])
    y_mean, y_std = y_train_log.mean(), y_train_log.std()
    y_train_arx = (y_train_log - y_mean) / y_std * 10
    X_train_raw = train[columnas_x_arx]
    x_mean, x_std = X_train_raw.mean(), X_train_raw.std()
    X_train_arx = (X_train_raw - x_mean) / x_std

    modelo_arx = arch_model(y_train_arx, x=X_train_arx, mean="ARX", lags=0, vol="GARCH", p=1, q=1, dist="normal")
    resultado_arx = modelo_arx.fit(disp="off", options={"maxiter": 500})

    X_test_arx = (test[columnas_x_arx] - x_mean) / x_std
    params_media = resultado_arx.params[["Const"] + columnas_x_arx]
    pred_escalado = params_media["Const"] + (X_test_arx * params_media[columnas_x_arx]).sum(axis=1)
    pred_arx = np.exp(pred_escalado / 10 * y_std + y_mean)
    for fh, real, pred in zip(test["fecha_hora"], test["precio_bolsa"], pred_arx.values):
        predicciones_crudas.append({"origen": o["nombre"], "fecha_hora": fh, "real": real,
                                     "modelo": "ARX+GARCH", "prediccion": pred})

    print(f"{o['nombre']}  Persistencia/XGBoost/ARX+GARCH listos")

df_pred_rapidos = pd.DataFrame(predicciones_crudas)
print("\nTotal filas (3 modelos rapidos):", len(df_pred_rapidos))

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx, NHITS

hist_exog_dl = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
futr_exog_continua_dl = ["oni"]
futr_exog_ya_escalada_dl = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
                             "es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
                             "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]
futr_exog_dl = futr_exog_continua_dl + futr_exog_ya_escalada_dl
INPUT_SIZE_DL = 168
MAX_STEPS_DL = 1000

predicciones_dl = []

for o in origenes:
    train = df_completo[df_completo["fecha_hora"] < o["corte_train"]].dropna(subset=columnas_features).copy()
    test = df_completo[(df_completo["fecha_hora"] >= o["test_inicio"]) & (df_completo["fecha_hora"] <= o["test_fin"])].copy()

    df_origen = pd.concat([train, test], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    mascara_train_o = df_origen["fecha_hora"] < o["corte_train"]

    df_nf_o = df_origen[["fecha_hora", "precio_bolsa"] + hist_exog_dl + futr_exog_dl].copy()
    for col in hist_exog_dl + futr_exog_continua_dl:
        mu, sigma = df_nf_o.loc[mascara_train_o, col].mean(), df_nf_o.loc[mascara_train_o, col].std()
        df_nf_o[col] = (df_nf_o[col] - mu) / sigma

    df_nf_o["unique_id"] = "precio_bolsa"
    df_nf_o = df_nf_o.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})
    df_nf_o = df_nf_o[["unique_id", "ds", "y"] + hist_exog_dl + futr_exog_dl]

    n_test_o = int((~mascara_train_o).sum())
    n_windows_o = n_test_o // 24

    m_nbeatsx = NBEATSx(h=24, input_size=INPUT_SIZE_DL, hist_exog_list=hist_exog_dl, futr_exog_list=futr_exog_dl,
                          max_steps=MAX_STEPS_DL, val_check_steps=100, random_seed=42, enable_progress_bar=False)
    m_nhits = NHITS(h=24, input_size=INPUT_SIZE_DL, hist_exog_list=hist_exog_dl, futr_exog_list=futr_exog_dl,
                      max_steps=MAX_STEPS_DL, val_check_steps=100, random_seed=42, enable_progress_bar=False)
    nf_o = NeuralForecast(models=[m_nbeatsx, m_nhits], freq="h")
    cv_o = nf_o.cross_validation(df=df_nf_o, n_windows=n_windows_o, step_size=24)

    for col_modelo, nombre_modelo in [("NBEATSx", "N-BEATSx"), ("NHITS", "N-HiTS")]:
        for fh, real, pred in zip(cv_o["ds"], cv_o["y"], cv_o[col_modelo]):
            predicciones_dl.append({"origen": o["nombre"], "fecha_hora": fh, "real": real,
                                     "modelo": nombre_modelo, "prediccion": pred})

    print(f"{o['nombre']}  N-BEATSx/N-HiTS listos")

df_pred_dl = pd.DataFrame(predicciones_dl)
df_predicciones = pd.concat([df_pred_rapidos, df_pred_dl], ignore_index=True)
df_predicciones.to_csv(RAIZ / "data" / "processed" / "resultados" / "walkforward_junio_predicciones_crudas.csv", index=False)
print("\nGuardado: walkforward_junio_predicciones_crudas.csv --", len(df_predicciones), "filas")

resumen = df_predicciones.groupby(["origen", "modelo"]).apply(
    lambda g: pd.Series(calcular_metricas(g["real"], g["prediccion"]), index=["mae", "rmse", "mape"])
).reset_index()
print("\n" + "=" * 70)
print("MAE por origen y modelo")
print("=" * 70)
print(resumen.pivot(index="origen", columns="modelo", values="mae").round(2).to_string())

print("\n" + "=" * 70)
print("RACHAS DE DIAS CONSECUTIVOS MALOS (MAPE diario > 25%), por origen, N-BEATSx")
print("=" * 70)
df_predicciones["ape"] = (df_predicciones["real"] - df_predicciones["prediccion"]).abs() / df_predicciones["real"] * 100
df_predicciones["dia"] = pd.to_datetime(df_predicciones["fecha_hora"]).dt.normalize()
for o in origenes:
    m = df_predicciones[(df_predicciones["origen"] == o["nombre"]) & (df_predicciones["modelo"] == "N-BEATSx")]
    por_dia = m.groupby("dia")["ape"].mean().sort_index()
    malo = por_dia > 25
    grupo = (malo != malo.shift()).cumsum()
    rachas = por_dia[malo].groupby(grupo[malo]).apply(lambda x: (x.index.min(), x.index.max(), len(x)))
    print(f"\n--- {o['nombre']} --- ({malo.sum()} dias con MAPE>25% de {len(por_dia)} dias totales)")
    if len(rachas) == 0:
        print("  (ninguna racha)")
    for inicio, fin, n in rachas:
        print(f"  {inicio.date()} a {fin.date()}  ({n} dias seguidos)")
