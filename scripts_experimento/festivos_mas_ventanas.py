# -*- coding: utf-8 -*-
"""
Combina las dos unicas ideas de la noche con señal real:
  - Festivos (CONFIRMADO, p=0.0064): mejora el error en horas festivas.
  - Ensamble de ventanas de calibracion (direccional, no significativo solo):
    1a+2a+3a+5a+completa dio el mejor MAE global (54.34).

Se entrena N-BEATSx CON features de festivo en esas mismas 5 ventanas (2 semillas c/u,
10 entrenamientos) para ver si el ensamble resultante mejora sobre el 54.34 ya conocido,
y si ademas hereda la mejora especifica en horas festivas.

REVERSIBLE: vive en scripts_experimento/. Resultados en
data/processed/resultados/festivos_mas_ventanas.csv
"""
import pandas as pd
import numpy as np
import holidays
import time
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_PRED = RAIZ / "data" / "processed" / "resultados" / "festivos_ventanas_predicciones.csv"
RUTA_MET = RAIZ / "data" / "processed" / "resultados" / "festivos_mas_ventanas.csv"

FECHA_TEST = pd.Timestamp("2026-01-01")
HORIZONTE, INPUT_SIZE, MAX_STEPS = 24, 168, 1000
VENTANAS = {"1a": 365, "2a": 730, "3a": 1095, "5a": 1825, "completa": 2557}
SEMILLAS = [42, 7]

HIST_EXOG = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUTR_EXOG_CONT = ["oni"]
FUTR_EXOG_BIN = ["es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos"]
FUTR_EXOG_FESTIVOS = ["es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
                       "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]


def cargar():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

    festivos = set(holidays.CO(years=range(2018, 2028)).keys())
    fecha = df["fecha_hora"].dt.normalize()

    def festivo_desplazado(dias):
        return (fecha - pd.Timedelta(days=dias)).dt.date.map(lambda d: d in festivos).astype(int)

    df["es_festivo"] = festivo_desplazado(0)
    df["festivo_lag24h"] = festivo_desplazado(1)
    df["festivo_lag48h"] = festivo_desplazado(2)
    df["festivo_lag72h"] = festivo_desplazado(3)
    df["festivo_lag168h"] = festivo_desplazado(7)
    df["mismatch_festivo_24h"] = (df["es_festivo"] != df["festivo_lag24h"]).astype(int)
    df["mismatch_festivo_168h"] = (df["es_festivo"] != df["festivo_lag168h"]).astype(int)
    return df


def correr_ventana(df, dias_ventana, semilla):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx

    inicio = FECHA_TEST - pd.Timedelta(days=dias_ventana)
    df_v = df[df["fecha_hora"] >= inicio].reset_index(drop=True)

    futr_exog = FUTR_EXOG_CONT + FUTR_EXOG_BIN + FUTR_EXOG_FESTIVOS
    df_nf = df_v[["fecha_hora", "precio_bolsa"] + HIST_EXOG + futr_exog].copy()
    mascara_train = df_v["fecha_hora"] < FECHA_TEST

    for col in HIST_EXOG + FUTR_EXOG_CONT:
        mu = df_nf.loc[mascara_train.values, col].mean()
        sigma = df_nf.loc[mascara_train.values, col].std()
        if sigma == 0 or pd.isna(sigma):
            sigma = 1.0
        df_nf[col] = (df_nf[col] - mu) / sigma

    df_nf["unique_id"] = "precio_bolsa"
    df_nf = df_nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})
    df_nf = df_nf[["unique_id", "ds", "y"] + HIST_EXOG + futr_exog]

    n_test = int((~mascara_train).sum())
    modelo = NBEATSx(h=HORIZONTE, input_size=INPUT_SIZE, hist_exog_list=HIST_EXOG,
                      futr_exog_list=futr_exog, max_steps=MAX_STEPS, val_check_steps=100,
                      random_seed=semilla, enable_progress_bar=False)
    nf = NeuralForecast(models=[modelo], freq="h")
    cv = nf.cross_validation(df=df_nf, n_windows=n_test // HORIZONTE, step_size=HORIZONTE)
    return cv[["ds", "y", "NBEATSx"]]


def metricas(y, pred):
    err = np.abs(np.asarray(y, float) - np.asarray(pred, float))
    return err.mean(), np.sqrt(((np.asarray(y, float) - np.asarray(pred, float)) ** 2).mean()), (err / np.asarray(y, float)).mean() * 100


def main():
    df = cargar()
    festivo_por_fecha = df.set_index("fecha_hora")["es_festivo"]
    filas_pred, filas_met = [], []
    t0 = time.time()

    for nombre, dias in VENTANAS.items():
        for semilla in SEMILLAS:
            t = time.time()
            cv = correr_ventana(df, dias, semilla)
            mae, rmse, mape = metricas(cv["y"].values, cv["NBEATSx"].values)
            mask_fest = festivo_por_fecha.reindex(cv["ds"]).values == 1
            err = np.abs(cv["y"].values - cv["NBEATSx"].values)

            filas_met.append({"ventana": nombre, "semilla": semilla, "mae": mae, "rmse": rmse,
                               "mape": mape, "mae_festivos": err[mask_fest].mean(),
                               "segundos": time.time() - t})
            tmp = cv.copy()
            tmp["ventana"] = nombre
            tmp["semilla"] = semilla
            filas_pred.append(tmp)

            print(f"  ventana {nombre:9s} semilla {semilla:4d} | MAE {mae:6.2f} | "
                  f"festivos {err[mask_fest].mean():6.2f} | {time.time()-t:.0f}s")
            sys.stdout.flush()
            pd.concat(filas_pred, ignore_index=True).to_csv(RUTA_PRED, index=False)
            pd.DataFrame(filas_met).to_csv(RUTA_MET, index=False)

    df_pred = pd.concat(filas_pred, ignore_index=True)
    print(f"\n{'='*70}\nENSAMBLE 1a+2a+3a+5a+completa (CON festivos)\n{'='*70}")
    pivote = df_pred.pivot_table(index="ds", columns="ventana", values="NBEATSx", aggfunc="mean")
    y_real = df_pred.groupby("ds")["y"].first().reindex(pivote.index)
    mask_fest_ens = festivo_por_fecha.reindex(pivote.index).values == 1

    pred_ensamble = pivote[list(VENTANAS.keys())].mean(axis=1)
    err_ens = np.abs(y_real.values - pred_ensamble.values)
    mae, rmse, mape = metricas(y_real.values, pred_ensamble.values)
    print(f"MAE global: {mae:.2f}  RMSE: {rmse:.2f}  MAPE: {mape:.2f}%")
    print(f"MAE en festivos: {err_ens[mask_fest_ens].mean():.2f}")
    print(f"MAE fuera de festivos: {err_ens[~mask_fest_ens].mean():.2f}")

    print("\nReferencias:")
    print("  Ensamble de ventanas SIN festivos (ya conocido): MAE global 54.34")
    print("  BASE (ventana completa) CON festivos, 5 semillas: MAE festivos 62.65 +- 6.05")
    print("  BASE (ventana completa) SIN festivos, 5 semillas: MAE festivos 74.07 +- 3.90")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
