# -*- coding: utf-8 -*-
"""
Igual que ventanas_calibracion.py pero para N-HiTS en vez de N-BEATSx, y solo con las 5
ventanas que dieron el mejor ensamble en N-BEATSx (1a, 2a, 3a, 5a, completa) -- se omite
6m porque ya se probo que es mala individualmente, para no gastar tiempo de mas.

Objetivo: ver si el mismo mecanismo de ensamble de ventanas de calibracion (Lago et al.
2021), que le dio a N-BEATSx un nuevo record (MAE 54.34 vs 56.11 de la ventana unica),
tambien funciona para N-HiTS -- el otro de los 3 mejores modelos del proyecto.
"""
import pandas as pd
import numpy as np
import time
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_PRED = RAIZ / "data" / "processed" / "resultados" / "ventanas_predicciones_nhits.csv"
RUTA_MET = RAIZ / "data" / "processed" / "resultados" / "ventanas_metricas_nhits.csv"

FECHA_TEST = pd.Timestamp("2026-01-01")
HORIZONTE = 24
INPUT_SIZE = 168
MAX_STEPS = 1000

VENTANAS = {"1a": 365, "2a": 730, "3a": 1095, "5a": 1825, "completa": 2557}
SEMILLAS = [42, 7]

HIST_EXOG = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUTR_EXOG_CONT = ["oni"]
FUTR_EXOG_BIN = ["es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos"]


def cargar():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    return pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)


def correr_ventana(df, dias_ventana, semilla):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    inicio = FECHA_TEST - pd.Timedelta(days=dias_ventana)
    df_v = df[df["fecha_hora"] >= inicio].reset_index(drop=True)

    futr_exog = FUTR_EXOG_CONT + FUTR_EXOG_BIN
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
    n_windows = n_test // HORIZONTE

    modelo = NHITS(h=HORIZONTE, input_size=INPUT_SIZE, hist_exog_list=HIST_EXOG,
                   futr_exog_list=futr_exog, max_steps=MAX_STEPS, val_check_steps=100,
                   random_seed=semilla, enable_progress_bar=False)
    nf = NeuralForecast(models=[modelo], freq="h")
    cv = nf.cross_validation(df=df_nf, n_windows=n_windows, step_size=HORIZONTE)
    return cv[["ds", "y", "NHITS"]], int(mascara_train.sum())


def metricas(y, pred):
    err = np.abs(np.asarray(y, float) - np.asarray(pred, float))
    return err.mean(), np.sqrt(((np.asarray(y, float) - np.asarray(pred, float)) ** 2).mean()), (err / np.asarray(y, float)).mean() * 100


def main():
    df = cargar()
    filas_pred, filas_met = [], []
    t0 = time.time()

    for nombre, dias in VENTANAS.items():
        for semilla in SEMILLAS:
            t = time.time()
            cv, n_train = correr_ventana(df, dias, semilla)
            mae, rmse, mape = metricas(cv["y"].values, cv["NHITS"].values)

            filas_met.append({"ventana": nombre, "dias": dias, "semilla": semilla,
                               "n_train": n_train, "mae": mae, "rmse": rmse, "mape": mape,
                               "segundos": time.time() - t})
            tmp = cv.copy()
            tmp["ventana"] = nombre
            tmp["semilla"] = semilla
            filas_pred.append(tmp)

            print(f"  ventana {nombre:9s} semilla {semilla:4d} | n_train {n_train:6,} | "
                  f"MAE {mae:6.2f} RMSE {rmse:6.2f} MAPE {mape:5.2f}% | {time.time()-t:.0f}s")
            sys.stdout.flush()

            pd.concat(filas_pred, ignore_index=True).to_csv(RUTA_PRED, index=False)
            pd.DataFrame(filas_met).to_csv(RUTA_MET, index=False)

    df_pred = pd.concat(filas_pred, ignore_index=True)
    df_met = pd.DataFrame(filas_met)

    print(f"\n{'='*70}\nINDIVIDUALES (media sobre semillas)\n{'='*70}")
    print(df_met.groupby("ventana")[["mae", "rmse", "mape"]].mean().round(2).sort_values("mae").to_string())

    print(f"\n{'='*70}\nENSAMBLES (promedio de predicciones, todas las semillas)\n{'='*70}")
    pivote = df_pred.pivot_table(index="ds", columns="ventana", values="NHITS", aggfunc="mean")
    y_real = df_pred.groupby("ds")["y"].first().reindex(pivote.index)

    import itertools
    combos = []
    nombres = list(VENTANAS.keys())
    for k in range(1, len(nombres) + 1):
        for combo in itertools.combinations(nombres, k):
            pred = pivote[list(combo)].mean(axis=1)
            mae, rmse, mape = metricas(y_real.values, pred.values)
            combos.append({"ensamble": "+".join(combo), "n": k, "mae": mae, "rmse": rmse, "mape": mape})

    df_combos = pd.DataFrame(combos).sort_values("mae")
    print("\nTop 10 ensambles por MAE:")
    print(df_combos.head(10).round(3).to_string(index=False))
    print("\nReferencia -- mejor ventana individual:")
    print(df_combos[df_combos["n"] == 1].sort_values("mae").round(3).to_string(index=False))
    print("\nReferencia del proyecto: N-HiTS ventana completa (oficial) = 57.08")
    print("Referencia N-BEATSx: ventana completa 56.11 -> ensamble 5 ventanas 54.34")

    df_combos.to_csv(RAIZ / "data/processed/resultados/ventanas_ensambles_nhits.csv", index=False)
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
