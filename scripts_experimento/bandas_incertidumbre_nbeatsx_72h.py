# -*- coding: utf-8 -*-
"""
Bandas de incertidumbre para N-BEATSx a horizonte COMPLETO 24-72h (no solo el punto extremo).

Hueco encontrado el 2026-09-10: las bandas de incertidumbre que ya existen (script hermano,
bandas_incertidumbre_nbeatsx.py) solo cubren h=24. El propio Anexo 1 (OE2.2) exige comparar los
modelos "con horizonte de 24 a 72 horas" -- el motor de decision (OE3) necesita incertidumbre en
TODO ese rango, no solo en la hora 24.

N-BEATSx es nativamente multi-horizonte: entrenado con h=72, cada ventana de cross_validation ya
pronostica las 72 horas siguientes de una sola pasada (no hace falta un modelo separado por paso).
Se recupera el "paso" de cada prediccion (ds - cutoff, en horas) para poder reportar cobertura y
error por tramo de horizonte (1-24h / 25-48h / 49-72h), que es donde se espera que la incertidumbre
crezca.

Incluye festivos (integrados al pipeline compartido el 2026-09-10, confirmado con significancia
estadistica) en los mismos futr_exog que el resto de la sesion.

Salida principal -- contrato completo para OE3:
  data/processed/resultados/pronostico_con_bandas_72h_2026.csv
  columnas: fecha_hora, cutoff, paso_horas, real, q10, q50, q90
"""
import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_CONTRATO = RAIZ / "data" / "processed" / "resultados" / "pronostico_con_bandas_72h_2026.csv"
RUTA_METRICAS = RAIZ / "data" / "processed" / "resultados" / "bandas_metricas_72h.csv"

FECHA_TEST = pd.Timestamp("2026-01-01")
HORIZONTE, INPUT_SIZE, MAX_STEPS = 72, 168, 1000
CUANTILES = [0.1, 0.5, 0.9]

HIST_EXOG = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUTR_EXOG_CONT = ["oni"]
# Festivos incluidos (integrados al pipeline compartido hoy, p=0.0064 en N-BEATSx a 24h).
FUTR_EXOG_BIN = ["es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
                  "es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
                  "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]


def perdida_pinball(y, pred, tau):
    d = np.asarray(y, float) - np.asarray(pred, float)
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def main():
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MQLoss

    t0 = time.time()
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

    futr_exog = FUTR_EXOG_CONT + FUTR_EXOG_BIN
    df_nf = df[["fecha_hora", "precio_bolsa"] + HIST_EXOG + futr_exog].copy()
    mascara_train = df["fecha_hora"] < FECHA_TEST
    for col in HIST_EXOG + FUTR_EXOG_CONT:
        mu = df_nf.loc[mascara_train.values, col].mean()
        sigma = df_nf.loc[mascara_train.values, col].std()
        df_nf[col] = (df_nf[col] - mu) / sigma

    df_nf["unique_id"] = "precio_bolsa"
    df_nf = df_nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})
    df_nf = df_nf[["unique_id", "ds", "y"] + HIST_EXOG + futr_exog]

    n_test = int((~mascara_train).sum())
    n_windows = n_test // HORIZONTE
    print(f"Entrenando N-BEATSx h={HORIZONTE} con MQLoss (cuantiles {CUANTILES})...")
    print(f"Train: {int(mascara_train.sum()):,} | Test 2026: {n_test:,} | Ventanas: {n_windows}")

    modelo = NBEATSx(h=HORIZONTE, input_size=INPUT_SIZE, hist_exog_list=HIST_EXOG,
                      futr_exog_list=futr_exog, max_steps=MAX_STEPS, val_check_steps=100,
                      loss=MQLoss(quantiles=CUANTILES), random_seed=42,
                      enable_progress_bar=False)
    nf = NeuralForecast(models=[modelo], freq="h")
    cv = nf.cross_validation(df=df_nf, n_windows=n_windows, step_size=HORIZONTE)

    cols_pred = [c for c in cv.columns if c.startswith("NBEATSx")]
    print("Columnas de prediccion:", cols_pred)

    def buscar(patrones):
        for p in patrones:
            for c in cols_pred:
                if p in c:
                    return c
        return None

    col_q10 = buscar(["ql0.1", "lo-80", "-lo-", "q-10"])
    col_q50 = buscar(["median", "ql0.5", "q-50"])
    col_q90 = buscar(["ql0.9", "hi-80", "-hi-", "q-90"])
    if col_q50 is None:
        col_q50 = cols_pred[len(cols_pred) // 2]
    print(f"Mapeo -> q10={col_q10}  q50={col_q50}  q90={col_q90}")

    cv["paso_horas"] = ((cv["ds"] - cv["cutoff"]).dt.total_seconds() / 3600).astype(int)

    salida = pd.DataFrame({
        "fecha_hora": cv["ds"].values,
        "cutoff": cv["cutoff"].values,
        "paso_horas": cv["paso_horas"].values,
        "real": cv["y"].values,
        "q10": cv[col_q10].values if col_q10 else np.nan,
        "q50": cv[col_q50].values,
        "q90": cv[col_q90].values if col_q90 else np.nan,
    })
    salida.to_csv(RUTA_CONTRATO, index=False)

    y = salida["real"].values
    mae_q50 = np.abs(y - salida["q50"].values).mean()
    print(f"\n{'='*70}\nMAE global (cuantil 50, todos los pasos 1-72h): {mae_q50:.2f}")
    print("Referencia -- N-BEATSx a 24h (punto fijo): MAE 56.11")

    filas_met = [{"tramo": "global", "mae": mae_q50, "n": len(salida)}]

    print(f"\n{'='*70}\nDESGLOSE POR TRAMO DE HORIZONTE (donde se espera que crezca el error)\n{'='*70}")
    tramos = [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]
    for nombre, ini, fin in tramos:
        sub = salida[(salida["paso_horas"] >= ini) & (salida["paso_horas"] <= fin)]
        if len(sub) == 0:
            continue
        mae_tramo = np.abs(sub["real"] - sub["q50"]).mean()
        dentro = ((sub["real"] >= sub["q10"]) & (sub["real"] <= sub["q90"])).mean() * 100
        ancho = (sub["q90"] - sub["q10"]).mean()
        print(f"{nombre:8s}  n={len(sub):5d}  MAE={mae_tramo:6.2f}  cobertura={dentro:5.1f}%  ancho_banda={ancho:6.1f}")
        filas_met.append({"tramo": nombre, "mae": mae_tramo, "cobertura": dentro, "ancho_banda": ancho, "n": len(sub)})

    print("\nComparacion directa con el punto fijo de 72h de XGBoost (notebook 06): MAE 85.18")
    mae_72_exacto = np.abs(salida.loc[salida["paso_horas"] == 72, "real"] - salida.loc[salida["paso_horas"] == 72, "q50"]).mean()
    print(f"N-BEATSx exactamente en el paso 72h: MAE {mae_72_exacto:.2f}")

    pd.DataFrame(filas_met).to_csv(RUTA_METRICAS, index=False)
    print(f"\nContrato completo 24-72h para OE3 guardado en: {RUTA_CONTRATO}")
    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
