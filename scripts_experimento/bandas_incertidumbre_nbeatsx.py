# -*- coding: utf-8 -*-
"""
Bandas de incertidumbre para N-BEATSx usando MQLoss (perdida multi-cuantil nativa de
neuralforecast). Cierra el hueco pendiente del backlog: hoy solo ARX+GARCH y XGBoost tienen
bandas, y el motor de decision (OE3) las necesita para definir reglas por percentil.

Salida principal -- el "contrato" que consume OE3:
  data/processed/resultados/pronostico_con_bandas_2026.csv
  columnas: fecha_hora, real, q10, q50, q90

Se valida la calidad de las bandas con:
  - COBERTURA: que fraccion de los valores reales cae dentro de [q10, q90].
    Deberia ser ~80%. Si es mucho menor, las bandas son demasiado angostas (exceso de
    confianza) y el motor de decision tomaria riesgos mal medidos.
  - PERDIDA PINBALL: metrica propia de pronostico por cuantiles.
  - MAE del cuantil 50 (mediana) como pronostico puntual, comparable con el N-BEATSx normal.
"""
import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_CONTRATO = RAIZ / "data" / "processed" / "resultados" / "pronostico_con_bandas_2026.csv"
RUTA_METRICAS = RAIZ / "data" / "processed" / "resultados" / "bandas_metricas.csv"

FECHA_TEST = pd.Timestamp("2026-01-01")
HORIZONTE, INPUT_SIZE, MAX_STEPS = 24, 168, 1000
CUANTILES = [0.1, 0.5, 0.9]

HIST_EXOG = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUTR_EXOG_CONT = ["oni"]
FUTR_EXOG_BIN = ["es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos"]


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
    print(f"Entrenando N-BEATSx con MQLoss (cuantiles {CUANTILES})...")
    print(f"Train: {int(mascara_train.sum()):,} | Test 2026: {n_test:,}")

    modelo = NBEATSx(h=HORIZONTE, input_size=INPUT_SIZE, hist_exog_list=HIST_EXOG,
                      futr_exog_list=futr_exog, max_steps=MAX_STEPS, val_check_steps=100,
                      loss=MQLoss(quantiles=CUANTILES), random_seed=42,
                      enable_progress_bar=False)
    nf = NeuralForecast(models=[modelo], freq="h")
    cv = nf.cross_validation(df=df_nf, n_windows=n_test // HORIZONTE, step_size=HORIZONTE)

    print("\nColumnas devueltas:", [c for c in cv.columns])

    # neuralforecast nombra las columnas segun el cuantil, ej NBEATSx-median, NBEATSx-lo-80...
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

    salida = pd.DataFrame({
        "fecha_hora": cv["ds"].values,
        "real": cv["y"].values,
        "q10": cv[col_q10].values if col_q10 else np.nan,
        "q50": cv[col_q50].values,
        "q90": cv[col_q90].values if col_q90 else np.nan,
    })
    salida.to_csv(RUTA_CONTRATO, index=False)

    y = salida["real"].values
    mae_q50 = np.abs(y - salida["q50"].values).mean()
    mape_q50 = (np.abs(y - salida["q50"].values) / y).mean() * 100

    print(f"\n{'='*66}")
    print(f"Pronostico puntual (cuantil 50):  MAE {mae_q50:.2f}  MAPE {mape_q50:.2f}%")
    print(f"  (referencia N-BEATSx normal: MAE 56.11, ensamble 54.89)")

    filas_met = [{"metrica": "mae_q50", "valor": mae_q50}, {"metrica": "mape_q50", "valor": mape_q50}]

    if col_q10 and col_q90:
        dentro = ((y >= salida["q10"].values) & (y <= salida["q90"].values)).mean() * 100
        ancho = (salida["q90"].values - salida["q10"].values).mean()
        print(f"\nCOBERTURA de la banda [q10, q90]: {dentro:.1f}%   (objetivo: 80%)")
        print(f"Ancho promedio de la banda: {ancho:.1f} COP/kWh")
        if dentro < 70:
            print("  -> Bandas DEMASIADO ANGOSTAS: el modelo esta sobre-confiado.")
        elif dentro > 90:
            print("  -> Bandas demasiado anchas: poco informativas para decidir.")
        else:
            print("  -> Cobertura razonable.")
        for tau, col in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
            p = perdida_pinball(y, salida[col].values, tau)
            print(f"  pinball tau={tau}: {p:.3f}")
            filas_met.append({"metrica": f"pinball_{tau}", "valor": p})
        filas_met += [{"metrica": "cobertura_80", "valor": dentro},
                       {"metrica": "ancho_banda", "valor": ancho}]

    pd.DataFrame(filas_met).to_csv(RUTA_METRICAS, index=False)
    print(f"\nContrato para OE3 guardado en: {RUTA_CONTRATO}")
    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
