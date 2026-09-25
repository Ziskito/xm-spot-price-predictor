# -*- coding: utf-8 -*-
"""
Medicion de costo antes de reintentar el TFT (Temporal Fusion Transformer).

Anoche el TFT se aborto tras 2+ horas sin terminar ni la primera fase de validacion cruzada.
Aqui se mide el tiempo por paso de entrenamiento de varias configuraciones con pocos pasos, para
extrapolar el costo de las 1000 iteraciones completas ANTES de lanzar el entrenamiento real.
Se incluyen tambien el LSTM (arquitectura recurrente, la que usa la tesis UNAD de Giraldo &
Ramirez 2023) y N-BEATSx como referencia de lo que ya se sabe que corre en ~2.5 minutos.
"""
import time
import inspect
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
HIST = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUT_CONT = ["oni"]
FUT_BIN = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos", "es_festivo", "festivo_lag24h",
           "festivo_lag48h", "festivo_lag72h", "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]
FUT = FUT_CONT + FUT_BIN
PASOS = 30


def datos_train():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    df = a[["fecha_hora", "precio_bolsa"] + HIST + FUT].copy()
    for c in HIST + FUT_CONT:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    df["unique_id"] = "precio_bolsa"
    return df.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})[["unique_id", "ds", "y"] + HIST + FUT]


def main():
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT, LSTM, NBEATSx
    print("LSTM acepta:", [p for p in inspect.signature(LSTM.__init__).parameters][:30])
    df = datos_train()
    comunes = dict(h=24, input_size=168, hist_exog_list=HIST, futr_exog_list=FUT, max_steps=PASOS,
                   random_seed=42, enable_progress_bar=False, val_check_steps=100000)
    configs = [
        ("TFT hidden=64, windows_batch=1024 (config de anoche)", lambda: TFT(hidden_size=64, n_head=4, dropout=0.1, **comunes)),
        ("TFT hidden=32, windows_batch=256", lambda: TFT(hidden_size=32, n_head=4, dropout=0.1, windows_batch_size=256, **comunes)),
        ("TFT hidden=32, windows_batch=128", lambda: TFT(hidden_size=32, n_head=4, dropout=0.1, windows_batch_size=128, **comunes)),
        ("LSTM (defaults)", lambda: LSTM(**comunes)),
        ("N-BEATSx (referencia)", lambda: NBEATSx(**comunes)),
    ]
    filas = []
    for nombre, crear in configs:
        nf = NeuralForecast(models=[crear()], freq="h")
        t0 = time.time()
        nf.fit(df=df)
        seg = time.time() - t0
        min_1000 = seg / PASOS * 1000 / 60
        print(f"{nombre:52s} {seg/PASOS:6.2f} s/paso  ->  1000 pasos ~ {min_1000:6.1f} min", flush=True)
        filas.append({"config": nombre, "seg_por_paso": seg / PASOS, "min_1000_pasos": min_1000})
    pd.DataFrame(filas).to_csv(RAIZ / "data/processed/resultados/tft_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
