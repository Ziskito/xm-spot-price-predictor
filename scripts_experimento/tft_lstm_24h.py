# -*- coding: utf-8 -*-
"""
Punto 5 -- TFT (Temporal Fusion Transformer) como votante del ensamble de 24h, y el intento pendiente
del LSTM (arquitectura recurrente que usa la tesis UNAD de Giraldo & Ramirez 2023).

Protocolo identico al Origen 6 del notebook 10 (o6_comun.py), para que las predicciones queden
alineadas hora a hora con los 5 votantes actuales. La configuracion del TFT se elige con
tft_benchmark.py y se pasa por argumentos: hidden_size, windows_batch_size, max_steps, modelos.
Uso: python tft_lstm_24h.py 32 256 1000 ambos|tft|lstm
"""
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, particion_o6, df_neural, HIST, FUT, RES

RUTA = RES / "tft_lstm_24h_2026.csv"


def main():
    hidden = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    wbs = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    pasos = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    cuales = sys.argv[4] if len(sys.argv) > 4 else "ambos"
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT, LSTM

    train, test = particion_o6(cargar_completo())
    nf_df, n_win = df_neural(train, test)
    comunes = dict(h=24, input_size=168, hist_exog_list=HIST, futr_exog_list=FUT, random_seed=42,
                   enable_progress_bar=False, val_check_steps=100)
    modelos = []
    if cuales in ("tft", "ambos"):
        modelos.append(("TFT", lambda: TFT(hidden_size=hidden, n_head=4, dropout=0.1, windows_batch_size=wbs,
                                           max_steps=pasos, **comunes)))
    if cuales in ("lstm", "ambos"):
        modelos.append(("LSTM", lambda: LSTM(max_steps=1000, **comunes)))

    salida = pd.read_csv(RUTA, parse_dates=["fecha_hora"]) if RUTA.exists() else None
    for nombre, crear in modelos:
        print(f"Entrenando {nombre} (ventanas={n_win})...", flush=True)
        t0 = time.time()
        cv = NeuralForecast(models=[crear()], freq="h").cross_validation(df=nf_df, n_windows=n_win, step_size=24)
        s = cv[["ds", "y", nombre]].rename(columns={"ds": "fecha_hora", "y": "real"})
        e = np.abs(s["real"] - s[nombre])
        print(f"{nombre}: {(time.time()-t0)/60:.1f} min  MAE={e.mean():.2f}  MAPE={(e/s['real']).mean()*100:.2f}%", flush=True)
        if salida is None:
            salida = s
        else:
            salida = salida.drop(columns=[nombre], errors="ignore").merge(s[["fecha_hora", nombre]], on="fecha_hora", how="outer")
        salida.to_csv(RUTA, index=False)
    print(f"Guardado: {RUTA}")


if __name__ == "__main__":
    main()
