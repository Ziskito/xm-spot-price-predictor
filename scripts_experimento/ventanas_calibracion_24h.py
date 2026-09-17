# -*- coding: utf-8 -*-
"""
Intento pendiente: promediar pronosticos entre VENTANAS DE CALIBRACION (Marcjasz, Serafin & Weron
2018, Energies 11(9):2364) con el pipeline ACTUAL. El experimento de ventanas del 2026-09-09 se hizo
antes de agregar los festivos (MAE ~55 contra 46.8 del pipeline actual) y no se puede reutilizar.

N-BEATSx y N-HiTS se reentrenan con solo los ultimos 3 y 5 anos antes de 2026. La ventana completa
(2019-2025) ya esta en walkforward_predicciones_crudas.csv. Idea del paper: una ventana larga capta
patrones estables y una corta se parece mas al regimen reciente; promediarlas es mas robusto que
elegir una sola -- justo lo que interesa para adaptarse a 2026.
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
from o6_comun import cargar_completo, particion_o6, df_neural, crudas_o6, HIST, FUT, RES

RUTA = RES / "ventanas_24h_actual_2026.csv"


def main():
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS
    df = cargar_completo()
    salida = None
    for anos in [3, 5]:
        inicio = pd.Timestamp("2026-01-01") - pd.DateOffset(years=anos)
        train, test = particion_o6(df, inicio_train=inicio)
        nf_df, n_win = df_neural(train, test)
        comunes = dict(h=24, input_size=168, hist_exog_list=HIST, futr_exog_list=FUT, max_steps=1000,
                       val_check_steps=100, random_seed=42, enable_progress_bar=False)
        t0 = time.time()
        cv = NeuralForecast(models=[NBEATSx(**comunes), NHITS(**comunes)], freq="h").cross_validation(
            df=nf_df, n_windows=n_win, step_size=24)
        s = cv[["ds", "NBEATSx", "NHITS"]].rename(columns={"ds": "fecha_hora", "NBEATSx": f"N-BEATSx_{anos}a",
                                                          "NHITS": f"N-HiTS_{anos}a"})
        print(f"Ventana {anos} anos (desde {inicio.date()}, {len(train):,} filas): {(time.time()-t0)/60:.1f} min", flush=True)
        salida = s if salida is None else salida.merge(s, on="fecha_hora")
        salida.to_csv(RUTA, index=False)

    c = crudas_o6()
    m = salida.set_index("fecha_hora").join(c[["N-BEATSx", "N-HiTS", "real"]], how="inner")
    real = m["real"]
    for base in ["N-BEATSx", "N-HiTS"]:
        m[f"{base}_prom"] = m[[base, f"{base}_3a", f"{base}_5a"]].mean(axis=1)
        for col in [base, f"{base}_3a", f"{base}_5a", f"{base}_prom"]:
            e = (real - m[col]).abs()
            print(f"  {col:16s} MAE={e.mean():6.2f}  MAPE={(e/real).mean()*100:5.2f}%")
    print(f"Guardado: {RUTA}")


if __name__ == "__main__":
    main()
