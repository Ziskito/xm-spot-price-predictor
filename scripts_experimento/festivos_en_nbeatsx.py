# -*- coding: utf-8 -*-
"""
Prueba de la mejor codificacion de festivos (F2: flags de contaminacion de rezagos) sobre
N-BEATSx, que es el modelo recomendado del proyecto -- no sobre XGBoost, que solo se uso
para tamizar configuraciones rapido.

Se corren VARIAS SEMILLAS por configuracion porque N-BEATSx es estocastico: una diferencia
de ~0.5 MAE entre dos corridas de una sola semilla puede ser puro ruido de inicializacion.
Con 3 semillas se puede comparar la media y ver si la diferencia sobrevive a la dispersion.

REVERSIBLE: todo en scripts_experimento/. Resultados en
data/processed/resultados/festivos_nbeatsx.csv
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
RUTA_SALIDA = RAIZ / "data" / "processed" / "resultados" / "festivos_nbeatsx.csv"

FECHA_TEST = "2026-01-01"
HORIZONTE = 24
INPUT_SIZE = 168
MAX_STEPS = 1000
SEMILLAS = [13, 99]  # extension de 3->5 semillas totales para asentar la lectura de festivos

# Exogenas del pipeline oficial de la notebook 09
HIST_EXOG_BASE = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUTR_EXOG_CONT_BASE = ["oni"]
FUTR_EXOG_BIN_BASE = ["es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos"]

# F2: los festivos son calendario, se conocen con anios de anticipacion -> futr_exog (no hay fuga)
FUTR_EXOG_FESTIVOS = ["es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
                       "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]


def cargar_datos():
    df_a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    df_b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([df_a, df_b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

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


def correr(df, usar_festivos, semilla):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx

    hist_exog = list(HIST_EXOG_BASE)
    futr_exog_cont = list(FUTR_EXOG_CONT_BASE)
    futr_exog_bin = list(FUTR_EXOG_BIN_BASE)
    if usar_festivos:
        futr_exog_bin = futr_exog_bin + FUTR_EXOG_FESTIVOS
    futr_exog = futr_exog_cont + futr_exog_bin

    df_nf = df[["fecha_hora", "precio_bolsa"] + hist_exog + futr_exog].copy()
    mascara_train = df["fecha_hora"] < FECHA_TEST
    for col in hist_exog + futr_exog_cont:
        mu = df_nf.loc[mascara_train.values, col].mean()
        sigma = df_nf.loc[mascara_train.values, col].std()
        df_nf[col] = (df_nf[col] - mu) / sigma

    df_nf["unique_id"] = "precio_bolsa"
    df_nf = df_nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})
    df_nf = df_nf[["unique_id", "ds", "y"] + hist_exog + futr_exog]

    n_test = int((~mascara_train).sum())
    n_windows = n_test // HORIZONTE

    modelo = NBEATSx(h=HORIZONTE, input_size=INPUT_SIZE, hist_exog_list=hist_exog,
                      futr_exog_list=futr_exog, max_steps=MAX_STEPS, val_check_steps=100,
                      random_seed=semilla, enable_progress_bar=False)
    nf = NeuralForecast(models=[modelo], freq="h")
    cv = nf.cross_validation(df=df_nf, n_windows=n_windows, step_size=HORIZONTE)
    return cv[["ds", "y", "NBEATSx"]]


def main():
    df = cargar_datos()
    festivo_por_fecha = df.set_index("fecha_hora")["es_festivo"]

    filas = []
    predicciones = {}
    t_global = time.time()

    for usar_festivos in [False, True]:
        etiqueta = "CON_FESTIVOS" if usar_festivos else "BASE"
        for semilla in SEMILLAS:
            t0 = time.time()
            cv = correr(df, usar_festivos, semilla)
            err = np.abs(cv["y"].values - cv["NBEATSx"].values)
            mask_fest = festivo_por_fecha.loc[cv["ds"]].values == 1

            fila = {
                "config": etiqueta,
                "semilla": semilla,
                "mae": err.mean(),
                "rmse": np.sqrt(((cv["y"].values - cv["NBEATSx"].values) ** 2).mean()),
                "mape": (err / cv["y"].values).mean() * 100,
                "mae_festivos": err[mask_fest].mean(),
                "mae_no_festivos": err[~mask_fest].mean(),
                "segundos": time.time() - t0,
            }
            filas.append(fila)
            predicciones[f"{etiqueta}_{semilla}"] = cv["NBEATSx"].values
            print(f"{etiqueta:14s} semilla {semilla:5d}  MAE {fila['mae']:6.2f}  "
                  f"MAPE {fila['mape']:5.2f}%  festivos {fila['mae_festivos']:6.2f}  "
                  f"({fila['segundos']:.0f}s)")
            sys.stdout.flush()
            pd.DataFrame(filas).to_csv(RUTA_SALIDA, index=False)

    df_res = pd.DataFrame(filas)

    ruta_parte1 = RAIZ / "data/processed/resultados/festivos_nbeatsx_parte1.csv"
    if ruta_parte1.exists():
        df_res = pd.concat([pd.read_csv(ruta_parte1), df_res], ignore_index=True)
        df_res.to_csv(RUTA_SALIDA, index=False)

    # Ensamble por configuracion (promedio de las 5 semillas), que es como se usa en la 09
    cv_ref = correr.__globals__  # no usado; se recalcula abajo con la ultima cv
    print("\n" + "=" * 70)
    print("RESUMEN POR CONFIGURACION (media +- desviacion sobre las 3 semillas)")
    print("=" * 70)
    resumen = df_res.groupby("config")[["mae", "mape", "mae_festivos", "mae_no_festivos"]].agg(["mean", "std"])
    print(resumen.round(3).to_string())

    base_mae = df_res[df_res["config"] == "BASE"]["mae"]
    fest_mae = df_res[df_res["config"] == "CON_FESTIVOS"]["mae"]
    print(f"\nMAE BASE:          {base_mae.mean():.2f} +- {base_mae.std():.2f}  (semillas: {list(base_mae.round(2))})")
    print(f"MAE CON_FESTIVOS:  {fest_mae.mean():.2f} +- {fest_mae.std():.2f}  (semillas: {list(fest_mae.round(2))})")
    print(f"Diferencia de medias: {fest_mae.mean() - base_mae.mean():+.2f} MAE")
    print("\nLectura: si la diferencia de medias es menor que la desviacion entre semillas,")
    print("el efecto no se distingue del ruido de inicializacion de la red.")

    df_res.to_csv(RUTA_SALIDA, index=False)
    print(f"\nLISTO en {(time.time()-t_global)/60:.1f} min | Guardado: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
