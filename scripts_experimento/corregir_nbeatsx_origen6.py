# -*- coding: utf-8 -*-
"""
Corrige el bug de corrimiento de 1 hora en N-BEATSx/N-HiTS del Origen 6.

Bug encontrado: `test_fin: "2026-08-05"` (sin hora, = 00:00) hace que el rango de prueba del
Origen 6 tenga 5185 horas -- no divisible entre 24. `neuralforecast.cross_validation` arma sus
216 ventanas de 24h CONTANDO DESDE EL FINAL del dataframe hacia atras, asi que con
n_windows = 5185 // 24 = 216 (division entera) se queda corto por exactamente 1 hora, y esa hora
que se pierde es la PRIMERA del rango (2026-01-01 00:00) -- no la ultima. Consecuencia: en las
216 ventanas del anio, la hora calendario "1" pasa a ser el primer paso de cada ventana (el mas
cercano al corte, el mas facil) y la hora calendario "0" pasa a ser el ULTIMO paso de la ventana
ANTERIOR (el mas lejano, el mas dificil) -- un corrimiento sistematico, no un patron real del
mercado a la medianoche.

Los otros 3 modelos del walk-forward (Persistencia, XGBoost, ARX+GARCH) se calculan con un bucle
explicito hora por hora, sin pasar por el mecanismo de ventanas de neuralforecast, asi que no
tienen este problema (cubren las 5185 horas completas, empezando en 2026-01-01 00:00).

Arreglo: usar test_fin = "2026-08-04 23:00" (mismo criterio que los Origenes 1-5, que SI
especifican la hora exacta del ultimo punto) en vez de "2026-08-05" a secas. Eso deja el rango en
exactamente 5184 horas = 216*24, multiplo exacto, y neuralforecast arma las 216 ventanas sin
perder ninguna hora del inicio. Se pierde solo la ultima hora (2026-08-05 00:00) del rango
original, que ya era una hora "extra" fuera del patron de los demas origenes.

Mismos hiperparametros que notebooks/10_diebold_mariano_juan.ipynb (la version con festivos, que
es la que alimenta el pipeline desplegado via walkforward_predicciones_crudas.csv): input_size=168,
max_steps=1000, semilla 42, hist_exog=[volumen_embalses, aportes_hidricos, demanda_lag24h],
futr_exog=[oni + calendario + festivos].
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
CRUDAS = RES / "walkforward_predicciones_crudas.csv"

CORTE_TRAIN = "2026-01-01"
TEST_INICIO = "2026-01-01"
TEST_FIN_ORIGINAL = "2026-08-05"          # bug: sin hora, = 00:00, da 5185 horas
TEST_FIN_CORREGIDO = "2026-08-04 23:00"   # arreglo: 5184 horas, multiplo exacto de 24

HIST_EXOG_DL = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUTR_EXOG_CONTINUA_DL = ["oni"]
FUTR_EXOG_YA_ESCALADA_DL = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
                            "es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
                            "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]
FUTR_EXOG_DL = FUTR_EXOG_CONTINUA_DL + FUTR_EXOG_YA_ESCALADA_DL
INPUT_SIZE_DL = 168
MAX_STEPS_DL = 1000


def main():
    t0 = time.time()
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS

    df_train = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    df_test = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df_completo = pd.concat([df_train, df_test], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    columnas_excluir = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora",
                        "dia_semana", "dia_anio"]
    columnas_features = [c for c in df_completo.columns if c not in columnas_excluir]
    print(f"Filas totales: {len(df_completo):,} | features: {len(columnas_features)}")

    train = df_completo[df_completo["fecha_hora"] < CORTE_TRAIN].dropna(subset=columnas_features).copy()
    test = df_completo[(df_completo["fecha_hora"] >= TEST_INICIO) &
                       (df_completo["fecha_hora"] <= TEST_FIN_CORREGIDO)].copy()
    n_test = len(test)
    print(f"Train: {len(train):,} filas | Test (corregido): {n_test:,} filas "
          f"({test['fecha_hora'].min()} .. {test['fecha_hora'].max()})")
    assert n_test % 24 == 0, f"n_test={n_test} sigue sin ser multiplo de 24, revisar"
    n_windows = n_test // 24
    print(f"n_windows = {n_windows} (exacto, sin resto)\n")

    df_origen = pd.concat([train, test], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    mascara_train = df_origen["fecha_hora"] < CORTE_TRAIN

    df_nf = df_origen[["fecha_hora", "precio_bolsa"] + HIST_EXOG_DL + FUTR_EXOG_DL].copy()
    for col in HIST_EXOG_DL + FUTR_EXOG_CONTINUA_DL:
        mu, sigma = df_nf.loc[mascara_train, col].mean(), df_nf.loc[mascara_train, col].std()
        df_nf[col] = (df_nf[col] - mu) / sigma

    df_nf["unique_id"] = "precio_bolsa"
    df_nf = df_nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})
    df_nf = df_nf[["unique_id", "ds", "y"] + HIST_EXOG_DL + FUTR_EXOG_DL]

    m_nbeatsx = NBEATSx(h=24, input_size=INPUT_SIZE_DL, hist_exog_list=HIST_EXOG_DL,
                        futr_exog_list=FUTR_EXOG_DL, max_steps=MAX_STEPS_DL, val_check_steps=100,
                        random_seed=42, enable_progress_bar=False)
    m_nhits = NHITS(h=24, input_size=INPUT_SIZE_DL, hist_exog_list=HIST_EXOG_DL,
                    futr_exog_list=FUTR_EXOG_DL, max_steps=MAX_STEPS_DL, val_check_steps=100,
                    random_seed=42, enable_progress_bar=False)
    nf = NeuralForecast(models=[m_nbeatsx, m_nhits], freq="h")

    print("Entrenando N-BEATSx + N-HiTS (Origen 6, rango corregido)...", flush=True)
    cv = nf.cross_validation(df=df_nf, n_windows=n_windows, step_size=24)
    print(f"Listo en {(time.time()-t0)/60:.1f} min. Filas de cv: {len(cv):,}")

    # ---------- verificacion del arreglo ----------
    print(f"\nVerificacion: min ds = {cv['ds'].min()} (debe ser 2026-01-01 00:00:00)")
    assert cv["ds"].min() == pd.Timestamp("2026-01-01 00:00:00"), "el arreglo no cubrio la hora 0 inicial"
    print(f"max ds = {cv['ds'].max()}")
    print(f"filas por modelo: NBEATSx={cv['NBEATSx'].notna().sum()}, NHITS={cv['NHITS'].notna().sum()}")

    # ---------- reconstruir filas en el formato de walkforward_predicciones_crudas.csv ----------
    nuevas = []
    for col_modelo, nombre_modelo in [("NBEATSx", "N-BEATSx"), ("NHITS", "N-HiTS")]:
        for fh, real, pred in zip(cv["ds"], cv["y"], cv[col_modelo]):
            nuevas.append({"origen": "Origen 6", "fecha_hora": fh, "real": real,
                           "modelo": nombre_modelo, "prediccion": pred})
    df_nuevas = pd.DataFrame(nuevas)
    df_nuevas.to_csv(RES / "nbeatsx_nhits_origen6_corregido.csv", index=False)

    # ---------- parchear walkforward_predicciones_crudas.csv ----------
    crudas = pd.read_csv(CRUDAS, parse_dates=["fecha_hora"])
    antes = len(crudas)
    mask_viejo = (crudas["origen"] == "Origen 6") & (crudas["modelo"].isin(["N-BEATSx", "N-HiTS"]))
    print(f"\nFilas viejas de N-BEATSx/N-HiTS en Origen 6 a reemplazar: {int(mask_viejo.sum())}")
    crudas_limpio = crudas[~mask_viejo].copy()
    crudas_final = pd.concat([crudas_limpio, df_nuevas], ignore_index=True).sort_values(
        ["origen", "modelo", "fecha_hora"]).reset_index(drop=True)
    crudas_final.to_csv(CRUDAS, index=False)
    print(f"Filas antes: {antes:,} | despues: {len(crudas_final):,} | "
          f"nuevas de N-BEATSx/N-HiTS: {len(df_nuevas):,}")
    print(f"\nGuardado: walkforward_predicciones_crudas.csv (parcheado) y "
          f"nbeatsx_nhits_origen6_corregido.csv (respaldo)")
    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
