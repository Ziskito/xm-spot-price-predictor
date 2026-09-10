# -*- coding: utf-8 -*-
"""
Prueba en N-BEATSx las variables nuevas que pasaron el tamiz de fuga/redundancia
(scripts_experimento/tamizar_variables_nuevas.py).

Todas se rezagan 24h antes de entrar al modelo, igual que la hidrologia y la demanda del
pipeline actual. Ese rezago es lo que hace legitimo usar variables como el costo marginal
del despacho programado: aunque se calculan en el proceso de despacho, su valor de hace
24h ya es informacion publica y disponible al momento de pronosticar.

Configuraciones: BASE, cada candidata por separado, y todas juntas.
Dos semillas por configuracion para poder distinguir efecto real de ruido de inicializacion.
"""
import pandas as pd
import numpy as np
import time
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_TAMIZ = RAIZ / "data" / "processed" / "resultados" / "tamiz_variables_nuevas.csv"
RUTA_SALIDA = RAIZ / "data" / "processed" / "resultados" / "variables_nuevas_nbeatsx.csv"

FECHA_TEST = pd.Timestamp("2026-01-01")
HORIZONTE, INPUT_SIZE, MAX_STEPS = 24, 168, 1000
SEMILLAS = [42, 7]
REZAGO = 24

HIST_EXOG_BASE = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUTR_EXOG_CONT = ["oni"]
FUTR_EXOG_BIN = ["es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos"]

ARCHIVOS = {
    "costo_marginal_despacho": "costo_marginal_despacho_2019_2026.csv",
    "generacion_ideal": "generacion_ideal_2019_2026.csv",
    "demanda_comercial": "demanda_comercial_2019_2026.csv",
    "restricciones_aliviadas": "restricciones_aliviadas_2019_2026.csv",
    "perdidas_energia": "perdidas_energia_2019_2026.csv",
    "importaciones_energia": "importaciones_energia_2019_2026.csv",
    "exportaciones_energia": "exportaciones_energia_2019_2026.csv",
    "precio_escasez": "precio_escasez_2019_2026.csv",
    "precio_escasez_marginal": "precio_escasez_marginal_2019_2026.csv",
    "aportes_media_historica": "aportes_media_historica_2019_2026.csv",
    "compras_arranque_parada": "compras_arranque_parada_2019_2026.csv",
    "disponibilidad_termica": "disponibilidad_por_tipo_2019_2026.csv",
}


def cargar_base():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    return pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)


def agregar_variable(df, nombre, archivo):
    """Une la variable, la rezaga 24h y agrega una media movil de 7 dias (tambien rezagada)."""
    ruta = RAIZ / "data" / archivo
    if not ruta.exists():
        return df, []

    nueva = pd.read_csv(ruta, parse_dates=["fecha_hora"])
    if nombre == "disponibilidad_termica":
        col_termica = next((c for c in nueva.columns if "TERMICA" in c.upper()), None)
        if col_termica is None:
            return df, []
        nueva = nueva[["fecha_hora", col_termica]].rename(columns={col_termica: nombre})
    else:
        col = [c for c in nueva.columns if c != "fecha_hora"][0]
        nueva = nueva[["fecha_hora", col]].rename(columns={col: nombre})

    es_diaria = nueva["fecha_hora"].dt.hour.nunique() == 1
    if es_diaria:
        df = pd.merge_asof(df.sort_values("fecha_hora"), nueva.sort_values("fecha_hora"),
                            on="fecha_hora", direction="backward")
    else:
        df = df.merge(nueva, on="fecha_hora", how="left")

    df[nombre] = df[nombre].ffill()
    df[f"{nombre}_lag"] = df[nombre].shift(REZAGO)
    df[f"{nombre}_media7d"] = df[nombre].rolling(168, min_periods=24).mean().shift(REZAGO)
    return df, [f"{nombre}_lag", f"{nombre}_media7d"]


def correr(df, hist_exog_extra, semilla):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx

    hist_exog = HIST_EXOG_BASE + hist_exog_extra
    futr_exog = FUTR_EXOG_CONT + FUTR_EXOG_BIN

    d = df.dropna(subset=hist_exog + futr_exog + ["precio_bolsa"]).reset_index(drop=True)
    df_nf = d[["fecha_hora", "precio_bolsa"] + hist_exog + futr_exog].copy()
    mascara_train = d["fecha_hora"] < FECHA_TEST

    for col in hist_exog + FUTR_EXOG_CONT:
        mu = df_nf.loc[mascara_train.values, col].mean()
        sigma = df_nf.loc[mascara_train.values, col].std()
        if sigma == 0 or pd.isna(sigma):
            sigma = 1.0
        df_nf[col] = (df_nf[col] - mu) / sigma

    df_nf["unique_id"] = "precio_bolsa"
    df_nf = df_nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})
    df_nf = df_nf[["unique_id", "ds", "y"] + hist_exog + futr_exog]

    n_test = int((~mascara_train).sum())
    modelo = NBEATSx(h=HORIZONTE, input_size=INPUT_SIZE, hist_exog_list=hist_exog,
                      futr_exog_list=futr_exog, max_steps=MAX_STEPS, val_check_steps=100,
                      random_seed=semilla, enable_progress_bar=False)
    nf = NeuralForecast(models=[modelo], freq="h")
    cv = nf.cross_validation(df=df_nf, n_windows=n_test // HORIZONTE, step_size=HORIZONTE)
    return cv[["ds", "y", "NBEATSx"]]


def main():
    df = cargar_base()

    # Que candidatas pasaron el tamiz
    if RUTA_TAMIZ.exists():
        tamiz = pd.read_csv(RUTA_TAMIZ)
        candidatas = tamiz[tamiz["veredicto"].str.contains("CANDIDATA|debil", na=False)]["variable"].tolist()
    else:
        candidatas = ["costo_marginal_despacho"]
    candidatas = [c for c in candidatas if c in ARCHIVOS]
    print(f"Candidatas a probar: {candidatas}\n")

    columnas_por_variable = {}
    for nombre in candidatas:
        df, cols = agregar_variable(df, nombre, ARCHIVOS[nombre])
        if cols:
            columnas_por_variable[nombre] = cols
            print(f"  {nombre}: agregadas {cols}")

    configuraciones = [("BASE", [])]
    for nombre, cols in columnas_por_variable.items():
        configuraciones.append((nombre, cols))
    if len(columnas_por_variable) > 1:
        todas = [c for cols in columnas_por_variable.values() for c in cols]
        configuraciones.append(("TODAS", todas))

    filas = []
    t0 = time.time()
    for etiqueta, extra in configuraciones:
        for semilla in SEMILLAS:
            t = time.time()
            cv = correr(df, extra, semilla)
            err = np.abs(cv["y"].values - cv["NBEATSx"].values)
            filas.append({"config": etiqueta, "semilla": semilla, "mae": err.mean(),
                           "rmse": np.sqrt(((cv["y"].values - cv["NBEATSx"].values) ** 2).mean()),
                           "mape": (err / cv["y"].values).mean() * 100,
                           "n_exog": len(HIST_EXOG_BASE) + len(extra),
                           "segundos": time.time() - t})
            print(f"  {etiqueta:28s} semilla {semilla:4d}  MAE {err.mean():6.2f}  "
                  f"MAPE {filas[-1]['mape']:5.2f}%  ({time.time()-t:.0f}s)")
            sys.stdout.flush()
            pd.DataFrame(filas).to_csv(RUTA_SALIDA, index=False)

    res = pd.DataFrame(filas)
    print(f"\n{'='*70}\nRESUMEN (media +- std sobre semillas)\n{'='*70}")
    resumen = res.groupby("config")["mae"].agg(["mean", "std"]).sort_values("mean")
    print(resumen.round(3).to_string())
    base = resumen.loc["BASE", "mean"]
    print(f"\nDiferencia contra BASE ({base:.2f}):")
    for cfg, fila in resumen.iterrows():
        if cfg != "BASE":
            print(f"  {cfg:28s} {fila['mean']-base:+6.2f}  (std entre semillas: {fila['std']:.2f})")
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
