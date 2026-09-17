# -*- coding: utf-8 -*-
"""
Preparacion de datos IDENTICA al Origen 6 del notebook 10 (walk-forward 2026): entrenamiento con todo
lo anterior a 2026 (sin las filas con features faltantes), prueba hasta 2026-08-05 00:00, ventanas de
24h de 01:00 a 00:00 con corte a las 00:00. Cualquier modelo nuevo de 24h que use estas funciones
queda alineado hora a hora con walkforward_predicciones_crudas.csv y puede entrar al ensamble.
"""
import pandas as pd
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
EXCLUIR = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora", "dia_semana", "dia_anio"]
HIST = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUT_CONT = ["oni"]
FUT_BIN = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos", "es_festivo", "festivo_lag24h",
           "festivo_lag48h", "festivo_lag72h", "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]
FUT = FUT_CONT + FUT_BIN
REGRESORAS_ARX = ["aportes_hidricos", "volumen_embalses", "oni", "es_pandemia", "hora_sin", "hora_cos",
                  "dia_semana_sin", "dia_semana_cos", "es_festivo", "festivo_lag24h", "festivo_lag48h",
                  "festivo_lag72h", "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]
CORTE = "2026-01-01"
TEST_FIN = "2026-08-05"


def cargar_completo():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    return pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)


def columnas_features(df):
    return [c for c in df.columns if c not in EXCLUIR]


def particion_o6(df, inicio_train=None):
    cf = columnas_features(df)
    train = df[df["fecha_hora"] < CORTE].dropna(subset=cf).copy()
    if inicio_train is not None:
        train = train[train["fecha_hora"] >= pd.Timestamp(inicio_train)]
    test = df[(df["fecha_hora"] >= CORTE) & (df["fecha_hora"] <= TEST_FIN)].copy()
    return train, test


def df_neural(train, test, hist=HIST, fut_cont=FUT_CONT, fut_bin=FUT_BIN):
    """Mismo escalado que el notebook 10: media/desv. solo del periodo de entrenamiento."""
    o = pd.concat([train, test], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    m = (o["fecha_hora"] < CORTE).to_numpy()
    fut = fut_cont + fut_bin
    nf = o[["fecha_hora", "precio_bolsa"] + hist + fut].copy()
    for c in hist + fut_cont:
        mu, sd = nf.loc[m, c].mean(), nf.loc[m, c].std()
        nf[c] = (nf[c] - mu) / sd
    nf["unique_id"] = "precio_bolsa"
    nf = nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})[["unique_id", "ds", "y"] + hist + fut]
    return nf, int((~m).sum()) // 24


def crudas_o6():
    c = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    o = c[c["origen"] == "Origen 6"]
    p = o.pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
    return p.join(o.groupby("fecha_hora")["real"].first())
