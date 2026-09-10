# -*- coding: utf-8 -*-
"""
Prueba de causalidad de Granger entre las variables hidrologicas del pipeline
(aportes_hidricos, volumen_embalses) y el precio de bolsa -- metodologia tomada de Emre (2025,
IJEPES), quien la uso para el mercado turco (sin causalidad significativa encontrada alli, por
razones estructurales especificas de su mercado -- YEKDEM). Aqui se replica sobre nuestros propios
datos para tener el mismo tipo de evidencia formal, no solo correlacion.

Se corre sobre la serie DIARIA (no horaria) para evitar el ruido/autocorrelacion excesiva de la
resolucion horaria, que puede inflar artificialmente la significancia en un test de Granger.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora")

df["fecha"] = df["fecha_hora"].dt.normalize()
diario = df.groupby("fecha").agg(
    precio_bolsa=("precio_bolsa", "mean"),
    aportes_hidricos=("aportes_hidricos", "mean"),
    volumen_embalses=("volumen_embalses", "mean"),
).dropna()

print(f"Serie diaria: {len(diario)} dias\n")

MAX_LAG = 7  # hasta una semana de rezago

for var in ["aportes_hidricos", "volumen_embalses"]:
    print(f"{'='*70}\nGranger: {var} -> precio_bolsa  (H0: {var} NO causa precio_bolsa)\n{'='*70}")
    datos = diario[["precio_bolsa", var]].dropna()
    resultado = grangercausalitytests(datos, maxlag=MAX_LAG)
    for lag in range(1, MAX_LAG + 1):
        p_valor = resultado[lag][0]["ssr_ftest"][1]
        marca = "  <-- significativo (p<0.05)" if p_valor < 0.05 else ""
        print(f"  lag={lag}d   p-value={p_valor:.4f}{marca}")
    print()

print("Referencia -- Emre (2025) en el mercado turco: TODOS los p-values > 0.16 (sin causalidad")
print("significativa para ninguna variable de nivel de agua). Comparar contra nuestros resultados.")
