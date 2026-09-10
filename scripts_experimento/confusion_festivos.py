# -*- coding: utf-8 -*-
"""
La diferencia cruda de precio entre festivo y dia normal (ej. martes festivo 108 vs martes
normal 348) es real o esta confundida por el periodo en que caen esos festivos?

Un festivo que cayo en 2021 (anio barato, media 150) se compara contra martes normales de
todos los anios incluyendo 2023-2024 (media 558-676). Eso solo ya produciria una diferencia
enorme sin que el festivo tenga nada que ver.

Prueba: comparar el precio del festivo contra el nivel LOCAL de precio (media de 7 dias
centrada en esa misma fecha). Si el efecto festivo es real, el cociente debe ser < 1 de
forma consistente. Si desaparece, la diferencia cruda era un artefacto del periodo.
"""
import pandas as pd
import numpy as np
import holidays
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

df_a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
df_b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
df = pd.concat([df_a, df_b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

festivos = set(holidays.CO(years=range(2018, 2028)).keys())
df["fecha"] = df["fecha_hora"].dt.date
df["es_festivo"] = df["fecha"].map(lambda d: d in festivos).astype(int)
df["dia_semana"] = df["fecha_hora"].dt.day_name()
df["anio"] = df["fecha_hora"].dt.year

print("=" * 78)
print("1) EN QUE ANIOS CAEN LOS FESTIVOS ENTRE SEMANA? (chequeo del confundidor)")
print("=" * 78)
entre_semana = df[~df["dia_semana"].isin(["Saturday", "Sunday"])]
festivos_semana = entre_semana[entre_semana["es_festivo"] == 1]
print("Horas festivas entre semana por anio:")
print(festivos_semana.groupby("anio").size().to_string())
print("\nPrecio medio del anio (todas las horas), para comparar:")
print(df.groupby("anio")["precio_bolsa"].mean().round(1).to_string())

print()
print("=" * 78)
print("2) MARTES FESTIVOS: cuando ocurrieron exactamente?")
print("=" * 78)
martes_fest = df[(df["dia_semana"] == "Tuesday") & (df["es_festivo"] == 1)]
resumen_martes = martes_fest.groupby("fecha")["precio_bolsa"].mean().round(1)
print(resumen_martes.to_string())

print()
print("=" * 78)
print("3) EFECTO FESTIVO CONTROLANDO POR NIVEL LOCAL DE PRECIO")
print("=" * 78)
# Nivel local: media movil centrada de 7 dias (168h), que captura el regimen de esos dias
serie = df.set_index("fecha_hora")["precio_bolsa"]
df["nivel_local"] = serie.rolling(337, min_periods=48, center=True).mean().values
df["ratio"] = df["precio_bolsa"] / df["nivel_local"]

comparacion = df[df["nivel_local"].notna()].groupby(
    ["dia_semana", "es_festivo"]
)["ratio"].agg(["mean", "count"]).round(3).unstack()
print("Precio relativo al nivel local de esa semana (1.0 = igual al promedio de su semana):")
print(comparacion.to_string())

print()
print("Global, controlando por nivel local:")
glob = df[df["nivel_local"].notna()].groupby("es_festivo")["ratio"].agg(["mean", "median", "count"]).round(3)
print(glob.to_string())

print()
print("Solo dias entre semana (lunes-viernes), controlando por nivel local:")
ent = df[(df["nivel_local"].notna()) & (~df["dia_semana"].isin(["Saturday", "Sunday"]))]
print(ent.groupby("es_festivo")["ratio"].agg(["mean", "median", "count"]).round(3).to_string())

print()
print("Domingos (referencia de 'dia no laboral' natural):")
dom = df[(df["nivel_local"].notna()) & (df["dia_semana"] == "Sunday")]
print(f"  ratio medio domingo: {dom['ratio'].mean():.3f}")

print()
print("=" * 78)
print("4) LO MISMO PERO EN DEMANDA (donde el efecto festivo deberia verse claro)")
print("=" * 78)
if "demanda_lag24h" in df.columns:
    serie_dem = df.set_index("fecha_hora")["demanda_lag24h"]
    df["nivel_dem"] = serie_dem.rolling(169, min_periods=24, center=True).mean().values
    df["ratio_dem"] = df["demanda_lag24h"] / df["nivel_dem"]
    # demanda_lag24h del dia siguiente refleja la demanda del dia festivo
    ent2 = df[(df["nivel_dem"].notna()) & (~df["dia_semana"].isin(["Saturday", "Sunday"]))]
    print("Demanda relativa al nivel local (rezagada 24h, entre semana):")
    print(ent2.groupby("es_dia_despues_festivo" if "es_dia_despues_festivo" in df.columns else "es_festivo")["ratio_dem"].agg(["mean", "count"]).round(3).to_string())
