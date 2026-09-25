# -*- coding: utf-8 -*-
"""
Mismo tratamiento que le dimos a junio-2026, pero para julio-2026 -- descubierto hoy via rMAE
como el mes donde N-BEATSx agrega MENOS valor relativo sobre el naive semanal (rMAE=0.584, peor
del anio), aunque su MAPE (10.50%) no se veia alarmante.

Origen 6 del walk-forward ya cubre julio 2026 (test hasta 2026-08-05), asi que no hace falta
reentrenar nada -- se reusa walkforward_predicciones_crudas.csv.
"""
import pandas as pd
import numpy as np
import holidays
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
festivos_co = set(holidays.CO(years=range(2019, 2028)).keys())

df = pd.read_csv(RAIZ / "data/processed/resultados/walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
nb = df[(df["origen"] == "Origen 6") & (df["modelo"] == "N-BEATSx")].copy()
nb = nb[(nb["fecha_hora"] >= "2026-07-01") & (nb["fecha_hora"] < "2026-08-01")].copy()

nb["ape"] = (nb["real"] - nb["prediccion"]).abs() / nb["real"] * 100
nb["error_signed"] = nb["real"] - nb["prediccion"]
nb["sesgo_pct"] = nb["error_signed"] / nb["real"] * 100
nb["dia_semana"] = nb["fecha_hora"].dt.day_name()
nb["es_domingo"] = nb["dia_semana"] == "Sunday"
nb["es_festivo"] = nb["fecha_hora"].dt.date.map(lambda d: d in festivos_co)
nb["hora"] = nb["fecha_hora"].dt.hour
nb["dia"] = nb["fecha_hora"].dt.normalize()

print("=" * 80)
print("JULIO 2026 -- N-BEATSx, diagnostico completo")
print("=" * 80)
print(f"\nMAPE general julio: {nb['ape'].mean():.2f}%")
print(f"MAPE domingos:       {nb[nb['es_domingo']]['ape'].mean():.2f}%")
print(f"MAPE festivos:       {nb[nb['es_festivo']]['ape'].mean():.2f}%  ({nb['es_festivo'].sum()} horas festivas)")
print(f"MAPE resto:          {nb[~nb['es_domingo'] & ~nb['es_festivo']]['ape'].mean():.2f}%")

print("\n--- MAPE por dia de la semana ---")
orden_dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
print(nb.groupby("dia_semana")["ape"].mean().reindex(orden_dias).round(2))

print("\n--- MAPE por hora del dia (top 5 peores) ---")
por_hora = nb.groupby("hora")["ape"].mean().sort_values(ascending=False)
print(por_hora.head(5).round(2))
print("\n--- MAPE por hora del dia (top 5 mejores) ---")
print(por_hora.tail(5).round(2))

print("\n--- Top 10 peores dias de julio (por MAPE diario) ---")
por_dia = nb.groupby("dia").agg(mape=("ape", "mean"), sesgo=("sesgo_pct", "mean")).sort_values("mape", ascending=False)
por_dia["dia_semana"] = por_dia.index.day_name()
por_dia["es_festivo"] = por_dia.index.map(lambda d: d.date() in festivos_co)
print(por_dia.head(10).round(2).to_string())

print("\n--- Rachas de dias consecutivos malos (MAPE>25%) ---")
por_dia_s = nb.groupby("dia")["ape"].mean().sort_index()
malo = por_dia_s > 25
grupo = (malo != malo.shift()).cumsum()
rachas = por_dia_s[malo].groupby(grupo[malo]).apply(lambda x: (x.index.min(), x.index.max(), len(x)))
print(f"({malo.sum()} dias con MAPE>25% de {len(por_dia_s)} dias totales)")
if len(rachas) == 0:
    print("  (ninguna racha)")
for inicio, fin, n in rachas:
    print(f"  {inicio.date()} a {fin.date()}  ({n} dias seguidos)")

print("\n--- Sesgo por decil de precio (se invierte en el decil mas caro?) ---")
nb["decil"] = pd.qcut(nb["real"], 10, labels=[f"D{i+1}" for i in range(10)], duplicates="drop")
resumen = nb.groupby("decil", observed=True)["sesgo_pct"].mean()
print(resumen.round(2))

# --- Contexto: comparar contra junio para ver si el patron es distinto en calidad, no solo en magnitud ---
jun = df[(df["origen"] == "Origen 6") & (df["modelo"] == "N-BEATSx")].copy()
jun = jun[(jun["fecha_hora"] >= "2026-06-01") & (jun["fecha_hora"] < "2026-07-01")].copy()
jun["ape"] = (jun["real"] - jun["prediccion"]).abs() / jun["real"] * 100
print(f"\n--- Comparacion rapida con junio ---")
print(f"MAPE junio: {jun['ape'].mean():.2f}%  |  MAPE julio: {nb['ape'].mean():.2f}%")
print(f"Desviacion estandar del error abs. junio: {(jun['real']-jun['prediccion']).abs().std():.2f}  |  julio: {(nb['real']-nb['prediccion']).abs().std():.2f}")
