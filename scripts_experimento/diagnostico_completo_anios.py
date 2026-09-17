# -*- coding: utf-8 -*-
"""
Diagnostico completo, replicado en los 6 origenes del walk-forward (2020-2026), para ver si
los patrones encontrados en 2026 (domingo/festivo dificiles, sesgo invertido en picos, etc.)
se repiten ano a ano o son especificos de 2026.

LIMITACION IMPORTANTE: los origenes 1-5 son ventanas de solo 3 meses, y NINGUNO cae en junio
-- el patron especifico de "junio" (posible efecto de vacaciones de mitad de ano) no se puede
verificar en otros anios con estos datos, solo en Origen 6.
"""
import pandas as pd
import numpy as np
import holidays
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
festivos_co = set(holidays.CO(years=range(2019, 2028)).keys())

df = pd.read_csv(RAIZ / "data/processed/resultados/walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
df["ape"] = (df["real"] - df["prediccion"]).abs() / df["real"] * 100
df["error_signed"] = df["real"] - df["prediccion"]
df["sesgo_pct"] = df["error_signed"] / df["real"] * 100
df["dia_semana"] = df["fecha_hora"].dt.day_name()
df["es_domingo"] = df["dia_semana"] == "Sunday"
df["es_festivo"] = df["fecha_hora"].dt.date.map(lambda d: d in festivos_co)
df["hora"] = df["fecha_hora"].dt.hour
df["dia"] = df["fecha_hora"].dt.normalize()

origenes = sorted(df["origen"].unique())

# --- Cargar demanda real para el chequeo de anomalia de demanda ---
a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
demanda = pd.concat([a, b], ignore_index=True)[["fecha_hora", "demanda_lag24h"]].sort_values("fecha_hora")
demanda["demanda_media_30d"] = demanda["demanda_lag24h"].rolling(24 * 30, min_periods=24 * 7).mean()
demanda["anomalia_demanda_pct"] = (demanda["demanda_lag24h"] - demanda["demanda_media_30d"]) / demanda["demanda_media_30d"] * 100

print("=" * 90)
print("1) UNIVERSALIDAD ENTRE MODELOS: domingo/festivo/hora9-10, por origen")
print("=" * 90)
for o in origenes:
    print(f"\n--- {o} ---")
    for modelo in ["Persistencia", "ARX+GARCH", "N-BEATSx", "N-HiTS", "XGBoost"]:
        m = df[(df["origen"] == o) & (df["modelo"] == modelo)]
        if len(m) == 0:
            continue
        gen = m["ape"].mean()
        dom = m[m["es_domingo"]]["ape"].mean()
        fest = m[m["es_festivo"]]["ape"].mean() if m["es_festivo"].sum() > 0 else np.nan
        h910 = m[m["hora"].isin([9, 10])]["ape"].mean()
        print(f"  {modelo:14s} general={gen:5.1f}%  domingo={dom:5.1f}%  festivo={fest:5.1f}%  hora9-10={h910:5.1f}%")

print("\n" + "=" * 90)
print("2) SESGO POR DECIL DE PRECIO (N-BEATSx), por origen -- se invierte en el decil mas caro?")
print("=" * 90)
for o in origenes:
    m = df[(df["origen"] == o) & (df["modelo"] == "N-BEATSx")].copy()
    if len(m) < 100:
        continue
    m["decil"] = pd.qcut(m["real"], 10, labels=[f"D{i+1}" for i in range(10)], duplicates="drop")
    resumen = m.groupby("decil", observed=True)["sesgo_pct"].mean()
    d1 = resumen.iloc[0]
    d_ultimo = resumen.iloc[-1]
    invierte = "SI" if (d1 < 0 and d_ultimo > 0) else ("parcial" if np.sign(d1) != np.sign(d_ultimo) else "NO")
    print(f"{o:10s} sesgo D1(barato)={d1:+6.2f}%  sesgo D-ultimo(caro)={d_ultimo:+6.2f}%  se invierte: {invierte}")

print("\n" + "=" * 90)
print("3) RACHAS DE DIAS CONSECUTIVOS MALOS (N-BEATSx, MAPE diario > 25%), por origen")
print("=" * 90)
for o in origenes:
    m = df[(df["origen"] == o) & (df["modelo"] == "N-BEATSx")].copy()
    if len(m) < 100:
        continue
    por_dia = m.groupby("dia")["ape"].mean().sort_index()
    malo = por_dia > 25
    # identificar rachas
    grupo = (malo != malo.shift()).cumsum()
    rachas = por_dia[malo].groupby(grupo[malo]).apply(lambda x: (x.index.min(), x.index.max(), len(x)))
    print(f"\n--- {o} --- ({malo.sum()} dias con MAPE>25% de {len(por_dia)} dias totales)")
    if len(rachas) == 0:
        print("  (ninguna racha)")
    for inicio, fin, n in rachas:
        print(f"  {inicio.date()} a {fin.date()}  ({n} dias seguidos)")

print("\n" + "=" * 90)
print("4) ANOMALIA DE DEMANDA vs ERROR (N-BEATSx), por origen -- correlacion")
print("=" * 90)
for o in origenes:
    m = df[(df["origen"] == o) & (df["modelo"] == "N-BEATSx")].copy()
    if len(m) < 100:
        continue
    m = m.merge(demanda[["fecha_hora", "anomalia_demanda_pct"]], on="fecha_hora", how="left")
    m = m.dropna(subset=["anomalia_demanda_pct"])
    corr_ape = m["anomalia_demanda_pct"].abs().corr(m["ape"])
    corr_signed = m["anomalia_demanda_pct"].corr(m["error_signed"])
    print(f"{o:10s} corr(|anomalia_demanda|, error_abs%) = {corr_ape:+.3f}   "
          f"corr(anomalia_demanda, error_signed) = {corr_signed:+.3f}")

print("\n" + "=" * 90)
print("5) TASA DE CAMBIO DE ONI vs ERROR, por origen (limitado: ventanas de 3 meses = pocos puntos)")
print("=" * 90)
oni = pd.read_csv(RAIZ / "data/external/oni_index.csv")
meses_cols = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]
oni_long = []
for _, r in oni.iterrows():
    for mi, t in enumerate(meses_cols, start=1):
        oni_long.append({"anio": int(r["anio"]), "mes": mi, "oni": r[t]})
oni_long = pd.DataFrame(oni_long).sort_values(["anio","mes"])
oni_long["oni_delta"] = oni_long["oni"].diff()

for o in origenes:
    m = df[(df["origen"] == o) & (df["modelo"] == "N-BEATSx")].copy()
    if len(m) < 100:
        continue
    m["anio"] = m["fecha_hora"].dt.year
    m["mes"] = m["fecha_hora"].dt.month
    m = m.merge(oni_long[["anio","mes","oni","oni_delta"]], on=["anio","mes"], how="left")
    por_mes = m.groupby(["anio","mes"]).agg(mape=("ape","mean"), oni=("oni","first"), oni_delta=("oni_delta","first")).reset_index()
    print(f"\n--- {o} ---")
    print(por_mes.round(2).to_string(index=False))
