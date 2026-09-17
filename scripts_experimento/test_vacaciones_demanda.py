# -*- coding: utf-8 -*-
"""
Test DIRECTO de la hipotesis de vacaciones de mitad de anio en Colombia, distinto del proxy
indirecto que usamos antes (correlacion anomalia de demanda vs error del modelo, que dio negativo
en los 6 origenes).

Aqui se prueba la premisa misma: hay una caida medible de demanda durante la ventana de
vacaciones de mitad de anio (aprox. 15 junio - 15 julio, colegios y universidades en Colombia)
en cada anio 2019-2026? Si la demanda NO cae de forma consistente en esa ventana, la hipotesis de
vacaciones queda descartada en su raiz, sin necesidad de seguir buscando su efecto en el error del
modelo.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
df = df[["fecha_hora", "demanda_lag24h"]].dropna()

df["demanda_media_30d"] = df["demanda_lag24h"].rolling(24 * 30, min_periods=24 * 7, center=True).mean()
df["anomalia_pct"] = (df["demanda_lag24h"] - df["demanda_media_30d"]) / df["demanda_media_30d"] * 100

print("=" * 90)
print("Anomalia de demanda (%) durante la ventana de vacaciones (15 jun - 15 jul) vs el resto del anio")
print("=" * 90)

resultados = []
for anio in range(2019, 2027):
    ini = pd.Timestamp(f"{anio}-06-15")
    fin = pd.Timestamp(f"{anio}-07-15")
    ventana = df[(df["fecha_hora"] >= ini) & (df["fecha_hora"] < fin)]
    resto = df[(df["fecha_hora"].dt.year == anio) & ~((df["fecha_hora"] >= ini) & (df["fecha_hora"] < fin))]
    if len(ventana) < 100 or len(resto) < 100:
        continue
    anom_vac = ventana["anomalia_pct"].mean()
    anom_resto = resto["anomalia_pct"].mean()

    # Prueba de significancia (HAC, mismo estilo que el resto del proyecto): la anomalia
    # promedio en la ventana de vacaciones, es significativamente distinta de cero?
    y = ventana["anomalia_pct"].dropna().values
    m = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
    p_valor = m.pvalues[0]
    marca = "cae SIG." if (p_valor < 0.05 and anom_vac < 0) else ("sube sig." if (p_valor < 0.05 and anom_vac > 0) else "sin diferencia sig.")

    resultados.append({"anio": anio, "anomalia_vacaciones_pct": anom_vac, "anomalia_resto_pct": anom_resto, "p_valor": p_valor, "lectura": marca})
    print(f"{anio}  anomalia en ventana vacaciones = {anom_vac:+6.2f}%   anomalia resto del anio = {anom_resto:+6.2f}%   p={p_valor:.4f}   {marca}")

df_res = pd.DataFrame(resultados)
print("\n" + "=" * 90)
print(f"Anios donde la demanda CAE de forma significativa en la ventana de vacaciones: "
      f"{(df_res['lectura']=='cae SIG.').sum()} de {len(df_res)}")
print(f"Promedio de la anomalia en ventana de vacaciones, across anios: {df_res['anomalia_vacaciones_pct'].mean():+.2f}%")

df_res.to_csv(RAIZ / "data" / "processed" / "resultados" / "test_vacaciones_demanda.csv", index=False)
print("\nGuardado: test_vacaciones_demanda.csv")
