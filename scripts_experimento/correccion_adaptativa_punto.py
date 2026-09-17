# -*- coding: utf-8 -*-
"""
Idea nueva #1: recalibracion ADAPTATIVA del punto central (no solo de las bandas de
incertidumbre, que ya usan esto). En vez de un ajuste de sesgo global fijo (ya probado y
descartado -- correccion_sesgo_2026.py, DM p=0.682), un ajuste que se RECALCULA cada dia con
el error real de los ultimos N dias -- se mueve con el sesgo reciente en vez de ser un numero
fijo. Ataca directo el "a veces sesga arriba, a veces abajo".

CRITICO: esto es un procedimiento SECUENCIAL de verdad (no un split aleatorio) -- la correccion
del dia t solo puede usar errores de dias ANTES de t, nunca de despues, para que sea honesto
(replicable en tiempo real).

Base: las predicciones ya validadas del ensamble de stacking (mejor resultado actual del
proyecto, MAE 44.51 / MAPE 12.05%), sacadas de stacking_ensamble_cv_2026.csv.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

df = pd.read_csv(RAIZ / "data/processed/resultados/stacking_ensamble_cv_2026.csv", parse_dates=["fecha_hora"])
df = df.sort_values("fecha_hora").reset_index(drop=True)
df["error_signed"] = df["real"] - df["pred_ensamble"]
df["dia"] = df["fecha_hora"].dt.normalize()

# --- Error promedio diario (para calcular la correccion a nivel de dia, mas estable que por hora) ---
error_diario = df.groupby("dia")["error_signed"].mean()

def calcular_metricas(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    err = y_real - y_pred
    mae = np.abs(err).mean()
    mape = (np.abs(err) / y_real).mean() * 100
    return mae, mape

def prueba_dm(e1, e2, horizonte=24):
    d = np.asarray(e1) - np.asarray(e2)
    m = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": horizonte - 1})
    return m.tvalues[0], m.pvalues[0]

resultados = []
for ventana_dias in [7, 14, 30]:
    # Correccion del dia t = promedio de error_signed en los `ventana_dias` dias ANTES de t
    correccion_por_dia = error_diario.rolling(window=ventana_dias, min_periods=ventana_dias).mean().shift(1)
    df[f"correccion_{ventana_dias}d"] = df["dia"].map(correccion_por_dia)

df_valido = df.dropna(subset=["correccion_30d"]).copy()  # recorta el primer mes (sin historia suficiente)
print(f"Filas evaluables (con al menos 30 dias de historia previa): {len(df_valido)} de {len(df)}")

mae_base, mape_base = calcular_metricas(df_valido["real"], df_valido["pred_ensamble"])
print(f"\nSin correccion (base, mismo periodo recortado): MAE {mae_base:.2f}  MAPE {mape_base:.2f}%")

for ventana_dias in [7, 14, 30]:
    pred_corregida = df_valido["pred_ensamble"] + df_valido[f"correccion_{ventana_dias}d"]
    mae_c, mape_c = calcular_metricas(df_valido["real"], pred_corregida)
    dm_stat, p_valor = prueba_dm(
        np.abs(df_valido["real"] - df_valido["pred_ensamble"]),
        np.abs(df_valido["real"] - pred_corregida),
    )
    concl = "CORRECCION GANA (sig.)" if (p_valor < 0.05 and dm_stat > 0) else ("CORRECCION PIERDE (sig.)" if (p_valor < 0.05 and dm_stat < 0) else "sin diferencia")
    print(f"\nVentana {ventana_dias}d -> MAE {mae_c:.2f}  MAPE {mape_c:.2f}%  "
          f"DM t={dm_stat:.3f} p={p_valor:.4f}  {concl}")
    resultados.append({"ventana_dias": ventana_dias, "mae": mae_c, "mape": mape_c,
                        "dm_stat": dm_stat, "p_valor": p_valor, "conclusion": concl})

# --- Foco en junio: es ahi donde mas se necesita la adaptabilidad ---
print(f"\n{'='*70}\nZoom a junio 2026 (el mes problematico)\n{'='*70}")
junio = df_valido[df_valido["dia"].dt.month == 6]
mae_junio_base, mape_junio_base = calcular_metricas(junio["real"], junio["pred_ensamble"])
print(f"Sin correccion: MAE {mae_junio_base:.2f}  MAPE {mape_junio_base:.2f}%")
for ventana_dias in [7, 14, 30]:
    pred_junio_c = junio["pred_ensamble"] + junio[f"correccion_{ventana_dias}d"]
    mae_jc, mape_jc = calcular_metricas(junio["real"], pred_junio_c)
    print(f"Ventana {ventana_dias}d -> MAE {mae_jc:.2f}  MAPE {mape_jc:.2f}%")

pd.DataFrame(resultados).to_csv(RAIZ / "data" / "processed" / "resultados" / "correccion_adaptativa_punto.csv", index=False)
print("\nGuardado: correccion_adaptativa_punto.csv")
