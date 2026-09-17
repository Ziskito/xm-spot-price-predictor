# -*- coding: utf-8 -*-
"""
Prueba si el sesgo de sobreestimacion visto en 2026 (N-BEATSx, sesgo promedio -5.25%) se puede
corregir de forma honesta -- calibrada y evaluada en tramos DISTINTOS de 2026, no en los mismos
datos (eso inflaria el resultado artificialmente).

IMPORTANTE (ver bitacora): el sesgo NO es estable entre anios -- en 2020 y 2022 el modelo
SUBESTIMABA, no sobreestimaba. Por eso esta correccion se calibra solo con datos de 2026, no con
todo el historico, y se advierte que podria no generalizar a anios futuros con un regimen distinto.

Split ALEATORIO ESTRATIFICADO por dia (no cronologico), para que calibracion y evaluacion cubran
por igual todos los regimenes de 2026 (el mismo problema que tuvimos anoche con un split
cronologico simple: dejaba El Nino fuera de la calibracion por completo).
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

df = pd.read_csv(RAIZ / "data/processed/resultados/walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
nb = df[(df["origen"] == "Origen 6") & (df["modelo"] == "N-BEATSx")].copy().sort_values("fecha_hora")
nb["dia"] = nb["fecha_hora"].dt.normalize()

rng = np.random.default_rng(42)
dias = nb["dia"].unique()
dias_calib = set(rng.choice(dias, size=len(dias) // 2, replace=False))

calib = nb[nb["dia"].isin(dias_calib)]
evalua = nb[~nb["dia"].isin(dias_calib)].copy()
print(f"Calibracion: {len(calib)} filas | Evaluacion: {len(evalua)} filas")

# --- Correccion lineal simple: real = a + b*pred, ajustada SOLO en calibracion ---
X = np.vstack([np.ones(len(calib)), calib["prediccion"].values]).T
coef, *_ = np.linalg.lstsq(X, calib["real"].values, rcond=None)
a, b = coef
print(f"\nCorreccion ajustada en calibracion: precio_corregido = {a:.2f} + {b:.4f} * prediccion")

evalua["pred_corregida"] = a + b * evalua["prediccion"]

def metricas(y, p, nombre):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = y - p
    mae = np.abs(err).mean()
    mape = (np.abs(err) / y).mean() * 100
    sesgo = (err / y).mean() * 100
    print(f"{nombre:20s} MAE={mae:6.2f}  MAPE={mape:5.2f}%  sesgo={sesgo:+.2f}%")
    return mae, mape, sesgo

print(f"\n{'='*66}\nRESULTADO EN EVALUACION (mitad de dias NO usada para calibrar)\n{'='*66}")
metricas(evalua["real"], evalua["prediccion"], "Cruda")
metricas(evalua["real"], evalua["pred_corregida"], "Corregida")

# --- Significancia: Diebold-Mariano sobre la mejora ---
import statsmodels.api as sm
e_cruda = np.abs(evalua["real"] - evalua["prediccion"])
e_corr = np.abs(evalua["real"] - evalua["pred_corregida"])
d = e_cruda.values - e_corr.values
m = sm.OLS(d, np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
print(f"\nDiferencia media de error (cruda-corregida): {d.mean():+.3f}")
print(f"t={m.tvalues[0]:.3f}  p={m.pvalues[0]:.5f}  {'SIGNIFICATIVO' if m.pvalues[0]<0.05 else 'no significativo'}")
