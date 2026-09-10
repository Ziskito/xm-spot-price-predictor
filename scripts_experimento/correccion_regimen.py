# -*- coding: utf-8 -*-
"""
Version barata del "regime-aware modeling" (punto 2, inspirado en arXiv:2508.00040 -- HMM +
modelo por regimen). Entrenar una red separada por regimen para El Nino (~10% de los datos) ya
se descarto anoche por resultados esperados malos (el intento de sobremuestreo DUP_NINO fallo por
la misma escasez de datos, y la propia literatura reconoce que es el caso dificil).

En vez de eso: CORRECCION POST-HOC de sesgo/escala especifica por regimen, aplicada sobre las
predicciones YA CALCULADAS del ensamble de ventanas de N-BEATSx. Es la version barata y honesta
de "el modelo se comporta distinto segun el regimen" -- no requiere reentrenar nada, solo ajustar
una regresion lineal simple.

Metodologia (para evitar la misma trampa que la calibracion de bandas de anoche -- ajustar y
evaluar en el mismo tramo infla los resultados):
  1. Partir 2026 en CALIBRACION (primera mitad) y EVALUACION (segunda mitad), igual que se hizo
     para las bandas de incertidumbre.
  2. En calibracion, ajustar por separado (con OLS: error = a + b*prediccion) una correccion para
     horas de El Nino y otra para el resto.
  3. Aplicar esa correccion en evaluacion y medir si el error de El Nino mejora AHI, no en la
     muestra de ajuste.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

df = pd.read_csv(RAIZ / "data/processed/resultados/ventanas_predicciones.csv", parse_dates=["ds"])
pivote = df.pivot_table(index="ds", columns="ventana", values="NBEATSx", aggfunc="mean")
y = df.groupby("ds")["y"].first().reindex(pivote.index)
pred = pivote[["1a", "2a", "3a", "5a", "completa"]].mean(axis=1)

base = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
oni = base["oni"].reindex(pred.index)

datos = pd.DataFrame({"y": y, "pred": pred, "oni": oni}).dropna()
datos["es_nino"] = datos["oni"] > 1.0

# Particion ALEATORIA ESTRATIFICADA por regimen (no temporal): el corte temporal 50/50 dejaba
# 0 horas de El Nino en calibracion (El Nino en 2026 solo arranca en junio) -- inutil para ajustar
# su propia correccion. Para una correccion post-hoc de 2 parametros (no reentrenar el modelo),
# un split aleatorio estratificado es defendible y evita ese problema.
rng = np.random.default_rng(42)
datos["dia"] = datos.index.normalize()
dias_nino = datos.loc[datos["es_nino"], "dia"].unique()
dias_resto = datos.loc[~datos["es_nino"], "dia"].unique()
dias_calib = set(rng.choice(dias_nino, size=len(dias_nino)//2, replace=False)) | \
             set(rng.choice(dias_resto, size=len(dias_resto)//2, replace=False))

calib = datos[datos["dia"].isin(dias_calib)]
evalua = datos[~datos["dia"].isin(dias_calib)]
print(f"Calibracion: {len(calib)} filas | Evaluacion: {len(evalua)} filas")
print(f"Horas El Nino en calibracion: {calib['es_nino'].sum()} | en evaluacion: {evalua['es_nino'].sum()}")


def ajustar_correccion(sub):
    """OLS simple: y = a + b*pred -- devuelve (a, b)."""
    if len(sub) < 30:
        return 0.0, 1.0
    X = np.vstack([np.ones(len(sub)), sub["pred"].values]).T
    coef, *_ = np.linalg.lstsq(X, sub["y"].values, rcond=None)
    return coef[0], coef[1]


a_nino, b_nino = ajustar_correccion(calib[calib["es_nino"]])
a_resto, b_resto = ajustar_correccion(calib[~calib["es_nino"]])
print(f"\nCorreccion El Nino:  y_corr = {a_nino:.2f} + {b_nino:.3f} * pred")
print(f"Correccion resto:    y_corr = {a_resto:.2f} + {b_resto:.3f} * pred")

evalua = evalua.copy()
evalua["pred_corregida"] = np.where(
    evalua["es_nino"], a_nino + b_nino * evalua["pred"], a_resto + b_resto * evalua["pred"]
)

mae_crudo_nino = (evalua.loc[evalua["es_nino"], "y"] - evalua.loc[evalua["es_nino"], "pred"]).abs().mean()
mae_corr_nino = (evalua.loc[evalua["es_nino"], "y"] - evalua.loc[evalua["es_nino"], "pred_corregida"]).abs().mean()
mae_crudo_resto = (evalua.loc[~evalua["es_nino"], "y"] - evalua.loc[~evalua["es_nino"], "pred"]).abs().mean()
mae_corr_resto = (evalua.loc[~evalua["es_nino"], "y"] - evalua.loc[~evalua["es_nino"], "pred_corregida"]).abs().mean()
mae_crudo_total = (evalua["y"] - evalua["pred"]).abs().mean()
mae_corr_total = (evalua["y"] - evalua["pred_corregida"]).abs().mean()

print(f"\n{'='*66}\nRESULTADO EN EVALUACION (fuera de la muestra de ajuste)\n{'='*66}")
print(f"El Nino  -- MAE crudo: {mae_crudo_nino:.2f}   MAE corregido: {mae_corr_nino:.2f}   "
      f"diferencia: {mae_corr_nino-mae_crudo_nino:+.2f}")
print(f"Resto    -- MAE crudo: {mae_crudo_resto:.2f}   MAE corregido: {mae_corr_resto:.2f}   "
      f"diferencia: {mae_corr_resto-mae_crudo_resto:+.2f}")
print(f"Global   -- MAE crudo: {mae_crudo_total:.2f}   MAE corregido: {mae_corr_total:.2f}   "
      f"diferencia: {mae_corr_total-mae_crudo_total:+.2f}")
