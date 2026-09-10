# -*- coding: utf-8 -*-
"""
Calibracion conforme (CQR - Conformalized Quantile Regression) de las bandas de N-BEATSx.

Problema encontrado: las bandas [q10,q90] crudas de MQLoss dieron 62.6% de cobertura en el
holdout 2026 completo, contra el 80% nominal -- estan sobre-confiadas. Esto es un fenomeno
conocido de MQLoss (la perdida cuantilica no garantiza calibracion frecuentista exacta).

Metodo (CQR, Romano et al. 2019 -- estandar para corregir esto sin reentrenar el modelo):
  1. Se parte el holdout 2026 en un tramo de CALIBRACION (primeros meses) y un tramo de
     EVALUACION FINAL (resto), que nunca se toca al calcular el ajuste.
  2. En calibracion, se mide el score de conformidad de cada punto:
         s = max(q10 - y, y - q90)
     (positivo si el valor real quedo fuera de la banda cruda, negativo si quedo dentro)
  3. Se toma el percentil ceil((n+1)*0.8)/n de esos scores -> margen de ajuste.
  4. Las bandas finales son [q10 - margen, q90 + margen]. Esto GARANTIZA cobertura >= 80%
     en el tramo de evaluacion, bajo el supuesto estandar de intercambiabilidad.

Esto es ademas realista para produccion: un sistema real recalibraria periodicamente sus
bandas con los errores mas recientes, exactamente como se hace aqui.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_CRUDO = RAIZ / "data" / "processed" / "resultados" / "pronostico_con_bandas_2026.csv"
RUTA_CALIBRADO = RAIZ / "data" / "processed" / "resultados" / "pronostico_con_bandas_2026_calibrado.csv"

NIVEL_OBJETIVO = 0.80

df = pd.read_csv(RUTA_CRUDO, parse_dates=["fecha_hora"])
print(f"Total: {len(df):,} filas | {df['fecha_hora'].min()} a {df['fecha_hora'].max()}")

fecha_corte = df["fecha_hora"].quantile(0.5, interpolation="nearest")
calib = df[df["fecha_hora"] < fecha_corte].copy()
evalua = df[df["fecha_hora"] >= fecha_corte].copy()
print(f"Calibracion: {len(calib):,} filas (hasta {calib['fecha_hora'].max()})")
print(f"Evaluacion final: {len(evalua):,} filas (desde {evalua['fecha_hora'].min()})")

cobertura_cruda_calib = ((calib["real"] >= calib["q10"]) & (calib["real"] <= calib["q90"])).mean()
cobertura_cruda_eval = ((evalua["real"] >= evalua["q10"]) & (evalua["real"] <= evalua["q90"])).mean()
print(f"\nCobertura CRUDA en calibracion: {cobertura_cruda_calib*100:.1f}%")
print(f"Cobertura CRUDA en evaluacion:  {cobertura_cruda_eval*100:.1f}%")

scores = np.maximum(calib["q10"] - calib["real"], calib["real"] - calib["q90"])
n = len(scores)
nivel_ajustado = min(np.ceil((n + 1) * NIVEL_OBJETIVO) / n, 1.0)
margen = np.quantile(scores, nivel_ajustado)
print(f"\nMargen de ajuste CQR (percentil {nivel_ajustado*100:.2f}% de los scores): {margen:.2f} COP/kWh")

for parte in [calib, evalua, df]:
    parte["q10_calibrado"] = parte["q10"] - margen
    parte["q90_calibrado"] = parte["q90"] + margen

cobertura_calibrada_eval = ((evalua["real"] >= evalua["q10_calibrado"]) &
                             (evalua["real"] <= evalua["q90_calibrado"])).mean()
ancho_crudo = (evalua["q90"] - evalua["q10"]).mean()
ancho_calibrado = (evalua["q90_calibrado"] - evalua["q10_calibrado"]).mean()

print(f"\nCobertura CALIBRADA en evaluacion (fuera de la muestra de ajuste): {cobertura_calibrada_eval*100:.1f}%")
print(f"Ancho de banda: {ancho_crudo:.1f} -> {ancho_calibrado:.1f} COP/kWh")

salida = df[["fecha_hora", "real", "q50"]].copy()
salida["q10"] = df["q10"] - margen
salida["q90"] = df["q90"] + margen
salida.to_csv(RUTA_CALIBRADO, index=False)
print(f"\nContrato CALIBRADO para OE3 guardado en: {RUTA_CALIBRADO}")
print("(usa esta version, no la cruda -- la cruda subestima el riesgo)")
