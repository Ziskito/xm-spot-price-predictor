# -*- coding: utf-8 -*-
"""
Calibracion conforme de las bandas de N-BEATSx a horizonte completo 24-72h.

Hallazgo que motiva esto: las bandas crudas (MQLoss) tienen cobertura que se desploma con el
horizonte -- 60.7% en 1-24h, 50.5% en 25-48h, 45.0% en 49-72h (objetivo: 80% en los 3 tramos).
El ancho de la banda casi no crece con el horizonte (136.6 -> 144.1 -> 148.8), muy por debajo de
lo que crece el error real (58.7 -> 76.2 -> 98.6) -- el modelo no esta ensanchando su incertidumbre
lo suficiente a medida que el pronostico se aleja.

Se calibra un margen INDEPENDIENTE por tramo de horizonte (1-24h, 25-48h, 49-72h), no uno solo
global, precisamente porque el problema es que la incertidumbre real varia con el horizonte.
Split conforme (mitad de las fechas para calibrar, mitad para evaluar), igual metodologia que
calibrar_bandas.py de anoche.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_CRUDO = RAIZ / "data" / "processed" / "resultados" / "pronostico_con_bandas_72h_2026.csv"
RUTA_CALIBRADO = RAIZ / "data" / "processed" / "resultados" / "pronostico_con_bandas_72h_2026_calibrado.csv"

NIVEL_OBJETIVO = 0.80

df = pd.read_csv(RUTA_CRUDO, parse_dates=["fecha_hora", "cutoff"])
print(f"Total: {len(df):,} filas | {df['fecha_hora'].min()} a {df['fecha_hora'].max()}")

tramos = [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]
partes_calibradas = []
resumen = []

for nombre, ini, fin in tramos:
    sub = df[(df["paso_horas"] >= ini) & (df["paso_horas"] <= fin)].copy()
    fecha_corte = sub["fecha_hora"].quantile(0.5, interpolation="nearest")
    calib = sub[sub["fecha_hora"] < fecha_corte]
    evalua = sub[sub["fecha_hora"] >= fecha_corte].copy()

    scores = np.maximum(calib["q10"] - calib["real"], calib["real"] - calib["q90"])
    n = len(scores)
    nivel_ajustado = min(np.ceil((n + 1) * NIVEL_OBJETIVO) / n, 1.0)
    margen = np.quantile(scores, nivel_ajustado)

    evalua["q10_cal"] = evalua["q10"] - margen
    evalua["q90_cal"] = evalua["q90"] + margen

    cob_cruda = ((evalua["real"] >= evalua["q10"]) & (evalua["real"] <= evalua["q90"])).mean() * 100
    cob_cal = ((evalua["real"] >= evalua["q10_cal"]) & (evalua["real"] <= evalua["q90_cal"])).mean() * 100
    ancho_cal = (evalua["q90_cal"] - evalua["q10_cal"]).mean()

    print(f"{nombre:8s}  margen={margen:6.1f}  cobertura cruda={cob_cruda:5.1f}%  "
          f"-> calibrada={cob_cal:5.1f}%  ancho_calibrado={ancho_cal:6.1f}")
    resumen.append({"tramo": nombre, "margen": margen, "cobertura_cruda": cob_cruda,
                     "cobertura_calibrada": cob_cal, "ancho_calibrado": ancho_cal, "n_eval": len(evalua)})

    salida_tramo = evalua[["fecha_hora", "cutoff", "paso_horas", "real", "q50"]].copy()
    salida_tramo["q10"] = evalua["q10_cal"]
    salida_tramo["q90"] = evalua["q90_cal"]
    partes_calibradas.append(salida_tramo)

salida = pd.concat(partes_calibradas, ignore_index=True).sort_values(["cutoff", "paso_horas"])
salida.to_csv(RUTA_CALIBRADO, index=False)
pd.DataFrame(resumen).to_csv(RAIZ / "data/processed/resultados/bandas_metricas_72h_calibrado.csv", index=False)

print(f"\nContrato calibrado 24-72h para OE3 guardado en: {RUTA_CALIBRADO}")
print("(cubre solo la mitad de evaluacion del test 2026 -- la otra mitad se uso para calibrar)")
