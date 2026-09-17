# -*- coding: utf-8 -*-
"""
Metricas en el mismo formato que reportan los papers de Noruega/Brasil (sMAPE, R2), para poder
comparar de verdad contra nuestros resultados -- nuestro MAPE convencional no es directamente
comparable con su sMAPE (acotado 0-200%, mas robusto a precios bajos con error relativo grande).

sMAPE = mean( |real-pred| / ((|real|+|pred|)/2) ) * 100
R2    = 1 - SS_res/SS_tot  (coeficiente de determinacion clasico)

Se calculan para los modelos/ensambles ya guardados en data/processed/resultados/, tanto a 24h
como a 72h, y para el naive-24h/naive-168h como referencia (los papers siempre reportan su naive).
"""
import numpy as np
import pandas as pd
from pathlib import Path

RES = Path(r"C:\Users\mgdbj\xm-spot-price-predictor") / "data/processed/resultados"


def smape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    denom = (np.abs(real) + np.abs(pred)) / 2
    return float(np.mean(np.abs(real - pred) / denom) * 100)


def r2(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    ss_res = np.sum((real - pred) ** 2)
    ss_tot = np.sum((real - real.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def mae(real, pred):
    return float(np.mean(np.abs(np.asarray(real, float) - np.asarray(pred, float))))


def fila(nombre, real, pred):
    return {"modelo": nombre, "mae": mae(real, pred), "mape": mape(real, pred),
            "smape": smape(real, pred), "r2": r2(real, pred), "n": len(real)}


def main():
    filas = []

    # ---------------- 24h ----------------
    c = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    o6 = c[c.origen == "Origen 6"].pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
    real24 = c[c.origen == "Origen 6"].groupby("fecha_hora")["real"].first().reindex(o6.index)
    for m in ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]:
        filas.append(fila(f"24h: {m}", real24, o6[m]))

    v1 = pd.read_csv(RES / "stacking_ensamble_cv_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    filas.append(fila("24h: Ensamble v1 (NNLS, 5 modelos)", v1["real"], v1["pred_ensamble"]))

    v4_path = RES / "bandas_24h_qra_causal_2026.csv"
    if v4_path.exists():
        v4 = pd.read_csv(v4_path, parse_dates=["fecha_hora"])
        filas.append(fila("24h: Ensamble v4 (QRA por franja, desplegable)", v4["real"], v4["q50"]))

    v4cv = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"])
    filas.append(fila("24h: Ensamble v4 (QRA por franja, CV)", v4cv["real"], v4cv["pred_ensamble"]))

    # naive-168h (persistencia semanal) como referencia de los papers
    naive168 = c[(c.origen == "Origen 6") & (c.modelo == "Persistencia")]  # ya es t-24; buscamos t-168 aparte
    ruta_feat = Path(r"C:\Users\mgdbj\xm-spot-price-predictor\data\processed\dataset_features_2026.csv")
    feat = pd.read_csv(ruta_feat, parse_dates=["fecha_hora"]).set_index("fecha_hora")
    common = o6.index.intersection(feat.index)
    filas.append(fila("24h: Naive-168h (semana pasada)", real24.loc[common], feat.loc[common, "precio_lag168h"]))

    # ---------------- 72h (por tramo) ----------------
    d72_path = RES / "pronostico_72h_diario_votantes_2026.csv"
    if d72_path.exists():
        d72 = pd.read_csv(d72_path, parse_dates=["fecha_hora", "cutoff"])
        for nombre, ini, fin in [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]:
            sub = d72[d72.paso_horas.between(ini, fin)]
            filas.append(fila(f"72h tramo {nombre}: N-BEATSx (linea base)", sub["real"], sub["NBX"]))
            if "Ridge-directo" in sub.columns:
                filas.append(fila(f"72h tramo {nombre}: Ridge directo", sub["real"], sub["Ridge-directo"]))

    ens72_path = RES / "pronostico_ensamble_72h_diario_2026.csv"
    if ens72_path.exists():
        ens72 = pd.read_csv(ens72_path, parse_dates=["fecha_hora", "cutoff"])
        for nombre, ini, fin in [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]:
            sub = ens72[ens72.paso_horas.between(ini, fin)].dropna(subset=["q50_v3_cv"])
            if len(sub):
                filas.append(fila(f"72h tramo {nombre}: Ensamble (CV)", sub["real"], sub["q50_v3_cv"]))

    tabla = pd.DataFrame(filas)
    print(tabla.round(3).to_string(index=False))
    tabla.to_csv(RES / "metricas_comparables_papers.csv", index=False)

    print(f"\n{'='*78}\nCOMPARACION DIRECTA CONTRA LOS PAPERS LEIDOS\n{'='*78}")
    print("Noruega (LightGBM, 5 zonas, 2025 test) -- sMAPE 12.07-25.32%,  R2 0.889-0.934")
    print("Nuestro mejor modelo de 24h (Ensamble v4, CV):")
    fv = tabla[tabla.modelo.str.contains("v4.*CV", regex=True)].iloc[0]
    print(f"  sMAPE={fv['smape']:.2f}%   R2={fv['r2']:.3f}   (MAPE convencional={fv['mape']:.2f}%)")


if __name__ == "__main__":
    main()
