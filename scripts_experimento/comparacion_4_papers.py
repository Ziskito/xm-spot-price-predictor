# -*- coding: utf-8 -*-
"""
Comparacion rigurosa contra los 4 papers que el usuario consiguio en PDF completo.

El problema que motiva este script: NO se pueden comparar los numeros tal como vienen. Al leer
los papers completos aparecen tres trampas de comparabilidad:

  1) Kapoor & Wichitaksorn 2023 (Nueva Zelanda, Applied Energy 347:121446) definen
        sMAPE = (100/T) * suma( |y-yhat| / (|y| + |yhat|) )
     SIN el /2 en el denominador. Nuestra definicion (y la mas comun) usa (|y|+|yhat|)/2.
     Es decir: su sMAPE es exactamente LA MITAD del nuestro sobre los mismos datos.
     Comparar 10.38% contra su 0.67 sin corregir es un error de factor 2.

  2) Ademas, sus MAE/RMSE/sMAPE estan calculados sobre la serie TRANSFORMADA (Box-Cox +
     min-max a [0,1], su Tabla 2), no sobre NZD/MWh: su MAE "0.3793" en una serie cuya media
     real es 107 NZD/MWh solo tiene sentido en escala transformada. El paper nunca dice que
     invierta la transformacion antes de medir. Por eso su UNICA metrica comparable con
     nosotros es la MASE, que es adimensional por construccion.

  3) Dias, Lira & Freire 2024 (Brasil, Energies 17:1864) reportan NRMSE = RMSE / media(y)
     y TAPI (aciertos de direccion), no sMAPE. Albani et al. 2025 reportan un "L2-error"
     sobre LOG-precios y acierto direccional. Nada de eso lo teniamos calculado.

Este script calcula sobre NUESTROS resultados, en 2026: MAE, MAPE, sMAPE (ambas convenciones),
R2, RMSE, NRMSE, MASE (escalada con naive-1h y con naive-24h) y TAPI/acierto direccional.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
CORTE = pd.Timestamp("2026-01-01")


def tapi(real, pred):
    """Indice de acierto de tendencia (Dias et al. 2024): % de pasos donde el SIGNO del cambio
    coincide. Solo se cuentan los pasos donde el precio real SI cambio (sign!=0)."""
    d_real, d_pred = np.diff(real), np.diff(pred)
    valido = np.sign(d_real) != 0
    if valido.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(d_real[valido]) == np.sign(d_pred[valido])) * 100)


def metricas(idx, real, pred, den_mase_1h, den_mase_24h):
    real = np.asarray(real, float)
    pred = np.asarray(pred, float)
    ok = ~(np.isnan(real) | np.isnan(pred))
    idx, real, pred = pd.DatetimeIndex(idx)[ok], real[ok], pred[ok]
    err = np.abs(real - pred)
    # sMAPE en las dos convenciones que aparecen en la literatura
    smape_std = float(np.mean(err / ((np.abs(real) + np.abs(pred)) / 2)) * 100)   # la nuestra
    smape_kap = float(np.mean(err / (np.abs(real) + np.abs(pred))) * 100)         # la de Kapoor
    rmse = float(np.sqrt(np.mean((real - pred) ** 2)))
    # TAPI a tres agregaciones: Dias et al. miden paso SEMANAL, no horario. Comparar su TAPI
    # semanal contra un TAPI horario nuestro seria tramposo, asi que se reportan los tres.
    s = pd.DataFrame({"real": real, "pred": pred}, index=idx)
    dia = s.resample("D").mean().dropna()
    sem = s.resample("W").mean().dropna()
    return {
        "MAE": float(err.mean()),
        "MAPE_%": float(np.mean(err / real) * 100),
        "sMAPE_nuestro_%": smape_std,
        "sMAPE_conv_Kapoor_%": smape_kap,
        "R2": float(1 - np.sum((real - pred) ** 2) / np.sum((real - real.mean()) ** 2)),
        "RMSE": rmse,
        "NRMSE_%": rmse / real.mean() * 100,          # convencion de Dias et al.
        "MASE_naive1h": float(err.mean()) / den_mase_1h,
        "MASE_naive24h": float(err.mean()) / den_mase_24h,
        "TAPI_hora_%": tapi(real, pred),
        "TAPI_dia_%": tapi(dia["real"].to_numpy(), dia["pred"].to_numpy()),
        "TAPI_semana_%": tapi(sem["real"].to_numpy(), sem["pred"].to_numpy()),
        "n": len(real),
    }


def main():
    # --- denominadores MASE: MAE en-muestra del naive de un paso, sobre el TRAIN (pre-2026) ---
    feat = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv",
                       parse_dates=["fecha_hora"]).set_index("fecha_hora").sort_index()
    p_tr = feat.loc[feat.index < CORTE, "precio_bolsa"].dropna()
    den_1h = float(np.abs(p_tr.diff().dropna()).mean())
    den_24h = float(np.abs(p_tr.diff(24).dropna()).mean())
    print(f"Denominadores MASE (MAE en-muestra del naive, train 2019-2025):")
    print(f"  naive 1 hora  : {den_1h:7.3f} COP/kWh   (referencia muy exigente: la hora anterior)")
    print(f"  naive 24 horas: {den_24h:7.3f} COP/kWh   (referencia justa: nuestro horizonte real)\n")

    filas = {}

    # ---------------- 24h ----------------
    v4 = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"]).sort_values("fecha_hora")
    filas["24h Ensamble v4 (QRA franja, CV)"] = metricas(
        v4["fecha_hora"], v4["real"], v4["pred_ensamble"], den_1h, den_24h)
    filas["24h Ensamble v4 (desplegable)"] = metricas(
        v4["fecha_hora"], v4["real"], v4["pred_causal"], den_1h, den_24h)

    crudas = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    o6 = crudas[crudas.origen == "Origen 6"]
    piv = o6.pivot_table(index="fecha_hora", columns="modelo", values="prediccion").sort_index()
    real24 = o6.groupby("fecha_hora")["real"].first().reindex(piv.index)
    for m in ["N-BEATSx", "XGBoost", "Persistencia"]:
        if m in piv.columns:
            filas[f"24h {m}"] = metricas(piv.index, real24, piv[m], den_1h, den_24h)

    # ---------------- 72h por tramo ----------------
    e72 = pd.read_csv(RES / "pronostico_ensamble_72h_diario_2026.csv",
                      parse_dates=["fecha_hora", "cutoff"]).sort_values(["cutoff", "paso_horas"])
    for nom, ini, fin in [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]:
        sub = e72[e72.paso_horas.between(ini, fin)].dropna(subset=["q50_v3_cv"])
        if len(sub):
            filas[f"72h tramo {nom} Ensamble (CV)"] = metricas(
                sub["fecha_hora"], sub["real"], sub["q50_v3_cv"], den_1h, den_24h)

    tabla = pd.DataFrame(filas).T
    tabla.index.name = "modelo"
    print(tabla.round(3).to_string())
    tabla.to_csv(RES / "comparacion_4_papers.csv")

    # ---------------- confrontacion ----------------
    n = tabla.loc["24h Ensamble v4 (QRA franja, CV)"]
    print(f"\n{'='*92}\nCONFRONTACION CON CADA PAPER (usando SU metrica, no la nuestra)\n{'='*92}")

    print("\n[1] Kapoor & Wichitaksorn 2023 -- Nueva Zelanda, precio DIARIO, 5 regiones, horizonte 1 dia")
    print("    Unica metrica comparable: MASE (las demas estan en escala transformada Box-Cox+minmax).")
    print("    Sus mejores modelos (LE-GARCH-t / LEAR): MASE 1.263 - 1.337  (peores hasta 2.51)")
    print(f"    Nosotros, 24h: MASE={n['MASE_naive24h']:.3f} (escalado naive-24h, mismo horizonte)")
    print(f"                   MASE={n['MASE_naive1h']:.3f} (escalado naive-1h, mucho mas exigente)")
    print("    Su sMAPE usa denominador SIN /2; en su convencion el nuestro seria "
          f"{n['sMAPE_conv_Kapoor_%']:.2f}% (no {n['sMAPE_nuestro_%']:.2f}%).")

    print("\n[2] Dias, Lira & Freire 2024 -- Brasil, PLD SEMANAL, horizonte 4 semanas, submercado SE/CO")
    print("    MLP:     MAPE 14.65%   NRMSE 24.70%   TAPI 68.75%  (TAPI medido paso a paso SEMANAL)")
    print("    DECOMP (modelo oficial de Brasil): MAPE 19.44%   NRMSE 53.89%   TAPI 56.25%")
    print(f"    Nosotros 24h:    MAPE {n['MAPE_%']:5.2f}%   NRMSE {n['NRMSE_%']:5.2f}%   "
          f"TAPI hora {n['TAPI_hora_%']:.1f}% / dia {n['TAPI_dia_%']:.1f}% / semana {n['TAPI_semana_%']:.1f}%")
    t72 = tabla.loc["72h tramo 49-72h Ensamble (CV)"]
    print(f"    Nosotros 49-72h: MAPE {t72['MAPE_%']:5.2f}%   NRMSE {t72['NRMSE_%']:5.2f}%   "
          f"TAPI hora {t72['TAPI_hora_%']:.1f}% / dia {t72['TAPI_dia_%']:.1f}% / semana {t72['TAPI_semana_%']:.1f}%")

    print("\n[3] Albani et al. 2025 -- Brasil, precio FORWARD mensual, horizonte 30 dias")
    print("    Acierto direccional: ~50% con todos los datos; ~60% solo cuando |corr(ENA,precio)|>=0.5")
    print("    No le gana al naive estacional: solo ~50% de las veces su RMSE es menor,")
    print("    y la diferencia es significativa apenas el 31-33% del tiempo.")
    print(f"    Nosotros 24h, acierto direccional a paso diario: {n['TAPI_dia_%']:.1f}%")

    print("\n    Nota metodologica: el TAPI solo cuenta los pasos donde el precio real SI cambio.")
    print("    Contar las horas de precio plano como fallo hunde el indice a ~40% (por debajo del")
    print("    azar) y es un artefacto, no un resultado: sign(0) nunca coincide con +1 ni con -1.")

    print("\n[4] Nunes Jr, Ferreira & Pinho 2024 -- Brasil, caudales -> DESSEM -> PLD horario")
    print("    No reporta MAE/MAPE de precio (solo boxplots de diferencia contra el PLD oficial).")
    print("    Su resultado comparable es CUALITATIVO y confirma nuestro hallazgo negativo:")
    print("    mejorar el pronostico de caudales (ANN-MCC mejor que ANN-MSE) NO movio el precio:")
    print("    'the boxplots are practically the same'.")


if __name__ == "__main__":
    main()
