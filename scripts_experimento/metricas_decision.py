# -*- coding: utf-8 -*-
"""
Metricas de evaluacion "conscientes de la decision" (Maciejowska, Lipiecki, Uniejewski --
"Statistical and economic evaluation of forecasts in electricity markets: beyond RMSE and MAE",
Energy Conversion and Management 2026 / arXiv:2511.13616).

Hallazgo del paper: RMSE/MAE correlacionan <0.20 con la ganancia real de arbitraje con bateria.
La metrica que si correlaciona fuerte (>0.80) es que tan bien el pronostico replica la FORMA de
la curva de precio del dia (Corr-f), no que tan chico es el error puntual promedio.

Esto importa directamente para el motor de decision (OE3): un modelo con MAE mediocre pero que
acierta LA FORMA del dia (cuando sube, cuando baja, donde esta el pico) puede ser mas util para
una regla de compra/venta que un modelo con mejor MAE pero que aplana la curva.

Metricas implementadas (las 3 mas accionables del paper, se omite Cov-e por ser mas compleja de
interpretar y menos correlacionada con la ganancia que Corr-f/MPD segun el propio paper):

  Corr-f (Asociacion): correlacion de Spearman entre el perfil horario pronosticado y el real,
                        PROMEDIADA POR DIA. La metrica mas correlacionada con ganancia real (~0.80+).
  MHD (Min-Max Hour Deviation): diferencia absoluta promedio, en horas, entre cuando el pronostico
                        dice que sera el minimo/maximo del dia y cuando realmente fue.
  MPD (Min-Max Price Deviation): diferencia promedio en el SPREAD diario (precio_max - precio_min)
                        entre lo pronosticado y lo real -- refleja directamente cuanta oportunidad
                        de arbitraje se pierde o se sobreestima.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def _por_dia(fecha_hora, real, pred):
    df = pd.DataFrame({"fecha_hora": pd.to_datetime(fecha_hora), "real": real, "pred": pred})
    df["fecha"] = df["fecha_hora"].dt.normalize()
    return df


def corr_f(fecha_hora, real, pred, horas_min=20):
    """Correlacion de Spearman entre perfil horario pronosticado y real, promediada por dia.
    Requiere al menos `horas_min` horas en el dia para calcularla (dias incompletos se ignoran)."""
    df = _por_dia(fecha_hora, real, pred)
    correlaciones = []
    for _, grupo in df.groupby("fecha"):
        if len(grupo) < horas_min or grupo["real"].std() == 0 or grupo["pred"].std() == 0:
            continue
        rho, _ = spearmanr(grupo["real"], grupo["pred"])
        if not np.isnan(rho):
            correlaciones.append(rho)
    return float(np.mean(correlaciones)) if correlaciones else np.nan


def mhd(fecha_hora, real, pred, horas_min=20):
    """Min-Max Hour Deviation (formula exacta de Maciejowska et al. 2026, ec. en Sec. 4.2.3):
    MHD = (1/T) * sum_t [ |h_min_real - h_min_pred| + |h_max_real - h_max_pred| ]
    Es la SUMA de las dos desviaciones por dia, promediada sobre los dias -- no el promedio de
    las 2T desviaciones sueltas (eso da la mitad del valor correcto)."""
    df = _por_dia(fecha_hora, real, pred)
    df["hora"] = df["fecha_hora"].dt.hour
    sumas_por_dia = []
    for _, grupo in df.groupby("fecha"):
        if len(grupo) < horas_min:
            continue
        hora_min_real = grupo.loc[grupo["real"].idxmin(), "hora"]
        hora_max_real = grupo.loc[grupo["real"].idxmax(), "hora"]
        hora_min_pred = grupo.loc[grupo["pred"].idxmin(), "hora"]
        hora_max_pred = grupo.loc[grupo["pred"].idxmax(), "hora"]
        sumas_por_dia.append(abs(hora_min_real - hora_min_pred) + abs(hora_max_real - hora_max_pred))
    return float(np.mean(sumas_por_dia)) if sumas_por_dia else np.nan


def mpd(fecha_hora, real, pred, horas_min=20):
    """Min-Max Price Deviation (formula exacta de Maciejowska et al. 2026, ec. en Sec. 4.2.3):
    MPD = (1/T) * sum_t [ |P_real[h_min_real] - P_real[h_min_pred]| + |P_real[h_max_real] - P_real[h_max_pred]| ]
    OJO: compara precios REALES en dos horas distintas (la hora optima real vs. la hora que el
    pronostico hubiera indicado operar) -- NO es la diferencia de amplitud (spread) entre el
    pronostico y lo real, que es una metrica distinta (mas parecida a Cov-e)."""
    df = _por_dia(fecha_hora, real, pred)
    df["hora"] = df["fecha_hora"].dt.hour
    sumas_por_dia = []
    for _, grupo in df.groupby("fecha"):
        if len(grupo) < horas_min:
            continue
        hora_min_real = grupo.loc[grupo["real"].idxmin(), "hora"]
        hora_max_real = grupo.loc[grupo["real"].idxmax(), "hora"]
        hora_min_pred = grupo.loc[grupo["pred"].idxmin(), "hora"]
        hora_max_pred = grupo.loc[grupo["pred"].idxmax(), "hora"]
        precio_en = lambda h: grupo.loc[grupo["hora"] == h, "real"].iloc[0]
        sumas_por_dia.append(abs(precio_en(hora_min_real) - precio_en(hora_min_pred)) +
                              abs(precio_en(hora_max_real) - precio_en(hora_max_pred)))
    return float(np.mean(sumas_por_dia)) if sumas_por_dia else np.nan


def evaluar_modelo(fecha_hora, real, pred, nombre=""):
    """Corre las 3 metricas de decision + MAE de referencia, en un solo dict."""
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    mae = np.abs(real - pred).mean()
    return {
        "modelo": nombre,
        "mae": mae,
        "corr_f": corr_f(fecha_hora, real, pred),
        "mhd_horas": mhd(fecha_hora, real, pred),
        "mpd_cop_kwh": mpd(fecha_hora, real, pred),
    }


if __name__ == "__main__":
    # --- Demostracion con datos ya disponibles: ensamble de ventanas N-BEATSx (anoche) vs persistencia ---
    from pathlib import Path
    RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

    df = pd.read_csv(RAIZ / "data/processed/resultados/ventanas_predicciones.csv", parse_dates=["ds"])
    pivote = df.pivot_table(index="ds", columns="ventana", values="NBEATSx", aggfunc="mean")
    y = df.groupby("ds")["y"].first().reindex(pivote.index)

    pred_completa = pivote["completa"]
    pred_ensamble = pivote[["1a", "2a", "3a", "5a", "completa"]].mean(axis=1)

    base = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    base = base.set_index("fecha_hora")
    persistencia = base["precio_lag24h"].reindex(pivote.index)

    resultados = [
        evaluar_modelo(pivote.index, y, persistencia, "Persistencia"),
        evaluar_modelo(pivote.index, y, pred_completa, "N-BEATSx (ventana completa)"),
        evaluar_modelo(pivote.index, y, pred_ensamble, "N-BEATSx (ensamble ventanas)"),
    ]

    res = pd.DataFrame(resultados)
    print("=" * 78)
    print("METRICAS DE DECISION -- holdout 2026")
    print("=" * 78)
    print(res.round(4).to_string(index=False))
    print("\nLectura: MAE mas bajo no siempre implica mejor Corr-f/MHD/MPD -- son cosas distintas.")
    print("Para el motor de decision (OE3), Corr-f alto y MPD bajo importan mas que el MAE solo.")

    res.to_csv(RAIZ / "data/processed/resultados/metricas_decision_demo.csv", index=False)
