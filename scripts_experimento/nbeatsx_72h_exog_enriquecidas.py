# -*- coding: utf-8 -*-
"""
MEJORA 72h #2: enriquecer las exogenas de N-BEATSx a 72h.

Hallazgo de la revision: el pipeline de 72h (bandas_incertidumbre_nbeatsx_72h.py) usa solo 3
variables historicas -- volumen_embalses, aportes_hidricos, demanda_lag24h -- de las 47 disponibles
en el dataset. Faltan justamente las versiones NORMALIZADAS de la hidrologia:

  - volumen_embalses en metros cubicos absolutos (~1.3e10) es NO ESTACIONARIA: deriva entre anios
    y su nivel absoluto no significa lo mismo en 2019 que en 2026.
  - volumen_embalses_vs_media30d (el mismo dato dividido por su media movil de 30 dias) SI es
    comparable entre regimenes: dice "que tan lleno esta el embalse respecto a su normal reciente",
    que es exactamente la senal de estres hidrico que mueve el precio.

Esto ataca directamente el problema que el usuario planteo (el modelo no se adapta a cambios de
regimen tipo El Nino): una variable estacionaria y relativa transfiere entre regimenes, una
variable de nivel absoluto no.

Ademas se agrega dia_anio_sin/cos (estacionalidad anual, que el pipeline de 24h si tenia y el de
72h no) y los armonicos de segundo orden de la hora, que ya demostraron aportar en el experimento
de Optuna de anoche.

Todo lo demas (horizonte, input_size, MQLoss, protocolo de validacion cruzada, semilla) se deja
IDENTICO al script original para que la comparacion sea limpia y atribuible solo a las exogenas.
"""
import pandas as pd
import numpy as np
import time
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

FECHA_TEST = pd.Timestamp("2026-01-01")
HORIZONTE, INPUT_SIZE, MAX_STEPS = 72, 168, 1000
CUANTILES = [0.1, 0.5, 0.9]

# ORIGINAL (3) + las normalizadas/derivadas que faltaban
HIST_EXOG = [
    "volumen_embalses", "aportes_hidricos", "demanda_lag24h",
    "volumen_embalses_vs_media30d", "aportes_hidricos_vs_media30d",
    "volumen_embalses_delta_7d", "aportes_hidricos_delta_7d",
    "demanda_media_24h", "generacion",
]
FUTR_EXOG_CONT = ["oni"]
FUTR_EXOG_BIN = [
    "es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
    "es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
    "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h",
    # nuevas: estacionalidad anual (faltaba en el pipeline de 72h) + armonicos de 2do orden
    "dia_anio_sin", "dia_anio_cos", "hora_sin2", "hora_cos2",
]


def main(semilla=42, etiqueta="exog"):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MQLoss

    ruta_salida = RAIZ / f"data/processed/resultados/pronostico_nbeatsx_72h_{etiqueta}_2026.csv"

    t0 = time.time()
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

    # armonicos de segundo orden de la hora del dia (serie de Fourier truncada)
    ang = 2 * np.pi * df["hora"] / 24
    df["hora_sin2"] = np.sin(2 * ang)
    df["hora_cos2"] = np.cos(2 * ang)

    futr_exog = FUTR_EXOG_CONT + FUTR_EXOG_BIN
    df_nf = df[["fecha_hora", "precio_bolsa"] + HIST_EXOG + futr_exog].copy()
    mascara_train = df["fecha_hora"] < FECHA_TEST
    for col in HIST_EXOG + FUTR_EXOG_CONT:
        mu = df_nf.loc[mascara_train.values, col].mean()
        sigma = df_nf.loc[mascara_train.values, col].std()
        df_nf[col] = (df_nf[col] - mu) / sigma
    df_nf[HIST_EXOG + futr_exog] = df_nf[HIST_EXOG + futr_exog].ffill().bfill()

    df_nf["unique_id"] = "precio_bolsa"
    df_nf = df_nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})
    df_nf = df_nf[["unique_id", "ds", "y"] + HIST_EXOG + futr_exog]

    n_test = int((~mascara_train).sum())
    n_windows = n_test // HORIZONTE
    print(f"N-BEATSx h={HORIZONTE} exogenas ENRIQUECIDAS ({len(HIST_EXOG)} hist + {len(futr_exog)} futr), semilla={semilla}")
    print(f"Train: {int(mascara_train.sum()):,} | Test 2026: {n_test:,} | Ventanas: {n_windows}")

    modelo = NBEATSx(h=HORIZONTE, input_size=INPUT_SIZE, hist_exog_list=HIST_EXOG,
                     futr_exog_list=futr_exog, max_steps=MAX_STEPS, val_check_steps=100,
                     loss=MQLoss(quantiles=CUANTILES), random_seed=semilla,
                     enable_progress_bar=False)
    nf = NeuralForecast(models=[modelo], freq="h")
    cv = nf.cross_validation(df=df_nf, n_windows=n_windows, step_size=HORIZONTE)

    cols_pred = [c for c in cv.columns if c.startswith("NBEATSx")]
    print("Columnas de prediccion:", cols_pred)

    def buscar(patrones):
        for p in patrones:
            for c in cols_pred:
                if p in c:
                    return c
        return None

    col_q10 = buscar(["ql0.1", "lo-80", "-lo-", "q-10"])
    col_q50 = buscar(["median", "ql0.5", "q-50"])
    col_q90 = buscar(["ql0.9", "hi-80", "-hi-", "q-90"])
    if col_q50 is None:
        col_q50 = cols_pred[len(cols_pred) // 2]
    print(f"Mapeo -> q10={col_q10}  q50={col_q50}  q90={col_q90}")

    cv["paso_horas"] = ((cv["ds"] - cv["cutoff"]).dt.total_seconds() / 3600).astype(int)
    salida = pd.DataFrame({
        "fecha_hora": cv["ds"].values, "cutoff": cv["cutoff"].values,
        "paso_horas": cv["paso_horas"].values, "real": cv["y"].values,
        "q10": cv[col_q10].values, "q50": cv[col_q50].values, "q90": cv[col_q90].values,
    })
    salida.to_csv(ruta_salida, index=False)

    ref = pd.read_csv(RAIZ / "data/processed/resultados/pronostico_con_bandas_72h_2026.csv",
                      parse_dates=["fecha_hora", "cutoff"])
    print(f"\n{'='*78}\nDESGLOSE POR TRAMO -- enriquecido vs original\n{'='*78}")
    print(f"{'tramo':9s} {'MAE nuevo':>11s} {'MAE orig':>11s} {'delta':>9s} {'cobertura':>10s}")
    import statsmodels.api as sm
    for nombre, ini, fin in [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]:
        s = salida[(salida.paso_horas >= ini) & (salida.paso_horas <= fin)]
        r = ref[(ref.paso_horas >= ini) & (ref.paso_horas <= fin)]
        mae_n = np.abs(s["real"] - s["q50"]).mean()
        mae_o = np.abs(r["real"] - r["q50"]).mean()
        cob = ((s["real"] >= s["q10"]) & (s["real"] <= s["q90"])).mean() * 100
        comp = s.set_index(["fecha_hora", "cutoff", "paso_horas"]).join(
            r.set_index(["fecha_hora", "cutoff", "paso_horas"])[["q50"]], rsuffix="_o", how="inner")
        d = (comp["real"] - comp["q50_o"]).abs().values - (comp["real"] - comp["q50"]).abs().values
        dm = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
        marca = "GANA(sig)" if dm.pvalues[0] < 0.05 and dm.tvalues[0] > 0 else (
            "PIERDE(sig)" if dm.pvalues[0] < 0.05 else "sin dif.")
        print(f"{nombre:9s} {mae_n:11.2f} {mae_o:11.2f} {mae_n-mae_o:+9.2f} {cob:9.1f}%  DM p={dm.pvalues[0]:.4f} {marca}")

    print(f"\nGuardado: {ruta_salida}")
    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    e = sys.argv[2] if len(sys.argv) > 2 else "exog"
    main(s, e)
