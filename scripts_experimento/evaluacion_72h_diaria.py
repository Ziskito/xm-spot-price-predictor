# -*- coding: utf-8 -*-
"""
Horizonte 72h con cortes DIARIOS (step_size=24): 214 ventanas de 72 horas en vez de 72 ventanas no
solapadas. Casi el triple de ventanas = mucha mas potencia estadistica en los tests.

Cortes a las 00:00 (la evaluacion anterior los tenia a las 23:00) para quedar alineados hora a hora
con el walk-forward de 24h (ventanas 01:00->00:00): asi el especialista de 24h (v4) y los pasos 1-24
del ensamble de 72h salen exactamente del mismo corte y el contrato unificado queda consistente.

Se completan aqui dos intentos pendientes de 72h: promedio de semillas de N-BEATSx (42, 7, 123) y
contexto mas largo (input_size=336 = dos semanas en vez de una).
Votantes de v3 que se omiten por peso ~0 en el ensamble: N-HiTS (peso 0 en los tres tramos) y
XGBoost directo v2 (peso <= 0.08 y 17 minutos de entrenamiento).
Salida: pronostico_72h_diario_votantes_2026.csv (una fila por corte x paso).
"""
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
sys.path.insert(0, str(Path(__file__).parent))
from xgboost_directo_72h import construir_largo, COLS_ESTADO, COLS_FUTURO

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
RUTA = RES / "pronostico_72h_diario_votantes_2026.csv"
H, N_WIN = 72, 214
PRIMER_CORTE = pd.Timestamp("2026-01-01 00:00")
FIN_DF = PRIMER_CORTE + pd.Timedelta(hours=(N_WIN - 1) * 24 + H)
CUANTILES = [0.1, 0.5, 0.9]

HIST = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
HIST_ENR = HIST + ["volumen_embalses_vs_media30d", "aportes_hidricos_vs_media30d", "volumen_embalses_delta_7d",
                   "aportes_hidricos_delta_7d", "demanda_media_24h", "generacion"]
FUT_CONT = ["oni"]
FUT_BIN = ["es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos", "es_festivo", "festivo_lag24h",
           "festivo_lag48h", "festivo_lag72h", "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]
FUT_BIN_ENR = FUT_BIN + ["dia_anio_sin", "dia_anio_cos", "hora_sin2", "hora_cos2"]
CLAVE = ["fecha_hora", "cutoff", "paso_horas"]


def cargar():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    ang = 2 * np.pi * df["hora"] / 24
    df["hora_sin2"], df["hora_cos2"] = np.sin(2 * ang), np.cos(2 * ang)
    return df


def redes(df):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MQLoss
    cont = sorted(set(HIST_ENR + FUT_CONT))
    todas = sorted(set(HIST_ENR + FUT_CONT + FUT_BIN_ENR))
    nf = df[df["fecha_hora"] <= FIN_DF][["fecha_hora", "precio_bolsa"] + todas].copy()
    m = (nf["fecha_hora"] < "2026-01-01").to_numpy()
    for c in cont:
        nf[c] = (nf[c] - nf.loc[m, c].mean()) / nf.loc[m, c].std()
    nf[todas] = nf[todas].ffill().bfill()
    nf["unique_id"] = "precio_bolsa"
    nf = nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})

    def nbx(alias, **kw):
        base = dict(h=H, input_size=168, hist_exog_list=HIST, futr_exog_list=FUT_CONT + FUT_BIN, max_steps=1000,
                    val_check_steps=100, loss=MQLoss(quantiles=CUANTILES), random_seed=42, enable_progress_bar=False)
        base.update(kw)
        return NBEATSx(alias=alias, **base)

    modelos = [nbx("NBX"), nbx("NBX_s7", random_seed=7), nbx("NBX_s123", random_seed=123),
               nbx("NBX_in336", input_size=336),
               nbx("NBX_exog", hist_exog_list=HIST_ENR, futr_exog_list=FUT_CONT + FUT_BIN_ENR)]
    t0 = time.time()
    cv = NeuralForecast(models=modelos, freq="h").cross_validation(df=nf, n_windows=N_WIN, step_size=24)
    print(f"Redes 72h (5 modelos): {(time.time()-t0)/60:.1f} min", flush=True)
    assert cv["cutoff"].min() == PRIMER_CORTE, f"primer corte inesperado: {cv['cutoff'].min()}"
    s = cv.rename(columns={"ds": "fecha_hora", "y": "real"})
    s["paso_horas"] = ((s["fecha_hora"] - s["cutoff"]).dt.total_seconds() / 3600).astype(int)
    out = s[CLAVE + ["real"]].copy()
    for a in ["NBX", "NBX_s7", "NBX_s123", "NBX_in336", "NBX_exog"]:
        out[a] = s[f"{a}-median"].to_numpy()
        out[f"{a}__q10"] = s[f"{a}-lo-80.0"].to_numpy()
        out[f"{a}__q90"] = s[f"{a}-hi-80.0"].to_numpy()
    return out


def directos(df):
    from catboost import CatBoostRegressor
    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    pos = pd.Series(df.index.values, index=df["fecha_hora"])
    cortes = pd.date_range(PRIMER_CORTE, periods=N_WIN, freq="24h")
    idx_test = pos.reindex(cortes).astype(int).to_numpy()
    idx_fin = int(pos[pd.Timestamp("2026-01-01")])
    pasos = np.arange(1, H + 1)
    cols = COLS_ESTADO + COLS_FUTURO + ["precio_mismo_hora_disp", "paso_horas"]

    te = construir_largo(df, idx_test, pasos).dropna().copy()
    te["paso_horas"] = te["paso_horas"].astype(int)
    out = te[["fecha_hora", "cutoff", "paso_horas"]].copy()
    out["NaiveEstac"] = te["precio_lag168h"].to_numpy()

    t0 = time.time()
    tr = construir_largo(df, np.arange(200, idx_fin - H, 4), pasos).dropna()
    print(f"Dataset directo: {len(tr):,} filas de entrenamiento, {len(te):,} de prueba", flush=True)
    m = xgb.XGBRegressor(n_estimators=1200, max_depth=8, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=5, objective="reg:absoluteerror", random_state=42, n_jobs=-1, tree_method="hist")
    out["XGB-directo"] = m.fit(tr[cols], tr["y"]).predict(te[cols])
    print(f"  XGB-directo listo ({(time.time()-t0)/60:.1f} min)", flush=True)
    m = CatBoostRegressor(n_estimators=1500, max_depth=8, learning_rate=0.05, loss_function="MAE",
                          random_seed=42, verbose=0, thread_count=-1)
    out["CatB-directo"] = m.fit(tr[cols], tr["y"]).predict(te[cols])
    print(f"  CatB-directo listo ({(time.time()-t0)/60:.1f} min)", flush=True)
    esc = StandardScaler().fit(tr[cols])
    out["Ridge-directo"] = Ridge(alpha=10.0).fit(esc.transform(tr[cols]), tr["y"]).predict(esc.transform(te[cols]))
    del tr

    tr = construir_largo(df, np.arange(200, idx_fin - H, 2), pasos).dropna()
    m = CatBoostRegressor(n_estimators=2000, max_depth=8, learning_rate=0.05, loss_function="MAE",
                          random_seed=42, verbose=0, thread_count=-1)
    out["CatB-denso"] = m.fit(tr[cols], tr["y"]).predict(te[cols])
    print(f"  CatB-denso listo ({(time.time()-t0)/60:.1f} min)", flush=True)
    return out


def main():
    df = cargar()
    r = redes(df)
    r.to_csv(RUTA, index=False)
    d = directos(df)
    todo = r.merge(d, on=CLAVE, how="inner").sort_values(["cutoff", "paso_horas"])
    todo.to_csv(RUTA, index=False)
    print(f"\nFilas: {len(todo):,} | cortes: {todo['cutoff'].nunique()} | pasos {todo['paso_horas'].min()}-{todo['paso_horas'].max()}")
    todo["NBX_semillas"] = todo[["NBX", "NBX_s7", "NBX_s123"]].mean(axis=1)
    votos = ["NBX", "NBX_s7", "NBX_s123", "NBX_semillas", "NBX_in336", "NBX_exog", "XGB-directo",
             "CatB-directo", "CatB-denso", "Ridge-directo", "NaiveEstac"]
    print(f"\n{'votante':14s} " + " ".join(f"{t:>8s}" for t in ["1-24h", "25-48h", "49-72h"]))
    for v in votos:
        maes = [np.abs(todo.loc[todo.paso_horas.between(i, f), "real"] - todo.loc[todo.paso_horas.between(i, f), v]).mean()
                for i, f in [(1, 24), (25, 48), (49, 72)]]
        print(f"{v:14s} " + " ".join(f"{x:8.2f}" for x in maes))
    print(f"Guardado: {RUTA}")


if __name__ == "__main__":
    main()
