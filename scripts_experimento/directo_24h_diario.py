# -*- coding: utf-8 -*-
"""
Los modelos de formulacion DIRECTA (CatBoost y Ridge) resultaron ser los mejores votantes nuevos
del ensamble de 72h. Aqui se reentrenan como especialistas de 24h (pasos 1-24 solamente) y se
evaluan con el MISMO protocolo day-ahead del ensamble ganador de 24h: un corte diario a las 23:00,
pronosticando las 24 horas del dia siguiente, todo 2026.

Objetivo: sumarlos como votantes nuevos al stacking de 24h. Aportan dos cosas que ese ensamble no
tiene: (1) un modelo lineal en NIVELES de precio (ARX+GARCH es lineal pero en logaritmos), y
(2) el precio observado en el propio corte (23:00), que los modelos recursivos no usan porque
estan anclados a precio_lag24h.
"""
import pandas as pd
import numpy as np
import time
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from xgboost_directo_72h import construir_largo, COLS_ESTADO, COLS_FUTURO, PASO_MUESTREO_TRAIN

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
H = 24


def main():
    t0 = time.time()
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    pos = pd.Series(df.index.values, index=df["fecha_hora"])

    cortes = pd.date_range("2025-12-31 23:00", df["fecha_hora"].max() - pd.Timedelta(hours=H), freq="24h")
    idx_test = pos.reindex(cortes).dropna().astype(int).to_numpy()
    idx_fin = int(pos[pd.Timestamp("2026-01-01")])
    idx_train = np.arange(200, idx_fin - H, PASO_MUESTREO_TRAIN)  # objetivos siempre < 2026
    print(f"Cortes de prueba (diarios, 23:00): {len(idx_test)} | cortes de entrenamiento: {len(idx_train)}")

    pasos = np.arange(1, H + 1)
    train = construir_largo(df, idx_train, pasos).dropna()
    test = construir_largo(df, idx_test, pasos).dropna().copy()
    cols_x = COLS_ESTADO + COLS_FUTURO + ["precio_mismo_hora_disp", "paso_horas"]
    print(f"Train: {len(train):,} filas | Test: {len(test):,} filas")

    cat = CatBoostRegressor(n_estimators=1500, max_depth=8, learning_rate=0.05,
                            loss_function="MAE", random_seed=42, verbose=0, thread_count=-1)
    cat.fit(train[cols_x], train["y"])
    esc = StandardScaler().fit(train[cols_x])
    ridge = Ridge(alpha=10.0).fit(esc.transform(train[cols_x]), train["y"])

    salida = test[["fecha_hora", "cutoff", "paso_horas", "y"]].rename(columns={"y": "real"})
    salida["paso_horas"] = salida["paso_horas"].astype(int)
    salida["pred_catb"] = cat.predict(test[cols_x])
    salida["pred_ridge"] = ridge.predict(esc.transform(test[cols_x]))
    salida.to_csv(RES / "pronostico_directo_24h_diario_2026.csv", index=False)

    crudas = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    nb = crudas[(crudas.origen == "Origen 6") & (crudas.modelo == "N-BEATSx")][["fecha_hora", "prediccion"]]
    m = salida.merge(nb, on="fecha_hora")
    print(f"\nMismas {len(m)} horas de 2026:")
    for col, nom in [("pred_catb", "CatBoost directo 24h"), ("pred_ridge", "Ridge directo 24h"),
                     ("prediccion", "N-BEATSx (referencia)")]:
        e = np.abs(m["real"] - m[col])
        print(f"  {nom:24s} MAE={e.mean():6.2f}  MAPE={(e/m['real']).mean()*100:5.2f}%")
    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
