# -*- coding: utf-8 -*-
"""
MEJORA 72h #4: CatBoost directo multi-horizonte -- quinto votante, mas diversidad de familia.

XGBoost directo resulto ser el mejor modelo individual en los tramos largos (25-48h y 49-72h),
asi que vale la pena reforzar esa familia con una implementacion de arboles DISTINTA. CatBoost con
boosting ordenado construye arboles simetricos y maneja el sesgo de prediccion de forma diferente
a XGBoost, asi que sus errores no deberian estar perfectamente correlacionados -- que es justo lo
que el meta-modelo NNLS puede explotar.

Reutiliza exactamente el mismo constructor de dataset largo (misma disciplina anti-fuga, mismos
cortes de prueba) que xgboost_directo_72h.py, para que la comparacion sea limpia.
"""
import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).parent))
from xgboost_directo_72h import construir_largo, COLS_ESTADO, COLS_FUTURO, HORIZONTE, PASO_MUESTREO_TRAIN

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"


def main():
    t0 = time.time()
    from catboost import CatBoostRegressor

    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    pos = pd.Series(df.index.values, index=df["fecha_hora"])

    ref = pd.read_csv(RES / "pronostico_con_bandas_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    idx_cortes_test = pos.reindex(pd.Index(sorted(ref["cutoff"].unique()))).dropna().astype(int).to_numpy()
    idx_fin_train = int(pos[pd.Timestamp("2026-01-01")])
    idx_cortes_train = np.arange(200, idx_fin_train - HORIZONTE, PASO_MUESTREO_TRAIN)

    pasos = np.arange(1, HORIZONTE + 1)
    train = construir_largo(df, idx_cortes_train, pasos).dropna()
    test = construir_largo(df, idx_cortes_test, pasos).dropna()
    cols_x = COLS_ESTADO + COLS_FUTURO + ["precio_mismo_hora_disp", "paso_horas"]
    print(f"Train: {len(train):,} filas | Test: {len(test):,} filas")

    modelo = CatBoostRegressor(n_estimators=1500, max_depth=8, learning_rate=0.05,
                               loss_function="MAE", random_seed=42, verbose=200,
                               thread_count=-1)
    print("Entrenando CatBoost directo multi-horizonte...")
    modelo.fit(train[cols_x], train["y"])
    test = test.copy()
    test["pred"] = modelo.predict(test[cols_x])

    salida = test[["fecha_hora", "cutoff", "paso_horas", "y", "pred"]].rename(columns={"y": "real"})
    salida["paso_horas"] = salida["paso_horas"].astype(int)
    salida.to_csv(RES / "pronostico_catboost_directo_72h_2026.csv", index=False)

    xgb_d = pd.read_csv(RES / "pronostico_xgb_directo_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    print(f"\n{'='*70}\n{'tramo':9s} {'CatB-dir':>10s} {'XGB-dir':>10s} {'N-BEATSx':>10s}")
    for nombre, ini, fin in [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]:
        s = salida[(salida.paso_horas >= ini) & (salida.paso_horas <= fin)]
        x = xgb_d[(xgb_d.paso_horas >= ini) & (xgb_d.paso_horas <= fin)]
        r = ref[(ref.paso_horas >= ini) & (ref.paso_horas <= fin)]
        print(f"{nombre:9s} {np.abs(s.real-s.pred).mean():10.2f} {np.abs(x.real-x.pred).mean():10.2f} "
              f"{np.abs(r.real-r.q50).mean():10.2f}")

    # correlacion de errores con XGBoost: si es < 1 hay diversidad explotable por el ensamble
    m = salida.merge(xgb_d, on=["fecha_hora", "cutoff", "paso_horas"], suffixes=("_cat", "_xgb"))
    e_cat = m["real_cat"] - m["pred_cat"]
    e_xgb = m["real_xgb"] - m["pred_xgb"]
    print(f"\nCorrelacion de errores CatBoost-directo vs XGB-directo: {np.corrcoef(e_cat, e_xgb)[0,1]:.4f}")
    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
