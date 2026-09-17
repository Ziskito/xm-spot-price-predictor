# -*- coding: utf-8 -*-
"""
MEJORA 72h #6: refuerzo de los votantes directos.

Dos ajustes sobre el ensamble v2, motivados por lo que mostraron los pesos NNLS:

1) CatBoost-directo se lleva el peso dominante en los tramos largos (0.49 en 49-72h) y es el mejor
   modelo individual del horizonte largo. Se reentrena con el DOBLE de cortes de entrenamiento
   (uno cada 2 horas en vez de cada 4): 2.2M filas en vez de 1.1M. El submuestreo original se
   eligio por memoria, no por criterio estadistico, asi que aqui se comprueba si estaba limitando.

2) Se agrega un modelo LINEAL directo (Ridge sobre las mismas regresoras, estandarizadas). No se
   espera que gane individualmente -- se espera que DIVERSIFIQUE. Hoy el ensamble tiene solo dos
   familias con peso real (redes neuronales y arboles), y la correlacion de errores entre XGBoost
   y CatBoost resulto ser 0.957, es decir, casi el mismo modelo. Un lineal tiene un sesgo de
   aprendizaje genuinamente distinto (extrapola, los arboles no), y en el ensamble ganador de 24h
   el votante lineal-econometrico (ARX+GARCH) si se llevo peso propio.
"""
import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).parent))
from xgboost_directo_72h import construir_largo, COLS_ESTADO, COLS_FUTURO, HORIZONTE

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
MUESTREO_DENSO = 2

TRAMOS = [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]


def main():
    t0 = time.time()
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    pos = pd.Series(df.index.values, index=df["fecha_hora"])

    ref = pd.read_csv(RES / "pronostico_con_bandas_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    idx_test = pos.reindex(pd.Index(sorted(ref["cutoff"].unique()))).dropna().astype(int).to_numpy()
    idx_fin = int(pos[pd.Timestamp("2026-01-01")])
    idx_train = np.arange(200, idx_fin - HORIZONTE, MUESTREO_DENSO)

    pasos = np.arange(1, HORIZONTE + 1)
    print(f"Construyendo dataset denso (1 corte cada {MUESTREO_DENSO}h)...")
    train = construir_largo(df, idx_train, pasos).dropna()
    test = construir_largo(df, idx_test, pasos).dropna()
    cols_x = COLS_ESTADO + COLS_FUTURO + ["precio_mismo_hora_disp", "paso_horas"]
    print(f"Train: {len(train):,} filas | Test: {len(test):,} filas")

    test = test.copy()
    base = test[["fecha_hora", "cutoff", "paso_horas", "y"]].rename(columns={"y": "real"})
    base["paso_horas"] = base["paso_horas"].astype(int)

    # ---- CatBoost denso ----
    print("\nEntrenando CatBoost directo con dataset denso...")
    cat = CatBoostRegressor(n_estimators=2000, max_depth=8, learning_rate=0.05,
                            loss_function="MAE", random_seed=42, verbose=500, thread_count=-1)
    cat.fit(train[cols_x], train["y"])
    sal_cat = base.copy()
    sal_cat["pred"] = cat.predict(test[cols_x])
    sal_cat.to_csv(RES / "pronostico_catboost_denso_72h_2026.csv", index=False)

    # ---- Ridge directo ----
    print("Entrenando Ridge directo...")
    esc = StandardScaler()
    Xtr = esc.fit_transform(train[cols_x])
    Xte = esc.transform(test[cols_x])
    ridge = Ridge(alpha=10.0)
    ridge.fit(Xtr, train["y"])
    sal_r = base.copy()
    sal_r["pred"] = ridge.predict(Xte)
    sal_r.to_csv(RES / "pronostico_ridge_directo_72h_2026.csv", index=False)

    # ---- comparacion ----
    cat_ant = pd.read_csv(RES / "pronostico_catboost_directo_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    print(f"\n{'='*76}\n{'tramo':9s} {'CatB-denso':>11s} {'CatB-orig':>11s} {'Ridge-dir':>11s} {'N-BEATSx':>10s}")
    for nombre, ini, fin in TRAMOS:
        c = sal_cat[(sal_cat.paso_horas >= ini) & (sal_cat.paso_horas <= fin)]
        o = cat_ant[(cat_ant.paso_horas >= ini) & (cat_ant.paso_horas <= fin)]
        r = sal_r[(sal_r.paso_horas >= ini) & (sal_r.paso_horas <= fin)]
        n = ref[(ref.paso_horas >= ini) & (ref.paso_horas <= fin)]
        print(f"{nombre:9s} {np.abs(c.real-c.pred).mean():11.2f} {np.abs(o.real-o.pred).mean():11.2f} "
              f"{np.abs(r.real-r.pred).mean():11.2f} {np.abs(n.real-n.q50).mean():10.2f}")

    m = sal_cat.merge(sal_r, on=["fecha_hora", "cutoff", "paso_horas"], suffixes=("_c", "_r"))
    print(f"\nCorrelacion de errores Ridge-directo vs CatBoost-denso: "
          f"{np.corrcoef(m.real_c-m.pred_c, m.real_r-m.pred_r)[0,1]:.4f}")
    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
