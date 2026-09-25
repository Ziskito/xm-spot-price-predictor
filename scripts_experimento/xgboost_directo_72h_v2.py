# -*- coding: utf-8 -*-
"""
MEJORA 72h #5: XGBoost directo v2 -- anclas estacionales multiples y objetivo cuantilico.

La v1 del modelo directo ya le gano a N-BEATSx en los tramos largos, y su variable mas importante
(23% de la importancia) fue "precio_mismo_hora_disp": el precio de la misma hora del dia mas
reciente que esta disponible en el corte. Eso sugiere que el modelo se apoya fuerte en anclas de
estacionalidad, asi que aqui se le dan mas y mejores:

  - precio en t-168, t-336, t-504 (misma hora hace 1, 2 y 3 semanas). Las tres son SIEMPRE conocidas
    en el corte porque el horizonte maximo es 72h < 168h, asi que no hay fuga.
  - la mediana de esas tres (ancla semanal robusta a un dia atipico aislado)
  - la pendiente entre ellas (tendencia semanal del nivel de precio de esa hora)
  - precio en el corte menos el ancla semanal: mide cuanto se ha despegado el precio actual de su
    normal semanal, que es justamente la senal de que hay un evento de regimen en curso.

Ademas se entrenan tres cuantiles (0.1/0.5/0.9) con objetivo pinball, para que el modelo de arboles
aporte BANDAS propias al ensamble y no dependa solo de la dispersion de las redes neuronales.
"""
import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
HORIZONTE = 72
PASO_MUESTREO_TRAIN = 4

COLS_ESTADO = [
    "precio_bolsa", "precio_media_24h", "precio_media_7d", "precio_media_30d",
    "precio_std_24h", "precio_std_7d", "precio_rango_24h", "ratio_volatilidad",
    "volumen_embalses", "volumen_embalses_vs_media30d", "volumen_embalses_delta_1d",
    "volumen_embalses_delta_7d", "aportes_hidricos", "aportes_hidricos_vs_media30d",
    "aportes_hidricos_delta_1d", "aportes_hidricos_delta_7d",
    "demanda", "demanda_media_24h", "generacion",
]
COLS_FUTURO = [
    "oni", "es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
    "dia_anio_sin", "dia_anio_cos", "es_festivo", "festivo_lag24h", "festivo_lag48h",
    "festivo_lag72h", "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h",
]
ANCLAS = [168, 336, 504]  # todas > 72, siempre conocidas en el corte


def construir(df, cortes_validos, pasos):
    n = len(df)
    estado = df[COLS_ESTADO].to_numpy(dtype=np.float32)
    futuro = df[COLS_FUTURO].to_numpy(dtype=np.float32)
    precio = df["precio_bolsa"].to_numpy(dtype=np.float32)
    fechas = df["fecha_hora"].to_numpy()
    set_cortes = set(cortes_validos.tolist())
    partes = []

    for h in pasos:
        t_idx = np.arange(max(ANCLAS) + 1, n)
        c_idx = t_idx - h
        mask = np.array([c in set_cortes for c in c_idx])
        if not mask.any():
            continue
        t_idx, c_idx = t_idx[mask], c_idx[mask]

        retro = 24 * int(np.ceil(h / 24))
        mismo = precio[t_idx - retro]
        anclas = np.stack([precio[t_idx - a] for a in ANCLAS], axis=1)
        mediana = np.median(anclas, axis=1)
        pendiente = anclas[:, 0] - anclas[:, 2]          # semana pasada vs hace 3 semanas
        despegue = precio[c_idx] - mediana                # precio actual vs su normal semanal

        bloque = np.hstack([
            estado[c_idx], futuro[t_idx],
            mismo.reshape(-1, 1), anclas, mediana.reshape(-1, 1),
            pendiente.reshape(-1, 1), despegue.reshape(-1, 1),
            np.full((len(t_idx), 1), h, dtype=np.float32),
        ])
        cols = (COLS_ESTADO + COLS_FUTURO + ["precio_mismo_hora_disp"]
                + [f"ancla_{a}h" for a in ANCLAS]
                + ["ancla_mediana", "ancla_pendiente", "despegue_semanal", "paso_horas"])
        sub = pd.DataFrame(bloque, columns=cols)
        sub["y"] = precio[t_idx]
        sub["fecha_hora"] = fechas[t_idx]
        sub["cutoff"] = fechas[c_idx]
        partes.append(sub)
    return pd.concat(partes, ignore_index=True)


def main():
    t0 = time.time()
    import xgboost as xgb

    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    pos = pd.Series(df.index.values, index=df["fecha_hora"])

    ref = pd.read_csv(RES / "pronostico_con_bandas_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    idx_test = pos.reindex(pd.Index(sorted(ref["cutoff"].unique()))).dropna().astype(int).to_numpy()
    idx_fin_train = int(pos[pd.Timestamp("2026-01-01")])
    idx_train = np.arange(max(ANCLAS) + 50, idx_fin_train - HORIZONTE, PASO_MUESTREO_TRAIN)

    pasos = np.arange(1, HORIZONTE + 1)
    print("Construyendo datasets...")
    train = construir(df, idx_train, pasos).dropna()
    test = construir(df, idx_test, pasos).dropna()
    cols_x = [c for c in train.columns if c not in ("y", "fecha_hora", "cutoff")]
    print(f"Train: {len(train):,} filas x {len(cols_x)} variables | Test: {len(test):,} filas")

    test = test.copy()
    salida = test[["fecha_hora", "cutoff", "paso_horas", "y"]].rename(columns={"y": "real"})
    salida["paso_horas"] = salida["paso_horas"].astype(int)

    for q, nom in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        print(f"Entrenando cuantil {q}...")
        m = xgb.XGBRegressor(
            n_estimators=1200, max_depth=8, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            objective="reg:quantileerror", quantile_alpha=q,
            random_state=42, n_jobs=-1, tree_method="hist",
        )
        m.fit(train[cols_x], train["y"])
        salida[nom] = m.predict(test[cols_x])
        if nom == "q50":
            imp = pd.Series(m.feature_importances_, index=cols_x).sort_values(ascending=False)

    salida.to_csv(RES / "pronostico_xgb_directo_v2_72h_2026.csv", index=False)

    v1 = pd.read_csv(RES / "pronostico_xgb_directo_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    print(f"\n{'='*72}\n{'tramo':9s} {'XGBdir-v2':>10s} {'XGBdir-v1':>10s} {'N-BEATSx':>10s} {'cobertura':>10s}")
    for nombre, ini, fin in [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]:
        s = salida[(salida.paso_horas >= ini) & (salida.paso_horas <= fin)]
        u = v1[(v1.paso_horas >= ini) & (v1.paso_horas <= fin)]
        r = ref[(ref.paso_horas >= ini) & (ref.paso_horas <= fin)]
        cob = ((s["real"] >= s["q10"]) & (s["real"] <= s["q90"])).mean() * 100
        print(f"{nombre:9s} {np.abs(s.real-s.q50).mean():10.2f} {np.abs(u.real-u.pred).mean():10.2f} "
              f"{np.abs(r.real-r.q50).mean():10.2f} {cob:9.1f}%")

    print(f"\nTop 12 variables:\n{imp.head(12).to_string()}")
    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
