# -*- coding: utf-8 -*-
"""
Prueba seria del efecto de los festivos colombianos, en 5 codificaciones distintas.

La prueba anterior (flag binario 'es_festivo' solo en el dia objetivo) empeoro el modelo.
La hipotesis de por que fallo: el problema no es solo que el dia objetivo sea festivo, sino
que los REZAGOS (t-24h, t-72h, t-168h) caigan sobre un festivo -- ahi la feature le esta
dando al modelo un precio anomalo como si fuera un dia normal. Un martes normal cuyo
precio_lag168h cae en un martes festivo ve 108 COP/kWh en vez de ~348.

Configuraciones:
  BASE  : features actuales del pipeline, sin nada de festivos (control)
  F1    : + flags del dia objetivo (es_festivo, vispera, dia despues)   <- lo que ya fallo
  F2    : + flags de CONTAMINACION de rezagos (festivo en t-24/48/72/168h) + mismatch
  F3    : + dia de semana EFECTIVO (festivo se codifica como domingo), reemplaza dia_semana_sin/cos
  F4    : + rezagos CORREGIDOS (si t-168h fue festivo, usa t-336h; referencia de dia similar)
  F5    : todo combinado

Se mide sobre el holdout de 2026, y ademas del MAE global se reporta el MAE en los
subconjuntos que deberian mejorar si la hipotesis es correcta:
  - horas que son festivo
  - horas cuyo rezago semanal (t-168h) cayo en festivo  <- el caso que motiva todo esto
  - horas cuyo rezago de 24h cayo en festivo

REVERSIBLE: todo vive en scripts_experimento/. Para deshacer, borrar la carpeta.
"""
import pandas as pd
import numpy as np
import holidays
import xgboost as xgb
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_SALIDA = RAIZ / "data" / "processed" / "resultados" / "experimento_festivos.csv"
FECHA_TEST = "2026-01-01"

# Config de XGBoost: la misma en todas las configuraciones, para que la comparacion sea limpia.
PARAMS_XGB = dict(n_estimators=2000, max_depth=3, learning_rate=0.01,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

FEATURES_BASE = [
    "volumen_embalses", "aportes_hidricos", "oni", "es_pandemia",
    "precio_lag24h", "precio_lag168h", "precio_media_24h", "precio_media_7d", "precio_media_30d",
    "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos", "dia_anio_sin", "dia_anio_cos",
    "volumen_embalses_delta_1d", "volumen_embalses_delta_7d", "volumen_embalses_media_7d",
    "volumen_embalses_media_30d", "volumen_embalses_vs_media30d",
    "aportes_hidricos_delta_1d", "aportes_hidricos_delta_7d", "aportes_hidricos_media_7d",
    "aportes_hidricos_media_30d", "aportes_hidricos_vs_media30d",
    "precio_std_24h", "precio_std_7d", "precio_rango_24h", "ratio_volatilidad",
    "demanda_lag24h", "demanda_lag48h", "demanda_lag72h", "demanda_media_24h",
]


def construir_features_festivos(df):
    """Agrega todas las variantes de features de festivo. No borra nada del original."""
    festivos = set(holidays.CO(years=range(2018, 2028)).keys())

    fecha = df["fecha_hora"].dt.normalize()

    def es_festivo_en(desplazamiento_dias):
        objetivo = (fecha - pd.Timedelta(days=desplazamiento_dias)).dt.date
        return objetivo.map(lambda d: d in festivos).astype(int)

    # --- Dia objetivo ---
    df["es_festivo"] = es_festivo_en(0)
    df["es_vispera_festivo"] = es_festivo_en(-1)      # manana es festivo
    df["es_dia_despues_festivo"] = es_festivo_en(1)   # ayer fue festivo

    # --- Contaminacion de rezagos: el rezago cae sobre un festivo? ---
    df["festivo_lag24h"] = es_festivo_en(1)
    df["festivo_lag48h"] = es_festivo_en(2)
    df["festivo_lag72h"] = es_festivo_en(3)
    df["festivo_lag168h"] = es_festivo_en(7)

    # "mismatch": el dia objetivo y el dia del rezago son de tipo distinto -> el rezago
    # no es comparable. Esta es la senal mas directa de la hipotesis.
    df["mismatch_festivo_24h"] = (df["es_festivo"] != df["festivo_lag24h"]).astype(int)
    df["mismatch_festivo_168h"] = (df["es_festivo"] != df["festivo_lag168h"]).astype(int)

    # --- Dia de semana efectivo: un festivo se comporta como domingo ---
    dia_semana = df["fecha_hora"].dt.dayofweek  # lunes=0 ... domingo=6
    dia_efectivo = dia_semana.where(df["es_festivo"] == 0, 6)
    df["dia_efectivo_sin"] = np.sin(2 * np.pi * dia_efectivo / 7)
    df["dia_efectivo_cos"] = np.cos(2 * np.pi * dia_efectivo / 7)

    # --- Rezagos corregidos ---
    # Serie de precio indexada por hora, para poder buscar t-336h.
    serie = df.set_index("fecha_hora")["precio_bolsa"]
    precio_lag336h = df["fecha_hora"].map(lambda t: serie.get(t - pd.Timedelta(hours=336), np.nan))

    # Si el rezago semanal cayo en un dia de tipo distinto al objetivo, usa el de 2 semanas atras.
    df["precio_lag168h_ajustado"] = np.where(
        df["mismatch_festivo_168h"] == 1,
        precio_lag336h.fillna(df["precio_lag168h"]),
        df["precio_lag168h"],
    )

    # Referencia de "dia similar": ultimo dia del mismo tipo (laboral vs no-laboral).
    # No-laboral = domingo o festivo. Se busca hacia atras hasta 14 dias, misma hora.
    es_no_laboral = ((dia_semana == 6) | (df["es_festivo"] == 1)).values
    mapa_no_laboral = pd.Series(es_no_laboral, index=df["fecha_hora"].values)

    def referencia_similar(t, objetivo_no_laboral):
        for dias in range(1, 15):
            candidato = t - pd.Timedelta(days=dias)
            tipo = mapa_no_laboral.get(candidato, None)
            if tipo is not None and bool(tipo) == objetivo_no_laboral:
                valor = serie.get(candidato, np.nan)
                if not pd.isna(valor):
                    return valor
        return np.nan

    df["precio_dia_similar"] = [
        referencia_similar(t, bool(nl)) for t, nl in zip(df["fecha_hora"], es_no_laboral)
    ]
    df["precio_dia_similar"] = df["precio_dia_similar"].fillna(df["precio_lag168h"])

    return df


CONFIGURACIONES = {
    "BASE": {"agregar": [], "quitar": []},
    "F1_dia_objetivo": {
        "agregar": ["es_festivo", "es_vispera_festivo", "es_dia_despues_festivo"],
        "quitar": [],
    },
    "F2_contaminacion_lags": {
        "agregar": ["es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
                     "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"],
        "quitar": [],
    },
    "F3_dia_efectivo": {
        "agregar": ["es_festivo", "dia_efectivo_sin", "dia_efectivo_cos"],
        "quitar": ["dia_semana_sin", "dia_semana_cos"],
    },
    "F4_lags_corregidos": {
        "agregar": ["es_festivo", "precio_lag168h_ajustado", "precio_dia_similar"],
        "quitar": ["precio_lag168h"],
    },
    "F5_todo": {
        "agregar": ["es_festivo", "es_vispera_festivo", "es_dia_despues_festivo",
                     "festivo_lag24h", "festivo_lag48h", "festivo_lag72h", "festivo_lag168h",
                     "mismatch_festivo_24h", "mismatch_festivo_168h",
                     "dia_efectivo_sin", "dia_efectivo_cos",
                     "precio_lag168h_ajustado", "precio_dia_similar"],
        "quitar": ["dia_semana_sin", "dia_semana_cos", "precio_lag168h"],
    },
}


def main():
    print("Cargando datos...")
    df_train = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    df_test = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([df_train, df_test], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

    print("Construyendo features de festivos...")
    df = construir_features_festivos(df)

    columnas_necesarias = FEATURES_BASE + [
        "es_festivo", "es_vispera_festivo", "es_dia_despues_festivo",
        "festivo_lag24h", "festivo_lag48h", "festivo_lag72h", "festivo_lag168h",
        "mismatch_festivo_24h", "mismatch_festivo_168h",
        "dia_efectivo_sin", "dia_efectivo_cos",
        "precio_lag168h_ajustado", "precio_dia_similar", "precio_bolsa",
    ]
    df = df.dropna(subset=columnas_necesarias).reset_index(drop=True)

    train = df[df["fecha_hora"] < FECHA_TEST]
    test = df[df["fecha_hora"] >= FECHA_TEST].reset_index(drop=True)
    print(f"Train: {len(train):,} | Test 2026: {len(test):,}")

    # Subconjuntos de interes dentro del test
    mask_festivo = test["es_festivo"].values == 1
    mask_lag168_contaminado = (test["mismatch_festivo_168h"].values == 1)
    mask_lag24_contaminado = (test["mismatch_festivo_24h"].values == 1)
    print(f"  Horas festivas en test: {mask_festivo.sum()}")
    print(f"  Horas con rezago 168h contaminado: {mask_lag168_contaminado.sum()}")
    print(f"  Horas con rezago 24h contaminado: {mask_lag24_contaminado.sum()}\n")

    y_test = test["precio_bolsa"].values
    resultados = []

    for nombre, cfg in CONFIGURACIONES.items():
        features = [f for f in FEATURES_BASE if f not in cfg["quitar"]] + cfg["agregar"]

        modelo = xgb.XGBRegressor(**PARAMS_XGB)
        modelo.fit(train[features], np.log(train["precio_bolsa"]))
        pred = np.exp(modelo.predict(test[features]))

        err = np.abs(y_test - pred)
        fila = {
            "config": nombre,
            "n_features": len(features),
            "mae": err.mean(),
            "mape": (err / y_test).mean() * 100,
            "rmse": np.sqrt(((y_test - pred) ** 2).mean()),
            "mae_festivos": err[mask_festivo].mean(),
            "mae_lag168_contaminado": err[mask_lag168_contaminado].mean(),
            "mae_lag24_contaminado": err[mask_lag24_contaminado].mean(),
        }
        resultados.append(fila)
        print(f"{nombre:24s} MAE {fila['mae']:6.2f}  MAPE {fila['mape']:5.2f}%  "
              f"| festivos {fila['mae_festivos']:6.2f}  "
              f"lag168-contam {fila['mae_lag168_contaminado']:6.2f}  "
              f"lag24-contam {fila['mae_lag24_contaminado']:6.2f}")

    df_res = pd.DataFrame(resultados)
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(RUTA_SALIDA, index=False)

    base = df_res[df_res["config"] == "BASE"].iloc[0]
    print("\n" + "=" * 70)
    print("DIFERENCIA CONTRA BASE (negativo = mejora)")
    print("=" * 70)
    for _, fila in df_res.iterrows():
        if fila["config"] == "BASE":
            continue
        print(f"{fila['config']:24s} MAE {fila['mae']-base['mae']:+6.2f}  "
              f"festivos {fila['mae_festivos']-base['mae_festivos']:+6.2f}  "
              f"lag168-contam {fila['mae_lag168_contaminado']-base['mae_lag168_contaminado']:+6.2f}")
    print(f"\nGuardado: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
