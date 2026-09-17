# -*- coding: utf-8 -*-
"""
MEJORA 72h #1: XGBoost DIRECTO multi-horizonte -- cierra el hueco estructural del ensamble de 72h.

Anoche el ensamble de 72h fallo (no le gano a N-BEATSx solo) porque solo tenia 2 votantes de la
MISMA familia (N-BEATSx y N-HiTS, ambos neuralforecast). La leccion central de toda la sesion es
que al stacking lo hace ganar la DIVERSIDAD ENTRE FAMILIAS, no el acierto individual (en el
ensamble de 24h, Persistencia tiene MAE mucho peor que N-BEATSx y aun asi se lleva 11% del peso).

Se habia documentado que XGBoost "no se puede extender a 25-72h porque esta atado a precio_lag24h".
Eso es cierto SOLO para la formulacion recursiva. La formulacion DIRECTA si se puede: se entrena un
unico modelo que recibe el paso del horizonte (h) como variable, y como regresoras usa unicamente
el estado conocido EN EL MOMENTO DEL CORTE (t-h) mas el calendario/ONI del instante objetivo (t).
Asi un solo modelo cubre los 72 pasos sin fuga de informacion y sin 71 reentrenamientos.

Anti-fuga (lo critico de este script):
  - Las variables de "estado" se toman en t-h (el corte), NUNCA en t.
  - precio_lag168h en t = precio en t-168, que siempre es conocido en el corte porque h<=72<168. SI se usa.
  - precio_lag24h en t = precio en t-24, que NO es conocido en el corte cuando h>24. NO se usa.
    En su lugar se construye "precio_mismo_hora_disp" = precio en t - 24*ceil(h/24), que es el
    valor mas reciente de la misma hora del dia que si esta disponible en el corte, para cualquier h.
  - demanda/generacion se toman en el corte (t-h), no en t.

Ademas se evalua la PERSISTENCIA ESTACIONAL (precio de la misma hora hace 7 dias) como cuarto
votante candidato: es gratis, es de una familia completamente distinta, y en el ensamble de 24h
la persistencia demostro aportar peso real.
"""
import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_REF_72H = RAIZ / "data/processed/resultados/pronostico_con_bandas_72h_2026.csv"
RUTA_SALIDA = RAIZ / "data/processed/resultados/pronostico_xgb_directo_72h_2026.csv"
RUTA_SALIDA_SN = RAIZ / "data/processed/resultados/pronostico_naive_estacional_72h_2026.csv"

HORIZONTE = 72
PASO_MUESTREO_TRAIN = 4  # usar 1 de cada 4 horas como corte de entrenamiento (control de memoria)

# Estado conocido EN EL CORTE (se desplaza h horas hacia atras)
COLS_ESTADO = [
    "precio_bolsa", "precio_media_24h", "precio_media_7d", "precio_media_30d",
    "precio_std_24h", "precio_std_7d", "precio_rango_24h", "ratio_volatilidad",
    "volumen_embalses", "volumen_embalses_vs_media30d", "volumen_embalses_delta_1d",
    "volumen_embalses_delta_7d", "aportes_hidricos", "aportes_hidricos_vs_media30d",
    "aportes_hidricos_delta_1d", "aportes_hidricos_delta_7d",
    "demanda", "demanda_media_24h", "generacion",
]

# Conocido para el instante objetivo t (calendario + ONI mensual + festivos)
COLS_FUTURO = [
    "oni", "es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
    "dia_anio_sin", "dia_anio_cos", "es_festivo", "festivo_lag24h", "festivo_lag48h",
    "festivo_lag72h", "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h",
    "precio_lag168h",  # = precio en t-168, siempre conocido en el corte porque h<=72
]


def construir_largo(df, cortes_validos, pasos):
    """Arma el dataset largo (un registro por par corte-paso) sin fuga de informacion."""
    n = len(df)
    idx_corte_de_t = {}
    partes = []
    estado = df[COLS_ESTADO].to_numpy(dtype=np.float32)
    futuro = df[COLS_FUTURO].to_numpy(dtype=np.float32)
    precio = df["precio_bolsa"].to_numpy(dtype=np.float32)
    fechas = df["fecha_hora"].to_numpy()

    for h in pasos:
        # t va de h..n-1 ; el corte es t-h
        t_idx = np.arange(h, n)
        c_idx = t_idx - h
        # filtrar por cortes validos
        mascara = np.isin(c_idx, cortes_validos)
        if not mascara.any():
            continue
        t_idx, c_idx = t_idx[mascara], c_idx[mascara]

        # precio de la misma hora del dia mas reciente disponible en el corte
        retro = 24 * int(np.ceil(h / 24))
        idx_mismo = t_idx - retro
        ok = idx_mismo >= 0
        t_idx, c_idx, idx_mismo = t_idx[ok], c_idx[ok], idx_mismo[ok]

        bloque = np.hstack([
            estado[c_idx],
            futuro[t_idx],
            precio[idx_mismo].reshape(-1, 1),
            np.full((len(t_idx), 1), h, dtype=np.float32),
        ])
        sub = pd.DataFrame(bloque, columns=COLS_ESTADO + COLS_FUTURO + ["precio_mismo_hora_disp", "paso_horas"])
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

    # La construccion usa desplazamientos posicionales: exige rejilla horaria continua
    saltos = df["fecha_hora"].diff().dropna().value_counts()
    print("Continuidad horaria:", dict(list(saltos.items())[:3]))
    assert saltos.index[0] == pd.Timedelta(hours=1), "La serie no es horaria continua"

    df = df.reset_index(drop=True)
    pos = pd.Series(df.index.values, index=df["fecha_hora"])

    # Cortes de prueba: EXACTAMENTE los mismos 72 de la validacion cruzada neuronal
    ref = pd.read_csv(RUTA_REF_72H, parse_dates=["fecha_hora", "cutoff"])
    cortes_test = pd.Index(sorted(ref["cutoff"].unique()))
    idx_cortes_test = pos.reindex(cortes_test).dropna().astype(int).to_numpy()
    print(f"Cortes de prueba alineados: {len(idx_cortes_test)} de {len(cortes_test)}")

    # Cortes de entrenamiento: todo lo anterior a 2026, submuestreado
    idx_fin_train = int(pos[pd.Timestamp("2026-01-01")])
    idx_cortes_train = np.arange(200, idx_fin_train - HORIZONTE, PASO_MUESTREO_TRAIN)
    print(f"Cortes de entrenamiento: {len(idx_cortes_train)}")

    pasos = np.arange(1, HORIZONTE + 1)
    print("Construyendo dataset largo de entrenamiento...")
    train = construir_largo(df, idx_cortes_train, pasos).dropna()
    print(f"  filas de entrenamiento: {len(train):,}")
    print("Construyendo dataset largo de prueba...")
    test = construir_largo(df, idx_cortes_test, pasos).dropna()
    print(f"  filas de prueba: {len(test):,}")

    cols_x = COLS_ESTADO + COLS_FUTURO + ["precio_mismo_hora_disp", "paso_horas"]

    modelo = xgb.XGBRegressor(
        n_estimators=1200, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="reg:absoluteerror", random_state=42, n_jobs=-1, tree_method="hist",
    )
    print("Entrenando XGBoost directo multi-horizonte...")
    modelo.fit(train[cols_x], train["y"])
    test = test.copy()
    test["pred"] = modelo.predict(test[cols_x])

    salida = test[["fecha_hora", "cutoff", "paso_horas", "y", "pred"]].rename(columns={"y": "real"})
    salida["paso_horas"] = salida["paso_horas"].astype(int)
    salida.to_csv(RUTA_SALIDA, index=False)

    # Persistencia estacional: el precio de la misma hora hace 7 dias (conocido siempre en el corte)
    sn = test[["fecha_hora", "cutoff", "paso_horas", "y", "precio_lag168h"]].rename(
        columns={"y": "real", "precio_lag168h": "pred"})
    sn["paso_horas"] = sn["paso_horas"].astype(int)
    sn.to_csv(RUTA_SALIDA_SN, index=False)

    print(f"\n{'='*78}\nRESULTADOS POR TRAMO -- contra N-BEATSx 72h (referencia actual)\n{'='*78}")
    ref_idx = ref.set_index(["fecha_hora", "cutoff", "paso_horas"])
    print(f"{'tramo':9s} {'XGB-directo':>12s} {'NaiveEstac':>12s} {'N-BEATSx':>12s}")
    filas = []
    for nombre, ini, fin in [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]:
        s = salida[(salida.paso_horas >= ini) & (salida.paso_horas <= fin)]
        s_sn = sn[(sn.paso_horas >= ini) & (sn.paso_horas <= fin)]
        r = ref[(ref.paso_horas >= ini) & (ref.paso_horas <= fin)]
        mae_x = np.abs(s["real"] - s["pred"]).mean()
        mae_s = np.abs(s_sn["real"] - s_sn["pred"]).mean()
        mae_n = np.abs(r["real"] - r["q50"]).mean()
        print(f"{nombre:9s} {mae_x:12.2f} {mae_s:12.2f} {mae_n:12.2f}")
        filas.append({"tramo": nombre, "mae_xgb_directo": mae_x, "mae_naive_estacional": mae_s,
                      "mae_nbeatsx": mae_n})
    pd.DataFrame(filas).to_csv(RAIZ / "data/processed/resultados/xgb_directo_72h_metricas.csv", index=False)

    imp = pd.Series(modelo.feature_importances_, index=cols_x).sort_values(ascending=False)
    print(f"\nTop 12 variables del XGBoost directo:\n{imp.head(12).to_string()}")
    print(f"\nGuardado: {RUTA_SALIDA}")
    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
