# -*- coding: utf-8 -*-
"""
Ataca la segunda idea del paper de Lago et al. (2021), Seccion 2.1 (citando De Marcos et al. 2020
y Nitka et al. 2021): en vez de calibrar con TODO el historico cronologico (o con un cutoff fijo),
calibrar solo con los periodos del pasado que MAS SE PARECEN al regimen actual, usando clustering/
vecinos-mas-cercanos sobre variables fundamentales.

Aqui se implementa con k-NN sobre el descriptor fuzzy del ONI (7 dimensiones: fuzzy_nino_debil/
moderado/fuerte/muy_fuerte, fuzzy_nina_debil/moderado, fuzzy_transicion_rapida -- las mismas que
ya se integraron al pipeline compartido hoy), en vez de duplicar/sobremuestrear un episodio fijo
(las 4 hipotesis de "atacar El Nino" que ya fallaron anoche).

Metodologia:
1. Se parte el historico (train, antes del corte de cada origen) en bloques NO solapados de 30
   dias, y se calcula el vector fuzzy PROMEDIO de cada bloque (descriptor de "que tan fuerte y que
   tan rapido estaba cambiando el regimen" en ese bloque).
2. El "snapshot" del regimen actual es el vector fuzzy promedio de los ultimos 30 dias antes del
   corte de cada origen (lo mas parecido a "lo que se sabe justo antes de pronosticar").
3. Se buscan los K bloques historicos mas cercanos (distancia euclidiana) al snapshot actual, y se
   entrena XGBoost SOLO con esas horas (no con todo el historico cronologico).
4. Se compara contra el XGBoost estandar (entrenado con TODO el historico antes del corte, ya
   guardado en walkforward_predicciones_crudas.csv) via MAE y prueba de Diebold-Mariano.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

FUZZY_COLS = ["fuzzy_nino_debil", "fuzzy_nino_moderado", "fuzzy_nino_fuerte", "fuzzy_nino_muy_fuerte",
              "fuzzy_nina_debil", "fuzzy_nina_moderado", "fuzzy_transicion_rapida"]

K_BLOQUES = 12          # ~360 dias de datos analogos (comparable en tamano a una ventana corta tipo LEAR)
DIAS_BLOQUE = 30

a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
df_completo = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

columnas_excluir = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora", "dia_semana", "dia_anio"]
columnas_features = [c for c in df_completo.columns if c not in columnas_excluir]

origenes = [
    {"nombre": "Origen 1", "regimen": "La Nina (inicio)",       "corte_train": "2020-07-01", "test_inicio": "2020-07-01", "test_fin": "2020-09-30"},
    {"nombre": "Origen 4", "regimen": "El Nino (fuerte)",       "corte_train": "2023-10-01", "test_inicio": "2023-10-01", "test_fin": "2023-12-31"},
    {"nombre": "Origen 6", "regimen": "El Nino 2026 (neutral->fuerte)", "corte_train": "2026-01-01", "test_inicio": "2026-01-01", "test_fin": "2026-08-05"},
]

def calcular_metricas(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    error = y_real - y_pred
    mae = np.abs(error).mean()
    mape = (np.abs(error) / y_real).mean() * 100
    return mae, mape

def prueba_dm(e1, e2, horizonte=24):
    d = np.asarray(e1) - np.asarray(e2)
    m = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": horizonte - 1})
    return m.tvalues[0], m.pvalues[0]

resultados = []
predicciones_analogas = []

for o in origenes:
    train = df_completo[df_completo["fecha_hora"] < o["corte_train"]].dropna(subset=columnas_features).copy()
    test = df_completo[(df_completo["fecha_hora"] >= o["test_inicio"]) & (df_completo["fecha_hora"] <= o["test_fin"])].copy()

    # --- Snapshot del regimen actual: promedio fuzzy de los ultimos 30 dias de train ---
    snapshot = train[train["fecha_hora"] >= train["fecha_hora"].max() - pd.Timedelta(days=DIAS_BLOQUE)][FUZZY_COLS].mean().values

    # --- Bloques no solapados de 30 dias en el resto del historico (excluyendo el propio snapshot) ---
    train_hist = train[train["fecha_hora"] < train["fecha_hora"].max() - pd.Timedelta(days=DIAS_BLOQUE)].copy()
    train_hist["bloque"] = ((train_hist["fecha_hora"] - train_hist["fecha_hora"].min()).dt.days // DIAS_BLOQUE)

    descriptores_bloque = train_hist.groupby("bloque")[FUZZY_COLS].mean()
    distancias = np.linalg.norm(descriptores_bloque.values - snapshot, axis=1)
    descriptores_bloque = descriptores_bloque.assign(distancia=distancias).sort_values("distancia")
    bloques_analogos = descriptores_bloque.head(K_BLOQUES).index.tolist()

    train_analogo = train_hist[train_hist["bloque"].isin(bloques_analogos)]
    print(f"\n--- {o['nombre']} [{o['regimen']}] ---")
    print(f"Train cronologico: {len(train)} filas | Train analogo (K={K_BLOQUES} bloques de {DIAS_BLOQUE}d): {len(train_analogo)} filas")
    fechas_bloques = train_hist.groupby("bloque")["fecha_hora"].agg(["min", "max"])
    print("Bloques analogos seleccionados (fecha inicio - fin, distancia):")
    for bl in bloques_analogos[:5]:
        print(f"  {fechas_bloques.loc[bl, 'min'].date()} a {fechas_bloques.loc[bl, 'max'].date()}  dist={descriptores_bloque.loc[bl, 'distancia']:.3f}")
    if len(bloques_analogos) > 5:
        print(f"  ... y {len(bloques_analogos) - 5} bloques mas")

    # --- XGBoost con train analogo ---
    X_train_a, y_train_a = train_analogo[columnas_features], np.log(train_analogo["precio_bolsa"])
    X_test = test[columnas_features]
    m_xgb_a = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)
    m_xgb_a.fit(X_train_a, y_train_a)
    pred_analogo = np.exp(m_xgb_a.predict(X_test))

    # --- XGBoost con train cronologico estandar (mismo criterio del proyecto) ---
    X_train_c, y_train_c = train[columnas_features], np.log(train["precio_bolsa"])
    m_xgb_c = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)
    m_xgb_c.fit(X_train_c, y_train_c)
    pred_cronologico = np.exp(m_xgb_c.predict(X_test))

    y_real = test["precio_bolsa"].values
    mae_a, mape_a = calcular_metricas(y_real, pred_analogo)
    mae_c, mape_c = calcular_metricas(y_real, pred_cronologico)

    dm_stat, p_valor = prueba_dm(np.abs(y_real - pred_analogo), np.abs(y_real - pred_cronologico))
    if p_valor < 0.05:
        conclusion = "ANALOGO GANA (sig.)" if dm_stat < 0 else "ANALOGO PIERDE (sig.)"
    else:
        conclusion = "sin diferencia significativa"

    print(f"XGBoost analogo (K-NN fuzzy):     MAE {mae_a:6.2f}  MAPE {mape_a:5.2f}%")
    print(f"XGBoost cronologico (estandar):   MAE {mae_c:6.2f}  MAPE {mape_c:5.2f}%")
    print(f"DM: t={dm_stat:.3f}  p={p_valor:.4f}  -> {conclusion}")

    resultados.append({"origen": o["nombre"], "regimen": o["regimen"],
                        "filas_train_analogo": len(train_analogo), "filas_train_cronologico": len(train),
                        "mae_analogo": mae_a, "mae_cronologico": mae_c,
                        "dm_stat": dm_stat, "p_valor": p_valor, "conclusion": conclusion})

    for fh, real, pa, pc in zip(test["fecha_hora"], y_real, pred_analogo, pred_cronologico):
        predicciones_analogas.append({"origen": o["nombre"], "fecha_hora": fh, "real": real,
                                       "pred_analogo": pa, "pred_cronologico": pc})

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(RAIZ / "data" / "processed" / "resultados" / "analogos_regimen_knn.csv", index=False)
pd.DataFrame(predicciones_analogas).to_csv(RAIZ / "data" / "processed" / "resultados" / "analogos_regimen_knn_predicciones.csv", index=False)

print("\n" + "=" * 90)
print("RESUMEN")
print("=" * 90)
print(df_resultados[["origen", "regimen", "mae_analogo", "mae_cronologico", "p_valor", "conclusion"]].to_string(index=False))
print("\nGuardado: analogos_regimen_knn.csv, analogos_regimen_knn_predicciones.csv")
