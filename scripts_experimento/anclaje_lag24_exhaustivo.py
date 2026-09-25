# -*- coding: utf-8 -*-
"""
Prueba exhaustiva del anclaje al lag-24 y de si el salto dia-a-dia es anticipable.

Contexto. Al analizar la Figura 4 se encontro que el error del ensamble en la hora 0 correlaciona
0.974 con |precio(h0, hoy) - precio(h0, ayer)|. Las pruebas iniciales fueron debiles: solo la hora
0, solo 217 dias (2026), un unico target binario en la mediana, y validacion cruzada por pliegues
estratificados que NO respeta el orden temporal.

Este script corrige todo eso y ademas somete el propio hallazgo a una prueba critica:

  PARTE 1 - ¿La correlacion 0.974 es especial de la hora 0 o es generica?
      Si el modelo fuera una persistencia pura, error = |P_hoy - P_ayer| EXACTAMENTE y la
      correlacion seria 1.0 por construccion, sin que eso dijera nada interesante. Se calcula la
      correlacion para las 24 horas y se compara cada hora contra la persistencia pura, para
      separar "hallazgo" de "tautologia".

  PARTE 2 - Predictibilidad del salto, en serio:
      - historia completa 2019-2026 (~2.900 dias) en vez de solo 217
      - las 24 horas, no solo la 0
      - tres targets: binario en mediana, binario en percentil 80 (saltos extremos), y regresion
        sobre la magnitud
      - validacion cruzada TEMPORAL (TimeSeriesSplit), que es lo correcto para series
      - tres familias de modelo (logistica, Random Forest, CatBoost)
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES

ESTADO = ["precio_bolsa", "precio_media_24h", "precio_media_7d", "precio_media_30d",
          "precio_std_24h", "precio_std_7d", "precio_rango_24h", "ratio_volatilidad",
          "volumen_embalses", "volumen_embalses_vs_media30d", "aportes_hidricos",
          "aportes_hidricos_vs_media30d", "demanda_lag24h", "demanda_media_24h",
          "oni", "es_festivo", "festivo_lag24h"]


def parte1_es_especial_la_hora_0():
    """La correlacion alta, ¿es un hallazgo o una tautologia de la persistencia?"""
    print("=" * 96)
    print("PARTE 1 - ¿La correlacion error~salto es especial de la hora 0, o generica?")
    print("=" * 96)
    d = pd.read_csv(RES / "combinador_optimo_mape.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora").sort_index()
    df = cargar_completo().set_index("fecha_hora").sort_index()

    filas = []
    for h in range(24):
        s = d[d.index.hour == h]
        if len(s) < 30:
            continue
        real = s["real"]
        pred = s["pred_smape"]
        ayer = df["precio_bolsa"].reindex(s.index - pd.Timedelta(days=1))
        ayer.index = s.index
        ok = ayer.notna()
        salto = (real - ayer).abs()[ok]
        err = (real - pred).abs()[ok]
        err_pers = (real - ayer).abs()[ok]          # error de la persistencia pura
        filas.append({
            "hora": h,
            "corr_err_salto": np.corrcoef(salto, err)[0, 1],
            "MAE_modelo": err.mean(),
            "MAE_persistencia": err_pers.mean(),
            "ganancia_vs_pers_%": (1 - err.mean() / err_pers.mean()) * 100,
        })
    t = pd.DataFrame(filas)
    print(t.round(3).to_string(index=False))
    print()
    print(f"  Correlacion media entre horas: {t['corr_err_salto'].mean():.3f}  "
          f"(min {t['corr_err_salto'].min():.3f} en h{int(t.loc[t['corr_err_salto'].idxmin(),'hora'])}, "
          f"max {t['corr_err_salto'].max():.3f} en h{int(t.loc[t['corr_err_salto'].idxmax(),'hora'])})")
    print(f"  Hora 0: corr={t.loc[t.hora==0,'corr_err_salto'].iloc[0]:.3f}  "
          f"-> percentil {int((t['corr_err_salto'] < t.loc[t.hora==0,'corr_err_salto'].iloc[0]).mean()*100)} entre las 24 horas")
    print()
    print("  LECTURA: si la correlacion es alta en TODAS las horas, no es un hallazgo sobre la")
    print("  hora 0 sino una propiedad generica (el error se parece al cambio dia-a-dia en todas).")
    print("  Lo que si distingue a cada hora es cuanto le gana el modelo a la persistencia pura.")
    return t


def parte2_predictibilidad():
    """¿Se puede anticipar el salto dia-a-dia con informacion del corte? Prueba seria."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from catboost import CatBoostClassifier
    from scipy.stats import spearmanr

    print("\n" + "=" * 96)
    print("PARTE 2 - ¿Es anticipable el salto dia-a-dia? (historia completa, CV temporal)")
    print("=" * 96)
    df = cargar_completo().set_index("fecha_hora").sort_index()
    disponibles = [c for c in ESTADO if c in df.columns]

    resultados = []
    for h in [0, 9, 19]:                       # una hora de cada regimen: medianoche, manana, pico
        s = df[df.index.hour == h].copy()
        ayer = df["precio_bolsa"].reindex(s.index - pd.Timedelta(days=1))
        ayer.index = s.index
        salto = (s["precio_bolsa"] - ayer).abs()
        X = s[disponibles].copy()
        X["dia_semana"] = s.index.dayofweek
        ok = X.notna().all(axis=1) & salto.notna()
        X, ysalto = X[ok], salto[ok]
        n = len(X)

        for etiqueta, ybin in [("mediana", (ysalto > ysalto.median()).astype(int)),
                               ("percentil 80", (ysalto > ysalto.quantile(0.80)).astype(int))]:
            tscv = TimeSeriesSplit(n_splits=5)
            for nombre, modelo in [
                ("logistica", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                ("RandomForest", RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)),
                ("CatBoost", CatBoostClassifier(n_estimators=400, max_depth=5, random_seed=42, verbose=0)),
            ]:
                aucs = []
                for tr, te in tscv.split(X):
                    if ybin.iloc[tr].nunique() < 2 or ybin.iloc[te].nunique() < 2:
                        continue
                    modelo.fit(X.iloc[tr], ybin.iloc[tr])
                    pr = modelo.predict_proba(X.iloc[te])[:, 1]
                    aucs.append(roc_auc_score(ybin.iloc[te], pr))
                resultados.append({"hora": h, "n": n, "target": etiqueta, "modelo": nombre,
                                   "AUC_medio": np.mean(aucs), "AUC_std": np.std(aucs)})

        # regresion sobre la magnitud del salto
        tscv = TimeSeriesSplit(n_splits=5)
        rhos = []
        for tr, te in tscv.split(X):
            m = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
            m.fit(X.iloc[tr], ysalto.iloc[tr])
            pr = m.predict(X.iloc[te])
            rhos.append(spearmanr(ysalto.iloc[te], pr).statistic)
        resultados.append({"hora": h, "n": n, "target": "magnitud (Spearman)", "modelo": "RF regresor",
                           "AUC_medio": np.mean(rhos), "AUC_std": np.std(rhos)})

    r = pd.DataFrame(resultados)
    print(r.round(3).to_string(index=False))
    print()
    print("  Referencia: AUC 0.5 = azar. >0.70 seria util para enrutar o avisar al operador.")
    print("  Para la fila de Spearman, 0 = sin relacion monotona con la magnitud real del salto.")
    return r


def main():
    t0 = time.time()
    t1 = parte1_es_especial_la_hora_0()
    t2 = parte2_predictibilidad()
    t1.to_csv(RES / "anclaje_por_hora.csv", index=False)
    t2.to_csv(RES / "predictibilidad_salto.csv", index=False)
    print(f"\nGuardado: anclaje_por_hora.csv, predictibilidad_salto.csv | "
          f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
