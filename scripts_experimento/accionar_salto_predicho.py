# -*- coding: utf-8 -*-
"""
El salto dia-a-dia SI es predecible (AUC 0.77-0.81). ¿Sirve de algo saberlo?

Antecedente. La prueba exhaustiva (`anclaje_lag24_exhaustivo.py`) corrigio dos conclusiones
previas equivocadas:
  1) En la hora 0 el modelo NO le gana a la persistencia pura (-1.3%), siendo la unica hora del
     dia donde eso pasa (en las horas 1-5 le gana 52-84%).
  2) El salto dia-a-dia SI es anticipable con informacion del corte: AUC 0.77-0.81 sobre saltos
     extremos (percentil 80), usando la historia completa (2.742 dias) y CV temporal. La prueba
     anterior daba 0.54-0.59 por usar solo 217 dias y pliegues que no respetan el orden temporal.

Pero "predecible" no implica "util". Aqui se prueba si esa senal se puede convertir en una mejora
del pronostico puntual. Tres usos posibles, de menor a mayor ambicion:

  A) ENRUTAR: en los dias donde se predice salto grande, usar un modelo distinto (o mezcla) en vez
     del ensamble. Se compara contra el oraculo (enrutar sabiendo el salto real) para separar
     "la senal no sirve" de "el modelo alterno no sirve".
  B) MEZCLAR POR CONFIANZA: mezclar ensamble y persistencia con un peso continuo segun la
     probabilidad predicha de salto.
  C) COTA SUPERIOR: cuanto se ganaria con un enrutamiento perfecto, para saber si vale la pena
     seguir por aqui.

Se mide sobre 2026 (el periodo de evaluacion del proyecto), entrenando el clasificador de salto
solo con datos anteriores a cada punto (sin fuga).
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES

ESTADO = ["precio_bolsa", "precio_media_24h", "precio_media_7d", "precio_media_30d",
          "precio_std_24h", "precio_std_7d", "precio_rango_24h", "ratio_volatilidad",
          "volumen_embalses", "volumen_embalses_vs_media30d", "aportes_hidricos",
          "aportes_hidricos_vs_media30d", "demanda_lag24h", "demanda_media_24h",
          "oni", "es_festivo", "festivo_lag24h"]


def dm(real, p1, p2, maxlags=7):
    d = np.abs(real - p1) - np.abs(real - p2)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def main():
    t0 = time.time()
    from catboost import CatBoostClassifier

    df = cargar_completo().set_index("fecha_hora").sort_index()
    ens = pd.read_csv(RES / "combinador_optimo_mape.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")

    for HORA in [0, 9, 19]:
        print("=" * 94)
        print(f"HORA {HORA}")
        print("=" * 94)

        s = df[df.index.hour == HORA].copy()
        ayer = df["precio_bolsa"].reindex(s.index - pd.Timedelta(days=1))
        ayer.index = s.index
        s["persistencia"] = ayer
        s["salto"] = (s["precio_bolsa"] - ayer).abs()

        cols = [c for c in ESTADO if c in s.columns]
        X = s[cols].copy()
        X["dia_semana"] = s.index.dayofweek
        ok = X.notna().all(axis=1) & s["salto"].notna()
        s, X = s[ok], X[ok]

        # clasificador de salto grande, entrenado SOLO con datos anteriores a 2026
        umbral = s.loc[s.index < "2026-01-01", "salto"].quantile(0.80)
        ybin = (s["salto"] > umbral).astype(int)
        tr = s.index < "2026-01-01"
        clf = CatBoostClassifier(n_estimators=400, max_depth=5, random_seed=42, verbose=0)
        clf.fit(X[tr], ybin[tr])
        p_salto = pd.Series(clf.predict_proba(X)[:, 1], index=s.index)

        # evaluacion solo en 2026, y solo donde hay prediccion del ensamble
        e = ens[ens.index.hour == HORA]
        idx = s.index.intersection(e.index)
        y = e.loc[idx, "real"].to_numpy()
        pred_ens = e.loc[idx, "pred_smape"].to_numpy()
        pred_pers = s.loc[idx, "persistencia"].to_numpy()
        prob = p_salto.reindex(idx).to_numpy()
        salto_real = s.loc[idx, "salto"].to_numpy()
        print(f"  n={len(idx)} dias de 2026 | umbral de salto grande (train) = {umbral:.1f}")
        print(f"  MAE ensamble={np.abs(y-pred_ens).mean():.2f}   MAE persistencia={np.abs(y-pred_pers).mean():.2f}")

        # A) enrutar: si prob alta -> persistencia, si no -> ensamble
        print("\n  A) ENRUTAR a persistencia cuando se predice salto grande:")
        for q in [0.5, 0.6, 0.7, 0.8]:
            corte = np.quantile(prob, q)
            usar_pers = prob >= corte
            mix = np.where(usar_pers, pred_pers, pred_ens)
            t, pv = dm(y, pred_ens, mix)
            print(f"     corte en percentil {int(q*100)}: MAE={np.abs(y-mix).mean():6.2f} "
                  f"(ensamble {np.abs(y-pred_ens).mean():.2f})  DM t={t:+.2f} p={pv:.4f}  "
                  f"[{int(usar_pers.sum())} dias enrutados]")

        # B) mezcla continua segun la probabilidad
        print("\n  B) MEZCLA CONTINUA  pred = (1-w)*ensamble + w*persistencia, w = prob de salto:")
        for escala in [0.5, 1.0]:
            w = np.clip(prob * escala, 0, 1)
            mix = (1 - w) * pred_ens + w * pred_pers
            t, pv = dm(y, pred_ens, mix)
            print(f"     escala {escala}: MAE={np.abs(y-mix).mean():6.2f}  DM t={t:+.2f} p={pv:.4f}")

        # C) cota superior con oraculo
        print("\n  C) COTA SUPERIOR (oraculo, usa el salto REAL, no desplegable):")
        for q in [0.5, 0.8]:
            corte = np.quantile(salto_real, q)
            usar_pers = salto_real >= corte
            mix = np.where(usar_pers, pred_pers, pred_ens)
            print(f"     oraculo percentil {int(q*100)}: MAE={np.abs(y-mix).mean():6.2f}")
        mejor = np.where(np.abs(y - pred_pers) < np.abs(y - pred_ens), pred_pers, pred_ens)
        print(f"     oraculo perfecto (elige el mejor cada dia): MAE={np.abs(y-mejor).mean():6.2f}")
        print()

    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
