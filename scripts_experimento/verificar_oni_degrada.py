# -*- coding: utf-8 -*-
"""
Verificacion de un hallazgo inesperado de la ablacion LOGO (ideas_papers_hidrologia.py):
QUITAR el ONI MEJORA el modelo directo de 24h en 2.82 COP/kWh de MAE, con p<0.0001.

Es una afirmacion fuerte -- el ONI es la variable estrella del marco conceptual del proyecto --
asi que antes de reportarla se verifica que no sea un artefacto:
  1) Rango del ONI en entrenamiento vs. 2026: si 2026 sale del rango visto, los arboles no pueden
     extrapolar y el corte aprendido se vuelve una regla equivocada (explicacion mecanica).
  2) Repetir base vs. sin-ONI con 3 semillas distintas: si el efecto se sostiene en las tres, no es
     ruido de una corrida.
  3) Repetirlo con Ridge (lineal): si tambien pasa ahi, no es una peculiaridad de los arboles;
     si NO pasa, confirma que es un problema de extrapolacion de arboles (que no extrapolan).
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
from ideas_papers_hidrologia import construir_datos, dm
from o6_comun import cargar_completo, RES

CORTE = pd.Timestamp("2026-01-01")


def main():
    t0 = time.time()
    df = cargar_completo()

    # ---------- 1) rango del ONI ----------
    tr_oni = df[df["fecha_hora"] < CORTE]["oni"]
    te_oni = df[df["fecha_hora"] >= CORTE]["oni"]
    print("1) RANGO DEL ONI")
    print(f"   entrenamiento (2019-2025): min={tr_oni.min():+.2f}  max={tr_oni.max():+.2f}  "
          f"media={tr_oni.mean():+.2f}")
    print(f"   prueba (2026):             min={te_oni.min():+.2f}  max={te_oni.max():+.2f}  "
          f"media={te_oni.mean():+.2f}")
    fuera = ((te_oni > tr_oni.max()) | (te_oni < tr_oni.min())).mean() * 100
    print(f"   horas de 2026 con ONI fuera del rango de entrenamiento: {fuera:.1f}%")
    p90_tr = tr_oni.quantile(0.90)
    print(f"   horas de 2026 por encima del percentil 90 de entrenamiento ({p90_tr:+.2f}): "
          f"{(te_oni > p90_tr).mean()*100:.1f}%")

    # ---------- 2) y 3) ----------
    tr, te, cols = construir_datos(df, [])
    cols_sin_oni = [c for c in cols if c != "oni"]
    real = te["y"].to_numpy()
    print(f"\n2) CATBOOST -- base vs sin ONI, 3 semillas ({len(tr):,} filas de entrenamiento)")
    from catboost import CatBoostRegressor
    for semilla in [42, 7, 123]:
        preds = {}
        for etiqueta, cc in [("base", cols), ("sin ONI", cols_sin_oni)]:
            m = CatBoostRegressor(n_estimators=800, max_depth=8, learning_rate=0.05,
                                  loss_function="MAE", random_seed=semilla, verbose=0, thread_count=-1)
            m.fit(tr[cc], tr["y"])
            preds[etiqueta] = m.predict(te[cc])
        mae_b = np.abs(real - preds["base"]).mean()
        mae_s = np.abs(real - preds["sin ONI"]).mean()
        t, pv = dm(real, preds["base"], preds["sin ONI"])
        print(f"   semilla {semilla:3d}: base={mae_b:6.2f}  sin ONI={mae_s:6.2f}  "
              f"({mae_b-mae_s:+.2f} a favor de quitarlo)  DM p={pv:.4f}")

    print("\n3) RIDGE (lineal, SI extrapola) -- base vs sin ONI")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    for etiqueta, cc in [("base", cols), ("sin ONI", cols_sin_oni)]:
        esc = StandardScaler().fit(tr[cc])
        r = Ridge(alpha=10.0).fit(esc.transform(tr[cc]), tr["y"])
        p = r.predict(esc.transform(te[cc]))
        globals()[f"p_{etiqueta.replace(' ', '_')}"] = p
        print(f"   {etiqueta:10s} MAE={np.abs(real - p).mean():6.2f}")
    t, pv = dm(real, globals()["p_base"], globals()["p_sin_ONI"])
    print(f"   DM (base vs sin ONI): p={pv:.4f}  "
          f"{'quitar ONI ayuda' if pv < 0.05 and t > 0 else 'sin diferencia significativa'}")
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
