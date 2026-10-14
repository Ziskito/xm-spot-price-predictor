# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: variantes del modelo de arboles con el predespacho por dentro (R+P+G,
objetivo log(precio/predespacho), entrenado hasta 2025-12-31, prueba 15-ene a 5-ago-2026).
  * CatBoost base (MAE, profundidad 6, 1500 iteraciones)
  * ponderacion por antiguedad (vida media de 1 y 2 años)
  * historia corta (desde 2021 / 2023)
  * mas profundo / mas lento
  * perdida RMSE y cuantil 0.5 con mas iteraciones
  * XGBoost y LightGBM con perdida absoluta
Salida: resultados/escenario_realista/mejoras/hiper_metricas.csv y hiper_predicciones.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402


def ajustar(tipo, Xtr, ytr, Xte, w=None, **kw):
    if tipo == "cat":
        from catboost import CatBoostRegressor
        p = dict(iterations=1500, depth=6, learning_rate=0.05, loss_function="MAE", random_seed=42, verbose=0, thread_count=6)
        p.update(kw)
        m = CatBoostRegressor(**p)
    elif tipo == "xgb":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=1500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                             objective="reg:absoluteerror", random_state=42, n_jobs=6)
    else:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, objective="l1", random_state=42, n_jobs=6, verbose=-1)
    m.fit(Xtr, ytr, sample_weight=w)
    return m.predict(Xte)


if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    s = conjuntos(d)
    cols = s["R+P"] + [c for c in d.columns if c.startswith("prog_") or c == "n_termicas"]
    tr0 = d.loc["2019-02-15":"2025-12-31 23:00"].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    te = d.loc["2026-01-15":"2026-08-05 00:00"].dropna(subset=["precio_bolsa", "pre_obj"])
    lp = lambda x: np.log(x["pre_obj"].clip(lower=1))
    edad = ((pd.Timestamp("2026-01-01") - tr0.index) / pd.Timedelta(days=365)).to_numpy()
    variantes = {
        "CatBoost base": ("cat", tr0, None, {}),
        "vida media 1 año": ("cat", tr0, 0.5 ** edad, {}),
        "vida media 2 años": ("cat", tr0, 0.5 ** (edad / 2), {}),
        "desde 2021": ("cat", tr0.loc["2021-01-01":], None, {}),
        "desde 2023": ("cat", tr0.loc["2023-01-01":], None, {}),
        "profundidad 8": ("cat", tr0, None, {"depth": 8}),
        "3000 iter lr 0.03": ("cat", tr0, None, {"iterations": 3000, "learning_rate": 0.03}),
        "perdida RMSE": ("cat", tr0, None, {"loss_function": "RMSE"}),
        "XGBoost L1": ("xgb", tr0, None, {}),
        "LightGBM L1": ("lgb", tr0, None, {}),
    }
    preds = pd.DataFrame({"real": te["precio_bolsa"]})
    for nom, (tipo, tr, w, kw) in variantes.items():
        try:
            z = ajustar(tipo, tr[cols], np.log(tr["precio_bolsa"]) - lp(tr), te[cols], w, **kw)
        except Exception as e:  # paquete no instalado, etc.
            print(f"  {nom}: fallo {e}", flush=True)
            continue
        preds[nom] = np.exp(z + lp(te))
        print(f"  {nom:22s} MAE {np.abs(preds[nom] - preds['real']).mean():6.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    preds.to_csv(rc.MEJ / "hiper_predicciones.csv")
    t = pd.DataFrame([{"variante": c, **rc.metricas(preds["real"], preds[c])} for c in preds.columns if c != "real"]).sort_values("MAE")
    t.to_csv(rc.MEJ / "hiper_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
