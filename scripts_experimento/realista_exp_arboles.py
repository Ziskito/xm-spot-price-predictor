# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: modelos de arboles con el predespacho POR DENTRO (no solo en la combinacion).

Entrena con 2019-2025 y prueba en 2026 (15-ene a 5-ago, mismas horas que el resto del escenario realista).
Conjuntos de variables (todas publicadas al corte, ver realista_comun.tabla_realista):
  R    variables realistas (precio y demanda con ~3 dias de rezago, embalses, aportes, ONI, calendario)
  P    predespacho de la hora objetivo y del dia (pre_*)
  D    delta ex post conocido (delta_*)
  H    precio de los ultimos dias lleno con el predespacho (h_*)
  G    despacho programado (prog_*), si esta disponible
Objetivos: log(precio) y log(precio / predespacho) (el modelo aprende la correccion del predespacho).
Modelos: XGBoost y CatBoost. Cada variante tambien entra como septimo votante al combinador realista.
Salida: resultados/escenario_realista/mejoras/arboles_metricas.csv y arboles_predicciones.csv
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
import escenario_realista_24h as er  # noqa: E402
from o6_comun import EXCLUIR  # noqa: E402

INI_TR, FIN_TR = "2019-02-15", "2025-12-31 23:00"
INI_TE, FIN_TE = "2026-01-15", "2026-08-05 00:00"


def conjuntos(d):
    base = [c for c in pd.read_csv(rc.RAIZ / "data/processed/dataset_features_2026.csv", nrows=1).columns if c not in EXCLUIR]
    P = ["pre_obj", "log_pre", "pre_media", "pre_min", "pre_max", "pre_rel", "pre_rango_rel", "pre_lag24", "pre_cambio24"]
    D = ["delta_d4", "delta_7d", "delta_30d"]
    H = [c for c in d.columns if c.startswith("h_")]
    G = [c for c in d.columns if c.startswith("prog_") or c == "n_termicas"]
    s = {"R+P": base + P, "R+P+D": base + P + D, "R+P+D+H": base + P + D + H}
    if G:
        s["R+P+D+H+G"] = base + P + D + H + G
    return s


def entrenar(modelo, Xtr, ytr, Xte):
    if modelo == "XGBoost":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=800, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=5, random_state=42, n_jobs=4)
    else:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.05, loss_function="MAE", random_seed=42, verbose=0, thread_count=4)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista()
    d = d.set_index("fecha_hora")
    tr = d.loc[INI_TR:FIN_TR].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    te = d.loc[INI_TE:FIN_TE].dropna(subset=["precio_bolsa", "pre_obj"])
    print(f"entrenamiento {len(tr)} h, prueba {len(te)} h", flush=True)
    preds = pd.DataFrame({"real": te["precio_bolsa"]})
    for nom, cols in conjuntos(d).items():
        for modelo in ("XGBoost", "CatBoost"):
            for objetivo in ("log precio", "log precio/predespacho"):
                ytr = np.log(tr["precio_bolsa"]) - (np.log(tr["pre_obj"].clip(lower=1)) if "/" in objetivo else 0)
                z = entrenar(modelo, tr[cols], ytr, te[cols])
                p = np.exp(z + (np.log(te["pre_obj"].clip(lower=1)) if "/" in objetivo else 0))
                clave = f"{modelo} · {nom} · {objetivo}"
                preds[clave] = p
                print(f"  {clave:55s} MAE {np.abs(p - te['precio_bolsa']).mean():6.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    preds.to_csv(rc.MEJ / "arboles_predicciones.csv")
    # como votante adicional en el combinador realista
    ref = rc.referencia()
    from ensamble_24h_enero import origen5
    from ensamble_con_predespacho_24h import imar
    filas = []
    y = preds["real"]
    ev_idx = preds.index.intersection(ref.dropna(subset=["ens7_realista", "ens6_actual"]).index)
    for c in [c for c in preds.columns if c != "real"]:
        f = {"variante": c, **rc.metricas(y[ev_idx], preds.loc[ev_idx, c])}
        filas.append(f)
    for c, n in (("ens7_realista", "REFERENCIA ensamble 7 realista"), ("ens6_actual", "REFERENCIA ensamble 6 optimista (anterior)"), ("Predespacho", "REFERENCIA predespacho solo")):
        filas.append({"variante": n, **rc.metricas(y[ev_idx], ref.loc[ev_idx, c])})
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / "arboles_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
