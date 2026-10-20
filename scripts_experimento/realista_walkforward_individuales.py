# -*- coding: utf-8 -*-
"""
Walk-forward del escenario realista para los modelos que pronostican el precio desde cero (20-oct-2026). Hasta hoy
esta validacion solo existia con informacion optimista (precio de ayer disponible); con lo publicado al corte solo se
habia medido 2026 (escenario_realista_24h.py).

Mismas funciones y configuracion de escenario_realista_24h.py (variables realistas: precio y demanda con 73 h sin
publicar, embalses y aportes de D-2, ONI de M-2; modelo fijo entrenado con todo lo anterior al inicio del periodo,
como el walk-forward original), aplicadas a los cinco periodos historicos (15-jul a 30-jun, 2020-2025) y a 2026:
  Persistencia (D-4), XGBoost, ARX+GARCH, GARCH-ged (LEAR), N-BEATSx y N-HiTS
Se juntan con las predicciones del benchmark de correctores (realista_benchmark_modelos.py: predespacho solo y
predespacho + CatBoost / LightGBM / XGBoost / RandomForest / Lasso) en las mismas horas y se comparan con MAE, rMAE y
Diebold-Mariano por pares.
Uso: python realista_walkforward_individuales.py [periodo ...]   (sin argumentos, todos; --evaluar solo junta)
Salida: resultados/escenario_realista/walkforward/pred_<periodo>.csv, metricas.csv, dm.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import escenario_realista_24h as er  # noqa: E402
import realista_comun as rc  # noqa: E402

SAL = rc.MEJ.parent / "walkforward"
SAL.mkdir(exist_ok=True)
BENCH = rc.MEJ.parent / "benchmark" / "predicciones_combinadas.csv"
PERIODOS = {"O1": ("Origen 1 (2020-21)", "2020-07-15", "2021-06-30"), "O2": ("Origen 2 (2021-22)", "2021-07-15", "2022-06-30"),
            "O3": ("Origen 3 (2022-23)", "2022-07-15", "2023-06-30"), "O4": ("Origen 4 (2023-24)", "2023-07-15", "2024-06-30"),
            "O5": ("Origen 5 (2024-25)", "2024-07-15", "2025-06-30"), "2026": ("2026", "2026-01-15", "2026-08-04")}
DESDE_CERO = ["Persistencia", "XGBoost", "ARX+GARCH", "GARCH-ged", "N-BEATSx", "N-HiTS"]


def correr(clave, df):
    t0 = time.time()
    nom, ini, fin = PERIODOS[clave]
    er.CORTE, er.FIN = pd.Timestamp(ini), pd.Timestamp(fin) + pd.Timedelta(hours=24)
    serie = df.set_index("fecha_hora")
    idx = pd.DatetimeIndex(df.loc[(df["fecha_hora"] > er.CORTE) & (df["fecha_hora"] <= er.FIN), "fecha_hora"])
    pred = pd.DataFrame(index=idx)
    pred["real"], pred["Persistencia"] = serie["precio_bolsa"], serie["precio_lag24h"]
    pred["XGBoost"] = er.xgboost(df)
    pred["ARX+GARCH"] = er.arx_garch(df)
    pred["GARCH-ged"] = er.garch_ged(df)
    print(f"  {nom}: XGBoost, ARX+GARCH, GARCH-ged ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pred["N-BEATSx"], pred["N-HiTS"] = er.neuronales(df)
    print(f"  {nom}: redes ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pred.to_csv(SAL / f"pred_{clave}.csv")


def evaluar():
    b = pd.read_csv(BENCH, index_col=0, parse_dates=True)
    corr = ["Predespacho", "Pre + CatBoost", "Pre + LightGBM", "Pre + XGBoost", "Pre + RandomForest", "Pre + Lasso",
            "Pre + Prom. 3 boosting"]
    filas, dms = [], []
    for clave, (nom, ini, fin) in PERIODOS.items():
        ruta = SAL / f"pred_{clave}.csv"
        if not ruta.exists():
            continue
        p = pd.read_csv(ruta, index_col=0, parse_dates=True).join(b[corr]).dropna()
        p = p.loc[pd.Timestamp(ini) + pd.Timedelta(hours=1):pd.Timestamp(fin) + pd.Timedelta(hours=24)]
        y = p["real"].to_numpy()
        mods = DESDE_CERO + corr
        mp = np.abs(y - p["Persistencia"]).mean()
        for m in mods:
            e = y - p[m].to_numpy()
            fila = {"periodo": nom, "horas": len(p), "modelo": m, "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
                    "MAPE": 100 * np.mean(np.abs(e) / np.maximum(y, 1)), "rMAE": np.abs(e).mean() / mp}
            if m != "Persistencia":
                fila["DM_t_vs_persistencia"], fila["DM_p"] = er.dm(y, p[m].to_numpy(), p["Persistencia"].to_numpy())
            filas.append(fila)
        for i, a in enumerate(mods):
            for c in mods[i + 1:]:
                t, pv = er.dm(y, p[a].to_numpy(), p[c].to_numpy())
                dms.append({"periodo": nom, "A": a, "B": c, "DM_t": t, "p": pv})
    t = pd.DataFrame(filas)
    t.to_csv(SAL / "metricas.csv", index=False)
    pd.DataFrame(dms).to_csv(SAL / "dm.csv", index=False)
    pd.set_option("display.width", 250)
    orden = [v[0] for v in PERIODOS.values() if v[0] in set(t["periodo"])]
    tab = t.pivot(index="modelo", columns="periodo", values="MAE")[orden]
    tab["media"] = tab.mean(axis=1)
    print(tab.sort_values("media").round(2).to_string())
    print("\nDM frente a la persistencia (t; negativo = el modelo tiene MENOS error):")
    print(t.pivot(index="modelo", columns="periodo", values="DM_t_vs_persistencia")[orden].round(2).to_string())


if __name__ == "__main__":
    if "--evaluar" in sys.argv:
        evaluar()
        sys.exit(0)
    claves = [a for a in sys.argv[1:] if a in PERIODOS] or list(PERIODOS)
    df = er.features_realistas()
    print(f"variables realistas listas ({len(df)} filas)", flush=True)
    for c in claves:
        correr(c, df)
    if BENCH.exists():
        evaluar()
