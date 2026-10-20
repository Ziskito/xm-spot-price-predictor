# -*- coding: utf-8 -*-
"""
Benchmark de los modelos que corrigen el predespacho en el escenario realista (19-oct-2026), con walk-forward en los
cinco periodos historicos (15-jul a 30-jun, 2020-2025) y en 2026 (15-ene hasta el ultimo precio publicado).

Todos los candidatos en las mismas condiciones del diseño en produccion: objetivo log(precio / predespacho), variables
R+P (realistas + predespacho), reentrenamiento mensual con lo publicado hasta 73 h antes del corte (ventana creciente
desde 2019-02-15), mismo presupuesto (1.500 arboles, profundidad 6, tasa 0,05, perdida absoluta cuando el modelo la
admite) y sin afinar hiperparametros para ninguno:
  CatBoost      perdida MAE (el de produccion)
  LightGBM      objetivo l1
  XGBoost       reg:absoluteerror, histograma
  RandomForest  300 arboles, hoja minima 10 (sklearn no ofrece perdida absoluta a esta escala; error cuadratico)
  Lasso         lineal regularizado tipo LEAR sobre variables estandarizadas
Cada modelo se evalua solo y combinado con el predespacho en el combinador causal de produccion (QRA objetivo MAE,
franjas de 6 h, margen de 5 dias), y se compara con la persistencia realista (D-4) y el predespacho solo, con
Diebold-Mariano (HAC) por pares.

Uso:
  python realista_benchmark_modelos.py <modelo>    entrena un modelo (CatBoost | LightGBM | XGBoost | RandomForest | Lasso)
  python realista_benchmark_modelos.py --evaluar   junta las predicciones y escribe las tablas
Salidas: resultados/escenario_realista/benchmark/pred_<modelo>.csv, metricas.csv, dm.csv
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
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN  # noqa: E402

SAL = rc.MEJ.parent / "benchmark"
SAL.mkdir(exist_ok=True)
MODELOS = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "Lasso"]
INICIO = pd.Timestamp("2020-05-01")              # 60+ dias antes del primer periodo: calienta el combinador
PERIODOS = [("Origen 1 (2020-21)", "2020-07-15", "2021-06-30"), ("Origen 2 (2021-22)", "2021-07-15", "2022-06-30"),
            ("Origen 3 (2022-23)", "2022-07-15", "2023-06-30"), ("Origen 4 (2023-24)", "2023-07-15", "2024-06-30"),
            ("Origen 5 (2024-25)", "2024-07-15", "2025-06-30"), ("2026", "2026-01-15", None)]


def ajustar(nombre, X, y):
    if nombre == "CatBoost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.05, loss_function="MAE", random_seed=42,
                              verbose=0, thread_count=4)
    elif nombre == "LightGBM":
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(objective="l1", n_estimators=1500, learning_rate=0.05, max_depth=6, num_leaves=63,
                          random_state=42, n_jobs=4, verbose=-1)
    elif nombre == "XGBoost":
        from xgboost import XGBRegressor
        m = XGBRegressor(objective="reg:absoluteerror", n_estimators=1500, learning_rate=0.05, max_depth=6,
                         tree_method="hist", random_state=42, n_jobs=4)
    elif nombre == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.33,
                                                random_state=42, n_jobs=4))
    elif nombre == "Lasso":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Lasso
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Lasso(alpha=1e-3, max_iter=20000))
    else:
        raise ValueError(nombre)
    m.fit(X, y)
    return m


def entrenar(nombre):
    t0 = time.time()
    d = rc.tabla_realista(produccion=True).set_index("fecha_hora")
    cols = conjuntos(d)["R+P"]
    fin = d["precio_bolsa"].last_valid_index()
    partes = []
    for mes in pd.date_range(INICIO, fin, freq="MS"):
        tr = d.loc[INI_TR:mes - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
        te = d.loc[mes + pd.Timedelta(hours=1):min(mes + pd.offsets.MonthBegin(1), fin)].dropna(subset=["pre_obj"])
        y = np.log(tr["precio_bolsa"].clip(lower=1)) - np.log(tr["pre_obj"].clip(lower=1))
        m = ajustar(nombre, tr[cols], y)
        partes.append(pd.Series(np.exp(m.predict(te[cols]) + np.log(te["pre_obj"].clip(lower=1))), index=te.index))
        print(f"  {nombre} {mes:%Y-%m}: {len(tr)} h ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pd.concat(partes).rename(nombre).to_csv(SAL / f"pred_{nombre}.csv")
    print(f"{nombre} listo en {(time.time() - t0) / 60:.1f} min", flush=True)


CALIENTA = pd.Timedelta(days=60)      # el combinador arranca de cero en cada periodo, 60 dias antes (como
                                       # realista_validacion_historica_v2.py): con toda la historia desde 2020 el QRA
                                       # diario crece mas que cuadratico (153 dias: 25 s; 426 dias: 339 s por modelo)


def ventana(p, ini, fin):
    return p.loc[pd.Timestamp(ini) + pd.Timedelta(hours=1):(pd.Timestamp(fin) + pd.Timedelta(hours=24)) if fin else None]


def combinar(args):
    q, m = args
    return causal(q, q.iloc[:0], ["Predespacho", m], "g4", objetivo="mae")


def evaluar():
    from multiprocessing import Pool
    d = rc.tabla_realista(produccion=True).set_index("fecha_hora")
    p = pd.concat([pd.read_csv(SAL / f"pred_{m}.csv", index_col=0, parse_dates=True).iloc[:, 0] for m in MODELOS], axis=1)
    p["real"], p["Predespacho"], p["Persistencia"] = d["precio_bolsa"], d["pre_obj"], d["precio_lag24h"]
    p = p.dropna(subset=MODELOS + ["Predespacho"]).sort_index()
    p["dia"], p["g4"] = (p.index - pd.Timedelta(hours=1)).normalize(), p.index.hour // 6
    arboles = ["CatBoost", "LightGBM", "XGBoost"]
    p["Prom. 3 boosting"] = p[arboles].mean(axis=1)
    solos = MODELOS + ["Prom. 3 boosting"]
    trozos = {nom: p.loc[pd.Timestamp(ini) - CALIENTA:(pd.Timestamp(fin) + pd.Timedelta(hours=24)) if fin else None].copy()
              for nom, ini, fin in PERIODOS}
    tareas = [(nom, m) for nom in trozos for m in solos]
    with Pool(12) as pool:
        salidas = pool.map(combinar, [(trozos[nom], m) for nom, m in tareas])
    for (nom, m), s in zip(tareas, salidas):
        trozos[nom][f"Pre + {m}"] = s
    todos = ["Persistencia", "Predespacho"] + solos + [f"Pre + {m}" for m in solos]
    pd.concat([ventana(trozos[nom], ini, fin)[["real"] + todos] for nom, ini, fin in PERIODOS]).to_csv(
        SAL / "predicciones_combinadas.csv")
    ref = "Pre + CatBoost"
    filas, dms = [], []
    for nom, ini, fin in PERIODOS:
        ev = ventana(trozos[nom], ini, fin).dropna(subset=todos + ["real"])
        y = ev["real"].to_numpy()
        mp = np.abs(y - ev["Persistencia"]).mean()
        for m in todos:
            e = y - ev[m].to_numpy()
            fila = {"periodo": nom, "horas": len(ev), "modelo": m, "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
                    "MAPE": 100 * np.mean(np.abs(e) / np.maximum(y, 1)), "rMAE": np.abs(e).mean() / mp}
            if m != ref:
                fila["DM_t_vs_PreCatBoost"], fila["DM_p"] = er.dm(y, ev[m].to_numpy(), ev[ref].to_numpy())
            filas.append(fila)
        for i, a in enumerate(todos):
            for b in todos[i + 1:]:
                t, pv = er.dm(y, ev[a].to_numpy(), ev[b].to_numpy())
                dms.append({"periodo": nom, "A": a, "B": b, "DM_t": t, "p": pv})
    t = pd.DataFrame(filas)
    t.to_csv(SAL / "metricas.csv", index=False)
    pd.DataFrame(dms).to_csv(SAL / "dm.csv", index=False)
    tab = t.pivot(index="modelo", columns="periodo", values="MAE").reindex(todos)[[x[0] for x in PERIODOS]]
    tab["media"] = tab.mean(axis=1)
    pd.set_option("display.width", 250)
    print(tab.round(2).to_string())
    print("\nDM frente a Pre + CatBoost (t; positivo = el modelo tiene MAS error):")
    print(t.pivot(index="modelo", columns="periodo", values="DM_t_vs_PreCatBoost").reindex(todos)[[x[0] for x in PERIODOS]].round(2).to_string())
    print("\nHoras por periodo:", t.groupby("periodo")["horas"].first().to_dict())


if __name__ == "__main__":
    if sys.argv[1] == "--evaluar":
        evaluar()
    else:
        entrenar(sys.argv[1])
