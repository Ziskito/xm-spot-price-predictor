# -*- coding: utf-8 -*-
"""
Reentrenamiento de los origenes historicos 1-5 del walk-forward con el ONI corregido (15-oct-2026, a pedido de Juan).

La actualizacion del 14-oct (actualizar_datos_oct2026.py) cambio el ONI en 4 trimestres historicos (2021: JFM -0,9 ->
-0,8 y JJA -0,3 -> -0,4; 2023: DJF -0,5 -> -0,6 y JAS 1,3 -> 1,2), que tocan 5.780 horas de feb-mar y jul-ago de 2021
y de ene-feb y ago-sep de 2023. Todos los modelos menos la persistencia usan el ONI, y esas horas caen en el
entrenamiento o en la prueba de los 5 origenes. El Origen 6 ya se reentreno ese dia (reentrenar_todo_oct2026.py).

Procedimiento identico al notebook 10 (celdas de prediccion), que genero walkforward_predicciones_crudas.csv:
Persistencia, XGBoost (depth 3, lr 0,01, 500 arboles), ARX+GARCH (15 regresoras con festivos) y N-BEATSx/N-HiTS
(input 168, 1000 pasos, semilla 42), entrenados con todo lo anterior a cada origen.
  --verificar  corre el Origen 1 con los datos anteriores a la correccion (backups_pre_actualizacion_14oct2026) y lo
               compara con lo guardado: debe reproducirlo antes de confiar en el reentreno.
Sin argumentos: reentrena los origenes 1-5 con los datos vigentes, reemplaza sus filas en
walkforward_predicciones_crudas.csv y recalcula diebold_mariano.csv (prueba del notebook 10, los 6 origenes).
Copias previas en resultados/backups_pre_origenes_oni_15oct2026/. Resumen en resultados/reentreno_oct2026/origenes_oni.txt
"""
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "data" / "processed"
RES = PROC / "resultados"
BK_DATOS = RES / "backups_pre_actualizacion_14oct2026"

ORIGENES = [
    {"nombre": "Origen 1", "regimen": "La Nina 2020-21", "corte_train": "2020-07-01", "test_inicio": "2020-07-01", "test_fin": "2021-06-30 23:00"},
    {"nombre": "Origen 2", "regimen": "La Nina 2021-22", "corte_train": "2021-07-01", "test_inicio": "2021-07-01", "test_fin": "2022-06-30 23:00"},
    {"nombre": "Origen 3", "regimen": "La Nina triple-dip -> neutral", "corte_train": "2022-07-01", "test_inicio": "2022-07-01", "test_fin": "2023-06-30 23:00"},
    {"nombre": "Origen 4", "regimen": "El Nino 2023-24 (episodio completo)", "corte_train": "2023-07-01", "test_inicio": "2023-07-01", "test_fin": "2024-06-30 23:00"},
    {"nombre": "Origen 5", "regimen": "Neutral / La Nina debil 2024-25", "corte_train": "2024-07-01", "test_inicio": "2024-07-01", "test_fin": "2025-06-30 23:00"},
    {"nombre": "Origen 6", "regimen": "El Nino 2026 (neutral->fuerte)", "corte_train": "2026-01-01", "test_inicio": "2026-01-01", "test_fin": "2026-08-05"},
]
EXCLUIR = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora", "dia_semana", "dia_anio"]
REGRESORAS_ARX = ["aportes_hidricos", "volumen_embalses", "oni", "es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin",
                  "dia_semana_cos", "es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h", "festivo_lag168h",
                  "mismatch_festivo_24h", "mismatch_festivo_168h"]
HIST_EXOG_DL = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUTR_EXOG_CONT = ["oni"]
FUTR_EXOG_ESC = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos", "es_festivo", "festivo_lag24h", "festivo_lag48h",
                 "festivo_lag72h", "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]


def cargar(carpeta):
    a = pd.read_csv(carpeta / "dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(carpeta / "dataset_features_2026.csv", parse_dates=["fecha_hora"])
    return pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)


def predecir(df, o):
    """Notebook 10, celdas 4 y 5, para un origen."""
    import xgboost as xgb
    from arch import arch_model
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS
    cols = [c for c in df.columns if c not in EXCLUIR]
    train = df[df["fecha_hora"] < o["corte_train"]].dropna(subset=cols).copy()
    test = df[(df["fecha_hora"] >= o["test_inicio"]) & (df["fecha_hora"] <= o["test_fin"])].copy()
    filas = [pd.DataFrame({"origen": o["nombre"], "fecha_hora": test["fecha_hora"].to_numpy(), "real": test["precio_bolsa"].to_numpy(),
                           "modelo": "Persistencia", "prediccion": test["precio_lag24h"].to_numpy()})]
    m = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, random_state=42)
    m.fit(train[cols], np.log(train["precio_bolsa"]))
    filas.append(pd.DataFrame({"origen": o["nombre"], "fecha_hora": test["fecha_hora"].to_numpy(), "real": test["precio_bolsa"].to_numpy(),
                               "modelo": "XGBoost", "prediccion": np.exp(m.predict(test[cols]))}))
    for x in (train, test):
        x["precio_lag24h_log"] = np.log(x["precio_lag24h"])
    cx = REGRESORAS_ARX + ["precio_lag24h_log"]
    yl = np.log(train["precio_bolsa"])
    ym, ys = yl.mean(), yl.std()
    xm, xs = train[cx].mean(), train[cx].std()
    r = arch_model((yl - ym) / ys * 10, x=(train[cx] - xm) / xs, mean="ARX", lags=0, vol="GARCH", p=1, q=1,
                   dist="normal").fit(disp="off", options={"maxiter": 500})
    pm = r.params[["Const"] + cx]
    pred = np.exp((pm["Const"] + ((test[cx] - xm) / xs * pm[cx]).sum(axis=1)) / 10 * ys + ym)
    filas.append(pd.DataFrame({"origen": o["nombre"], "fecha_hora": test["fecha_hora"].to_numpy(), "real": test["precio_bolsa"].to_numpy(),
                               "modelo": "ARX+GARCH", "prediccion": pred.to_numpy()}))
    futr = FUTR_EXOG_CONT + FUTR_EXOG_ESC
    dfo = pd.concat([train, test], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    mtr = dfo["fecha_hora"] < o["corte_train"]
    nf = dfo[["fecha_hora", "precio_bolsa"] + HIST_EXOG_DL + futr].copy()
    for c in HIST_EXOG_DL + FUTR_EXOG_CONT:
        mu, sg = nf.loc[mtr, c].mean(), nf.loc[mtr, c].std()
        nf[c] = (nf[c] - mu) / sg
    nf["unique_id"] = "precio_bolsa"
    nf = nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})[["unique_id", "ds", "y"] + HIST_EXOG_DL + futr]
    n_win = int((~mtr).sum()) // 24
    mods = [NBEATSx(h=24, input_size=168, hist_exog_list=HIST_EXOG_DL, futr_exog_list=futr, max_steps=1000, val_check_steps=100,
                    random_seed=42, enable_progress_bar=False),
            NHITS(h=24, input_size=168, hist_exog_list=HIST_EXOG_DL, futr_exog_list=futr, max_steps=1000, val_check_steps=100,
                  random_seed=42, enable_progress_bar=False)]
    cv = NeuralForecast(models=mods, freq="h").cross_validation(df=nf, n_windows=n_win, step_size=24)
    for col, nom in (("NBEATSx", "N-BEATSx"), ("NHITS", "N-HiTS")):
        filas.append(pd.DataFrame({"origen": o["nombre"], "fecha_hora": cv["ds"].to_numpy(), "real": cv["y"].to_numpy(),
                                   "modelo": nom, "prediccion": cv[col].to_numpy()}))
    return pd.concat(filas, ignore_index=True)


def dm(e1, e2, horizonte=24):
    d = np.asarray(e1) - np.asarray(e2)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": horizonte - 1})
    return r.tvalues[0], r.pvalues[0]


def diebold_mariano(crudas):
    """Notebook 10, ultima celda."""
    pares = [("N-BEATSx", "ARX+GARCH"), ("N-BEATSx", "Persistencia"), ("N-HiTS", "ARX+GARCH"), ("N-HiTS", "Persistencia"),
             ("ARX+GARCH", "Persistencia"), ("XGBoost", "Persistencia")]
    filas = []
    for o in ORIGENES:
        d = crudas[crudas["origen"] == o["nombre"]]
        for m1, m2 in pares:
            mg = d[d["modelo"] == m1].sort_values("fecha_hora").merge(d[d["modelo"] == m2].sort_values("fecha_hora"),
                                                                       on="fecha_hora", suffixes=("_1", "_2"))
            t, p = dm(np.abs(mg["real_1"] - mg["prediccion_1"]), np.abs(mg["real_2"] - mg["prediccion_2"]))
            concl = (f"{m1} PIERDE (sig.)" if t > 0 else f"{m1} GANA (sig.)") if p < 0.05 else "sin diferencia significativa"
            filas.append({"origen": o["nombre"], "regimen": o["regimen"], "modelo_1": m1, "modelo_2": m2,
                          "dm_stat": t, "p_valor": p, "conclusion": concl})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    t0 = time.time()
    crudas = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    if "--verificar" in sys.argv:
        n = predecir(cargar(BK_DATOS), ORIGENES[0])
        v = crudas[crudas["origen"] == "Origen 1"]
        for mdl in ("Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"):
            a = n[n.modelo == mdl].set_index("fecha_hora")["prediccion"]
            b = v[v.modelo == mdl].set_index("fecha_hora")["prediccion"].reindex(a.index)
            print(f"  {mdl:12s} max |dif| {np.abs(a - b).max():.2e}  ({len(a)} h)", flush=True)
        print(f"Listo en {(time.time() - t0) / 60:.1f} min")
        sys.exit(0)
    bk = RES / "backups_pre_origenes_oni_15oct2026"
    bk.mkdir(exist_ok=True)
    for f in ("walkforward_predicciones_crudas.csv", "diebold_mariano.csv"):
        if not (bk / f).exists():
            shutil.copy2(RES / f, bk / f)
    df = cargar(PROC)
    nuevos = []
    for o in ORIGENES[:5]:
        nuevos.append(predecir(df, o))
        print(f"  {o['nombre']} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    nuevo = pd.concat(nuevos, ignore_index=True)
    viejo = crudas[crudas["origen"].isin([o["nombre"] for o in ORIGENES[:5]])]
    lineas = ["MAE por origen y modelo (antes -> con el ONI corregido):"]
    mae = lambda x: x.groupby(["origen", "modelo"]).apply(lambda g: np.abs(g["real"] - g["prediccion"]).mean()).unstack()
    a, b = mae(viejo), mae(nuevo)
    for org in a.index:
        lineas.append(f"  {org}: " + " | ".join(f"{m} {a.loc[org, m]:.2f} -> {b.loc[org, m]:.2f}" for m in a.columns))
    crudas = pd.concat([crudas[~crudas["origen"].isin(nuevo["origen"].unique())], nuevo], ignore_index=True)
    crudas = crudas.sort_values(["origen", "modelo", "fecha_hora"])
    crudas.to_csv(RES / "walkforward_predicciones_crudas.csv", index=False)
    dmv = pd.read_csv(RES / "diebold_mariano.csv")
    dmn = diebold_mariano(crudas)
    dmn.to_csv(RES / "diebold_mariano.csv", index=False)
    cambia = dmv.merge(dmn, on=["origen", "modelo_1", "modelo_2"], suffixes=("_antes", ""))
    cambia = cambia[cambia["conclusion_antes"] != cambia["conclusion"]]
    lineas.append(f"Diebold-Mariano: {len(cambia)} de {len(dmn)} conclusiones cambian" + (":" if len(cambia) else "."))
    for _, f in cambia.iterrows():
        lineas.append(f"  {f['origen']} {f['modelo_1']} vs {f['modelo_2']}: {f['conclusion_antes']} -> {f['conclusion']} (t = {f['dm_stat']:.2f})")
    texto = "\n".join(lineas)
    (RES / "reentreno_oct2026").mkdir(exist_ok=True)
    (RES / "reentreno_oct2026" / "origenes_oni.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
