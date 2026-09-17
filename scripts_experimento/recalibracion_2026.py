# -*- coding: utf-8 -*-
"""
Problema 1, adaptado a 2026: RECALIBRACION de los modelos base durante el ano de publicacion.

Hasta ahora cada modelo se entrena una sola vez con datos hasta el 31-dic-2025 y se usa congelado
todo 2026. Recalibrar = antes de pronosticar cada ventana, volver a estimar el modelo con todo lo
observado hasta el corte. Es el estandar del benchmark de Lago et al. (2021) y la forma directa de
que el modelo incorpore el regimen de 2026 en vez de corregirlo desde afuera (que ya fallo).

  - ARX+GARCH: recalibracion DIARIA (se reestima antes de cada una de las 216 ventanas).
  - XGBoost:   diaria si cada ajuste tarda < 6 s; si no, semanal.
  - N-BEATSx y N-HiTS: reentrenamiento cada REFIT ventanas (argumento; 30 = mensual), con el
    parametro `refit` de NeuralForecast.cross_validation. Diario costaria ~18 horas en esta CPU.

Protocolo del Origen 6 del notebook 10 (o6_comun.py). Control de reproducibilidad: antes del primer
reentrenamiento, las redes deben reproducir las predicciones congeladas de walkforward_predicciones_crudas.csv.
Uso: python recalibracion_2026.py [REFIT]
"""
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import (cargar_completo, particion_o6, df_neural, columnas_features, crudas_o6,
                      HIST, FUT, REGRESORAS_ARX, RES)

RUTA = RES / "recalibracion_2026_predicciones.csv"


def arx_garch(train, test):
    from arch import arch_model
    train, test = train.copy(), test.copy()
    train["lp"] = np.log(train["precio_lag24h"])
    test["lp"] = np.log(test["precio_lag24h"])
    cols = REGRESORAS_ARX + ["lp"]
    yl = np.log(train["precio_bolsa"])
    ym, ys = yl.mean(), yl.std()
    xm, xs = train[cols].mean(), train[cols].std()
    r = arch_model((yl - ym) / ys * 10, x=(train[cols] - xm) / xs, mean="ARX", lags=0, vol="GARCH",
                   p=1, q=1, dist="normal").fit(disp="off", options={"maxiter": 500})
    pm = r.params[["Const"] + cols]
    esc = pm["Const"] + ((test[cols] - xm) / xs * pm[cols]).sum(axis=1)
    return np.exp(esc.to_numpy() / 10 * ys + ym)


def xgboost(train, test, cf):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.8,
                         colsample_bytree=0.8, random_state=42)
    m.fit(train[cf], np.log(train["precio_bolsa"]))
    return m


def main():
    refit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    df = cargar_completo()
    cf = columnas_features(df)
    train0, test = particion_o6(df)
    nf_df, n_win = df_neural(train0, test)
    horas = test["fecha_hora"].iloc[-n_win * 24:].reset_index(drop=True)
    ventanas = [horas.iloc[i * 24:(i + 1) * 24] for i in range(n_win)]
    dfi = df.set_index("fecha_hora")
    out = pd.DataFrame({"fecha_hora": horas})
    out["real"] = dfi.loc[horas, "precio_bolsa"].to_numpy()
    out["Persistencia"] = dfi.loc[horas, "precio_lag24h"].to_numpy()

    # ---------------- ARX+GARCH diario ----------------
    t0 = time.time()
    pred = []
    for i, w in enumerate(ventanas):
        corte = w.iloc[0] - pd.Timedelta(hours=1)
        tr = df[df["fecha_hora"] <= corte].dropna(subset=cf)
        pred.append(arx_garch(tr, dfi.loc[w].reset_index()))
        if i == 0:
            print(f"ARX+GARCH: {time.time()-t0:.1f} s por ajuste -> ~{(time.time()-t0)*n_win/60:.0f} min", flush=True)
    out["ARX_rec"] = np.concatenate(pred)
    out.to_csv(RUTA, index=False)
    print(f"ARX+GARCH diario listo ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---------------- XGBoost ----------------
    t0 = time.time()
    tr0 = df[df["fecha_hora"] <= ventanas[0].iloc[0] - pd.Timedelta(hours=1)].dropna(subset=cf)
    xgboost(tr0, None, cf)
    seg = time.time() - t0
    cada = 1 if seg < 6 else 7
    print(f"XGBoost: {seg:.1f} s por ajuste -> recalibracion cada {cada} dia(s)", flush=True)
    pred, modelo = [], None
    for i, w in enumerate(ventanas):
        if i % cada == 0:
            corte = w.iloc[0] - pd.Timedelta(hours=1)
            modelo = xgboost(df[df["fecha_hora"] <= corte].dropna(subset=cf), None, cf)
        pred.append(np.exp(modelo.predict(dfi.loc[w, cf])))
    out["XGB_rec"] = np.concatenate(pred)
    out.to_csv(RUTA, index=False)
    print(f"XGBoost listo ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---------------- Redes con refit ----------------
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS
    comunes = dict(h=24, input_size=168, hist_exog_list=HIST, futr_exog_list=FUT, max_steps=1000,
                   val_check_steps=100, random_seed=42, enable_progress_bar=False)
    t0 = time.time()
    cv = NeuralForecast(models=[NBEATSx(**comunes), NHITS(**comunes)], freq="h").cross_validation(
        df=nf_df, n_windows=n_win, step_size=24, refit=refit)
    cv = cv.rename(columns={"ds": "fecha_hora"}).set_index("fecha_hora")
    out["NBEATSx_rec"] = cv.loc[horas, "NBEATSx"].to_numpy()
    out["NHITS_rec"] = cv.loc[horas, "NHITS"].to_numpy()
    out.to_csv(RUTA, index=False)
    print(f"Redes con refit={refit} listas ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---------------- controles y resumen ----------------
    c = crudas_o6()
    m = out.set_index("fecha_hora").join(c[["ARX+GARCH", "XGBoost", "N-BEATSx", "N-HiTS"]], how="inner")
    antes = m.index < ventanas[min(refit, n_win - 1)].iloc[0]
    for rec, cong in [("NBEATSx_rec", "N-BEATSx"), ("NHITS_rec", "N-HiTS")]:
        dif = (m.loc[antes, rec] - m.loc[antes, cong]).abs().max()
        print(f"Reproducibilidad {cong} antes del primer reentrenamiento: diferencia maxima = {dif:.6f}")
    real = m["real"]
    print(f"\n{'modelo':12s} {'congelado':>10s} {'recalibrado':>12s}")
    for rec, cong in [("ARX_rec", "ARX+GARCH"), ("XGB_rec", "XGBoost"), ("NBEATSx_rec", "N-BEATSx"), ("NHITS_rec", "N-HiTS")]:
        print(f"{cong:12s} {(real-m[cong]).abs().mean():10.2f} {(real-m[rec]).abs().mean():12.2f}")
    print(f"Guardado: {RUTA}")


if __name__ == "__main__":
    main()
