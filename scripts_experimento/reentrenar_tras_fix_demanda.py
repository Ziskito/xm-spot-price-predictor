# -*- coding: utf-8 -*-
"""
Reentrenamiento tras el parche de demanda del 04-05 agosto 2026 (ver
parchar_demanda_04_05ago2026.py y la bitacora del 2026-09-24).

ALCANCE, verificado antes de escribir este script (no se regenera lo que no cambia):
  Persistencia   usa precio_lag24h -- NO usa demanda. Sin cambios.
  ARX+GARCH      su lista de regresoras (REGRESORAS_ARX) no incluye ninguna variable de
                 demanda. Sin cambios.
  XGBoost        usa TODAS las columnas menos EXCLUIR, que excluye 'demanda' (contemporanea)
                 pero NO excluye demanda_lag24h/48h/72h/media_24h. SE REGENERA.
  N-BEATSx/HiTS  demanda_lag24h esta en su hist_exog_list. SE REGENERA.
  GARCH-ged      su diseño (lear_24h.construir_para_paso) incluye demanda_media_24h en
                 EXOG_CORTE. SE REGENERA.
  LEAR (comparaciones historicas de sesiones anteriores) -- no forma parte del ensamble
                 final de 6 votantes, no se toca.

El entrenamiento (2019-2025) NUNCA se toco por el bug -- es exclusivo de 2 dias de 2026. Asi que
"reentrenar" aqui es exacto solo para XGBoost (se re-fitea con la MISMA data de entrenamiento,
deberia dar pesos identicos, y predice con las features de test ya corregidas). Para GARCH-ged y
N-BEATSx/N-HiTS, que se ajustan sobre walk-forward con datos hasta el corte, el efecto se limita a
los pasos cuyo corte o cuya ventana de contexto toca las 48h corregidas.

Mismos hiperparametros que ya producen los numeros actuales del proyecto -- unico input que
cambia es dataset_features_2026.csv, ya parchado.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import RAIZ, RES

CRUDAS = RES / "walkforward_predicciones_crudas.csv"
TEST_INICIO = "2026-01-01"
CORTE_TRAIN = "2026-01-01"

# Rangos EXACTOS que ya existen hoy en vivo para cada modelo (verificados antes de escribir este
# script, no supuestos): asi la regeneracion reproduce la misma estructura de filas, y solo
# cambian los VALORES de las features en las horas parchadas. Si algo no calza, se aborta antes
# de guardar nada (ver verificar_rango en main()).
XGB_FIN = "2026-08-05 00:00:00"                  # igual al rango vivo de XGBoost/Persistencia/ARX+GARCH
GARCH_FIN = "2026-08-05 23:00:00"                # igual al rango vivo de pronostico_GARCH-ged_24h_2026.csv

# N-BEATSx/N-HiTS pronostican en BLOQUES de 24h (ventana 01:00 -> 00:00 del dia siguiente, corte a
# las 00:00 -- el protocolo day-ahead documentado en la bitacora del 2026-09-23/24), a diferencia
# de XGBoost que predice cada hora suelta. Primer intento uso el mismo corte que XGBoost (00:00 del
# 1-ene) y salio corrido 1h respecto al rango vivo (train hasta 00:00 del 1-ene inclusive, no
# exclusive). Corregido aqui con limites propios para la red:
NEURAL_TRAIN_FIN = "2026-01-01 00:00:00"          # train INCLUYE esta hora (el corte, la ultima
                                                  # informacion disponible antes del primer objetivo)
NEURAL_TEST_INICIO = "2026-01-01 01:00:00"        # primer objetivo pronosticado
NEURAL_TEST_FIN = "2026-08-05 00:00:00"           # ultimo objetivo (igual al rango vivo)

EXCLUIR = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora",
           "dia_semana", "dia_anio"]
HIST_EXOG_DL = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUTR_EXOG_CONT = ["oni"]
FUTR_EXOG_BIN = ["es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
                 "es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
                 "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]


def cargar_completo():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    return pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)


def xgboost_origen6(df):
    import xgboost as xgb
    cols = [c for c in df.columns if c not in EXCLUIR]
    train = df[df["fecha_hora"] < CORTE_TRAIN].dropna(subset=cols).copy()
    test = df[(df["fecha_hora"] >= TEST_INICIO) & (df["fecha_hora"] <= XGB_FIN)].copy()
    m = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
    m.fit(train[cols], np.log(train["precio_bolsa"]))
    pred = np.exp(m.predict(test[cols]))
    return pd.DataFrame({"origen": "Origen 6", "fecha_hora": test["fecha_hora"].to_numpy(),
                         "real": test["precio_bolsa"].to_numpy(), "modelo": "XGBoost",
                         "prediccion": pred})


def neuronales_origen6(df):
    """Misma logica de corregir_nbeatsx_origen6.py, sobre los datos ya corregidos."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS

    train = df[df["fecha_hora"] <= NEURAL_TRAIN_FIN].dropna(subset=[c for c in df.columns if c not in EXCLUIR]).copy()
    test = df[(df["fecha_hora"] >= NEURAL_TEST_INICIO) & (df["fecha_hora"] <= NEURAL_TEST_FIN)].copy()
    n_test = len(test)
    assert n_test % 24 == 0, f"n_test={n_test} no es multiplo de 24"
    n_windows = n_test // 24
    print(f"  train {len(train):,} filas | test {n_test:,} filas | n_windows={n_windows}")

    origen = pd.concat([train, test], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    assert not origen["fecha_hora"].duplicated().any(), "hay fechas duplicadas al unir train+test (revisar limites)"
    mask_train = origen["fecha_hora"] <= NEURAL_TRAIN_FIN

    fut = FUTR_EXOG_CONT + FUTR_EXOG_BIN
    nf = origen[["fecha_hora", "precio_bolsa"] + HIST_EXOG_DL + fut].copy()
    for c in HIST_EXOG_DL + FUTR_EXOG_CONT:
        mu, sd = nf.loc[mask_train, c].mean(), nf.loc[mask_train, c].std()
        nf[c] = (nf[c] - mu) / sd
    nf["unique_id"] = "precio_bolsa"
    nf = nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})[["unique_id", "ds", "y"] + HIST_EXOG_DL + fut]

    kw = dict(h=24, input_size=168, hist_exog_list=HIST_EXOG_DL, futr_exog_list=fut,
             max_steps=1000, val_check_steps=100, random_seed=42, enable_progress_bar=False)
    model_nf = NeuralForecast(models=[NBEATSx(**kw), NHITS(**kw)], freq="h")
    t0 = time.time()
    cv = model_nf.cross_validation(df=nf, n_windows=n_windows, step_size=24)
    print(f"  entrenado en {(time.time()-t0)/60:.1f} min, primer objetivo {cv['ds'].min()}, "
          f"ultimo {cv['ds'].max()}")
    esperado_min, esperado_max = pd.Timestamp("2026-01-01 01:00:00"), pd.Timestamp("2026-08-05 00:00:00")
    assert cv["ds"].min() == esperado_min and cv["ds"].max() == esperado_max, (
        f"El rango de salida ({cv['ds'].min()} a {cv['ds'].max()}) NO coincide con el rango vivo "
        f"actual ({esperado_min} a {esperado_max}). Se aborta SIN guardar nada -- revisar antes de "
        f"reintentar.")

    salidas = []
    for col_modelo, nombre in [("NBEATSx", "N-BEATSx"), ("NHITS", "N-HiTS")]:
        salidas.append(pd.DataFrame({"origen": "Origen 6", "fecha_hora": cv["ds"], "real": cv["y"],
                                     "modelo": nombre, "prediccion": cv[col_modelo]}))
    return pd.concat(salidas, ignore_index=True)


def garch_ged_origen6(df):
    """Mismo diseño y misma variante que familia_garch_24h.py, filtrada a GARCH-ged."""
    from lear_24h import construir_para_paso
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    from arch import arch_model

    dfx = df.reset_index(drop=True)
    fechas = dfx["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(dfx.index.values, index=fechas)[pd.Timestamp("2026-01-01")])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - 24)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]

    partes, alphas, fallos = [], [], 0
    for h in range(1, 25):
        Xtr, ytr, _, _ = construir_para_paso(dfx, h, cortes_tr)
        Xte, yte, fo, _ = construir_para_paso(dfx, h, cortes_te)
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        if h <= 4:
            lc = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr)
            alphas.append(lc.alpha_)
            sel = np.abs(lc.coef_) > 1e-8
        else:
            la = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr)
            sel = np.abs(la.coef_) > 1e-8
        if sel.sum() == 0:
            sel[:] = True
        A, B = Ztr[:, sel], Zte[:, sel]
        respaldo = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr).predict(Zte)
        try:
            res = arch_model(ytr, x=A, mean="LS", rescale=False, vol="GARCH", p=1, q=1,
                             dist="ged").fit(disp="off", show_warning=False)
            par = res.params
            pred = par.iloc[0] + B @ par.iloc[1:1 + A.shape[1]].to_numpy()
            if not np.isfinite(pred).all():
                raise ValueError
        except Exception:
            pred, fallos = respaldo, fallos + 1
        partes.append(pd.DataFrame({"fecha_hora": fo, "real": yte, "pred": pred}))
    r = pd.concat(partes, ignore_index=True).drop_duplicates("fecha_hora").sort_values("fecha_hora")
    r = r[(r["fecha_hora"] >= TEST_INICIO) & (r["fecha_hora"] <= GARCH_FIN)]
    print(f"  GARCH-ged: no convergio en {fallos}/24 pasos")
    return r


def main():
    t0 = time.time()
    df = cargar_completo()
    print(f"Filas totales (2019-2025 + 2026 ya corregido): {len(df):,}\n")

    # ---------- antes (para comparar) ----------
    crudas_antes = pd.read_csv(CRUDAS, parse_dates=["fecha_hora"])
    o6_antes = crudas_antes[crudas_antes["origen"] == "Origen 6"]
    print("ANTES del parche (metricas de cada modelo, Origen 6):")
    for m in ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]:
        s = o6_antes[o6_antes["modelo"] == m]
        if len(s):
            mae = (s["real"] - s["prediccion"]).abs().mean()
            print(f"  {m:14s} MAE={mae:7.2f}  n={len(s)}")

    print("\n" + "=" * 90)
    print("1) XGBoost -- reentrenado con el train identico, prediciendo con features corregidas")
    print("=" * 90)
    xgb_nuevo = xgboost_origen6(df)
    xgb_antes = o6_antes[o6_antes.modelo == "XGBoost"]
    assert len(xgb_nuevo) == len(xgb_antes) and xgb_nuevo["fecha_hora"].min() == xgb_antes["fecha_hora"].min() \
        and xgb_nuevo["fecha_hora"].max() == xgb_antes["fecha_hora"].max(), \
        f"XGBoost: rango nuevo ({len(xgb_nuevo)} filas) no coincide con el vivo ({len(xgb_antes)} filas). Se aborta."
    print(f"  MAE nuevo: {(xgb_nuevo['real']-xgb_nuevo['prediccion']).abs().mean():.2f}  "
          f"(antes: {(xgb_antes['real']-xgb_antes['prediccion']).abs().mean():.2f})")

    print("\n" + "=" * 90)
    print("2) N-BEATSx + N-HiTS -- walk-forward completo sobre 2026 con demanda_lag24h corregida")
    print("=" * 90)
    dl_nuevo = neuronales_origen6(df)
    for m in ["N-BEATSx", "N-HiTS"]:
        s = dl_nuevo[dl_nuevo["modelo"] == m]
        s_antes = o6_antes[o6_antes.modelo == m]
        assert len(s) == len(s_antes), f"{m}: {len(s)} filas nuevas vs {len(s_antes)} vivas. Se aborta."
        mae = (s["real"] - s["prediccion"]).abs().mean()
        mae_antes = (s_antes["real"] - s_antes["prediccion"]).abs().mean()
        print(f"  {m}: MAE nuevo={mae:.2f}  (antes={mae_antes:.2f})")

    print("\n" + "=" * 90)
    print("3) GARCH-ged")
    print("=" * 90)
    garch_nuevo = garch_ged_origen6(df)
    garch_antes = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"])
    assert len(garch_nuevo) == len(garch_antes) and garch_nuevo["fecha_hora"].min() == garch_antes["fecha_hora"].min() \
        and garch_nuevo["fecha_hora"].max() == garch_antes["fecha_hora"].max(), \
        f"GARCH-ged: rango nuevo ({len(garch_nuevo)} filas) no coincide con el vivo ({len(garch_antes)} filas). Se aborta."
    comun = garch_nuevo["fecha_hora"].isin(garch_antes["fecha_hora"])
    mae_g_antes = (garch_antes.set_index("fecha_hora").loc[garch_nuevo.loc[comun, "fecha_hora"], "real"] -
                  garch_antes.set_index("fecha_hora").loc[garch_nuevo.loc[comun, "fecha_hora"], "pred"]).abs().mean()
    print(f"  MAE nuevo: {(garch_nuevo['real']-garch_nuevo['pred']).abs().mean():.2f}  (antes: {mae_g_antes:.2f})")

    # ---------- parchar walkforward_predicciones_crudas.csv (solo Origen 6, solo estos 3 modelos) ----------
    print("\n" + "=" * 90)
    print("4) Guardando: parchar walkforward_predicciones_crudas.csv y pronostico_GARCH-ged_24h_2026.csv")
    print("=" * 90)
    viejo = crudas_antes[~((crudas_antes["origen"] == "Origen 6") &
                          (crudas_antes["modelo"].isin(["XGBoost", "N-BEATSx", "N-HiTS"])))]
    nuevo_completo = pd.concat([viejo, xgb_nuevo, dl_nuevo], ignore_index=True).sort_values(
        ["origen", "modelo", "fecha_hora"]).reset_index(drop=True)
    nuevo_completo.to_csv(CRUDAS, index=False)
    print(f"  walkforward_predicciones_crudas.csv: {len(crudas_antes):,} -> {len(nuevo_completo):,} filas")

    garch_out = garch_nuevo.rename(columns={"pred": "pred"})
    garch_out.to_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", index=False)
    print(f"  pronostico_GARCH-ged_24h_2026.csv sobreescrito ({len(garch_out)} filas)")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
