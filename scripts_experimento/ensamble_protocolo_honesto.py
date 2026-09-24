# -*- coding: utf-8 -*-
"""
EL ENSAMBLE DE 24h CON EL CORTE DAY-AHEAD HONESTO: cuanto vale de verdad.

CONTEXTO (ver `protocolo_dayahead_honesto.py` y la bitacora del 2026-09-23). El dia de despacho en
Colombia va de 00:00 a 23:00 -- la unica transicion horaria sin repeticiones exactas de precio es
23:00 -> 00:00. El protocolo del proyecto corta a las 00:00 del dia D, que es la PRIMERA hora de ese
mismo dia de despacho, y pronostica 01:00 de D a 00:00 de D+1. Medido con el LEAR, el corte honesto
(23:00 del dia previo) sube el MAE de 53.27 a 61.65 y lo deja por detras de la persistencia.

QUE VOTANTES ESTAN AFECTADOS. Se mide el MAE de cada votante en la hora 1 (la mas cercana al corte)
y en la hora 0 (el unico objetivo de un dia de despacho nuevo):

    votante        hora 1   hora 0
    Persistencia     47.1     46.8     <- honesto: no usa el corte
    XGBoost          51.1     52.7     <- honesto: variables con rezago de 24h
    ARX+GARCH        46.0     46.0     <- honesto
    N-BEATSx         10.3     52.3     <- usa el precio del mismo dia de despacho
    N-HiTS           10.4     49.0     <- idem
    GARCH-ged         7.5     47.6     <- idem (disenos de lear_24h con corte a las 00:00)

Los tres primeros no cambian. Los tres ultimos se REENTRENAN aqui con el corte a las 23:00 del dia
previo, con los mismos hiperparametros que sus versiones actuales:
  - N-BEATSx y N-HiTS: ventanas 00:00 -> 23:00 (es exactamente lo que hacia la "correccion" de
    `corregir_nbeatsx_origen6.py`, que se revirtio por empeorar el modelo).
  - GARCH-ged: el mismo LASSO + GARCH(1,1) con errores GED de `familia_garch_24h.py`, con cortes a
    las 23:00 en vez de a las 00:00.

NO SE TOCA NINGUN ARCHIVO DEL PROYECTO. Todo se escribe con sufijo `_corte23`. En particular NO se
modifica `walkforward_predicciones_crudas.csv` (la vez anterior se sobrescribio sin respaldo).
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES, crudas_o6
from lear_24h import construir_para_paso
from combinador_optimo_mape import qra_cv, mape, smape

CORTE = pd.Timestamp("2026-01-01")
TEST_FIN = "2026-08-04 23:00"
SEMILLAS = list(range(10))
PICO = [0, 8, 9, 10, 18, 19]
HONESTOS = ["Persistencia", "XGBoost", "ARX+GARCH"]

HIST_EXOG_DL = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUTR_CONT = ["oni"]
FUTR_BIN = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos", "es_festivo", "festivo_lag24h",
            "festivo_lag48h", "festivo_lag72h", "festivo_lag168h", "mismatch_festivo_24h",
            "mismatch_festivo_168h"]


def neuronales_corte23():
    """N-BEATSx y N-HiTS con ventanas 00:00 -> 23:00 (corte a las 23:00 del dia previo)."""
    salida = RES / "nbeatsx_nhits_corte23.csv"
    if salida.exists():
        print(f"  {salida.name} ya existe, se reutiliza")
        return pd.read_csv(salida, parse_dates=["fecha_hora"])
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS

    df = cargar_completo()
    excl = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora", "dia_semana", "dia_anio"]
    feats = [c for c in df.columns if c not in excl]
    train = df[df["fecha_hora"] < CORTE].dropna(subset=feats)
    test = df[(df["fecha_hora"] >= CORTE) & (df["fecha_hora"] <= TEST_FIN)]
    assert len(test) % 24 == 0 and test["fecha_hora"].min().hour == 0
    o = pd.concat([train, test]).sort_values("fecha_hora").reset_index(drop=True)
    m = o["fecha_hora"] < CORTE
    nf_df = o[["fecha_hora", "precio_bolsa"] + HIST_EXOG_DL + FUTR_CONT + FUTR_BIN].copy()
    for c in HIST_EXOG_DL + FUTR_CONT:
        nf_df[c] = (nf_df[c] - nf_df.loc[m, c].mean()) / nf_df.loc[m, c].std()
    nf_df["unique_id"] = "precio_bolsa"
    nf_df = nf_df.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})
    kw = dict(h=24, input_size=168, hist_exog_list=HIST_EXOG_DL, futr_exog_list=FUTR_CONT + FUTR_BIN,
              max_steps=1000, val_check_steps=100, random_seed=42, enable_progress_bar=False)
    nf = NeuralForecast(models=[NBEATSx(**kw), NHITS(**kw)], freq="h")
    t0 = time.time()
    cv = nf.cross_validation(df=nf_df, n_windows=len(test) // 24, step_size=24)
    print(f"  N-BEATSx + N-HiTS entrenados en {(time.time()-t0)/60:.1f} min; "
          f"primer objetivo {cv['ds'].min()}, ultimo {cv['ds'].max()}")
    assert cv["ds"].min() == pd.Timestamp("2026-01-01 00:00")
    r = cv.rename(columns={"ds": "fecha_hora", "NBEATSx": "N-BEATSx", "NHITS": "N-HiTS"})[
        ["fecha_hora", "N-BEATSx", "N-HiTS"]]
    r.to_csv(salida, index=False)
    return r


def garch_ged_corte23():
    """LASSO + GARCH(1,1)-GED de familia_garch_24h.py, con cortes a las 23:00."""
    salida = RES / "pronostico_GARCH-ged_24h_2026_corte23.csv"
    if salida.exists():
        print(f"  {salida.name} ya existe, se reutiliza")
        return pd.read_csv(salida, parse_dates=["fecha_hora"])
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    from arch import arch_model

    df = cargar_completo().reset_index(drop=True)
    f = df["fecha_hora"]
    idx23 = np.where(f.dt.hour.to_numpy() == 23)[0]
    idx_fin = int(pd.Series(df.index.values, index=f)[CORTE])
    cortes_tr = idx23[(idx23 >= 200) & (idx23 < idx_fin - 24)]
    cortes_te = idx23[idx23 >= idx_fin - 1]           # el primero es 2025-12-31 23:00
    partes, alphas, fallos = [], [], 0
    for h in range(1, 25):
        Xtr, ytr, _, _ = construir_para_paso(df, h, cortes_tr)
        Xte, yte, fte, _ = construir_para_paso(df, h, cortes_te)
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        if h <= 4:
            lc = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr)
            alphas.append(lc.alpha_)
            sel = np.abs(lc.coef_) > 1e-8
        else:
            sel = np.abs(Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42)
                         .fit(Ztr, ytr).coef_) > 1e-8
        if sel.sum() == 0:
            sel[:] = True
        A, B = Ztr[:, sel], Zte[:, sel]
        try:
            res = arch_model(ytr, x=A, mean="LS", rescale=False, vol="GARCH", p=1, q=1,
                             dist="ged").fit(disp="off", show_warning=False)
            pred = res.params.iloc[0] + B @ res.params.iloc[1:1 + A.shape[1]].to_numpy()
            if not np.isfinite(pred).all():
                raise ValueError
        except Exception:
            fallos += 1
            pred = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr).predict(Zte)
        partes.append(pd.DataFrame({"fecha_hora": fte, "pred": pred}))
    r = pd.concat(partes).drop_duplicates("fecha_hora").sort_values("fecha_hora")
    r = r[r["fecha_hora"] >= CORTE]
    print(f"  GARCH-ged con corte 23:00: {len(r):,} pronosticos, GARCH no convergio en {fallos}/24 pasos")
    r.to_csv(salida, index=False)
    return r


def main():
    t0 = time.time()
    print("Reentrenando los tres votantes que usaban el mismo dia de despacho...", flush=True)
    g = garch_ged_corte23().set_index("fecha_hora")["pred"].rename("GARCH-ged")
    nn = neuronales_corte23().set_index("fecha_hora")

    base = crudas_o6()                                          # protocolo actual (A)
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    A = base.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    B = base[HONESTOS + ["real"]].join(nn[["N-BEATSx", "N-HiTS"]], how="inner").join(g, how="inner")
    VOT = HONESTOS + ["N-BEATSx", "N-HiTS", "GARCH-ged"]
    comun = A.dropna(subset=VOT + ["real"]).index.intersection(B.dropna(subset=VOT + ["real"]).index)
    A, B = A.loc[comun].copy(), B.loc[comun].copy()
    for d in (A, B):
        d["dia"] = d.index.normalize()
        d["g4"] = d.index.hour // 6
    y = A["real"].to_numpy()
    hh = comun.hour
    print(f"\nInstantes objetivo comunes: {len(comun):,}\n")

    print("=" * 100)
    print("VOTANTES SUELTOS: protocolo actual (A, corte 00:00) contra honesto (B, corte 23:00)")
    print("=" * 100)
    print(f"  {'votante':14s} {'MAE A':>8s} {'MAE B':>8s} {'hora 1 A':>9s} {'hora 1 B':>9s}")
    for v in VOT:
        a, b = A[v].to_numpy(), B[v].to_numpy()
        print(f"  {v:14s} {np.abs(y-a).mean():8.2f} {np.abs(y-b).mean():8.2f} "
              f"{np.abs(y[hh==1]-a[hh==1]).mean():9.2f} {np.abs(y[hh==1]-b[hh==1]).mean():9.2f}")

    print("\n" + "=" * 100)
    print("EL ENSAMBLE (6 votantes, combinador sMAPE, 10 particiones)")
    print("=" * 100)
    MA = np.array([[np.abs(y-p).mean(), mape(y, p), smape(y, p)]
                   for p in (qra_cv(A, VOT, "smape", s_) for s_ in SEMILLAS)])
    MB = np.array([[np.abs(y-p).mean(), mape(y, p), smape(y, p)]
                   for p in (qra_cv(B, VOT, "smape", s_) for s_ in SEMILLAS)])
    pers = A["Persistencia"].to_numpy()
    for nom, M in [("A  protocolo actual (corte 00:00)", MA), ("B  day-ahead honesto (corte 23:00)", MB)]:
        print(f"  {nom:38s} MAE {M[:,0].mean():7.2f}+/-{M[:,0].std():.2f}   "
              f"MAPE {M[:,1].mean():6.2f}%   sMAPE {M[:,2].mean():6.2f}%")
    print(f"  {'persistencia':38s} MAE {np.abs(y-pers).mean():7.2f}         "
          f"MAPE {mape(y, pers):6.2f}%   sMAPE {smape(y, pers):6.2f}%")
    # MASE con el ingenuo de 24h
    ing = np.abs(y - pers).mean()
    print(f"\n  MASE (frente a persistencia 24h):  A = {MA[:,0].mean()/ing:.3f}   B = {MB[:,0].mean()/ing:.3f}")

    pA, pB = qra_cv(A, VOT, "smape"), qra_cv(B, VOT, "smape")
    print(f"\n  POR HORA DEL RELOJ (MAE)")
    print(f"  {'hora':>5s} {'A':>8s} {'B':>8s} {'persist.':>9s}")
    filas = []
    for h in range(24):
        gm = hh == h
        a, b, q = (np.abs(y[gm]-pA[gm]).mean(), np.abs(y[gm]-pB[gm]).mean(), np.abs(y[gm]-pers[gm]).mean())
        filas.append(dict(hora=h, MAE_A=a, MAE_B=b, MAE_persist=q))
        print(f"  {h:5d} {a:8.2f} {b:8.2f} {q:9.2f}")
    pd.DataFrame(filas).to_csv(RES / "ensamble_protocolo_honesto_por_hora.csv", index=False)
    pd.DataFrame({"fecha_hora": comun, "real": y, "pred_A": pA, "pred_B": pB}).to_csv(
        RES / "ensamble_protocolo_honesto.csv", index=False)
    print(f"\nGuardado: ensamble_protocolo_honesto.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
