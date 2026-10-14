# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: N-BEATSx y N-HiTS con el predespacho como variable futura conocida.

Igual que escenario_realista_24h.neuronales() (97 pasos desde la ultima hora publicada, 23:00 de D-4; se guardan los
ultimos 24), pero con el predespacho alineado al corte (la hora 00:00 usa el valor de las 23:00 del dia anterior) y el
delta ex post conocido como variables futuras, y el predespacho relativo al dia. Ademas, variante con objetivo
log(precio) y variante con mas pasos de entrenamiento.
Salida: resultados/escenario_realista/mejoras/neuronales_predicciones.csv y neuronales_metricas.csv
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
from reentrenar_tras_fix_demanda import FUTR_EXOG_BIN, FUTR_EXOG_CONT, HIST_EXOG_DL  # noqa: E402

CORTE, FIN, OCULTO = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-08-05 00:00"), 73


def correr(d, futr_extra, log=False, pasos=1000, semilla=42):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS
    H = OCULTO + 24
    o = d[d["fecha_hora"] <= FIN].dropna(subset=["precio_bolsa"]).reset_index(drop=True)
    n_dias = int(((o["fecha_hora"] > CORTE) & (o["fecha_hora"] <= FIN)).sum() // 24)
    fut = FUTR_EXOG_CONT + FUTR_EXOG_BIN + futr_extra
    nf = o[["fecha_hora", "precio_bolsa"] + HIST_EXOG_DL + fut].copy()
    mtr = o["fecha_hora"] <= CORTE - pd.Timedelta(hours=OCULTO)
    for c in HIST_EXOG_DL + FUTR_EXOG_CONT + futr_extra:
        nf[c] = (nf[c] - nf.loc[mtr, c].mean()) / nf.loc[mtr, c].std()
    nf[HIST_EXOG_DL + fut] = nf[HIST_EXOG_DL + fut].ffill().bfill()
    if log:
        nf["precio_bolsa"] = np.log(nf["precio_bolsa"])
    nf["unique_id"] = "p"
    nf = nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})[["unique_id", "ds", "y"] + HIST_EXOG_DL + fut]
    kw = dict(h=H, input_size=168, hist_exog_list=HIST_EXOG_DL, futr_exog_list=fut, max_steps=pasos, val_check_steps=100,
              random_seed=semilla, enable_progress_bar=False)
    cv = NeuralForecast(models=[NBEATSx(**kw), NHITS(**kw)], freq="h").cross_validation(df=nf, n_windows=n_dias, step_size=24)
    cv = cv[(cv["ds"] - cv["cutoff"]) > pd.Timedelta(hours=OCULTO)].drop_duplicates("ds", keep="last").set_index("ds")
    if log:
        cv[["NBEATSx", "NHITS"]] = np.exp(cv[["NBEATSx", "NHITS"]])
    return cv["NBEATSx"], cv["NHITS"]


if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().copy()
    d["pre_feat"] = d["pre_obj"]
    d["pre_rel_feat"] = d["pre_rel"]
    d["delta_feat"] = d["delta_7d"].fillna(0)
    preds = pd.DataFrame()
    for nom, extra, log, pasos in (("pre", ["pre_feat"], False, 1000), ("pre+rel+delta", ["pre_feat", "pre_rel_feat", "delta_feat"], False, 1000),
                                   ("pre+rel+delta log", ["pre_feat", "pre_rel_feat", "delta_feat"], True, 1000),
                                   ("pre+rel+delta 2000 pasos", ["pre_feat", "pre_rel_feat", "delta_feat"], False, 2000)):
        nb, nh = correr(d, extra, log, pasos)
        preds[f"N-BEATSx · {nom}"], preds[f"N-HiTS · {nom}"] = nb, nh
        print(f"  {nom} listo ({(time.time() - t0) / 60:.1f} min)", flush=True)
    real = d.set_index("fecha_hora")["precio_bolsa"]
    preds["real"] = real.reindex(preds.index)
    preds.to_csv(rc.MEJ / "neuronales_predicciones.csv")
    ref = rc.referencia()
    ev = preds[preds.index >= "2026-01-15"].dropna().index.intersection(ref.dropna(subset=["ens7_realista"]).index)
    filas = [{"variante": c, **rc.metricas(preds.loc[ev, "real"], preds.loc[ev, c])} for c in preds.columns if c != "real"]
    for c, n in (("N-BEATSx", "REFERENCIA N-BEATSx realista sin predespacho"), ("ens7_realista", "REFERENCIA ensamble 7 realista")):
        filas.append({"variante": n, **rc.metricas(preds.loc[ev, "real"], ref.loc[ev, c])})
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / "neuronales_metricas.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
