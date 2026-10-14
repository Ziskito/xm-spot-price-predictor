# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: frecuencia de reentrenamiento y semillas del CatBoost con el predespacho por dentro.

Mismo CatBoost de realista_exp_votantes.py (R+P, objetivo log(precio/predespacho)), reentrenado
  semanal          cada 7 dias desde el 1-ene, con todo lo publicado (hasta 73 h antes del corte)
  mensual 3 sem.   cada mes, promedio de 3 semillas
y combinado con el predespacho (combinador causal realista, franjas de 6 h).
Salida: resultados/escenario_realista/mejoras/frecuencia_predicciones.csv y frecuencia_metricas.csv
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
from realista_exp_votantes import INI_TR, MARGEN, catboost  # noqa: E402


def reentrenado(d, cols, inicios, fin, semillas=(42,)):
    partes = []
    for k, ini in enumerate(inicios):
        tr = d.loc[INI_TR:ini - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
        hasta = inicios[k + 1] if k + 1 < len(inicios) else pd.Timestamp(fin)
        te = d.loc[ini + pd.Timedelta(hours=1):hasta].dropna(subset=["pre_obj"])
        if len(te):
            partes.append(sum(catboost(tr, te, cols, s) for s in semillas) / len(semillas))
    return pd.concat(partes)


if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    cols = conjuntos(d)["R+P"]
    v = pd.read_csv(rc.MEJ / "votantes_predicciones.csv", parse_dates=[0], index_col=0)
    pan = v[["real", "dia", "g4", "Predespacho", "Persistencia", "N-HiTS", "ens7_realista", "CatBoost_mensual", "Pre+Cat mensual"]].copy()
    pan["dia"] = pd.to_datetime(pan["dia"])
    fin = pan.index.max()
    pan["CatBoost_semanal"] = reentrenado(d, cols, list(pd.date_range("2026-01-01", fin, freq="7D")), fin).reindex(pan.index)
    print(f"  semanal ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pan["CatBoost_mensual_3s"] = reentrenado(d, cols, list(pd.date_range("2026-01-01", fin, freq="MS")), fin, (42, 7, 2026)).reindex(pan.index)
    print(f"  mensual 3 semillas ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # previa del combinador: CatBoost entrenado antes de la previa
    from ensamble_24h_enero import origen5
    from ensamble_con_predespacho_24h import imar
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    tr = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    previa["Cat"] = catboost(tr, d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"]), cols).reindex(previa.index)
    previa = previa.dropna(subset=["Predespacho", "Cat"])
    previa["CatBoost_semanal"] = previa["CatBoost_mensual_3s"] = previa["Cat"]
    for nom, vs in (("Pre+Cat semanal", ["Predespacho", "CatBoost_semanal"]), ("Pre+Cat mensual 3s", ["Predespacho", "CatBoost_mensual_3s"]),
                    ("Pers+Pre+Cat semanal", ["Persistencia", "Predespacho", "CatBoost_semanal"])):
        pan[nom] = er.causal_con_margen(pan, previa, vs, "g4")
        print(f"  {nom} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pan.to_csv(rc.MEJ / "frecuencia_predicciones.csv")
    ev = pan[pan.index >= "2026-01-15"].dropna()
    filas = []
    for c in ["ens7_realista", "Pre+Cat mensual", "Pre+Cat semanal", "Pre+Cat mensual 3s", "Pers+Pre+Cat semanal",
              "CatBoost_mensual", "CatBoost_semanal", "CatBoost_mensual_3s"]:
        f = {"variante": c, **rc.metricas(ev["real"], ev[c])}
        f["DM_t_vs_ens7"], f["DM_p"] = er.dm(ev["real"].to_numpy(), ev[c].to_numpy(), ev["ens7_realista"].to_numpy())
        filas.append(f)
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / "frecuencia_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    mes = ev.index.to_period("M")
    print(pd.DataFrame({c: (ev[c] - ev["real"]).abs().groupby(mes).mean() for c in t["variante"]}).round(1).to_string())
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
