# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: CatBoost con el predespacho por dentro como votante del combinador.

CatBoost (perdida MAE, objetivo log(precio/predespacho), variables R+P de realista_comun) en dos versiones:
  fijo      entrenado hasta 2025-12-31 (como realista_exp_arboles.py)
  mensual   reentrenado al inicio de cada mes con todo lo publicado (hasta 4 dias antes)
Para la previa del combinador (origen 5, 2024-07 a 2025-06) se entrena hasta 2024-06-26 y se predice ese año.
Combinaciones probadas con er.causal_con_margen (margen de 4 dias, franjas de 6 h).
Salida: resultados/escenario_realista/mejoras/votantes_predicciones.csv, votantes_metricas.csv, votantes_sesgo_mes.csv
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

INI_TR = "2019-02-15"
MARGEN = pd.Timedelta(hours=er.OCULTO)


def catboost(tr, te, cols, semilla=42):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.05, loss_function="MAE", random_seed=semilla, verbose=0, thread_count=4)
    m.fit(tr[cols], np.log(tr["precio_bolsa"]) - np.log(tr["pre_obj"].clip(lower=1)))
    return pd.Series(np.exp(m.predict(te[cols]) + np.log(te["pre_obj"].clip(lower=1))), index=te.index)


def mensual(d, cols, ini, fin):
    partes = []
    for mes in pd.date_range(pd.Timestamp(ini).normalize().replace(day=1), fin, freq="MS"):
        # la ventana del dia D (01:00 D a 00:00 D+1) usa lo publicado hasta las 00:00 de D menos 73 h
        corte = mes - MARGEN
        tr = d.loc[INI_TR:corte].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
        te = d.loc[mes + pd.Timedelta(hours=1):min(mes + pd.offsets.MonthBegin(1), pd.Timestamp(fin))].dropna(subset=["pre_obj"])
        partes.append(catboost(tr, te, cols))
        print(f"    mes {mes:%Y-%m}: {len(tr)} h de entrenamiento", flush=True)
    return pd.concat(partes)


if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    cols = conjuntos(d)["R+P"]
    ref = rc.referencia()
    from ensamble_24h_enero import origen5
    from ensamble_con_predespacho_24h import imar
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    previa = previa.dropna(subset=["Predespacho"])
    # CatBoost en la previa (entrenado antes de la previa)
    tr = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    te = d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"])
    previa["CatBoost"] = catboost(tr, te, cols).reindex(previa.index)
    previa = previa.dropna(subset=["CatBoost"])
    print(f"  CatBoost previa ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # 2026
    pan = ref.dropna(subset=er.VOT6 + ["real", "Predespacho"]).copy()
    pan["dia"], pan["g4"] = (pan.index - pd.Timedelta(hours=1)).normalize(), pan.index.hour // 6
    tr = d.loc[INI_TR:"2025-12-31 23:00"].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    pan["CatBoost_fijo"] = catboost(tr, d.loc[pan.index], cols).reindex(pan.index)
    print(f"  CatBoost fijo ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pan["CatBoost_mensual"] = mensual(d, cols, pan.index.min(), pan.index.max()).reindex(pan.index)
    print(f"  CatBoost mensual ({(time.time() - t0) / 60:.1f} min)", flush=True)

    combos = {
        "VOT6+Pre (referencia)": er.VOT6 + ["Predespacho"],
        "VOT6+Pre+Cat fijo": er.VOT6 + ["Predespacho", "CatBoost_fijo"],
        "VOT6+Pre+Cat mensual": er.VOT6 + ["Predespacho", "CatBoost_mensual"],
        "VOT6+Cat mensual": er.VOT6 + ["CatBoost_mensual"],
        "Pers+Pre+Cat mensual+N-HiTS": ["Persistencia", "Predespacho", "CatBoost_mensual", "N-HiTS"],
        "Pre+Cat mensual": ["Predespacho", "CatBoost_mensual"],
    }
    for nom, vs in combos.items():
        pv = previa.rename(columns={"CatBoost": "CatBoost_fijo"}).assign(CatBoost_mensual=previa["CatBoost"])
        pan[nom] = er.causal_con_margen(pan, pv, vs, "g4")
        print(f"  combinador {nom} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pan.to_csv(rc.MEJ / "votantes_predicciones.csv")

    ev = pan[pan.index >= "2026-01-15"].dropna(subset=list(combos) + ["CatBoost_mensual"])
    filas = []
    for c in list(combos) + ["CatBoost_fijo", "CatBoost_mensual", "Predespacho", "ens7_realista"]:
        f = {"variante": c, **rc.metricas(ev["real"], ev[c])}
        if c != "ens7_realista":
            f["DM_t_vs_ens7"], f["DM_p"] = er.dm(ev["real"].to_numpy(), ev[c].to_numpy(), ev["ens7_realista"].to_numpy())
        filas.append(f)
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / "votantes_metricas.csv", index=False)
    mes = ev.index.to_period("M")
    sesgo = pd.DataFrame({c: (ev[c] - ev["real"]).groupby(mes).mean() for c in ["ens7_realista"] + list(combos)[1:] + ["CatBoost_mensual"]})
    mae = pd.DataFrame({c: (ev[c] - ev["real"]).abs().groupby(mes).mean() for c in ["ens7_realista"] + list(combos)[1:] + ["CatBoost_mensual"]})
    pd.concat({"sesgo": sesgo, "MAE": mae}, axis=1).to_csv(rc.MEJ / "votantes_sesgo_mes.csv")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print("\nMAE por mes\n", mae.round(1).to_string())
    print("\nSesgo por mes\n", sesgo.round(1).to_string())
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
