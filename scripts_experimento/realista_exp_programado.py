# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: despacho programado de XM (dDEC, publicado D-1 ~14:30) en el CatBoost.

1) CatBoost fijo (entrenado hasta 2025-12-31) con R+P y con R+P+G (G = prog_*, n_termicas).
2) CatBoost semanal con R+P+G y su combinacion con el predespacho (objetivo MAE y sMAPE, franjas de 6 h).
Salida: resultados/escenario_realista/mejoras/programado_predicciones.csv y programado_metricas.csv
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
from realista_exp_frecuencia import reentrenado  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN, catboost  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    s = conjuntos(d)
    RP, RPG = s["R+P"], s["R+P"] + [c for c in d.columns if c.startswith("prog_") or c == "n_termicas"]
    f = pd.read_csv(rc.MEJ / "frecuencia_predicciones.csv", parse_dates=[0], index_col=0)
    pan = f[["real", "dia", "Predespacho", "ens7_realista", "CatBoost_semanal", "Pre+Cat semanal"]].copy()
    pan["dia"] = pd.to_datetime(pan["dia"])
    tr = d.loc[INI_TR:"2025-12-31 23:00"].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    for nom, cols in (("CatBoost_fijo R+P", RP), ("CatBoost_fijo R+P+G", RPG)):
        pan[nom] = catboost(tr, d.loc[pan.index], cols).reindex(pan.index)
        print(f"  {nom}: MAE {np.abs(pan[nom] - pan['real']).mean():.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    fin = pan.index.max()
    pan["CatBoost_semanal G"] = reentrenado(d, RPG, list(pd.date_range("2026-01-01", fin, freq="7D")), fin).reindex(pan.index)
    print(f"  semanal R+P+G: MAE {np.abs(pan['CatBoost_semanal G'] - pan['real']).mean():.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    from ensamble_24h_enero import origen5
    from ensamble_con_predespacho_24h import imar
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    trp = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    tep = d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"])
    previa["CatBoost_semanal"] = catboost(trp, tep, RP).reindex(previa.index)
    previa["CatBoost_semanal G"] = catboost(trp, tep, RPG).reindex(previa.index)
    previa = previa.dropna(subset=["Predespacho", "CatBoost_semanal", "CatBoost_semanal G"])
    for p in (pan, previa):
        p["g4"] = p.index.hour // 6
    nuevos = []
    for vs_nom, vs in (("Pre+Cat semanal", ["Predespacho", "CatBoost_semanal"]), ("Pre+Cat semanal G", ["Predespacho", "CatBoost_semanal G"]),
                       ("Pre+Cat semanal+Cat semanal G", ["Predespacho", "CatBoost_semanal", "CatBoost_semanal G"])):
        for obj in ("mae", "smape"):
            nom = f"{vs_nom} · {obj}"
            pan[nom] = causal(pan, previa, vs, "g4", objetivo=obj)
            nuevos.append(nom)
            print(f"  {nom} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pan.to_csv(rc.MEJ / "programado_predicciones.csv")
    ev = pan[pan.index >= "2026-01-15"].dropna()
    filas = []
    for c in ["ens7_realista", "CatBoost_fijo R+P", "CatBoost_fijo R+P+G", "CatBoost_semanal", "CatBoost_semanal G"] + nuevos:
        fl = {"variante": c, **rc.metricas(ev["real"], ev[c])}
        fl["DM_t_vs_ens7"], fl["DM_p"] = er.dm(ev["real"].to_numpy(), ev[c].to_numpy(), ev["ens7_realista"].to_numpy())
        filas.append(fl)
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / "programado_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    mes = ev.index.to_period("M")
    print(pd.DataFrame({c: (ev[c] - ev["real"]).mean() for c in []} or {c: (ev[c] - ev["real"]).groupby(mes).mean() for c in t["variante"][:6]}).round(1).to_string())
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
