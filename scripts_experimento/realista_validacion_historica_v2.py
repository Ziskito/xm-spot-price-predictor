# -*- coding: utf-8 -*-
"""
Validacion historica (2020-2025) de la variante "predespacho + promedio de 3 CatBoost", elegida mirando 2026
(realista_exp_forma.py y realista_exp_errorpre.py), para medir si su ganancia se sostiene fuera de ese año.

Tres CatBoost mensuales (mismo esquema de realista_validacion_historica.py) con variables distintas:
  R+P        realistas + predespacho
  R+P+F+C    + forma intradia del predespacho + costo marginal y demanda del despacho programado (dMAR/dDEM)
  R+P+E      + error reciente del predespacho (ventanas ya publicadas, D-5)
Se promedian y se combinan con el predespacho (objetivo MAE, franjas de 6 h, margen de 5 dias, sin previa). Se compara
con la version simple (predespacho + CatBoost R+P) en las mismas horas.
Salida: resultados/escenario_realista/mejoras/validacion_historica_v2.csv
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
import realista_exp_errorpre as rep  # noqa: E402
import realista_exp_forma as rf  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN, catboost  # noqa: E402
from ensamble_con_predespacho_24h import imar, imar_con_respaldo  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    F, C = rf.forma(d), None
    d = d.join(F)
    C = rf.cm_programado(d)
    d = d.join(C)
    E = rep.errores(d)
    d = d.join(E)
    RP = conjuntos(d)["R+P"]
    sets = {"R+P": RP, "R+P+F+C": RP + list(F.columns) + list(C.columns), "R+P+E": RP + list(E.columns)}
    pre = pd.Series(rc.alinear(d.index, imar_con_respaldo(imar(), d.index.max())), index=d.index)
    hist = pd.read_csv(rc.RAIZ / "data/processed/resultados/validacion_historica_predespacho.csv")
    filas = []
    for _, h in hist.iterrows():
        desde, hasta = pd.Timestamp(h["desde"]), pd.Timestamp(h["hasta"]) + pd.Timedelta(hours=24)
        cats = {k: [] for k in sets}
        for mes in pd.date_range((desde - pd.Timedelta(days=60)).replace(day=1), hasta, freq="MS"):
            tr = d.loc[INI_TR:mes - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
            te = d.loc[mes + pd.Timedelta(hours=1):min(mes + pd.offsets.MonthBegin(1), hasta)].dropna(subset=["pre_obj"])
            for k, cols in sets.items():
                cats[k].append(catboost(tr, te, cols))
        p = pd.DataFrame({"real": d["precio_bolsa"], "Predespacho": pre, **{k: pd.concat(v) for k, v in cats.items()}})
        p = p.loc[pd.concat(cats["R+P"]).index].dropna()
        p["prom3"] = p[list(sets)].mean(axis=1)
        p["dia"], p["g4"] = (p.index - pd.Timedelta(hours=1)).normalize(), p.index.hour // 6
        vacia = p.iloc[:0]
        p["simple"] = causal(p, vacia, ["Predespacho", "R+P"], "g4", objetivo="mae")
        p["v2"] = causal(p, vacia, ["Predespacho", "prom3"], "g4", objetivo="mae")
        ev = p.loc[desde + pd.Timedelta(hours=1):hasta].dropna()
        y = ev["real"].to_numpy()
        fila = {"origen": h["origen"], "horas": len(ev), "MAE_predespacho": np.abs(y - ev["Predespacho"]).mean(),
                "MAE_simple": np.abs(y - ev["simple"]).mean(), "MAE_v2": np.abs(y - ev["v2"]).mean(),
                **{f"MAE_cat_{k}": np.abs(y - ev[k]).mean() for k in sets}, "MAE_cat_prom3": np.abs(y - ev["prom3"]).mean()}
        fila["DM_t_v2_vs_simple"], fila["DM_p"] = er.dm(y, ev["v2"].to_numpy(), ev["simple"].to_numpy())
        filas.append(fila)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()}, f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / "validacion_historica_v2.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
