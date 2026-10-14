# -*- coding: utf-8 -*-
"""
Validacion historica del escenario realista con el predespacho + CatBoost (2020-2025, mismos 5 periodos de
validacion_historica_predespacho.py: 15-jul a 30-jun de cada año).

Por periodo: CatBoost (perdida MAE, objetivo log(precio/predespacho), variables R+P+G de realista_comun, ventana
creciente desde 2019-02-15) reentrenado cada mes con lo publicado (hasta 73 h antes del corte), desde 60 dias antes del
periodo (calentamiento del combinador), y combinado con el predespacho en el combinador causal realista (objetivo MAE,
franjas de 6 h, margen de 4 dias, sin previa). Se compara con el predespacho solo y la persistencia realista (D-4).
Uso: realista_validacion_historica.py [sin_G]   (sin_G: sin las variables del despacho programado)
Salida: resultados/escenario_realista/mejoras/validacion_historica_realista.csv (o _sin_G.csv)
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
from realista_exp_votantes import INI_TR, MARGEN, catboost  # noqa: E402
from ensamble_con_predespacho_24h import imar, imar_con_respaldo  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    con_g = "sin_G" not in sys.argv          # G (despacho programado por tipo de recurso) choca con la limitacion OE1-3
    cols = conjuntos(d)["R+P"] + ([c for c in d.columns if c.startswith("prog_") or c == "n_termicas"] if con_g else [])
    pre = pd.Series(rc.alinear(d.index, imar_con_respaldo(imar(), d.index.max())), index=d.index)
    hist = pd.read_csv(rc.RAIZ / "data/processed/resultados/validacion_historica_predespacho.csv")
    filas = []
    for _, h in hist.iterrows():
        desde, hasta = pd.Timestamp(h["desde"]), pd.Timestamp(h["hasta"]) + pd.Timedelta(hours=24)
        partes = []
        for mes in pd.date_range((desde - pd.Timedelta(days=60)).replace(day=1), hasta, freq="MS"):
            tr = d.loc[INI_TR:mes - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
            te = d.loc[mes + pd.Timedelta(hours=1):min(mes + pd.offsets.MonthBegin(1), hasta)].dropna(subset=["pre_obj"])
            partes.append(catboost(tr, te, cols))
        cat = pd.concat(partes)
        p = pd.DataFrame({"real": d["precio_bolsa"], "Predespacho": pre, "CatBoost": cat, "Persistencia": d["precio_lag24h"]}).loc[cat.index].dropna()
        p["dia"], p["g4"] = (p.index - pd.Timedelta(hours=1)).normalize(), p.index.hour // 6
        vacia = p.iloc[:0]
        p["Pre+Cat"] = causal(p, vacia, ["Predespacho", "CatBoost"], "g4", objetivo="mae")
        ev = p.loc[desde + pd.Timedelta(hours=1):hasta].dropna()
        y = ev["real"].to_numpy()
        fila = {"origen": h["origen"], "desde": desde.date(), "hasta": (hasta - pd.Timedelta(hours=1)).date(), "horas": len(ev),
                "MAE_persistencia_D4": np.abs(y - ev["Persistencia"]).mean(), "MAE_predespacho": np.abs(y - ev["Predespacho"]).mean(),
                "MAE_catboost": np.abs(y - ev["CatBoost"]).mean(), "MAE_pre_cat": np.abs(y - ev["Pre+Cat"]).mean(),
                "MAE_ens7_optimista": h["MAE_ens7"], "MAE_ens6_optimista": h["MAE_ens6"]}
        fila["DM_t_pre_cat_vs_pre"], fila["DM_p_pre_cat_vs_pre"] = er.dm(y, ev["Pre+Cat"].to_numpy(), ev["Predespacho"].to_numpy())
        fila["cambio_vs_predespacho_%"] = (fila["MAE_pre_cat"] / fila["MAE_predespacho"] - 1) * 100
        filas.append(fila)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()}, f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / ("validacion_historica_realista.csv" if con_g else "validacion_historica_realista_sin_G.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
