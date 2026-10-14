# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: cruce de los mejores votantes (predespacho + CatBoost reentrenado) con las mejores
variantes del combinador (objetivo MAE, franjas de 6 h o 12 h). Usa frecuencia_predicciones.csv.
Salida: resultados/escenario_realista/mejoras/mezcla_predicciones.csv y mezcla_metricas.csv
"""
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
import escenario_realista_24h as er  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN, catboost  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    ARCH = sys.argv[1] if len(sys.argv) > 1 else "frecuencia_predicciones.csv"
    SAL = sys.argv[2] if len(sys.argv) > 2 else "mezcla"
    pan = pd.read_csv(rc.MEJ / ARCH, parse_dates=[0], index_col=0)
    pan["dia"] = pd.to_datetime(pan["dia"])
    ref = rc.referencia()
    for c in er.VOT6:
        if c not in pan:
            pan[c] = ref[c].reindex(pan.index)
    cats = [c for c in pan.columns if c.startswith("CatBoost_")]
    d = rc.tabla_realista().set_index("fecha_hora")
    from ensamble_24h_enero import origen5
    from ensamble_con_predespacho_24h import imar
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    tr = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    cat_prev = catboost(tr, d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"]), conjuntos(d)["R+P"]).reindex(previa.index)
    for c in cats:
        previa[c] = cat_prev
    previa = previa.dropna(subset=["Predespacho"] + cats)
    for p in (pan, previa):
        p["g4"], p["g2"] = p.index.hour // 6, p.index.hour // 12
    nuevos = []
    for cat in cats:
        for vs_nom, vs in (("Pre+", ["Predespacho", cat]), ("VOT6+Pre+", er.VOT6 + ["Predespacho", cat])):
            for obj in ("mae", "smape"):
                for g in ("g4", "g2"):
                    nom = f"{vs_nom}{cat.replace('CatBoost_', 'Cat ')} · {obj} · {g}"
                    pan[nom] = causal(pan, previa, vs, g, objetivo=obj)
                    nuevos.append(nom)
                    print(f"  {nom} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pan.to_csv(rc.MEJ / f"{SAL}_predicciones.csv")
    ev = pan[pan.index >= "2026-01-15"].dropna(subset=nuevos + ["ens7_realista"])
    filas = []
    for c in ["ens7_realista"] + nuevos + cats:
        f = {"variante": c, **rc.metricas(ev["real"], ev[c])}
        f["DM_t_vs_ens7"], f["DM_p"] = er.dm(ev["real"].to_numpy(), ev[c].to_numpy(), ev["ens7_realista"].to_numpy())
        filas.append(f)
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / f"{SAL}_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
