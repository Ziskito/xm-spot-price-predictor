# -*- coding: utf-8 -*-
"""
Verificacion de rigor del escenario realista: margen del combinador de 4 a 5 dias.

Al corte de D (00:00) el precio publicado llega hasta las 23:00 de D-4, pero la ventana de D-4 termina a las 00:00
de D-3. Con margen = 4 el combinador entrena con esa ultima hora, que aun no es publica (1 de 24 horas del dia mas
reciente). Aqui se recalculan el ensamble 7 realista y el predespacho + CatBoost mensual con margen = 5.
Salida: resultados/escenario_realista/mejoras/margen5_metricas.csv
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
    m = pd.read_csv(rc.MEJ / "mezcla_predicciones.csv", parse_dates=[0], index_col=0)
    m["dia"] = pd.to_datetime(m["dia"])
    ref = rc.referencia()
    for c in er.VOT6:
        if c not in m:
            m[c] = ref[c].reindex(m.index)
    d = rc.tabla_realista().set_index("fecha_hora")
    from ensamble_24h_enero import origen5
    from ensamble_con_predespacho_24h import imar
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    trp = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    previa["CatBoost_mensual"] = catboost(trp, d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"]), conjuntos(d)["R+P"]).reindex(previa.index)
    previa = previa.dropna(subset=["Predespacho", "CatBoost_mensual"])
    for p in (m, previa):
        p["g4"] = p.index.hour // 6
    m["ens7 realista · margen 5"] = er.causal_con_margen(m, previa, er.VOT6 + ["Predespacho"], "g4", margen=5)
    print(f"  ens7 ({(time.time() - t0) / 60:.1f} min)", flush=True)
    m["Pre+Cat mensual · mae · margen 5"] = causal(m, previa, ["Predespacho", "CatBoost_mensual"], "g4", margen=5, objetivo="mae")
    ev = m[m.index >= "2026-01-15"].dropna(subset=["ens7 realista · margen 5", "Pre+Cat mensual · mae · margen 5", "ens7_realista"])
    filas = [{"variante": c, **rc.metricas(ev["real"], ev[c])} for c in
             ["ens7_realista", "ens7 realista · margen 5", "Pre+Cat mensual · mae · g4", "Pre+Cat mensual · mae · margen 5"]]
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / "margen5_metricas.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
