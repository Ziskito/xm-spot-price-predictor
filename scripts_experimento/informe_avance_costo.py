# -*- coding: utf-8 -*-
"""
Costo computacional de cada modelo (criterio de comparacion del Anexo 1, OE2 alcance 3).

Mide el tiempo de entrenamiento + prediccion del Origen 6 (train 2019-2025, prueba ene-ago 2026) en
este mismo equipo, con las mismas funciones que produjeron los pronosticos del proyecto. Las redes
neuronales no se re-entrenan aqui: su tiempo ya quedo medido el 2026-09-24 en logs_reentrenar_demanda.txt
(mismo equipo, misma configuracion). No escribe sobre ningun archivo del proyecto salvo su propia salida.
"""
import sys
import json
import time
import platform
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, particion_o6, columnas_features, RES


def cronometro(f, *a, repeticiones=1):
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        out = f(*a)
        tiempos.append(time.perf_counter() - t0)
    return out, float(np.median(tiempos))


def main():
    df = cargar_completo()
    cf = columnas_features(df)
    train, test = particion_o6(df)
    res = {"filas_train": int(len(train)), "filas_test": int(len(test)),
           "cpu": platform.processor(), "python": platform.python_version()}

    _, res["Persistencia_s"] = cronometro(lambda: test["precio_lag24h"].to_numpy().copy(), repeticiones=5)

    from recalibracion_2026 import xgboost, arx_garch

    def xgb_fp():
        m = xgboost(train, test, cf)
        return np.exp(m.predict(test[cf]))
    _, res["XGBoost_s"] = cronometro(xgb_fp, repeticiones=3)
    _, res["ARX+GARCH_s"] = cronometro(lambda: arx_garch(train, test), repeticiones=3)

    from reentrenar_tras_fix_demanda import garch_ged_origen6, cargar_completo as cc2
    df2 = cc2()
    _, res["GARCH-ged_s"] = cronometro(lambda: garch_ged_origen6(df2), repeticiones=1)

    from combinador_optimo_mape import qra_cv
    from stacking_24h_v2 import cargar as cargar24, BASE5
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred"].rename("GARCH-ged"), how="left").dropna()
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    _, res["Combinador_sMAPE_s"] = cronometro(lambda: qra_cv(d, BASE5 + ["GARCH-ged"], "smape", 42), repeticiones=1)

    res["N-BEATSx+N-HiTS_s"] = 6.8 * 60
    res["N-BEATSx+N-HiTS_fuente"] = "logs_reentrenar_demanda.txt (2026-09-24, 1000 pasos cada red, CPU)"
    out = RES / "informe_avance" / "costo_computacional.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
