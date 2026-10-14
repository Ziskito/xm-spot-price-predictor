# -*- coding: utf-8 -*-
"""
Reentrenamiento de los modelos de 24 h de 2026 (Origen 6) tras la actualizacion de datos del 14-oct-2026
(actualizar_datos_oct2026.py: ONI de NOAA vigente y revision de XM del 1-5 de agosto).

Mismos modelos e hiperparametros que producen los numeros del proyecto, con funciones que se verificaron
reproduciendo exactamente las predicciones guardadas con los datos anteriores (diferencias < 1e-12):
  Persistencia   precio_lag24h
  XGBoost        reentrenar_tras_fix_demanda.xgboost_origen6
  ARX+GARCH      escenario_realista_24h.arx_garch (misma configuracion del proyecto)
  N-BEATSx/HiTS  reentrenar_tras_fix_demanda.neuronales_origen6
  GARCH-ged      reentrenar_tras_fix_demanda.garch_ged_origen6
Reemplaza las filas del Origen 6 en walkforward_predicciones_crudas.csv y pronostico_GARCH-ged_24h_2026.csv
(copias previas en resultados/backups_pre_actualizacion_14oct2026/).
"""
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import escenario_realista_24h as er  # noqa: E402
import reentrenar_tras_fix_demanda as rt  # noqa: E402
from o6_comun import RES, cargar_completo  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    bk = RES / "backups_pre_actualizacion_14oct2026"
    bk.mkdir(parents=True, exist_ok=True)
    for f in ("walkforward_predicciones_crudas.csv", "pronostico_GARCH-ged_24h_2026.csv"):
        if not (bk / f).exists():
            shutil.copy2(RES / f, bk / f)
    df = cargar_completo()
    crudas = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    o6 = crudas[crudas["origen"] == "Origen 6"]
    ref = o6.pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
    real = df.set_index("fecha_hora")["precio_bolsa"]
    nuevos = []
    idx = ref.index
    pers = df.set_index("fecha_hora")["precio_lag24h"].reindex(idx)
    nuevos.append(pd.DataFrame({"origen": "Origen 6", "fecha_hora": idx, "real": real.reindex(idx).to_numpy(), "modelo": "Persistencia", "prediccion": pers.to_numpy()}))
    nuevos.append(rt.xgboost_origen6(df))
    print(f"  XGBoost ({(time.time() - t0) / 60:.1f} min)", flush=True)
    a = er.arx_garch(df).reindex(idx)
    nuevos.append(pd.DataFrame({"origen": "Origen 6", "fecha_hora": idx, "real": real.reindex(idx).to_numpy(), "modelo": "ARX+GARCH", "prediccion": a.to_numpy()}))
    print(f"  ARX+GARCH ({(time.time() - t0) / 60:.1f} min)", flush=True)
    g = rt.garch_ged_origen6(df)
    print(f"  GARCH-ged ({(time.time() - t0) / 60:.1f} min)", flush=True)
    nuevos.append(rt.neuronales_origen6(df))
    print(f"  N-BEATSx y N-HiTS ({(time.time() - t0) / 60:.1f} min)", flush=True)
    nuevo = pd.concat(nuevos, ignore_index=True)
    nuevo["real"] = real.reindex(pd.DatetimeIndex(nuevo["fecha_hora"])).to_numpy()
    filas = []
    for m in ("Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"):
        s_n, s_v = nuevo[nuevo.modelo == m].set_index("fecha_hora"), o6[o6.modelo == m].set_index("fecha_hora")
        assert len(s_n) == len(s_v), (m, len(s_n), len(s_v))
        c = s_n.index.intersection(s_v.index)
        filas.append(f"  {m:12s} MAE antes {np.abs(s_v.loc[c, 'prediccion'] - s_v.loc[c, 'real']).mean():7.2f} -> ahora {np.abs(s_n.loc[c, 'prediccion'] - s_n.loc[c, 'real']).mean():7.2f}")
    gv = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"])
    g["real"] = real.reindex(pd.DatetimeIndex(g["fecha_hora"])).to_numpy()
    filas.append(f"  GARCH-ged    MAE antes {np.abs(gv['pred'] - gv['real']).mean():7.2f} -> ahora {np.abs(g['pred'] - g['real']).mean():7.2f}")
    crudas = pd.concat([crudas[crudas["origen"] != "Origen 6"], nuevo], ignore_index=True).sort_values(["origen", "modelo", "fecha_hora"])
    crudas.to_csv(RES / "walkforward_predicciones_crudas.csv", index=False)
    g.to_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", index=False)
    print("\n".join(filas))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
