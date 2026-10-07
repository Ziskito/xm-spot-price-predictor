# -*- coding: utf-8 -*-
"""
Validacion fuera de muestra del predespacho ideal de XM en los origenes historicos 1-5 (2020-2025).

En cada origen (ventanas de 24 h de 00:00 a 23:00 con corte a las 23:00 del dia anterior) se combinan con el
combinador causal de la version desplegable (QRA por franja de 6 h, objetivo sMAPE, solo dias anteriores,
14 dias de calentamiento; ensamble_origenes_historicos.py):
  * ensamble de 6: Persistencia, XGBoost, ARX+GARCH, N-BEATSx, N-HiTS, GARCH-ged
  * ensamble de 7: los 6 + el costo marginal del predespacho ideal del mismo dia (solo archivos creados y
    modificados antes de las 23:00 del dia anterior; si falta, se usa el ensamble de 6)
y se comparan con Diebold-Mariano (varianza HAC), en total y en las horas de rampa.
El GARCH-ged de cada origen se guarda en garch_ged_origen{n}_24h.csv para no recalcularlo.
Salida: data/processed/resultados/validacion_historica_predespacho.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import RES, cargar_completo  # noqa: E402
from franja_medianoche_24h import causal  # noqa: E402
from ensamble_con_predespacho_24h import dm, imar  # noqa: E402

VOT6 = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS", "GARCH-ged"]
ORIGENES = ["Origen 1", "Origen 2", "Origen 3", "Origen 4", "Origen 5"]


def garch(o, d, df_completo):
    ruta = RES / f"garch_ged_origen{o[-1]}_24h.csv"
    if ruta.exists():
        return pd.read_csv(ruta, parse_dates=["fecha_hora"]).set_index("fecha_hora")["GARCH-ged"]
    from ensamble_origenes_historicos import garch_ged
    t0 = time.time()
    g, fallos = garch_ged(df_completo(), d.index.min(), d.index.max())
    g.rename("GARCH-ged").to_frame().to_csv(ruta)
    print(f"  GARCH-ged {o}: {(time.time() - t0) / 60:.1f} min, {fallos} pasos con respaldo LASSO", flush=True)
    return g


def rampas(d, col):
    dr = d.groupby("dia")["real"].diff(3)
    dp = d.groupby("dia")[col].diff(3)
    top = (dr.abs() >= dr.abs().quantile(.9)) & dr.notna()
    return (np.abs(d["real"] - d[col])[top].mean(), (dp[top] / dr[top]).median(), top)


if __name__ == "__main__":
    t0 = time.time()
    crudas = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    cm = imar()
    cache = {}
    df_completo = lambda: cache.setdefault("df", cargar_completo().reset_index(drop=True))
    filas = []
    for o in ORIGENES:
        c = crudas[crudas["origen"] == o]
        d = c.pivot_table(index="fecha_hora", columns="modelo", values="prediccion").join(c.groupby("fecha_hora")["real"].first())
        d = d.join(garch(o, d, df_completo).rename("GARCH-ged"), how="left").dropna(subset=VOT6 + ["real"])
        d["dia"], d["g4"] = d.index.normalize(), d.index.hour // 6
        d["Predespacho"] = cm.reindex(d.index).to_numpy()
        d["ens6"] = causal(d, VOT6, "g4", "smape")
        hay = d["Predespacho"].notna()
        d["ens7"] = d["ens6"]
        d.loc[hay, "ens7"] = causal(d[hay], VOT6 + ["Predespacho"], "g4", "smape")
        d = d.dropna(subset=["ens6", "ens7"])
        y = d["real"].to_numpy()
        t, p = dm(y, d["ens6"].to_numpy(), d["ens7"].to_numpy())
        r6, f6, top = rampas(d, "ens6")
        r7, f7, _ = rampas(d, "ens7")
        tr, pr = dm(y[top.to_numpy()], d["ens6"].to_numpy()[top.to_numpy()], d["ens7"].to_numpy()[top.to_numpy()])
        per = d["Persistencia"].to_numpy()
        fila = {"origen": o, "desde": d.index.min().date(), "hasta": d.index.max().date(), "horas": len(d),
                "horas_con_predespacho_%": hay.reindex(d.index).mean() * 100,
                "MAE_ens6": np.abs(y - d.ens6).mean(), "MAE_ens7": np.abs(y - d.ens7).mean(),
                "MAE_predespacho_solo": np.abs(y - d["Predespacho"].fillna(d.ens6)).mean(),
                "MAPE_ens6_%": np.mean(np.abs(y - d.ens6) / y) * 100, "MAPE_ens7_%": np.mean(np.abs(y - d.ens7) / y) * 100,
                "rMAE_ens6": np.abs(y - d.ens6).mean() / np.abs(y - per).mean(),
                "rMAE_ens7": np.abs(y - d.ens7).mean() / np.abs(y - per).mean(),
                "DM_t": t, "DM_p": p, "MAE_rampa_ens6": r6, "MAE_rampa_ens7": r7, "DM_t_rampas": tr, "DM_p_rampas": pr,
                "rampa_fraccion_ens6": f6, "rampa_fraccion_ens7": f7}
        fila["cambio_MAE_%"] = (fila["MAE_ens7"] - fila["MAE_ens6"]) / fila["MAE_ens6"] * 100
        filas.append(fila)
        print(f"{o}: MAE {fila['MAE_ens6']:.2f} -> {fila['MAE_ens7']:.2f} ({fila['cambio_MAE_%']:+.1f} %), DM t={t:.2f} p={p:.4f} | "
              f"rampas {r6:.1f} -> {r7:.1f} | {(time.time() - t0) / 60:.1f} min", flush=True)
    tb = pd.DataFrame(filas)
    tb.to_csv(RES / "validacion_historica_predespacho.csv", index=False)
    pd.set_option("display.width", 250)
    print(tb.round(3).to_string(index=False))
