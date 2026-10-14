# -*- coding: utf-8 -*-
"""
Variante D del experimento de amplitud (realista_exp_amplitud.py): encoger el pico del pronostico SOLO cuando el detector
de eventos realista no lo respalda.

Para cada hora: dev = pronostico - media del pronostico del dia. Si dev > 0 se usa P(pico) de la hora; si dev < 0,
P(desplome). Factor k = k0 + (1 - k0) * min(1, P / p_ref): con probabilidad alta (>= p_ref) el cambio se conserva
entero; con probabilidad cero se reduce a k0.
(k0, p_ref) se aprenden sobre la marcha con las ventanas ya publicadas de 2026 (hasta d-5), grilla
k0 en {0.3, 0.5, 0.7, 0.85, 1.0} y p_ref en {0.1, 0.2, 0.3, 0.5}; los primeros 21 dias sin cambio.
Las probabilidades son las del detector realista en produccion (probabilidades_eventos_24h_realista_2026.csv).
Salida: resultados/escenario_realista/mejoras/amplitud_eventos_metricas.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
import escenario_realista_24h as er  # noqa: E402
from realista_exp_amplitud import SENALADOS, ramp_metrics  # noqa: E402

if __name__ == "__main__":
    R = rc.RAIZ / "data/processed/resultados"
    e = pd.read_csv(R / "ensamble_24h_realista_2026.csv", parse_dates=["fecha_hora"])
    e["dia"] = pd.to_datetime(e["dia"])
    e = e[e.groupby("dia")["real"].transform("size") == 24].reset_index(drop=True)
    pr = pd.read_csv(R / "probabilidades_eventos_24h_realista_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    e["p_pico"] = pr["p_pico"].reindex(e["fecha_hora"]).fillna(0).to_numpy()
    e["p_desplome"] = pr["p_desplome"].reindex(e["fecha_hora"]).fillna(0).to_numpy()
    base = e["pred_realista"]
    m = base.groupby(e["dia"]).transform("mean")
    dev = base - m
    P = np.where(dev > 0, e["p_pico"], e["p_desplome"])
    combos = [(k0, pref) for k0 in (0.3, 0.5, 0.7, 0.85, 1.0) for pref in (0.1, 0.2, 0.3, 0.5)]
    pred = {}
    for k0, pref in combos:
        k = k0 + (1 - k0) * np.minimum(1, P / pref)
        pred[(k0, pref)] = m + k * dev
    err = pd.DataFrame({c: np.abs(pred[c] - e["real"]).groupby(e["dia"]).sum() for c in combos})
    dias = np.sort(e["dia"].unique())
    elegido = {}
    for i, dd in enumerate(dias):
        conocidos = dias[:max(0, i - 4)]
        elegido[dd] = err.loc[conocidos].sum().idxmin() if len(conocidos) >= 21 else (1.0, 0.3)
    e["D_eventos"] = [pred[elegido[dd]][i] for i, dd in enumerate(e["dia"])]
    sel = pd.Series(elegido)
    print("parametros elegidos (k0, p_ref) por mes:", sel.groupby(pd.DatetimeIndex(sel.index).month).agg(lambda x: x.mode().iloc[0]).to_dict())
    # referencia optimista (NO causal): el mejor par fijo mirando 2026 completo, solo para ver el techo
    mejor = min(combos, key=lambda c: err[c].sum())
    e["D_techo_no_causal"] = pred[mejor]
    sen = e["dia"].isin([pd.Timestamp(2026, mm, x) for mm, xs in SENALADOS.items() for x in xs])
    volatil = e.groupby("dia")["real"].transform(lambda x: x.max() - x.min()) > 300
    ev = e["dia"] >= "2026-01-15"
    filas = []
    for c in ("pred_realista", "D_eventos", "D_techo_no_causal"):
        q = e[ev]
        mr, fr = ramp_metrics(q, c)
        f = {"variante": c, "MAE": np.abs(q[c] - q["real"]).mean(), "MAE_43_dias": np.abs(q[c] - q["real"])[sen[ev]].mean(),
             "MAE_dias_volatiles": np.abs(q[c] - q["real"])[volatil[ev]].mean(), "MAE_rampas": mr, "rampa_anticipada": fr}
        f["DM_t"], f["DM_p"] = er.dm(q["real"].to_numpy(), q[c].to_numpy(), q["pred_realista"].to_numpy())
        filas.append(f)
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / "amplitud_eventos_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"techo no causal: k0={mejor[0]}, p_ref={mejor[1]}")
    print(t.round(3).to_string(index=False))
