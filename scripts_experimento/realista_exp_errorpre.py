# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: error reciente del predespacho como variables del CatBoost.

E: log(precio real / predespacho) ya publicado al corte (ventanas completas hasta D-5):
  err_hora_d4      misma hora, ventana de D-5 (la ultima completa; el nombre se conserva)
  err_dia_d4       media de la ventana de D-5
  err_7d, err_30d  media de los ultimos 7 y 30 dias publicados
  err_hora_7d      media de la misma hora en los ultimos 7 dias publicados
CatBoost mensual con R+P y R+P+E, cada uno combinado con el predespacho (objetivo MAE, franjas de 6 h).
Salida: resultados/escenario_realista/mejoras/errorpre_predicciones.csv y errorpre_metricas.csv
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
from realista_exp_votantes import INI_TR, MARGEN, catboost, mensual  # noqa: E402
from ensamble_con_predespacho_24h import imar  # noqa: E402


def errores(d):
    """d indexado por hora objetivo, con precio_bolsa, pre_obj y dia (dia de la ventana)."""
    e = np.log(d["precio_bolsa"].clip(lower=1)) - np.log(d["pre_obj"].clip(lower=1))
    dia = pd.DatetimeIndex(d["dia"])
    pos = d.index.hour.to_numpy()                                   # hora objetivo (la 00:00 es la ultima de la ventana)
    tab = pd.DataFrame({"e": e.to_numpy(), "dia": dia, "pos": pos})
    por_dia = tab.groupby("dia")["e"].mean()
    por_hora = tab.pivot_table(index="dia", columns="pos", values="e")
    # al corte de D (00:00) el precio llega hasta las 23:00 de D-4: la ventana de D-4 (01:00 D-4 a 00:00 D-3) aun no
    # esta completa, asi que la ultima ventana completa es la de D-5
    k = 5
    out = pd.DataFrame(index=d.index)
    out["err_dia_d4"] = por_dia.shift(k, freq="D").reindex(dia).to_numpy()
    out["err_7d"] = por_dia.rolling(7, min_periods=3).mean().shift(k, freq="D").reindex(dia).to_numpy()
    out["err_30d"] = por_dia.rolling(30, min_periods=10).mean().shift(k, freq="D").reindex(dia).to_numpy()
    ph = por_hora.shift(k, freq="D")
    ph7 = por_hora.rolling(7, min_periods=3).mean().shift(k, freq="D")
    ix = ph.index.get_indexer(dia)
    ix7 = ph7.index.get_indexer(dia)
    col = ph.columns.get_indexer(pos)
    out["err_hora_d4"] = np.where(ix >= 0, ph.to_numpy()[np.clip(ix, 0, None), col], np.nan)
    out["err_hora_7d"] = np.where(ix7 >= 0, ph7.to_numpy()[np.clip(ix7, 0, None), col], np.nan)
    return out


if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    E = errores(d)
    d = d.join(E)
    RP = conjuntos(d)["R+P"]
    m = pd.read_csv(rc.MEJ / "mezcla_predicciones.csv", parse_dates=[0], index_col=0)
    pan = m[["real", "dia", "Predespacho", "ens7_realista", "Pre+Cat mensual · mae · g4"]].copy()
    pan["dia"] = pd.to_datetime(pan["dia"])
    from ensamble_24h_enero import origen5
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    trp = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    tep = d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"])
    cols = RP + list(E.columns)
    pan["Cat R+P+E"] = mensual(d, cols, pan.index.min(), pan.index.max()).reindex(pan.index)
    previa["Cat R+P+E"] = catboost(trp, tep, cols).reindex(previa.index)
    previa = previa.dropna(subset=["Predespacho", "Cat R+P+E"])
    for p in (pan, previa):
        p["g4"] = p.index.hour // 6
    pan["Pre+Cat R+P+E · mae"] = causal(pan, previa, ["Predespacho", "Cat R+P+E"], "g4", objetivo="mae")
    pan.to_csv(rc.MEJ / "errorpre_predicciones.csv")
    ev = pan[pan.index >= "2026-01-15"].dropna()
    filas = []
    for c in ["ens7_realista", "Pre+Cat mensual · mae · g4", "Cat R+P+E", "Pre+Cat R+P+E · mae"]:
        fl = {"variante": c, **rc.metricas(ev["real"], ev[c])}
        fl["DM_t_vs_PreCat"], fl["DM_p"] = er.dm(ev["real"].to_numpy(), ev[c].to_numpy(), ev["Pre+Cat mensual · mae · g4"].to_numpy())
        filas.append(fl)
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / "errorpre_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
