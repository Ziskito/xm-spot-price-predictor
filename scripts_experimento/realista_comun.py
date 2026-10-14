# -*- coding: utf-8 -*-
"""
Base comun de los experimentos del escenario realista (solo datos publicos a las 00:00 del dia D), 7-oct-2026.

tabla_realista() junta, por hora objetivo t (ventanas 01:00 de D a 00:00 de D+1, corte a las 00:00 de D):
  * variables realistas de escenario_realista_24h.features_realistas() (precio y demanda con ~3 dias de rezago,
    embalses y aportes de D-2, ONI(M-2), calendario)
  * h_*: variables de precio con el hueco de 73 h lleno con el predespacho (escenario_realista_relleno_24h)
  * pre_*: predespacho ideal de la hora objetivo y del dia (publicado D-1; la hora 00:00 de D+1 usa las 23:00 de D;
    dias sin archivo valido: el del dia anterior)
  * delta_*: delta ex post (precio - maximo precio de oferta) conocido al corte (hasta D-4)
  * prog_*: despacho programado (dDEC, publicado D-1 ~14:30), si existe data/processed/despacho_programado_xm.csv
Se guarda en cache (resultados/escenario_realista/mejoras/tabla_realista.pkl).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import escenario_realista_24h as er  # noqa: E402
import escenario_realista_relleno_24h as rr  # noqa: E402
from ensamble_con_predespacho_24h import imar, imar_con_respaldo  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
MEJ = RAIZ / "data" / "processed" / "resultados" / "escenario_realista" / "mejoras"
CACHE = MEJ / "tabla_realista.pkl"


def alinear(idx, serie):
    """Valor disponible al corte para cada hora objetivo (la 00:00 de D+1 toma las 23:00 de D)."""
    ventana = (idx - pd.Timedelta(hours=1)).normalize()
    fuente = np.where(idx.normalize() == ventana, idx, ventana + pd.Timedelta(hours=23))
    return serie.reindex(pd.DatetimeIndex(fuente)).to_numpy()


def tabla_realista(forzar=False):
    if CACHE.exists() and not forzar:
        return pd.read_pickle(CACHE)
    MEJ.mkdir(parents=True, exist_ok=True)
    base = er.features_realistas()
    rel, _, _ = rr.features_relleno(base)
    d = base.copy()
    for c in ["precio_lag24h", "precio_media_24h", "precio_media_7d", "precio_media_30d", "precio_std_24h", "precio_std_7d",
              "precio_rango_24h", "ratio_volatilidad"]:
        d[f"h_{c}"] = rel[c].to_numpy()
    idx = pd.DatetimeIndex(d["fecha_hora"])
    cm = imar_con_respaldo(imar(), idx.max())
    d["pre_obj"] = alinear(idx, cm)
    dia = (idx - pd.Timedelta(hours=1)).normalize()
    horas = np.stack([cm.reindex(dia + pd.Timedelta(hours=k)).to_numpy() for k in range(24)], axis=1)
    d["pre_media"], d["pre_min"], d["pre_max"] = np.nanmean(horas, 1), np.nanmin(horas, 1), np.nanmax(horas, 1)
    d["pre_rel"] = d["pre_obj"] / d["pre_media"]
    d["pre_rango_rel"] = (d["pre_max"] - d["pre_min"]) / d["pre_media"]
    d["pre_lag24"] = cm.reindex(idx - pd.Timedelta(hours=24)).to_numpy()
    d["pre_cambio24"] = d["pre_obj"] / d["pre_lag24"]
    d["log_pre"] = np.log(d["pre_obj"].clip(lower=1))
    # delta ex post conocido al corte
    ruta = RAIZ / "data" / "raw" / "delta_expost_2018_2026.csv"
    if not ruta.exists():
        ruta = RAIZ / "data" / "raw" / "delta_expost_2024_2026.csv"
    dl = pd.read_csv(ruta, parse_dates=[0], index_col=0)["delta_expost"]
    ddia = dl.groupby(dl.index.normalize()).mean()
    d["delta_d4"] = ddia.shift(4, freq="D").reindex(dia).to_numpy()
    d["delta_7d"] = ddia.rolling(7, min_periods=3).mean().shift(4, freq="D").reindex(dia).to_numpy()
    d["delta_30d"] = ddia.rolling(30, min_periods=10).mean().shift(4, freq="D").reindex(dia).to_numpy()
    d["pre_mas_delta"] = d["pre_obj"] + d["delta_7d"].fillna(0)
    # despacho programado
    rp = RAIZ / "data" / "processed" / "despacho_programado_xm.csv"
    if rp.exists():
        pg = pd.read_csv(rp, parse_dates=["fecha_hora"])
        pg = pg[pg["a_tiempo"]].set_index("fecha_hora")
        for c in ("prog_total", "prog_hidraulica", "prog_termica", "prog_solar", "prog_eolica", "n_termicas"):
            s = pg[c].reindex(pd.date_range(pg.index.min(), idx.max(), freq="h"))
            for k in (1, 2):
                s = s.fillna(pg[c].reindex(s.index - pd.Timedelta(days=k)).set_axis(s.index))
            d[c] = alinear(idx, s)
        d["prog_termica_frac"] = d["prog_termica"] / d["prog_total"]
        d["prog_renov_frac"] = (d["prog_solar"] + d["prog_eolica"]) / d["prog_total"]
        tot_dia = pd.Series(d["prog_total"].to_numpy(), index=idx).groupby(dia).transform("mean").to_numpy()
        d["prog_total_rel"] = d["prog_total"] / tot_dia
        ter_dia = pd.Series(d["prog_termica"].to_numpy(), index=idx).groupby(dia).transform("mean").to_numpy()
        d["prog_termica_dia"] = ter_dia
    d["dia"] = dia
    d.to_pickle(CACHE)
    return d


def metricas(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "MAPE_%": np.mean(np.abs(e) / y) * 100,
            "sMAPE_%": np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p))) * 100, "sesgo": e.mean(),
            "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)}


def referencia():
    """Predicciones del escenario realista ya calculadas (ensambles y modelos)."""
    return pd.read_csv(er.OUT / "predicciones.csv", parse_dates=[0], index_col=0)
