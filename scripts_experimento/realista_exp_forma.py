# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: forma intradia del predespacho como variables del CatBoost.

Variables F (todas del predespacho del dia de la ventana, publicado D-1):
  pre_d1 / pre_d1_sig   cambio frente a la hora anterior / siguiente (relativo a la media del dia)
  pre_rango_hora        posicion de la hora en el dia (0 = la mas barata, 1 = la mas cara)
  pre_z                 (pre - media del dia) / desviacion del dia
  pre_max_hora / pre_min_hora   hora (0-23) del maximo y del minimo del dia
CatBoost mensual (realista_exp_votantes.mensual) con R+P y con R+P+F; cada uno combinado con el predespacho
(combinador causal, objetivo MAE, franjas de 6 h). Uso: realista_exp_forma.py [extra] donde extra puede ser
"cm_prog" para agregar ademas el costo marginal del despacho programado y la demanda programada (dMAR/dDEM).
Salida: resultados/escenario_realista/mejoras/forma_predicciones.csv y forma_metricas.csv (o *_cmprog)
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
from ensamble_con_predespacho_24h import imar, imar_con_respaldo  # noqa: E402


def forma(d):
    idx = pd.DatetimeIndex(d.index)
    cm = imar_con_respaldo(imar(), idx.max())
    dia = (idx - pd.Timedelta(hours=1)).normalize()
    H = np.stack([cm.reindex(dia + pd.Timedelta(hours=k)).to_numpy() for k in range(24)], axis=1)   # 24 h del dia
    h = np.where(idx.normalize() == dia, idx.hour, 23)          # hora del dia usada (la 00:00 de D+1 usa las 23)
    r = np.arange(len(d))
    med, sd = np.nanmean(H, 1), np.nanstd(H, 1)
    prev = H[r, np.clip(h - 1, 0, 23)]
    sig = H[r, np.clip(h + 1, 0, 23)]
    act = H[r, h]
    F = pd.DataFrame(index=d.index)
    F["pre_d1"] = (act - prev) / med
    F["pre_d1_sig"] = (sig - act) / med
    F["pre_rango_hora"] = (np.argsort(np.argsort(np.nan_to_num(H, nan=-1), 1), 1)[r, h]) / 23.0
    F["pre_z"] = (act - med) / np.where(sd > 0, sd, np.nan)
    F["pre_max_hora"] = np.nanargmax(np.nan_to_num(H, nan=-np.inf), 1)
    F["pre_min_hora"] = np.nanargmin(np.nan_to_num(H, nan=np.inf), 1)
    return F


def cm_programado(d):
    t = pd.read_csv(rc.RAIZ / "data/processed/despacho_programado_cm_demanda_xm.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    idx = pd.DatetimeIndex(d.index)
    out = pd.DataFrame(index=d.index)
    for col, ok in (("cm_programado", "a_tiempo_mar"), ("demanda_programada", "a_tiempo_dem")):
        s = t.loc[t[ok], col]
        s = s.reindex(pd.date_range(s.index.min(), idx.max(), freq="h"))
        for k in (1, 2):
            s = s.fillna(t.loc[t[ok], col].reindex(s.index - pd.Timedelta(days=k)).set_axis(s.index))
        out[col] = rc.alinear(idx, s)
    dia = (idx - pd.Timedelta(hours=1)).normalize()
    out["cm_prog_rel_pre"] = out["cm_programado"] / d["pre_obj"].clip(lower=1)
    out["cm_prog_media"] = pd.Series(out["cm_programado"].to_numpy(), index=idx).groupby(dia).transform("mean").to_numpy()
    dem_dia = pd.Series(out["demanda_programada"].to_numpy(), index=idx).groupby(dia).transform("mean").to_numpy()
    out["dem_prog_rel"] = out["demanda_programada"] / dem_dia
    out["dem_prog_dia"] = dem_dia
    return out


if __name__ == "__main__":
    t0 = time.time()
    extra = "cm_prog" in sys.argv
    d = rc.tabla_realista().set_index("fecha_hora")
    F = forma(d)
    d = d.join(F)
    RP = conjuntos(d)["R+P"]
    sets = {"R+P": RP, "R+P+F": RP + list(F.columns)}
    if extra:
        C = cm_programado(d)
        d = d.join(C)
        sets = {"R+P+C": RP + list(C.columns), "R+P+F+C": RP + list(F.columns) + list(C.columns)}
    m = pd.read_csv(rc.MEJ / "mezcla_predicciones.csv", parse_dates=[0], index_col=0)
    pan = m[["real", "dia", "Predespacho", "ens7_realista", "CatBoost_mensual", "Pre+Cat mensual · mae · g4"]].copy()
    pan["dia"] = pd.to_datetime(pan["dia"])
    from ensamble_24h_enero import origen5
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    trp = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    tep = d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"])
    nuevos = []
    for nom, cols in sets.items():
        c = f"Cat {nom}"
        pan[c] = mensual(d, cols, pan.index.min(), pan.index.max()).reindex(pan.index)
        previa[c] = catboost(trp, tep, cols).reindex(previa.index)
        print(f"  {c}: MAE {np.abs(pan[c] - pan['real']).mean():.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        nuevos.append(c)
    if extra:
        # dias sin dMAR a tiempo: el predespacho ideal de la misma hora (el combinador no admite vacios)
        pan["CM programado"] = d["cm_programado"].reindex(pan.index).fillna(pan["Predespacho"])
        previa["CM programado"] = d["cm_programado"].reindex(previa.index).fillna(previa["Predespacho"])
    previa = previa.dropna(subset=["Predespacho"] + nuevos + (["CM programado"] if extra else []))
    for p in (pan, previa):
        p["g4"] = p.index.hour // 6
    combos = {f"Pre+{c} · mae": ["Predespacho", c] for c in nuevos}
    if extra:
        combos.update({f"Pre+CMprog+{c} · mae": ["Predespacho", "CM programado", c] for c in nuevos})
        combos["Pre+CMprog · mae"] = ["Predespacho", "CM programado"]
    for nom, vs in combos.items():
        pan[nom] = causal(pan, previa, vs, "g4", objetivo="mae")
        print(f"  {nom} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    suf = "_cmprog" if extra else ""
    pan.to_csv(rc.MEJ / f"forma_predicciones{suf}.csv")
    ev = pan[pan.index >= "2026-01-15"].dropna()
    filas = []
    for c in ["ens7_realista", "Pre+Cat mensual · mae · g4", "CatBoost_mensual"] + nuevos + list(combos) + (["CM programado"] if extra else []):
        fl = {"variante": c, **rc.metricas(ev["real"], ev[c])}
        fl["DM_t_vs_PreCat"], fl["DM_p"] = er.dm(ev["real"].to_numpy(), ev[c].to_numpy(), ev["Pre+Cat mensual · mae · g4"].to_numpy())
        filas.append(fl)
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / f"forma_metricas{suf}.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    mes = ev.index.to_period("M")
    print(pd.DataFrame({c: (ev[c] - ev["real"]).abs().groupby(mes).mean() for c in t["variante"][:6]}).round(1).to_string())
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
