# -*- coding: utf-8 -*-
"""
Experimento del escenario realista (14-oct-2026): disponibilidad declarada de las ofertas iniciales (OFEI) como variable.

Los precios de oferta NO se publican a tiempo (descargar_disponibilidad_ofei.py: XM los agrega al mes siguiente);
la disponibilidad declarada si (D-1, ~08:10). Variables A, del dia de la ventana, todas del sistema:
  disp_total     disponibilidad declarada total de la hora (MW)
  margen         disponibilidad - demanda programada del despacho (dDEM), MW
  margen_rel     disponibilidad / demanda programada - 1
  margen_min_dia margen relativo minimo del dia; disp_rel_dia: disponibilidad media del dia frente a la de hace 7 dias
Se agregan a cada uno de los tres CatBoost del modelo realista (R+P, R+P+F+C, R+P+E) y se compara el combinado con el
predespacho (mismo esquema de pronostico_realista_24h.py) con y sin A, en las mismas horas.
Salida: resultados/escenario_realista/mejoras/disponibilidad_metricas.csv
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
import realista_exp_errorpre as rep  # noqa: E402
import realista_exp_forma as rf  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN, catboost, mensual  # noqa: E402
from ensamble_con_predespacho_24h import imar, imar_con_respaldo  # noqa: E402

INI, FIN = pd.Timestamp("2026-01-01 01:00"), pd.Timestamp("2026-08-05 00:00")


def disponibilidad(d):
    t = pd.read_csv(rc.RAIZ / "data/processed/disponibilidad_ofei_xm.csv", parse_dates=["fecha_hora"])
    t = t[t["a_tiempo"] & (t["unidades"] > 50)].set_index("fecha_hora")["disp_total_mw"]
    idx = pd.DatetimeIndex(d.index)
    s = t.reindex(pd.date_range(t.index.min(), idx.max(), freq="h"))
    for k in (1, 2):
        s = s.fillna(t.reindex(s.index - pd.Timedelta(days=k)).set_axis(s.index))
    out = pd.DataFrame(index=d.index)
    out["disp_total"] = rc.alinear(idx, s)
    out["margen"] = out["disp_total"] - d["demanda_programada"]
    out["margen_rel"] = out["disp_total"] / d["demanda_programada"] - 1
    dia = (idx - pd.Timedelta(hours=1)).normalize()
    out["margen_min_dia"] = pd.Series(out["margen_rel"].to_numpy(), index=idx).groupby(dia).transform("min").to_numpy()
    md = pd.Series(out["disp_total"].to_numpy(), index=idx).groupby(dia).mean()
    out["disp_rel_dia"] = (md / md.shift(7, freq="D").reindex(md.index)).reindex(dia).to_numpy()
    return out


if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    F = rf.forma(d)
    d = d.join(F)
    C = rf.cm_programado(d)
    d = d.join(C)
    E = rep.errores(d)
    d = d.join(E)
    A = disponibilidad(d)
    d = d.join(A)
    print("cobertura de A en 2019-2026:", round(A.loc["2019-02-15":].notna().all(axis=1).mean() * 100, 1), "%", flush=True)
    RP = conjuntos(d)["R+P"]
    base = {"RP": RP, "RPFC": RP + list(F.columns) + list(C.columns), "RPE": RP + list(E.columns)}
    idx = d.loc[INI:FIN].dropna(subset=["precio_bolsa"]).index
    pan = pd.DataFrame(index=idx)
    pan["real"] = d.loc[idx, "precio_bolsa"]
    pan["Predespacho"] = rc.alinear(idx, imar_con_respaldo(imar(), idx.max()))
    from ensamble_24h_enero import origen5
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    trp = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    tep = d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"])
    for sufijo, extra in (("", []), ("+A", list(A.columns))):
        cats = []
        for k, cols in base.items():
            nom = f"Cat {k}{sufijo}"
            pan[nom] = mensual(d, cols + extra, INI, FIN).reindex(idx)
            print(f"  {nom}: MAE {np.abs(pan[nom] - pan['real']).mean():.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            cats.append(nom)
        pan[f"prom{sufijo}"] = pan[cats].mean(axis=1)
        previa[f"prom{sufijo}"] = np.mean([catboost(trp, tep, cols + extra).reindex(previa.index) for cols in base.values()], axis=0)
    previa = previa.dropna(subset=["Predespacho", "prom", "prom+A"])
    pan["dia"] = (pan.index - pd.Timedelta(hours=1)).normalize()
    for p in (pan, previa):
        p["g4"] = p.index.hour // 6
    for sufijo in ("", "+A"):
        pan[f"Pre+prom{sufijo}"] = causal(pan, previa, ["Predespacho", f"prom{sufijo}"], "g4", objetivo="mae")
    pan.to_csv(rc.MEJ / "disponibilidad_predicciones.csv")
    ev = pan[pan.index >= "2026-01-15"].dropna()
    filas = []
    for c in ["Pre+prom", "Pre+prom+A", "prom", "prom+A"] + [c for c in pan.columns if c.startswith("Cat ")]:
        fl = {"variante": c, **rc.metricas(ev["real"], ev[c])}
        fl["DM_t_vs_Pre+prom"], fl["DM_p"] = er.dm(ev["real"].to_numpy(), ev[c].to_numpy(), ev["Pre+prom"].to_numpy())
        filas.append(fl)
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / "disponibilidad_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    mes = ev.index.to_period("M")
    print(pd.DataFrame({c: (ev[c] - ev["real"]).abs().groupby(mes).mean() for c in ("Pre+prom", "Pre+prom+A")}).round(1).to_string())
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
