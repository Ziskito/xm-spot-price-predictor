# -*- coding: utf-8 -*-
"""
Experimento (19-oct-2026): adaptar el pronostico de 24 h al cambio de reglas de la Resolucion CREG 101 127 de 2026
(medidas transitorias ante El Nino 2026-2027, activadas de forma preventiva el 9-sep-2026). Desde esa fecha el MAE de
augur y del predespacho se duplico (33 -> 66 COP/kWh) y la banda bajo a 62 % de cobertura.

Variantes del CatBoost (mismo diseno de produccion: 3 conjuntos R+P, R+P+F+C, R+P+E; perdida MAE; objetivo
log(precio/predespacho); reentreno mensual con lo publicado; combinador causal con el predespacho; previa 2024-07 a
2025-06 con la misma variante):
  base       el de produccion
  rec365     pesos por antiguedad, vida media de 365 dias (lo reciente pesa mas)
  rec120     vida media de 120 dias
  curva50    + precio de referencia de la curva de ofertas hidraulicas de la CREG 101 127 (X = 50 % del volumen util)
             + su relacion con el predespacho + indicador de la medida activa
  curva82    igual con X = 82 % (el umbral ISO-GT de fin de agosto que simulo la CREG)
La curva (Documento CREG 901 431 de 2026): PES x 1,10 en el minimo tecnico (0 % del volumen util), PES + 0,01 en X %
y los otros costos variables (OCV) en el nivel de vertimiento (100 %). Supuestos: se aplica al embalse agregado del SIN
(no por planta), el PES historico es el precio marginal de escasez de XM (PrecEscaMarg; igual al PES desde 2025) y los
OCV se aproximan con el CERE del mes anterior. El X % real lo fija la CREG por circular y no esta publicado.
Uso: python realista_exp_regimen_2026.py <variante>   (escribe resultados/escenario_realista/regimen_2026/<variante>.csv)
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
import realista_exp_errorpre as rep  # noqa: E402
import realista_exp_forma as rf  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN  # noqa: E402
from ensamble_con_predespacho_24h import imar, imar_con_respaldo  # noqa: E402

SAL = rc.MEJ.parent / "regimen_2026"
INI = pd.Timestamp("2026-01-01 01:00")
ACTIVACION = pd.Timestamp("2026-09-10")      # anunciada el 9-sep; al corte de las 00:00 del 10 ya era publica
VARIANTES = {"base": {}, "rec365": {"vida": 365}, "rec120": {"vida": 120},
             "curva50": {"curva": 0.50}, "curva82": {"curva": 0.82}}


def catboost(tr, te, cols, vida=None, semilla=42):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.05, loss_function="MAE", random_seed=semilla, verbose=0,
                          thread_count=4)
    peso = None
    if vida:
        edad = (tr.index.max() - tr.index) / pd.Timedelta(days=1)
        peso = 0.5 ** (np.asarray(edad, float) / vida)
    m.fit(tr[cols], np.log(tr["precio_bolsa"]) - np.log(tr["pre_obj"].clip(lower=1)), sample_weight=peso)
    return pd.Series(np.exp(m.predict(te[cols]) + np.log(te["pre_obj"].clip(lower=1))), index=te.index)


def mensual(d, cols, ini, fin, vida=None):
    partes = []
    for mes in pd.date_range(pd.Timestamp(ini).normalize().replace(day=1), fin, freq="MS"):
        corte = mes - MARGEN
        tr = d.loc[INI_TR:corte].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
        te = d.loc[mes + pd.Timedelta(hours=1):min(mes + pd.offsets.MonthBegin(1), pd.Timestamp(fin))].dropna(subset=["pre_obj"])
        if len(te):
            partes.append(catboost(tr, te, cols, vida))
    return pd.concat(partes)


def variables_curva(d, x):
    """Precio de referencia de la curva de ofertas hidraulicas (CREG 101 127 de 2026) con el embalse agregado."""
    esc = pd.read_csv(rc.RAIZ / "data" / "raw" / "escasez_capacidad_xm.csv", parse_dates=[0], index_col=0)
    idx = pd.DatetimeIndex(d.index)
    dia = idx.normalize()
    cap = esc["CapaUtilDiarEner"].ffill().reindex(dia, method="ffill").to_numpy()
    pct = np.clip(d["volumen_embalses"].to_numpy() / cap, 0, 1)          # volumen_embalses ya viene con su rezago (D-2)
    pes = esc["PrecEscaMarg"].ffill().reindex(dia, method="ffill").to_numpy() / 1000 if esc["PrecEscaMarg"].max() > 1e4 \
        else esc["PrecEscaMarg"].ffill().reindex(dia, method="ffill").to_numpy()
    ocv = cere_mes_anterior(dia)
    p1, p2 = 1.10 * pes, pes + 0.01
    curva = np.where(pct <= x, p1 + (p2 - p1) * pct / x, p2 + (ocv - p2) * (pct - x) / (1 - x))
    out = pd.DataFrame(index=d.index)
    out["curva_ref"] = curva
    out["curva_log_vs_pre"] = np.log(curva / d["pre_obj"].clip(lower=1).to_numpy())
    out["medida_101127"] = (dia >= ACTIVACION).astype(int)
    return out


def cere_mes_anterior(dia):
    """OCV aproximados con el CERE (API de XM, mensual) del mes anterior; respaldo: 80 COP/kWh."""
    ruta = rc.RAIZ / "data" / "raw" / "cere_xm.csv"
    if not ruta.exists():
        try:
            from pydataxm.pydataxm import ReadDB
            import datetime as dt
            r = ReadDB()
            partes = [r.request_data("CERE", "Sistema", dt.date(a, 1, 1), dt.date(a + 1, 12, 31)) for a in (2018, 2020, 2022, 2024, 2026)]
            x = pd.concat([p for p in partes if p is not None])
            s = x.set_index(pd.to_datetime(x["Date"]))["Value"].sort_index()
            s[~s.index.duplicated()].to_csv(ruta)
        except Exception as e:
            print("  CERE no disponible:", str(e)[:80])
            return np.full(len(dia), 80.0)
    s = pd.read_csv(ruta, parse_dates=[0], index_col=0).iloc[:, 0]
    s = s / 1000 if s.max() > 1e4 else s
    mes = (pd.DatetimeIndex(dia).to_period("M") - 1).to_timestamp()
    return s.resample("MS").mean().reindex(pd.date_range(s.index.min(), mes.max(), freq="MS")).ffill().reindex(mes).fillna(80).to_numpy()


def previa(d, sets, vida=None):
    from ensamble_24h_enero import origen5
    pv = origen5()
    pv["Predespacho"] = imar().reindex(pv.index).to_numpy()
    trp = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    tep = d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"])
    pv["Cat_prom"] = np.mean([catboost(trp, tep, cols, vida).reindex(pv.index) for cols in sets.values()], axis=0)
    return pv.dropna(subset=["Predespacho", "Cat_prom"])[["real", "Predespacho", "Cat_prom", "dia"]]


def main(nombre):
    t0 = time.time()
    conf = VARIANTES[nombre]
    d = rc.tabla_realista(produccion=True).set_index("fecha_hora")
    for f in (rf.forma, rf.cm_programado, rep.errores):
        d = d.join(f(d))
    RP = conjuntos(d)["R+P"]
    F = [c for c in rf.forma(d).columns]
    C = [c for c in rf.cm_programado(d).columns]
    E = [c for c in rep.errores(d).columns]
    extra = []
    if "curva" in conf:
        V = variables_curva(d, conf["curva"])
        d = d.join(V)
        extra = list(V.columns)
    sets = {"Cat_RP": RP + extra, "Cat_RPFC": RP + F + C + extra, "Cat_RPE": RP + E + extra}
    fin = imar().index.max().normalize() + pd.Timedelta(days=1)
    idx = d.loc[INI:fin].dropna(subset=["pre_obj"]).index
    pan = pd.DataFrame(index=idx)
    pan["Predespacho"] = rc.alinear(idx, imar_con_respaldo(imar(), idx.max()))
    for k, cols in sets.items():
        pan[k] = mensual(d, cols, idx.min(), fin, conf.get("vida")).reindex(idx)
        print(f"  {nombre} {k} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pan["Cat_prom"] = pan[list(sets)].mean(axis=1)
    pan["real"] = d.loc[idx, "precio_bolsa"]
    pan["dia"] = (pan.index - pd.Timedelta(hours=1)).normalize()
    pv = previa(d, sets, conf.get("vida"))
    for p in (pan, pv):
        p["g4"] = p.index.hour // 6
    pan["pred"] = causal(pan, pv, ["Predespacho", "Cat_prom"], "g4", objetivo="mae")
    pan["pred"] = pan["pred"].fillna(pan["Predespacho"])
    SAL.mkdir(parents=True, exist_ok=True)
    pan.index.name = "fecha_hora"
    pan.to_csv(SAL / f"{nombre}.csv")
    ok = pan.dropna(subset=["real"]).loc["2026-01-15":]
    print(f"{nombre}: MAE {np.abs(ok['pred'] - ok['real']).mean():.2f} | listo en {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
