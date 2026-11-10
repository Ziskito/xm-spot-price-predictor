# -*- coding: utf-8 -*-
"""
¿Hay una mejor forma de usar el ENSO que el ONI trimestral? (10-nov-2026, pedido de Juan)

Sobre el CatBoost R+P de produccion (perdida MAE, objetivo log(precio/predespacho), reentreno mensual con lo publicado
hasta 73 h antes del corte), se cambia la representacion del ENSO. Fuentes de NOAA CPC, cada una con su rezago de
publicacion (al corte de las 00:00 del dia D solo se usa lo ya publicado):
  * semanal  wksst9120.for: anomalia de Niño 3.4 y Niño 1+2 por semana (centrada en miercoles); se publica el lunes
             siguiente -> disponible si el centro de la semana es <= D - 7 dias
  * mensual  sstoi.indices: anomalia mensual (OISST) de Niño 3.4 y Niño 1+2; se publica los primeros dias del mes
             siguiente -> el mes M-1 desde el dia 6 del mes M, antes el M-2
Variantes:
  base       ONI actual
  sin_oni    sin ninguna variable del ENSO
  semanal    ONI reemplazado por Niño 3.4 semanal (ultima semana, media de 4 semanas, cambio en 12 semanas)
  mensual    ONI reemplazado por Niño 3.4 mensual (ultimo mes, media de 3 meses)
  tendencia  ONI + cambio de la anomalia mensual en 3 meses + media de 12 meses (sequia acumulada)
  costero    ONI + Niño 1+2 semanal (ultima semana y media de 4 semanas): la region frente a Ecuador y Peru
  fase       ONI + fase (El Niño 1, La Niña -1, neutral 0, con la media de 3 meses de la anomalia mensual y umbral
             0,5 °C) y meses consecutivos en esa fase
Evaluacion: 2026 (15-ene a 5-ago) y operacion (6-ago a 31-oct), DM diario (HAC) contra base.
Uso: python realista_exp_oni_variantes.py <variante>   (luego: python realista_exp_oni_variantes.py --evaluar)
Salida: resultados/escenario_realista/mejoras/oni_variantes/<variante>.csv y metricas.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_votantes import mensual  # noqa: E402

SAL = rc.MEJ / "oni_variantes"
CPC = "https://www.cpc.ncep.noaa.gov/data/indices/"
VARIANTES = ("base", "sin_oni", "semanal", "mensual", "tendencia", "costero", "fase")


def semanal_cpc():
    filas = []
    for l in requests.get(CPC + "wksst9120.for", timeout=60).text.splitlines():
        if len(l) < 49 or not l[1:3].isdigit():
            continue
        f = pd.Timestamp(l[1:10])
        filas.append((f, float(l[19:23]), float(l[45:49])))        # SSTA Niño 1+2, SSTA Niño 3.4
    s = pd.DataFrame(filas, columns=["semana", "n12", "n34"]).set_index("semana").sort_index()
    s.to_csv(rc.RAIZ / "data" / "processed" / "nino_semanal_cpc.csv")
    return s


def mensual_cpc():
    filas = []
    for l in requests.get(CPC + "sstoi.indices", timeout=60).text.splitlines()[1:]:
        p = l.split()
        if len(p) == 10:
            filas.append((pd.Timestamp(int(p[0]), int(p[1]), 1), float(p[3]), float(p[9])))
    s = pd.DataFrame(filas, columns=["mes", "n12", "n34"]).set_index("mes").sort_index()
    s.to_csv(rc.RAIZ / "data" / "processed" / "nino_mensual_cpc.csv")
    return s


def variables(idx):
    """Variables del ENSO publicadas al corte (00:00 del dia D) para cada hora objetivo."""
    dia = (pd.DatetimeIndex(idx) - pd.Timedelta(hours=1)).normalize()
    dias = pd.DatetimeIndex(sorted(set(dia)))
    w, m = semanal_cpc(), mensual_cpc()
    # semanal: ultima semana con centro <= D - 7 dias
    w4, w12 = w.rolling(4).mean(), w["n34"] - w["n34"].shift(12)
    pos = w.index.searchsorted(dias - pd.Timedelta(days=7), side="right") - 1
    S = pd.DataFrame({"n34_sem": w["n34"].to_numpy()[pos], "n34_sem_4": w4["n34"].to_numpy()[pos],
                      "n34_sem_cambio12": w12.to_numpy()[pos], "n12_sem": w["n12"].to_numpy()[pos],
                      "n12_sem_4": w4["n12"].to_numpy()[pos]}, index=dias)
    # mensual: M-1 desde el dia 6, si no M-2
    m3, m12 = m["n34"].rolling(3).mean(), m["n34"].rolling(12).mean()
    fase = np.sign(m3.where(m3.abs() >= 0.5, 0.0))
    racha = fase.groupby((fase != fase.shift()).cumsum()).cumcount() + 1
    mes = dias.to_period("M").to_timestamp()
    clave = [mm - pd.DateOffset(months=1 if d.day >= 6 else 2) for mm, d in zip(mes, dias)]
    M = pd.DataFrame({"n34_mes": m["n34"].reindex(clave).to_numpy(), "n34_mes_3": m3.reindex(clave).to_numpy(),
                      "n34_mes_cambio3": (m["n34"] - m["n34"].shift(3)).reindex(clave).to_numpy(),
                      "n34_mes_12": m12.reindex(clave).to_numpy(), "enso_fase": fase.reindex(clave).to_numpy(),
                      "enso_fase_meses": racha.reindex(clave).to_numpy()}, index=dias)
    return S.join(M).reindex(dia).set_axis(pd.DatetimeIndex(idx))


def columnas(variante, RP):
    sin = [c for c in RP if c != "oni"]
    return {"base": RP, "sin_oni": sin,
            "semanal": sin + ["n34_sem", "n34_sem_4", "n34_sem_cambio12"],
            "mensual": sin + ["n34_mes", "n34_mes_3"],
            "tendencia": RP + ["n34_mes_cambio3", "n34_mes_12"],
            "costero": RP + ["n12_sem", "n12_sem_4"],
            "fase": RP + ["enso_fase", "enso_fase_meses"]}[variante]


def dm(y, a, b, idx):
    d = pd.Series(np.abs(a - y) - np.abs(b - y), index=idx).groupby((idx - pd.Timedelta(hours=1)).normalize()).mean()
    r = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return float(r.tvalues[0]), float(r.pvalues[0])


def evaluar():
    pan = pd.concat({v: pd.read_csv(SAL / f"{v}.csv", parse_dates=[0], index_col=0)["pred"] for v in VARIANTES}, axis=1)
    pan["real"] = pd.read_csv(SAL / "base.csv", parse_dates=[0], index_col=0)["real"]
    ev = pan[pan.index >= "2026-01-15"].dropna()
    tramos = {"2026 (15-ene a 5-ago)": ev.index <= "2026-08-05 00:00", "operación (6-ago a 31-oct)": ev.index > "2026-08-05 00:00",
              "may-ago (El Niño)": (ev.index >= "2026-05-01") & (ev.index <= "2026-08-05 00:00")}
    filas = []
    for tr, sel in tramos.items():
        e = ev[sel]
        y = e["real"].to_numpy()
        for v in VARIANTES:
            t, p = (np.nan, np.nan) if v == "base" else dm(y, e[v].to_numpy(), e["base"].to_numpy(), e.index)
            filas.append({"tramo": tr, "variante": v, "horas": len(e), "MAE": float(np.abs(e[v] - e["real"]).mean()),
                          "RMSE": float(np.sqrt(((e[v] - e["real"]) ** 2).mean())), "DM_t_vs_base": t, "DM_p": p})
    t = pd.DataFrame(filas)
    t.to_csv(SAL / "metricas.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "--evaluar":
        evaluar()
        sys.exit()
    v = sys.argv[1]
    t0 = time.time()
    d = rc.tabla_realista(produccion=True).set_index("fecha_hora").sort_index()
    d = d.join(variables(d.index))
    cols = columnas(v, conjuntos(d)["R+P"])
    ini, fin = pd.Timestamp("2026-01-01 01:00"), pd.Timestamp("2026-10-31 23:00")
    idx = d.loc[ini:fin].dropna(subset=["precio_bolsa", "pre_obj"]).index
    out = pd.DataFrame({"real": d.loc[idx, "precio_bolsa"], "pred": mensual(d, cols, ini, fin).reindex(idx)})
    SAL.mkdir(parents=True, exist_ok=True)
    out.to_csv(SAL / f"{v}.csv")
    print(f"{v} ({len(cols)} variables): listo en {(time.time() - t0) / 60:.1f} min", flush=True)
