# -*- coding: utf-8 -*-
"""
Evaluacion de la curva de la CREG 101 127 activa solo desde la regla (10-nov-2026, pedido de Juan).

Variantes (realista_exp_regimen_2026.py, mismo diseño de produccion, reentreno mensual con lo publicado):
  base      produccion
  regla50   + curva de referencia de las ofertas hidraulicas (X = 50 % del volumen util) y su relacion con el
            predespacho, multiplicadas por el indicador de la medida: valen 0 antes del 10-sep-2026
  regla82   igual con X = 82 %
  medida    solo el indicador de la medida (0/1), sin la curva
  regla50w  regla50 con peso 5 en las horas con la medida activa (son pocas frente a siete años sin ella)
  corr_*    sin reentrenar: con la medida activa, el pronostico se corrige por la mediana del error en escala log de los
            ultimos 14 dias con la medida y precio publicado (hasta D-4); hace falta un minimo de 5 dias
  piso50/82 sin reentrenar: con la medida activa, el pronostico de base no baja de la curva de referencia (si la planta
            marginal es hidraulica, su oferta no puede estar por debajo de la curva)
Como el reentreno es mensual con 73 h de rezago, el modelo de septiembre no vio ningun dia con la regla: solo desde
octubre puede aprender de ella (el de octubre con ~18 dias, el de noviembre con ~7 semanas). Por eso se separa:
  ene-ago (control: no deberia cambiar), 10-30 sep (sin ejemplos de la regla), 1-oct en adelante (con ejemplos).
Salida: resultados/escenario_realista/regimen_2026/regla_metricas.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
import realista_exp_regimen_2026 as rg  # noqa: E402

SAL = rg.SAL
FIN = "2026-11-01 00:00"          # corte del informe: 31 de octubre


def corregir(p, activa, ventana=14, minimo=5, rezago=4):
    """Correccion causal del sesgo durante la medida: pred * exp(mediana de log(real/pred)) de dias publicados."""
    dia = (p.index - pd.Timedelta(hours=1)).normalize()
    r = pd.Series(np.log(p["real"] / p["pred"]).to_numpy(), index=p.index).where(activa)
    med_dia = r.groupby(dia).median()
    out = p["pred"].copy()
    for D in pd.DatetimeIndex(sorted(set(dia[activa]))):
        prev = med_dia[(med_dia.index <= D - pd.Timedelta(days=rezago))].dropna().iloc[-ventana:]
        if len(prev) >= minimo:
            sel = (dia == D) & activa
            out[sel] = out[sel] * np.exp(prev.median())
    return p.assign(pred=out)


def dm(y, a, b, idx):
    d = pd.Series(np.abs(a - y) - np.abs(b - y), index=idx).groupby((idx - pd.Timedelta(hours=1)).normalize()).mean()
    if (d == 0).all():
        return 0.0, 1.0
    r = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return float(r.tvalues[0]), float(r.pvalues[0])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    P = {v: pd.read_csv(SAL / f"{v}.csv", parse_dates=[0], index_col=0) for v in ("base", "regla50", "regla82", "medida", "regla50w")}
    base = P["base"]
    d = rc.tabla_realista(produccion=True).set_index("fecha_hora")
    for x in (0.50, 0.82):
        V = rg.variables_curva(d.loc[base.index.min():], x).reindex(base.index)
        activa = V["medida_101127"].to_numpy() == 1
        if x == 0.50:
            for v in ("base", "regla50"):
                P[f"corr_{v}"] = corregir(P[v], activa)
        P[f"piso{int(x * 100)}"] = base.assign(pred=np.where(activa, np.maximum(base["pred"], V["curva_ref"]), base["pred"]))
    tramos = {"ene-ago (control)": ("2026-01-15", "2026-08-31 23:00"), "10-30 sep (sin ejemplos)": ("2026-09-10 01:00", "2026-09-30 23:00"),
              "1-31 oct (con ejemplos)": ("2026-10-01 01:00", FIN), "10-sep a 31-oct": ("2026-09-10 01:00", FIN)}
    filas = []
    for nom, (a, b) in tramos.items():
        sel = base.loc[a:b].dropna(subset=["real"]).index
        y = base.loc[sel, "real"].to_numpy()
        for v, p in P.items():
            q = p.loc[sel, "pred"].to_numpy()
            t, pv = dm(y, q, base.loc[sel, "pred"].to_numpy(), sel) if v != "base" else (np.nan, np.nan)
            filas.append({"tramo": nom, "variante": v, "horas": len(sel), "hasta": str(sel.max()), "MAE": np.abs(q - y).mean(),
                          "sesgo": (q - y).mean(), "DM_t_vs_base": t, "DM_p": pv})
    t = pd.DataFrame(filas)
    t.to_csv(SAL / "regla_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(2).to_string(index=False))
