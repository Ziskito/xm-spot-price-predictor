# -*- coding: utf-8 -*-
"""
¿Combinador o promedio de los CatBoost en los dias 2 y 3 del pronostico de 72 h? (10-nov-2026, pedido de Juan)

Vigente: q50 = QRA causal entre la referencia (predespacho del dia del corte) y el promedio de los tres CatBoost.
Alternativa: q50 = promedio de los tres CatBoost, sin combinador. En los dos casos la banda del 80 % es la conforme en
escala log por hora (pronostico_realista_72h.bandas, 60 dias, rezago de 5 + k dias), calculada sobre su propio q50.
Periodos: los cinco años historicos (realista_validacion_historica_72h.py), 2026 (15-ene a 5-ago) y la operacion
(6-ago a 31-oct), con los votantes de produccion (resultados/produccion/votantes_72h.csv, cortes diarios).
Medidas: MAE, DM diario (HAC; negativo = el promedio se equivoca menos), cobertura, ancho y puntaje de Winkler (alfa 0,2).
Salida: resultados/escenario_realista/mejoras/combinador_72h.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import pronostico_realista_72h as p72  # noqa: E402
import realista_comun as rc  # noqa: E402


def winkler(y, lo, hi, a=0.2):
    return float(np.mean((hi - lo) + 2 / a * (lo - y) * (y < lo) + 2 / a * (y - hi) * (y > hi)))


def dm(y, a, b, idx):
    d = pd.Series(np.abs(a - y) - np.abs(b - y), index=idx).groupby((idx - pd.Timedelta(hours=1)).normalize()).mean()
    r = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return float(r.tvalues[0]), float(r.pvalues[0])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    h = pd.read_csv(rc.MEJ / "validacion_historica_72h_predicciones.csv", parse_dates=[0], index_col=0)
    v = pd.read_csv(rc.RAIZ / "data/processed/resultados/produccion/votantes_72h.csv", parse_dates=["fecha_hora", "cutoff"]).set_index("fecha_hora")
    v = v[v["real"].notna() & (v.index <= "2026-10-31 23:00")]
    tramos = [(per, h[(h["periodo"] == per)]) for per in ("2020-21", "2021-22", "2022-23", "2023-24", "2024-25")]
    tramos += [("2026 (15-ene a 5-ago)", v[(v.index >= "2026-01-15") & (v.index <= "2026-08-05 00:00")]),
               ("operación (6-ago a 31-oct)", v[v.index > "2026-08-05 00:00"])]
    filas = []
    for per, x in tramos:
        for k in (1, 2):
            p = x[x["k"] == k].copy().sort_index()
            p["dia"] = (p.index - pd.Timedelta(hours=1)).normalize()
            # banda conforme sobre el promedio de los CatBoost (misma receta que la del combinador)
            alt = p[["real", "dia"]].assign(q50=p["Cat_prom"])
            lo_a, hi_a, _ = p72.bandas(alt, k)
            ev = p.index[p.index.isin(lo_a.index)]
            y = p.loc[ev, "real"].to_numpy()
            qc, qa = p.loc[ev, "q50"].to_numpy(), p.loc[ev, "Cat_prom"].to_numpy()
            lc, hc = p.loc[ev, "q10"].to_numpy(), p.loc[ev, "q90"].to_numpy()
            la, ha = lo_a.loc[ev].to_numpy(), hi_a.loc[ev].to_numpy()
            t, pv = dm(y, qa, qc, ev)
            filas.append({"periodo": per, "dia": k + 1, "horas": len(ev),
                          "MAE_combinador": np.abs(qc - y).mean(), "MAE_promedio": np.abs(qa - y).mean(), "DM_t": t, "DM_p": pv,
                          "cob_combinador": ((y >= lc) & (y <= hc)).mean() * 100, "cob_promedio": ((y >= la) & (y <= ha)).mean() * 100,
                          "ancho_combinador": (hc - lc).mean(), "ancho_promedio": (ha - la).mean(),
                          "winkler_combinador": winkler(y, lc, hc), "winkler_promedio": winkler(y, la, ha)})
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / "combinador_72h.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
