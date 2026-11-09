# -*- coding: utf-8 -*-
"""
Comparacion de los pronosticos de 24 y 72 h (9-nov-2026, pedido de Juan). Para cada hora objetivo hay tres pronosticos:
el emitido el mismo dia (dia 1 = modelo de 24 h), el de la vispera (dia 2 = 72 h, pasos 25-48) y el de dos dias antes
(dia 3 = 72 h, pasos 49-72). Mismas horas para los tres, del 15-ene al 31-oct-2026 (corte del informe), separadas en 2026
(hasta el 5-ago) y operacion (6-ago a 31-oct).
Medidas: MAE, MAPE, sesgo, cobertura y ancho de la banda del 80 %, persistencia al mismo corte (t - 96 - 24k), DM de cada dia
frente al dia 1 (diferencia diaria, HAC), parecido de la curva del dia (correlacion y hora pico) y coincidencia de las
senales del motor (rodante 30 dias + filtro intradia + filtro de contrato MC).
Salidas: resultados/escenario_realista/mejoras/comparacion_horizontes.csv y comparacion_horizontes_resumen.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_motor_umbrales_alternativos as u  # noqa: E402

FIN = pd.Timestamp("2026-10-31 23:00")
OFICIAL = pd.Timestamp("2026-08-05 00:00")


def dm_diario(y, a, b, idx):
    d = pd.Series(np.abs(a - y) - np.abs(b - y), index=idx).groupby((idx - pd.Timedelta(hours=1)).normalize()).mean()
    r = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return float(r.tvalues[0]), float(r.pvalues[0])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    c = pd.read_csv(u.RES / "pronostico_realista_72h_diario_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    c = c[c["real"].notna() & (c["fecha_hora"] <= FIN) & (c["fecha_hora"] >= "2026-01-15")]
    w = c.pivot_table(index="fecha_hora", columns="k", values=["q50", "q10", "q90"]).dropna()
    real = c.groupby("fecha_hora")["real"].first().reindex(w.index)
    precio = u.precio_real()
    filas = []
    for nom, sel in (("2026", w.index <= OFICIAL), ("operacion", w.index > OFICIAL)):
        idx, y = w.index[sel], real[sel].to_numpy()
        for k in (0, 1, 2):
            q, lo, hi = (w[v][k][sel].to_numpy() for v in ("q50", "q10", "q90"))
            pers = precio.reindex(idx - pd.to_timedelta(np.where(idx.hour == 0, 120, 96) + 24 * k, unit="h")).to_numpy()
            t, p = dm_diario(y, q, w["q50"][0][sel].to_numpy(), idx) if k else (np.nan, np.nan)
            filas.append({"periodo": nom, "dia": k + 1, "horas": int(sel.sum()), "MAE": np.abs(q - y).mean(),
                          "MAPE": (np.abs(q - y) / y).mean() * 100, "sesgo": (q - y).mean(),
                          "cobertura": ((y >= lo) & (y <= hi)).mean() * 100, "ancho": (hi - lo).mean(),
                          "MAE_pers": np.nanmean(np.abs(pers - y)), "DM_t_vs_dia1": t, "DM_p": p})
    t = pd.DataFrame(filas)
    t.to_csv(u.MEJ / "comparacion_horizontes.csv", index=False)
    dia = (w.index - pd.Timedelta(hours=1)).normalize()
    res = {"desde": str(w.index.min()), "hasta": str(w.index.max()), "forma": {}, "senales": {}}
    for k in (1, 2):
        grupos = [g for _, g in w.groupby(dia) if len(g) == 24]
        res["forma"][f"dia{k + 1}"] = {"correlacion": float(np.mean([np.corrcoef(g["q50"][0], g["q50"][k])[0, 1] for g in grupos])),
                                       "misma_hora_pico": float(np.mean([g["q50"][0].idxmax() == g["q50"][k].idxmax() for g in grupos]) * 100)}
    mc = u.mc_disponible(w.index)
    for rol, fav in (("generador", "vender"), ("comercializador", "comprar")):
        S = {}
        for k in (0, 1, 2):
            df = pd.DataFrame({"fecha_hora": w.index, "real": real.to_numpy(), "q50": w["q50"][k].to_numpy()})
            b, a = u.rodante(df, precio[precio.index <= df["fecha_hora"].min() - pd.Timedelta(hours=73)], 30)
            s = u.senal(df, b, a, rol)
            no = (df["q50"] > mc) if rol == "comercializador" else (df["q50"] < mc)
            s[(s == fav) & no] = "esperar"
            S[k] = s.to_numpy()
        res["senales"][rol] = {f"dia{k + 1}": float(np.mean(S[k] == S[0]) * 100) for k in (1, 2)}
    (u.MEJ / "comparacion_horizontes_resumen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(t.round(2).to_string(index=False)); print(json.dumps(res, indent=1))
