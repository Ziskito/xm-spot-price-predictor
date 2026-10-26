# -*- coding: utf-8 -*-
"""
Validacion de todo 2026 (periodo oficial + operacion) para el informe final, con el mismo metodo de
informe_final_resultados.py: pronostico de 24 h publicado (q50 y banda), predespacho disponible al corte y prueba de
Diebold-Mariano con varianza HAC. Corta donde corta la operacion del informe (cifras.json) para que las cifras del
mismo parrafo coincidan. Pedido de Juan, 26-oct-2026.
Salida: scripts_experimento/informe_final_assets/cifras_todo2026.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "data" / "processed" / "resultados"
OUT = AQUI / "informe_final_assets"


def dm(y, a, b, lags=24):
    """t > 0: a tiene MENOR error absoluto que b (misma convencion de informe_final_resultados.py)."""
    d = np.abs(np.asarray(b) - np.asarray(y)) - np.abs(np.asarray(a) - np.asarray(y))
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": lags - 1})
    return float(r.tvalues[0]), float(r.pvalues[0] / 2 if r.tvalues[0] > 0 else 1 - r.pvalues[0] / 2)


fin = pd.Timestamp(json.loads((OUT / "cifras.json").read_text(encoding="utf-8"))["operacion"]["hasta"])
c24 = pd.read_csv(RES / "pronostico_realista_bandas_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
pre = pd.read_csv(RES / "ensamble_24h_realista_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")["Predespacho"]
q = c24.loc[:fin].dropna(subset=["real"])
p = pre.reindex(q.index)
R = {"desde": str(q.index.min()), "hasta": str(q.index.max()), "horas": int(len(q)),
     "MAE": float((q.q50 - q.real).abs().mean()), "MAPE": float(((q.q50 - q.real).abs() / q.real).mean() * 100),
     "MAE_pre": float((p - q.real).abs().mean()),
     "cobertura": float(((q.real >= q.q10) & (q.real <= q.q90)).mean() * 100),
     "DM_vs_pre": dm(q.real, q.q50, p)}
(OUT / "cifras_todo2026.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
print(R)
