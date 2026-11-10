# -*- coding: utf-8 -*-
"""
Curva de la CREG 101 127 en el pronostico de 72 h (dias 2 y 3), 10-nov-2026, pedido de Juan antes de adoptarla.

Mismo diseño de produccion (pronostico_realista_72h.py): para el dia D+k (k = 1, 2) cada CatBoost usa la fila del dia
del corte D, la referencia es el predespacho de D, combinador QRA causal con margen de 5 + k dias y banda conforme.
Variantes (las de realista_exp_regimen_2026.py): base, regla50 (embalse agregado, X = 50 %) y planta (planta por planta,
X = NEP, tercer punto en el NPV). Las variables valen 0 antes del 10-sep-2026, asi que los modelos de los meses
anteriores a octubre son identicos al de produccion (columna constante): solo se reentrenan los cortes desde el
1-oct-2026; la historia del combinador y de la banda (cortes anteriores) es la publicada en produccion
(resultados/produccion/votantes_72h.csv) y la previa del combinador es la de produccion.
Uso: python realista_exp_regla_72h.py <variante>   ->  resultados/escenario_realista/regimen_2026/72h_<variante>.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import pronostico_realista_72h as p72  # noqa: E402
import realista_comun as rc  # noqa: E402
import realista_exp_errorpre as rep  # noqa: E402
import realista_exp_forma as rf  # noqa: E402
import realista_exp_regimen_2026 as rg  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402
from ensamble_con_predespacho_24h import imar  # noqa: E402

DESDE = pd.Timestamp("2026-10-01")


def extra(d, variante):
    if variante == "regla50":
        V = rg.variables_curva(d, 0.50)
        V["curva_ref"] *= V["medida_101127"]
        V["curva_log_vs_pre"] *= V["medida_101127"]
        return V
    if variante == "planta":
        return rg.variables_planta(d)
    if variante == "ambas":
        return extra(d, "regla50").join(rg.variables_planta(d))
    return pd.DataFrame(index=d.index)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    var = sys.argv[1]
    t0 = time.time()
    d = rc.tabla_realista(produccion=True).set_index("fecha_hora")
    F = rf.forma(d); d = d.join(F)
    C = rf.cm_programado(d); d = d.join(C)
    E = rep.errores(d); d = d.join(E)
    V = extra(d, var); d = d.join(V)
    RP = conjuntos(d)["R+P"] + list(V.columns)
    sets = {"Cat_RP": RP, "Cat_RPFC": RP + list(F.columns) + list(C.columns), "Cat_RPE": RP + list(E.columns)}
    corte_fin = imar().index.max().normalize()
    vot = pd.read_csv(p72.VOTANTES, parse_dates=["fecha_hora", "cutoff"], float_precision="round_trip").set_index("fecha_hora")
    salidas = []
    for k in (1, 2):
        X = p72.tabla_k(d, k, sets)
        cols = {n: [c for c in s if c in X.columns] for n, s in sets.items()}
        hist = vot[(vot["k"] == k) & (vot["cutoff"] < DESDE)][list(cols) + ["Cat_prom"]]
        sel = (X["corte"] >= DESDE) & (X["corte"] <= corte_fin) & X["ref"].notna()
        nuevos = X.index[sel]
        p = pd.DataFrame(index=nuevos)
        for n, c in cols.items():
            p[n] = p72.mensual(X, c, DESDE, corte_fin).reindex(nuevos)
            print(f"  {var} k={k} {n} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        p["Cat_prom"] = p[list(cols)].mean(axis=1)
        p = pd.concat([hist, p]).sort_index()
        p["real"], p["ref"] = X["precio"].reindex(p.index), X["ref"].reindex(p.index)
        previa = pd.read_csv(p72.PROD / f"previa_72h_k{k}.csv", parse_dates=["fecha_hora"], float_precision="round_trip").set_index("fecha_hora")
        for q in (p, previa):
            q["dia"] = (q.index - pd.Timedelta(hours=1)).normalize()
            q["g4"] = q.index.hour // 6
        p = p.dropna(subset=["ref", "Cat_prom"])
        p["q50"] = pd.Series(causal(p, previa, ["ref", "Cat_prom"], "g4", margen=5 + k, objetivo="mae"), index=p.index).fillna(p["ref"])
        p["q10"], p["q90"], p["margen"] = p72.bandas(p, k)
        p["k"] = k
        salidas.append(p)
    out = pd.concat(salidas)
    out.index.name = "fecha_hora"
    out[["k", "real", "ref", "Cat_prom", "q50", "q10", "q90", "margen"]].to_csv(rg.SAL / f"72h_{var}.csv")
    ok = out[(out.index >= "2026-10-02") & out["real"].notna()]
    print(f"{var}: MAE 72h (k=1,2) desde oct {np.abs(ok['q50'] - ok['real']).mean():.2f} | listo en {(time.time() - t0) / 60:.1f} min", flush=True)
