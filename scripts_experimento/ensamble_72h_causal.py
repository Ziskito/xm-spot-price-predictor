# -*- coding: utf-8 -*-
"""
Ensamble de 72h: pesos CAUSALES (desplegables) y meta-modelo LAD.

El ensamble v2 de 72h (ensamble_72h_v2.py) fija los pesos NNLS con validacion cruzada por dias:
evaluacion fuera de muestra valida, pero usa dias FUTUROS para ponderar dias pasados, cosa que en
operacion real no existe. Aqui se recalculan los pesos usando SOLO los cortes anteriores a cada
corte (ventana expansiva y ventanas moviles), que es exactamente como se operaria el sistema.

Causalidad: los cortes estan espaciados 72h, asi que todos los objetivos del corte i-1 (hasta
cutoff_{i-1}+72h = cutoff_i) ya son observados en el corte i. Entrenar con los cortes < i es honesto.

Tambien se compara NNLS contra LAD (minima desviacion absoluta, pesos >= 0 que suman 1), que
optimiza la misma metrica con la que se evalua el proyecto.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from ensamble_72h_v2 import cargar, TRAMOS, RES
from stacking_24h_v2 import pesos_nnls, pesos_lad

MIN_CORTES = 5  # 15 dias de calentamiento


def dm(real, p_ref, p_nuevo):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 71})
    return r.tvalues[0], r.pvalues[0]


def cv_dias(sub, cols, ajustar):
    dias = np.sort(sub["dia"].unique())
    pliegues = np.array_split(np.random.default_rng(42).permutation(dias), 5)
    pred = pd.Series(np.nan, index=sub.index)
    for dt in pliegues:
        m = sub["dia"].isin(dt)
        w = ajustar(sub.loc[~m, cols].values, sub.loc[~m, "real"].values)
        pred[m] = sub.loc[m, cols].values @ w
    return pred


def causal(sub, cols, ajustar, ventana=None):
    cortes = np.sort(sub["cutoff"].unique())
    pred = pd.Series(np.nan, index=sub.index)
    for i in range(MIN_CORTES, len(cortes)):
        ini = 0 if ventana is None else max(0, i - ventana)
        tr = sub["cutoff"].isin(cortes[ini:i])
        te = sub["cutoff"] == cortes[i]
        w = ajustar(sub.loc[tr, cols].values, sub.loc[tr, "real"].values)
        pred[te] = sub.loc[te, cols].values @ w
    return pred


def main():
    df, modelos, _ = cargar()
    df["dia"] = df["fecha_hora"].dt.normalize()
    filas = []
    for nombre, ini, fin in TRAMOS:
        sub = df[(df.paso_horas >= ini) & (df.paso_horas <= fin)].reset_index(drop=True)
        v = {
            "NNLS-CV (v2 actual)": cv_dias(sub, modelos, pesos_nnls),
            "LAD-CV": cv_dias(sub, modelos, pesos_lad),
            "causal NNLS expansiva": causal(sub, modelos, pesos_nnls),
            "causal NNLS movil 10 cortes (30d)": causal(sub, modelos, pesos_nnls, 10),
            "causal NNLS movil 20 cortes (60d)": causal(sub, modelos, pesos_nnls, 20),
            "causal LAD expansiva": causal(sub, modelos, pesos_lad),
            "causal LAD movil 20 cortes (60d)": causal(sub, modelos, pesos_lad, 20),
        }
        comun = pd.concat(v, axis=1).dropna().index
        real = sub.loc[comun, "real"].values
        base = sub.loc[comun, "N-BEATSx"].values
        print(f"\n--- Tramo {nombre} ({len(comun)} filas comunes tras calentamiento) ---")
        print(f"  {'N-BEATSx (linea base)':36s} MAE={np.abs(real-base).mean():6.2f}")
        for k, p in v.items():
            pk = p[comun].values
            mae = np.abs(real - pk).mean()
            t, pv = dm(real, base, pk)
            marca = "GANA (sig.)" if pv < 0.05 and t > 0 else ("PIERDE (sig.)" if pv < 0.05 else "sin dif.")
            print(f"  {k:36s} MAE={mae:6.2f}  vs N-BEATSx: {100*(np.abs(real-base).mean()-mae)/np.abs(real-base).mean():+5.1f}%  p={pv:.4f} {marca}")
            filas.append({"tramo": nombre, "variante": k, "mae": mae,
                          "mae_nbeatsx": np.abs(real - base).mean(), "dm_p_vs_nbeatsx": pv})
    pd.DataFrame(filas).to_csv(RES / "ensamble_72h_causal_metricas.csv", index=False)
    print("\nGuardado: ensamble_72h_causal_metricas.csv")


if __name__ == "__main__":
    main()
