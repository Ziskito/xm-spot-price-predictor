# -*- coding: utf-8 -*-
"""
Pesos del ensamble distintos por FRANJA HORARIA (madrugada / manana / tarde / noche).

Motivacion: el paper de Uniejewski (2026) aplica todos sus esquemas de seleccion y promedio de
forma independiente para cada hora del dia, porque la dificultad de pronosticar cambia mucho a lo
largo del dia (el precio colombiano tiene picos de demanda en la noche y el despacho hidraulico
opera distinto en la madrugada). Si N-BEATSx es mejor en unas franjas y la persistencia o el lineal
en otras, un solo vector de pesos para las 24 horas desperdicia esa informacion.

Se usan 4 franjas de 6 horas (no 24 horas sueltas) para no multiplicar por 24 los parametros del
meta-modelo con ~200 dias de datos. Mismo protocolo que los ensambles: validacion cruzada de 5
pliegues por dia, DM en orden cronologico.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import cargar as cargar24, BASE5, pesos_lad, pesos_nnls
from ensamble_72h_v2 import cargar as cargar72, TRAMOS, RES


def cv(df, cols, ajustar, grupo=None):
    dias = np.sort(df["dia"].unique())
    pliegues = np.array_split(np.random.default_rng(42).permutation(dias), 5)
    pred = np.full(len(df), np.nan)
    X, y = df[cols].to_numpy(), df["real"].to_numpy()
    g = np.zeros(len(df), int) if grupo is None else df[grupo].to_numpy()
    for dt in pliegues:
        mte = df["dia"].isin(dt).to_numpy()
        for gg in np.unique(g):
            tr, te = (~mte) & (g == gg), mte & (g == gg)
            pred[te] = X[te] @ ajustar(X[tr], y[tr])
    return pred


def dm(real, p_ref, p_nuevo, maxlags):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def reportar(etiqueta, real, variantes, ref, maxlags, filas):
    for nom, p in variantes.items():
        mae = np.abs(real - p).mean()
        mape = (np.abs(real - p) / real).mean() * 100
        if nom == ref:
            print(f"  {nom:34s} MAE={mae:6.2f}  MAPE={mape:5.2f}%   (referencia)")
            continue
        t, pv = dm(real, variantes[ref], p, maxlags)
        marca = "GANA (sig.)" if pv < 0.05 and t > 0 else ("PIERDE (sig.)" if pv < 0.05 else "sin dif.")
        print(f"  {nom:34s} MAE={mae:6.2f}  MAPE={mape:5.2f}%   vs ref p={pv:.4f} {marca}")
        filas.append({"caso": etiqueta, "variante": nom, "mae": mae, "mape": mape, "dm_p_vs_ref": pv})


def main():
    filas = []
    print("=" * 84 + "\n24h -- 5 modelos base\n" + "=" * 84)
    d24 = cargar24().sort_index()
    d24["franja"] = d24.index.hour // 6
    real = d24["real"].to_numpy()
    v = {
        "NNLS global (actual)": cv(d24, BASE5, pesos_nnls),
        "LAD global": cv(d24, BASE5, pesos_lad),
        "NNLS por franja": cv(d24, BASE5, pesos_nnls, "franja"),
        "LAD por franja": cv(d24, BASE5, pesos_lad, "franja"),
    }
    reportar("24h", real, v, "NNLS global (actual)", 23, filas)

    print("\n" + "=" * 84 + "\n72h -- 9 votantes, por tramo\n" + "=" * 84)
    d72, modelos, _ = cargar72()
    d72["dia"] = d72["fecha_hora"].dt.normalize()
    d72["franja"] = d72["fecha_hora"].dt.hour // 6
    for nombre, ini, fin in TRAMOS:
        sub = (d72[(d72.paso_horas >= ini) & (d72.paso_horas <= fin)]
               .sort_values(["cutoff", "paso_horas"]).reset_index(drop=True))
        print(f"\n--- Tramo {nombre} ---")
        real = sub["real"].to_numpy()
        v = {
            "N-BEATSx (linea base)": sub["N-BEATSx"].to_numpy(),
            "LAD global (v3)": cv(sub, modelos, pesos_lad),
            "LAD por franja": cv(sub, modelos, pesos_lad, "franja"),
        }
        reportar(f"72h {nombre}", real, v, "LAD global (v3)", 71, filas)
    pd.DataFrame(filas).to_csv(RES / "pesos_por_franja_metricas.csv", index=False)
    print("\nGuardado: pesos_por_franja_metricas.csv")


if __name__ == "__main__":
    main()
