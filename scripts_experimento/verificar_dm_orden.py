# -*- coding: utf-8 -*-
"""
Verificacion de un posible bug en los tests de Diebold-Mariano de los ensambles.

El estimador HAC (Newey-West) de la varianza de la diferencia de perdidas asume que la serie esta
en ORDEN TEMPORAL: asi puede medir su autocorrelacion. Si la serie se arma concatenando los
pliegues de la validacion cruzada (dias en orden aleatorio), la autocorrelacion real queda rota,
el HAC la subestima, y los p-valores salen artificialmente pequenos.

Aqui se recalculan los p-valores de los ensambles en orden cronologico y en el orden de pliegues,
para medir cuanto cambia cada conclusion. Tambien se prueba LAD-CV vs NNLS-CV con el orden correcto.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from ensamble_72h_v2 import cargar, TRAMOS, RES
from stacking_24h_v2 import pesos_nnls, pesos_lad
from ensamble_72h_causal import cv_dias

BASE5 = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]


def dm(real, p_ref, p_nuevo, maxlags):
    d = np.abs(np.asarray(real) - np.asarray(p_ref)) - np.abs(np.asarray(real) - np.asarray(p_nuevo))
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def main():
    print("=" * 84 + "\nENSAMBLE DE 24h (stacking_ensamble_cv_2026.csv)\n" + "=" * 84)
    ens = pd.read_csv(RES / "stacking_ensamble_cv_2026.csv", parse_dates=["fecha_hora"])
    print(f"Archivo en orden cronologico: {ens['fecha_hora'].is_monotonic_increasing}")
    c = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    piv = c[c["origen"] == "Origen 6"].pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
    m = ens.merge(piv.reset_index(), on="fecha_hora", how="inner")
    ms = m.sort_values("fecha_hora")
    print(f"{'modelo':14s} {'p orden archivo':>16s} {'p cronologico':>15s}")
    for mod in BASE5:
        _, pa = dm(m["real"], m[mod], m["pred_ensamble"], 23)
        t, pb = dm(ms["real"], ms[mod], ms["pred_ensamble"], 23)
        print(f"{mod:14s} {pa:16.5f} {pb:15.5f}  (t={t:.2f})")

    s = pd.read_csv(RES / "stacking_24h_v2_2026.csv", parse_dates=["fecha_hora"]).sort_values("fecha_hora")
    t, p = dm(s["real"], s["pred_A"], s["pred_C"], 23)
    print(f"\n24h LAD-CV todos vs NNLS-CV 5 modelos (cronologico, todo 2026): t={t:.3f} p={p:.4f}")

    print("\n" + "=" * 84 + "\nENSAMBLE DE 72h v2 (9 votantes, NNLS-CV por tramo)\n" + "=" * 84)
    df, modelos, _ = cargar()
    df["dia"] = df["fecha_hora"].dt.normalize()
    filas = []
    for nombre, ini, fin in TRAMOS:
        sub = (df[(df.paso_horas >= ini) & (df.paso_horas <= fin)]
               .sort_values(["cutoff", "paso_horas"]).reset_index(drop=True))
        p_nnls = cv_dias(sub, modelos, pesos_nnls).to_numpy()
        p_lad = cv_dias(sub, modelos, pesos_lad).to_numpy()
        real = sub["real"].to_numpy()
        base = sub["N-BEATSx"].to_numpy()

        dias = np.sort(sub["dia"].unique())
        pl = np.array_split(np.random.default_rng(42).permutation(dias), 5)
        orden_pliegues = np.concatenate([np.where(sub["dia"].isin(dt))[0] for dt in pl])

        _, p_bug = dm(real[orden_pliegues], base[orden_pliegues], p_nnls[orden_pliegues], 71)
        t_ok, p_ok = dm(real, base, p_nnls, 71)
        t_l, p_l = dm(real, base, p_lad, 71)
        t_ln, p_ln = dm(real, p_nnls, p_lad, 71)
        mae_b, mae_n, mae_l = (np.abs(real - x).mean() for x in (base, p_nnls, p_lad))
        print(f"\nTramo {nombre}: MAE N-BEATSx={mae_b:.2f} | NNLS-CV={mae_n:.2f} | LAD-CV={mae_l:.2f}")
        print(f"  NNLS-CV vs N-BEATSx: p orden pliegues={p_bug:.4f}  ->  p cronologico={p_ok:.4f} (t={t_ok:.2f})")
        print(f"  LAD-CV  vs N-BEATSx: p cronologico={p_l:.4f} (t={t_l:.2f})")
        print(f"  LAD-CV  vs NNLS-CV : p cronologico={p_ln:.4f} (t={t_ln:.2f})")
        filas.append({"tramo": nombre, "mae_nbeatsx": mae_b, "mae_nnls_cv": mae_n, "mae_lad_cv": mae_l,
                      "p_nnls_orden_pliegues": p_bug, "p_nnls_cronologico": p_ok,
                      "p_lad_vs_nbeatsx": p_l, "p_lad_vs_nnls": p_ln})
    pd.DataFrame(filas).to_csv(RES / "verificacion_dm_orden_72h.csv", index=False)


if __name__ == "__main__":
    main()
