# -*- coding: utf-8 -*-
"""
Separar la hora 0 en su propio grupo de pesos del combinador.

Motivacion (hallazgo del 2026-09-18). El precio de bolsa da un salto grande al cambiar de dia de
despacho: 56.5 COP/kWh de cambio medio entre las 23:00 y las 00:00, contra 7.4 entre las 00:00 y
la 01:00. Con el corte del pronostico a las 00:00, eso parte la primera franja en dos regimenes
muy distintos:

  - horas 1-5: las redes explotan el precio recien observado del corte y son dominantes
               (N-BEATSx MAE 10.3 en la hora 1, contra 46-51 de ARX+GARCH / XGBoost / Persistencia)
  - hora 0:    es el paso 24, al otro lado del salto de medianoche, y ahi las redes PIERDEN su
               ventaja (N-BEATSx 52.3, N-HiTS 49.0) frente a ARX+GARCH (46.0) y Persistencia (46.8)

Como los pesos se ajustan por franja de 6 horas (hora // 6), la hora 0 comparte pesos con las
horas 1-5, donde las redes mandan. Consecuencia medible: en la hora 0 el ensamble (MAE 47.4) es
PEOR que dos de sus propios votantes.

Se prueba separar la hora 0 en su propio grupo, dejando el resto igual. Es un cambio minimo (5
grupos en vez de 4) y motivado por un mecanismo fisico concreto, no por buscar granularidad a
ciegas -- la granularidad por si sola ya se probo (1,2,3,4,6,8,12,24 grupos) sin significancia.

Se verifica con 10 particiones y prueba de Diebold-Mariano, y se mide tambien el efecto en modo
desplegable.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import cargar as cargar24, BASE5
from combinador_optimo_mape import qra_pesos, mape, smape
from o6_comun import RES

SEMILLAS = list(range(10))


def qra_cv_grupo(d, cols, grupo, modo="smape", semilla=42, iteraciones=3):
    """Igual que qra_cv de combinador_optimo_mape, pero con la columna de grupo parametrizable."""
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d[grupo].to_numpy(), d["dia"].to_numpy()
    P = np.full(len(d), np.nan)
    pliegues = np.array_split(np.random.default_rng(semilla).permutation(np.sort(np.unique(dia))), 5)
    for dt in pliegues:
        te_m = np.isin(dia, dt)
        for gg in np.unique(g[te_m]):
            tr, te = (~te_m) & (g == gg), te_m & (g == gg)
            if tr.sum() < len(cols) + 5:
                continue
            w = 1.0 / np.maximum(y[tr], 1e-6)
            ww, b = qra_pesos(X[tr], y[tr], w)
            if modo == "smape":
                for _ in range(iteraciones):
                    pr = X[tr] @ ww + b
                    w = 2.0 / np.maximum(np.abs(y[tr]) + np.abs(pr), 1e-6)
                    ww, b = qra_pesos(X[tr], y[tr], w)
            P[te] = X[te] @ ww + b
    return P


def causal(d, cols, grupo, modo="smape", min_dias=14, iteraciones=3):
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d[grupo].to_numpy(), d["dia"].to_numpy()
    dias = np.sort(np.unique(dia))
    P = np.full(len(d), np.nan)
    for i in range(min_dias, len(dias)):
        tr_m, te_m = np.isin(dia, dias[:i]), dia == dias[i]
        for gg in np.unique(g[te_m]):
            tr, te = tr_m & (g == gg), te_m & (g == gg)
            if tr.sum() < len(cols) + 5:
                continue
            w = 1.0 / np.maximum(y[tr], 1e-6)
            ww, b = qra_pesos(X[tr], y[tr], w)
            if modo == "smape":
                for _ in range(iteraciones):
                    pr = X[tr] @ ww + b
                    w = 2.0 / np.maximum(np.abs(y[tr]) + np.abs(pr), 1e-6)
                    ww, b = qra_pesos(X[tr], y[tr], w)
            P[te] = X[te] @ ww + b
    return P


def dm(real, p_ref, p_new, metrica="mape", maxlags=24):
    real = np.asarray(real, float)
    if metrica == "mape":
        l1, l2 = np.abs(real - p_ref) / real, np.abs(real - p_new) / real
    else:
        l1, l2 = np.abs(real - p_ref), np.abs(real - p_new)
    dd = l1 - l2
    r = sm.OLS(dd, np.ones_like(dd)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def main():
    t0 = time.time()
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    col = "pred" if "pred" in s.columns else s.columns[-1]
    d = d.join(s[col].rename("GARCH-ged"), how="left")
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d["hora"] // 6                                  # vigente: 4 franjas de 6h
    d["g5"] = np.where(d["hora"] == 0, 4, d["hora"] // 6)     # nuevo: hora 0 aparte
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["real"])
    y = d["real"].to_numpy()
    print(f"Filas: {len(d):,} | votantes: {MEJOR}\n")

    print(f"{'='*96}\nSEPARAR LA HORA 0 EN SU PROPIO GRUPO DE PESOS\n{'='*96}")
    print(f"{'configuracion':34s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s} {'MAE hora 0':>11s}")
    p_base = qra_cv_grupo(d, MEJOR, "g4")
    p_new = qra_cv_grupo(d, MEJOR, "g5")
    h0 = (d["hora"] == 0).to_numpy()
    for nom, p in [("vigente (4 franjas de 6h)", p_base), ("hora 0 aparte (5 grupos)", p_new)]:
        print(f"{nom:34s} {np.abs(y-p).mean():8.2f} {mape(y,p):8.2f} {smape(y,p):8.2f} "
              f"{np.abs(y[h0]-p[h0]).mean():11.2f}")
    t, pv = dm(y, p_base, p_new, "mape")
    t2, pv2 = dm(y, p_base, p_new, "mae")
    print(f"\n  DM global: MAPE p={pv:.4f}  |  MAE p={pv2:.4f}   (t>0 = el nuevo gana)")
    t3, pv3 = dm(y[h0], p_base[h0], p_new[h0], "mae", maxlags=7)
    print(f"  DM solo en la hora 0: MAE t={t3:.3f} p={pv3:.4f}")

    print(f"\n{'='*96}\nVERIFICACION 10 PARTICIONES\n{'='*96}")
    A, B = [], []
    for s_ in SEMILLAS:
        pa, pb = qra_cv_grupo(d, MEJOR, "g4", semilla=s_), qra_cv_grupo(d, MEJOR, "g5", semilla=s_)
        A.append([np.abs(y-pa).mean(), mape(y, pa), smape(y, pa)])
        B.append([np.abs(y-pb).mean(), mape(y, pb), smape(y, pb)])
    A, B = np.array(A), np.array(B)
    for j, nom in enumerate(["MAE", "MAPE %", "sMAPE %"]):
        dl = B[:, j] - A[:, j]
        print(f"  {nom:8s} vigente={A[:,j].mean():7.3f}+/-{A[:,j].std():.3f}  "
              f"nuevo={B[:,j].mean():7.3f}+/-{B[:,j].std():.3f}  "
              f"delta={dl.mean():+7.3f}, mejor en {int((dl<0).sum())}/{len(SEMILLAS)}")
    solido = all((B[:, j] - A[:, j] < 0).all() for j in range(3))
    print(f"  VEREDICTO: {'SOLIDO en las tres metricas' if solido else 'no concluyente'}")

    print(f"\n{'='*96}\nMODO DESPLEGABLE (pesos solo con dias anteriores)\n{'='*96}")
    ca, cb = causal(d, MEJOR, "g4"), causal(d, MEJOR, "g5")
    ok = ~np.isnan(ca) & ~np.isnan(cb)
    print(f"  vigente  MAE={np.abs(y[ok]-ca[ok]).mean():7.2f}  MAPE={mape(y[ok],ca[ok]):6.2f}%")
    print(f"  nuevo    MAE={np.abs(y[ok]-cb[ok]).mean():7.2f}  MAPE={mape(y[ok],cb[ok]):6.2f}%")
    t4, pv4 = dm(y[ok], ca[ok], cb[ok], "mape")
    print(f"  DM desplegable sobre MAPE: t={t4:.3f} p={pv4:.4f}")

    pd.DataFrame({"fecha_hora": d.index, "real": y, "pred_vigente": p_base,
                  "pred_hora0_aparte": p_new}).to_csv(RES / "franja_medianoche_24h.csv", index=False)
    print(f"\nGuardado: franja_medianoche_24h.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
