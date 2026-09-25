# -*- coding: utf-8 -*-
"""
Hora 0: reemplazar el combinador ponderado por un promedio simple de votantes.

Motivacion teorica (Ziel & Weron 2018, Energy Economics 70:396-420). En un mercado day-ahead los
precios de las 24 horas de un dia se revelan de una vez en una sola subasta, asi que el vector de
24 precios es la unidad natural, no la serie horaria continua. El "salto" entre las 23:00 y las
00:00 no es un defecto: son subastas distintas. Ese mismo trabajo reporta que combinar pronosticos
mejora la precision "incluso con un esquema de promedio simple de solo dos modelos", lo que sugiere
probar el promedio simple donde los pesos aprendidos tengan poca informacion.

Motivacion empirica. En la hora 0 (n=216 dias) el promedio simple de los 7 votantes da MAE 45.40,
mientras el combinador ponderado da 47.37. ADVERTENCIA que hay que respetar: el error estandar del
MAE a esa hora es +/-4.20, asi que esa diferencia de ~2 NO es concluyente por si sola. Por eso se
prueba formalmente con Diebold-Mariano y multiples particiones, y se mide el efecto GLOBAL, no
solo en la hora 0.

Hipotesis del mecanismo: los pesos se estiman por franja de 6 horas; la franja 0-5 esta dominada
por las horas 1-5, donde las redes son muy superiores porque explotan el precio recien observado
en el corte. En la hora 0 esa ventaja no existe (esta al otro lado del salto), asi que los pesos
heredados de la franja son inadecuados y un promedio sin pesos podria ser mas robusto.
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
from combinador_optimo_mape import qra_cv, mape, smape
from o6_comun import RES

SEMILLAS = list(range(10))


def dm(real, p_ref, p_new, metrica="mae", maxlags=24):
    real = np.asarray(real, float)
    if metrica == "mape":
        l1, l2 = np.abs(real - p_ref) / real, np.abs(real - p_new) / real
    else:
        l1, l2 = np.abs(real - p_ref), np.abs(real - p_new)
    d = l1 - l2
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def main():
    t0 = time.time()
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    col = "pred" if "pred" in s.columns else s.columns[-1]
    d = d.join(s[col].rename("GARCH-ged"), how="left")
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d["hora"] // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["real"])
    y = d["real"].to_numpy()
    h0 = (d["hora"] == 0).to_numpy()
    prom = d[MEJOR].mean(axis=1).to_numpy()
    print(f"Filas: {len(d):,} | votantes: {MEJOR} | dias con hora 0: {h0.sum()}\n")

    print(f"{'='*94}\nPROMEDIO SIMPLE EN LA HORA 0 (resto del dia sin cambios)\n{'='*94}")
    variantes = {}
    for s_ in SEMILLAS:
        base = qra_cv(d, MEJOR, "smape", s_)
        nuevo = base.copy()
        nuevo[h0] = prom[h0]
        variantes[s_] = (base, nuevo)

    A = np.array([[np.abs(y - b).mean(), mape(y, b), smape(y, b), np.abs(y[h0] - b[h0]).mean()]
                  for b, _ in variantes.values()])
    B = np.array([[np.abs(y - n).mean(), mape(y, n), smape(y, n), np.abs(y[h0] - n[h0]).mean()]
                  for _, n in variantes.values()])
    print(f"{'metrica':14s} {'combinador':>18s} {'prom. simple en h0':>20s} {'delta':>9s} {'mejor en':>10s}")
    for j, nom in enumerate(["MAE global", "MAPE %", "sMAPE %", "MAE hora 0"]):
        dl = B[:, j] - A[:, j]
        print(f"{nom:14s} {A[:,j].mean():10.3f}+/-{A[:,j].std():.3f} "
              f"{B[:,j].mean():12.3f}+/-{B[:,j].std():.3f} {dl.mean():+9.3f} "
              f"{int((dl<0).sum()):7d}/{len(SEMILLAS)}")

    base42, nuevo42 = variantes[42 if 42 in variantes else SEMILLAS[0]]
    t1, p1 = dm(y, base42, nuevo42, "mae")
    t2, p2 = dm(y, base42, nuevo42, "mape")
    t3, p3 = dm(y[h0], base42[h0], nuevo42[h0], "mae", maxlags=7)
    print(f"\n  DM global   : MAE t={t1:+.3f} p={p1:.4f}  |  MAPE t={t2:+.3f} p={p2:.4f}")
    print(f"  DM en hora 0: MAE t={t3:+.3f} p={p3:.4f}   (t>0 = el promedio simple gana)")
    print(f"  (n=216 en la hora 0; el error estandar del MAE ahi es ~4.2, "
          f"asi que hace falta un efecto grande para ser detectable)")

    # cuanto pesa la hora 0 en el global: 1/24 de las observaciones
    print(f"\n  Nota de magnitud: la hora 0 es {h0.sum()/len(d)*100:.1f}% de las observaciones, "
          f"asi que aunque mejorara mucho ahi,\n  el efecto global queda dividido por ~24.")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
