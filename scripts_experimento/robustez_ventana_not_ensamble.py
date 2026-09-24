# -*- coding: utf-8 -*-
"""
ROBUSTEZ DE LA GANANCIA DE NOT EN EL ENSAMBLE: ¿es el metodo o fue la eleccion del candidato?

El problema que resuelve. En `ventana_calibracion_not_24h.py` el votante que entro al ensamble se
eligio como "el de mejor MAPE entre las variantes de ventana" -- y esa eleccion se hizo mirando el
MISMO periodo de prueba donde despues se midio la mejora. Es exactamente el sesgo de seleccion que
el proyecto ya se ha cazado antes: con ~26 candidatos, que uno luzca bien no dice gran cosa.

La prueba. Se meten al ensamble, uno por uno, TODOS los pronosticos basados en seleccion por puntos
de cambio, y ademas los de ventana fija que sirven de control. Si la ganancia viene del metodo,
varios candidatos deben mejorar; si vino de la eleccion, solo mejorara el que se eligio y los
controles de ventana fija se comportaran igual de bien, lo que indicaria que lo que ayuda es
simplemente "un votante lineal mas" y no la seleccion de subperiodos.

Los controles son la parte importante: LEAR Win(728) y LEAR Win(todo) son el MISMO modelo con la
MISMA informacion, cambiando solo la regla de que historia usar. Si esos tambien mejoran igual,
entonces NOT no aporta nada y la ganancia es del votante extra.

Estandar del proyecto: 10 particiones del combinador + Diebold-Mariano sobre la perdida de cada
metrica. Un candidato solo se declara bueno si gana en 10/10 en las tres metricas.
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
from o6_comun import RES
from stacking_24h_v2 import cargar as cargar24, BASE5
from combinador_optimo_mape import qra_cv, mape, smape

SEMILLAS = list(range(10))

# (etiqueta, archivo, es_control)
CANDIDATOS = [
    ("ARX NOTH(728)",  "vc_ARX_NOTH728.csv",  False),
    ("ARX NOT(728)",   "vc_ARX_NOT728.csv",   False),
    ("ARX Av(NOTH)",   "vc_ARX_AvNOTH.csv",   False),
    ("LEAR NOT(728)",  "vc_LEAR_NOT728.csv",  False),
    ("LEAR NOTH(728)", "vc_LEAR_NOTH728.csv", False),
    ("LEAR WinH(728)", "vc_LEAR_WinH728.csv", True),
    ("LEAR Win(728)",  "vc_LEAR_Win728.csv",  True),
    ("LEAR Win(todo)", "vc_LEAR_Wintodo.csv", True),
]


def dm_metrica(real, p_ref, p_nuevo, metrica, maxlags=24):
    real = np.asarray(real, float)
    if metrica == "mape":
        l1, l2 = np.abs(real - p_ref) / real, np.abs(real - p_nuevo) / real
    elif metrica == "smape":
        l1 = np.abs(real - p_ref) / ((np.abs(real) + np.abs(p_ref)) / 2)
        l2 = np.abs(real - p_nuevo) / ((np.abs(real) + np.abs(p_nuevo)) / 2)
    else:
        l1, l2 = np.abs(real - p_ref), np.abs(real - p_nuevo)
    d = l1 - l2
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.pvalues[0])


def main():
    t0 = time.time()
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")

    disponibles = []
    for etq, arch, ctrl in CANDIDATOS:
        ruta = RES / arch
        if not ruta.exists():
            print(f"  (falta {arch}, se omite {etq})")
            continue
        v = pd.read_csv(ruta, parse_dates=["fecha_hora"]).drop_duplicates("fecha_hora")
        d = d.join(v.set_index("fecha_hora")["pred"].rename(etq), how="left")
        disponibles.append((etq, ctrl))

    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + [e for e, _ in disponibles] + ["real"])
    y = d["real"].to_numpy()
    print(f"Filas alineadas: {len(d):,} | votantes base: {MEJOR}\n")

    # linea base, una sola vez, con las mismas 10 particiones para todos
    A = np.array([[np.abs(y - p).mean(), mape(y, p), smape(y, p)]
                  for p in (qra_cv(d, MEJOR, "smape", s_) for s_ in SEMILLAS)])
    print(f"Ensamble actual (6 votantes): MAE={A[:,0].mean():.3f}+/-{A[:,0].std():.3f}  "
          f"MAPE={A[:,1].mean():.3f}+/-{A[:,1].std():.3f}  sMAPE={A[:,2].mean():.3f}+/-{A[:,2].std():.3f}\n")

    print("=" * 108)
    print("CADA CANDIDATO COMO SEPTIMO VOTANTE  (delta negativo = mejora; 10/10 = gana en las 10 particiones)")
    print("=" * 108)
    print(f"{'candidato':17s} {'tipo':9s} {'dMAE':>8s} {'gana':>6s} {'dMAPE':>8s} {'gana':>6s} "
          f"{'dsMAPE':>8s} {'gana':>6s}  {'p(MAE)':>8s} {'p(MAPE)':>8s}  veredicto")
    filas = []
    for etq, ctrl in disponibles:
        B = np.array([[np.abs(y - p).mean(), mape(y, p), smape(y, p)]
                      for p in (qra_cv(d, MEJOR + [etq], "smape", s_) for s_ in SEMILLAS)])
        dl = B - A
        gana = (dl < 0).sum(axis=0)
        solido = bool((gana == 10).all())
        pm = dm_metrica(y, qra_cv(d, MEJOR, "smape"), qra_cv(d, MEJOR + [etq], "smape"), "mae")
        pp = dm_metrica(y, qra_cv(d, MEJOR, "smape"), qra_cv(d, MEJOR + [etq], "smape"), "mape")
        print(f"{etq:17s} {'control' if ctrl else 'NOT':9s} "
              f"{dl[:,0].mean():+8.3f} {gana[0]:>4d}/10 {dl[:,1].mean():+8.3f} {gana[1]:>4d}/10 "
              f"{dl[:,2].mean():+8.3f} {gana[2]:>4d}/10  {pm:8.4f} {pp:8.4f}  "
              f"{'SOLIDO' if solido else '-'}", flush=True)
        filas.append(dict(candidato=etq, control=ctrl, dMAE=dl[:, 0].mean(), dMAPE=dl[:, 1].mean(),
                          dsMAPE=dl[:, 2].mean(), gana_MAE=gana[0], gana_MAPE=gana[1],
                          gana_sMAPE=gana[2], p_MAE=pm, p_MAPE=pp, solido=solido))

    R = pd.DataFrame(filas)
    R.to_csv(RES / "robustez_ventana_not_ensamble.csv", index=False)

    print("\n" + "=" * 108)
    print("LECTURA")
    print("=" * 108)
    n_not = int(R[~R["control"]]["solido"].sum())
    n_ctrl = int(R[R["control"]]["solido"].sum())
    print(f"  candidatos NOT que pasan el estandar 10/10: {n_not} de {(~R['control']).sum()}")
    print(f"  controles de ventana fija que lo pasan:      {n_ctrl} de {R['control'].sum()}")
    if n_not >= 2 and n_ctrl == 0:
        print("  -> La ganancia es del METODO, no de la eleccion del candidato ni de sumar un votante mas.")
    elif n_ctrl > 0:
        print("  -> Ojo: algun control de ventana fija tambien mejora. Parte de la ganancia puede ser")
        print("     simplemente tener un votante lineal adicional, no la seleccion por puntos de cambio.")
    else:
        print("  -> Solo el candidato elegido mejora: compatible con seleccion post-hoc. No declarar mejora.")

    # ¿mejoran juntos? el mejor NOT del ARX y el mejor del LEAR son familias distintas
    mejores = R[~R["control"] & R["solido"]].nsmallest(2, "dMAPE")["candidato"].tolist()
    if len(mejores) == 2:
        print(f"\n  Los dos mejores juntos ({mejores[0]} + {mejores[1]}):")
        B = np.array([[np.abs(y - p).mean(), mape(y, p), smape(y, p)]
                      for p in (qra_cv(d, MEJOR + mejores, "smape", s_) for s_ in SEMILLAS)])
        dl = B - A
        for j, nm in enumerate(["MAE", "MAPE %", "sMAPE %"]):
            print(f"    {nm:8s} {A[:,j].mean():7.3f} -> {B[:,j].mean():7.3f}  "
                  f"delta={dl[:,j].mean():+.3f}, mejor en {int((dl[:,j]<0).sum())}/10")

    print(f"\nGuardado: robustez_ventana_not_ensamble.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
