# -*- coding: utf-8 -*-
"""
Verificacion del combinador alineado con la metrica, con atencion especial a un resultado
sospechoso que exige escrutinio.

Lo que salio en combinador_optimo_mape.py:
  objetivo MAE    -> MAE 41.45  MAPE 10.86%  sMAPE 10.08%
  objetivo MAPE   -> MAE 41.58  MAPE 10.71%  sMAPE 10.11%   (compromiso normal: gana MAPE, pierde MAE)
  objetivo sMAPE  -> MAE 41.19  MAPE 10.72%  sMAPE 10.04%   (gana en LAS TRES)

La fila del objetivo MAPE se comporta como se espera: mejora la metrica que optimiza y empeora la
otra. Pero la del objetivo sMAPE gana tambien en MAE, que NO es lo que optimiza -- y el combinador
por MAE deberia ser, por construccion, el mejor en MAE dentro de cada pliegue de entrenamiento.
Que pierda fuera de muestra apunta a un efecto de REGULARIZACION: la reponderacion 2/(|y|+|pred|)
comprime el rango efectivo de la funcion de perdida y evita que unos pocos dias de precio alto
dominen el ajuste de los pesos, lo que reduce sobreajuste del meta-modelo. Es una explicacion
plausible, pero hay que comprobarla en vez de asumirla.

Se verifica:
  1) Las tres metricas x 10 semillas de particion, para los tres objetivos.
  2) Que dentro de la MUESTRA DE ENTRENAMIENTO el objetivo MAE si gana en MAE (si no, hay un bug).
  3) Dispersion de los pesos estimados: si la hipotesis de regularizacion es correcta, los pesos
     del combinador por sMAPE deberian ser mas estables entre pliegues que los del combinador MAE.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog
from scipy import sparse

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import cargar as cargar24, BASE5
from combinador_optimo_mape import qra_pesos, qra_cv, mape, smape
from o6_comun import RES

SEMILLAS = list(range(10))


def main():
    t0 = time.time()
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    col = "pred" if "pred" in s.columns else s.columns[-1]
    d = d.join(s[col].rename("GARCH-ged"), how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    d = d.dropna(subset=BASE5 + ["GARCH-ged", "real"])
    y = d["real"].to_numpy()
    MEJOR = BASE5 + ["GARCH-ged"]
    print(f"Filas: {len(d):,} | votantes: {len(MEJOR)}\n")

    # ---------- 1) tres metricas x 10 semillas ----------
    print(f"{'='*96}\n1) LAS TRES METRICAS, 10 SEMILLAS DE PARTICION\n{'='*96}")
    print(f"{'objetivo':18s} {'MAE':>16s} {'MAPE %':>16s} {'sMAPE %':>16s}")
    res = {}
    for modo in ["mae", "mape", "smape"]:
        ms = np.array([[np.abs(y - p).mean(), mape(y, p), smape(y, p)]
                       for p in (qra_cv(d, MEJOR, modo, s_) for s_ in SEMILLAS)])
        res[modo] = ms
        print(f"{modo:18s} {ms[:,0].mean():8.2f}+/-{ms[:,0].std():.3f} "
              f"{ms[:,1].mean():8.3f}+/-{ms[:,1].std():.3f} "
              f"{ms[:,2].mean():8.3f}+/-{ms[:,2].std():.3f}")
    print("\n  delta del objetivo sMAPE frente al objetivo MAE (negativo = mejor):")
    for j, nom in enumerate(["MAE", "MAPE", "sMAPE"]):
        dl = res["smape"][:, j] - res["mae"][:, j]
        print(f"    {nom:6s} {dl.mean():+7.3f} +/- {dl.std():.3f}, negativo en "
              f"{int((dl<0).sum())}/{len(SEMILLAS)} semillas")

    # ---------- 2) control: dentro de la muestra, el objetivo MAE debe ganar en MAE ----------
    print(f"\n{'='*96}\n2) CONTROL EN MUESTRA (el objetivo MAE debe ganar en MAE; si no, hay bug)\n{'='*96}")
    X, yy = d[MEJOR].to_numpy(), y
    g = d["g4"].to_numpy()
    for modo in ["mae", "smape"]:
        err = []
        for gg in np.unique(g):
            m = g == gg
            if modo == "mae":
                w = None
            else:
                w = np.ones(m.sum())
                ww, b = qra_pesos(X[m], yy[m], w)
                for _ in range(3):
                    pr = X[m] @ ww + b
                    w = 2.0 / np.maximum(np.abs(yy[m]) + np.abs(pr), 1e-6)
                    ww, b = qra_pesos(X[m], yy[m], w)
            if modo == "mae":
                ww, b = qra_pesos(X[m], yy[m], None)
            err.append(np.abs(yy[m] - (X[m] @ ww + b)))
        e = np.concatenate(err)
        print(f"  objetivo {modo:6s}: MAE EN MUESTRA = {e.mean():.3f}")
    print("  (si el objetivo MAE no gana aqui, el programa lineal esta mal planteado)")

    # ---------- 3) estabilidad de los pesos ----------
    print(f"\n{'='*96}\n3) ESTABILIDAD DE LOS PESOS ENTRE PLIEGUES (hipotesis de regularizacion)\n{'='*96}")
    dias = np.sort(d["dia"].unique())
    dia = d["dia"].to_numpy()
    for modo in ["mae", "smape"]:
        todos = []
        for s_ in SEMILLAS[:5]:
            for dt in np.array_split(np.random.default_rng(s_).permutation(dias), 5):
                te = np.isin(dia, dt)
                for gg in np.unique(g):
                    tr = (~te) & (g == gg)
                    if tr.sum() < len(MEJOR) + 5:
                        continue
                    if modo == "mae":
                        w = None
                    else:
                        w = 1.0 / np.maximum(yy[tr], 1e-6)
                    ww, b = qra_pesos(X[tr], yy[tr], w)
                    if modo == "smape":
                        for _ in range(3):
                            pr = X[tr] @ ww + b
                            w = 2.0 / np.maximum(np.abs(yy[tr]) + np.abs(pr), 1e-6)
                            ww, b = qra_pesos(X[tr], yy[tr], w)
                    todos.append(ww)
        W = np.array(todos)
        print(f"  objetivo {modo:6s}: desviacion media de los pesos entre ajustes = {W.std(0).mean():.4f}")
    print("  (menor = pesos mas estables = menos sobreajuste del meta-modelo)")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
