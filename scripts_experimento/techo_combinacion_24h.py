# -*- coding: utf-8 -*-
"""
Cuanto margen queda REALMENTE en combinar modelos de 24h? Techos con oraculo.

Hasta ahora todas las mejoras probadas (votantes nuevos, granularidad de pesos, LEAR24,
especialistas de pico, demanda objetivo) fracasaron. Antes de seguir buscando conviene saber si el
problema es que no encontramos el buen combinador, o que los votantes ya no tienen nada mas que dar.

Se calculan cuatro techos, todos con informacion perfecta (NO desplegables, son cotas superiores):
  1) ORACULO DE SELECCION: para cada hora, elegir el votante que menos se equivoco.
  2) ORACULO DE PESOS POR HORA DEL DIA: la mejor combinacion convexa fija por hora, ajustada
     sobre los propios datos de prueba (cota de cualquier esquema de pesos por hora).
  3) ORACULO DE PESOS POR DIA: la mejor combinacion convexa por dia.
  4) ORACULO TOTAL: para cada hora, la mejor combinacion convexa de esa misma hora (limite absoluto
     de cualquier mezcla lineal de estos votantes).
Ademas se mide la CORRELACION DE ERRORES entre votantes en las horas pico: si todos se equivocan
igual, ninguna combinacion puede salvarlos, y eso zanja la discusion.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import cargar as cargar24, BASE5, pesos_lad
from o6_comun import RES

PICO = [18, 19, 20]


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def main():
    d = cargar24().sort_index()
    lear = pd.read_csv(RES / "pronostico_lear_24h_2026.csv", parse_dates=["fecha_hora"])
    d = d.join(lear.set_index("fecha_hora")["LEAR24"], how="left")
    v4 = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(v4["pred_ensamble"].rename("v4"), how="left").dropna(subset=["v4"])
    d["hora"] = d.index.hour
    d["dia"] = d.index.normalize()
    votantes = [c for c in BASE5 + ["LEAR24", "CatBoost", "MarkovSw"] if c in d.columns]
    y = d["real"].to_numpy()
    X = d[votantes].to_numpy()
    pk = d["hora"].isin(PICO).to_numpy()
    print(f"Filas: {len(d):,} | votantes: {votantes}")
    print(f"Ensamble v4 actual: MAE={np.abs(y-d['v4']).mean():.2f}  MAPE={mape(y, d['v4']):.2f}%")
    print(f"  en horas pico:    MAE={np.abs(y[pk]-d['v4'].to_numpy()[pk]).mean():.2f}  "
          f"MAPE={mape(y[pk], d['v4'].to_numpy()[pk]):.2f}%\n")

    print(f"{'='*88}\nTECHOS CON ORACULO (no desplegables: son cotas superiores)\n{'='*88}")

    # 1) oraculo de seleccion
    err = np.abs(y[:, None] - X)
    sel = X[np.arange(len(X)), err.argmin(1)]
    print(f"1) Elegir el MEJOR VOTANTE cada hora        MAE={np.abs(y-sel).mean():6.2f}  "
          f"MAPE={mape(y, sel):5.2f}%   (pico {mape(y[pk], sel[pk]):5.2f}%)")

    # 2) pesos optimos por hora del dia
    p2 = np.full(len(y), np.nan)
    for h in range(24):
        m = (d["hora"] == h).to_numpy()
        w = pesos_lad(X[m], y[m])
        p2[m] = X[m] @ w
    print(f"2) Pesos OPTIMOS por hora del dia           MAE={np.abs(y-p2).mean():6.2f}  "
          f"MAPE={mape(y, p2):5.2f}%   (pico {mape(y[pk], p2[pk]):5.2f}%)")

    # 3) pesos optimos por dia
    p3 = np.full(len(y), np.nan)
    for dia, m in d.groupby("dia").groups.items():
        mm = d.index.isin(m)
        if mm.sum() >= len(votantes) + 1:
            p3[mm] = X[mm] @ pesos_lad(X[mm], y[mm])
    ok3 = ~np.isnan(p3)
    print(f"3) Pesos OPTIMOS por dia                    MAE={np.abs(y[ok3]-p3[ok3]).mean():6.2f}  "
          f"MAPE={mape(y[ok3], p3[ok3]):5.2f}%   (pico {mape(y[pk&ok3], p3[pk&ok3]):5.2f}%)")

    # 4) mejor mezcla convexa posible hora a hora (limite absoluto): equivale a proyectar y sobre
    # el simplejo de los votantes en cada punto -> el mejor es min sobre combinaciones, que para un
    # solo punto es el votante mas cercano si y esta fuera del rango, o y exacto si esta dentro.
    lo, hi = X.min(1), X.max(1)
    p4 = np.clip(y, lo, hi)
    print(f"4) Mejor mezcla convexa hora a hora (limite) MAE={np.abs(y-p4).mean():6.2f}  "
          f"MAPE={mape(y, p4):5.2f}%   (pico {mape(y[pk], p4[pk]):5.2f}%)")
    dentro = ((y >= lo) & (y <= hi)).mean() * 100
    print(f"\n   El precio real cae DENTRO del rango de los votantes en {dentro:.1f}% de las horas")
    print(f"   (en horas pico: {((y[pk] >= lo[pk]) & (y[pk] <= hi[pk])).mean()*100:.1f}%)")

    # ---------- correlacion de errores ----------
    print(f"\n{'='*88}\nCORRELACION DE ERRORES ENTRE VOTANTES\n{'='*88}")
    E = y[:, None] - X
    for etiqueta, m in [("todas las horas", np.ones(len(y), bool)), ("horas pico 18-20", pk)]:
        C = np.corrcoef(E[m], rowvar=False)
        tri = C[np.triu_indices(len(votantes), 1)]
        print(f"  {etiqueta:18s} correlacion media de errores = {tri.mean():.3f} "
              f"(min {tri.min():.3f}, max {tri.max():.3f})")
    print("\n  Matriz en horas pico:")
    print(pd.DataFrame(np.corrcoef(E[pk], rowvar=False), index=votantes, columns=votantes).round(2).to_string())

    # sesgo comun: todos subestiman a la vez?
    print(f"\n  En horas pico, fraccion de horas en que TODOS los votantes subestiman: "
          f"{(E[pk] > 0).all(1).mean()*100:.1f}%")
    print(f"  ...en que TODOS sobreestiman: {(E[pk] < 0).all(1).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
