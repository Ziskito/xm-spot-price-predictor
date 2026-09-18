# -*- coding: utf-8 -*-
"""
"Podemos cambiarlo por un modelo mas sencillo, como sugiere la literatura?"

La literatura de pronostico de precios electricos (Lago et al. 2021; Kapoor & Wichitaksorn 2023)
sostiene que modelos simples y bien regularizados igualan o superan al deep learning. Nuestro
ensamble de 72h tiene 7 votantes, de los cuales DOS son redes neuronales (N-BEATSx base y N-BEATSx
con exogenas enriquecidas) que son, con diferencia, lo mas caro de entrenar y mantener: son las
que obligan a tener el entorno de neuralforecast/PyTorch y las que hacen que un reentrenamiento
completo tome horas en esta maquina.

La pregunta practica es: cuanto aportan de verdad? Si el ensamble sin ellas rinde igual, el
sistema se vuelve entrenable en minutos, desplegable sin GPU ni PyTorch, y mucho mas facil de
defender y reproducir.

Esta prueba es casi gratis porque las predicciones de todos los votantes ya estan guardadas:
solo cambia QUE columnas entran al combinador LAD. Se evalua:
  1) subconjuntos fijos de interes (todos, sin redes, solo arboles, minimo, etc.);
  2) seleccion greedy hacia adelante: que votante agrega mas en cada paso, para ver cuantos
     hacen falta de verdad antes de que la curva se aplane.
Todo con DM HAC contra el ensamble completo vigente.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ensamble_72h_diario import TRAMOS, lad_cv, lad_causal, hac_p, RES

COMPLETO = ["NBX", "NBX_exog", "XGB-directo", "CatB-directo", "CatB-denso", "Ridge-directo", "NaiveEstac"]
REDES = ["NBX", "NBX_exog"]

SUBCONJUNTOS = {
    "completo (7 votantes, vigente)": COMPLETO,
    "sin redes neuronales (5)": [c for c in COMPLETO if c not in REDES],
    "solo arboles + naive (4)": ["XGB-directo", "CatB-directo", "CatB-denso", "NaiveEstac"],
    "arboles + Ridge (4)": ["XGB-directo", "CatB-directo", "CatB-denso", "Ridge-directo"],
    "minimo: Ridge + XGB + naive (3)": ["Ridge-directo", "XGB-directo", "NaiveEstac"],
    "muy minimo: Ridge + naive (2)": ["Ridge-directo", "NaiveEstac"],
    "una red + Ridge + naive (3)": ["NBX_exog", "Ridge-directo", "NaiveEstac"],
}


def evaluar(sub, cols, fun):
    y = sub["real"].to_numpy()
    P, _ = fun(sub, cols)
    ok = ~np.isnan(P)
    return P, ok, float(np.abs(y[ok] - P[ok]).mean())


def main():
    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v["dia"] = v["fecha_hora"].dt.normalize()
    print(f"Filas: {len(v):,} | cortes: {v['cutoff'].nunique()}\n")

    filas = []
    for modo, fun in [("CV", lad_cv), ("desplegable", lad_causal)]:
        print(f"{'='*104}\nMODO {modo.upper()} -- MAE por tramo y DM contra el ensamble completo\n{'='*104}")
        print(f"{'subconjunto':34s} {'1-24h':>16s} {'25-48h':>16s} {'49-72h':>16s}")
        ref = {}
        for nombre, ini, fin in TRAMOS:
            sub = v[v["paso_horas"].between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
            ref[nombre] = (sub, *evaluar(sub, COMPLETO, fun))
        for etiqueta, cols in SUBCONJUNTOS.items():
            linea, fila = f"{etiqueta:34s}", {"modo": modo, "subconjunto": etiqueta, "n_votantes": len(cols)}
            for nombre, ini, fin in TRAMOS:
                sub, P0, ok0, mae0 = ref[nombre]
                y = sub["real"].to_numpy()
                P, ok, mae = evaluar(sub, cols, fun)
                oo = ok & ok0
                if etiqueta.startswith("completo"):
                    linea += f" {mae:9.2f}       "
                else:
                    t, pv = hac_p(np.abs(y[oo] - P0[oo]) - np.abs(y[oo] - P[oo]))
                    marca = "+" if (pv < 0.05 and t > 0) else ("-" if pv < 0.05 else " ")
                    linea += f" {mae:9.2f}({mae-mae0:+5.2f}){marca}"
                fila[f"mae_{nombre}"] = mae
            print(linea)
            filas.append(fila)

        # ---------- seleccion greedy hacia adelante (solo en el tramo largo, el mas dificil) ----------
        nombre, ini, fin = TRAMOS[-1]
        sub = v[v["paso_horas"].between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
        print(f"\nSeleccion greedy hacia adelante en el tramo {nombre} (modo {modo}):")
        elegidos, restantes = [], list(COMPLETO)
        while restantes:
            mejor, mejor_mae = None, np.inf
            for c in restantes:
                _, _, mae = evaluar(sub, elegidos + [c], fun)
                if mae < mejor_mae:
                    mejor, mejor_mae = c, mae
            elegidos.append(mejor)
            restantes.remove(mejor)
            print(f"  {len(elegidos)}: +{mejor:14s} -> MAE {mejor_mae:.2f}")
        print()

    pd.DataFrame(filas).to_csv(RES / "simplificar_ensamble_72h.csv", index=False)
    print("Guardado: simplificar_ensamble_72h.csv")
    print("(+ = el subconjunto MEJORA significativamente vs completo; - = empeora; vacio = sin diferencia)")


if __name__ == "__main__":
    main()
