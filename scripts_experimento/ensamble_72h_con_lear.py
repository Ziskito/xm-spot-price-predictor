# -*- coding: utf-8 -*-
"""
LEAR como votante del ensamble de 72h, y busqueda del ensamble MAS SIMPLE que iguale al vigente.

Contexto: lear_72h.py mostro que un LEAR (LASSO por paso de horizonte, con estructura rica de
rezagos) es el MEJOR modelo individual del proyecto -- MAE global 65.11, por encima de las dos
redes N-BEATSx (68.79 / 68.33), de todos los arboles y de Ridge-directo (71.71) -- y que a 49-72h
le gana a N-BEATSx con significancia (-9.34, p=0.0225) y empata con el ensamble completo de 7
votantes (p=0.26). Esto reproduce en Colombia lo que la literatura reporta para los mercados mas
estudiados: un lineal bien regularizado compite de tu a tu con el deep learning.

Aqui se prueban las dos consecuencias practicas:
  A) Agregar LEAR al ensamble vigente, y sustituir con el a Ridge-directo (el lineal que ya estaba).
  B) Buscar el ensamble MAS SIMPLE que no sea significativamente peor que el completo, ahora que
     hay un votante lineal mucho mas fuerte. Si LEAR + una red igualan a los 7 votantes, el sistema
     se simplifica de forma sustancial.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ensamble_72h_diario import TRAMOS, lad_cv, lad_causal, hac_p, RES

COMPLETO = ["NBX", "NBX_exog", "XGB-directo", "CatB-directo", "CatB-denso", "Ridge-directo", "NaiveEstac"]

CONJUNTOS = {
    "completo vigente (7)": COMPLETO,
    "completo + LEAR (8)": COMPLETO + ["LEAR"],
    "completo, LEAR en vez de Ridge (7)": [c if c != "Ridge-directo" else "LEAR" for c in COMPLETO],
    "LEAR + 2 redes + naive (4)": ["LEAR", "NBX", "NBX_exog", "NaiveEstac"],
    "LEAR + 1 red + naive (3)": ["LEAR", "NBX_exog", "NaiveEstac"],
    "LEAR + 1 red (2)": ["LEAR", "NBX_exog"],
    "LEAR + arboles + naive (5)": ["LEAR", "XGB-directo", "CatB-directo", "CatB-denso", "NaiveEstac"],
    "solo LEAR + naive (2)": ["LEAR", "NaiveEstac"],
}


def main():
    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    lear = pd.read_csv(RES / "pronostico_lear_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v = v.merge(lear[["fecha_hora", "cutoff", "paso_horas", "LEAR"]],
                on=["fecha_hora", "cutoff", "paso_horas"], how="inner")
    v["dia"] = v["fecha_hora"].dt.normalize()
    print(f"Filas: {len(v):,} | cortes: {v['cutoff'].nunique()}\n")

    filas = []
    for modo, fun in [("CV", lad_cv), ("desplegable", lad_causal)]:
        print(f"{'='*112}\nMODO {modo.upper()} -- MAE por tramo, con DM contra el ensamble completo vigente\n{'='*112}")
        print(f"{'conjunto':36s} {'n':>2s} {'1-24h':>17s} {'25-48h':>17s} {'49-72h':>17s}")
        ref = {}
        for nom, ini, fin in TRAMOS:
            sub = v[v.paso_horas.between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
            P, _ = fun(sub, COMPLETO)
            ref[nom] = (sub, P)
        for etiqueta, cols in CONJUNTOS.items():
            linea = f"{etiqueta:36s} {len(cols):2d}"
            fila = {"modo": modo, "conjunto": etiqueta, "n_votantes": len(cols)}
            for nom, ini, fin in TRAMOS:
                sub, P0 = ref[nom]
                y = sub["real"].to_numpy()
                P, _ = fun(sub, cols)
                oo = ~(np.isnan(P) | np.isnan(P0))
                mae, mae0 = np.abs(y[oo] - P[oo]).mean(), np.abs(y[oo] - P0[oo]).mean()
                fila[f"mae_{nom}"] = mae
                if etiqueta.startswith("completo vigente"):
                    linea += f" {mae:10.2f}        "
                else:
                    t, pv = hac_p(np.abs(y[oo] - P0[oo]) - np.abs(y[oo] - P[oo]))
                    marca = "MEJOR" if (pv < 0.05 and t > 0) else ("peor " if pv < 0.05 else "  =  ")
                    linea += f" {mae:10.2f}({mae-mae0:+5.2f}){marca}"
            print(linea)
            filas.append(fila)
        print()

    pd.DataFrame(filas).to_csv(RES / "ensamble_72h_con_lear.csv", index=False)
    print("Guardado: ensamble_72h_con_lear.csv")
    print("MEJOR/peor = diferencia significativa al 5% contra el completo vigente;  '=' = sin diferencia")


if __name__ == "__main__":
    main()
