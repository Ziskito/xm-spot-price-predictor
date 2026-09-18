# -*- coding: utf-8 -*-
"""
Otra via para el margen de 25-72h: variantes de N-BEATSx ya descargadas y nunca usadas como
votantes propios.

La linea de arboles especialistas por tramo (`especialistas_por_tramo_72h.py`) fallo: los arboles
generalistas y especialistas correlacionan demasiado con lo que ya vota el ensamble (misma
informacion, distinto algoritmo, pero no aportan angulo nuevo). Antes de darla por agotada falta
probar la diversidad mas barata que existe: `pronostico_72h_diario_votantes_2026.csv` ya trae tres
variantes de N-BEATSx que nunca entraron al ensamble como votantes independientes -- solo se uso
`NBX` (semilla base):

    NBX_s7      -- misma arquitectura, semilla 7
    NBX_s123    -- misma arquitectura, semilla 123
    NBX_in336   -- misma arquitectura, ventana de entrada de 336h (14 dias) en vez de la base

La correlacion entre ellas es alta (0.97-0.98) pero no perfecta, y el mecanismo por el que ayudan a
un ensamble (Lago et al., "diversidad de errores", no diversidad de arquitectura) no exige baja
correlacion en el nivel, sino errores parcialmente independientes. Se prueba:

  (1) agregar las 3 variantes al conjunto de 4 votantes vigente (LAD y combinador sMAPE).
  (2) un "sub-ensamble" de las 4 redes NBX (promedio simple) como un solo votante nuevo, para ver
      si diversificar DENTRO de la familia neuronal ya captura la mayor parte de la ganancia sin
      inflar el numero de parametros del combinador.

Se hace en los tres tramos, no solo los lejanos, porque el mecanismo (diversidad intra-familia) no
tiene razon a priori para depender de la distancia al corte -- a diferencia del GARCH-t, que si la
tenia (se apagaba con la distancia).
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import RES
from ensamble_72h_diario import TRAMOS
from puente_24h_a_72h import mape, smape, dm_metrica, lad_cv, causal

NUEVO_72 = ["LEAR", "NBX", "NBX_exog", "NaiveEstac"]
VARIANTES_NBX = ["NBX_s7", "NBX_s123", "NBX_in336"]
SEMILLAS = list(range(10))


def main():
    t0 = time.time()
    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    lear = pd.read_csv(RES / "pronostico_lear_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v = v.merge(lear[["fecha_hora", "cutoff", "paso_horas", "LEAR"]],
                on=["fecha_hora", "cutoff", "paso_horas"], how="inner")
    v["dia"] = v["fecha_hora"].dt.normalize()
    v = v.dropna(subset=NUEVO_72 + VARIANTES_NBX + ["real"])
    v["sub_nbx"] = v[["NBX", "NBX_exog"] + VARIANTES_NBX].mean(axis=1)
    print(f"Filas: {len(v):,}\n")

    print(f"{'='*100}\nINDIVIDUALES: variantes de N-BEATSx contra la base\n{'='*100}")
    print(f"{'modelo':12s} {'MAE':>8s} {'MAPE %':>8s}")
    for c in ["NBX", "NBX_exog"] + VARIANTES_NBX + ["sub_nbx"]:
        print(f"{c:12s} {np.abs(v['real']-v[c]).mean():8.2f} {mape(v['real'], v[c]):8.2f}")

    CONFS = [
        ("A  4 votantes (vigente)", NUEVO_72, "lad"),
        ("B  + 3 variantes NBX, LAD", NUEVO_72 + VARIANTES_NBX, "lad"),
        ("C  + 3 variantes NBX, sMAPE", NUEVO_72 + VARIANTES_NBX, "smape"),
        ("D  LEAR + sub_nbx + naive (5->3 votantes)", ["LEAR", "sub_nbx", "NaiveEstac"], "lad"),
        ("E  LEAR + sub_nbx + naive, sMAPE", ["LEAR", "sub_nbx", "NaiveEstac"], "smape"),
    ]

    filas = []
    for nom, ini, fin in TRAMOS:
        print(f"\n{'='*100}\nTRAMO {nom}\n{'='*100}")
        s = v[v.paso_horas.between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
        y = s["real"].to_numpy()
        base = lad_cv(s, NUEVO_72, "lad")
        preds = {}
        print(f"{'configuracion':44s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s}  {'p(MAE)':>8s} {'p(MAPE)':>8s}")
        for etq, cols, modo in CONFS:
            pr = lad_cv(s, cols, modo)
            preds[etq] = pr
            linea = f"{etq:44s} {np.abs(y-pr).mean():8.2f} {mape(y,pr):8.2f} {smape(y,pr):8.2f}"
            if not etq.startswith("A "):
                p1, p2 = dm_metrica(y, base, pr, "mae")[1], dm_metrica(y, base, pr, "mape")[1]
                linea += f"  {p1:8.4f} {p2:8.4f}"
            print(linea)
            filas.append({"tramo": nom, "config": etq, "MAE": np.abs(y-pr).mean(),
                          "MAPE": mape(y, pr), "sMAPE": smape(y, pr)})

        cand = [c for c in preds if not c.startswith("A ")]
        ganador = min(cand, key=lambda c: mape(y, preds[c]))
        if mape(y, preds[ganador]) < mape(y, base):
            cols_g, modo_g = next((c, m) for e, c, m in CONFS if e == ganador)
            print(f"\n  Verificacion 10 semillas: '{ganador}'")
            A, G = [], []
            for s_ in SEMILLAS:
                pa, pg = lad_cv(s, NUEVO_72, "lad", s_), lad_cv(s, cols_g, modo_g, s_)
                A.append([np.abs(y-pa).mean(), mape(y, pa), smape(y, pa)])
                G.append([np.abs(y-pg).mean(), mape(y, pg), smape(y, pg)])
            A, G = np.array(A), np.array(G)
            for j, mn in enumerate(["MAE", "MAPE %", "sMAPE %"]):
                dl = G[:, j] - A[:, j]
                print(f"    {mn:8s} A={A[:,j].mean():7.3f}  nuevo={G[:,j].mean():7.3f}  "
                      f"delta={dl.mean():+7.3f}, mejor en {int((dl<0).sum())}/{len(SEMILLAS)}")
            ca, cg = causal(s, NUEVO_72, "lad"), causal(s, cols_g, modo_g)
            ok = ~np.isnan(ca) & ~np.isnan(cg)
            print(f"    desplegable: A MAPE={mape(y[ok],ca[ok]):.2f}%  nuevo MAPE={mape(y[ok],cg[ok]):.2f}%  "
                  f"DM p={dm_metrica(y[ok],ca[ok],cg[ok],'mape')[1]:.4f}")

    pd.DataFrame(filas).to_csv(RES / "variantes_nbx_72h.csv", index=False)
    print(f"\nGuardado: variantes_nbx_72h.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
