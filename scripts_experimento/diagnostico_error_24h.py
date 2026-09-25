# -*- coding: utf-8 -*-
"""
Diagnostico: DONDE vive el error del ensamble de 24h, y que haria falta para llegar al 8%.

Motivo: mejorar_ensamble_24h.py mostro que el ensamble esta en meseta -- agregar cualquiera de los
9 votantes extra disponibles (incluido el LEAR24 nuevo) no da diferencia significativa, la seleccion
greedy gana apenas 0.21 COP/kWh (p=0.26) y afinar la granularidad de los pesos gana 0.28 (p=0.29).
Es decir: el limite ya no esta en COMO se combinan los modelos, sino en los modelos base.

Antes de seguir probando combinaciones a ciegas conviene medir donde se concentra el error, porque
eso dice si el objetivo de 8% es alcanzable y por que via. Se analiza:
  1) por hora del dia (hipotesis: el pico vespertino domina);
  2) por nivel de precio (hipotesis: los picos de precio dominan);
  3) por mes (hipotesis: hay regimenes peores);
  4) cuanto baja el MAPE global si se "arreglara" el X% peor de las horas -- esto acota cuanto
     margen real hay y donde tendria que venir.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import RES

OBJETIVO = 8.0


def main():
    v4 = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"]).sort_values("fecha_hora")
    d = v4.dropna(subset=["real", "pred_ensamble"]).copy()
    d["err"] = (d["real"] - d["pred_ensamble"]).abs()
    d["ape"] = d["err"] / d["real"] * 100
    d["hora"] = d["fecha_hora"].dt.hour
    d["mes"] = d["fecha_hora"].dt.to_period("M").astype(str)
    mape_global = d["ape"].mean()
    print(f"Horas: {len(d):,} | MAPE global = {mape_global:.2f}%  (objetivo {OBJETIVO}%)\n")

    # ---------- 1) por hora del dia ----------
    print(f"{'='*88}\n1) ERROR POR HORA DEL DIA\n{'='*88}")
    g = d.groupby("hora").agg(MAE=("err", "mean"), MAPE=("ape", "mean"),
                              precio=("real", "mean"), n=("err", "size"))
    g["aporte_%"] = d.groupby("hora")["ape"].sum() / d["ape"].sum() * 100
    print(g.round(2).to_string())
    peores = g["MAPE"].nlargest(6).index.tolist()
    print(f"\nLas 6 horas peores ({sorted(peores)}) aportan "
          f"{g.loc[peores, 'aporte_%'].sum():.1f}% del error total (serian 25% si fuera parejo)")

    # ---------- 2) por nivel de precio ----------
    print(f"\n{'='*88}\n2) ERROR POR NIVEL DE PRECIO (deciles del precio real)\n{'='*88}")
    d["decil"] = pd.qcut(d["real"], 10, labels=False, duplicates="drop") + 1
    g2 = d.groupby("decil").agg(precio_medio=("real", "mean"), MAE=("err", "mean"),
                                MAPE=("ape", "mean"), n=("err", "size"))
    g2["aporte_%"] = d.groupby("decil")["ape"].sum() / d["ape"].sum() * 100
    print(g2.round(2).to_string())

    # ---------- 3) por mes ----------
    print(f"\n{'='*88}\n3) ERROR POR MES\n{'='*88}")
    g3 = d.groupby("mes").agg(precio_medio=("real", "mean"), MAE=("err", "mean"),
                              MAPE=("ape", "mean"), n=("err", "size"))
    g3["aporte_%"] = d.groupby("mes")["ape"].sum() / d["ape"].sum() * 100
    print(g3.round(2).to_string())

    # ---------- 4) cuanto margen hay ----------
    print(f"\n{'='*88}\n4) QUE HARIA FALTA PARA LLEGAR AL {OBJETIVO}%\n{'='*88}")
    ape = d["ape"].to_numpy()
    orden = np.sort(ape)[::-1]
    print("Si se eliminara por completo el error de las peores X% de horas:")
    for pct in [1, 2, 5, 10, 20]:
        k = int(len(ape) * pct / 100)
        nuevo = ape.copy()
        nuevo[np.argsort(-ape)[:k]] = 0
        print(f"  peores {pct:2d}% ({k:4d} horas, todas con APE > {orden[k-1]:6.1f}%)"
              f" -> MAPE global {nuevo.mean():5.2f}%")
    reduccion = (mape_global - OBJETIVO) / mape_global * 100
    print(f"\nPara pasar de {mape_global:.2f}% a {OBJETIVO}% hace falta reducir el error un "
          f"{reduccion:.1f}% en terminos relativos.")
    # cuanta reduccion uniforme haria falta
    print(f"Repartido de forma uniforme, cada prediccion tendria que mejorar un {reduccion:.1f}%.")
    # que pasa si solo se mejora el pico vespertino
    msk = d["hora"].isin(peores)
    for mejora in [0.25, 0.5, 1.0]:
        nuevo = d["ape"].to_numpy().copy()
        nuevo[msk.to_numpy()] *= (1 - mejora)
        print(f"  si SOLO las 6 horas peores mejoraran un {int(mejora*100):3d}% -> "
              f"MAPE global {nuevo.mean():5.2f}%")

    g.to_csv(RES / "diagnostico_error_24h_por_hora.csv")
    g2.to_csv(RES / "diagnostico_error_24h_por_decil.csv")
    print("\nGuardado: diagnostico_error_24h_por_hora.csv, diagnostico_error_24h_por_decil.csv")


if __name__ == "__main__":
    main()
