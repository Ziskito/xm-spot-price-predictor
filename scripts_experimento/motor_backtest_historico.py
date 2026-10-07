# -*- coding: utf-8 -*-
"""
Backtest del motor de decision en los origenes historicos 1-5 (pendiente de OE4).

Usa las predicciones del ensamble de ensamble_origenes_historicos.py (mediana y banda empirica
[q10, q90] con los residuos de los 30 dias previos, por hora). En cada origen, la referencia historica
de los metodos fijo/banda son SOLO los precios anteriores al inicio del origen (no 2019-2025 completo).

Preguntas:
  1. ¿Que metodo elige el criterio de estabilidad en cada año?
  2. ¿Se sostienen fuera de 2026 los metodos elegidos con 2026 (hibrido para el generador, rodante para
     el comercializador)?
  3. Prueba fuera de muestra de la hipotesis del grid search (formulada solo con 2026): ¿el generador
     gana con umbrales mas extremos y ventanas mas largas (rodante 10/90 con 60 dias; rodante 15/85
     con 14 dias e histeresis del 30 %)?
Salida: data/processed/resultados/motor_backtest_historico.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from motor_decision import comparar_metodos_estable, elegir_mejor_metodo_estable  # noqa: E402
from motor_grid_search import DEFECTO, Senales, mitad, valido  # noqa: E402

RES = RAIZ / "data" / "processed" / "resultados"
ELEGIDO_2026 = {"generador": "hibrido", "comercializador": "rodante"}
HIPOTESIS = {"rodante 10/90 v60d": ("rodante", 10, 90, 60, None, 0.0),
             "rodante 15/85 v14d hist30%": ("rodante", 15, 85, 14, None, 0.3)}

if __name__ == "__main__":
    pred = pd.read_csv(RES / "ensamble_origenes_historicos_predicciones.csv", parse_dates=["fecha_hora"])
    precios = pd.concat([pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", parse_dates=["fecha_hora"]),
                         pd.read_csv(RAIZ / "data/processed/dataset_maestro_2026.csv", parse_dates=["fecha_hora"])])
    filas = []
    for origen, df in pred.groupby("origen"):
        df = df.dropna(subset=["q10", "q90"]).sort_values("fecha_hora").reset_index(drop=True)
        hist = precios.loc[precios["fecha_hora"] < df["fecha_hora"].min(), "precio_bolsa"].to_numpy()
        # precio real anterior al origen: calienta la ventana de 30 dias de "rodante"/"hibrido"
        previo = precios.set_index("fecha_hora")["precio_bolsa"].sort_index()
        corte = df["fecha_hora"].min() + (df["fecha_hora"].max() - df["fecha_hora"].min()) / 2
        m1 = (df["fecha_hora"] < corte).to_numpy()
        gen = Senales(df, hist, previo=hist)
        for rol in ("generador", "comercializador"):
            t = comparar_metodos_estable(df, rol, hist, corte=corte, precio_previo=previo)
            elegido, estable = elegir_mejor_metodo_estable(t)
            politicas = {f"base {m}": DEFECTO[m] for m in DEFECTO} | HIPOTESIS
            for nombre, cfg in politicas.items():
                s = gen.generar(rol, *cfg)
                v1, f1 = mitad(df, s, rol, m1)
                v2, f2 = mitad(df, s, rol, ~m1)
                va, fa = mitad(df, s, rol, np.ones(len(df), bool))
                metodo = nombre.replace("base ", "")
                filas.append({"origen": origen, "rol": rol, "politica": nombre,
                              "elegida_por_estabilidad_este_año": metodo == elegido,
                              "elegida_con_2026": metodo == ELEGIDO_2026[rol],
                              "ventaja": va, "frecuencia": fa, "ventaja_H1": v1, "ventaja_H2": v2,
                              "frec_H1": f1, "frec_H2": f2,
                              "peor_mitad": np.nanmin([v1, v2]) if not (np.isnan(v1) and np.isnan(v2)) else np.nan,
                              "valida_ambas_mitades": valido(f1) and valido(f2)})
            print(f"{origen} · {rol}: el criterio de estabilidad elige {elegido} (estable={estable})", flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(RES / "motor_backtest_historico.csv", index=False)

    pd.set_option("display.width", 250)
    print("\nVentaja (COP/kWh) y peor mitad por política y año:")
    for rol in ("generador", "comercializador"):
        g = t[t["rol"] == rol]
        tab = g.pivot_table(index="politica", columns="origen", values="ventaja").round(1)
        peor = g.pivot_table(index="politica", columns="origen", values="peor_mitad").round(1)
        val = g.groupby("politica")["valida_ambas_mitades"].sum().rename("años válidos (de 5)")
        pos = g.assign(p=g["peor_mitad"] > 0).groupby("politica")["p"].sum().rename("años con peor mitad > 0")
        print(f"\n=== {rol}: ventaja del año")
        print(tab.assign(media=tab.mean(axis=1).round(1)).join(val).join(pos).to_string())
        print(f"--- {rol}: peor mitad")
        print(peor.assign(media=peor.mean(axis=1).round(1)).to_string())
