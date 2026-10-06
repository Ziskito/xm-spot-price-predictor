# -*- coding: utf-8 -*-
"""
Calibracion conforme por hora del dia ("Mondrian") de la banda de 24 h del motor.

La banda vigente (conectar_motor_ensamble.py) se calibra con un unico margen por dia para las 24 horas,
y su cobertura queda muy dispareja: 54 % a las 00:00, 66 % a las 19:00 y 91-94 % de 01:00 a 04:00
(objetivo 80 %). Como el motor mide la "confianza" por el ancho de banda, eso la hace poco honesta
en las horas extremas. Aqui se calibra un margen por hora del dia, con ventanas de 30 y 60 dias, y se
compara con la version vigente en cobertura (global y por hora), ancho y backtest del motor.
Salida: data/processed/resultados/calibracion_por_hora_24h.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import conectar_motor_ensamble as cme  # noqa: E402
from motor_decision import comparar_metodos_estable, elegir_mejor_metodo_estable, evaluar_backtest, generar_senales  # noqa: E402


def base_24h():
    ens = pd.read_csv(cme.RES / "informe_avance" / "ensamble_24h_desplegable.csv", parse_dates=["fecha_hora"])
    nb = pd.read_csv(cme.RES / "pronostico_con_bandas_2026_adaptativo.csv", parse_dates=["fecha_hora"])
    nb["off10"] = (nb["q10"] + nb["margen"]) - nb["q50"]
    nb["off90"] = (nb["q90"] - nb["margen"]) - nb["q50"]
    d = ens.merge(nb[["fecha_hora", "off10", "off90"]], on="fecha_hora", how="inner")
    d["q50"] = d["pred_desplegable"]
    d["q10_base"] = d["q50"] + np.minimum(d["off10"], 0)
    d["q90_base"] = d["q50"] + np.maximum(d["off90"], 0)
    d["hora"] = d["fecha_hora"].dt.hour
    return d


def evaluar(df, nombre, hist):
    df = df.sort_values("fecha_hora").reset_index(drop=True)
    y = df["real"]
    cubre = (y >= df["q10"]) & (y <= df["q90"])
    por_hora = cubre.groupby(df["fecha_hora"].dt.hour).mean() * 100
    corte = df["fecha_hora"].min() + (df["fecha_hora"].max() - df["fecha_hora"].min()) / 2
    fila = {"variante": nombre, "cobertura_pct": cubre.mean() * 100, "cob_hora_min": por_hora.min(),
            "cob_hora_max": por_hora.max(), "desv_cob_hora": por_hora.std(),
            "ancho_medio": (df["q90"] - df["q10"]).mean()}
    for rol in ("generador", "comercializador"):
        t = comparar_metodos_estable(df, rol, hist, corte=corte)
        m, ok = elegir_mejor_metodo_estable(t)
        bt = evaluar_backtest(df, generar_senales(df, m, rol, hist), rol)
        fila[f"{rol}_metodo"] = m
        fila[f"{rol}_ventaja"] = bt["ventaja_cop_kwh"]
        fila[f"{rol}_peor_mitad"] = float(t.loc[t.metodo == m, "peor_mitad"].iloc[0])
    return fila


if __name__ == "__main__":
    hist = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
    vigente = pd.read_csv(cme.RES / "pronostico_ensamble_bandas_24h_2026.csv", parse_dates=["fecha_hora"])
    filas = [evaluar(vigente, "contrato activo", hist)]
    for dias in (30, 60):
        cme.VENTANA_DIAS = dias
        cal = cme.calibrar(base_24h(), "hora")
        filas.append(evaluar(cal, f"por hora del dia, {dias} d", hist))
    t = pd.DataFrame(filas)
    t.to_csv(cme.RES / "calibracion_por_hora_24h.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
