# -*- coding: utf-8 -*-
"""
¿Decide mejor el motor si usa la probabilidad de desplome y de pico? (prueba del 7-oct-2026)

Sobre la senal del metodo elegido por el criterio de estabilidad, las horas con probabilidad >= u se fuerzan:
  generador:        P(pico) >= u -> "vender"         P(desplome) >= u -> "retener"
  comercializador:  P(desplome) >= u -> "comprar"    P(pico) >= u -> "evitar_compra"
(si las dos superan u, gana la mayor). Tambien se prueba la politica "solo eventos" (sin umbrales de precio).
Las probabilidades son causales (reentrenamiento semanal, eventos_escenarios_24h.py). Se mide igual que el
motor: ventaja y frecuencia de accion en cada mitad del periodo, peor mitad y estabilidad.
Uso: python motor_con_eventos.py [archivo_de_probabilidades]
Salida: data/processed/resultados/motor_con_eventos.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import motor_decision as md  # noqa: E402

RES = RAIZ / "data" / "processed" / "resultados"
ACCION = {"generador": ("vender", "retener"), "comercializador": ("comprar", "evitar_compra")}


def con_eventos(senal, pd_, pp, rol, u, solo=False):
    s = pd.Series("esperar", index=senal.index) if solo else senal.copy()
    favorable, desfavorable = ACCION[rol]
    if rol == "generador":
        sube, baja = pp >= u, pd_ >= u
        s[sube & (pp >= pd_)] = favorable
        s[baja & (pd_ > pp)] = desfavorable
    else:
        baja, sube = pd_ >= u, pp >= u
        s[baja & (pd_ >= pp)] = favorable
        s[sube & (pp > pd_)] = desfavorable
    return s


def evaluar(df, s, rol, corte):
    m1 = df["fecha_hora"] < corte
    h1 = md.evaluar_backtest(df[m1], s[m1], rol)
    h2 = md.evaluar_backtest(df[~m1], s[~m1], rol)
    tot = md.evaluar_backtest(df, s, rol)
    ok = lambda f: 0.10 <= f <= 0.40
    return {"ventaja_año": tot["ventaja_cop_kwh"], "frecuencia_año_%": tot["frecuencia_accion"] * 100,
            "ventaja_H1": h1["ventaja_cop_kwh"], "frec_H1_%": h1["frecuencia_accion"] * 100,
            "ventaja_H2": h2["ventaja_cop_kwh"], "frec_H2_%": h2["frecuencia_accion"] * 100,
            "peor_mitad": np.nanmin([h1["ventaja_cop_kwh"], h2["ventaja_cop_kwh"]]),
            "estable": bool(h1["ventaja_cop_kwh"] >= 0 and h2["ventaja_cop_kwh"] >= 0
                            and ok(h1["frecuencia_accion"]) and ok(h2["frecuencia_accion"]))}


if __name__ == "__main__":
    archivo = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "probabilidades_eventos_24h_2026.csv"
    prob = pd.read_csv(archivo, parse_dates=["fecha_hora"]).drop_duplicates("fecha_hora").set_index("fecha_hora")
    hist = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
    previo = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])
                        for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]
                       ).drop_duplicates("fecha_hora").set_index("fecha_hora")["precio_bolsa"].sort_index()
    df, _ = md.cargar_fuente_pronostico(RAIZ, "24h")
    pd_ = pd.Series(prob["p_desplome"].reindex(df["fecha_hora"]).fillna(0).to_numpy(), index=df.index)
    pp = pd.Series(prob["p_pico"].reindex(df["fecha_hora"]).fillna(0).to_numpy(), index=df.index)
    corte = df["fecha_hora"].min() + (df["fecha_hora"].max() - df["fecha_hora"].min()) / 2
    filas = []
    for rol in ("generador", "comercializador"):
        met, _ = md.elegir_mejor_metodo_estable(md.comparar_metodos_estable(df, rol, hist, precio_previo=previo))
        base = md.generar_senales(df, met, rol, hist, precio_previo=previo)
        filas.append({"rol": rol, "politica": f"{met} (vigente)", **evaluar(df, base, rol, corte)})
        for u in (0.3, 0.5):
            filas.append({"rol": rol, "politica": f"{met} + eventos (p >= {u})", **evaluar(df, con_eventos(base, pd_, pp, rol, u), rol, corte)})
            filas.append({"rol": rol, "politica": f"solo eventos (p >= {u})", **evaluar(df, con_eventos(base, pd_, pp, rol, u, solo=True), rol, corte)})
    t = pd.DataFrame(filas)
    t.to_csv(RES / "motor_con_eventos.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"probabilidades: {archivo.name}")
    print(t.round(1).to_string(index=False))
