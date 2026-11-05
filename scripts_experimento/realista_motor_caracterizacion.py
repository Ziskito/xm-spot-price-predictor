# -*- coding: utf-8 -*-
"""
Motor de decision: variantes sugeridas por la caracterizacion (notebooks 03 y 03b, 5-nov-2026).

La caracterizacion muestra un ciclo semanal claro (fines de semana y festivos ~12 % mas baratos) y un ciclo diario cuya
amplitud cambia con el regimen. El motor vigente (umbral rodante de 30 dias + filtro intradia) compara cada hora con
todas las horas del ultimo mes, sin distinguir el tipo de dia. Se prueban, sobre los mismos contratos de 24 h de
realista_motor.py (C predespacho + CatBoost, el pronostico en produccion, y D precio real como techo):
  base     rodante + intradia (vigente)
  V1       umbral rodante por tipo de dia: percentiles de las horas del ultimo mes del mismo tipo (laborable / no laborable)
  V2       filtro semanal: la senal favorable solo en el tipo de dia favorable (generador vende en laborables, el
           comercializador compra en no laborables) y la desfavorable solo en el opuesto; analogo al filtro intradia
  V3       umbral rodante por hora: percentiles de la misma hora en el ultimo mes (quita el ciclo diario del umbral)
Mismo criterio de estabilidad del motor (peor mitad, frecuencia 10-40 % en cada mitad).
Salida: resultados/escenario_realista/mejoras/motor_caracterizacion.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import motor_decision as md  # noqa: E402
import realista_comun as rc  # noqa: E402
from realista_motor import bandas  # noqa: E402

VENTANA = 30


def rodante_por_grupo(q50, previo_fecha, previo_val, fechas, grupo_de, min_horas=24 * 5):
    """Percentiles 25/75 causales de las ultimas VENTANA*24 horas, solo entre horas del mismo grupo."""
    s = pd.concat([pd.Series(previo_val, index=previo_fecha), pd.Series(q50.to_numpy(), index=fechas)])
    g = grupo_de(s.index)
    bajo, alto = pd.Series(np.nan, index=s.index), pd.Series(np.nan, index=s.index)
    tiempos = s.index.to_numpy()
    for k in np.unique(g):
        sel = g == k
        sub = s[sel]
        # ventana de tiempo (no de conteo): el grupo ocupa una fraccion de las horas del mes
        r = sub.shift(1).rolling(f"{VENTANA}D", min_periods=min_horas)
        bajo[sel], alto[sel] = r.quantile(0.25).to_numpy(), r.quantile(0.75).to_numpy()
    n = len(previo_val)
    return bajo.iloc[n:].to_numpy(), alto.iloc[n:].to_numpy()


def senal(df, bajo, alto, rol):
    p = df["q50"].to_numpy()
    caro, barato = p >= alto, p <= bajo
    et = ["vender", "retener"] if rol == "generador" else ["evitar_compra", "comprar"]
    return pd.Series(np.select([caro, barato], et, default="esperar"), index=df.index)


def filtro_semanal(df, s, rol, no_lab):
    favorable, desfavorable = ("vender", "retener") if rol == "generador" else ("comprar", "evitar_compra")
    bien = ~no_lab if rol == "generador" else no_lab
    s = s.copy()
    s[(s == favorable).to_numpy() & ~bien] = "esperar"
    s[(s == desfavorable).to_numpy() & bien] = "esperar"
    return s


def evaluar(df, s, rol):
    corte = df["fecha_hora"].min() + (df["fecha_hora"].max() - df["fecha_hora"].min()) / 2
    m1 = (df["fecha_hora"] < corte).to_numpy()
    h1, h2 = md.evaluar_backtest(df[m1], s[m1], rol), md.evaluar_backtest(df[~m1], s[~m1], rol)
    tot = md.evaluar_backtest(df, s, rol)
    ok = all(0.10 <= h["frecuencia_accion"] <= 0.40 and h["ventaja_cop_kwh"] >= 0 for h in (h1, h2))
    return {"ventaja_año": tot["ventaja_cop_kwh"], "frecuencia_%": tot["frecuencia_accion"] * 100,
            "ventaja_H1": h1["ventaja_cop_kwh"], "ventaja_H2": h2["ventaja_cop_kwh"],
            "frec_H1_%": h1["frecuencia_accion"] * 100, "frec_H2_%": h2["frecuencia_accion"] * 100,
            "peor_mitad": min(h1["ventaja_cop_kwh"], h2["ventaja_cop_kwh"]), "estable": ok}


if __name__ == "__main__":
    m = pd.read_csv(rc.MEJ / "mezcla_predicciones.csv", parse_dates=[0], index_col=0)
    ref = rc.referencia()
    vig = pd.read_csv(RAIZ / "data/processed/resultados/pronostico_ensamble_bandas_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    idx = m.index[(m.index >= "2026-01-15")].intersection(vig.index).intersection(ref.dropna(subset=["ens7_realista"]).index)
    real = m.loc[idx, "real"]
    fuentes = {"C predespacho + CatBoost": m.loc[idx, "Pre+Cat mensual · mae · g4"], "D precio real (techo)": real}
    hist = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
    previo = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])
                        for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]
                       ).drop_duplicates("fecha_hora").set_index("fecha_hora")["precio_bolsa"].sort_index()
    previo = previo[previo.index <= pd.Timestamp("2026-01-15") - pd.Timedelta(hours=73)].iloc[-VENTANA * 24:]
    fest = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, usecols=["fecha_hora", "es_festivo"], parse_dates=["fecha_hora"])
                      for f in ("dataset_features_2019_2025.csv", "dataset_features_2026.csv")]
                     ).drop_duplicates("fecha_hora").set_index("fecha_hora")["es_festivo"]
    # el dia operativo va de 01:00 a 24:00 (la hora 00:00 es la ultima del dia anterior)
    dia_op = lambda ix: (pd.DatetimeIndex(ix) - pd.Timedelta(hours=1)).normalize()  # noqa: E731
    fest_dia = fest.groupby(fest.index.normalize()).max()
    no_laborable = lambda ix: ((dia_op(ix).dayofweek >= 5) | (fest_dia.reindex(dia_op(ix)).fillna(0).to_numpy() > 0)).astype(int)  # noqa: E731
    por_hora = lambda ix: pd.DatetimeIndex(ix).hour.to_numpy()  # noqa: E731

    filas = []
    for nom, q in fuentes.items():
        df = bandas(idx, real.to_numpy(), q.to_numpy())
        nl = no_laborable(df["fecha_hora"]).astype(bool)
        print(f"{nom}: precio real medio laborable {df.loc[~nl, 'real'].mean():.0f}, no laborable {df.loc[nl, 'real'].mean():.0f}")
        for rol in ("generador", "comercializador"):
            base = md.generar_senales(df, "rodante", rol, hist, precio_previo=previo, intradia=True)
            b1, a1 = rodante_por_grupo(df["q50"], previo.index, previo.to_numpy(), df["fecha_hora"], no_laborable)
            b3, a3 = rodante_por_grupo(df["q50"], previo.index, previo.to_numpy(), df["fecha_hora"], por_hora, min_horas=20)
            variantes = {
                "base: rodante + intradia (vigente)": base,
                "V1 umbral por tipo de dia": md.filtro_intradia(df, senal(df, b1, a1, rol), rol),
                "V2 base + filtro semanal": filtro_semanal(df, base, rol, nl),
                "V3 umbral por hora": md.filtro_intradia(df, senal(df, b3, a3, rol), rol),
            }
            # control: el vigente mas selectivo, con la frecuencia de V2 (sin mirar el tipo de dia)
            obj = (variantes["V2 base + filtro semanal"] == ("vender" if rol == "generador" else "comprar")).mean()
            mejor = None
            for k in np.arange(0, 16, 0.5):
                pb, pa = (25, 75 + k) if rol == "generador" else (25 - k, 75)
                s = md.generar_senales(df, "rodante", rol, hist, p_bajo=pb, p_alto=pa, precio_previo=previo, intradia=True)
                f = (s == ("vender" if rol == "generador" else "comprar")).mean()
                if mejor is None or abs(f - obj) < mejor[0]:
                    mejor = (abs(f - obj), s, pb, pa)
            variantes[f"control: vigente con p{mejor[2]:g}/p{mejor[3]:g} (frecuencia de V2)"] = mejor[1]
            for v, s in variantes.items():
                filas.append({"pronostico": nom, "rol": rol, "variante": v, **evaluar(df, s, rol)})
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / "motor_caracterizacion.csv", index=False)
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    print(t.round(1).to_string(index=False))
