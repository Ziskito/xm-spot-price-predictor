# -*- coding: utf-8 -*-
"""
Motor de decision en el escenario realista (solo datos publicos a las 00:00 de D).

Contratos de 24 h en las mismas horas (15-ene a 5-ago-2026), todos con bandas del mismo tipo para aislar el efecto
del pronostico: q50 = pronostico; [q10, q90] = q50 * exp(-+m), con m el cuantil conforme 80 % de |log(real/q50)| de la
misma hora en los 60 dias previos con precio ya publicado (ventanas hasta D-5).
  A  ensamble 7 optimista (contrato vigente del motor)
  B  ensamble 7 realista
  C  predespacho + CatBoost mensual (combinador MAE)
  D  precio real (techo: pronostico perfecto)
Para cada rol: metodo elegido por el criterio de estabilidad (con y sin filtro intradia), ventaja del año y peor
mitad. El calentamiento de los umbrales rodantes usa el precio real de los 30 dias previos al 15-ene, que en el
escenario realista llega hasta el 11-ene (diferencia de 4 dias, despreciable para un percentil de 30 dias).
Salida: resultados/escenario_realista/mejoras/motor_realista.csv
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

NIVEL, VENTANA, RETRASO = 0.80, 60, 5


def bandas(fecha, real, q50):
    d = pd.DataFrame({"fecha_hora": fecha, "real": real, "q50": q50}).sort_values("fecha_hora").reset_index(drop=True)
    d["dia"] = (d["fecha_hora"] - pd.Timedelta(hours=1)).dt.normalize()
    d["hora"] = d["fecha_hora"].dt.hour
    d["s"] = np.abs(np.log(d["real"].clip(lower=1)) - np.log(d["q50"].clip(lower=1)))
    m = np.full(len(d), np.nan)
    for dia in d["dia"].unique():
        hist = d[(d["dia"] <= dia - pd.Timedelta(days=RETRASO)) & (d["dia"] > dia - pd.Timedelta(days=RETRASO + VENTANA))]
        hoy = (d["dia"] == dia).to_numpy()
        for h, g in hist.groupby("hora"):
            sel = hoy & (d["hora"] == h).to_numpy()
            if len(g) >= 7:
                n = len(g)
                m[sel] = np.quantile(g["s"], min(np.ceil((n + 1) * NIVEL) / n, 1.0))
    d["m"] = pd.Series(m).ffill().bfill().fillna(0.3).to_numpy()
    d["q10"], d["q90"] = d["q50"] * np.exp(-d["m"]), d["q50"] * np.exp(d["m"])
    return d[["fecha_hora", "real", "q10", "q50", "q90"]]


if __name__ == "__main__":
    m = pd.read_csv(rc.MEJ / "mezcla_predicciones.csv", parse_dates=[0], index_col=0)
    ref = rc.referencia()
    vig = pd.read_csv(RAIZ / "data/processed/resultados/pronostico_ensamble_bandas_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    idx = m.index[(m.index >= "2026-01-15")].intersection(vig.index).intersection(ref.dropna(subset=["ens7_realista"]).index)
    real = m.loc[idx, "real"]
    fuentes = {"A ens7 optimista": vig.loc[idx, "q50"], "B ens7 realista": ref.loc[idx, "ens7_realista"],
               "C predespacho + CatBoost": m.loc[idx, "Pre+Cat mensual · mae · g4"], "D precio real (techo)": real}
    hist = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
    previo = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])
                        for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]
                       ).drop_duplicates("fecha_hora").set_index("fecha_hora")["precio_bolsa"].sort_index()
    previo = previo[previo.index <= pd.Timestamp("2026-01-15") - pd.Timedelta(hours=73)]
    filas = []
    for nom, q in fuentes.items():
        df = bandas(idx, real.to_numpy(), q.to_numpy())
        cob = ((df["real"] >= df["q10"]) & (df["real"] <= df["q90"])).mean() * 100
        for rol in ("generador", "comercializador"):
            for intradia in (False, True):
                t = md.comparar_metodos_estable(df, rol, hist, precio_previo=previo, intradia=intradia)
                met, ok = md.elegir_mejor_metodo_estable(t)
                fila = t[t["metodo"] == met].iloc[0]
                s = md.generar_senales(df, met, rol, hist, precio_previo=previo, intradia=intradia)
                tot = md.evaluar_backtest(df, s, rol)
                filas.append({"pronostico": nom, "MAE": float(np.abs(df["q50"] - df["real"]).mean()), "cobertura_%": cob,
                              "rol": rol, "filtro_intradia": intradia, "metodo": met, "estable": ok,
                              "ventaja_año": tot["ventaja_cop_kwh"], "frecuencia_%": tot["frecuencia_accion"] * 100,
                              "ventaja_H1": fila["ventaja_H1"], "ventaja_H2": fila["ventaja_H2"], "peor_mitad": fila["peor_mitad"]})
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / "motor_realista.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(1).to_string(index=False))
