# -*- coding: utf-8 -*-
"""
Metricas de FORMA del dia (Maciejowska, Lipiecki & Uniejewski 2026) sobre 2026 para todos los
modelos de 24h y los ensambles. Nunca se habian medido para los ensambles v1 y v4.

  Corr-f : correlacion de Spearman entre el perfil horario real y el pronosticado, promedio por dia
           (1 = el pronostico ordena las 24 horas igual que la realidad).
  MHD    : error en HORAS al ubicar el minimo y el maximo del dia (suma de ambos, promedio por dia).
  MPD    : cuanto dinero se deja en la mesa por operar en la hora equivocada: diferencia entre el
           precio real en la hora optima y el precio real en la hora que el pronostico senalo.
  Ratio de amplitud: (max-min) pronosticado / (max-min) real, promedio por dia. < 1 = curva aplanada.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metricas_decision import evaluar_modelo

RES = Path(r"C:\Users\mgdbj\xm-spot-price-predictor") / "data/processed/resultados"


def ratio_amplitud(fecha_hora, real, pred):
    d = pd.DataFrame({"f": pd.to_datetime(fecha_hora).normalize(), "r": real, "p": pred})
    g = d.groupby("f").agg(rmax=("r", "max"), rmin=("r", "min"), pmax=("p", "max"), pmin=("p", "min"))
    g = g[(g.rmax - g.rmin) > 1]
    return float(((g.pmax - g.pmin) / (g.rmax - g.rmin)).median())


def cargar_24h():
    c = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    o6 = c[c["origen"] == "Origen 6"]
    df = o6.pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
    df = df.join(o6.groupby("fecha_hora")["real"].first())
    v1 = pd.read_csv(RES / "stacking_ensamble_cv_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    v4 = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    df = df.join(v1["pred_ensamble"].rename("Ensamble v1 (NNLS)")).join(v4["pred_ensamble"].rename("Ensamble v4 (QRA franjas)"))
    return df.dropna()


def main():
    df = cargar_24h()
    modelos = [c for c in df.columns if c != "real"]
    filas = []
    for m in modelos:
        r = evaluar_modelo(df.index, df["real"].values, df[m].values, m)
        r["ratio_amplitud"] = ratio_amplitud(df.index, df["real"].values, df[m].values)
        filas.append(r)
    res = pd.DataFrame(filas).sort_values("mae")
    print(res.round(3).to_string(index=False))
    res.to_csv(RES / "metricas_forma_2026.csv", index=False)


if __name__ == "__main__":
    main()
