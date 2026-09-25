# -*- coding: utf-8 -*-
"""
Problema 1 visto SOLO en 2026 (el ano que se va a publicar): sesgo mensual y por nivel de precio de
cada modelo de 24h y de los ensambles. Sesgo = pronostico - real (positivo = sobreestima).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metricas_forma_2026 import cargar_24h

RES = Path(r"C:\Users\mgdbj\xm-spot-price-predictor") / "data/processed/resultados"
MODELOS = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS", "v1", "v4"]


def main():
    df = cargar_24h().rename(columns={"Ensamble v1 (NNLS)": "v1", "Ensamble v4 (QRA franjas)": "v4"})
    sesgo = df[MODELOS].sub(df["real"], axis=0)
    mes = df.index.month
    print("Sesgo medio mensual (COP/kWh, pronostico - real; positivo = sobreestima)")
    t = sesgo.groupby(mes).mean()
    t["precio medio real"] = df["real"].groupby(mes).mean()
    print(t.round(1).to_string())
    print("\nSesgo medio mensual en % del precio real")
    print((sesgo.div(df["real"], axis=0) * 100).groupby(mes).mean().round(1).to_string())
    tercil = pd.qcut(df["real"], 3, labels=["bajo", "medio", "alto"])
    print("\nSesgo medio por tercil de precio real (COP/kWh)")
    print(sesgo.groupby(tercil, observed=True).mean().round(1).to_string())
    t.to_csv(RES / "sesgo_2026_mensual.csv")


if __name__ == "__main__":
    main()
