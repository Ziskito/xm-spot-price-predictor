# -*- coding: utf-8 -*-
"""
Experimento (19-oct-2026): bandas de incertidumbre que reaccionan mas rapido a un cambio de regimen. Desde la
activacion de la Resolucion CREG 101 127 de 2026 (9-sep) la banda de 24 h cubrio el 62 % en vez del 80 %, porque su
ancho sale de los errores de los 60 dias anteriores.

Mismo esquema de produccion (realista_motor.bandas): para cada dia y hora, m = cuantil conforme de |log(real/q50)| de
la misma hora en los dias ya publicados (retraso de 5 dias), banda q50 x exp(+-m). Variantes:
  ventana de 60 (produccion), 45, 30, 20 y 14 dias
  pesos exponenciales por antiguedad sobre 90 dias (vida media de 10 y 20 dias): cuantil ponderado
Metricas por periodo: cobertura, ancho medio y puntaje de intervalo (Winkler, alfa = 0,2; menor es mejor: penaliza el
ancho y cada vez que el precio real cae fuera).
Entrada: data/processed/resultados/pronostico_realista_bandas_24h_2026.csv (q50 publicado y precio real)
Salida: data/processed/resultados/escenario_realista/regimen_2026/bandas_metricas.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402

NIVEL, RETRASO, ALFA = 0.80, 5, 0.20
SAL = rc.MEJ.parent / "regimen_2026"
PERIODOS = {"2026 (15-ene a 15-oct)": ("2026-01-15", "2026-12-31"), "ene-ago": ("2026-01-15", "2026-08-31"),
            "1-8 sep": ("2026-09-01", "2026-09-08 23:00"), "desde 9 sep (CREG 101 127)": ("2026-09-09", "2026-12-31")}


def cuantil_ponderado(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return x[min(np.searchsorted(c, q), len(x) - 1)]


def bandas(fecha, real, q50, ventana=60, vida=None):
    d = pd.DataFrame({"fecha_hora": fecha, "real": real, "q50": q50}).sort_values("fecha_hora").reset_index(drop=True)
    d["dia"] = (d["fecha_hora"] - pd.Timedelta(hours=1)).dt.normalize()
    d["hora"] = d["fecha_hora"].dt.hour
    d["s"] = np.abs(np.log(d["real"].clip(lower=1)) - np.log(d["q50"].clip(lower=1)))
    m = np.full(len(d), np.nan)
    for dia in d["dia"].unique():
        hist = d[(d["dia"] <= dia - pd.Timedelta(days=RETRASO)) & (d["dia"] > dia - pd.Timedelta(days=RETRASO + ventana))]
        hist = hist[hist["s"].notna()]
        hoy = (d["dia"] == dia).to_numpy()
        for h, g in hist.groupby("hora"):
            if len(g) < 7:
                continue
            sel = hoy & (d["hora"] == h).to_numpy()
            if vida is None:
                n = len(g)
                m[sel] = np.quantile(g["s"], min(np.ceil((n + 1) * NIVEL) / n, 1.0))
            else:
                edad = ((dia - g["dia"]) / pd.Timedelta(days=1)).to_numpy()
                m[sel] = cuantil_ponderado(g["s"].to_numpy(), 0.5 ** (edad / vida), NIVEL)
    d["m"] = pd.Series(m).ffill().bfill().fillna(0.3).to_numpy()
    d["q10"], d["q90"] = d["q50"] * np.exp(-d["m"]), d["q50"] * np.exp(d["m"])
    return d.set_index("fecha_hora")


def metricas(b):
    filas = []
    for nombre, (a, z) in PERIODOS.items():
        x = b.loc[a:z].dropna(subset=["real"])
        y, lo, hi = x["real"], x["q10"], x["q90"]
        ancho = hi - lo
        winkler = ancho + (2 / ALFA) * ((lo - y).clip(lower=0) + (y - hi).clip(lower=0))
        filas.append({"periodo": nombre, "horas": len(x), "cobertura_%": ((y >= lo) & (y <= hi)).mean() * 100,
                      "ancho_medio": ancho.mean(), "winkler": winkler.mean()})
    return filas


if __name__ == "__main__":
    c = pd.read_csv(rc.RAIZ / "data/processed/resultados/pronostico_realista_bandas_24h_2026.csv", parse_dates=["fecha_hora"])
    variantes = {"60 dias (produccion)": dict(ventana=60), "45 dias": dict(ventana=45), "30 dias": dict(ventana=30),
                 "20 dias": dict(ventana=20), "14 dias": dict(ventana=14),
                 "90 dias, vida media 20": dict(ventana=90, vida=20), "90 dias, vida media 10": dict(ventana=90, vida=10)}
    filas = []
    for nombre, kw in variantes.items():
        b = bandas(c["fecha_hora"], c["real"].to_numpy(), c["q50"].to_numpy(), **kw)
        filas += [{"variante": nombre, **f} for f in metricas(b)]
    t = pd.DataFrame(filas)
    SAL.mkdir(parents=True, exist_ok=True)
    t.to_csv(SAL / "bandas_metricas.csv", index=False)
    pd.set_option("display.width", 200)
    for p in PERIODOS:
        print("\n" + p)
        print(t[t["periodo"] == p].drop(columns="periodo").round(1).to_string(index=False))
