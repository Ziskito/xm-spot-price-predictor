# -*- coding: utf-8 -*-
"""
Grid search de hiperparametros del motor de decision, con validacion cruzada entre mitades de 2026.

Rejilla (sobre los contratos activos de fuentes_pronostico.json):
  referencia      fijo (percentiles 2019-2025) | rodante (ventana movil causal de q50)
  percentiles     10/90, 15/85, 20/80, 25/75, 30/70, 35/65
  ventana         7, 14, 30, 60, 90 dias (solo rodante)
  filtro banda    ninguno | esperar si ancho > percentil 60, 70, 75, 80, 90 del ancho
  histeresis      0, 10, 20, 30 % de la distancia entre umbrales
Los 4 metodos actuales son puntos de la rejilla (fijo, rodante, banda = fijo+filtro 75,
hibrido = rodante+filtro 75, todos con 25/75, 30 dias y sin histeresis).

Protocolo honesto: se elige la mejor configuracion en una mitad (maxima ventaja entre las que
actuan 10-40 % de las horas en esa mitad) y se mide en la OTRA mitad; se compara con el metodo
que el motor elige hoy (criterio de estabilidad, valores por defecto) en esa misma mitad.
Como referencia optimista se reporta tambien la mejor configuracion por maximin sobre el año.
Salida: data/processed/resultados/motor_grid_search.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from motor_decision import (cargar_fuente_pronostico, comparar_metodos_estable,  # noqa: E402
                            elegir_mejor_metodo_estable, evaluar_backtest, generar_senales)

PERCENTILES = [(10, 90), (15, 85), (20, 80), (25, 75), (30, 70), (35, 65)]
VENTANAS = [7, 14, 30, 60, 90]
FILTROS = [None, 60, 70, 75, 80, 90]
HISTERESIS = [0.0, 0.1, 0.2, 0.3]
ACCION = {"generador": ("vender", "retener"), "comercializador": ("evitar_compra", "comprar")}
DEFECTO = {"fijo": ("fijo", 25, 75, 30, None, 0.0), "rodante": ("rodante", 25, 75, 30, None, 0.0),
           "banda": ("fijo", 25, 75, 30, 75, 0.0), "hibrido": ("rodante", 25, 75, 30, 75, 0.0)}


class Senales:
    def __init__(self, df, hist):
        self.df, self.hist = df, hist
        self.q = df["q50"]
        self.ancho = (df["q90"] - df["q10"]).to_numpy()
        self._rod = {}

    def umbrales(self, ref, pb, pa, ventana):
        if ref == "fijo":
            n = len(self.q)
            return np.full(n, np.nanpercentile(self.hist, pb)), np.full(n, np.nanpercentile(self.hist, pa))
        for p in (pb, pa):
            if (ventana, p) not in self._rod:
                h = ventana * 24
                self._rod[(ventana, p)] = self.q.shift(1).rolling(h, min_periods=h).quantile(p / 100).to_numpy()
        return self._rod[(ventana, pb)], self._rod[(ventana, pa)]

    def generar(self, rol, ref, pb, pa, ventana, filtro, h):
        caro_a, barato_a = ACCION[rol]
        lo, hi = self.umbrales(ref, pb, pa, ventana)
        q = self.q.to_numpy()
        if h == 0:
            s = np.select([q >= hi, q <= lo], [caro_a, barato_a], default="esperar")
        else:
            s, estado = np.empty(len(q), dtype=object), "esperar"
            for i in range(len(q)):
                if np.isnan(lo[i]) or np.isnan(hi[i]):
                    estado = "esperar"
                else:
                    d = h * (hi[i] - lo[i])
                    if estado == caro_a and q[i] < hi[i] - d:
                        estado = "esperar"
                    elif estado == barato_a and q[i] > lo[i] + d:
                        estado = "esperar"
                    if estado == "esperar":
                        estado = caro_a if q[i] >= hi[i] else (barato_a if q[i] <= lo[i] else "esperar")
                s[i] = estado
        s = pd.Series(s, index=self.df.index)
        if filtro is not None:
            s = s.mask(self.ancho > np.nanpercentile(self.ancho, filtro), "esperar")
        return s


def mitad(df, s, rol, m):
    b = evaluar_backtest(df[m], s[m], rol)
    return b["ventaja_cop_kwh"], b["frecuencia_accion"]


def valido(f):
    return 0.10 <= f <= 0.40


def nombre(cfg):
    ref, pb, pa, v, filt, h = cfg
    txt = f"{ref} {pb}/{pa}" + (f" v{v}d" if ref == "rodante" else "")
    return txt + (f" filtro{filt}" if filt else "") + (f" hist{int(h * 100)}%" if h else "")


if __name__ == "__main__":
    hist = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
    configs = [("fijo", pb, pa, 30, f, h) for (pb, pa), f, h in itertools.product(PERCENTILES, FILTROS, HISTERESIS)]
    configs += [("rodante", pb, pa, v, f, h) for (pb, pa), v, f, h in itertools.product(PERCENTILES, VENTANAS, FILTROS, HISTERESIS)]
    print(f"{len(configs)} configuraciones por caso")
    filas, resumen = [], []
    for hz in ("24h", "72h"):
        df, _ = cargar_fuente_pronostico(RAIZ, hz)
        df = df.sort_values("fecha_hora").reset_index(drop=True)
        corte = df["fecha_hora"].min() + (df["fecha_hora"].max() - df["fecha_hora"].min()) / 2
        m1 = (df["fecha_hora"] < corte).to_numpy()
        gen = Senales(df, hist)
        for met, cfg in DEFECTO.items():  # el generador reproduce exactamente al motor
            for rol in ACCION:
                assert (gen.generar(rol, *cfg) == generar_senales(df, met, rol, hist)).all(), (hz, met, rol)
        for rol in ACCION:
            res = []
            for cfg in configs:
                s = gen.generar(rol, *cfg)
                v1, f1 = mitad(df, s, rol, m1)
                v2, f2 = mitad(df, s, rol, ~m1)
                res.append({"horizonte": hz, "rol": rol, "config": nombre(cfg), "ventaja_H1": v1, "frec_H1": f1,
                            "ventaja_H2": v2, "frec_H2": f2})
            t = pd.DataFrame(res)
            filas.append(t)
            met_hoy, _ = elegir_mejor_metodo_estable(comparar_metodos_estable(df, rol, hist, corte=corte))
            hoy = t[t["config"] == nombre(DEFECTO[met_hoy])].iloc[0]
            v1ok, v2ok = t["frec_H1"].apply(valido), t["frec_H2"].apply(valido)
            sel1 = t[v1ok].sort_values("ventaja_H1", ascending=False).iloc[0]
            sel2 = t[v2ok].sort_values("ventaja_H2", ascending=False).iloc[0]
            ambos = t[v1ok & v2ok].assign(peor=lambda x: np.minimum(x["ventaja_H1"], x["ventaja_H2"]))
            opt = ambos.sort_values("peor", ascending=False).iloc[0]
            mejor_que_hoy = int(((ambos["ventaja_H1"] > hoy["ventaja_H1"]) & (ambos["ventaja_H2"] > hoy["ventaja_H2"])).sum())
            resumen.append({
                "caso": f"{hz} {rol}", "metodo_hoy": met_hoy,
                "hoy_H1": hoy["ventaja_H1"], "hoy_H2": hoy["ventaja_H2"],
                "ajuste_H1": sel1["config"], "ajuste_H1_en_H2": sel1["ventaja_H2"], "ajuste_H1_frec_H2": sel1["frec_H2"],
                "ajuste_H2": sel2["config"], "ajuste_H2_en_H1": sel2["ventaja_H1"], "ajuste_H2_frec_H1": sel2["frec_H1"],
                "optimista_anio": opt["config"], "optimista_H1": opt["ventaja_H1"], "optimista_H2": opt["ventaja_H2"],
                "validas_ambas_mitades": int(len(ambos)), "mejores_que_hoy_en_ambas": mejor_que_hoy})
    pd.concat(filas).to_csv(RAIZ / "data/processed/resultados/motor_grid_search.csv", index=False)
    r = pd.DataFrame(resumen)
    pd.set_option("display.width", 300)
    for _, x in r.iterrows():
        print(f"\n=== {x['caso']} | método actual: {x['metodo_hoy']} (H1 {x['hoy_H1']:.1f}, H2 {x['hoy_H2']:.1f})")
        print(f"  ajustado en H1: {x['ajuste_H1']:42} -> en H2: {x['ajuste_H1_en_H2']:7.1f} (frec {x['ajuste_H1_frec_H2']:.0%})  vs actual {x['hoy_H2']:.1f}")
        print(f"  ajustado en H2: {x['ajuste_H2']:42} -> en H1: {x['ajuste_H2_en_H1']:7.1f} (frec {x['ajuste_H2_frec_H1']:.0%})  vs actual {x['hoy_H1']:.1f}")
        print(f"  optimista (año, maximin): {x['optimista_anio']:32} H1 {x['optimista_H1']:.1f} / H2 {x['optimista_H2']:.1f}")
        print(f"  configs válidas en ambas mitades: {x['validas_ambas_mitades']} | mejores que la actual en ambas: {x['mejores_que_hoy_en_ambas']}")
    r.to_csv(RAIZ / "data/processed/resultados/motor_grid_search_resumen.csv", index=False)
