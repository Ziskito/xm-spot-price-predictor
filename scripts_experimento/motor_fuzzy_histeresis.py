# -*- coding: utf-8 -*-
"""
Experimento: histeresis (disparador Schmitt) y logica difusa en el motor de decision.

Pregunta (sugerida al mostrar el dashboard): con umbrales de corte duro la senal puede saltar entre
"actuar" y "esperar" de una hora a otra cuando la mediana ronda el umbral. ¿Lo corrige la
histeresis? ¿Mejora la decision una regla difusa en lugar de cortes duros?

Variantes, aplicadas sobre la referencia rodante (la que elige hoy el criterio de estabilidad):
  * histeresis h: entra a la accion al cruzar el umbral y sale solo al volver mas alla de
    umbral -/+ h*(alto - bajo). Causal: cada hora depende del estado de la anterior.
    "hibrido+histeresis" aplica ademas el filtro de banda ancha de 'hibrido'.
  * difusa alpha: pertenencia a caro (rampa p65->p85) y a barato (rampa p35->p15) sobre la
    ventana movil causal de 30 dias; confianza alta = rampa del ancho de banda (1 hasta su p60,
    0 desde su p90). Regla Mamdani: grado = min(caro, confianza) ; se actua si grado >= alpha.

No cambia src/motor_decision.py. Corre sobre los contratos activos de fuentes_pronostico.json.
Salida: data/processed/resultados/motor_fuzzy_histeresis.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from motor_decision import cargar_fuente_pronostico, evaluar_backtest, generar_senales, umbrales_rodantes  # noqa: E402

VENTANA_H = 30 * 24
ACCION = {"generador": ("vender", "retener"), "comercializador": ("evitar_compra", "comprar")}


def percentil_rodante(serie, p):
    return serie.shift(1).rolling(VENTANA_H, min_periods=VENTANA_H).quantile(p / 100)


def senal_histeresis(df, rol, h_frac, filtro_banda):
    caro_a, barato_a = ACCION[rol]
    bajo, alto = umbrales_rodantes(df["q50"], 30, 25, 75)
    q = df["q50"].to_numpy()
    lo, hi = bajo.to_numpy(), alto.to_numpy()
    estado, salida = "esperar", []
    for i in range(len(q)):
        if np.isnan(lo[i]) or np.isnan(hi[i]):
            estado = "esperar"
        else:
            h = h_frac * (hi[i] - lo[i])
            if estado == caro_a:
                estado = caro_a if q[i] >= hi[i] - h else "esperar"
            elif estado == barato_a:
                estado = barato_a if q[i] <= lo[i] + h else "esperar"
            if estado == "esperar":
                if q[i] >= hi[i]:
                    estado = caro_a
                elif q[i] <= lo[i]:
                    estado = barato_a
        salida.append(estado)
    s = pd.Series(salida, index=df.index)
    if filtro_banda:
        ancho = df["q90"] - df["q10"]
        s = s.mask(ancho > np.nanpercentile(ancho, 75), "esperar")
    return s


def rampa(x, x0, x1):
    """0 en x0 y 1 en x1 (sirve creciente o decreciente segun el orden de x0 y x1)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((x - x0) / (x1 - x0), 0, 1)


def senal_difusa(df, rol, alpha):
    caro_a, barato_a = ACCION[rol]
    q = df["q50"]
    mu_caro = rampa(q, percentil_rodante(q, 65), percentil_rodante(q, 85))
    mu_barato = rampa(q, percentil_rodante(q, 35), percentil_rodante(q, 15))
    ancho = df["q90"] - df["q10"]
    mu_conf = rampa(ancho, np.nanpercentile(ancho, 90), np.nanpercentile(ancho, 60))
    g_caro, g_barato = np.minimum(mu_caro, mu_conf), np.minimum(mu_barato, mu_conf)
    s = np.where((g_caro >= alpha) & (g_caro >= g_barato), caro_a,
                 np.where(g_barato >= alpha, barato_a, "esperar"))
    s = np.where(q.shift(1).rolling(VENTANA_H, min_periods=VENTANA_H).count() < VENTANA_H, "esperar", s)
    return pd.Series(s, index=df.index)


def metricas(df, s, rol):
    corte = df["fecha_hora"].min() + (df["fecha_hora"].max() - df["fecha_hora"].min()) / 2
    m1 = df["fecha_hora"] < corte
    bt = evaluar_backtest(df, s, rol)
    b1, b2 = evaluar_backtest(df[m1], s[m1], rol), evaluar_backtest(df[~m1], s[~m1], rol)
    validas = all(0.10 <= b["frecuencia_accion"] <= 0.40 for b in (b1, b2))
    peor = min(b1["ventaja_cop_kwh"], b2["ventaja_cop_kwh"])
    cambios = (s != s.shift(1)).iloc[1:].sum() / (len(s) / 24)
    accion = ACCION[rol][0] if rol == "generador" else "comprar"
    a = (s == accion).astype(int)
    rachas = a.groupby((a != a.shift()).cumsum()).sum()
    racha = rachas[rachas > 0].mean() if (rachas > 0).any() else np.nan
    ult = df["fecha_hora"] >= df["fecha_hora"].max() - pd.Timedelta(days=30)
    bu = evaluar_backtest(df[ult], s[ult], rol)
    pesos = bu["ventaja_cop_kwh"] * 100 * bu["horas_accion"] / 1e6 if bu["horas_accion"] else 0.0
    return {"ventaja": bt["ventaja_cop_kwh"], "frecuencia_pct": bt["frecuencia_accion"] * 100,
            "ventaja_H1": b1["ventaja_cop_kwh"], "ventaja_H2": b2["ventaja_cop_kwh"],
            "frec_H1_pct": b1["frecuencia_accion"] * 100, "frec_H2_pct": b2["frecuencia_accion"] * 100,
            "peor_mitad": peor, "estable": bool(validas and peor >= 0),
            "cambios_por_dia": cambios, "racha_media_h": racha, "ultimo_mes_MCOP": pesos}


if __name__ == "__main__":
    hist = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
    filas = []
    for hz in ("24h", "72h"):
        df, _ = cargar_fuente_pronostico(RAIZ, hz)
        df = df.sort_values("fecha_hora").reset_index(drop=True)
        for rol in ("generador", "comercializador"):
            variantes = {f"base {m}": generar_senales(df, m, rol, hist) for m in ("fijo", "rodante", "banda", "hibrido")}
            for h in (0.1, 0.2, 0.3):
                variantes[f"rodante+histeresis {h:.0%}"] = senal_histeresis(df, rol, h, False)
                variantes[f"hibrido+histeresis {h:.0%}"] = senal_histeresis(df, rol, h, True)
            for a in (0.3, 0.5, 0.7):
                variantes[f"difusa alpha={a}"] = senal_difusa(df, rol, a)
            for nombre, s in variantes.items():
                filas.append({"horizonte": hz, "rol": rol, "variante": nombre, **metricas(df, s, rol)})
    t = pd.DataFrame(filas)
    t.to_csv(RAIZ / "data/processed/resultados/motor_fuzzy_histeresis.csv", index=False)
    pd.set_option("display.width", 250)
    for (hz, rol), g in t.groupby(["horizonte", "rol"], sort=False):
        print(f"\n=== {hz} · {rol}")
        print(g.drop(columns=["horizonte", "rol"]).round(2).to_string(index=False))
