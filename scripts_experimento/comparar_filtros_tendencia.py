# -*- coding: utf-8 -*-
"""
Comparacion de filtros de tendencia frente al Savitzky-Golay de la caracterizacion (OE1).

Metricas, las mismas del notebook 03 (sin modificarlo) mas una de suavizado:
  * retraso: horas hasta que la tendencia alcanza el 90 % de la subida de El Nino 2023-2024,
    contadas desde 60 dias antes del pico del tramo jun-2023 a mar-2024 (notebook 03, celda 29);
  * residuo: desviacion estandar de precio - tendencia (notebook 03, celda 29);
  * fuga: cuanto ciclo corto (diario y semanal) se cuela en la tendencia = desv. estandar de
    tendencia - promedio movil centrado de 168 h de la tendencia;
  * retraso medio: desfase (h) que maximiza la correlacion de la tendencia con una referencia sin desfase
    (Butterworth de fase cero con corte de 720 h) en toda la serie. El retraso de la celda 29 mide un solo
    cruce de umbral y puede adelantarse por un pico aislado; este promedia todo el periodo.

Comparar con un solo ajuste por filtro es injusto: el que suaviza menos siempre reacciona antes y deja
menos residuo. Por eso cada familia se barre en su parametro de suavizado y se reporta el punto con la
MISMA fuga que el Savitzky-Golay de referencia (721 h, orden 3): a igual suavizado, gana el que llega antes.

Familias fuera de linea (usan pasado y futuro; sirven para caracterizar): promedio movil centrado,
Savitzky-Golay, Butterworth de fase cero (filtfilt), wavelet sym8 (aproximacion de nivel L) y suavizador
de Kalman (tendencia local lineal, RTS). Familias causales (solo pasado; usables en tiempo real):
promedio movil, media exponencial, Savitzky-Golay causal (polinomio evaluado en el extremo), Butterworth
causal y filtro de Kalman.
Salidas: data/processed/resultados/filtros_tendencia.csv y filtros_tendencia.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, lfilter, savgol_coeffs, savgol_filter

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "data" / "processed" / "resultados"
ZOOM = slice("2023-06-01", "2024-03-01")


def cargar_precio():
    p = pd.concat([pd.read_csv(RAIZ / "data/precio_bolsa_2019_2025.csv"),
                   pd.read_csv(RAIZ / "data/precio_bolsa_2026.csv")], ignore_index=True)
    p["fecha_hora"] = pd.to_datetime(p["fecha_hora"], format="%Y-%m-%d %H:%M:%S")
    return p.sort_values("fecha_hora").set_index("fecha_hora")["precio_bolsa_cop_kwh"]


REF = None


def retraso_medio(tend, ref, max_h=1200, paso=4):
    ok = ~np.isnan(tend) & ~np.isnan(ref)
    a, b = tend[ok], ref[ok]
    mejor, k_mejor = -2.0, 0
    for k in range(0, max_h + 1, paso):
        c = np.corrcoef(a[k:], b[:len(b) - k])[0, 1]
        if c > mejor:
            mejor, k_mejor = c, k
    return k_mejor


def metricas(p, tend):
    tend = pd.Series(np.asarray(tend, float), index=p.index)
    sub = p.loc[ZOOM]
    pico = sub.idxmax()
    previo = sub.loc[:pico - pd.Timedelta(days=45)].mean()
    nivel = sub.loc[pico - pd.Timedelta(days=3): pico + pd.Timedelta(days=3)].mean()
    umbral = previo + 0.9 * (nivel - previo)
    inicio = pico - pd.Timedelta(days=60)
    cruce = tend.loc[inicio:][tend.loc[inicio:] >= umbral]
    retraso = (cruce.index[0] - inicio).total_seconds() / 3600 if len(cruce) else np.nan
    residuo = (p - tend).dropna().std()
    fuga = (tend - tend.rolling(168, center=True, min_periods=84).mean()).dropna().std()
    return retraso, residuo, fuga, retraso_medio(tend.to_numpy(), REF)


def kalman_tendencia(y, q, suavizar):
    """Tendencia local lineal: nivel y pendiente, con ruido solo en la pendiente (tendencia suave)."""
    n = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.0, q]])
    R = 1.0
    x, P = np.array([y[0], 0.0]), np.eye(2) * 1e3
    xs, Ps, xp_s, Pp_s = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2)), np.zeros((n, 2, 2))
    for t in range(n):
        xp, Pp = F @ x, F @ P @ F.T + Q
        k = Pp[:, 0] / (Pp[0, 0] + R)
        x = xp + k * (y[t] - xp[0])
        P = Pp - np.outer(k, Pp[0, :])
        xs[t], Ps[t], xp_s[t], Pp_s[t] = x, P, xp, Pp
    if not suavizar:
        return xs[:, 0]
    xsm = xs.copy()
    for t in range(n - 2, -1, -1):
        C = Ps[t] @ F.T @ np.linalg.inv(Pp_s[t + 1])
        xsm[t] = xs[t] + C @ (xsm[t + 1] - xp_s[t + 1])
    return xsm[:, 0]


def wavelet_aprox(y, nivel, ond="sym8"):
    coefs = pywt.wavedec(y, ond, level=nivel, mode="symmetric")
    coefs = [coefs[0]] + [np.zeros_like(c) for c in coefs[1:]]
    return pywt.waverec(coefs, ond, mode="symmetric")[:len(y)]


def sg_causal(y, ventana, orden=2):
    c = savgol_coeffs(ventana, orden, pos=ventana - 1, use="conv")
    out = np.convolve(y, c, mode="full")[:len(y)]
    out[:ventana - 1] = np.nan
    return out


def familias(p):
    y = p.to_numpy(float)
    yn = (y - y.mean()) / y.std()
    desn = lambda z: z * y.std() + y.mean()
    impar = lambda w: int(w) | 1
    fl = {
        "Promedio móvil centrado": ("ventana (h)", [168, 240, 336, 480, 720, 960, 1200, 1440],
                                    lambda w: p.rolling(int(w), center=True, min_periods=int(w) // 2).mean()),
        "Savitzky-Golay (orden 3)": ("ventana (h)", [241, 361, 481, 601, 721, 961, 1201, 1441, 1921],
                                     lambda w: savgol_filter(y, impar(w), 3)),
        "Butterworth fase cero (orden 2)": ("periodo de corte (h)", [168, 240, 336, 480, 720, 960, 1440, 2160],
                                            lambda pc: filtfilt(*butter(2, 2.0 / pc), y)),
        "Wavelet sym8 (aproximación)": ("nivel", [6, 7, 8, 9, 10, 11], lambda L: wavelet_aprox(y, int(L))),
        "Kalman (suavizador RTS)": ("q (ruido de pendiente)", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11],
                                    lambda q: desn(kalman_tendencia(yn, q, True))),
    }
    ca = {
        "Promedio móvil causal": ("ventana (h)", [24, 48, 96, 168, 240, 336, 480, 720],
                                  lambda w: p.rolling(int(w), min_periods=int(w)).mean()),
        "Media exponencial": ("vida media (h)", [12, 24, 48, 96, 168, 240, 336, 480],
                              lambda hl: p.ewm(halflife=hl, adjust=False).mean()),
        "Savitzky-Golay causal (orden 2)": ("ventana (h)", [337, 481, 721, 961, 1441, 2161, 2881, 3601, 4321],
                                           lambda w: sg_causal(y, impar(w))),
        "Butterworth causal (orden 2)": ("periodo de corte (h)", [48, 96, 168, 240, 336, 480, 720, 960],
                                         lambda pc: lfilter(*butter(2, 2.0 / pc), y)),
        "Kalman (filtro causal)": ("q (ruido de pendiente)", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12],
                                   lambda q: desn(kalman_tendencia(yn, q, False))),
    }
    return fl, ca


def interpolar_a_fuga(t, fuga_obj):
    """Metricas de la familia al nivel de fuga objetivo (interpolacion lineal en log-fuga)."""
    t = t.dropna(subset=["retraso"]).sort_values("fuga")
    if not (t["fuga"].min() <= fuga_obj <= t["fuga"].max()):
        return np.nan, np.nan, np.nan
    x = np.log(t["fuga"].to_numpy())
    return tuple(float(np.interp(np.log(fuga_obj), x, t[c].to_numpy())) for c in ("retraso", "residuo", "retraso_medio_h"))


if __name__ == "__main__":
    p = cargar_precio()
    REF = filtfilt(*butter(2, 2.0 / 720), p.to_numpy(float))
    ref = metricas(p, savgol_filter(p.to_numpy(float), 721, 3))
    ma = metricas(p, p.rolling(720, center=True, min_periods=360).mean())
    print(f"Reproduccion del notebook 03 -> SG 721/3: retraso {ref[0]:.0f} h (medio {ref[3]} h), residuo {ref[1]:.2f} | "
          f"promedio movil 30 d: retraso {ma[0]:.0f} h, residuo {ma[1]:.2f}")
    fl, ca = familias(p)
    filas = []
    for tipo, fams in (("fuera de linea", fl), ("causal", ca)):
        for nombre, (param, valores, f) in fams.items():
            for v in valores:
                r, s, g, rm = metricas(p, f(v))
                filas.append({"tipo": tipo, "familia": nombre, "parametro": param, "valor": v,
                              "retraso_h": r, "residuo": s, "fuga": g, "retraso_medio_h": rm})
            print(f"  {tipo:14} {nombre} listo", flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(RES / "filtros_tendencia.csv", index=False)

    print(f"\nA igual fuga que el Savitzky-Golay de referencia ({ref[2]:.2f} COP/kWh):")
    resumen = []
    for (tipo, fam), g in t.groupby(["tipo", "familia"], sort=False):
        r, s, rm = interpolar_a_fuga(g.rename(columns={"retraso_h": "retraso"}), ref[2])
        resumen.append({"tipo": tipo, "familia": fam, "retraso_h_igual_fuga": r, "residuo_igual_fuga": s,
                        "retraso_medio_h_igual_fuga": rm})
        print(f"  {tipo:14} {fam:34} retraso El Niño {r:6.0f} h | retraso medio {rm:6.0f} h | residuo {s:7.2f}")
    pd.DataFrame(resumen).to_csv(RES / "filtros_tendencia_resumen.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    for ax, tipo in zip(axs, ("fuera de linea", "causal")):
        for fam, g in t[t["tipo"] == tipo].groupby("familia", sort=False):
            g = g.dropna(subset=["retraso_h"]).sort_values("fuga")
            ax.plot(g["fuga"], g["retraso_h"], "o-", ms=3.5, lw=1.4, label=fam)
        ax.axvline(ref[2], color="#888", ls="--", lw=1)
        if tipo == "fuera de linea":
            ax.plot(ref[2], ref[0], "k*", ms=13, label="Savitzky-Golay de OE1 (721 h)")
        ax.set_xscale("log")
        ax.set_xlabel("Fuga de ciclos cortos en la tendencia (COP/kWh, escala log.)")
        ax.set_ylabel("Horas hasta el 90 % de la subida")
        ax.set_title(("Fuera de línea" if tipo == "fuera de linea" else "Causales (tiempo real)")
                     + ": más abajo y a la izquierda es mejor", fontsize=11)
        ax.set_ylim(0, 1500)
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RES / "filtros_tendencia.png", dpi=160, facecolor="white")
    print("Guardado: filtros_tendencia.csv, filtros_tendencia_resumen.csv y filtros_tendencia.png")
