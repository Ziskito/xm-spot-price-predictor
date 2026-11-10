# -*- coding: utf-8 -*-
"""
Evaluacion completa de la curva de la CREG 101 127 antes de adoptarla (10-nov-2026, pedido de Juan): que mejore el
pronostico de 24 h y de 72 h y que no empeore las bandas ni las senales del motor.

Variantes: base (produccion), regla50 (embalse agregado, X = 50 %), planta (planta por planta: X = NEP de cada
planta, tercer punto en su NPV) y ambas (regla50 + planta). Todas valen 0 antes del 10-sep-2026: lo anterior es identico.
  1) 24 h, pronostico final (combinador): MAE, RMSE, sesgo y DM diario (HAC) contra base
  2) 24 h, banda conforme del 80 % (realista_motor.bandas, la de produccion, sobre el q50 de cada variante): cobertura,
     ancho y puntaje de Winkler (alfa 0,2)
  3) 72 h (dias 2 y 3): lo mismo con realista_exp_regla_72h.py
  4) motor: senales de produccion (rodante 30 dias + filtro intradia + filtro de contrato con el MC de XM) sobre el
     q50 de cada variante, para generador y comercializador: ventaja en precio real, frecuencia de accion, ganancia
     frente al MC por hora y % de horas en que la senal cambia frente a base
Tramos: 1-oct al ultimo precio publicado (con ejemplos de la regla) y 10-sep en adelante.
Salida: resultados/escenario_realista/regimen_2026/regla_evaluacion_completa.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_exp_regimen_2026 as rg  # noqa: E402
from realista_motor import bandas  # noqa: E402
import realista_motor_umbrales_alternativos as ua  # noqa: E402

VARS = ("base", "regla50", "planta", "ambas")
TRAMOS = {"1-oct en adelante": "2026-10-01 01:00", "10-sep en adelante": "2026-09-10 01:00"}


def dm(y, a, b, idx):
    d = pd.Series(np.abs(a - y) - np.abs(b - y), index=idx).groupby((idx - pd.Timedelta(hours=1)).normalize()).mean()
    if (d == 0).all():
        return 0.0, 1.0
    r = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return float(r.tvalues[0]), float(r.pvalues[0])


def winkler(y, lo, hi, a=0.2):
    return float(np.mean((hi - lo) + 2 / a * (lo - y) * (y < lo) + 2 / a * (y - hi) * (y > hi)))


def puntual_y_bandas(P, horizonte, filas, ini):
    """P: {variante: DataFrame con real, q50, q10, q90}, mismo indice."""
    b = P["base"]
    sel = b.index[(b.index >= ini) & b["real"].notna()]
    y = b.loc[sel, "real"].to_numpy()
    for v, p in P.items():
        q = p.loc[sel, "q50"].to_numpy()
        lo, hi = p.loc[sel, "q10"].to_numpy(), p.loc[sel, "q90"].to_numpy()
        t, pv = (np.nan, np.nan) if v == "base" else dm(y, q, b.loc[sel, "q50"].to_numpy(), sel)
        filas.append({"horizonte": horizonte, "desde": str(ini)[:10], "variante": v, "horas": len(sel), "hasta": str(sel.max()),
                      "MAE": np.abs(q - y).mean(), "RMSE": np.sqrt(((q - y) ** 2).mean()), "sesgo": (q - y).mean(),
                      "DM_t": t, "DM_p": pv, "cobertura_%": ((y >= lo) & (y <= hi)).mean() * 100,
                      "ancho": (hi - lo).mean(), "winkler": winkler(y, lo, hi)})


def motor(P, filas, ini, real):
    base_s = {}
    for v, p in P.items():
        df = p.loc["2026-01-15":, ["real", "q50"]].dropna(subset=["q50"]).rename_axis("fecha_hora").reset_index()
        previo = real[real.index <= df["fecha_hora"].min() - pd.Timedelta(hours=73)]
        bajo, alto = ua.rodante(df, previo, 30)
        mc = pd.Series(ua.mc_disponible(df["fecha_hora"]), index=df.index)
        ev = (df["fecha_hora"] >= ini).to_numpy()
        for rol in ("generador", "comercializador"):
            s = ua.senal(df, bajo, alto, rol)
            fav = "comprar" if rol == "comercializador" else "vender"
            no_mc = (df["q50"] > mc) if rol == "comercializador" else (df["q50"] < mc)
            s[(s == fav) & no_mc] = "esperar"
            dfe, se = df[ev], s[ev]
            r = ua.md.evaluar_backtest(dfe, se, rol)
            act = (se == fav).to_numpy() & dfe["real"].notna().to_numpy()
            gana = (mc[ev] - dfe["real"]).to_numpy() if rol == "comercializador" else (dfe["real"] - mc[ev]).to_numpy()
            con = dfe["real"].notna().to_numpy()
            if v == "base":
                base_s[rol] = se.to_numpy()
            filas.append({"horizonte": "motor 24 h", "desde": str(ini)[:10], "variante": v, "rol": rol,
                          "ventaja_cop_kwh": r["ventaja_cop_kwh"], "frecuencia_accion_%": 100 * r["frecuencia_accion"],
                          "vs_MC_por_hora": float(np.where(act, gana, 0.0)[con].mean()),
                          "senal_distinta_a_base_%": float((se.to_numpy() != base_s[rol]).mean() * 100)})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    filas = []
    # 24 h
    P24 = {}
    for v in VARS:
        x = pd.read_csv(rg.SAL / f"{v}.csv", parse_dates=[0], index_col=0)
        x = x[x.index >= "2026-01-15"].dropna(subset=["pred"])
        bd = bandas(x.index, x["real"].to_numpy(), x["pred"].to_numpy()).set_index("fecha_hora")
        P24[v] = bd[["real", "q10", "q50", "q90"]]
    comun = P24["base"].index
    for v in VARS:
        comun = comun.intersection(P24[v].index)
    P24 = {v: p.loc[comun] for v, p in P24.items()}
    for ini in TRAMOS.values():
        puntual_y_bandas(P24, "24 h", filas, pd.Timestamp(ini))
    # 72 h
    for k in (1, 2):
        P72 = {}
        for v in VARS:
            x = pd.read_csv(rg.SAL / f"72h_{v}.csv", parse_dates=[0], index_col=0)
            P72[v] = x[x["k"] == k][["real", "q10", "q50", "q90"]]
        comun = P72["base"].index
        for v in VARS:
            comun = comun.intersection(P72[v].index)
        P72 = {v: p.loc[comun] for v, p in P72.items()}
        puntual_y_bandas(P72, f"72 h, dia {k + 1}", filas, pd.Timestamp("2026-10-01 01:00") + pd.Timedelta(days=k))
    # motor
    real = ua.precio_real()
    for ini in TRAMOS.values():
        motor(P24, filas, pd.Timestamp(ini), real)
    t = pd.DataFrame(filas)
    t.to_csv(rg.SAL / "regla_evaluacion_completa.csv", index=False)
    pd.set_option("display.width", 260)
    pron = t[~t["horizonte"].str.startswith("motor")].drop(columns=["rol", "ventaja_cop_kwh", "frecuencia_accion_%", "vs_MC_por_hora",
                                                                       "senal_distinta_a_base_%"], errors="ignore")
    print(pron.round(2).to_string(index=False))
    mot = t[t["horizonte"].str.startswith("motor")][["desde", "rol", "variante", "ventaja_cop_kwh", "frecuencia_accion_%", "vs_MC_por_hora",
                                                     "senal_distinta_a_base_%"]]
    print(mot.round(2).to_string(index=False))
