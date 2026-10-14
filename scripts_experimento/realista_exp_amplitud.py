# -*- coding: utf-8 -*-
"""
Experimento (14-oct-2026, a pedido de Juan): reducir la magnitud de los picos que el predespacho de XM exagera.

Diagnostico previo: en 18 de los 43 dias que Juan senalo, el pronostico acierta la forma del dia pero sus cambios son 2 a 15
veces mas grandes que los reales; el predespacho trae ese mismo exceso (p. ej. 9-mar: 16x) y el modelo lo hereda.

Todas las variantes son causales y se aplican sobre el pronostico realista de 24 h en produccion
(ensamble_24h_realista_2026.csv). Para cada dia d (ventana de 01:00 a 00:00):
  m_d = media del pronostico del dia;  s_d = desviacion del pronostico dentro del dia
  S_d = mediana de la desviacion diaria del precio REAL en las 30 ventanas ya publicadas (hasta d-5)
  r_d = s_d / S_d  (cuanto mas grande viene el dia que lo normal)
  pronostico' = m_d + k_d * (pronostico - m_d)
Variantes de k_d:
  A  regla: k = 1 si r <= tau, si no (tau / r)^alpha. (tau, alpha) se eligen en 2020-2025 aplicando la regla al
     predespacho solo (minimo MAE); no se mira 2026.
  B  k unico aprendido sobre la marcha: cada dia, el k de una grilla que minimiza el MAE de las ventanas publicadas
     de 2026 (hasta d-5), con 1 los primeros 21 dias.
  C  regresion de amplitud entrenada con 2019-2025 sobre el predespacho: (real - media real del dia) ~ b(r) x
     (predespacho - media del predespacho), con b lineal en log r; se aplica al pronostico de 2026.
Metricas: MAE global, MAE en los 43 dias senalados, MAE en dias de alta volatilidad real (rango > 300), MAE en rampas
(10 % mayor de cambios de 3 h), fraccion de rampa anticipada y amplitud; DM frente al pronostico sin cambio.
Salida: resultados/escenario_realista/mejoras/amplitud_metricas.csv y amplitud_predicciones.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
import escenario_realista_24h as er  # noqa: E402

SENALADOS = {1: [17, 20, 21, 22, 26, 27, 28], 2: [6, 11, 12, 19, 20, 23, 24, 25, 26, 28],
             3: [1, 2, 4, 5, 6, 7, 8, 9, 14, 17, 18, 19], 4: [11, 17, 23, 25, 30], 5: [2, 3, 22, 29], 6: [16, 23],
             7: [9, 21, 27], 8: [2, 3]}


def amplitud_normal(real_dia_std, retraso=5, ventana=30):
    """S_d: mediana de la desviacion diaria del real en las ventanas ya publicadas (hasta d - retraso)."""
    return real_dia_std.rolling(ventana, min_periods=10).median().shift(retraso)


def aplicar(p, dia, k):
    m = p.groupby(dia).transform("mean")
    return m + k * (p - m)


def ramp_metrics(df, col):
    dr = df.groupby("dia")["real"].diff(3)
    dp = df.groupby("dia")[col].diff(3)
    top = (dr.abs() >= dr.abs().quantile(0.9)) & dr.notna()
    return np.abs(df["real"] - df[col])[top].mean(), (dp[top] / dr[top]).median()


if __name__ == "__main__":
    # ---------------- historia 2019-2025: predespacho y real por ventana ----------------
    d = rc.tabla_realista()[["fecha_hora", "precio_bolsa", "pre_obj"]].dropna()
    d["dia"] = (d["fecha_hora"] - pd.Timedelta(hours=1)).dt.normalize()
    d = d[d.groupby("dia")["precio_bolsa"].transform("size") == 24]
    h = d[d["dia"] < "2026-01-01"].copy()
    h_std_real = h.groupby("dia")["precio_bolsa"].std()
    S_h = amplitud_normal(h_std_real)
    s_pre = h.groupby("dia")["pre_obj"].transform("std")
    h["r"] = (s_pre / h["dia"].map(S_h)).to_numpy()
    h = h.dropna(subset=["r"])
    h = h[h["dia"] >= "2020-01-01"]
    # A: grilla de (tau, alpha) en 2020-2025
    mejor = (np.abs(h["pre_obj"] - h["precio_bolsa"]).mean(), 99, 0)
    for tau in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        for alpha in (0.25, 0.5, 0.75, 1.0):
            k = np.where(h["r"] <= tau, 1.0, (tau / h["r"]) ** alpha)
            mae = np.abs(aplicar(h["pre_obj"], h["dia"], pd.Series(k, index=h.index)) - h["precio_bolsa"]).mean()
            if mae < mejor[0]:
                mejor = (mae, tau, alpha)
    print(f"A: en 2020-2025 el predespacho sin cambio tiene MAE {np.abs(h['pre_obj'] - h['precio_bolsa']).mean():.2f}; "
          f"mejor regla tau={mejor[1]}, alpha={mejor[2]} -> {mejor[0]:.2f}", flush=True)
    TAU, ALPHA = mejor[1], mejor[2]
    # C: regresion de amplitud en 2019-2025 (desviacion real ~ b(r) * desviacion predespacho, b = a + c log r)
    dev_r = h["precio_bolsa"] - h.groupby("dia")["precio_bolsa"].transform("mean")
    dev_p = h["pre_obj"] - h.groupby("dia")["pre_obj"].transform("mean")
    lr = np.log(h["r"].clip(lower=0.05))
    X = np.column_stack([dev_p, dev_p * lr])
    coef = np.linalg.lstsq(X, dev_r, rcond=None)[0]
    print(f"C: b(r) = {coef[0]:.3f} + {coef[1]:.3f} log r (r=1 -> {coef[0]:.2f}; r=5 -> {coef[0] + coef[1] * np.log(5):.2f})", flush=True)

    # ---------------- 2026: pronostico en produccion ----------------
    e = pd.read_csv(rc.RAIZ / "data/processed/resultados/ensamble_24h_realista_2026.csv", parse_dates=["fecha_hora"])
    e["dia"] = pd.to_datetime(e["dia"])
    e = e[e.groupby("dia")["real"].transform("size") == 24].reset_index(drop=True)
    real_std = pd.concat([h_std_real, e.groupby("dia")["real"].std()]).sort_index()
    real_std = real_std[~real_std.index.duplicated()]
    S = amplitud_normal(real_std)
    s_pred = e.groupby("dia")["pred_realista"].transform("std")
    e["r"] = (s_pred / e["dia"].map(S)).to_numpy()
    base = e["pred_realista"]
    kA = np.where(e["r"] <= TAU, 1.0, (TAU / e["r"]) ** ALPHA)
    e["A_regla"] = aplicar(base, e["dia"], pd.Series(kA, index=e.index))
    bC = np.clip(coef[0] + coef[1] * np.log(e["r"].clip(lower=0.05)), 0.0, 1.2)
    e["C_regresion"] = aplicar(base, e["dia"], pd.Series(bC, index=e.index))
    # B: k unico aprendido sobre la marcha con las ventanas publicadas de 2026
    grilla = np.round(np.arange(0.5, 1.21, 0.05), 2)
    dias = np.sort(e["dia"].unique())
    m_d = base.groupby(e["dia"]).transform("mean")
    errores = {k: np.abs(m_d + k * (base - m_d) - e["real"]).groupby(e["dia"]).sum() for k in grilla}
    errores = pd.DataFrame(errores)
    kB = pd.Series(1.0, index=dias)
    for i, dd in enumerate(dias):
        conocidos = dias[:max(0, i - 4)]
        if len(conocidos) >= 21:
            kB[dd] = errores.loc[conocidos].sum().idxmin()
    e["B_aprendido"] = m_d + e["dia"].map(kB).to_numpy() * (base - m_d)
    print("B: k aprendido por mes:", kB.groupby(pd.DatetimeIndex(kB.index).month).mean().round(2).to_dict(), flush=True)

    sen = e["dia"].isin([pd.Timestamp(2026, m, x) for m, xs in SENALADOS.items() for x in xs])
    rango = e.groupby("dia")["real"].transform(lambda x: x.max() - x.min())
    volatil = rango > 300
    ev = e["dia"] >= "2026-01-15"
    filas = []
    for c in ("pred_realista", "A_regla", "B_aprendido", "C_regresion", "Predespacho"):
        q = e[ev]
        mr, fr = ramp_metrics(q, c)
        amp = (q.groupby("dia")[c].std() / q.groupby("dia")["real"].std()).median()
        f = {"variante": c, "MAE": np.abs(q[c] - q["real"]).mean(),
             "MAE_43_dias": np.abs(q[c] - q["real"])[sen[ev]].mean(),
             "MAE_dias_volatiles": np.abs(q[c] - q["real"])[volatil[ev]].mean(),
             "MAE_rampas": mr, "rampa_anticipada": fr, "amplitud_mediana": amp}
        f["DM_t"], f["DM_p"] = er.dm(q["real"].to_numpy(), q[c].to_numpy(), q["pred_realista"].to_numpy())
        filas.append(f)
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / "amplitud_metricas.csv", index=False)
    e.to_csv(rc.MEJ / "amplitud_predicciones.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    print("dias con encogimiento (regla A):", int((pd.Series(kA).groupby(e["dia"].values).first() < 1).sum()), "de", e["dia"].nunique())
    q = e[sen]
    por = q.groupby("dia").apply(lambda g: pd.Series({c: np.abs(g[c] - g["real"]).mean() for c in ("pred_realista", "A_regla", "B_aprendido", "C_regresion")}))
    print(por.round(1).to_string())
