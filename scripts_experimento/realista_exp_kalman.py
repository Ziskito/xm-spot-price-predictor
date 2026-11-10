# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: combinador de Kalman frente al QRA causal vigente (5-nov-2026).

Vigente: y = w_pre * predespacho + w_cat * CatBoost + b, pesos por franja de 6 h (g4) reestimados cada dia por QRA
(regresion de la mediana) con TODOS los dias ya publicados (margen de 5 dias). Pesos que cambian lento.
Kalman: el mismo modelo, pero los pesos son un estado que camina al azar (w_t = w_t-1 + ruido de proceso Q) y se corrigen
con cada hora nueva publicada (ruido de medicion R). Q/R = q fija cuanto "persigue" el filtro lo reciente.
  nivel   modelo en COP/kWh
  log     modelo en log(precio) (errores mas parecidos a gaussianos, que es lo que supone Kalman)
  robusto log + innovacion recortada a 3 desviaciones (Huber): una hora atipica no arrastra los pesos
q se elige en la previa (origen 5, jul-2024 a jun-2025, ultimos 6 meses) y se aplica tal cual a 2026; tambien se
muestra la rejilla completa en 2026 como sensibilidad (no para elegir).
Mismas horas (15-ene a 5-ago-2026), Diebold-Mariano contra el vigente.
Salida: resultados/escenario_realista/mejoras/kalman_predicciones.csv y kalman_metricas.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
import escenario_realista_24h as er  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN, catboost  # noqa: E402

VIG = "Pre+Cat mensual · mae · g4"
COLS = ["Predespacho", "CatBoost_mensual"]
MARGEN_DIAS = 5
REJILLA = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]


def kalman(d, q, log=False, robusto=False):
    """d: horas en orden, con COLS, real, dia, g4. Devuelve la prediccion causal (solo dias con precio publicado)."""
    X = d[COLS].to_numpy(float)
    y = d["real"].to_numpy(float)
    if log:
        X, y = np.log(np.clip(X, 1, None)), np.log(np.clip(y, 1, None))
    X = np.column_stack([X, np.ones(len(X))])
    dia, g = d["dia"].to_numpy(), d["g4"].to_numpy()
    dias = np.sort(np.unique(dia))
    k = X.shape[1]
    # estado inicial: mitad y mitad, sin intercepto; R inicial de la varianza de y (se re-estima en linea)
    w = {gg: np.array([0.5, 0.5, 0.0]) for gg in np.unique(g)}
    P0 = np.diag([1.0, 1.0, (np.nanstd(y) ** 2) * 0.01])
    Pm = {gg: P0.copy() for gg in w}
    R = {gg: np.nanvar(y) * 0.05 for gg in w}
    pred = np.full(len(d), np.nan)
    pos = {x: np.flatnonzero(dia == x) for x in dias}
    for i, hoy in enumerate(dias):
        if i >= MARGEN_DIAS:                       # se publica el precio del dia i-5: actualizar con sus horas
            for j in pos[dias[i - MARGEN_DIAS]]:
                if np.isnan(y[j]) or np.isnan(X[j]).any():
                    continue
                gg, x = g[j], X[j]
                Pm[gg] = Pm[gg] + q * R[gg] * np.eye(k)
                e = y[j] - x @ w[gg]
                S = x @ Pm[gg] @ x + R[gg]
                if robusto:
                    e = np.clip(e, -3 * np.sqrt(S), 3 * np.sqrt(S))
                K = Pm[gg] @ x / S
                w[gg] = w[gg] + K * e
                Pm[gg] = Pm[gg] - np.outer(K, x) @ Pm[gg]
                R[gg] = 0.99 * R[gg] + 0.01 * min(e * e, 9 * S)     # varianza de medicion, olvido lento
        for j in pos[hoy]:
            pred[j] = X[j] @ w[g[j]]
    return np.exp(pred) if log else pred


if __name__ == "__main__":
    t0 = time.time()
    pan = pd.read_csv(rc.MEJ / "mezcla_predicciones.csv", parse_dates=[0], index_col=0)
    pan["dia"] = pd.to_datetime(pan["dia"])
    pan["g4"] = pan.index.hour // 6
    # previa (origen 5) con el CatBoost entrenado antes de ella, como en realista_exp_mezcla.py
    d = rc.tabla_realista().set_index("fecha_hora")
    from ensamble_24h_enero import origen5
    from ensamble_con_predespacho_24h import imar
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    tr = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag96h"])
    previa["CatBoost_mensual"] = catboost(tr, d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"]),
                                          conjuntos(d)["R+P"]).reindex(previa.index)
    previa = previa.dropna(subset=COLS + ["real"]).sort_index()
    previa["dia"] = (previa.index - pd.Timedelta(hours=1)).normalize()
    previa["g4"] = previa.index.hour // 6
    print(f"previa lista ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # 1) elegir q en la previa (ultimos 6 meses, tras calentar el filtro)
    selp = previa.index >= "2025-01-01"
    eleccion = {}
    filas_prev = []
    for nom, kw in (("nivel", {}), ("log", {"log": True}), ("robusto", {"log": True, "robusto": True})):
        maes = {}
        for q in REJILLA:
            p = kalman(previa, q, **kw)
            maes[q] = float(np.nanmean(np.abs(p[selp] - previa["real"].to_numpy()[selp])))
            filas_prev.append({"variante": nom, "q": q, "MAE_previa": maes[q]})
        eleccion[nom] = min(maes, key=maes.get)
        print(f"  previa {nom}: " + ", ".join(f"q={q:g} {m:.2f}" for q, m in maes.items()), flush=True)

    # 2) 2026: el filtro arranca con lo aprendido en la previa (stream continuo previa -> 2026)
    pan_k = pan.dropna(subset=COLS).sort_index()
    flujo = pd.concat([previa[COLS + ["real", "dia", "g4"]], pan_k[COLS + ["real", "dia", "g4"]]])
    n_prev = len(previa)
    filas = []
    ev = pan_k.index >= "2026-01-15"
    real = pan_k["real"].to_numpy()
    base = pan_k[VIG].to_numpy()
    sel = ev & ~np.isnan(real) & ~np.isnan(base)
    filas.append({"variante": "vigente QRA diario (g4, MAE)", "q": np.nan, **rc.metricas(real[sel], base[sel]),
                  "DM_t_vs_vigente": np.nan, "DM_p": np.nan})
    out = pd.DataFrame({"real": real, VIG: base}, index=pan_k.index)
    for nom, kw in (("nivel", {}), ("log", {"log": True}), ("robusto", {"log": True, "robusto": True})):
        for q in REJILLA:
            p = kalman(flujo, q, **kw)[n_prev:]
            elegido = q == eleccion[nom]
            col = f"Kalman {nom} q={q:g}" + (" (elegido en previa)" if elegido else "")
            out[col] = p
            ok = sel & ~np.isnan(p)
            t, pv = er.dm(real[ok], p[ok], base[ok])
            filas.append({"variante": col, "q": q, **rc.metricas(real[ok], p[ok]), "DM_t_vs_vigente": t, "DM_p": pv})
    # por tramo (ene-abr / may-ago) para el vigente y los elegidos
    t = pd.DataFrame(filas)
    for nom_t, m in (("MAE ene-abr", out.index < "2026-05-01"), ("MAE may-ago", out.index >= "2026-05-01")):
        t[nom_t] = [float(np.nanmean(np.abs(out.loc[m & sel, c if c in out else VIG] - out.loc[m & sel, "real"])))
                    for c in [VIG] + [r for r in t["variante"][1:]]]
    out.to_csv(rc.MEJ / "kalman_predicciones.csv")
    t.to_csv(rc.MEJ / "kalman_metricas.csv", index=False)
    pd.DataFrame(filas_prev).to_csv(rc.MEJ / "kalman_previa.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
