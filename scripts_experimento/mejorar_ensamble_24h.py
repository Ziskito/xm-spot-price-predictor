# -*- coding: utf-8 -*-
"""
Busqueda sistematica de mejoras al ensamble de 24h (v4 = QRA por franja sobre 5 votantes,
MAE 42.51 / MAPE 11.27% / sMAPE 10.38%). Objetivo pedido: acercarse a 8% de error.

Tres frentes, todos sobre predicciones ya guardadas (no se reentrena ningun modelo base):

  A) VOTANTES. Al ensamble v4 solo entran 5 modelos (BASE5), pero en data/processed/resultados hay
     8 modelos EXTRA guardados (CatBoost, Hurdle, Markov-Switching, RecursiveLS day-ahead, XGB y
     CatBoost afinados con Optuna, y los directos CatB/Ridge de 24h) mas el LEAR24 recien
     construido. Se prueba agregarlos: de a uno (para ver cual aporta), y por seleccion greedy
     hacia adelante (para ver cuantos hacen falta antes de que la curva se aplane).

  B) GRANULARIDAD DE LOS PESOS. v4 usa 4 franjas horarias. Se prueban 1, 2, 3, 4, 6, 8, 12 y 24
     grupos (24 = un peso por hora del dia, que es como se hace en la literatura day-ahead).
     Mas grupos = mas especializacion pero menos datos por grupo; hay un optimo intermedio.

  C) LEAR24 POR FRANJA. LEAR24 es malo en promedio (MAE 53.64) pero muy bueno en las primeras
     horas del horizonte (MAE 33-41 en los pasos 2-6) y pesimo en el pico vespertino (99-123 en
     los pasos 18-20). Un combinador por franja deberia poder aprovechar solo la parte buena, asi
     que se mide explicitamente cuanto peso le da QRA en cada franja.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import cargar as cargar24, BASE5
from qra_bandas import ajustar_cuantil
from pesos_por_franja import dm
from o6_comun import RES


def qra_cv(d, cols, grupo, semilla=42):
    """QRA (cuantil 0.5) por grupo, con 5 pliegues por dia. Mismo esquema que el v4 vigente."""
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g = d[grupo].to_numpy()
    dia = d["dia"].to_numpy()
    P = np.full(len(d), np.nan)
    dias = np.sort(np.unique(dia))
    for dt in np.array_split(np.random.default_rng(semilla).permutation(dias), 5):
        te_m = np.isin(dia, dt)
        for gg in np.unique(g[te_m]):
            tr, te = (~te_m) & (g == gg), te_m & (g == gg)
            if tr.sum() < 20:
                continue
            w, b = ajustar_cuantil(X[tr], y[tr], 0.5)
            P[te] = X[te] @ w + b
    return P


def metricas(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    e = real - pred
    return {"MAE": float(np.abs(e).mean()),
            "MAPE_%": float(np.mean(np.abs(e) / real) * 100),
            "sMAPE_%": float(np.mean(np.abs(e) / ((np.abs(real) + np.abs(pred)) / 2)) * 100),
            "R2": float(1 - np.sum(e ** 2) / np.sum((real - real.mean()) ** 2))}


def main():
    d = cargar24().sort_index()
    lear = pd.read_csv(RES / "pronostico_lear_24h_2026.csv", parse_dates=["fecha_hora"])
    d = d.join(lear.set_index("fecha_hora")["LEAR24"], how="inner")
    d["dia"] = d.index.normalize()
    for n in [1, 2, 3, 4, 6, 8, 12, 24]:
        d[f"g{n}"] = d.index.hour // (24 // n)
    real = d["real"].to_numpy()
    disponibles = [c for c in d.columns if c not in ("real", "dia") and not c.startswith("g")]
    print(f"Filas: {len(d):,} | votantes disponibles: {disponibles}\n")

    base = qra_cv(d, BASE5, "g4")
    m0 = metricas(real, base)
    print(f"LINEA BASE (v4: QRA por 4 franjas, 5 votantes): MAE={m0['MAE']:.2f} "
          f"MAPE={m0['MAPE_%']:.2f}% sMAPE={m0['sMAPE_%']:.2f}% R2={m0['R2']:.3f}\n")

    # ---------- B) granularidad de los pesos ----------
    print(f"{'='*92}\nB) GRANULARIDAD DE LOS PESOS (con los 5 votantes actuales)\n{'='*92}")
    print(f"{'grupos':>7s} {'h/grupo':>8s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s}   DM vs v4")
    for n in [1, 2, 3, 4, 6, 8, 12, 24]:
        p = qra_cv(d, BASE5, f"g{n}")
        m = metricas(real, p)
        if n == 4:
            print(f"{n:7d} {len(d)//n:8d} {m['MAE']:8.2f} {m['MAPE_%']:8.2f} {m['sMAPE_%']:8.2f}   (linea base)")
        else:
            t, pv = dm(real, base, p, 24)
            marca = ("MEJOR" if t > 0 else "peor") if pv < 0.05 else "="
            print(f"{n:7d} {len(d)//n:8d} {m['MAE']:8.2f} {m['MAPE_%']:8.2f} {m['sMAPE_%']:8.2f}   "
                  f"{m['MAE']-m0['MAE']:+6.2f} p={pv:.4f} {marca}")

    # ---------- A) agregar votantes de a uno ----------
    print(f"\n{'='*92}\nA1) AGREGAR CADA VOTANTE EXTRA AL v4 (de a uno)\n{'='*92}")
    extras = [c for c in disponibles if c not in BASE5]
    print(f"{'votante agregado':18s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s}   DM vs v4")
    for c in extras:
        p = qra_cv(d, BASE5 + [c], "g4")
        m = metricas(real, p)
        t, pv = dm(real, base, p, 24)
        marca = ("MEJOR" if t > 0 else "peor") if pv < 0.05 else "="
        print(f"{c:18s} {m['MAE']:8.2f} {m['MAPE_%']:8.2f} {m['sMAPE_%']:8.2f}   "
              f"{m['MAE']-m0['MAE']:+6.2f} p={pv:.4f} {marca}")

    # ---------- A2) seleccion greedy sobre TODOS los votantes ----------
    print(f"\n{'='*92}\nA2) SELECCION GREEDY HACIA ADELANTE (todos los votantes, 4 franjas)\n{'='*92}")
    elegidos, restantes = [], list(disponibles)
    historial = []
    while restantes:
        mejor, mejor_m, mejor_p = None, None, None
        for c in restantes:
            p = qra_cv(d, elegidos + [c], "g4")
            m = metricas(real, p)
            if mejor_m is None or m["MAE"] < mejor_m["MAE"]:
                mejor, mejor_m, mejor_p = c, m, p
        elegidos.append(mejor)
        restantes.remove(mejor)
        historial.append({"n": len(elegidos), "agregado": mejor, **mejor_m})
        print(f"  {len(elegidos):2d}: +{mejor:18s} MAE={mejor_m['MAE']:6.2f} "
              f"MAPE={mejor_m['MAPE_%']:5.2f}% sMAPE={mejor_m['sMAPE_%']:5.2f}%")
    hist = pd.DataFrame(historial)
    mejor_fila = hist.loc[hist["MAE"].idxmin()]
    mejores = hist.loc[:mejor_fila.name, "agregado"].tolist()
    print(f"\nMejor conjunto ({int(mejor_fila['n'])} votantes): {mejores}")
    print(f"  MAE={mejor_fila['MAE']:.2f}  MAPE={mejor_fila['MAPE_%']:.2f}%  "
          f"sMAPE={mejor_fila['sMAPE_%']:.2f}%  R2={mejor_fila['R2']:.3f}")

    # ---------- combinar lo mejor de A y B ----------
    print(f"\n{'='*92}\nA+B) EL MEJOR CONJUNTO CON DISTINTAS GRANULARIDADES\n{'='*92}")
    print(f"{'grupos':>7s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s} {'R2':>7s}   DM vs v4")
    filas = []
    for n in [1, 2, 3, 4, 6, 8, 12, 24]:
        p = qra_cv(d, mejores, f"g{n}")
        m = metricas(real, p)
        t, pv = dm(real, base, p, 24)
        marca = ("MEJOR" if t > 0 else "peor") if pv < 0.05 else "="
        print(f"{n:7d} {m['MAE']:8.2f} {m['MAPE_%']:8.2f} {m['sMAPE_%']:8.2f} {m['R2']:7.3f}   "
              f"{m['MAE']-m0['MAE']:+6.2f} p={pv:.4f} {marca}")
        filas.append({"grupos": n, "votantes": " + ".join(mejores), **m})
    pd.DataFrame(filas).to_csv(RES / "mejorar_ensamble_24h.csv", index=False)

    # ---------- C) cuanto peso recibe LEAR24 en cada franja ----------
    if "LEAR24" in mejores:
        print(f"\n{'='*92}\nC) PESO QUE RECIBE LEAR24 EN CADA FRANJA (ajuste sobre todo 2026)\n{'='*92}")
        X, y = d[mejores].to_numpy(), real
        for gg in range(4):
            msk = (d["g4"] == gg).to_numpy()
            w, b = ajustar_cuantil(X[msk], y[msk], 0.5)
            pesos = {c: round(float(wi), 3) for c, wi in zip(mejores, w) if abs(wi) > 0.005}
            horas = f"{gg*6:02d}:00-{gg*6+5:02d}:59"
            print(f"  franja {horas}: " + ", ".join(f"{k}={v}" for k, v in
                                                    sorted(pesos.items(), key=lambda x: -x[1])))
    print("\nGuardado: mejorar_ensamble_24h.csv")


if __name__ == "__main__":
    main()
