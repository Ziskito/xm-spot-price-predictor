# -*- coding: utf-8 -*-
"""
Mejora 1 de las tres pendientes: QRA en vez de LAD como combinador del ensamble de 72h.

Motivo: en 24h, QRA por franja horaria supero a LAD por franja (MAE 42.51 vs 43.24) y por eso el
ensamble v4 de 24h usa QRA. Pero el ensamble de 72h (ensamble_72h_diario.py) nunca se probo con
QRA: su punto (q50) se arma con LAD por tramo de horizonte. Es la mejora mas barata disponible
porque NO hay que reentrenar ningun votante -- las predicciones ya estan guardadas en
pronostico_72h_diario_votantes_2026.csv y solo cambia como se combinan.

Diferencia entre los dos combinadores:
  LAD: min sum|y - Xw|   s.a. w >= 0, sum(w) = 1     (sin intercepto, pesos suman 1)
  QRA: min sum rho_0.5(y - Xw - b)  s.a. w >= 0, b libre   (con intercepto, pesos libres)
El intercepto puede absorber un sesgo de nivel comun a todos los votantes -- justo el tipo de
sesgo que el proyecto ya documento (sobreestimacion en precios bajos) -- y a 48-72h el sesgo de
nivel deberia ser MAYOR que a 24h, asi que la hipotesis es que QRA ayude mas aqui que alla.

Se evalua en los dos modos que usa el ensamble vigente (CV y desplegable/causal) y en los tres
tramos, con test Diebold-Mariano HAC contra el LAD correspondiente.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import pesos_lad
from qra_bandas import ajustar_cuantil
from ensamble_72h_diario import CONJ, TRAMOS, hac_p, RES


def combinar_cv(sub, cols, metodo):
    """5 pliegues por dia objetivo (mismo esquema que lad_cv del ensamble vigente)."""
    X, y = sub[cols].to_numpy(), sub["real"].to_numpy()
    P = np.full(len(sub), np.nan)
    dias = np.sort(sub["dia"].unique())
    for dt in np.array_split(np.random.default_rng(42).permutation(dias), 5):
        te = sub["dia"].isin(dt).to_numpy()
        if metodo == "lad":
            w = pesos_lad(X[~te], y[~te])
            P[te] = X[te] @ w
        else:
            w, b = ajustar_cuantil(X[~te], y[~te], 0.5)
            P[te] = X[te] @ w + b
    return P


def combinar_causal(sub, cols, metodo, cada=7, min_filas=15 * 24):
    """Version desplegable: en el corte i solo se usan filas cuyo instante objetivo ya ocurrio."""
    X, y = sub[cols].to_numpy(), sub["real"].to_numpy()
    obj, cor = sub["fecha_hora"].to_numpy(), sub["cutoff"].to_numpy()
    P = np.full(len(sub), np.nan)
    w = b = None
    for i, c in enumerate(np.sort(np.unique(cor))):
        tr = obj <= c
        if tr.sum() < min_filas:
            continue
        if w is None or i % cada == 0:
            if metodo == "lad":
                w, b = pesos_lad(X[tr], y[tr]), 0.0
            else:
                w, b = ajustar_cuantil(X[tr], y[tr], 0.5)
        te = cor == c
        P[te] = X[te] @ w + b
    return P


def main():
    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v["NBX_semillas"] = v[["NBX", "NBX_s7", "NBX_s123"]].mean(axis=1)
    v["dia"] = v["fecha_hora"].dt.normalize()
    print(f"Filas: {len(v):,} | cortes: {v['cutoff'].nunique()}\n")

    filas = []
    for nombre, ini, fin in TRAMOS:
        sub = v[v["paso_horas"].between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
        y = sub["real"].to_numpy()
        print(f"--- Tramo {nombre} ({len(sub):,} filas) ---")
        for cn, cols in CONJ.items():
            for modo, fun in [("CV", combinar_cv), ("desplegable", combinar_causal)]:
                p_lad = fun(sub, cols, "lad")
                p_qra = fun(sub, cols, "qra")
                ok = ~(np.isnan(p_lad) | np.isnan(p_qra))
                mae_l = np.abs(y[ok] - p_lad[ok]).mean()
                mae_q = np.abs(y[ok] - p_qra[ok]).mean()
                # d > 0 => QRA mejor (menor error) que LAD
                t, pv = hac_p(np.abs(y[ok] - p_lad[ok]) - np.abs(y[ok] - p_qra[ok]))
                delta = mae_q - mae_l
                if pv < 0.05:
                    marca = "QRA GANA" if t > 0 else "QRA PIERDE"
                else:
                    marca = "sin diferencia"
                print(f"  {cn:30s} {modo:11s} LAD={mae_l:6.2f}  QRA={mae_q:6.2f}  "
                      f"delta={delta:+6.2f}  p={pv:.4f}  {marca}")
                filas.append({"tramo": nombre, "conjunto": cn, "modo": modo, "mae_lad": mae_l,
                              "mae_qra": mae_q, "delta": delta, "p": pv, "veredicto": marca})
        print()

    t = pd.DataFrame(filas)
    t.to_csv(RES / "qra_72h.csv", index=False)
    gana = (t.veredicto == "QRA GANA").sum()
    pierde = (t.veredicto == "QRA PIERDE").sum()
    print(f"{'='*80}\nRESUMEN: QRA gana en {gana}/{len(t)} combinaciones, pierde en {pierde}, "
          f"sin diferencia en {len(t)-gana-pierde}")
    print(f"Delta medio (negativo = QRA mejor): {t.delta.mean():+.3f} COP/kWh")
    print("Guardado: qra_72h.csv")


if __name__ == "__main__":
    main()
