# -*- coding: utf-8 -*-
"""
Ultimo frente para el ensamble de 24h: meta-modelo NO LINEAL y pesos condicionados al ESTADO.

Recorrido hasta aqui (todo negativo):
  - votantes nuevos, incluido LEAR24: sin diferencia significativa;
  - granularidad de pesos (1 a 24 grupos): sin diferencia;
  - especialistas de hora pico (LASSO y CatBoost): peores que el ensamble general;
  - demanda de la hora objetivo, incluso con ORACULO: empeora;
  - pesos adaptativos (ventanas moviles de 7 a 90 dias, y Hedge con descuento): ninguno mejora.

Pero techo_combinacion_24h.py mostro que los pesos optimos POR DIA darian 8.82% frente al 11.27%
actual. Que los esquemas adaptativos fallen significa que ese optimo movil NO es predecible a
partir del rendimiento pasado de cada votante. Queda la otra posibilidad: que sea predecible a
partir del ESTADO OBSERVABLE en el corte (nivel de precio, volatilidad reciente, hidrologia, tipo
de dia). Eso es un ensamble "con compuerta" (gated), y no se ha probado.

Dos formas, ambas causales y tambien en validacion cruzada por dias:
  A) META-MODELO NO LINEAL: un modelo (CatBoost / bosque) que recibe las predicciones de los
     votantes MAS variables de estado, y devuelve directamente el precio. Puede aprender reglas del
     tipo "si la volatilidad reciente es alta, confia mas en N-BEATSx".
  B) PESOS POR CELDA DE ESTADO: LAD ajustado dentro de celdas definidas por volatilidad reciente y
     nivel de precio (no por hidrologia, que ya fallo en 72h), ademas de la franja horaria.

Anti-fuga: todas las variables de estado se toman en el corte (00:00 del dia anterior), nunca del
instante objetivo.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import cargar as cargar24, BASE5, pesos_lad
from o6_comun import cargar_completo, RES

PICO = [18, 19, 20]
ESTADO = ["precio_bolsa", "precio_media_24h", "precio_media_7d", "precio_std_24h", "precio_std_7d",
          "ratio_volatilidad", "precio_rango_24h", "volumen_embalses_vs_media30d",
          "aportes_hidricos_vs_media30d", "demanda_media_24h"]


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def main():
    t0 = time.time()
    from catboost import CatBoostRegressor
    from sklearn.ensemble import RandomForestRegressor

    d = cargar24().sort_index()
    lear = pd.read_csv(RES / "pronostico_lear_24h_2026.csv", parse_dates=["fecha_hora"])
    d = d.join(lear.set_index("fecha_hora")["LEAR24"], how="left")
    votantes = [c for c in BASE5 + ["LEAR24", "CatBoost", "MarkovSw"] if c in d.columns]

    # --- estado EN EL CORTE: 00:00 del dia anterior al objetivo ---
    full = cargar_completo().set_index("fecha_hora").sort_index()
    d["dia"] = d.index.normalize()
    # corte = 00:00 del mismo dia para horas 01:00-23:00; para la hora 00:00 es el dia anterior
    corte = np.where(d.index.hour == 0, d["dia"] - pd.Timedelta(days=1), d["dia"])
    est = full[ESTADO].reindex(pd.DatetimeIndex(corte))
    est.index = d.index
    d = pd.concat([d, est.add_suffix("_corte")], axis=1)
    cols_est = [f"{c}_corte" for c in ESTADO]
    d["hora"] = d.index.hour
    d["g4"] = d.index.hour // 6
    d = d.dropna(subset=votantes + cols_est + ["real"])
    y = d["real"].to_numpy()
    pk = d["hora"].isin(PICO).to_numpy()
    dias = np.sort(d["dia"].unique())
    print(f"Filas: {len(d):,} | votantes: {len(votantes)} | variables de estado: {len(cols_est)}\n")

    # referencia: QRA/LAD por franja en CV (equivalente al v4)
    def lad_cv_grupo(cols, grupo, semilla=42):
        X = d[cols].to_numpy()
        g = d[grupo].to_numpy()
        dia = d["dia"].to_numpy()
        P = np.full(len(d), np.nan)
        for dt in np.array_split(np.random.default_rng(semilla).permutation(dias), 5):
            te_m = np.isin(dia, dt)
            for gg in np.unique(g[te_m]):
                tr, te = (~te_m) & (g == gg), te_m & (g == gg)
                if tr.sum() >= len(cols) + 5:
                    P[te] = X[te] @ pesos_lad(X[tr], y[tr])
        return P

    ref = lad_cv_grupo(BASE5, "g4")
    print(f"REFERENCIA (LAD por franja, CV, 5 votantes): MAE={np.abs(y-ref).mean():.2f}  "
          f"MAPE={mape(y, ref):.2f}%  pico={mape(y[pk], ref[pk]):.2f}%\n")

    resultados = {"referencia": ref}

    # ---------- A) meta-modelo no lineal ----------
    print(f"{'='*96}\nA) META-MODELO NO LINEAL (votantes + estado -> precio), CV por dias\n{'='*96}")
    print(f"{'meta-modelo':46s} {'MAE':>8s} {'MAPE %':>8s} {'pico %':>8s}   DM vs referencia")
    configs = {
        "CatBoost (votantes solo)": (votantes, "catboost"),
        "CatBoost (votantes + estado)": (votantes + cols_est, "catboost"),
        "CatBoost (votantes + estado + hora)": (votantes + cols_est + ["hora"], "catboost"),
        "RandomForest (votantes + estado + hora)": (votantes + cols_est + ["hora"], "rf"),
    }
    for etiqueta, (cc, tipo) in configs.items():
        X = d[cc].to_numpy()
        P = np.full(len(d), np.nan)
        dia = d["dia"].to_numpy()
        for dt in np.array_split(np.random.default_rng(42).permutation(dias), 5):
            te = np.isin(dia, dt)
            if tipo == "catboost":
                m = CatBoostRegressor(n_estimators=800, max_depth=6, learning_rate=0.05,
                                      loss_function="MAE", random_seed=42, verbose=0, thread_count=-1)
            else:
                m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
            m.fit(X[~te], y[~te])
            P[te] = m.predict(X[te])
        t, pv = dm(y, ref, P)
        marca = ("MEJORA" if t > 0 else "peor") if pv < 0.05 else "n.s."
        print(f"{etiqueta:46s} {np.abs(y-P).mean():8.2f} {mape(y, P):8.2f} {mape(y[pk], P[pk]):8.2f}"
              f"   p={pv:.4f} {marca}")
        resultados[etiqueta] = P

    # ---------- B) pesos por celda de estado ----------
    print(f"\n{'='*96}\nB) PESOS LAD POR CELDA DE ESTADO (volatilidad / nivel de precio), CV por dias\n{'='*96}")
    print(f"{'celda':46s} {'MAE':>8s} {'MAPE %':>8s} {'pico %':>8s}   DM vs referencia")
    # terciles calculados solo con dias anteriores seria lo ideal; en CV se usan los del conjunto de
    # entrenamiento de cada pliegue, que es el equivalente correcto
    for nombre, base_col in [("volatilidad reciente", "precio_std_7d_corte"),
                             ("nivel de precio", "precio_media_7d_corte"),
                             ("ratio de volatilidad", "ratio_volatilidad_corte")]:
        for n_cel in [2, 3]:
            X = d[votantes].to_numpy()
            dia = d["dia"].to_numpy()
            P = np.full(len(d), np.nan)
            for dt in np.array_split(np.random.default_rng(42).permutation(dias), 5):
                te_m = np.isin(dia, dt)
                cortes = np.quantile(d.loc[~te_m, base_col], np.linspace(0, 1, n_cel + 1)[1:-1])
                cel = np.digitize(d[base_col].to_numpy(), cortes)
                comb = cel * 10 + d["g4"].to_numpy()
                for gg in np.unique(comb[te_m]):
                    tr, te = (~te_m) & (comb == gg), te_m & (comb == gg)
                    if tr.sum() >= len(votantes) + 5:
                        P[te] = X[te] @ pesos_lad(X[tr], y[tr])
            oo = ~np.isnan(P)
            t, pv = dm(y[oo], ref[oo], P[oo])
            marca = ("MEJORA" if t > 0 else "peor") if pv < 0.05 else "n.s."
            et = f"{nombre} x franja ({n_cel} celdas x 4)"
            print(f"{et:46s} {np.abs(y[oo]-P[oo]).mean():8.2f} {mape(y[oo], P[oo]):8.2f} "
                  f"{mape(y[oo&pk], P[oo&pk]):8.2f}   p={pv:.4f} {marca}")
            resultados[et] = P

    mejor = min((k for k in resultados if k != "referencia"),
                key=lambda k: mape(y[~np.isnan(resultados[k])], resultados[k][~np.isnan(resultados[k])]))
    pm = resultados[mejor]
    oo = ~np.isnan(pm)
    t, pv = dm(y[oo], ref[oo], pm[oo])
    print(f"\n{'='*96}\nMEJOR: {mejor}")
    print(f"  MAE={np.abs(y[oo]-pm[oo]).mean():.2f}  MAPE={mape(y[oo], pm[oo]):.2f}%  "
          f"pico={mape(y[oo&pk], pm[oo&pk]):.2f}%  DM p={pv:.4f}")
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
