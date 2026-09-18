# -*- coding: utf-8 -*-
"""
Explotar la senal de riesgo: predecir_fallo_24h.py mostro que el fallo del ensamble de 24h SI es
anticipable en el corte (AUC 0.818, frente a 0.568 de la heuristica "es hora pico"; de las horas
senaladas como riesgosas, 49.4% estan de verdad en el peor 20%, 2.5x sobre el azar).

Eso abre dos usos, y se prueban los dos porque sirven para cosas distintas:

  A) ENRUTAMIENTO REAL. En las horas de alto riesgo, sustituir o mezclar el ensamble con un modelo
     alterno. OJO: la cota de predecir_fallo_24h suponia un especialista PERFECTO. Aqui se usan los
     modelos que existen de verdad. Si ninguno mejora en esas horas, el enrutamiento queda cerrado
     aunque el riesgo sea predecible -- porque no hay a donde enrutar.

  B) BANDAS DE INCERTIDUMBRE MODULADAS POR RIESGO. Aunque no exista mejor pronostico PUNTUAL, saber
     que una hora es riesgosa permite ensanchar la banda ahi y estrecharla donde el modelo es
     confiable. Eso mejora la calidad del intervalo (cobertura y ancho) sin tocar el punto, y es
     justo lo que el motor de decision (OE3) necesita: no solo el precio, sino cuanto fiarse.
     Se compara contra la banda conforme adaptativa vigente.
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
from o6_comun import cargar_completo, RES
from stacking_24h_v2 import cargar as cargar24

ESTADO = ["precio_bolsa", "precio_media_24h", "precio_media_7d", "precio_media_30d",
          "precio_std_24h", "precio_std_7d", "precio_rango_24h", "ratio_volatilidad",
          "volumen_embalses", "volumen_embalses_vs_media30d", "volumen_embalses_delta_1d",
          "aportes_hidricos", "aportes_hidricos_vs_media30d", "demanda", "demanda_media_24h",
          "generacion", "oni"]


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def interval_score(y, lo, hi, alfa=0.2):
    """Interval score (menor es mejor): penaliza ancho y penaliza quedarse corto."""
    return float(np.mean((hi - lo) + (2 / alfa) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))))


def main():
    t0 = time.time()
    from catboost import CatBoostRegressor

    v4 = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"]).sort_values("fecha_hora")
    d = v4.dropna(subset=["real", "pred_ensamble"]).copy().set_index("fecha_hora")
    d["hora"] = d.index.hour
    d["dia"] = d.index.normalize()
    otros = cargar24()
    lear = pd.read_csv(RES / "pronostico_lear_24h_2026.csv", parse_dates=["fecha_hora"])
    d = d.join(otros.drop(columns=[c for c in otros.columns if c in ("real", "dia")]), how="left")
    d = d.join(lear.set_index("fecha_hora")["LEAR24"], how="left")

    full = cargar_completo().set_index("fecha_hora").sort_index()
    corte = np.where(d["hora"] == 0, d["dia"] - pd.Timedelta(days=1), d["dia"])
    est = full[ESTADO].reindex(pd.DatetimeIndex(corte))
    est.index = d.index
    d = pd.concat([d, est.add_suffix("_c")], axis=1)
    d = d.dropna(subset=["real", "pred_ensamble"] + [f"{c}_c" for c in ESTADO])
    cols = [f"{c}_c" for c in ESTADO] + ["hora"]
    y = d["real"].to_numpy()
    base = d["pred_ensamble"].to_numpy()
    err = np.abs(y - base)
    dias = np.sort(d["dia"].unique())
    dia = d["dia"].to_numpy()
    print(f"Filas: {len(d):,} | MAPE del ensamble: {mape(y, base):.2f}%\n")

    # --- riesgo predicho, en validacion cruzada por dias (causal en el sentido de fuera de muestra) ---
    riesgo = np.full(len(d), np.nan)
    X = d[cols].to_numpy()
    for dt in np.array_split(np.random.default_rng(42).permutation(dias), 5):
        te = np.isin(dia, dt)
        m = CatBoostRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, loss_function="MAE",
                             random_seed=42, verbose=0, thread_count=-1).fit(X[~te], err[~te])
        riesgo[te] = m.predict(X[te])
    riesgo = np.maximum(riesgo, 1e-6)

    # ---------- A) enrutamiento con modelos REALES ----------
    print(f"{'='*94}\nA) ENRUTAR LAS HORAS DE ALTO RIESGO A UN MODELO ALTERNO (modelos reales)\n{'='*94}")
    alternos = [c for c in ["N-BEATSx", "N-HiTS", "LEAR24", "CatBoost", "MarkovSw", "ARX+GARCH"]
                if c in d.columns and d[c].notna().all()]
    print(f"{'alterno':12s} {'pct':>4s} {'MAPE global':>12s} {'MAPE en las enrutadas':>22s}  DM vs ensamble")
    for pct in [10, 20]:
        k = int(len(d) * pct / 100)
        sel = np.argsort(-riesgo)[:k]
        m_ens_sel = mape(y[sel], base[sel])
        for c in alternos:
            p = base.copy()
            p[sel] = d[c].to_numpy()[sel]
            t, pv = dm(y, base, p)
            marca = ("MEJORA" if t > 0 else "peor") if pv < 0.05 else "n.s."
            print(f"{c:12s} {pct:4d} {mape(y, p):12.2f} {mape(y[sel], d[c].to_numpy()[sel]):22.2f}"
                  f"  p={pv:.4f} {marca}   (ensamble ahi: {m_ens_sel:.2f})")
        # mezcla 50/50 con el mejor alterno
        print()

    # ---------- B) bandas moduladas por riesgo ----------
    print(f"{'='*94}\nB) BANDAS DE INCERTIDUMBRE MODULADAS POR EL RIESGO PREDICHO\n{'='*94}")
    bandas = pd.read_csv(RES / "bandas_24h_qra_causal_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    b = d.join(bandas[["q10", "q90"]], how="inner").dropna(subset=["q10", "q90"])
    yb = b["real"].to_numpy()
    lo0, hi0 = b["q10"].to_numpy(), b["q90"].to_numpy()
    r_b = pd.Series(riesgo, index=d.index).reindex(b.index).to_numpy()
    semi0 = (hi0 - lo0) / 2
    centro = (hi0 + lo0) / 2
    cob0 = float(np.mean((yb >= lo0) & (yb <= hi0)) * 100)
    print(f"  banda vigente (conforme adaptativa): cobertura {cob0:5.1f}%  "
          f"ancho medio {np.mean(hi0-lo0):7.1f}  interval score {interval_score(yb, lo0, hi0):8.1f}")
    # modular: semiancho proporcional al riesgo, renormalizado para conservar el ancho medio
    for mezcla in [0.5, 1.0]:
        factor = (r_b / r_b.mean()) ** mezcla
        factor = factor / factor.mean()
        semi = semi0 * factor
        lo, hi = centro - semi, centro + semi
        cob = float(np.mean((yb >= lo) & (yb <= hi)) * 100)
        print(f"  modulada (exponente {mezcla}):            cobertura {cob:5.1f}%  "
              f"ancho medio {np.mean(hi-lo):7.1f}  interval score {interval_score(yb, lo, hi):8.1f}")

    pd.DataFrame({"fecha_hora": d.index, "riesgo_predicho": riesgo,
                  "error_real": err}).to_csv(RES / "riesgo_24h.csv", index=False)
    print(f"\nGuardado: riesgo_24h.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
