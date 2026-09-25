# -*- coding: utf-8 -*-
"""
Stacking de 24h v2: tres mejoras sobre el ensamble ganador (MAE 44.51, MAPE 12.05%).

1) FUSION PENDIENTE. Anoche se probaron 6 modelos de 24h por separado (CatBoost, hurdle,
   Markov-Switching, RecursiveLS, XGBoost y CatBoost afinados con Optuna) y se concluyo que
   "fusionar es simplemente agregarlos al stacking" -- pero nunca se hizo. Aqui se hace, junto con
   los dos modelos directos nuevos (CatBoost y Ridge especialistas de 24h). RecursiveLS entra en
   su version day-ahead corregida (sin la fuga intradia de la version original).

2) META-MODELO ALINEADO CON LA METRICA. NNLS minimiza error CUADRATICO, pero el proyecto se evalua
   con MAE. Un meta-modelo LAD (minima desviacion absoluta) con pesos no negativos que suman 1
   optimiza exactamente la metrica reportada, y es mas robusto a los picos de precio.

3) PESOS CAUSALES (DESPLEGABLES). La validacion cruzada por pliegues de dias es una evaluacion
   fuera de muestra valida, pero usa dias FUTUROS para fijar los pesos de dias pasados -- en
   produccion eso no existe. Aqui se calculan pesos solo con los dias anteriores (ventana
   expansiva y ventanas moviles de 30/60 dias). Es la cifra que un jurado deberia ver: cuanto
   rinde el ensamble si se opera en tiempo real. Si las ventanas moviles ganan, ademas es
   evidencia de que re-ponderar sigue los cambios de regimen (la preocupacion recurrente del
   usuario sobre junio 2026), en la linea de la agregacion adaptativa de expertos de la literatura.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import nnls, linprog
from scipy import sparse
import statsmodels.api as sm

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"

BASE5 = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]
EXTRAS = {
    "CatBoost": ("catboost_2026.csv", "prediccion"),
    "Hurdle": ("hurdle_dos_etapas_2026.csv", "prediccion"),
    "MarkovSw": ("markov_switching_2026.csv", "prediccion"),
    "RLS-dayahead": ("recursive_ls_dayahead_2026.csv", "pred_recursivo_dayahead"),
    "XGB-Optuna": ("optuna_armonicos_xgb_catboost_2026.csv", "pred_xgb_afinado"),
    "CatB-Optuna": ("optuna_armonicos_xgb_catboost_2026.csv", "pred_catboost_afinado"),
    "CatB-dir24": ("pronostico_directo_24h_diario_2026.csv", "pred_catb"),
    "Ridge-dir24": ("pronostico_directo_24h_diario_2026.csv", "pred_ridge"),
}


def cargar():
    c = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    o6 = c[c["origen"] == "Origen 6"]
    df = o6.pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
    df = df.join(o6.groupby("fecha_hora")["real"].first())
    for nom, (arch, col) in EXTRAS.items():
        ruta = RES / arch
        if not ruta.exists():
            print(f"  (falta {arch}, se omite {nom})")
            continue
        e = pd.read_csv(ruta, parse_dates=["fecha_hora"]).drop_duplicates("fecha_hora").set_index("fecha_hora")[col]
        df = df.join(e.rename(nom), how="inner")
    df = df.dropna()
    df["dia"] = df.index.normalize()
    return df


def pesos_nnls(X, y):
    w, _ = nnls(X, y)
    return w / w.sum() if w.sum() > 0 else np.full(X.shape[1], 1 / X.shape[1])


def pesos_lad(X, y):
    """min sum|y - Xw|  s.a. w>=0, sum(w)=1  (programacion lineal exacta)."""
    n, k = X.shape
    c = np.concatenate([np.zeros(k), np.ones(2 * n)])
    A_eq = sparse.vstack([
        sparse.hstack([sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)]),
        sparse.hstack([sparse.csr_matrix(np.ones((1, k))), sparse.csr_matrix((1, 2 * n))]),
    ]).tocsr()
    b_eq = np.concatenate([y, [1.0]])
    r = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return r.x[:k] if r.success else pesos_nnls(X, y)


def cv_dias(df, cols, ajustar, k=5):
    dias = np.sort(df["dia"].unique())
    pliegues = np.array_split(np.random.default_rng(42).permutation(dias), k)
    pred = pd.Series(np.nan, index=df.index)
    pesos = []
    for dt in pliegues:
        m = df["dia"].isin(dt)
        w = ajustar(df.loc[~m, cols].values, df.loc[~m, "real"].values)
        pesos.append(w)
        pred[m] = df.loc[m, cols].values @ w
    return pred, np.mean(pesos, axis=0)


def causal(df, cols, ajustar, ventana=None, min_dias=14):
    dias = np.sort(df["dia"].unique())
    pred = pd.Series(np.nan, index=df.index)
    ultimo_w = None
    for i in range(min_dias, len(dias)):
        ini = 0 if ventana is None else max(0, i - ventana)
        tr = df["dia"].isin(dias[ini:i])
        te = df["dia"] == dias[i]
        w = ajustar(df.loc[tr, cols].values, df.loc[tr, "real"].values)
        pred[te] = df.loc[te, cols].values @ w
        ultimo_w = w
    return pred, ultimo_w


def metricas(real, pred):
    e = np.abs(real - pred)
    return e.mean(), (e / real).mean() * 100


def dm(real, p_ref, p_nuevo):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
    return r.tvalues[0], r.pvalues[0]


def main():
    df = cargar()
    todos = BASE5 + [m for m in EXTRAS if m in df.columns]
    print(f"Horas alineadas: {len(df):,} | votantes: {len(todos)}")
    print("\nMAE individual (2026):")
    for m in todos:
        print(f"  {m:14s} {metricas(df['real'], df[m])[0]:7.2f}")

    variantes = {}
    variantes["A NNLS-CV 5 modelos (actual)"], wA = cv_dias(df, BASE5, pesos_nnls)
    variantes["B NNLS-CV todos"], wB = cv_dias(df, todos, pesos_nnls)
    variantes["C LAD-CV todos"], wC = cv_dias(df, todos, pesos_lad)
    variantes["D causal NNLS 5 modelos (expansiva)"], _ = causal(df, BASE5, pesos_nnls)
    variantes["E causal NNLS todos (expansiva)"], _ = causal(df, todos, pesos_nnls)
    variantes["F causal NNLS todos (movil 60d)"], _ = causal(df, todos, pesos_nnls, ventana=60)
    variantes["G causal NNLS todos (movil 30d)"], _ = causal(df, todos, pesos_nnls, ventana=30)
    variantes["H causal LAD todos (movil 60d)"], wH = causal(df, todos, pesos_lad, ventana=60)

    comun = pd.concat(variantes, axis=1).dropna().index
    real_c = df.loc[comun, "real"]
    print(f"\n{'='*84}\nComparacion sobre las mismas {len(comun):,} horas (tras 14 dias de calentamiento)\n{'='*84}")
    ref = variantes["A NNLS-CV 5 modelos (actual)"]
    filas = []
    for nom, p in variantes.items():
        mae, mape = metricas(real_c, p[comun])
        t, pv = dm(real_c, ref[comun], p[comun]) if not nom.startswith("A ") else (np.nan, np.nan)
        marca = "" if nom.startswith("A ") else (
            "GANA (sig.)" if pv < 0.05 and t > 0 else ("PIERDE (sig.)" if pv < 0.05 else "sin dif."))
        print(f"  {nom:40s} MAE={mae:6.2f}  MAPE={mape:5.2f}%   DM vs A: p={pv:.4f} {marca}")
        filas.append({"variante": nom, "mae": mae, "mape": mape, "dm_t_vs_A": t, "dm_p_vs_A": pv})

    t, pv = dm(real_c, variantes["D causal NNLS 5 modelos (expansiva)"][comun],
               variantes["E causal NNLS todos (expansiva)"][comun])
    print(f"\n  Desplegable, efecto de sumar votantes (E vs D): t={t:.3f} p={pv:.4f}")

    print("\nMAE en todo 2026 (solo variantes con validacion cruzada, cubren el 100%):")
    for nom in ["A NNLS-CV 5 modelos (actual)", "B NNLS-CV todos", "C LAD-CV todos"]:
        mae, mape = metricas(df["real"], variantes[nom])
        print(f"  {nom:40s} MAE={mae:6.2f}  MAPE={mape:5.2f}%")

    print("\nPesos promedio B (NNLS-CV todos):  " + " | ".join(f"{m}={w:.3f}" for m, w in zip(todos, wB) if w > 0.005))
    print("Pesos promedio C (LAD-CV todos):   " + " | ".join(f"{m}={w:.3f}" for m, w in zip(todos, wC) if w > 0.005))
    print("Pesos ultimo dia H (LAD movil 60d): " + " | ".join(f"{m}={w:.3f}" for m, w in zip(todos, wH) if w > 0.005))

    out = pd.DataFrame({"real": df["real"]})
    for nom, p in variantes.items():
        out["pred_" + nom.split(" ")[0]] = p
    out.reset_index().to_csv(RES / "stacking_24h_v2_2026.csv", index=False)
    pd.DataFrame(filas).to_csv(RES / "stacking_24h_v2_metricas.csv", index=False)
    print("\nGuardado: stacking_24h_v2_2026.csv / stacking_24h_v2_metricas.csv")


if __name__ == "__main__":
    main()
