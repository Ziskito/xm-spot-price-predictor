# -*- coding: utf-8 -*-
"""
Mejoras 2 y 3 de las tres pendientes: SELECCION DE VARIABLES sobre Ridge-directo a 72h.

Ridge-directo es nuestro mejor modelo individual a horizonte largo (48-72h), donde los arboles
no pueden extrapolar y lo lineal si. Es justamente la familia de modelos que segun la literatura
se beneficia mas de una seleccion agresiva de variables, y nunca se la hemos aplicado.

De donde salen las dos ideas:
  (2) LASSO agresivo -- Kapoor & Wichitaksorn (2023, Applied Energy 347:121446): sus modelos
      estadisticos CON TODAS las variables son los PEORES del estudio, y con variables filtradas
      por LASSO pasan a ser los MEJORES, por encima de DNN/LSTM/GRU/XGBoost, con mejoras de
      40-45%. Nosotros hemos estado AGREGANDO variables, nunca reduciendo.
      Dato que ahorra trabajo, de su misma tabla: RFE es el PEOR de los tres metodos de seleccion
      e informacion mutua queda segunda, asi que aqui se prueba LASSO (y MI como contraste), no RFE.
  (3) Poda por multicolinealidad -- Dias, Lira & Freire (2024, Energies 17:1864): eliminan una de
      cada par de variables con |Pearson| > 0.8 antes de modelar. Nuestro conjunto tiene muchas
      variables construidas unas de otras (medias moviles de 24h/7d/30d, deltas, anomalias, ratios)
      que casi seguro violan ese umbral, y la colinealidad castiga especificamente a los lineales.

Anti-fuga: TODA la seleccion (coeficientes LASSO, matriz de correlacion, ranking de informacion
mutua, media/desviacion del escalador) se estima SOLO con el periodo de entrenamiento.
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
from xgboost_directo_72h import construir_largo, COLS_ESTADO, COLS_FUTURO
from o6_comun import cargar_completo, RES

CORTE = pd.Timestamp("2026-01-01")
PASOS = list(range(1, 73))
TRAMOS = [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]


def dm(real, p_ref, p_nuevo, maxlags=71):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def podar_colineales(Xtr, cols, umbral=0.8, ytr=None):
    """Dias et al.: de cada par con |Pearson| > umbral se conserva UNA sola variable. El paper no
    dice cual de las dos, y la eleccion importa mucho:
      - ytr=None  -> se conserva la que aparece primero en la lista (orden arbitrario de columnas).
      - ytr dado  -> se conserva la MAS correlacionada con el objetivo, que es lo sensato: de otro
        modo se puede descartar una variable muy predictiva solo por venir despues en la lista.
    Se reportan las dos variantes porque la diferencia entre ellas es informativa por si misma."""
    C = np.nan_to_num(np.corrcoef(Xtr, rowvar=False))
    if ytr is None:
        prioridad = np.arange(len(cols))                      # menor indice = se conserva
    else:
        cy = np.array([abs(np.corrcoef(Xtr[:, k], ytr)[0, 1]) for k in range(len(cols))])
        cy = np.nan_to_num(cy)
        prioridad = np.argsort(np.argsort(-cy))               # mayor |corr con y| = se conserva
    quitar = set()
    for i in np.argsort(prioridad):                           # recorre de mayor a menor prioridad
        if i in quitar:
            continue
        for j in range(len(cols)):
            if j != i and j not in quitar and prioridad[j] > prioridad[i] and abs(C[i, j]) > umbral:
                quitar.add(j)
    return [c for k, c in enumerate(cols) if k not in quitar], [cols[k] for k in sorted(quitar)]


def main():
    t0 = time.time()
    from sklearn.linear_model import Ridge, LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import mutual_info_regression

    df = cargar_completo().reset_index(drop=True)
    cortes_validos = np.arange(len(df))
    largo = construir_largo(df, cortes_validos, PASOS)
    cols = COLS_ESTADO + COLS_FUTURO + ["precio_mismo_hora_disp", "paso_horas"]

    tr = largo[largo["cutoff"] < CORTE]
    te = largo[(largo["cutoff"] >= CORTE) & (largo["fecha_hora"] >= CORTE)]
    print(f"Entrenamiento: {len(tr):,} filas | Prueba: {len(te):,} filas | {len(cols)} variables")

    Xtr_raw, ytr = tr[cols].to_numpy(np.float64), tr["y"].to_numpy(np.float64)
    Xte_raw, yte = te[cols].to_numpy(np.float64), te["y"].to_numpy(np.float64)
    esc = StandardScaler().fit(Xtr_raw)          # ajustado SOLO en entrenamiento
    Xtr, Xte = esc.transform(Xtr_raw), esc.transform(Xte_raw)

    conjuntos = {"todas (linea base)": cols}

    # ---------- (3) poda por multicolinealidad ----------
    for u in [0.8, 0.9]:
        quedan, fuera = podar_colineales(Xtr, cols, u)
        conjuntos[f"poda |r|>{u} (por orden)"] = quedan
        print(f"\nPoda |r|>{u} conservando la PRIMERA de cada par: quedan {len(quedan)}/{len(cols)}."
              f"\n  descartadas: {fuera}")
        quedan2, fuera2 = podar_colineales(Xtr, cols, u, ytr)
        conjuntos[f"poda |r|>{u} (guiada por y)"] = quedan2
        print(f"Poda |r|>{u} conservando la MAS CORRELACIONADA con el precio: quedan {len(quedan2)}/{len(cols)}."
              f"\n  descartadas: {fuera2}")

    # ---------- (2) LASSO ----------
    # La SELECCION (que coeficientes quedan en cero) no necesita los 4.4M de filas: se estima sobre
    # una submuestra aleatoria de 300k. Con las 4.4M filas y 5 pliegues en paralelo, LassoCV agota
    # la memoria de esta maquina (WinError 1450). El modelo Ridge final si se ajusta con TODO.
    print("\nAjustando LASSO con validacion cruzada sobre submuestra de 300k filas...")
    rng = np.random.default_rng(42)
    sub_l = rng.choice(len(Xtr), size=min(300_000, len(Xtr)), replace=False)
    lc = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=5000).fit(Xtr[sub_l], ytr[sub_l])
    sel_lasso = [c for c, coef in zip(cols, lc.coef_) if abs(coef) > 1e-8]
    conjuntos["LASSO (alpha por CV)"] = sel_lasso
    print(f"LASSO alpha={lc.alpha_:.4f}: selecciona {len(sel_lasso)}/{len(cols)} variables")
    print(f"  descarta: {[c for c in cols if c not in sel_lasso]}")

    # LASSO mas agresivo: alpha x10 (la 'parsimonia' que segun NZ es lo que hace la diferencia)
    from sklearn.linear_model import Lasso
    for mult in [10, 50]:
        la = Lasso(alpha=lc.alpha_ * mult, max_iter=5000, random_state=42).fit(Xtr[sub_l], ytr[sub_l])
        sel = [c for c, coef in zip(cols, la.coef_) if abs(coef) > 1e-8]
        if sel and len(sel) < len(cols):
            conjuntos[f"LASSO agresivo (alpha x{mult})"] = sel
            print(f"LASSO alpha x{mult}: {len(sel)} variables -> {sel}")

    # ---------- informacion mutua (contraste; 2o mejor metodo segun NZ) ----------
    print("\nCalculando informacion mutua (submuestra de 60k filas por costo)...")
    sm_idx = np.random.default_rng(42).choice(len(Xtr), size=min(60000, len(Xtr)), replace=False)
    mi = mutual_info_regression(Xtr[sm_idx], ytr[sm_idx], random_state=42)
    orden = np.argsort(mi)[::-1]
    for k in [10, 20]:
        conjuntos[f"info. mutua top-{k}"] = [cols[i] for i in orden[:k]]
    print("  top-10 por MI: " + ", ".join(cols[i] for i in orden[:10]))

    # ---------- evaluacion ----------
    print(f"\n{'='*100}\nRIDGE DIRECTO 72h -- MAE por tramo segun el conjunto de variables\n{'='*100}")
    print(f"{'conjunto':32s} {'#var':>5s} {'1-24h':>9s} {'25-48h':>9s} {'49-72h':>9s} {'global':>9s}")
    preds, filas = {}, []
    for nombre, cc in conjuntos.items():
        ii = [cols.index(c) for c in cc]
        m = Ridge(alpha=10.0).fit(Xtr[:, ii], ytr)
        p = m.predict(Xte[:, ii])
        preds[nombre] = p
        fila = {"conjunto": nombre, "n_variables": len(cc)}
        linea = f"{nombre:32s} {len(cc):5d}"
        for tn, ini, fin in TRAMOS:
            msk = te["paso_horas"].between(ini, fin).to_numpy()
            mae = np.abs(yte[msk] - p[msk]).mean()
            fila[f"mae_{tn}"] = mae
            linea += f" {mae:9.2f}"
        fila["mae_global"] = np.abs(yte - p).mean()
        linea += f" {fila['mae_global']:9.2f}"
        print(linea)
        filas.append(fila)

    base = preds["todas (linea base)"]
    print(f"\n{'='*100}\nDM contra 'todas las variables', por tramo (negativo = el conjunto reducido mejora)\n{'='*100}")
    for nombre, p in preds.items():
        if nombre == "todas (linea base)":
            continue
        partes = []
        for tn, ini, fin in TRAMOS:
            msk = te["paso_horas"].between(ini, fin).to_numpy()
            d = np.abs(yte[msk] - p[msk]).mean() - np.abs(yte[msk] - base[msk]).mean()
            _, pv = dm(yte[msk], base[msk], p[msk])
            est = "*" if pv < 0.05 else " "
            partes.append(f"{tn}: {d:+6.2f} (p={pv:.3f}){est}")
        print(f"{nombre:32s} " + "  ".join(partes))

    pd.DataFrame(filas).to_csv(RES / "seleccion_variables_ridge_72h.csv", index=False)
    print(f"\nGuardado: seleccion_variables_ridge_72h.csv | LISTO en {(time.time()-t0)/60:.1f} min")
    print("(* = diferencia significativa al 5%)")


if __name__ == "__main__":
    main()
