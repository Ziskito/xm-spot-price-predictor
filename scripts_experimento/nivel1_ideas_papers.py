# -*- coding: utf-8 -*-
"""
NIVEL 1 de las 4 ideas de los papers desbloqueados (README 2026-09-?? "cuatro ideas... sin probar",
orden de costo/beneficio del propio texto): las tres mas baratas, con predicciones ya guardadas.

(1) COMPUERTA POR REGIMEN HIDROLOGICO (Albani et al. 2025). Su modelo se degrada cuando se rompe
    la correlacion ENA-precio; proponen apagar el componente hidrologico fuera de esa zona. Aqui:
    en vez de pesos del ensamble por FRANJA HORARIA (g4, lo que hace hoy el combinador), pesos por
    CELDA DE REGIMEN (embalse alto/bajo x ONI alto/bajo). El error ya triplica entre celdas
    (28.5 a 97.1 de MAE, `estratificacion_regimen_24h.csv`), pero esa celda solo se usa hoy como
    etiqueta de confianza, nunca para repesar el ensamble.

    IMPORTANTE: `estratificacion_regimen_24h.csv` corta "embalse alto/bajo" en la MEDIANA DE TODO
    2026 -- no es causal (usa el futuro para definir el corte del presente) y no es desplegable.
    Aqui se corrige: anomalia de volumen de embalse contra su propia climatologia (ya existe en el
    dataset como `volumen_embalses_vs_media30d`) y el corte se hace por SIGNO (>=0 alto, <0 bajo),
    que no necesita ninguna ventana futura. Ademas se prueba una version con umbral movil de 90
    dias (la correccion que se le debe a Rafael para la senal de confianza del dashboard).

(2) RLS-DAYAHEAD COMO VOTANTE. Documentado en README: "candidato razonable para sumar como sexto
    input... no se hizo todavia". El sexto puesto lo ocupo despues GARCH-ged. Se prueba si RLS
    aporta ADEMAS de GARCH-ged (septimo votante) o en su lugar.

(3) PODA POR MULTICOLINEALIDAD (Dias et al. 2024). Eliminar una de cada par de variables con
    |Pearson| > 0.8, quedarse con la de mayor informacion mutua con el precio. Se aplica sobre el
    diseño del LEAR (62 variables, muchas construidas unas de otras: medias moviles, deltas,
    anomalias) y se mide si un LEAR mas chico iguala o mejora al actual.

Estandar del proyecto: DM test + 10 particiones donde aplique.
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
from stacking_24h_v2 import cargar as cargar24, BASE5, EXTRAS
from combinador_optimo_mape import qra_cv, qra_pesos, mape, smape
from lear_24h import construir_para_paso

SEMILLAS = list(range(10))
PICO = [0, 8, 9, 10, 18, 19]


def dm(y, p_ref, p_nuevo, metrica="mae", maxlags=24):
    y = np.asarray(y, float)
    if metrica == "mape":
        l1, l2 = np.abs(y - p_ref) / y, np.abs(y - p_nuevo) / y
    else:
        l1, l2 = np.abs(y - p_ref), np.abs(y - p_nuevo)
    d = l1 - l2
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.pvalues[0])


def qra_cv_por_grupo(d, cols, col_grupo, modo="smape", semilla=42, iteraciones=3):
    """Igual que qra_cv pero el pliegue de pesos se hace por `col_grupo` en vez de por 'g4'."""
    from combinador_optimo_mape import qra_pesos
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d[col_grupo].to_numpy(), d["dia"].to_numpy()
    P = np.full(len(d), np.nan)
    pliegues = np.array_split(np.random.default_rng(semilla).permutation(np.sort(np.unique(dia))), 5)
    for dt in pliegues:
        te_m = np.isin(dia, dt)
        for gg in pd.unique(g[te_m]):
            tr, te = (~te_m) & (g == gg), te_m & (g == gg)
            if tr.sum() < len(cols) + 5:
                tr = ~te_m           # celda con pocos datos: usa todo el resto como respaldo
            w = 1.0 / np.maximum(y[tr], 1e-6) if modo != "mae" else None
            ww, b = qra_pesos(X[tr], y[tr], w)
            if modo == "smape":
                for _ in range(iteraciones):
                    pred_tr = X[tr] @ ww + b
                    w = 2.0 / np.maximum(np.abs(y[tr]) + np.abs(pred_tr), 1e-6)
                    ww, b = qra_pesos(X[tr], y[tr], w)
            P[te] = X[te] @ ww + b
    return P


def preparar_base():
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    # RLS-dayahead ya viene incluido por cargar24() (esta en EXTRAS de stacking_24h_v2.py)
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d.index.hour // 6

    # --- regimen causal: signo de la anomalia de embalse (>=media30d -> alto) x signo del ONI
    df = cargar_completo().set_index("fecha_hora")
    d["vol_anom"] = df["volumen_embalses_vs_media30d"].reindex(d.index)
    d["oni"] = df["oni"].reindex(d.index)
    d["celda_regimen"] = (np.where(d["vol_anom"] >= 0, "embAlto", "embBajo") + "_" +
                          np.where(d["oni"] >= 0, "oniAlto", "oniBajo"))
    # version con umbral MOVIL de 90 dias (causal, solo pasado), la correccion que se le debe a Rafael
    vol_roll = d["vol_anom"].rolling("90D", min_periods=30).median().shift(1)
    oni_roll = d["oni"].rolling("90D", min_periods=30).median().shift(1)
    d["celda_regimen_movil"] = (np.where(d["vol_anom"] >= vol_roll.fillna(0), "embAlto", "embBajo") + "_" +
                                np.where(d["oni"] >= oni_roll.fillna(0), "oniAlto", "oniBajo"))
    return d


def parte1_regimen(d):
    print("=" * 100)
    print("(1) COMPUERTA POR REGIMEN HIDROLOGICO -- pesos del ensamble por celda en vez de por franja")
    print("=" * 100)
    MEJOR = BASE5 + ["GARCH-ged"]
    dd = d.dropna(subset=MEJOR + ["real", "celda_regimen", "celda_regimen_movil"]).copy()
    y = dd["real"].to_numpy()
    print(f"  filas: {len(dd):,}")
    print(f"  reparto celda de regimen (fija, sobre anomalia):")
    print(dd["celda_regimen"].value_counts().to_string())
    print(f"  reparto celda de regimen (movil 90d, causal):")
    print(dd["celda_regimen_movil"].value_counts().to_string())

    base = qra_cv(dd, MEJOR, "smape")
    print(f"\n  {'esquema de pesos':38s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s}  DM(MAE) DM(MAPE)")
    print(f"  {'actual (por franja horaria g4)':38s} {np.abs(y-base).mean():8.2f} {mape(y,base):8.2f} "
          f"{smape(y,base):8.2f}")
    for col, nom in [("celda_regimen", "por celda de regimen (fija)"),
                     ("celda_regimen_movil", "por celda de regimen (movil 90d)")]:
        p = qra_cv_por_grupo(dd, MEJOR, col, "smape")
        print(f"  {nom:38s} {np.abs(y-p).mean():8.2f} {mape(y,p):8.2f} {smape(y,p):8.2f}  "
              f"{dm(y,base,p,'mae'):.4f}  {dm(y,base,p,'mape'):.4f}")

    # interaccion: franja horaria x regimen (celdas mas finas)
    dd["celda_franja_regimen"] = dd["g4"].astype(str) + "_" + dd["celda_regimen_movil"]
    p = qra_cv_por_grupo(dd, MEJOR, "celda_franja_regimen", "smape")
    print(f"  {'franja horaria x regimen (movil)':38s} {np.abs(y-p).mean():8.2f} {mape(y,p):8.2f} "
          f"{smape(y,p):8.2f}  {dm(y,base,p,'mae'):.4f}  {dm(y,base,p,'mape'):.4f}")

    mejor_esquema = "celda_regimen_movil"
    print(f"\n  Verificacion 10 particiones -- '{mejor_esquema}' contra franja horaria:")
    A, B = [], []
    for s_ in SEMILLAS:
        pa = qra_cv(dd, MEJOR, "smape", s_)
        pb = qra_cv_por_grupo(dd, MEJOR, mejor_esquema, "smape", s_)
        A.append([np.abs(y-pa).mean(), mape(y,pa), smape(y,pa)])
        B.append([np.abs(y-pb).mean(), mape(y,pb), smape(y,pb)])
    A, B = np.array(A), np.array(B)
    for j, nm in enumerate(["MAE", "MAPE %", "sMAPE %"]):
        dl = B[:,j] - A[:,j]
        print(f"    {nm:8s} actual={A[:,j].mean():7.3f}+/-{A[:,j].std():.3f}  "
              f"regimen={B[:,j].mean():7.3f}+/-{B[:,j].std():.3f}  "
              f"delta={dl.mean():+7.3f}, mejor en {int((dl<0).sum())}/10")


def parte2_rls(d):
    print("\n" + "=" * 100)
    print("(2) RLS-DAYAHEAD COMO VOTANTE ADICIONAL / ALTERNATIVA A GARCH-GED")
    print("=" * 100)
    if "RLS-dayahead" not in d.columns:
        print("  (no se encontro recursive_ls_dayahead_2026.csv; se omite)")
        return
    MEJOR = BASE5 + ["GARCH-ged"]
    dd = d.dropna(subset=MEJOR + ["RLS-dayahead", "real"]).copy()
    y = dd["real"].to_numpy()
    print(f"  filas: {len(dd):,}")
    print(f"  MAE de RLS-dayahead suelto: {np.abs(y - dd['RLS-dayahead']).mean():.2f}")

    combos = {
        "actual (6: BASE5+GARCH-ged)": MEJOR,
        "+ RLS-dayahead (7 votantes)": MEJOR + ["RLS-dayahead"],
        "BASE5 + RLS-dayahead (sin GARCH-ged)": BASE5 + ["RLS-dayahead"],
    }
    base = None
    print(f"\n  {'combinacion':38s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s}  DM(MAE) DM(MAPE)")
    preds = {}
    for nom, cols in combos.items():
        p = qra_cv(dd, cols, "smape")
        preds[nom] = p
        if base is None:
            base = p
        extra = f"  {dm(y,base,p,'mae'):.4f}  {dm(y,base,p,'mape'):.4f}" if nom != "actual (6: BASE5+GARCH-ged)" else ""
        print(f"  {nom:38s} {np.abs(y-p).mean():8.2f} {mape(y,p):8.2f} {smape(y,p):8.2f}{extra}")

    mejor = min([k for k in combos if k != "actual (6: BASE5+GARCH-ged)"],
               key=lambda k: mape(y, preds[k]))
    print(f"\n  Verificacion 10 particiones -- '{mejor}':")
    A, B = [], []
    for s_ in SEMILLAS:
        pa = qra_cv(dd, MEJOR, "smape", s_)
        pb = qra_cv(dd, combos[mejor], "smape", s_)
        A.append([np.abs(y-pa).mean(), mape(y,pa), smape(y,pa)])
        B.append([np.abs(y-pb).mean(), mape(y,pb), smape(y,pb)])
    A, B = np.array(A), np.array(B)
    for j, nm in enumerate(["MAE", "MAPE %", "sMAPE %"]):
        dl = B[:,j] - A[:,j]
        print(f"    {nm:8s} actual={A[:,j].mean():7.3f}+/-{A[:,j].std():.3f}  "
              f"con RLS={B[:,j].mean():7.3f}+/-{B[:,j].std():.3f}  "
              f"delta={dl.mean():+7.3f}, mejor en {int((dl<0).sum())}/10")


def parte3_multicolinealidad():
    print("\n" + "=" * 100)
    print("(3) PODA POR MULTICOLINEALIDAD sobre el diseno del LEAR (|Pearson| > 0.8)")
    print("=" * 100)
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import mutual_info_regression

    CORTE = pd.Timestamp("2026-01-01")
    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - 24)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]

    # matriz de diseno de un paso representativo (h=12) para decidir que variables podar;
    # la poda de columnas se aplica igual a los 24 pasos (mismas columnas en todos)
    Xrep, yrep, _, _ = construir_para_paso(df, 12, cortes_tr)
    C = Xrep.corr().abs()
    cols = list(Xrep.columns)
    mi = pd.Series(mutual_info_regression(Xrep, yrep, random_state=42), index=cols)

    eliminar = set()
    for i, a in enumerate(cols):
        if a in eliminar:
            continue
        for b in cols[i+1:]:
            if b in eliminar:
                continue
            if C.loc[a, b] > 0.8:
                eliminar.add(b if mi[a] >= mi[b] else a)
    conservar = [c for c in cols if c not in eliminar]
    print(f"  variables originales: {len(cols)}  |  eliminadas por colinealidad: {len(eliminar)}  "
          f"|  conservadas: {len(conservar)}")
    print(f"  eliminadas: {sorted(eliminar)}")

    filas_res = []
    for etiqueta, usar_todas in [("LEAR actual (62 variables)", True), ("LEAR podado (MI, |r|<=0.8)", False)]:
        partes, alphas = [], []
        for h in range(1, 25):
            Xtr, ytr, _, _ = construir_para_paso(df, h, cortes_tr)
            Xte, yte, fo, _ = construir_para_paso(df, h, cortes_te)
            if not usar_todas:
                Xtr, Xte = Xtr[conservar], Xte[conservar]
            sc = StandardScaler().fit(Xtr)
            if h <= 4 or h % 6 == 0:
                alphas.append(LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000)
                              .fit(sc.transform(Xtr), ytr).alpha_)
            m = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(sc.transform(Xtr), ytr)
            partes.append(pd.DataFrame({"fecha_hora": fo, "real": yte, "pred": m.predict(sc.transform(Xte))}))
        r = pd.concat(partes).drop_duplicates("fecha_hora")
        y, p = r["real"].to_numpy(), r["pred"].to_numpy()
        mae = np.abs(y - p).mean()
        print(f"  {etiqueta:32s} MAE={mae:7.2f}  MAPE={mape(y,p):6.2f}%  sMAPE={smape(y,p):6.2f}%")
        filas_res.append((etiqueta, r))
    (_, r1), (_, r2) = filas_res
    comun = r1.set_index("fecha_hora").index.intersection(r2.set_index("fecha_hora").index)
    y = r1.set_index("fecha_hora").loc[comun, "real"].to_numpy()
    p1 = r1.set_index("fecha_hora").loc[comun, "pred"].to_numpy()
    p2 = r2.set_index("fecha_hora").loc[comun, "pred"].to_numpy()
    print(f"  DM (podado vs actual): p={dm(y, p1, p2):.4f}")


def main():
    t0 = time.time()
    d = preparar_base()
    parte1_regimen(d)
    parte2_rls(d)
    parte3_multicolinealidad()
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
