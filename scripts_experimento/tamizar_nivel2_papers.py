# -*- coding: utf-8 -*-
"""
NIVEL 2: tamiz + prueba en modelo de las 4 variables con respaldo de papers, nunca probadas.

    porcentaje_volumen_util   equivalente al EAR de Dias, Lira & Freire (2024, Brasil), #1 en
                              informacion mutua con el precio en su Tabla 1
    precio_contratos_regulado
    precio_contratos_no_regulado
                              equivalente a los forward prices del paper de Nueva Zelanda
                              (Kapoor & Wichitaksorn 2023), de las variables mas usadas por sus
                              modelos ganadores
    aportes_caudal            la variable que pronostica Nunes Jr., Ferreira & Pinho (2024,
                              Brasil) en su cadena caudal -> precio; el proyecto tiene aportes
                              en ENERGIA, no en caudal

MISMO ESTANDAR QUE `tamizar_variables_nuevas.py`. Primero el tamiz de correlacion parcial
(descontando precio_lag24h, la prueba decisiva del proyecto para separar fuga/redundancia de
aporte real). Solo lo que pase el tamiz se prueba en el modelo real (LEAR + ensamble, DM test),
que es donde han fallado las variables candidatas anteriores (costo_marginal_despacho tenia
parcial 0.178 y aun asi empeoro el ensamble).

ANTI-FUGA. Las 4 son variables EXOGENAS al precio (volumen de embalses, contratos, caudal), no
construidas a partir del precio, asi que no hay fuga por diseno. Se usan en el CORTE (t-24h o
antes, igual que aportes_hidricos/volumen_embalses ya en el pipeline), nunca en el instante
objetivo. Ademas de la variable en nivel, se construyen version "vs media 30d" y "delta 1d",
igual que las variables hidrologicas que ya estan en el proyecto (aportan mas que el nivel solo,
segun el experimento de memoria hidrologica).
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
from o6_comun import cargar_completo, RES, RAIZ
from lear_24h import construir_para_paso, EXOG_CORTE, REZAGOS_CORTE, MISMA_HORA, EXOG_OBJ
from stacking_24h_v2 import cargar as cargar24, BASE5
from combinador_optimo_mape import qra_cv, mape, smape

CORTE = pd.Timestamp("2026-01-01")
ARCHIVOS = {
    "porcentaje_volumen_util": "porcentaje_volumen_util_2019_2026.csv",
    "precio_contratos_regulado": "precio_contratos_regulado_2019_2026.csv",
    "precio_contratos_no_regulado": "precio_contratos_no_regulado_2019_2026.csv",
    "aportes_caudal": "aportes_caudal_2019_2026.csv",
}


def correlacion_parcial(x, y, z):
    datos = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(datos) < 100:
        return np.nan
    def residuo(v):
        A = np.vstack([np.ones(len(datos)), datos["z"].values]).T
        coef, *_ = np.linalg.lstsq(A, datos[v].values, rcond=None)
        return datos[v].values - A @ coef
    return float(np.corrcoef(residuo("x"), residuo("y"))[0, 1])


def dm(y, p_ref, p_nuevo, maxlags=24):
    y = np.asarray(y, float)
    d = np.abs(y - p_ref) - np.abs(y - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.pvalues[0])


def tamiz():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    base = pd.concat([a, b], ignore_index=True)[["fecha_hora", "precio_bolsa"]].sort_values("fecha_hora")
    base["precio_lag24h"] = base["precio_bolsa"].shift(24)
    ref = base["precio_lag24h"].corr(base["precio_bolsa"])
    print(f"REFERENCIA -- corr(precio_t-24h, precio_t) = {ref:.4f}\n")

    series, filas = {}, []
    print(f"{'variable':30s} {'n':>8s} {'corr_contemp':>13s} {'corr_lag24h':>12s} {'corr_parcial':>13s}  veredicto")
    for nombre, archivo in ARCHIVOS.items():
        df = pd.read_csv(RAIZ / "data" / archivo, parse_dates=["fecha_hora"])
        col = [c for c in df.columns if c != "fecha_hora"][0]
        es_diaria = df["fecha_hora"].dt.hour.nunique() == 1
        if es_diaria:
            m = pd.merge_asof(base.sort_values("fecha_hora"), df.sort_values("fecha_hora"),
                              on="fecha_hora", direction="backward")
        else:
            m = base.merge(df, on="fecha_hora", how="left")
        m = m.sort_values("fecha_hora")
        m["x_lag24"] = m[col].shift(24)
        cc = m[col].corr(m["precio_bolsa"])
        cl = m["x_lag24"].corr(m["precio_bolsa"])
        cp = correlacion_parcial(m["x_lag24"], m["precio_bolsa"], m["precio_lag24h"])
        veredicto = ("FUGA (no usar)" if abs(cc) > 0.95 else
                     "REDUNDANTE" if abs(cp) < 0.05 else ">>> CANDIDATA REAL <<<")
        print(f"{nombre:30s} {m[col].notna().sum():8d} {cc:13.4f} {cl:12.4f} {cp:13.4f}  {veredicto}")
        filas.append(dict(variable=nombre, n=int(m[col].notna().sum()), corr_contemp=cc,
                          corr_lag24h=cl, corr_parcial=cp, veredicto=veredicto))
        series[nombre] = m.set_index("fecha_hora")[col]
    pd.DataFrame(filas).to_csv(RES / "tamiz_nivel2_papers.csv", index=False)
    return series, [f["variable"] for f in filas if "CANDIDATA" in f["veredicto"]]


def construir_features(series, candidatas):
    """Nivel + anomalia vs media 30d + delta 1d, como las variables hidrologicas ya en el pipeline."""
    feats = {}
    for nom in candidatas:
        s = series[nom]
        # llevar a rejilla horaria completa (ffill si es diaria) alineada con el resto del pipeline
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h")).ffill()
        media30d = s.rolling(24 * 30, min_periods=24).mean()
        feats[f"{nom}"] = s
        feats[f"{nom}_vs_media30d"] = s - media30d
        feats[f"{nom}_delta_1d"] = s - s.shift(24)
    return pd.DataFrame(feats)


def probar_en_lear(df_feats, candidatas):
    """Añade las features nuevas al diseño del LEAR (mismo protocolo EXOG_CORTE) y compara."""
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler

    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    for c in df_feats.columns:
        df[c] = df_feats[c].reindex(fechas).to_numpy()

    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - 24)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]

    nuevas_cols = [c for nom in candidatas for c in
                   (nom, f"{nom}_vs_media30d", f"{nom}_delta_1d")]

    def construir_extendido(h, cortes, con_nuevas):
        precio = df["precio_bolsa"].to_numpy(np.float64)
        n = len(df)
        c = cortes[(cortes + h < n) & (cortes - max(REZAGOS_CORTE) >= 0)]
        t = c + h
        col, nom = [], []
        for k in REZAGOS_CORTE:
            col.append(precio[c - k]); nom.append(f"p_corte_lag{k}")
        for m in MISMA_HORA:
            idx = t - m
            col.append(precio[idx]); nom.append(f"p_obj_lag{m}")
        ventana = np.stack([precio[c - k] for k in range(1, 25)], axis=1)
        col += [ventana.min(1), ventana.max(1), ventana.mean(1), ventana.std(1)]
        nom += ["p_min24_corte", "p_max24_corte", "p_media24_corte", "p_std24_corte"]
        for cc in EXOG_CORTE:
            col.append(df[cc].to_numpy(np.float64)[c]); nom.append(f"{cc}_corte")
        for cc in EXOG_OBJ:
            col.append(df[cc].to_numpy(np.float64)[t]); nom.append(f"{cc}_obj")
        if con_nuevas:
            for cc in nuevas_cols:
                col.append(df[cc].to_numpy(np.float64)[c]); nom.append(f"{cc}_corte")
        X = np.column_stack(col)
        ok = ~np.isnan(X).any(1)
        return pd.DataFrame(X[ok], columns=nom), precio[t][ok], fechas.to_numpy()[t][ok]

    resultados = {}
    for etiqueta, con_nuevas in [("LEAR actual", False), ("LEAR + variables nuevas", True)]:
        partes, alphas = [], []
        for h in range(1, 25):
            Xtr, ytr, _ = construir_extendido(h, cortes_tr, con_nuevas)
            Xte, yte, fo = construir_extendido(h, cortes_te, con_nuevas)
            sc = StandardScaler().fit(Xtr)
            if h <= 4 or h % 6 == 0:
                alphas.append(LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000)
                              .fit(sc.transform(Xtr), ytr).alpha_)
            m = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(sc.transform(Xtr), ytr)
            partes.append(pd.DataFrame({"fecha_hora": fo, "real": yte,
                                        "pred": m.predict(sc.transform(Xte))}))
        r = pd.concat(partes).drop_duplicates("fecha_hora").set_index("fecha_hora").sort_index()
        resultados[etiqueta] = r
        print(f"  {etiqueta:26s} MAE={np.abs(r['real']-r['pred']).mean():7.2f}  "
              f"MAPE={mape(r['real'],r['pred']):6.2f}%  sMAPE={smape(r['real'],r['pred']):6.2f}%")
    comun = resultados["LEAR actual"].index.intersection(resultados["LEAR + variables nuevas"].index)
    y = resultados["LEAR actual"].loc[comun, "real"].to_numpy()
    p1 = resultados["LEAR actual"].loc[comun, "pred"].to_numpy()
    p2 = resultados["LEAR + variables nuevas"].loc[comun, "pred"].to_numpy()
    print(f"  DM (con nuevas vs actual): p={dm(y, p1, p2):.4f}")
    return resultados["LEAR + variables nuevas"]


def probar_en_ensamble(lear_nuevo):
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    d = d.join(lear_nuevo["pred"].rename("LEAR-nivel2"), how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["LEAR-nivel2", "real"])
    y = d["real"].to_numpy()
    base = qra_cv(d, MEJOR, "smape")
    conv = qra_cv(d, MEJOR + ["LEAR-nivel2"], "smape")
    print(f"\n  {'ensamble actual (6 votantes)':32s} MAE={np.abs(y-base).mean():7.2f}  MAPE={mape(y,base):6.2f}%")
    print(f"  {'+ LEAR con variables nuevas':32s} MAE={np.abs(y-conv).mean():7.2f}  MAPE={mape(y,conv):6.2f}%  "
          f"DM p={dm(y, base, conv):.4f}")


def main():
    t0 = time.time()
    print("=" * 100)
    print("TAMIZ DE CORRELACION PARCIAL (mismo metodo que tamizar_variables_nuevas.py)")
    print("=" * 100)
    series, candidatas = tamiz()
    print(f"\nCandidatas que pasan el tamiz: {candidatas if candidatas else '(ninguna)'}")

    if not candidatas:
        print("\nNinguna variable tiene correlacion parcial no trivial. No se prueban en el modelo")
        print("(mismo criterio que el resto del proyecto: solo lo que pasa el tamiz llega al modelo).")
        print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")
        return

    print("\n" + "=" * 100)
    print(f"PRUEBA EN EL MODELO: {candidatas}")
    print("=" * 100)
    feats = construir_features(series, candidatas)
    lear_nuevo = probar_en_lear(feats, candidatas)
    probar_en_ensamble(lear_nuevo)
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
