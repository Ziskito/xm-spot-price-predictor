# -*- coding: utf-8 -*-
"""
Ensamble de 72h con DIVERSIDAD REAL entre familias.

El ensamble de 72h original (ensamble_72h.py) fallo -- no le gano a N-BEATSx solo en ningun tramo --
porque solo tenia 2 votantes de la MISMA familia (neuralforecast). Aqui se agregan votantes de
familias genuinamente distintas, disponibles para todo el horizonte gracias a la formulacion
DIRECTA (un modelo que recibe el paso h como variable y usa solo el estado conocido en el corte):
redes neuronales (N-BEATSx, N-HiTS), arboles (XGBoost, CatBoost), lineal (Ridge) y naive estacional.

Metodologia: pesos ajustados POR TRAMO de horizonte, validacion cruzada de 5 pliegues por dia para
que toda prediccion evaluada sea fuera de muestra, y test de Diebold-Mariano (HAC) contra el mejor
votante individual y contra la linea base de produccion (N-BEATSx).

Meta-modelo:
  - por defecto (v2): NNLS -- pesos no negativos que minimizan error cuadratico.
  - con --lad (v3): LAD -- pesos no negativos que suman 1 y minimizan error ABSOLUTO, es decir, la
    misma metrica con la que se evalua el proyecto. Es mas robusto a picos. Resulto mejor que NNLS
    con significancia en 1-24h y 25-48h (verificar_dm_orden.py).

Correccion 2026-09-11: el test DM se calculaba sobre la serie concatenada en el orden de los
pliegues (dias en orden aleatorio). El estimador HAC asume orden temporal; al romperlo subestima la
varianza y los p-valores salen artificialmente pequenos. Ahora se ordena cronologicamente antes.

Las bandas se construyen sobre el punto del ensamble usando la dispersion cuantilica de los modelos
neuronales, y se recalibran con calibracion conforme ADAPTATIVA (ventana movil de 30 dias).
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from scipy.optimize import nnls
import statsmodels.api as sm
from stacking_24h_v2 import pesos_lad

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
META = "lad" if "--lad" in sys.argv else "nnls"
VERSION = "v3" if META == "lad" else "v2"

# nombre -> (archivo, columna de punto, columna q10, columna q90)
CANDIDATOS = {
    "N-BEATSx": ("pronostico_con_bandas_72h_2026.csv", "q50", "q10", "q90"),
    "N-BEATSx-exog": ("pronostico_nbeatsx_72h_exog_2026.csv", "q50", "q10", "q90"),
    "N-HiTS": ("pronostico_nhits_72h_2026.csv", "q50", "q10", "q90"),
    "XGB-directo": ("pronostico_xgb_directo_72h_2026.csv", "pred", None, None),
    "XGB-dir-v2": ("pronostico_xgb_directo_v2_72h_2026.csv", "q50", "q10", "q90"),
    "CatB-directo": ("pronostico_catboost_directo_72h_2026.csv", "pred", None, None),
    "CatB-denso": ("pronostico_catboost_denso_72h_2026.csv", "pred", None, None),
    "Ridge-directo": ("pronostico_ridge_directo_72h_2026.csv", "pred", None, None),
    "NaiveEstac": ("pronostico_naive_estacional_72h_2026.csv", "pred", None, None),
}
TRAMOS = [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]
CLAVE = ["fecha_hora", "cutoff", "paso_horas"]


def pesos_nnls_norm(X, y):
    w, _ = nnls(X, y)
    return w / w.sum() if w.sum() > 0 else np.full(X.shape[1], 1 / X.shape[1])


AJUSTAR = pesos_lad if META == "lad" else pesos_nnls_norm


def cargar(excluir=()):
    base, bandas = None, {}
    usados = []
    for nombre, (arch, c50, c10, c90) in CANDIDATOS.items():
        ruta = RES / arch
        if not ruta.exists() or nombre in excluir:
            continue
        d = pd.read_csv(ruta, parse_dates=["fecha_hora", "cutoff"])
        d["paso_horas"] = d["paso_horas"].astype(int)
        cols = CLAVE + ["real", c50] + ([c10, c90] if c10 else [])
        d = d[cols].rename(columns={c50: nombre})
        if c10:
            d = d.rename(columns={c10: f"{nombre}__q10", c90: f"{nombre}__q90"})
            bandas[nombre] = (f"{nombre}__q10", f"{nombre}__q90")
        base = d if base is None else base.merge(d.drop(columns=["real"]), on=CLAVE, how="inner")
        usados.append(nombre)
    print(f"Votantes cargados ({len(usados)}): {', '.join(usados)}")
    print(f"Filas alineadas: {len(base):,}")
    return base, usados, bandas


def main(excluir=()):
    print(f"Meta-modelo: {META.upper()}  ->  archivos {VERSION}")
    df, modelos, bandas = cargar(excluir)
    df["dia"] = df["fecha_hora"].dt.normalize()

    filas_res, partes = [], []
    for nombre, ini, fin in TRAMOS:
        sub = df[(df.paso_horas >= ini) & (df.paso_horas <= fin)].copy()
        maes = {m: np.abs(sub["real"] - sub[m]).mean() for m in modelos}
        mejor = min(maes, key=maes.get)

        dias = np.sort(sub["dia"].unique())
        rng = np.random.default_rng(42)
        pliegues = np.array_split(rng.permutation(dias), 5)

        oos, pesos_todos = [], []
        for dias_test in pliegues:
            tr = sub[~sub["dia"].isin(dias_test)]
            te = sub[sub["dia"].isin(dias_test)]
            w = AJUSTAR(tr[modelos].values, tr["real"].values)
            pesos_todos.append(w)
            p = te[modelos].values @ w
            # banda del ensamble: mismo peso aplicado a los cuantiles de quien los tenga,
            # renormalizado sobre el subconjunto de modelos con bandas
            idx_b = [i for i, m in enumerate(modelos) if m in bandas]
            wb = w[idx_b]
            wb = wb / wb.sum() if wb.sum() > 0 else np.full(len(idx_b), 1 / len(idx_b))
            q10 = sum(te[bandas[modelos[i]][0]].values * ww for i, ww in zip(idx_b, wb))
            q90 = sum(te[bandas[modelos[i]][1]].values * ww for i, ww in zip(idx_b, wb))
            semi = (q90 - q10) / 2
            oos.append(pd.DataFrame({
                "fecha_hora": te["fecha_hora"].values, "cutoff": te["cutoff"].values,
                "paso_horas": te["paso_horas"].values, "real": te["real"].values,
                "q50": p, "q10": p - semi, "q90": p + semi,
            }))

        dft = pd.concat(oos, ignore_index=True)
        partes.append(dft)
        mae_ens = np.abs(dft["real"] - dft["q50"]).mean()
        w_prom = np.mean(pesos_todos, axis=0)

        # el HAC del test DM exige la serie en orden temporal, no en el orden de los pliegues
        comp = (dft.set_index(CLAVE).join(sub.set_index(CLAVE)[modelos], how="inner")
                .sort_index(level=["cutoff", "paso_horas"]))

        def prueba_dm(col):
            d = (comp["real"] - comp[col]).abs().values - (comp["real"] - comp["q50"]).abs().values
            return sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 71})

        dm = prueba_dm(mejor)
        gana = dm.pvalues[0] < 0.05 and dm.tvalues[0] > 0
        base_prod = "N-BEATSx"
        dm_base = prueba_dm(base_prod) if base_prod in modelos else None

        print(f"\n--- Tramo {nombre} ---")
        print("  MAE individuales: " + " | ".join(f"{m}={maes[m]:.2f}" for m in modelos))
        print("  Pesos:            " + " | ".join(f"{m}={w:.3f}" for m, w in zip(modelos, w_prom)))
        print(f"  MAE ensamble: {mae_ens:.2f}   (mejor individual: {mejor} = {maes[mejor]:.2f}, "
              f"mejora {maes[mejor]-mae_ens:+.2f})")
        print(f"  DM vs mejor votante ({mejor}): t={dm.tvalues[0]:.3f} p={dm.pvalues[0]:.5f}  "
              f"{'ENSAMBLE GANA (sig.)' if gana else 'sin diferencia / pierde'}")
        if dm_base is not None:
            g2 = dm_base.pvalues[0] < 0.05 and dm_base.tvalues[0] > 0
            print(f"  DM vs LINEA BASE ({base_prod}, modelo actual de produccion): "
                  f"t={dm_base.tvalues[0]:.3f} p={dm_base.pvalues[0]:.5f}  "
                  f"{'ENSAMBLE GANA (sig.)' if g2 else 'sin diferencia / pierde'}  "
                  f"[mejora {maes[base_prod]-mae_ens:+.2f} MAE = {100*(maes[base_prod]-mae_ens)/maes[base_prod]:+.1f}%]")

        fila = {"tramo": nombre, "mae_ensamble": mae_ens, "mejor_individual": mejor,
                "mae_mejor_individual": maes[mejor], "dm_p": dm.pvalues[0], "gana": gana,
                "dm_p_vs_base": dm_base.pvalues[0] if dm_base is not None else np.nan}
        fila.update({f"mae_{m}": maes[m] for m in modelos})
        fila.update({f"peso_{m}": w for m, w in zip(modelos, w_prom)})
        filas_res.append(fila)

    ens = pd.concat(partes, ignore_index=True).sort_values(["cutoff", "paso_horas"])
    ens.to_csv(RES / f"pronostico_ensamble_72h_{VERSION}_2026.csv", index=False)
    pd.DataFrame(filas_res).to_csv(RES / f"ensamble_72h_{VERSION}_metricas.csv", index=False)

    mae_glob = np.abs(ens["real"] - ens["q50"]).mean()
    mape_glob = (np.abs(ens["real"] - ens["q50"]) / ens["real"]).mean() * 100
    print(f"\n{'='*78}\nGLOBAL 1-72h  MAE={mae_glob:.2f}  MAPE={mape_glob:.2f}%")

    # ---------- calibracion conforme adaptativa (ventana movil 30 dias) ----------
    print(f"\n{'='*78}\nCalibracion conforme adaptativa sobre el ensamble {VERSION}\n{'='*78}")
    NIVEL, VENTANA, CALENT = 0.80, 30, 30
    ens["score"] = np.maximum(ens["q10"] - ens["real"], ens["real"] - ens["q90"])
    ens["fecha"] = ens["fecha_hora"].dt.normalize()

    partes_cal, resumen = [], []
    for nombre, ini, fin in TRAMOS:
        s = ens[(ens.paso_horas >= ini) & (ens.paso_horas <= fin)].sort_values("fecha_hora").reset_index(drop=True)
        margenes = {}
        for fecha in sorted(s["fecha"].unique()):
            hist = s[(s["fecha"] < fecha) & (s["fecha"] >= fecha - pd.Timedelta(days=VENTANA))]
            if len(hist) < 120:
                prev = s[s["fecha"] < fecha]
                margenes[fecha] = prev["score"].quantile(0.90) if len(prev) > 24 else 0.0
            else:
                n = len(hist)
                margenes[fecha] = hist["score"].quantile(min(np.ceil((n + 1) * NIVEL) / n, 1.0))
        s["margen"] = s["fecha"].map(margenes).fillna(0.0)
        s["q10a"], s["q90a"] = s["q10"] - s["margen"], s["q90"] + s["margen"]

        post = s[s["fecha"] >= s["fecha"].min() + pd.Timedelta(days=CALENT)]
        cc = ((post["real"] >= post["q10"]) & (post["real"] <= post["q90"])).mean() * 100
        ca = ((post["real"] >= post["q10a"]) & (post["real"] <= post["q90a"])).mean() * 100
        an = (post["q90a"] - post["q10a"]).mean()
        print(f"{nombre:8s} n={len(post):5d}  cruda={cc:5.1f}% -> adaptativa={ca:5.1f}%  ancho={an:6.1f}")
        resumen.append({"tramo": nombre, "cobertura_cruda": cc, "cobertura_adaptativa": ca,
                        "ancho_adaptativo": an, "n_eval": len(post)})
        o = s[CLAVE + ["real", "q50", "margen"]].copy()
        o["q10"], o["q90"] = s["q10a"], s["q90a"]
        partes_cal.append(o)

    final = pd.concat(partes_cal, ignore_index=True).sort_values(["cutoff", "paso_horas"])
    ruta_final = RES / f"pronostico_ensamble_72h_{VERSION}_adaptativo.csv"
    final.to_csv(ruta_final, index=False)
    pd.DataFrame(resumen).to_csv(RES / f"ensamble_72h_{VERSION}_metricas_adaptativo.csv", index=False)
    print(f"\nContrato 24-72h {VERSION} guardado: {ruta_final}")


if __name__ == "__main__":
    main(tuple(a for a in sys.argv[1:] if not a.startswith("--")))
