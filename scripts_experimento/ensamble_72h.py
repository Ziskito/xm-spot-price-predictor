# -*- coding: utf-8 -*-
"""
Extension del ensamble de stacking (que gano a 24h, MAPE 12.05%) al horizonte completo 24-72h --
pedido explicito del usuario antes de dormir, para no dejar el trabajo de hoy solo en el punto de
24h mientras el Anexo 1 (OE2.2) exige comparar en todo el rango 24-72h.

Limitacion documentada explicitamente (no oculta): Persistencia, XGBoost y ARX+GARCH estan
estructuralmente atados a "precio_lag24h" como regresor ancla -- extenderlos a cada paso de
25-72h requeriria 71 reentrenamientos adicionales por modelo (uno por paso), fuera de alcance
para esta ronda. Por eso el ensamble de 24-72h combina SOLO los modelos nativamente multi-horizonte
(N-BEATSx y N-HiTS, entrenados con h=72 y MQLoss) -- los unicos capaces de producir el horizonte
completo de una sola pasada. El ensamble de 5-6 modelos completo se mantiene como el mejor
resultado especificamente en el punto de 24h (documentado aparte).

Metodologia:
1. Combinar N-BEATSx y N-HiTS (cuantil 50) via NNLS, con pesos AJUSTADOS POR TRAMO de horizonte
   por separado (1-24h/25-48h/49-72h) -- el peso relativo de cada modelo puede cambiar con el
   horizonte (ya vimos que N-BEATSx y N-HiTS tienen fuerzas distintas segun el paso).
2. Validacion cruzada de 5 pliegues por dia, mismo criterio que el ensamble de 24h.
3. Bandas: se combinan tambien q10/q90 de ambos modelos con los MISMOS pesos NNLS del punto, y
   se recalibra con calibracion conforme ADAPTATIVA (ventana movil 30 dias) -- exactamente el
   mismo metodo que ya funciono mejor que la version estatica para las bandas de N-BEATSx solo.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import nnls
import statsmodels.api as sm

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_NBEATSX = RAIZ / "data/processed/resultados/pronostico_con_bandas_72h_2026.csv"
RUTA_NHITS = RAIZ / "data/processed/resultados/pronostico_nhits_72h_2026.csv"

nbeatsx = pd.read_csv(RUTA_NBEATSX, parse_dates=["fecha_hora", "cutoff"])
nhits = pd.read_csv(RUTA_NHITS, parse_dates=["fecha_hora", "cutoff"])

nbeatsx = nbeatsx.rename(columns={"q10": "q10_nbeatsx", "q50": "q50_nbeatsx", "q90": "q90_nbeatsx"})
nhits = nhits.rename(columns={"q10": "q10_nhits", "q50": "q50_nhits", "q90": "q90_nhits"})

df = nbeatsx.merge(nhits[["fecha_hora", "cutoff", "paso_horas", "q10_nhits", "q50_nhits", "q90_nhits"]],
                     on=["fecha_hora", "cutoff", "paso_horas"], how="inner")
print(f"Filas alineadas entre N-BEATSx y N-HiTS: {len(df)} de {len(nbeatsx)} (N-BEATSx) / {len(nhits)} (N-HiTS)")

df["dia"] = df["fecha_hora"].dt.normalize()
tramos = [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]

resultados_tramo = []
partes_ensamble = []

for nombre, ini, fin in tramos:
    sub = df[(df["paso_horas"] >= ini) & (df["paso_horas"] <= fin)].copy()
    dias = np.sort(sub["dia"].unique())
    rng = np.random.default_rng(42)
    dias_shuffled = rng.permutation(dias)
    pliegues = np.array_split(dias_shuffled, 5)

    filas_oos = []
    pesos_por_pliegue = []
    for dias_test in pliegues:
        dias_train = np.setdiff1d(dias, dias_test)
        train_p = sub[sub["dia"].isin(dias_train)]
        test_p = sub[sub["dia"].isin(dias_test)]

        X_train = train_p[["q50_nbeatsx", "q50_nhits"]].values
        y_train = train_p["real"].values
        pesos, _ = nnls(X_train, y_train)
        pesos_n = pesos / pesos.sum() if pesos.sum() > 0 else np.array([0.5, 0.5])
        pesos_por_pliegue.append(pesos_n)

        q50_ens = test_p["q50_nbeatsx"].values * pesos_n[0] + test_p["q50_nhits"].values * pesos_n[1]
        q10_ens = test_p["q10_nbeatsx"].values * pesos_n[0] + test_p["q10_nhits"].values * pesos_n[1]
        q90_ens = test_p["q90_nbeatsx"].values * pesos_n[0] + test_p["q90_nhits"].values * pesos_n[1]

        for fh, cut, ph, real_v, q10v, q50v, q90v in zip(
            test_p["fecha_hora"], test_p["cutoff"], test_p["paso_horas"], test_p["real"],
            q10_ens, q50_ens, q90_ens
        ):
            filas_oos.append({"fecha_hora": fh, "cutoff": cut, "paso_horas": ph, "real": real_v,
                               "q10": q10v, "q50": q50v, "q90": q90v})

    df_tramo = pd.DataFrame(filas_oos)
    partes_ensamble.append(df_tramo)

    mae_ens = np.abs(df_tramo["real"] - df_tramo["q50"]).mean()
    mae_nbeatsx = np.abs(sub["real"] - sub["q50_nbeatsx"]).mean()
    mae_nhits = np.abs(sub["real"] - sub["q50_nhits"]).mean()
    pesos_promedio = np.mean(pesos_por_pliegue, axis=0)

    print(f"\n--- Tramo {nombre} ---")
    print(f"  Pesos promedio: N-BEATSx={pesos_promedio[0]:.3f}  N-HiTS={pesos_promedio[1]:.3f}")
    print(f"  MAE N-BEATSx solo: {mae_nbeatsx:.2f}  |  MAE N-HiTS solo: {mae_nhits:.2f}  |  MAE ensamble: {mae_ens:.2f}")

    # DM test: ensamble vs el mejor de los dos individuales en este tramo
    mejor_individual_col = "q50_nbeatsx" if mae_nbeatsx < mae_nhits else "q50_nhits"
    sub_idx = sub.set_index(["fecha_hora", "cutoff", "paso_horas"])
    tramo_idx = df_tramo.set_index(["fecha_hora", "cutoff", "paso_horas"])
    comp = tramo_idx.join(sub_idx[[mejor_individual_col]], how="inner")
    e_ens = (comp["real"] - comp["q50"]).abs().values
    e_ind = (comp["real"] - comp[mejor_individual_col]).abs().values
    dm = sm.OLS(e_ind - e_ens, np.ones_like(e_ens)).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
    concl = "ENSAMBLE GANA (sig.)" if (dm.pvalues[0] < 0.05 and dm.tvalues[0] > 0) else "sin diferencia / pierde"
    print(f"  DM (ensamble vs mejor individual, {mejor_individual_col}): t={dm.tvalues[0]:.3f}  p={dm.pvalues[0]:.5f}  {concl}")

    resultados_tramo.append({"tramo": nombre, "mae_nbeatsx": mae_nbeatsx, "mae_nhits": mae_nhits,
                              "mae_ensamble": mae_ens, "peso_nbeatsx": pesos_promedio[0],
                              "peso_nhits": pesos_promedio[1], "dm_p": dm.pvalues[0], "conclusion": concl})

df_ensamble_72h = pd.concat(partes_ensamble, ignore_index=True).sort_values(["cutoff", "paso_horas"])
RUTA_ENSAMBLE = RAIZ / "data/processed/resultados/pronostico_ensamble_72h_2026.csv"
df_ensamble_72h.to_csv(RUTA_ENSAMBLE, index=False)
pd.DataFrame(resultados_tramo).to_csv(RAIZ / "data/processed/resultados/ensamble_72h_metricas.csv", index=False)

print(f"\n{'='*70}\nRESUMEN\n{'='*70}")
print(pd.DataFrame(resultados_tramo).to_string(index=False))
print(f"\nGuardado: {RUTA_ENSAMBLE}")

# ================== Calibracion conforme ADAPTATIVA sobre el ensamble ==================
print(f"\n{'='*70}\nCalibracion conforme adaptativa sobre el ensamble 24-72h\n{'='*70}")

NIVEL_OBJETIVO = 0.80
VENTANA_DIAS = 30
DIAS_CALENTAMIENTO = 30

df_c = df_ensamble_72h.copy()
df_c["score"] = np.maximum(df_c["q10"] - df_c["real"], df_c["real"] - df_c["q90"])
df_c["fecha"] = df_c["fecha_hora"].dt.normalize()

partes_cal = []
resumen_cal = []
for nombre, ini, fin in tramos:
    sub = df_c[(df_c["paso_horas"] >= ini) & (df_c["paso_horas"] <= fin)].copy().sort_values("fecha_hora").reset_index(drop=True)
    fechas_unicas = sorted(sub["fecha"].unique())
    margenes = {}
    for fecha in fechas_unicas:
        ventana_ini = fecha - pd.Timedelta(days=VENTANA_DIAS)
        historia = sub[(sub["fecha"] < fecha) & (sub["fecha"] >= ventana_ini)]
        if len(historia) < 24 * 5:
            hist_previa = sub[sub["fecha"] < fecha]
            margenes[fecha] = hist_previa["score"].quantile(0.90) if len(hist_previa) > 24 else 0.0
            continue
        n = len(historia)
        nivel_ajustado = min(np.ceil((n + 1) * NIVEL_OBJETIVO) / n, 1.0)
        margenes[fecha] = historia["score"].quantile(nivel_ajustado)

    sub["margen"] = sub["fecha"].map(margenes).fillna(0.0)
    sub["q10_adapt"] = sub["q10"] - sub["margen"]
    sub["q90_adapt"] = sub["q90"] + sub["margen"]

    inicio_eval = fechas_unicas[0] + pd.Timedelta(days=DIAS_CALENTAMIENTO)
    post = sub[sub["fecha"] >= inicio_eval]
    cob_cruda = ((post["real"] >= post["q10"]) & (post["real"] <= post["q90"])).mean() * 100
    cob_adapt = ((post["real"] >= post["q10_adapt"]) & (post["real"] <= post["q90_adapt"])).mean() * 100
    ancho_adapt = (post["q90_adapt"] - post["q10_adapt"]).mean()
    print(f"{nombre:8s}  n_eval={len(post):5d}  cobertura cruda={cob_cruda:5.1f}%  -> adaptativa={cob_adapt:5.1f}%  ancho={ancho_adapt:6.1f}")
    resumen_cal.append({"tramo": nombre, "cobertura_cruda": cob_cruda, "cobertura_adaptativa": cob_adapt,
                         "ancho_adaptativo": ancho_adapt, "n_eval": len(post)})

    salida_tramo = sub[["fecha_hora", "cutoff", "paso_horas", "real", "q50", "margen"]].copy()
    salida_tramo["q10"] = sub["q10_adapt"]
    salida_tramo["q90"] = sub["q90_adapt"]
    partes_cal.append(salida_tramo)

salida_final = pd.concat(partes_cal, ignore_index=True).sort_values(["cutoff", "paso_horas"])
RUTA_FINAL = RAIZ / "data/processed/resultados/pronostico_ensamble_72h_2026_adaptativo.csv"
salida_final.to_csv(RUTA_FINAL, index=False)
pd.DataFrame(resumen_cal).to_csv(RAIZ / "data/processed/resultados/ensamble_72h_metricas_adaptativo.csv", index=False)

print(f"\nContrato FINAL 24-72h (ensamble N-BEATSx+N-HiTS, calibracion adaptativa) para OE3:")
print(f"  {RUTA_FINAL}")
print("\nDone.")
