# -*- coding: utf-8 -*-
"""
La arquitectura de dos modelos que pidio el usuario: uno para los dias NORMALES y otro ENTRENADO
ESPECIFICAMENTE con los dias dificiles.

Por que esto NO es lo que ya se probo. Hasta ahora se habia probado:
  - especialistas por HORA DEL RELOJ (18-20): entrenados solo con esas horas -> peores;
  - ENRUTAR las horas de alto riesgo a modelos YA EXISTENTES (N-BEATSx, CatBoost, LEAR...): todos
    peores que el ensamble justamente en esas horas.
Lo que falta, y es lo que se pidio, es distinto en dos cosas: (1) la unidad es el DIA, no la hora;
(2) el especialista se ENTRENA con dias dificiles del historico, en vez de enrutar a un modelo
generalista ya entrenado.

Problema de diseno que hay que resolver: para entrenar un especialista "con los dias donde falla el
ensamble" haria falta el error del ensamble en 2019-2025, pero el ensamble solo existe para 2026
(Origen 6). Solucion: definir la dificultad con un criterio OBSERVABLE y calculable en train y en
test por igual, y verificar despues que ese criterio si se corresponde con los dias donde el
ensamble falla en 2026. Se usan dos definiciones:
    D1) dia con alta VOLATILIDAD intradiaria del precio (rango/media de las 24h del dia);
    D2) dia con gran SALTO respecto a la semana previa (|media del dia - media de los 7 dias
        anteriores| / media de los 7 dias anteriores).

Arquitectura evaluada, toda causal:
  1. Clasificador de dia dificil: predice EN EL CORTE (00:00) si el dia sera dificil.
  2. Modelo general: entrenado con dias normales.
  3. Modelo especialista: entrenado SOLO con dias dificiles del historico.
  4. En prediccion: el clasificador decide a cual se enruta cada dia.
Se compara contra el ensamble v4 y contra el especialista aplicado con un ORACULO de dificultad
(que conoce la etiqueta real), para separar "el especialista no sirve" de "el clasificador no
acierta".
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
from lear_24h import construir_para_paso

CORTE = pd.Timestamp("2026-01-01")
H = 24


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def etiquetas_dificultad(df):
    """Dificultad por dia, con criterios observables (calculables en train y test igual)."""
    d = df[["fecha_hora", "precio_bolsa"]].copy()
    d["dia"] = d["fecha_hora"].dt.normalize()
    g = d.groupby("dia")["precio_bolsa"].agg(["mean", "max", "min"])
    g["volatilidad"] = (g["max"] - g["min"]) / g["mean"].clip(lower=1)
    media7 = g["mean"].rolling(7, min_periods=3).mean().shift(1)
    g["salto"] = (g["mean"] - media7).abs() / media7.clip(lower=1)
    return g


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    df = cargar_completo().reset_index(drop=True)
    g = etiquetas_dificultad(df)

    # --- verificar que los criterios se corresponden con los fallos reales del ensamble en 2026 ---
    v4 = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"])
    v4 = v4.dropna(subset=["real", "pred_ensamble"])
    v4["dia"] = v4["fecha_hora"].dt.normalize()
    err_dia = (v4.assign(ape=(v4["real"] - v4["pred_ensamble"]).abs() / v4["real"] * 100)
                 .groupby("dia")["ape"].mean())
    comp = g.join(err_dia.rename("mape_ensamble"), how="inner").dropna()
    print(f"{'='*94}\nVALIDEZ DE LOS CRITERIOS DE DIFICULTAD (contra el error real del ensamble en 2026)\n{'='*94}")
    for crit in ["volatilidad", "salto"]:
        print(f"  correlacion {crit:12s} vs MAPE diario del ensamble: "
              f"{comp[crit].corr(comp['mape_ensamble']):+.3f}")
    print(f"  dias evaluados: {len(comp)}")

    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - H)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]

    for CRIT in ["volatilidad", "salto"]:
        print(f"\n{'='*94}\nARQUITECTURA DE DOS MODELOS -- dificultad por '{CRIT}'\n{'='*94}")
        umbral = g.loc[g.index < CORTE, CRIT].quantile(0.75)   # umbral del TRAIN, sin fuga
        dificil = (g[CRIT] >= umbral)
        print(f"  umbral (percentil 75 del train): {umbral:.3f} | "
              f"dias dificiles en train: {dificil[dificil.index < CORTE].mean()*100:.0f}%, "
              f"en 2026: {dificil[dificil.index >= CORTE].mean()*100:.0f}%")

        # ---------- 1) clasificador causal de dia dificil, desde el estado en el corte ----------
        est_cols = ["precio_bolsa", "precio_media_24h", "precio_media_7d", "precio_std_24h",
                    "precio_std_7d", "precio_rango_24h", "ratio_volatilidad",
                    "volumen_embalses_vs_media30d", "aportes_hidricos_vs_media30d",
                    "demanda_media_24h", "oni"]
        E = df.set_index("fecha_hora")[est_cols]
        Xc = E.reindex(g.index).dropna()
        yc = dificil.reindex(Xc.index).astype(int)
        m_tr = Xc.index < CORTE
        clf = CatBoostClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                                 random_seed=42, verbose=0, thread_count=-1)
        clf.fit(Xc[m_tr], yc[m_tr])
        p_dif = pd.Series(clf.predict_proba(Xc[~m_tr])[:, 1], index=Xc.index[~m_tr])
        auc = roc_auc_score(yc[~m_tr], p_dif)
        print(f"  clasificador de dia dificil: AUC en 2026 = {auc:.3f}")

        # ---------- 2 y 3) modelos general y especialista ----------
        dias_tr = pd.DatetimeIndex(fechas.to_numpy()[cortes_tr]).normalize()
        es_dif_tr = dificil.reindex(dias_tr).fillna(False).to_numpy()
        salidas = {}
        for nombre, mascara in [("general (dias normales)", ~es_dif_tr),
                                ("especialista (dias dificiles)", es_dif_tr),
                                ("unico (todos los dias)", np.ones(len(cortes_tr), bool))]:
            cortes = cortes_tr[mascara]
            if len(cortes) < 200:
                continue
            partes, alphas = [], []
            for h in range(1, H + 1):
                Xtr, ytr, _, _ = construir_para_paso(df, h, cortes)
                Xte, yte, f_te, _ = construir_para_paso(df, h, cortes_te)
                sc = StandardScaler().fit(Xtr)
                if h <= 4:
                    mm = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(sc.transform(Xtr), ytr)
                    alphas.append(mm.alpha_)
                else:
                    mm = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(
                        sc.transform(Xtr), ytr)
                partes.append(pd.DataFrame({"fecha_hora": f_te, "real": yte,
                                            "pred": mm.predict(sc.transform(Xte))}))
            r = pd.concat(partes, ignore_index=True)
            salidas[nombre] = r[r["fecha_hora"] >= CORTE].set_index("fecha_hora").sort_index()
            print(f"  {nombre:32s} entrenado con {len(cortes):4d} dias", flush=True)

        # ---------- 4) enrutamiento ----------
        base = salidas["unico (todos los dias)"]
        y = base["real"].to_numpy()
        dias_te = base.index.normalize()
        print(f"\n  {'esquema':46s} {'MAE':>8s} {'MAPE %':>8s}   DM vs modelo unico")
        print(f"  {'modelo unico (referencia)':46s} {np.abs(y-base['pred']).mean():8.2f} "
              f"{mape(y, base['pred']):8.2f}")
        gen, esp = salidas.get("general (dias normales)"), salidas.get("especialista (dias dificiles)")
        if gen is None or esp is None:
            print("  (no hay suficientes dias para alguno de los dos modelos)")
            continue
        for etiqueta, sel in [
            ("enrutado por el CLASIFICADOR", pd.Series(p_dif >= 0.5).reindex(dias_te).fillna(False).to_numpy()),
            ("enrutado por ORACULO de dificultad", dificil.reindex(dias_te).fillna(False).to_numpy()),
            ("siempre el especialista", np.ones(len(base), bool)),
            ("siempre el general", np.zeros(len(base), bool)),
        ]:
            p = np.where(sel, esp["pred"].to_numpy(), gen["pred"].to_numpy())
            t, pv = dm(y, base["pred"].to_numpy(), p)
            marca = ("MEJORA" if t > 0 else "peor") if pv < 0.05 else "n.s."
            print(f"  {etiqueta:46s} {np.abs(y-p).mean():8.2f} {mape(y, p):8.2f}   p={pv:.4f} {marca}")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
