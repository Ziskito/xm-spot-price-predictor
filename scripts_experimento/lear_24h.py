# -*- coding: utf-8 -*-
"""
LEAR de 24h: el benchmark canonico day-ahead, aplicado al especialista de 24 horas.

En 72h el LEAR resulto ser el mejor modelo individual del proyecto y, metido al ensamble, permitio
sacar los tres arboles y mejorar los tres tramos (ver bitacora). Aqui se hace lo mismo para 24h.

Detalle que hace este caso MAS canonico que el de 72h: el protocolo day-ahead del proyecto usa
ventanas 01:00 -> 00:00 con corte a las 00:00, asi que el paso h y la HORA DEL DIA estan en
correspondencia 1 a 1 (h=1 <-> 01:00, ..., h=24 <-> 00:00). Por tanto "un modelo LASSO por paso"
ES "un modelo LASSO por hora del dia", que es exactamente como se construye el LEAR en la
literatura de day-ahead (Lago et al. 2021). No es una adaptacion: es la receta original.

Anti-fuga: para el objetivo t, el corte es c = ultimo 00:00 anterior a t (es decir,
c = floor_dia(t - 1h)), y solo se usa informacion disponible en c. Los rezagos "misma hora de dias
previos" (t-24, t-48, t-72, t-168) son todos validos porque h <= 24 implica t-24 <= c.
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

CORTE = pd.Timestamp("2026-01-01")
H = 24

# Rezagos de precio medidos DESDE EL CORTE hacia atras (siempre conocidos en el corte)
REZAGOS_CORTE = [0, 1, 2, 3, 4, 5, 6, 12, 18, 24, 48, 72, 96, 120, 144, 168]
# "Misma hora" de dias previos, relativos al OBJETIVO (validos porque h<=24)
MISMA_HORA = [24, 48, 72, 96, 120, 144, 168]

EXOG_CORTE = ["volumen_embalses", "volumen_embalses_vs_media30d", "volumen_embalses_delta_1d",
              "aportes_hidricos", "aportes_hidricos_vs_media30d", "aportes_hidricos_delta_1d",
              "demanda", "demanda_media_24h", "generacion",
              "precio_media_24h", "precio_media_7d", "precio_media_30d",
              "precio_std_24h", "precio_std_7d", "ratio_volatilidad"]
EXOG_OBJ = ["oni", "es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
            "dia_anio_sin", "dia_anio_cos", "es_festivo", "festivo_lag24h", "festivo_lag168h",
            "mismatch_festivo_24h", "mismatch_festivo_168h"]


def construir_para_paso(df, h, idx_cortes):
    """Matriz de diseno del LEAR para un paso h fijo (equivalente a una hora del dia)."""
    n = len(df)
    precio = df["precio_bolsa"].to_numpy(np.float64)
    fechas = df["fecha_hora"].to_numpy()
    c = idx_cortes[(idx_cortes + h < n) & (idx_cortes - max(REZAGOS_CORTE) >= 0)]
    t = c + h

    col, nom = [], []
    for k in REZAGOS_CORTE:
        col.append(precio[c - k]); nom.append(f"p_corte_lag{k}")
    for m in MISMA_HORA:
        idx = t - m
        if (idx < 0).any() or (idx > c).any():          # nunca debe pasar con h<=24, se valida igual
            continue
        col.append(precio[idx]); nom.append(f"p_obj_lag{m}")
    ventana = np.stack([precio[c - k] for k in range(1, 25)], axis=1)
    col += [ventana.min(1), ventana.max(1), ventana.mean(1), ventana.std(1)]
    nom += ["p_min24_corte", "p_max24_corte", "p_media24_corte", "p_std24_corte"]
    for cc in EXOG_CORTE:
        col.append(df[cc].to_numpy(np.float64)[c]); nom.append(f"{cc}_corte")
    for cc in EXOG_OBJ:
        col.append(df[cc].to_numpy(np.float64)[t]); nom.append(f"{cc}_obj")
    X = np.column_stack(col)
    ok = ~np.isnan(X).any(1)
    return pd.DataFrame(X[ok], columns=nom), precio[t][ok], fechas[t][ok], fechas[c][ok]


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def metricas(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    e = real - pred
    return {"MAE": float(np.abs(e).mean()),
            "RMSE": float(np.sqrt((e ** 2).mean())),
            "MAPE_%": float(np.mean(np.abs(e) / real) * 100),
            "sMAPE_%": float(np.mean(np.abs(e) / ((np.abs(real) + np.abs(pred)) / 2)) * 100),
            "R2": float(1 - np.sum(e ** 2) / np.sum((real - real.mean()) ** 2))}


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler

    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    # cortes = todos los 00:00 disponibles
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - H)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]

    print(f"Cortes de entrenamiento: {len(cortes_tr)} | de prueba: {len(cortes_te)}")
    print(f"Un modelo LASSO por cada uno de los {H} pasos (= horas del dia)\n")

    partes, alphas, n_sel = [], [], []
    for h in range(1, H + 1):
        Xtr, ytr, _, _ = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_obj, f_cor = construir_para_paso(df, h, cortes_te)
        esc = StandardScaler().fit(Xtr)
        Ztr, Zte = esc.transform(Xtr), esc.transform(Xte)
        if h <= 4 or h % 6 == 0:
            m = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr)
            alphas.append(m.alpha_)
        else:
            m = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr)
        n_sel.append(int((np.abs(m.coef_) > 1e-8).sum()))
        partes.append(pd.DataFrame({"fecha_hora": f_obj, "cutoff": f_cor, "paso_horas": h,
                                    "real": yte, "LEAR24": m.predict(Zte)}))
        print(f"  paso {h:2d}/24 ({(time.time()-t0)/60:4.1f} min) alpha={float(np.median(alphas)):6.3f} "
              f"activas={n_sel[-1]:2d} MAE={np.abs(yte - m.predict(Zte)).mean():6.2f}", flush=True)

    lear = pd.concat(partes, ignore_index=True).sort_values("fecha_hora")
    lear = lear[lear["fecha_hora"] >= CORTE]
    lear.to_csv(RES / "pronostico_lear_24h_2026.csv", index=False)
    print(f"\nLEAR24: {len(lear):,} horas | variables activas mediana={int(np.median(n_sel))}")

    # ---------- comparacion contra los votantes actuales y el ensamble v4 ----------
    from stacking_24h_v2 import cargar as cargar24, BASE5
    d = cargar24()
    v4 = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    j = d.join(lear.set_index("fecha_hora")["LEAR24"], how="inner")
    j = j.join(v4["pred_ensamble"].rename("Ensamble v4"), how="left")
    print(f"Filas comparables: {len(j):,}\n")

    print(f"{'='*88}\nMODELOS INDIVIDUALES DE 24h vs LEAR24\n{'='*88}")
    print(f"{'modelo':18s} {'MAE':>8s} {'RMSE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s} {'R2':>7s}")
    filas = []
    for c in ["LEAR24"] + BASE5 + ["CatB-dir24", "Ridge-dir24", "Ensamble v4"]:
        if c not in j.columns:
            continue
        s = j.dropna(subset=[c])
        m = metricas(s["real"], s[c])
        m["modelo"] = c
        filas.append(m)
        print(f"{c:18s} {m['MAE']:8.2f} {m['RMSE']:8.2f} {m['MAPE_%']:8.2f} {m['sMAPE_%']:8.2f} {m['R2']:7.3f}")
    pd.DataFrame(filas).to_csv(RES / "lear_24h_metricas.csv", index=False)

    print(f"\n{'='*88}\nDM: LEAR24 contra cada referencia\n{'='*88}")
    for ref in ["N-BEATSx", "XGBoost", "Ridge-dir24", "Ensamble v4"]:
        if ref not in j.columns:
            continue
        s = j.dropna(subset=["LEAR24", ref])
        y = s["real"].to_numpy()
        t, pv = dm(y, s[ref].to_numpy(), s["LEAR24"].to_numpy())
        delta = np.abs(y - s["LEAR24"]).mean() - np.abs(y - s[ref]).mean()
        marca = ("LEAR24 GANA" if t > 0 else "LEAR24 pierde") if pv < 0.05 else "sin diferencia"
        print(f"  vs {ref:14s} delta={delta:+7.2f}  p={pv:.4f}  {marca}")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
