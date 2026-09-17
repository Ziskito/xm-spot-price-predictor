# -*- coding: utf-8 -*-
"""
Transformaciones estabilizadoras de varianza (VST) para el Ridge directo, a 24h y a 72h.

Base bibliografica:
  - Uniejewski, Weron & Ziel (2018), "Variance stabilizing transformations for electricity spot
    price forecasting", IEEE Trans. Power Systems 33:2219-2229 -- estandarizar con mediana/MAD y
    aplicar asinh antes de ajustar el modelo.
  - Uniejewski (2026), "Variance stabilizing transformations for electricity price forecasting in
    periods of increased volatility", Electric Power Systems Research 257:112992 -- propone un asinh
    PARAMETRIZADO por la pendiente en el origen c, y el esquema AVGroll (promedio movil de las mejores
    combinaciones de transformaciones), con ganancias de hasta -17.7% y especialmente en regimenes
    volatiles. Para modelos lineales (LEAR) reporta hasta -14.6%.

Por que Ridge y no CatBoost: con perdida MAE los arboles estiman la mediana condicional, que es
equivariante ante transformaciones monotonas del objetivo, asi que una VST casi no los cambia. El
Ridge minimiza error CUADRATICO, dominado por los picos: ahi es donde estabilizar la varianza
deberia pesar. Y Ridge directo es el mejor modelo individual del proyecto a horizonte largo.

Transformaciones (sobre z = (p - mediana)/MAD, mediana y MAD de los OBJETIVOS de entrenamiento):
  - id           : z (equivale al Ridge directo actual -- control de cordura)
  - asinh(c)     : sgn(z) [asinh(|z| + k) - asinh(k)],  k = sqrt(1/c^2 - 1),  c in {0.1..1.0}
                   (c=1 es el asinh estandar; la pendiente en el origen es exactamente c)
  - boxcox(l)    : sgn(z) ((|z|+1)^l - 1)/l,  l=0 -> sgn(z) log(|z|+1)   (variante robusta de Sakia)
Se transforman el objetivo y las regresoras de NIVEL de precio con la misma transformacion.

Esquemas de seleccion (causales, siguiendo el paper, ventana movil de 56 dias):
  - SELroll: en cada corte, para cada grupo, la configuracion con menor MAE en los 56 dias previos.
  - AVGroll: igual, pero entre todas las combinaciones de 1, 2 o 3 configuraciones (promedio simple).
  Grupo = hora del dia (24h, como el paper) o tramo de horizonte (72h, adaptacion: con cortes cada
  3 dias no hay suficientes observaciones por hora y por paso).
Durante el calentamiento (menos de 14 dias de historia) se usa el asinh estandar (c=1).
"""
import sys
import time
import itertools
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from xgboost_directo_72h import construir_largo, COLS_ESTADO, COLS_FUTURO, PASO_MUESTREO_TRAIN

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
COLS_NIVEL = ["precio_bolsa", "precio_media_24h", "precio_media_7d", "precio_media_30d",
              "precio_lag168h", "precio_mismo_hora_disp"]
CONFIGS = ([("id", None)]
           + [("asinh", round(c, 1)) for c in np.arange(0.1, 1.01, 0.1)]
           + [("boxcox", round(l, 1)) for l in np.arange(0.0, 1.01, 0.1)])
VENTANA_DIAS = 56


def etiqueta(tipo, par):
    return "id" if tipo == "id" else f"{tipo}_{par:.1f}"


def transformar(v, med, mad, tipo, par):
    z = (v - med) / mad
    if tipo == "id":
        return z
    if tipo == "asinh":
        k = np.sqrt(1 / par ** 2 - 1)
        return np.sign(z) * (np.arcsinh(np.abs(z) + k) - np.arcsinh(k))
    a = np.abs(z) + 1
    return np.sign(z) * (np.log(a) if par == 0 else (a ** par - 1) / par)


def invertir(y, med, mad, tipo, par):
    if tipo == "id":
        z = y
    elif tipo == "asinh":
        k = np.sqrt(1 / par ** 2 - 1)
        z = np.sign(y) * (np.sinh(np.abs(y) + np.arcsinh(k)) - k)
    else:
        a = np.exp(np.abs(y)) if par == 0 else np.maximum(par * np.abs(y) + 1, 0) ** (1 / par)
        z = np.sign(y) * (a - 1)
    return z * mad + med


def ajustar_configs(tr, te):
    from sklearn.linear_model import Ridge
    cols_x = COLS_ESTADO + COLS_FUTURO + ["precio_mismo_hora_disp", "paso_horas"]
    fijas = [c for c in cols_x if c not in COLS_NIVEL]
    A_tr = tr[fijas].to_numpy(np.float64)
    mu, sd = A_tr.mean(0), A_tr.std(0)
    sd[sd == 0] = 1
    A_tr = (A_tr - mu) / sd
    A_te = (te[fijas].to_numpy(np.float64) - mu) / sd
    y = tr["y"].to_numpy(np.float64)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    tope = 2 * y.max()  # red de seguridad numerica ante sinh/potencias explosivas
    print(f"  mediana={med:.1f}  MAD={mad:.1f}  (objetivos de entrenamiento)")
    P = {}
    for tipo, par in CONFIGS:
        B_tr = np.column_stack([transformar(tr[c].to_numpy(np.float64), med, mad, tipo, par) for c in COLS_NIVEL])
        B_te = np.column_stack([transformar(te[c].to_numpy(np.float64), med, mad, tipo, par) for c in COLS_NIVEL])
        m2, s2 = B_tr.mean(0), B_tr.std(0)
        X_tr = np.hstack([A_tr, (B_tr - m2) / s2])
        X_te = np.hstack([A_te, (B_te - m2) / s2])
        r = Ridge(alpha=10.0).fit(X_tr, transformar(y, med, mad, tipo, par))
        pred = invertir(r.predict(X_te), med, mad, tipo, par)
        P[etiqueta(tipo, par)] = np.clip(np.nan_to_num(pred, nan=med, posinf=tope), 0, tope)
        del X_tr, X_te
    return pd.DataFrame(P)


def esquemas(te, P, min_unidades):
    M = P.to_numpy()
    y = te["real"].to_numpy()
    E = M - y[:, None]
    k = M.shape[1]
    c2 = np.array(list(itertools.combinations(range(k), 2)))
    c3 = np.array(list(itertools.combinations(range(k), 3)))
    lit = list(P.columns).index("asinh_1.0")
    uni = te["unidad"].to_numpy()
    gru = te["grupo"].to_numpy()
    sel = np.full(len(y), np.nan)
    avg = np.full(len(y), np.nan)
    calent = np.zeros(len(y), bool)
    for u in np.sort(np.unique(uni)):
        fu = uni == u
        hist = (uni < u) & (uni >= u - np.timedelta64(VENTANA_DIAS, "D"))
        if len(np.unique(uni[hist])) < min_unidades:
            sel[fu] = M[fu, lit]
            avg[fu] = M[fu, lit]
            calent[fu] = True
            continue
        for g in np.unique(gru[fu]):
            filas = fu & (gru == g)
            Eh = E[hist & (gru == g)]
            m1 = np.abs(Eh).mean(0)
            m2 = np.abs((Eh[:, c2[:, 0]] + Eh[:, c2[:, 1]]) / 2).mean(0)
            m3 = np.abs((Eh[:, c3[:, 0]] + Eh[:, c3[:, 1]] + Eh[:, c3[:, 2]]) / 3).mean(0)
            sel[filas] = M[filas, np.argmin(m1)]
            mejor = min((m1.min(), [np.argmin(m1)]), (m2.min(), list(c2[np.argmin(m2)])),
                        (m3.min(), list(c3[np.argmin(m3)])), key=lambda t: t[0])
            avg[filas] = M[filas][:, mejor[1]].mean(1)
    return sel, avg, calent


def dm(real, p_ref, p_nuevo, maxlags):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return r.tvalues[0], r.pvalues[0]


def main():
    t0 = time.time()
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    pos = pd.Series(df.index.values, index=df["fecha_hora"])
    idx_fin = int(pos[pd.Timestamp("2026-01-01")])

    ref72 = pd.read_csv(RES / "pronostico_con_bandas_72h_2026.csv", parse_dates=["cutoff"])
    idx72 = pos.reindex(pd.Index(sorted(ref72["cutoff"].unique()))).dropna().astype(int).to_numpy()
    cortes24 = pd.date_range("2025-12-31 23:00", df["fecha_hora"].max() - pd.Timedelta(hours=24), freq="24h")
    idx24 = pos.reindex(cortes24).dropna().astype(int).to_numpy()

    specs = [
        ("72h", 72, idx72, [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)], 71, 5),
        ("24h", 24, idx24, [("1-24h", 1, 24)], 23, 14),
    ]
    filas_met = []
    for nombre, H, idx_test, tramos, maxlags, min_uni in specs:
        print(f"\n{'='*84}\nHORIZONTE {nombre}\n{'='*84}")
        idx_train = np.arange(200, idx_fin - H, PASO_MUESTREO_TRAIN)
        pasos = np.arange(1, H + 1)
        tr = construir_largo(df, idx_train, pasos).dropna().reset_index(drop=True)
        te = construir_largo(df, idx_test, pasos).dropna().reset_index(drop=True)
        print(f"  train {len(tr):,} filas | test {len(te):,} filas | {len(CONFIGS)} configuraciones")
        P = ajustar_configs(tr, te)

        te["real"] = te["y"].astype(float)
        te["paso_horas"] = te["paso_horas"].astype(int)
        te["unidad"] = te["cutoff"]
        if nombre == "72h":
            te["grupo"] = pd.cut(te["paso_horas"], [0, 24, 48, 72], labels=[0, 1, 2]).astype(int)
        else:
            te["grupo"] = pd.to_datetime(te["fecha_hora"]).dt.hour
        sel, avg, calent = esquemas(te, P, min_uni)

        ok = ~calent
        print(f"  filas evaluadas tras calentamiento: {ok.sum():,} de {len(te):,}")
        mae_cfg = (P[ok].sub(te.loc[ok, "real"], axis=0)).abs().mean()
        print("  MAE por configuracion fija: " + " | ".join(f"{c}={v:.1f}" for c, v in mae_cfg.items()))
        mejor_fijo = mae_cfg.idxmin()

        variantes = {
            "Ridge sin VST (id = actual)": P["id"].to_numpy(),
            "asinh estandar c=1 (literatura)": P["asinh_1.0"].to_numpy(),
            f"mejor fijo ex post [{mejor_fijo}] (NO desplegable)": P[mejor_fijo].to_numpy(),
            "SELroll (56d)": sel,
            "AVGroll (56d)": avg,
        }
        for tnom, ini, fin in tramos:
            m = ok & (te["paso_horas"].to_numpy() >= ini) & (te["paso_horas"].to_numpy() <= fin)
            real = te.loc[m, "real"].to_numpy()
            base = variantes["Ridge sin VST (id = actual)"][m]
            mae_b = np.abs(real - base).mean()
            print(f"\n  --- {nombre} tramo {tnom} ({m.sum():,} filas) ---")
            for vnom, p in variantes.items():
                pm = p[m]
                mae = np.abs(real - pm).mean()
                mape = (np.abs(real - pm) / real).mean() * 100
                if vnom.startswith("Ridge sin VST"):
                    print(f"    {vnom:46s} MAE={mae:6.2f} MAPE={mape:5.2f}%")
                    continue
                t, pv = dm(real, base, pm, maxlags)
                marca = "GANA (sig.)" if pv < 0.05 and t > 0 else ("PIERDE (sig.)" if pv < 0.05 else "sin dif.")
                print(f"    {vnom:46s} MAE={mae:6.2f} MAPE={mape:5.2f}%  {100*(mae_b-mae)/mae_b:+5.1f}%  p={pv:.4f} {marca}")
                filas_met.append({"horizonte": nombre, "tramo": tnom, "variante": vnom, "mae": mae,
                                  "mape": mape, "mae_ridge_id": mae_b, "dm_p_vs_id": pv})

        salida = te[["fecha_hora", "cutoff", "paso_horas", "real"]].copy()
        salida["pred_id"] = P["id"].to_numpy()
        salida["pred_asinh1"] = P["asinh_1.0"].to_numpy()
        salida["pred_selroll"] = sel
        salida["pred_avgroll"] = avg
        salida["calentamiento"] = calent
        salida.to_csv(RES / f"pronostico_ridge_vst_{nombre}_2026.csv", index=False)

    pd.DataFrame(filas_met).to_csv(RES / "vst_ridge_metricas.csv", index=False)
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
