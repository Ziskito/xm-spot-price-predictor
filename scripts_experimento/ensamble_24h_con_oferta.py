# -*- coding: utf-8 -*-
"""
Prueba decisiva: el LEAR24 enriquecido con variables de OFERTA, como votante del ensamble de 24h.

margen_reserva_pico_24h.py mostro que agregar margen de reserva, costo marginal y cercania al
precio de escasez (todas medidas EN EL CORTE) mejora el modelo LEAR de 24h de MAPE 16.59% a 15.05%
(DM p=0.0011). Pero el ensamble v4 esta en 11.27%: la pregunta que importa es si ese votante
enriquecido le aporta algo al ensamble, o si la informacion que trae ya estaba cubierta por los
otros votantes.

Se construye LEAR24-oferta con el mismo diseno del LEAR24 original y se prueba:
  1) como votante adicional del v4;
  2) sustituyendo al LEAR24 sin oferta;
  3) en el mejor conjunto encontrado antes (N-BEATSx + MarkovSw + N-HiTS + LEAR24 + CatBoost);
todo con QRA por franja en validacion cruzada por dias, y DM contra el v4 vigente.

Nota de alcance: se usa la SUMA TOTAL de disponibilidad, nunca el desglose por tipo de recurso
(restringido por el Alcance del Anexo 1), y NO se usa max_precio_oferta (fuga demostrada).
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
from margen_reserva_pico_24h import cargar_oferta
from stacking_24h_v2 import cargar as cargar24, BASE5
from qra_bandas import ajustar_cuantil

CORTE = pd.Timestamp("2026-01-01")
H, PICO = 24, [18, 19, 20]


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def qra_cv(d, cols, grupo, semilla=42):
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d[grupo].to_numpy(), d["dia"].to_numpy()
    P = np.full(len(d), np.nan)
    for dt in np.array_split(np.random.default_rng(semilla).permutation(np.sort(np.unique(dia))), 5):
        te_m = np.isin(dia, dt)
        for gg in np.unique(g[te_m]):
            tr, te = (~te_m) & (g == gg), te_m & (g == gg)
            if tr.sum() >= len(cols) + 5:
                w, b = ajustar_cuantil(X[tr], y[tr], 0.5)
                P[te] = X[te] @ w + b
    return P


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler

    df = cargar_completo().reset_index(drop=True)
    of = cargar_oferta()
    idx = pd.DatetimeIndex(df["fecha_hora"])
    dem = df["demanda"].to_numpy(np.float64)
    disp = of["disponibilidad_total"].reindex(idx).to_numpy(np.float64)
    cmg = of["costo_marginal"].reindex(idx).to_numpy(np.float64)
    esc = of["precio_escasez"].reindex(idx).to_numpy(np.float64)
    precio = df["precio_bolsa"].to_numpy(np.float64)
    margen = disp - dem
    derivadas = {"margen_reserva": margen, "margen_relativo": margen / np.maximum(dem, 1),
                 "costo_marginal": cmg, "cercania_escasez": precio / np.maximum(esc, 1)}

    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - H)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]
    pos = pd.Series(np.arange(len(df)), index=fechas)

    print("Construyendo LEAR24-oferta (un LASSO por hora, variables de oferta en el corte)...")
    partes, alphas = [], []
    for h in range(1, H + 1):
        Xtr, ytr, f_tr, c_tr = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_te, c_te = construir_para_paso(df, h, cortes_te)
        ictr = pos.reindex(pd.DatetimeIndex(c_tr)).to_numpy()
        icte = pos.reindex(pd.DatetimeIndex(c_te)).to_numpy()

        def bloque(ii):
            b = {}
            for nom, v in derivadas.items():
                b[f"{nom}_corte"] = v[ii]
                b[f"{nom}_corte_24h"] = v[np.maximum(ii - 24, 0)]
            return pd.DataFrame(b)

        A = pd.concat([Xtr.reset_index(drop=True), bloque(ictr)], axis=1)
        B = pd.concat([Xte.reset_index(drop=True), bloque(icte)], axis=1)
        A, B = A.fillna(A.median()), B.fillna(A.median())
        sc = StandardScaler().fit(A)
        if h <= 4:
            m = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(sc.transform(A), ytr)
            alphas.append(m.alpha_)
        else:
            m = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(
                sc.transform(A), ytr)
        partes.append(pd.DataFrame({"fecha_hora": f_te, "LEAR24_of": m.predict(sc.transform(B))}))
    lear_of = pd.concat(partes, ignore_index=True)
    lear_of = lear_of[lear_of["fecha_hora"] >= CORTE]
    lear_of.to_csv(RES / "pronostico_lear24_oferta_2026.csv", index=False)

    # ---------- ensamble ----------
    d = cargar24().sort_index()
    lear = pd.read_csv(RES / "pronostico_lear_24h_2026.csv", parse_dates=["fecha_hora"])
    d = d.join(lear.set_index("fecha_hora")["LEAR24"], how="left")
    d = d.join(lear_of.set_index("fecha_hora")["LEAR24_of"], how="left")
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d.index.hour // 6
    d = d.dropna(subset=BASE5 + ["LEAR24", "LEAR24_of", "real"])
    y = d["real"].to_numpy()
    pk = d["hora"].isin(PICO).to_numpy()
    print(f"Filas: {len(d):,}\n")
    print(f"LEAR24-oferta como modelo suelto: MAE={np.abs(y-d['LEAR24_of']).mean():.2f} "
          f"MAPE={mape(y, d['LEAR24_of']):.2f}%  (LEAR24 sin oferta: "
          f"MAE={np.abs(y-d['LEAR24']).mean():.2f} MAPE={mape(y, d['LEAR24']):.2f}%)\n")

    MEJOR5 = [c for c in ["N-BEATSx", "MarkovSw", "N-HiTS", "LEAR24", "CatBoost"] if c in d.columns]
    conjuntos = {
        "v4 vigente (5 votantes)": BASE5,
        "v4 + LEAR24_of": BASE5 + ["LEAR24_of"],
        "v4 + LEAR24 + LEAR24_of": BASE5 + ["LEAR24", "LEAR24_of"],
        "mejor5 (con LEAR24)": MEJOR5,
        "mejor5, LEAR24_of en vez de LEAR24": [c if c != "LEAR24" else "LEAR24_of" for c in MEJOR5],
        "mejor5 + LEAR24_of": MEJOR5 + ["LEAR24_of"],
    }
    base = qra_cv(d, BASE5, "g4")
    print(f"{'='*104}\nENSAMBLE DE 24h (QRA por franja, CV por dias)\n{'='*104}")
    print(f"{'conjunto':38s} {'n':>2s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s} {'pico %':>8s}   DM vs v4")
    filas = []
    for etiqueta, cols in conjuntos.items():
        p = qra_cv(d, cols, "g4")
        ok = ~np.isnan(p)
        sm_ = float(np.mean(np.abs(y[ok]-p[ok]) / ((np.abs(y[ok])+np.abs(p[ok]))/2)) * 100)
        linea = (f"{etiqueta:38s} {len(cols):2d} {np.abs(y[ok]-p[ok]).mean():8.2f} "
                 f"{mape(y[ok], p[ok]):8.2f} {sm_:8.2f} {mape(y[ok&pk], p[ok&pk]):8.2f}")
        if etiqueta != "v4 vigente (5 votantes)":
            t, pv = dm(y[ok], base[ok], p[ok])
            marca = ("MEJORA" if t > 0 else "peor") if pv < 0.05 else "n.s."
            linea += f"   p={pv:.4f} {marca}"
        print(linea)
        filas.append({"conjunto": etiqueta, "n": len(cols), "MAE": np.abs(y[ok]-p[ok]).mean(),
                      "MAPE": mape(y[ok], p[ok]), "sMAPE": sm_, "MAPE_pico": mape(y[ok&pk], p[ok&pk])})
    pd.DataFrame(filas).to_csv(RES / "ensamble_24h_con_oferta.csv", index=False)
    print(f"\nGuardado: ensamble_24h_con_oferta.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
