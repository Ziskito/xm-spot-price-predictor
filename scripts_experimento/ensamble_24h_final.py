# -*- coding: utf-8 -*-
"""
Comparacion FINAL y limpia del ensamble de 24h, con procedimiento de ajuste identico en todas las
variantes.

Motivo: afinar_24h_final.py reajusto el LE-GARCH-t con una ruta de seleccion de alpha distinta a la
de le_garch_t_24h.py (la lista de alphas se llenaba solo con 4 valores en vez de la ruta completa),
y por eso ahi el mismo modelo daba MAPE 11.14% en vez de 10.89%. Esa diferencia es de procedimiento,
no de modelo, y contaminaba la comparacion de los contratos. Aqui se rehace todo con UN SOLO
procedimiento de ajuste, el de le_garch_t_24h.py, para que las variantes sean comparables entre si.

Variantes comparadas como votante adicional del v4:
  - GARCH-t          (la mejora ya verificada: MAPE 10.89%, sMAPE 10.09%)
  - GARCH-ged        (colas pesadas por otra via; empataba a GARCH-t)
  - GARCH-t + contratos  (precios de contratos en la ecuacion de media, equivalente colombiano de
                          los forward prices del paper de Nueva Zelanda)
  - combinaciones de las anteriores

Nota sobre `prima_sobre_contrato`: su correlacion parcial contemporanea salio 0.89, pero eso es un
artefacto -- la variable es precio/precio_contratos y por tanto CONTIENE el precio contemporaneo.
En el modelo se usa medida EN EL CORTE (donde el precio ya es conocido), asi que no hay fuga, pero
ese 0.89 no debe leerse como evidencia de utilidad.
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
from stacking_24h_v2 import cargar as cargar24, BASE5
from qra_bandas import ajustar_cuantil

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
CORTE = pd.Timestamp("2026-01-01")
H, PICO = 24, [18, 19, 20]
SEMILLAS = list(range(10))


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def smape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / ((np.abs(real) + np.abs(pred)) / 2)) * 100)


def qra_cv(d, cols, semilla=42):
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d["g4"].to_numpy(), d["dia"].to_numpy()
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
    from arch import arch_model

    df = cargar_completo().reset_index(drop=True)
    idx = pd.DatetimeIndex(df["fecha_hora"])
    precio = df["precio_bolsa"].to_numpy(np.float64)
    contratos = {}
    for n in ["precio_contratos_regulado", "precio_contratos_no_regulado"]:
        s = pd.read_csv(RAIZ / f"data/{n}_2019_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
        contratos[n] = s.iloc[:, 0].reindex(idx).to_numpy(np.float64)
    contratos["prima_sobre_contrato"] = precio / np.maximum(contratos["precio_contratos_regulado"], 1)

    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - H)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]
    pos = pd.Series(np.arange(len(df)), index=fechas)

    # UN SOLO procedimiento de ajuste para todas las variantes (el de le_garch_t_24h.py)
    CONFIG = {"GARCH-t": dict(vol="GARCH", p=1, q=1, dist="t"),
              "GARCH-ged": dict(vol="GARCH", p=1, q=1, dist="ged")}
    acc = {"GARCH-t": [], "GARCH-ged": [], "GARCH-t+contratos": []}
    alphas = []
    print("Ajustando las variantes con procedimiento identico...")
    for h in range(1, H + 1):
        Xtr, ytr, f_tr, c_tr = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_te, c_te = construir_para_paso(df, h, cortes_te)
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        if h <= 4 or h % 6 == 0:
            lc = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr)
            alphas.append(lc.alpha_)
        al = float(np.median(alphas))
        la = Lasso(alpha=al, max_iter=10000, random_state=42).fit(Ztr, ytr)
        sel = np.abs(la.coef_) > 1e-8
        if sel.sum() == 0:
            sel[:] = True
        A, B = Ztr[:, sel], Zte[:, sel]
        respaldo = la.predict(Zte)

        for et, kw in CONFIG.items():
            try:
                res = arch_model(ytr, x=A, mean="LS", rescale=False, **kw).fit(disp="off", show_warning=False)
                par = res.params
                pred = par.iloc[0] + B @ par.iloc[1:1 + A.shape[1]].to_numpy()
                if not np.isfinite(pred).all():
                    raise ValueError
            except Exception:
                pred = respaldo
            acc[et].append(pd.DataFrame({"fecha_hora": f_te, "real": yte, "pred": pred}))

        # variante con contratos, MISMO procedimiento
        ictr = pos.reindex(pd.DatetimeIndex(c_tr)).to_numpy()
        icte = pos.reindex(pd.DatetimeIndex(c_te)).to_numpy()
        etr = pd.DataFrame({f"{n}_c": v[ictr] for n, v in contratos.items()})
        ete = pd.DataFrame({f"{n}_c": v[icte] for n, v in contratos.items()})
        P2 = pd.concat([Xtr.reset_index(drop=True), etr], axis=1)
        Q2 = pd.concat([Xte.reset_index(drop=True), ete], axis=1)
        P2, Q2 = P2.fillna(P2.median()), Q2.fillna(P2.median())
        sc2 = StandardScaler().fit(P2)
        Z2, W2 = sc2.transform(P2), sc2.transform(Q2)
        la2 = Lasso(alpha=al, max_iter=10000, random_state=42).fit(Z2, ytr)
        sel2 = np.abs(la2.coef_) > 1e-8
        if sel2.sum() == 0:
            sel2[:] = True
        try:
            res = arch_model(ytr, x=Z2[:, sel2], mean="LS", vol="GARCH", p=1, q=1, dist="t",
                             rescale=False).fit(disp="off", show_warning=False)
            par = res.params
            pred = par.iloc[0] + W2[:, sel2] @ par.iloc[1:1 + sel2.sum()].to_numpy()
            if not np.isfinite(pred).all():
                raise ValueError
        except Exception:
            pred = la2.predict(W2)
        acc["GARCH-t+contratos"].append(pd.DataFrame({"fecha_hora": f_te, "real": yte, "pred": pred}))

    modelos = {}
    for et, lst in acc.items():
        r = pd.concat(lst, ignore_index=True)
        modelos[et] = r[r["fecha_hora"] >= CORTE].set_index("fecha_hora").sort_index()
        modelos[et].to_csv(RES / f"pronostico_{et.replace('+','_')}_24h_2026.csv")

    print(f"\n{'='*92}\nMODELOS SUELTOS (mismo procedimiento)\n{'='*92}")
    y0 = modelos["GARCH-t"]["real"].to_numpy()
    for et, r in modelos.items():
        p = r["pred"].to_numpy()
        print(f"  {et:22s} MAE={np.abs(y0-p).mean():7.2f}  MAPE={mape(y0,p):6.2f}%  sMAPE={smape(y0,p):6.2f}%")

    d = cargar24().sort_index()
    for et, r in modelos.items():
        d = d.join(r["pred"].rename(et), how="left")
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d.index.hour // 6
    extras = list(modelos.keys())
    d = d.dropna(subset=BASE5 + extras + ["real"])
    y = d["real"].to_numpy()
    pk = d["hora"].isin(PICO).to_numpy()
    base = qra_cv(d, BASE5)
    ref = qra_cv(d, BASE5 + ["GARCH-t"])

    conjuntos = {
        "v4 vigente": BASE5,
        "v4 + GARCH-t (verificado)": BASE5 + ["GARCH-t"],
        "v4 + GARCH-ged": BASE5 + ["GARCH-ged"],
        "v4 + GARCH-t+contratos": BASE5 + ["GARCH-t+contratos"],
        "v4 + GARCH-t + contratos (2 votantes)": BASE5 + ["GARCH-t", "GARCH-t+contratos"],
        "v4 + GARCH-t + GED": BASE5 + ["GARCH-t", "GARCH-ged"],
        "v4 + los 3": BASE5 + extras,
    }
    print(f"\n{'='*92}\nENSAMBLE (QRA por franja, CV)\n{'='*92}")
    print(f"{'conjunto':40s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s} {'pico %':>8s}   DM vs v4 | vs +GARCH-t")
    resultados = {}
    for et, cols in conjuntos.items():
        p = qra_cv(d, cols)
        resultados[et] = (cols, p)
        linea = (f"{et:40s} {np.abs(y-p).mean():8.2f} {mape(y,p):8.2f} {smape(y,p):8.2f} "
                 f"{mape(y[pk],p[pk]):8.2f}")
        if et != "v4 vigente":
            _, p1 = dm(y, base, p)
            _, p2 = dm(y, ref, p)
            linea += f"   p={p1:.4f} | p={p2:.4f}"
        print(linea)

    # verificacion multi-semilla del mejor
    mejor = min((k for k in resultados if k != "v4 vigente"),
                key=lambda k: mape(y, resultados[k][1]))
    cols_mejor = resultados[mejor][0]
    print(f"\n{'='*92}\nVERIFICACION MULTI-SEMILLA DEL MEJOR: {mejor}\n{'='*92}")
    mv = np.array([np.abs(y - qra_cv(d, BASE5, s)).mean() for s in SEMILLAS])
    mn = np.array([np.abs(y - qra_cv(d, cols_mejor, s)).mean() for s in SEMILLAS])
    delta = mn - mv
    print(f"  v4 vigente : MAE {mv.mean():.2f} +/- {mv.std():.3f}")
    print(f"  mejor      : MAE {mn.mean():.2f} +/- {mn.std():.3f}")
    print(f"  delta      : {delta.mean():+.2f} +/- {delta.std():.3f}, negativo en "
          f"{int((delta<0).sum())}/{len(SEMILLAS)} semillas")
    print(f"  VEREDICTO: {'SOLIDO' if (delta<0).all() and abs(delta.mean())>2*mv.std() else 'NO concluyente'}")
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
