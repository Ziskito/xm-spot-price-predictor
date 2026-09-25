# -*- coding: utf-8 -*-
"""
Barrido sistematico de la familia GARCH para el ensamble de 24h, partiendo del exito de LE-GARCH-t.

Que se aprendio y por que este barrido. LE-GARCH-t bajo el ensamble de MAPE 11.27% a 10.89%
(sMAPE 10.38% -> 10.09%), y el experimento aislo el mecanismo: con errores NORMALES no mejora nada
(n.s.), con errores t mejora mucho. Es decir, lo que aporta es el supuesto de COLAS PESADAS al
estimar la media, que descuenta los dias volatiles. Si ese es el mecanismo, hay tres extensiones
naturales que el proyecto nunca ha probado:

  1) DISTRIBUCION ASIMETRICA (skew-t). El precio de bolsa es fuertemente asimetrico a la derecha:
     los picos son hacia arriba, no hacia abajo. Una t simetrica captura las colas pero no la
     asimetria; una skew-t captura ambas. Es un cambio de un solo parametro sobre lo que ya funciona.

  2) VOLATILIDAD ASIMETRICA (GJR-GARCH y EGARCH). En GARCH(1,1) un shock positivo y uno negativo
     del mismo tamano elevan igual la volatilidad futura. En electricidad eso es falso: una subida
     brusca anuncia mas turbulencia que una bajada equivalente. GJR y EGARCH modelan esa asimetria.
     El propio paper de Nueva Zelanda deja EGARCH como trabajo futuro explicito.

  3) LE-SV-t (volatilidad estocastica con t). Es el OTRO ganador del paper: queda 2o o 3o por MASE
     en las cinco regiones, y en el paper de comparacion que citan (Chan & Grant 2016) los modelos
     SV casi siempre le ganan a sus equivalentes GARCH. El proyecto nunca ha probado SV.
     Se implementa como aproximacion practica: la varianza latente se estima por suavizado
     exponencial del cuadrado de los residuos y se usa para re-estimar la media por minimos
     cuadrados ponderados, iterando. No es MCMC como en el paper (seria mucho mas costoso), pero
     captura lo esencial: varianza que evoluciona de forma suave y estimacion ponderada por ella.

Cada variante se evalua como modelo suelto y como VOTANTE ADICIONAL del ensamble, contra el v4
vigente y contra el v4+LE-GARCH-t ya validado.
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

CORTE = pd.Timestamp("2026-01-01")
H, PICO = 24, [18, 19, 20]

# (etiqueta, kwargs de arch_model)  -- o=1 activa el termino asimetrico (GJR)
VARIANTES = [
    ("GARCH-t",        dict(vol="GARCH",  p=1, q=1,        dist="t")),
    ("GARCH-skewt",    dict(vol="GARCH",  p=1, q=1,        dist="skewt")),
    ("GARCH-ged",      dict(vol="GARCH",  p=1, q=1,        dist="ged")),
    ("GJR-t",          dict(vol="GARCH",  p=1, o=1, q=1,   dist="t")),
    ("GJR-skewt",      dict(vol="GARCH",  p=1, o=1, q=1,   dist="skewt")),
    ("EGARCH-t",       dict(vol="EGARCH", p=1, q=1,        dist="t")),
    ("EGARCH-skewt",   dict(vol="EGARCH", p=1, q=1,        dist="skewt")),
]


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


def sv_t_aprox(A, ytr, B, iteraciones=6, lam=0.94):
    """Aproximacion practica de SV-t: minimos cuadrados ponderados por una varianza latente que
    evoluciona por suavizado exponencial de los residuos al cuadrado, iterando hasta estabilizar.
    Los pesos se acotan para que ningun dia domine (equivalente practico a las colas pesadas)."""
    X1 = np.column_stack([np.ones(len(A)), A])
    beta = np.linalg.lstsq(X1, ytr, rcond=None)[0]
    for _ in range(iteraciones):
        res2 = (ytr - X1 @ beta) ** 2
        h = np.empty_like(res2)
        h[0] = res2.mean()
        for i in range(1, len(res2)):           # varianza latente suavizada
            h[i] = lam * h[i - 1] + (1 - lam) * res2[i - 1]
        w = 1.0 / np.maximum(h, np.percentile(h, 5))
        w = np.clip(w / w.mean(), 0.1, 10.0)    # acotar para no dar peso extremo a un dia
        W = np.sqrt(w)[:, None]
        beta = np.linalg.lstsq(X1 * W, ytr * np.sqrt(w), rcond=None)[0]
    return np.column_stack([np.ones(len(B)), B]) @ beta


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    from arch import arch_model

    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - H)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]

    acc = {et: [] for et, _ in VARIANTES}
    acc["SV-t (aprox)"] = []
    alphas, fallos = [], {et: 0 for et, _ in VARIANTES}
    print(f"Ajustando {len(VARIANTES)+1} variantes x 24 horas...\n")
    for h in range(1, H + 1):
        Xtr, ytr, f_tr, _ = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_te, _ = construir_para_paso(df, h, cortes_te)
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        if h <= 4:
            lc = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr)
            alphas.append(lc.alpha_)
            sel = np.abs(lc.coef_) > 1e-8
        else:
            la = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr)
            sel = np.abs(la.coef_) > 1e-8
        if sel.sum() == 0:
            sel[:] = True
        A, B = Ztr[:, sel], Zte[:, sel]
        respaldo = Lasso(alpha=float(np.median(alphas)), max_iter=10000,
                         random_state=42).fit(Ztr, ytr).predict(Zte)

        for et, kw in VARIANTES:
            try:
                res = arch_model(ytr, x=A, mean="LS", rescale=False, **kw).fit(disp="off", show_warning=False)
                par = res.params
                pred = par.iloc[0] + B @ par.iloc[1:1 + A.shape[1]].to_numpy()
                if not np.isfinite(pred).all():
                    raise ValueError
            except Exception:
                pred, fallos[et] = respaldo, fallos[et] + 1
            acc[et].append(pd.DataFrame({"fecha_hora": f_te, "real": yte, "pred": pred}))
        acc["SV-t (aprox)"].append(pd.DataFrame({"fecha_hora": f_te, "real": yte,
                                                 "pred": sv_t_aprox(A, ytr, B)}))
        if h % 8 == 0:
            print(f"  h={h:2d}/24 ({(time.time()-t0)/60:.1f} min)", flush=True)

    modelos = {}
    for et, lst in acc.items():
        r = pd.concat(lst, ignore_index=True)
        modelos[et] = r[r["fecha_hora"] >= CORTE].set_index("fecha_hora").sort_index()

    print(f"\n{'='*96}\nMODELOS SUELTOS\n{'='*96}")
    print(f"{'variante':18s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s} {'pico %':>8s}  no convergio")
    ref = modelos["GARCH-t"]
    y0 = ref["real"].to_numpy()
    hh = np.isin(ref.index.hour, PICO)
    for et, r in modelos.items():
        p = r["pred"].to_numpy()
        print(f"{et:18s} {np.abs(y0-p).mean():8.2f} {mape(y0, p):8.2f} {smape(y0, p):8.2f} "
              f"{mape(y0[hh], p[hh]):8.2f}  {fallos.get(et, 0):>3d}/24")

    # ---------- en el ensamble ----------
    d = cargar24().sort_index()
    for et, r in modelos.items():
        d = d.join(r["pred"].rename(et), how="left")
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d.index.hour // 6
    nuevos = list(modelos.keys())
    d = d.dropna(subset=BASE5 + nuevos + ["real"])
    y = d["real"].to_numpy()
    pk = d["hora"].isin(PICO).to_numpy()
    base = qra_cv(d, BASE5)
    ref_lg = qra_cv(d, BASE5 + ["GARCH-t"])
    print(f"\n{'='*96}\nEN EL ENSAMBLE (QRA por franja, CV) -- referencia: v4 y v4+GARCH-t\n{'='*96}")
    print(f"{'conjunto':34s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s} {'pico %':>8s}   DM vs v4 | vs v4+GARCH-t")
    print(f"{'v4 vigente':34s} {np.abs(y-base).mean():8.2f} {mape(y, base):8.2f} "
          f"{smape(y, base):8.2f} {mape(y[pk], base[pk]):8.2f}")
    filas = []
    for et in nuevos:
        p = qra_cv(d, BASE5 + [et])
        _, pv1 = dm(y, base, p)
        _, pv2 = dm(y, ref_lg, p)
        print(f"{'v4 + ' + et:34s} {np.abs(y-p).mean():8.2f} {mape(y, p):8.2f} {smape(y, p):8.2f} "
              f"{mape(y[pk], p[pk]):8.2f}   p={pv1:.4f} | p={pv2:.4f}")
        filas.append({"conjunto": f"v4 + {et}", "MAE": np.abs(y-p).mean(), "MAPE": mape(y, p),
                      "sMAPE": smape(y, p), "p_vs_v4": pv1, "p_vs_v4_garcht": pv2})

    # combinaciones de varias variantes (diversidad)
    print(f"\n{'='*96}\nCOMBINACIONES DE VARIAS VARIANTES\n{'='*96}")
    combos = {
        "v4 + GARCH-t + GJR-skewt": ["GARCH-t", "GJR-skewt"],
        "v4 + GARCH-t + SV-t": ["GARCH-t", "SV-t (aprox)"],
        "v4 + GARCH-t + EGARCH-t": ["GARCH-t", "EGARCH-t"],
        "v4 + GARCH-t + GJR-skewt + SV-t": ["GARCH-t", "GJR-skewt", "SV-t (aprox)"],
        "v4 + las 8 variantes": nuevos,
    }
    for et, cc in combos.items():
        cc = [c for c in cc if c in d.columns]
        p = qra_cv(d, BASE5 + cc)
        _, pv1 = dm(y, base, p)
        _, pv2 = dm(y, ref_lg, p)
        print(f"{et:34s} {np.abs(y-p).mean():8.2f} {mape(y, p):8.2f} {smape(y, p):8.2f} "
              f"{mape(y[pk], p[pk]):8.2f}   p={pv1:.4f} | p={pv2:.4f}")
        filas.append({"conjunto": et, "MAE": np.abs(y-p).mean(), "MAPE": mape(y, p),
                      "sMAPE": smape(y, p), "p_vs_v4": pv1, "p_vs_v4_garcht": pv2})

    pd.DataFrame(filas).to_csv(RES / "familia_garch_24h.csv", index=False)
    for et, r in modelos.items():
        r.to_csv(RES / f"pronostico_{et.replace(' ','_').replace('(','').replace(')','')}_24h_2026.csv")
    print(f"\nGuardado: familia_garch_24h.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
