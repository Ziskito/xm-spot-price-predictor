# -*- coding: utf-8 -*-
"""
CRITERIO DE MAXIMA CORRENTROPIA (MCC) como funcion de perdida robusta -- el unico pendiente
genuinamente nunca probado de las ideas de los papers (Nunes Jr., Ferreira & Pinho 2024).

LA IDEA. La correntropia mide la similitud entre prediccion y real con un kernel gaussiano sobre
el error: maximizar  sum exp(-e_i^2 / (2 sigma^2)).  A diferencia del error cuadratico (que
castiga los errores grandes cuadraticamente) y del absoluto (que los castiga linealmente), MCC es
REDESCENDENTE: un error muy grande casi no pesa. Nunes Jr. et al. la programaron como perdida
propia en Keras para caudales y le gano a MSE en la mayoria de embalses; el precio de bolsa es un
objetivo igual de no gaussiano, con colas pesadas.

COMO SE OPTIMIZA AQUI. Por la equivalencia de la optimizacion "half-quadratic" (Yang et al.,
He et al. -- el metodo estandar para MCC): maximizar la correntropia equivale a resolver minimos
cuadrados reponderados iterativamente con pesos  w_i = exp(-e_i^2 / (2 sigma^2)),  recalculados
con los residuos de la iteracion anterior. Asi se usan perdidas nativas y rapidas (con
sample_weight) en vez de una perdida propia lenta. Se parte de un ajuste con perdida absoluta
(robusto) y se hacen 4 iteraciones.

SIGMA (ancho del kernel). Decide que cuenta como "atipico". Se elige en validacion temporal
DENTRO del entrenamiento (ultimo 20% de los cortes previos a 2026), entre multiplos de la
escala robusta de los residuos (MAD/0.6745), nunca mirando 2026.

ADVERTENCIA A PRIORI, LIGADA AL OBJETIVO DEL USUARIO. MCC le quita peso justamente a los errores
grandes -- y los errores grandes del proyecto viven en las rampas. Es plausible que MCC mejore
las horas tranquilas y EMPEORE las rampas. Por eso se reporta el efecto por tipo de hora.

Protocolo de 24h vigente (corte 00:00), mismo diseño que el LEAR, entrenamiento < 2026.
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
from combinador_optimo_mape import qra_cv, mape, smape

CORTE = pd.Timestamp("2026-01-01")
SEMILLAS = list(range(10))
PICO = [0, 8, 9, 10, 18, 19]
MULT_SIGMA = [0.5, 1.0, 2.0, 4.0]
ITER = 4


def dm(y, a, b, maxlags=24):
    y = np.asarray(y, float)
    d = np.abs(y - a) - np.abs(y - b)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": min(maxlags, len(d)//3)})
    return float(r.pvalues[0])


def escala(e):
    return max(float(np.median(np.abs(e - np.median(e)))) / 0.6745, 1e-6)


# ------------------------------------------------------------------ ajustadores
def gbm(loss, Xtr, ytr, w=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss=loss, max_iter=300, max_depth=6, learning_rate=0.05,
                                         random_state=42).fit(Xtr, ytr, sample_weight=w)


def lasso(Xtr, ytr, w=None, alpha=1.0):
    from sklearn.linear_model import Lasso
    return Lasso(alpha=alpha, max_iter=10000, random_state=42).fit(Xtr, ytr, sample_weight=w)


def mcc_gbm(Xtr, ytr, mult):
    m = gbm("absolute_error", Xtr, ytr)
    for _ in range(ITER):
        e = ytr - m.predict(Xtr)
        s = mult * escala(e)
        w = np.exp(-e ** 2 / (2 * s ** 2)) + 1e-6
        m = gbm("squared_error", Xtr, ytr, w)
    return m


def mcc_lasso(Xtr, ytr, mult, alpha):
    m = lasso(Xtr, ytr, None, alpha)
    for _ in range(ITER):
        e = ytr - m.predict(Xtr)
        s = mult * escala(e)
        w = np.exp(-e ** 2 / (2 * s ** 2)) + 1e-6
        m = lasso(Xtr, ytr, w * len(w) / w.sum(), alpha)
    return m


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    df = cargar_completo().reset_index(drop=True)
    f = df["fecha_hora"]
    idx00 = np.where(f.dt.hour.to_numpy() == 0)[0]
    fin = int(pd.Series(df.index.values, index=f)[CORTE])
    tr_c = idx00[(idx00 >= 200) & (idx00 < fin - 24)]
    te_c = idx00[idx00 >= fin - 1]
    corte_val = tr_c[int(len(tr_c) * 0.8)]            # validacion temporal para sigma

    # ---------- 1) elegir sigma en validacion (pasos representativos)
    print("Eligiendo sigma en validacion temporal (ultimo 20% del entrenamiento)...", flush=True)
    votos_g, votos_l, alphas = [], [], []
    for h in [1, 6, 9, 12, 19, 24]:
        X, y, _, fc = construir_para_paso(df, h, tr_c)
        fcut = pd.DatetimeIndex(fc)
        v = fcut >= f.iloc[corte_val]
        sc = StandardScaler().fit(X[~v])
        Zt, Zv = sc.transform(X[~v]), sc.transform(X[v])
        a = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Zt, y[~v]).alpha_
        alphas.append(a)
        eg = [np.abs(y[v] - mcc_gbm(X[~v], y[~v], m).predict(X[v])).mean() for m in MULT_SIGMA]
        el = [np.abs(y[v] - mcc_lasso(Zt, y[~v], m, a).predict(Zv)).mean() for m in MULT_SIGMA]
        votos_g.append(MULT_SIGMA[int(np.argmin(eg))])
        votos_l.append(MULT_SIGMA[int(np.argmin(el))])
    mg, ml, alpha = float(np.median(votos_g)), float(np.median(votos_l)), float(np.median(alphas))
    print(f"  sigma elegido: GBM {mg} x escala, LASSO {ml} x escala (votos {votos_g} / {votos_l})\n")

    # ---------- 2) ajuste final por paso, 2026
    print("Ajustando 24 pasos x 6 variantes...", flush=True)
    filas = []
    for h in range(1, 25):
        Xtr, ytr, _, _ = construir_para_paso(df, h, tr_c)
        Xte, yte, fo, _ = construir_para_paso(df, h, te_c)
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        fila = pd.DataFrame({"fecha_hora": fo, "real": yte})
        fila["GBM cuadratica"] = gbm("squared_error", Xtr, ytr).predict(Xte)
        fila["GBM absoluta (MAE)"] = gbm("absolute_error", Xtr, ytr).predict(Xte)
        fila["GBM MCC"] = mcc_gbm(Xtr, ytr, mg).predict(Xte)
        fila["LEAR (LASSO)"] = lasso(Ztr, ytr, None, alpha).predict(Zte)
        fila["LEAR MCC"] = mcc_lasso(Ztr, ytr, ml, alpha).predict(Zte)
        filas.append(fila)
        if h % 6 == 0:
            print(f"  paso {h}/24 ({(time.time()-t0)/60:.1f} min)", flush=True)
    R = pd.concat(filas).drop_duplicates("fecha_hora").set_index("fecha_hora").sort_index()
    R = R[R.index >= CORTE]
    R.to_csv(RES / "mcc_perdida_robusta_24h.csv")

    serie = df.set_index("fecha_hora")["precio_bolsa"]
    prev = serie.reindex(R.index - pd.Timedelta(hours=1)).to_numpy()
    y = R["real"].to_numpy()
    rampa = np.abs(y - prev)
    fu, me, pk = rampa >= 50, rampa == 0, np.isin(R.index.hour, PICO)

    print("\n" + "=" * 104)
    print("MODELOS SUELTOS (2026) -- con el efecto en las RAMPAS, que es el objetivo")
    print("=" * 104)
    print(f"  {'variante':20s} {'MAE':>7s} {'MAPE':>7s} {'sMAPE':>7s} {'rampa>=50':>10s} {'meseta':>8s} {'pico':>7s}")
    for c in ["GBM cuadratica", "GBM absoluta (MAE)", "GBM MCC", "LEAR (LASSO)", "LEAR MCC"]:
        p = R[c].to_numpy()
        print(f"  {c:20s} {np.abs(y-p).mean():7.2f} {mape(y,p):7.2f} {smape(y,p):7.2f} "
              f"{np.abs(y-p)[fu].mean():10.2f} {np.abs(y-p)[me].mean():8.2f} {np.abs(y-p)[pk].mean():7.2f}")
    print(f"\n  DM  GBM MCC vs GBM absoluta: p={dm(y, R['GBM absoluta (MAE)'], R['GBM MCC']):.4f}")
    print(f"  DM  GBM MCC vs GBM cuadratica: p={dm(y, R['GBM cuadratica'], R['GBM MCC']):.4f}")
    print(f"  DM  LEAR MCC vs LEAR: p={dm(y, R['LEAR (LASSO)'], R['LEAR MCC']):.4f}")
    for base, nuevo in [("GBM absoluta (MAE)", "GBM MCC"), ("LEAR (LASSO)", "LEAR MCC")]:
        print(f"  en rampa fuerte, {nuevo} vs {base}: DM p={dm(y[fu], R[base].to_numpy()[fu], R[nuevo].to_numpy()[fu], 12):.4f}")

    # ---------- 3) como votante del ensamble
    print("\n" + "=" * 104)
    print("COMO SEPTIMO VOTANTE DEL ENSAMBLE (10 particiones)")
    print("=" * 104)
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    d = d.join(R[["GBM MCC", "LEAR MCC", "GBM absoluta (MAE)"]], how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["GBM MCC", "LEAR MCC", "GBM absoluta (MAE)", "real"])
    yy = d["real"].to_numpy()
    pv = serie.reindex(d.index - pd.Timedelta(hours=1)).to_numpy()
    fu2 = np.abs(yy - pv) >= 50
    A = np.array([[np.abs(yy-p).mean(), mape(yy,p), smape(yy,p), np.abs(yy-p)[fu2].mean()]
                  for p in (qra_cv(d, MEJOR, "smape", s_) for s_ in SEMILLAS)])
    print(f"  {'ensamble actual':26s} MAE {A[:,0].mean():7.3f}  MAPE {A[:,1].mean():6.3f}  "
          f"sMAPE {A[:,2].mean():6.3f}  rampa {A[:,3].mean():7.2f}")
    for c in ["GBM MCC", "LEAR MCC", "GBM absoluta (MAE)"]:
        B = np.array([[np.abs(yy-p).mean(), mape(yy,p), smape(yy,p), np.abs(yy-p)[fu2].mean()]
                      for p in (qra_cv(d, MEJOR + [c], "smape", s_) for s_ in SEMILLAS)])
        dl = B - A
        print(f"  {'+ ' + c:26s} MAE {B[:,0].mean():7.3f}  MAPE {B[:,1].mean():6.3f}  "
              f"sMAPE {B[:,2].mean():6.3f}  rampa {B[:,3].mean():7.2f}   "
              f"mejor en {int((dl[:,0]<0).sum())}/{int((dl[:,1]<0).sum())}/{int((dl[:,2]<0).sum())} (MAE/MAPE/sMAPE)",
              flush=True)
    print("  ('GBM absoluta' se incluye como control: si MCC ayuda solo por ser un GBM mas, el control")
    print("   deberia ayudar igual; si ayuda por la perdida, el control no.)")
    print(f"\nGuardado: mcc_perdida_robusta_24h.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
