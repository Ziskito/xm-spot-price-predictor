# -*- coding: utf-8 -*-
"""
CALIBRACION DE AMPLITUD DE LA CURVA INTRADIA. Atacar directamente el aplanamiento en las rampas.

EL DIAGNOSTICO QUE LO MOTIVA (`diagnostico_forma_intradia.py`, sobre el ensamble actual en 2026):

    el 98% del error esta en la FORMA del dia, no en el nivel
    desviacion tipica intradia:  real 90.22  vs  pronosticada 65.18   ->  razon 0.723
    rango intradia (max-min):    real 327.8  vs  pronosticado 230.1   ->  razon 0.702
    en el 70% de los dias la curva pronosticada es MAS PLANA que la real
    en las 6 horas peores la razon baja a 0.690
    NO hay desfase temporal (el mejor desplazamiento es 0 h)

O sea: el modelo acierta la forma pero la dibuja comprimida. Se cubre quedandose cerca del promedio
del dia, que es lo optimo en error cuadratico cuando hay incertidumbre, pero deja corto justo en los
picos y valles -- que es donde viven las rampas y el 35% del error.

DE DONDE SE TRAE LA SOLUCION (literatura fuera de mercados electricos, como pidio el usuario).
En meteorologia esto es un problema clasico y tiene nombre: pronosticos SUBDISPERSOS. Se corrige con
inflacion de varianza y con EMOS / regresion gaussiana no homogenea (Gneiting et al.), que recalibra
a la vez el sesgo y la dispersion del pronostico. La adaptacion aqui es directa: en vez de recalibrar
la dispersion de un ensamble probabilistico, se recalibra la AMPLITUD de la forma intradia:

    p_calibrado(d,h) = nivel_d + k * (p(d,h) - nivel_d)          nivel_d = media del dia pronosticada

con k > 1 expandiendo la curva. Si k=1 se recupera el pronostico actual, asi que la correccion es
reversible por construccion y no puede "romper" el modelo: en el peor caso el ajuste devuelve k=1.

CUIDADO, Y ES IMPORTANTE. Igualar la varianza (k = 0.723^-1 = 1.38) seria un ERROR: cuando la forma
pronosticada esta correlacionada imperfectamente con la real (aqui r=0.78), el pronostico optimo en
error cuadratico ESTA encogido a proposito -- es regresion a la media, no un defecto. Expandir hasta
igualar varianzas empeora el error. La pendiente de la regresion forma_real ~ forma_pronosticada es
1.095, no 1.38: ese es el margen real. Por eso k se ajusta MINIMIZANDO LA METRICA, no igualando
varianzas. Se incluye la version "igualar varianza" justamente para mostrar que pierde.

VARIANTES
  1. k global, uno solo para todo.
  2. k por hora del dia (24 factores): las rampas no estan repartidas parejo.
  3. k por hora, encogido hacia 1 (regularizado), por si 24 factores sobreajustan.
  4. k = razon de desviaciones tipicas (la version "igualar varianza", como control negativo).

TODO CAUSAL. Para el dia d, k se estima solo con dias anteriores a d (ventana movil y expansiva).
Nunca se usa el propio dia ni dias futuros para fijar su correccion.
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
from stacking_24h_v2 import cargar as cargar24, BASE5
from combinador_optimo_mape import qra_cv, mape, smape

SEMILLAS = list(range(10))
PICO = [0, 8, 9, 10, 18, 19]
ORDEN = list(range(1, 24)) + [0]        # la ventana de pronostico: 01:00 -> 00:00 del dia siguiente
REJILLA = np.round(np.arange(0.80, 1.61, 0.01), 2)
MIN_DIAS = 30                            # dias previos minimos para estimar k


def dm(y, p_ref, p_nuevo, metrica="mae", maxlags=24):
    y = np.asarray(y, float)
    if metrica == "mape":
        l1, l2 = np.abs(y - p_ref) / y, np.abs(y - p_nuevo) / y
    elif metrica == "smape":
        l1 = np.abs(y - p_ref) / ((np.abs(y) + np.abs(p_ref)) / 2)
        l2 = np.abs(y - p_nuevo) / ((np.abs(y) + np.abs(p_nuevo)) / 2)
    else:
        l1, l2 = np.abs(y - p_ref), np.abs(y - p_nuevo)
    d = l1 - l2
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.pvalues[0])


def perdida(y, p, metrica):
    if metrica == "mape":
        return np.abs(y - p) / np.maximum(y, 1e-6)
    if metrica == "smape":
        return np.abs(y - p) / np.maximum((np.abs(y) + np.abs(p)) / 2, 1e-6)
    return np.abs(y - p)


def calibrar(Y, P, modo, metrica="smape", ventana=None, encoge=0.0):
    """
    Devuelve la matriz calibrada. Para cada dia t, k se estima con los dias [t0, t) solamente.

    modo: 'global' | 'por_hora' | 'varianza'
    ventana: numero de dias previos a usar (None = expansiva)
    encoge: peso hacia k=1 (0 = sin encoger, 1 = no corregir nada)
    """
    n, H = Y.shape
    niv_p = P.mean(1, keepdims=True)
    Fp = P - niv_p
    Fy = Y - Y.mean(1, keepdims=True)
    out = P.copy()
    ks = np.full((n, H), np.nan)
    for t in range(n):
        ini = 0 if ventana is None else max(0, t - ventana)
        if t - ini < MIN_DIAS:
            continue
        ytr, ptr, ftr = Y[ini:t], P[ini:t], Fp[ini:t]
        ntr = ptr.mean(1, keepdims=True)
        if modo == "varianza":
            k = np.full(H, Fy[ini:t].std() / max(ftr.std(), 1e-9))
        elif modo == "global":
            perd = [perdida(ytr, ntr + kk * ftr, metrica).mean() for kk in REJILLA]
            k = np.full(H, REJILLA[int(np.argmin(perd))])
        else:                                   # por hora
            k = np.ones(H)
            for j in range(H):
                perd = [perdida(ytr[:, j], ntr[:, 0] + kk * ftr[:, j], metrica).mean()
                        for kk in REJILLA]
                k[j] = REJILLA[int(np.argmin(perd))]
        k = (1 - encoge) * k + encoge * 1.0
        ks[t] = k
        out[t] = niv_p[t] + k * Fp[t]
    return out, ks


def evaluar(nombre, Y, P, Pc, rampa, horas, fila_pico):
    y, p, pc = Y.ravel(), P.ravel(), Pc.ravel()
    ok = ~np.isnan(pc)
    y, p, pc = y[ok], p[ok], pc[ok]
    r, hh = rampa.ravel()[ok], horas.ravel()[ok]
    fuerte = r >= 50
    pico = np.isin(hh, PICO)
    print(f"  {nombre:34s} {np.abs(y-pc).mean():7.2f} {mape(y,pc):7.2f} {smape(y,pc):7.2f} "
          f"{np.abs(y[fuerte]-pc[fuerte]).mean():9.2f} {np.abs(y[pico]-pc[pico]).mean():8.2f} "
          f"{dm(y,p,pc,'mae'):8.4f} {dm(y,p,pc,'smape'):8.4f}")
    return dict(variante=nombre, MAE=np.abs(y - pc).mean(), MAPE=mape(y, pc), sMAPE=smape(y, pc),
                MAE_rampa=np.abs(y[fuerte] - pc[fuerte]).mean(),
                MAE_pico=np.abs(y[pico] - pc[pico]).mean(),
                p_MAE=dm(y, p, pc, "mae"), p_sMAPE=dm(y, p, pc, "smape"))


def main():
    t0 = time.time()
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["real"])
    d["hora"] = d.index.hour
    d["ventana"] = (d.index - pd.Timedelta(hours=1)).normalize()
    comp = d.groupby("ventana")["real"].count()
    d = d[d["ventana"].isin(comp[comp == 24].index)].copy()

    serie = cargar_completo().set_index("fecha_hora")["precio_bolsa"]
    prev = serie.reindex(d.index - pd.Timedelta(hours=1)); prev.index = d.index
    d["rampa"] = (d["real"] - prev).abs()

    d["pred"] = qra_cv(d, MEJOR, "smape")
    piv = lambda c: d.pivot_table(index="ventana", columns="hora", values=c)[ORDEN].to_numpy()
    Y, P, R = piv("real"), piv("pred"), piv("rampa")
    HH = np.tile(np.array(ORDEN), (len(Y), 1))
    print(f"Ventanas completas: {len(Y)}  |  horas: {Y.size:,}\n")

    print("=" * 110)
    print("CALIBRACION DE AMPLITUD  (k ajustado solo con dias anteriores; k=1 = sin corregir)")
    print("=" * 110)
    print(f"  {'variante':34s} {'MAE':>7s} {'MAPE':>7s} {'sMAPE':>7s} {'MAE rampa':>9s} "
          f"{'MAE pico':>8s} {'p(MAE)':>8s} {'p(sMAPE)':>8s}")
    base = evaluar("sin calibrar (ensamble actual)", Y, P, P, R, HH, None)

    filas = [base]
    configs = [("k global, expansiva", "global", None, 0.0),
               ("k global, ventana 90 dias", "global", 90, 0.0),
               ("k por hora, expansiva", "por_hora", None, 0.0),
               ("k por hora, ventana 90 dias", "por_hora", 90, 0.0),
               ("k por hora, encogido 50% a 1", "por_hora", None, 0.5),
               ("k = igualar varianza (control)", "varianza", None, 0.0)]
    guardado = {}
    for nom, modo, ven, enc in configs:
        Pc, ks = calibrar(Y, P, modo, "smape", ven, enc)
        guardado[nom] = (Pc, ks)
        filas.append(evaluar(nom, Y, P, Pc, R, HH, None))

    print("\n" + "=" * 110)
    print("¿QUE VALOR DE k ELIGE EL AJUSTE?  (media sobre los dias en que ya hay datos suficientes)")
    print("=" * 110)
    for nom in ["k global, expansiva", "k por hora, expansiva", "k = igualar varianza (control)"]:
        ks = guardado[nom][1]
        v = ks[~np.isnan(ks).any(1)]
        if len(v) == 0:
            continue
        print(f"  {nom:34s} media {v.mean():.3f}   min {v.min():.2f}   max {v.max():.2f}")
    ks = guardado["k por hora, expansiva"][1]
    v = ks[~np.isnan(ks).any(1)]
    if len(v):
        print(f"\n  k final por hora (ultimo dia), en orden de la ventana 01:00 -> 00:00:")
        ult = ks[~np.isnan(ks).any(1)][-1]
        for i in range(0, 24, 8):
            print("    " + "  ".join(f"h{ORDEN[j]:02d}={ult[j]:.2f}" for j in range(i, min(i + 8, 24))))

    T = pd.DataFrame(filas)
    T.to_csv(RES / "calibracion_amplitud_intradia.csv", index=False)

    # ------------------------------------------------- ¿se sostiene en las 10 particiones?
    mejor = T.iloc[1:].nsmallest(1, "sMAPE")["variante"].iloc[0]
    cfg = dict((n, (m, v, e)) for n, m, v, e in configs)[mejor]
    print("\n" + "=" * 110)
    print(f"VERIFICACION 10 PARTICIONES DEL COMBINADOR: '{mejor}'")
    print("=" * 110)
    A, B = [], []
    for s_ in SEMILLAS:
        d["pred"] = qra_cv(d, MEJOR, "smape", s_)
        Ps = d.pivot_table(index="ventana", columns="hora", values="pred")[ORDEN].to_numpy()
        Pc, _ = calibrar(Y, Ps, cfg[0], "smape", cfg[1], cfg[2])
        m = ~np.isnan(Pc).any(1)
        y = Y[m].ravel()
        A.append([np.abs(y - Ps[m].ravel()).mean(), mape(y, Ps[m].ravel()), smape(y, Ps[m].ravel())])
        B.append([np.abs(y - Pc[m].ravel()).mean(), mape(y, Pc[m].ravel()), smape(y, Pc[m].ravel())])
    A, B = np.array(A), np.array(B)
    for j, nm in enumerate(["MAE", "MAPE %", "sMAPE %"]):
        dl = B[:, j] - A[:, j]
        print(f"  {nm:8s} sin calibrar={A[:,j].mean():7.3f}+/-{A[:,j].std():.3f}  "
              f"calibrado={B[:,j].mean():7.3f}+/-{B[:,j].std():.3f}  "
              f"delta={dl.mean():+7.3f}, mejor en {int((dl<0).sum())}/10")
    print(f"  VEREDICTO: {'SOLIDO' if all((B[:,j]-A[:,j]<0).all() for j in range(3)) else 'no concluyente'}")

    print(f"\nGuardado: calibracion_amplitud_intradia.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
