# -*- coding: utf-8 -*-
"""
¿QUEDA INFORMACION SIN EXPLOTAR EN EL ERROR DE LAS RAMPAS? Prueba de cierre de la linea.

POR QUE ESTA PRUEBA. Cuatro intentos distintos de bajar el error en los puntos de cambio intradia
han fallado, y lo interesante es COMO fallaron:

  modelo de rampas (predecir el cambio)      el Delta predicho captura solo el 29% de la magnitud
  calibracion de amplitud (meteorologia)     el factor optimo es 0.98, no 1.38: el aplanamiento YA
                                             es la respuesta correcta, no un defecto corregible
  FPCA de Hyndman-Ullah-Shang (forma)        empata al ensamble en rampa (93.7 vs 93.1) siendo un
                                             modelo radicalmente mas simple, y pierde en el resto
  scores funcionales con exogenas            PEOR que sin exogenas: la hidrologia no informa la
                                             forma del dia

Cuatro metodos de familias distintas chocando contra el mismo numero (~93 de MAE de forma en rampa)
sugiere un LIMITE DE INFORMACION y no un limite de modelado. Pero "sugiere" no es "demuestra". Esta
prueba lo mide directamente.

LA PRUEBA, Y DE DONDE VIENE. Es el esquema de Short-Term Bias Compensation de la literatura
financiera (Warsaw Stock Exchange, Expert Systems with Applications), que el propio paper resume
como pasar de "build a better predictor" a "correct a known predictor's drift": en vez de mejorar el
modelo, se entrena un SEGUNDO modelo para predecir el ERROR del primero, y se resta.

Si el error del ensamble en las horas de rampa fuera predecible con la informacion disponible en el
corte, este segundo modelo lo encontraria y la correccion bajaria el MAE. Si no baja, entonces no
queda informacion que extraer con los datos que el proyecto puede usar, y la linea se cierra con
evidencia en vez de con cansancio.

TRES NIVELES, de menos a mas ambicioso:
  1. ¿Es predecible el error con firma (para restarlo)?  -> R2 fuera de muestra del error del ensamble
  2. ¿Mejora el MAE al restar el error predicho?          -> la correccion STBC completa
  3. ORACULO: ¿cuanto se ganaria conociendo el SIGNO del error? -> cota superior de cualquier
     correccion de sesgo imaginable. Si hasta el oraculo del signo deja el error alto, ninguna
     correccion de sesgo puede resolver esto.

Todo con validacion temporal (TimeSeriesSplit): el corrector solo ve el pasado.
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
PICO = [0, 8, 9, 10, 18, 19]


def dm(y, p_ref, p_nuevo, maxlags=24):
    y = np.asarray(y, float)
    d = np.abs(y - p_ref) - np.abs(y - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": min(maxlags, len(d)//3)})
    return float(r.pvalues[0])


def main():
    t0 = time.time()
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    # ---------------------------------------------------- ensamble y su error
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv",
                    parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["real"])
    d["pred"] = qra_cv(d, MEJOR, "smape")
    d["error"] = d["real"] - d["pred"]

    serie = cargar_completo().set_index("fecha_hora")["precio_bolsa"]
    prev = serie.reindex(d.index - pd.Timedelta(hours=1))
    prev.index = d.index
    d["rampa"] = (d["real"] - prev).abs()

    # ---------------------------------------------------- estado en el corte, por paso
    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    cortes = idx_00[idx_00 >= 200]
    trozos = []
    for h in range(1, 25):
        X, y, f_obj, f_cor = construir_para_paso(df, h, cortes)
        X = X.copy()
        X["fecha_hora"] = pd.DatetimeIndex(f_obj)
        X["paso"] = h
        trozos.append(X)
    E = pd.concat(trozos, ignore_index=True).drop_duplicates("fecha_hora").set_index("fecha_hora")

    D = d.join(E, how="inner").dropna()
    D = D.sort_index()
    cols = [c for c in E.columns if c != "paso"] + ["paso"]
    X = D[cols].to_numpy(float)
    err = D["error"].to_numpy()
    y = D["real"].to_numpy()
    p = D["pred"].to_numpy()
    ramp = D["rampa"].to_numpy()
    hh = D.index.hour.to_numpy()
    fuerte = ramp >= 50
    pico = np.isin(hh, PICO)
    print(f"Filas con estado completo: {len(D):,}  |  de rampa fuerte: {int(fuerte.sum()):,}")
    print(f"Variables del estado en el corte: {len(cols)}\n")

    print("=" * 100)
    print("1) ¿ES PREDECIBLE EL ERROR DEL ENSAMBLE CON LA INFORMACION DEL CORTE?")
    print("=" * 100)
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"  {'subconjunto':22s} {'modelo':16s} {'R2 fuera de muestra':>20s} {'corr':>8s}")
    pred_err = {}
    for etq, m in [("todas las horas", np.ones(len(D), bool)),
                   ("rampa fuerte", fuerte), ("6 horas pico", pico)]:
        for nom, hacer in [("Ridge", lambda: RidgeCV(alphas=np.logspace(-2, 3, 20))),
                           ("GradBoost", lambda: HistGradientBoostingRegressor(
                               max_iter=300, max_depth=5, random_state=42))]:
            Xi, ei = X[m], err[m]
            pe = np.full(len(ei), np.nan)
            for tr, te in tscv.split(Xi):
                if nom == "Ridge":
                    sc = StandardScaler().fit(Xi[tr])
                    mod = hacer().fit(sc.transform(Xi[tr]), ei[tr])
                    pe[te] = mod.predict(sc.transform(Xi[te]))
                else:
                    pe[te] = hacer().fit(Xi[tr], ei[tr]).predict(Xi[te])
            ok = ~np.isnan(pe)
            r2 = 1 - np.sum((ei[ok] - pe[ok]) ** 2) / np.sum((ei[ok] - ei[ok].mean()) ** 2)
            cc = float(np.corrcoef(ei[ok], pe[ok])[0, 1])
            print(f"  {etq:22s} {nom:16s} {r2:20.4f} {cc:8.3f}")
            pred_err[(etq, nom)] = (m, pe, ok)
    print("  R2 <= 0 significa que el error del ensamble es ruido respecto de lo que se sabe en el corte.")

    print("\n" + "=" * 100)
    print("2) CORRECCION STBC: restar el error predicho, ¿baja el MAE?")
    print("=" * 100)
    print(f"  {'subconjunto':22s} {'modelo':12s} {'MAE antes':>10s} {'MAE despues':>12s} {'DM p':>8s}")
    for (etq, nom), (m, pe, ok) in pred_err.items():
        yy, pp, ee = y[m][ok], p[m][ok], pe[ok]
        corregido = pp + ee
        print(f"  {etq:22s} {nom:12s} {np.abs(yy-pp).mean():10.2f} "
              f"{np.abs(yy-corregido).mean():12.2f} {dm(yy, pp, corregido):8.4f}")

    print("\n" + "=" * 100)
    print("3) ORACULO DEL SIGNO: cota superior de CUALQUIER correccion de sesgo")
    print("=" * 100)
    print(f"  {'subconjunto':22s} {'MAE actual':>11s} {'oraculo del signo':>18s} {'oraculo total':>14s}")
    for etq, m in [("todas las horas", np.ones(len(D), bool)),
                   ("rampa fuerte", fuerte), ("6 horas pico", pico)]:
        yy, pp = y[m], p[m]
        e = yy - pp
        # oraculo del signo: se conoce el signo del error y se corrige una magnitud fija optima
        cands = np.arange(0, 121, 1.0)
        mejor = min(cands, key=lambda c: np.abs(yy - (pp + np.sign(e) * c)).mean())
        mae_signo = np.abs(yy - (pp + np.sign(e) * mejor)).mean()
        print(f"  {etq:22s} {np.abs(e).mean():11.2f} {mae_signo:18.2f} {0.0:14.2f}"
              f"   (magnitud optima {mejor:.0f})")
    print("  El oraculo del signo es inalcanzable: supone conocer si el modelo se queda corto o largo.")
    print("  Si aun ASI el error de rampa sigue alto, no hay correccion de sesgo que resuelva esto.")

    print("\n" + "=" * 100)
    print("4) ¿CUANTO DEL ERROR DE RAMPA ES SIMPLEMENTE MAGNITUD IMPREVISIBLE?")
    print("=" * 100)
    print(f"  {'grupo':26s} {'n':>6s} {'|rampa| media':>14s} {'MAE ensamble':>13s} {'razon':>8s}")
    for lo, hi, et in [(0, 0, "meseta"), (0, 10, "cambio chico"), (10, 50, "cambio medio"),
                       (50, 150, "rampa 50-150"), (150, 300, "rampa 150-300"), (300, 1e9, "rampa >300")]:
        g = (ramp == 0) if lo == hi else ((ramp > lo) & (ramp <= hi))
        if g.sum() < 10:
            continue
        print(f"  {et:26s} {int(g.sum()):6d} {ramp[g].mean():14.1f} "
              f"{np.abs(y[g]-p[g]).mean():13.2f} {np.abs(y[g]-p[g]).mean()/max(ramp[g].mean(),1):8.2f}")
    print("  Si el MAE crece proporcional a la magnitud de la rampa, el modelo no esta 'fallando' en")
    print("  las rampas: esta enfrentando un objetivo que se mueve mas, y su error relativo es estable.")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
