# -*- coding: utf-8 -*-
"""
¿LA GANANCIA DE LA SELECCION POR PUNTOS DE CAMBIO SE SOSTIENE EN TODO EL WALK-FORWARD?

EL PROBLEMA. La ganancia de `ventana_calibracion_not_24h.py` esta medida sobre 2026 (Origen 6), el
mismo periodo sobre el que se ha evaluado todo lo demas del proyecto. Un metodo que elige que
historia usar es, por construccion, sensible al regimen: puede funcionar en un anio y no en otro.
Medirlo en un solo anio no distingue "el metodo sirve" de "2026 le vino bien".

LA PRUEBA. Se repite la comparacion en los 6 origenes del walk-forward, que cubren regimenes ENSO
distintos (La Nina, El Nino 2023-24, la transicion de 2026):

    Origen 1  2020-07-01 .. 2021-06-30      Origen 4  2023-07-01 .. 2024-06-30
    Origen 2  2021-07-01 .. 2022-06-30      Origen 5  2024-07-01 .. 2025-06-30
    Origen 3  2022-07-01 .. 2023-06-30      Origen 6  2026-01-01 .. 2026-08-05

Tres comparaciones, cada una con el MISMO modelo y la MISMA informacion, cambiando solo la regla de
que historia se usa para calibrar:

    ARX  NOT(C0)   contra  ARX  Win(C0)      <- la comparacion cruda del paper
    ARX  NOTH(C0)  contra  ARX  WinH(C0)     <- la del paper, sobre datos asinh
    LEAR NOT(C0)   contra  LEAR Win(todo)    <- la que le importa al proyecto: contra entrenar
                                                con el historico completo, que es lo que hace hoy

VENTANA INICIAL C0. El paper usa 728 dias. El Origen 1 empieza en 2020-07 y el historico arranca en
2019, asi que no hay 728 dias disponibles; para ese origen C0 se recorta a lo que haya. Se reporta
el C0 efectivo de cada origen.

REFRESCO SEMANAL DE LAS MASCARAS. El paper rehace la deteccion cada dia. Aqui se rehace cada 7 dias,
lo que baja el coste de ~12 h a ~2 h. NO es una suposicion: se midio sobre el Origen 6, donde estan
las mascaras diarias, y la diferencia es indistinguible --

    NOTH diario           MAE 50.525   MAPE 13.161%
    NOTH refresco 7 dias  MAE 50.914   MAPE 13.523%   DM contra el diario: p=0.4848
    NOTH refresco 14 dias MAE 51.927   MAPE 13.877%   DM contra el diario: p=0.0808

o sea que a 7 dias no se pierde nada medible y a 14 ya empieza a notarse. Coherente con lo que
observan los autores: los puntos de cambio se detectan en los mismos dias o vecinos para ventanas
contiguas.

ADVERTENCIA DE LECTURA. Aqui el modelo se reajusta cada dia (esquema rodante), mientras que el
walk-forward oficial del proyecto entrena una vez por origen. Por eso los MAE de este script NO son
comparables con los de `walkforward_5origenes.csv`. La comparacion valida es la interna: regla de
ventana contra regla de ventana, dentro de cada fila.

Se guarda el resultado tras CADA origen, para que una interrupcion no pierda lo hecho.
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
from o6_comun import RES
import not_changepoint as notcp
from ventana_calibracion_not_24h import (armar_disenos, asinh_par, met, pronosticar,
                                         indices_calibracion, calibrar_alphas, TAU_MAX)

REFRESCO = 7
SALIDA = RES / "not_walkforward_origenes.csv"
ORIGENES = [("Origen 1", "2020-07-01", "2021-06-30"), ("Origen 2", "2021-07-01", "2022-06-30"),
            ("Origen 3", "2022-07-01", "2023-06-30"), ("Origen 4", "2023-07-01", "2024-06-30"),
            ("Origen 5", "2024-07-01", "2025-06-30"), ("Origen 6", "2026-01-01", "2026-08-05")]


def dm(y, p_ref, p_nuevo, maxlags=24):
    y = np.asarray(y, float)
    d = np.abs(y - p_ref) - np.abs(y - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.pvalues[0])


def recortar(dis, ini, fin):
    """Copia de los disenos con i0/i1 puestos en el periodo de prueba de un origen."""
    sub = {}
    for h, d in dis.items():
        fc = d["f_cor"]
        i0 = int(np.searchsorted(fc.to_numpy(), np.datetime64(pd.Timestamp(ini))))
        i1 = int(np.searchsorted(fc.to_numpy(), np.datetime64(pd.Timestamp(fin)), side="right"))
        sub[h] = dict(d, i0=i0, i1=min(i1, len(d["y"])))
    return sub


def mascaras_origen(sub, c0):
    """Mascaras NOT del origen, recalculadas cada REFRESCO dias (validado contra el diario)."""
    n = max(d["i1"] - d["i0"] for d in sub.values())
    cruda = np.ones((n, 24, c0), dtype=bool)
    asnh = np.ones((n, 24, c0), dtype=bool)
    for hi, h in enumerate(range(1, 25)):
        y, i0, i1 = sub[h]["y"], sub[h]["i0"], sub[h]["i1"]
        ultimo_c = ultimo_a = None
        for k in range(i1 - i0):
            if k % REFRESCO == 0:
                v = y[i0 + k - c0:i0 + k]
                ultimo_c = notcp.seleccionar_subperiodos(v, semilla=0)
                a, b = asinh_par(v)
                ultimo_a = notcp.seleccionar_subperiodos(np.arcsinh((v - a) / b), semilla=0)
            cruda[k, hi], asnh[k, hi] = ultimo_c, ultimo_a
    return cruda, asnh


def pronosticar_rango(sub, mc, ma, familia, transf, ventana, c0, alphas=None):
    """Igual que `pronosticar` pero acotado al rango [i0, i1) del origen y con ventana C0 propia."""
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from ventana_calibracion_not_24h import ARX_COLS, PRECIO_COLS_PREFIJO, ols

    preds, reales, fechas = [], [], []
    for hi, h in enumerate(range(1, 25)):
        d = sub[h]
        cols = ARX_COLS + [f"dow{k}" for k in range(7)] if familia == "ARX" else list(d["X"].columns)
        Xf = d["X"][cols].to_numpy(float)
        y, i0, i1 = d["y"], d["i0"], d["i1"]
        es_precio = np.array([c.startswith(PRECIO_COLS_PREFIJO) for c in cols])
        for k in range(i1 - i0):
            i = i0 + k
            if ventana == "NOT":
                fila = (ma if transf == "asinh" else mc)[k, hi]
                idx = np.arange(i - c0, i)[fila]
                if len(idx) < 60:
                    idx = np.arange(i - 60, i)
            else:
                idx = indices_calibracion(i, ventana, None)
            Xtr, ytr, Xte = Xf[idx], y[idx], Xf[i:i + 1]
            if transf == "asinh":
                a, b = asinh_par(ytr)
                ytr = np.arcsinh((ytr - a) / b)
                Xtr, Xte = Xtr.copy(), Xte.copy()
                Xtr[:, es_precio] = np.arcsinh((Xtr[:, es_precio] - a) / b)
                Xte[:, es_precio] = np.arcsinh((Xte[:, es_precio] - a) / b)
            if familia == "ARX":
                p = float((Xte @ ols(Xtr, ytr))[0])
            else:
                sc = StandardScaler().fit(Xtr)
                p = float(Lasso(alpha=alphas[h], max_iter=5000, random_state=42)
                          .fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))[0])
            if transf == "asinh":
                p = a + b * np.sinh(np.clip(p, -30, 30))
            preds.append(p); reales.append(y[i]); fechas.append(d["f_obj"][i])
    return pd.DataFrame({"fecha_hora": fechas, "real": reales, "pred": preds})


def alphas_origen(sub, ventana, transf, c0):
    """alpha del LASSO calibrado en el ultimo corte ANTERIOR al periodo de prueba del origen."""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    vals = []
    for h in [1, 2, 3, 4, 6, 12, 18, 24]:
        d = sub[h]
        i = d["i0"]
        if ventana == "NOT":
            v = d["y"][i - c0:i]
            if transf == "asinh":
                a, b = asinh_par(v)
                v = np.arcsinh((v - a) / b)
            fila = notcp.seleccionar_subperiodos(v, semilla=0)
            idx = np.arange(i - c0, i)[fila]
            if len(idx) < 60:
                idx = np.arange(i - 60, i)
        else:
            idx = indices_calibracion(i, ventana, None)
        Xtr, ytr = d["X"].to_numpy(float)[idx], d["y"][idx]
        if transf == "asinh":
            a, b = asinh_par(ytr)
            ytr = np.arcsinh((ytr - a) / b)
            es_precio = np.array([c.startswith(("p_obj_lag", "p_corte_lag", "p_min24", "p_max24",
                                                "p_media24")) for c in d["X"].columns])
            Xtr = Xtr.copy()
            Xtr[:, es_precio] = np.arcsinh((Xtr[:, es_precio] - a) / b)
        sc = StandardScaler().fit(Xtr)
        vals.append(LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=8000)
                    .fit(sc.transform(Xtr), ytr).alpha_)
    return {h: float(np.median(vals)) for h in range(1, 25)}


def main():
    t0 = time.time()
    if not notcp.autoprueba():
        print("El detector no paso la autoprueba. Se aborta.")
        return

    print("Armando disenos...", flush=True)
    dis = armar_disenos()
    hechos = set()
    if SALIDA.exists():
        previo = pd.read_csv(SALIDA)
        hechos = set(previo["origen"])
        print(f"  ya calculados: {sorted(hechos)}")
    filas = [] if not hechos else pd.read_csv(SALIDA).to_dict("records")

    for nom, ini, fin in ORIGENES:
        if nom in hechos:
            continue
        sub = recortar(dis, ini, fin)
        n_test = sub[1]["i1"] - sub[1]["i0"]
        c0 = int(min(TAU_MAX, sub[1]["i0"]))
        if n_test < 30 or c0 < 120:
            print(f"{nom}: historia o prueba insuficiente (C0={c0}, n={n_test}); se omite")
            continue
        print(f"\n{'='*104}\n{nom}  {ini} .. {fin}   dias de prueba {n_test}   C0 efectivo {c0}"
              f"{'  (recortado: no hay 728 dias de historia)' if c0 < TAU_MAX else ''}\n{'='*104}", flush=True)

        t1 = time.time()
        mc, ma = mascaras_origen(sub, c0)
        print(f"  mascaras listas en {(time.time()-t1)/60:.1f} min  "
              f"(selecciona {100*mc.mean():.0f}% cruda / {100*ma.mean():.0f}% asinh)", flush=True)

        preds = {}
        for etq, fam, tr, ven in [(f"ARX Win({c0})", "ARX", "ninguna", c0),
                                  (f"ARX NOT({c0})", "ARX", "ninguna", "NOT"),
                                  (f"ARX WinH({c0})", "ARX", "asinh", c0),
                                  (f"ARX NOTH({c0})", "ARX", "asinh", "NOT")]:
            preds[etq] = pronosticar_rango(sub, mc, ma, fam, tr, ven, c0)
        for etq, ven in [("LEAR Win(todo)", "todo"), (f"LEAR NOT({c0})", "NOT")]:
            al = alphas_origen(sub, ven, "ninguna", c0)
            preds[etq] = pronosticar_rango(sub, mc, ma, "LEAR", "ninguna", ven, c0, al)

        comparaciones = [(f"ARX Win({c0})", f"ARX NOT({c0})"),
                         (f"ARX WinH({c0})", f"ARX NOTH({c0})"),
                         ("LEAR Win(todo)", f"LEAR NOT({c0})")]
        print(f"  {'referencia':18s} {'con puntos de cambio':22s} {'MAE':>16s} {'MAPE %':>16s} {'DM p':>8s}")
        for a, b in comparaciones:
            ra, rb = preds[a], preds[b]
            y = ra["real"].to_numpy()
            ma_, mb = met(y, ra["pred"]), met(y, rb["pred"])
            p = dm(y, ra["pred"].to_numpy(), rb["pred"].to_numpy())
            print(f"  {a:18s} {b:22s} {ma_['MAE']:7.2f}->{mb['MAE']:7.2f} "
                  f"{ma_['MAPE_%']:7.2f}->{mb['MAPE_%']:7.2f} {p:8.4f}"
                  f"{'  MEJORA' if mb['MAE'] < ma_['MAE'] else '  empeora'}", flush=True)
            filas.append(dict(origen=nom, inicio=ini, fin=fin, n=len(ra), C0=c0,
                              referencia=a, variante=b,
                              MAE_ref=ma_["MAE"], MAE_not=mb["MAE"],
                              MAPE_ref=ma_["MAPE_%"], MAPE_not=mb["MAPE_%"],
                              RMSE_ref=ma_["RMSE"], RMSE_not=mb["RMSE"], DM_p=p))
        pd.DataFrame(filas).to_csv(SALIDA, index=False)          # se guarda tras cada origen
        print(f"  guardado ({(time.time()-t0)/60:.0f} min acumulados)", flush=True)

    R = pd.DataFrame(filas)
    print(f"\n{'='*104}\nRESUMEN: ¿en cuantos origenes mejora cada comparacion?\n{'='*104}")
    R["mejora"] = R["MAE_not"] < R["MAE_ref"]
    R["mejora_mape"] = R["MAPE_not"] < R["MAPE_ref"]
    for (ref, var), g in R.groupby([R["referencia"].str.replace(r"\(\d+\)", "(C0)", regex=True),
                                    R["variante"].str.replace(r"\(\d+\)", "(C0)", regex=True)]):
        print(f"  {var:20s} contra {ref:18s}: MAE mejor en {int(g['mejora'].sum())}/{len(g)} origenes, "
              f"MAPE mejor en {int(g['mejora_mape'].sum())}/{len(g)}, "
              f"significativo (p<0.05) en {int((g['DM_p']<0.05).sum())}/{len(g)}")
    print(f"\nGuardado: {SALIDA.name}  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
