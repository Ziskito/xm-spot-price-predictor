# -*- coding: utf-8 -*-
"""
EL PRECIO "PEGAJOSO": pronosticar el ATOMO del precio del corte, no un valor continuo.

EL HALLAZGO QUE LO MOTIVA. El precio de bolsa colombiano es escalonado: cada dia tiene en promedio
solo 7.2 niveles distintos en 24 horas (las plantas ofertan un precio diario y el precio horario lo
marca la planta marginal). Y hay una regularidad muy fuerte en la frontera entre dias:

    hora   P(precio == precio del corte EXACTO)     (todo / 2026)
     01          73.4% / 70.8%
     02          60.6% / 61.1%
     03-04       ~55-60%
     05          48%
     06-08       24-41%  /  34-38%
     00          21.9% / 38.4%

O sea: la planta que marca el precio a las 00:00 sigue marcandolo en las horas siguientes con alta
probabilidad. Si se queda, el precio es EXACTAMENTE igual, no parecido.

POR QUE ESTO ES UN PROBLEMA DE PUNTOS DE CAMBIO, Y POR QUE LOS INTENTOS ANTERIORES LO PERDIERON.
La pregunta "¿cambia la planta marginal entre el corte y la hora h?" es exactamente la pregunta de
si hay un punto de cambio intradia. La distribucion del precio en la hora h es una MEZCLA:

    con probabilidad pi_h:      precio = precio del corte           (un atomo)
    con probabilidad 1 - pi_h:  precio ~ alguna distribucion continua

Para error absoluto (MAE, y aproximadamente MAPE) el pronostico optimo es la MEDIANA de esa mezcla,
y la mediana de una mezcla con un atomo de masa pi > 0.5 ES EL ATOMO. Ningun regresor continuo
entrenado con MAE produce ese valor exacto: se queda cerca y paga error en todas las horas donde el
precio no se movio (se midio: en tramos planos el real se mueve 0.00 por hora y el ensamble 6.62).

Y explica el fracaso del modelo hurdle de rampas: aquel multiplicaba P(se mueve) x magnitud, que es
la ESPERANZA de la mezcla -- optima para error cuadratico, no para MAE. Encogia hacia cero en vez de
saltar al atomo.

LA PRUEBA
  1. Un clasificador por hora estima pi_h = P(precio_h == precio del corte) con informacion del
     corte. Variables: el diseno del LEAR mas dos que miden la "inercia" de la planta marginal:
     cuantas horas seguidas lleva el precio en el nivel del corte, y si el corte es el minimo del
     dia (la planta base nocturna).
  2. Regla de mediana: si pi_h > tau, el pronostico es EXACTAMENTE el precio del corte; si no, se
     deja el del ensamble. Con tau = 0.5 la regla sale de la teoria, sin ajustar nada. Tambien se
     reporta tau elegido en un periodo de validacion.
  3. Oraculo: lo que se ganaria sabiendo de antemano en que horas se queda. Cota superior.

TODO CAUSAL: el clasificador se entrena solo con dias anteriores a 2026 y el precio del corte es,
por definicion, el ultimo precio conocido.
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
FIN_VALID = pd.Timestamp("2026-03-01")
SEMILLAS = list(range(10))
PICO = [0, 8, 9, 10, 18, 19]


def dm(y, p_ref, p_nuevo, metrica="mae", maxlags=24):
    y = np.asarray(y, float)
    if metrica == "mape":
        l1, l2 = np.abs(y - p_ref) / y, np.abs(y - p_nuevo) / y
    else:
        l1, l2 = np.abs(y - p_ref), np.abs(y - p_nuevo)
    d = l1 - l2
    if np.allclose(d, 0):
        return np.nan
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": min(maxlags, len(d)//3)})
    return float(r.pvalues[0])


def disenos_con_inercia():
    """Diseno del LEAR por paso + variables de inercia de la planta marginal en el corte."""
    df = cargar_completo().reset_index(drop=True)
    precio = df["precio_bolsa"].to_numpy(float).round(4)
    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    cortes = idx_00[idx_00 >= 200]

    # racha en el corte: cuantas horas seguidas, hacia atras, el precio vale lo mismo que en el corte
    racha = np.zeros(len(precio))
    for i in range(1, len(precio)):
        racha[i] = racha[i - 1] + 1 if precio[i] == precio[i - 1] else 0

    salida = {}
    for h in range(1, 25):
        X, y, f_obj, f_cor = construir_para_paso(df, h, cortes)
        pos = pd.Series(df.index.values, index=fechas).reindex(pd.DatetimeIndex(f_cor)).to_numpy()
        pc = precio[pos]
        v24 = np.stack([precio[pos - k] for k in range(0, 24)], axis=1)
        X = X.copy()
        X["racha_corte"] = racha[pos]
        X["corte_es_min24"] = (pc <= v24.min(1) + 1e-9).astype(float)
        X["corte_es_max24"] = (pc >= v24.max(1) - 1e-9).astype(float)
        X["niveles_24h"] = [len(np.unique(r)) for r in v24]
        # ¿se quedo pegado ayer a esta misma hora? (el objetivo de hace 24h vs el corte de ayer)
        pos_ayer = pos - 24
        X["pegado_ayer"] = (precio[pos_ayer + h] == precio[pos_ayer]).astype(float)
        salida[h] = dict(X=X, y=np.round(y, 4), f_obj=pd.DatetimeIndex(f_obj),
                         f_cor=pd.DatetimeIndex(f_cor), pc=pc)
    return salida


def main():
    t0 = time.time()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss

    print("Construyendo disenos con variables de inercia...", flush=True)
    dis = disenos_con_inercia()

    filas = []
    print("\n" + "=" * 100)
    print("1) ¿SE PUEDE ANTICIPAR SI EL PRECIO SE QUEDA EN EL DEL CORTE?  (entrenado < 2026, medido en 2026)")
    print("=" * 100)
    print(f"  {'paso':>4s} {'P base 2026':>12s} {'AUC':>7s} {'Brier':>7s} {'Brier ingenuo':>14s}")
    for h in range(1, 25):
        d = dis[h]
        pega = (d["y"] == d["pc"]).astype(int)
        tr = d["f_cor"] < CORTE
        te = ~tr
        clf = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05,
                                             random_state=42).fit(d["X"][tr], pega[tr])
        pi = clf.predict_proba(d["X"][te])[:, 1]
        auc = roc_auc_score(pega[te], pi) if 0 < pega[te].mean() < 1 else np.nan
        br = brier_score_loss(pega[te], pi)
        br0 = brier_score_loss(pega[te], np.full(te.sum(), pega[tr].mean()))
        if h <= 8 or h in (12, 18, 24):
            print(f"  {h:4d} {100*pega[te].mean():11.1f}% {auc:7.3f} {br:7.3f} {br0:14.3f}")
        filas.append(pd.DataFrame({"fecha_hora": d["f_obj"][te], "pi": pi, "pc": d["pc"][te],
                                   "pega": pega[te], "paso": h}))
    C = pd.concat(filas).drop_duplicates("fecha_hora").set_index("fecha_hora")

    # ------------------------------------------------ ensamble
    e = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    e = e.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    e["dia"] = e.index.normalize()
    e["g4"] = e.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    e = e.dropna(subset=MEJOR + ["real"]).join(C, how="inner")
    e["pred"] = qra_cv(e, MEJOR, "smape")
    y, p = e["real"].to_numpy(), e["pred"].to_numpy()
    pi, pc, pega = e["pi"].to_numpy(), e["pc"].to_numpy(), e["pega"].to_numpy().astype(bool)
    val = (e.index < FIN_VALID)
    tes = ~val
    serie = cargar_completo().set_index("fecha_hora")["precio_bolsa"]
    prev = serie.reindex(e.index - pd.Timedelta(hours=1)); prev.index = e.index
    rampa = (e["real"] - prev).abs().to_numpy()
    hh = e.index.hour.to_numpy()

    print("\n" + "=" * 100)
    print("2) REGLA DE LA MEDIANA: si pi > tau, pronosticar EXACTAMENTE el precio del corte")
    print("=" * 100)
    base_mae = np.abs(y - p).mean()
    print(f"  ensamble actual: MAE {base_mae:.2f}  MAPE {mape(y,p):.2f}%  sMAPE {smape(y,p):.2f}%")
    print(f"\n  {'regla':30s} {'MAE':>7s} {'MAPE':>7s} {'sMAPE':>7s} {'horas cambiadas':>16s} "
          f"{'acierto':>8s} {'p(MAE)':>8s} {'p(MAPE)':>8s}")
    # tau elegido en validacion
    taus = np.round(np.arange(0.3, 0.96, 0.05), 2)
    perd = [np.abs(y[val] - np.where(pi[val] > t, pc[val], p[val])).mean() for t in taus]
    tau_val = float(taus[int(np.argmin(perd))])
    resultados = {}
    for nom, t in [("tau = 0.5 (teoria, sin ajustar)", 0.5), (f"tau = {tau_val} (validacion)", tau_val),
                   ("tau = 0.7", 0.7), ("tau = 0.8", 0.8)]:
        q = np.where(pi > t, pc, p)
        cam = pi > t
        print(f"  {nom:30s} {np.abs(y-q).mean():7.2f} {mape(y,q):7.2f} {smape(y,q):7.2f} "
              f"{int(cam.sum()):16d} {100*pega[cam].mean() if cam.any() else 0:7.1f}% "
              f"{dm(y,p,q):8.4f} {dm(y,p,q,'mape'):8.4f}")
        resultados[nom] = q
    q_or = np.where(pega, pc, p)
    print(f"  {'ORACULO (sabe donde se queda)':30s} {np.abs(y-q_or).mean():7.2f} {mape(y,q_or):7.2f} "
          f"{smape(y,q_or):7.2f} {int(pega.sum()):16d}")

    print("\n  Lo mismo SOLO en el periodo de prueba (mar-ago 2026), con tau elegido en ene-feb:")
    q = np.where(pi > tau_val, pc, p)
    print(f"    ensamble {np.abs(y[tes]-p[tes]).mean():.2f} -> regla {np.abs(y[tes]-q[tes]).mean():.2f}  "
          f"(DM p={dm(y[tes], p[tes], q[tes]):.4f})")

    print("\n" + "=" * 100)
    print("3) ¿DONDE ACTUA?  Por hora del reloj, con tau = 0.5")
    print("=" * 100)
    q = resultados["tau = 0.5 (teoria, sin ajustar)"]
    print(f"  {'hora':>5s} {'P pega':>7s} {'cambiadas':>10s} {'MAE antes':>10s} {'MAE despues':>12s}")
    for h in [1, 2, 3, 4, 5, 6, 7, 8, 0]:
        g = hh == h
        print(f"  {h:5d} {100*pega[g].mean():6.0f}% {int((pi[g]>0.5).sum()):10d} "
              f"{np.abs(y[g]-p[g]).mean():10.2f} {np.abs(y[g]-q[g]).mean():12.2f}")
    fu = rampa >= 50
    print(f"\n  en rampa fuerte: {np.abs(y[fu]-p[fu]).mean():.2f} -> {np.abs(y[fu]-q[fu]).mean():.2f}")
    print(f"  en mesetas:      {np.abs(y[rampa==0]-p[rampa==0]).mean():.2f} -> "
          f"{np.abs(y[rampa==0]-q[rampa==0]).mean():.2f}")

    print("\n" + "=" * 100)
    print("4) VERIFICACION 10 PARTICIONES DEL COMBINADOR  (tau = 0.5, sin ajustar)")
    print("=" * 100)
    A, B = [], []
    for s_ in SEMILLAS:
        ps = qra_cv(e, MEJOR, "smape", s_)
        qs = np.where(pi > 0.5, pc, ps)
        A.append([np.abs(y - ps).mean(), mape(y, ps), smape(y, ps)])
        B.append([np.abs(y - qs).mean(), mape(y, qs), smape(y, qs)])
    A, B = np.array(A), np.array(B)
    for j, nm in enumerate(["MAE", "MAPE %", "sMAPE %"]):
        dl = B[:, j] - A[:, j]
        print(f"  {nm:8s} ensamble={A[:,j].mean():7.3f}+/-{A[:,j].std():.3f}  "
              f"con regla={B[:,j].mean():7.3f}+/-{B[:,j].std():.3f}  "
              f"delta={dl.mean():+7.3f}, mejor en {int((dl<0).sum())}/10")
    print(f"  VEREDICTO: {'SOLIDO' if all((B[:,j]-A[:,j]<0).all() for j in range(3)) else 'no concluyente'}")

    e.assign(pred_regla=resultados["tau = 0.5 (teoria, sin ajustar)"])[
        ["real", "pred", "pred_regla", "pi", "pc", "pega"]].to_csv(RES / "precio_pegajoso_mediana.csv")
    print(f"\nGuardado: precio_pegajoso_mediana.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
