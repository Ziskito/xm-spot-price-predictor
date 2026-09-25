# -*- coding: utf-8 -*-
"""
El primer dia del producto de 72h se esta prediciendo mucho peor que con el modelo de 24h propio.

Hallazgo que destapa la corrida de mejoras_a_72h.py. Sobre EXACTAMENTE las mismas horas objetivo:

    ensamble de 72h, tramo 1-24h .... MAE 46.43   MAPE 13.03%
    ensamble de 24h del proyecto .... MAE 40.72   MAPE 10.54%

Y el de 24h lo consigue con MENOS informacion, no con mas: su protocolo es corte a las 23:00 del dia
anterior, mientras el de 72h corta a las 00:00 del propio dia. O sea el de 24h va una hora por
detras y aun asi gana 2.5 puntos de MAPE. Se verifico que ninguna fila tiene el corte de 24h
posterior al corte de 72h (0 de 4922), asi que no hay fuga: es una perdida real del canal de 72h.

La razon es de construccion, no de datos. El ensamble de 72h vota entre 4 modelos generalistas
(LEAR, dos redes y un naive) entrenados para cubrir los 72 pasos, con pesos por tramo. El de 24h
vota entre 13 especialistas day-ahead (XGBoost, CatBoost, hurdle, Markov-Switching, N-BEATSx,
N-HiTS, RecursiveLS day-ahead, los afinados con Optuna, los directos) mas el GARCH-ged, con pesos
por franja de 6 horas y combinador alineado a la metrica. El canal de 72h estaba re-derivando el
dia 1 desde cero, peor que el canal que ya existia para ese mismo dia.

ALCANCE DEL ARREGLO. El pronostico de 24h para la hora t se emite en el corte 23:00 del dia
anterior a t. Ese corte es <= el corte de 72h solo si t cae dentro del propio dia del corte, lo que
cubre los pasos 1..23 (01:00 a 23:00). El paso 24 (00:00 del dia siguiente) tendria que usar un
pronostico emitido 23 horas DESPUES del corte de 72h, asi que se excluye y se deja como esta. Los
tramos 25-48h y 49-72h no se tocan: no existe pronostico day-ahead valido a esa distancia.

Se prueban seis configuraciones para los pasos 1-23 y se mide el efecto sobre el global de 72h.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog
from scipy import sparse

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import RES
from stacking_24h_v2 import cargar as cargar24, BASE5
from ensamble_72h_diario import TRAMOS, hac_p

NUEVO_72 = ["LEAR", "NBX", "NBX_exog", "NaiveEstac"]
SEMILLAS = list(range(10))
PASO_MAX_PUENTE = 23


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def smape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / ((np.abs(real) + np.abs(pred)) / 2)) * 100)


def dm_metrica(real, p_ref, p_new, metrica, maxlags=48):
    import statsmodels.api as sm
    real = np.asarray(real, float)
    if metrica == "mape":
        l1, l2 = np.abs(real - p_ref) / real, np.abs(real - p_new) / real
    elif metrica == "smape":
        l1 = np.abs(real - p_ref) / ((np.abs(real) + np.abs(p_ref)) / 2)
        l2 = np.abs(real - p_new) / ((np.abs(real) + np.abs(p_new)) / 2)
    else:
        l1, l2 = np.abs(real - p_ref), np.abs(real - p_new)
    d = l1 - l2
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def pesos_lad(X, y, w=None):
    """min sum w_i |y_i - Xw|  s.a. w>=0, suman 1. Mismo LAD del proyecto, con ponderacion."""
    n, k = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    c = np.concatenate([np.zeros(k), w, w])
    A_eq = sparse.vstack([
        sparse.hstack([sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)]),
        sparse.hstack([sparse.csr_matrix(np.ones((1, k))), sparse.csr_matrix((1, 2 * n))]),
    ]).tocsr()
    r = linprog(c, A_eq=A_eq, b_eq=np.concatenate([y, [1.0]]), bounds=(0, None), method="highs")
    return r.x[:k] if r.success else np.full(k, 1 / k)


def lad_cv(sub, cols, modo="lad", semilla=42, iteraciones=3, devolver_pesos=False):
    X, y = sub[cols].to_numpy(), sub["real"].to_numpy()
    P = np.full(len(sub), np.nan)
    dias = np.sort(sub["dia"].unique())
    acum = []
    for dt in np.array_split(np.random.default_rng(semilla).permutation(dias), 5):
        te = sub["dia"].isin(dt).to_numpy()
        tr = ~te
        w = None if modo == "lad" else 1.0 / np.maximum(y[tr], 1e-6)
        ww = pesos_lad(X[tr], y[tr], w)
        if modo == "smape":
            for _ in range(iteraciones):
                pr = X[tr] @ ww
                w = 2.0 / np.maximum(np.abs(y[tr]) + np.abs(pr), 1e-6)
                ww = pesos_lad(X[tr], y[tr], w)
        P[te] = X[te] @ ww
        acum.append(ww)
    return (P, np.mean(acum, axis=0)) if devolver_pesos else P


def causal(sub, cols, modo="lad", ventana=60, min_dias=14, iteraciones=3):
    """Pesos con solo dias anteriores: la cifra desplegable en tiempo real."""
    X, y = sub[cols].to_numpy(), sub["real"].to_numpy()
    dia = sub["dia"].to_numpy()
    dias = np.sort(np.unique(dia))
    P = np.full(len(sub), np.nan)
    for i in range(min_dias, len(dias)):
        ini = max(0, i - ventana)
        tr = np.isin(dia, dias[ini:i])
        te = dia == dias[i]
        w = None if modo == "lad" else 1.0 / np.maximum(y[tr], 1e-6)
        ww = pesos_lad(X[tr], y[tr], w)
        if modo == "smape":
            for _ in range(iteraciones):
                pr = X[tr] @ ww
                w = 2.0 / np.maximum(np.abs(y[tr]) + np.abs(pr), 1e-6)
                ww = pesos_lad(X[tr], y[tr], w)
        P[te] = X[te] @ ww
    return P


def cargar_datos():
    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    lear = pd.read_csv(RES / "pronostico_lear_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v = v.merge(lear[["fecha_hora", "cutoff", "paso_horas", "LEAR"]],
                on=["fecha_hora", "cutoff", "paso_horas"], how="inner")
    gar = RES / "pronostico_garcht_72h_2026.csv"
    if gar.exists():
        g = pd.read_csv(gar, parse_dates=["fecha_hora", "cutoff"])
        v = v.merge(g[["fecha_hora", "cutoff", "paso_horas", "GARCHt72"]],
                    on=["fecha_hora", "cutoff", "paso_horas"], how="left")
    v["dia"] = v["fecha_hora"].dt.normalize()

    # ---- canal de 24h: votantes individuales, GARCH-ged y el ensamble ya combinado ----
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    col = "pred" if "pred" in s.columns else s.columns[-1]
    d = d.join(s[col].rename("GARCH-ged"), how="left")
    votantes24 = [c for c in d.columns if c not in ("real", "dia")]
    d24 = d[votantes24].add_prefix("v24_").reset_index()

    e24 = pd.read_csv(RES / "combinador_optimo_mape.csv", parse_dates=["fecha_hora"])
    d24 = d24.merge(e24[["fecha_hora", "pred_smape"]].rename(columns={"pred_smape": "ENS24"}),
                    on="fecha_hora", how="inner")
    cols24 = ["ENS24"] + [f"v24_{c}" for c in votantes24]
    return v, d24, cols24


def main():
    t0 = time.time()
    v, d24, cols24 = cargar_datos()

    # ---- union con verificacion de corte ----
    p = v[v.paso_horas.between(1, PASO_MAX_PUENTE)].merge(d24, on="fecha_hora", how="inner")
    corte24 = p["fecha_hora"].dt.normalize() - pd.Timedelta(hours=1)
    fuga = int((corte24 > p["cutoff"]).sum())
    print(f"Pasos 1-{PASO_MAX_PUENTE}: {len(p):,} filas | filas con corte de 24h posterior al de 72h: {fuga}")
    if fuga:
        raise SystemExit("FUGA detectada, abortado")
    p = p.dropna(subset=NUEVO_72 + cols24 + ["real"]).sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
    y = p["real"].to_numpy()
    print(f"Tras dropna: {len(p):,} filas | dias: {p['dia'].nunique()}\n")

    # ---- modelos individuales de referencia ----
    print(f"{'='*104}\nINDIVIDUALES SOBRE LAS MISMAS HORAS (pasos 1-{PASO_MAX_PUENTE})\n{'='*104}")
    print(f"{'modelo':28s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s}")
    for c in NUEVO_72 + ["ENS24"]:
        print(f"{c:28s} {np.abs(y-p[c]).mean():8.2f} {mape(y,p[c]):8.2f} {smape(y,p[c]):8.2f}")

    # ---- configuraciones ----
    V24 = [c for c in cols24 if c != "ENS24"]
    CONFIGS = [
        ("A  4 votantes 72h (vigente)",              NUEVO_72,                  "lad"),
        ("B  solo ENS24",                            ["ENS24"],                 "lad"),
        ("C  4 votantes + ENS24, LAD",               NUEVO_72 + ["ENS24"],      "lad"),
        ("D  4 votantes + ENS24, comb. sMAPE",       NUEVO_72 + ["ENS24"],      "smape"),
        ("E  4 votantes + 14 votantes 24h, LAD",     NUEVO_72 + V24,            "lad"),
        ("F  4 votantes + 14 votantes 24h, sMAPE",   NUEVO_72 + V24,            "smape"),
        ("G  todo (ENS24 + individuales), sMAPE",    NUEVO_72 + ["ENS24"] + V24, "smape"),
    ]

    print(f"\n{'='*104}\nPUENTE 24h -> 72h, PASOS 1-{PASO_MAX_PUENTE} (DM contra A)\n{'='*104}")
    print(f"{'configuracion':42s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s}  "
          f"{'p(MAE)':>8s} {'p(MAPE)':>8s} {'p(sMAPE)':>9s}")
    preds, filas = {}, []
    for nom, cols, modo in CONFIGS:
        pr = lad_cv(p, cols, modo)
        preds[nom] = pr
        linea = f"{nom:42s} {np.abs(y-pr).mean():8.2f} {mape(y,pr):8.2f} {smape(y,pr):8.2f}"
        if not nom.startswith("A "):
            ps = [dm_metrica(y, preds[CONFIGS[0][0]], pr, m)[1] for m in ("mae", "mape", "smape")]
            linea += f"  {ps[0]:8.4f} {ps[1]:8.4f} {ps[2]:9.4f}"
        print(linea)
        filas.append({"config": nom, "MAE": np.abs(y-pr).mean(), "MAPE": mape(y, pr), "sMAPE": smape(y, pr)})

    # ---- ganador y verificacion multisemilla ----
    ganador = min([c for c in preds if not c.startswith("A ")], key=lambda c: mape(y, preds[c]))
    print(f"\n{'='*104}\nVERIFICACION MULTISEMILLA: '{ganador}' contra A ({len(SEMILLAS)} particiones)\n{'='*104}")
    cols_g, modo_g = next((c, m) for n, c, m in CONFIGS if n == ganador)
    M = {"A": [], "G": []}
    for s_ in SEMILLAS:
        pa, pg = lad_cv(p, NUEVO_72, "lad", s_), lad_cv(p, cols_g, modo_g, s_)
        M["A"].append([np.abs(y-pa).mean(), mape(y, pa), smape(y, pa)])
        M["G"].append([np.abs(y-pg).mean(), mape(y, pg), smape(y, pg)])
    A, G = np.array(M["A"]), np.array(M["G"])
    for j, nom in enumerate(["MAE", "MAPE %", "sMAPE %"]):
        dl = G[:, j] - A[:, j]
        print(f"  {nom:8s} A={A[:,j].mean():7.3f}+/-{A[:,j].std():.3f}  "
              f"ganador={G[:,j].mean():7.3f}+/-{G[:,j].std():.3f}  "
              f"delta={dl.mean():+7.3f}, mejor en {int((dl<0).sum())}/{len(SEMILLAS)}")
    solido = all((G[:, j] - A[:, j] < 0).all() for j in range(3))
    print(f"  VEREDICTO: {'SOLIDO en las tres metricas' if solido else 'revisar'}")

    # ---- desplegable (pesos causales, ventana movil de 60 dias) ----
    print(f"\n{'='*104}\nMODO DESPLEGABLE (pesos solo con dias anteriores, ventana movil 60d)\n{'='*104}")
    ca, cg = causal(p, NUEVO_72, "lad"), causal(p, cols_g, modo_g)
    ok = ~np.isnan(ca) & ~np.isnan(cg)
    print(f"  {'A vigente':24s} MAE={np.abs(y[ok]-ca[ok]).mean():7.2f}  MAPE={mape(y[ok],ca[ok]):6.2f}%  "
          f"sMAPE={smape(y[ok],cg[ok]*0+ca[ok]):6.2f}%")
    print(f"  {'ganador':24s} MAE={np.abs(y[ok]-cg[ok]).mean():7.2f}  MAPE={mape(y[ok],cg[ok]):6.2f}%  "
          f"sMAPE={smape(y[ok],cg[ok]):6.2f}%")
    _, pv = dm_metrica(y[ok], ca[ok], cg[ok], "mape")
    print(f"  DM sobre MAPE, desplegable: p={pv:.4f}")

    # ---- pesos que asigna el combinador ----
    _, w = lad_cv(p, cols_g, modo_g, 42, devolver_pesos=True)
    print(f"\n  Pesos medios del ganador: " + " | ".join(
        f"{c}={x:.3f}" for c, x in zip(cols_g, w) if x > 0.005))

    # ---- efecto sobre el global de 72h ----
    print(f"\n{'='*104}\nEFECTO SOBRE EL PRODUCTO COMPLETO DE 72h\n{'='*104}")
    p["pred_nueva"] = preds[ganador]
    llaves = ["cutoff", "paso_horas"]
    resto = v[~v.paso_horas.between(1, PASO_MAX_PUENTE) | ~v.set_index(llaves).index.isin(
        p.set_index(llaves).index)].copy()
    resto = resto.dropna(subset=NUEVO_72 + ["real"])
    filas_g = []
    for etiqueta in ("vigente", "con puente"):
        trozos = []
        for nom, ini, fin in TRAMOS:
            s = resto[resto.paso_horas.between(ini, fin)].sort_values(llaves).reset_index(drop=True)
            if len(s):
                trozos.append(pd.DataFrame({"real": s["real"].to_numpy(),
                                            "pred": lad_cv(s, NUEVO_72, "lad"),
                                            "tramo": nom}))
        s = p.sort_values(llaves).reset_index(drop=True)
        trozos.append(pd.DataFrame({"real": s["real"].to_numpy(),
                                    "pred": preds[CONFIGS[0][0]] if etiqueta == "vigente" else s["pred_nueva"].to_numpy(),
                                    "tramo": "1-24h"}))
        t = pd.concat(trozos, ignore_index=True).dropna()
        print(f"\n  {etiqueta}:")
        for nom, _, _ in TRAMOS:
            q = t[t.tramo == nom]
            print(f"    {nom:9s} MAE={np.abs(q['real']-q['pred']).mean():7.2f}  "
                  f"MAPE={mape(q['real'],q['pred']):6.2f}%  sMAPE={smape(q['real'],q['pred']):6.2f}%")
        print(f"    {'GLOBAL':9s} MAE={np.abs(t['real']-t['pred']).mean():7.2f}  "
              f"MAPE={mape(t['real'],t['pred']):6.2f}%  sMAPE={smape(t['real'],t['pred']):6.2f}%  "
              f"R2={1-((t['real']-t['pred'])**2).sum()/((t['real']-t['real'].mean())**2).sum():.4f}")
        filas_g.append({"escenario": etiqueta, "MAE": np.abs(t['real']-t['pred']).mean(),
                        "MAPE": mape(t['real'], t['pred']), "sMAPE": smape(t['real'], t['pred'])})

    pd.DataFrame(filas).to_csv(RES / "puente_24h_a_72h.csv", index=False)
    pd.DataFrame(filas_g).to_csv(RES / "puente_24h_a_72h_global.csv", index=False)
    pd.DataFrame({"fecha_hora": p["fecha_hora"], "cutoff": p["cutoff"], "paso_horas": p["paso_horas"],
                  "real": y, "pred_vigente": preds[CONFIGS[0][0]], "pred_puente": preds[ganador]}
                 ).to_csv(RES / "pronostico_puente_72h_2026.csv", index=False)
    print(f"\nGuardado: puente_24h_a_72h.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
