# -*- coding: utf-8 -*-
"""
Pronosticar los escalones del precio como EVENTOS (pruebas del 6-oct-2026).

El diagnostico (diagnostico_amplitud_24h.py) mostro que el pronostico acierta la direccion de las rampas
(96 %) pero solo anticipa el 40 % de su magnitud, y que estirar la mediana empeora el MAE: el
amortiguamiento es la respuesta optima de un pronostico puntual cuando no se sabe si ni cuando habra
salto. Aqui se pronostica directamente la probabilidad del salto.

Eventos, por hora y relativos a la mediana del precio real de la ventana de pronostico (24 h):
    desplome: real <= 0,7 x mediana        pico: real >= 1,3 x mediana

Metodos (todos causales; evaluados en 2026 con reentrenamiento semanal):
  A. clasificador (HistGradientBoosting) con la forma del pronostico de los 5 votantes, su desacuerdo,
     el ancho de banda, la forma del precio real de dias anteriores, calendario e hidrologia rezagada;
     se entrena con los origenes historicos 1-5 (2020-2025) mas las semanas previas de 2026.
     Variante A+XM: agrega variables de XM (despacho programado, disponibilidad declarada, ofertas con
     7 dias de rezago) si existe data/processed/xm_despacho_ofertas_2026.csv (solo 2026: se entrena
     dentro de 2026).
  B. escenarios: la mediana del dia multiplicada por las curvas de error relativo de las 120 ventanas
     anteriores (conservan los escalones reales); probabilidad = fraccion de escenarios con el evento.
  C. referencias: frecuencia historica por hora (climatologia), probabilidad implicita en la banda
     (normal partida con q10/q50/q90) y la mediana sola (0/1).
Metricas: Brier y su mejora sobre la climatologia (BSS), area ROC, precision promedio (PR) y cuantos
eventos se detectan con probabilidad >= 0,5.
Salidas: data/processed/resultados/eventos_escenarios/ (metricas.csv, probabilidades_2026.csv, figuras)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "data" / "processed" / "resultados"
OUT = RES / ("eventos_escenarios" if "--sin-predespacho" in sys.argv else "eventos_escenarios_predespacho")
VOT = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]
USAR_PREDESPACHO = "--sin-predespacho" not in sys.argv
UMBRAL = {"desplome": 0.7, "pico": 1.3}


def panel():
    crudas = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    v = crudas.pivot_table(index=["origen", "fecha_hora"], columns="modelo", values="prediccion").reset_index()
    hist = pd.read_csv(RES / "ensamble_origenes_historicos_predicciones.csv", parse_dates=["fecha_hora"])
    hist = hist.merge(v, on=["origen", "fecha_hora"], how="inner")
    hist["ventana"] = hist["fecha_hora"].dt.normalize()                      # cortes a las 23:00: 00..23 h
    c26 = pd.read_csv(RES / "pronostico_ensamble_bandas_24h_2026.csv", parse_dates=["fecha_hora"])
    c26 = c26.merge(v[v["origen"] == "Origen 6"].drop(columns="origen"), on="fecha_hora", how="inner")
    c26["origen"] = "Origen 6"
    c26["ventana"] = (c26["fecha_hora"] - pd.Timedelta(hours=1)).dt.normalize()   # corte 00:00: 01..00 h
    d = pd.concat([hist, c26], ignore_index=True).dropna(subset=["real", "q50", "q10", "q90"] + VOT)
    d = d[d.groupby(["origen", "ventana"])["real"].transform("size") == 24].sort_values(["origen", "fecha_hora"])
    d["paso"] = d.groupby(["origen", "ventana"]).cumcount()
    med = d.groupby(["origen", "ventana"])["real"].transform("median")
    for e, u in UMBRAL.items():
        d[e] = (d["real"] <= u * med) if e == "desplome" else (d["real"] >= u * med)
    return d.reset_index(drop=True)


def variables(d):
    g = d.groupby(["origen", "ventana"])
    m50 = g["q50"].transform("median")
    X = pd.DataFrame(index=d.index)
    X["hora"], X["paso"], X["dow"] = d["fecha_hora"].dt.hour, d["paso"], d["ventana"].dt.dayofweek
    X["mes"] = d["fecha_hora"].dt.month
    X["q50_rel"] = d["q50"] / m50                                   # forma del pronostico
    X["q50_min_rel"] = g["q50"].transform("min") / m50
    X["q50_max_rel"] = g["q50"].transform("max") / m50
    X["q50_cambio"] = g["q50"].diff().fillna(0) / m50
    X["nivel_log"] = np.log(m50.clip(lower=1))
    for c in VOT:
        X[f"{c}_rel"] = d[c] / m50
    X["desacuerdo"] = d[VOT].std(axis=1) / m50
    X["min_votantes_rel"] = d[VOT].min(axis=1) / m50
    X["max_votantes_rel"] = d[VOT].max(axis=1) / m50
    X["banda_rel"] = (d["q90"] - d["q10"]) / m50
    # forma del precio real de ventanas anteriores (misma posicion horaria), conocida al corte
    for k in (1, 7):
        X[f"real_rel_lag{k}"] = d.groupby(["origen", "paso"])["real"].shift(k) / \
            d.assign(r=g["real"].transform("median")).groupby(["origen", "paso"])["r"].shift(k)
    for e in UMBRAL:
        X[f"{e}_frec7"] = d.groupby(["origen", "paso"])[e].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
        X[f"{e}_frec7_dia"] = d.assign(z=d[e].astype(float)).groupby(["origen", "ventana"])["z"].transform("mean")
        X[f"{e}_frec7_dia"] = X[f"{e}_frec7_dia"].groupby([d["origen"], d["paso"]]).shift(1)
    m = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, parse_dates=["fecha_hora"])
                   for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]).drop_duplicates("fecha_hora")
    m = m.set_index("fecha_hora").sort_index()[["volumen_embalses", "aportes_hidricos", "demanda", "oni"]]
    rez = m.shift(48, freq="h")                                      # rezago de 2 dias: seguro al corte
    rez["aportes_vs_30d"] = rez["aportes_hidricos"] / rez["aportes_hidricos"].rolling(720, min_periods=24).mean()
    X = X.join(rez.reindex(d["fecha_hora"]).reset_index(drop=True))
    # predespacho ideal de XM (publicado el dia anterior; ver descargar_predespacho_ideal.py), alineado al corte:
    # en 2026 la hora 00:00 del dia siguiente usa el valor de las 23:00 (el iMAR de ese dia sale despues del corte)
    ruta = RAIZ / "data" / "processed" / "predespacho_ideal_xm.csv"
    if ruta.exists() and USAR_PREDESPACHO:
        cm = pd.read_csv(ruta, parse_dates=["fecha_hora"]).set_index("fecha_hora")["costo_marginal"]
        t = d["fecha_hora"]
        es26 = (d["origen"] == "Origen 6").to_numpy()
        fuente = t.where(~es26 | (t.dt.normalize() == d["ventana"]), d["ventana"] + pd.Timedelta(hours=23))
        pre = pd.Series(cm.reindex(pd.DatetimeIndex(fuente)).to_numpy(), index=d.index)
        mpre = pre.groupby([d["origen"], d["ventana"]]).transform("median")
        X["pre_rel"] = pre / mpre
        X["pre_vs_q50"] = pre / d["q50"]
        X["pre_min_rel"] = pre.groupby([d["origen"], d["ventana"]]).transform("min") / mpre
        X["pre_max_rel"] = pre.groupby([d["origen"], d["ventana"]]).transform("max") / mpre
        X["pre_cambio"] = pre.groupby([d["origen"], d["ventana"]]).diff().fillna(0) / mpre
    return X


def prob_banda(d, u, evento):
    """P(evento) suponiendo una normal partida con q10/q50/q90 y umbral u x mediana del pronostico."""
    m50 = d.groupby(["origen", "ventana"])["q50"].transform("median")
    x = u * m50
    sl = ((d["q50"] - d["q10"]) / 1.2816).clip(lower=1e-6)
    su = ((d["q90"] - d["q50"]) / 1.2816).clip(lower=1e-6)
    cdf = np.where(x <= d["q50"], 0.5 * 2 * norm.cdf((x - d["q50"]) / sl), 0.5 + 0.5 * (2 * norm.cdf((x - d["q50"]) / su) - 1))
    return cdf if evento == "desplome" else 1 - cdf


def escenarios(d, ventanas_previas=120):
    """Curvas de error relativo de ventanas anteriores sobre la mediana del dia -> P(evento) y rango del dia."""
    d = d.copy()
    d["err"] = d["real"] / d["q50"].clip(lower=1) - 1
    curvas = d.pivot_table(index=["origen", "ventana"], columns="paso", values="err").sort_index()
    q50 = d.pivot_table(index=["origen", "ventana"], columns="paso", values="q50").sort_index()
    orden = curvas.index.get_level_values("ventana")
    salida = {}
    banco = curvas.to_numpy()
    for i, (o, v) in enumerate(q50.index):
        if o != "Origen 6":
            continue
        previas = banco[(orden < v)][-ventanas_previas:]                  # historico + 2026 anterior
        sc = q50.loc[(o, v)].to_numpy()[None, :] * (1 + previas)          # (n, 24)
        med = np.median(sc, axis=1, keepdims=True)
        salida[v] = {"desplome": (sc <= UMBRAL["desplome"] * med).mean(axis=0),
                     "pico": (sc >= UMBRAL["pico"] * med).mean(axis=0),
                     "rangos": sc.max(axis=1) - sc.min(axis=1)}
    return salida


def metricas(y, p, clim):
    ok = np.isfinite(p) & np.isfinite(clim)          # semanas sin datos de entrenamiento: fuera
    y, p, clim = np.asarray(y, float)[ok], np.asarray(p)[ok], np.asarray(clim)[ok]
    b, bc = np.mean((p - y) ** 2), np.mean((clim - y) ** 2)
    det = p >= 0.5
    return {"Brier": b, "BSS_vs_climatologia": 1 - b / bc, "AUC": roc_auc_score(y, p),
            "precision_promedio": average_precision_score(y, p),
            "eventos": int(y.sum()), "detectados_p05": int((det & (y == 1)).sum()),
            "falsas_alarmas_p05": int((det & (y == 0)).sum()), "horas_evaluadas": int(len(y))}


def evaluar(d, X, solo26=False):
    """Reentrena cada semana de 2026 con todo lo anterior (o solo con 2026 anterior si solo26)."""
    es26 = (d["origen"] == "Origen 6").to_numpy()
    semanas = d.loc[es26, "ventana"].dt.to_period("W").astype(str)
    res = {e: np.full(len(d), np.nan) for e in UMBRAL}
    clim = {e: np.full(len(d), np.nan) for e in UMBRAL}
    cols = list(X.columns)
    for sem in sorted(semanas.unique()):
        te = np.zeros(len(d), bool)
        te[np.where(es26)[0][(semanas == sem).to_numpy()]] = True
        ini = d.loc[te, "ventana"].min()
        tr = (d["ventana"] < ini).to_numpy() & (es26 if solo26 else True)
        for e in UMBRAL:
            y = d.loc[tr, e].astype(int).to_numpy()
            fr = pd.Series(y).groupby(X.loc[tr, "hora"].to_numpy()).mean()
            clim[e][te] = X.loc[te, "hora"].map(fr).fillna(y.mean()).to_numpy()
            if y.sum() < 10 or len(y) < 24 * 21:
                res[e][te] = clim[e][te]
                continue
            mdl = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=4,
                                                 l2_regularization=1.0, random_state=0)
            mdl.fit(X.loc[tr, cols], y)
            res[e][te] = mdl.predict_proba(X.loc[te, cols])[:, 1]
    return res, clim


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d = panel()
    X = variables(d)
    es26 = (d["origen"] == "Origen 6").to_numpy()
    print(f"panel: {len(d)} horas ({(~es26).sum()} historicas 2020-2025, {es26.sum()} de 2026)", flush=True)
    clf, clim = evaluar(d, X)
    sc = escenarios(d)
    filas, probs = [], d.loc[es26, ["fecha_hora", "ventana", "paso", "real", "q50", "q10", "q90"]].copy()
    for e, u in UMBRAL.items():
        y = d.loc[es26, e].astype(int).to_numpy()
        c = clim[e][es26]
        p_sc = np.array([sc[v][e][p] for v, p in zip(d.loc[es26, "ventana"], d.loc[es26, "paso"])])
        p_band = prob_banda(d, u, e)[es26]
        m50 = d.groupby(["origen", "ventana"])["q50"].transform("median")
        p_pt = ((d["q50"] <= u * m50) if e == "desplome" else (d["q50"] >= u * m50)).astype(float).to_numpy()[es26]
        for nombre, p in (("climatologia por hora", c), ("mediana sola (0/1)", p_pt), ("implicita en la banda", p_band),
                          ("escenarios (120 ventanas)", p_sc), ("clasificador", clf[e][es26])):
            filas.append({"evento": e, "metodo": nombre, **metricas(y, np.clip(p, 0, 1), c)})
        probs[f"y_{e}"], probs[f"p_{e}_clasificador"], probs[f"p_{e}_escenarios"] = y, clf[e][es26], p_sc
    # variante con variables de XM (solo 2026, porque la descarga cubre dic-2025 a ago-2026)
    ruta_xm = RAIZ / "data" / "processed" / "xm_despacho_ofertas_2026.csv"
    if ruta_xm.exists() and "--con-xm" in sys.argv:   # variantes API de XM (prueba del 6-oct)
        xm = pd.read_csv(ruta_xm, parse_dates=["fecha_hora"]).set_index("fecha_hora")
        ofertas = [c for c in xm.columns if c.startswith("oferta")]
        xm[ofertas] = xm[ofertas].shift(24 * 7, freq="h").reindex(xm.index)       # rezago real de publicacion
        prog = [c for c in xm.columns if c.startswith("prog_")]
        dispo = [c for c in xm.columns if c.startswith("dispo_")]
        tot_p, tot_d = xm[prog].sum(axis=1), xm[dispo].sum(axis=1)
        xm["prog_termica_frac"] = xm.get("prog_termica", 0) / tot_p
        xm["prog_solar_frac"] = xm.get("prog_solar", 0) / tot_p
        xm["prog_hidraulica_frac"] = xm.get("prog_hidraulica", 0) / tot_p
        xm["margen_dispo"] = (tot_d - tot_p) / tot_d
        grupos = {"despacho programado": ["prog_termica_frac", "prog_solar_frac", "prog_hidraulica_frac"] + prog,
                  "disponibilidad declarada": dispo + ["margen_dispo"],
                  "ofertas (rezago 7 dias)": ofertas}
        Xb = X.copy()
        for cols in grupos.values():
            for c in cols:
                Xb[c] = xm[c].reindex(d["fecha_hora"]).to_numpy()
        variantes = [("2026 solo, sin XM", [])] + [(f"2026 solo, + {k}", v) for k, v in grupos.items()]             + [("2026 solo, + todo XM", sum(grupos.values(), []))]
        for nom, cols in variantes:
            r, cl = evaluar(d, Xb[list(X.columns) + cols], solo26=True)
            for e in UMBRAL:
                y = d.loc[es26, e].astype(int).to_numpy()
                filas.append({"evento": e, "metodo": f"clasificador {nom}", **metricas(y, r[e][es26], cl[e][es26])})
            print("listo:", nom, flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(OUT / "metricas.csv", index=False)
    probs.to_csv(OUT / "probabilidades_2026.csv", index=False)
    # archivo que lee el dashboard (solo con la version con predespacho, la que se despliega)
    if USAR_PREDESPACHO:
        probs[["fecha_hora", "p_desplome_clasificador", "p_pico_clasificador"]].rename(
            columns={"p_desplome_clasificador": "p_desplome", "p_pico_clasificador": "p_pico"}).to_csv(
            RES / "probabilidades_eventos_24h_2026.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
