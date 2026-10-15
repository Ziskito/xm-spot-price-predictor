# -*- coding: utf-8 -*-
"""
Detector de eventos (desplome / pico) en el escenario realista (solo lo publicado a las 00:00 de D).

Mismos eventos que eventos_escenarios_24h.py, por hora y relativos a la mediana del precio real de la ventana
(01:00 de D a 00:00 de D+1): desplome real <= 0,7 x mediana; pico real >= 1,3 x mediana.
Variables (todas publicadas al corte): forma del predespacho del dia (relativo a su mediana, minimo, maximo,
cambio, z, posicion de la hora; realista_exp_forma.forma), nivel del predespacho, forma del precio real de la ventana
de D-5 en la misma hora (la ultima completa) y frecuencia de eventos en las 7 ventanas publicadas, embalses y aportes
de D-2, calendario. Clasificador HistGradientBoosting entrenado con 2019-02 a 2025 y reentrenado cada semana de 2026
con lo publicado (ventanas hasta D-5). Se compara con la climatologia por hora y con el predespacho solo (0/1).
Salida: resultados/escenario_realista/mejoras/eventos_realista.csv y eventos_realista_prob.csv; con --produccion
(desde el 1-ene) tambien resultados/probabilidades_eventos_24h_realista_2026.csv, el archivo que lee el dashboard.
Con --produccion --operacion (actualizar_produccion.py): datos hasta hoy y probabilidades hasta la ventana del ultimo
predespacho publicado; las ventanas aun sin precio real completo no se etiquetan (no entrenan ni se evaluan, pero se
predicen) y las ya empezadas (dia <= hoy) quedan como se publicaron; solo se recalculan las semanas con ventanas nuevas.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
import realista_exp_forma as rf  # noqa: E402
from eventos_escenarios_24h import UMBRAL, metricas  # noqa: E402

RETRASO = 5          # ventanas completas publicadas al corte de D: hasta la de D-5

if __name__ == "__main__":
    OPER = "--operacion" in sys.argv
    if OPER:
        from ensamble_con_predespacho_24h import imar
        from produccion_datos import hoy
        d = rc.tabla_realista(produccion=True).set_index("fecha_hora")
        d = d[(d.index >= "2019-02-15") & (d.index <= imar().index.max().normalize() + pd.Timedelta(days=1))].dropna(subset=["pre_obj"])
    else:
        d = rc.tabla_realista().set_index("fecha_hora")
        d = d[d.index >= "2019-02-15"].dropna(subset=["precio_bolsa", "pre_obj"])
    d = d.join(rf.forma(d))
    d["dia"] = pd.to_datetime(d["dia"])
    d = d[d.groupby("dia")["pre_obj" if OPER else "precio_bolsa"].transform("size") == 24]
    completo = (d.groupby("dia")["precio_bolsa"].transform("count") == 24).to_numpy()     # ventana con precio real completo
    d["paso"] = d.groupby("dia").cumcount()
    med = d.groupby("dia")["precio_bolsa"].transform("median").where(completo)
    for e, u in UMBRAL.items():
        d[e] = ((d["precio_bolsa"] <= u * med) if e == "desplome" else (d["precio_bolsa"] >= u * med)).where(completo)
    mpre = d.groupby("dia")["pre_obj"].transform("median")
    X = pd.DataFrame(index=d.index)
    X["hora"], X["paso"], X["dow"], X["mes"] = d.index.hour, d["paso"], d["dia"].dt.dayofweek, d.index.month
    X["pre_rel_med"] = d["pre_obj"] / mpre
    X["pre_min_rel"] = d.groupby("dia")["pre_obj"].transform("min") / mpre
    X["pre_max_rel"] = d.groupby("dia")["pre_obj"].transform("max") / mpre
    X["nivel_log"] = np.log(mpre.clip(lower=1))
    for c in ("pre_d1", "pre_d1_sig", "pre_rango_hora", "pre_z", "pre_max_hora", "pre_min_hora"):
        X[c] = d[c]
    # forma del precio real ya publicado: ventana de D-5 en la misma posicion, y frecuencia de eventos en 7 ventanas
    real_rel = (d["precio_bolsa"] / med).rename("r")
    tab = pd.DataFrame({"dia": d["dia"], "paso": d["paso"], "r": real_rel.to_numpy(),
                        **{e: d[e].astype(float).to_numpy() for e in UMBRAL}})[completo]
    piv = tab.pivot_table(index="dia", columns="paso", values="r").asfreq("D")
    for k in (RETRASO, RETRASO + 2):
        X[f"real_rel_lag{k}"] = piv.shift(k).stack().reindex(pd.MultiIndex.from_arrays([d["dia"], d["paso"]])).to_numpy()
    for e in UMBRAL:
        pe = tab.pivot_table(index="dia", columns="paso", values=e).asfreq("D")
        X[f"{e}_frec7"] = pe.rolling(7, min_periods=3).mean().shift(RETRASO).stack().reindex(
            pd.MultiIndex.from_arrays([d["dia"], d["paso"]])).to_numpy()
        pd_ = tab.groupby("dia")[e].mean().asfreq("D")
        X[f"{e}_frec7_dia"] = pd_.rolling(7, min_periods=3).mean().shift(RETRASO).reindex(d["dia"]).to_numpy()
    for c in ("volumen_embalses", "aportes_hidricos", "oni"):
        if c in d:
            X[c] = d[c]
    PROD = "--produccion" in sys.argv       # para el dashboard: desde el 1-ene (las metricas siguen igual de causales)
    SALIDA = rc.RAIZ / "data/processed/resultados/probabilidades_eventos_24h_realista_2026.csv"
    es26 = np.asarray(d.index >= ("2026-01-01 01:00" if PROD else "2026-01-15"))
    semanas = d.loc[es26, "dia"].dt.to_period("W").astype(str)
    calcular = sorted(semanas.unique())
    viejo = None
    if OPER and SALIDA.exists():              # ventanas ya empezadas: congeladas; solo las semanas con ventanas nuevas
        viejo = pd.read_csv(SALIDA, parse_dates=["fecha_hora"], float_precision="round_trip").set_index("fecha_hora")
        viejo = viejo[(viejo.index - pd.Timedelta(hours=1)).normalize() <= hoy().normalize()]
        nuevas = ~d.index[es26].isin(viejo.index)
        calcular = sorted(semanas[nuevas].unique())
    res = {e: np.full(len(d), np.nan) for e in UMBRAL}
    clim = {e: np.full(len(d), np.nan) for e in UMBRAL}
    for sem in calcular:
        te = np.zeros(len(d), bool)
        te[np.where(es26)[0][(semanas == sem).to_numpy()]] = True
        ini = d.loc[te, "dia"].min()
        tr = (d["dia"] <= ini - pd.Timedelta(days=RETRASO)).to_numpy() & completo
        for e in UMBRAL:
            y = d.loc[tr, e].astype(int).to_numpy()
            fr = pd.Series(y).groupby(X.loc[tr, "hora"].to_numpy()).mean()
            clim[e][te] = X.loc[te, "hora"].map(fr).fillna(y.mean()).to_numpy()
            mdl = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=0)
            mdl.fit(X.loc[tr], y)
            res[e][te] = mdl.predict_proba(X.loc[te])[:, 1]
    filas = []
    prob = d.loc[es26, ["dia", "paso", "precio_bolsa", "pre_obj"]].copy()
    ev = es26 & completo & ~np.isnan(res["pico"])                  # horas etiquetadas y calculadas en esta corrida
    for e, u in UMBRAL.items():
        y = d.loc[ev, e].astype(int).to_numpy()
        p_pre = ((d["pre_obj"] <= u * mpre) if e == "desplome" else (d["pre_obj"] >= u * mpre)).astype(float).to_numpy()[ev]
        if len(y) and 0 < y.sum() < len(y):
            for nom, p in (("climatologia por hora", clim[e][ev]), ("predespacho solo (0/1)", p_pre), ("clasificador realista", res[e][ev])):
                filas.append({"evento": e, "metodo": nom, **metricas(y, p, clim[e][ev])})
        prob[f"p_{e}"] = res[e][es26]
    if viejo is not None:
        quedan = prob.index.intersection(viejo.index)
        prob.loc[quedan, ["p_desplome", "p_pico"]] = viejo.loc[quedan, ["p_desplome", "p_pico"]]
    t = pd.DataFrame(filas)
    if OPER:
        prob.index.name = "fecha_hora"
        prob[["p_desplome", "p_pico"]].to_csv(SALIDA)
        print(f"probabilidades de eventos: {len(calcular)} semanas recalculadas, hasta {prob.index.max()}")
        sys.exit(0)
    suf = "_produccion" if PROD else ""
    t.to_csv(rc.MEJ / f"eventos_realista{suf}.csv", index=False)
    prob.to_csv(rc.MEJ / f"eventos_realista_prob{suf}.csv")
    if PROD:
        prob.index.name = "fecha_hora"
        prob[["p_desplome", "p_pico"]].to_csv(SALIDA)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
