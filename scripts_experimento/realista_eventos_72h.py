# -*- coding: utf-8 -*-
"""
Detector de eventos (desplome / pico) por dia de anticipacion en el escenario realista, con validacion historica
(20-oct-2026, a pedido de Juan: en 72 h el dashboard no mostraba picos ni desplomes porque el detector solo existia
para 24 h).

Mismos eventos que realista_eventos.py: en la ventana objetivo T (01:00 de T a 00:00 de T+1), desplome si el precio
real <= 0,7 x mediana de la ventana y pico si >= 1,3 x mediana. El corte es a las 00:00 de D = T - k:
  k = 0  pasos 1-24  (el detector de 24 h actual)
  k = 1  pasos 25-48
  k = 2  pasos 49-72
Variables, todas publicadas al corte de D: forma y nivel del predespacho de D en la misma hora (el de T todavia no
existe si k > 0), embalses, aportes y ONI tal como se conocian en D, forma del precio real y frecuencia de eventos
de las ventanas completas hasta D-5, y el calendario de T (hora, dia de la semana, mes, festivo). Con k = 0 son las
mismas variables de realista_eventos.py.
Clasificador HistGradientBoosting (mismos hiperparametros del de 24 h) reentrenado con las ventanas ya publicadas
(hasta D-5 = T-k-5): cada semana en 2026, como en produccion, y cada mes en los periodos historicos (por tiempo de
computo; un modelo menos fresco solo puede jugar en contra del detector). Se compara con la climatologia por hora (frecuencia del evento a esa hora en el
entrenamiento) y con la regla del predespacho de D (0/1). Significancia del Brier frente a la climatologia con
Diebold-Mariano (HAC, 24 rezagos) sobre la perdida cuadratica.
Periodos: los 5 historicos (15-jul a 30-jun, 2020-2025) y 2026 (15-ene hasta la ultima ventana completa).
Salida: resultados/escenario_realista/eventos_72h/metricas.csv y prob_k{0,1,2}.csv
"""
import json
import sys
import time
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

RETRASO = 5
SAL = rc.MEJ.parent / "eventos_72h"
SAL.mkdir(exist_ok=True)
PERIODOS = [("Origen 1 (2020-21)", "2020-07-15", "2021-06-30"), ("Origen 2 (2021-22)", "2021-07-15", "2022-06-30"),
            ("Origen 3 (2022-23)", "2022-07-15", "2023-06-30"), ("Origen 4 (2023-24)", "2023-07-15", "2024-06-30"),
            ("Origen 5 (2024-25)", "2024-07-15", "2025-06-30"), ("2026", "2026-01-15", "2099-01-01")]


def dm_brier(y, p, c):
    import statsmodels.api as sm
    dd = (p - y) ** 2 - (c - y) ** 2
    r = sm.OLS(dd, np.ones_like(dd)).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    return float(r.tvalues[0]), float(r.pvalues[0])


def tabla(operacion=False):
    """operacion: conserva las ventanas futuras sin predespacho propio (D+1, D+2), que se predicen con el de D."""
    d = rc.tabla_realista(produccion=True).set_index("fecha_hora")
    d = d[d.index >= "2019-02-15"]
    if not operacion:
        d = d.dropna(subset=["pre_obj"])
    d = d.join(rf.forma(d))
    d["dia"] = pd.to_datetime(d["dia"])
    d = d[d.groupby("dia")["dia"].transform("size") == 24]
    d["paso"] = d.groupby("dia").cumcount()
    completo = d.groupby("dia")["precio_bolsa"].transform("count") == 24
    med = d.groupby("dia")["precio_bolsa"].transform("median").where(completo)
    for e, u in UMBRAL.items():
        d[e] = ((d["precio_bolsa"] <= u * med) if e == "desplome" else (d["precio_bolsa"] >= u * med)).where(completo)
    d["completo"] = completo
    d["real_rel"] = d["precio_bolsa"] / med
    return d


def variables(d, k):
    """Variables por (ventana objetivo T, paso) con lo publicado al corte de D = T - k."""
    clave = pd.MultiIndex.from_arrays([d["dia"], d["paso"]])
    mpre = d.groupby("dia")["pre_obj"].transform("median")
    info = pd.DataFrame({"pre_rel_med": (d["pre_obj"] / mpre).to_numpy(),
                         "pre_min_rel": (d.groupby("dia")["pre_obj"].transform("min") / mpre).to_numpy(),
                         "pre_max_rel": (d.groupby("dia")["pre_obj"].transform("max") / mpre).to_numpy(),
                         "nivel_log": np.log(mpre.clip(lower=1)).to_numpy(),
                         **{c: d[c].to_numpy() for c in ("pre_d1", "pre_d1_sig", "pre_rango_hora", "pre_z",
                                                         "pre_max_hora", "pre_min_hora")},
                         **{c: d[c].to_numpy() for c in ("volumen_embalses", "aportes_hidricos", "oni") if c in d}},
                        index=clave)
    X = pd.DataFrame(index=d.index)
    X["hora"], X["paso"], X["dow"], X["mes"] = d.index.hour, d["paso"], d["dia"].dt.dayofweek, d.index.month
    if "es_festivo" in d:
        X["es_festivo"] = d["es_festivo"]
    # informacion del dia D = T - k en la misma posicion de la ventana
    origen = pd.MultiIndex.from_arrays([d["dia"] - pd.Timedelta(days=k), d["paso"]])
    for c in info.columns:
        X[c] = info[c].reindex(origen).to_numpy()
    X["k"] = k
    # precio real ya publicado: ventanas completas hasta D-5 = T-k-5
    tab = d.loc[d["completo"], ["dia", "paso", "real_rel"] + list(UMBRAL)]
    piv = tab.pivot_table(index="dia", columns="paso", values="real_rel").asfreq("D")
    for j in (RETRASO + k, RETRASO + k + 2):
        X[f"real_rel_lag{j - k}"] = piv.shift(j).stack().reindex(clave).to_numpy()
    for e in UMBRAL:
        pe = tab.pivot_table(index="dia", columns="paso", values=e).asfreq("D")
        X[f"{e}_frec7"] = pe.rolling(7, min_periods=3).mean().shift(RETRASO + k).stack().reindex(clave).to_numpy()
        pdia = tab.groupby("dia")[e].mean().asfreq("D")
        X[f"{e}_frec7_dia"] = pdia.rolling(7, min_periods=3).mean().shift(RETRASO + k).reindex(d["dia"]).to_numpy()
    regla = {e: ((X["pre_rel_med"] <= u) if e == "desplome" else (X["pre_rel_med"] >= u)).astype(float)
             for e, u in UMBRAL.items()}
    return X, regla


def predecir(d, X, k, ini, fin, freq="W"):
    """Reentrena cada semana (freq="W") o cada mes ("M") de [ini, fin] con las ventanas publicadas al corte del primer
    dia del bloque."""
    sel = ((d["dia"] >= ini) & (d["dia"] <= fin)).to_numpy()
    semanas = d.loc[sel, "dia"].dt.to_period(freq).astype(str)
    res = {e: np.full(len(d), np.nan) for e in UMBRAL}
    clim = {e: np.full(len(d), np.nan) for e in UMBRAL}
    for sem in sorted(semanas.unique()):
        te = np.zeros(len(d), bool)
        te[np.where(sel)[0][(semanas == sem).to_numpy()]] = True
        t0 = d.loc[te, "dia"].min()
        tr = ((d["dia"] <= t0 - pd.Timedelta(days=k + RETRASO)) & d["completo"]).to_numpy()
        for e in UMBRAL:
            y = d.loc[tr, e].astype(int).to_numpy()
            fr = pd.Series(y).groupby(X.loc[tr, "hora"].to_numpy()).mean()
            clim[e][te] = X.loc[te, "hora"].map(fr).fillna(y.mean()).to_numpy()
            m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
                                               random_state=0)
            m.fit(X.loc[tr], y)
            res[e][te] = m.predict_proba(X.loc[te])[:, 1]
    return res, clim


SALIDA = rc.RAIZ / "data/processed/resultados/probabilidades_eventos_72h_realista_2026.csv"


def operacion():
    """Produccion (actualizar_produccion.py, despues de pronostico_realista_72h.py): probabilidad de PICO para los pasos
    25-72 de cada corte del contrato de 72 h. Solo pico: en la validacion historica el detector de desplome a 25-72 h
    no supera de forma consistente a la climatologia (20-oct-2026, benchmark/RESUMEN.md). Reentreno semanal como en
    2026; las ventanas ya empezadas (dia <= hoy) quedan como se publicaron y solo se calculan las semanas con horas
    nuevas."""
    from produccion_datos import hoy
    res_dir = rc.RAIZ / "data/processed/resultados"
    nombre = json.loads((res_dir / "fuentes_pronostico.json").read_text(encoding="utf-8"))["72h"]["archivo"]
    c72 = pd.read_csv(res_dir / nombre, parse_dates=["fecha_hora"]).drop_duplicates("fecha_hora").set_index("fecha_hora")
    c72 = c72[c72["paso_horas"] > 24]
    c72["k"] = (c72["paso_horas"] - 1) // 24
    viejo = pd.read_csv(SALIDA, parse_dates=["fecha_hora"]).set_index("fecha_hora") if SALIDA.exists() else None
    congelar = pd.DatetimeIndex([])
    if viejo is not None:
        congelar = viejo.index[(viejo.index - pd.Timedelta(hours=1)).normalize() <= hoy().normalize()]
    faltan = c72.index.difference(congelar)
    d = tabla(operacion=True)
    p = pd.Series(np.nan, index=c72.index, name="p_pico")
    for k in (1, 2):
        horas = faltan[c72.loc[faltan, "k"] == k]
        if not len(horas):
            continue
        dias = (horas - pd.Timedelta(hours=1)).normalize()
        X, _ = variables(d, k)
        res, _ = predecir(d, X, k, dias.min(), dias.max())
        p.loc[horas] = pd.Series(res["pico"], index=d.index).reindex(horas).to_numpy()
    if viejo is not None:
        quedan = congelar.intersection(p.index)
        p.loc[quedan] = viejo.loc[quedan, "p_pico"]
    out = pd.DataFrame({"p_pico": p, "p_desplome": np.nan})
    out.index.name = "fecha_hora"
    out.to_csv(SALIDA)
    print(f"probabilidad de pico a 25-72 h: {len(faltan)} horas calculadas, hasta {out.index.max()}")


if __name__ == "__main__":
    if "--operacion" in sys.argv:
        operacion()
        sys.exit(0)
    t0 = time.time()
    d = tabla()
    filas = []
    for k in (0, 1, 2):
        X, regla = variables(d, k)
        prob = pd.DataFrame(index=d.index)
        for nom, ini, fin in PERIODOS:
            res, clim = predecir(d, X, k, pd.Timestamp(ini), pd.Timestamp(fin), "W" if nom == "2026" else "M")
            ev = (np.isfinite(res["pico"]) & d["completo"].to_numpy())
            for e in UMBRAL:
                prob.loc[np.isfinite(res[e]), f"p_{e}"] = res[e][np.isfinite(res[e])]
                y = d.loc[ev, e].astype(int).to_numpy()
                c, p, r = clim[e][ev], res[e][ev], regla[e].to_numpy()[ev]
                for met, pp in (("climatologia por hora", c), ("predespacho de D (0/1)", r), ("clasificador", p)):
                    fila = {"k": k, "pasos": f"{24 * k + 1}-{24 * k + 24}", "periodo": nom, "evento": e, "metodo": met,
                            **metricas(y, pp, c)}
                    if met != "climatologia por hora":
                        fila["DM_t_Brier_vs_clim"], fila["DM_p"] = dm_brier(y, pp, c)
                    filas.append(fila)
            print(f"  k={k} {nom} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        prob.dropna(how="all").to_csv(SAL / f"prob_k{k}.csv")
        pd.DataFrame(filas).to_csv(SAL / "metricas.csv", index=False)
    t = pd.DataFrame(filas)
    pd.set_option("display.width", 250)
    for e in UMBRAL:
        v = t[(t["evento"] == e) & (t["metodo"] == "clasificador")]
        print(f"\n{e}: BSS frente a la climatologia (positivo = mejor) y AUC del clasificador")
        print(v.pivot(index="periodo", columns="pasos", values="BSS_vs_climatologia").round(3).to_string())
        print(v.pivot(index="periodo", columns="pasos", values="AUC").round(3).to_string())
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
