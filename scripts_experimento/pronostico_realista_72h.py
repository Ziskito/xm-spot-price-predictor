# -*- coding: utf-8 -*-
"""
Pronostico de 72 h del ESCENARIO REALISTA para el motor y el dashboard (14-oct-2026).

Corte a las 00:00 del dia D, solo con lo publicado a esa hora (precio y demanda hasta las 23:00 de D-4; embalses y aportes
de D-2; ONI de M-2; predespacho ideal, dMAR y dDEM del dia D, publicados el dia anterior). Ventanas de 01:00 a 00:00:
  pasos 1-24  (dia D)    el pronostico realista de 24 h en produccion (pronostico_realista_24h.py)
  pasos 25-72 (D+1, D+2) no hay predespacho de esos dias. Para el objetivo t del dia D+k (k = 1, 2):
              * referencia: el predespacho del dia D a la misma hora (estado del mercado mas reciente)
              * tres CatBoost por k (perdida MAE, objetivo log(precio / referencia)) con las variables publicadas al
                corte (las de la fila t - 24k de la tabla realista: precio y demanda rezagados, hidrologia, ONI,
                predespacho del dia D y su forma, dMAR/dDEM de D, error reciente del predespacho) y el calendario del dia
                objetivo; reentrenados cada mes con lo publicado (objetivos hasta 73 h antes del corte); se promedian
              * combinador causal con la referencia (QRA, objetivo MAE, franjas de 6 h); solo aprende de ventanas ya
                publicadas (margen de 5 + k dias) y usa como previa jul-2024 a jun-2025 los primeros 14 dias
              * banda del 80 %: q50 x exp(+-m), m = cuantil conforme de |log(real/q50)| de la misma hora y el mismo k en los
                60 dias anteriores con precio ya publicado
Contrato: mismos cortes que el contrato anterior (uno cada 3 dias desde el 2-ene, 72 pasos), para que las ventanas no se
solapen. Se evalua ademas con cortes diarios.
Salidas (resultados/): pronostico_realista_72h_diario_2026.csv (todos los cortes), pronostico_realista_bandas_72h_2026.csv
(contrato) y metricas en reentreno_oct2026/realista_72h.txt. Con --activar (o --solo-activar, sin reentrenar), fuentes_pronostico.json apunta el horizonte
de 72 h a este contrato.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
import escenario_realista_24h as er  # noqa: E402
import realista_exp_errorpre as rep  # noqa: E402
import realista_exp_forma as rf  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402

RES = rc.RAIZ / "data" / "processed" / "resultados"
INI_TR = pd.Timestamp("2019-02-15")
OCULTO = pd.Timedelta(hours=73)
CORTE_INI, CORTE_FIN = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-08-03")     # cortes diarios evaluados
CALENDARIO = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos", "dia_anio_sin", "dia_anio_cos", "es_festivo",
              "festivo_lag24h", "festivo_lag48h", "festivo_lag72h", "festivo_lag168h", "mismatch_festivo_24h",
              "mismatch_festivo_168h", "es_pandemia"]


def tabla_k(d, k, sets):
    """Filas por hora objetivo t (ventana del dia D+k): variables de la fila t - 24k (publicadas al corte de D) y el
    calendario de t. La referencia es el predespacho de D a la misma hora (pre_obj de la fila t - 24k)."""
    src = d.shift(24 * k, freq="h").reindex(d.index)              # fila t - 24k alineada con t
    info = sorted({c for cols in sets.values() for c in cols} - set(CALENDARIO))
    X = src[info].copy()
    for c in CALENDARIO:
        if c in d:
            X[c] = d[c]
    X["ref"] = src["pre_obj"]
    X["precio"] = d["precio_bolsa"]
    X["corte"] = (d.index - pd.Timedelta(hours=1)).normalize() - pd.Timedelta(days=k)     # 00:00 del dia D
    return X


def catboost(tr, te, cols, semilla=42):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.05, loss_function="MAE", random_seed=semilla, verbose=0, thread_count=6)
    m.fit(tr[cols], np.log(tr["precio"]) - np.log(tr["ref"].clip(lower=1)))
    return pd.Series(np.exp(m.predict(te[cols]) + np.log(te["ref"].clip(lower=1))), index=te.index)


def limpio(x):
    return x.dropna(subset=["precio", "ref", "precio_lag24h"])


def mensual(X, cols, ini, fin):
    partes = []
    for mes in pd.date_range(ini.replace(day=1), fin, freq="MS"):
        tr = limpio(X[(X.index >= INI_TR) & (X.index <= mes - OCULTO)])
        te = X[(X["corte"] >= mes) & (X["corte"] < mes + pd.offsets.MonthBegin(1)) & (X["corte"] <= fin)].dropna(subset=["ref"])
        if len(te):
            partes.append(catboost(tr, te, cols))
    return pd.concat(partes)


def bandas(p, k, ventana=60, nivel=0.80):
    """Banda conforme en escala log por hora del dia; solo ventanas publicadas (5 + k dias antes del dia objetivo)."""
    p = p.sort_index().copy()
    p["s"] = np.abs(np.log(p["real"].clip(lower=1)) - np.log(p["q50"].clip(lower=1)))
    p["hora"] = p.index.hour
    m = np.full(len(p), np.nan)
    retraso = 5 + k
    for dia in p["dia"].unique():
        hist = p[(p["dia"] <= dia - pd.Timedelta(days=retraso)) & (p["dia"] > dia - pd.Timedelta(days=retraso + ventana))]
        hoy = (p["dia"] == dia).to_numpy()
        for h, g in hist.groupby("hora"):
            n = len(g)
            if n >= 7:
                m[hoy & (p["hora"] == h).to_numpy()] = np.quantile(g["s"], min(np.ceil((n + 1) * nivel) / n, 1.0))
    m = pd.Series(m, index=p.index).ffill().bfill().fillna(0.35)
    return p["q50"] * np.exp(-m), p["q50"] * np.exp(m), m


def main():
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    F = rf.forma(d)
    d = d.join(F)
    C = rf.cm_programado(d)
    d = d.join(C)
    E = rep.errores(d)
    d = d.join(E)
    RP = conjuntos(d)["R+P"]
    sets = {"Cat_RP": RP, "Cat_RPFC": RP + list(F.columns) + list(C.columns), "Cat_RPE": RP + list(E.columns)}
    salidas, lineas = [], []
    for k in (1, 2):
        X = tabla_k(d, k, sets)
        cols = {n: [c for c in s if c in X.columns] for n, s in sets.items()}
        sel = (X["corte"] >= CORTE_INI) & (X["corte"] <= CORTE_FIN)
        p = pd.DataFrame(index=X.index[sel & X["precio"].notna()])
        p["real"], p["ref"] = X.loc[p.index, "precio"], X.loc[p.index, "ref"]
        for n, c in cols.items():
            p[n] = mensual(X, c, CORTE_INI, CORTE_FIN).reindex(p.index)
            print(f"  k={k} {n}: MAE {np.abs(p[n] - p['real']).mean():.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        p["Cat_prom"] = p[list(cols)].mean(axis=1)
        # previa del combinador: jul-2024 a jun-2025, modelos entrenados antes de ese periodo
        pv_idx = X.index[(X.index >= "2024-07-01") & (X.index < "2025-07-01")]
        trp = limpio(X[(X.index >= INI_TR) & (X.index <= pd.Timestamp("2024-07-01") - OCULTO - pd.Timedelta(days=k))])
        tep = X.loc[pv_idx].dropna(subset=["ref"])
        previa = pd.DataFrame({"real": X.loc[tep.index, "precio"], "ref": tep["ref"],
                               "Cat_prom": np.mean([catboost(trp, tep, c) for c in cols.values()], axis=0)}).dropna()
        for q in (p, previa):
            q["dia"] = (q.index - pd.Timedelta(hours=1)).normalize()
            q["g4"] = q.index.hour // 6
        p = p.dropna(subset=["ref", "Cat_prom"])
        p["q50"] = causal(p, previa, ["ref", "Cat_prom"], "g4", margen=5 + k, objetivo="mae")
        p["q50"] = p["q50"].fillna(p["ref"])
        p["q10"], p["q90"], p["margen"] = bandas(p, k)
        p["k"] = k
        p["cutoff"] = p["dia"] - pd.Timedelta(days=k)
        salidas.append(p)
        y = p["real"]
        lineas.append(f"k={k} (pasos {24 * k + 1}-{24 * k + 24}): MAE q50 {np.abs(p['q50'] - y).mean():.2f} | referencia (predespacho de D) "
                      f"{np.abs(p['ref'] - y).mean():.2f} | promedio CatBoost {np.abs(p['Cat_prom'] - y).mean():.2f} | cobertura "
                      f"{((y >= p['q10']) & (y <= p['q90'])).mean() * 100:.1f} % | {len(p)} h")
        print(lineas[-1], flush=True)
    # pasos 1-24: contrato realista de 24 h
    c24 = pd.read_csv(RES / "pronostico_realista_bandas_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    c24["k"], c24["dia"] = 0, (c24.index - pd.Timedelta(hours=1)).normalize()
    c24["cutoff"] = c24["dia"]
    todo = pd.concat([c24[["real", "q50", "q10", "q90", "margen", "k", "cutoff"]]] + [s[["real", "q50", "q10", "q90", "margen", "k", "cutoff"]] for s in salidas])
    todo.index.name = "fecha_hora"
    todo = todo.reset_index()
    todo["paso_horas"] = ((todo["fecha_hora"] - todo["cutoff"]) / pd.Timedelta(hours=1)).astype(int)
    todo = todo[todo["paso_horas"].between(1, 72)].sort_values(["cutoff", "paso_horas"])
    todo.to_csv(RES / "pronostico_realista_72h_diario_2026.csv", index=False)
    # contrato: mismos cortes que el contrato anterior (uno cada 3 dias), 72 pasos completos
    viejo = pd.read_csv(RES / "pronostico_ensamble_bandas_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    con = todo[todo["cutoff"].isin(viejo["cutoff"].unique())]
    con = con[con.groupby("cutoff")["paso_horas"].transform("size") == 72]
    con = con[["fecha_hora", "cutoff", "paso_horas", "real", "q50", "q10", "q90", "margen"]]
    assert con["fecha_hora"].is_unique
    con.to_csv(RES / "pronostico_realista_bandas_72h_2026.csv", index=False)
    comun = con.merge(viejo[["fecha_hora", "cutoff", "q50"]].rename(columns={"q50": "q50_opt"}), on=["fecha_hora", "cutoff"])
    lineas.append(f"Contrato (cortes cada 3 dias): {con['cutoff'].nunique()} cortes, {len(con)} h, cobertura "
                  f"{((con.real >= con.q10) & (con.real <= con.q90)).mean() * 100:.1f} %, ancho medio {(con.q90 - con.q10).mean():.0f}")
    for a, b in ((1, 24), (25, 48), (49, 72), (1, 72)):
        s = comun[comun.paso_horas.between(a, b)]
        t, pv = er.dm(s["real"].to_numpy(), s["q50"].to_numpy(), s["q50_opt"].to_numpy())
        lineas.append(f"  pasos {a}-{b}: MAE realista {np.abs(s.q50 - s.real).mean():.2f} | optimista anterior {np.abs(s.q50_opt - s.real).mean():.2f} "
                      f"(DM t = {t:.2f}, p = {pv:.3f})")
    for a, b in ((1, 24), (25, 48), (49, 72)):
        s = todo[todo.paso_horas.between(a, b)]
        lineas.append(f"  cortes diarios, pasos {a}-{b}: MAE {np.abs(s.q50 - s.real).mean():.2f}, {len(s)} h")
    texto = "\n".join(lineas)
    (RES / "reentreno_oct2026").mkdir(exist_ok=True)
    (RES / "reentreno_oct2026" / "realista_72h.txt").write_text(texto, encoding="utf-8")
    print(texto)
    if "--activar" in sys.argv:
        activar(((con.real >= con.q10) & (con.real <= con.q90)).mean() * 100)
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")


def activar(cob):
    if True:
        ruta = RES / "fuentes_pronostico.json"
        conf = json.loads(ruta.read_text(encoding="utf-8"))
        conf["72h"] = {"archivo": "pronostico_realista_bandas_72h_2026.csv",
                       "modelo": "Escenario realista: pasos 1-24 = pronostico realista de 24 h; pasos 25-72 = predespacho del dia del "
                                 "corte + promedio de 3 CatBoost por horizonte reentrenados cada mes (combinador causal, objetivo MAE) "
                                 "+ bandas conformes por hora en escala log (60 dias)",
                       "calibrado": True, "cobertura_objetivo_pct": 80, "cobertura_medida_pct": round(cob, 1),
                       "generado": pd.Timestamp.now().strftime("%Y-%m-%d"),
                       "notebook_origen": "scripts_experimento/pronostico_realista_72h.py",
                       "notas": "Solo usa lo publicado a las 00:00 del dia del corte. Contrato anterior (escenario optimista) en "
                                "fuentes_pronostico_optimista.json."}
        ruta.write_text(json.dumps(conf, indent=2, ensure_ascii=False), encoding="utf-8")
        print("fuentes_pronostico.json: 72h -> contrato realista")


if __name__ == "__main__":
    if "--solo-activar" in sys.argv:          # activa el contrato ya calculado, sin reentrenar
        c = pd.read_csv(RES / "pronostico_realista_bandas_72h_2026.csv")
        activar(((c["real"] >= c["q10"]) & (c["real"] <= c["q90"])).mean() * 100)
    else:
        main()
