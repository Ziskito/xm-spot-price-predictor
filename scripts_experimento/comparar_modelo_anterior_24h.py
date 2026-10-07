# -*- coding: utf-8 -*-
"""
Comparacion del modelo de 24 h anterior (ensamble de 6 votantes) con el nuevo (6 + predespacho ideal de XM),
sobre exactamente las mismas horas de 2026.

Archivos comparados (contratos del motor, con bandas calibradas):
  anterior: data/processed/resultados/pronostico_ensamble_bandas_24h_2026_6votantes.csv
  nuevo:    data/processed/resultados/pronostico_ensamble_bandas_24h_2026.csv
Tablas (data/processed/resultados/comparacion_modelo_anterior/):
  1_global.csv        MAE, RMSE, MAPE, sMAPE, sesgo, rMAE frente a la persistencia, Diebold-Mariano
                      (todo 2026 y desde el 15-ene, el periodo de la cifra del informe)
  2_por_mes.csv       MAE por mes
  3_por_hora.csv      MAE por hora del dia
  4_regimen.csv       precio normal (ene-abr) frente a El Niño (may-ago)
  5_rampas_forma.csv  rampas, amplitud y descomposicion del error (nivel / amplitud / forma)
  6_bandas.csv        cobertura, ancho, puntaje de intervalo (Winkler) y cobertura por hora
  7_eventos.csv       deteccion de desplomes y picos (mediana y clasificador)
  8_motor.csv         decisiones del motor con cada pronostico, sin y con filtro intradia
Figura: comparacion_modelo_anterior.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "data" / "processed" / "resultados"
OUT = RES / "comparacion_modelo_anterior"
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import motor_decision as md  # noqa: E402
from ensamble_24h_enero import datos_2026  # noqa: E402

NOMBRES = {"ant": "anterior (6 modelos)", "nue": "nuevo (6 + predespacho)"}


def dm(y, a, b, maxlags=24):
    """t > 0: el segundo pronostico (b) tiene menor error absoluto que el primero (a)."""
    d = np.abs(y - a) - np.abs(y - b)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def basicas(y, p):
    e = p - y
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "MAPE_%": np.mean(np.abs(e) / y) * 100,
            "sMAPE_%": np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p))) * 100, "sesgo": e.mean()}


def cambio(a, b):
    return (b - a) / a * 100


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ant = pd.read_csv(RES / "pronostico_ensamble_bandas_24h_2026_6votantes.csv", parse_dates=["fecha_hora"])
    nue = pd.read_csv(RES / "pronostico_ensamble_bandas_24h_2026.csv", parse_dates=["fecha_hora"])
    d = ant.merge(nue, on=["fecha_hora", "real"], suffixes=("_ant", "_nue"))
    per = datos_2026()["Persistencia"]
    d["persistencia"] = per.reindex(d["fecha_hora"]).to_numpy()
    d["dia"] = (d["fecha_hora"] - pd.Timedelta(hours=1)).dt.normalize()
    d["hora"] = d["fecha_hora"].dt.hour
    y = d["real"].to_numpy()
    pd.set_option("display.width", 250)

    # 1. global
    filas = []
    for periodo, m in (("2026 completo (01-ene a 05-ago)", np.ones(len(d), bool)),
                       ("desde 15-ene (periodo del informe)", (d["fecha_hora"] >= "2026-01-15").to_numpy())):
        ok = m & d["persistencia"].notna().to_numpy()
        for k in ("ant", "nue"):
            p = d[f"q50_{k}"].to_numpy()
            f = {"periodo": periodo, "modelo": NOMBRES[k], "horas": int(m.sum()), **basicas(y[m], p[m]),
                 "rMAE_vs_persistencia": np.abs(y[ok] - p[ok]).mean() / np.abs(y[ok] - d["persistencia"].to_numpy()[ok]).mean()}
            f["DM_t_vs_persistencia"], f["DM_p_vs_persistencia"] = dm(y[ok], d["persistencia"].to_numpy()[ok], p[ok])
            if k == "nue":
                f["DM_t_vs_anterior"], f["DM_p_vs_anterior"] = dm(y[m], d["q50_ant"].to_numpy()[m], p[m])
            filas.append(f)
    t1 = pd.DataFrame(filas)
    t1.to_csv(OUT / "1_global.csv", index=False)
    print("1. GLOBAL\n" + t1.round(3).to_string(index=False))

    # 2. por mes y 3. por hora
    def tabla_mae(clave):
        g = d.groupby(clave).apply(lambda x: pd.Series({"horas": len(x), "MAE_anterior": np.abs(x.real - x.q50_ant).mean(),
                                                         "MAE_nuevo": np.abs(x.real - x.q50_nue).mean(),
                                                         "MAPE_anterior_%": np.mean(np.abs(x.real - x.q50_ant) / x.real) * 100,
                                                         "MAPE_nuevo_%": np.mean(np.abs(x.real - x.q50_nue) / x.real) * 100}))
        g["cambio_MAE_%"] = cambio(g.MAE_anterior, g.MAE_nuevo)
        return g
    t2 = tabla_mae(d["fecha_hora"].dt.to_period("M").astype(str))
    t2.to_csv(OUT / "2_por_mes.csv")
    print("\n2. POR MES\n" + t2.round(2).to_string())
    t3 = tabla_mae("hora")
    t3.to_csv(OUT / "3_por_hora.csv")
    print("\n3. POR HORA DEL DIA\n" + t3.round(1).to_string())

    # 4. regimen
    nino = (d["fecha_hora"] >= "2026-05-01").to_numpy()
    filas = []
    for nom, m in (("normal (ene-abr)", ~nino), ("El Niño (may-ago)", nino)):
        f = {"regimen": nom, "horas": int(m.sum())}
        for k in ("ant", "nue"):
            b = basicas(y[m], d[f"q50_{k}"].to_numpy()[m])
            f[f"MAE_{k}"], f[f"MAPE_{k}_%"] = b["MAE"], b["MAPE_%"]
        f["DM_t"], f["DM_p"] = dm(y[m], d["q50_ant"].to_numpy()[m], d["q50_nue"].to_numpy()[m])
        filas.append(f)
    t4 = pd.DataFrame(filas)
    t4.to_csv(OUT / "4_regimen.csv", index=False)
    print("\n4. REGIMEN\n" + t4.round(2).to_string(index=False))

    # 5. rampas y forma
    filas = []
    dr = d.groupby("dia")["real"].diff(3)
    top = (dr.abs() >= dr.abs().quantile(.9)) & dr.notna()
    for k in ("ant", "nue"):
        p = d[f"q50_{k}"]
        dp = p.groupby(d["dia"]).diff(3)
        sd = d.assign(p=p).groupby("dia").agg(sp=("p", "std"), sy=("real", "std"))
        rho = d.assign(p=p).groupby("dia").apply(lambda x: x["p"].corr(x["real"]))
        niv = (p.groupby(d["dia"]).mean() - d.groupby("dia")["real"].mean()) ** 2
        amp, forma = (sd.sp - sd.sy) ** 2, 2 * sd.sp * sd.sy * (1 - rho)
        tot = niv.sum() + amp.sum() + forma.sum()
        filas.append({"modelo": NOMBRES[k], "MAE_horas_de_rampa": np.abs(d.real - p)[top].mean(),
                      "rampas_direccion_%": (np.sign(dr[top]) == np.sign(dp[top])).mean() * 100,
                      "rampas_fraccion_anticipada": (dp[top] / dr[top]).median(),
                      "amplitud_dentro_del_dia": (sd.sp / sd.sy).median(), "correlacion_horaria": rho.median(),
                      "error_cuadratico_total": tot, "err_nivel_%": niv.sum() / tot * 100,
                      "err_amplitud_%": amp.sum() / tot * 100, "err_forma_%": forma.sum() / tot * 100})
    t5 = pd.DataFrame(filas)
    t5.to_csv(OUT / "5_rampas_forma.csv", index=False)
    print("\n5. RAMPAS Y FORMA\n" + t5.round(2).to_string(index=False))

    # 6. bandas
    filas = []
    for k in ("ant", "nue"):
        lo, hi = d[f"q10_{k}"], d[f"q90_{k}"]
        cub = (d.real >= lo) & (d.real <= hi)
        alfa = 0.2
        winkler = (hi - lo) + 2 / alfa * ((lo - d.real).clip(lower=0) + (d.real - hi).clip(lower=0))
        ph = cub.groupby(d["hora"]).mean() * 100
        filas.append({"modelo": NOMBRES[k], "cobertura_%": cub.mean() * 100, "ancho_medio": (hi - lo).mean(),
                      "ancho_relativo_%": ((hi - lo) / d[f"q50_{k}"]).mean() * 100, "puntaje_Winkler": winkler.mean(),
                      "cobertura_hora_min_%": ph.min(), "cobertura_hora_max_%": ph.max()})
    t6 = pd.DataFrame(filas)
    t6.to_csv(OUT / "6_bandas.csv", index=False)
    print("\n6. BANDAS (objetivo 80 %)\n" + t6.round(2).to_string(index=False))

    # 7. eventos
    med = d.groupby("dia")["real"].transform("median")
    filas = []
    for ev, cond in (("desplome", lambda s, m: s <= .7 * m), ("pico", lambda s, m: s >= 1.3 * m)):
        yev = cond(d.real, med)
        for k in ("ant", "nue"):
            mp = d.groupby("dia")[f"q50_{k}"].transform("median")
            al = cond(d[f"q50_{k}"], mp)
            filas.append({"evento": ev, "detector": f"mediana · {NOMBRES[k]}", "eventos": int(yev.sum()),
                          "detectados_%": (al & yev).sum() / yev.sum() * 100, "falsas_alarmas": int((al & ~yev).sum()),
                          "acierto_de_alertas_%": (al & yev).sum() / max(al.sum(), 1) * 100})
    t7 = pd.DataFrame(filas)
    for carpeta, nom in (("eventos_escenarios", "anterior"), ("eventos_escenarios_predespacho", "nuevo")):
        r = RES / carpeta / "metricas.csv"
        if r.exists():
            m = pd.read_csv(r)
            m = m[m["metodo"] == "clasificador"]
            for _, x in m.iterrows():
                t7 = pd.concat([t7, pd.DataFrame([{"evento": x.evento, "detector": f"clasificador · {nom}", "eventos": x.eventos,
                                                    "AUC": x.AUC, "BSS_vs_climatologia": x.BSS_vs_climatologia,
                                                    "precision_promedio": x.precision_promedio}])])
    t7.to_csv(OUT / "7_eventos.csv", index=False)
    print("\n7. EVENTOS\n" + t7.round(3).to_string(index=False))

    # 8. motor
    hist = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
    previo = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])
                        for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]
                       ).drop_duplicates("fecha_hora").set_index("fecha_hora")["precio_bolsa"].sort_index()
    filas = []
    for k, b in (("ant", ant), ("nue", nue)):
        dia = b["fecha_hora"].dt.normalize()
        rel = b["real"] - b["real"].groupby(dia).transform("mean")
        for rol in ("generador", "comercializador"):
            for intra in (False, True):
                t = md.comparar_metodos_estable(b, rol, hist, precio_previo=previo, intradia=intra)
                met, est = md.elegir_mejor_metodo_estable(t)
                f = t[t.metodo == met].iloc[0]
                s = md.generar_senales(b, met, rol, hist, precio_previo=previo, intradia=intra)
                bt = md.evaluar_backtest(b, s, rol)
                acc = s == ("vender" if rol == "generador" else "comprar")
                filas.append({"pronostico": NOMBRES[k], "rol": rol, "filtro_intradia": intra, "metodo_elegido": met,
                              "estable": est, "ventaja_año": bt["ventaja_cop_kwh"], "peor_mitad": f.peor_mitad,
                              "ventaja_intradia": (1 if rol == "generador" else -1) * rel[acc].mean(),
                              "frecuencia_%": bt["frecuencia_accion"] * 100})
    t8 = pd.DataFrame(filas)
    t8.to_csv(OUT / "8_motor.csv", index=False)
    print("\n8. MOTOR (24 h)\n" + t8.round(1).to_string(index=False))

    # figura
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    x = np.arange(len(t2))
    ax[0, 0].bar(x - .2, t2.MAE_anterior, .4, label=NOMBRES["ant"], color="#94A3B8")
    ax[0, 0].bar(x + .2, t2.MAE_nuevo, .4, label=NOMBRES["nue"], color="#D97706")
    ax[0, 0].set_xticks(x, t2.index, rotation=0)
    ax[0, 0].set_title("MAE por mes (COP/kWh)")
    ax[0, 0].legend()
    ax[0, 1].plot(t3.index, t3.MAE_anterior, color="#94A3B8", lw=2, label=NOMBRES["ant"])
    ax[0, 1].plot(t3.index, t3.MAE_nuevo, color="#D97706", lw=2, label=NOMBRES["nue"])
    ax[0, 1].set_title("MAE por hora del día")
    ax[0, 1].set_xticks(range(0, 24, 2))
    ax[0, 1].legend()
    comp = ["err_nivel_%", "err_amplitud_%", "err_forma_%"]
    izq = np.zeros(2)
    for c, col in zip(comp, ("#5B6B82", "#D97706", "#0B1F3A")):
        v = (t5[c] / 100 * t5["error_cuadratico_total"] / len(d)).to_numpy()
        ax[1, 0].barh([NOMBRES["ant"], NOMBRES["nue"]], v, left=izq, color=col, label=c.replace("err_", "").replace("_%", ""))
        izq += v
    ax[1, 0].set_title("Error cuadrático medio por día, descompuesto")
    ax[1, 0].legend()
    ej = d[(d["fecha_hora"] >= "2026-07-26") & (d["fecha_hora"] < "2026-08-01")]
    ax[1, 1].plot(ej.fecha_hora, ej.real, color="#0B1F3A", lw=1.6, label="real")
    ax[1, 1].plot(ej.fecha_hora, ej.q50_ant, color="#94A3B8", lw=1.6, label=NOMBRES["ant"])
    ax[1, 1].plot(ej.fecha_hora, ej.q50_nue, color="#D97706", lw=1.6, label=NOMBRES["nue"])
    ax[1, 1].set_title("Ejemplo: 26 al 31 de julio de 2026")
    ax[1, 1].legend(fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "comparacion_modelo_anterior.png", dpi=120)
