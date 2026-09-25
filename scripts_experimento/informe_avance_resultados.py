# -*- coding: utf-8 -*-
"""
Cifras y graficas del informe de avance (plantilla ABET, sustentacion 2026-09-25).

Todo se recalcula desde los archivos vigentes del proyecto (ya con el arreglo de la demanda del
4-5 de agosto de 2026), para que el informe no mezcle numeros de fechas distintas. No sobrescribe
ningun archivo existente: las tablas van a data/processed/resultados/informe_avance/ y las figuras
a scripts_experimento/informe_avance_assets/.
"""
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.signal import savgol_filter

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
OUT = RES / "informe_avance"
FIG = RAIZ / "scripts_experimento/informe_avance_assets"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts_experimento"))

AZUL, NARANJA, VERDE, ROJO, GRIS, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#898781", "#4a3aa7"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
                     "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
CLAVES = {}


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIG / nombre, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mae(r, p):
    return float(np.mean(np.abs(np.asarray(r, float) - np.asarray(p, float))))


def rmse(r, p):
    return float(np.sqrt(np.mean((np.asarray(r, float) - np.asarray(p, float)) ** 2)))


def mape(r, p):
    r, p = np.asarray(r, float), np.asarray(p, float)
    return float(np.mean(np.abs(r - p) / r) * 100)


def smape(r, p):
    r, p = np.asarray(r, float), np.asarray(p, float)
    return float(np.mean(np.abs(r - p) / ((np.abs(r) + np.abs(p)) / 2)) * 100)


def dm(real, p_ref, p_new, maxlags=24):
    """Diebold-Mariano con varianza HAC (Newey-West). d>0 => el modelo nuevo tiene menor error.
    Devuelve estadistico t, p bilateral y p unilateral (H1: nuevo mejor que referencia)."""
    real = np.asarray(real, float)
    d = np.abs(real - np.asarray(p_ref, float)) - np.abs(real - np.asarray(p_new, float))
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    t, p2 = float(r.tvalues[0]), float(r.pvalues[0])
    p1 = p2 / 2 if t > 0 else 1 - p2 / 2
    return t, p2, p1


# ============================================================================================
# OE1 -- caracterizacion 2019-2026
# ============================================================================================
def oe1():
    p = pd.concat([pd.read_csv(RAIZ / "data/precio_bolsa_2019_2025.csv"),
                   pd.read_csv(RAIZ / "data/precio_bolsa_2026.csv")], ignore_index=True)
    p["fecha_hora"] = pd.to_datetime(p["fecha_hora"])
    p = p.sort_values("fecha_hora").set_index("fecha_hora")["precio_bolsa_cop_kwh"]
    CLAVES["oe1_filas"] = int(len(p))
    CLAVES["oe1_rango"] = f"{p.index.min()} a {p.index.max()}"
    CLAVES["oe1_huecos"] = int((p.index.max() - p.index.min()) / pd.Timedelta(hours=1) + 1 - len(p))
    desc = p.describe(percentiles=[.05, .5, .95, .99])
    CLAVES["oe1_desc"] = {k: float(v) for k, v in desc.items()}

    # periodograma del residuo (se quita la tendencia de 30 dias, igual que el notebook 03)
    tend = p.rolling(720, min_periods=360, center=True).mean()
    res = (p - tend).dropna()
    f, pot = signal.periodogram(res.values, fs=1.0, detrend="constant")
    per = np.divide(1, f, out=np.full_like(f, np.inf), where=f != 0)
    # picos: maximos locales entre 2 h y 2000 h
    m = (per >= 2) & (per <= 2000)
    idx = signal.find_peaks(pot * m, distance=5)[0]
    top = idx[np.argsort(pot[idx])[::-1][:6]]
    picos = sorted([(float(per[i]), float(pot[i] / pot[m].sum() * 100)) for i in top], key=lambda x: -x[1])
    CLAVES["oe1_picos"] = picos

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.plot(per[m], pot[m], lw=0.6, color=AZUL)
    ax.set_xscale("log"); ax.set_yscale("log")
    for h, t in [(12, "12 h"), (24, "24 h"), (168, "168 h")]:
        ax.axvline(h, color=ROJO, ls="--", lw=0.9, alpha=0.7)
        ax.annotate(t, xy=(h, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", color=ROJO, fontsize=9)
    ax.set_xlabel("Período (horas, escala logarítmica)")
    ax.set_ylabel("Densidad espectral de potencia\n((COP/kWh)$^2$·h)")
    ax.set_title("Periodograma del precio de bolsa sin tendencia, ene-2019 a ago-2026", pad=16)
    guardar(fig, "f_periodograma.png")

    # Savitzky-Golay vs promedio movil, tramo super El Nino
    sg = pd.Series(savgol_filter(p.values, 721, 3), index=p.index)
    z = slice("2023-06-01", "2024-03-01")
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.plot(p.loc[z].index, p.loc[z].values, color=GRIS, lw=0.4, alpha=0.6, label="Precio horario")
    ax.plot(tend.loc[z].index, tend.loc[z].values, color=NARANJA, lw=2, label="Promedio móvil (30 días)")
    ax.plot(sg.loc[z].index, sg.loc[z].values, color=AZUL, lw=2, ls="--", label="Savitzky-Golay (721 h, orden 3)")
    ax.set_ylabel("Precio de bolsa (COP/kWh)")
    ax.set_xlabel("Fecha")
    ax.set_title("Filtros de tendencia en el tramo de El Niño 2023-2024")
    ax.legend(loc="upper left", frameon=False)
    guardar(fig, "f_filtros.png")

    # serie completa con regimenes ONI
    oni = pd.read_csv(RAIZ / "data/external/oni_index.csv")
    cols = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    ol = oni.melt(id_vars="anio", value_vars=cols, var_name="t", value_name="oni")
    ol["mes"] = ol["t"].map({c: i + 1 for i, c in enumerate(cols)})
    ol["fecha"] = pd.to_datetime(dict(year=ol.anio, month=ol.mes, day=15))
    ol = ol.sort_values("fecha").set_index("fecha")["oni"].loc["2019":"2026-08"]
    diario = p.resample("D").mean()
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for fch, v in ol.items():
        if v >= 0.5:
            ax.axvspan(fch - pd.Timedelta(days=15), fch + pd.Timedelta(days=15), color=NARANJA, alpha=0.13, lw=0)
        elif v <= -0.5:
            ax.axvspan(fch - pd.Timedelta(days=15), fch + pd.Timedelta(days=15), color=AZUL, alpha=0.10, lw=0)
    ax.plot(diario.index, diario.values, color="#222222", lw=0.7)
    ax.set_ylabel("Precio promedio diario (COP/kWh)")
    ax.set_xlabel("Fecha")
    ax.set_title("Precio de bolsa 2019-2026 (naranja: El Niño, ONI ≥ 0,5; azul: La Niña, ONI ≤ −0,5)")
    guardar(fig, "f_serie_oni.png")

    # correlaciones
    cc = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "volumen_embalses", "aportes_hidricos", "oni"]
    dfc = pd.concat([pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=cc),
                     pd.read_csv(RAIZ / "data/processed/dataset_maestro_2026.csv", usecols=cc)])
    cor = dfc.drop(columns="fecha_hora").corr()["precio_bolsa"].drop("precio_bolsa")
    CLAVES["oe1_corr"] = {k: float(v) for k, v in cor.items()}


# ============================================================================================
# OE2 -- modelos, holdout 2026 y walk-forward
# ============================================================================================
def oe2():
    c = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    ens = pd.read_csv(RES / "combinador_optimo_mape.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    g = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    o6 = c[c.origen == "Origen 6"].pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
    idx = ens.index
    real = ens["real"]
    preds = {m: o6[m].reindex(idx) for m in ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]}
    preds["GARCH-ged"] = g["pred"].reindex(idx)
    preds["Ensamble (6 votantes)"] = ens["pred_smape"]
    assert all(v.notna().all() for v in preds.values()), "faltan predicciones en el indice comun"
    filas = []
    for nom, pr in preds.items():
        t, p2, p1 = dm(real, preds["Persistencia"], pr) if nom != "Persistencia" else (np.nan,) * 3
        filas.append({"modelo": nom, "MAE": mae(real, pr), "RMSE": rmse(real, pr), "MAPE": mape(real, pr),
                      "sMAPE": smape(real, pr), "rMAE": mae(real, pr) / mae(real, preds["Persistencia"]),
                      "DM_t": t, "DM_p_unilateral": p1, "n": len(real)})
    tab = pd.DataFrame(filas)
    tab.to_csv(OUT / "holdout_2026_24h.csv", index=False)
    CLAVES["oe2_holdout"] = tab.round(4).to_dict("records")

    # error por hora del dia
    eh = pd.DataFrame({"h": idx.hour, "ens": np.abs(real - preds["Ensamble (6 votantes)"]),
                       "per": np.abs(real - preds["Persistencia"])}).groupby("h").mean()
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot(eh.index, eh["per"], "o-", color=GRIS, ms=3.5, label="Persistencia (t − 24 h)")
    ax.plot(eh.index, eh["ens"], "o-", color=AZUL, ms=3.5, label="Ensamble de 6 votantes")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("MAE (COP/kWh)")
    ax.set_title("Error absoluto medio por hora del día, prueba ene-ago 2026")
    ax.legend(frameon=False)
    guardar(fig, "f_error_hora.png")

    # semana ejemplo real vs pronostico con bandas N-BEATSx
    b = pd.read_csv(RES / "pronostico_con_bandas_2026_adaptativo.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    sem = slice("2026-07-06", "2026-07-19 23:00")
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.fill_between(b.loc[sem].index, b.loc[sem, "q10"], b.loc[sem, "q90"], color=AZUL, alpha=0.15,
                    label="Banda [q10, q90] de N-BEATSx calibrada")
    ax.plot(real.loc[sem].index, real.loc[sem].values, color="#111111", lw=1.2, label="Precio real")
    ax.plot(ens.loc[sem].index, ens.loc[sem, "pred_smape"], color=NARANJA, lw=1.2, label="Pronóstico 24 h (ensamble)")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    ax.set_ylabel("Precio de bolsa (COP/kWh)")
    ax.set_xlabel("Fecha (día-mes de 2026)")
    ax.set_title("Pronóstico de 24 h frente al precio real, 6 al 19 de julio de 2026")
    ax.legend(loc="upper left", frameon=False, ncol=3)
    guardar(fig, "f_real_vs_pronostico.png")

    # walk-forward 6 origenes
    wf = []
    for o, d in c.groupby("origen"):
        pv = d.pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
        rr = d.groupby("fecha_hora")["real"].first().reindex(pv.index)
        pv = pv.dropna()
        rr = rr.reindex(pv.index)
        for m in pv.columns:
            fila = {"origen": o, "modelo": m, "MAE": mae(rr, pv[m]), "MAPE": mape(rr, pv[m]),
                    "inicio": str(pv.index.min()), "fin": str(pv.index.max()), "n": len(pv)}
            if m != "Persistencia" and "Persistencia" in pv.columns:
                t, p2, p1 = dm(rr, pv["Persistencia"], pv[m])
                fila.update({"DM_t": t, "DM_p_bilateral": p2})
            wf.append(fila)
    wf = pd.DataFrame(wf)
    wf.to_csv(OUT / "walkforward_6_origenes.csv", index=False)
    piv = wf.pivot(index="origen", columns="modelo", values="MAE")
    CLAVES["oe2_wf_mae"] = piv.round(2).to_dict()
    CLAVES["oe2_wf_rangos"] = wf.groupby("origen")[["inicio", "fin", "n"]].first().to_dict("index")
    resumen = []
    for m in piv.columns:
        v = piv[m]
        sub = wf[(wf.modelo == m)]
        gana = int(((sub["DM_t"] > 0) & (sub["DM_p_bilateral"] < 0.05)).sum()) if "DM_t" in sub else 0
        pierde = int(((sub["DM_t"] < 0) & (sub["DM_p_bilateral"] < 0.05)).sum()) if "DM_t" in sub else 0
        resumen.append({"modelo": m, "MAE_medio": v.mean(), "CV": v.std() / v.mean(),
                        "rMAE_medio": (v / piv["Persistencia"]).mean(), "gana_sig": gana, "pierde_sig": pierde})
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(OUT / "walkforward_resumen.csv", index=False)
    CLAVES["oe2_wf_resumen"] = resumen.round(3).to_dict("records")

    orden = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]
    orden = [m for m in orden if m in piv.columns]
    colores = [GRIS, NARANJA, VIOLETA, AZUL, VERDE]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(piv.index))
    w = 0.16
    for i, m in enumerate(orden):
        ax.bar(x + (i - 2) * w, piv[m].values, w, label=m, color=colores[i])
    etiquetas = {"Origen 1": "O1 · 2020-21\nLa Niña", "Origen 2": "O2 · 2021-22\nLa Niña",
                 "Origen 3": "O3 · 2022-23\nLa Niña → El Niño", "Origen 4": "O4 · 2023-24\nEl Niño fuerte",
                 "Origen 5": "O5 · 2024-25\nneutro", "Origen 6": "O6 · 2026\nneutro → El Niño"}
    ax.set_xticks(x)
    ax.set_xticklabels([etiquetas.get(o, o) for o in piv.index], fontsize=9)
    ax.set_ylabel("MAE (COP/kWh)")
    ax.set_title("Validación walk-forward: MAE por origen temporal y régimen ENSO")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    guardar(fig, "f_walkforward.png")

    # 72h: producto completo armado igual que puente_24h_a_72h.py (pasos 1-23 = puente con el
    # ensamble de 24h; el resto = combinador LAD por tramo sobre los 4 votantes de 72h)
    from puente_24h_a_72h import cargar_datos, lad_cv, NUEVO_72
    from ensamble_72h_diario import TRAMOS
    v, _, _ = cargar_datos()
    pz = pd.read_csv(RES / "pronostico_puente_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    llaves = ["cutoff", "paso_horas"]
    resto = v[~v.paso_horas.between(1, 23) | ~v.set_index(llaves).index.isin(pz.set_index(llaves).index)].copy()
    resto = resto.dropna(subset=NUEVO_72 + ["real"])
    trozos = []
    for nom, a, bb in TRAMOS:
        s = resto[resto.paso_horas.between(a, bb)].sort_values(llaves).reset_index(drop=True)
        if len(s):
            trozos.append(pd.DataFrame({"fecha_hora": s.fecha_hora, "cutoff": s.cutoff, "paso_horas": s.paso_horas,
                                        "real": s.real, "pred": lad_cv(s, NUEVO_72, "lad")}))
    trozos.append(pd.DataFrame({"fecha_hora": pz.fecha_hora, "cutoff": pz.cutoff, "paso_horas": pz.paso_horas,
                                "real": pz.real, "pred": pz.pred_puente}))
    full = pd.concat(trozos, ignore_index=True).dropna()
    # persistencia honesta desde el corte: misma hora, ultimo dia observado en o antes del corte
    pr = pd.concat([pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["fecha_hora", "precio_bolsa"]),
                    pd.read_csv(RAIZ / "data/processed/dataset_maestro_2026.csv", usecols=["fecha_hora", "precio_bolsa"])])
    pr["fecha_hora"] = pd.to_datetime(pr["fecha_hora"])
    pr = pr.set_index("fecha_hora")["precio_bolsa"].sort_index()
    horas = (full.fecha_hora - full.cutoff) / pd.Timedelta(hours=1)
    lag = np.ceil(horas / 24).astype(int) * 24
    full["naive"] = pr.reindex(full.fecha_hora - pd.to_timedelta(lag, unit="h")).to_numpy()
    full = full.dropna()
    full.to_csv(OUT / "producto_72h_reconstruido.csv", index=False)
    tr = []
    for nom, (a, bb) in {"1-24 h": (1, 24), "25-48 h": (25, 48), "49-72 h": (49, 72), "Global 1-72 h": (1, 72)}.items():
        s = full[full.paso_horas.between(a, bb)]
        t, p2, p1 = dm(s.real, s.naive, s.pred, maxlags=72)
        tr.append({"tramo": nom, "MAE": mae(s.real, s.pred), "RMSE": rmse(s.real, s.pred),
                   "MAPE": mape(s.real, s.pred), "sMAPE": smape(s.real, s.pred),
                   "MAE_persistencia": mae(s.real, s.naive), "MAPE_persistencia": mape(s.real, s.naive),
                   "rMAE": mae(s.real, s.pred) / mae(s.real, s.naive), "DM_t": t, "DM_p_unilateral": p1, "n": len(s)})
    tr = pd.DataFrame(tr)
    tr.to_csv(OUT / "ensamble_72h_tramos.csv", index=False)
    CLAVES["oe2_72h"] = tr.round(4).to_dict("records")
    pz = full.sort_values("fecha_hora").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    t3 = tr.iloc[:3]
    barras = ax.bar(t3["tramo"], t3["MAPE"], color=[AZUL, "#5b9be0", "#9cc2ef"])
    for bb_, v in zip(barras, t3["MAPE"]):
        ax.text(bb_.get_x() + bb_.get_width() / 2, v + 0.3, f"{v:.1f} %", ha="center", fontsize=10)
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Tramo del horizonte de pronóstico")
    ax.set_title("Error del pronóstico de 72 h por tramo, prueba 2026")
    guardar(fig, "f_72h_tramos.png")

    # literatura: MASE (naive-24h en muestra, train 2019-2025), NRMSE, sMAPE conv. Kapoor, TAPI diario
    from comparacion_4_papers import metricas
    feat = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    ptr = feat.loc[feat.index < pd.Timestamp("2026-01-01"), "precio_bolsa"].dropna()
    d1, d24 = float(np.abs(ptr.diff()).dropna().mean()), float(np.abs(ptr.diff(24)).dropna().mean())
    lit = metricas(idx, real, preds["Ensamble (6 votantes)"], d1, d24)
    lit72 = metricas(pz["fecha_hora"], pz["real"], pz["pred"], d1, d24)
    CLAVES["oe2_literatura_24h"] = lit
    CLAVES["oe2_literatura_72h"] = lit72

    # bandas
    b72 = pd.read_csv(RES / "pronostico_con_bandas_72h_2026_adaptativo.csv")
    cob = {"24h": float(((b.real >= b.q10) & (b.real <= b.q90)).mean() * 100),
           "24h_ancho": float((b.q90 - b.q10).mean())}
    for nom, (a, bb) in {"72h_1-24": (1, 24), "72h_25-48": (25, 48), "72h_49-72": (49, 72), "72h": (1, 72)}.items():
        s = b72[(b72.paso_horas >= a) & (b72.paso_horas <= bb)]
        cob[nom] = float(((s.real >= s.q10) & (s.real <= s.q90)).mean() * 100)
        cob[nom + "_ancho"] = float((s.q90 - s.q10).mean())
    CLAVES["oe2_bandas"] = cob

    # hora de lanzamiento
    lz = pd.read_csv(RES / "pronostico_movil_24_horas_lanzamiento.csv")
    CLAVES["oe2_lanzamiento_lear"] = {"MAE_00": float(lz.loc[lz.hora_lanzamiento == 0, "MAE"].iloc[0]),
                                     "MAE_23": float(lz.loc[lz.hora_lanzamiento == 23, "MAE"].iloc[0]),
                                     "MAE_prom": float(lz["MAE"].mean())}
    ph = pd.read_csv(RES / "ensamble_protocolo_honesto.csv", parse_dates=["fecha_hora"])
    CLAVES["oe2_ens_corte23"] = {"MAE": mae(ph.real, ph.pred_B), "MAPE": mape(ph.real, ph.pred_B)}


# ============================================================================================
# OE3 / OE4 -- motor de decision (solo lectura de src/motor_decision.py)
# ============================================================================================
def oe3():
    from motor_decision import cargar_fuente_pronostico, comparar_metodos, elegir_mejor_metodo, generar_senales
    hist = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
    filas, bandas, gan = [], {}, {}
    for hz in ["24h", "72h"]:
        df, meta = cargar_fuente_pronostico(RAIZ, hz)
        bandas[hz] = df
        for rol in ["generador", "comercializador"]:
            t = comparar_metodos(df, rol, hist)
            met, _ = elegir_mejor_metodo(t)
            gan[(hz, rol)] = met
            for _, r in t.iterrows():
                filas.append({"horizonte": hz, "rol": rol, **r.to_dict(), "ganador": r["metodo"] == met})
    tb = pd.DataFrame(filas)
    tb.to_csv(OUT / "motor_backtest.csv", index=False)
    CLAVES["oe3_backtest"] = tb.round(3).to_dict("records")
    CLAVES["oe3_precio_medio_2026"] = float(bandas["24h"]["real"].mean())

    df = bandas["24h"]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True)
    colores = {"comprar": VERDE, "vender": VERDE, "evitar_compra": ROJO, "retener": ROJO}
    nombres = {"comprar": "Comprar", "vender": "Vender", "evitar_compra": "Evitar compra", "retener": "Retener"}
    for ax, rol in zip(axes, ["comercializador", "generador"]):
        s = generar_senales(df, gan[("24h", rol)], rol, hist)
        d = df.assign(senal=s.values)
        d = d[d["fecha_hora"] >= d["fecha_hora"].max() - pd.Timedelta(days=14)]
        ax.fill_between(d.fecha_hora, d.q10, d.q90, color=AZUL, alpha=0.15, label="Banda [q10, q90]")
        ax.plot(d.fecha_hora, d.real, color="#111111", lw=1, label="Precio real")
        for sv, col in colores.items():
            q = d[d.senal == sv]
            if len(q):
                ax.scatter(q.fecha_hora, q.real, s=12, color=col, zorder=3, label=nombres[sv])
        ax.set_ylabel("COP/kWh")
        ax.set_title(f"Rol {rol} (método de umbral: {gan[('24h', rol)]})", fontsize=11)
        ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8.5)
    import matplotlib.dates as mdates
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    axes[1].set_xlabel("Fecha (día-mes de 2026)")
    guardar(fig, "f_motor_senales.png")


def desplegable():
    """Ensamble de 24h con pesos causales (solo dias anteriores, ventana expansiva): la cifra que
    tendria el sistema operando en tiempo real, sin validacion cruzada dentro de 2026."""
    from stacking_24h_v2 import cargar as cargar24, BASE5
    from franja_medianoche_24h import causal
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred"].rename("GARCH-ged"), how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    cols = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=cols + ["real"])
    p = causal(d, cols, "g4", "smape")
    ok = ~np.isnan(p)
    ens = pd.read_csv(RES / "combinador_optimo_mape.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    r = d["real"].to_numpy()[ok]
    pcv = ens["pred_smape"].reindex(d.index).to_numpy()[ok]
    per = d["Persistencia"].to_numpy()[ok]
    t, p2, p1 = dm(r, per, p[ok])
    CLAVES["oe2_desplegable"] = {"n": int(ok.sum()), "desde": str(d.index[ok].min()),
                                 "MAE": mae(r, p[ok]), "MAPE": mape(r, p[ok]), "sMAPE": smape(r, p[ok]),
                                 "MAE_cv_mismas_horas": mae(r, pcv), "MAPE_cv_mismas_horas": mape(r, pcv),
                                 "MAE_persistencia_mismas_horas": mae(r, per),
                                 "MAPE_persistencia_mismas_horas": mape(r, per),
                                 "DM_vs_persistencia": [t, p2, p1]}
    pd.DataFrame({"fecha_hora": d.index[ok], "real": r, "pred_desplegable": p[ok]}).to_csv(
        OUT / "ensamble_24h_desplegable.csv", index=False)


if __name__ == "__main__":
    oe1()
    oe2()
    oe3()
    desplegable()
    with open(OUT / "claves_informe.json", "w", encoding="utf-8") as fh:
        json.dump(CLAVES, fh, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(CLAVES, ensure_ascii=False, indent=1, default=str))
