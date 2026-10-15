# -*- coding: utf-8 -*-
"""
Cifras y figuras del informe final (ABET), 15-oct-2026. Todo sale de los archivos del proyecto y del codigo real del
motor (src/motor_decision.py), sin numeros escritos a mano. Lo lee generar_informe_final.py.

Periodo oficial (Anexo 1): prueba de 2026 hasta el 5-ago-2026; las comparaciones entre modelos usan las mismas horas
(15-ene a 5-ago, despues del calentamiento de 14 dias del combinador). Escenario realista: solo lo publicado a las 00:00
del dia D (precio hasta las 23:00 de D-4). Persistencia realista: el precio publicado mas reciente de la misma hora
(hace 96 h; 120 h para la hora 00:00). Operacion: del 6-ago al ultimo precio publicado (data/produccion/).
Salida: scripts_experimento/informe_final_assets/cifras.json y figuras .png
"""
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(RAIZ / "src"))
import realista_comun as rc  # noqa: E402
from motor_decision import (  # noqa: E402
    comparar_metodos, comparar_metodos_estable, elegir_mejor_metodo_estable, evaluar_backtest, generar_senales,
    precio_previo_ventana,
)

RES = RAIZ / "data" / "processed" / "resultados"
OUT = AQUI / "informe_final_assets"
OUT.mkdir(exist_ok=True)
FIN_OFICIAL = pd.Timestamp("2026-08-05 00:00")
INI_COMUN = pd.Timestamp("2026-01-15 01:00")
AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
                     "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
R = {}


def metr(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())), "MAPE": float(np.mean(np.abs(e) / y) * 100),
            "sMAPE": float(np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p))) * 100), "sesgo": float(e.mean())}


def dm(y, a, b, lags=24):
    """t > 0: a tiene MENOR error absoluto que b (convencion de la Tabla 14 del informe de avance)."""
    d = np.abs(np.asarray(b) - np.asarray(y)) - np.abs(np.asarray(a) - np.asarray(y))
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": lags - 1})
    return float(r.tvalues[0]), float(r.pvalues[0] / 2 if r.tvalues[0] > 0 else 1 - r.pvalues[0] / 2)


def boot_media(x, bloque=7, n=4000, semilla=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(semilla)
    k = int(np.ceil(len(x) / bloque))
    medias = []
    for _ in range(n):
        ini = rng.integers(0, len(x) - bloque + 1, k)
        medias.append(np.concatenate([x[i:i + bloque] for i in ini])[:len(x)].mean())
    return [float(np.quantile(medias, 0.025)), float(np.quantile(medias, 0.975))]


# ------------------------------------------------------------------------------------------------------------------
# 1) Pronostico de 24 h en el escenario realista: modelos y disenos alternativos en las mismas horas
# ------------------------------------------------------------------------------------------------------------------
t0 = time.time()
tab = rc.tabla_realista().set_index("fecha_hora")
pers_real = tab["precio_lag24h"]                       # realista: 96 h (120 h para la hora 00:00)
pers_opt = tab["precio_bolsa"].shift(24)               # la del informe de avance (precio de ayer, no publico al corte)
esc = pd.read_csv(RES / "escenario_realista" / "predicciones.csv", parse_dates=[0], index_col=0)
prod = pd.read_csv(RES / "ensamble_24h_realista_2026.csv", parse_dates=["fecha_hora", "dia"]).set_index("fecha_hora")
idx = esc.index[(esc.index >= INI_COMUN) & (esc.index <= FIN_OFICIAL)].intersection(prod.index)
y = prod.loc[idx, "real"]
modelos = {
    "Persistencia (precio publicado más reciente)": pers_real.reindex(idx),
    "XGBoost": esc.loc[idx, "XGBoost"], "ARX+GARCH": esc.loc[idx, "ARX+GARCH"], "N-BEATSx": esc.loc[idx, "N-BEATSx"],
    "N-HiTS": esc.loc[idx, "N-HiTS"], "GARCH-ged": esc.loc[idx, "GARCH-ged"],
    "Ensamble de 6 (diseño del informe de avance)": esc.loc[idx, "ens6_realista"],
    "Predespacho ideal de XM": prod.loc[idx, "Predespacho"],
    "Ensamble de 6 + predespacho": esc.loc[idx, "ens7_realista"],
    "Promedio de 3 CatBoost": prod.loc[idx, "Cat_prom"],
    "Predespacho + 3 CatBoost (diseño definitivo)": prod.loc[idx, "pred_realista"],
}
ref = modelos["Persistencia (precio publicado más reciente)"]
tabla24 = []
for nom, p in modelos.items():
    f = {"modelo": nom, **metr(y, p), "rMAE": float(np.abs(p - y).mean() / np.abs(ref - y).mean())}
    f["DM_t"], f["DM_p1"] = (None, None) if p is ref else dm(y, p, ref)
    tabla24.append(f)
R["tabla24"] = tabla24
R["horas_comunes"] = int(len(idx))
R["desde_comun"], R["hasta_comun"] = str(idx.min()), str(idx.max())
R["persistencia_optimista_mae"] = float(np.abs(pers_opt.reindex(idx) - y).mean())
R["ens6_optimista_mae"] = float(np.abs(esc.loc[idx, "ens6_actual"] - y).mean()) if "ens6_actual" in esc else None
R["ens7_optimista_mae"] = float(np.abs(esc.loc[idx, "ens7_actual"] - y).mean()) if "ens7_actual" in esc else None
fin = prod["pred_realista"]
R["dm_definitivo_vs_ens7_realista"] = dm(y, fin.loc[idx], esc.loc[idx, "ens7_realista"])
R["dm_definitivo_vs_predespacho"] = dm(y, fin.loc[idx], prod.loc[idx, "Predespacho"])
# contrato completo (1-ene a 5-ago) y bandas
c24 = pd.read_csv(RES / "pronostico_realista_bandas_24h_2026.csv", parse_dates=["fecha_hora"])
c24o = c24[(c24["fecha_hora"] <= FIN_OFICIAL)]
R["contrato24"] = {"horas": int(len(c24o)), **metr(c24o["real"], c24o["q50"]),
                   "cobertura": float(((c24o.real >= c24o.q10) & (c24o.real <= c24o.q90)).mean() * 100),
                   "ancho": float((c24o.q90 - c24o.q10).mean())}
pc = pers_real.reindex(c24o["fecha_hora"]).to_numpy()
prc = prod["Predespacho"].reindex(c24o["fecha_hora"]).to_numpy()
R["contrato24"]["MAE_pers"] = float(np.nanmean(np.abs(pc - c24o["real"].to_numpy())))
R["contrato24"]["MAE_pre"] = float(np.nanmean(np.abs(prc - c24o["real"].to_numpy())))
R["contrato24"]["DM_vs_pers"] = dm(c24o["real"].to_numpy(), c24o["q50"].to_numpy(), pc)
R["contrato24"]["DM_vs_pre"] = dm(c24o["real"].to_numpy(), c24o["q50"].to_numpy(), prc)
# matriz de decision: peor rMAE mensual de cada alternativa frente a ambas persistencias
alternativas = {"S1": esc.loc[idx, "ens6_realista"], "S2": esc.loc[idx, "ens7_realista"], "S3": prod.loc[idx, "pred_realista"]}
po = pers_opt.reindex(idx)
mes_idx = idx.to_period("M")
R["alternativas"] = {}
for k_, p in alternativas.items():
    e = np.abs(p - y)
    R["alternativas"][k_] = {"MAE": float(e.mean()), "rMAE_opt": float(e.mean() / np.abs(po - y).mean()),
                             "rMAE_real": float(e.mean() / np.abs(ref - y).mean()),
                             "peor_mes_opt": float((e.groupby(mes_idx).mean() / np.abs(po - y).groupby(mes_idx).mean()).max()),
                             "peor_mes_real": float((e.groupby(mes_idx).mean() / np.abs(ref - y).groupby(mes_idx).mean()).max())}
# robustez mensual (rMAE frente a la persistencia realista) y regimenes
q = pd.DataFrame({"y": y, "p": fin.loc[idx], "r": ref, "pre": prod.loc[idx, "Predespacho"]})
q["mes"] = q.index.to_period("M").astype(str)
R["rmae_mensual"] = {m: float(np.abs(g.p - g.y).mean() / np.abs(g.r - g.y).mean()) for m, g in q.groupby("mes")}
R["mae_mensual"] = {m: float(np.abs(g.p - g.y).mean()) for m, g in q.groupby("mes")}
for nom, sel in (("normal_ene_abr", q.index < "2026-05-01"), ("el_nino_may_ago", q.index >= "2026-05-01")):
    g = q[sel]
    R[f"regimen_{nom}"] = {"MAE": float(np.abs(g.p - g.y).mean()), "MAE_pre": float(np.abs(g.pre - g.y).mean()),
                           "MAE_pers": float(np.abs(g.r - g.y).mean()), "DM_vs_pre": dm(g.y, g.p, g.pre)}
# error diario con intervalos por bootstrap de bloques de 7 dias
q["dia"] = (q.index - pd.Timedelta(hours=1)).normalize()
dd = q.groupby("dia").apply(lambda g: pd.Series({"mae": np.abs(g.p - g.y).mean(), "mae_r": np.abs(g.r - g.y).mean(),
                                                 "mape": (np.abs(g.p - g.y) / g.y).mean() * 100, "sesgo": (g.p - g.y).mean()}))
R["diario"] = {"dias": int(len(dd)), "media": float(dd.mae.mean()), "desv": float(dd.mae.std()),
               "ic": boot_media(dd.mae), "ic_pers": boot_media(dd.mae_r), "media_pers": float(dd.mae_r.mean()),
               "dif_media": float((dd.mae_r - dd.mae).mean()), "ic_dif": boot_media(dd.mae_r - dd.mae),
               "frac_mejor": float((dd.mae < dd.mae_r).mean() * 100), "p10": float(dd.mae.quantile(0.1)),
               "p90": float(dd.mae.quantile(0.9)), "menor50": float((dd.mae < 50).mean() * 100),
               "menor50_pers": float((dd.mae_r < 50).mean() * 100), "mape_media": float(dd.mape.mean()),
               "ic_mape": boot_media(dd.mape), "sesgo": float(dd.sesgo.mean()), "ic_sesgo": boot_media(dd.sesgo),
               "autocorr": float(dd.mae.autocorr())}
print(f"24 h listo ({(time.time() - t0) / 60:.1f} min)", flush=True)

# ------------------------------------------------------------------------------------------------------------------
# 2) 72 h realista (contrato de cortes cada 3 dias del periodo oficial) frente a la persistencia realista del corte
# ------------------------------------------------------------------------------------------------------------------
c72 = pd.read_csv(RES / "pronostico_realista_bandas_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
c72 = c72[c72["cutoff"] <= FIN_OFICIAL.normalize() - pd.Timedelta(days=3)]
k = (c72["paso_horas"] - 1) // 24
c72["pers"] = pers_real.reindex(c72["fecha_hora"] - pd.to_timedelta(24 * k, unit="h")).to_numpy()
tramos = []
for a, b in ((1, 24), (25, 48), (49, 72), (1, 72)):
    s = c72[c72.paso_horas.between(a, b)].dropna(subset=["pers", "real"])
    m, mp = metr(s.real, s.q50), metr(s.real, s.pers)
    tramos.append({"tramo": f"{a}-{b} h", "MAE": m["MAE"], "MAPE": m["MAPE"], "MAE_pers": mp["MAE"], "MAPE_pers": mp["MAPE"],
                   "rMAE": m["MAE"] / mp["MAE"], "DM_t": dm(s.real, s.q50, s.pers)[0],
                   "cobertura": float(((s.real >= s.q10) & (s.real <= s.q90)).mean() * 100), "ancho": float((s.q90 - s.q10).mean()),
                   "horas": int(len(s))})
R["tabla72"] = tramos
R["cortes72"] = int(c72["cutoff"].nunique())

# ------------------------------------------------------------------------------------------------------------------
# 3) Operacion del 6-ago al ultimo precio publicado (pronostico ya publicado, datos de data/produccion/)
# ------------------------------------------------------------------------------------------------------------------
op = c24[(c24["fecha_hora"] > FIN_OFICIAL) & c24["real"].notna()]
pre_op = prod["Predespacho"].reindex(op["fecha_hora"]).to_numpy()
R["operacion"] = {"desde": str(op.fecha_hora.min()), "hasta": str(op.fecha_hora.max()), "horas": int(len(op)), **metr(op.real, op.q50),
                  "MAE_pre": float(np.nanmean(np.abs(pre_op - op.real.to_numpy()))),
                  "cobertura": float(((op.real >= op.q10) & (op.real <= op.q90)).mean() * 100),
                  "mensual": {str(m): float(np.abs(g.q50 - g.real).mean()) for m, g in op.groupby(op.fecha_hora.dt.to_period("M"))}}
try:
    R["estado_produccion"] = json.loads((RES / "produccion" / "estado.json").read_text(encoding="utf-8"))
except OSError:
    R["estado_produccion"] = None

# ------------------------------------------------------------------------------------------------------------------
# 4) Eventos (detector realista, mismas 5.184 h) e historicos
# ------------------------------------------------------------------------------------------------------------------
ev = pd.read_csv(RES / "escenario_realista" / "mejoras" / "eventos_realista_produccion.csv")
R["eventos"] = ev.to_dict(orient="records")
crudas = pd.read_csv(RES / "walkforward_predicciones_crudas.csv")
R["walkforward_mae"] = {f"{o}|{m}": float(np.abs(g.real - g.prediccion).mean()) for (o, m), g in crudas.groupby(["origen", "modelo"])}
dmw = pd.read_csv(RES / "diebold_mariano.csv")
R["walkforward_dm"] = {f"{m1}|{m2}": {"gana": int(g.conclusion.str.contains("GANA").sum()), "pierde": int(g.conclusion.str.contains("PIERDE").sum())}
                       for (m1, m2), g in dmw.groupby(["modelo_1", "modelo_2"])}
R["ensamble_historico"] = pd.read_csv(RES / "ensamble_origenes_historicos.csv").to_dict(orient="records")
R["validacion_predespacho"] = pd.read_csv(RES / "validacion_historica_predespacho.csv").to_dict(orient="records")
v2 = RES / "escenario_realista" / "mejoras" / "validacion_historica_v2.csv"
R["validacion_definitivo"] = pd.read_csv(v2).to_dict(orient="records")
R["validacion_definitivo_fecha"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(v2.stat().st_mtime))
# persistencia realista en las mismas horas de cada origen historico (ventanas 01:00-00:00, desde el 15 del primer mes)
pers_hist = {}
for o, (a, b) in {"Origen 1": ("2020-07-15 01:00", "2021-07-01 00:00"), "Origen 2": ("2021-07-15 01:00", "2022-07-01 00:00"),
                  "Origen 3": ("2022-07-15 01:00", "2023-07-01 00:00"), "Origen 4": ("2023-07-15 01:00", "2024-07-01 00:00"),
                  "Origen 5": ("2024-07-15 01:00", "2025-07-01 00:00")}.items():
    s = tab.loc[a:b, ["precio_bolsa", "precio_lag24h"]].dropna()
    pers_hist[o] = float(np.abs(s.precio_bolsa - s.precio_lag24h).mean())
R["persistencia_realista_historica"] = pers_hist
mb = pd.read_csv(RES / "motor_backtest_historico.csv")
col = [c for c in mb.columns if c.startswith("elegida_por_estabilidad")][0]
R["motor_historico_elegidos"] = mb[mb[col] == True][["origen", "rol", "politica", "ventaja", "peor_mitad"]].to_dict(orient="records")  # noqa: E712
print(f"historicos listos ({(time.time() - t0) / 60:.1f} min)", flush=True)

# ------------------------------------------------------------------------------------------------------------------
# 5) Motor de decision (como lo usa el dashboard: calentamiento con el precio previo y filtro intradia)
# ------------------------------------------------------------------------------------------------------------------
HIST = pd.read_csv(RAIZ / "data" / "processed" / "dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
PREVIO = pd.concat([pd.read_csv(RAIZ / "data" / "processed" / f, usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])
                    for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]).drop_duplicates("fecha_hora")
PREVIO = PREVIO.set_index("fecha_hora")["precio_bolsa"].sort_index()
bandas = {"24h": c24o[["fecha_hora", "real", "q10", "q50", "q90"]].reset_index(drop=True),
          "72h": c72[["fecha_hora", "real", "q10", "q50", "q90"]].sort_values("fecha_hora").reset_index(drop=True)}
R["motor"] = {"precio_medio": float(bandas["24h"]["real"].mean())}
for h, b in bandas.items():
    for rol in ("generador", "comercializador"):
        tp = comparar_metodos(b, rol, HIST, precio_previo=PREVIO, intradia=True).set_index("metodo")
        te = comparar_metodos_estable(b, rol, HIST, precio_previo=PREVIO, intradia=True)
        m_est, estable = elegir_mejor_metodo_estable(te)
        te = te.set_index("metodo")
        sin = comparar_metodos_estable(b, rol, HIST, precio_previo=PREVIO, intradia=False)
        m_sin, _ = elegir_mejor_metodo_estable(sin)
        sin = sin.set_index("metodo")
        nan = lambda v: None if pd.isna(v) else float(v)  # noqa: E731
        R["motor"][f"{h}|{rol}"] = {
            "metodo_estable": m_est, "es_estable": bool(estable), "metodo_sin_filtro": m_sin,
            "peor_mitad_sin_filtro": nan(min(sin.loc[m_sin, "ventaja_H1"], sin.loc[m_sin, "ventaja_H2"])),
            "metodos": {m: {"ventaja_anio": nan(tp.loc[m, "ventaja_cop_kwh"]), "frec_anio": float(tp.loc[m, "frecuencia_accion"]),
                            "ventaja_h1": nan(te.loc[m, "ventaja_H1"]), "ventaja_h2": nan(te.loc[m, "ventaja_H2"]),
                            "frec_h1": float(te.loc[m, "frecuencia_H1"]), "frec_h2": float(te.loc[m, "frecuencia_H2"]),
                            "estable": bool(te.loc[m, "estable"])} for m in tp.index}}
b24 = bandas["24h"]
R["motor"]["corte_mitades"] = str((b24.fecha_hora.min() + (b24.fecha_hora.max() - b24.fecha_hora.min()) / 2).date())
metodos = ["fijo", "rodante", "banda", "hibrido"]
caso = R["motor"]["24h|generador"]["metodos"]
x = np.arange(4)
fig, ax = plt.subplots(figsize=(8.2, 3.4))
b1 = ax.bar(x - 0.19, [caso[m]["frec_h1"] * 100 for m in metodos], 0.36, color="#9cc2ef", label=f"1.ª mitad (hasta {R['motor']['corte_mitades']})")
b2 = ax.bar(x + 0.19, [caso[m]["frec_h2"] * 100 for m in metodos], 0.36, color=AZUL, label="2.ª mitad")
ax.axhspan(10, 40, color=VERDE, alpha=0.10, lw=0)
ax.text(3.55, 41.5, "rango válido 10-40 %", color=VERDE, fontsize=9, ha="right")
for bars in (b1, b2):
    for bb in bars:
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 1.0, f"{bb.get_height():.0f} %", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["fijo", "rodante", "banda", "híbrido"])
ax.set_ylabel("Horas con acción (%)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "frecuencia_mitades.png")
plt.close(fig)
# barrido de percentiles con el metodo elegido para el generador a 24 h
m_g = R["motor"]["24h|generador"]["metodo_estable"]
barrido = []
for pb, pa in [(10, 90), (15, 85), (20, 80), (25, 75), (30, 70), (35, 65)]:
    s = generar_senales(b24, m_g, "generador", HIST, p_bajo=pb, p_alto=pa, precio_previo=PREVIO, intradia=True)
    bt = evaluar_backtest(b24, s, "generador")
    barrido.append({"par": f"{pb}/{pa}", "ventaja": float(bt["ventaja_cop_kwh"]), "frec": float(bt["frecuencia_accion"] * 100)})
R["motor"]["barrido"] = {"metodo": m_g, "pares": barrido}
fig, ax = plt.subplots(figsize=(8.2, 3.4))
ax.plot([r["frec"] for r in barrido], [r["ventaja"] for r in barrido], color=GRIS, lw=1.2, zorder=1)
for r in barrido:
    act = r["par"] == "25/75"
    ax.scatter(r["frec"], r["ventaja"], s=70 if act else 40, color=AZUL if act else GRIS, zorder=2)
    ax.annotate(r["par"] + (" (actual)" if act else ""), (r["frec"], r["ventaja"]), textcoords="offset points", xytext=(6, 6),
                fontsize=9, color=AZUL if act else "#52514e", fontweight="bold" if act else "normal")
ax.set_xlabel("Horas con acción (%)")
ax.set_ylabel("Ventaja (COP/kWh)")
fig.tight_layout()
fig.savefig(OUT / "barrido_percentiles.png")
plt.close(fig)
# ultimo mes del periodo oficial, en pesos, para un cliente de 100 kW
fin24 = b24.fecha_hora.max()
mask = b24.fecha_hora > fin24 - pd.Timedelta(days=30)
sub = b24[mask]
um = {"inicio": str((fin24 - pd.Timedelta(days=30) + pd.Timedelta(hours=1)).date()), "fin": str(fin24.date()),
      "cobertura": float(((sub.real >= sub.q10) & (sub.real <= sub.q90)).mean() * 100), "mae": float((sub.real - sub.q50).abs().mean()),
      "metodos": {}}
for m in metodos:
    s = generar_senales(b24, m, "generador", HIST, precio_previo=PREVIO, intradia=True)
    bt = evaluar_backtest(b24[mask], s[mask], "generador")
    v = bt["ventaja_cop_kwh"]
    um["metodos"][m] = {"ventaja": None if np.isnan(v) else float(v), "frec": float(bt["frecuencia_accion"] * 100),
                        "horas": int(bt["horas_accion"]), "cop": 0.0 if np.isnan(v) else float(v * bt["horas_accion"] * 100)}
R["motor"]["ultimo_mes"] = um
fig, ax = plt.subplots(figsize=(8.2, 3.4))
vals = [um["metodos"][m]["cop"] / 1e6 for m in metodos]
bars = ax.bar(["fijo", "rodante", "banda", "híbrido"], vals, color=[GRIS, AZUL, NARANJA, VERDE], width=0.55)
ax.axhline(0, color="#222222", lw=0.8)
rg = max(max(vals), 0) - min(min(vals), 0) or 1
for bb, v in zip(bars, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, v + (rg * 0.04 if v >= 0 else -rg * 0.04), f"{v:+.2f} M COP".replace(".", ","),
            ha="center", va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
ax.set_ylabel("Millones de COP")
fig.tight_layout()
fig.savefig(OUT / "ultimo_mes_pesos.png")
plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------
# 6) Costo computacional del diseno definitivo (en este equipo): un CatBoost y la actualizacion diaria
# ------------------------------------------------------------------------------------------------------------------
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN, catboost  # noqa: E402
d = tab.copy()
cols = conjuntos(d)["R+P"]
mes = pd.Timestamp("2026-08-01")
tr = d.loc[INI_TR:mes - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
te = d.loc[mes + pd.Timedelta(hours=1):FIN_OFICIAL].dropna(subset=["pre_obj"])
t1 = time.time()
catboost(tr, te, cols)
R["costo"] = {"catboost_s": time.time() - t1, "filas_entrenamiento": int(len(tr)), "variables": len(cols)}
(OUT / "cifras.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
print(f"Listo en {(time.time() - t0) / 60:.1f} min -> {OUT / 'cifras.json'}")
