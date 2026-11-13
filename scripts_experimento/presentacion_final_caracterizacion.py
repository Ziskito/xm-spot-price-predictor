# -*- coding: utf-8 -*-
"""
Figuras de caracterizacion (OE1) para la PRESENTACION FINAL, con la identidad visual de Augur.

Mismo analisis y mismos datos de los notebooks de Juan (03_caracterizacion_precio_juan.ipynb y
03b_caracterizacion_variables_juan.ipynb); solo cambian los colores, la tipografia y el tamano, para que las figuras
coincidan con la pagina web y se lean proyectadas (Lexend/IBM Plex Sans y los colores de .streamlit/config.toml).

Usa el MISMO periodo oficial del informe final (1-ene-2019 a 5-ago-2026, dataset maestro, 66.576 h) que
informe_final_caracterizacion.py, para que las cifras de las diapositivas coincidan con las del informe.

Salida: scripts_experimento/presentacion_final_assets/ (figuras F1 a F7, A1 y A2, y cifras_caracterizacion.json).
Uso: python scripts_experimento/presentacion_final_caracterizacion.py
"""
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy import signal  # noqa: E402
from scipy.signal import butter, filtfilt, savgol_filter  # noqa: E402

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
OUT = AQUI / "presentacion_final_assets"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- identidad de Augur (.streamlit/config.toml)
INK, AMBER, BLUE, GREEN, RUST = "#0B1F3A", "#D97706", "#1E3A8A", "#047857", "#C2410C"
MUTED, GRAY, LINE = "#5B6B82", "#8193AD", "#DDE3EC"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["IBM Plex Sans", "Segoe UI", "Calibri", "DejaVu Sans"],
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.labelsize": 10.5, "axes.labelcolor": MUTED, "axes.edgecolor": LINE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": LINE, "grid.alpha": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "legend.fontsize": 9.5,
    "lines.linewidth": 1.6, "figure.facecolor": "white", "savefig.facecolor": "white",
})
CIFRAS = {}


def guardar(fig, nombre):
    fig.savefig(OUT / f"{nombre}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figura:", nombre)


# ---------------------------------------------------------------- datos
# MISMO periodo oficial del informe final (1-ene-2019 a 5-ago-2026, solo dataset maestro, 66.576 h), igual que
# scripts_experimento/informe_final_caracterizacion.py de Juan, para que las cifras de las diapositivas coincidan
# con las del informe. No se usa la serie de operacion.
CORTE = "2026-08-05 23:00"
df = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, parse_dates=["fecha_hora"])
                for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")])
df = df.drop_duplicates("fecha_hora").set_index("fecha_hora").sort_index().loc[:CORTE]
precio = df["precio_bolsa"]
FIN = precio.index[-1]
assert len(df) == 66576 and precio.notna().all(), (len(df), precio.isna().sum())

TRIM = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
oni = (pd.read_csv(RAIZ / "data/produccion/oni_index.csv")
       .melt(id_vars="anio", value_vars=TRIM, var_name="trim", value_name="oni")
       .assign(fecha=lambda t: pd.to_datetime(dict(year=t["anio"],
                                                   month=t["trim"].map({k: i + 1 for i, k in enumerate(TRIM)}), day=1))))
oni = oni[oni["fecha"] + pd.DateOffset(months=1) < FIN.to_period("M").to_timestamp()].dropna(subset=["oni"])
oni = oni.set_index("fecha")["oni"].sort_index().loc["2019-01-01":]

CIFRAS["corte"] = f"{FIN:%d/%m/%Y %H:%M}"
CIFRAS["horas"] = int(len(df))
CIFRAS["faltantes"] = int(len(pd.date_range(df.index.min(), FIN, freq="h").difference(df.index)))
CIFRAS["duplicadas"] = int(df.index.duplicated().sum())
print(f"Periodo: {df.index.min():%d/%m/%Y} a {FIN:%d/%m/%Y} | {len(df):,} horas | "
      f"faltantes {CIFRAS['faltantes']} | duplicadas {CIFRAS['duplicadas']}")


def franjas_enso(ax):
    """Una franja continua por cada racha de meses del mismo episodio."""
    signo = np.sign(oni.where(oni.abs() >= 0.5, 0))
    for _, g in signo.groupby((signo != signo.shift()).cumsum()):
        if g.iloc[0] != 0:
            ax.axvspan(g.index[0], g.index[-1] + pd.offsets.MonthBegin(1), lw=0,
                       color=AMBER if g.iloc[0] > 0 else BLUE, alpha=0.13 if g.iloc[0] > 0 else 0.10)


# ================================================================ F1. El reloj del precio: ciclo diario
tipo = np.where(df.index.dayofweek < 5, "Laboral", np.where(df.index.dayofweek == 5, "Sábado", "Domingo"))
colores = {"Laboral": INK, "Sábado": AMBER, "Domingo": GRAY}
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0))
for nombre, color in colores.items():
    sub = df[tipo == nombre]
    g = sub.groupby(sub.index.hour)
    axs[0].plot(range(24), g["precio_bolsa"].median(), color=color, lw=2.2, label=nombre, zorder=3)
    if nombre == "Laboral":  # una sola banda: tres superpuestas ensucian la lectura
        axs[0].fill_between(range(24), g["precio_bolsa"].quantile(.25), g["precio_bolsa"].quantile(.75),
                            color=INK, alpha=.09, lw=0, label="Laboral: percentiles 25-75")
    axs[1].plot(range(24), g["demanda"].mean() / 1e3, color=color, lw=2.2, label=nombre)
axs[0].set_title("Precio por hora (mediana)")
axs[0].set_ylabel("COP/kWh")
axs[1].set_title("Demanda por hora (promedio)")
axs[1].set_ylabel("MW")
for ax in axs:
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hora del día")
    ax.legend()
guardar(fig, "F1_ciclo_diario")

lab = df[tipo == "Laboral"]
pp, pd_ = lab.groupby(lab.index.hour)["precio_bolsa"].mean(), lab.groupby(lab.index.hour)["demanda"].mean()
med = df.groupby(tipo)[["precio_bolsa", "demanda"]].mean()
CIFRAS["F1"] = {
    "hora_max_precio": int(pp.idxmax()), "precio_max": round(float(pp.max())),
    "hora_min_precio": int(pp.idxmin()), "precio_min": round(float(pp.min())),
    "hora_max_demanda": int(pd_.idxmax()),
    "corr_perfiles": round(float(np.corrcoef(pp, pd_)[0, 1]), 2),
    "domingo_precio_%": round(float(med.loc["Domingo", "precio_bolsa"] / med.loc["Laboral", "precio_bolsa"] - 1) * 100, 1),
    "domingo_demanda_%": round(float(med.loc["Domingo", "demanda"] / med.loc["Laboral", "demanda"] - 1) * 100, 1),
}

# ================================================================ F2. El mismo reloj, en frecuencia
y = precio.interpolate().to_numpy(float)
tendencia_sg = savgol_filter(y, 721, 3)
residuo = y - tendencia_sg
f, pxx = signal.welch(residuo, fs=1.0, window="hann", nperseg=24 * 7 * 16, noverlap=24 * 7 * 8, detrend="constant")
per, pot = 1 / f[1:], pxx[1:]
fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.loglog(per, pot, color=BLUE, lw=1.2)
ax.invert_xaxis()
ax.set_ylim(top=pot.max() * 12)  # aire para que la etiqueta del pico diario no toque el título
for p, nombre, color, dx in ((168, "168 h\n(semana)", MUTED, 0), (24, "24 h (día)", AMBER, -46),
                             (12, "12 h", MUTED, 0), (8, "8 h", MUTED, 0), (6, "6 h", MUTED, 0)):
    i = np.argmin(abs(per - p))
    ax.plot(per[i], pot[i], "o", ms=7 if p == 24 else 5, color=color, zorder=3)
    ax.annotate(nombre, (per[i], pot[i]), xytext=(dx, 11), textcoords="offset points",
                ha="right" if dx else "center", fontsize=9.5, color=color,
                fontweight="bold" if p == 24 else "normal")
ax.set_xlabel("Período del ciclo (horas, escala logarítmica)")
ax.set_ylabel("Densidad espectral ((COP/kWh)²·h)")
ax.set_title("Espectro del precio sin tendencia (Welch)")
guardar(fig, "F2_espectro")

picos, _ = signal.find_peaks(np.log(pot), prominence=1.0)
picos = picos[per[picos] <= 200]
tp = pd.DataFrame({"periodo_h": per[picos], "potencia": pot[picos]}).sort_values("potencia", ascending=False).head(5)
p24 = pot[np.argmin(abs(per - 24))]
CIFRAS["F2"] = {"picos": [{"periodo_h": round(float(r.periodo_h), 1),
                           "relativa_al_diario_%": round(float(r.potencia / p24) * 100, 1)}
                          for r in tp.itertuples()]}

# ================================================================ F3. La amplitud del ciclo diario crece en El Nino
nper = 24 * 28
f_s, t_s, Z = signal.stft(residuo, fs=1.0, window="hann", nperseg=nper, noverlap=nper - 24 * 7, detrend="constant",
                          boundary=None, padded=False)
fechas_s = precio.index[0] + pd.to_timedelta(t_s, unit="h")
amp = 2 * np.abs(Z)
i24, i12 = np.argmin(abs(f_s - 1 / 24)), np.argmin(abs(f_s - 1 / 12))
fig, ax = plt.subplots(figsize=(11.5, 3.8))
franjas_enso(ax)
ax.plot(fechas_s, amp[i24], color=INK, lw=1.8, label="Ciclo de 24 h")
ax.plot(fechas_s, amp[i12], color=AMBER, lw=1.4, label="Armónico de 12 h")
ax.set_ylabel("Amplitud (COP/kWh)")
ax.set_title("Diferencia entre el día y la noche, en ventanas de 4 semanas")
ax.legend(loc="upper left", ncol=2)
ax.set_xlim(fechas_s[0], fechas_s[-1])
guardar(fig, "F3_amplitud_diaria")

a24 = pd.Series(amp[i24], index=fechas_s)
oni_v = a24.index.map(lambda d: oni.asof(d) if d >= oni.index[0] else np.nan)
CIFRAS["F3"] = {
    "amplitud_nino": round(float(a24[oni_v >= .5].mean())),
    "amplitud_neutral": round(float(a24[(oni_v > -.5) & (oni_v < .5)].mean())),
    "amplitud_nina": round(float(a24[oni_v <= -.5].mean())),
    "maxima": round(float(a24.max())), "maxima_fecha": f"{a24.idxmax() + pd.Timedelta(days=14):%m/%Y}",
}

# ================================================================ F4. Que filtro sigue mejor el cambio de regimen
p = precio.interpolate()
sg = pd.Series(savgol_filter(p.to_numpy(float), 721, 3), index=p.index)
mm = p.rolling(720, center=True, min_periods=360).mean()
bw = pd.Series(filtfilt(*butter(2, 2.0 / 720), p.to_numpy(float)), index=p.index)
ZOOMS = {"El Niño 2023-2024 (subida rápida)": slice("2023-06-01", "2024-03-01"),
         "El Niño 2026 (subida gradual)": slice("2026-02-01", None)}
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
for ax, (evento, z) in zip(axs, ZOOMS.items()):
    sub = p.loc[z]
    ax.plot(sub.index, sub.rolling(24, center=True).mean(), color=LINE, lw=1.0, label="Precio (media de 24 h)")
    ax.plot(sub.index, mm.loc[z], color=GRAY, lw=2.0, label="Promedio móvil 720 h")
    ax.plot(sub.index, bw.loc[z], color=GREEN, lw=2.0, label="Butterworth fase cero 720 h")
    ax.plot(sub.index, sg.loc[z], color=AMBER, lw=2.4, label="Savitzky-Golay 721 h")
    ax.set_title(evento)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
axs[0].set_ylabel("COP/kWh")
axs[0].legend(loc="upper left", fontsize=9)
guardar(fig, "F4_filtros")

fig, ax = plt.subplots(figsize=(7.6, 4.3))
z = ZOOMS["El Niño 2023-2024 (subida rápida)"]
sub = p.loc[z]
ax.plot(sub.index, sub.rolling(24, center=True).mean(), color=LINE, lw=1.0, label="Precio (media de 24 h)")
ax.plot(sub.index, mm.loc[z], color=GRAY, lw=2.0, label="Promedio móvil (837 h)")
ax.plot(sub.index, bw.loc[z], color=GREEN, lw=2.0, label="Butterworth (388 h)")
ax.plot(sub.index, sg.loc[z], color=AMBER, lw=2.6, label="Savitzky-Golay (348 h)")
ax.set_ylabel("COP/kWh")
ax.set_title("Cuál detecta antes la subida de El Niño 2023-2024")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
ax.legend(loc="upper left", fontsize=9.5)
guardar(fig, "F4b_filtros_un_panel")

# ================================================================ F5. El ONI se adelanta a la hidrologia
dia = pd.DataFrame({"precio": precio.resample("D").mean(),
                    "demanda": (df["demanda"] / 1e3).resample("D").mean(),
                    "generacion": (df["generacion"] / 1e3).resample("D").mean(),
                    "embalses": (df["volumen_embalses"] / 1e9).resample("D").mean(),
                    "aportes": (df["aportes_hidricos"] / 1e6).resample("D").mean()})
mens = dia[["aportes", "embalses", "precio"]].resample("MS").mean()
rez = range(0, 13)
t = pd.DataFrame({v: [mens[v].corr(oni.shift(k).reindex(mens.index)) for k in rez] for v in mens.columns},
                 index=pd.Index(rez, name="rezago"))
fig, ax = plt.subplots(figsize=(7.6, 4.3))
for v, color, n in (("aportes", GREEN, "Aportes hídricos"), ("embalses", BLUE, "Volumen de embalses"),
                    ("precio", AMBER, "Precio")):
    ax.plot(t.index, t[v], "o-", ms=5, color=color, label=n)
    k = int(t[v].abs().idxmax())
    ax.annotate(f"{t[v].loc[k]:+.2f}".replace(".", ","), (k, t[v].loc[k]), xytext=(0, -16 if t[v].loc[k] < 0 else 10),
                textcoords="offset points", ha="center", fontsize=9.5, color=color, fontweight="bold")
ax.axhline(0, color=MUTED, lw=.9)
ax.set_xlabel("Meses que el ONI se adelanta")
ax.set_ylabel("Correlación")
ax.set_title("El ONI anticipa la hidrología y el precio")
ax.legend()
guardar(fig, "F5_oni_cadena")
CIFRAS["F5"] = {v: {"mejor_rezago_meses": int(t[v].abs().idxmax()),
                    "correlacion": round(float(t[v].loc[t[v].abs().idxmax()]), 2)} for v in t.columns}

# ================================================================ F6. Que dice cada variable del precio
dd = dia.copy()
dd["oni"] = [oni.asof(x) if x >= oni.index[0] else np.nan for x in dd.index]
rezd = range(0, 31)
corr = pd.DataFrame({v: [dd["precio"].corr(dd[v].shift(k)) for k in rezd]
                     for v in ("oni", "demanda", "generacion", "embalses", "aportes")},
                    index=pd.Index(rezd, name="rezago"))
fig, ax = plt.subplots(figsize=(7.6, 4.3))
for v, color, n in (("oni", AMBER, "ONI"), ("demanda", BLUE, "Demanda"), ("generacion", GRAY, "Generación"),
                    ("embalses", GREEN, "Embalses"), ("aportes", RUST, "Aportes")):
    ax.plot(corr.index, corr[v], "-", color=color, lw=2.0, label=n)
ax.axhline(0, color=MUTED, lw=.9)
ax.set_xlabel("Rezago de la variable (días)")
ax.set_ylabel("Correlación con el precio del día")
ax.set_title("Qué dice cada variable del precio")
ax.legend(ncol=3)
guardar(fig, "F6_correlacion_rezago")
CIFRAS["F6"] = {v: {"rezago_0": round(float(corr[v].iloc[0]), 2),
                    "mejor_rezago_dias": int(corr[v].abs().idxmax()),
                    "correlacion": round(float(corr[v].loc[corr[v].abs().idxmax()]), 2)} for v in corr.columns}

# ================================================================ F7. La serie completa (respaldo / problematica)
diario = precio.resample("D").mean()
fig, ax = plt.subplots(figsize=(11.5, 3.6))
franjas_enso(ax)
ax.plot(diario.index, diario, color=GRAY, lw=0.7, alpha=.75, label="Promedio diario")
ax.plot(diario.index, diario.rolling(30, center=True).mean(), color=INK, lw=2.0, label="Media móvil de 30 días")
ax.set_ylabel("COP/kWh")
ax.set_xlim(diario.index.min(), diario.index.max())
ax.legend(handles=ax.get_legend_handles_labels()[0] + [plt.Rectangle((0, 0), 1, 1, color=AMBER, alpha=.2),
                                                       plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=.18)],
          labels=["Promedio diario", "Media móvil de 30 días", "El Niño", "La Niña"], loc="upper left", ncol=4)
guardar(fig, "F7_serie_enso")
epis = diario.groupby(np.select(
    [diario.index.map(lambda d: oni.asof(d) if d >= oni.index[0] else np.nan) >= .5,
     diario.index.map(lambda d: oni.asof(d) if d >= oni.index[0] else np.nan) <= -.5],
    ["El Niño", "La Niña"], "Neutral")).mean()
CIFRAS["F7"] = {k: round(float(v)) for k, v in epis.items()}
CIFRAS["F7"]["maximo_diario"] = round(float(diario.max()))
CIFRAS["F7"]["maximo_fecha"] = f"{diario.idxmax():%d/%m/%Y}"

# ================================================================ apendice didactico (A1 y A2)
tend = pd.Series(tendencia_sg, index=precio.index)
resid = pd.Series(residuo, index=precio.index)
kw = dict(fs=1.0, window="hann", nperseg=24 * 7 * 16, noverlap=24 * 7 * 8, detrend="constant")
f_con, p_con = signal.welch(precio.to_numpy(float), **kw)
f_sin, p_sin = signal.welch(resid.to_numpy(float), **kw)

# ================================================================ A1. que es quitar la tendencia
z = slice("2024-07-01", "2024-12-31")
fig, axs = plt.subplots(2, 1, figsize=(11.5, 5.6), sharex=True, gridspec_kw={"hspace": 0.28})
axs[0].plot(precio.loc[z].index, precio.loc[z], color=GRAY, lw=0.7, label="Precio hora a hora")
axs[0].plot(tend.loc[z].index, tend.loc[z], color=AMBER, lw=3, label="Tendencia de fondo (Savitzky-Golay, 721 h)")
axs[0].set_ylabel("COP/kWh")
axs[0].set_title("1. El precio y su tendencia de fondo")
axs[0].legend(loc="upper right")
axs[1].axhline(0, color=MUTED, lw=1)
axs[1].plot(resid.loc[z].index, resid.loc[z], color=BLUE, lw=0.7)
axs[1].set_ylabel("COP/kWh")
axs[1].set_title("2. Lo que queda al restarla: solo los ciclos (esto es lo que entra al espectro)")
axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
guardar(fig, "A1_que_es_quitar_la_tendencia")

# ================================================================ A2. por que hay que quitarla
per = 1 / f_con[1:]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.loglog(per, p_con[1:], color=GRAY, lw=2.6, label="Precio tal cual")
ax.loglog(per, p_sin[1:], color=BLUE, lw=1.2, label="Precio sin la tendencia")
ax.invert_xaxis()
ax.set_ylim(top=p_con[1:].max() * 25)
i24 = np.argmin(abs(per - 24))
ax.plot(per[i24], p_sin[1:][i24], "o", ms=8, color=AMBER, zorder=3)
ax.annotate("24 h: el pico\nno se mueve", (per[i24], p_sin[1:][i24]), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=10, color=AMBER, fontweight="bold")
ax.axvspan(per[0], 1000, color=AMBER, alpha=0.07, lw=0)
ax.annotate("Aquí se nota la diferencia:\nla tendencia aporta el 43 % de\nla potencia y se va casi toda",
            xy=(1700, 1.4e7), xytext=(620, 2.6e3), color=MUTED, fontsize=9.5, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, connectionstyle="arc3,rad=-0.2"))
ax.set_xlabel("Período del ciclo (horas, escala logarítmica)")
ax.set_ylabel("Densidad espectral ((COP/kWh)²·h)")
ax.set_title("Quitar la tendencia solo borra la parte lenta del espectro")
ax.legend(loc="upper right")
guardar(fig, "A2_por_que_quitar_la_tendencia")

# ---------------------------------------------------------------- cifras para la explicacion
var_total = float(precio.var())
var_resid = float(resid.var())
largos = per > 1000
print(f"Desviación estándar del precio: {precio.std():,.0f} COP/kWh | del residuo (sin tendencia): {resid.std():,.0f}")
print(f"La tendencia explica el {(1 - var_resid / var_total) * 100:.0f} % de la variación total del precio.")
print(f"Potencia en ciclos de más de 1.000 h — con tendencia: {p_con[1:][largos].sum() / p_con[1:].sum() * 100:.1f} % "
      f"| sin tendencia: {p_sin[1:][largos].sum() / p_sin[1:].sum() * 100:.1f} %")
print(f"Altura del pico de 24 h frente al nivel de fondo en 20-30 h — "
      f"con tendencia: x{p_con[1:][i24] / np.median(p_con[1:][(per > 20) & (per < 30)]):.0f} | "
      f"sin tendencia: x{p_sin[1:][i24] / np.median(p_sin[1:][(per > 20) & (per < 30)]):.0f}")

(OUT / "cifras_caracterizacion.json").write_text(
    json.dumps(CIFRAS, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nCifras:")
print(json.dumps(CIFRAS, indent=2, ensure_ascii=False))
