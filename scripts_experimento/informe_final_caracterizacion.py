# -*- coding: utf-8 -*-
"""
Figuras de caracterizacion del precio para el informe final (OE1), 12-nov-2026. Son las del notebook
03_caracterizacion_precio_juan.ipynb, pero con el periodo oficial del informe (1-ene-2019 a 5-ago-2026, dataset maestro)
para que las cifras coincidan con el texto. Salida: informe_final_assets/carac_*.png y cifras_caracterizacion.json
Uso: python scripts_experimento/informe_final_caracterizacion.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy import signal  # noqa: E402
from scipy.signal import butter, filtfilt, savgol_filter  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
OUT = AQUI / "informe_final_assets"
AZUL, NARANJA, AQUA, GRIS, TINTA = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a66", "#1f1f1e"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Calibri", "Arial"], "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.edgecolor": "#b9b8b2",
                     "axes.labelcolor": TINTA, "xtick.color": GRIS, "ytick.color": GRIS, "legend.frameon": False,
                     "savefig.dpi": 200})

# ------------------------------------------------------------------------------------------------ datos
df = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, parse_dates=["fecha_hora"])
                for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")])
df = df.drop_duplicates("fecha_hora").set_index("fecha_hora").sort_index().loc[:"2026-08-05 23:00"]
precio = df["precio_bolsa"]
FIN = precio.index[-1]
assert len(df) == 66576 and precio.notna().all(), (len(df), precio.isna().sum())

TRIM = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
ancho = pd.read_csv(RAIZ / "data/produccion/oni_index.csv")
oni = ancho.melt(id_vars="anio", value_vars=TRIM, var_name="trim", value_name="oni")
oni["fecha"] = pd.to_datetime(dict(year=oni["anio"], month=oni["trim"].map({k: i + 1 for i, k in enumerate(TRIM)}), day=1))
oni = oni[oni["fecha"] + pd.DateOffset(months=1) < FIN.to_period("M").to_timestamp()].dropna(subset=["oni"])
oni = oni.set_index("fecha")["oni"].sort_index().loc["2019-01-01":]
cif = {}


def franjas_enso(ax):
    s = np.sign(oni.where(oni.abs() >= 0.5, 0))
    for _, g in s.groupby((s != s.shift()).cumsum()):
        if g.iloc[0] != 0:
            ax.axvspan(g.index[0], g.index[-1] + pd.offsets.MonthBegin(1), lw=0,
                       color=NARANJA if g.iloc[0] > 0 else AZUL, alpha=0.12)


# ------------------------------------------------------------------------------------------------ 1) ciclo diario
tipo = np.where(df.index.dayofweek < 5, "Laboral", np.where(df.index.dayofweek == 5, "Sábado", "Domingo"))
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
for nombre, color in {"Laboral": AZUL, "Sábado": NARANJA, "Domingo": AQUA}.items():
    g = df[tipo == nombre].groupby(df[tipo == nombre].index.hour)
    axs[0].plot(range(24), g["precio_bolsa"].median(), color=color, lw=2, label=nombre)
    axs[0].fill_between(range(24), g["precio_bolsa"].quantile(.25), g["precio_bolsa"].quantile(.75), color=color,
                        alpha=.12, lw=0)
    axs[1].plot(range(24), g["demanda"].mean() / 1e3, color=color, lw=2, label=nombre)
axs[0].set_title("Precio por hora (mediana y percentiles 25-75)"); axs[0].set_ylabel("COP/kWh")
axs[1].set_title("Demanda promedio por hora"); axs[1].set_ylabel("MWh")
for ax in axs:
    ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("Hora del día"); ax.legend(fontsize=8.5)
plt.tight_layout(); fig.savefig(OUT / "carac_ciclo_diario.png"); plt.close(fig)
lab = df[tipo == "Laboral"]
pp, pd_ = lab.groupby(lab.index.hour)["precio_bolsa"].mean(), lab.groupby(lab.index.hour)["demanda"].mean()
med = df.groupby(tipo)[["precio_bolsa", "demanda"]].mean()
cif["ciclo"] = {"h_max_p": int(pp.idxmax()), "p_max": pp.max(), "h_min_p": int(pp.idxmin()), "p_min": pp.min(),
                "h_max_d": int(pd_.idxmax()), "corr": float(np.corrcoef(pp, pd_)[0, 1]),
                **{f"{k}_{c}": float(med.loc[k, c] / med.loc["Laboral", c] - 1) for k in ("Sábado", "Domingo")
                   for c in ("precio_bolsa", "demanda")}}

# ------------------------------------------------------------------------------------------------ 2) mapa de calor ano-mes
g_mes = precio.groupby([precio.index.year, precio.index.month])
tabla = g_mes.mean().where(g_mes.size() >= 20 * 24).unstack()
escala_xm = LinearSegmentedColormap.from_list("xm", [(0.0, "#2A7FC0"), (0.22, "#8BBCD6"), (0.45, "#E4ECDF"),
                                                     (0.5, "#F4EED6"), (0.62, "#F6CFA4"), (0.8, "#EF8A45"),
                                                     (1.0, "#E3511B")])
v = tabla.to_numpy()
norma = TwoSlopeNorm(vmin=np.nanmin(v), vcenter=np.nanmean(v), vmax=np.nanmax(v))
fig, ax = plt.subplots(figsize=(9.6, 3.5))
im = ax.imshow(v, cmap=escala_xm, norm=norma, aspect="auto")
ax.set_xticks(range(12), ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])
ax.set_yticks(range(len(tabla)), tabla.index); ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
for (i, j), x in np.ndenumerate(v):
    if not np.isnan(x):
        r, g, b, _ = escala_xm(norma(x))
        ax.text(j, i, f"{x:,.0f}".replace(",", "."), ha="center", va="center", fontsize=7.5,
                color="white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else TINTA)
fig.colorbar(im, ax=ax, label="COP/kWh", shrink=.85)
plt.tight_layout(); fig.savefig(OUT / "carac_calor_mes.png"); plt.close(fig)
mx = tabla.stack().idxmax()
mn = tabla.stack().idxmin()
cif["calor"] = {"media": float(np.nanmean(v)), "max_anio": int(mx[0]), "max_mes": int(mx[1]), "max": float(tabla.stack().max()),
                "min_anio": int(mn[0]), "min_mes": int(mn[1]), "min": float(tabla.stack().min()),
                "anual": {int(a): float(x) for a, x in precio.groupby(precio.index.year).mean().items()}}

# ------------------------------------------------------------------------------------------------ 3) espectrograma
y = precio.to_numpy(float)
tend_sg = savgol_filter(y, 721, 3)
residuo = y - tend_sg
nper = 24 * 28
f_s, t_s, Z = signal.stft(residuo, fs=1.0, window="hann", nperseg=nper, noverlap=nper - 24 * 7, detrend="constant",
                          boundary=None, padded=False)
fechas = precio.index[0] + pd.to_timedelta(t_s, unit="h")
amp = 2 * np.abs(Z)
i24, i12 = np.argmin(abs(f_s - 1 / 24)), np.argmin(abs(f_s - 1 / 12))
azules = LinearSegmentedColormap.from_list("azules", ["#e8f1fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
fig, axs = plt.subplots(2, 1, figsize=(9.6, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
banda = (f_s > 0) & (f_s <= 1 / 4)
axs[0].pcolormesh(fechas, f_s[banda] * 24, amp[banda], cmap=azules, shading="auto", vmax=np.percentile(amp[banda], 99.5))
axs[0].set_ylabel("Ciclos por día"); axs[0].set_title("Espectrograma del precio sin tendencia"); axs[0].grid(False)
franjas_enso(axs[1])
axs[1].plot(fechas, amp[i24], color=AZUL, lw=1.6, label="Ciclo de 24 h")
axs[1].plot(fechas, amp[i12], color=NARANJA, lw=1.6, label="Armónico de 12 h")
axs[1].set_ylabel("Amplitud (COP/kWh)"); axs[1].legend(loc="upper left", fontsize=8.5)
axs[1].set_title("Amplitud del ciclo diario en ventanas de 4 semanas")
axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout(); fig.savefig(OUT / "carac_espectrograma.png"); plt.close(fig)
a24 = pd.Series(amp[i24], index=fechas)
fase = a24.index.map(lambda d: oni.asof(d))
cif["espectro"] = {"nino": float(a24[fase >= .5].mean()), "neutral": float(a24[(fase > -.5) & (fase < .5)].mean()),
                   "nina": float(a24[fase <= -.5].mean()), "max": float(a24.max()),
                   "max_fecha": str((a24.idxmax() + pd.Timedelta(days=14)).date())}

# ------------------------------------------------------------------------------------------------ 4) filtros de tendencia
tend = {"Savitzky-Golay 721 h": (tend_sg, AZUL),
        "Promedio móvil centrado 720 h": (precio.rolling(720, center=True, min_periods=360).mean().to_numpy(), NARANJA),
        "Butterworth fase cero (orden 2)": (filtfilt(*butter(2, 2.0 / 720), y), AQUA)}
zoom = slice("2023-06-01", "2024-03-01")
fig, ax = plt.subplots(figsize=(9.6, 3.4))
sub = precio.loc[zoom]
ax.plot(sub.index, sub.rolling(24, center=True).mean(), color="#b9b8b2", lw=.8, label="Precio (media de 24 h)")
for nombre, (t, color) in tend.items():
    ax.plot(sub.index, pd.Series(t, index=precio.index).loc[zoom], color=color, lw=2, label=nombre)
ax.set_ylabel("COP/kWh"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
ax.set_title("Tendencia del precio en El Niño 2023-2024"); ax.legend(fontsize=8.5, loc="upper left")
plt.tight_layout(); fig.savefig(OUT / "carac_filtros.png"); plt.close(fig)

# ------------------------------------------------------------------------------------------------ 5) correlaciones
d = df[["precio_bolsa", "demanda", "generacion", "volumen_embalses", "aportes_hidricos"]].copy()
d["oni"] = d.index.map(lambda x: oni.asof(x) if x >= oni.index[0] else np.nan)
nombres = {"precio_bolsa": "Precio", "demanda": "Demanda", "generacion": "Generación", "volumen_embalses": "Embalses",
           "aportes_hidricos": "Aportes", "oni": "ONI"}
m = d.rename(columns=nombres).corr()
dd = d.resample("D").mean()
lags = range(0, 31)
cruz = {c: [dd["precio_bolsa"].corr(dd[c].shift(k)) for k in lags] for c in ("aportes_hidricos", "volumen_embalses")}
divergente = LinearSegmentedColormap.from_list("div", [NARANJA, "#f0efec", AZUL])
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
im = axs[0].imshow(m, cmap=divergente, vmin=-1, vmax=1); axs[0].grid(False)
axs[0].set_xticks(range(len(m)), m.columns, rotation=45, ha="right"); axs[0].set_yticks(range(len(m)), m.index)
for (i, j), x in np.ndenumerate(m.to_numpy()):
    axs[0].text(j, i, f"{x:.2f}".replace(".", ","), ha="center", va="center", fontsize=7.5, color=TINTA)
axs[0].set_title("Correlación horaria"); fig.colorbar(im, ax=axs[0], shrink=.8)
for c, color in (("aportes_hidricos", AZUL), ("volumen_embalses", NARANJA)):
    axs[1].plot(lags, cruz[c], "o-", ms=3, color=color, label=nombres[c])
axs[1].axhline(0, color=GRIS, lw=.8)
axs[1].set_xlabel("Rezago (días)"); axs[1].set_ylabel("Correlación con el precio")
axs[1].set_title("Correlación cruzada con la hidrología (diaria)"); axs[1].legend(fontsize=8.5)
plt.tight_layout(); fig.savefig(OUT / "carac_correlacion.png"); plt.close(fig)
ap = pd.Series(cruz["aportes_hidricos"])
cif["corr"] = {"horaria": {nombres[k]: float(m.loc["Precio", nombres[k]]) for k in nombres if k != "precio_bolsa"},
               "aportes_0": float(ap[0]), "aportes_min": float(ap.min()), "aportes_min_lag": int(ap.idxmin()),
               "embalses_0": float(cruz["volumen_embalses"][0]), "embalses_30": float(cruz["volumen_embalses"][30])}

(OUT / "cifras_caracterizacion.json").write_text(json.dumps(cif, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(cif, ensure_ascii=False, indent=1))
