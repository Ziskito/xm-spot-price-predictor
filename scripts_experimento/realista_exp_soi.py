# -*- coding: utf-8 -*-
"""
¿Aporta el SOI al pronostico de 24 h? (10-nov-2026, pregunta del experto: ONI, Niño 3.4 y SOI)

El SOI (Indice de Oscilacion del Sur, NOAA CPC, presion estandarizada Tahiti - Darwin) mide el ENSO desde la atmosfera;
el ONI que ya usa Augur lo mide desde la temperatura del mar en Niño 3.4. Sobre el CatBoost R+P de produccion (perdida
MAE, objetivo log(precio/predespacho), reentreno mensual con lo publicado hasta 73 h antes del corte) se agregan:
  S   SOI disponible al corte (mes M-1 desde el dia 10, M-2 antes: CPC lo publica en la primera semana) y su media
      de tres meses
  S+R lo anterior mas el SOI con 3 y 8 meses mas de rezago (lo que tardan aportes y embalses, como la variable C del ONI)
Evaluacion: 2026 (15-ene a 5-ago) y operacion (6-ago a 31-oct), mismas horas, DM diario (HAC) contra R+P.
Si alguna variante mejora con significancia se pasa al walk-forward historico; si no, no se adopta.
Salida: resultados/escenario_realista/mejoras/soi_predicciones.csv, soi_metricas.csv y data/processed/soi_cpc.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_votantes import mensual  # noqa: E402

RUTA = rc.RAIZ / "data" / "processed" / "soi_cpc.csv"


def soi_mensual():
    """Serie mensual del SOI estandarizado (segunda tabla del archivo de CPC)."""
    txt = requests.get("https://www.cpc.ncep.noaa.gov/data/indices/soi", timeout=60).text.splitlines()
    ini = next(i for i, l in enumerate(txt) if "STANDARDIZED" in l)
    filas = []
    for l in txt[ini:]:
        if len(l) < 10 or not l[:4].isdigit():
            continue
        anio = int(l[:4])
        for m in range(12):
            v = float(l[4 + 6 * m: 10 + 6 * m])
            if v > -999:
                filas.append((pd.Timestamp(anio, m + 1, 1), v))
    s = pd.Series(dict(filas), name="soi").sort_index()
    s.rename_axis("mes").to_csv(RUTA)
    return s


def variables(idx, s):
    """Para cada hora objetivo, el SOI publicado al corte (00:00 del dia D)."""
    dia = (idx - pd.Timedelta(hours=1)).normalize()
    mes = dia.to_period("M").to_timestamp()
    atras = np.where(dia.day >= 10, 1, 2)
    m3 = s.rolling(3).mean()

    def toma(serie, extra):
        claves = [m - pd.DateOffset(months=int(a) + extra) for m, a in zip(mes, atras)]
        return serie.reindex(claves).to_numpy()

    return pd.DataFrame({"soi": toma(s, 0), "soi_media3m": toma(m3, 0),
                         "soi_mas3m": toma(s, 3), "soi_mas8m": toma(s, 8)}, index=idx)


def dm(y, a, b, idx):
    d = pd.Series(np.abs(a - y) - np.abs(b - y), index=idx).groupby((idx - pd.Timedelta(hours=1)).normalize()).mean()
    r = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return float(r.tvalues[0]), float(r.pvalues[0])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    s = soi_mensual()
    print(f"SOI: {s.index.min():%Y-%m} a {s.index.max():%Y-%m}", flush=True)
    d = rc.tabla_realista(produccion=True).set_index("fecha_hora").sort_index()
    d = d.join(variables(d.index, s))
    RP = conjuntos(d)["R+P"]
    variantes = {"R+P": RP, "R+P+S": RP + ["soi", "soi_media3m"], "R+P+S+R": RP + ["soi", "soi_media3m", "soi_mas3m", "soi_mas8m"]}
    ini, fin = pd.Timestamp("2026-01-01 01:00"), pd.Timestamp("2026-10-31 23:00")
    idx = d.loc[ini:fin].dropna(subset=["precio_bolsa", "pre_obj"]).index
    pan = pd.DataFrame({"real": d.loc[idx, "precio_bolsa"]})
    for nombre, cols in variantes.items():
        print(f"== {nombre} ({len(cols)} variables)", flush=True)
        pan[nombre] = mensual(d, cols, ini, fin).reindex(idx)
        print(f"   listo ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pan.to_csv(rc.MEJ / "soi_predicciones.csv")
    ev = pan[pan.index >= "2026-01-15"].dropna()
    tramos = {"2026 (15-ene a 5-ago)": ev.index <= "2026-08-05 00:00", "operación (6-ago a 31-oct)": ev.index > "2026-08-05 00:00",
              "ene-abr": ev.index < "2026-05-01", "may-ago (El Niño)": (ev.index >= "2026-05-01") & (ev.index <= "2026-08-05 00:00")}
    filas = []
    for tr, sel in tramos.items():
        e = ev[sel]
        y = e["real"].to_numpy()
        for c in variantes:
            t, p = (np.nan, np.nan) if c == "R+P" else dm(y, e[c].to_numpy(), e["R+P"].to_numpy(), e.index)
            filas.append({"tramo": tr, "variante": c, "horas": len(e), "MAE": float(np.abs(e[c] - e["real"]).mean()),
                          "RMSE": float(np.sqrt(((e[c] - e["real"]) ** 2).mean())), "DM_t_vs_R+P": t, "DM_p": p})
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / "soi_metricas.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
