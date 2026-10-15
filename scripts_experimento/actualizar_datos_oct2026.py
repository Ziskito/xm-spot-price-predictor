# -*- coding: utf-8 -*-
"""
Actualizacion de datos del proyecto (14-oct-2026, a pedido de Juan). El periodo del proyecto no cambia (hasta el
5-ago-2026 23:00); solo se corrigen los valores que XM y NOAA revisaron despues de la descarga original.

1) ONI: data/external/oni_index.csv se reemplaza por la tabla oficial de NOAA (oni.ascii.txt, anomalia redondeada a
   un decimal, como el archivo original). Diferencias con el archivo anterior: +-0,1 en 4 trimestres de 2021 y 2023, y
   2026, que estaba congelado en 1,4 desde JJA (NOAA: AMJ 0,9 -> 1,0, JJA 1,8, JAS 2,2). Los trimestres de 2026 sin
   dato aun (ASO a NDJ) repiten el ultimo publicado, la misma convencion del archivo anterior.
2) XM revisó el 1-5 de agosto de 2026 (precio de bolsa hasta 86 % en una hora, aportes hasta 18 %): se descargan de
   nuevo precio, demanda, generacion, volumen de embalses y aportes de esos dias (API publica servapibi) con el mismo
   procedimiento del notebook 02 (interpolacion lineal de las diarias a horas; las diarias se piden desde el
   25-jul y solo hasta el 5-ago, para no usar el 6-ago).
3) Se recalcula la columna oni de los dos archivos maestros con la rampa causal del notebook 02 y se rearman los dos
   archivos de variables con el procedimiento del notebook 05 (reproducido aqui; diferencias < 2e-6 con los archivos
   originales). Se verifica que fuera de las filas afectadas nada cambie.
Los archivos anteriores quedan en git (commit afc3a4f) y en resultados/backups_pre_actualizacion_14oct2026/.
Salida: data/external/oni_index.csv, data/processed/dataset_{maestro,features}_{2019_2025,2026}.csv y un resumen
en resultados/actualizacion_datos_14oct2026.txt
"""
import shutil
import sys
from pathlib import Path

import time

import holidays
import numpy as np
import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
PROC, EXT = RAIZ / "data" / "processed", RAIZ / "data" / "external"
RES = PROC / "resultados"
API = "https://servapibi.xm.com.co"


# ---- mismo procedimiento de los notebooks 02 (descarga, rampa causal del ONI) y 05 (variables) ----
def bajar(metrica, frecuencia, ini, fin):
    filas, a = [], ini
    while a <= fin:
        b = min(a + pd.Timedelta(days=27), fin)
        for intento in range(4):
            try:
                r = requests.post(f"{API}/{frecuencia}", timeout=60, json={"MetricId": metrica, "StartDate": f"{a:%Y-%m-%d}",
                                                                            "EndDate": f"{b:%Y-%m-%d}", "Entity": "Sistema"})
                r.raise_for_status()
                break
            except Exception as e:
                print(f"  reintento {metrica} {a:%Y-%m-%d}: {str(e)[:60]}", flush=True)
                time.sleep(5)
        for it in r.json().get("Items", []):
            if frecuencia == "hourly":
                v = it["HourlyEntities"][0]["Values"]
                for h in range(1, 25):
                    x = v.get(f"Hour{h:02d}")
                    if x not in (None, ""):
                        filas.append((pd.Timestamp(it["Date"]) + pd.Timedelta(hours=h - 1), float(x)))
            else:
                filas.append((pd.Timestamp(it["Date"]), float(it["DailyEntities"][0]["Value"])))
        a = b + pd.Timedelta(days=1)
    return pd.Series(dict(filas)).sort_index()


def oni_causal(indice, ruta=None):
    """Misma rampa causal del notebook 02 (celda 30), con el archivo del proyecto (o la tabla ONI de `ruta`)."""
    ancho = pd.read_csv(ruta or EXT / "oni_index.csv")
    temp = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    largo = ancho.melt(id_vars="anio", value_vars=temp, var_name="t", value_name="oni")
    largo["mes"] = largo["t"].map({t: i + 1 for i, t in enumerate(temp)})
    largo = largo.sort_values(["anio", "mes"]).reset_index(drop=True)
    largo["ant"] = largo["oni"].shift(1).fillna(largo["oni"])
    h = pd.DataFrame({"fecha_hora": indice})
    h["anio"], h["mes"] = h["fecha_hora"].dt.year, h["fecha_hora"].dt.month
    h = h.merge(largo[["anio", "mes", "oni", "ant"]], on=["anio", "mes"], how="left")
    inicio = h["fecha_hora"].values.astype("datetime64[M]").astype("datetime64[ns]")
    frac = ((h["fecha_hora"] - inicio) / pd.Timedelta(hours=1) / (h["fecha_hora"].dt.days_in_month * 24)).clip(0, 1)
    return (h["ant"] + frac * (h["oni"] - h["ant"])).to_numpy()


def features(m):
    """Notebook 05 (celdas 2 a 8) sobre la serie completa."""
    d = m.sort_values("fecha_hora").reset_index(drop=True).copy()
    for k in (24, 168):
        d[f"precio_lag{k}h"] = d["precio_bolsa"].shift(k)
    d["precio_media_24h"] = d["precio_bolsa"].shift(24).rolling(24).mean()
    d["precio_media_7d"] = d["precio_bolsa"].shift(24).rolling(24 * 7).mean()
    d["precio_media_30d"] = d["precio_bolsa"].shift(24).rolling(24 * 30).mean()
    d["hora"] = d["fecha_hora"].dt.hour
    d["hora_sin"], d["hora_cos"] = np.sin(2 * np.pi * d["hora"] / 24), np.cos(2 * np.pi * d["hora"] / 24)
    d["dia_semana"] = d["fecha_hora"].dt.dayofweek
    d["dia_semana_sin"], d["dia_semana_cos"] = np.sin(2 * np.pi * d["dia_semana"] / 7), np.cos(2 * np.pi * d["dia_semana"] / 7)
    d["dia_anio"] = d["fecha_hora"].dt.dayofyear
    d["dia_anio_sin"], d["dia_anio_cos"] = np.sin(2 * np.pi * d["dia_anio"] / 365.25), np.cos(2 * np.pi * d["dia_anio"] / 365.25)
    for col in ("volumen_embalses", "aportes_hidricos"):
        base = d[col].shift(24)
        d[col] = base
        d[f"{col}_delta_1d"], d[f"{col}_delta_7d"] = base.diff(24), base.diff(24 * 7)
        d[f"{col}_media_7d"], d[f"{col}_media_30d"] = base.rolling(24 * 7).mean(), base.rolling(24 * 30).mean()
        d[f"{col}_vs_media30d"] = base / base.rolling(24 * 30).mean()
    pb = d["precio_bolsa"].shift(24)
    d["precio_std_24h"], d["precio_std_7d"] = pb.rolling(24).std(), pb.rolling(24 * 7).std()
    d["precio_rango_24h"] = pb.rolling(24).max() - pb.rolling(24).min()
    d["ratio_volatilidad"] = d["precio_std_24h"] / d["precio_std_7d"]
    db = d["demanda"].shift(24)
    d["demanda_lag24h"], d["demanda_lag48h"], d["demanda_lag72h"] = db, d["demanda"].shift(48), d["demanda"].shift(72)
    d["demanda_media_24h"] = db.rolling(24).mean()
    fest = set(holidays.CO(years=range(2018, 2028)).keys())
    fn = d["fecha_hora"].dt.normalize()
    desp = lambda k: (fn - pd.Timedelta(days=k)).dt.date.map(lambda x: x in fest).astype(int)
    d["es_festivo"], d["festivo_lag24h"], d["festivo_lag48h"] = desp(0), desp(1), desp(2)
    d["festivo_lag72h"], d["festivo_lag168h"] = desp(3), desp(7)
    d["mismatch_festivo_24h"] = (d["es_festivo"] != d["festivo_lag24h"]).astype(int)
    d["mismatch_festivo_168h"] = (d["es_festivo"] != d["festivo_lag168h"]).astype(int)
    return d



TEMP = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
ARCH = ["dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv", "dataset_features_2019_2025.csv", "dataset_features_2026.csv"]


def oni_noaa():
    txt = requests.get("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", timeout=60).text
    filas = [l.split() for l in txt.splitlines()[1:] if l.strip()]
    n = pd.DataFrame(filas, columns=["SEAS", "YR", "TOTAL", "ANOM"]).astype({"YR": int, "ANOM": float})
    w = n.pivot_table(index="YR", columns="SEAS", values="ANOM")[TEMP]
    viejo = pd.read_csv(EXT / "oni_index.csv").set_index("anio")
    nuevo = w.loc[viejo.index.min():viejo.index.max()].round(1)
    nuevo = nuevo.T.ffill().T                          # trimestres aun sin publicar: repite el ultimo
    return viejo, nuevo.reset_index().rename(columns={"YR": "anio"})


if __name__ == "__main__":
    lineas = []
    bk = RES / "backups_pre_actualizacion_14oct2026"
    bk.mkdir(parents=True, exist_ok=True)
    for a in ARCH:
        shutil.copy2(PROC / a, bk / a)
    shutil.copy2(EXT / "oni_index.csv", bk / "oni_index.csv")

    # 1) ONI
    viejo, nuevo = oni_noaa()
    dif = (nuevo.set_index("anio")[TEMP] - viejo[TEMP]).round(2)
    lineas.append("ONI (NOAA - archivo anterior), trimestres distintos:")
    for anio, fila in dif.iterrows():
        cambios = {t: v for t, v in fila.items() if abs(v) > 1e-9}
        if cambios:
            lineas.append(f"  {anio}: " + ", ".join(f"{t} {viejo.loc[anio, t]} -> {nuevo.set_index('anio').loc[anio, t]}" for t in cambios))
    nuevo.to_csv(EXT / "oni_index.csv", index=False)

    # 2) 1-5 de agosto de 2026
    ini, fin = pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-05")
    precio = bajar("PrecBolsNaci", "hourly", ini, fin)
    demanda = bajar("DemaReal", "hourly", ini, fin)
    generacion = bajar("Gene", "hourly", ini, fin)
    volumen = bajar("VoluUtilDiarEner", "daily", pd.Timestamp("2026-07-25"), fin)
    aportes = bajar("AporEner", "daily", pd.Timestamp("2026-07-25"), fin)
    m26 = pd.read_csv(PROC / "dataset_maestro_2026.csv", parse_dates=["fecha_hora"])
    idx = pd.DatetimeIndex(m26.loc[m26["fecha_hora"] >= ini, "fecha_hora"])
    assert len(idx) == 120 and precio.reindex(idx).notna().all(), "faltan horas del 1-5 de agosto"
    interp = lambda s: s.reindex(s.index.union(idx)).interpolate(method="linear", limit_direction="both").reindex(idx)
    nuevos = {"precio_bolsa": precio.reindex(idx), "demanda": demanda.reindex(idx), "generacion": generacion.reindex(idx),
              "volumen_embalses": interp(volumen), "aportes_hidricos": interp(aportes)}
    sel = m26["fecha_hora"] >= ini
    lineas.append("1-5 de agosto de 2026 (120 h), revision de XM frente al archivo anterior:")
    for c, s in nuevos.items():
        antes = m26.loc[sel, c].to_numpy()
        d = np.abs(s.to_numpy() - antes) / np.abs(antes).clip(1e-9)
        lineas.append(f"  {c:18s} horas que cambian {int((d > 1e-6).sum()):3d}, diferencia relativa mediana {np.median(d) * 100:.3f} %, maxima {d.max() * 100:.2f} %")
        m26.loc[sel, c] = s.to_numpy()
    lineas.append(f"  precio medio 1-5 ago: {pd.read_csv(bk / 'dataset_maestro_2026.csv')['precio_bolsa'].tail(120).mean():.2f} -> {m26.loc[sel, 'precio_bolsa'].mean():.2f}")

    # 3) oni de los maestros y variables
    m19 = pd.read_csv(PROC / "dataset_maestro_2019_2025.csv", parse_dates=["fecha_hora"])
    m = pd.concat([m19, m26], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    oni_antes = m["oni"].to_numpy().copy()
    m["oni"] = oni_causal(m["fecha_hora"])
    cambia = np.abs(m["oni"].to_numpy() - oni_antes) > 1e-9
    lineas.append(f"Columna oni de los maestros: cambian {cambia.sum()} horas de {len(m)} "
                  f"(2026: {cambia[m['fecha_hora'].dt.year.to_numpy() == 2026].sum()}); oni del 5-ago 23:00 {oni_antes[-1]} -> {m['oni'].iloc[-1]:.3f}")
    col_m = list(m19.columns)
    m.loc[m["fecha_hora"].dt.year <= 2025, col_m].to_csv(PROC / "dataset_maestro_2019_2025.csv", index=False)
    m.loc[m["fecha_hora"].dt.year == 2026, list(m26.columns)].to_csv(PROC / "dataset_maestro_2026.csv", index=False)
    f = features(m.drop(columns=[c for c in ("es_pandemia",) if c not in m26.columns]))
    for a, cond in (("dataset_features_2019_2025.csv", f["fecha_hora"].dt.year <= 2025), ("dataset_features_2026.csv", f["fecha_hora"].dt.year == 2026)):
        viejo_f = pd.read_csv(bk / a, parse_dates=["fecha_hora"])
        nuevo_f = f.loc[cond].reset_index(drop=True)
        faltan = [c for c in viejo_f.columns if c not in nuevo_f.columns]
        assert not faltan, f"columnas que no reproduce el procedimiento: {faltan}"
        nuevo_f = nuevo_f[viejo_f.columns]
        # el archivo original empieza donde las ventanas de 30 dias ya estan completas
        nuevo_f = nuevo_f[nuevo_f["fecha_hora"].isin(viejo_f["fecha_hora"])].reset_index(drop=True)
        assert len(nuevo_f) == len(viejo_f), (a, len(nuevo_f), len(viejo_f))
        num = [c for c in viejo_f.columns if c != "fecha_hora"]
        dd = np.abs(nuevo_f[num].to_numpy(float) - viejo_f[num].to_numpy(float))
        fuera = ~(nuevo_f["fecha_hora"] >= ini).to_numpy()
        cols_cambian = [c for j, c in enumerate(num) if np.nanmax(dd[:, j]) > 1e-4]
        sin_oni = [j for j, c in enumerate(num) if c != "oni"]
        lineas.append(f"{a}: columnas que cambian {cols_cambian}; maxima diferencia fuera del 1-5 ago sin contar oni "
                      f"{np.nanmax(dd[fuera][:, sin_oni]):.2e}")
        nuevo_f.to_csv(PROC / a, index=False)
    texto = "\n".join(lineas)
    (RES / "actualizacion_datos_14oct2026.txt").write_text(texto, encoding="utf-8")
    print(texto)
