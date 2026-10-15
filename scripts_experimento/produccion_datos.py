# -*- coding: utf-8 -*-
"""
Datos de produccion para la operacion diaria del dashboard (desde el 14-oct-2026).

El dataset oficial del proyecto (data/processed/dataset_maestro_*.csv y dataset_features_*.csv, ene-2019 a 5-ago-2026,
el periodo del Anexo 1) NO se modifica: las cifras de los informes siguen siendo reproducibles. Lo publicado despues va
a data/produccion/ y se une al oficial solo para pronosticar (maestro_produccion()):
  * serie_xm.csv: precio de bolsa, demanda y generacion horarios, volumen de embalses y aportes (diarios, interpolados a
    horas como en el notebook 02) y ONI (rampa causal del notebook 02) desde el 6-ago-2026, tal como esten publicados en
    la API de XM el dia de la corrida. Se descarga completo en cada corrida (pocas consultas) para recoger las revisiones
    de XM; si una consulta falla, se conserva lo que ya habia.
  * oni_index.csv: tabla ONI de NOAA CPC (los trimestres aun sin publicar repiten el ultimo, como el archivo del proyecto)
  * predespacho ideal (iMAR) y dMAR/dDEM: se agregan a data/processed/predespacho_ideal_xm.csv y
    despacho_programado_cm_demanda_xm.csv los dias nuevos (se vuelven a leer los meses del ultimo dia guardado a manana),
    con la marca a_tiempo (archivo creado y modificado antes de las 23:00 del dia anterior; validar_predespacho_ideal.py)
Uso: python produccion_datos.py   (lo llama actualizar_produccion.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import actualizar_datos_oct2026 as ad  # noqa: E402
import descargar_dmar_ddem as ddm  # noqa: E402
import descargar_predespacho_ideal as dpi  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "data" / "processed"
PROD = RAIZ / "data" / "produccion"
SERIE = PROD / "serie_xm.csv"
ONI = PROD / "oni_index.csv"
INICIO = pd.Timestamp("2026-08-06")              # primera hora despues del dataset oficial
TEMP = ad.TEMP


def hoy():
    """Fecha y hora de Colombia (el runner de GitHub esta en UTC)."""
    return pd.Timestamp.now(tz="America/Bogota").tz_localize(None)


def oni_tabla():
    txt = requests.get("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", timeout=60).text
    filas = [l.split() for l in txt.splitlines()[1:] if l.strip()]
    n = pd.DataFrame(filas, columns=["SEAS", "YR", "TOTAL", "ANOM"]).astype({"YR": int, "ANOM": float})
    n["k"] = n["YR"] * 12 + n["SEAS"].map({t: i for i, t in enumerate(TEMP)})
    s = n.set_index("k")["ANOM"].round(1)
    fin = (hoy() + pd.Timedelta(days=4)).year
    s = s.reindex(range(2018 * 12, (fin + 1) * 12)).ffill()      # trimestres sin publicar: repite el ultimo
    t = pd.DataFrame({"anio": s.index // 12, "t": [TEMP[k % 12] for k in s.index], "oni": s.to_numpy()})
    return t.pivot(index="anio", columns="t", values="oni")[TEMP].reset_index()


def actualizar_serie():
    fin = hoy().normalize()
    horarias = {c: ad.bajar(m, "hourly", INICIO, fin) for c, m in
                (("precio_bolsa", "PrecBolsNaci"), ("demanda", "DemaReal"), ("generacion", "Gene"))}
    diarias = {c: ad.bajar(m, "daily", INICIO - pd.Timedelta(days=12), fin) for c, m in
               (("volumen_embalses", "VoluUtilDiarEner"), ("aportes_hidricos", "AporEner"))}
    ultimo = max([s.index.max() for s in horarias.values()] + [s.index.max() + pd.Timedelta(hours=23) for s in diarias.values()])
    idx = pd.date_range(INICIO, ultimo, freq="h")
    n = pd.DataFrame(index=idx)
    for c, s in horarias.items():
        n[c] = s.reindex(idx)
    for c, s in diarias.items():      # notebook 02: diaria a las 00:00 e interpolacion lineal a horas
        n[c] = s.reindex(s.index.union(idx)).interpolate(method="linear", limit_direction="both").reindex(idx)
        n.loc[idx > s.index.max() + pd.Timedelta(hours=23), c] = np.nan
    n["oni"] = ad.oni_causal(pd.Series(idx), ruta=ONI)
    n["es_pandemia"] = 0
    n.index.name = "fecha_hora"
    if SERIE.exists():                # lo que no llego en esta corrida se conserva
        viejo = pd.read_csv(SERIE, parse_dates=["fecha_hora"], float_precision="round_trip").set_index("fecha_hora")
        n = n.combine_first(viejo)[n.columns]
    n.to_csv(SERIE)
    return {c: str(n[c].last_valid_index()) for c in ("precio_bolsa", "volumen_embalses")}


def meses_a_releer(ultimo):
    return [m.strftime("%Y-%m") for m in pd.period_range((ultimo - pd.Timedelta(days=2)).to_period("M"),
                                                         (hoy() + pd.Timedelta(days=1)).to_period("M"), freq="M")]


def actualizar_imar():
    ruta = PROC / "predespacho_ideal_xm.csv"
    t = pd.read_csv(ruta, parse_dates=["fecha_hora"], float_precision="round_trip")
    nuevos = [x for x in (dpi.leer_mes(m) for m in meses_a_releer(t["fecha_hora"].max())) if x is not None]
    if not nuevos:
        return str(t["fecha_hora"].max())
    n = pd.concat(nuevos).drop_duplicates("fecha_hora")
    corte = n["fecha_hora"].dt.normalize() - pd.Timedelta(hours=1)
    mod = pd.to_datetime(n["modificado"].str[:19])
    n["a_tiempo"] = (pd.to_datetime(n["publicado"].str[:19]) < corte) & (mod.isna() | (mod < corte))
    dias = n["fecha_hora"].dt.normalize().unique()
    t = pd.concat([t[~t["fecha_hora"].dt.normalize().isin(dias)], n[t.columns]]).sort_values("fecha_hora")
    t.to_csv(ruta, index=False)
    return str(t.loc[t["a_tiempo"], "fecha_hora"].max())


def actualizar_dmar_ddem():
    ruta = PROC / "despacho_programado_cm_demanda_xm.csv"
    t = pd.read_csv(ruta, parse_dates=["fecha_hora"], float_precision="round_trip")
    nuevos = [x for x in (ddm.leer_mes(m) for m in meses_a_releer(t["fecha_hora"].max())) if x is not None]
    if not nuevos:
        return str(t["fecha_hora"].max())
    n = ddm.marcar_a_tiempo(pd.concat(nuevos).drop_duplicates("fecha_hora").sort_values("fecha_hora"))
    dias = n["fecha_hora"].dt.normalize().unique()
    t = pd.concat([t[~t["fecha_hora"].dt.normalize().isin(dias)], n[t.columns]]).sort_values("fecha_hora")
    t.to_csv(ruta, index=False)
    return str(t.loc[t["a_tiempo_mar"], "fecha_hora"].max())


def maestro_produccion():
    """Maestro oficial (hasta el 5-ago-2026) + serie de produccion (desde el 6-ago), horas consecutivas."""
    m = pd.concat([pd.read_csv(PROC / f, parse_dates=["fecha_hora"])
                   for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]).drop_duplicates("fecha_hora")
    if SERIE.exists():
        s = pd.read_csv(SERIE, parse_dates=["fecha_hora"], float_precision="round_trip")
        m = pd.concat([m, s[s["fecha_hora"] > m["fecha_hora"].max()][m.columns]])
    return m.sort_values("fecha_hora").reset_index(drop=True)


def actualizar():
    """Corre las cuatro actualizaciones; un fallo en una fuente no detiene las demas. Devuelve el estado."""
    PROD.mkdir(parents=True, exist_ok=True)
    estado = {}
    pasos = [("oni", lambda: oni_tabla().to_csv(ONI, index=False) or "ok"), ("serie_xm", actualizar_serie),
             ("predespacho_ideal", actualizar_imar), ("dmar_ddem", actualizar_dmar_ddem)]
    for nombre, f in pasos:
        try:
            estado[nombre] = f()
        except Exception as e:                         # noqa: BLE001
            estado[nombre] = f"ERROR: {str(e)[:200]}"
        print(f"  {nombre}: {estado[nombre]}", flush=True)
    if not ONI.exists():
        raise RuntimeError("sin tabla ONI de produccion")
    return estado


if __name__ == "__main__":
    actualizar()
