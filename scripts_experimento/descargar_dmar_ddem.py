# -*- coding: utf-8 -*-
"""
Costo marginal del despacho programado (dMARmmdd.txt) y demanda pronosticada del despacho (dDEMmmdd.txt) de XM,
publicados el dia anterior hacia las 14:30 en la biblioteca publica del portal (carpeta DESPACHO, misma API que
descargar_predespacho_ideal.py). dMAR trae una fila "Costo Marginal" con 24 valores ($/MWh); dDEM, la demanda
pronosticada por subarea y hora (MW), que se suma para el sistema.
Salida: data/processed/despacho_programado_cm_demanda_xm.csv (fecha_hora, cm_programado, demanda_programada,
        creado_mar, modificado_mar, creado_dem, modificado_dem, a_tiempo_mar, a_tiempo_dem)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import descargar_predespacho_ideal as dpi  # noqa: E402
from descargar_despacho_programado import RUTA, listar  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
CACHE = RAIZ / "data" / "raw" / "despacho_cm_demanda"
SALIDA = RAIZ / "data" / "processed" / "despacho_programado_cm_demanda_xm.csv"


def numeros(linea):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", linea.split(",", 1)[1])] if "," in linea else []


def leer_mar(texto):
    for linea in texto.splitlines():
        if "costo marginal" in linea.lower():
            v = numeros(linea)
            if len(v) >= 24:
                return np.array(v[:24]) / 1000.0
    return None


def leer_dem(texto):
    filas = [numeros(l) for l in texto.splitlines() if l.strip() and not l.lower().startswith(("total", '"total'))]
    filas = [v[:24] for v in filas if len(v) >= 24]
    return np.sum(filas, axis=0) if filas else None


def texto(zz, nombre):
    crudo = zz.read(nombre)
    try:
        return crudo.decode("utf-8")
    except UnicodeDecodeError:
        return crudo.decode("latin-1")


def leer_mes(mes):
    """Costo marginal y demanda programados de los dias de un mes (carpeta AAAA-MM de XM); None si no hay archivos."""
    fs = [f for f in listar(f"{RUTA}/{mes}", "dMAR") + listar(f"{RUTA}/{mes}", "dDEM")
          if re.fullmatch(r"(?i)d(mar|dem)\d{4}\.txt", f["nombre"])]
    if not fs:
        return None
    pub = {f["nombre"].lower(): (f["fechaCreacion"], f["fechaModificacion"]) for f in fs}
    z = dpi.descargar_zip([f["id"] for f in fs])
    en_zip = {Path(n).name.lower() for n in z.namelist()}
    extra = [dpi.descargar_zip([f["id"]]) for f in fs if f["nombre"].lower() not in en_zip]
    dias = {}
    for nombre, zz in [(n, z) for n in z.namelist()] + [(n, e) for e in extra for n in e.namelist()]:
        m = re.fullmatch(r"(?i)d(mar|dem)(\d{2})(\d{2})\.txt", Path(nombre).name)
        if not m:
            continue
        anio = int(mes[:4]) - (1 if (mes.endswith("-01") and int(m.group(2)) == 12) else 0)
        dia = pd.Timestamp(anio, int(m.group(2)), int(m.group(3)))
        tipo = m.group(1).lower()
        v = leer_mar(texto(zz, nombre)) if tipo == "mar" else leer_dem(texto(zz, nombre))
        if v is None:
            continue
        c, mo = pub.get(Path(nombre).name.lower(), (None, None))
        dias.setdefault(dia, {})[tipo] = (v, c, mo)
    filas = []
    for dia, x in sorted(dias.items()):
        df = pd.DataFrame({"fecha_hora": pd.date_range(dia, periods=24, freq="h")})
        for tipo, col in (("mar", "cm_programado"), ("dem", "demanda_programada")):
            v, c, mo = x.get(tipo, (np.full(24, np.nan), None, None))
            df[col], df[f"creado_{tipo}"], df[f"modificado_{tipo}"] = v, c, mo
        filas.append(df)
    return pd.concat(filas) if filas else None


def marcar_a_tiempo(t):
    """a_tiempo_*: archivo creado y modificado antes de las 23:00 del dia anterior."""
    corte = t["fecha_hora"].dt.normalize() - pd.Timedelta(hours=1)
    for tipo in ("mar", "dem"):
        mod = pd.to_datetime(t[f"modificado_{tipo}"].str[:19])
        # 13-feb-2025: migracion en bloque del repositorio (ver descargar_despacho_programado.py)
        t[f"a_tiempo_{tipo}"] = ((pd.to_datetime(t[f"creado_{tipo}"].str[:19]) < corte)
                                 & ((mod < corte) | (mod.dt.normalize() == pd.Timestamp("2025-02-13"))))
    return t


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    meses = sorted(f["nombre"] for f in dpi.listar(RUTA) if re.fullmatch(r"\d{4}-\d{2}", f["nombre"]) and f["nombre"] >= "2018-01")
    if "inverso" in sys.argv:      # segundo proceso en paralelo, desde el final (comparten el cache por mes)
        meses = meses[::-1]
    partes = []
    for mes in meses:
        cache = CACHE / f"mar_dem_{mes}.csv"
        if cache.exists() and mes < pd.Timestamp.now().strftime("%Y-%m"):
            partes.append(pd.read_csv(cache, parse_dates=["fecha_hora"]))
            continue
        mdf = leer_mes(mes)
        if mdf is None:
            continue
        mdf.to_csv(cache, index=False)
        partes.append(mdf)
        print(f"  {mes}: {mdf['fecha_hora'].dt.normalize().nunique()} dias", flush=True)
    t = marcar_a_tiempo(pd.concat(partes).drop_duplicates("fecha_hora").sort_values("fecha_hora"))
    t.to_csv(SALIDA, index=False)
    print("guardado", SALIDA, len(t), t["fecha_hora"].min(), t["fecha_hora"].max(),
          "| a tiempo mar:", round(t["a_tiempo_mar"].mean() * 100, 2), "% dem:", round(t["a_tiempo_dem"].mean() * 100, 2), "%")
