# -*- coding: utf-8 -*-
"""
Disponibilidad declarada de las ofertas iniciales de XM (OFERTAS/INICIAL/OFEImmdd.txt, biblioteca publica del portal).

Hallazgo del 14-oct-2026: el archivo del dia D se crea el dia D-1 hacia las 08:10 (cierre de ofertas a las 08:00,
Codigo de Operacion, Resolucion CREG 025 de 1995) pero en ese momento SOLO trae la disponibilidad declarada por unidad
y hora (lineas "D"); los precios de oferta (lineas "P") se agregan cuando XM reescribe el archivo el primer dia del
mes siguiente (los 14 archivos de octubre de 2026 revisados el 14-oct: 0 lineas P; el del 1-oct, ya reescrito: 77).
Por eso aqui solo se usa la disponibilidad, que si es publica a tiempo. Se suma para todo el sistema (sin desglose
por tipo de recurso, limitacion OE1-3). Advertencia: el historico disponible es el archivo reescrito; no se puede
comprobar que la disponibilidad no cambie en la reescritura.
Uso: descargar_disponibilidad_ofei.py [desde YYYY-MM] [hasta YYYY-MM] [inverso]
Salida: data/processed/disponibilidad_ofei_xm.csv (fecha_hora, disp_total_mw, unidades, lineas_precio, creado)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import descargar_predespacho_ideal as dpi  # noqa: E402
from descargar_despacho_programado import listar  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RUTA = "/M:/InformacionAgentes/Usuarios/Publico/OFERTAS/INICIAL"
CACHE = RAIZ / "data" / "raw" / "ofei_disponibilidad"
SALIDA = RAIZ / "data" / "processed" / "disponibilidad_ofei_xm.csv"


def leer(texto, dia):
    tot, n, n_p = np.zeros(24), 0, 0
    for linea in texto.splitlines():
        partes = [x.strip() for x in linea.split(",")]
        if len(partes) < 3:
            continue
        if partes[1] == "P":
            n_p += 1
        if partes[1] != "D":
            continue
        v = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", ",".join(partes[2:]))]
        if len(v) >= 24:
            tot += np.array(v[:24])
            n += 1
    return pd.DataFrame({"fecha_hora": pd.date_range(dia, periods=24, freq="h"), "disp_total_mw": tot, "unidades": n, "lineas_precio": n_p})


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    args = [a for a in sys.argv[1:] if a != "inverso"]
    desde, hasta = (args + ["2019-01", "2099-12"])[:2] if len(args) < 2 else args[:2]
    meses = sorted(f["nombre"] for f in dpi.listar(RUTA) if re.fullmatch(r"\d{4}-\d{2}", f["nombre"]) and desde <= f["nombre"] <= hasta)
    if "inverso" in sys.argv:
        meses = meses[::-1]
    for mes in meses:
        cache = CACHE / f"ofei_{mes}.csv"
        if cache.exists() and mes < pd.Timestamp.now().strftime("%Y-%m"):
            continue
        fs = [f for f in listar(f"{RUTA}/{mes}", "OFEI") if re.fullmatch(r"(?i)ofei\d{4}\.txt", f["nombre"])]
        if not fs:
            continue
        pub = {f["nombre"].lower(): f["fechaCreacion"] for f in fs}
        # en grupos de 8 archivos: el mes completo en un solo zip a veces agota el tiempo del servidor (504)
        zips = [dpi.descargar_zip([f["id"] for f in fs[i:i + 8]]) for i in range(0, len(fs), 8)]
        en_zip = {Path(n).name.lower() for z in zips for n in z.namelist()}
        zips += [dpi.descargar_zip([f["id"]]) for f in fs if f["nombre"].lower() not in en_zip]
        dfs = []
        for nombre, zz in [(n, z) for z in zips for n in z.namelist()]:
            m = re.fullmatch(r"(?i)ofei(\d{2})(\d{2})\.txt", Path(nombre).name)
            if not m:
                continue
            anio = int(mes[:4]) - (1 if (mes.endswith("-01") and int(m.group(1)) == 12) else 0)
            d = leer(zz.read(nombre).decode("latin-1"), pd.Timestamp(anio, int(m.group(1)), int(m.group(2))))
            d["creado"] = pub.get(Path(nombre).name.lower())
            dfs.append(d)
        pd.concat(dfs).sort_values("fecha_hora").to_csv(cache, index=False)
        print(f"  {mes}: {len(dfs)} dias", flush=True)
    t = pd.concat([pd.read_csv(f, parse_dates=["fecha_hora"]) for f in sorted(CACHE.glob("ofei_*.csv"))])
    t = t.drop_duplicates("fecha_hora").sort_values("fecha_hora")
    corte = t["fecha_hora"].dt.normalize() - pd.Timedelta(hours=1)
    t["a_tiempo"] = pd.to_datetime(t["creado"].str[:19]) < corte          # creado antes de las 23:00 de D-1
    t.to_csv(SALIDA, index=False)
    print("guardado", SALIDA, len(t), t["fecha_hora"].min(), t["fecha_hora"].max(), "| a tiempo:", round(t["a_tiempo"].mean() * 100, 2), "%")
