# -*- coding: utf-8 -*-
"""
Redespacho de XM (rDECmmdd.txt, carpeta Redespacho de la biblioteca publica del portal): generacion redespachada por
planta y hora del dia D, con la ultima modificacion hacia las 21:40 del mismo dia D, antes del corte de las 00:00 de
D+1. Refleja los ajustes por la demanda y la disponibilidad reales del dia. Se agrega igual que el despacho
programado (descargar_despacho_programado.leer: total y por tipo de recurso).
Uso: descargar_redespacho.py [desde YYYY-MM] [hasta YYYY-MM]
Salida: data/processed/redespacho_xm.csv (fecha_hora, red_total, red_hidraulica, red_termica, ..., creado, modificado,
        a_tiempo = modificado antes de las 00:00 del dia siguiente)
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import descargar_predespacho_ideal as dpi  # noqa: E402
from descargar_despacho_programado import leer, listar, tipos  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RUTA = "/M:/InformacionAgentes/Usuarios/Publico/Redespacho"
CACHE = RAIZ / "data" / "raw" / "redespacho"
SALIDA = RAIZ / "data" / "processed" / "redespacho_xm.csv"

if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    desde = sys.argv[1] if len(sys.argv) > 1 else "2020-01"
    hasta = sys.argv[2] if len(sys.argv) > 2 else "2099-12"
    tipo, norm = tipos()
    meses = sorted(f["nombre"] for f in dpi.listar(RUTA) if re.fullmatch(r"\d{4}-\d{2}", f["nombre"]) and desde <= f["nombre"] <= hasta)
    if "inverso" in sys.argv:
        meses = meses[::-1]
    for mes in meses:
        cache = CACHE / f"rdec_{mes}.csv"
        if cache.exists() and mes < pd.Timestamp.now().strftime("%Y-%m"):
            continue
        fs = [f for f in listar(f"{RUTA}/{mes}", "rDEC") if re.fullmatch(r"(?i)rdec\d{4}\.txt", f["nombre"])]
        if not fs:
            continue
        pub = {f["nombre"].lower(): (f["fechaCreacion"], f["fechaModificacion"]) for f in fs}
        z = dpi.descargar_zip([f["id"] for f in fs])
        en_zip = {Path(n).name.lower() for n in z.namelist()}
        extra = [dpi.descargar_zip([f["id"]]) for f in fs if f["nombre"].lower() not in en_zip]
        dfs = []
        for nombre, zz in [(n, z) for n in z.namelist()] + [(n, e) for e in extra for n in e.namelist()]:
            m = re.fullmatch(r"(?i)rdec(\d{2})(\d{2})\.txt", Path(nombre).name)
            if not m:
                continue
            anio = int(mes[:4]) - (1 if (mes.endswith("-01") and int(m.group(1)) == 12) else 0)
            crudo = zz.read(nombre)
            try:
                texto = crudo.decode("utf-8")
            except UnicodeDecodeError:
                texto = crudo.decode("latin-1")
            d = leer(texto, pd.Timestamp(anio, int(m.group(1)), int(m.group(2))), tipo, norm)
            d.columns = [c.replace("prog_", "red_") for c in d.columns]
            c, mo = pub.get(Path(nombre).name.lower(), (None, None))
            d["creado"], d["modificado"] = c, mo
            dfs.append(d)
        pd.concat(dfs).sort_values("fecha_hora").to_csv(cache, index=False)
        print(f"  {mes}: {len(dfs)} dias", flush=True)
    t = pd.concat([pd.read_csv(f, parse_dates=["fecha_hora"]) for f in sorted(CACHE.glob("rdec_*.csv"))])
    t = t.drop_duplicates("fecha_hora").sort_values("fecha_hora")
    corte = t["fecha_hora"].dt.normalize() + pd.Timedelta(days=1)        # usable al corte de las 00:00 de D+1
    t["a_tiempo"] = pd.to_datetime(t["modificado"].str[:19]) < corte
    t.to_csv(SALIDA, index=False)
    print("guardado", SALIDA, len(t), t["fecha_hora"].min(), t["fecha_hora"].max(), "| a tiempo:", round(t["a_tiempo"].mean() * 100, 2), "%")
