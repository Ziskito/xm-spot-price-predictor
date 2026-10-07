# -*- coding: utf-8 -*-
"""
Descarga el predespacho ideal de XM (archivos iMARmmdd.txt) desde la biblioteca publica del portal.

Hallazgo del 7-oct-2026: XM publica sin usuario, en https://www.xm.com.co/generacion/informes-despacho/predespacho-ideal,
para cada dia D y el DIA ANTERIOR (D-1, entre las 10:00 y las 12:00), el archivo iMAR con tres filas por hora:
  * "Costo Marginal": precio de oferta del recurso marginal en el predespacho ideal ($/MWh)
  * "MPO": maximo precio ofertado horario ($/MWh)
  * "Delta" ($/MWh)
El predespacho ideal usa las ofertas reales del dia D y la demanda pronosticada por XM, asi que es una
estimacion ex ante del precio de bolsa, disponible antes de nuestro corte (00:00 de D).

La biblioteca se consulta por la API publica del portal (la misma que usa la pagina):
  listar:    GET https://api-portalxm.xm.com.co/administracion-archivos/ficheros?ruta=...&contenedor=storageportalxm
  descargar: GET https://api-portalxm.xm.com.co/administracion-archivos/ficheros/descargar-archivos?ficheros=<ids>&nombreBlobContainer=storageportalxm
             (devuelve un zip; con varios ids, un solo zip por mes)
Salida: data/processed/predespacho_ideal_xm.csv (fecha_hora, costo_marginal, mpo, delta, publicado) en COP/kWh
"""
import io
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
API = "https://api-portalxm.xm.com.co/administracion-archivos"
RUTA = "/M:/InformacionAgentes/Usuarios/Publico/PredespachoIdeal"
CACHE = RAIZ / "data" / "raw" / "predespacho_ideal"
SALIDA = RAIZ / "data" / "processed" / "predespacho_ideal_xm.csv"


def listar(ruta, n=200):
    for intento in range(4):
        try:
            r = requests.get(f"{API}/ficheros", params={"ruta": ruta, "contenedor": "storageportalxm", "ordenarPor": "nombre",
                                                         "orden": "DESC", "pagina": 1, "resultadosPorPagina": n, "nombre": ""}, timeout=60)
            r.raise_for_status()
            return r.json().get("ficheros") or []
        except Exception as e:
            print(f"  reintento listar {ruta}: {str(e)[:80]}", flush=True)
            time.sleep(5)
    raise RuntimeError(ruta)


def descargar_zip(ids):
    for intento in range(4):
        try:
            r = requests.get(f"{API}/ficheros/descargar-archivos", timeout=180,
                             params={"ficheros": ",".join(map(str, ids)), "nombreBlobContainer": "storageportalxm"})
            r.raise_for_status()
            return zipfile.ZipFile(io.BytesIO(r.content))
        except Exception as e:
            print(f"  reintento descarga: {str(e)[:80]}", flush=True)
            time.sleep(5)
    raise RuntimeError("descarga")


def leer_imar(texto, dia):
    filas = {}
    for linea in texto.splitlines():
        # etiqueta entre comillas y 24 numeros; tolera la coma faltante tras la etiqueta (p. ej. iMAR0223 de 2023)
        m = re.match(r'\s*"([^"]+)"\s*,?(.*)', linea)
        if not m:
            continue
        numeros = re.findall(r"-?\d+(?:\.\d+)?", m.group(2))
        if len(numeros) >= 24:
            filas[m.group(1).strip().lower()] = [float(x) / 1000 for x in numeros[:24]]   # $/MWh -> COP/kWh
    horas = pd.date_range(dia, periods=24, freq="h")
    return pd.DataFrame({"fecha_hora": horas, "costo_marginal": filas.get("costo marginal"),
                         "mpo": filas.get("mpo"), "delta": filas.get("delta")})


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    meses = sorted(f["nombre"] for f in listar(RUTA) if re.fullmatch(r"\d{4}-\d{2}", f["nombre"]))
    print(f"{len(meses)} meses: {meses[0]} a {meses[-1]}", flush=True)
    partes = []
    for mes in meses:
        ruta_cache = CACHE / f"imar_{mes}.csv"
        if ruta_cache.exists() and mes < pd.Timestamp.now().strftime("%Y-%m"):
            partes.append(pd.read_csv(ruta_cache, parse_dates=["fecha_hora"]))
            continue
        # hasta abr-2025 el archivo se llama iMARmmdd_NAL.TXT; desde may-2025, iMARmmdd.txt
        fs = [f for f in listar(f"{RUTA}/{mes}") if re.fullmatch(r"(?i)imar\d{4}(_NAL)?\.txt", f["nombre"])]
        if not fs:
            continue
        publicado = {f["nombre"].lower(): f["fechaCreacion"] for f in fs}
        z = descargar_zip([f["id"] for f in fs])
        anio = int(mes[:4])
        dfs = []
        # el zip de varios archivos a veces omite algunos (p. ej. 25-27 ene 2026): se piden uno por uno
        en_zip = {Path(n).name.lower() for n in z.namelist()}
        extra = [descargar_zip([f["id"]]) for f in fs if f["nombre"].lower() not in en_zip]
        archivos = [(n, z) for n in z.namelist()] + [(n, zz) for zz in extra for n in zz.namelist()]
        for nombre, z in archivos:
            m = re.fullmatch(r"(?i)imar(\d{2})(\d{2})(?:_NAL)?\.txt", Path(nombre).name)
            if not m:
                continue
            mm, dd = int(m.group(1)), int(m.group(2))
            dia = pd.Timestamp(anio - (1 if (mes.endswith("-01") and mm == 12) else 0), mm, dd)
            df = leer_imar(z.read(nombre).decode("latin-1"), dia)
            df["publicado"] = publicado.get(Path(nombre).name.lower())
            dfs.append(df)
        mesdf = pd.concat(dfs).sort_values("fecha_hora")
        mesdf.to_csv(ruta_cache, index=False)
        partes.append(mesdf)
        print(f"  {mes}: {len(dfs)} dias", flush=True)
    t = pd.concat(partes).drop_duplicates("fecha_hora").sort_values("fecha_hora")
    t.to_csv(SALIDA, index=False)
    print("guardado", SALIDA, len(t), t["fecha_hora"].min(), t["fecha_hora"].max())
