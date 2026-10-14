# -*- coding: utf-8 -*-
"""
Despacho programado de XM (archivos dDECmmdd.txt), publicado el dia anterior hacia las 14:30, desde la biblioteca
publica del portal (misma API que descargar_predespacho_ideal.py). Cada archivo trae, por planta y hora, los MW
programados. Se agrega por tipo de recurso (ListadoRecursos de la API de XM) y se guarda la hora de publicacion.
Salida: data/processed/despacho_programado_xm.csv (fecha_hora, prog_total, prog_hidraulica, prog_termica, prog_solar, prog_importacion,
        prog_eolica, prog_otro, n_termicas, publicado)
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import descargar_predespacho_ideal as dpi  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RUTA = "/M:/InformacionAgentes/Usuarios/Publico/DESPACHO"
CACHE = RAIZ / "data" / "raw" / "despacho_programado"
SALIDA = RAIZ / "data" / "processed" / "despacho_programado_xm.csv"


def listar(ruta, nombre):
    import requests
    for _ in range(4):
        try:
            r = requests.get(f"{dpi.API}/ficheros", timeout=60, params={"ruta": ruta, "contenedor": "storageportalxm", "ordenarPor": "nombre",
                                                                      "orden": "ASC", "pagina": 1, "resultadosPorPagina": 400, "nombre": nombre}).json()
            return r.get("ficheros") or []
        except Exception:
            pass
    return []


def tipos():
    from pydataxm.pydataxm import ReadDB
    rec = ReadDB().request_data("ListadoRecursos", "Sistema", pd.Timestamp.now().date(), pd.Timestamp.now().date())
    norm = lambda s: re.sub(r"\s+", " ", str(s).upper().strip())
    return {norm(n): str(t).upper() for n, t in zip(rec["Values_Name"], rec["Values_Type"])}, norm


def clasificar(nombre, tipo, norm):
    """Tipo del recurso: nombre exacto en ListadoRecursos, luego prefijo comun (p. ej. 'AUTOG ARGOS YUMBO' vs
    'AUTOG ARGOS YUMBO 1'), luego palabras clave del nombre."""
    n = norm(nombre)
    if n.startswith(("AUTOG ", "AG_")):
        return "AUTOGENERADOR"
    if n in tipo:
        return tipo[n]
    if n.startswith("ECUADOR") or n.startswith("VENEZUELA"):
        return "IMPORTACION"
    cand = {t for k, t in tipo.items() if k.startswith(n + " ") or n.startswith(k + " ")}
    if len(cand) == 1:
        return cand.pop()
    for clave, t in (("SOLAR", "SOLAR"), ("EOLIC", "EOLICA"), ("TERMO", "TERMICA"), ("TERMICA", "TERMICA"), ("HIDRO", "HIDRAULICA"),
                     ("POWER PLANT", "TERMICA"), ("BIO", "COGENERADOR")):
        if clave in n:
            return t
    # archivos anteriores a ~2024: codigos de despacho (TEBSA, PAIPA1, MGUAVIO...) en vez de nombres
    if " " not in n.replace("FLORES IV", "FLORESIV"):
        if n.startswith(("VENEZUE",)):
            return "IMPORTACION"
        if n.startswith(TERMICAS_COD):
            return "TERMICA"
        if n.startswith("RN_"):
            return "SOLAR"
        if n == "MJEPIRAC":
            return "EOLICA"
        if n.startswith("AG_"):
            return "AUTOGENERADOR"
        if n.startswith(("CMA", "COINC", "CORIS", "CPRO", "CRIO", "CSAN", "C_")):
            return "COGENERADOR"
        return "HIDRAULICA"
    return "OTRO"


# codigos de despacho de plantas termicas (formato anterior a ~2024)
TERMICAS_COD = ("TEBSA", "FLORES", "GUAJIR", "TCENTRO", "PAIPA", "TASAJER", "TCANDEL", "TEMCALI", "TSIERRA", "TVALLE", "TYOPAL",
                "ZIPAEMG", "GECELCA", "BARRANQ", "CTGEMG", "PROELECT", "TPIEDRAS", "MTMECHERO", "MTBOLIVAR", "MTCAPACHOS",
                "MDONAJUANA", "MERILEC", "COROZO", "DORADA", "TMERILEC", "TERMO", "TFLORES", "TPROELEC", "TCARIBE")


def leer(texto, dia, tipo, norm):
    filas = []
    for linea in texto.splitlines():
        m = re.match(r'\s*"([^"]+)"\s*,?(.*)', linea)
        if not m or norm(m.group(1)) == "TOTAL":
            continue
        num = re.findall(r"-?\d+(?:\.\d+)?", m.group(2))
        if len(num) >= 24:
            filas.append((clasificar(m.group(1), tipo, norm), [float(x) for x in num[:24]]))
    df = pd.DataFrame({"fecha_hora": pd.date_range(dia, periods=24, freq="h")})
    imp = [x for tt, x in filas if tt == "IMPORTACION"]
    df["prog_importacion"] = np.sum(imp, axis=0) if imp else 0.0
    filas = [(tt, x) for tt, x in filas if tt != "IMPORTACION"]
    for t in ("HIDRAULICA", "TERMICA", "SOLAR", "EOLICA", "OTRO"):
        v = [x for tt, x in filas if tt == t or (t == "OTRO" and tt not in ("HIDRAULICA", "TERMICA", "SOLAR", "EOLICA"))]
        df[f"prog_{t.lower()}"] = np.sum(v, axis=0) if v else 0.0
    df["n_termicas"] = np.sum([np.array(x) > 0 for tt, x in filas if tt == "TERMICA"], axis=0) if any(tt == "TERMICA" for tt, _ in filas) else 0
    df["prog_total"] = df[[c for c in df.columns if c.startswith("prog_") and c != "prog_importacion"]].sum(axis=1)
    return df


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    tipo, norm = tipos()
    meses = sorted(f["nombre"] for f in dpi.listar(RUTA) if re.fullmatch(r"\d{4}-\d{2}", f["nombre"]) and f["nombre"] >= "2018-01")
    partes = []
    for mes in meses:
        cache = CACHE / f"ddec_{mes}.csv"
        if cache.exists() and mes < pd.Timestamp.now().strftime("%Y-%m"):
            partes.append(pd.read_csv(cache, parse_dates=["fecha_hora"]))
            continue
        fs = [f for f in listar(f"{RUTA}/{mes}", "dDEC") if re.fullmatch(r"(?i)ddec\d{4}\.txt", f["nombre"])]
        if not fs:
            continue
        pub = {f["nombre"].lower(): (f["fechaCreacion"], f["fechaModificacion"]) for f in fs}
        z = dpi.descargar_zip([f["id"] for f in fs])
        en_zip = {Path(n).name.lower() for n in z.namelist()}
        extra = [dpi.descargar_zip([f["id"]]) for f in fs if f["nombre"].lower() not in en_zip]
        dfs = []
        for nombre, zz in [(n, z) for n in z.namelist()] + [(n, e) for e in extra for n in e.namelist()]:
            m = re.fullmatch(r"(?i)ddec(\d{2})(\d{2})\.txt", Path(nombre).name)
            if not m:
                continue
            anio = int(mes[:4]) - (1 if (mes.endswith("-01") and int(m.group(1)) == 12) else 0)
            crudo = zz.read(nombre)
            try:
                texto = crudo.decode("utf-8")
            except UnicodeDecodeError:
                texto = crudo.decode("latin-1")
            d = leer(texto,pd.Timestamp(anio, int(m.group(1)), int(m.group(2))), tipo, norm)
            c, mo = pub.get(Path(nombre).name.lower(), (None, None))
            d["creado"], d["modificado"] = c, mo
            dfs.append(d)
        mdf = pd.concat(dfs).sort_values("fecha_hora")
        mdf.to_csv(cache, index=False)
        partes.append(mdf)
        print(f"  {mes}: {len(dfs)} dias", flush=True)
    t = pd.concat(partes).drop_duplicates("fecha_hora").sort_values("fecha_hora")
    corte = t["fecha_hora"].dt.normalize() - pd.Timedelta(hours=1)
    # XM toco en bloque todos los archivos anteriores a 2025 el 13-feb-2025 (2468 de 2473 modificaciones tardias son
    # de ese dia, con la fecha de creacion intacta): migracion del repositorio, no revision del contenido
    mod = pd.to_datetime(t["modificado"].str[:19])
    t["a_tiempo"] = (pd.to_datetime(t["creado"].str[:19]) < corte) & ((mod < corte) | (mod.dt.normalize() == pd.Timestamp("2025-02-13")))
    t.to_csv(SALIDA, index=False)
    print("guardado", SALIDA, len(t), t["fecha_hora"].min(), t["fecha_hora"].max(), "| a tiempo:", round(t["a_tiempo"].mean() * 100, 2), "%",
          "| sin tipo (OTRO) medio MW:", round(t["prog_otro"].mean(), 1))
