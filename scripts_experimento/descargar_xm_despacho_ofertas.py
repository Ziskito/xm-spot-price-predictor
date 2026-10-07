# -*- coding: utf-8 -*-
"""
Descarga de variables de XM que podrian anticipar los escalones del precio (pruebas del 6-oct-2026):
  * GeneProgDesp   generacion programada por recurso (despacho del dia siguiente)
  * DispoDeclarada disponibilidad declarada por recurso (se reporta con la oferta)
  * PrecOferDesp   precio de oferta por recurso (la API lo publica con ~6 dias de rezago)
Se agregan por hora y tipo de recurso (hidraulica, termica, solar, eolica, ...) con ListadoRecursos.

OJO con la disponibilidad en tiempo real (consulta del 6-oct-2026 a las 21:49): la API tenia
GeneProgDesp hasta el 5-oct, DispoDeclarada hasta el 6-oct y PrecOferDesp hasta el 30-sep. Aqui se
mide si ayudarian SI estuvieran a tiempo; usarlas en operacion exige confirmar su hora de publicacion.
Salida: data/processed/xm_despacho_ofertas_2026.csv (una fila por hora)
"""
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydataxm.pydataxm import ReadDB

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "data" / "processed" / "xm_despacho_ofertas_2026.csv"
CACHE = RAIZ / "data" / "raw" / "xm_despacho_ofertas"
INICIO, FIN = dt.date(2025, 12, 1), dt.date(2026, 8, 6)
METRICAS = ("GeneProgDesp", "DispoDeclarada", "PrecOferDesp")


def bloques(ini, fin, dias=28):
    a = ini
    while a <= fin:
        b = min(a + dt.timedelta(days=dias - 1), fin)
        yield a, b
        a = b + dt.timedelta(days=1)


def descargar(api, metrica):
    partes = []
    for a, b in bloques(INICIO, FIN):
        ruta = CACHE / f"{metrica}_{a:%Y%m%d}_{b:%Y%m%d}.csv"
        if ruta.exists():
            partes.append(pd.read_csv(ruta))
            continue
        for intento in range(3):
            try:
                df = api.request_data(metrica, "Recurso", a, b)
                break
            except Exception as e:  # la API a veces corta la conexion
                print(f"  {metrica} {a}: reintento {intento + 1} ({str(e)[:80]})", flush=True)
                time.sleep(5)
        else:
            raise RuntimeError(f"no se pudo descargar {metrica} {a}-{b}")
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta, index=False)
        partes.append(df)
        print(f"  {metrica} {a} a {b}: {len(df)} filas", flush=True)
    return pd.concat(partes, ignore_index=True)


def a_largo(df):
    col = "Values_code" if "Values_code" in df.columns else "Values_Code"
    horas = [c for c in df.columns if c.startswith("Values_Hour")]
    lg = df.melt(id_vars=["Date", col], value_vars=horas, var_name="h", value_name="valor")
    lg["fecha_hora"] = pd.to_datetime(lg["Date"]) + pd.to_timedelta(lg["h"].str[-2:].astype(int) - 1, unit="h")
    return lg.rename(columns={col: "codigo"})[["fecha_hora", "codigo", "valor"]]


if __name__ == "__main__":
    api = ReadDB()
    rec = api.request_data("ListadoRecursos", "Sistema", FIN, FIN)
    tipo = rec.set_index("Values_Code")["Values_Type"].to_dict()
    out = []
    for m in METRICAS:
        lg = a_largo(descargar(api, m))
        lg["tipo"] = lg["codigo"].map(tipo).fillna("OTRO").str.upper()
        lg["valor"] = pd.to_numeric(lg["valor"], errors="coerce")
        if m == "PrecOferDesp":
            agg = lg.groupby(["fecha_hora", "tipo"])["valor"].agg(lambda v: np.nanpercentile(v, 75)).unstack()
            agg.columns = [f"oferta_p75_{c.lower()}" for c in agg.columns]
            agg["oferta_p90_todas"] = lg.groupby("fecha_hora")["valor"].quantile(.9)
        else:
            agg = lg.groupby(["fecha_hora", "tipo"])["valor"].sum().unstack() / 1000  # MWh o MW
            agg.columns = [f"{'prog' if m == 'GeneProgDesp' else 'dispo'}_{c.lower()}" for c in agg.columns]
        out.append(agg)
    t = pd.concat(out, axis=1).sort_index()
    t.index.name = "fecha_hora"
    t.to_csv(SALIDA)
    print("guardado", SALIDA, t.shape, t.index.min(), t.index.max(), file=sys.stderr)
    print(list(t.columns))
