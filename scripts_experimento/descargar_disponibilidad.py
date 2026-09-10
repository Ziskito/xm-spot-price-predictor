# -*- coding: utf-8 -*-
"""
Descarga disponibilidad declarada por recurso y la agrega por tipo de planta
(hidraulica / termica), usando el catalogo ListadoRecursos.

Por que esta variable es distinta a las que ya descartamos:
  - Es FISICA (MW disponibles), no un precio -> no puede ser el precio disfrazado.
  - Se DECLARA POR ANTICIPADO al despacho -> es informacion forward-looking genuina.
  - La disponibilidad termica en particular determina cuanto respaldo caro hay disponible,
    que es justo lo que fija el precio cuando la hidrologia esta mal.

Genera: data/disponibilidad_por_tipo_2019_2026.csv
"""
import requests
import pandas as pd
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
URL_HOURLY = "https://servapibi.xm.com.co/hourly"
URL_LISTS = "https://servapibi.xm.com.co/lists"

INICIO = datetime(2019, 1, 1)
FIN = datetime(2026, 8, 7)
DIAS_BLOQUE = 30
METRICA = "DispoDeclarada"


def pedir(url, payload, reintentos=3):
    for intento in range(reintentos):
        try:
            r = requests.post(url, json=payload, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if intento == reintentos - 1:
                print(f"    fallo: {e}")
                return None
            time.sleep(3 * (intento + 1))
    return None


def descargar_catalogo():
    datos = pedir(URL_LISTS, {"MetricId": "ListadoRecursos", "StartDate": "2026-01-01",
                                "EndDate": "2026-01-01", "Entity": "Sistema"})
    filas = []
    for item in datos.get("Items", []):
        for ent in item.get("ListEntities", []):
            v = ent.get("Values", {})
            if isinstance(v, dict):
                filas.append(v)
    cat = pd.DataFrame(filas)
    ruta = RAIZ / "data" / "catalogo_recursos_xm.csv"
    cat.to_csv(ruta, index=False, encoding="utf-8-sig")

    col_cod = next((c for c in cat.columns if c.lower() in ("code", "codigo", "values_code")), cat.columns[0])
    col_tipo = next((c for c in cat.columns if "type" in c.lower() or "tipo" in c.lower()), None)
    mapa = dict(zip(cat[col_cod], cat[col_tipo])) if col_tipo else {}
    print(f"Catalogo: {len(cat)} recursos | columna codigo='{col_cod}' tipo='{col_tipo}'")
    if col_tipo:
        print(cat[col_tipo].value_counts().head(10).to_string())
    return mapa


def agregar_bloque(datos, mapa_tipos):
    acumulado = {}
    for item in datos.get("Items", []):
        fecha = item.get("Date")
        if not fecha:
            continue
        for ent in item.get("HourlyEntities", []):
            valores = ent.get("Values", {})
            codigo = valores.get("code") or ent.get("Id")
            tipo = str(mapa_tipos.get(codigo, "DESCONOCIDO")).upper()
            for hora in range(1, 25):
                v = valores.get(f"Hour{hora:02d}")
                if v is None or v == "":
                    continue
                try:
                    valor = float(v)
                except (TypeError, ValueError):
                    continue
                fh = pd.Timestamp(fecha) + pd.Timedelta(hours=hora - 1)
                acumulado.setdefault(fh, {}).setdefault(tipo, 0.0)
                acumulado[fh][tipo] += valor
    filas = [{"fecha_hora": fh, **tipos} for fh, tipos in acumulado.items()]
    return pd.DataFrame(filas)


def main():
    t0 = time.time()
    mapa = descargar_catalogo()

    partes, fallidos, actual, bloque = [], 0, INICIO, 0
    while actual <= FIN:
        fin_b = min(actual + timedelta(days=DIAS_BLOQUE - 1), FIN)
        datos = pedir(URL_HOURLY, {"MetricId": METRICA, "StartDate": actual.strftime("%Y-%m-%d"),
                                    "EndDate": fin_b.strftime("%Y-%m-%d"), "Entity": "Recurso"})
        if datos is None:
            fallidos += 1
        else:
            df_b = agregar_bloque(datos, mapa)
            if len(df_b):
                partes.append(df_b)
        bloque += 1
        if bloque % 10 == 0:
            print(f"  [{bloque}] {actual.date()} | {(time.time()-t0)/60:.1f} min")
            sys.stdout.flush()
            if partes:
                pd.concat(partes, ignore_index=True).to_csv(
                    RAIZ / "data" / "disponibilidad_PARCIAL.csv", index=False)
        actual = fin_b + timedelta(days=1)

    df = pd.concat(partes, ignore_index=True).groupby("fecha_hora").sum().reset_index()
    df = df.sort_values("fecha_hora").reset_index(drop=True)

    ruta = RAIZ / "data" / "disponibilidad_por_tipo_2019_2026.csv"
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min | {len(df):,} filas | fallidos: {fallidos}")
    print("Columnas:", df.columns.tolist())
    print(df.head(3).to_string())
    print("\nMedias por tipo:")
    for c in df.columns:
        if c != "fecha_hora":
            print(f"  {c}: {df[c].mean():,.0f}")


if __name__ == "__main__":
    main()
