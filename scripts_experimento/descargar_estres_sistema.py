# -*- coding: utf-8 -*-
"""
Descarga tres metricas HORARIAS del API de XM que el proyecto nunca bajo y que son indicadores
DIRECTOS de estres del sistema -- exactamente el tipo de senal que el diagnostico del error de 24h
senalo como faltante (los picos de precio son de OFERTA, no de demanda).

  GeneFueraMerito  Generacion Fuera de Merito por Recurso   horaria
      Se activa cuando el despacho tiene que salirse del orden de merito por restricciones.
      Es un indicador directo de que el sistema esta tensionado.

  RestSinAliv      Restricciones sin Alivios por Sistema    horaria
      El proyecto ya tiene "restricciones aliviadas"; las NO aliviadas son las que de verdad
      quedan sin resolver y presionan el precio.

  GeneProgDesp     Generacion Programada de Despacho        horaria
      Es el programa de despacho del dia siguiente: informacion CONOCIDA EN EL CORTE y
      genuinamente anticipativa, a diferencia de casi todo lo demas que es realizado.

Nota de alcance (Anexo 1): las tres se agregan a TOTAL DEL SISTEMA. No se guarda ni se usa el
desglose por tipo de recurso, que es lo que el Alcance restringe.

Nota de fuga: se descargan crudas. El rezago (uso solo del valor en el corte) se aplica despues,
en la construccion de features, igual que se hizo con demanda/generacion.
"""
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from descargar_variables_nuevas import pedir, parsear_horaria

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
SALIDA = RAIZ / "data"
URL = "https://servapibi.xm.com.co/hourly"
INICIO = datetime(2019, 1, 1)
FIN = datetime(2026, 8, 7)
DIAS_BLOQUE = 30

# metric_id -> (nombre de archivo, entidad que pide el API)
METRICAS = {
    "GeneFueraMerito": ("generacion_fuera_merito", "Recurso"),
    "RestSinAliv": ("restricciones_sin_alivios", "Sistema"),
    "GeneProgDesp": ("generacion_programada_despacho", "Recurso"),
}


def descargar(metric_id, nombre, entidad):
    print(f"\n=== {metric_id} ({nombre}, entidad={entidad}) ===", flush=True)
    trozos, fecha = [], INICIO
    while fecha <= FIN:
        hasta = min(fecha + timedelta(days=DIAS_BLOQUE - 1), FIN)
        datos = pedir(URL, metric_id, fecha, hasta, entity=entidad)
        if datos:
            df = parsear_horaria(datos, "valor")
            if len(df):
                # agregar a TOTAL del sistema (suma sobre recursos); sin desglose por tipo
                trozos.append(df.groupby("fecha_hora", as_index=False)["valor"].sum())
        fecha = hasta + timedelta(days=1)
        if len(trozos) % 12 == 0 and trozos:
            print(f"  {trozos[-1]['fecha_hora'].max().date()} ...", flush=True)
        time.sleep(0.3)
    if not trozos:
        print(f"  SIN DATOS para {metric_id}")
        return None
    out = (pd.concat(trozos, ignore_index=True)
             .drop_duplicates("fecha_hora").sort_values("fecha_hora")
             .rename(columns={"valor": nombre}))
    ruta = SALIDA / f"{nombre}_2019_2026.csv"
    out.to_csv(ruta, index=False)
    print(f"  guardado {ruta.name}: {len(out):,} horas "
          f"({out['fecha_hora'].min()} .. {out['fecha_hora'].max()})")
    return out


def main():
    print("Probando conexion con el API de XM...")
    prueba = pedir(URL, "RestSinAliv", datetime(2026, 1, 1), datetime(2026, 1, 2))
    if prueba is None:
        print("NO hay conexion con el API de XM (o la metrica no responde). Se aborta.")
        return
    print("Conexion OK.\n")
    for mid, (nombre, entidad) in METRICAS.items():
        descargar(mid, nombre, entidad)


if __name__ == "__main__":
    main()
