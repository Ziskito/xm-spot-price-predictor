# -*- coding: utf-8 -*-
"""
Descarga las metricas de XM que son EQUIVALENTES a variables que los papers leidos identificaron
como utiles, y que el proyecto nunca tuvo.

Mapeo paper -> XM:

  PrecPromContRegu / PrecPromContNoRegu (horaria)
      Equivalente colombiano de los FORWARD PRICES del paper de Nueva Zelanda (Kapoor &
      Wichitaksorn 2023). Alli los precios forward del mercado de derivados eran de las variables
      mas usadas por sus modelos ganadores (60 de sus 423 features). En Colombia el analogo es el
      precio promedio del mercado de contratos: refleja la expectativa del mercado sobre el precio
      futuro, informacion que el precio de bolsa rezagado NO contiene.

  PorcVoluUtilDiar (diaria)
      Equivalente del EAR (energia almacenada en embalses) de Dias, Lira & Freire (2024), que en su
      Tabla 1 quedo en el puesto #1 por informacion mutua con el precio. El proyecto tiene
      volumen_embalses en valor ABSOLUTO; el porcentaje de volumen util es lo que ellos usan y es
      comparable en el tiempo (no se distorsiona cuando cambia la capacidad instalada).

  DesvGenVariableDesp (horaria)
      Desviacion de la generacion variable (solar/eolica) respecto al despacho programado. No sale
      de un paper concreto, pero es el mecanismo fisico detras de los picos: cuando la renovable no
      entrega lo programado, hay que arrancar termica cara. Es informacion de estres del sistema.

  AporCaudal (diaria)
      Aportes de caudal por rio: es literalmente la variable que pronostica Nunes Jr., Ferreira &
      Pinho (2024) en su cadena caudal -> precio. El proyecto tiene aportes en ENERGIA, no en caudal.

Alcance (Anexo 1): todas se agregan a total del sistema; no se guarda desglose por tipo de recurso.
"""
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from descargar_variables_nuevas import pedir, parsear_horaria, parsear_diaria

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
SALIDA = RAIZ / "data"
INICIO = datetime(2019, 1, 1)
FIN = datetime(2026, 8, 7)
DIAS_BLOQUE = 30

HORARIAS = {
    "PrecPromContRegu": ("precio_contratos_regulado", "Sistema"),
    "PrecPromContNoRegu": ("precio_contratos_no_regulado", "Sistema"),
    "DesvGenVariableDesp": ("desviacion_generacion_variable", "Sistema"),
}
DIARIAS = {
    "PorcVoluUtilDiar": ("porcentaje_volumen_util", "Sistema"),
    "AporCaudal": ("aportes_caudal", "Rio"),
}


def bajar(metric_id, nombre, entidad, url, parser):
    print(f"\n=== {metric_id} -> {nombre} (entidad={entidad}) ===", flush=True)
    trozos, fecha = [], INICIO
    while fecha <= FIN:
        hasta = min(fecha + timedelta(days=DIAS_BLOQUE - 1), FIN)
        datos = pedir(url, metric_id, fecha, hasta, entity=entidad)
        if datos:
            df = parser(datos, "valor")
            if len(df):
                # agregacion a total del sistema (suma sobre entidades); sin desglose
                trozos.append(df.groupby("fecha_hora", as_index=False)["valor"].sum())
        fecha = hasta + timedelta(days=1)
        time.sleep(0.3)
    if not trozos:
        print(f"  SIN DATOS para {metric_id}")
        return
    out = (pd.concat(trozos, ignore_index=True).drop_duplicates("fecha_hora")
             .sort_values("fecha_hora").rename(columns={"valor": nombre}))
    ruta = SALIDA / f"{nombre}_2019_2026.csv"
    out.to_csv(ruta, index=False)
    print(f"  guardado {ruta.name}: {len(out):,} registros "
          f"({out['fecha_hora'].min()} .. {out['fecha_hora'].max()})", flush=True)


def main():
    for mid, (nombre, ent) in HORARIAS.items():
        bajar(mid, nombre, ent, "https://servapibi.xm.com.co/hourly", parsear_horaria)
    for mid, (nombre, ent) in DIARIAS.items():
        bajar(mid, nombre, ent, "https://servapibi.xm.com.co/daily", parsear_diaria)


if __name__ == "__main__":
    main()
