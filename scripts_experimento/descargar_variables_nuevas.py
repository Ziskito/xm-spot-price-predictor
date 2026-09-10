# -*- coding: utf-8 -*-
"""
Descarga las variables nuevas de nivel Sistema (rapidas, una serie cada una) que el
proyecto no estaba usando. Todas son candidatas a aportar informacion que el precio
rezagado NO contiene, a diferencia de la relacion hidro/termica que ya probamos.

  MaxPrecOferNal  Maximo precio de oferta nacional        horaria   <- oferta, no precio pasado
  CostMargDesp    Costo marginal del despacho programado  horaria   <- senal del despacho
  GeneIdea        Generacion ideal                        horaria
  DemaCome        Demanda comercial                       horaria
  RestAliv        Restricciones aliviadas                 horaria
  PerdidasEner    Perdidas de energia                     horaria
  ImpoEner        Importaciones energia (Ecuador)         horaria
  ExpoEner        Exportaciones energia (Ecuador)         horaria
  PrecEsca        Precio de escasez                       diaria    <- techo regulatorio
  PrecEscaMarg    Precio marginal de escasez              diaria
  AporEnerMediHist Aportes media historica                diaria    <- permite anomalia de aportes
  CompNetArranPara Compras netas arranque y parada        diaria

NOTA DE FUGA DE DATOS: varias de estas (CostMargDesp, GeneProgDesp) se determinan en el
mismo proceso de despacho que el precio. Se descargan crudas aqui; el rezago de >=24h se
aplica despues, en la construccion de features, igual que con demanda/generacion.
"""
import requests
import pandas as pd
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
SALIDA = RAIZ / "data"

INICIO = datetime(2019, 1, 1)
FIN = datetime(2026, 8, 7)
DIAS_BLOQUE = 30

VARIABLES_HORARIAS = {
    "MaxPrecOferNal": "max_precio_oferta",
    "CostMargDesp": "costo_marginal_despacho",
    "GeneIdea": "generacion_ideal",
    "DemaCome": "demanda_comercial",
    "RestAliv": "restricciones_aliviadas",
    "PerdidasEner": "perdidas_energia",
    "ImpoEner": "importaciones_energia",
    "ExpoEner": "exportaciones_energia",
}

VARIABLES_DIARIAS = {
    "PrecEsca": "precio_escasez",
    "PrecEscaMarg": "precio_escasez_marginal",
    "AporEnerMediHist": "aportes_media_historica",
    "CompNetArranPara": "compras_arranque_parada",
}


def pedir(url, metric_id, inicio, fin, entity="Sistema", reintentos=3):
    payload = {"MetricId": metric_id, "StartDate": inicio.strftime("%Y-%m-%d"),
               "EndDate": fin.strftime("%Y-%m-%d"), "Entity": entity}
    for intento in range(reintentos):
        try:
            r = requests.post(url, json=payload, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if intento == reintentos - 1:
                print(f"    fallo tras {reintentos} intentos: {e}")
                return None
            time.sleep(3 * (intento + 1))
    return None


def parsear_horaria(datos, nombre_col):
    filas = []
    for item in datos.get("Items", []):
        fecha = item.get("Date")
        if not fecha:
            continue
        for entidad in item.get("HourlyEntities", []):
            valores = entidad.get("Values", {})
            for hora in range(1, 25):
                v = valores.get(f"Hour{hora:02d}")
                if v is None or v == "":
                    continue
                try:
                    valor = float(v)
                except (TypeError, ValueError):
                    continue
                fh = pd.Timestamp(fecha) + pd.Timedelta(hours=hora - 1)
                filas.append({"fecha_hora": fh, nombre_col: valor})
    return pd.DataFrame(filas)


def parsear_diaria(datos, nombre_col):
    filas = []
    for item in datos.get("Items", []):
        fecha = item.get("Date")
        if not fecha:
            continue
        for entidad in item.get("DailyEntities", []):
            v = entidad.get("Value")
            if v is None or v == "":
                continue
            try:
                valor = float(v)
            except (TypeError, ValueError):
                continue
            filas.append({"fecha_hora": pd.Timestamp(fecha), nombre_col: valor})
    return pd.DataFrame(filas)


def descargar(metric_id, nombre_col, es_horaria):
    url = "https://servapibi.xm.com.co/hourly" if es_horaria else "https://servapibi.xm.com.co/daily"
    parsear = parsear_horaria if es_horaria else parsear_diaria

    partes, fallidos, actual, bloque = [], 0, INICIO, 0
    while actual <= FIN:
        fin_bloque = min(actual + timedelta(days=DIAS_BLOQUE - 1), FIN)
        datos = pedir(url, metric_id, actual, fin_bloque)
        if datos is None:
            fallidos += 1
        else:
            df_bloque = parsear(datos, nombre_col)
            if len(df_bloque):
                partes.append(df_bloque)
        bloque += 1
        actual = fin_bloque + timedelta(days=1)

    if not partes:
        return None, fallidos
    df = (pd.concat(partes, ignore_index=True)
          .drop_duplicates("fecha_hora")
          .sort_values("fecha_hora")
          .reset_index(drop=True))
    return df, fallidos


def main():
    t0 = time.time()
    resumen = []

    for grupo, variables, es_horaria in [("HORARIAS", VARIABLES_HORARIAS, True),
                                           ("DIARIAS", VARIABLES_DIARIAS, False)]:
        print(f"\n{'='*70}\n### {grupo}\n{'='*70}")
        for metric_id, nombre_col in variables.items():
            t_var = time.time()
            df, fallidos = descargar(metric_id, nombre_col, es_horaria)
            if df is None or len(df) == 0:
                print(f"  {metric_id:18s} SIN DATOS (fallidos: {fallidos})")
                resumen.append({"metrica": metric_id, "filas": 0, "fallidos": fallidos})
                sys.stdout.flush()
                continue
            ruta = SALIDA / f"{nombre_col}_2019_2026.csv"
            df.to_csv(ruta, index=False, encoding="utf-8-sig")
            print(f"  {metric_id:18s} {len(df):6,} filas | {df['fecha_hora'].min()} a "
                  f"{df['fecha_hora'].max()} | fallidos: {fallidos} | {time.time()-t_var:.0f}s")
            print(f"    -> {ruta.name} | media: {df[nombre_col].mean():,.2f}")
            resumen.append({"metrica": metric_id, "filas": len(df), "fallidos": fallidos})
            sys.stdout.flush()

    print(f"\n{'='*70}\nLISTO en {(time.time()-t0)/60:.1f} min")
    print(pd.DataFrame(resumen).to_string(index=False))


if __name__ == "__main__":
    main()
