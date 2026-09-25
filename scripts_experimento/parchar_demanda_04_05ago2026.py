# -*- coding: utf-8 -*-
"""
Parche puntual: la demanda del 4 y 5 de agosto de 2026 estaba mal en el proyecto (70-86% por
debajo del valor real), descubierto al comparar el notebook 02 corrido en una copia aislada
contra los datos en vivo. precio_bolsa, generacion, volumen_embalses y aportes_hidricos estaban
bien esos mismos dias -- solo demanda.

Alcance del parche, medido con precision antes de tocar nada (ver bitacora):
  demanda_real_2026.csv        48 horas (04-ago 00:00 a 05-ago 23:00)
  dataset_maestro_2026.csv     las mismas 48 horas, columna 'demanda'
  dataset_features_2026.csv    'demanda' en esas 48h, y ademas 'demanda_lag24h' y
                                'demanda_media_24h' en las 24h del 05-ago (se derivan con
                                shift(24)/rolling(24) de la propia demanda, asi que el 05-ago
                                hereda el error del 04-ago). demanda_lag48h y demanda_lag72h NO
                                se tocan: sus ventanas caen fuera del rango del archivo (termina
                                el 05-ago 23:00).

Los valores correctos vienen de una descarga fresca de la API de XM (usando la misma funcion
`descargar_precio_bolsa`-equivalente del notebook 02), hecha en una COPIA AISLADA del proyecto,
nunca sobre datos en vivo. 2019-2025 no se toca: la anomalia esta confirmada exclusiva a estas
48 horas de 2026.

Respaldo ya hecho antes de correr esto: data/processed/resultados/backups_pre_fix_demanda_04_05ago2026/
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import RAIZ

SCRATCH = Path("C:/Users/mgdbj/AppData/Local/Temp/claude/c--Users-mgdbj-xm-spot-price-predictor"
               "/158da6d1-99be-43de-b42e-63eface76398/scratchpad/prueba_notebooks")
VENTANA_INI = pd.Timestamp("2026-08-04 00:00:00")
VENTANA_FIN = pd.Timestamp("2026-08-05 23:00:00")


def main():
    # ---------- valores correctos, de la descarga fresca en la copia aislada ----------
    fresco = pd.read_csv(SCRATCH / "data/demanda_real_2026.csv", parse_dates=["fecha_hora"])
    fresco = fresco[(fresco["fecha_hora"] >= VENTANA_INI) & (fresco["fecha_hora"] <= VENTANA_FIN)]
    assert len(fresco) == 48, f"se esperaban 48 horas frescas, hay {len(fresco)}"
    correctos = fresco.set_index("fecha_hora")["demanda_kwh"]
    print(f"Valores correctos obtenidos: {len(correctos)} horas, "
          f"media={correctos.mean():,.0f}  min={correctos.min():,.0f}  max={correctos.max():,.0f}")

    # ---------- 1) demanda_real_2026.csv ----------
    p1 = RAIZ / "data/demanda_real_2026.csv"
    d1 = pd.read_csv(p1, parse_dates=["fecha_hora"])
    antes = d1.set_index("fecha_hora").loc[VENTANA_INI:VENTANA_FIN, "demanda_kwh"].copy()
    mask = (d1["fecha_hora"] >= VENTANA_INI) & (d1["fecha_hora"] <= VENTANA_FIN)
    d1.loc[mask, "demanda_kwh"] = d1.loc[mask, "fecha_hora"].map(correctos).to_numpy()
    assert mask.sum() == 48
    d1.to_csv(p1, index=False)
    print(f"\n1) demanda_real_2026.csv parchado: 48 filas, media antes={antes.mean():,.0f} "
          f"-> despues={correctos.mean():,.0f}")

    # ---------- 2) dataset_maestro_2026.csv ----------
    p2 = RAIZ / "data/processed/dataset_maestro_2026.csv"
    d2 = pd.read_csv(p2, parse_dates=["fecha_hora"])
    mask2 = (d2["fecha_hora"] >= VENTANA_INI) & (d2["fecha_hora"] <= VENTANA_FIN)
    assert mask2.sum() == 48, f"dataset_maestro: se esperaban 48 filas en la ventana, hay {mask2.sum()}"
    d2.loc[mask2, "demanda"] = d2.loc[mask2, "fecha_hora"].map(correctos).to_numpy()
    d2.to_csv(p2, index=False)
    print(f"2) dataset_maestro_2026.csv parchado: 48 filas (columna 'demanda' unicamente)")

    # ---------- 3) dataset_features_2026.csv ----------
    # Se recalculan demanda_lag24h/48h/72h/media_24h con la MISMA formula de 05_features_compartidas
    # (H=24), usando una serie continua construida a partir del maestro 2019-2025 (intacto) + el
    # 2026 ya corregido, exactamente como hace el notebook (concatena antes de derivar).
    m25 = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", parse_dates=["fecha_hora"])
    serie = pd.concat([m25[["fecha_hora", "demanda"]], d2[["fecha_hora", "demanda"]]],
                      ignore_index=True).sort_values("fecha_hora").set_index("fecha_hora")["demanda"]
    H = 24
    demanda_base = serie.shift(H)
    recalc = pd.DataFrame({
        "demanda_lag24h": demanda_base,
        "demanda_lag48h": serie.shift(H + 24),
        "demanda_lag72h": serie.shift(H + 48),
        "demanda_media_24h": demanda_base.rolling(24).mean(),
    })

    p3 = RAIZ / "data/processed/dataset_features_2026.csv"
    d3 = pd.read_csv(p3, parse_dates=["fecha_hora"])
    cols = ["demanda_lag24h", "demanda_lag48h", "demanda_lag72h", "demanda_media_24h"]
    antes3 = d3.set_index("fecha_hora")[["demanda"] + cols].copy()

    d3 = d3.set_index("fecha_hora")
    filas_cambiadas = {}
    en_ventana = (d3.index >= VENTANA_INI) & (d3.index <= VENTANA_FIN)
    assert en_ventana.sum() == 48, f"dataset_features: se esperaban 48 filas en la ventana, hay {en_ventana.sum()}"
    d3.loc[en_ventana, "demanda"] = pd.Series(d3.index[en_ventana]).map(correctos).to_numpy()

    for c in cols:
        nuevo_c = recalc[c].reindex(d3.index)
        difiere = (d3[c] - nuevo_c).abs() > 1
        difiere = difiere.fillna(False)
        filas_cambiadas[c] = int(difiere.sum())
        d3.loc[difiere, c] = nuevo_c.loc[difiere]

    d3 = d3.reset_index()
    d3.to_csv(p3, index=False)
    print(f"3) dataset_features_2026.csv parchado:")
    print(f"   demanda: {int(en_ventana.sum())} filas")
    for c in cols:
        print(f"   {c}: {filas_cambiadas[c]} filas cambiadas")

    # ---------- verificacion final ----------
    print("\n===== VERIFICACION =====")
    v3 = pd.read_csv(p3, parse_dates=["fecha_hora"])
    ventana3 = v3[(v3["fecha_hora"] >= VENTANA_INI) & (v3["fecha_hora"] <= VENTANA_FIN + pd.Timedelta(hours=24))]
    print(ventana3[["fecha_hora", "demanda", "demanda_lag24h", "demanda_media_24h"]].to_string())


if __name__ == "__main__":
    main()
