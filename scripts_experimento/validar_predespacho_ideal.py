# -*- coding: utf-8 -*-
"""
Validacion de data/processed/predespacho_ideal_xm.csv (descargar_predespacho_ideal.py).

Revisa: cobertura de dias, 24 horas por dia, valores vacios, consistencia interna (costo marginal = MPO + delta),
rangos y unidades estables por año frente al precio de bolsa real, hora de publicacion frente a nuestro corte
(un valor solo se puede usar si se publico antes de las 23:00 del dia anterior) y una relectura de 15 dias al
azar directamente desde XM, comparada con lo guardado. Agrega la columna "a_tiempo" al archivo.
Salida: data/processed/resultados/validacion_predespacho_ideal.txt
"""
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import descargar_predespacho_ideal as dpi  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "data" / "processed" / "predespacho_ideal_xm.csv"
INFORME = RAIZ / "data" / "processed" / "resultados" / "validacion_predespacho_ideal.txt"

if __name__ == "__main__":
    d = pd.read_csv(RUTA, parse_dates=["fecha_hora"])
    lineas = []
    p = lambda s: (lineas.append(s), print(s))
    d["dia"] = d["fecha_hora"].dt.normalize()
    dias = pd.DatetimeIndex(d["dia"].unique())
    todos = pd.date_range(dias.min(), dias.max())
    faltan = todos.difference(dias)
    p(f"1. Cobertura: {len(dias)} de {len(todos)} dias ({dias.min():%d/%m/%Y} a {dias.max():%d/%m/%Y}); faltan {len(faltan)}: "
      + ", ".join(f"{x:%d/%m/%Y}" for x in faltan)
      + ". XM los lista en la biblioteca, pero su descarga devuelve un zip vacio (archivo daniado en el servidor).")
    horas = d.groupby("dia").size()
    p(f"2. Horas por dia: {(horas == 24).sum()} dias con 24 h; distintos de 24: {dict(horas[horas != 24])}")
    p(f"   Duplicados de fecha_hora: {int(d['fecha_hora'].duplicated().sum())}; vacios: "
      f"costo_marginal {int(d['costo_marginal'].isna().sum())}, mpo {int(d['mpo'].isna().sum())}, delta {int(d['delta'].isna().sum())}")
    dif = (d["costo_marginal"] - (d["mpo"] + d["delta"])).abs()
    p(f"3. Consistencia costo marginal = MPO + delta: diferencia maxima {dif.max():.4f} COP/kWh; "
      f"filas con diferencia > 0,01: {int((dif > 0.01).sum())}")
    p(f"   Valores <= 0: {int((d['costo_marginal'] <= 0).sum())}; minimo {d['costo_marginal'].min():.1f}, maximo {d['costo_marginal'].max():.1f} COP/kWh")
    real = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])
                      for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]).drop_duplicates("fecha_hora")
    t = d.merge(real, on="fecha_hora", how="inner")
    p("4. Frente al precio de bolsa real, por año (si cambiaran las unidades, el cociente saltaria):")
    for anio, g in t.groupby(t["fecha_hora"].dt.year):
        p(f"   {anio}: {len(g):5d} h | mediana predespacho {g.costo_marginal.median():7.1f} | mediana real {g.precio_bolsa.median():7.1f} | "
          f"cociente mediano {np.median(g.costo_marginal / g.precio_bolsa):.3f} | correlacion {g.costo_marginal.corr(g.precio_bolsa):.3f} | "
          f"MAE {np.abs(g.costo_marginal - g.precio_bolsa).mean():6.1f}")
    pub = pd.to_datetime(d["publicado"].str[:19], errors="coerce")
    corte = d["dia"] - pd.Timedelta(hours=1)                  # 23:00 del dia anterior (el corte mas temprano)
    # tambien cuenta la fecha de MODIFICACION: si XM corrigio el archivo despues del corte, el contenido
    # guardado no existia al momento del pronostico (metadatos de la biblioteca, 3.196 archivos)
    meta = pd.read_csv(RAIZ / "data" / "processed" / "resultados" / "predespacho_ideal_metadatos.csv",
                       parse_dates=["dia", "creado", "modificado"]).drop_duplicates("dia").set_index("dia")
    modif = d["dia"].map(meta["modificado"])
    d["a_tiempo"] = (pub < corte) & (modif.isna() | (modif < corte))
    p(f"   Archivos modificados despues del corte (se excluyen aunque se hayan creado a tiempo): "
      + ", ".join(f"{x:%d/%m/%Y}" for x in sorted(meta.index[meta["modificado"] >= meta.index - pd.Timedelta(hours=1)])))
    tarde = d.loc[~d["a_tiempo"]].groupby("dia")["publicado"].first()
    p(f"5. Publicacion: mediana {(pub - pub.dt.normalize()).median()} despues de la medianoche del dia anterior; {d['a_tiempo'].mean() * 100:.2f} % de las horas "
      f"publicadas antes de las 23:00 del dia anterior. Dias publicados tarde (se marcan a_tiempo = False y no se usan): {len(tarde)}")
    for dia, pb in tarde.items():
        p(f"   {dia:%d/%m/%Y}: publicado {pb[:16]}")
    # relectura de 15 dias al azar desde XM
    random.seed(7)
    muestra = sorted(random.sample(list(dias), 15))
    errores = 0
    for dia in muestra:
        mes = f"{dia:%Y-%m}"
        fs = [f for f in dpi.listar(f"{dpi.RUTA}/{mes}") if re.fullmatch(rf"(?i)imar{dia:%m%d}(_NAL)?\.txt", f["nombre"])]
        z = dpi.descargar_zip([fs[0]["id"]])
        nuevo = dpi.leer_imar(z.read(z.namelist()[0]).decode("latin-1"), dia).set_index("fecha_hora")["costo_marginal"]
        guardado = d.set_index("fecha_hora").loc[nuevo.index, "costo_marginal"]
        errores += int((np.abs(nuevo - guardado) > 1e-6).sum())
    p(f"6. Relectura desde XM de 15 dias al azar ({', '.join(f'{x:%d/%m/%Y}' for x in muestra)}): {errores} horas distintas de lo guardado")
    d.drop(columns="dia").to_csv(RUTA, index=False)
    INFORME.write_text("\n".join(lineas), encoding="utf-8")
