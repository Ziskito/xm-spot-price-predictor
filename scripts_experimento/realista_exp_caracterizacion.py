# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: variables sugeridas por la caracterizacion (notebooks 03 y 03b, 5-nov-2026).

Sobre el CatBoost R+P del modelo en produccion (perdida MAE, objetivo log(precio/predespacho), reentrenado cada mes con
lo publicado hasta 73 h antes del corte), se agregan:
  A  anomalia estacional de los embalses: volumen menos la mediana de ese dia del año en los años ANTERIORES (03b: ciclo
     anual fuerte y relacion no lineal con el precio)
  B  aportes acumulados de 180 dias (03b: el embalse integra los aportes; mejor ajuste con ~180 dias)
  C  ONI con 1 y 6 meses mas de rezago que el disponible (~3 y ~8 meses tras el trimestre: lo que tardan en responder
     los aportes y los embalses, 03b)
Todas salen de columnas que ya respetan la publicacion (embalses y aportes de D-2, ONI de M-2), asi que no hay fuga.
Evaluacion: 15-ene a 5-ago-2026, mismas horas, Diebold-Mariano contra el CatBoost R+P sin variables nuevas.
Salida: resultados/escenario_realista/mejoras/caracterizacion_predicciones.csv y caracterizacion_metricas.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
import escenario_realista_24h as er  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_votantes import mensual  # noqa: E402


def climatologia_previa(diaria):
    """Para cada dia, la mediana del mismo dia del año (suavizada +-7 dias) en los años anteriores; NaN sin 2 años."""
    out = pd.Series(np.nan, index=diaria.index)
    for anio in sorted(set(diaria.index.year)):
        previa = diaria[diaria.index.year < anio]
        if previa.index.year.nunique() < 2:
            continue
        med = previa.groupby(previa.index.dayofyear).median().reindex(range(1, 367)).interpolate(limit_direction="both")
        med = pd.concat([med.iloc[-7:], med, med.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
        sel = diaria.index.year == anio
        out[sel] = med.reindex(diaria.index[sel].dayofyear).to_numpy()
    return out


def nuevas(d):
    idx = d.index
    dia = idx.normalize()
    v = d["volumen_embalses"].groupby(dia).mean()
    A = pd.DataFrame({"embalses_anomalia": (v - climatologia_previa(v)).reindex(dia).to_numpy()}, index=idx)
    a = d["aportes_hidricos"].groupby(dia).mean()
    B = pd.DataFrame({"aportes_media_180d": a.rolling(180, min_periods=90).mean().reindex(dia).to_numpy()}, index=idx)
    oni = d["oni"]
    C = pd.DataFrame({f"oni_mas{k}m": oni.reindex(idx - pd.DateOffset(months=k), method="ffill").to_numpy() for k in (1, 6)},
                     index=idx)
    return A, B, C


if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora").sort_index()
    A, B, C = nuevas(d)
    d = d.join(A).join(B).join(C)
    RP = conjuntos(d)["R+P"]
    variantes = {"R+P": RP, "R+P+A": RP + list(A.columns), "R+P+B": RP + list(B.columns),
                 "R+P+C": RP + list(C.columns), "R+P+A+B+C": RP + list(A.columns) + list(B.columns) + list(C.columns)}
    ini, fin = pd.Timestamp("2026-01-01 01:00"), pd.Timestamp("2026-08-05 00:00")
    idx = d.loc[ini:fin].dropna(subset=["precio_bolsa", "pre_obj"]).index
    pan = pd.DataFrame({"real": d.loc[idx, "precio_bolsa"]})
    for nombre, cols in variantes.items():
        print(f"== {nombre} ({len(cols)} variables)", flush=True)
        pan[nombre] = mensual(d, cols, ini, fin).reindex(idx)
        print(f"   listo ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pan.to_csv(rc.MEJ / "caracterizacion_predicciones.csv")
    ev = pan[pan.index >= "2026-01-15"].dropna()
    filas = []
    for c in variantes:
        fl = {"variante": c, **rc.metricas(ev["real"], ev[c])}
        fl["DM_t_vs_R+P"], fl["DM_p"] = (np.nan, np.nan) if c == "R+P" else er.dm(ev["real"].to_numpy(), ev[c].to_numpy(), ev["R+P"].to_numpy())
        for nom, sel in (("ene-abr", ev.index < "2026-05-01"), ("may-ago (El Niño)", ev.index >= "2026-05-01")):
            fl[f"MAE {nom}"] = float(np.abs(ev.loc[sel, c] - ev.loc[sel, "real"]).mean())
        filas.append(fl)
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / "caracterizacion_metricas.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
