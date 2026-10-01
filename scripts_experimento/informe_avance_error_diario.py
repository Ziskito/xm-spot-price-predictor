# -*- coding: utf-8 -*-
"""Error diario del pronóstico de 24 h en 2026: media, desviación estándar e intervalo de confianza del 95 % para la
media, y proporción de días bajo varios umbrales, para el ensamble y la persistencia.

El MAE diario tiene autocorrelación alta (los días malos vienen en rachas), así que el intervalo de la media se
calcula con bootstrap por bloques móviles de 7 días; el intervalo t clásico, que supone días independientes, se
guarda solo como referencia y sale más estrecho de lo que corresponde.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data" / "processed" / "resultados"
SALIDA = RES / "informe_avance" / "error_diario_2026.json"
BLOQUE_DIAS, REMUESTRAS, SEMILLA = 7, 20000, 0
UMBRALES = [25, 50, 75, 100]

ens = pd.read_csv(RES / "combinador_optimo_mape.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
c = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
per = c[(c.origen == "Origen 6") & (c.modelo == "Persistencia")].set_index("fecha_hora")["prediccion"]
per = per.reindex(ens.index)
assert per.notna().all() and len(ens) % 24 == 0
y = ens["real"]
e_ens, e_per = y - ens["pred_smape"], y - per
# cada pronóstico cubre las 24 horas siguientes al corte de las 00:00: un día = 24 filas consecutivas
dia = np.arange(len(y)) // 24
d = pd.DataFrame({"mae_ens": e_ens.abs().values, "mae_per": e_per.abs().values,
                  "mape_ens": (e_ens.abs() / y).values * 100, "sesgo_ens": e_ens.values}).groupby(dia).mean()
d["dif"] = d["mae_per"] - d["mae_ens"]
rng = np.random.default_rng(SEMILLA)


def ic_bloques(x):
    x = np.asarray(x)
    n = len(x)
    k = int(np.ceil(n / BLOQUE_DIAS))
    inicios = rng.integers(0, n - BLOQUE_DIAS + 1, size=(REMUESTRAS, k))
    idx = (inicios[:, :, None] + np.arange(BLOQUE_DIAS)).reshape(REMUESTRAS, -1)[:, :n]
    return [float(v) for v in np.percentile(x[idx].mean(axis=1), [2.5, 97.5])]


def resumen(x):
    x = pd.Series(x)
    n, m, s = len(x), float(x.mean()), float(x.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * s / np.sqrt(n))
    return {"media": m, "de": s, "ic95_bloques": ic_bloques(x), "ic95_t": [m - h, m + h],
            "mediana": float(x.median()), "p10": float(x.quantile(0.10)), "p90": float(x.quantile(0.90))}


res = {
    "n_dias": int(len(d)), "n_horas": int(len(y)), "bloque_dias": BLOQUE_DIAS,
    "acf1_mae_diario_ensamble": float(d["mae_ens"].autocorr(1)),
    "mae_ensamble": resumen(d["mae_ens"]), "mae_persistencia": resumen(d["mae_per"]),
    "mape_ensamble": resumen(d["mape_ens"]), "sesgo_ensamble": resumen(d["sesgo_ens"]),
    "diferencia_per_menos_ens": resumen(d["dif"]),
    "pct_dias_ensamble_mejor": float((d["dif"] > 0).mean() * 100),
    "pct_dias_bajo_umbral": {str(u): {"ensamble": float((d["mae_ens"] < u).mean() * 100),
                                      "persistencia": float((d["mae_per"] < u).mean() * 100)} for u in UMBRALES},
}
SALIDA.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False)[:1500])
