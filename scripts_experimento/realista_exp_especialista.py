# -*- coding: utf-8 -*-
"""
Experimento (14-oct-2026, a pedido de Juan): votante especialista en dias baratos + combinador por regimen.

Motivacion: en los dias baratos y planos (ene-mar 2026) el error es sobre todo de nivel y los CatBoost de produccion,
entrenados con todos los dias, estan dominados por los dias caros de El Niño.
  * Especialista: los mismos 3 CatBoost de produccion (R+P, R+P+F+C, R+P+E; perdida MAE, objetivo log(precio /
    predespacho), reentrenados cada mes con lo publicado) pero entrenados SOLO con ventanas cuyo predespacho medio del dia
    es < UMBRAL COP/kWh; se promedian (Cat_bajo).
  * Regimen del dia: barato si la media del predespacho del dia (conocido al corte) es < UMBRAL.
  * Combinador causal de produccion (QRA, objetivo MAE, margen 5 dias), con grupos = franja de 6 h, o franja x regimen.
Variantes: produccion (Pre + Cat_prom, franjas) | Pre + Cat_prom por regimen | Pre + Cat_prom + Cat_bajo (franjas) |
Pre + Cat_prom + Cat_bajo por regimen. UMBRAL = 250 COP/kWh, fijado antes de mirar resultados.
Salida: resultados/escenario_realista/mejoras/especialista_metricas.csv y especialista_predicciones.csv
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
import realista_exp_errorpre as rep  # noqa: E402
import realista_exp_forma as rf  # noqa: E402
from realista_exp_amplitud import SENALADOS, ramp_metrics  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN, catboost  # noqa: E402
from ensamble_con_predespacho_24h import imar  # noqa: E402

UMBRAL = 250.0
INI, FIN = pd.Timestamp("2026-01-01 01:00"), pd.Timestamp("2026-08-05 00:00")


def limpio(x):
    return x.dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])


def mensual_bajo(d, cols):
    partes = []
    for mes in pd.date_range(INI.normalize().replace(day=1), FIN, freq="MS"):
        tr = limpio(d.loc[INI_TR:mes - MARGEN])
        tr = tr[tr["pre_media"] < UMBRAL]
        te = d.loc[mes + pd.Timedelta(hours=1):min(mes + pd.offsets.MonthBegin(1), FIN)].dropna(subset=["pre_obj"])
        partes.append(catboost(tr, te, cols))
    return pd.concat(partes)


if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    F = rf.forma(d)
    d = d.join(F)
    C = rf.cm_programado(d)
    d = d.join(C)
    E = rep.errores(d)
    d = d.join(E)
    RP = conjuntos(d)["R+P"]
    sets = [RP, RP + list(F.columns) + list(C.columns), RP + list(E.columns)]
    print("filas de entrenamiento en regimen barato (2019-2025):",
          int((limpio(d.loc[INI_TR:"2025-12-31 23:00"])["pre_media"] < UMBRAL).sum()), flush=True)
    e = pd.read_csv(rc.RAIZ / "data/processed/resultados/ensamble_24h_realista_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    e["dia"] = pd.to_datetime(e["dia"])
    e["Cat_bajo"] = np.mean([mensual_bajo(d, c).reindex(e.index) for c in sets], axis=0)
    print(f"  especialista listo ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # previa del combinador (origen 5): modelos entrenados antes de jul-2024
    from ensamble_24h_enero import origen5
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    trp = limpio(d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN])
    tep = d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"])
    previa["Cat_prom"] = np.mean([catboost(trp, tep, c).reindex(previa.index) for c in sets], axis=0)
    previa["Cat_bajo"] = np.mean([catboost(trp[trp["pre_media"] < UMBRAL], tep, c).reindex(previa.index) for c in sets], axis=0)
    previa = previa.dropna(subset=["Predespacho", "Cat_prom", "Cat_bajo"])
    for p in (e, previa):
        dia = (p.index - pd.Timedelta(hours=1)).normalize()
        media_pre = p["Predespacho"].groupby(dia).transform("mean")
        p["barato"] = (media_pre < UMBRAL).astype(int).to_numpy()
        p["g4"] = p.index.hour // 6
        p["g4r"] = p["g4"] * 2 + p["barato"]
        if "dia" not in p:
            p["dia"] = dia
    print("dias baratos en 2026:", int(e.groupby("dia")["barato"].first().sum()), "de", e["dia"].nunique(), flush=True)
    variantes = {"produccion (recalculada)": (["Predespacho", "Cat_prom"], "g4"),
                 "Pre+Cat por regimen": (["Predespacho", "Cat_prom"], "g4r"),
                 "Pre+Cat+especialista": (["Predespacho", "Cat_prom", "Cat_bajo"], "g4"),
                 "Pre+Cat+especialista por regimen": (["Predespacho", "Cat_prom", "Cat_bajo"], "g4r")}
    for nom, (vs, g) in variantes.items():
        e[nom] = causal(e, previa, vs, g, objetivo="mae")
        e[nom] = e[nom].fillna(e["pred_realista"])
        print(f"  {nom} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    e.to_csv(rc.MEJ / "especialista_predicciones.csv")
    q = e[(e["dia"] >= "2026-01-15") & (e.groupby("dia")["real"].transform("size") == 24)].reset_index()
    sen = q["dia"].isin([pd.Timestamp(2026, m, x) for m, xs in SENALADOS.items() for x in xs])
    vol = q.groupby("dia")["real"].transform(lambda x: x.max() - x.min()) > 300
    filas = []
    for c in ["pred_realista"] + list(variantes) + ["Cat_bajo", "Cat_prom"]:
        mr, fr = ramp_metrics(q, c)
        f = {"variante": c, "MAE": np.abs(q[c] - q["real"]).mean(), "MAE_43_dias": np.abs(q[c] - q["real"])[sen].mean(),
             "MAE_dias_baratos": np.abs(q[c] - q["real"])[q["barato"] == 1].mean(),
             "MAE_dias_volatiles": np.abs(q[c] - q["real"])[vol].mean(), "MAE_rampas": mr, "rampa_anticipada": fr}
        f["DM_t"], f["DM_p"] = er.dm(q["real"].to_numpy(), q[c].to_numpy(), q["pred_realista"].to_numpy())
        filas.append(f)
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / "especialista_metricas.csv", index=False)
    pd.set_option("display.width", 230)
    print(t.round(3).to_string(index=False))
    mes = q["fecha_hora"].dt.to_period("M")
    print(pd.DataFrame({c: (q[c] - q["real"]).abs().groupby(mes).mean() for c in ["pred_realista"] + list(variantes)[1:]}).round(1).T.to_string())
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
