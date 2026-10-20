# -*- coding: utf-8 -*-
"""
Validacion historica (2020-2025) de la variante curva50 (realista_exp_regimen_2026.py), que en 2026 mejora al diseño
en produccion (MAE 38,17 frente a 38,43, DM t = -2,65; desde el 9-sep 64,76 frente a 65,35). Se eligio mirando 2026,
asi que antes de proponerla hay que ver que no empeore en los años en que la CREG 101 127 no existia (20-oct-2026).

Mismo protocolo de realista_validacion_historica_v2.py: por periodo (15-jul a 30-jun), los tres CatBoost mensuales
(R+P, R+P+F+C, R+P+E) se promedian y se combinan con el predespacho (QRA objetivo MAE, franjas de 6 h, margen de 5 dias,
sin previa, 60 dias de calentamiento). base = produccion; curva50 = las mismas variables + precio de referencia de la
curva de ofertas hidraulicas con X = 50 % (variables_curva). Ambas en las mismas horas y Diebold-Mariano entre ellas.
Uso: python realista_validacion_historica_curva.py <O1..O5>   (un periodo por proceso)  |  --evaluar
Salida: resultados/escenario_realista/regimen_2026/historica_<periodo>.csv y historica_metricas.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import escenario_realista_24h as er  # noqa: E402
import realista_comun as rc  # noqa: E402
import realista_exp_errorpre as rep  # noqa: E402
import realista_exp_forma as rf  # noqa: E402
from ensamble_con_predespacho_24h import imar, imar_con_respaldo  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402
from realista_exp_regimen_2026 import SAL, variables_curva  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN, catboost  # noqa: E402

PERIODOS = {"O1": ("Origen 1 (2020-21)", "2020-07-15", "2021-06-30"), "O2": ("Origen 2 (2021-22)", "2021-07-15", "2022-06-30"),
            "O3": ("Origen 3 (2022-23)", "2022-07-15", "2023-06-30"), "O4": ("Origen 4 (2023-24)", "2023-07-15", "2024-06-30"),
            "O5": ("Origen 5 (2024-25)", "2024-07-15", "2025-06-30")}


def correr(clave):
    t0 = time.time()
    nom, ini, fin = PERIODOS[clave]
    desde, hasta = pd.Timestamp(ini), pd.Timestamp(fin) + pd.Timedelta(hours=24)
    d = rc.tabla_realista(produccion=True).set_index("fecha_hora")
    for f in (rf.forma, rf.cm_programado, rep.errores):
        d = d.join(f(d))
    RP = conjuntos(d)["R+P"]
    F, C, E = list(rf.forma(d).columns), list(rf.cm_programado(d).columns), list(rep.errores(d).columns)
    V = variables_curva(d, 0.50)
    d = d.join(V)
    variantes = {"base": [], "curva50": list(V.columns)}
    pre = pd.Series(rc.alinear(d.index, imar_con_respaldo(imar(), d.index.max())), index=d.index)
    cats = {(v, k): [] for v in variantes for k in ("RP", "RPFC", "RPE")}
    for mes in pd.date_range((desde - pd.Timedelta(days=60)).replace(day=1), hasta, freq="MS"):
        tr = d.loc[INI_TR:mes - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
        te = d.loc[mes + pd.Timedelta(hours=1):min(mes + pd.offsets.MonthBegin(1), hasta)].dropna(subset=["pre_obj"])
        for v, extra in variantes.items():
            for k, cols in (("RP", RP), ("RPFC", RP + F + C), ("RPE", RP + E)):
                cats[(v, k)].append(catboost(tr, te, cols + extra))
        print(f"  {clave} {mes:%Y-%m} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    p = pd.DataFrame({"real": d["precio_bolsa"], "Predespacho": pre})
    for v in variantes:
        p[v + "_prom3"] = pd.concat([pd.concat(cats[(v, k)]) for k in ("RP", "RPFC", "RPE")], axis=1).mean(axis=1)
    p = p.dropna(subset=["Predespacho", "base_prom3", "curva50_prom3"])
    p["dia"], p["g4"] = (p.index - pd.Timedelta(hours=1)).normalize(), p.index.hour // 6
    vacia = p.iloc[:0]
    for v in variantes:
        p[v] = causal(p, vacia, ["Predespacho", v + "_prom3"], "g4", objetivo="mae")
    p.loc[desde + pd.Timedelta(hours=1):hasta].to_csv(SAL / f"historica_{clave}.csv")
    print(f"{clave} listo en {(time.time() - t0) / 60:.1f} min", flush=True)


def evaluar():
    filas = []
    for clave, (nom, ini, fin) in PERIODOS.items():
        ruta = SAL / f"historica_{clave}.csv"
        if not ruta.exists():
            continue
        p = pd.read_csv(ruta, index_col=0, parse_dates=True).dropna(subset=["real", "base", "curva50", "Predespacho"])
        y = p["real"].to_numpy()
        t, pv = er.dm(y, p["curva50"].to_numpy(), p["base"].to_numpy())
        filas.append({"periodo": nom, "horas": len(p), "MAE_predespacho": np.abs(y - p["Predespacho"]).mean(),
                      "MAE_base": np.abs(y - p["base"]).mean(), "MAE_curva50": np.abs(y - p["curva50"]).mean(),
                      "DM_t_curva50_vs_base": t, "DM_p": pv})
    t = pd.DataFrame(filas)
    t.to_csv(SAL / "historica_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    print("\nDM t > 0: curva50 tiene MAS error que base; < 0: menos.")


if __name__ == "__main__":
    if sys.argv[1] == "--evaluar":
        evaluar()
    else:
        correr(sys.argv[1])
