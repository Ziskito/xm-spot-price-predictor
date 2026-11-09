# -*- coding: utf-8 -*-
"""
Walk-forward historico del pronostico de 72 h realista (pasos 25-72: dias 2 y 3), 9-nov-2026 (pedido de Juan).

Mismo protocolo que la validacion historica de 24 h (realista_validacion_historica_v2.py): cinco periodos del 15 de julio
al 30 de junio (2020-21 a 2024-25), solo informacion publicada al corte de cada dia D, CatBoost reentrenados cada mes y
combinador causal sin previa (los primeros dias, sin historia publicada, usan la referencia). Mismo modelo que en
produccion (pronostico_realista_72h.py): para el dia D+k (k = 1, 2) la referencia es el predespacho de D a la misma hora;
tres CatBoost (R+P, R+P+F+C, R+P+E) con las variables de la fila t - 24k y el calendario de t, promediados; combinador QRA
(objetivo MAE, franjas de 6 h, margen de 5 + k dias) y banda conforme del 80 %.
Comparaciones (mismas horas): persistencia (ultimo precio publicado de la misma hora, t - 96 - 24k), referencia sola
(predespacho de D) y el dia 1 (validacion de 24 h, mejoras/validacion_historica_v2.csv).
Salida: resultados/escenario_realista/mejoras/validacion_historica_72h.csv (metricas) y _predicciones.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import pronostico_realista_72h as p72  # noqa: E402
import realista_comun as rc  # noqa: E402
import escenario_realista_24h as er  # noqa: E402
import realista_exp_errorpre as rep  # noqa: E402
import realista_exp_forma as rf  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402

PERIODOS = [("2020-21", "2020-07-15", "2021-06-30"), ("2021-22", "2021-07-15", "2022-06-30"),
            ("2022-23", "2022-07-15", "2023-06-30"), ("2023-24", "2023-07-15", "2024-06-30"),
            ("2024-25", "2024-07-15", "2025-06-30")]

if __name__ == "__main__":
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    F = rf.forma(d); d = d.join(F)
    C = rf.cm_programado(d); d = d.join(C)
    E = rep.errores(d); d = d.join(E)
    RP = conjuntos(d)["R+P"]
    sets = {"Cat_RP": RP, "Cat_RPFC": RP + list(F.columns) + list(C.columns), "Cat_RPE": RP + list(E.columns)}
    precio = d["precio_bolsa"]
    filas, preds = [], []
    for nom, ini, fin in PERIODOS:
        ini, fin = pd.Timestamp(ini), pd.Timestamp(fin)
        for k in (1, 2):
            X = p72.tabla_k(d, k, sets)
            cols = {n: [c for c in s if c in X.columns] for n, s in sets.items()}
            sel = (X["corte"] >= ini - pd.Timedelta(days=k)) & (X["corte"] <= fin - pd.Timedelta(days=k)) & X["ref"].notna() & X["precio"].notna()
            idx = X.index[sel]
            idx = idx[((idx - pd.Timedelta(hours=1)).normalize() >= ini) & ((idx - pd.Timedelta(hours=1)).normalize() <= fin)]
            p = pd.DataFrame(index=idx)
            for n, c in cols.items():
                p[n] = p72.mensual(X, c, X.loc[idx, "corte"].min(), X.loc[idx, "corte"].max()).reindex(idx)
                print(f"  {nom} k={k} {n} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            p["Cat_prom"] = p[list(cols)].mean(axis=1)
            p["real"], p["ref"] = X.loc[idx, "precio"], X.loc[idx, "ref"]
            p["dia"] = (p.index - pd.Timedelta(hours=1)).normalize()
            p["g4"] = p.index.hour // 6
            previa = p.iloc[:0][["real", "ref", "Cat_prom", "dia", "g4"]].copy()     # vacia, con los mismos tipos (fechas)
            p = p.dropna(subset=["ref", "Cat_prom"])
            p["q50"] = pd.Series(causal(p, previa, ["ref", "Cat_prom"], "g4", margen=5 + k, objetivo="mae"),
                                 index=p.index).fillna(p["ref"])
            p["q10"], p["q90"], _ = p72.bandas(p, k)
            p["pers"] = precio.reindex(p.index - pd.to_timedelta(np.where(p.index.hour == 0, 120, 96) + 24 * k, unit="h")).to_numpy()
            ev = p.dropna(subset=["pers"])
            y = ev["real"].to_numpy()
            t, pv = er.dm(y, ev["q50"].to_numpy(), ev["pers"].to_numpy())
            filas.append({"periodo": nom, "dia": k + 1, "horas": len(ev), "MAE": np.abs(ev["q50"] - y).mean(),
                          "MAPE": (np.abs(ev["q50"] - y) / y).mean() * 100, "sesgo": (ev["q50"] - y).mean(),
                          "MAE_ref": np.abs(ev["ref"] - y).mean(), "MAE_cat_prom": np.abs(ev["Cat_prom"] - y).mean(),
                          "MAE_pers": np.abs(ev["pers"] - y).mean(),
                          "cobertura": ((y >= ev["q10"]) & (y <= ev["q90"])).mean() * 100, "ancho": (ev["q90"] - ev["q10"]).mean(),
                          "DM_t_vs_pers": t, "DM_p": pv})
            preds.append(ev.assign(periodo=nom, k=k)[["periodo", "k", "real", "ref", "Cat_prom", "q50", "q10", "q90", "pers"]])
            print("   ", {x: (round(v, 2) if isinstance(v, float) else v) for x, v in filas[-1].items()}, flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(rc.MEJ / "validacion_historica_72h.csv", index=False)
    pd.concat(preds).to_csv(rc.MEJ / "validacion_historica_72h_predicciones.csv")
    pd.set_option("display.width", 220)
    print(t.round(2).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
