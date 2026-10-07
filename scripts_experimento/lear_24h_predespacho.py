# -*- coding: utf-8 -*-
"""
Benchmark para el articulo: ¿el predespacho ideal de XM mejora tambien al LEAR (Lago et al., 2021)?

Mismo LEAR de lear_24h.py (un LASSO por hora del dia, entrenado con 2019-2025 y probado en 2026, corte a las
00:00) con las mismas variables, mas el predespacho ideal disponible al corte:
  * pre_obj: costo marginal del predespacho de la hora objetivo (la hora 00:00 del dia siguiente usa el valor
    de las 23:00, porque el predespacho de ese dia sale despues del corte)
  * pre_media / pre_min / pre_max: media, minimo y maximo del predespacho del dia de la ventana
Solo archivos creados y modificados antes de las 23:00 del dia anterior (validar_predespacho_ideal.py).
Se compara en las mismas horas con la persistencia, el LEAR, el predespacho solo y los ensambles de 6 y 7.
Salidas: data/processed/resultados/pronostico_lear_predespacho_24h_2026.csv y benchmark_predespacho_24h.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import RES, cargar_completo  # noqa: E402
from lear_24h import CORTE, H, construir_para_paso, dm, metricas  # noqa: E402
from ensamble_con_predespacho_24h import imar  # noqa: E402


def con_predespacho(df, h, cortes, cm):
    X, y, f_obj, f_cor = construir_para_paso(df, h, cortes)
    f_obj, f_cor = pd.DatetimeIndex(f_obj), pd.DatetimeIndex(f_cor)
    dia = f_cor.normalize()                                   # dia de la ventana (corte a las 00:00)
    fuente = f_obj.where(f_obj.normalize() == dia, dia + pd.Timedelta(hours=23))
    X = X.copy()
    X["pre_obj"] = cm.reindex(fuente).to_numpy()
    horas = np.stack([cm.reindex(dia + pd.Timedelta(hours=k)).to_numpy() for k in range(24)], axis=1)
    X["pre_media"], X["pre_min"], X["pre_max"] = np.nanmean(horas, 1), np.nanmin(horas, 1), np.nanmax(horas, 1)
    ok = ~X.isna().any(axis=1).to_numpy()
    return X[ok], y[ok], np.asarray(f_obj)[ok], np.asarray(f_cor)[ok]


if __name__ == "__main__":
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    cm = imar()
    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - H)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]
    partes, alphas = [], []
    for h in range(1, H + 1):
        Xtr, ytr, _, _ = con_predespacho(df, h, cortes_tr, cm)
        Xte, yte, f_obj, f_cor = con_predespacho(df, h, cortes_te, cm)
        esc = StandardScaler().fit(Xtr)
        Ztr, Zte = esc.transform(Xtr), esc.transform(Xte)
        if h <= 4 or h % 6 == 0:
            m = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr)
            alphas.append(m.alpha_)
        else:
            m = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr)
        coef_pre = float(m.coef_[list(Xtr.columns).index("pre_obj")])
        partes.append(pd.DataFrame({"fecha_hora": f_obj, "real": yte, "LEAR24_predespacho": m.predict(Zte)}))
        print(f"  paso {h:2d}/24 ({(time.time() - t0) / 60:4.1f} min) MAE={np.abs(yte - m.predict(Zte)).mean():6.2f} "
              f"coef pre_obj={coef_pre:7.2f} activas={(np.abs(m.coef_) > 1e-8).sum()}", flush=True)
    lx = pd.concat(partes).sort_values("fecha_hora")
    lx = lx[lx["fecha_hora"] >= CORTE]
    lx.to_csv(RES / "pronostico_lear_predespacho_24h_2026.csv", index=False)

    # comparacion en las mismas horas
    lear = pd.read_csv(RES / "pronostico_lear_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")["LEAR24"]
    from ensamble_24h_enero import datos_2026
    per = datos_2026()["Persistencia"]
    e6 = pd.read_csv(RES / "pronostico_ensamble_bandas_24h_2026_6votantes.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")["q50"]
    e7 = pd.read_csv(RES / "pronostico_ensamble_bandas_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")["q50"]
    j = lx.set_index("fecha_hora").join([per.rename("Persistencia"), lear.rename("LEAR"), e6.rename("Ensamble 6"),
                                         e7.rename("Ensamble 7 (+ predespacho)")], how="inner")
    fuente = pd.DatetimeIndex(np.where(((j.index - pd.Timedelta(hours=1)).normalize() == j.index.normalize()), j.index,
                                       (j.index - pd.Timedelta(hours=1)).normalize() + pd.Timedelta(hours=23)))
    j["Predespacho solo"] = cm.reindex(fuente).to_numpy()
    j = j.rename(columns={"LEAR24_predespacho": "LEAR + predespacho"}).dropna()
    y = j["real"].to_numpy()
    filas = []
    for c in ["Persistencia", "LEAR", "Predespacho solo", "LEAR + predespacho", "Ensamble 6", "Ensamble 7 (+ predespacho)"]:
        f = {"modelo": c, **metricas(y, j[c])}
        f["rMAE_vs_persistencia"] = f["MAE"] / np.abs(y - j["Persistencia"]).mean()
        if c != "Persistencia":
            f["DM_t_vs_persistencia"], f["DM_p_vs_persistencia"] = dm(y, j["Persistencia"].to_numpy(), j[c].to_numpy())
        filas.append(f)
    t = pd.DataFrame(filas)
    for a, b in (("LEAR", "LEAR + predespacho"), ("Ensamble 6", "Ensamble 7 (+ predespacho)"), ("Predespacho solo", "LEAR + predespacho"),
                 ("Predespacho solo", "Ensamble 7 (+ predespacho)"), ("LEAR + predespacho", "Ensamble 7 (+ predespacho)")):
        tt, pp = dm(y, j[a].to_numpy(), j[b].to_numpy())
        print(f"DM {b} frente a {a}: t={tt:.2f} p={pp:.4f}")
    t.to_csv(RES / "benchmark_predespacho_24h.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\nhoras comparables: {len(j)}\n" + t.round(3).to_string(index=False))
