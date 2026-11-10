# -*- coding: utf-8 -*-
"""
Backtest de la demanda incompleta del ultimo dia publicado (5-nov-2026).

En produccion, XM publica el ultimo dia de demanda con ~25 % del valor (se completa al dia siguiente). Ese dia es D-4
para la ventana del dia D, y entra en demanda_lag96h (todas las horas) y en demanda_media_24h_hasta96h (en parte). El CatBoost se
entrena con demanda completa (las corridas posteriores la corrigen), asi que solo el dato de prediccion esta mal.
Se simulan, sobre 15-ene a 5-ago-2026 (CatBoost R+P reentrenado cada mes + combinador QRA vigente con el predespacho):
  completa    lo ideal (el dataset oficial)
  parcial     lo que pasa hoy: el dia D-4 escalado hora a hora por la razon publicada/final observada en las 22
              corridas del 11-oct al 4-nov (git de data/produccion/serie_xm.csv)
  completada  el arreglo de produccion_datos.completar_demanda_parcial: generacion de D-4 x la mediana de
              demanda/generacion de los 30 dias anteriores
Salida: resultados/escenario_realista/mejoras/demanda_parcial_metricas.csv y demanda_parcial_razon_horaria.csv
"""
import io
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
import escenario_realista_24h as er  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN, catboost  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SERIE = "data/produccion/serie_xm.csv"


def razon_observada():
    """Razon publicada/final por hora del dia del ultimo dia de demanda, en cada corrida guardada en git."""
    log = subprocess.run(["git", "log", "--format=%h", "--", SERIE], capture_output=True, text=True, cwd=RAIZ).stdout.split()
    lee = lambda h: pd.read_csv(io.StringIO(subprocess.run(["git", "show", f"{h}:{SERIE}"], capture_output=True, text=True,  # noqa: E731
                                                          cwd=RAIZ).stdout), parse_dates=["fecha_hora"]).set_index("fecha_hora")["demanda"]
    final = lee(log[0])
    filas = []
    for h in log[:40]:
        s = lee(h)
        ult = s.last_valid_index().normalize()
        f = final.loc[str(ult.date())]
        if ult >= final.last_valid_index().normalize() or f.isna().any():
            continue                                    # sin dato final todavia
        r = (s.loc[str(ult.date())] / f).to_numpy()
        if r.mean() < 0.9:
            filas.append(pd.Series(r, index=range(24), name=str(ult.date())))
    return pd.DataFrame(filas)


def variables_demanda(dem, gen, fechas, modo, perfil):
    """demanda_lag96h y demanda_media_24h_hasta96h de cada hora objetivo con el dia D-4 modificado segun `modo`."""
    fechas = pd.DatetimeIndex(fechas)
    src = fechas - pd.to_timedelta(np.where(fechas.hour == 0, 120, 96), unit="h")   # como er.por_hora
    dia_p = src.normalize()                                                           # D-4: el dia incompleto
    if modo == "parcial":
        mod = dem * perfil[dem.index.hour]
    else:                                                                             # completada
        razon = (dem / gen).groupby(dem.index.normalize()).mean()
        fac = razon.rolling(30).median().shift(1)                                     # 30 dias anteriores
        mod = gen * fac.reindex(dem.index.normalize()).to_numpy()
    dia = dem.index.normalize()
    cum_r, cum_m = dem.groupby(dia).cumsum(), mod.groupby(dia).cumsum()
    roll = dem.rolling(24).sum()
    lag = mod.reindex(src).to_numpy()
    media = (roll.reindex(src).to_numpy() - cum_r.reindex(src).to_numpy() + cum_m.reindex(src).to_numpy()) / 24
    assert (src.normalize() == dia_p).all()
    return lag, media


if __name__ == "__main__":
    t0 = time.time()
    rz = razon_observada()
    rz.to_csv(rc.MEJ / "demanda_parcial_razon_horaria.csv")
    perfil = rz.mean().to_numpy()
    print(f"{len(rz)} dias incompletos observados; razon media {perfil.mean():.3f} (hora min {perfil.min():.3f}, max {perfil.max():.3f})")

    d = rc.tabla_realista().set_index("fecha_hora").sort_index()
    m = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, usecols=["fecha_hora", "demanda", "generacion"], parse_dates=["fecha_hora"])
                   for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]).drop_duplicates("fecha_hora")
    m = m.set_index("fecha_hora").sort_index()
    cols = conjuntos(d)["R+P"]
    ini, fin = pd.Timestamp("2026-01-01 01:00"), pd.Timestamp("2026-08-05 00:00")
    pred = {k: [] for k in ("completa", "parcial", "completada")}
    for mes in pd.date_range(ini.normalize(), fin, freq="MS"):
        tr = d.loc[INI_TR:mes - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag96h"])
        te = d.loc[mes + pd.Timedelta(hours=1):min(mes + pd.offsets.MonthBegin(1), fin)].dropna(subset=["pre_obj"])
        from catboost import CatBoostRegressor
        mod = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.05, loss_function="MAE", random_seed=42,
                                verbose=0, thread_count=4)
        mod.fit(tr[cols], np.log(tr["precio_bolsa"]) - np.log(tr["pre_obj"].clip(lower=1)))
        for k in pred:
            x = te[cols].copy()
            if k != "completa":
                x["demanda_lag96h"], x["demanda_media_24h_hasta96h"] = variables_demanda(m["demanda"], m["generacion"], te.index, k, perfil)
            pred[k].append(pd.Series(np.exp(mod.predict(x) + np.log(te["pre_obj"].clip(lower=1))), index=te.index))
        print(f"  mes {mes:%Y-%m} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pred = {k: pd.concat(v) for k, v in pred.items()}

    # combinador vigente (QRA MAE g4 con el predespacho), con la misma previa que realista_exp_mezcla.py
    pan = pd.read_csv(rc.MEJ / "mezcla_predicciones.csv", parse_dates=[0], index_col=0)[["real", "dia", "Predespacho"]]
    pan["dia"] = pd.to_datetime(pan["dia"])
    pan["g4"] = pan.index.hour // 6
    from ensamble_24h_enero import origen5
    from ensamble_con_predespacho_24h import imar
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    tr = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag96h"])
    previa["Cat"] = catboost(tr, d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"]), cols).reindex(previa.index)
    previa = previa.dropna(subset=["Predespacho", "Cat"])
    previa["g4"] = previa.index.hour // 6
    filas = []
    out = pd.DataFrame({"real": pan["real"]})
    for k, p in pred.items():
        pan["Cat"] = p.reindex(pan.index)
        q = pan.dropna(subset=["Cat", "Predespacho"])
        out[f"CatBoost {k}"] = q["Cat"]
        out[f"combinado {k}"] = pd.Series(causal(q, previa, ["Predespacho", "Cat"], "g4", objetivo="mae"), index=q.index)
    ev = out[out.index >= "2026-01-15"].dropna()
    for etapa in ("CatBoost", "combinado"):
        ref = ev[f"{etapa} completa"].to_numpy()
        for k in pred:
            c = ev[f"{etapa} {k}"].to_numpy()
            t, pv = (np.nan, np.nan) if k == "completa" else er.dm(ev["real"].to_numpy(), c, ref)
            filas.append({"etapa": etapa, "demanda": k, **rc.metricas(ev["real"], c),
                          "cambio_medio_%": float(np.mean(c / ref - 1) * 100), "cambio_max_%": float(np.max(np.abs(c / ref - 1)) * 100),
                          "DM_t_vs_completa": t, "DM_p": pv})
    t = pd.DataFrame(filas)
    out.to_csv(rc.MEJ / "demanda_parcial_predicciones.csv")
    t.to_csv(rc.MEJ / "demanda_parcial_metricas.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
