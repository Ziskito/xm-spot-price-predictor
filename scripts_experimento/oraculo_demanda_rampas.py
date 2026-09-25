# -*- coding: utf-8 -*-
"""
¿ES LA CURVA DE DEMANDA LA INFORMACION QUE FALTA PARA LAS RAMPAS? Prueba de oraculo.

HIPOTESIS. El precio horario colombiano es escalonado (7.2 niveles por dia en promedio): cada dia
las plantas ofertan un precio, y el precio de cada hora lo marca la planta MARGINAL. Un punto de
cambio intradia es entonces un cambio de planta marginal, y eso ocurre cuando la DEMANDA cruza el
umbral de capacidad de la planta que esta marginando. Si es asi, la forma del dia -- y sobre todo
el momento y el tamanio de las rampas -- deberia estar dictada por la curva de demanda de ese dia.

El proyecto NO usa la demanda del instante objetivo (seria contemporanea). Solo usa la demanda en
el corte y rezagos. Pero XM publica un PRONOSTICO OFICIAL de demanda day-ahead, que si seria
informacion legitima. Antes de ir a buscar ese dato, conviene saber si vale la pena.

LA PRUEBA, en tres escalones sobre el mismo LEAR:
  base       el LEAR tal cual.
  +ayer      + la demanda de la MISMA HORA de AYER y su cambio hora a hora (legitimo hoy).
  ORACULO    + la demanda REAL del instante objetivo y su cambio hora a hora (cota superior de lo
             que aportaria un pronostico de demanda perfecto).
Se mide todo, en particular EN LAS RAMPAS. Si ni el oraculo baja el error de rampa, la demanda no es
la pieza que falta y no vale la pena buscar el pronostico de XM. Si lo baja mucho, es la
recomendacion de dato mas concreta que el proyecto puede hacer.

Se corre con los dos cortes (00:00 actual y 23:00 honesto) para no depender del hallazgo de
protocolo.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES
from lear_24h import construir_para_paso

CORTE = pd.Timestamp("2026-01-01")
FIN = pd.Timestamp("2026-08-05")
PICO = [0, 8, 9, 10, 18, 19]


def correr(df, hora_corte, variante, modelo="lear"):
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    f = df["fecha_hora"]
    dem = df["demanda"].to_numpy(float)
    pos = pd.Series(df.index.values, index=f)
    idx = np.where(f.dt.hour.to_numpy() == hora_corte)[0]
    idx = idx[idx >= 200]
    fc = f.iloc[idx].to_numpy()
    tr = idx[fc < np.datetime64(CORTE - pd.Timedelta(days=1))]
    te = idx[(fc >= np.datetime64(CORTE - pd.Timedelta(hours=1))) & (fc < np.datetime64(FIN))]
    partes, alphas = [], []
    for h in range(1, 25):
        out = []
        for cc in (tr, te):
            X, y, fo, _ = construir_para_paso(df, h, cc)
            t = pos.reindex(pd.DatetimeIndex(fo)).to_numpy()
            X = X.copy()
            if variante in ("ayer", "oraculo"):
                X["dem_ayer"] = dem[t - 24]
                X["ddem_ayer"] = dem[t - 24] - dem[t - 25]
            if variante == "oraculo":
                X["dem_obj"] = dem[t]
                X["ddem_obj"] = dem[t] - dem[t - 1]
                X["dem_obj_rel"] = dem[t] / np.maximum(dem[t - 24], 1)
            ok = ~np.isnan(X.to_numpy(float)).any(1)
            out.append((X[ok], y[ok], np.asarray(fo)[ok]))
        (Xtr, ytr, _), (Xte, yte, fo) = out
        if modelo == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor
            g = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, max_depth=6,
                                              learning_rate=0.05, random_state=42).fit(Xtr, ytr)
            partes.append(pd.DataFrame({"fecha_hora": fo, "real": yte, "pred": g.predict(Xte)}))
            continue
        sc = StandardScaler().fit(Xtr)
        if h <= 4 or h % 6 == 0:
            alphas.append(LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000)
                          .fit(sc.transform(Xtr), ytr).alpha_)
        m = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(sc.transform(Xtr), ytr)
        partes.append(pd.DataFrame({"fecha_hora": fo, "real": yte, "pred": m.predict(sc.transform(Xte))}))
    r = pd.concat(partes).drop_duplicates("fecha_hora").set_index("fecha_hora").sort_index()
    return r[(r.index >= CORTE) & (r.index <= FIN)]


def main():
    t0 = time.time()
    df = cargar_completo().reset_index(drop=True)
    serie = df.set_index("fecha_hora")["precio_bolsa"]
    print(f"{'corte':>10s} {'variante':>9s} {'MAE':>8s} {'MAPE':>7s} {'MAE rampa':>10s} "
          f"{'MAE meseta':>11s} {'MAE pico':>9s}")
    filas = []
    for hc, nom_c, mod in [(0, "00:00", "lear"), (23, "23:00", "lear"),
                           (0, "00:00 gbm", "gbm"), (23, "23:00 gbm", "gbm")]:
        res = {v: correr(df, hc, v, mod) for v in ["base", "ayer", "oraculo"]}
        comun = res["base"].index.intersection(res["ayer"].index).intersection(res["oraculo"].index)
        prev = serie.reindex(comun - pd.Timedelta(hours=1)).to_numpy()
        for v, r in res.items():
            r = r.loc[comun]
            y, p = r["real"].to_numpy(), r["pred"].to_numpy()
            ramp = np.abs(y - prev)
            fu, me, pk = ramp >= 50, ramp == 0, np.isin(comun.hour, PICO)
            fila = dict(corte=nom_c, variante=v, MAE=np.abs(y-p).mean(),
                        MAPE=100*np.mean(np.abs(y-p)/y), MAE_rampa=np.abs(y-p)[fu].mean(),
                        MAE_meseta=np.abs(y-p)[me].mean(), MAE_pico=np.abs(y-p)[pk].mean())
            filas.append(fila)
            print(f"{nom_c:>10s} {v:>9s} {fila['MAE']:8.2f} {fila['MAPE']:7.2f} {fila['MAE_rampa']:10.2f} "
                  f"{fila['MAE_meseta']:11.2f} {fila['MAE_pico']:9.2f}", flush=True)
        print()
    pd.DataFrame(filas).to_csv(RES / "oraculo_demanda_rampas.csv", index=False)
    print(f"Guardado: oraculo_demanda_rampas.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
