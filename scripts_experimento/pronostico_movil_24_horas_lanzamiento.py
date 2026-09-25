# -*- coding: utf-8 -*-
"""
ERROR DE UN PRONOSTICO MOVIL DE 24h SEGUN LA HORA EN QUE SE LANZA.

Aclaracion del usuario (2026-09-24): el pronostico de 24h se cuenta desde la hora real en que se
emite, y el precio de la ultima hora esta disponible de inmediato. Entonces el corte de las 00:00
que usa todo el proyecto es solo UNA de las 24 horas posibles de lanzamiento -- y es la mas
favorable, porque 23 de sus 24 objetivos pertenecen al dia de despacho que ya empezo a observarse
(el dia de ofertas va de 00:00 a 23:00). Lanzado a las 23:00 es el peor caso (0 de 24).

Aqui se mide la curva completa con el MISMO LEAR (mismas variables, misma regularizacion, mismo
periodo 2026), cambiando solo la hora del corte: 00:00, 01:00, ..., 23:00. El promedio sobre las
24 horas es la cifra representativa de un pronostico movil lanzado a cualquier hora.

Para el ensamble completo no se hace (exigiria reentrenar las redes 24 veces), pero se escala: la
razon (MAE promedio / MAE a las 00:00) del LEAR se aplica al 41.29 del ensamble como estimacion, y
se contrasta con los dos extremos que si estan medidos en el ensamble (00:00 = 41.29,
23:00 = 51.82).
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


def lear_desde(df, hora):
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    f = df["fecha_hora"]
    idx = np.where(f.dt.hour.to_numpy() == hora)[0]
    idx = idx[idx >= 200]
    fc = f.iloc[idx].to_numpy()
    tr = idx[fc < np.datetime64(CORTE - pd.Timedelta(days=1))]
    te = idx[(fc >= np.datetime64(CORTE - pd.Timedelta(days=1))) & (fc < np.datetime64(FIN))]
    partes, alphas = [], []
    for h in range(1, 25):
        Xtr, ytr, _, _ = construir_para_paso(df, h, tr)
        Xte, yte, fo, _ = construir_para_paso(df, h, te)
        sc = StandardScaler().fit(Xtr)
        if h <= 4 or h % 6 == 0:
            alphas.append(LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000)
                          .fit(sc.transform(Xtr), ytr).alpha_)
        m = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(sc.transform(Xtr), ytr)
        partes.append(pd.DataFrame({"fecha_hora": fo, "real": yte, "pred": m.predict(sc.transform(Xte))}))
    r = pd.concat(partes, ignore_index=True)
    return r[(r["fecha_hora"] >= CORTE) & (r["fecha_hora"] < FIN)]


def main():
    t0 = time.time()
    df = cargar_completo().reset_index(drop=True)
    filas = []
    print(f"{'lanzado a las':>14s} {'objetivos del dia ya abierto':>29s} {'MAE':>8s} {'MAPE %':>8s}")
    for hora in range(24):
        r = lear_desde(df, hora)
        y, p = r["real"].to_numpy(), r["pred"].to_numpy()
        mismo_dia = 23 - hora if hora < 23 else 0          # objetivos que caen antes de las 00:00 siguientes
        filas.append(dict(hora_lanzamiento=hora, objetivos_mismo_dia=mismo_dia, n=len(r),
                          MAE=float(np.abs(y - p).mean()), MAPE=float(np.mean(np.abs(y - p) / y) * 100)))
        print(f"{hora:>11d}:00 {mismo_dia:>29d} {filas[-1]['MAE']:8.2f} {filas[-1]['MAPE']:8.2f}", flush=True)
    R = pd.DataFrame(filas)
    R.to_csv(RES / "pronostico_movil_24_horas_lanzamiento.csv", index=False)

    m00 = R.loc[R.hora_lanzamiento == 0, "MAE"].iloc[0]
    m23 = R.loc[R.hora_lanzamiento == 23, "MAE"].iloc[0]
    prom = R["MAE"].mean()
    print(f"\nLEAR: 00:00 = {m00:.2f}   23:00 = {m23:.2f}   PROMEDIO 24 horas = {prom:.2f}")
    print(f"  mejor hora de lanzamiento: {int(R.loc[R.MAE.idxmin(),'hora_lanzamiento'])}:00 "
          f"({R.MAE.min():.2f}); peor: {int(R.loc[R.MAE.idxmax(),'hora_lanzamiento'])}:00 ({R.MAE.max():.2f})")
    # escalado al ensamble, anclado en sus dos extremos medidos
    ens00, ens23 = 41.29, 51.82
    frac = (prom - m00) / (m23 - m00) if m23 != m00 else np.nan
    print(f"\nESTIMACION para el ensamble (interpolando entre sus extremos medidos 41.29 y 51.82 con la")
    print(f"  posicion relativa del promedio del LEAR, {100*frac:.0f}% del camino): "
          f"MAE ~ {ens00 + frac*(ens23-ens00):.2f}")
    print(f"\nGuardado: pronostico_movil_24_horas_lanzamiento.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
