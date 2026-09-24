# -*- coding: utf-8 -*-
"""
¿CUANTO DEL DESEMPENO DE 24h DEPENDE DE QUE EL CORTE CAIGA DENTRO DEL DIA QUE SE PRONOSTICA?

EL HALLAZGO. La igualdad EXACTA entre el precio de una hora y el de la anterior es de 29-77% en
todas las transiciones del dia, y de exactamente 0% en la transicion 23:00 -> 00:00:

    00:  0%  01: 73%  02: 71%  03: 77%  04: 72%  05: 57%  06: 68% ... 22: 35%  23: 34%

En Colombia las plantas ofertan un precio por DIA; el precio horario salta entre esos niveles. Que
la unica transicion sin repeticiones sea 23->00 dice que el dia de ofertas (dia de despacho) va de
00:00 a 23:00.

EL PROBLEMA. El protocolo day-ahead del proyecto usa ventanas 01:00 -> 00:00 del dia siguiente con
CORTE a las 00:00. Entonces:
  - el corte (00:00 del dia D) es la PRIMERA hora del dia de despacho D;
  - 23 de los 24 objetivos (01:00 a 23:00 de D) son del MISMO dia de despacho que el corte;
  - solo el ultimo objetivo (00:00 de D+1) es de un dia nuevo.
El modelo conoce un precio formado con las mismas ofertas que sus objetivos. No es fuga del
objetivo -- ningun valor objetivo entra como variable --, pero si es informacion que un pronostico
day-ahead real no tendria: el precio de las 00:00 de D se conoce, como pronto, al terminar esa hora.

Esto explicaria tres cosas ya observadas:
  1. las horas 1-5 tienen MAE bajisimo (7.6 a 20.9);
  2. la hora 0 (00:00 de D+1, el unico objetivo de un dia nuevo) es la peor hora no-pico y la unica
     donde el modelo NO le gana a la persistencia;
  3. la "correccion" de ventana del N-BEATSx que lo empeoro de 46.82 a 57.38 -- probablemente era el
     protocolo honesto, y lo que se "perdio" era esta informacion del mismo dia.

LA PRUEBA. El MISMO LEAR (mismas variables, misma regularizacion, mismo periodo) con dos cortes:
  A (actual)  corte 00:00 de D   -> objetivos 01:00 de D ... 00:00 de D+1
  B (honesto) corte 23:00 de D-1 -> objetivos 00:00 de D ... 23:00 de D   (solo dias de despacho
              anteriores; es un day-ahead de verdad respecto del dia de ofertas)
Se comparan hora por hora del reloj sobre los mismos instantes objetivo.

NO SE CAMBIA NADA DEL PROYECTO. Esto solo mide. La decision de que protocolo reportar es del usuario.
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


def lear(df, idx_cortes, etiqueta):
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    fechas = df["fecha_hora"]
    fc_all = fechas.iloc[idx_cortes].to_numpy()
    tr_c = idx_cortes[fc_all < np.datetime64(CORTE - pd.Timedelta(days=1))]
    te_c = idx_cortes[(fc_all >= np.datetime64(CORTE - pd.Timedelta(hours=1))) & (fc_all < np.datetime64(FIN))]
    partes, alphas = [], []
    for h in range(1, 25):
        Xtr, ytr, _, _ = construir_para_paso(df, h, tr_c)
        Xte, yte, fo, fcut = construir_para_paso(df, h, te_c)
        sc = StandardScaler().fit(Xtr)
        if h <= 4 or h % 6 == 0:
            alphas.append(LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000)
                          .fit(sc.transform(Xtr), ytr).alpha_)
        m = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(sc.transform(Xtr), ytr)
        partes.append(pd.DataFrame({"fecha_hora": fo, "real": yte, "pred": m.predict(sc.transform(Xte)),
                                    "paso": h}))
    r = pd.concat(partes, ignore_index=True)
    print(f"  {etiqueta}: {len(r):,} pronosticos", flush=True)
    return r


def main():
    t0 = time.time()
    df = cargar_completo().reset_index(drop=True)
    hora = df["fecha_hora"].dt.hour.to_numpy()
    idx = np.arange(len(df))
    cortes_A = idx[(hora == 0) & (idx >= 200)]
    cortes_B = idx[(hora == 23) & (idx >= 200)]
    print("Ajustando el mismo LEAR con dos cortes...")
    A = lear(df, cortes_A, "A corte 00:00 (actual)")
    B = lear(df, cortes_B, "B corte 23:00 del dia previo (honesto)")

    pers = df.set_index("fecha_hora")["precio_bolsa"]
    comun = pd.Index(A["fecha_hora"]).intersection(pd.Index(B["fecha_hora"]))
    A = A.set_index("fecha_hora").loc[comun]
    B = B.set_index("fecha_hora").loc[comun]
    y = A["real"].to_numpy()
    p24 = pers.reindex(comun - pd.Timedelta(hours=24)).to_numpy()
    hh = comun.hour

    print(f"\nInstantes objetivo comunes: {len(comun):,}")
    print("=" * 92)
    print(f"  {'':24s} {'MAE':>8s} {'MAPE %':>8s}")
    for nom, p in [("A  corte 00:00 (actual)", A["pred"].to_numpy()),
                   ("B  corte 23:00 (honesto)", B["pred"].to_numpy()),
                   ("persistencia (t-24h)", p24)]:
        ok = ~np.isnan(p)
        print(f"  {nom:24s} {np.abs(y[ok]-p[ok]).mean():8.2f} {100*np.mean(np.abs(y[ok]-p[ok])/y[ok]):8.2f}")

    print("\n  POR HORA DEL RELOJ")
    print(f"  {'hora':>5s} {'A actual':>9s} {'B honesto':>10s} {'B-A':>7s} {'persist.':>9s} "
          f"{'A gana a pers.':>15s} {'B gana a pers.':>15s}")
    filas = []
    for h in range(24):
        g = hh == h
        a = np.abs(y[g] - A["pred"].to_numpy()[g]).mean()
        b = np.abs(y[g] - B["pred"].to_numpy()[g]).mean()
        q = np.nanmean(np.abs(y[g] - p24[g]))
        filas.append(dict(hora=h, MAE_A=a, MAE_B=b, MAE_persist=q))
        print(f"  {h:5d} {a:9.2f} {b:10.2f} {b-a:+7.2f} {q:9.2f} {100*(1-a/q):+14.1f}% {100*(1-b/q):+14.1f}%")
    R = pd.DataFrame(filas)
    R.to_csv(RES / "protocolo_dayahead_honesto.csv", index=False)
    print(f"\n  Ganancia de A sobre B concentrada en horas 1-5: "
          f"{(R.loc[R.hora.between(1,5),'MAE_B']-R.loc[R.hora.between(1,5),'MAE_A']).mean():+.2f} de MAE medio; "
          f"horas 6-23: {(R.loc[R.hora.between(6,23),'MAE_B']-R.loc[R.hora.between(6,23),'MAE_A']).mean():+.2f}")
    print(f"\nGuardado: protocolo_dayahead_honesto.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
