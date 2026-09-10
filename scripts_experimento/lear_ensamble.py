# -*- coding: utf-8 -*-
"""
Ensemble LEAR: la configuracion que Lago et al. (2021) proponen como estado del arte para
pronostico de precio electrico. No es LEAR "a secas" sino el promedio de varios LEAR
calibrados con ventanas de distinta longitud.

Dos diferencias importantes frente a nuestros otros modelos:

1. RECALIBRACION RODANTE. Una ventana corta (56 dias) solo tiene sentido si se reentrena
   periodicamente; entrenarla una vez en diciembre y predecir hasta agosto contradice su
   proposito. Aqui se reentrena cada 7 dias. (El paper reentrena diario; semanal es una
   aproximacion para que el costo sea manejable, y se deja anotado como desviacion.)

2. UN MODELO POR HORA. Cada hora del dia tiene su propio LASSO, porque la formacion de
   precio a las 3am y a las 7pm responde a dinamicas distintas.

Ventanas probadas (siguiendo el paper: unas cortas y unas largas):
   56d, 84d  -> cortas, se adaptan rapido a cambios de regimen
   365d      -> intermedia
   728d      -> ~2 anios, la que el paper senala como mejor larga
   1456d     -> ~4 anios
   completa  -> toda la historia (lo que hace hoy el resto del proyecto)

Salidas:
   data/processed/resultados/lear_predicciones.csv   (crudo por ventana)
   data/processed/resultados/lear_ensambles.csv      (metricas individuales y de ensambles)
"""
import pandas as pd
import numpy as np
import time
import sys
import itertools
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_PRED = RAIZ / "data" / "processed" / "resultados" / "lear_predicciones.csv"
RUTA_MET = RAIZ / "data" / "processed" / "resultados" / "lear_ensambles.csv"

FECHA_TEST = pd.Timestamp("2026-01-01")
DIAS_ENTRE_RECALIBRACIONES = 7

VENTANAS = {"56d": 56, "84d": 84, "365d": 365, "728d": 728, "1456d": 1456, "completa": 2557}


def cargar():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    return pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)


def construir_matriz(df):
    """Una fila por (dia, hora) con los regresores tipicos del LEAR."""
    df = df.copy()
    df["fecha"] = df["fecha_hora"].dt.normalize()
    df["hora"] = df["fecha_hora"].dt.hour

    ancho = df.pivot_table(index="fecha", columns="hora", values="precio_bolsa").sort_index()
    agregados = pd.DataFrame({
        "pmin": ancho.min(axis=1), "pmax": ancho.max(axis=1), "pmed": ancho.mean(axis=1),
    })
    exog = df.groupby("fecha")[["volumen_embalses", "aportes_hidricos", "oni",
                                  "demanda_lag24h", "es_pandemia"]].mean()

    fechas = ancho.index
    registros = []
    for i, fecha in enumerate(fechas):
        if i < 7:
            continue
        base = {"fecha": fecha}
        for r in [1, 2, 3, 7]:
            fr = fechas[i - r]
            for c in agregados.columns:
                base[f"{c}_d{r}"] = agregados.loc[fr, c]
        for c in exog.columns:
            base[c] = exog.loc[fecha, c] if fecha in exog.index else np.nan
        dow = fecha.dayofweek
        for d in range(7):
            base[f"dow_{d}"] = 1.0 if dow == d else 0.0
        for h in range(24):
            fila = dict(base)
            fila["hora"] = h
            for r in [1, 2, 3, 7]:
                fr = fechas[i - r]
                fila[f"p_d{r}_h"] = ancho.loc[fr, h] if h in ancho.columns else np.nan
            fila["y"] = ancho.loc[fecha, h] if h in ancho.columns else np.nan
            registros.append(fila)
    return pd.DataFrame(registros).dropna().reset_index(drop=True)


def correr_ventana(matriz, columnas_x, dias_ventana, nombre):
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler

    test_fechas = sorted(matriz.loc[matriz["fecha"] >= FECHA_TEST, "fecha"].unique())
    bloques = [test_fechas[i:i + DIAS_ENTRE_RECALIBRACIONES]
               for i in range(0, len(test_fechas), DIAS_ENTRE_RECALIBRACIONES)]

    alphas_por_hora = {}
    salidas = []
    t0 = time.time()

    for nb, bloque in enumerate(bloques):
        inicio_bloque = pd.Timestamp(bloque[0])
        limite_inferior = inicio_bloque - pd.Timedelta(days=dias_ventana)
        train = matriz[(matriz["fecha"] < inicio_bloque) & (matriz["fecha"] >= limite_inferior)]
        test = matriz[matriz["fecha"].isin(bloque)]
        if len(train) < 24 * 14:
            continue

        for h in range(24):
            tr = train[train["hora"] == h]
            te = test[test["hora"] == h]
            if len(te) == 0 or len(tr) < 14:
                continue

            y_tr = np.arcsinh(tr["y"].values)
            esc = StandardScaler()
            X_tr = esc.fit_transform(tr[columnas_x].values)
            X_te = esc.transform(te[columnas_x].values)

            # El alpha se elige una vez por hora (primera recalibracion) y se reusa,
            # para que el costo no explote con 6 ventanas x 31 recalibraciones x 24 horas.
            if h not in alphas_por_hora:
                cv_modelo = LassoCV(cv=3, alphas=25, max_iter=3000, random_state=42, n_jobs=1)
                cv_modelo.fit(X_tr, y_tr)
                alphas_por_hora[h] = cv_modelo.alpha_
                modelo = cv_modelo
            else:
                modelo = Lasso(alpha=alphas_por_hora[h], max_iter=3000)
                modelo.fit(X_tr, y_tr)

            pred = np.sinh(modelo.predict(X_te))
            salidas.append(pd.DataFrame({
                "fecha": te["fecha"].values, "hora": h,
                "real": te["y"].values, "pred": pred, "ventana": nombre,
            }))

        if (nb + 1) % 10 == 0:
            print(f"    {nombre}: bloque {nb+1}/{len(bloques)} | {(time.time()-t0)/60:.1f} min")
            sys.stdout.flush()

    return pd.concat(salidas, ignore_index=True) if salidas else None


def metricas(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = np.abs(y - p)
    return e.mean(), np.sqrt(((y - p) ** 2).mean()), (e / y).mean() * 100


def main():
    t0 = time.time()
    print("Construyendo matriz LEAR...")
    matriz = construir_matriz(cargar())
    columnas_x = [c for c in matriz.columns if c not in ["fecha", "hora", "y"]]
    print(f"Matriz: {len(matriz):,} filas | {len(columnas_x)} regresores")
    print(f"Recalibracion cada {DIAS_ENTRE_RECALIBRACIONES} dias\n")

    partes = []
    for nombre, dias in VENTANAS.items():
        t = time.time()
        res = correr_ventana(matriz, columnas_x, dias, nombre)
        if res is None:
            print(f"  {nombre:10s} sin resultados")
            continue
        mae, rmse, mape = metricas(res["real"], res["pred"])
        print(f"  {nombre:10s} MAE {mae:6.2f}  RMSE {rmse:6.2f}  MAPE {mape:5.2f}%  "
              f"| n={len(res):,} | {(time.time()-t)/60:.1f} min")
        sys.stdout.flush()
        partes.append(res)
        pd.concat(partes, ignore_index=True).to_csv(RUTA_PRED, index=False)

    df = pd.concat(partes, ignore_index=True)
    df["ts"] = pd.to_datetime(df["fecha"]) + pd.to_timedelta(df["hora"], unit="h")
    pivote = df.pivot_table(index="ts", columns="ventana", values="pred")
    y_real = df.groupby("ts")["real"].first().reindex(pivote.index)

    filas = []
    nombres = [n for n in VENTANAS if n in pivote.columns]
    for k in range(1, len(nombres) + 1):
        for combo in itertools.combinations(nombres, k):
            sub = pivote[list(combo)].dropna()
            if len(sub) == 0:
                continue
            yy = y_real.reindex(sub.index)
            mae, rmse, mape = metricas(yy.values, sub.mean(axis=1).values)
            filas.append({"ensamble": "+".join(combo), "n_ventanas": k,
                           "mae": mae, "rmse": rmse, "mape": mape, "n_obs": len(sub)})

    res = pd.DataFrame(filas).sort_values("mae")
    res.to_csv(RUTA_MET, index=False)

    print(f"\n{'='*74}\nMEJORES 15 COMBINACIONES\n{'='*74}")
    print(res.head(15).round(3).to_string(index=False))
    print(f"\n{'='*74}\nVENTANAS INDIVIDUALES\n{'='*74}")
    print(res[res["n_ventanas"] == 1].round(3).to_string(index=False))
    print("\nReferencia del proyecto (holdout 2026):")
    print("  Persistencia 56.31 | ARX+GARCH 55.76 | N-BEATSx 56.11 | N-BEATSx ensamble 54.89")
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
