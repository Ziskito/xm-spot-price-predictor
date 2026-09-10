# -*- coding: utf-8 -*-
"""
LEAR (LASSO Estimated AutoRegressive), el benchmark lineal estandar del area de pronostico
de precio electrico (Lago et al. 2021). Hoy el proyecto no lo tiene: nuestro benchmark
"clasico" es ARX+GARCH, que no es el estandar del campo. Tenerlo fortalece mucho la
comparacion frente a un jurado tecnico.

Caracteristicas del LEAR, siguiendo el paper:
  - UN MODELO POR HORA del dia (24 modelos independientes), no un modelo global
  - Estimacion con LASSO (seleccion automatica de variables entre muchos regresores)
  - Transformacion estabilizadora de varianza (asinh) sobre el precio, no log, porque
    asinh maneja bien valores cercanos a cero
  - Regresores: precios de dias anteriores a la misma hora, agregados diarios (min/max/media)
    de dias anteriores, exogenas, y dummies de dia de la semana

Se evalua sobre el mismo holdout de 2026 que todos los demas modelos del proyecto.
"""
import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_SALIDA = RAIZ / "data" / "processed" / "resultados" / "lear_2026.csv"
FECHA_TEST = pd.Timestamp("2026-01-01")


def cargar():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
    return df


def construir_matriz_lear(df):
    """Arma la matriz de regresores del LEAR: un registro por (dia, hora)."""
    df = df.copy()
    df["fecha"] = df["fecha_hora"].dt.normalize()
    df["hora"] = df["fecha_hora"].dt.hour

    # Precio en formato ancho: filas = dia, columnas = hora
    ancho = df.pivot_table(index="fecha", columns="hora", values="precio_bolsa")
    ancho = ancho.sort_index()

    # Agregados diarios del precio
    agregados = pd.DataFrame({
        "precio_min_dia": ancho.min(axis=1),
        "precio_max_dia": ancho.max(axis=1),
        "precio_media_dia": ancho.mean(axis=1),
    })

    registros = []
    fechas = ancho.index

    # Exogenas: se toman a nivel dia (media del dia) para las que ya vienen rezagadas
    exogenas_dia = df.groupby("fecha")[
        ["volumen_embalses", "aportes_hidricos", "oni", "demanda_lag24h", "es_pandemia"]
    ].mean()

    for i, fecha in enumerate(fechas):
        if i < 7:  # necesitamos hasta d-7
            continue
        fila_base = {"fecha": fecha}
        # Agregados de dias anteriores
        for retardo in [1, 2, 3, 7]:
            f_ret = fechas[i - retardo]
            for col in agregados.columns:
                fila_base[f"{col}_d{retardo}"] = agregados.loc[f_ret, col]
        # Exogenas del dia objetivo (ya rezagadas en el pipeline base)
        for col in exogenas_dia.columns:
            fila_base[col] = exogenas_dia.loc[fecha, col] if fecha in exogenas_dia.index else np.nan
        # Dummies de dia de la semana
        dow = fecha.dayofweek
        for d in range(7):
            fila_base[f"dow_{d}"] = 1.0 if dow == d else 0.0
        # Precios de dias anteriores, por hora
        for h in range(24):
            fila = dict(fila_base)
            fila["hora"] = h
            for retardo in [1, 2, 3, 7]:
                f_ret = fechas[i - retardo]
                fila[f"precio_d{retardo}_h"] = ancho.loc[f_ret, h] if h in ancho.columns else np.nan
            fila["y"] = ancho.loc[fecha, h] if h in ancho.columns else np.nan
            registros.append(fila)

    return pd.DataFrame(registros)


def main():
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    print("Cargando y armando matriz LEAR...")
    df = cargar()
    matriz = construir_matriz_lear(df)
    matriz = matriz.dropna().reset_index(drop=True)
    print(f"Matriz: {len(matriz):,} registros (dia x hora)")

    columnas_x = [c for c in matriz.columns if c not in ["fecha", "hora", "y"]]

    train = matriz[matriz["fecha"] < FECHA_TEST]
    test = matriz[matriz["fecha"] >= FECHA_TEST]
    print(f"Train: {len(train):,} | Test 2026: {len(test):,}")
    print(f"Regresores: {len(columnas_x)}\n")

    predicciones, reales, horas_ref, fechas_ref = [], [], [], []
    n_coef_activos = []

    for h in range(24):
        tr = train[train["hora"] == h]
        te = test[test["hora"] == h]
        if len(te) == 0:
            continue

        # Transformacion estabilizadora de varianza (asinh), como recomienda el paper
        y_tr = np.arcsinh(tr["y"].values)

        escalador = StandardScaler()
        X_tr = escalador.fit_transform(tr[columnas_x].values)
        X_te = escalador.transform(te[columnas_x].values)

        modelo = LassoCV(cv=5, max_iter=5000, alphas=50, random_state=42, n_jobs=-1)
        modelo.fit(X_tr, y_tr)

        pred = np.sinh(modelo.predict(X_te))
        predicciones.extend(pred)
        reales.extend(te["y"].values)
        horas_ref.extend([h] * len(te))
        fechas_ref.extend(te["fecha"].values)
        n_coef_activos.append(int((modelo.coef_ != 0).sum()))
        print(f"  hora {h:02d}: alpha={modelo.alpha_:.5f}  coeficientes activos="
              f"{n_coef_activos[-1]}/{len(columnas_x)}")

    reales = np.array(reales, float)
    predicciones = np.array(predicciones, float)
    err = np.abs(reales - predicciones)
    mae = err.mean()
    rmse = np.sqrt(((reales - predicciones) ** 2).mean())
    mape = (err / reales).mean() * 100

    print(f"\n{'='*60}")
    print(f"LEAR en holdout 2026:  MAE {mae:.2f}  RMSE {rmse:.2f}  MAPE {mape:.2f}%")
    print(f"Coeficientes activos promedio: {np.mean(n_coef_activos):.1f} de {len(columnas_x)}")
    print(f"{'='*60}")
    print("\nReferencia del proyecto (mismo holdout):")
    print("  Persistencia 56.31 | ARX+GARCH 55.76 | N-BEATSx 56.11 | N-BEATSx ens. 54.89")

    salida = pd.DataFrame({"fecha": fechas_ref, "hora": horas_ref,
                            "real": reales, "prediccion": predicciones})
    salida.to_csv(RUTA_SALIDA, index=False)
    print(f"\nGuardado: {RUTA_SALIDA}")
    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
