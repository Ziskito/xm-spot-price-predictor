# -*- coding: utf-8 -*-
"""
Tamiz de fuga/redundancia para las variables nuevas.

Aprendizaje de esta sesion (dos veces ya): una correlacion alta con el precio NO significa
que la variable sirva. Hay dos formas de que no sirva:

  FUGA        la variable se determina en el mismo despacho que el precio, asi que su valor
              contemporaneo "ya contiene" el precio. Ej: MaxPrecOferNal, corr 0.9998.
  REDUNDANCIA rezagada, la variable no aporta nada que precio_lag24h no tenga ya. Ej: la
              relacion hidro/termica, o esa misma MaxPrecOferNal rezagada (corr 0.9306,
              identica a la autocorrelacion del propio precio).

Criterio de tamizaje, para cada variable X:
  1. corr(X_t, precio_t)          muy alta (>0.95) -> sospecha fuerte de fuga
  2. corr(X_{t-24}, precio_t)     comparada con corr(precio_{t-24}, precio_t)=0.9306
                                   si es parecida o menor, no aporta por si sola
  3. CORRELACION PARCIAL: corr(X_{t-24}, precio_t | precio_{t-24}) -- lo que realmente
     importa. Mide cuanta informacion queda en X DESPUES de descontar lo que el precio
     rezagado ya explica. Esta es la prueba decisiva.

Solo las variables con correlacion parcial no trivial pasan a probarse en los modelos.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

ARCHIVOS = {
    "max_precio_oferta": "max_precio_oferta_2019_2026.csv",
    "costo_marginal_despacho": "costo_marginal_despacho_2019_2026.csv",
    "generacion_ideal": "generacion_ideal_2019_2026.csv",
    "demanda_comercial": "demanda_comercial_2019_2026.csv",
    "restricciones_aliviadas": "restricciones_aliviadas_2019_2026.csv",
    "perdidas_energia": "perdidas_energia_2019_2026.csv",
    "importaciones_energia": "importaciones_energia_2019_2026.csv",
    "exportaciones_energia": "exportaciones_energia_2019_2026.csv",
    "precio_escasez": "precio_escasez_2019_2026.csv",
    "precio_escasez_marginal": "precio_escasez_marginal_2019_2026.csv",
    "aportes_media_historica": "aportes_media_historica_2019_2026.csv",
    "compras_arranque_parada": "compras_arranque_parada_2019_2026.csv",
}


def correlacion_parcial(x, y, z):
    """corr(x, y | z): correlacion entre x e y descontando la influencia de z."""
    datos = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(datos) < 100:
        return np.nan
    # Residuos de regresar x y y sobre z
    def residuo(v):
        A = np.vstack([np.ones(len(datos)), datos["z"].values]).T
        coef, *_ = np.linalg.lstsq(A, datos[v].values, rcond=None)
        return datos[v].values - A @ coef
    rx, ry = residuo("x"), residuo("y")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    base = pd.concat([a, b], ignore_index=True)[["fecha_hora", "precio_bolsa"]].sort_values("fecha_hora")

    # Referencia: cuanto explica el precio rezagado por si solo
    base["precio_lag24h"] = base["precio_bolsa"].shift(24)
    ref = base["precio_lag24h"].corr(base["precio_bolsa"])
    print(f"REFERENCIA -- corr(precio_t-24h, precio_t) = {ref:.4f}")
    print("Una variable solo aporta si su correlacion PARCIAL (ultima columna) no es trivial.\n")

    filas = []
    for nombre, archivo in ARCHIVOS.items():
        ruta = RAIZ / "data" / archivo
        if not ruta.exists():
            print(f"  {nombre:26s} (aun no descargada)")
            continue

        df = pd.read_csv(ruta, parse_dates=["fecha_hora"])
        col = [c for c in df.columns if c != "fecha_hora"][0]

        # Las diarias se expanden a horaria por merge_asof hacia adelante
        es_diaria = df["fecha_hora"].dt.hour.nunique() == 1
        if es_diaria:
            m = pd.merge_asof(base.sort_values("fecha_hora"), df.sort_values("fecha_hora"),
                               on="fecha_hora", direction="backward")
        else:
            m = base.merge(df, on="fecha_hora", how="left")

        m = m.sort_values("fecha_hora")
        m["x_lag24"] = m[col].shift(24)

        corr_cont = m[col].corr(m["precio_bolsa"])
        corr_lag = m["x_lag24"].corr(m["precio_bolsa"])
        corr_parc = correlacion_parcial(m["x_lag24"], m["precio_bolsa"], m["precio_lag24h"])

        if pd.isna(corr_cont):
            veredicto = "SIN DATOS UTILES"
        elif abs(corr_cont) > 0.95:
            veredicto = "FUGA (no usar contemporanea)"
        elif pd.isna(corr_parc) or abs(corr_parc) < 0.05:
            veredicto = "REDUNDANTE"
        elif abs(corr_parc) < 0.15:
            veredicto = "aporte debil"
        else:
            veredicto = ">>> CANDIDATA REAL <<<"

        filas.append({"variable": nombre, "n": int(m[col].notna().sum()),
                       "corr_contemp": corr_cont, "corr_lag24h": corr_lag,
                       "corr_parcial": corr_parc, "veredicto": veredicto})
        print(f"  {nombre:26s} n={int(m[col].notna().sum()):6,}  contemp {corr_cont:+.4f}  "
              f"lag24h {corr_lag:+.4f}  PARCIAL {corr_parc:+.4f}   {veredicto}")

    if filas:
        res = pd.DataFrame(filas)
        ruta_out = RAIZ / "data/processed/resultados/tamiz_variables_nuevas.csv"
        res.to_csv(ruta_out, index=False)
        print(f"\nGuardado: {ruta_out}")
        candidatas = res[res["veredicto"].str.contains("CANDIDATA|debil")]
        print(f"\nCandidatas a probar en los modelos: {list(candidatas['variable'])}")


if __name__ == "__main__":
    main()
