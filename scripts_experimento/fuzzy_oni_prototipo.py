# -*- coding: utf-8 -*-
"""
Prototipo de logica difusa sobre el indice ONI, para revisar antes de integrarlo al pipeline
compartido (05). NO modifica dataset_features todavia -- solo genera una vista previa.

Motivacion (idea del usuario 2026-09-10): "oni" ya es una variable continua en el pipeline (no
binaria), pero el modelo tiene muy pocos episodios extremos en el historico para aprender solos
la no-linealidad de los umbrales NOAA. Codificar el grado de pertenencia a cada categoria de
fuerza (debil/moderado/fuerte/muy fuerte) como funciones de membresia triangulares le da esa
pista explicita, en vez de forzarlo a inferir el umbral con pocos ejemplos.

Umbrales NOAA estandar para el ONI (valor absoluto):
  debil     0.5 - 0.9
  moderado  1.0 - 1.4
  fuerte    1.5 - 1.9
  muy fuerte >= 2.0

Ademas se agrega 'velocidad_cambio_oni': tasa de cambio del ONI en una ventana de 30 dias,
motivada por el hallazgo de la sesion (Origen 4 y Origen 6: las rachas de dias malos coinciden
con transiciones RAPIDAS de regimen, no solo con el nivel absoluto del ONI -- correlacion pooled
de |cambio ONI| vs MAPE = 0.254, mas alta que la de nivel ONI vs MAPE = 0.134).
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")


def membresia_triangular(x, a, b, c):
    """Funcion de membresia triangular estandar: 0 fuera de [a,c], pico=1 en b."""
    x = np.asarray(x, dtype=float)
    izq = (x - a) / (b - a) if b > a else np.ones_like(x)
    der = (c - x) / (c - b) if c > b else np.ones_like(x)
    return np.clip(np.minimum(izq, der), 0, 1)


def construir_features_fuzzy_oni(oni):
    """oni: Serie con el indice ONI (positivo=Nino, negativo=Nina). Devuelve DataFrame con las
    columnas de membresia difusa + velocidad de cambio."""
    oni = pd.Series(oni).astype(float)
    intensidad = oni.abs()
    signo_nino = (oni > 0).astype(float)
    signo_nina = (oni < 0).astype(float)

    # Umbrales NOAA como funciones triangulares solapadas (transicion suave entre categorias,
    # en vez de un corte duro tipo "si ONI>=1.5 entonces fuerte")
    debil     = membresia_triangular(intensidad, 0.3, 0.7, 1.1)
    moderado  = membresia_triangular(intensidad, 0.8, 1.2, 1.6)
    fuerte    = membresia_triangular(intensidad, 1.3, 1.7, 2.1)
    muy_fuerte = np.clip((intensidad - 1.9) / 0.4, 0, 1)  # rampa abierta, no baja despues de 2.0+

    df = pd.DataFrame({
        "fuzzy_nino_debil":      debil * signo_nino,
        "fuzzy_nino_moderado":   moderado * signo_nino,
        "fuzzy_nino_fuerte":     fuerte * signo_nino,
        "fuzzy_nino_muy_fuerte": muy_fuerte * signo_nino,
        "fuzzy_nina_debil":      debil * signo_nina,
        "fuzzy_nina_moderado":   moderado * signo_nina,
        "fuzzy_nina_fuerte":     fuerte * signo_nina,
        "fuzzy_nina_muy_fuerte": muy_fuerte * signo_nina,
    })

    # Velocidad de cambio: derivada suavizada del ONI en ventana de 30 dias (asumiendo datos
    # horarios, 24*30 filas), en valor absoluto normalizado 0-1 por el rango historico observado
    ventana = 24 * 30
    cambio_30d = oni.diff(ventana)
    df["velocidad_cambio_oni"] = cambio_30d
    rango_hist = cambio_30d.abs().quantile(0.99)
    df["fuzzy_transicion_rapida"] = np.clip(cambio_30d.abs() / rango_hist, 0, 1)

    return df


if __name__ == "__main__":
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

    fuzzy = construir_features_fuzzy_oni(df["oni"])
    vista = pd.concat([df[["fecha_hora", "oni"]], fuzzy], axis=1)

    print("Vista previa en fechas clave (episodios conocidos):\n")
    fechas_interes = ["2020-08-01", "2021-08-01", "2022-11-01", "2023-11-15",
                       "2024-02-01", "2026-06-15", "2026-07-01"]
    for f in fechas_interes:
        fila = vista[vista["fecha_hora"] == pd.Timestamp(f) + pd.Timedelta(hours=12)]
        if len(fila):
            r = fila.iloc[0]
            print(f"{f}  oni={r['oni']:+.2f}  "
                  f"nino[d/m/f/mf]=[{r['fuzzy_nino_debil']:.2f}/{r['fuzzy_nino_moderado']:.2f}/{r['fuzzy_nino_fuerte']:.2f}/{r['fuzzy_nino_muy_fuerte']:.2f}]  "
                  f"nina[d/m/f/mf]=[{r['fuzzy_nina_debil']:.2f}/{r['fuzzy_nina_moderado']:.2f}/{r['fuzzy_nina_fuerte']:.2f}/{r['fuzzy_nina_muy_fuerte']:.2f}]  "
                  f"transicion_rapida={r['fuzzy_transicion_rapida']:.2f}")

    print(f"\nFilas totales: {len(vista)}  |  NaN en velocidad_cambio_oni (primeros 30 dias sin historia): {vista['velocidad_cambio_oni'].isna().sum()}")

    ruta_out = RAIZ / "data" / "processed" / "resultados" / "fuzzy_oni_preview.csv"
    vista.to_csv(ruta_out, index=False)
    print(f"\nGuardado: {ruta_out}")
