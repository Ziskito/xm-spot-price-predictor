# -*- coding: utf-8 -*-
"""
Problema 1, adaptado a 2026: RECALIBRACION de los modelos base durante el ano de publicacion.
VERSION 2 -- corrige DOS bugs encontrados en recalibracion_2026.py (v1, ver bitacora 2026-09-11
tarde) y un tercero encontrado al depurar esta v2 (ver bitacora 2026-09-11 noche).

Bug 1 (v1): usaba el parametro nativo `refit=N` de NeuralForecast.cross_validation, que hace
warm-start (continua desde los pesos ya ajustados) en vez de reiniciar el modelo -- degradaba el
resultado en vez de mejorarlo. Corregido creando una instancia NUEVA de NBEATSx/NHITS en cada punto
de reentrenamiento (mismo criterio que nbeatsx_reentrenado_junio.py).

Bug 2 (encontrado al escribir esta v2): `predict(futr_df=futr)` de neuralforecast SOLO pronostica
los h pasos INMEDIATAMENTE siguientes al final de los datos con los que el modelo fue entrenado (o
de los que se le pasen explicitamente en `df=`) -- no "la ventana que se le pida" segun las fechas
de `futr_df`. Reentrenar cada 30 dias pero seguir llamando a `predict(futr_df=...)` con ventanas de
dias intermedios (sin pasar `df=`) revienta con "missing combinations of ids and times" en la
SEGUNDA ventana tras cada reentrenamiento (la primera coincide por casualidad con el fin del
entrenamiento, y por eso una prueba aislada de una sola ventana no lo detectaba). Corregido pasando
siempre `df=` con el historial real observado hasta el corte de esa ventana (normalizado con las
MISMAS estadisticas del ultimo reentrenamiento) -- exactamente lo que hace por dentro
`NeuralForecast.cross_validation(..., refit=N)` para las ventanas que no reentrena
(`predict_df = train` en su codigo fuente), pero aqui con pesos congelados entre reentrenamientos.

Bug 3: la normalizacion de "oni" (unica exogena futura continua) se aplicaba a los datos de
ENTRENAMIENTO pero nunca a `futr_df` -- el modelo entrenaba viendo "oni" en escala normalizada
(z-score) pero al predecir recibia el ONI crudo (rango -1 a 2), una escala completamente distinta.
Esto no producia un error (ninguna validacion de neuralforecast lo detecta), simplemente predicciones
mal calibradas -- el tipo de bug mas peligroso porque no avisa. Corregido normalizando "oni" en
`futr_df` con las mismas estadisticas congeladas del ultimo reentrenamiento.

Costo esperado: ~n_win/REFIT reentrenamientos completos (7-8 con REFIT=30) + 216 llamadas a predict
(baratas). Uso: python recalibracion_2026_v2.py [REFIT]
"""
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, particion_o6, columnas_features, crudas_o6, HIST, FUT_CONT, FUT_BIN, RES

FUT = FUT_CONT + FUT_BIN
RUTA = RES / "recalibracion_2026_v2_predicciones.csv"


def normalizar(tr_o_futr, stats, cols):
    out = tr_o_futr.copy()
    for c in cols:
        mu, sd = stats[c]
        out[c] = (out[c] - mu) / sd
    return out


def preparar(tr, corte, cols_norm=None, stats=None):
    """Arma el DataFrame en formato neuralforecast (unique_id/ds/y + exogenas). Si stats es None,
    calcula media/desviacion de tr y las devuelve (para congelarlas); si se pasan, las reutiliza."""
    o = tr[["fecha_hora", "precio_bolsa"] + HIST + FUT].copy()
    if stats is None:
        stats = {c: (o[c].mean(), o[c].std()) for c in HIST + FUT_CONT}
    o = normalizar(o, stats, HIST + FUT_CONT)
    o["unique_id"] = "precio_bolsa"
    o = o.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})
    return o[["unique_id", "ds", "y"] + HIST + FUT], stats


def main():
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS

    refit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    df = cargar_completo()
    cf = columnas_features(df)
    _, test = particion_o6(df)
    n_win = len(test) // 24
    horas = test["fecha_hora"].reset_index(drop=True)
    ventanas = [horas.iloc[i * 24:(i + 1) * 24] for i in range(n_win)]
    dfi = df.set_index("fecha_hora")
    out = pd.DataFrame({"fecha_hora": horas.iloc[:n_win * 24]})
    out["real"] = dfi.loc[out["fecha_hora"], "precio_bolsa"].to_numpy()

    comunes = dict(h=24, input_size=168, hist_exog_list=HIST, futr_exog_list=FUT, max_steps=1000,
                   val_check_steps=100, random_seed=42, enable_progress_bar=False)
    pred_nbx, pred_nhits = [], []
    modelo, stats = None, None
    t0 = time.time()
    for i, w in enumerate(ventanas):
        corte = w.iloc[0]
        tr = df[df["fecha_hora"] < corte].dropna(subset=cf + ["precio_bolsa"])
        if i % refit == 0:
            nf, stats = preparar(tr, corte)
            modelo = NeuralForecast(models=[NBEATSx(**comunes), NHITS(**comunes)], freq="h")
            modelo.fit(df=nf)
            print(f"  reentrenado en ventana {i} (corte {corte.date()}, {time.time()-t0:.0f}s acumulados)", flush=True)
            df_hist = None  # el modelo ya tiene esta historia internamente tras fit()
        else:
            df_hist, _ = preparar(tr, corte, stats=stats)  # extiende con datos reales, pesos congelados

        futr = df.loc[df["fecha_hora"].isin(w), ["fecha_hora"] + FUT].rename(columns={"fecha_hora": "ds"})
        futr["unique_id"] = "precio_bolsa"
        futr = normalizar(futr, stats, FUT_CONT)  # bug 3: "oni" tambien debe normalizarse
        p = modelo.predict(df=df_hist, futr_df=futr)
        pred_nbx.append(p["NBEATSx"].to_numpy())
        pred_nhits.append(p["NHITS"].to_numpy())
        if i in (0, 1, refit, refit + 1):
            print(f"    ventana {i}: OK (df_hist={'interno' if df_hist is None else len(df_hist)})", flush=True)
    out["NBEATSx_rec"] = np.concatenate(pred_nbx)
    out["NHITS_rec"] = np.concatenate(pred_nhits)
    out.to_csv(RUTA, index=False)
    print(f"Redes recalibradas desde cero (refit={refit}): {(time.time()-t0)/60:.1f} min", flush=True)

    c = crudas_o6()
    m = out.set_index("fecha_hora").join(c[["N-BEATSx", "N-HiTS"]], how="inner")
    real = m["real"]
    print(f"\n{'modelo':10s} {'congelado':>10s} {'recalibrado v2':>15s}")
    for rec, cong in [("NBEATSx_rec", "N-BEATSx"), ("NHITS_rec", "N-HiTS")]:
        print(f"{cong:10s} {(real-m[cong]).abs().mean():10.2f} {(real-m[rec]).abs().mean():15.2f}")
    print(f"Guardado: {RUTA}")


if __name__ == "__main__":
    main()
