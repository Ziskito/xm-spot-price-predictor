# -*- coding: utf-8 -*-
"""
AUDITORIA: el ONI del pipeline contiene informacion futura. Cuanto importa?

EL PROBLEMA ENCONTRADO (2026-09-18). El archivo `data/external/oni_index.csv` viene en el formato
estandar de NOAA, con columnas DJF, JFM, FMA, ... que son medias moviles de 3 meses CENTRADAS.
El pipeline (notebook 02, `construir_oni_horario_causal`) mapea cada columna a su mes central
(JJA -> julio) y construye una rampa lineal desde ONI(M-1) al inicio del mes M hasta ONI(M) al
final. El problema es que:

    ONI(M)   = media de SST de los meses M-1, M, M+1   -> incluye el mes SIGUIENTE
    ONI(M-1) = media de SST de los meses M-2, M-1, M   -> incluye el mes EN CURSO

O sea que los dos extremos de la rampa contienen informacion que todavia no ha ocurrido: hasta
~1.5 meses de futuro. La funcion se llama "causal" pero no lo es respecto al dato subyacente.
(Aparte de eso, NOAA publica el valor de un mes a comienzos del mes siguiente, asi que ni siquiera
el valor del mes anterior esta disponible en tiempo real.)

QUE SE MIDE AQUI. Se construye un ONI genuinamente causal -- para una hora del mes M se usa
ONI(M-2), que es media de los meses M-3, M-2, M-1: enteramente pasado, y ya publicado -- y se
reentrena el mismo modelo con las dos versiones, todo lo demas identico.

Hipotesis previa: el efecto deberia ser despreciable, porque la ablacion LOGO del proyecto ya
mostro que quitar el ONI por completo mueve el MAE dentro del ruido (-1.36, con el signo cambiando
entre semillas). Si eso se confirma, la fuga es real pero inocua. Si NO se confirma, hay un
problema serio que obliga a reentrenar el pipeline.

Se prueba sobre XGBoost (barato y usa el conjunto completo de features) en el Origen 6.
"""
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

warnings.filterwarnings("ignore")
RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"

CORTE = "2026-01-01"
TEST_FIN = "2026-08-05"


def oni_causal(indice_ref, ruta_oni, meses_atras=2):
    """ONI sin futuro: para el mes M usa el valor de ONI(M - meses_atras).

    Con meses_atras=2, el valor usado es la media de los meses M-3, M-2, M-1: todo pasado, y ya
    publicado por NOAA al momento de pronosticar.
    """
    ancho = pd.read_csv(ruta_oni)
    cols = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    mes_central = {t: i + 1 for i, t in enumerate(cols)}
    largo = ancho.melt(id_vars="anio", value_vars=cols, var_name="temporada", value_name="oni")
    largo["mes"] = largo["temporada"].map(mes_central)
    largo = largo[["anio", "mes", "oni"]].sort_values(["anio", "mes"]).reset_index(drop=True)
    # desplazar el calendario: el valor de la fila pasa a regir 'meses_atras' meses despues
    largo["periodo"] = pd.PeriodIndex(year=largo["anio"], month=largo["mes"], freq="M") + meses_atras
    mapa = largo.set_index("periodo")["oni"]

    h = pd.DataFrame({"fecha_hora": pd.DatetimeIndex(indice_ref)})
    per = h["fecha_hora"].dt.to_period("M")
    h["oni_causal"] = per.map(mapa).astype(float)
    # rampa suave entre el valor del mes anterior y el del mes actual (ambos ya pasados)
    prev = (per - 1).map(mapa).astype(float)
    ini = h["fecha_hora"].values.astype("datetime64[M]").astype("datetime64[ns]")
    frac = (((h["fecha_hora"] - ini) / pd.Timedelta(hours=1)) /
            (h["fecha_hora"].dt.days_in_month * 24)).clip(0, 1)
    h["oni_causal"] = prev + frac * (h["oni_causal"] - prev)
    return h.set_index("fecha_hora")["oni_causal"]


def dm(real, p1, p2, maxlags=23):
    d = np.abs(real - p1) - np.abs(real - p2)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def main():
    t0 = time.time()
    import xgboost as xgb

    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

    oni_c = oni_causal(df["fecha_hora"], RAIZ / "data/external/oni_index.csv")
    df["oni_causal"] = oni_c.to_numpy()

    print("=" * 84)
    print("COMPARACION DE LAS DOS VERSIONES DEL ONI")
    print("=" * 84)
    cmp = df[["fecha_hora", "oni", "oni_causal"]].dropna()
    print(f"  correlacion entre ambas: {cmp['oni'].corr(cmp['oni_causal']):.4f}")
    print(f"  diferencia media absoluta: {(cmp['oni'] - cmp['oni_causal']).abs().mean():.3f}")
    print(f"  diferencia maxima:         {(cmp['oni'] - cmp['oni_causal']).abs().max():.3f}")
    m26 = cmp[cmp.fecha_hora >= "2026-01-01"].set_index("fecha_hora").resample("ME").last()
    print("\n  Valores al cierre de mes en 2026 (el periodo de prueba):")
    print(m26.round(2).to_string())

    excl = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora",
            "dia_semana", "dia_anio", "oni_causal"]
    feats = [c for c in df.columns if c not in excl]

    train = df[df["fecha_hora"] < CORTE].dropna(subset=feats).copy()
    test = df[(df["fecha_hora"] >= CORTE) & (df["fecha_hora"] <= TEST_FIN)].copy()
    print(f"\nTrain: {len(train):,}  Test: {len(test):,}  Features: {len(feats)}")

    resultados = {}
    for etiqueta, col_oni in [("ONI actual (con futuro)", "oni"), ("ONI causal (sin futuro)", "oni_causal")]:
        f = [c if c != "oni" else col_oni for c in feats]
        Xtr, ytr = train[f], np.log(train["precio_bolsa"])
        m = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.8,
                             colsample_bytree=0.8, random_state=42)
        m.fit(Xtr, ytr)
        pred = np.exp(m.predict(test[f]))
        y = test["precio_bolsa"].to_numpy()
        resultados[etiqueta] = pred
        print(f"\n{etiqueta}:  MAE={np.abs(y-pred).mean():.2f}  "
              f"MAPE={(np.abs(y-pred)/y).mean()*100:.2f}%")
        imp = pd.Series(m.feature_importances_, index=f).sort_values(ascending=False)
        pos = list(imp.index).index(col_oni) + 1
        print(f"   importancia de {col_oni}: {imp[col_oni]:.4f}  (puesto {pos} de {len(f)})")

    y = test["precio_bolsa"].to_numpy()
    p1, p2 = resultados["ONI actual (con futuro)"], resultados["ONI causal (sin futuro)"]
    t, pv = dm(y, p1, p2)
    print(f"\n{'='*84}")
    print(f"Diebold-Mariano entre ambas versiones: t={t:+.3f}  p={pv:.4f}")
    print(f"  (t>0 = la version causal es MEJOR; p>0.05 = la fuga no cambia nada medible)")
    print(f"  diferencia de MAE: {np.abs(y-p2).mean() - np.abs(y-p1).mean():+.3f} COP/kWh")
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
