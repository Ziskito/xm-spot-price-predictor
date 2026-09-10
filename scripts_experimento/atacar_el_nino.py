# -*- coding: utf-8 -*-
"""
Ataque combinado al error de El Nino en N-BEATSx, con las 3 ideas de mayor impacto
esperado encontradas en el diagnostico de esta sesion:

  HALLAZGO 1: El Nino es 10.3% del entrenamiento (2019-2025) pero 30.4% del holdout 2026.
              El modelo ve poco de lo que mas le toca predecir.
  HALLAZGO 2: costo_marginal_despacho y compras_arranque_parada tienen correlacion parcial
              (descontando el precio rezagado) mas fuerte DENTRO de El Nino que en el resto:
                costo_marginal_despacho:  global 0.178 -> el_nino 0.199
                compras_arranque_parada:  global -0.027 -> el_nino -0.102 (invisible en el promedio)
  HALLAZGO 3 (hipotesis, no probada aun): una interaccion explicita oni x hidrologia podria
              ayudarle a la red a aprender el cambio de regimen con pocos datos de El Nino.

Configuraciones (todas parten del pipeline oficial, se van sumando ideas):
  BASE            pipeline actual, sin cambios
  DUP_NINO        BASE + el episodio de El Nino de entrenamiento (ago2023-abr2024, 6215h)
                  duplicado como una SERIE ADICIONAL (unique_id distinto) en el panel de
                  entrenamiento -- el modelo lo ve dos veces por epoca sin romper ventanas
  VARS_NINO       BASE + costo_marginal_despacho + compras_arranque_parada (rezagadas 24h)
  INTERACCION     BASE + oni*aportes_hidricos_media_7d, oni*volumen_embalses_vs_media30d
  TODO            las 3 ideas juntas

NOTA TECNICA: se intento primero ponderar las horas de El Nino en la funcion de perdida
(weight_col), pero neuralforecast 3.2.1 no expone ese parametro en NBEATSx ni en
NeuralForecast.fit -- se verifico por inspeccion de firma antes de gastar computo en eso.
La duplicacion de serie logra un efecto de sobremuestreo equivalente sin esa API.

Metrica clave: MAE especificamente en las horas de El Nino del holdout 2026, no solo el
MAE global -- que es lo que estamos tratando de bajar.

REVERSIBLE: vive en scripts_experimento/. Resultados en
data/processed/resultados/ataque_el_nino.csv
"""
import pandas as pd
import numpy as np
import time
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_SALIDA = RAIZ / "data" / "processed" / "resultados" / "ataque_el_nino.csv"

FECHA_TEST = pd.Timestamp("2026-01-01")
HORIZONTE, INPUT_SIZE, MAX_STEPS = 24, 168, 1000
SEMILLAS = [42, 7, 2024]
PESO_EL_NINO = 2.5

HIST_EXOG_BASE = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
FUTR_EXOG_CONT_BASE = ["oni"]
FUTR_EXOG_BIN_BASE = ["es_pandemia", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos"]


def cargar_todo():
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    df = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

    # Variables nuevas de El Nino (H2)
    cmd = pd.read_csv(RAIZ / "data/costo_marginal_despacho_2019_2026.csv", parse_dates=["fecha_hora"])
    cap = pd.read_csv(RAIZ / "data/compras_arranque_parada_2019_2026.csv", parse_dates=["fecha_hora"])
    df = df.merge(cmd, on="fecha_hora", how="left")
    df = pd.merge_asof(df.sort_values("fecha_hora"), cap.sort_values("fecha_hora"),
                        on="fecha_hora", direction="backward")
    df["costo_marginal_despacho"] = df["costo_marginal_despacho"].ffill()
    df["compras_arranque_parada"] = df["compras_arranque_parada"].ffill()
    df["costo_marginal_lag24"] = df["costo_marginal_despacho"].shift(24)
    df["compras_arranque_lag24"] = df["compras_arranque_parada"].shift(24)

    # Interaccion oni x hidrologia (H3)
    df["oni_x_aportes"] = df["oni"] * df["aportes_hidricos_media_7d"]
    df["oni_x_embalses"] = df["oni"] * df["volumen_embalses_vs_media30d"]

    return df


INICIO_EPISODIO_NINO = pd.Timestamp("2023-08-01")
FIN_EPISODIO_NINO = pd.Timestamp("2024-04-15 23:00:00")
N_COPIAS_NINO = 2  # cuantas veces se duplica el episodio (2 = el modelo lo ve 3x en total)


def correr(df, hist_exog_extra, futr_exog_extra, duplicar_nino, semilla):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx

    hist_exog = HIST_EXOG_BASE + hist_exog_extra
    futr_exog = FUTR_EXOG_CONT_BASE + FUTR_EXOG_BIN_BASE + futr_exog_extra
    cols_necesarias = hist_exog + futr_exog + ["precio_bolsa"]

    d = df.dropna(subset=cols_necesarias).reset_index(drop=True)
    df_nf = d[["fecha_hora", "precio_bolsa"] + hist_exog + futr_exog].copy()
    mascara_train = d["fecha_hora"] < FECHA_TEST

    for col in hist_exog + FUTR_EXOG_CONT_BASE + [c for c in futr_exog_extra if "oni_x" not in c]:
        mu = df_nf.loc[mascara_train.values, col].mean()
        sigma = df_nf.loc[mascara_train.values, col].std()
        if sigma == 0 or pd.isna(sigma):
            sigma = 1.0
        df_nf[col] = (df_nf[col] - mu) / sigma
    for col in [c for c in futr_exog_extra if "oni_x" in c]:
        sigma = df_nf.loc[mascara_train.values, col].std()
        df_nf[col] = df_nf[col] / (sigma if sigma and not pd.isna(sigma) else 1.0)

    df_nf["unique_id"] = "precio_bolsa"
    df_nf = df_nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})
    df_nf = df_nf[["unique_id", "ds", "y"] + hist_exog + futr_exog]

    if duplicar_nino:
        # El episodio ya esta normalizado con las mismas mu/sigma del train -- se duplica
        # como series independientes ("precio_bolsa_nino_copia1", "...copia2") que TERMINAN
        # antes de 2026, asi que cross_validation nunca genera pronosticos de prueba para
        # ellas (solo alimentan el entrenamiento). Las fechas se mantienen identicas: dos
        # series distintas pueden compartir "ds" sin problema, se distinguen por unique_id.
        episodio = df_nf[(df_nf["ds"] >= INICIO_EPISODIO_NINO) & (df_nf["ds"] <= FIN_EPISODIO_NINO)]
        copias = []
        for i in range(1, N_COPIAS_NINO + 1):
            copia = episodio.copy()
            copia["unique_id"] = f"precio_bolsa_nino_copia{i}"
            copias.append(copia)
        df_nf = pd.concat([df_nf] + copias, ignore_index=True)

    n_test = int((~mascara_train).sum())
    modelo = NBEATSx(h=HORIZONTE, input_size=INPUT_SIZE, hist_exog_list=hist_exog,
                      futr_exog_list=futr_exog, max_steps=MAX_STEPS, val_check_steps=100,
                      random_seed=semilla, enable_progress_bar=False)
    nf = NeuralForecast(models=[modelo], freq="h")
    cv = nf.cross_validation(df=df_nf, n_windows=n_test // HORIZONTE, step_size=HORIZONTE,
                              id_col="unique_id")
    cv = cv[cv["unique_id"] == "precio_bolsa"]  # descarta cualquier ventana de las copias, por seguridad

    oni_por_fecha = d.set_index("fecha_hora")["oni"]
    return cv[["ds", "y", "NBEATSx"]], oni_por_fecha


CONFIGURACIONES = [
    ("BASE", [], [], False),
    ("DUP_NINO", [], [], True),
    ("VARS_NINO", ["costo_marginal_lag24", "compras_arranque_lag24"], [], False),
    ("INTERACCION", [], ["oni_x_aportes", "oni_x_embalses"], False),
    ("TODO", ["costo_marginal_lag24", "compras_arranque_lag24"],
             ["oni_x_aportes", "oni_x_embalses"], True),
]


def main():
    df = cargar_todo()
    print(f"Datos cargados: {len(df):,} filas\n")

    filas = []
    t0 = time.time()

    for nombre, hist_extra, futr_extra, duplicar in CONFIGURACIONES:
        for semilla in SEMILLAS:
            t = time.time()
            try:
                cv, oni_ref = correr(df, hist_extra, futr_extra, duplicar, semilla)
            except Exception as e:
                print(f"  {nombre:14s} semilla {semilla:5d}  FALLO: {e}")
                sys.stdout.flush()
                continue

            oni_test = oni_ref.reindex(cv["ds"]).values
            es_nino = oni_test > 1.0
            err = np.abs(cv["y"].values - cv["NBEATSx"].values)

            fila = {
                "config": nombre, "semilla": semilla,
                "mae": err.mean(), "mape": (err / cv["y"].values).mean() * 100,
                "mae_el_nino": err[es_nino].mean() if es_nino.sum() else np.nan,
                "mae_resto": err[~es_nino].mean() if (~es_nino).sum() else np.nan,
                "n_el_nino": int(es_nino.sum()),
                "segundos": time.time() - t,
            }
            filas.append(fila)
            print(f"  {nombre:14s} semilla {semilla:5d}  MAE {fila['mae']:6.2f}  "
                  f"| El Nino {fila['mae_el_nino']:6.2f}  resto {fila['mae_resto']:6.2f}  "
                  f"({fila['segundos']:.0f}s)")
            sys.stdout.flush()
            pd.DataFrame(filas).to_csv(RUTA_SALIDA, index=False)

    res = pd.DataFrame(filas)
    print(f"\n{'='*74}\nRESUMEN (media +- std sobre semillas)\n{'='*74}")
    resumen = res.groupby("config")[["mae", "mae_el_nino", "mae_resto"]].agg(["mean", "std"])
    print(resumen.round(3).to_string())

    base_nino = res[res["config"] == "BASE"]["mae_el_nino"]
    print(f"\nBASE en El Nino: {base_nino.mean():.2f} +- {base_nino.std():.2f}")
    for cfg in res["config"].unique():
        if cfg == "BASE":
            continue
        sub = res[res["config"] == cfg]["mae_el_nino"]
        print(f"{cfg:14s} en El Nino: {sub.mean():.2f} +- {sub.std():.2f}  "
              f"(diferencia: {sub.mean()-base_nino.mean():+.2f})")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
