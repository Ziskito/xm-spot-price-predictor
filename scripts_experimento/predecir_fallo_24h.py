# -*- coding: utf-8 -*-
"""
Se puede PREDECIR, en el momento del corte, cuando va a fallar el ensamble de 24h?

Por que esta es la pregunta correcta. El usuario pidio "un modelo que se use especificamente en
esos momentos donde falla el nuestro". Ya se probo la version ingenua -- especialistas entrenados
para las horas 18-20 -- y salieron PEORES que el ensamble general. Pero esa version identifica el
momento por la hora del reloj, que es un proxy grueso: no todas las tardes fallan, y tambien hay
fallos fuera del pico.

Un enrutador de verdad necesita algo mas fuerte: saber EN EL CORTE que la prediccion de dentro de
N horas va a salir mal. Si eso es predecible, se puede enrutar a un especialista (o al menos
avisar al operador). Si no lo es, entonces ningun esquema de especializacion es desplegable, y eso
cierra formalmente la linea con evidencia en vez de con intuicion.

Se mide:
  A) REGRESION del error absoluto del ensamble a partir del estado en el corte.
  B) CLASIFICACION del 20% peor de horas (AUC). Un AUC cercano a 0.5 significa que el fallo es
     indistinguible de antemano.
  C) Valor practico: si se enrutara el X% de horas con mayor riesgo predicho a un modelo alterno,
     cuanto bajaria el MAPE global? Se compara contra el enrutamiento por hora del reloj (18-20) y
     contra el enrutamiento ORACULO (el que conoce el error de verdad), que es la cota superior.
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

PICO = [18, 19, 20]
ESTADO = ["precio_bolsa", "precio_media_24h", "precio_media_7d", "precio_media_30d",
          "precio_std_24h", "precio_std_7d", "precio_rango_24h", "ratio_volatilidad",
          "volumen_embalses", "volumen_embalses_vs_media30d", "volumen_embalses_delta_1d",
          "aportes_hidricos", "aportes_hidricos_vs_media30d", "demanda", "demanda_media_24h",
          "generacion", "oni"]


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def main():
    t0 = time.time()
    from catboost import CatBoostRegressor, CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    v4 = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"]).sort_values("fecha_hora")
    d = v4.dropna(subset=["real", "pred_ensamble"]).copy().set_index("fecha_hora")
    d["err"] = (d["real"] - d["pred_ensamble"]).abs()
    d["ape"] = d["err"] / d["real"] * 100
    d["hora"] = d.index.hour
    d["dia"] = d.index.normalize()

    # estado EN EL CORTE (00:00 del dia anterior al objetivo), nunca del instante objetivo
    full = cargar_completo().set_index("fecha_hora").sort_index()
    corte = np.where(d["hora"] == 0, d["dia"] - pd.Timedelta(days=1), d["dia"])
    est = full[ESTADO].reindex(pd.DatetimeIndex(corte))
    est.index = d.index
    d = pd.concat([d, est.add_suffix("_c")], axis=1).dropna()
    cols = [f"{c}_c" for c in ESTADO] + ["hora"]
    dias = np.sort(d["dia"].unique())
    print(f"Filas: {len(d):,} | variables de estado en el corte: {len(cols)}")
    print(f"MAPE del ensamble: {d['ape'].mean():.2f}%\n")

    # ---------- A) regresion del error ----------
    X = d[cols].to_numpy()
    y_err = d["err"].to_numpy()
    pred_err = np.full(len(d), np.nan)
    dia = d["dia"].to_numpy()
    for dt in np.array_split(np.random.default_rng(42).permutation(dias), 5):
        te = np.isin(dia, dt)
        m = CatBoostRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, loss_function="MAE",
                              random_seed=42, verbose=0, thread_count=-1).fit(X[~te], y_err[~te])
        pred_err[te] = m.predict(X[te])
    corr = np.corrcoef(pred_err, y_err)[0, 1]
    print(f"{'='*88}\nA) REGRESION DEL ERROR DEL ENSAMBLE (desde el estado en el corte)\n{'='*88}")
    print(f"  correlacion entre error predicho y error real: {corr:.3f}")
    print(f"  (0 = el fallo es impredecible; 1 = perfectamente anticipable)")

    # ---------- B) clasificacion del 20% peor ----------
    umbral = np.quantile(y_err, 0.80)
    y_bin = (y_err > umbral).astype(int)
    p_bin = np.full(len(d), np.nan)
    for dt in np.array_split(np.random.default_rng(42).permutation(dias), 5):
        te = np.isin(dia, dt)
        m = CatBoostClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                               random_seed=42, verbose=0, thread_count=-1).fit(X[~te], y_bin[~te])
        p_bin[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y_bin, p_bin)
    print(f"\n{'='*88}\nB) CLASIFICACION DEL 20% PEOR DE HORAS\n{'='*88}")
    print(f"  AUC = {auc:.3f}   (0.5 = azar puro; >0.7 seria util para enrutar)")
    # que tan bien separa: precision en el top 20% predicho
    top = np.argsort(-p_bin)[: int(0.2 * len(d))]
    print(f"  De las {len(top)} horas senaladas como mas riesgosas, {y_bin[top].mean()*100:.1f}% "
          f"eran de verdad del 20% peor (azar daria 20%)")
    # comparacion: la hora del reloj sola
    auc_hora = roc_auc_score(y_bin, d["hora"].isin(PICO).astype(int))
    print(f"  Para referencia, usar solo 'es hora pico 18-20': AUC = {auc_hora:.3f}")

    # ---------- C) valor practico del enrutamiento ----------
    print(f"\n{'='*88}\nC) SI SE ENRUTARA EL X% MAS RIESGOSO A UN MODELO PERFECTO\n{'='*88}")
    print("  (cota superior: supone que el especialista acierta EXACTO en las horas enrutadas)")
    ape = d["ape"].to_numpy()
    for pct in [5, 10, 20]:
        k = int(len(d) * pct / 100)
        # enrutamiento por el clasificador
        sel_clf = np.argsort(-p_bin)[:k]
        a1 = ape.copy(); a1[sel_clf] = 0
        # enrutamiento oraculo
        sel_or = np.argsort(-ape)[:k]
        a2 = ape.copy(); a2[sel_or] = 0
        # enrutamiento por hora del reloj
        pk = d["hora"].isin(PICO).to_numpy()
        idx_pk = np.where(pk)[0][:k]
        a3 = ape.copy(); a3[idx_pk] = 0
        print(f"  {pct:2d}% -> clasificador: {a1.mean():5.2f}%  |  por hora 18-20: {a3.mean():5.2f}%  "
              f"|  ORACULO: {a2.mean():5.2f}%")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
