# -*- coding: utf-8 -*-
"""
Hueco detectado en los datos: la DEMANDA de la hora objetivo nunca se usa como variable.

Hoy el modelo de 24h conoce la demanda EN EL CORTE (00:00) y su media de 24h, pero no tiene ninguna
estimacion de cuanta demanda habra a las 19:00, que es justamente la hora que peor predice. En la
literatura day-ahead (Lago et al. 2021) el pronostico de carga es LA variable exogena estandar, y
la demanda es mucho mas predecible que el precio (ciclo diario y semanal muy marcado).

Diseno del experimento, en dos etapas para no gastar esfuerzo a ciegas:

  ETAPA 1 - TECHO CON ORACULO. Se agrega la demanda REAL de la hora objetivo como variable. Eso es
  fuga de informacion y NO es desplegable: sirve solo para medir el techo. Si ni con la demanda
  perfecta mejora, un pronostico de demanda tampoco va a mejorar y la linea se cierra aqui.

  ETAPA 2 - VERSION HONESTA. Solo si la etapa 1 promete: se entrena un pronosticador de demanda con
  la misma formulacion directa y sin fuga (estado en el corte -> demanda en t), y se usan SUS
  predicciones como variable del modelo de precio.

Se mide el efecto global y, por separado, en las horas pico 18-20, que es donde vive el error.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES
from lear_24h import construir_para_paso, REZAGOS_CORTE

CORTE = pd.Timestamp("2026-01-01")
H, PICO = 24, [18, 19, 20]


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def construir_demanda(df, h, idx_cortes):
    """Mismo diseno que para el precio, pero el objetivo es la DEMANDA en t (no el precio)."""
    n = len(df)
    dem = df["demanda"].to_numpy(np.float64)
    precio = df["precio_bolsa"].to_numpy(np.float64)
    fechas = df["fecha_hora"].to_numpy()
    c = idx_cortes[(idx_cortes + h < n) & (idx_cortes - 168 >= 0)]
    t = c + h
    col, nom = [], []
    for k in [0, 1, 2, 3, 6, 12, 24, 48, 72, 168]:
        col.append(dem[c - k]); nom.append(f"dem_corte_lag{k}")
    for m in [24, 48, 72, 168]:                      # demanda a la misma hora de dias previos
        col.append(dem[t - m]); nom.append(f"dem_obj_lag{m}")
    col.append(precio[c]); nom.append("precio_corte")
    for cc in ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
               "dia_anio_sin", "dia_anio_cos", "es_festivo", "es_pandemia"]:
        col.append(df[cc].to_numpy(np.float64)[t]); nom.append(f"{cc}_obj")
    X = np.column_stack(col)
    ok = ~np.isnan(X).any(1)
    return pd.DataFrame(X[ok], columns=nom), dem[t][ok], fechas[t][ok]


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV, Lasso, Ridge
    from sklearn.preprocessing import StandardScaler

    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - H)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]
    dem_all = df["demanda"].to_numpy(np.float64)
    gen_all = df["generacion"].to_numpy(np.float64)

    # ---------------- ETAPA 2 (se entrena igual, se usa despues): pronosticador de demanda ----------
    print(f"{'='*94}\nPRONOSTICADOR DE DEMANDA (formulacion directa, sin fuga)\n{'='*94}")
    pred_dem = {}
    for h in range(1, H + 1):
        Xtr, ytr, _ = construir_demanda(df, h, cortes_tr)
        Xte, yte, f_obj = construir_demanda(df, h, cortes_te)
        esc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=1.0).fit(esc.transform(Xtr), ytr)
        p = m.predict(esc.transform(Xte))
        pred_dem[h] = pd.DataFrame({"fecha_hora": f_obj, "dem_pron": p, "dem_real": yte})
        if h in (1, 12, 19, 24):
            print(f"  h={h:2d}: MAPE de la demanda = {mape(yte, p):.2f}%")
    pdem = pd.concat(pred_dem.values(), ignore_index=True)
    print(f"  MAPE global del pronostico de demanda: {mape(pdem['dem_real'], pdem['dem_pron']):.2f}% "
          f"(muy por debajo del ~11% del precio: la demanda si es predecible)")

    # ---------------- modelo de precio, tres variantes ----------------
    print(f"\n{'='*94}\nMODELO DE PRECIO CON/SIN DEMANDA DE LA HORA OBJETIVO\n{'='*94}")
    partes = []
    alphas = []
    for h in range(1, H + 1):
        Xtr, ytr, f_tr, c_tr = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_te, c_te = construir_para_paso(df, h, cortes_te)
        # indices de las filas que sobrevivieron, para alinear demanda/generacion del objetivo
        ctr = cortes_tr[(cortes_tr + h < len(df)) & (cortes_tr - max(REZAGOS_CORTE) >= 0)]
        cte = cortes_te[(cortes_te + h < len(df)) & (cortes_te - max(REZAGOS_CORTE) >= 0)]
        # reconstruir mascara de validos igual que construir_para_paso
        m_tr = pd.Series(f_tr).isin(pd.Series(fechas.to_numpy()[ctr + h])).to_numpy()
        ttr, tte = ctr + h, cte + h
        ttr = ttr[: len(Xtr)] if len(ttr) >= len(Xtr) else ttr
        # alinear por fecha (robusto): mapa fecha -> indice
        pos = pd.Series(np.arange(len(df)), index=fechas)
        itr = pos.reindex(pd.DatetimeIndex(f_tr)).to_numpy()
        ite = pos.reindex(pd.DatetimeIndex(f_te)).to_numpy()

        variantes = {
            "base (sin demanda objetivo)": (Xtr.copy(), Xte.copy()),
            "+ demanda REAL objetivo (oraculo)": (
                Xtr.assign(dem_obj=dem_all[itr], gen_obj=gen_all[itr]),
                Xte.assign(dem_obj=dem_all[ite], gen_obj=gen_all[ite])),
        }
        # version honesta: pronostico de demanda
        dmap = pdem.set_index("fecha_hora")["dem_pron"]
        # para entrenamiento se usa la demanda real (es lo que se hace en la practica: se entrena
        # con la realizada y se predice con el pronostico), variante estandar en la literatura
        variantes["+ demanda PRONOSTICADA (desplegable)"] = (
            Xtr.assign(dem_obj=dem_all[itr]),
            Xte.assign(dem_obj=dmap.reindex(pd.DatetimeIndex(f_te)).to_numpy()))

        fila = {"fecha_hora": f_te, "paso": h, "real": yte}
        for etiqueta, (A, B) in variantes.items():
            esc = StandardScaler().fit(A)
            if h <= 4:
                mm = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(esc.transform(A), ytr)
                alphas.append(mm.alpha_)
            else:
                mm = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(
                    esc.transform(A), ytr)
            fila[etiqueta] = mm.predict(esc.transform(B))
        partes.append(pd.DataFrame(fila))
    r = pd.concat(partes, ignore_index=True)
    r = r[r["fecha_hora"] >= CORTE].dropna()
    r["hora"] = pd.DatetimeIndex(r["fecha_hora"]).hour

    cols = [c for c in r.columns if c not in ("fecha_hora", "paso", "real", "hora")]
    print(f"{'variante':40s} {'MAE':>8s} {'MAPE %':>8s} {'MAE pico':>10s} {'MAPE pico':>10s}")
    base = r[cols[0]].to_numpy()
    pk = r["hora"].isin(PICO).to_numpy()
    for c in cols:
        p = r[c].to_numpy()
        y = r["real"].to_numpy()
        linea = (f"{c:40s} {np.abs(y-p).mean():8.2f} {mape(y,p):8.2f} "
                 f"{np.abs(y[pk]-p[pk]).mean():10.2f} {mape(y[pk],p[pk]):10.2f}")
        if c != cols[0]:
            t, pv = dm(y, base, p)
            marca = ("MEJORA" if t > 0 else "empeora") if pv < 0.05 else "n.s."
            linea += f"   DM p={pv:.4f} {marca}"
        print(linea)

    r.to_csv(RES / "demanda_objetivo_24h.csv", index=False)
    print(f"\nGuardado: demanda_objetivo_24h.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
