# -*- coding: utf-8 -*-
"""
Ataque dirigido al pico vespertino, que es donde vive el error del ensamble de 24h.

Diagnostico previo (diagnostico_error_24h.py): las horas 18, 19 y 20 tienen MAPE 15.7%, 16.5% y
14.3% frente al 11.27% global, y las 6 horas peores concentran el 37.8% del error total. Para
llegar al objetivo de 8% hace falta bajar el error un 29%: ni siquiera acertando perfecto esas 6
horas se llega (daria 7.01%), asi que el pico es condicion necesaria.

La pregunta previa a cualquier modelo: ese error es SESGO (el ensamble se equivoca de forma
sistematica y se puede corregir) o VARIANZA (los picos son impredecibles y no hay nada que hacer)?
Se mide primero eso, y solo despues se prueban correcciones:

  A) Descomposicion sesgo/varianza por hora: error medio con signo vs desviacion del error.
  B) Correccion de sesgo por hora, ESTIMADA DE FORMA CAUSAL (media movil de los errores de los
     ultimos N dias, solo con dias anteriores) -- si el sesgo es persistente, esto lo captura.
  C) Especialista entrenado SOLO con horas pico, para ver si un modelo dedicado a ese regimen
     mejora sobre el ensamble general.
  D) Cuanto de la mejora sobrevive al medirla sobre el MAPE global.
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
from lear_24h import construir_para_paso

CORTE = pd.Timestamp("2026-01-01")
PICO = [18, 19, 20]


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def mape(real, pred):
    return float(np.mean(np.abs(np.asarray(real, float) - np.asarray(pred, float)) /
                         np.asarray(real, float)) * 100)


def main():
    t0 = time.time()
    v4 = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"]).sort_values("fecha_hora")
    d = v4.dropna(subset=["real", "pred_ensamble"]).copy().reset_index(drop=True)
    d["hora"] = d["fecha_hora"].dt.hour
    d["dia"] = d["fecha_hora"].dt.normalize()
    d["e_signo"] = d["real"] - d["pred_ensamble"]      # >0 = el modelo SUBESTIMA
    base_mape = mape(d["real"], d["pred_ensamble"])
    print(f"Linea base: MAPE global {base_mape:.2f}%\n")

    # ---------- A) sesgo vs varianza por hora ----------
    print(f"{'='*94}\nA) SESGO vs VARIANZA POR HORA (e = real - pronostico; e>0 = subestima)\n{'='*94}")
    g = d.groupby("hora")["e_signo"].agg(sesgo="mean", desv="std")
    g["|sesgo|/desv"] = (g["sesgo"].abs() / g["desv"]).round(3)
    g["MAE"] = d.groupby("hora")["e_signo"].apply(lambda s: s.abs().mean())
    print(g.round(2).to_string())
    pico_g = g.loc[PICO]
    print(f"\nEn el pico ({PICO}): sesgo medio {pico_g['sesgo'].mean():+.1f} COP/kWh, "
          f"desviacion {pico_g['desv'].mean():.1f}")
    print(f"El sesgo explica solo {(pico_g['sesgo'].abs().mean()/pico_g['MAE'].mean()*100):.1f}% "
          f"del MAE del pico -- el resto es varianza.")

    # ---------- B) correccion de sesgo causal por hora ----------
    print(f"\n{'='*94}\nB) CORRECCION DE SESGO POR HORA, CAUSAL (media de los ultimos N dias)\n{'='*94}")
    piv = d.pivot_table(index="dia", columns="hora", values="e_signo")
    mejor = None
    for N in [7, 14, 30, 60]:
        corr = piv.rolling(N, min_periods=max(3, N // 3)).mean().shift(1)   # shift: solo dias previos
        c = corr.stack().rename("ajuste").reset_index()
        c.columns = ["dia", "hora", "ajuste"]
        dd = d.merge(c, on=["dia", "hora"], how="left")
        dd["ajuste"] = dd["ajuste"].fillna(0.0)
        nuevo = dd["pred_ensamble"] + dd["ajuste"]
        m = mape(dd["real"], nuevo)
        t, pv = dm(dd["real"].to_numpy(), dd["pred_ensamble"].to_numpy(), nuevo.to_numpy())
        marca = ("MEJORA" if t > 0 else "empeora") if pv < 0.05 else "n.s."
        print(f"  ventana {N:2d} dias: MAPE {m:5.2f}% ({m-base_mape:+.2f}) p={pv:.4f} {marca}")
        if mejor is None or m < mejor[1]:
            mejor = (N, m, nuevo.to_numpy(), dd)
    print(f"  mejor ventana: {mejor[0]} dias -> MAPE {mejor[1]:.2f}%")

    # ---------- C) especialista entrenado solo con horas pico ----------
    print(f"\n{'='*94}\nC) ESPECIALISTA ENTRENADO SOLO CON HORAS PICO (LASSO y CatBoost)\n{'='*94}")
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostRegressor

    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - 24)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]

    pred_pico = {}
    for h in PICO:                       # paso h == hora del dia h (corte a las 00:00)
        Xtr, ytr, _, _ = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_obj, _ = construir_para_paso(df, h, cortes_te)
        esc = StandardScaler().fit(Xtr)
        Ztr, Zte = esc.transform(Xtr), esc.transform(Xte)
        ml = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr)
        mc = CatBoostRegressor(n_estimators=1500, max_depth=8, learning_rate=0.05,
                               loss_function="MAE", random_seed=42, verbose=0, thread_count=-1).fit(Xtr, ytr)
        pred_pico[h] = pd.DataFrame({"fecha_hora": f_obj, "esp_lasso": ml.predict(Zte),
                                     "esp_catb": mc.predict(Xte)})
        print(f"  hora {h}: LASSO MAE={np.abs(yte-ml.predict(Zte)).mean():6.2f} | "
              f"CatBoost MAE={np.abs(yte-mc.predict(Xte)).mean():6.2f}", flush=True)
    esp = pd.concat(pred_pico.values(), ignore_index=True)

    dd = d.merge(esp, on="fecha_hora", how="left")
    msk = dd["hora"].isin(PICO) & dd["esp_lasso"].notna()
    print(f"\n  horas pico con especialista: {int(msk.sum())}")
    print(f"  MAE en el pico -- ensamble v4: {np.abs(dd.loc[msk,'real']-dd.loc[msk,'pred_ensamble']).mean():.2f}")
    for c in ["esp_lasso", "esp_catb"]:
        print(f"                   {c:10s}: {np.abs(dd.loc[msk,'real']-dd.loc[msk,c]).mean():.2f}")

    # mezcla 50/50 y sustitucion, medidas sobre el MAPE GLOBAL
    print(f"\n{'='*94}\nD) EFECTO SOBRE EL MAPE GLOBAL\n{'='*94}")
    variantes = {}
    for c in ["esp_lasso", "esp_catb"]:
        p = dd["pred_ensamble"].to_numpy().copy()
        p[msk.to_numpy()] = dd.loc[msk, c].to_numpy()
        variantes[f"sustituir pico por {c}"] = p
        p2 = dd["pred_ensamble"].to_numpy().copy()
        p2[msk.to_numpy()] = 0.5 * p2[msk.to_numpy()] + 0.5 * dd.loc[msk, c].to_numpy()
        variantes[f"mezcla 50/50 con {c}"] = p2
    variantes[f"correccion de sesgo ({mejor[0]}d)"] = mejor[2]
    for etiqueta, p in variantes.items():
        m = mape(dd["real"], p)
        t, pv = dm(dd["real"].to_numpy(), dd["pred_ensamble"].to_numpy(), p)
        marca = ("MEJORA" if t > 0 else "empeora") if pv < 0.05 else "n.s."
        print(f"  {etiqueta:34s} MAPE {m:5.2f}% ({m-base_mape:+.2f})  p={pv:.4f}  {marca}")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
