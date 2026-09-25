# -*- coding: utf-8 -*-
"""
Modelo de RAMPAS: pronosticar la transicion hora a hora en vez del nivel.

IDEA (del usuario, respaldada por la literatura de eventos de rampa en energia eolica, p.ej.
"Ramp Event Directional Forecasting ... A Regime-Stratified Ensemble Framework with
Direction-Focused Training", Energies 2026). Cuando una serie tiene puntos de cambio marcados,
modelar la TRANSICION puede funcionar mejor que modelar el NIVEL, porque el error se concentra
justo en los quiebres.

DIAGNOSTICO QUE LO MOTIVA (medido sobre el ensamble actual, 2026):

    tipo de hora            % del tiempo    MAE del ensamble
    precio plano (cambio 0)     50.8 %           28.5
    cambio chico (<10)          18.6 %           28.4
    cambio medio (10-50)        16.8 %           40.5
    RAMPA FUERTE (>=50)         13.9 %          105.5     <- 3.7x peor

Y las 6 horas de mayor error (0, 8, 9, 10, 18, 19) tienen rampa media de 46.6 contra 24.7 de las
otras 18. El error vive en los quiebres, no en las mesetas.

ESTRUCTURA INFLADA EN CERO. Mas de la mitad de las horas el precio no cambia nada (la misma unidad
marca el precio varias horas seguidas). Por eso se prueban dos variantes del modelo de rampa:
  (a) directa:  un LASSO por paso sobre Delta = p(t_h) - p(t_{h-1})
  (b) hurdle:   clasificador de "¿se mueve el precio?" x regresor de "¿cuanto?", que respeta la
                masa en cero -- es el mismo esquema de dos partes que el proyecto ya uso sobre el
                NIVEL (hurdle_dos_etapas.py), aplicado aqui sobre la TRANSICION, que es nuevo.

RECONSTRUCCION. El nivel se reconstruye anclando en el precio del corte, que si se conoce:
    p_hat(h) = p(corte) + suma acumulada de Delta_hat hasta h

USO. No se propone reemplazar el ensamble, sino ofrecerlo como VOTANTE ADICIONAL y dejar que el
combinador decida -- la leccion repetida del proyecto es que lo que hace ganar al ensamble es la
diversidad entre familias, no el acierto individual. Un modelo de transiciones es una familia que
el ensamble no tiene.

Verificacion: 10 particiones + Diebold-Mariano, el estandar del proyecto.
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
from stacking_24h_v2 import cargar as cargar24, BASE5
from combinador_optimo_mape import qra_cv, mape, smape

CORTE = pd.Timestamp("2026-01-01")
SEMILLAS = list(range(10))


def dm(real, p1, p2, metrica="mae", maxlags=24):
    real = np.asarray(real, float)
    if metrica == "mape":
        l1, l2 = np.abs(real - p1) / real, np.abs(real - p2) / real
    else:
        l1, l2 = np.abs(real - p1), np.abs(real - p2)
    d = l1 - l2
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def construir_rampas():
    """Para cada paso h, arma X (estado en el corte) y el objetivo Delta = p(t_h) - p(t_{h-1})."""
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier

    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    precio = df["precio_bolsa"].to_numpy(float)
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - 24)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]

    salida_directa, salida_hurdle = [], []
    acum_tr = None   # nivel acumulado no hace falta; se reconstruye al final
    alphas = []

    for h in range(1, 25):
        Xtr, ytr, _, ctr = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_obj, cte = construir_para_paso(df, h, cortes_te)
        # objetivo: cambio respecto a la hora anterior del propio horizonte
        pos = pd.Series(df.index.values, index=fechas)
        i_obj_tr = pos.reindex(pd.DatetimeIndex(f_obj)).to_numpy() if False else None
        # p(t_{h-1}) = precio en la hora anterior al objetivo
        prev_tr = precio[pos.reindex(pd.DatetimeIndex(ctr)).to_numpy() + (h - 1)]
        prev_te = precio[pos.reindex(pd.DatetimeIndex(cte)).to_numpy() + (h - 1)]
        dtr, dte = ytr - prev_tr, yte - prev_te

        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

        # (a) directa: LASSO sobre el Delta
        if h <= 4 or h % 8 == 0:
            alphas.append(LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=6000).fit(Ztr, dtr).alpha_)
        al = float(np.median(alphas))
        lm = Lasso(alpha=al, max_iter=6000, random_state=42).fit(Ztr, dtr)
        d_directa = lm.predict(Zte)

        # (b) hurdle: P(se mueve) x magnitud si se mueve
        mueve_tr = (np.abs(dtr) > 1e-9).astype(int)
        if mueve_tr.min() == mueve_tr.max():
            d_hurdle = d_directa
        else:
            clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42).fit(Ztr, mueve_tr)
            p_mueve = clf.predict_proba(Zte)[:, 1]
            m = mueve_tr == 1
            lm2 = Lasso(alpha=al, max_iter=6000, random_state=42).fit(Ztr[m], dtr[m])
            d_hurdle = p_mueve * lm2.predict(Zte)

        salida_directa.append(pd.DataFrame({"fecha_hora": f_obj, "cutoff": cte, "paso": h,
                                            "delta": d_directa, "prev_real": prev_te, "real": yte}))
        salida_hurdle.append(pd.DataFrame({"fecha_hora": f_obj, "cutoff": cte, "paso": h,
                                           "delta": d_hurdle}))
        if h % 8 == 0:
            print(f"    paso {h}/24 listo", flush=True)

    A = pd.concat(salida_directa, ignore_index=True)
    B = pd.concat(salida_hurdle, ignore_index=True)
    A = A.merge(B[["fecha_hora", "cutoff", "paso", "delta"]], on=["fecha_hora", "cutoff", "paso"],
                suffixes=("_dir", "_hur"))
    # reconstruccion: nivel = precio del corte + suma acumulada de deltas dentro del dia
    precio_corte = pd.Series(precio, index=fechas).reindex(pd.DatetimeIndex(A["cutoff"])).to_numpy()
    A["precio_corte"] = precio_corte
    A = A.sort_values(["cutoff", "paso"])
    for col in ["dir", "hur"]:
        A[f"nivel_{col}"] = A["precio_corte"] + A.groupby("cutoff")[f"delta_{col}"].cumsum()
    return A


def main():
    t0 = time.time()
    print("Construyendo modelo de rampas (24 pasos, dos variantes)...")
    A = construir_rampas()
    A.to_csv(RES / "modelo_rampas_24h.csv", index=False)
    y = A["real"].to_numpy()
    print(f"\nFilas: {len(A):,}")
    print(f"{'='*90}\nEL MODELO DE RAMPAS COMO PRONOSTICO SUELTO\n{'='*90}")
    for col, nom in [("nivel_dir", "rampa directa (LASSO sobre Delta)"),
                     ("nivel_hur", "rampa hurdle (mueve? x cuanto)")]:
        p = A[col].to_numpy()
        print(f"  {nom:36s} MAE={np.abs(y-p).mean():7.2f}  MAPE={mape(y,p):6.2f}%")
    # referencia: persistencia del corte
    print(f"  {'(referencia) precio del corte fijo':36s} "
          f"MAE={np.abs(y-A['precio_corte']).mean():7.2f}")

    # ---- como votante adicional del ensamble ----
    print(f"\n{'='*90}\nCOMO VOTANTE ADICIONAL DEL ENSAMBLE DE 24h\n{'='*90}")
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    c = "pred" if "pred" in s.columns else s.columns[-1]
    d = d.join(s[c].rename("GARCH-ged"), how="left")
    for col, nom in [("nivel_dir", "Rampa"), ("nivel_hur", "RampaH")]:
        r = A.set_index("fecha_hora")[col].rename(nom)
        d = d.join(r[~r.index.duplicated()], how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["Rampa", "RampaH", "real"])
    yy = d["real"].to_numpy()
    print(f"Filas alineadas: {len(d):,}")

    combos = {
        "ensamble actual (6 votantes)": MEJOR,
        "+ Rampa directa": MEJOR + ["Rampa"],
        "+ Rampa hurdle": MEJOR + ["RampaH"],
        "+ ambas": MEJOR + ["Rampa", "RampaH"],
    }
    base = None
    print(f"{'configuracion':32s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s}  {'p(MAE)':>8s} {'p(MAPE)':>8s}")
    preds = {}
    for nom, cols in combos.items():
        p = qra_cv(d, cols, "smape")
        preds[nom] = p
        if base is None:
            base = p
        linea = f"{nom:32s} {np.abs(yy-p).mean():8.2f} {mape(yy,p):8.2f} {smape(yy,p):8.2f}"
        if nom != "ensamble actual (6 votantes)":
            linea += f"  {dm(yy,base,p,'mae')[1]:8.4f} {dm(yy,base,p,'mape')[1]:8.4f}"
        print(linea)

    # verificacion multisemilla del mejor candidato
    cand = [k for k in combos if k != "ensamble actual (6 votantes)"]
    ganador = min(cand, key=lambda k: mape(yy, preds[k]))
    print(f"\n{'='*90}\nVERIFICACION 10 PARTICIONES: '{ganador}'\n{'='*90}")
    A_, B_ = [], []
    for s_ in SEMILLAS:
        pa = qra_cv(d, MEJOR, "smape", s_)
        pb = qra_cv(d, combos[ganador], "smape", s_)
        A_.append([np.abs(yy-pa).mean(), mape(yy, pa), smape(yy, pa)])
        B_.append([np.abs(yy-pb).mean(), mape(yy, pb), smape(yy, pb)])
    A_, B_ = np.array(A_), np.array(B_)
    for j, nm in enumerate(["MAE", "MAPE %", "sMAPE %"]):
        dl = B_[:, j] - A_[:, j]
        print(f"  {nm:8s} actual={A_[:,j].mean():7.3f}+/-{A_[:,j].std():.3f}  "
              f"con rampa={B_[:,j].mean():7.3f}+/-{B_[:,j].std():.3f}  "
              f"delta={dl.mean():+7.3f}, mejor en {int((dl<0).sum())}/10")
    print(f"  VEREDICTO: {'SOLIDO' if all((B_[:,j]-A_[:,j]<0).all() for j in range(3)) else 'no concluyente'}")

    # ¿mejora especificamente en las horas de rampa fuerte?
    print(f"\n{'='*90}\nEFECTO EN LAS HORAS DE RAMPA FUERTE (donde vive el error)\n{'='*90}")
    prev = cargar_completo().set_index("fecha_hora")["precio_bolsa"].reindex(d.index - pd.Timedelta(hours=1))
    prev.index = d.index
    rampa = (d["real"] - prev).abs()
    fuerte = (rampa >= 50).to_numpy()
    pb = preds[ganador]
    print(f"  rampa fuerte (n={int(fuerte.sum())}): actual={np.abs(yy[fuerte]-base[fuerte]).mean():7.2f}  "
          f"con rampa={np.abs(yy[fuerte]-pb[fuerte]).mean():7.2f}  "
          f"DM p={dm(yy[fuerte], base[fuerte], pb[fuerte], 'mae', 12)[1]:.4f}")
    plano = (rampa == 0).to_numpy()
    print(f"  precio plano (n={int(plano.sum())}): actual={np.abs(yy[plano]-base[plano]).mean():7.2f}  "
          f"con rampa={np.abs(yy[plano]-pb[plano]).mean():7.2f}")
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
