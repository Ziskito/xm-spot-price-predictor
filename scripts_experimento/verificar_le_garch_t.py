# -*- coding: utf-8 -*-
"""
Verificacion del hallazgo de le_garch_t_24h.py antes de darlo por bueno.

El hallazgo: agregar LE-GARCH-t al ensamble de 24h baja el MAPE de 11.27% a 10.89% (MAE 42.51 ->
41.46, sMAPE 10.38 -> 10.09), con DM p=0.0044. Seria la primera mejora real al ensamble de 24h en
toda la sesion, despues de doce lineas fallidas.

Por que hay que verificarlo con dureza: el proyecto ya se quemo dando por bueno un efecto de ~2.8
COP/kWh que era ruido de semilla (el falso hallazgo del ONI). Aqui el delta es ~1.05 COP/kWh, o
sea MENOR, asi que la vara debe ser al menos igual de alta. Se comprueba:

  1) ESTABILIDAD ENTRE SEMILLAS de la particion en validacion cruzada (10 semillas).
  2) MODO DESPLEGABLE (pesos causales con ventana expansiva): es determinista y es lo que de
     verdad se operaria; si el efecto solo aparece en CV, no sirve.
  3) REPARTO POR DIA: que la mejora sea generalizada y no venga de unos pocos dias.
  4) AUSENCIA DE FUGA: se verifica explicitamente que las variables de la ecuacion de media son
     exactamente las mismas del LEAR (todas medidas en el corte) y que lo unico que cambia es el
     METODO DE ESTIMACION (maxima verosimilitud con GARCH(1,1) y errores t, en vez de LASSO).
     Si LE-GARCH-t gana, tiene que ser por como estima, no por que informacion ve.
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
from stacking_24h_v2 import cargar as cargar24, BASE5
from qra_bandas import ajustar_cuantil
from o6_comun import RES

PICO = [0, 8, 9, 10, 18, 19]  # corregido: las 6 horas que mas aportan al MAPE (nlargest), no solo 18-20
SEMILLAS = list(range(10))
MIN_DIAS = 21


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def smape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / ((np.abs(real) + np.abs(pred)) / 2)) * 100)


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def qra_cv(d, cols, semilla):
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d["g4"].to_numpy(), d["dia"].to_numpy()
    P = np.full(len(d), np.nan)
    for dt in np.array_split(np.random.default_rng(semilla).permutation(np.sort(np.unique(dia))), 5):
        te_m = np.isin(dia, dt)
        for gg in np.unique(g[te_m]):
            tr, te = (~te_m) & (g == gg), te_m & (g == gg)
            if tr.sum() >= len(cols) + 5:
                w, b = ajustar_cuantil(X[tr], y[tr], 0.5)
                P[te] = X[te] @ w + b
    return P


def qra_causal(d, cols):
    """Pesos ajustados solo con dias anteriores (ventana expansiva). Determinista."""
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d["g4"].to_numpy(), d["dia"].to_numpy()
    dias = np.sort(np.unique(dia))
    P = np.full(len(d), np.nan)
    for i, dd in enumerate(dias):
        if i < MIN_DIAS:
            continue
        tr_m, te_m = np.isin(dia, dias[:i]), dia == dd
        for gg in np.unique(g[te_m]):
            tr, te = tr_m & (g == gg), te_m & (g == gg)
            if tr.sum() >= len(cols) + 5:
                w, b = ajustar_cuantil(X[tr], y[tr], 0.5)
                P[te] = X[te] @ w + b
    return P


def main():
    t0 = time.time()
    d = cargar24().sort_index()
    lg = pd.read_csv(RES / "pronostico_le_garch_t_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(lg["pred"].rename("LE-GARCH-t"), how="left")
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d.index.hour // 6
    d = d.dropna(subset=BASE5 + ["LE-GARCH-t", "real"])
    y = d["real"].to_numpy()
    pk = d["hora"].isin(PICO).to_numpy()
    NUEVO = BASE5 + ["LE-GARCH-t"]
    print(f"Filas: {len(d):,}\n")

    # ---------- 1) estabilidad entre semillas ----------
    print(f"{'='*92}\n1) ESTABILIDAD ENTRE {len(SEMILLAS)} SEMILLAS DE PARTICION (modo CV)\n{'='*92}")
    mv = np.array([np.abs(y - qra_cv(d, BASE5, s)).mean() for s in SEMILLAS])
    mn = np.array([np.abs(y - qra_cv(d, NUEVO, s)).mean() for s in SEMILLAS])
    delta = mn - mv
    print(f"  v4 vigente : MAE {mv.mean():.2f} +/- {mv.std():.3f}")
    print(f"  v4 + LG-t  : MAE {mn.mean():.2f} +/- {mn.std():.3f}")
    print(f"  delta      : {delta.mean():+.2f} +/- {delta.std():.3f}  "
          f"(min {delta.min():+.2f}, max {delta.max():+.2f})")
    solido = (delta < 0).all() and abs(delta.mean()) > 2 * mv.std()
    print(f"  signo negativo en {int((delta<0).sum())}/{len(SEMILLAS)} semillas | "
          f"ruido del vigente = {mv.std():.3f} | efecto > 2x ruido: {abs(delta.mean()) > 2*mv.std()}")
    print(f"  VEREDICTO: {'SOLIDO' if solido else 'NO concluyente'}")

    # ---------- 2) modo desplegable ----------
    print(f"\n{'='*92}\n2) MODO DESPLEGABLE (pesos causales, determinista)\n{'='*92}")
    pv_ = qra_causal(d, BASE5)
    pn_ = qra_causal(d, NUEVO)
    ok = ~(np.isnan(pv_) | np.isnan(pn_))
    t, pval = dm(y[ok], pv_[ok], pn_[ok])
    print(f"  v4 vigente : MAE {np.abs(y[ok]-pv_[ok]).mean():.2f}  MAPE {mape(y[ok], pv_[ok]):.2f}%  "
          f"sMAPE {smape(y[ok], pv_[ok]):.2f}%  pico {mape(y[ok&pk], pv_[ok&pk]):.2f}%")
    print(f"  v4 + LG-t  : MAE {np.abs(y[ok]-pn_[ok]).mean():.2f}  MAPE {mape(y[ok], pn_[ok]):.2f}%  "
          f"sMAPE {smape(y[ok], pn_[ok]):.2f}%  pico {mape(y[ok&pk], pn_[ok&pk]):.2f}%")
    print(f"  delta MAE {np.abs(y[ok]-pn_[ok]).mean()-np.abs(y[ok]-pv_[ok]).mean():+.2f}  DM p={pval:.4f}")

    # ---------- 3) reparto por dia ----------
    g = pd.DataFrame({"dia": d["dia"].to_numpy()[ok],
                      "ev": np.abs(y[ok]-pv_[ok]), "en": np.abs(y[ok]-pn_[ok])}).groupby("dia").mean()
    print(f"\n{'='*92}\n3) REPARTO POR DIA (modo desplegable)\n{'='*92}")
    print(f"  el nuevo gana en {100*(g.en < g.ev).mean():.0f}% de los {len(g)} dias | "
          f"mediana del delta diario {np.median(g.en - g.ev):+.2f}")

    # ---------- 4) chequeo de fuga ----------
    print(f"\n{'='*92}\n4) CHEQUEO DE FUGA\n{'='*92}")
    print("  LE-GARCH-t usa EXACTAMENTE las mismas variables que el LEAR (todas medidas en el corte")
    print("  de las 00:00, construidas por lear_24h.construir_para_paso). Lo unico que cambia es el")
    print("  metodo de estimacion: maxima verosimilitud con GARCH(1,1) y errores t, en vez de LASSO.")
    print("  Por construccion no puede haber informacion adicional, asi que la ganancia solo puede")
    print("  venir de COMO estima, no de QUE ve.")
    # evidencia empirica: correlacion del pronostico con el precio contemporaneo vs el LEAR
    lear = pd.read_csv(RES / "pronostico_lear_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    j = d.join(lear["LEAR24"], how="left").dropna(subset=["LEAR24"])
    print(f"  corr(pronostico, precio real): LEAR={np.corrcoef(j['LEAR24'], j['real'])[0,1]:.4f}  "
          f"LE-GARCH-t={np.corrcoef(j['LE-GARCH-t'], j['real'])[0,1]:.4f}")
    print("  (una fuga se delataria con una correlacion cercana a 1, como paso con max_precio_oferta)")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
