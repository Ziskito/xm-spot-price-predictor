# -*- coding: utf-8 -*-
"""
Ensamble de 24 h desplegable extendido a la primera quincena de enero de 2026.

El combinador causal (QRA por franja de 6 h, objetivo sMAPE) aprende los pesos solo con dias anteriores
y necesita 14 dias de calentamiento, por eso la version desplegable arranca el 15-ene-2026. Aqui, para
los dias 1 a 14 de enero, los pesos se aprenden con el ultimo año historico del walk-forward (origen 5,
jul-2024 a jun-2025, los mismos 6 votantes) mas los dias de 2026 ya transcurridos. Es informacion
anterior a cada dia, asi que sigue siendo causal. Desde el 15-ene se usa exactamente el combinador de
siempre (solo dias de 2026), y el script verifica que esas horas coinciden con
informe_avance/ensamble_24h_desplegable.csv.

El sexto votante (GARCH-ged) del origen 5 se genera con el mismo procedimiento de
ensamble_origenes_historicos.py y se guarda en garch_ged_origen5_24h.csv para no recalcularlo.
Salida: data/processed/resultados/ensamble_24h_operativo_2026.csv (fecha_hora, real, pred_desplegable)
No modifica los archivos del informe.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import RES, cargar_completo  # noqa: E402
from stacking_24h_v2 import BASE5, cargar as cargar24  # noqa: E402
from combinador_optimo_mape import qra_pesos  # noqa: E402

VOTANTES = BASE5 + ["GARCH-ged"]
CACHE_GARCH = RES / "garch_ged_origen5_24h.csv"


def origen5():
    c = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    c = c[c["origen"] == "Origen 5"]
    d = c.pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
    d = d.join(c.groupby("fecha_hora")["real"].first())
    if CACHE_GARCH.exists():
        g = pd.read_csv(CACHE_GARCH, parse_dates=["fecha_hora"]).set_index("fecha_hora")["GARCH-ged"]
    else:
        from ensamble_origenes_historicos import garch_ged
        t0 = time.time()
        g, fallos = garch_ged(cargar_completo().reset_index(drop=True), d.index.min(), d.index.max())
        g.rename("GARCH-ged").to_frame().to_csv(CACHE_GARCH)
        print(f"GARCH-ged del origen 5 generado en {(time.time() - t0) / 60:.1f} min ({fallos} pasos con respaldo LASSO)")
    d = d.join(g, how="left").dropna(subset=VOTANTES + ["real"])
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    return d


def datos_2026():
    """Igual que desplegable() en informe_avance_resultados.py."""
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred"].rename("GARCH-ged"), how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    return d.dropna(subset=VOTANTES + ["real"])


def causal_con_previa(d, previa, cols, grupo, min_dias=14, iteraciones=3):
    """Combinador causal de franja_medianoche_24h.causal; mientras 2026 tenga menos de `min_dias` dias,
    entrena con `previa` (origen 5) mas los dias de 2026 ya transcurridos."""
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d[grupo].to_numpy(), d["dia"].to_numpy()
    Xp, yp, gp = previa[cols].to_numpy(), previa["real"].to_numpy(), previa[grupo].to_numpy()
    dias = np.sort(np.unique(dia))
    P = np.full(len(d), np.nan)
    for i in range(len(dias)):
        tr_m, te_m = np.isin(dia, dias[:i]), dia == dias[i]
        for gg in np.unique(g[te_m]):
            tr, te = tr_m & (g == gg), te_m & (g == gg)
            Xt, yt = X[tr], y[tr]
            if i < min_dias:
                Xt = np.vstack([Xp[gp == gg], Xt])
                yt = np.concatenate([yp[gp == gg], yt])
            if len(yt) < len(cols) + 5:
                continue
            w = 1.0 / np.maximum(yt, 1e-6)
            ww, b = qra_pesos(Xt, yt, w)
            for _ in range(iteraciones):
                pr = Xt @ ww + b
                w = 2.0 / np.maximum(np.abs(yt) + np.abs(pr), 1e-6)
                ww, b = qra_pesos(Xt, yt, w)
            P[te] = X[te] @ ww + b
    return P


if __name__ == "__main__":
    previa = origen5()
    d = datos_2026()
    d["pred_desplegable"] = causal_con_previa(d, previa, VOTANTES, "g4")
    d = d.dropna(subset=["pred_desplegable"])

    vigente = pd.read_csv(RES / "informe_avance" / "ensamble_24h_desplegable.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    comun = vigente.index.intersection(d.index)
    dif = np.abs(vigente.loc[comun, "pred_desplegable"] - d.loc[comun, "pred_desplegable"]).max()
    print(f"Desde el 15-ene: {len(comun)} horas, diferencia maxima con la version desplegable = {dif:.2e}")
    assert dif < 1e-6, "el combinador cambio fuera de la quincena de enero"

    ene = d[d.index < vigente.index.min()]
    y = ene["real"]
    print(f"1 al 14 de enero: {len(ene)} horas | MAE ensamble {np.abs(y - ene['pred_desplegable']).mean():.2f} "
          f"| persistencia {np.abs(y - ene['Persistencia']).mean():.2f} "
          f"| mejor votante {min(VOTANTES[1:], key=lambda m: np.abs(y - ene[m]).mean())}")
    out = d.reset_index()[["fecha_hora", "real", "pred_desplegable"]]
    out.to_csv(RES / "ensamble_24h_operativo_2026.csv", index=False)
    print("guardado ensamble_24h_operativo_2026.csv:", out["fecha_hora"].min(), "a", out["fecha_hora"].max(), len(out), "horas")
