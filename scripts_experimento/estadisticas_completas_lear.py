# -*- coding: utf-8 -*-
"""
Cuadro COMPLETO de estadisticas para el informe: ensamble vigente (7 votantes) vs ensamble nuevo
(LEAR + 2 redes + naive, 4 votantes), en los dos modos (CV y desplegable) y en los tres tramos,
con MAE, RMSE, MAPE, sMAPE y R2 -- las mismas metricas usadas en metricas_comparables_papers.py
para poder citar ambas comparaciones (contra los papers y entre nuestros propios modelos) con el
mismo criterio.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ensamble_72h_diario import TRAMOS, lad_cv, lad_causal, RES

VIGENTE = ["NBX", "NBX_exog", "XGB-directo", "CatB-directo", "CatB-denso", "Ridge-directo", "NaiveEstac"]
NUEVO = ["LEAR", "NBX", "NBX_exog", "NaiveEstac"]


def metricas(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    err = real - pred
    smape = np.mean(np.abs(err) / ((np.abs(real) + np.abs(pred)) / 2)) * 100
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE_%": float(np.mean(np.abs(err) / real) * 100),
        "sMAPE_%": float(smape),
        "R2": float(1 - np.sum(err ** 2) / np.sum((real - real.mean()) ** 2)),
        "n": len(real),
    }


def main():
    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    lear = pd.read_csv(RES / "pronostico_lear_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v = v.merge(lear[["fecha_hora", "cutoff", "paso_horas", "LEAR"]],
                on=["fecha_hora", "cutoff", "paso_horas"], how="inner")
    v["dia"] = v["fecha_hora"].dt.normalize()

    filas = []
    for modo, fun in [("CV", lad_cv), ("desplegable", lad_causal)]:
        for nom, ini, fin in TRAMOS:
            sub = v[v.paso_horas.between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
            y = sub["real"].to_numpy()
            for etiqueta, cols in [("vigente (7 votantes)", VIGENTE), ("nuevo (LEAR+2 redes+naive, 4)", NUEVO)]:
                P, _ = fun(sub, cols)
                ok = ~np.isnan(P)
                m = metricas(y[ok], P[ok])
                m.update({"modo": modo, "tramo": nom, "ensamble": etiqueta})
                filas.append(m)

    t = pd.DataFrame(filas)[["modo", "tramo", "ensamble", "MAE", "RMSE", "MAPE_%", "sMAPE_%", "R2", "n"]]
    print(t.round(3).to_string(index=False))
    t.to_csv(RES / "estadisticas_completas_lear.csv", index=False)

    # tambien: metricas globales (los 3 tramos juntos) en modo desplegable, que es lo que se cita mas
    print(f"\n{'='*90}\nMETRICAS GLOBALES (72h completas, modo desplegable -- el que se despliega)\n{'='*90}")
    for etiqueta, cols in [("vigente (7 votantes)", VIGENTE), ("nuevo (LEAR+2 redes+naive, 4)", NUEVO)]:
        reales, preds = [], []
        for nom, ini, fin in TRAMOS:
            sub = v[v.paso_horas.between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
            P, _ = lad_causal(sub, cols)
            ok = ~np.isnan(P)
            reales.append(sub["real"].to_numpy()[ok]); preds.append(P[ok])
        m = metricas(np.concatenate(reales), np.concatenate(preds))
        print(f"{etiqueta:32s} MAE={m['MAE']:6.2f}  RMSE={m['RMSE']:6.2f}  MAPE={m['MAPE_%']:5.2f}%  "
              f"sMAPE={m['sMAPE_%']:5.2f}%  R2={m['R2']:.3f}  n={m['n']}")

    print("\nGuardado: estadisticas_completas_lear.csv")


if __name__ == "__main__":
    main()
