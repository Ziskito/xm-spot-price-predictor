# -*- coding: utf-8 -*-
"""
Evaluador comun de conjuntos de votantes para el ensamble de 24h, con el metodo de v4 (QRA por
franja horaria de 6h): validacion cruzada de 5 pliegues por dia, version desplegable (ventana
expansiva), DM en orden cronologico contra v4 actual y metricas de forma del dia.
Decide si TFT, LSTM, las ventanas de calibracion o los modelos recalibrados entran al ensamble.
Los conjuntos cuyos archivos todavia no existen se omiten.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import crudas_o6, RES
from stacking_24h_v2 import BASE5
from validar_qra_24h import qra
from pesos_por_franja import dm
from forma_del_dia import por_dia

FUENTES = {
    "TFT": ("tft_lstm_24h_2026.csv", "TFT"),
    "LSTM": ("tft_lstm_24h_2026.csv", "LSTM"),
    "N-BEATSx_3a": ("ventanas_24h_actual_2026.csv", "N-BEATSx_3a"),
    "N-BEATSx_5a": ("ventanas_24h_actual_2026.csv", "N-BEATSx_5a"),
    "N-HiTS_3a": ("ventanas_24h_actual_2026.csv", "N-HiTS_3a"),
    "N-HiTS_5a": ("ventanas_24h_actual_2026.csv", "N-HiTS_5a"),
    # ARX+GARCH/XGBoost SI se reentrenaban desde cero en la v1 (loop propio, sin el bug de warm-start
    # de neuralforecast) -- se mantienen apuntando a ese archivo. NBEATSx/NHITS usan la v2 corregida
    # (recalibracion_2026_v2.py, reentrenamiento desde cero -- ver bitacora 2026-09-11 tarde).
    "ARX_rec": ("recalibracion_2026_predicciones.csv", "ARX_rec"),
    "XGB_rec": ("recalibracion_2026_predicciones.csv", "XGB_rec"),
    "NBEATSx_rec": ("recalibracion_2026_v2_predicciones.csv", "NBEATSx_rec"),
    "NHITS_rec": ("recalibracion_2026_v2_predicciones.csv", "NHITS_rec"),
}
REF = "v4 actual (5 modelos)"
CONJUNTOS = {
    REF: BASE5,
    "v4 + TFT": BASE5 + ["TFT"],
    "v4 + LSTM": BASE5 + ["LSTM"],
    "v4 + TFT + LSTM": BASE5 + ["TFT", "LSTM"],
    "ventanas promediadas (N-BEATSx y N-HiTS)": ["Persistencia", "XGBoost", "ARX+GARCH", "NBX_prom", "NHITS_prom"],
    "recalibrados": ["Persistencia", "XGB_rec", "ARX_rec", "NBEATSx_rec", "NHITS_rec"],
    "recalibrados + congelados": BASE5 + ["XGB_rec", "ARX_rec", "NBEATSx_rec", "NHITS_rec"],
}


def cargar():
    d = crudas_o6()
    for nom, (arch, col) in FUENTES.items():
        ruta = RES / arch
        if ruta.exists():
            e = pd.read_csv(ruta, parse_dates=["fecha_hora"]).drop_duplicates("fecha_hora").set_index("fecha_hora")
            if col in e.columns:
                d = d.join(e[col].rename(nom), how="left")
    if {"N-BEATSx_3a", "N-BEATSx_5a"}.issubset(d.columns):
        d["NBX_prom"] = d[["N-BEATSx", "N-BEATSx_3a", "N-BEATSx_5a"]].mean(axis=1)
    if {"N-HiTS_3a", "N-HiTS_5a"}.issubset(d.columns):
        d["NHITS_prom"] = d[["N-HiTS", "N-HiTS_3a", "N-HiTS_5a"]].mean(axis=1)
    d = d.sort_index()
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    return d


def main():
    d = cargar()
    disponibles = {k: v for k, v in CONJUNTOS.items() if set(v).issubset(d.columns)}
    todas = sorted({c for v in disponibles.values() for c in v})
    d = d.dropna(subset=todas + ["real"])
    real = d["real"].to_numpy()
    print(f"Horas: {len(d):,} | conjuntos evaluables: {list(disponibles)}")
    print("\nMAE individual de cada votante:")
    for c in todas:
        e = np.abs(real - d[c])
        print(f"  {c:14s} MAE={e.mean():7.2f}  MAPE={(e/real).mean()*100:5.2f}%")

    res = {nom: (qra(d, cols, "g4", [0.5])[:, 0], qra(d, cols, "g4", [0.5], modo="causal")[:, 0])
           for nom, cols in disponibles.items()}
    ref_cv, ref_c = res[REF]
    ok = ~np.isnan(ref_c)
    filas = []
    print(f"\n{'conjunto':42s} {'MAE CV':>7s} {'MAPE':>6s} {'p':>7s}  {'MAE despl.':>10s} {'p':>7s}  {'Corr-f':>6s} {'MHD':>5s} {'MPD':>6s}")
    for nom, (qcv, qc) in res.items():
        e = np.abs(real - qcv)
        mae_c = np.abs(real[ok] - qc[ok]).mean()
        p1 = dm(real, ref_cv, qcv, 23)[1] if nom != REF else np.nan
        p2 = dm(real[ok], ref_c[ok], qc[ok], 23)[1] if nom != REF else np.nan
        f = por_dia(d.index, real, qcv)
        print(f"{nom:42s} {e.mean():7.2f} {(e/real).mean()*100:5.2f}% {p1:7.4f}  {mae_c:10.2f} {p2:7.4f}  "
              f"{f['corr_f'].mean():6.3f} {f['mhd'].mean():5.2f} {f['mpd'].mean():6.2f}")
        filas.append({"conjunto": nom, "votantes": "+".join(disponibles[nom]), "mae_cv": e.mean(),
                      "mape_cv": (e / real).mean() * 100, "p_vs_v4_cv": p1, "mae_desplegable": mae_c,
                      "p_vs_v4_desplegable": p2, "corr_f": f["corr_f"].mean(), "mhd": f["mhd"].mean(),
                      "mpd": f["mpd"].mean()})
    pd.DataFrame(filas).to_csv(RES / "evaluacion_votantes_24h.csv", index=False)
    pred = pd.DataFrame({"real": real, **{f"{k} [CV]": v[0] for k, v in res.items()},
                         **{f"{k} [desplegable]": v[1] for k, v in res.items()}}, index=d.index)
    pred.reset_index().to_csv(RES / "evaluacion_votantes_24h_predicciones.csv", index=False)
    print("\nGuardado: evaluacion_votantes_24h.csv")


if __name__ == "__main__":
    main()
