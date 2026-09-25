# -*- coding: utf-8 -*-
"""
Robustez del ensamble de 24h con LAD por franja horaria (MAE 43.24, MAPE 11.70%, DM p=0.0006
contra el ensamble NNLS global) antes de adoptarlo como resultado principal.

Tres preguntas:
  1) Es la ganancia un artefacto de haber elegido 4 franjas de 6h? Se repite con 1, 3, 4, 6, 12 y
     24 grupos. Si solo funciona con una particion, es seleccion afortunada; si es estable, es real.
  2) Sobrevive en version DESPLEGABLE? Pesos con ventana expansiva usando solo dias anteriores.
     (Mas parametros que el NNLS global = mas riesgo de que la CV sea optimista.)
  3) Que modelo domina en cada franja? (interpretacion para la sustentacion)
Ademas se recalcula la precision por tercil y se guarda el contrato v3 de 24h.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import cargar as cargar24, BASE5, pesos_lad, pesos_nnls
from pesos_por_franja import cv, dm

RES = Path(r"C:\Users\mgdbj\xm-spot-price-predictor") / "data/processed/resultados"


def causal(df, cols, ajustar, grupo=None, min_dias=14):
    dias = np.sort(df["dia"].unique())
    X, y = df[cols].to_numpy(), df["real"].to_numpy()
    g = np.zeros(len(df), int) if grupo is None else df[grupo].to_numpy()
    dia = df["dia"].to_numpy()
    pred = np.full(len(df), np.nan)
    for i in range(min_dias, len(dias)):
        tr_d, te_d = dia < dias[i], dia == dias[i]
        for gg in np.unique(g[te_d]):
            tr, te = tr_d & (g == gg), te_d & (g == gg)
            pred[te] = X[te] @ ajustar(X[tr], y[tr])
    return pred


def main():
    d = cargar24().sort_index()
    real = d["real"].to_numpy()
    ref = cv(d, BASE5, pesos_nnls)
    mae_ref = np.abs(real - ref).mean()
    print(f"Referencia: NNLS global (ensamble actual)  MAE={mae_ref:.2f}  "
          f"MAPE={(np.abs(real-ref)/real).mean()*100:.2f}%")

    print("\n1) Sensibilidad a la particion horaria (LAD, validacion cruzada 5 pliegues por dia)")
    filas = []
    preds = {}
    for n_grupos in [1, 3, 4, 6, 12, 24]:
        col = f"g{n_grupos}"
        d[col] = d.index.hour // (24 // n_grupos)
        p = cv(d, BASE5, pesos_lad, col)
        preds[n_grupos] = p
        mae = np.abs(real - p).mean()
        mape = (np.abs(real - p) / real).mean() * 100
        t, pv = dm(real, ref, p, 23)
        print(f"   {n_grupos:2d} grupo(s) de {24//n_grupos:2d}h  MAE={mae:6.2f}  MAPE={mape:5.2f}%  "
              f"{100*(mae_ref-mae)/mae_ref:+5.1f}%  DM p={pv:.4f} {'GANA' if pv<0.05 and t>0 else ''}")
        filas.append({"prueba": "particion", "grupos": n_grupos, "mae": mae, "mape": mape, "dm_p": pv})

    print("\n2) Version desplegable (pesos con ventana expansiva, solo dias anteriores)")
    c_nnls = causal(d, BASE5, pesos_nnls)
    c_lad1 = causal(d, BASE5, pesos_lad)
    c_lad4 = causal(d, BASE5, pesos_lad, "g4")
    ok = ~np.isnan(c_nnls)
    r = real[ok]
    for nom, p in [("causal NNLS global", c_nnls), ("causal LAD global", c_lad1), ("causal LAD 4 franjas", c_lad4)]:
        mae = np.abs(r - p[ok]).mean()
        t, pv = dm(r, c_nnls[ok], p[ok], 23) if nom != "causal NNLS global" else (np.nan, np.nan)
        print(f"   {nom:24s} MAE={mae:6.2f}  MAPE={(np.abs(r-p[ok])/r).mean()*100:5.2f}%  vs causal NNLS p={pv:.4f}")
        filas.append({"prueba": "causal", "grupos": nom, "mae": mae, "dm_p": pv})
    t, pv = dm(r, ref[ok], c_lad4[ok], 23)
    print(f"   causal LAD 4 franjas vs ensamble actual (CV): t={t:.3f} p={pv:.4f}")

    print("\n3) Pesos LAD por franja (ajuste descriptivo sobre todo 2026)")
    nombres = ["madrugada 0-5h", "manana 6-11h", "tarde 12-17h", "noche 18-23h"]
    X, y = d[BASE5].to_numpy(), d["real"].to_numpy()
    for gg, nom in enumerate(nombres):
        m = d["g4"].to_numpy() == gg
        w = pesos_lad(X[m], y[m])
        print(f"   {nom:16s} " + " | ".join(f"{mo}={wi:.2f}" for mo, wi in zip(BASE5, w)))

    print("\n4) Precision por tercil (cortes sobre el precio real de 2026)")
    cortes = np.quantile(real, [1 / 3, 2 / 3])
    t_real = np.digitize(real, cortes)
    for nom, p in [("NNLS global (actual)", ref), ("LAD 4 franjas", preds[4])]:
        t_pred = np.digitize(p, cortes)
        grave = (np.abs(t_real - t_pred) == 2).mean() * 100
        print(f"   {nom:22s} acierto exacto={(t_real == t_pred).mean()*100:5.2f}%  errores graves={grave:.2f}%")

    out = pd.DataFrame({"fecha_hora": d.index, "real": real, "pred_ensamble": preds[4], "pred_causal": c_lad4})
    out.to_csv(RES / "stacking_24h_v3_2026.csv", index=False)
    pd.DataFrame(filas).to_csv(RES / "robustez_franjas_24h.csv", index=False)
    rmse = np.sqrt(((real - preds[4]) ** 2).mean())
    print(f"\nContrato 24h v3 (LAD 4 franjas): MAE={np.abs(real-preds[4]).mean():.2f} RMSE={rmse:.2f} "
          f"MAPE={(np.abs(real-preds[4])/real).mean()*100:.2f}%  -> stacking_24h_v3_2026.csv")


if __name__ == "__main__":
    main()
