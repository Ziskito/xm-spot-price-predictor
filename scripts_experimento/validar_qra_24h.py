# -*- coding: utf-8 -*-
"""
Validacion de QRA por franja horaria como ensamble de 24h (MAE 42.51, MAPE 11.27% en la primera
corrida de qra_bandas.py) antes de adoptarlo sobre LAD por franja (MAE 43.24).

QRA en el cuantil 0.5 difiere del LAD solo en dos cosas: tiene intercepto y los pesos no estan
obligados a sumar 1. El intercepto puede absorber un sesgo de nivel comun a todos los votantes --
justo el tipo de sesgo que el proyecto documento (sobreestimacion en precios bajos) -- pero tambien
agrega parametros. Se verifica:
  1) sensibilidad a la particion horaria (1, 3, 4, 6 franjas);
  2) DM contra el ensamble actual y contra LAD por franja;
  3) version DESPLEGABLE: QRA con ventana expansiva (solo dias anteriores) para los tres cuantiles,
     mas calibracion conforme causal, y sus bandas contra el contrato de bandas de 24h vigente.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import cargar as cargar24, BASE5, pesos_nnls, pesos_lad
from pesos_por_franja import cv, dm
from robustez_franjas_24h import causal
from qra_bandas import ajustar_cuantil, margen_por_corte, metricas_banda, TAUS, CALENT

RES = Path(r"C:\Users\mgdbj\xm-spot-price-predictor") / "data/processed/resultados"


def qra(df, cols, grupo, taus, modo="cv", min_dias=14):
    X, y = df[cols].to_numpy(), df["real"].to_numpy()
    g = df[grupo].to_numpy()
    dia = df["dia"].to_numpy()
    dias = np.sort(np.unique(dia))
    Q = np.full((len(df), len(taus)), np.nan)
    if modo == "cv":
        pasos = [(~df["dia"].isin(dt).to_numpy(), df["dia"].isin(dt).to_numpy())
                 for dt in np.array_split(np.random.default_rng(42).permutation(dias), 5)]
    else:
        pasos = [(dia < dias[i], dia == dias[i]) for i in range(min_dias, len(dias))]
    for tr_m, te_m in pasos:
        for gg in np.unique(g[te_m]):
            tr, te = tr_m & (g == gg), te_m & (g == gg)
            for j, tau in enumerate(taus):
                w, b = ajustar_cuantil(X[tr], y[tr], tau)
                Q[te, j] = X[te] @ w + b
    if len(taus) == 3:
        Q[:, 0] = np.minimum(Q[:, 0], Q[:, 1])
        Q[:, 2] = np.maximum(Q[:, 2], Q[:, 1])
    return Q


def main():
    d = cargar24().sort_index()
    d["dia"] = d.index.normalize()
    real = d["real"].to_numpy()
    for n in [1, 3, 4, 6]:
        d[f"g{n}"] = d.index.hour // (24 // n)
    ref = cv(d, BASE5, pesos_nnls)
    lad4 = cv(d, BASE5, pesos_lad, "g4")
    mae_ref = np.abs(real - ref).mean()

    print("1-2) QRA q50 por particion horaria (validacion cruzada)")
    q50 = {}
    for n in [1, 3, 4, 6]:
        q50[n] = qra(d, BASE5, f"g{n}", [0.5])[:, 0]
        mae = np.abs(real - q50[n]).mean()
        _, p_ref = dm(real, ref, q50[n], 23)
        _, p_lad = dm(real, lad4, q50[n], 23)
        print(f"   {n} franja(s): MAE={mae:6.2f} MAPE={(np.abs(real-q50[n])/real).mean()*100:5.2f}%  "
              f"{100*(mae_ref-mae)/mae_ref:+5.1f}% vs actual (p={p_ref:.4f})  vs LAD 4 franjas p={p_lad:.4f}")

    print("\n3) Version desplegable (ventana expansiva)")
    Qc = qra(d, BASE5, "g4", TAUS, modo="causal")
    c_nnls = causal(d, BASE5, pesos_nnls)
    c_lad4 = causal(d, BASE5, pesos_lad, "g4")
    ok = ~np.isnan(Qc[:, 1])
    r = real[ok]
    for nom, p in [("causal NNLS global (actual desplegable)", c_nnls), ("causal LAD 4 franjas", c_lad4),
                   ("causal QRA 4 franjas", Qc[:, 1])]:
        print(f"   {nom:40s} MAE={np.abs(r-p[ok]).mean():6.2f}  MAPE={(np.abs(r-p[ok])/r).mean()*100:5.2f}%")
    _, p1 = dm(r, c_nnls[ok], Qc[ok, 1], 23)
    _, p2 = dm(r, c_lad4[ok], Qc[ok, 1], 23)
    print(f"   DM causal QRA vs causal NNLS: p={p1:.4f} | vs causal LAD 4 franjas: p={p2:.4f}")

    obj = d.index.to_numpy()
    cor = (d.index.normalize() - pd.Timedelta(hours=1)).to_numpy()
    score = np.maximum(Qc[:, 0] - real, real - Qc[:, 2])
    score_ok = np.where(ok, score, np.nan)
    m = np.zeros(len(real))
    m[ok] = margen_por_corte(obj[ok], cor[ok], score[ok])
    ev = ok & (obj >= obj[ok].min() + np.timedelta64(CALENT, "D"))
    nb = (pd.read_csv(RES / "pronostico_con_bandas_2026_adaptativo.csv", parse_dates=["fecha_hora"])
          .drop_duplicates("fecha_hora").set_index("fecha_hora").reindex(d.index))
    print(f"\n   Bandas desplegables sobre {ev.sum()} horas:")
    for nom, (l, u) in {"N-BEATSx solo, adaptativa (contrato vigente)": (nb["q10"].to_numpy(), nb["q90"].to_numpy()),
                        "QRA causal + calibracion causal": (Qc[:, 0] - m, Qc[:, 2] + m)}.items():
        cob, ancho, isc = metricas_banda(real[ev], l[ev], u[ev])
        print(f"   {nom:46s} cobertura={cob:5.1f}%  ancho={ancho:6.1f}  IS={isc.mean():6.1f}")

    cortes = np.quantile(real, [1 / 3, 2 / 3])
    t_real = np.digitize(real, cortes)
    for nom, p in [("NNLS global", ref), ("LAD 4 franjas", lad4), ("QRA 4 franjas", q50[4])]:
        tp = np.digitize(p, cortes)
        print(f"   tercil {nom:14s} acierto={(t_real==tp).mean()*100:5.2f}%  graves={(np.abs(t_real-tp)==2).mean()*100:.2f}%")

    pd.DataFrame({"fecha_hora": d.index, "real": real, "pred_ensamble": q50[4],
                  "pred_causal": Qc[:, 1], "q10_causal": np.minimum(Qc[:, 0] - m, Qc[:, 1]),
                  "q90_causal": np.maximum(Qc[:, 2] + m, Qc[:, 1])}).to_csv(
        RES / "stacking_24h_v4_qra_2026.csv", index=False)
    print("\nGuardado: stacking_24h_v4_qra_2026.csv")


if __name__ == "__main__":
    main()
