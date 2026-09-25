# -*- coding: utf-8 -*-
"""
Contrato unificado 1-72h para OE3: el especialista de 24h (ensamble v4, QRA por franja horaria) en
los pasos 1-24 y el ensamble v3 de 72h (LAD por tramo) en los pasos 25-72.

Es legitimo combinarlos porque los cortes de 72h (cada 3 dias a las 23:00) son un subconjunto de
los cortes diarios de 24h (23:00): para h=1..24 ambos pronostican exactamente las mismas horas desde
el mismo instante de corte. Antes de unificar se verifica, en esas mismas filas, que el especialista
de 24h realmente es mejor en punto y en bandas; si no lo fuera, se conserva v3 completo.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from qra_bandas import metricas_banda, CALENT

RES = Path(r"C:\Users\mgdbj\xm-spot-price-predictor") / "data/processed/resultados"


def hac_p(d, maxlags):
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def main():
    v3 = pd.read_csv(RES / "pronostico_ensamble_72h_v3_adaptativo.csv", parse_dates=["fecha_hora", "cutoff"])
    b24 = pd.read_csv(RES / "bandas_24h_qra_causal_2026.csv", parse_dates=["fecha_hora"])
    t1 = (v3[v3["paso_horas"] <= 24]
          .merge(b24.rename(columns={"real": "real_24", "q10": "q10_24", "q50": "q50_24", "q90": "q90_24"}),
                 on="fecha_hora", how="left")
          .sort_values(["cutoff", "paso_horas"]).reset_index(drop=True))
    assert t1["q50_24"].notna().all(), "faltan horas del especialista de 24h"
    assert np.allclose(t1["real"], t1["real_24"]), "los valores reales no coinciden"
    y = t1["real"].to_numpy()

    e_v3 = np.abs(y - t1["q50"].to_numpy())
    e_24 = np.abs(y - t1["q50_24"].to_numpy())
    t, p = hac_p(e_v3 - e_24, 23)
    print(f"Pasos 1-24 en los {t1['cutoff'].nunique()} cortes de 72h ({len(t1)} horas):")
    print(f"  v3 (LAD 72h)            MAE={e_v3.mean():6.2f}  MAPE={(e_v3/y).mean()*100:5.2f}%")
    print(f"  v4 (QRA 24h, franjas)   MAE={e_24.mean():6.2f}  MAPE={(e_24/y).mean()*100:5.2f}%   DM t={t:.2f} p={p:.4f}")

    ev = (t1["fecha_hora"] >= t1["fecha_hora"].min() + pd.Timedelta(days=CALENT)).to_numpy()
    _, _, is_v3 = metricas_banda(y[ev], t1["q10"].to_numpy()[ev], t1["q90"].to_numpy()[ev])
    _, _, is_24 = metricas_banda(y[ev], t1["q10_24"].to_numpy()[ev], t1["q90_24"].to_numpy()[ev])
    for nom, (l, u) in {"bandas v3": ("q10", "q90"), "bandas v4 QRA": ("q10_24", "q90_24")}.items():
        cob, ancho, isc = metricas_banda(y[ev], t1[l].to_numpy()[ev], t1[u].to_numpy()[ev])
        print(f"  {nom:22s} cobertura={cob:5.1f}%  ancho={ancho:6.1f}  IS={isc.mean():6.1f}")
    ti, pi = hac_p(is_v3 - is_24, 23)
    print(f"  Interval Score v4 vs v3: t={ti:.2f} p={pi:.4f}")

    usar_24 = t > 0 and p < 0.10 and ti >= 0
    print(f"\nDecision: {'usar el especialista de 24h en los pasos 1-24' if usar_24 else 'conservar v3 completo'}")

    if usar_24:
        u1 = t1[["fecha_hora", "cutoff", "paso_horas", "real"]].copy()
        u1["q50"], u1["q10"], u1["q90"] = t1["q50_24"], t1["q10_24"], t1["q90_24"]
        u1["margen"] = np.nan  # la calibracion del especialista se hizo por corte en qra_bandas.py
        u1["fuente"] = "v4_qra_24h"
        u2 = v3[v3["paso_horas"] > 24].copy()
        u2["fuente"] = "v3_lad_72h"
        uni = pd.concat([u1, u2[u1.columns]], ignore_index=True).sort_values(["cutoff", "paso_horas"])
    else:
        uni = v3.copy()
        uni["fuente"] = "v3_lad_72h"
    uni = uni[["fecha_hora", "cutoff", "paso_horas", "real", "q50", "margen", "q10", "q90", "fuente"]]
    uni.to_csv(RES / "pronostico_unificado_72h_2026.csv", index=False)

    print("\nContrato unificado por tramo:")
    for nom, ini, fin in [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]:
        s = uni[(uni.paso_horas >= ini) & (uni.paso_horas <= fin)]
        e = np.abs(s["real"] - s["q50"])
        print(f"  {nom:7s} MAE={e.mean():6.2f}  MAPE={(e/s['real']).mean()*100:5.2f}%")
    e = np.abs(uni["real"] - uni["q50"])
    print(f"  global  MAE={e.mean():6.2f}  MAPE={(e/uni['real']).mean()*100:5.2f}%")
    print("Guardado: pronostico_unificado_72h_2026.csv")


if __name__ == "__main__":
    main()
