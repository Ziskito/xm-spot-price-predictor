# -*- coding: utf-8 -*-
"""
Resumen del walk-forward con origenes de 12 meses (notebook 10 reejecutado el 2026-09-11).
Para cada origen y modelo: MAE, MAPE, rMAE contra el naive semanal (Lago et al. 2021), sesgo medio
y por tercil de precio (el "sesgo invertido"), y metricas de forma del dia (Corr-f, MHD, MPD).
Compara con la version de 3 meses y verifica que el Origen 6 (2026) no cambio al reejecutar.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES
from forma_del_dia import por_dia

MODELOS = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]


def main():
    nuevo = pd.read_csv(RES / "walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
    viejo = pd.read_csv(RES / "walkforward_predicciones_crudas_3meses_backup.csv", parse_dates=["fecha_hora"])
    piv = lambda x: x[x["origen"] == "Origen 6"].pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
    dif = (piv(nuevo) - piv(viejo)).abs().max()
    print("Origen 6: diferencia maxima entre la corrida nueva y la anterior, por modelo:")
    print("  " + " | ".join(f"{m}={v:.6f}" for m, v in dif.items()))

    lag168 = cargar_completo().set_index("fecha_hora")["precio_lag168h"]
    filas = []
    for o, g in nuevo.groupby("origen"):
        p = g.pivot_table(index="fecha_hora", columns="modelo", values="prediccion").dropna()
        real = g.groupby("fecha_hora")["real"].first().reindex(p.index)
        mae_naive = (real - lag168.reindex(p.index)).abs().mean()
        terc = pd.qcut(real, 3, labels=["bajo", "medio", "alto"])
        for m in MODELOS:
            e = real - p[m]
            f = por_dia(p.index, real.to_numpy(), p[m].to_numpy())
            st = (p[m] - real).groupby(terc, observed=True).mean()
            filas.append({"origen": o, "modelo": m, "desde": p.index.min(), "hasta": p.index.max(), "n_horas": len(p),
                          "mae": e.abs().mean(), "mape": (e.abs() / real).mean() * 100, "rmae": e.abs().mean() / mae_naive,
                          "sesgo": (p[m] - real).mean(), "sesgo_pct": ((p[m] - real) / real).mean() * 100,
                          "sesgo_tercil_bajo": st["bajo"], "sesgo_tercil_alto": st["alto"],
                          "corr_f": f["corr_f"].mean(), "mhd": f["mhd"].mean(), "mpd": f["mpd"].mean()})
    r = pd.DataFrame(filas)
    r.to_csv(RES / "walkforward_anual_resumen.csv", index=False)
    rango = r.groupby("origen")[["desde", "hasta", "n_horas"]].first()
    print("\nVentanas de prueba:\n" + rango.to_string())
    for col, titulo, dec in [("mae", "MAE", 2), ("mape", "MAPE %", 2), ("rmae", "rMAE (>1 = pierde contra el naive semanal)", 3),
                             ("sesgo", "Sesgo medio COP/kWh (pronostico - real)", 1),
                             ("sesgo_tercil_bajo", "Sesgo en el tercil BAJO de precio", 1),
                             ("sesgo_tercil_alto", "Sesgo en el tercil ALTO de precio", 1),
                             ("corr_f", "Corr-f", 3), ("mhd", "MHD (horas)", 2), ("mpd", "MPD (COP/kWh)", 1)]:
        t = r.pivot(index="origen", columns="modelo", values=col)[MODELOS]
        print(f"\n{titulo}\n" + t.round(dec).to_string())

    v = viejo.groupby(["origen", "modelo"]).apply(lambda g: (g["real"] - g["prediccion"]).abs().mean()).unstack()[MODELOS]
    print("\nMAE con la version ANTERIOR (3 meses), para comparar:\n" + v.round(2).to_string())

    dm = pd.read_csv(RES / "diebold_mariano.csv")
    print("\nDiebold-Mariano por par (orígenes de 12 meses):")
    for (m1, m2), g in dm.groupby(["modelo_1", "modelo_2"], sort=False):
        print(f"  {m1:10s} vs {m2:12s}  gana sig.: {g['conclusion'].str.contains('GANA').sum()}/6"
              f"  pierde sig.: {g['conclusion'].str.contains('PIERDE').sum()}/6")


if __name__ == "__main__":
    main()
