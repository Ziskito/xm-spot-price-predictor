# -*- coding: utf-8 -*-
"""
Ensamble de 24 h con el predespacho ideal de XM como septimo votante (prueba del 7-oct-2026).

El predespacho ideal (archivo iMAR, descargar_predespacho_ideal.py) trae el costo marginal hora a hora del
dia D, calculado con las ofertas reales y la demanda pronosticada; XM lo publica el dia D-1 (mediana 10:00,
maximo observado 17:07). Alineacion causal con nuestras ventanas:
  * 2026 (corte 00:00 de D, horas 01:00 de D a 00:00 de D+1): horas 01-23 con el iMAR de D; la hora 00:00
    de D+1 usa el valor de las 23:00 del iMAR de D (el iMAR de D+1 sale despues del corte).
  * origen 5 (corte 23:00 de D-1, horas 00-23 de D): iMAR de D completo.
Mismo combinador causal de la version desplegable (QRA por franja de 6 h, objetivo sMAPE); del 1 al 14 de
enero los pesos salen del origen 5 (ensamble_24h_enero.py). Se compara con el ensamble vigente de 6 votantes.
Salidas: data/processed/resultados/ensamble_con_predespacho_24h.csv (metricas) y
         ensamble_24h_operativo_predespacho_2026.csv (prediccion por hora)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from ensamble_24h_enero import VOTANTES, causal_con_previa, datos_2026, origen5  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "data" / "processed" / "resultados"


def imar():
    t = pd.read_csv(RAIZ / "data" / "processed" / "predespacho_ideal_xm.csv", parse_dates=["fecha_hora"])
    t = t[t["a_tiempo"]]          # solo lo publicado antes de las 23:00 del dia anterior (validar_predespacho_ideal.py)
    return t.set_index("fecha_hora")["costo_marginal"]


def alinear_2026(idx, cm):
    """Valor del predespacho disponible al corte (00:00 del dia de la ventana) para cada hora."""
    ventana = (idx - pd.Timedelta(hours=1)).normalize()
    misma = idx.normalize() == ventana
    fuente = np.where(misma, idx, ventana + pd.Timedelta(hours=23))
    return cm.reindex(pd.DatetimeIndex(fuente)).to_numpy()


def dm(y, a, b, maxlags=24):
    d = np.abs(y - a) - np.abs(y - b)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def metricas(y, p, dias):
    g = pd.DataFrame({"y": y, "p": p, "d": dias})
    sd = g.groupby("d").agg(sp=("p", "std"), sy=("y", "std"))
    rho = g.groupby("d").apply(lambda x: x["p"].corr(x["y"]))
    niv = (g.groupby("d")["p"].mean() - g.groupby("d")["y"].mean()) ** 2
    amp = (sd.sp - sd.sy) ** 2
    forma = 2 * sd.sp * sd.sy * (1 - rho)
    tot = niv.sum() + amp.sum() + forma.sum()
    dr, dp = g.groupby("d")["y"].diff(3), g.groupby("d")["p"].diff(3)
    m = dr.notna()
    top = dr[m].abs() >= dr[m].abs().quantile(.9)
    med_y, med_p = g.groupby("d")["y"].transform("median"), g.groupby("d")["p"].transform("median")
    ev, al = g.y <= .7 * med_y, g.p <= .7 * med_p
    return {"MAE": np.abs(y - p).mean(), "MAPE_%": np.mean(np.abs(y - p) / y) * 100,
            "amplitud": (sd.sp / sd.sy).median(), "rampa_dir_%": (np.sign(dr[m][top]) == np.sign(dp[m][top])).mean() * 100,
            "rampa_fraccion": (dp[m][top] / dr[m][top]).median(),
            "err_nivel_%": niv.sum() / tot * 100, "err_amplitud_%": amp.sum() / tot * 100, "err_forma_%": forma.sum() / tot * 100,
            "desplomes_marcados_%": (ev & al).sum() / ev.sum() * 100, "falsas_alarmas_desplome": int((al & ~ev).sum())}


if __name__ == "__main__":
    cm = imar()
    d = datos_2026()
    d["Predespacho"] = alinear_2026(d.index, cm)
    previa = origen5()
    previa["Predespacho"] = cm.reindex(previa.index).to_numpy()
    print(f"predespacho disponible: 2026 {d['Predespacho'].notna().mean() * 100:.1f} % de las horas, "
          f"origen 5 {previa['Predespacho'].notna().mean() * 100:.1f} %")
    previa = previa.dropna(subset=["Predespacho"])
    con = VOTANTES + ["Predespacho"]
    d["ens6"] = causal_con_previa(d, previa, VOTANTES, "g4")
    hay = d["Predespacho"].notna()
    d["ens7"] = d["ens6"]
    d.loc[hay, "ens7"] = causal_con_previa(d[hay], previa, con, "g4")   # dias sin predespacho: ensamble de 6
    d["Predespacho"] = d["Predespacho"].fillna(d["ens6"])
    print(f"horas de 2026 sin predespacho a tiempo (usan el ensamble de 6): {int((~hay).sum())}")
    # predespacho solo, con correccion de sesgo causal (mediana del cociente real/predespacho de los 30 dias previos)
    dias = np.sort(d["dia"].unique())
    k = pd.Series(1.0, index=d.index)
    for i, dia in enumerate(dias):
        prev = d["dia"].isin(dias[max(0, i - 30):i])
        if prev.sum() >= 24 * 7:
            k[d["dia"] == dia] = np.median(d.loc[prev, "real"] / d.loc[prev, "Predespacho"].clip(lower=1))
    d["pred_corr"] = d["Predespacho"] * k
    d = d.dropna(subset=["ens6", "ens7"])
    y, dias = d["real"].to_numpy(), d["dia"].to_numpy()
    filas = []
    for nombre, col in (("ensamble vigente (6 votantes)", "ens6"), ("predespacho ideal de XM solo", "Predespacho"),
                        ("predespacho con correccion de sesgo", "pred_corr"), ("ensamble + predespacho (7 votantes)", "ens7")):
        f = {"modelo": nombre, **metricas(y, d[col].to_numpy(), dias)}
        if col != "ens6":
            f["DM_t_vs_vigente"], f["DM_p"] = dm(y, d["ens6"].to_numpy(), d[col].to_numpy())
        filas.append(f)
    # por mitades (para ver estabilidad) y en las horas de rampa
    mitad = d.index < d.index.min() + (d.index.max() - d.index.min()) / 2
    for nom, m in (("1a mitad", mitad), ("2a mitad", ~mitad)):
        for col in ("ens6", "ens7"):
            filas.append({"modelo": f"{'vigente' if col == 'ens6' else '+ predespacho'} · {nom}",
                          "MAE": np.abs(y[m] - d[col].to_numpy()[m]).mean()})
    dr = d.groupby("dia")["real"].diff(3).abs()
    rampa = (dr >= dr.quantile(.9)).to_numpy()
    for col in ("ens6", "ens7"):
        filas.append({"modelo": f"{'vigente' if col == 'ens6' else '+ predespacho'} · solo horas de rampa (10 % mayores)",
                      "MAE": np.abs(y[rampa] - d[col].to_numpy()[rampa]).mean()})
    t = pd.DataFrame(filas)
    t.to_csv(RES / "ensamble_con_predespacho_24h.csv", index=False)
    d.reset_index()[["fecha_hora", "real", "ens7", "Predespacho"]].rename(columns={"ens7": "pred_desplegable"}).to_csv(
        RES / "ensamble_24h_operativo_predespacho_2026.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"horas evaluadas: {len(d)} ({d.index.min():%d/%m/%Y} a {d.index.max():%d/%m/%Y})")
    print(t.round(2).to_string(index=False))
