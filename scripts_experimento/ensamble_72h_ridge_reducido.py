# -*- coding: utf-8 -*-
"""
Prueba DECISIVA de la mejora encontrada en seleccion_variables_ridge_72h.py.

Ahi se vio que reducir las variables de Ridge-directo lo mejora como modelo individual
(MAE global 74.70 con las 37 variables -> 72.93 con las 20 mejores por informacion mutua,
y 73.56 con las 6 que sobreviven a un LASSO muy agresivo), con DM significativo.

Pero Ridge-directo es UNO de los 7 votantes del ensamble de 72h, y el ensamble ya corrige
parte de los errores de cada votante. La pregunta que importa para el proyecto no es si mejora
Ridge aislado, sino si mejora EL ENSAMBLE. Este script sustituye el votante Ridge-directo por
su version reducida, sobre la MISMA rejilla de 214 cortes diarios, y vuelve a correr el
combinador LAD por tramo en los dos modos (CV y desplegable), con DM contra el ensamble vigente.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from xgboost_directo_72h import construir_largo, COLS_ESTADO, COLS_FUTURO
from ensamble_72h_diario import CONJ, TRAMOS, lad_cv, lad_causal, hac_p, RES
from o6_comun import cargar_completo

PRIMER_CORTE = pd.Timestamp("2026-01-01 00:00")
N_WIN, H = 214, 72

# Conjuntos ganadores de seleccion_variables_ridge_72h.py
MI20 = ["precio_lag168h", "dia_anio_cos", "dia_anio_sin", "oni", "precio_mismo_hora_disp",
        "precio_bolsa", "precio_media_30d", "precio_rango_24h", "precio_media_7d",
        "volumen_embalses", "precio_media_24h", "precio_std_7d", "precio_std_24h",
        "aportes_hidricos", "volumen_embalses_vs_media30d", "demanda_media_24h",
        "aportes_hidricos_vs_media30d", "ratio_volatilidad", "demanda", "hora_cos"]
LASSO6 = ["precio_bolsa", "precio_media_24h", "aportes_hidricos_vs_media30d",
          "dia_semana_sin", "precio_lag168h", "precio_mismo_hora_disp"]


def main():
    t0 = time.time()
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    df = cargar_completo().reset_index(drop=True)
    pos = pd.Series(df.index.values, index=df["fecha_hora"])
    cortes = pd.date_range(PRIMER_CORTE, periods=N_WIN, freq="24h")
    idx_test = pos.reindex(cortes).astype(int).to_numpy()
    idx_fin = int(pos[pd.Timestamp("2026-01-01")])
    pasos = np.arange(1, H + 1)
    cols = COLS_ESTADO + COLS_FUTURO + ["precio_mismo_hora_disp", "paso_horas"]

    te = construir_largo(df, idx_test, pasos).dropna().copy()
    te["paso_horas"] = te["paso_horas"].astype(int)
    tr = construir_largo(df, np.arange(200, idx_fin - H, 4), pasos).dropna()
    print(f"Ridge: {len(tr):,} filas de entrenamiento, {len(te):,} de prueba", flush=True)

    # Ridge en sus tres versiones, misma rejilla y mismo alpha que el votante vigente
    variantes = {"Ridge-directo": cols,
                 "Ridge-MI20": MI20 + ["paso_horas"],
                 "Ridge-LASSO6": LASSO6 + ["paso_horas"]}
    pred_ridge = {}
    for nombre, cc in variantes.items():
        esc = StandardScaler().fit(tr[cc])
        m = Ridge(alpha=10.0).fit(esc.transform(tr[cc]), tr["y"])
        pred_ridge[nombre] = m.predict(esc.transform(te[cc]))
        mae = np.abs(te["y"].to_numpy() - pred_ridge[nombre]).mean()
        print(f"  {nombre:16s} ({len(cc):2d} var) MAE aislado en la rejilla diaria = {mae:.2f}", flush=True)

    clave = te[["fecha_hora", "cutoff", "paso_horas"]].copy()
    for nombre, p in pred_ridge.items():
        clave[nombre] = p

    # --- unir con los votantes ya guardados ---
    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v["NBX_semillas"] = v[["NBX", "NBX_s7", "NBX_s123"]].mean(axis=1)
    v = v.merge(clave[["fecha_hora", "cutoff", "paso_horas", "Ridge-MI20", "Ridge-LASSO6"]],
                on=["fecha_hora", "cutoff", "paso_horas"], how="left")
    v["dia"] = v["fecha_hora"].dt.normalize()
    faltan = v[["Ridge-MI20", "Ridge-LASSO6"]].isna().sum().sum()
    print(f"\nFilas unidas: {len(v):,} | sin pareja: {faltan}")
    v = v.dropna(subset=["Ridge-MI20", "Ridge-LASSO6"])

    # --- ensamble con cada version del votante Ridge ---
    print(f"\n{'='*104}\nENSAMBLE DE 72h sustituyendo el votante Ridge-directo\n{'='*104}")
    print(f"{'tramo':8s} {'conjunto':22s} {'modo':12s} {'Ridge orig':>11s} {'Ridge-MI20':>11s} "
          f"{'Ridge-LASSO6':>13s}   DM MI20 / DM LASSO6")
    filas = []
    for nombre, ini, fin in TRAMOS:
        sub = v[v["paso_horas"].between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
        y = sub["real"].to_numpy()
        for cn, cols_v in CONJ.items():
            for modo, fun in [("CV", lad_cv), ("desplegable", lad_causal)]:
                res = {}
                for etiqueta, reemplazo in [("orig", "Ridge-directo"), ("MI20", "Ridge-MI20"),
                                            ("LASSO6", "Ridge-LASSO6")]:
                    cc = [reemplazo if c == "Ridge-directo" else c for c in cols_v]
                    P, _ = fun(sub, cc)
                    res[etiqueta] = P
                ok = ~np.isnan(res["orig"])
                maes = {k: np.abs(y[ok] - p[ok]).mean() for k, p in res.items()}
                txt = []
                for k in ["MI20", "LASSO6"]:
                    t, pv = hac_p(np.abs(y[ok] - res["orig"][ok]) - np.abs(y[ok] - res[k][ok]))
                    marca = ("MEJORA" if t > 0 else "empeora") if pv < 0.05 else "n.s."
                    txt.append(f"{maes[k]-maes['orig']:+.2f} p={pv:.3f} {marca}")
                print(f"{nombre:8s} {cn[:22]:22s} {modo:12s} {maes['orig']:11.2f} {maes['MI20']:11.2f} "
                      f"{maes['LASSO6']:13.2f}   " + " | ".join(txt))
                filas.append({"tramo": nombre, "conjunto": cn, "modo": modo,
                              "mae_ridge_orig": maes["orig"], "mae_ridge_mi20": maes["MI20"],
                              "mae_ridge_lasso6": maes["LASSO6"]})

    pd.DataFrame(filas).to_csv(RES / "ensamble_72h_ridge_reducido.csv", index=False)
    print(f"\nGuardado: ensamble_72h_ridge_reducido.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
