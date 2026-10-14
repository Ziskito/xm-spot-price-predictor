# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: LEAR realista (un LASSO por hora, al estilo Lago et al. 2021) con el predespacho.

Variables de realista_comun.tabla_realista (solo lo publicado a las 00:00 de D), transformacion asinh del precio
como en el LEAR del proyecto, y dos objetivos: asinh(precio) y log(precio/predespacho). Entrena 2019-02-15 a 2025-12-31,
prueba 2026-01-15 a 2026-08-05.
Salida: resultados/escenario_realista/mejoras/lineal_predicciones.csv y lineal_metricas.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402

INI_TR, FIN_TR = "2019-02-15", "2025-12-31 23:00"
INI_TE, FIN_TE = "2026-01-15", "2026-08-05 00:00"

if __name__ == "__main__":
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    d = rc.tabla_realista().set_index("fecha_hora")
    # en escala log las variables de precio entran mejor en un modelo lineal
    for c in [c for c in d.columns if (c.startswith("precio_") or c.startswith("h_precio_") or c.startswith("pre_")) and "std" not in c and c != "precio_bolsa"
              and "rel" not in c and "cambio" not in c and "ratio" not in c and "rango" not in c]:
        d[f"asinh_{c}"] = np.arcsinh(d[c])
    sets = conjuntos(d)
    extra = [c for c in d.columns if c.startswith("asinh_")]
    sets = {"R+P": sets["R+P"] + extra, "R+P+D": sets["R+P+D"] + extra}
    tr = d.loc[INI_TR:FIN_TR].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    te = d.loc[INI_TE:FIN_TE].dropna(subset=["precio_bolsa", "pre_obj"])
    preds = pd.DataFrame({"real": te["precio_bolsa"]})
    for nom, cols in sets.items():
        for objetivo in ("asinh precio", "log precio/predespacho"):
            p = pd.Series(np.nan, index=te.index)
            for h in range(24):
                a, b = tr[tr.index.hour == h], te[te.index.hour == h]
                Xa, Xb = a[cols].replace([np.inf, -np.inf], np.nan), b[cols].replace([np.inf, -np.inf], np.nan)
                med = Xa.median()
                Xa, Xb = Xa.fillna(med), Xb.fillna(med)
                esc = StandardScaler().fit(Xa)
                if "/" in objetivo:
                    y = np.log(a["precio_bolsa"]) - np.log(a["pre_obj"].clip(lower=1))
                else:
                    y = np.arcsinh(a["precio_bolsa"])
                m = LassoCV(cv=5, random_state=42, n_jobs=4, max_iter=20000).fit(esc.transform(Xa), y)
                z = m.predict(esc.transform(Xb))
                p[b.index] = np.exp(z) * b["pre_obj"].clip(lower=1) if "/" in objetivo else np.sinh(z)
            clave = f"LEAR realista · {nom} · {objetivo}"
            preds[clave] = p
            print(f"  {clave:50s} MAE {np.abs(p - te['precio_bolsa']).mean():6.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    preds.to_csv(rc.MEJ / "lineal_predicciones.csv")
    ref = rc.referencia()
    ev = preds.index.intersection(ref.dropna(subset=["ens7_realista"]).index)
    filas = [{"variante": c, **rc.metricas(preds.loc[ev, "real"], preds.loc[ev, c])} for c in preds.columns if c != "real"]
    filas.append({"variante": "REFERENCIA ensamble 7 realista", **rc.metricas(preds.loc[ev, "real"], ref.loc[ev, "ens7_realista"])})
    filas.append({"variante": "REFERENCIA predespacho solo", **rc.metricas(preds.loc[ev, "real"], ref.loc[ev, "Predespacho"])})
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / "lineal_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
