# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: candidato final del CatBoost con el predespacho por dentro.

CatBoost (perdida MAE, objetivo log(precio/predespacho)) con R+P+G, entrenado desde 2021, 3000 iteraciones con tasa
0.03, reentrenado cada 7 dias con lo publicado (hasta 73 h antes del corte), promedio de las semillas dadas.
Se combina con el predespacho en el combinador causal realista (objetivo MAE, franjas de 6 h).
Uso: realista_exp_candidato.py [semillas separadas por coma] (por defecto 42)
Salida: resultados/escenario_realista/mejoras/candidato_predicciones.csv y candidato_metricas.csv
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
import escenario_realista_24h as er  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402

INI_TR = "2021-01-01"
MARGEN = pd.Timedelta(hours=er.OCULTO)


def modelo(tr, te, cols, semillas):
    from catboost import CatBoostRegressor
    z = 0
    for s in semillas:
        m = CatBoostRegressor(iterations=3000, depth=6, learning_rate=0.03, loss_function="MAE", random_seed=s, verbose=0, thread_count=6)
        m.fit(tr[cols], np.log(tr["precio_bolsa"]) - np.log(tr["pre_obj"].clip(lower=1)))
        z = z + m.predict(te[cols])
    return pd.Series(np.exp(z / len(semillas) + np.log(te["pre_obj"].clip(lower=1))), index=te.index)


def limpio(x):
    return x.dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])


if __name__ == "__main__":
    t0 = time.time()
    semillas = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "42").split(",")]
    d = rc.tabla_realista().set_index("fecha_hora")
    cols = conjuntos(d)["R+P"] + [c for c in d.columns if c.startswith("prog_") or c == "n_termicas"]
    f = pd.read_csv(rc.MEJ / "programado_predicciones.csv", parse_dates=[0], index_col=0)
    pan = f[["real", "dia", "Predespacho", "ens7_realista", "CatBoost_semanal G", "Pre+Cat semanal · mae"]].copy()
    pan["dia"] = pd.to_datetime(pan["dia"])
    fin = pan.index.max()
    inicios = list(pd.date_range("2026-01-01", fin, freq="7D"))
    partes = []
    for k, ini in enumerate(inicios):
        hasta = inicios[k + 1] if k + 1 < len(inicios) else fin
        te = d.loc[ini + pd.Timedelta(hours=1):hasta].dropna(subset=["pre_obj"])
        partes.append(modelo(limpio(d.loc[INI_TR:ini - MARGEN]), te, cols, semillas))
    pan["Candidato"] = pd.concat(partes).reindex(pan.index)
    print(f"  candidato semanal ({(time.time() - t0) / 60:.1f} min)", flush=True)
    from ensamble_24h_enero import origen5
    from ensamble_con_predespacho_24h import imar
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    previa["Candidato"] = modelo(limpio(d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN]),
                                 d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"]), cols, semillas).reindex(previa.index)
    previa = previa.dropna(subset=["Predespacho", "Candidato"])
    for p in (pan, previa):
        p["g4"] = p.index.hour // 6
    pan["Pre+Candidato · mae"] = causal(pan, previa, ["Predespacho", "Candidato"], "g4", objetivo="mae")
    pan["Pre+Candidato · smape"] = causal(pan, previa, ["Predespacho", "Candidato"], "g4", objetivo="smape")
    sufijo = "_".join(map(str, semillas))
    pan.to_csv(rc.MEJ / f"candidato_predicciones_{sufijo}.csv")
    ev = pan[pan.index >= "2026-01-15"].dropna()
    filas = []
    for c in ["ens7_realista", "Pre+Cat semanal · mae", "CatBoost_semanal G", "Candidato", "Pre+Candidato · mae", "Pre+Candidato · smape"]:
        fl = {"variante": c, **rc.metricas(ev["real"], ev[c])}
        fl["DM_t_vs_ens7"], fl["DM_p"] = er.dm(ev["real"].to_numpy(), ev[c].to_numpy(), ev["ens7_realista"].to_numpy())
        filas.append(fl)
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / f"candidato_metricas_{sufijo}.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    mes = ev.index.to_period("M")
    print(pd.concat({"MAE": pd.DataFrame({c: (ev[c] - ev["real"]).abs().groupby(mes).mean() for c in t["variante"]}),
                     "sesgo": pd.DataFrame({c: (ev[c] - ev["real"]).groupby(mes).mean() for c in t["variante"]})}, axis=1).round(1).T.to_string())
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
