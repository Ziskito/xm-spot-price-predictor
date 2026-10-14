# -*- coding: utf-8 -*-
"""
Experimento del escenario realista: variantes del combinador causal (votantes del ensamble 7 realista).

Mismo esquema que escenario_realista_24h.causal_con_margen (solo dias con precio ya publicado, margen de 4 dias,
previa del origen 5 los primeros 14 dias), variando:
  * grupos: franjas de 6 h (referencia), hora a hora, franjas de 3 h y 12 h, un solo grupo
  * objetivo: sMAPE iterativo (referencia) o MAE
  * olvido: ponderacion exponencial por antiguedad (vida media de 30, 60, 120 dias)
y combinaciones fijas (media, mediana, media recortada).
Salida: resultados/escenario_realista/mejoras/combinador_metricas.csv
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
from combinador_optimo_mape import qra_pesos  # noqa: E402


def causal(d, previa, cols, grupo, margen=5, min_dias=14, iteraciones=3, objetivo="smape", vida=None):
    # margen = 5 desde el 7-oct (noche): con 4, el combinador entrenaba con la hora 00:00 de D-3 (ultima de la ventana
    # de D-4), que al corte de D aun no se publica. Efecto medido en realista_margen5.py: +0,04 a +0,10 de MAE.
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d[grupo].to_numpy(), d["dia"].to_numpy()
    Xp, yp, gp, diap = previa[cols].to_numpy(), previa["real"].to_numpy(), previa[grupo].to_numpy(), previa["dia"].to_numpy()
    dias = np.sort(np.unique(dia))
    P = np.full(len(d), np.nan)
    for i in range(len(dias)):
        conocidos = dias[:max(0, i - margen + 1)]
        tr_m, te_m = np.isin(dia, conocidos), dia == dias[i]
        for gg in np.unique(g[te_m]):
            tr, te = tr_m & (g == gg), te_m & (g == gg)
            Xt, yt, dt = X[tr], y[tr], dia[tr]
            if len(conocidos) < min_dias:
                Xt, yt = np.vstack([Xp[gp == gg], Xt]), np.concatenate([yp[gp == gg], yt])
                dt = np.concatenate([diap[gp == gg], dt])
            if len(yt) < len(cols) + 5:
                continue
            r = np.ones(len(yt)) if vida is None else 0.5 ** (((dias[i] - dt) / np.timedelta64(1, "D")) / vida)
            if objetivo == "mae":
                ww, b = qra_pesos(Xt, yt, r)
            else:
                w = r / np.maximum(yt, 1e-6)
                ww, b = qra_pesos(Xt, yt, w)
                for _ in range(iteraciones):
                    w = 2.0 * r / np.maximum(np.abs(yt) + np.abs(Xt @ ww + b), 1e-6)
                    ww, b = qra_pesos(Xt, yt, w)
            P[te] = X[te] @ ww + b
    return P


if __name__ == "__main__":
    t0 = time.time()
    from ensamble_24h_enero import origen5
    from ensamble_con_predespacho_24h import imar
    VOT = er.VOT6 + ["Predespacho"]
    pan = rc.referencia().dropna(subset=VOT + ["real"]).copy()
    pan["dia"] = (pan.index - pd.Timedelta(hours=1)).normalize()
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    previa = previa.dropna(subset=["Predespacho"])
    for p in (pan, previa):
        h = p.index.hour
        p["g24"], p["g8"], p["g4"], p["g2"], p["g1"] = h, h // 3, h // 6, h // 12, 0
    variantes = {
        "franjas 3 h": dict(grupo="g8"), "franjas 12 h": dict(grupo="g2"), "un grupo": dict(grupo="g1"),
        "hora a hora": dict(grupo="g24"), "objetivo MAE": dict(grupo="g4", objetivo="mae"),
        "olvido 30 d": dict(grupo="g4", vida=30), "olvido 60 d": dict(grupo="g4", vida=60), "olvido 120 d": dict(grupo="g4", vida=120),
        "referencia recalculada": dict(grupo="g4"),
    }
    for nom, kw in variantes.items():
        pan[nom] = causal(pan, previa, VOT, **kw)
        print(f"  {nom} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pan["media 7"] = pan[VOT].mean(axis=1)
    pan["mediana 7"] = pan[VOT].median(axis=1)
    pan["media recortada 7"] = pan[VOT].apply(lambda r: np.sort(r)[1:-1].mean(), axis=1, raw=True)
    pan["mitad ens6 + mitad predespacho"] = 0.5 * pan["ens6_realista"] + 0.5 * pan["Predespacho"]
    pan.to_csv(rc.MEJ / "combinador_predicciones.csv")
    ev = pan[pan.index >= "2026-01-15"].dropna(subset=list(variantes))
    filas = []
    for c in ["ens7_realista"] + list(variantes) + ["media 7", "mediana 7", "media recortada 7", "mitad ens6 + mitad predespacho"]:
        f = {"variante": c, **rc.metricas(ev["real"], ev[c])}
        if c != "ens7_realista":
            f["DM_t"], f["DM_p"] = er.dm(ev["real"].to_numpy(), ev[c].to_numpy(), ev["ens7_realista"].to_numpy())
        filas.append(f)
    t = pd.DataFrame(filas).sort_values("MAE")
    t.to_csv(rc.MEJ / "combinador_metricas.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
