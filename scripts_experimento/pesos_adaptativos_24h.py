# -*- coding: utf-8 -*-
"""
Pesos ADAPTATIVOS (agregacion de expertos en linea) para el ensamble de 24h.

De donde sale la idea: techo_combinacion_24h.py midio los techos con oraculo y encontro que
  - pesos optimos POR HORA DEL DIA dan MAPE 11.38%, o sea NO mejoran el 11.27% actual (agotado);
  - pesos optimos POR DIA dan 8.82%, es decir 2.4 puntos de margen.
Conclusion: la mezcla optima de votantes CAMBIA DIA A DIA, y el esquema vigente (pesos fijos por
franja, ajustados sobre todo el historico) no sigue ese cambio. La correlacion de errores entre
votantes es alta (0.78-0.83) pero no 1.0, y en 64% de las horas el precio real cae DENTRO del
rango de los votantes: hay diversidad explotable.

Ese techo de 8.82% usa informacion futura y no es alcanzable, pero la literatura de agregacion
adaptativa de expertos (Cesa-Bianchi & Lugosi; en pronostico de precios, la familia de metodos
"online expert aggregation") ofrece esquemas CAUSALES que persiguen ese optimo movil. Se prueban:

  A) VENTANAS MOVILES CORTAS de LAD: re-ajustar los pesos con solo los ultimos N dias.
  B) HEDGE / promedio exponencialmente ponderado: peso_i proporcional a exp(-eta * perdida_acum_i),
     con la perdida acumulada descontada, de modo que los votantes que vienen fallando pierden peso.
  C) HEDGE CON DESCUENTO (olvido geometrico): la perdida vieja se descuenta por gamma cada dia.
  D) Combinacion: ventana movil corta por franja horaria.

Todo estrictamente causal: los pesos del dia D se calculan solo con dias < D.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import cargar as cargar24, BASE5, pesos_lad
from o6_comun import RES

PICO = [18, 19, 20]
MIN_DIAS = 21


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def lad_ventana(d, cols, ventana=None, grupo=None):
    """LAD causal: para cada dia, pesos ajustados con los dias anteriores (todos, o los ultimos N)."""
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    dia = d["dia"].to_numpy()
    dias = np.sort(np.unique(dia))
    g = d[grupo].to_numpy() if grupo else np.zeros(len(d), int)
    P = np.full(len(d), np.nan)
    for i, dd in enumerate(dias):
        if i < MIN_DIAS:
            continue
        prev = dias[max(0, i - ventana):i] if ventana else dias[:i]
        tr_m, te_m = np.isin(dia, prev), dia == dd
        for gg in np.unique(g[te_m]):
            tr, te = tr_m & (g == gg), te_m & (g == gg)
            if tr.sum() < len(cols) + 5:
                tr = tr_m
            if tr.sum() < len(cols) + 5:
                continue
            P[te] = X[te] @ pesos_lad(X[tr], y[tr])
    return P


def hedge(d, cols, eta, gamma=1.0):
    """Promedio exponencialmente ponderado con descuento.
    peso_i ∝ exp(-eta * L_i), L_i = perdida absoluta acumulada y descontada del votante i.
    gamma<1 olvida el pasado: L <- gamma*L + perdida_del_dia."""
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    dia = d["dia"].to_numpy()
    dias = np.sort(np.unique(dia))
    L = np.zeros(len(cols))
    P = np.full(len(d), np.nan)
    for i, dd in enumerate(dias):
        te = dia == dd
        if i >= 1:
            w = np.exp(-eta * (L - L.min()))
            w = w / w.sum()
            P[te] = X[te] @ w
        # actualizar perdidas con el dia que acaba de ocurrir (solo despues de predecirlo)
        perd = np.abs(y[te][:, None] - X[te]).mean(0)
        L = gamma * L + perd / max(np.abs(y[te]).mean(), 1e-9)   # normalizada para que eta sea comparable
    return P


def main():
    d = cargar24().sort_index()
    lear = pd.read_csv(RES / "pronostico_lear_24h_2026.csv", parse_dates=["fecha_hora"])
    d = d.join(lear.set_index("fecha_hora")["LEAR24"], how="left")
    d["dia"] = d.index.normalize()
    d["hora"] = d.index.hour
    d["g4"] = d.index.hour // 6
    cols = [c for c in BASE5 + ["LEAR24", "CatBoost", "MarkovSw"] if c in d.columns]
    d = d.dropna(subset=cols + ["real"])
    y = d["real"].to_numpy()
    pk = d["hora"].isin(PICO).to_numpy()
    print(f"Filas: {len(d):,} | votantes ({len(cols)}): {cols}\n")

    # referencia: pesos causales con ventana expansiva (lo mas parecido al v4 desplegable)
    ref = lad_ventana(d, BASE5, None, "g4")
    ok = ~np.isnan(ref)
    print(f"REFERENCIA (LAD causal expansivo por franja, 5 votantes): "
          f"MAE={np.abs(y[ok]-ref[ok]).mean():.2f}  MAPE={mape(y[ok], ref[ok]):.2f}%  "
          f"pico {mape(y[ok&pk], ref[ok&pk]):.2f}%\n")

    resultados = {}
    print(f"{'='*100}\nA) VENTANAS MOVILES CORTAS DE LAD (causal)\n{'='*100}")
    print(f"{'esquema':44s} {'MAE':>8s} {'MAPE %':>8s} {'pico %':>8s}   DM vs referencia")
    for votantes, etiqueta_v in [(BASE5, "5 votantes"), (cols, f"{len(cols)} votantes")]:
        for ventana in [None, 90, 60, 30, 21, 14, 7]:
            for grupo in ["g4", None]:
                p = lad_ventana(d, votantes, ventana, grupo)
                oo = ok & ~np.isnan(p)
                et = (f"{etiqueta_v}, ventana "
                      f"{'expansiva' if ventana is None else f'{ventana}d':>9s}"
                      f", {'por franja' if grupo else 'global   '}")
                t, pv = dm(y[oo], ref[oo], p[oo])
                marca = ("MEJORA" if t > 0 else "peor") if pv < 0.05 else "n.s."
                print(f"{et:44s} {np.abs(y[oo]-p[oo]).mean():8.2f} {mape(y[oo], p[oo]):8.2f} "
                      f"{mape(y[oo&pk], p[oo&pk]):8.2f}   p={pv:.4f} {marca}")
                resultados[et] = p

    print(f"\n{'='*100}\nB/C) HEDGE (promedio exponencialmente ponderado) con y sin descuento\n{'='*100}")
    print(f"{'esquema':44s} {'MAE':>8s} {'MAPE %':>8s} {'pico %':>8s}   DM vs referencia")
    for eta in [1, 5, 20, 50]:
        for gamma in [1.0, 0.95, 0.85]:
            p = hedge(d, cols, eta, gamma)
            oo = ok & ~np.isnan(p)
            et = f"Hedge eta={eta:<3d} gamma={gamma}"
            t, pv = dm(y[oo], ref[oo], p[oo])
            marca = ("MEJORA" if t > 0 else "peor") if pv < 0.05 else "n.s."
            print(f"{et:44s} {np.abs(y[oo]-p[oo]).mean():8.2f} {mape(y[oo], p[oo]):8.2f} "
                  f"{mape(y[oo&pk], p[oo&pk]):8.2f}   p={pv:.4f} {marca}")
            resultados[et] = p

    mejor = min(resultados, key=lambda k: mape(y[ok & ~np.isnan(resultados[k])],
                                               resultados[k][ok & ~np.isnan(resultados[k])]))
    pm = resultados[mejor]
    oo = ok & ~np.isnan(pm)
    print(f"\n{'='*100}\nMEJOR ESQUEMA: {mejor}")
    print(f"  MAE={np.abs(y[oo]-pm[oo]).mean():.2f}  MAPE={mape(y[oo], pm[oo]):.2f}%  "
          f"pico={mape(y[oo&pk], pm[oo&pk]):.2f}%")
    t, pv = dm(y[oo], ref[oo], pm[oo])
    print(f"  DM contra la referencia: p={pv:.4f}")
    pd.DataFrame({"fecha_hora": d.index[oo], "real": y[oo], "pred": pm[oo]}).to_csv(
        RES / "pesos_adaptativos_24h.csv", index=False)
    print("\nGuardado: pesos_adaptativos_24h.csv")


if __name__ == "__main__":
    main()
