# -*- coding: utf-8 -*-
"""
Ensamble de 72h sobre la evaluacion con cortes DIARIOS (214 ventanas, evaluacion_72h_diaria.py).
Mismo metodo que v3 (LAD por tramo de horizonte), con los ajustes que exigen las ventanas solapadas:
  - version desplegable: en el corte i solo se usan filas cuyo instante objetivo ya ocurrio (<= corte i);
    los pesos se actualizan cada 7 cortes (semanal) para acotar el costo;
  - calibracion conforme POR CORTE (qra_bandas.margen_por_corte): con ventanas solapadas el metodo por
    dia objetivo si usaria scores posteriores al corte.
Arma tambien el contrato unificado (v4 de 24h en los pasos 1-24; los cortes coinciden a las 00:00) y
un contrato para OE3 con ventanas NO solapadas (uno de cada 3 cortes), mismo formato que el actual.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import pesos_lad
from qra_bandas import margen_por_corte, metricas_banda

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
TRAMOS = [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]
CONJ = {
    "v3 (mismos votantes)": ["NBX", "NBX_exog", "XGB-directo", "CatB-directo", "CatB-denso", "Ridge-directo", "NaiveEstac"],
    "v3 + semillas + contexto 336h": ["NBX_semillas", "NBX_in336", "NBX_exog", "XGB-directo", "CatB-directo",
                                      "CatB-denso", "Ridge-directo", "NaiveEstac"],
}
CON_BANDAS = ["NBX", "NBX_exog", "NBX_semillas", "NBX_in336"]


def hac_p(d, maxlags=71):
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def lad_cv(sub, cols):
    X, y = sub[cols].to_numpy(), sub["real"].to_numpy()
    P, W = np.full(len(sub), np.nan), np.full((len(sub), len(cols)), np.nan)
    dias = np.sort(sub["dia"].unique())
    for dt in np.array_split(np.random.default_rng(42).permutation(dias), 5):
        te = sub["dia"].isin(dt).to_numpy()
        w = pesos_lad(X[~te], y[~te])
        P[te], W[te] = X[te] @ w, w
    return P, W


def lad_causal(sub, cols, cada=7, min_filas=15 * 24):
    X, y = sub[cols].to_numpy(), sub["real"].to_numpy()
    obj, cor = sub["fecha_hora"].to_numpy(), sub["cutoff"].to_numpy()
    P, W = np.full(len(sub), np.nan), np.full((len(sub), len(cols)), np.nan)
    w = None
    for i, c in enumerate(np.sort(np.unique(cor))):
        tr = obj <= c
        if tr.sum() < min_filas:
            continue
        if w is None or i % cada == 0:
            w = pesos_lad(X[tr], y[tr])
        te = cor == c
        P[te], W[te] = X[te] @ w, w
    return P, W


def bandas(sub, cols, P, W):
    """Bandas del ensamble = mismos pesos LAD del punto, renormalizados sobre el subconjunto de
    modelos con cuantiles propios (CON_BANDAS). Si el optimizador le dio peso 0 a TODOS los modelos
    con bandas en una fila (el punto se armo solo con arboles/lineal, que no traen cuantiles), no
    hay de donde sacar la banda -- se usa peso igual entre los modelos con bandas solo para ESE
    calculo, sin tocar el pronostico puntual (que sigue siendo el de la mezcla completa)."""
    idx = [i for i, c in enumerate(cols) if c in CON_BANDAS]
    wb = W[:, idx]
    suma = wb.sum(1, keepdims=True)
    sin_banda = (suma == 0) | np.isnan(suma)
    wb = np.divide(wb, suma, out=np.full_like(wb, 1 / len(idx)), where=~sin_banda)
    q10 = sum(sub[f"{cols[i]}__q10"].to_numpy() * wb[:, j] for j, i in enumerate(idx))
    q90 = sum(sub[f"{cols[i]}__q90"].to_numpy() * wb[:, j] for j, i in enumerate(idx))
    semi = (q90 - q10) / 2
    l, u = P - semi, P + semi
    y = sub["real"].to_numpy()
    ok = ~np.isnan(P)
    m = np.full(len(P), np.nan)
    m[ok] = margen_por_corte(sub["fecha_hora"].to_numpy()[ok], sub["cutoff"].to_numpy()[ok],
                             np.maximum(l - y, y - u)[ok])
    return np.minimum(l - m, P), np.maximum(u + m, P), m


def main():
    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v["NBX_semillas"] = v[["NBX", "NBX_s7", "NBX_s123"]].mean(axis=1)
    v["NBX_semillas__q10"] = v[["NBX__q10", "NBX_s7__q10", "NBX_s123__q10"]].mean(axis=1)
    v["NBX_semillas__q90"] = v[["NBX__q90", "NBX_s7__q90", "NBX_s123__q90"]].mean(axis=1)
    v["dia"] = v["fecha_hora"].dt.normalize()
    v4 = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    b24 = pd.read_csv(RES / "bandas_24h_qra_causal_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    print(f"Filas: {len(v):,} | cortes: {v['cutoff'].nunique()} ({v['cutoff'].min()} .. {v['cutoff'].max()})")

    filas, partes = [], []
    for nombre, ini, fin in TRAMOS:
        sub = v[v["paso_horas"].between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
        y = sub["real"].to_numpy()
        base = sub["NBX"].to_numpy()
        mae_b = np.abs(y - base).mean()
        print(f"\n--- Tramo {nombre} ({len(sub):,} filas, {sub['cutoff'].nunique()} cortes) --- N-BEATSx: MAE={mae_b:.2f}")
        maes = {c: np.abs(y - sub[c]).mean() for c in sorted({c for cs in CONJ.values() for c in cs})}
        print("  votantes: " + " | ".join(f"{c}={m:.1f}" for c, m in maes.items()))
        salida = sub[["fecha_hora", "cutoff", "paso_horas", "real"]].copy()
        for cn, cols in CONJ.items():
            for modo, fun in [("CV", lad_cv), ("desplegable", lad_causal)]:
                P, W = fun(sub, cols)
                ok = ~np.isnan(P)
                t, p = hac_p(np.abs(y[ok] - base[ok]) - np.abs(y[ok] - P[ok]))
                mae = np.abs(y[ok] - P[ok]).mean()
                mae_bo = np.abs(y[ok] - base[ok]).mean()
                l, u, _ = bandas(sub, cols, P, W)
                ev = ok & (sub["fecha_hora"] >= sub["fecha_hora"].min() + pd.Timedelta(days=30)).to_numpy()
                cob, ancho, isc = metricas_banda(y[ev], l[ev], u[ev])
                marca = "GANA" if p < 0.05 and t > 0 else ("pierde" if p < 0.05 else "n.s.")
                print(f"  {cn:30s} {modo:11s} MAE={mae:6.2f} ({100*(mae_bo-mae)/mae_bo:+5.1f}% vs N-BEATSx, p={p:.4f} {marca})"
                      f"  bandas: {cob:4.1f}% ancho={ancho:5.1f} IS={isc.mean():5.1f}")
                filas.append({"tramo": nombre, "conjunto": cn, "modo": modo, "mae": mae, "mae_nbeatsx": mae_bo,
                              "p_vs_nbeatsx": p, "cobertura": cob, "ancho": ancho, "interval_score": isc.mean()})
                tag = f"{'v3' if cn.startswith('v3 (') else 'plus'}_{'cv' if modo == 'CV' else 'dsp'}"
                salida[f"q50_{tag}"], salida[f"q10_{tag}"], salida[f"q90_{tag}"] = P, l, u
        partes.append(salida)

    todo = pd.concat(partes, ignore_index=True).sort_values(["cutoff", "paso_horas"])
    todo.to_csv(RES / "pronostico_ensamble_72h_diario_2026.csv", index=False)
    pd.DataFrame(filas).to_csv(RES / "ensamble_72h_diario_metricas.csv", index=False)

    # -------- contrato unificado: v4 (24h) en los pasos 1-24 --------
    print("\n=== Contrato unificado (v4 de 24h en los pasos 1-24) ===")
    mejor = min(["v3_cv", "plus_cv"], key=lambda k: np.nanmean(np.abs(todo["real"] - todo[f"q50_{k}"])))
    mejor_d = mejor.replace("_cv", "_dsp")
    print(f"Conjunto elegido para los pasos 25-72: {mejor} (desplegable: {mejor_d})")
    uni = todo[["fecha_hora", "cutoff", "paso_horas", "real"]].copy()
    t1 = (uni["paso_horas"] <= 24).to_numpy()
    fh = uni.loc[t1, "fecha_hora"]
    for sufijo, fuente, q50, q10, q90 in [
        ("cv", mejor, b24["q50"], b24["q10"], b24["q90"]),
        ("dsp", mejor_d, v4["pred_causal"], v4["q10_causal"], v4["q90_causal"]),
    ]:
        uni[f"q50_{sufijo}"] = todo[f"q50_{fuente}"].to_numpy()
        uni[f"q10_{sufijo}"] = todo[f"q10_{fuente}"].to_numpy()
        uni[f"q90_{sufijo}"] = todo[f"q90_{fuente}"].to_numpy()
        uni.loc[t1, f"q50_{sufijo}"] = q50.reindex(fh).to_numpy()
        uni.loc[t1, f"q10_{sufijo}"] = q10.reindex(fh).to_numpy()
        uni.loc[t1, f"q90_{sufijo}"] = q90.reindex(fh).to_numpy()
    for nombre, ini, fin in TRAMOS + [("global", 1, 72)]:
        s = uni[uni["paso_horas"].between(ini, fin)]
        s72 = todo[todo["paso_horas"].between(ini, fin)]
        lin = [f"{nombre:7s}"]
        for suf in ["cv", "dsp"]:
            ok = s[f"q50_{suf}"].notna()
            e = np.abs(s.loc[ok, "real"] - s.loc[ok, f"q50_{suf}"])
            lin.append(f"{'CV' if suf == 'cv' else 'desplegable'}: MAE={e.mean():6.2f} MAPE={(e/s.loc[ok,'real']).mean()*100:5.2f}%")
        lin.append(f"| N-BEATSx solo: {np.abs(s72['real'] - s72['NBX'] if 'NBX' in s72 else np.nan).mean() if 'NBX' in s72 else ''}")
        print("  " + "  ".join(lin[:3]))
    uni.to_csv(RES / "pronostico_unificado_72h_diario_2026.csv", index=False)

    # -------- contrato para OE3: ventanas no solapadas (1 de cada 3 cortes) --------
    cortes = np.sort(uni["cutoff"].unique())
    sel = uni[uni["cutoff"].isin(cortes[::3])].copy()
    contrato = sel[["fecha_hora", "cutoff", "paso_horas", "real"]].copy()
    contrato["q50"], contrato["q10"], contrato["q90"] = sel["q50_cv"], sel["q10_cv"], sel["q90_cv"]
    contrato["margen"] = np.nan
    contrato = contrato[["fecha_hora", "cutoff", "paso_horas", "real", "q50", "margen", "q10", "q90"]]
    contrato.to_csv(RES / "pronostico_unificado_72h_v2_2026.csv", index=False)
    sys.path.insert(0, str(RAIZ / "src"))
    import motor_decision as md
    md.validar_contrato_pronostico(contrato, nombre_archivo="pronostico_unificado_72h_v2_2026.csv")
    print(f"\nContrato OE3 (ventanas no solapadas, cortes 00:00): {len(contrato):,} filas, "
          f"fecha_hora unica={contrato['fecha_hora'].is_unique}, cortes={contrato['cutoff'].nunique()}")


if __name__ == "__main__":
    main()
