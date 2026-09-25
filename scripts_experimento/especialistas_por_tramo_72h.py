# -*- coding: utf-8 -*-
"""
Llevar la leccion del puente a los tramos 25-48h y 49-72h: especializar por tramo.

Lo que acaba de quedar demostrado en el tramo 1-24h: el producto de 72h perdia 2.3 puntos de MAPE
por usar modelos GENERALISTAS (entrenados para cubrir los 72 pasos a la vez) donde existia un canal
ESPECIALISTA para ese rango. El puente al ensamble de 24h bajo ese tramo de 13.00% a 10.84%.

La pregunta natural: los tramos 25-48h y 49-72h sufren lo mismo? Ahi no existe un canal day-ahead
que se pueda reutilizar, asi que la especializacion hay que construirla. Dos vias, ninguna probada:

  (1) ARBOLES ESPECIALISTAS POR TRAMO. Los votantes de arbol del canal de 72h (XGB-directo,
      CatB-directo, Ridge-directo, CatB-denso) se entrenaron con los 72 pasos juntos, recibiendo el
      paso como una variable mas. Eso obliga a un solo juego de particiones a servir horizontes con
      relaciones muy distintas: a 1-24h manda el precio del corte, a 49-72h mandan el embalse y el
      calendario. Aqui se reentrena cada familia usando SOLO los pasos de su tramo. Si la
      especializacion es el mecanismo real, deberian ganarle a su version generalista.

  (2) PESOS POR PASO EN VEZ DE POR TRAMO. El combinador fija un juego de pesos para las 24 horas de
      cada tramo, pero el error crece de forma monotona dentro del tramo (a 25h no se parece a 48h).
      Se prueba un juego de pesos por paso de horizonte y una version intermedia por bloques de 6
      pasos. En 24h la granularidad por hora del dia ya estaba agotada, pero ahi el eje era la hora
      del dia, no la distancia al corte, que es un eje distinto y no se ha probado.

Se cierra con las combinaciones de ambas vias, DM contra el vigente, verificacion de 10 particiones
y modo desplegable con pesos causales.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES
from xgboost_directo_72h import construir_largo, COLS_ESTADO, COLS_FUTURO, PASO_MUESTREO_TRAIN
from ensamble_72h_diario import hac_p
from puente_24h_a_72h import mape, smape, dm_metrica, lad_cv, causal

CORTE = pd.Timestamp("2026-01-01")
PRIMER_CORTE = pd.Timestamp("2026-01-01 00:00")
N_WIN, H = 214, 72
NUEVO_72 = ["LEAR", "NBX", "NBX_exog", "NaiveEstac"]
TRAMOS_LEJANOS = [("25-48h", 25, 48), ("49-72h", 49, 72)]
SEMILLAS = list(range(10))


def lad_cv_por_grupo(sub, cols, grupo, modo="lad", semilla=42):
    """Como lad_cv pero con un juego de pesos independiente por valor de `grupo`."""
    P = np.full(len(sub), np.nan)
    g = sub[grupo].to_numpy()
    for gg in np.unique(g):
        m = g == gg
        s = sub.loc[m]
        if s["dia"].nunique() < 10:
            continue
        P[m] = lad_cv(s.reset_index(drop=True), cols, modo, semilla)
    return P


def entrenar_especialistas(df, idx_train, idx_test, ini, fin):
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    pasos = np.arange(ini, fin + 1)
    tr = construir_largo(df, idx_train, pasos).dropna()
    te = construir_largo(df, idx_test, pasos).dropna().copy()
    cols_x = COLS_ESTADO + COLS_FUTURO + ["precio_mismo_hora_disp", "paso_horas"]
    print(f"    train {len(tr):,} filas | test {len(te):,} filas", flush=True)

    out = te[["fecha_hora", "cutoff", "paso_horas"]].copy()
    out["paso_horas"] = out["paso_horas"].astype(int)

    cat = CatBoostRegressor(n_estimators=2000, max_depth=8, learning_rate=0.05,
                            loss_function="MAE", random_seed=42, verbose=0, thread_count=-1)
    cat.fit(tr[cols_x], tr["y"])
    out["CatB_esp"] = cat.predict(te[cols_x])

    xgb = XGBRegressor(n_estimators=1200, max_depth=7, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, objective="reg:absoluteerror", random_state=42,
                       n_jobs=-1, tree_method="hist")
    xgb.fit(tr[cols_x], tr["y"])
    out["XGB_esp"] = xgb.predict(te[cols_x])

    esc = StandardScaler().fit(tr[cols_x])
    out["Ridge_esp"] = Ridge(alpha=10.0).fit(esc.transform(tr[cols_x]), tr["y"]).predict(
        esc.transform(te[cols_x]))
    return out


def main():
    t0 = time.time()
    df = cargar_completo().reset_index(drop=True)
    pos = pd.Series(df.index.values, index=df["fecha_hora"])
    idx_test = pos.reindex(pd.date_range(PRIMER_CORTE, periods=N_WIN, freq="24h")).astype(int).to_numpy()
    idx_fin = int(pos[CORTE])
    idx_train = np.arange(200, idx_fin - H, PASO_MUESTREO_TRAIN)
    print(f"Cortes: {len(idx_test)} prueba | {len(idx_train):,} entrenamiento\n")

    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    lear = pd.read_csv(RES / "pronostico_lear_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v = v.merge(lear[["fecha_hora", "cutoff", "paso_horas", "LEAR"]],
                on=["fecha_hora", "cutoff", "paso_horas"], how="inner")
    g = RES / "pronostico_garcht_72h_2026.csv"
    if g.exists():
        gg = pd.read_csv(g, parse_dates=["fecha_hora", "cutoff"])
        v = v.merge(gg[["fecha_hora", "cutoff", "paso_horas", "GARCHt72"]],
                    on=["fecha_hora", "cutoff", "paso_horas"], how="left")
    v["dia"] = v["fecha_hora"].dt.normalize()
    GEN = [c for c in ["XGB-directo", "CatB-directo", "Ridge-directo", "CatB-denso"] if c in v.columns]

    filas, guardar = [], []
    for nom, ini, fin in TRAMOS_LEJANOS:
        print(f"{'='*104}\nTRAMO {nom}\n{'='*104}")
        print("  (1) entrenando arboles especialistas de este tramo...")
        esp = entrenar_especialistas(df, idx_train, idx_test, ini, fin)
        ESP = ["CatB_esp", "XGB_esp", "Ridge_esp"]

        s = v[v.paso_horas.between(ini, fin)].merge(esp, on=["fecha_hora", "cutoff", "paso_horas"],
                                                    how="inner")
        s = s.dropna(subset=NUEVO_72 + ESP + ["real"]).sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
        s["bloque6"] = (s["paso_horas"] - ini) // 6
        y = s["real"].to_numpy()
        guardar.append(s[["fecha_hora", "cutoff", "paso_horas"] + ESP])
        print(f"    filas alineadas: {len(s):,}\n")

        print(f"  INDIVIDUALES (especialista contra su generalista)")
        print(f"    {'modelo':18s} {'MAE':>8s} {'MAPE %':>8s}")
        for c in NUEVO_72 + GEN + ESP:
            if c in s.columns and s[c].notna().all():
                print(f"    {c:18s} {np.abs(y-s[c]).mean():8.2f} {mape(y,s[c]):8.2f}")

        print(f"\n  ENSAMBLES (DM contra el vigente)")
        print(f"    {'configuracion':46s} {'MAE':>8s} {'MAPE %':>8s} {'sMAPE %':>8s}  "
              f"{'p(MAE)':>8s} {'p(MAPE)':>8s}")
        base = lad_cv(s, NUEVO_72, "lad")
        CONFS = [
            ("A  4 votantes, pesos por tramo (vigente)", NUEVO_72, "lad", None),
            ("B  4 votantes, pesos por PASO",            NUEVO_72, "lad", "paso_horas"),
            ("C  4 votantes, pesos por bloque de 6",     NUEVO_72, "lad", "bloque6"),
            ("D  + arboles especialistas, por tramo",    NUEVO_72 + ESP, "lad", None),
            ("E  + arboles especialistas, por PASO",     NUEVO_72 + ESP, "lad", "paso_horas"),
            ("F  + arboles especialistas, comb. sMAPE",  NUEVO_72 + ESP, "smape", None),
            ("G  + arboles generalistas (control)",      NUEVO_72 + GEN, "lad", None),
        ]
        preds = {}
        for etq, cols, modo, grupo in CONFS:
            if any(c not in s.columns or s[c].isna().any() for c in cols):
                continue
            pr = lad_cv(s, cols, modo) if grupo is None else lad_cv_por_grupo(s, cols, grupo, modo)
            ok = ~np.isnan(pr)
            preds[etq] = pr
            linea = (f"    {etq:46s} {np.abs(y[ok]-pr[ok]).mean():8.2f} {mape(y[ok],pr[ok]):8.2f} "
                     f"{smape(y[ok],pr[ok]):8.2f}")
            if not etq.startswith("A "):
                p1 = dm_metrica(y[ok], base[ok], pr[ok], "mae")[1]
                p2 = dm_metrica(y[ok], base[ok], pr[ok], "mape")[1]
                linea += f"  {p1:8.4f} {p2:8.4f}"
            print(linea)
            filas.append({"tramo": nom, "config": etq, "MAE": np.abs(y[ok]-pr[ok]).mean(),
                          "MAPE": mape(y[ok], pr[ok]), "sMAPE": smape(y[ok], pr[ok])})

        # ---- verificacion del mejor ----
        cand = [c for c in preds if not c.startswith("A ")]
        ganador = min(cand, key=lambda c: mape(y[~np.isnan(preds[c])], preds[c][~np.isnan(preds[c])]))
        cols_g, modo_g, grupo_g = next((c, m, gr) for e, c, m, gr in CONFS if e == ganador)
        print(f"\n  VERIFICACION 10 PARTICIONES: '{ganador}'")
        A, G = [], []
        for s_ in SEMILLAS:
            pa = lad_cv(s, NUEVO_72, "lad", s_)
            pg = lad_cv(s, cols_g, modo_g, s_) if grupo_g is None else lad_cv_por_grupo(s, cols_g, grupo_g, modo_g, s_)
            ok = ~np.isnan(pa) & ~np.isnan(pg)
            A.append([np.abs(y[ok]-pa[ok]).mean(), mape(y[ok], pa[ok]), smape(y[ok], pa[ok])])
            G.append([np.abs(y[ok]-pg[ok]).mean(), mape(y[ok], pg[ok]), smape(y[ok], pg[ok])])
        A, G = np.array(A), np.array(G)
        for j, mn in enumerate(["MAE", "MAPE %", "sMAPE %"]):
            dl = G[:, j] - A[:, j]
            print(f"    {mn:8s} A={A[:,j].mean():7.3f}+/-{A[:,j].std():.3f}  "
                  f"nuevo={G[:,j].mean():7.3f}+/-{G[:,j].std():.3f}  delta={dl.mean():+7.3f}, "
                  f"mejor en {int((dl<0).sum())}/{len(SEMILLAS)}")
        print(f"    VEREDICTO: {'SOLIDO' if all((G[:,j]-A[:,j]<0).all() for j in range(3)) else 'NO concluyente'}")

        ca = causal(s, NUEVO_72, "lad")
        cg = causal(s, cols_g, modo_g)
        ok = ~np.isnan(ca) & ~np.isnan(cg)
        print(f"    desplegable (causal 60d): A MAPE={mape(y[ok],ca[ok]):.2f}%  "
              f"nuevo MAPE={mape(y[ok],cg[ok]):.2f}%  DM p={dm_metrica(y[ok],ca[ok],cg[ok],'mape')[1]:.4f}\n")

    pd.DataFrame(filas).to_csv(RES / "especialistas_por_tramo_72h.csv", index=False)
    pd.concat(guardar, ignore_index=True).to_csv(RES / "pronostico_especialistas_tramo_72h_2026.csv", index=False)
    print(f"Guardado: especialistas_por_tramo_72h.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
