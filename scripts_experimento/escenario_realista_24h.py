# -*- coding: utf-8 -*-
"""
Escenario realista de disponibilidad de datos publicos (7-oct-2026).

Medicion del 7-oct-2026 (10:26) sobre la API publica de XM y NOAA: precio de bolsa, demanda y generacion
se publican con ~3 dias de rezago (a las 00:00 del dia D solo se conoce hasta las 23:00 de D-4); embalses y
aportes con ~1 dia (a las 00:00 de D se conoce el valor diario de D-2); el ONI con ~2 meses. El protocolo del
proyecto supone el precio de la ultima hora "disponible de inmediato" (bitacora del 24-sep), cierto quiza para
los agentes del mercado pero no para las fuentes publicas del Anexo 1. El predespacho ideal si llega antes del
dia (D-1, ~10:30).

Aqui se rehace todo el ensamble de 24 h de 2026 con solo lo publico a las 00:00 de D:
  * variables de precio y demanda: "hace 24 h" pasa a "hace 96 h" (120 h para la hora 00:00 del dia
    siguiente), y los promedios moviles se desplazan igual; precio_lag168h no cambia
  * embalses y aportes: valor diario de D-2 y sus derivados diarios; ONI(M-2) (auditoria_oni_causal.py)
  * Persistencia = mismo valor de hace 96 h (120 h para la hora 00:00)
  * XGBoost, ARX+GARCH: misma configuracion que el proyecto con las variables realistas
  * GARCH-ged: diseño LEAR con rezagos desde el corte >= 73 h
  * N-BEATSx / N-HiTS: horizonte de 97 pasos desde la ultima hora publicada (23:00 de D-4), se usan las
    ultimas 24 (01:00 de D a 00:00 de D+1)
  * combinador causal (QRA por franja de 6 h, sMAPE) que solo aprende de ventanas cuyo precio real ya estaba
    publicado (4 ventanas de margen); ensamble de 6 y de 7 (+ predespacho)
Se compara con el protocolo actual en las mismas horas.
Salidas: data/processed/resultados/escenario_realista/ (predicciones.csv, metricas.csv, resumen.txt)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import EXCLUIR, REGRESORAS_ARX, RAIZ, RES  # noqa: E402
from reentrenar_tras_fix_demanda import FUTR_EXOG_BIN, FUTR_EXOG_CONT, HIST_EXOG_DL  # noqa: E402
from lear_24h import EXOG_CORTE, EXOG_OBJ  # noqa: E402

OUT = RES / "escenario_realista"
CORTE = pd.Timestamp("2026-01-01")
FIN = pd.Timestamp("2026-08-05 00:00")
OCULTO = 73                                  # horas sin publicar antes del corte (precio y demanda)
VOT6 = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS", "GARCH-ged"]


def por_hora(serie, horas_dia, k=96):
    """Valor desplazado k horas (k+24 para la hora 00:00, ultima de la ventana)."""
    return np.where(horas_dia == 0, serie.shift(k + 24), serie.shift(k))


def features_realistas():
    m = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, parse_dates=["fecha_hora"])
                   for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]).drop_duplicates("fecha_hora")
    m = m.sort_values("fecha_hora").reset_index(drop=True)
    actual = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, parse_dates=["fecha_hora"])
                        for f in ("dataset_features_2019_2025.csv", "dataset_features_2026.csv")])
    d = m[["fecha_hora", "precio_bolsa"]].copy()
    hd = d["fecha_hora"].dt.hour.to_numpy()
    p, dem = m["precio_bolsa"], m["demanda"]
    d["precio_lag24h"] = por_hora(p, hd)
    d["precio_lag168h"] = p.shift(168)
    for nombre, ven in (("precio_media_24h", 24), ("precio_media_7d", 168), ("precio_media_30d", 720)):
        d[nombre] = por_hora(p.rolling(ven).mean(), hd)
    d["precio_std_24h"] = por_hora(p.rolling(24).std(), hd)
    d["precio_std_7d"] = por_hora(p.rolling(168).std(), hd)
    d["precio_rango_24h"] = por_hora(p.rolling(24).max() - p.rolling(24).min(), hd)
    d["ratio_volatilidad"] = d["precio_std_24h"] / d["precio_std_7d"]
    d["demanda_lag24h"] = por_hora(dem, hd)
    d["demanda_lag48h"] = por_hora(dem, hd, 120)
    d["demanda_lag72h"] = por_hora(dem, hd, 144)
    d["demanda_media_24h"] = por_hora(dem.rolling(24).mean(), hd)
    # crudas (las usa el diseño LEAR en el corte): lo publicado a esa hora
    d["demanda"], d["generacion"] = dem.shift(OCULTO), m["generacion"].shift(OCULTO)
    # embalses y aportes: valor diario de D-2 (D = dia del corte) y derivados diarios
    dia_corte = (d["fecha_hora"] - pd.Timedelta(hours=1)).dt.normalize()
    for col in ("volumen_embalses", "aportes_hidricos"):
        diario = m.groupby(m["fecha_hora"].dt.normalize())[col].mean()
        base = diario.shift(2)
        der = pd.DataFrame({col: base, f"{col}_delta_1d": base.diff(1), f"{col}_delta_7d": base.diff(7),
                            f"{col}_media_7d": base.rolling(7).mean(), f"{col}_media_30d": base.rolling(30).mean()})
        der[f"{col}_vs_media30d"] = der[col] / der[f"{col}_media_30d"]
        for c in der.columns:
            d[c] = der[c].reindex(dia_corte).to_numpy()
    # ONI(M-2): valor del mes de hace dos meses (ya publicado)
    oni_mes = m.groupby(m["fecha_hora"].dt.to_period("M"))["oni"].mean()
    d["oni"] = oni_mes.shift(2).reindex(d["fecha_hora"].dt.to_period("M")).to_numpy()
    # calendario y festivos: identicos al proyecto
    cal = [c for c in actual.columns if c not in d.columns]
    d = d.merge(actual[["fecha_hora"] + cal].drop_duplicates("fecha_hora"), on="fecha_hora", how="left")
    return d[actual.columns.tolist()].sort_values("fecha_hora").reset_index(drop=True)


def xgboost(df):
    import xgboost as xgb
    cols = [c for c in df.columns if c not in EXCLUIR]
    tr = df[df["fecha_hora"] < CORTE].dropna(subset=cols)
    te = df[(df["fecha_hora"] >= CORTE) & (df["fecha_hora"] <= FIN)]
    mdl = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, random_state=42)
    mdl.fit(tr[cols], np.log(tr["precio_bolsa"]))
    return pd.Series(np.exp(mdl.predict(te[cols])), index=te["fecha_hora"].to_numpy())


def arx_garch(df):
    from arch import arch_model
    cols = [c for c in df.columns if c not in EXCLUIR]
    tr = df[df["fecha_hora"] < CORTE].dropna(subset=cols).copy()
    te = df[(df["fecha_hora"] >= CORTE) & (df["fecha_hora"] <= FIN)].copy()
    cx = REGRESORAS_ARX + ["lp"]
    for x in (tr, te):
        x["lp"] = np.log(x["precio_lag24h"])
    yl = np.log(tr["precio_bolsa"])
    ym, ys = yl.mean(), yl.std()
    xm, xs = tr[cx].mean(), tr[cx].std().replace(0, 1)
    r = arch_model((yl - ym) / ys * 10, x=(tr[cx] - xm) / xs, mean="ARX", lags=0, vol="GARCH", p=1, q=1,
                   dist="normal").fit(disp="off", options={"maxiter": 500})
    pm = r.params[["Const"] + cx]
    pred = pm["Const"] + ((te[cx] - xm) / xs * pm[cx]).sum(axis=1)
    return pd.Series(np.exp(pred.to_numpy() / 10 * ys + ym), index=te["fecha_hora"].to_numpy())


REZ_CORTE = [73, 74, 75, 76, 77, 78, 84, 90, 96, 120, 144, 168]


def diseño(df, h, cortes):
    """Diseño LEAR con solo lo publicado al corte: rezagos de precio >= 73 h desde el corte."""
    n = len(df)
    precio = df["precio_bolsa"].to_numpy(np.float64)
    c = cortes[(cortes + h < n) & (cortes - 200 >= 0)]
    t = c + h
    col = [precio[c - k] for k in REZ_CORTE]
    for mm in (96, 120, 144, 168):
        if (t - mm <= c - OCULTO).all():
            col.append(precio[t - mm])
    ven = np.stack([precio[c - k] for k in range(OCULTO, OCULTO + 24)], axis=1)
    col += [ven.min(1), ven.max(1), ven.mean(1), ven.std(1)]
    for cc in EXOG_CORTE:
        col.append(df[cc].to_numpy(np.float64)[c])
    for cc in EXOG_OBJ:
        col.append(df[cc].to_numpy(np.float64)[t])
    X = np.column_stack(col)
    ok = ~np.isnan(X).any(1)
    return X[ok], precio[t][ok], df["fecha_hora"].to_numpy()[t][ok]


def garch_ged(df):
    from arch import arch_model
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler
    f = df["fecha_hora"]
    idx00 = np.where(f.dt.hour.to_numpy() == 0)[0]
    ic = int(np.where(f == CORTE)[0][0])
    ctr, cte = idx00[(idx00 >= 200) & (idx00 < ic - 24)], idx00[idx00 >= ic - 1]
    partes, alphas = [], []
    for h in range(1, 25):
        Xtr, ytr, _ = diseño(df, h, ctr)
        Xte, _, fo = diseño(df, h, cte)
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        if h <= 4:
            lc = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr)
            alphas.append(lc.alpha_)
            sel = np.abs(lc.coef_) > 1e-8
        else:
            sel = np.abs(Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr).coef_) > 1e-8
        sel = sel if sel.any() else np.ones_like(sel)
        try:
            pr = arch_model(ytr, x=Ztr[:, sel], mean="LS", rescale=False, vol="GARCH", p=1, q=1, dist="ged").fit(disp="off", show_warning=False).params
            pred = pr.iloc[0] + Zte[:, sel] @ pr.iloc[1:1 + sel.sum()].to_numpy()
            assert np.isfinite(pred).all()
        except Exception:
            pred = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(Ztr, ytr).predict(Zte)
        partes.append(pd.Series(pred, index=pd.DatetimeIndex(fo)))
    s = pd.concat(partes).sort_index()
    s = s[~s.index.duplicated()]
    return s[(s.index > CORTE) & (s.index <= FIN)]


def neuronales(df):
    """97 pasos desde la ultima hora publicada (23:00 de D-4); se guardan los ultimos 24 de cada ventana."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS
    H = OCULTO + 24
    o = df[df["fecha_hora"] <= FIN].dropna(subset=["precio_bolsa"]).reset_index(drop=True)
    n_test_dias = int(((o["fecha_hora"] > CORTE) & (o["fecha_hora"] <= FIN)).sum() // 24)
    mtr = o["fecha_hora"] <= CORTE - pd.Timedelta(hours=OCULTO)
    fut = FUTR_EXOG_CONT + FUTR_EXOG_BIN
    nf = o[["fecha_hora", "precio_bolsa"] + HIST_EXOG_DL + fut].copy()
    for c in HIST_EXOG_DL + FUTR_EXOG_CONT:
        nf[c] = (nf[c] - nf.loc[mtr, c].mean()) / nf.loc[mtr, c].std()
    nf[HIST_EXOG_DL + fut] = nf[HIST_EXOG_DL + fut].ffill().bfill()
    nf["unique_id"] = "precio_bolsa"
    nf = nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})[["unique_id", "ds", "y"] + HIST_EXOG_DL + fut]
    kw = dict(h=H, input_size=168, hist_exog_list=HIST_EXOG_DL, futr_exog_list=fut, max_steps=1000,
              val_check_steps=100, random_seed=42, enable_progress_bar=False)
    cv = NeuralForecast(models=[NBEATSx(**kw), NHITS(**kw)], freq="h").cross_validation(df=nf, n_windows=n_test_dias, step_size=24)
    cv = cv[(cv["ds"] - cv["cutoff"]) > pd.Timedelta(hours=OCULTO)]          # ultimos 24 pasos de cada ventana
    cv = cv.drop_duplicates("ds", keep="last").set_index("ds")
    return cv["NBEATSx"], cv["NHITS"]


def causal_con_margen(d, previa, cols, grupo, margen=4, min_dias=14, iteraciones=3):
    """Combinador causal de produccion, pero solo con ventanas cuyo precio real ya estaba publicado."""
    from combinador_optimo_mape import qra_pesos
    X, y = d[cols].to_numpy(), d["real"].to_numpy()
    g, dia = d[grupo].to_numpy(), d["dia"].to_numpy()
    Xp, yp, gp = previa[cols].to_numpy(), previa["real"].to_numpy(), previa[grupo].to_numpy()
    dias = np.sort(np.unique(dia))
    P = np.full(len(d), np.nan)
    for i in range(len(dias)):
        conocidos = dias[:max(0, i - margen + 1)]
        tr_m, te_m = np.isin(dia, conocidos), dia == dias[i]
        for gg in np.unique(g[te_m]):
            tr, te = tr_m & (g == gg), te_m & (g == gg)
            Xt, yt = X[tr], y[tr]
            if len(conocidos) < min_dias:
                Xt, yt = np.vstack([Xp[gp == gg], Xt]), np.concatenate([yp[gp == gg], yt])
            if len(yt) < len(cols) + 5:
                continue
            w = 1.0 / np.maximum(yt, 1e-6)
            ww, b = qra_pesos(Xt, yt, w)
            for _ in range(iteraciones):
                w = 2.0 / np.maximum(np.abs(yt) + np.abs(Xt @ ww + b), 1e-6)
                ww, b = qra_pesos(Xt, yt, w)
            P[te] = X[te] @ ww + b
    return P


def dm(y, a, b):
    import statsmodels.api as sm
    dd = np.abs(y - a) - np.abs(y - b)
    r = sm.OLS(dd, np.ones_like(dd)).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    return float(r.tvalues[0]), float(r.pvalues[0])


if __name__ == "__main__":
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = features_realistas()
    print(f"variables realistas listas ({len(df)} filas)", flush=True)
    pred = pd.DataFrame(index=pd.DatetimeIndex(df.loc[(df["fecha_hora"] > CORTE) & (df["fecha_hora"] <= FIN), "fecha_hora"]))
    serie = df.set_index("fecha_hora")
    pred["real"] = serie["precio_bolsa"]
    pred["Persistencia"] = serie["precio_lag24h"]
    pred["XGBoost"] = xgboost(df); print(f"  XGBoost ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pred["ARX+GARCH"] = arx_garch(df); print(f"  ARX+GARCH ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pred["GARCH-ged"] = garch_ged(df); print(f"  GARCH-ged ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pred["N-BEATSx"], pred["N-HiTS"] = neuronales(df); print(f"  N-BEATSx y N-HiTS ({(time.time() - t0) / 60:.1f} min)", flush=True)

    from ensamble_24h_enero import origen5
    from ensamble_con_predespacho_24h import alinear_2026, imar
    cm = imar()
    pan = pred.dropna(subset=VOT6 + ["real"]).copy()
    pan["dia"], pan["g4"] = (pan.index - pd.Timedelta(hours=1)).normalize(), pan.index.hour // 6
    pan["Predespacho"] = alinear_2026(pan.index, cm)
    previa = origen5()
    previa["Predespacho"] = cm.reindex(previa.index).to_numpy()
    previa = previa.dropna(subset=["Predespacho"])
    pan["ens6_realista"] = causal_con_margen(pan, previa, VOT6, "g4")
    hay = pan["Predespacho"].notna()
    pan["ens7_realista"] = pan["ens6_realista"]
    pan.loc[hay, "ens7_realista"] = causal_con_margen(pan[hay], previa, VOT6 + ["Predespacho"], "g4")
    # protocolo actual (archivos del proyecto) en las mismas horas
    pan["ens6_actual"] = pd.read_csv(RES / "pronostico_ensamble_bandas_24h_2026_6votantes.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")["q50"]
    pan["ens7_actual"] = pd.read_csv(RES / "pronostico_ensamble_bandas_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")["q50"]
    from ensamble_24h_enero import datos_2026
    act = datos_2026()
    pan["Persistencia_actual"] = act["Persistencia"]
    pan.to_csv(OUT / "predicciones.csv")
    ev = pan[pan.index >= "2026-01-15"].dropna(subset=["ens6_realista", "ens7_realista", "ens6_actual", "ens7_actual", "Persistencia_actual"])
    y = ev["real"].to_numpy()
    ev["Predespacho_solo"] = ev["Predespacho"].fillna(ev["ens6_realista"])
    filas = []
    for c, nom in (("Persistencia_actual", "Persistencia · protocolo actual (precio de ayer)"), ("ens6_actual", "Ensamble 6 · protocolo actual"),
                   ("ens7_actual", "Ensamble 7 · protocolo actual"), ("Persistencia", "Persistencia · realista (precio de hace 4 días)"),
                   ("XGBoost", "XGBoost · realista"), ("ARX+GARCH", "ARX+GARCH · realista"), ("N-BEATSx", "N-BEATSx · realista"),
                   ("N-HiTS", "N-HiTS · realista"), ("GARCH-ged", "GARCH-ged · realista"), ("Predespacho_solo", "Predespacho solo"),
                   ("ens6_realista", "Ensamble 6 · realista"), ("ens7_realista", "Ensamble 7 · realista (+ predespacho)")):
        p_ = ev[c].to_numpy()
        f = {"modelo": nom, "MAE": np.abs(y - p_).mean(), "RMSE": np.sqrt(((y - p_) ** 2).mean()), "MAPE_%": np.mean(np.abs(y - p_) / y) * 100,
             "R2": 1 - np.sum((y - p_) ** 2) / np.sum((y - y.mean()) ** 2)}
        filas.append(f)
    t = pd.DataFrame(filas)
    pruebas = []
    for a, b in (("ens6_actual", "ens6_realista"), ("ens7_actual", "ens7_realista"), ("ens6_realista", "ens7_realista"),
                 ("Persistencia", "ens7_realista"), ("Predespacho_solo", "ens7_realista"), ("ens6_actual", "ens7_realista")):
        tt, pp = dm(y, ev[a].to_numpy(), ev[b].to_numpy())
        pruebas.append(f"DM {b} frente a {a}: t = {tt:.2f}, p = {pp:.4f} (t > 0: {b} tiene menor error)")
    t.to_csv(OUT / "metricas.csv", index=False)
    pd.set_option("display.width", 220)
    txt = f"Horas evaluadas: {len(ev)} ({ev.index.min():%d/%m/%Y} a {ev.index.max():%d/%m/%Y})\n" + t.round(3).to_string(index=False) + "\n\n" + "\n".join(pruebas)
    txt += f"\n\nTiempo: {(time.time() - t0) / 60:.1f} min"
    (OUT / "resumen.txt").write_text(txt, encoding="utf-8")
    print(txt)
