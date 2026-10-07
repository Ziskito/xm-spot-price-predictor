# -*- coding: utf-8 -*-
"""
Escenario realista con el hueco de precio lleno con el predespacho ideal de XM (7-oct-2026).

Complemento de escenario_realista_24h.py (correr ese primero). A las 00:00 del dia D el precio de bolsa publico
llega solo hasta las 23:00 de D-4; las 73 horas siguientes no estan publicadas, pero el predespacho ideal de
esas horas SI (sale el dia anterior a cada dia). Aqui el precio no publicado se reemplaza por el predespacho:
  * precio_lag24h, medias, desviaciones, rango y ratio de volatilidad: con la serie hibrida (precio real hasta
    las 23:00 de D-4, predespacho despues), calculada para cada corte
  * Persistencia = predespacho de la misma hora del dia anterior
  * GARCH-ged: diseño LEAR original (rezagos desde el corte 0..168 h) con la serie hibrida
  * N-BEATSx / N-HiTS: entrenadas como en el proyecto (horizonte 24) y pronosticadas cada dia con la ventana
    de entrada hibrida
  * demanda, embalses, aportes y ONI quedan con su rezago realista (no hay sustituto publico equivalente)
  * mismo combinador con 4 ventanas de margen; ensambles de 6 y 7 (+ predespacho del dia objetivo)
Compara: protocolo actual, realista sin relleno y realista con relleno, en las mismas horas.
Salidas: data/processed/resultados/escenario_realista/relleno_predicciones.csv, comparacion_3_escenarios.csv,
         resumen_3_escenarios.txt
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import escenario_realista_24h as er  # noqa: E402
from o6_comun import EXCLUIR, RES  # noqa: E402
from lear_24h import EXOG_CORTE, EXOG_OBJ, MISMA_HORA, REZAGOS_CORTE  # noqa: E402
from reentrenar_tras_fix_demanda import FUTR_EXOG_BIN, FUTR_EXOG_CONT, HIST_EXOG_DL  # noqa: E402
from ensamble_con_predespacho_24h import alinear_2026, imar  # noqa: E402

OUT = RES / "escenario_realista"
CORTE, FIN, OCULTO = er.CORTE, er.FIN, er.OCULTO
VOT6 = er.VOT6


def series(df):
    """P: precio real; I: predespacho a la misma hora (huecos: ultimo valor previo, luego el precio de hace 96 h)."""
    cm = imar()
    P = df["precio_bolsa"].to_numpy(np.float64)
    I = pd.Series(cm.reindex(df["fecha_hora"]).to_numpy()).ffill(limit=24)
    I = I.fillna(pd.Series(P).shift(96)).to_numpy(np.float64)
    return P, I


def corte_de(fechas):
    """Indice del corte (00:00 del dia de la ventana) para cada fila."""
    dia = (fechas - pd.Timedelta(hours=1)).dt.normalize()
    pos = pd.Series(np.arange(len(fechas)), index=fechas)
    return pos.reindex(dia).to_numpy()


def suma_hibrida(cP, cI, a, b, frontera):
    """Suma de la serie hibrida en (a, b]: real hasta 'frontera', predespacho despues (indices enteros)."""
    m = np.clip(frontera, a, b)
    return (cP[m] - cP[a]) + (cI[b] - cI[m])


def features_relleno(df):
    d = df.copy()
    P, I = series(d)
    n = len(d)
    c = corte_de(d["fecha_hora"])
    ok = ~np.isnan(c)
    c = np.where(ok, c, 0).astype(int)
    fr = c - OCULTO                                         # ultimo indice con precio publicado
    t = np.arange(n)
    Pz, Iz = np.nan_to_num(P), np.nan_to_num(I)
    cP, cI = np.concatenate([[0], np.cumsum(Pz)]), np.concatenate([[0], np.cumsum(Iz)])
    cP2, cI2 = np.concatenate([[0], np.cumsum(Pz ** 2)]), np.concatenate([[0], np.cumsum(Iz ** 2)])
    def ventana(L):
        b = t - 24 + 1                                      # (t-24-L, t-24] en indices de suma acumulada
        a = np.clip(b - L, 0, None)
        b = np.clip(b, 0, None)
        frontera = np.clip(fr + 1, 0, None)
        s = suma_hibrida(cP, cI, a, b, frontera)
        s2 = suma_hibrida(cP2, cI2, a, b, frontera)
        media = s / L
        var = np.maximum(s2 / L - media ** 2, 0) * L / max(L - 1, 1)
        return media, np.sqrt(var)
    lag24 = np.where(t - 24 > fr, I[np.clip(t - 24, 0, None)], P[np.clip(t - 24, 0, None)])
    m24, s24 = ventana(24)
    m7, s7 = ventana(168)
    m30, _ = ventana(720)
    rmax = pd.Series(I).rolling(24).max().shift(24).to_numpy()          # (t-48, t-24]: todo despues del corte-73
    rmin = pd.Series(I).rolling(24).min().shift(24).to_numpy()
    d["precio_lag24h"], d["precio_media_24h"], d["precio_media_7d"], d["precio_media_30d"] = lag24, m24, m7, m30
    d["precio_std_24h"], d["precio_std_7d"], d["precio_rango_24h"] = s24, s7, rmax - rmin
    d["ratio_volatilidad"] = d["precio_std_24h"] / d["precio_std_7d"]
    cols = ["precio_lag24h", "precio_media_24h", "precio_media_7d", "precio_media_30d", "precio_std_24h",
            "precio_std_7d", "precio_rango_24h", "ratio_volatilidad"]
    d.loc[~ok | (t < 800), cols] = np.nan
    return d, P, I


def diseño_relleno(df, P, I, h, cortes):
    """Diseño LEAR original con la serie hibrida vista desde cada corte."""
    n = len(df)
    c = cortes[(cortes + h < n) & (cortes - 200 >= 0)]
    t = c + h
    hib = lambda idx: np.where(idx > c - OCULTO, I[idx], P[idx])
    col = [hib(c - k) for k in REZAGOS_CORTE]
    col += [hib(t - mm) for mm in MISMA_HORA]
    ven = np.stack([hib(c - k) for k in range(1, 25)], axis=1)
    col += [ven.min(1), ven.max(1), ven.mean(1), ven.std(1)]
    for cc in EXOG_CORTE:
        col.append(df[cc].to_numpy(np.float64)[c])
    for cc in EXOG_OBJ:
        col.append(df[cc].to_numpy(np.float64)[t])
    X = np.column_stack(col)
    ok = ~np.isnan(X).any(1)
    return X[ok], P[t][ok], df["fecha_hora"].to_numpy()[t][ok]


def garch_ged(df, P, I):
    from arch import arch_model
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler
    f = df["fecha_hora"]
    idx00 = np.where(f.dt.hour.to_numpy() == 0)[0]
    ic = int(np.where(f == CORTE)[0][0])
    ctr, cte = idx00[(idx00 >= 800) & (idx00 < ic - 24)], idx00[idx00 >= ic - 1]
    partes, alphas = [], []
    for h in range(1, 25):
        Xtr, ytr, _ = diseño_relleno(df, P, I, h, ctr)
        Xte, _, fo = diseño_relleno(df, P, I, h, cte)
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


def neuronales(df, P, I):
    """Entrenadas como en el proyecto (precio real, horizonte 24); cada dia se pronostica con la ventana hibrida."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS
    fut = FUTR_EXOG_CONT + FUTR_EXOG_BIN
    base = df[["fecha_hora"] + HIST_EXOG_DL + fut].copy()
    base["y"] = P
    mtr = base["fecha_hora"] <= CORTE
    for c in HIST_EXOG_DL + FUTR_EXOG_CONT:
        base[c] = (base[c] - base.loc[mtr, c].mean()) / base.loc[mtr, c].std()
    base[HIST_EXOG_DL + fut] = base[HIST_EXOG_DL + fut].ffill().bfill()
    base["unique_id"] = "precio_bolsa"
    base = base.rename(columns={"fecha_hora": "ds"})
    tr = base[mtr & base["y"].notna()].dropna()
    kw = dict(h=24, input_size=168, hist_exog_list=HIST_EXOG_DL, futr_exog_list=fut, max_steps=1000,
              val_check_steps=100, random_seed=42, enable_progress_bar=False)
    nf = NeuralForecast(models=[NBEATSx(**kw), NHITS(**kw)], freq="h")
    nf.fit(df=tr[["unique_id", "ds", "y"] + HIST_EXOG_DL + fut])
    salidas = []
    idx00 = np.where((base["ds"].dt.hour == 0).to_numpy() & (base["ds"] >= CORTE).to_numpy() & (base["ds"] < FIN).to_numpy())[0]
    for c in idx00:
        h = base.iloc[c - 400:c + 1].copy()
        oculto = np.arange(len(h)) > len(h) - 1 - OCULTO
        h.loc[oculto, "y"] = I[c - 400:c + 1][oculto]
        futuro = base.iloc[c + 1:c + 25][["unique_id", "ds"] + fut]
        p = nf.predict(df=h[["unique_id", "ds", "y"] + HIST_EXOG_DL + fut], futr_df=futuro)
        salidas.append(p.set_index("ds")[["NBEATSx", "NHITS"]])
    s = pd.concat(salidas)
    return s["NBEATSx"], s["NHITS"]


if __name__ == "__main__":
    t0 = time.time()
    df, P, I = features_relleno(er.features_realistas())
    print(f"variables con relleno listas ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pred = pd.DataFrame(index=pd.DatetimeIndex(df.loc[(df["fecha_hora"] > CORTE) & (df["fecha_hora"] <= FIN), "fecha_hora"]))
    serie = df.set_index("fecha_hora")
    pred["real"] = serie["precio_bolsa"]
    pred["Persistencia"] = serie["precio_lag24h"]
    pred["XGBoost"] = er.xgboost(df); print(f"  XGBoost ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pred["ARX+GARCH"] = er.arx_garch(df); print(f"  ARX+GARCH ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pred["GARCH-ged"] = garch_ged(df, P, I); print(f"  GARCH-ged ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pred["N-BEATSx"], pred["N-HiTS"] = neuronales(df, P, I); print(f"  N-BEATSx y N-HiTS ({(time.time() - t0) / 60:.1f} min)", flush=True)

    from ensamble_24h_enero import origen5
    cm = imar()
    pan = pred.dropna(subset=VOT6 + ["real"]).copy()
    pan["dia"], pan["g4"] = (pan.index - pd.Timedelta(hours=1)).normalize(), pan.index.hour // 6
    pan["Predespacho"] = alinear_2026(pan.index, cm)
    previa = origen5()
    previa["Predespacho"] = cm.reindex(previa.index).to_numpy()
    previa = previa.dropna(subset=["Predespacho"])
    pan["ens6"] = er.causal_con_margen(pan, previa, VOT6, "g4")
    hay = pan["Predespacho"].notna()
    pan["ens7"] = pan["ens6"]
    pan.loc[hay, "ens7"] = er.causal_con_margen(pan[hay], previa, VOT6 + ["Predespacho"], "g4")
    pan.to_csv(OUT / "relleno_predicciones.csv")

    sin = pd.read_csv(OUT / "predicciones.csv", parse_dates=[0], index_col=0)
    j = pan[["real", "ens6", "ens7"]].rename(columns={"ens6": "ens6_relleno", "ens7": "ens7_relleno"}).join(
        sin[["ens6_actual", "ens7_actual", "ens6_realista", "ens7_realista", "Persistencia_actual"]], how="inner")
    j = j[j.index >= "2026-01-15"].dropna()
    y = j["real"].to_numpy()
    filas = []
    for c, esc, mod in (("ens6_actual", "1. protocolo actual (precio hasta la hora anterior)", "ensamble 6"),
                        ("ens7_actual", "1. protocolo actual (precio hasta la hora anterior)", "ensamble 7"),
                        ("ens6_realista", "2. realista sin relleno (precio hasta hace ~3 dias)", "ensamble 6"),
                        ("ens7_realista", "2. realista sin relleno (precio hasta hace ~3 dias)", "ensamble 7"),
                        ("ens6_relleno", "3. realista, hueco lleno con predespacho", "ensamble 6"),
                        ("ens7_relleno", "3. realista, hueco lleno con predespacho", "ensamble 7")):
        p_ = j[c].to_numpy()
        filas.append({"escenario": esc, "modelo": mod, "MAE": np.abs(y - p_).mean(), "RMSE": np.sqrt(((y - p_) ** 2).mean()),
                      "MAPE_%": np.mean(np.abs(y - p_) / y) * 100, "R2": 1 - np.sum((y - p_) ** 2) / np.sum((y - y.mean()) ** 2),
                      "rMAE_vs_persistencia_actual": np.abs(y - p_).mean() / np.abs(y - j["Persistencia_actual"]).mean()})
    t = pd.DataFrame(filas)
    t.to_csv(OUT / "comparacion_3_escenarios.csv", index=False)
    pruebas = []
    for a, b in (("ens7_realista", "ens7_relleno"), ("ens6_relleno", "ens7_relleno"), ("ens7_actual", "ens7_relleno"), ("ens6_actual", "ens7_relleno")):
        tt, pp = er.dm(y, j[a].to_numpy(), j[b].to_numpy())
        pruebas.append(f"DM {b} frente a {a}: t = {tt:.2f}, p = {pp:.4f} (t > 0: {b} tiene menor error)")
    pd.set_option("display.width", 220)
    txt = (f"Horas evaluadas: {len(j)} ({j.index.min():%d/%m/%Y} a {j.index.max():%d/%m/%Y})\n" + t.round(3).to_string(index=False)
           + "\n\n" + "\n".join(pruebas) + f"\n\nTiempo: {(time.time() - t0) / 60:.1f} min")
    (OUT / "resumen_3_escenarios.txt").write_text(txt, encoding="utf-8")
    print(txt)
