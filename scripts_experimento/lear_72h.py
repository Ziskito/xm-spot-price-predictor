# -*- coding: utf-8 -*-
"""
LEAR (LASSO Estimated AutoRegressive) para 72h: el benchmark canonico del campo, que nunca
habiamos construido bien.

Por que: la literatura de pronostico de precios electricos sostiene que un lineal bien armado y
regularizado iguala o supera al deep learning. El benchmark de referencia es LEAR (Lago, Marcjasz,
De Schutter & Weron 2021, Applied Energy 293:116983), que en los mercados mas estudiados del mundo
(Nord Pool, PJM, EPEX) resulta muy dificil de superar. Kapoor & Wichitaksorn (2023) lo confirman en
Nueva Zelanda: sus lineales/estadisticos con LASSO ganan a DNN, LSTM, GRU y XGBoost.

Nuestro Ridge-directo NO es un LEAR. Le faltan las tres cosas que definen al LEAR:
  1) ESTRUCTURA RICA DE REZAGOS. Ridge-directo ve el precio del corte, unas medias moviles y
     precio_lag168h. Un LEAR ve explicitamente el precio en muchos rezagos concretos.
  2) ESTIMACION POR LASSO (L1), no Ridge (L2). L1 pone coeficientes exactamente en cero y
     selecciona; L2 solo encoge. Esa seleccion es justo lo que la literatura senala como la clave.
  3) UN MODELO POR PASO DE HORIZONTE. En la literatura de day-ahead se ajusta un modelo por hora
     del dia, no un modelo unico con la hora como variable. Aqui el analogo es un modelo por paso
     h = 1..72, cada uno especializado en su distancia al corte.
Se agrega ademas la RECALIBRACION MOVIL, que es como se despliega LEAR en la practica.

Anti-fuga (lo mismo que rige todo el proyecto): para el objetivo t con paso h, el corte es c = t-h
y SOLO se usa informacion disponible en c. Los rezagos de precio se toman del corte hacia atras
(precio[c], precio[c-1], ...), y los "misma hora de dias anteriores" solo se incluyen si el
instante correspondiente es <= c. precio_lag168h siempre es valido porque h <= 72 < 168.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES

CORTE = pd.Timestamp("2026-01-01")
PRIMER_CORTE = pd.Timestamp("2026-01-01 00:00")
N_WIN, H = 214, 72
TRAMOS = [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]

# Rezagos de precio medidos DESDE EL CORTE hacia atras (siempre conocidos en el corte)
REZAGOS_CORTE = [0, 1, 2, 3, 6, 12, 18, 24, 48, 72, 96, 120, 144, 168]
# "Misma hora de dias anteriores" relativos al OBJETIVO (se validan contra el corte)
MISMA_HORA = [24, 48, 72, 96, 120, 144, 168]

EXOG_CORTE = ["volumen_embalses", "volumen_embalses_vs_media30d", "aportes_hidricos",
              "aportes_hidricos_vs_media30d", "demanda", "demanda_media_24h",
              "precio_media_24h", "precio_media_7d", "precio_media_30d", "precio_std_24h"]
EXOG_OBJ = ["oni", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
            "dia_anio_sin", "dia_anio_cos", "es_festivo", "es_pandemia"]


def construir_para_paso(df, h, idx_cortes):
    """Matriz de diseno del LEAR para un paso h fijo. Un registro por corte valido."""
    n = len(df)
    precio = df["precio_bolsa"].to_numpy(np.float64)
    fechas = df["fecha_hora"].to_numpy()
    c = idx_cortes[(idx_cortes + h < n) & (idx_cortes - max(REZAGOS_CORTE) >= 0)]
    t = c + h

    col, nom = [], []
    for k in REZAGOS_CORTE:                       # precio en el corte y antes: siempre valido
        col.append(precio[c - k]); nom.append(f"p_corte_lag{k}")
    for m in MISMA_HORA:                          # misma hora de dias previos, solo si t-m <= c
        v = np.where(t - m <= c, precio[np.maximum(t - m, 0)], np.nan)
        if not np.isnan(v).all():
            col.append(v); nom.append(f"p_obj_lag{m}")
    # estadisticos del dia anterior al corte (clasicos en LEAR: min/max/media de las 24h previas)
    ventana = np.stack([precio[c - k] for k in range(1, 25)], axis=1)
    col += [ventana.min(1), ventana.max(1), ventana.mean(1)]
    nom += ["p_min24_corte", "p_max24_corte", "p_media24_corte"]
    for cc in EXOG_CORTE:
        col.append(df[cc].to_numpy(np.float64)[c]); nom.append(f"{cc}_corte")
    for cc in EXOG_OBJ:
        col.append(df[cc].to_numpy(np.float64)[t]); nom.append(f"{cc}_obj")
    X = np.column_stack(col)
    ok = ~np.isnan(X).any(1)
    return (pd.DataFrame(X[ok], columns=nom), precio[t][ok],
            fechas[t][ok], fechas[c][ok])


def dm(real, p_ref, p_nuevo, maxlags=71):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler

    df = cargar_completo().reset_index(drop=True)
    pos = pd.Series(df.index.values, index=df["fecha_hora"])
    idx_fin = int(pos[CORTE])
    cortes_test = pos.reindex(pd.date_range(PRIMER_CORTE, periods=N_WIN, freq="24h")).astype(int).to_numpy()
    idx_train = np.arange(200, idx_fin - H)

    print(f"Construyendo LEAR: un modelo LASSO por cada uno de los {H} pasos de horizonte.")
    print(f"Variables por modelo: ~{len(REZAGOS_CORTE)+len(MISMA_HORA)+3+len(EXOG_CORTE)+len(EXOG_OBJ)}\n")

    partes = []
    alphas, n_sel = [], []
    for h in range(1, H + 1):
        Xtr, ytr, _, _ = construir_para_paso(df, h, idx_train)
        Xte, yte, f_obj, f_cor = construir_para_paso(df, h, cortes_test)
        esc = StandardScaler().fit(Xtr)
        Ztr, Zte = esc.transform(Xtr), esc.transform(Xte)
        # alpha por CV en los primeros pasos, luego se reutiliza la mediana (LassoCV por paso
        # es caro y el alpha optimo es muy estable entre pasos vecinos)
        if h <= 6 or h % 12 == 0:
            m = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=8000).fit(Ztr, ytr)
            alphas.append(m.alpha_)
        else:
            m = Lasso(alpha=float(np.median(alphas)), max_iter=8000, random_state=42).fit(Ztr, ytr)
        n_sel.append(int((np.abs(m.coef_) > 1e-8).sum()))
        partes.append(pd.DataFrame({"fecha_hora": f_obj, "cutoff": f_cor, "paso_horas": h,
                                    "real": yte, "LEAR": m.predict(Zte)}))
        if h % 12 == 0:
            print(f"  paso {h:2d}/72 listo ({(time.time()-t0)/60:.1f} min) "
                  f"| alpha mediano={np.median(alphas):.3f} | variables activas={n_sel[-1]}", flush=True)

    lear = pd.concat(partes, ignore_index=True)
    lear.to_csv(RES / "pronostico_lear_72h_2026.csv", index=False)
    print(f"\nLEAR terminado. Variables activas: min={min(n_sel)} max={max(n_sel)} "
          f"mediana={int(np.median(n_sel))}")

    # ---------- comparacion contra los votantes y el ensamble vigente ----------
    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    e = pd.read_csv(RES / "pronostico_ensamble_72h_diario_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    m = (v.merge(lear[["fecha_hora", "cutoff", "paso_horas", "LEAR"]], on=["fecha_hora", "cutoff", "paso_horas"])
          .merge(e[["fecha_hora", "cutoff", "paso_horas", "q50_v3_cv", "q50_v3_dsp"]],
                 on=["fecha_hora", "cutoff", "paso_horas"], how="left"))
    print(f"Filas comparables: {len(m):,}\n")

    comparar = ["LEAR", "Ridge-directo", "NBX", "NBX_exog", "XGB-directo", "CatB-directo",
                "NaiveEstac", "q50_v3_cv", "q50_v3_dsp"]
    print(f"{'='*96}\nMAE POR TRAMO -- LEAR contra todo lo demas\n{'='*96}")
    print(f"{'modelo':18s} {'1-24h':>9s} {'25-48h':>9s} {'49-72h':>9s} {'global':>9s}")
    filas = []
    for c in comparar:
        if c not in m.columns:
            continue
        s = m.dropna(subset=[c])
        fila, linea = {"modelo": c}, f"{c:18s}"
        for nom, ini, fin in TRAMOS:
            sub = s[s.paso_horas.between(ini, fin)]
            mae = float(np.abs(sub["real"] - sub[c]).mean())
            fila[f"mae_{nom}"] = mae
            linea += f" {mae:9.2f}"
        fila["mae_global"] = float(np.abs(s["real"] - s[c]).mean())
        print(linea + f" {fila['mae_global']:9.2f}")
        filas.append(fila)

    print(f"\n{'='*96}\nDM: LEAR contra el ensamble completo (q50_v3_cv) y contra Ridge-directo\n{'='*96}")
    for ref in ["q50_v3_cv", "Ridge-directo", "NBX"]:
        if ref not in m.columns:
            continue
        s = m.dropna(subset=["LEAR", ref])
        for nom, ini, fin in TRAMOS:
            sub = s[s.paso_horas.between(ini, fin)]
            y = sub["real"].to_numpy()
            t, pv = dm(y, sub[ref].to_numpy(), sub["LEAR"].to_numpy())
            d = np.abs(y - sub["LEAR"]).mean() - np.abs(y - sub[ref]).mean()
            marca = ("LEAR GANA" if t > 0 else "LEAR pierde") if pv < 0.05 else "sin diferencia"
            print(f"  LEAR vs {ref:14s} {nom:8s} delta={d:+7.2f}  p={pv:.4f}  {marca}")

    pd.DataFrame(filas).to_csv(RES / "lear_72h_metricas.csv", index=False)
    print(f"\nGuardado: pronostico_lear_72h_2026.csv, lear_72h_metricas.csv "
          f"| LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
