# -*- coding: utf-8 -*-
"""
POR QUE FALLO EL MODELO DE RAMPAS. Descomposicion de las dos causas candidatas.

El modelo de rampas (modelo_rampas_24h.py) predice la TRANSICION en vez del NIVEL:
    Delta_h = p(t_h) - p(t_{h-1}),   y luego   p_hat(h) = p(corte) + suma_{j<=h} Delta_hat_j

Fallo (0/10 semillas como votante, p=0.0000 en la reconstruccion anclada). La idea era buena --
el error del proyecto SI vive en las rampas (MAE 105.5 en rampa fuerte vs 28.5 en meseta). Asi que
la pregunta no es "si fallo" sino "por que", y hay exactamente dos sospechosos que se pueden
separar experimentalmente:

  CAUSA A - ACUMULACION. La reconstruccion es una suma acumulada de 24 predicciones. Si cada
    Delta_hat trae error e_j, el nivel en el paso h arrastra la suma de los h errores. Aunque cada
    e_j sea chico e insesgado, la suma se comporta como una caminata aleatoria y su desviacion
    crece como sqrt(h). El paso 24 hereda 24 errores.

  CAUSA B - RESTRICCION (mala especificacion). Anclar en p(corte) y sumar equivale a IMPONER que
    el coeficiente del ancla sea exactamente 1.0:   d(nivel)/d(p_corte) = 1.
    Pero el precio revierte a la media: el nivel de manana no se mueve 1 a 1 con el de hoy. Si la
    carga optima del ancla es, digamos, 0.6, forzarla a 1.0 es un error de especificacion que NO
    tiene nada que ver con lo bien o mal que se predigan las rampas.

    Dicho de otro modo: el modelo de NIVEL (LEAR) ya tiene p(corte) entre sus variables y es libre
    de ponerle el coeficiente que quiera. El modelo de rampas vive en un SUBESPACIO del modelo de
    nivel -- el mismo modelo con un coeficiente clavado a mano. Un subespacio no puede contener
    informacion nueva; solo puede perder grados de libertad.

COMO SE SEPARAN. Se agrega una tercera variante que tiene la CAUSA B pero no la CAUSA A:

    "ancla directa":  Delta_h = p(t_h) - p(corte)   predicho de una sola vez para cada paso h,
                      reconstruido como p(corte) + Delta_hat_h.   (restriccion si, acumulacion no)

Con las tres variantes + el LEAR de nivel (sin restriccion ni acumulacion) queda una tabla 2x2 que
identifica al culpable. Ademas se mide directamente la carga optima del ancla por paso, que es la
prueba directa de la CAUSA B, y el encogimiento de los Delta predichos.

Todo se mide sobre el mismo periodo 2026 y los mismos pasos que el resto del proyecto.
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
from lear_24h import construir_para_paso

CORTE = pd.Timestamp("2026-01-01")


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV, Lasso, LinearRegression
    from sklearn.preprocessing import StandardScaler

    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    precio = df["precio_bolsa"].to_numpy(float)
    pos = pd.Series(df.index.values, index=fechas)
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pos[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - 24)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]

    filas, cargas, alphas = [], [], []
    print("Ajustando por paso (nivel libre vs ancla forzada)...", flush=True)
    for h in range(1, 25):
        Xtr, ytr, _, ctr = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_obj, cte = construir_para_paso(df, h, cortes_te)
        a_tr = precio[pos.reindex(pd.DatetimeIndex(ctr)).to_numpy()]      # ancla = precio del corte
        a_te = precio[pos.reindex(pd.DatetimeIndex(cte)).to_numpy()]

        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        if h <= 4 or h % 6 == 0:
            alphas.append(LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(Ztr, ytr).alpha_)
        al = float(np.median(alphas))

        # (1) NIVEL libre: el LEAR. El ancla es una variable mas, con coeficiente libre.
        m_niv = Lasso(alpha=al, max_iter=10000, random_state=42).fit(Ztr, ytr)
        p_nivel = m_niv.predict(Zte)

        # (2) ANCLA DIRECTA: objetivo = y - ancla, un solo paso (restriccion SI, acumulacion NO)
        m_anc = Lasso(alpha=al, max_iter=10000, random_state=42).fit(Ztr, ytr - a_tr)
        p_ancla = a_te + m_anc.predict(Zte)

        # carga implicita del propio LEAR: derivada de la prediccion respecto al ancla
        j = list(Xtr.columns).index("p_corte_lag0")
        carga_lear = float(m_niv.coef_[j] / sc.scale_[j])
        # carga total del ancla via OLS de y sobre [ancla, resto de la prediccion]
        resto = p_nivel * 0  # placeholder, se calcula sobre train abajo
        pred_tr = m_niv.predict(Ztr)
        base = LinearRegression().fit(np.column_stack([a_tr, pred_tr - carga_lear * a_tr]), ytr)
        carga_ols = float(base.coef_[0])
        cargas.append({"paso": h, "carga_LEAR": carga_lear, "carga_OLS": carga_ols,
                       "carga_forzada_rampa": 1.0})

        filas.append(pd.DataFrame({"fecha_hora": f_obj, "paso": h, "real": yte,
                                   "ancla": a_te, "nivel_libre": p_nivel, "ancla_directa": p_ancla}))
        if h % 8 == 0:
            print(f"    paso {h}/24", flush=True)

    D = pd.concat(filas, ignore_index=True)
    R = pd.read_csv(RES / "modelo_rampas_24h.csv", parse_dates=["fecha_hora"])
    D = D.merge(R[["fecha_hora", "nivel_dir", "nivel_hur", "delta_dir", "prev_real", "precio_corte"]],
                on="fecha_hora", how="inner")
    y = D["real"].to_numpy()

    print(f"\n{'='*94}")
    print(f"TABLA 2x2: RESTRICCION x ACUMULACION   (n={len(D):,}, 2026)")
    print(f"{'='*94}")
    print(f"{'variante':38s} {'restriccion':>12s} {'acumulacion':>12s} {'MAE':>9s}")
    tabla = [("LEAR de nivel (ancla con coef. libre)", "no", "no", "nivel_libre"),
             ("Ancla directa (coef. forzado a 1.0)",  "SI", "no", "ancla_directa"),
             ("Rampas, suma acumulada (directa)",     "SI", "SI", "nivel_dir"),
             ("Rampas, suma acumulada (hurdle)",      "SI", "SI", "nivel_hur")]
    m_ref = None
    for nom, res, acu, col in tabla:
        v = mae(y, D[col])
        if m_ref is None:
            m_ref = v
        print(f"{nom:38s} {res:>12s} {acu:>12s} {v:9.2f}   ({v-m_ref:+.2f} vs LEAR)")
    print(f"{'(referencia) ancla fija = persistencia':38s} {'-':>12s} {'-':>12s} "
          f"{mae(y, D['ancla']):9.2f}")

    print(f"\n{'='*94}")
    print("CAUSA B, PRUEBA DIRECTA: cuanto vale de verdad el coeficiente del ancla")
    print(f"{'='*94}")
    C = pd.DataFrame(cargas)
    print("  El modelo de rampas lo fuerza a 1.0 en los 24 pasos. El LEAR, libre, elige:")
    print(f"    media={C['carga_LEAR'].mean():.3f}   min={C['carga_LEAR'].min():.3f} "
          f"(paso {int(C.loc[C['carga_LEAR'].idxmin(),'paso'])})   "
          f"max={C['carga_LEAR'].max():.3f} (paso {int(C.loc[C['carga_LEAR'].idxmax(),'paso'])})")
    for h in [1, 2, 4, 8, 12, 18, 24]:
        r = C[C["paso"] == h].iloc[0]
        print(f"    paso {h:2d}:  LEAR={r['carga_LEAR']:6.3f}   rampas={1.0:5.2f}   "
              f"exceso impuesto = {1.0-r['carga_LEAR']:+6.3f}")

    print(f"\n{'='*94}")
    print("CAUSA A, PRUEBA DIRECTA: como crece el error con el paso")
    print(f"{'='*94}")
    print(f"{'paso':>5s} {'LEAR nivel':>11s} {'ancla directa':>14s} {'rampas cumsum':>14s} "
          f"{'cumsum/LEAR':>12s}")
    g = D.groupby("paso").apply(lambda s: pd.Series({
        "lear": mae(s["real"], s["nivel_libre"]),
        "ancla": mae(s["real"], s["ancla_directa"]),
        "cum": mae(s["real"], s["nivel_dir"])}), include_groups=False)
    for h in [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]:
        r = g.loc[h]
        print(f"{h:5d} {r['lear']:11.2f} {r['ancla']:14.2f} {r['cum']:14.2f} {r['cum']/r['lear']:12.2f}x")
    cr = np.corrcoef(g.index.to_numpy(float), g["cum"].to_numpy())[0, 1]
    cr_l = np.corrcoef(g.index.to_numpy(float), g["lear"].to_numpy())[0, 1]
    print(f"\n  correlacion (paso, MAE):  rampas cumsum = {cr:+.3f}   |   LEAR nivel = {cr_l:+.3f}")
    print("  Si el cumsum crece con el paso y el LEAR no, la acumulacion es real.")

    print(f"\n{'='*94}")
    print("ENCOGIMIENTO: el Delta predicho ni siquiera intenta las rampas")
    print(f"{'='*94}")
    d_real = (D["real"] - D["prev_real"]).to_numpy()
    d_pred = D["delta_dir"].to_numpy()
    print(f"  desv. est. del Delta REAL     = {d_real.std():8.2f}")
    print(f"  desv. est. del Delta PREDICHO = {d_pred.std():8.2f}   "
          f"({100*d_pred.std()/d_real.std():.1f}% de la real)")
    print(f"  correlacion Delta real vs predicho = {np.corrcoef(d_real, d_pred)[0,1]:+.3f}")
    print(f"  R2 del Delta = {1 - np.sum((d_real-d_pred)**2)/np.sum((d_real-d_real.mean())**2):+.3f}  "
          "(negativo = peor que predecir siempre el promedio)")
    ceros = float(np.mean(np.abs(d_real) < 1e-9) * 100)
    print(f"  % de Delta reales exactamente 0 = {ceros:.1f}%")
    fuerte = np.abs(d_real) >= 50
    print(f"  en rampa fuerte (n={int(fuerte.sum())}): |Delta| real medio = {np.abs(d_real[fuerte]).mean():.1f}, "
          f"|Delta| predicho medio = {np.abs(d_pred[fuerte]).mean():.1f}  "
          f"-> captura el {100*np.abs(d_pred[fuerte]).mean()/np.abs(d_real[fuerte]).mean():.0f}%")

    D.to_csv(RES / "diagnostico_rampas_porque_fallo.csv", index=False)
    C.to_csv(RES / "diagnostico_rampas_carga_ancla.csv", index=False)
    print("\nGuardado: diagnostico_rampas_porque_fallo.csv, diagnostico_rampas_carga_ancla.csv")
    print(f"LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
