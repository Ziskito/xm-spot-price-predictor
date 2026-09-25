# -*- coding: utf-8 -*-
"""
MARGEN DE RESERVA como variable para el pico vespertino del modelo de 24h.

Contexto y por que esto NO repite el tamiz previo. En la sesion del 2026-09-09 se descargaron 12
variables del API de XM y se tamizaron por correlacion parcial (descontando precio_lag24h),
concluyendo que 9 de 12 eran redundantes. Pero ese tamiz probo las variables CRUDAS. Nunca se
construyo el MARGEN DE RESERVA = disponibilidad total - demanda, que es la magnitud con sentido
fisico para la escasez: el precio no se dispara porque haya mucha o poca capacidad en absoluto,
sino porque la capacidad disponible se acerca a la demanda. Una correlacion parcial sobre la
disponibilidad cruda no puede ver eso, porque el efecto vive en la DIFERENCIA con la demanda.

Ademas, aquel tamiz y la prueba posterior se hicieron:
  - sobre el MAE global o el de horas de El Nino, nunca sobre las horas PICO 18-20, que es donde
    diagnostico_error_24h.py localizo el 37.8% del error;
  - dentro de N-BEATSx, nunca en la formulacion lineal/directa.

Restricciones que se respetan:
  - `max_precio_oferta` NO se usa: el tamiz previo demostro que es fuga pura (corr. contemporanea
    0.9998, es literalmente el precio marginal).
  - NO se desglosa generacion ni disponibilidad POR TIPO de recurso: el Alcance del Anexo 1 lo
    restringe. Aqui se usa unicamente la SUMA TOTAL de disponibilidad, que no es un desglose.
  - Anti-fuga temporal: todas las variables se toman EN EL CORTE (00:00 del dia anterior) o antes,
    nunca en el instante objetivo.
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
from lear_24h import construir_para_paso, REZAGOS_CORTE

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
CORTE = pd.Timestamp("2026-01-01")
H, PICO = 24, [18, 19, 20]


def dm(real, p_ref, p_nuevo, maxlags=24):
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred) / real) * 100)


def cargar_oferta():
    """Variables del lado de la oferta, agregadas a TOTAL (sin desglose por tipo)."""
    dis = pd.read_csv(RAIZ / "data/disponibilidad_por_tipo_2019_2026.csv", parse_dates=["fecha_hora"])
    tipos = [c for c in dis.columns if c != "fecha_hora"]
    dis["disponibilidad_total"] = dis[tipos].sum(axis=1)          # SUMA, no desglose
    dis = dis[["fecha_hora", "disponibilidad_total"]].set_index("fecha_hora")

    cmg = pd.read_csv(RAIZ / "data/costo_marginal_despacho_2019_2026.csv", parse_dates=["fecha_hora"])
    cmg = cmg.set_index("fecha_hora").iloc[:, [0]]
    cmg.columns = ["costo_marginal"]

    esc = pd.read_csv(RAIZ / "data/precio_escasez_2019_2026.csv", parse_dates=["fecha_hora"])
    esc = esc.set_index("fecha_hora").iloc[:, [0]]
    esc.columns = ["precio_escasez"]
    esc = esc.resample("1h").ffill()                              # es diario -> se propaga a horario

    o = dis.join(cmg, how="outer").join(esc, how="outer").sort_index()
    return o


def main():
    t0 = time.time()
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler

    df = cargar_completo().reset_index(drop=True)
    of = cargar_oferta()
    idx = pd.DatetimeIndex(df["fecha_hora"])
    dem = df["demanda"].to_numpy(np.float64)
    disp = of["disponibilidad_total"].reindex(idx).to_numpy(np.float64)
    cmg = of["costo_marginal"].reindex(idx).to_numpy(np.float64)
    esc = of["precio_escasez"].reindex(idx).to_numpy(np.float64)
    precio = df["precio_bolsa"].to_numpy(np.float64)

    print(f"Cobertura de las variables nuevas sobre las {len(df):,} horas del proyecto:")
    for nom, v in [("disponibilidad_total", disp), ("costo_marginal", cmg), ("precio_escasez", esc)]:
        print(f"  {nom:22s} {100*np.mean(~np.isnan(v)):5.1f}% de las horas")

    # --- variables derivadas con sentido fisico (todas se usaran EN EL CORTE) ---
    margen = disp - dem                       # holgura absoluta
    margen_rel = margen / np.maximum(dem, 1)  # holgura relativa a la demanda
    cerca_esc = precio / np.maximum(esc, 1)   # que tan cerca esta el precio del de escasez
    derivadas = {"margen_reserva": margen, "margen_relativo": margen_rel,
                 "costo_marginal": cmg, "cercania_escasez": cerca_esc}
    print("\nCorrelacion de las derivadas con el precio, en horas PICO vs el resto:")
    hora = idx.hour.to_numpy()
    pk_all = np.isin(hora, PICO)
    for nom, v in derivadas.items():
        ok = ~(np.isnan(v) | np.isnan(precio))
        c_pk = np.corrcoef(v[ok & pk_all], precio[ok & pk_all])[0, 1]
        c_no = np.corrcoef(v[ok & ~pk_all], precio[ok & ~pk_all])[0, 1]
        print(f"  {nom:20s} pico={c_pk:+.3f}  resto={c_no:+.3f}")

    # ---------- modelo de precio con/sin las nuevas, por paso ----------
    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes_tr = idx_00[(idx_00 >= 200) & (idx_00 < idx_fin - H)]
    cortes_te = idx_00[idx_00 >= idx_fin - 1]
    pos = pd.Series(np.arange(len(df)), index=fechas)

    print(f"\n{'='*100}\nMODELO DE 24h CON LAS VARIABLES DE OFERTA (todas medidas EN EL CORTE)\n{'='*100}")
    partes, alphas = [], []
    for h in range(1, H + 1):
        Xtr, ytr, f_tr, c_tr = construir_para_paso(df, h, cortes_tr)
        Xte, yte, f_te, c_te = construir_para_paso(df, h, cortes_te)
        # indices del CORTE de cada fila (no del objetivo): asi no hay fuga
        ictr = pos.reindex(pd.DatetimeIndex(c_tr)).to_numpy()
        icte = pos.reindex(pd.DatetimeIndex(c_te)).to_numpy()

        def bloque(ii):
            b = {}
            for nom, v in derivadas.items():
                b[f"{nom}_corte"] = v[ii]
                b[f"{nom}_corte_24h"] = v[np.maximum(ii - 24, 0)]     # mismo instante, dia anterior
            # margen de reserva a la MISMA HORA del dia anterior (proxy del margen que habra en t)
            return pd.DataFrame(b)

        A, B = bloque(ictr), bloque(icte)
        variantes = {"base (sin variables de oferta)": (Xtr, Xte),
                     "+ margen de reserva y oferta": (pd.concat([Xtr.reset_index(drop=True), A], axis=1),
                                                      pd.concat([Xte.reset_index(drop=True), B], axis=1))}
        fila = {"fecha_hora": f_te, "paso": h, "real": yte}
        for etiqueta, (P, Q) in variantes.items():
            P2, Q2 = P.fillna(P.median()), Q.fillna(P.median())
            esc_ = StandardScaler().fit(P2)
            if h <= 4:
                mm = LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=10000).fit(esc_.transform(P2), ytr)
                alphas.append(mm.alpha_)
            else:
                mm = Lasso(alpha=float(np.median(alphas)), max_iter=10000, random_state=42).fit(
                    esc_.transform(P2), ytr)
            fila[etiqueta] = mm.predict(esc_.transform(Q2))
        partes.append(pd.DataFrame(fila))

    r = pd.concat(partes, ignore_index=True)
    r = r[r["fecha_hora"] >= CORTE].dropna()
    r["hora"] = pd.DatetimeIndex(r["fecha_hora"]).hour
    y = r["real"].to_numpy()
    pk = r["hora"].isin(PICO).to_numpy()
    cols = [c for c in r.columns if c not in ("fecha_hora", "paso", "real", "hora")]
    base = r[cols[0]].to_numpy()
    print(f"{'variante':34s} {'MAE':>8s} {'MAPE %':>8s} {'MAE pico':>10s} {'MAPE pico':>10s}   DM")
    for c in cols:
        p = r[c].to_numpy()
        linea = (f"{c:34s} {np.abs(y-p).mean():8.2f} {mape(y,p):8.2f} "
                 f"{np.abs(y[pk]-p[pk]).mean():10.2f} {mape(y[pk],p[pk]):10.2f}")
        if c != cols[0]:
            t, pv = dm(y, base, p)
            marca = ("MEJORA" if t > 0 else "empeora") if pv < 0.05 else "n.s."
            linea += f"   p={pv:.4f} {marca}"
            t2, pv2 = dm(y[pk], base[pk], p[pk])
            linea += f" | solo pico p={pv2:.4f}"
        print(linea)

    r.to_csv(RES / "margen_reserva_pico_24h.csv", index=False)
    print(f"\nGuardado: margen_reserva_pico_24h.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
