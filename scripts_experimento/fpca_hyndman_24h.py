# -*- coding: utf-8 -*-
"""
ANALISIS FUNCIONAL DE LA CURVA DIARIA (Hyndman-Ullah-Shang), para atacar la FORMA intradia.

POR QUE ESTA FAMILIA Y NO OTRA. El diagnostico (`diagnostico_forma_intradia.py`) dejo tres cosas
fijadas sobre el ensamble actual en 2026:

    el 98% del error esta en la FORMA del dia, no en el nivel
    no hay desfase temporal (el mejor desplazamiento de la curva es 0 h)
    la curva pronosticada esta aplanada (razon de amplitud 0.72)

y la calibracion de amplitud (`calibracion_amplitud_intradia.py`) mostro que ese aplanamiento NO se
puede arreglar por post-proceso: el factor optimo en metrica es ~0.98, no 1.38. El encogimiento es
la respuesta estadisticamente correcta a una forma que solo se conoce con correlacion 0.78. Para
mejorar las rampas hace falta INFORMACION DE FORMA MEJOR, no reescalar la que ya hay.

De ahi esta familia: en vez de tratar las 24 horas como 24 problemas independientes (que es lo que
hace el LEAR y, de hecho, casi todo el ensamble), se trata el dia como UNA CURVA y se modela su
forma directamente.

EL METODO. Gallon & Barrientos (2021, IJEEP 11(2), 67-74) aplican al mercado COLOMBIANO el enfoque
de Hyndman & Ullah (2007) y Hyndman & Shang (2009), basado en la descomposicion de Karhunen-Loeve:

    f_t(x) = mu(x) + suma_k beta_{t,k} * phi_k(x) + e_t(x)

  mu(x)    funcion media, estimada con media ponderada que da mas peso a los dias recientes
  phi_k    componentes principales funcionales (base ortonormal de FORMAS del dia)
  beta_tk  scores: cuanto pesa cada forma en el dia t

Se obtienen por SVD de la matriz de curvas centradas (el "enfoque por discretizacion" que ellos
usan, apropiado aqui porque las 24 horas ya son una rejilla densa y regular). Pronosticar el dia de
manana = pronosticar los K scores y reconstruir. Ellos ajustan una serie funcional POR DIA DE LA
SEMANA y eligen K por el metodo de Shang (2013); aqui se prueban ambas cosas en vez de fijarlas.

TRES FORMAS DE PRONOSTICAR LOS SCORES
  (a) 'media'  media ponderada de los scores recientes. Es la version mas parecida a una
               climatologia de formas, y sirve de piso.
  (b) 'ar'     autorregresivo sobre la serie de cada score (lo que hacen Hyndman-Ullah-Shang).
  (c) 'exog'   Ridge sobre el ESTADO DEL SISTEMA en el corte (embalses, aportes, ONI, precios
               recientes, calendario). Es una extension propia: el metodo original es univariante
               sobre las curvas, y aqui hay hidrologia que deberia informar la forma del dia.

QUE SE ESPERA, HONESTAMENTE. El ensamble ya le gana a una climatologia de formas por 38%, asi que
un modelo funcional puro dificilmente lo supere por si solo. El valor esperado esta en que aporta
una FAMILIA que el ensamble no tiene -- un modelo de forma, no de niveles hora a hora -- y la
leccion repetida del proyecto es que lo que hace ganar al ensamble es la diversidad entre familias.
Por eso se evalua como votante adicional, y sobre todo se mira si mejora EN LAS RAMPAS.

ANTI-FUGA. Para el corte c solo se usan curvas de ventanas anteriores, cuyos precios ya estan todos
realizados en c. La media ponderada, la SVD y el modelo de scores se reajustan en cada corte.
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
from stacking_24h_v2 import cargar as cargar24, BASE5
from combinador_optimo_mape import qra_cv, mape, smape

CORTE = pd.Timestamp("2026-01-01")
FIN = pd.Timestamp("2026-08-05")
ORDEN = list(range(1, 24)) + [0]          # ventana de pronostico 01:00 -> 00:00 del dia siguiente
SEMILLAS = list(range(10))
PICO = [0, 8, 9, 10, 18, 19]
MEDIA_VIDA = 180.0                        # dias; peso exponencial de la media ponderada
MIN_HIST = 120                            # ventanas previas minimas para ajustar

ESTADO = ["volumen_embalses", "volumen_embalses_vs_media30d", "aportes_hidricos",
          "aportes_hidricos_vs_media30d", "demanda_media_24h", "precio_media_24h",
          "precio_media_7d", "precio_media_30d", "precio_std_7d", "oni"]


def dm(y, p_ref, p_nuevo, metrica="mae", maxlags=24):
    y = np.asarray(y, float)
    if metrica == "smape":
        l1 = np.abs(y - p_ref) / ((np.abs(y) + np.abs(p_ref)) / 2)
        l2 = np.abs(y - p_nuevo) / ((np.abs(y) + np.abs(p_nuevo)) / 2)
    else:
        l1, l2 = np.abs(y - p_ref), np.abs(y - p_nuevo)
    d = l1 - l2
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.pvalues[0])


def armar_curvas(solo_forma=True):
    """
    Matriz de curvas diarias (una fila por ventana 01:00->00:00) y el estado en cada corte.

    Con `solo_forma`, cada curva se centra en su propia media: la matriz queda con la FORMA del dia
    y el nivel se saca aparte. Esto NO es un detalle: la primera version modelaba la curva completa
    (nivel incluido, que es lo que hace el metodo original) y daba MAE 210, porque el nivel del
    precio en Colombia no es estacionario y la media ponderada historica llega tarde. Y ademas es lo
    que dice el diagnostico propio: el 98% del error esta en la forma, asi que pedirle al modelo
    funcional que tambien acierte el nivel es cargarlo con el problema equivocado.
    """
    df = cargar_completo()
    df["ventana"] = (df["fecha_hora"] - pd.Timedelta(hours=1)).dt.normalize()
    df["hora"] = df["fecha_hora"].dt.hour
    comp = df.groupby("ventana")["precio_bolsa"].count()
    df = df[df["ventana"].isin(comp[comp == 24].index)]
    C = df.pivot_table(index="ventana", columns="hora", values="precio_bolsa")[ORDEN]
    C = C.dropna()
    nivel = C.mean(axis=1)
    if solo_forma:
        C = C.sub(nivel, axis=0)
    # estado en el corte de cada ventana = fila de las 00:00 de esa fecha
    est = df[df["hora"] == 0].set_index(df.loc[df["hora"] == 0, "fecha_hora"].dt.normalize())
    cols = [c for c in ESTADO if c in est.columns]
    E = est[cols].reindex(C.index)
    return C, E.ffill(), nivel


def pronosticar_fpca(C, E, K, modo, por_dia_semana, ridge_alpha=10.0):
    """Un pronostico por ventana de prueba. Devuelve DataFrame largo (fecha_hora, real, pred)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    fechas = C.index
    M = C.to_numpy(float)
    Xe = E.to_numpy(float)
    dow = fechas.dayofweek.to_numpy()
    test = np.where((fechas >= CORTE) & (fechas <= FIN))[0]

    filas = []
    for i in test:
        # --- historia usable: solo ventanas anteriores (sus precios ya ocurrieron en el corte)
        hist = np.arange(0, i)
        if por_dia_semana:
            hist = hist[dow[hist] == dow[i]]
        if len(hist) < MIN_HIST // (7 if por_dia_semana else 1):
            continue
        H = M[hist]

        # --- funcion media ponderada (mas peso a lo reciente), como Hyndman & Shang
        edad = (fechas[i] - fechas[hist]).days.to_numpy(float)
        w = 0.5 ** (edad / MEDIA_VIDA)
        w = w / w.sum()
        mu = (w[:, None] * H).sum(0)

        # --- base de formas por SVD de las curvas centradas
        Z = H - mu
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        k = min(K, Vt.shape[0])
        phi = Vt[:k]                       # (k x 24) componentes = formas del dia
        B = Z @ phi.T                      # (n x k) scores historicos

        # --- pronostico de los scores
        if modo == "media":
            b = (w[:, None] * B).sum(0)
        elif modo == "ar":
            b = np.zeros(k)
            for j in range(k):
                y = B[:, j]
                p = 7
                if len(y) <= p + 10:
                    b[j] = float(y[-5:].mean())
                    continue
                Xl = np.column_stack([y[p - l - 1:len(y) - l - 1] for l in range(p)] +
                                     [np.ones(len(y) - p)])
                coef, *_ = np.linalg.lstsq(Xl, y[p:], rcond=None)
                b[j] = float(np.r_[y[-1:-p - 1:-1], 1.0] @ coef)
        else:                              # exog: los scores en funcion del estado del sistema
            ok = ~np.isnan(Xe[hist]).any(1)
            if ok.sum() < 60 or np.isnan(Xe[i]).any():
                b = (w[:, None] * B).sum(0)
            else:
                sc = StandardScaler().fit(Xe[hist][ok])
                Zt = sc.transform(Xe[hist][ok])
                b = np.array([float(Ridge(alpha=ridge_alpha).fit(Zt, B[ok, j])
                                    .predict(sc.transform(Xe[i:i + 1]))[0]) for j in range(k)])

        curva = mu + b @ phi
        for j, h in enumerate(ORDEN):
            ts = fechas[i] + pd.Timedelta(hours=h if h != 0 else 24)
            filas.append((ts, M[i, j], curva[j]))
    return pd.DataFrame(filas, columns=["fecha_hora", "real", "pred"])




def pronosticar_nivel(nivel, E, fechas_test):
    """
    Nivel medio del dia, pronosticado aparte y de forma causal.

    Hace falta porque el modelo funcional ahora solo produce FORMA. Se usa una Ridge sobre el estado
    del sistema en el corte mas los niveles recientes; es deliberadamente simple, porque el nivel no
    es donde esta el problema y lo unico que se necesita es que no lo estropee.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    idx = nivel.index
    v = nivel.to_numpy(float)
    Xe = E.to_numpy(float)
    rez = np.column_stack([np.roll(v, l) for l in (1, 2, 3, 7, 14)])
    X = np.column_stack([Xe, rez])
    out = pd.Series(index=idx, dtype=float)
    pos = {f: i for i, f in enumerate(idx)}
    for f in fechas_test:
        i = pos[f]
        hist = np.arange(14, i)
        ok = ~np.isnan(X[hist]).any(1)
        if ok.sum() < 120 or np.isnan(X[i]).any():
            out[f] = v[i - 1]
            continue
        sc = StandardScaler().fit(X[hist][ok])
        m = Ridge(alpha=10.0).fit(sc.transform(X[hist][ok]), v[hist][ok])
        out[f] = float(m.predict(sc.transform(X[i:i + 1]))[0])
    return out


def met(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(MAE=float(np.abs(y - p).mean()), MAPE=mape(y, p), sMAPE=smape(y, p))


def main():
    t0 = time.time()
    C, E, nivel = armar_curvas(solo_forma=True)
    fechas_test = C.index[(C.index >= CORTE) & (C.index <= FIN)]
    print(f"Curvas diarias: {len(C)}  ({C.index.min().date()} a {C.index.max().date()})")
    print(f"Ventanas de prueba: {len(fechas_test)}")
    print("La matriz esta CENTRADA por dia: el modelo funcional pronostica FORMA, no nivel.\n")

    # ---------------------------------------------------------------- referencia: forma del ensamble
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv",
                    parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["real"])
    d["hora"] = d.index.hour
    d["ventana"] = (d.index - pd.Timedelta(hours=1)).normalize()
    comp = d.groupby("ventana")["real"].count()
    d = d[d["ventana"].isin(comp[comp == 24].index)].copy()
    d["pred"] = qra_cv(d, MEJOR, "smape")
    Yv = d.pivot_table(index="ventana", columns="hora", values="real")[ORDEN]
    Pv = d.pivot_table(index="ventana", columns="hora", values="pred")[ORDEN]
    comun = Yv.index.intersection(fechas_test)
    Yc = Yv.loc[comun].to_numpy()
    Fy = Yc - Yc.mean(1, keepdims=True)
    Pc = Pv.loc[comun].to_numpy()
    Fp = Pc - Pc.mean(1, keepdims=True)
    mae_forma_ens = float(np.abs(Fy - Fp).mean())
    col_pico = [ORDEN.index(h) for h in PICO]
    rampa = np.abs(np.diff(Yc, axis=1, prepend=Yc[:, :1]))
    fu = rampa >= 50

    print("=" * 104)
    print("LA PREGUNTA CENTRAL: ¿da el modelo funcional MEJOR FORMA que el ensamble?")
    print("=" * 104)
    print(f"  referencia -- MAE de forma del ENSAMBLE actual                {mae_forma_ens:8.2f}")
    print(f"  referencia -- MAE de forma de la CLIMATOLOGIA (hora x dow x mes)   65.11")
    print(f"  (el ensamble en rampa fuerte: {np.abs(Fy - Fp)[fu].mean():.2f} | "
          f"en las 6 horas pico: {np.abs(Fy[:, col_pico] - Fp[:, col_pico]).mean():.2f})\n")
    print(f"  {'variante funcional':28s} {'MAE forma':>10s} {'vs ensamble':>12s} "
          f"{'en rampa':>10s} {'en pico':>9s}")

    filas, guardado = [], {}
    for K in [1, 2, 3, 4, 6]:
        for modo in ["media", "ar", "exog"]:
            for pds in [False, True]:
                r = pronosticar_fpca(C, E, K, modo, pds)
                if len(r) == 0:
                    continue
                ts = pd.DatetimeIndex(r["fecha_hora"])
                vent = ts.normalize() - pd.to_timedelta((ts.hour == 0).astype(int), unit="D")
                Ff = pd.DataFrame({"v": vent, "h": ts.hour, "p": r["pred"].to_numpy()}) \
                    .pivot_table(index="v", columns="h", values="p")
                Ff = Ff.reindex(index=comun, columns=ORDEN)
                if Ff.isna().any().any():
                    continue
                F = Ff.to_numpy()
                m = float(np.abs(Fy - F).mean())
                nom = f"K={K} {modo}{' x dow' if pds else ''}"
                guardado[nom] = F
                filas.append(dict(variante=nom, mae_forma=m,
                                  mae_rampa=float(np.abs(Fy - F)[fu].mean()),
                                  mae_pico=float(np.abs(Fy[:, col_pico] - F[:, col_pico]).mean())))
                print(f"  {nom:28s} {m:10.2f} {100*(m-mae_forma_ens)/mae_forma_ens:+11.1f}% "
                      f"{np.abs(Fy - F)[fu].mean():10.2f} "
                      f"{np.abs(Fy[:, col_pico] - F[:, col_pico]).mean():9.2f}", flush=True)

    T = pd.DataFrame(filas).sort_values("mae_forma")
    T.to_csv(RES / "fpca_hyndman_forma.csv", index=False)
    if len(T) == 0:
        print("  (ninguna variante produjo salida utilizable)")
        return
    mejores = T.head(3)["variante"].tolist()
    print(f"\n  Mejores en forma: {mejores}")

    # ---------------------------------------------------------------- votante completo
    print("\n" + "=" * 104)
    print("COMO VOTANTE ADICIONAL  (forma funcional + nivel pronosticado aparte)")
    print("=" * 104)
    niv_hat = pronosticar_nivel(nivel, E, comun)
    print(f"  MAE del nivel diario pronosticado: "
          f"{float(np.abs(nivel.loc[comun] - niv_hat.loc[comun]).mean()):.2f}")

    serie = cargar_completo().set_index("fecha_hora")["precio_bolsa"]
    for nom in mejores:
        curva = guardado[nom] + niv_hat.loc[comun].to_numpy()[:, None]
        largo = []
        for a, f in enumerate(comun):
            for j, h in enumerate(ORDEN):
                largo.append((f + pd.Timedelta(hours=h if h != 0 else 24), curva[a, j]))
        v = pd.DataFrame(largo, columns=["fecha_hora", "pred"]).drop_duplicates("fecha_hora")
        v.to_csv(RES / f"fpca_votante_{nom.replace(' ', '_').replace('=', '')}.csv", index=False)
        d = d.join(v.set_index("fecha_hora")["pred"].rename(f"FPCA {nom}"), how="left")

    cols_f = [f"FPCA {n}" for n in mejores]
    d2 = d.dropna(subset=MEJOR + cols_f + ["real"])
    y = d2["real"].to_numpy()
    prev = serie.reindex(d2.index - pd.Timedelta(hours=1))
    prev.index = d2.index
    ramp = (d2["real"] - prev).abs().to_numpy()
    fuv, pkv = ramp >= 50, np.isin(d2.index.hour, PICO)
    A = np.array([[np.abs(y - p).mean(), mape(y, p), smape(y, p)]
                  for p in (qra_cv(d2, MEJOR, "smape", s_) for s_ in SEMILLAS)])
    base = qra_cv(d2, MEJOR, "smape")
    print(f"\n  filas alineadas: {len(d2):,}")
    print(f"  {'candidato':26s} {'MAE suelto':>11s} {'dMAE ens':>9s} {'gana':>6s} {'dMAPE':>8s} "
          f"{'gana':>6s} {'MAE rampa':>10s} {'MAE pico':>9s}")
    print(f"  {'ensamble actual':26s} {'':>11s} {'':>9s} {'':>6s} {'':>8s} {'':>6s} "
          f"{np.abs(y[fuv] - base[fuv]).mean():10.2f} {np.abs(y[pkv] - base[pkv]).mean():9.2f}")
    for nom, c in zip(mejores, cols_f):
        B = np.array([[np.abs(y - p).mean(), mape(y, p), smape(y, p)]
                      for p in (qra_cv(d2, MEJOR + [c], "smape", s_) for s_ in SEMILLAS)])
        dl = B - A
        nuevo = qra_cv(d2, MEJOR + [c], "smape")
        print(f"  {c:26s} {np.abs(y - d2[c].to_numpy()).mean():11.2f} {dl[:, 0].mean():+9.3f} "
              f"{int((dl[:, 0] < 0).sum()):4d}/10 {dl[:, 1].mean():+8.3f} "
              f"{int((dl[:, 1] < 0).sum()):4d}/10 "
              f"{np.abs(y[fuv] - nuevo[fuv]).mean():10.2f} "
              f"{np.abs(y[pkv] - nuevo[pkv]).mean():9.2f}", flush=True)

    print(f"\nGuardado: fpca_hyndman_forma.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
