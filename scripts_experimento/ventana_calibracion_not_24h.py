# -*- coding: utf-8 -*-
"""
SELECCION DE VENTANA DE CALIBRACION POR PUNTOS DE CAMBIO (NOT), aplicada al especialista de 24h.

DE DONDE SALE. Nasiadka, Nitka & Weron, "Calibration window selection based on change-point
detection for forecasting electricity prices". La idea rompe con la costumbre del area: en vez de
entrenar con las observaciones MAS RECIENTES (una ventana movil de tau dias), se usa un detector de
puntos de cambio para partir el historico en tramos estacionarios y se entrena SOLO con los tramos
cuyo comportamiento se parece al del momento actual -- aunque sean viejos, y saltandose los que no
se parecen aunque sean recientes.

Algoritmo de seleccion (pasos 1-5 del paper, implementados en not_changepoint.py):
  1. Ventana inicial C0 de tau dias.
  2. NOT detecta Nc <= 12 puntos de cambio en C0.
  3. Si Nc = 0, se usa C0 entera. Si no, se toman los cuantiles q0.025 y q0.975 del tramo entre el
     ULTIMO punto de cambio y el final de C0 (es decir, del regimen vigente).
  4. Para cada tramo entre puntos de cambio consecutivos se calcula su mediana.
  5. Se conservan los tramos cuya mediana cae dentro de esos cuantiles.

POR QUE ESTE EXPERIMENTO Y NO OTRO. Los autores cierran el paper diciendo, textualmente, que si
esto sirve para modelos mas complejos que el autorregresivo que ellos usan, "e.g., LASSO-estimated
AR (LEAR) and deep neural networks, is left for future work". Este proyecto tiene exactamente un
LEAR de 24h. O sea que la pregunta que ellos dejan abierta se puede contestar aqui.

EXPECTATIVA CALIBRADA DE ANTEMANO (para no venderse humo). En el mercado aleman la ganancia
atribuible a NOT es de 7.7286 -> 7.5994 de RMSE, un 1.7%, y los propios autores advierten que
"NOT-selection yields further improvement, although not statistically significant if considered on
its own". La mejora grande de su paper (8.2860 -> 7.7286) viene de OTRA cosa: la transformacion
asinh. Por eso aqui se prueban las dos por separado, para no atribuirle a NOT lo que sea del asinh.

ESTRUCTURA
  Capa 1 - REPLICA. Modelo ARX igual al del paper (Ec. 1), con ventanas fijas tau, con asinh, con
           NOT, y con los promedios de ventanas Av(Win) y Av(NOT). Contesta: ¿el efecto existe en
           el mercado colombiano?
  Capa 2 - LO NUEVO. El mismo juego sobre el LEAR del proyecto. Contesta la pregunta que el paper
           deja abierta.
  Capa 3 - INTEGRACION. Si algo gana, entra al ensamble de 6 votantes y se valida con el estandar
           del proyecto (10 particiones + Diebold-Mariano).

ANTI-FUGA. Para el corte c del dia de prueba, la ventana son los c anteriores y sus objetivos ya
realizados (para el corte c-1 y paso h, el objetivo cae en c-1+h <= c, o sea ya ocurrio). La
deteccion de puntos de cambio solo ve esa ventana. Se verifica explicitamente en la salida.

COSTO. La deteccion de puntos de cambio se rehace cada dia y cada paso (5.232 veces), a ~440 ms
cada una. Como las mascaras NO dependen del modelo, se calculan una vez y se cachean en disco;
todas las variantes de modelo despues son baratas.
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
from lear_24h import construir_para_paso
import not_changepoint as notcp

CORTE = pd.Timestamp("2026-01-01")
TAU_MAX = 728                       # ventana inicial C0 del paper (2 anios)
TAUS = [56, 84, 112, 182, 364, 700, 714, 728]
AV_CORTAS = [56, 84, 112]
AV_LARGAS = [700, 714, 728]
CACHE = RES / "not_mascaras_24h.npz"

ARX_COLS = ["p_obj_lag24", "p_obj_lag48", "p_obj_lag168",      # AR: misma hora a 1, 2 y 7 dias
            "p_corte_lag0",                                    # ultimo precio conocido
            "p_min24_corte", "p_max24_corte",                  # rango de ayer
            "demanda_media_24h_corte", "aportes_hidricos_corte", "volumen_embalses_corte"]
PRECIO_COLS_PREFIJO = ("p_obj_lag", "p_corte_lag", "p_min24", "p_max24", "p_media24")


# --------------------------------------------------------------------------------------- utilidades
def met(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    return {"MAE": float(np.abs(e).mean()),
            "RMSE": float(np.sqrt((e ** 2).mean())),
            "MAPE_%": float(np.mean(np.abs(e) / y) * 100),
            "sMAPE_%": float(np.mean(np.abs(e) / ((np.abs(y) + np.abs(p)) / 2)) * 100)}


def dm(y, p_ref, p_nuevo, maxlags=24):
    d = np.abs(np.asarray(y, float) - p_ref) - np.abs(np.asarray(y, float) - p_nuevo)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def asinh_par(y):
    """
    Parametros de la transformacion asinh: mediana y escala robusta de la ventana de calibracion.

    El paper dice literalmente "b is median absolute deviation". Aqui se usa la MAD NORMALIZADA
    (dividida por 0.6745, el estimador robusto estandar de la desviacion tipica), porque la MAD
    cruda comprime de mas y la inversion con sinh amplifica el error. Medido sobre el ARX:

        tau      MAD cruda    MAD normalizada    sin transformar
        182         56.72          54.12              51.02
        364         56.78          53.73              49.33
        728         53.23          52.30              57.97

    Las dos versiones de asinh dan la misma conclusion (ayuda en ventana larga, estorba en corta);
    la normalizada es uniformemente mejor, asi que es la que se usa.
    """
    a = float(np.median(y))
    b = float(np.median(np.abs(y - a))) / 0.6745
    return a, max(b, 1e-6)


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# ------------------------------------------------------------------------- diseno comun por paso
def armar_disenos():
    """Para cada paso h devuelve X completo, y, fechas objetivo y de corte, y el indice del 1er test."""
    df = cargar_completo().reset_index(drop=True)
    fechas = df["fecha_hora"]
    idx_00 = np.where(fechas.dt.hour.to_numpy() == 0)[0]
    idx_fin = int(pd.Series(df.index.values, index=fechas)[CORTE])
    cortes = idx_00[idx_00 >= 200]                      # todos los cortes utilizables, en orden
    out = {}
    for h in range(1, 25):
        X, y, f_obj, f_cor = construir_para_paso(df, h, cortes)
        X = X.copy()
        dow = pd.DatetimeIndex(f_obj).dayofweek
        for k in range(7):
            X[f"dow{k}"] = (dow == k).astype(float)
        i0 = int(np.searchsorted(pd.DatetimeIndex(f_cor).to_numpy(), np.datetime64(CORTE)))
        out[h] = dict(X=X, y=y, f_obj=pd.DatetimeIndex(f_obj), f_cor=pd.DatetimeIndex(f_cor), i0=i0)
    return out


# ------------------------------------------------------------------- capa 0: mascaras NOT cacheadas
def calcular_mascaras(dis):
    if CACHE.exists():
        z = np.load(CACHE)
        # El cache se escribe tras cada paso, asi que puede quedar a medias si se interrumpe la
        # corrida. Solo se reusa si esta marcado como completo; si no, se recalcula entero.
        listos = int(z["pasos_listos"])
        if listos == 24:
            print(f"Mascaras NOT leidas de {CACHE.name}  {z['cruda'].shape}")
            return z["cruda"], z["asinh"]
        # Quedo a medias (se apago el equipo, se interrumpio la corrida...). No se descarta: se
        # retoma desde el paso siguiente, que son ~1.5 min de calculo por paso que no hay que repetir.
        print(f"  ({CACHE.name} quedo en el paso {listos}/24; se retoma desde el {listos+1})")
        reanudar = (listos, z["cruda"].copy(), z["asinh"].copy())
    else:
        reanudar = (0, None, None)
    # OJO: no todos los pasos tienen el mismo numero de dias de prueba (el paso 24 tiene uno menos,
    # porque su objetivo cae ya fuera del periodo). Se dimensiona con el maximo y se rellena con
    # True (= usar la ventana entera) lo que no se calcule, que nunca se llega a leer.
    n_test = max(len(dis[h]["y"]) - dis[h]["i0"] for h in range(1, 25))
    print(f"Calculando mascaras NOT: {n_test} dias x 24 pasos x 2 series = {n_test*48:,} detecciones")
    print("(no dependen del modelo; se cachean y se reusan en todas las variantes)", flush=True)
    listos, cruda, asnh = reanudar
    if cruda is None:
        cruda = np.ones((n_test, 24, TAU_MAX), dtype=bool)
        asnh = np.ones((n_test, 24, TAU_MAX), dtype=bool)
    t0 = time.time()
    for hi, h in enumerate(range(1, 25)):
        if hi < listos:                      # ya calculado en una corrida anterior
            continue
        y, i0 = dis[h]["y"], dis[h]["i0"]
        for k in range(len(y) - i0):
            i = i0 + k
            v = y[i - TAU_MAX:i]                        # SOLO objetivos ya realizados en el corte
            cruda[k, hi] = notcp.seleccionar_subperiodos(v, semilla=0)
            a, b = asinh_par(v)
            asnh[k, hi] = notcp.seleccionar_subperiodos(np.arcsinh((v - a) / b), semilla=0)
        np.savez_compressed(CACHE, cruda=cruda, asinh=asnh, pasos_listos=hi + 1)  # tras cada paso
        el = time.time() - t0
        print(f"    paso {h:2d}/24 guardado  ({el/60:.1f} min transcurridos, "
              f"~{el/(hi+1)*(24-hi-1)/60:.0f} min restantes)", flush=True)
    print(f"  mascaras completas en {CACHE.name}")
    return cruda, asnh


# --------------------------------------------------------------- capa 1 y 2: pronosticos rodantes
def indices_calibracion(i, ventana, mask_fila):
    """Filas de entrenamiento para el corte i, segun la regla de ventana. Unico punto donde se
    decide que historia se usa, para que la calibracion de alpha y el pronostico no se separen."""
    if ventana == "todo":
        return np.arange(0, i)
    if ventana == "NOT":
        idx = np.arange(i - TAU_MAX, i)[mask_fila]
        return idx if len(idx) >= 60 else np.arange(i - 60, i)
    return np.arange(max(0, i - int(ventana)), i)


def pronosticar(dis, mask_cruda, mask_asinh, familia, transf, ventana, alphas=None):
    """
    familia: 'ARX' (OLS sobre ARX_COLS) o 'LEAR' (LASSO sobre el diseno completo)
    transf : 'ninguna' o 'asinh'
    ventana: entero tau (ventana movil), 'NOT' (subperiodos NOT de 728 dias), o 'todo'
    """
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    preds, reales, fechas = [], [], []
    for hi, h in enumerate(range(1, 25)):
        d = dis[h]
        cols = ARX_COLS + [f"dow{k}" for k in range(7)] if familia == "ARX" else list(d["X"].columns)
        Xf = d["X"][cols].to_numpy(float)
        y, i0 = d["y"], d["i0"]
        es_precio = np.array([c.startswith(PRECIO_COLS_PREFIJO) for c in cols])
        n_test = len(y) - i0

        for k in range(n_test):
            i = i0 + k
            fila = (mask_asinh if transf == "asinh" else mask_cruda)[k, hi]
            idx = indices_calibracion(i, ventana, fila)
            Xtr, ytr = Xf[idx], y[idx]
            Xte = Xf[i:i + 1]
            if transf == "asinh":
                a, b = asinh_par(ytr)
                ytr = np.arcsinh((ytr - a) / b)
                Xtr = Xtr.copy(); Xte = Xte.copy()
                Xtr[:, es_precio] = np.arcsinh((Xtr[:, es_precio] - a) / b)
                Xte[:, es_precio] = np.arcsinh((Xte[:, es_precio] - a) / b)

            if familia == "ARX":
                beta = ols(Xtr, ytr)
                p = float((Xte @ beta)[0])
            else:
                sc = StandardScaler().fit(Xtr)
                m_ = Lasso(alpha=alphas[h], max_iter=5000, random_state=42).fit(sc.transform(Xtr), ytr)
                p = float(m_.predict(sc.transform(Xte))[0])
            if transf == "asinh":
                p = a + b * np.sinh(np.clip(p, -30, 30))
            preds.append(p); reales.append(y[i]); fechas.append(d["f_obj"][i])
    return pd.DataFrame({"fecha_hora": fechas, "real": reales, "pred": preds}).sort_values("fecha_hora")


def calibrar_alphas(dis, ventana, transf):
    """
    alpha del LASSO calibrado CON LA MISMA REGLA DE VENTANA con la que se va a pronosticar.

    Hace falta hacerlo por variante y no una sola vez: alpha no es invariante de escala, asi que
    un alpha calibrado sobre precios crudos aplicado a datos asinh anula todos los coeficientes
    (se midio: MAE 216 en vez de 56). Y un alpha calibrado sobre 2.517 dias no es el adecuado para
    una ventana de 728. Se calibra en el ultimo corte de entrenamiento, que es informacion
    disponible antes del periodo de prueba, y se usa la mediana entre pasos como hace el resto del
    proyecto.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    vals = []
    for h in [1, 2, 3, 4, 6, 12, 18, 24]:
        d = dis[h]
        i = d["i0"]                                  # ultimo corte de entrenamiento (exclusivo)
        fila = notcp.seleccionar_subperiodos(
            np.arcsinh((d["y"][i - TAU_MAX:i] - asinh_par(d["y"][i - TAU_MAX:i])[0])
                       / asinh_par(d["y"][i - TAU_MAX:i])[1]) if transf == "asinh"
            else d["y"][i - TAU_MAX:i], semilla=0) if ventana == "NOT" else None
        idx = indices_calibracion(i, ventana, fila)
        Xtr = d["X"].to_numpy(float)[idx]
        ytr = d["y"][idx]
        if transf == "asinh":
            a, b = asinh_par(ytr)
            ytr = np.arcsinh((ytr - a) / b)
            es_precio = np.array([c.startswith(PRECIO_COLS_PREFIJO) for c in d["X"].columns])
            Xtr = Xtr.copy()
            Xtr[:, es_precio] = np.arcsinh((Xtr[:, es_precio] - a) / b)
        sc = StandardScaler().fit(Xtr)
        vals.append(LassoCV(cv=5, random_state=42, n_jobs=1, max_iter=8000)
                    .fit(sc.transform(Xtr), ytr).alpha_)
    return {h: float(np.median(vals)) for h in range(1, 25)}


# --------------------------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    if not notcp.autoprueba():
        print("El detector no paso la autoprueba. Se aborta antes de reportar cualquier numero.")
        return

    print("Armando disenos por paso...", flush=True)
    dis = armar_disenos()
    d1 = dis[1]
    print(f"  cortes totales {len(d1['y']):,} | primer corte de prueba "
          f"{d1['f_cor'][d1['i0']].date()} | dias de prueba {len(d1['y'])-d1['i0']}")
    print(f"  ANTI-FUGA: para el corte de prueba i, la ventana son los indices [i-{TAU_MAX}, i), "
          f"cuyos objetivos ya ocurrieron en el corte.\n")

    mc, ma = calcular_mascaras(dis)
    print(f"  seleccion media de NOT: cruda {100*mc.mean():.0f}% de los 728 dias, "
          f"asinh {100*ma.mean():.0f}%\n")

    resultados, guardados = {}, {}

    print("=" * 104)
    print("CAPA 1 - REPLICA CON EL MODELO ARX DEL PAPER")
    print("=" * 104)
    for transf, etq in [("ninguna", "Win"), ("asinh", "WinH")]:
        for tau in TAUS:
            nom = f"ARX {etq}({tau})"
            r = pronosticar(dis, mc, ma, "ARX", transf, tau)
            guardados[nom] = r; resultados[nom] = met(r["real"], r["pred"])
            print(f"  {nom:22s} " + "  ".join(f"{k}={v:8.3f}" for k, v in resultados[nom].items()), flush=True)
    for transf, etq in [("ninguna", "NOT"), ("asinh", "NOTH")]:
        nom = f"ARX {etq}({TAU_MAX})"
        r = pronosticar(dis, mc, ma, "ARX", transf, "NOT")
        guardados[nom] = r; resultados[nom] = met(r["real"], r["pred"])
        print(f"  {nom:22s} " + "  ".join(f"{k}={v:8.3f}" for k, v in resultados[nom].items()), flush=True)

    # promedios de ventanas
    def promedio(nombres, etiqueta):
        base = guardados[nombres[0]][["fecha_hora", "real"]].copy()
        base["pred"] = np.mean([guardados[n]["pred"].to_numpy() for n in nombres], axis=0)
        guardados[etiqueta] = base; resultados[etiqueta] = met(base["real"], base["pred"])
        print(f"  {etiqueta:22s} " + "  ".join(f"{k}={v:8.3f}" for k, v in resultados[etiqueta].items()))

    promedio([f"ARX Win({t})" for t in AV_CORTAS + AV_LARGAS], "ARX Av(Win)")
    promedio([f"ARX WinH({t})" for t in AV_CORTAS + AV_LARGAS], "ARX Av(WinH)")
    promedio([f"ARX WinH({t})" for t in AV_CORTAS] + [f"ARX NOTH({TAU_MAX})"] * 3, "ARX Av(NOTH)")

    print("\n  PRUEBAS DE DIEBOLD-MARIANO (la pregunta del paper, aislada):")
    for a, b in [(f"ARX Win({TAU_MAX})", f"ARX WinH({TAU_MAX})"),
                 (f"ARX WinH({TAU_MAX})", f"ARX NOTH({TAU_MAX})"),
                 (f"ARX Win({TAU_MAX})", f"ARX NOT({TAU_MAX})"),
                 ("ARX Av(WinH)", "ARX Av(NOTH)")]:
        t_, p_ = dm(guardados[a]["real"], guardados[a]["pred"], guardados[b]["pred"])
        signo = "mejora" if resultados[b]["MAE"] < resultados[a]["MAE"] else "empeora"
        print(f"    {b:22s} vs {a:22s}: {signo}, t={t_:+6.2f}  p={p_:.4f}")

    print("\n" + "=" * 104)
    print("CAPA 2 - LO QUE EL PAPER DEJA ABIERTO: ¿SIRVE SOBRE EL LEAR?")
    print("=" * 104)
    for transf, etq, ven in [("ninguna", f"LEAR Win({TAU_MAX})", TAU_MAX),
                             ("ninguna", "LEAR Win(todo)", "todo"),
                             ("ninguna", f"LEAR NOT({TAU_MAX})", "NOT"),
                             ("asinh", f"LEAR WinH({TAU_MAX})", TAU_MAX),
                             ("asinh", f"LEAR NOTH({TAU_MAX})", "NOT")]:
        alphas = calibrar_alphas(dis, ven, transf)
        r = pronosticar(dis, mc, ma, "LEAR", transf, ven, alphas)
        guardados[etq] = r; resultados[etq] = met(r["real"], r["pred"])
        print(f"  {etq:22s} " + "  ".join(f"{k}={v:8.3f}" for k, v in resultados[etq].items())
              + f"   (alpha={alphas[1]:.4g})", flush=True)

    print("\n  PRUEBAS DE DIEBOLD-MARIANO:")
    for a, b in [(f"LEAR Win({TAU_MAX})", f"LEAR NOT({TAU_MAX})"),
                 (f"LEAR Win({TAU_MAX})", f"LEAR WinH({TAU_MAX})"),
                 (f"LEAR WinH({TAU_MAX})", f"LEAR NOTH({TAU_MAX})"),
                 ("LEAR Win(todo)", f"LEAR NOT({TAU_MAX})")]:
        t_, p_ = dm(guardados[a]["real"], guardados[a]["pred"], guardados[b]["pred"])
        signo = "mejora" if resultados[b]["MAE"] < resultados[a]["MAE"] else "empeora"
        print(f"    {b:22s} vs {a:22s}: {signo}, t={t_:+6.2f}  p={p_:.4f}")

    guardar_parcial(resultados, guardados)

    print("\n" + "=" * 104)
    print("CAPA 3 - INTEGRACION AL ENSAMBLE DE 6 VOTANTES")
    print("=" * 104)
    try:
        integrar(guardados, resultados)
    except Exception as e:                       # la capa 3 no debe tumbar lo ya calculado
        print(f"  (no se pudo completar la integracion: {type(e).__name__}: {e})")

    tabla = guardar_parcial(resultados, guardados)
    print("\n" + "=" * 104)
    print("RESUMEN ORDENADO POR MAE")
    print("=" * 104)
    print(tabla.round(3).to_string())
    print(f"\nGuardado: ventana_calibracion_not_24h.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


def guardar_parcial(resultados, guardados):
    tabla = pd.DataFrame(resultados).T.sort_values("MAE")
    tabla.to_csv(RES / "ventana_calibracion_not_24h.csv")
    for nom, r in guardados.items():
        if nom.startswith("LEAR") or "NOT" in nom:
            seguro = nom.replace(" ", "_").replace("(", "").replace(")", "")
            r.to_csv(RES / f"vc_{seguro}.csv", index=False)
    return tabla


def integrar(guardados, resultados):
    """El mejor candidato de ventana entra como VOTANTE ADICIONAL y se valida con el estandar del
    proyecto: 10 particiones del combinador + Diebold-Mariano sobre la perdida de cada metrica."""
    from stacking_24h_v2 import cargar as cargar24, BASE5
    from combinador_optimo_mape import qra_cv, mape, smape

    cands = [n for n in guardados if n.startswith("LEAR") or "NOT" in n]
    ganador = min(cands, key=lambda n: resultados[n]["MAPE_%"])
    print(f"  candidato con mejor MAPE entre las variantes de ventana: {ganador} "
          f"({resultados[ganador]['MAPE_%']:.2f}%)")

    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    v = guardados[ganador].drop_duplicates("fecha_hora").set_index("fecha_hora")["pred"].rename("VentanaNOT")
    d = d.join(v, how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["VentanaNOT", "real"])
    y = d["real"].to_numpy()
    print(f"  filas alineadas: {len(d):,}")

    base = qra_cv(d, MEJOR, "smape")
    nuevo = qra_cv(d, MEJOR + ["VentanaNOT"], "smape")
    print(f"  {'ensamble actual (6 votantes)':32s} MAE={np.abs(y-base).mean():7.2f}  "
          f"MAPE={mape(y,base):6.2f}%  sMAPE={smape(y,base):6.2f}%")
    print(f"  {'+ ventana por puntos de cambio':32s} MAE={np.abs(y-nuevo).mean():7.2f}  "
          f"MAPE={mape(y,nuevo):6.2f}%  sMAPE={smape(y,nuevo):6.2f}%  "
          f"(DM p={dm(y, base, nuevo)[1]:.4f})")

    A, B = [], []
    for s_ in range(10):
        pa, pb = qra_cv(d, MEJOR, "smape", s_), qra_cv(d, MEJOR + ["VentanaNOT"], "smape", s_)
        A.append([np.abs(y - pa).mean(), mape(y, pa), smape(y, pa)])
        B.append([np.abs(y - pb).mean(), mape(y, pb), smape(y, pb)])
    A, B = np.array(A), np.array(B)
    print("\n  VERIFICACION 10 PARTICIONES:")
    for j, nm in enumerate(["MAE", "MAPE %", "sMAPE %"]):
        dl = B[:, j] - A[:, j]
        print(f"    {nm:8s} actual={A[:,j].mean():7.3f}+/-{A[:,j].std():.3f}  "
              f"con ventana={B[:,j].mean():7.3f}+/-{B[:,j].std():.3f}  "
              f"delta={dl.mean():+7.3f}, mejor en {int((dl<0).sum())}/10")
    solido = all((B[:, j] - A[:, j] < 0).all() for j in range(3))
    print(f"    VEREDICTO: {'SOLIDO' if solido else 'no concluyente'}")


if __name__ == "__main__":
    main()
