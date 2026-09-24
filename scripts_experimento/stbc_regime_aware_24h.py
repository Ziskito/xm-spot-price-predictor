# -*- coding: utf-8 -*-
"""
RA-STBC: compensacion de sesgo de corto plazo condicionada al regimen (Singh, 2027; Chang et al., 2024).

CORRECCION DE UNA PRUEBA MAL HECHA. En `limite_informacion_rampas.py` se dijo haber probado el
metodo STBC y se concluyo que fallaba. **Esa prueba no era STBC.** Ahi se entreno un modelo de
aprendizaje (Ridge, GradBoost) para predecir el error del ensamble a partir de las 62 variables de
estado en el corte. Es una pregunta distinta y mas ambiciosa: "¿es el error predecible a partir del
estado del sistema?". La respuesta a ESA pregunta sigue siendo no (R2 <= 0), y ese resultado se
mantiene. Pero no es lo que hace el paper.

QUE HACE DE VERDAD EL METODO (Singh 2027, seccion 4.5, siguiendo a Chang et al. 2024):

    p_t = real_t - predicho_t                       error de prediccion en cada paso
    B_t = (1/k) * suma de los ultimos k errores     sesgo local, con k = 10
    corregido_t = predicho_t + lambda * B_t         se corrige en la direccion del sesgo reciente

Sin variables explicativas y sin modelo aprendido. Es solo la MEDIA MOVIL DE LOS ULTIMOS ERRORES.
La pregunta que responde es otra: "¿esta el error AUTOCORRELACIONADO a corto plazo?". Puede haber
autocorrelacion del error aunque el error no sea predecible desde el estado, asi que hay que
probarlo aparte.

LA APORTACION DEL PAPER sobre Chang et al. es hacer lambda dependiente del REGIMEN, detectado con
una mezcla de gaussianas (GMM de 3 componentes) sobre retorno y volatilidad estandarizados,
etiquetadas por volatilidad creciente. Sus valores: lambda = 0.2 (calma), 0.5 (intermedio), 0.8
(crisis). Su argumento es que en crisis los errores son mayores y mas persistentes, asi que admiten
correccion mas agresiva.

ADAPTACION A ESTE PROYECTO. Aqui el pronostico es day-ahead de 24 horas de golpe, no de un paso. El
sesgo local se estima POR HORA DEL RELOJ: para la hora h del dia d, se usan los errores de la hora h
de los k dias anteriores, todos realizados antes del corte de las 00:00 del dia d. Es causal por
construccion y se verifica en la salida.

EXPECTATIVA CALIBRADA, LEYENDO EL PAPER CON CUIDADO. Sus numeros piden cautela:
  - El STBC de lambda fijo alcanza 92.27% de reduccion de MSE, pero la exactitud direccional se
    queda en 49.23% (el azar) y el rendimiento de cartera es -16.39%.
  - Pronostican NIVEL de precio de un indice bursatil, que es casi un paseo aleatorio. Corregir
    hacia los errores recientes acerca el pronostico al ultimo valor observado, o sea que empuja
    hacia la PERSISTENCIA. Eso explica que el MSE se desplome sin que mejore la direccion.
  - Su propia conclusion lo admite: "the primary contribution ... is in price-level forecasting
    accuracy", no en señal.
En este proyecto la persistencia ya es uno de los seis votantes del ensamble y el combinador le
asigna el peso que merece. Asi que la hipotesis realista es que STBC aporte poco: seria redescubrir
un votante que ya esta dentro. Se prueba igual, porque es barato y porque la pregunta de la
autocorrelacion del error no se habia hecho.

PROTOCOLO. lambda se elige por busqueda en rejilla sobre un periodo de VALIDACION anterior al de
prueba (como hace el paper con su held-out), nunca sobre el periodo donde se reporta. Se mide el
efecto en las rampas, que es el objetivo real.
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

SEMILLAS = list(range(10))
PICO = [0, 8, 9, 10, 18, 19]
K_VENTANA = 10                      # k del paper: 10 pasos
REJILLA_L = np.round(np.arange(0.0, 1.01, 0.05), 2)
FIN_VALID = pd.Timestamp("2026-04-01")      # validacion: ene-mar 2026; prueba: abr-ago 2026


def dm(y, p_ref, p_nuevo, metrica="mae", maxlags=24):
    y = np.asarray(y, float)
    if metrica == "smape":
        l1 = np.abs(y - p_ref) / ((np.abs(y) + np.abs(p_ref)) / 2)
        l2 = np.abs(y - p_nuevo) / ((np.abs(y) + np.abs(p_nuevo)) / 2)
    else:
        l1, l2 = np.abs(y - p_ref), np.abs(y - p_nuevo)
    d = l1 - l2
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": min(maxlags, len(d)//3)})
    return float(r.pvalues[0])


def sesgo_local(df, k=K_VENTANA):
    """
    B_t por hora del reloj: media de los k errores anteriores de ESA hora.

    Causal: para el objetivo del dia d a la hora h, el ultimo error usable es el del dia d-1 a la
    hora h, que ya ocurrio antes del corte de las 00:00 del dia d. Se desplaza 1 antes de promediar.
    """
    e = df.pivot_table(index="ventana", columns="hora", values="error").sort_index()
    B = e.shift(1).rolling(k, min_periods=k).mean()
    return B.stack().rename("B").reset_index()


def regimenes_gmm(precio_diario, n_comp=3, fin_ajuste=None):
    """
    Regimenes por mezcla de gaussianas sobre retorno y volatilidad (seccion 4.3 del paper).

    La GMM se ajusta SOLO con datos anteriores a `fin_ajuste` y despues se aplica al resto sin
    reajustar, que es exactamente lo que hace el paper para evitar mirar al futuro. Las etiquetas se
    ordenan por volatilidad media creciente: 0 = calma, 1 = intermedio, 2 = crisis.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    ret = precio_diario.pct_change()
    vol = ret.rolling(20).std()
    X = pd.concat([ret, vol], axis=1).dropna()
    X.columns = ["ret", "vol"]
    tr = X.index < fin_ajuste if fin_ajuste is not None else np.ones(len(X), bool)
    sc = StandardScaler().fit(X[tr])
    g = GaussianMixture(n_components=n_comp, random_state=42, n_init=5).fit(sc.transform(X[tr]))
    et = pd.Series(g.predict(sc.transform(X)), index=X.index)
    orden = X.groupby(et)["vol"].mean().sort_values().index.tolist()
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
    return et.map(mapa), X


def aplicar(pred, B, lam):
    return pred + lam * B


def main():
    t0 = time.time()
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
    d["error"] = d["real"] - d["pred"]

    B = sesgo_local(d)
    d = d.reset_index().merge(B, on=["ventana", "hora"], how="left").set_index("fecha_hora").sort_index()

    serie = cargar_completo().set_index("fecha_hora")["precio_bolsa"]
    prev = serie.reindex(d.index - pd.Timedelta(hours=1))
    prev.index = d.index
    d["rampa"] = (d["real"] - prev).abs()

    # regimenes con GMM, ajustada solo con datos previos a 2026
    diario = serie.resample("D").mean().dropna()
    reg, _ = regimenes_gmm(diario, 3, fin_ajuste=pd.Timestamp("2026-01-01"))
    d["regimen"] = reg.reindex(d["ventana"]).to_numpy()
    d = d.dropna(subset=["B", "regimen"])
    d["regimen"] = d["regimen"].astype(int)

    y = d["real"].to_numpy()
    p = d["pred"].to_numpy()
    b = d["B"].to_numpy()
    reg_v = d["regimen"].to_numpy()
    fu = (d["rampa"] >= 50).to_numpy()
    pk = np.isin(d["hora"].to_numpy(), PICO)
    val = (d["ventana"] < FIN_VALID).to_numpy()
    tes = ~val
    print(f"Filas: {len(d):,}  |  validacion {int(val.sum()):,}  prueba {int(tes.sum()):,}")
    print(f"Reparto de regimenes (GMM ajustada solo con < 2026): "
          + ", ".join(f"{k}={v}" for k, v in d['regimen'].value_counts().sort_index().items()))
    print(f"Autocorrelacion del error del ensamble con su media movil B: "
          f"{np.corrcoef(d['error'], b)[0,1]:+.3f}\n")

    print("=" * 100)
    print("1) STBC DE LAMBDA FIJO  (lambda elegido en validacion, reportado en prueba)")
    print("=" * 100)
    perd = [np.abs(y[val] - aplicar(p[val], b[val], l)).mean() for l in REJILLA_L]
    l_opt = float(REJILLA_L[int(np.argmin(perd))])
    print(f"  lambda optimo en validacion: {l_opt:.2f}  "
          f"(MAE de validacion {min(perd):.2f} contra {np.abs(y[val]-p[val]).mean():.2f} sin corregir)")
    print(f"\n  {'lambda':>7s} {'MAE prueba':>11s} {'MAPE':>7s} {'sMAPE':>7s} {'MAE rampa':>10s} "
          f"{'MAE pico':>9s} {'DM p':>8s}")
    for l in [0.0, 0.1, 0.2, 0.5, 0.9, l_opt]:
        pc = aplicar(p, b, l)
        marca = "  <- elegido" if l == l_opt and l != 0.0 else ("  <- sin corregir" if l == 0 else "")
        print(f"  {l:7.2f} {np.abs(y[tes]-pc[tes]).mean():11.2f} {mape(y[tes],pc[tes]):7.2f} "
              f"{smape(y[tes],pc[tes]):7.2f} "
              f"{np.abs(y[tes&fu]-pc[tes&fu]).mean():10.2f} "
              f"{np.abs(y[tes&pk]-pc[tes&pk]).mean():9.2f} "
              f"{dm(y[tes], p[tes], pc[tes]) if l>0 else np.nan:8.4f}{marca}")

    print("\n" + "=" * 100)
    print("2) RA-STBC: lambda por regimen  (rejilla monotona lambda_0 <= lambda_1 <= lambda_2)")
    print("=" * 100)
    mejor = None
    for l0 in REJILLA_L[::2]:
        for l1 in REJILLA_L[::2]:
            for l2 in REJILLA_L[::2]:
                if not (l0 <= l1 <= l2):
                    continue
                lam = np.array([l0, l1, l2])[reg_v]
                m = np.abs(y[val] - aplicar(p[val], b[val], lam[val])).mean()
                if mejor is None or m < mejor[0]:
                    mejor = (m, l0, l1, l2)
    _, l0, l1, l2 = mejor
    lam = np.array([l0, l1, l2])[reg_v]
    pc = aplicar(p, b, lam)
    print(f"  lambda por regimen elegido en validacion: calma={l0:.2f} intermedio={l1:.2f} crisis={l2:.2f}")
    print(f"  (el paper propone 0.20 / 0.50 / 0.80)")
    print(f"\n  {'variante':28s} {'MAE prueba':>11s} {'MAPE':>7s} {'MAE rampa':>10s} {'MAE pico':>9s} {'DM p':>8s}")
    print(f"  {'sin corregir':28s} {np.abs(y[tes]-p[tes]).mean():11.2f} {mape(y[tes],p[tes]):7.2f} "
          f"{np.abs(y[tes&fu]-p[tes&fu]).mean():10.2f} {np.abs(y[tes&pk]-p[tes&pk]).mean():9.2f}")
    for nom, ll in [("RA-STBC ajustado aqui", np.array([l0, l1, l2])),
                    ("RA-STBC del paper (.2/.5/.8)", np.array([0.2, 0.5, 0.8]))]:
        lam = ll[reg_v]
        pc = aplicar(p, b, lam)
        print(f"  {nom:28s} {np.abs(y[tes]-pc[tes]).mean():11.2f} {mape(y[tes],pc[tes]):7.2f} "
              f"{np.abs(y[tes&fu]-pc[tes&fu]).mean():10.2f} {np.abs(y[tes&pk]-pc[tes&pk]).mean():9.2f} "
              f"{dm(y[tes], p[tes], pc[tes]):8.4f}")

    print("\n" + "=" * 100)
    print("3) ¿ES STBC SOLO UN EMPUJON HACIA LA PERSISTENCIA?")
    print("=" * 100)
    pers = d["Persistencia"].to_numpy() if "Persistencia" in d.columns else None
    if pers is not None:
        pc = aplicar(p, b, l_opt if l_opt > 0 else 0.2)
        print(f"  correlacion (correccion STBC, persistencia - ensamble): "
              f"{np.corrcoef(pc - p, pers - p)[0,1]:+.3f}")
        print("  Alta y positiva => STBC esta moviendo el pronostico hacia la persistencia, que ya es")
        print("  uno de los seis votantes; entonces no aporta informacion nueva al ensamble.")

    print("\n" + "=" * 100)
    print("4) VERIFICACION 10 PARTICIONES DEL COMBINADOR")
    print("=" * 100)
    A, Bm = [], []
    for s_ in SEMILLAS:
        d["pred"] = qra_cv(d, MEJOR, "smape", s_)
        d["error"] = d["real"] - d["pred"]
        bb = sesgo_local(d)
        dd = d.reset_index().merge(bb, on=["ventana", "hora"], how="left",
                                   suffixes=("_v", "")).set_index("fecha_hora").sort_index()
        m = dd["B"].notna().to_numpy() & tes
        yy, pp, bv = dd["real"].to_numpy(), dd["pred"].to_numpy(), dd["B"].fillna(0).to_numpy()
        lam = np.array([l0, l1, l2])[reg_v]
        pcc = pp + lam * bv
        A.append([np.abs(yy[m]-pp[m]).mean(), mape(yy[m], pp[m]), smape(yy[m], pp[m])])
        Bm.append([np.abs(yy[m]-pcc[m]).mean(), mape(yy[m], pcc[m]), smape(yy[m], pcc[m])])
    A, Bm = np.array(A), np.array(Bm)
    for j, nm in enumerate(["MAE", "MAPE %", "sMAPE %"]):
        dl = Bm[:, j] - A[:, j]
        print(f"  {nm:8s} sin corregir={A[:,j].mean():7.3f}+/-{A[:,j].std():.3f}  "
              f"RA-STBC={Bm[:,j].mean():7.3f}+/-{Bm[:,j].std():.3f}  "
              f"delta={dl.mean():+7.3f}, mejor en {int((dl<0).sum())}/10")
    print(f"  VEREDICTO: {'SOLIDO' if all((Bm[:,j]-A[:,j]<0).all() for j in range(3)) else 'no concluyente'}")

    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
