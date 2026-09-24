# -*- coding: utf-8 -*-
"""
¿DONDE MEJORA LA SELECCION DE VENTANA POR PUNTOS DE CAMBIO? El error vive en las rampas: ¿mejora ahi?

POR QUE HACE FALTA ESTA PRUEBA (observacion del usuario, y es correcta). Toda esta linea de trabajo
nacio de un diagnostico muy concreto: el error del proyecto se concentra en los PUNTOS DE CAMBIO
INTRADIA -- las rampas de la curva diaria, donde el MAE es 105.5 contra 28.5 en las mesetas. Pero
el metodo de Nasiadka/Nitka/Weron detecta puntos de cambio en otra serie completamente distinta:

    puntos de cambio INTRADIA   -> quiebres dentro de la curva de un dia (hora a hora). AHI DUELE.
    puntos de cambio INTER-DIA  -> quiebres en la serie de precios de UNA hora fija a lo largo de
                                   los dias (cambios de regimen del mercado). ESO es lo que NOT
                                   detecta, y solo lo usa para decidir con que historia entrenar.

Son cosas diferentes. Que NOT baje el error global NO implica que lo baje donde el proyecto falla.
Puede perfectamente estar mejorando las horas faciles y dejando intactas las dificiles, lo cual
seria una mejora real pero irrelevante para el problema que motivo la busqueda. Esta prueba lo
decide con datos en vez de suponerlo.

QUE SE MIDE
  1. Descomposicion por tipo de hora: meseta / cambio chico / medio / RAMPA FUERTE, con los mismos
     cortes del diagnostico original, tanto en el modelo suelto como en el ensamble.
  2. Descomposicion por hora del reloj, marcando las 6 horas que mas aportan al error (0, 8, 9, 10,
     18, 19), que son las que tienen rampa media alta.
  3. Diebold-Mariano DENTRO del subconjunto de rampa fuerte, que es la pregunta directa.
  4. El caso de la hora 0 (el salto de medianoche), que es un punto de cambio por construccion:
     frontera entre dos subastas diarias.

Sobre el ensamble se usan las 10 particiones del combinador, no una sola, para que los deltas por
subconjunto no sean ruido de particion.
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
from combinador_optimo_mape import qra_cv

SEMILLAS = list(range(10))
PICO = [0, 8, 9, 10, 18, 19]
CORTES_RAMPA = [(0, 0, "meseta (cambio 0)"), (0, 10, "cambio chico (<10)"),
                (10, 50, "cambio medio (10-50)"), (50, np.inf, "RAMPA FUERTE (>=50)")]


def dm(real, p_ref, p_nuevo, maxlags=24):
    real = np.asarray(real, float)
    d = np.abs(real - p_ref) - np.abs(real - p_nuevo)
    if len(d) < 30 or np.allclose(d, 0):
        return np.nan
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": min(maxlags, len(d) // 3)})
    return float(r.pvalues[0])


def rampa_de(idx):
    """|precio(t) - precio(t-1)| para cada hora objetivo: la magnitud del quiebre intradia."""
    serie = cargar_completo().set_index("fecha_hora")["precio_bolsa"]
    prev = serie.reindex(idx - pd.Timedelta(hours=1))
    prev.index = idx
    return (serie.reindex(idx) - prev).abs()


def tabla_por_grupo(y, base, nuevo, grupos, etiquetas, titulo):
    print(f"\n{titulo}")
    print(f"  {'subconjunto':24s} {'n':>6s} {'% error':>8s} {'MAE antes':>10s} {'MAE despues':>12s} "
          f"{'delta':>8s} {'%':>7s} {'DM p':>8s}")
    err_tot = np.abs(y - base).sum()
    for g, et in zip(grupos, etiquetas):
        if g.sum() == 0:
            continue
        a, b = np.abs(y[g] - base[g]).mean(), np.abs(y[g] - nuevo[g]).mean()
        apo = 100 * np.abs(y[g] - base[g]).sum() / err_tot
        p = dm(y[g], base[g], nuevo[g])
        print(f"  {et:24s} {int(g.sum()):6d} {apo:7.1f}% {a:10.2f} {b:12.2f} "
              f"{b-a:+8.2f} {100*(b-a)/a:+6.1f}% {p:8.4f}")


def main():
    t0 = time.time()

    # ---------------------------------------------------------------- modelos sueltos
    print("=" * 104)
    print("PARTE 1 - EL MODELO SUELTO: ¿la seleccion de ventana mejora en las rampas?")
    print("=" * 104)
    pares = [("ARX WinH(728)", "vc_ARX_WinH728.csv", "ARX NOTH(728)", "vc_ARX_NOTH728.csv"),
             ("LEAR Win(todo)", "vc_LEAR_Wintodo.csv", "LEAR NOT(728)", "vc_LEAR_NOT728.csv")]
    for na, fa, nb, fb in pares:
        ra, rb = RES / fa, RES / fb
        if not ra.exists() or not rb.exists():
            print(f"  (falta {fa if not ra.exists() else fb}; se omite {na} vs {nb})")
            continue
        A = pd.read_csv(ra, parse_dates=["fecha_hora"]).drop_duplicates("fecha_hora").set_index("fecha_hora")
        B = pd.read_csv(rb, parse_dates=["fecha_hora"]).drop_duplicates("fecha_hora").set_index("fecha_hora")
        idx = A.index.intersection(B.index)
        y = A.loc[idx, "real"].to_numpy()
        pa, pb = A.loc[idx, "pred"].to_numpy(), B.loc[idx, "pred"].to_numpy()
        r = rampa_de(idx).to_numpy()
        grupos = [(r == 0) if lo == hi else ((r > lo) & (r < hi)) if np.isfinite(hi) else (r >= lo)
                  for lo, hi, _ in CORTES_RAMPA]
        tabla_por_grupo(y, pa, pb, grupos, [e for _, _, e in CORTES_RAMPA],
                        f"{nb} contra {na}   (n={len(idx):,})")

    # ---------------------------------------------------------------- ensamble
    print("\n" + "=" * 104)
    print("PARTE 2 - EL ENSAMBLE: ¿la mejora de 41.29 -> 40.99 de donde sale?")
    print("=" * 104)
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    v = pd.read_csv(RES / "vc_ARX_NOTH728.csv", parse_dates=["fecha_hora"]).drop_duplicates("fecha_hora")
    d = d.join(v.set_index("fecha_hora")["pred"].rename("VentanaNOT"), how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["VentanaNOT", "real"])
    y = d["real"].to_numpy()
    r = rampa_de(d.index).to_numpy()
    hora = d.index.hour.to_numpy()

    print(f"  calculando 10 particiones del combinador (con y sin el votante nuevo)...", flush=True)
    PA = np.array([qra_cv(d, MEJOR, "smape", s_) for s_ in SEMILLAS])
    PB = np.array([qra_cv(d, MEJOR + ["VentanaNOT"], "smape", s_) for s_ in SEMILLAS])
    base, nuevo = PA.mean(0), PB.mean(0)

    grupos = [(r == 0) if lo == hi else ((r > lo) & (r < hi)) if np.isfinite(hi) else (r >= lo)
              for lo, hi, _ in CORTES_RAMPA]
    tabla_por_grupo(y, base, nuevo, grupos, [e for _, _, e in CORTES_RAMPA],
                    f"POR TIPO DE HORA   (n={len(d):,}; prediccion = media de las 10 particiones)")

    # robustez del delta por subconjunto: ¿mejora en las 10 particiones dentro de cada grupo?
    print(f"\n  ¿Se sostiene en las 10 particiones DENTRO de cada subconjunto?")
    print(f"  {'subconjunto':24s} {'delta medio':>12s} {'gana':>8s}")
    for g, (_, _, et) in zip(grupos, CORTES_RAMPA):
        dl = np.array([np.abs(y[g] - PB[i][g]).mean() - np.abs(y[g] - PA[i][g]).mean()
                       for i in range(len(SEMILLAS))])
        print(f"  {et:24s} {dl.mean():+12.3f} {int((dl<0).sum()):6d}/10")

    gh = [hora == h for h in range(24)]
    tabla_por_grupo(y, base, nuevo, [np.isin(hora, PICO), ~np.isin(hora, PICO)],
                    ["6 horas peores", "las otras 18"],
                    "POR FRANJA DE ERROR   (las 6 peores son 0, 8, 9, 10, 18, 19)")
    tabla_por_grupo(y, base, nuevo, gh, [f"hora {h:02d}" for h in range(24)],
                    "POR HORA DEL RELOJ")

    filas = []
    for h in range(24):
        g = hora == h
        filas.append(dict(hora=h, n=int(g.sum()), rampa_media=float(r[g].mean()),
                          mae_antes=float(np.abs(y[g] - base[g]).mean()),
                          mae_despues=float(np.abs(y[g] - nuevo[g]).mean())))
    R = pd.DataFrame(filas)
    R["delta"] = R["mae_despues"] - R["mae_antes"]
    R.to_csv(RES / "not_donde_mejora_24h.csv", index=False)

    cc = np.corrcoef(R["rampa_media"], R["delta"])[0, 1]
    print(f"\n  Correlacion entre (rampa media de la hora) y (delta de MAE): {cc:+.3f}")
    print("  Negativa => la mejora es MAYOR justo en las horas de mas rampa, que es lo que se busca.")
    print("  Cerca de 0 => la mejora es pareja y NO ataca especificamente los puntos de cambio.")

    print(f"\nGuardado: not_donde_mejora_24h.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
