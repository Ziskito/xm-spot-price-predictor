# -*- coding: utf-8 -*-
"""
¿POR QUE FALLA EL ENSAMBLE EN LAS RAMPAS? Descomposicion NIVEL / FORMA / AMPLITUD / DESFASE.

El objetivo real del usuario es bajar el error en los PUNTOS DE CAMBIO INTRADIA. Antes de elegir
metodo conviene saber por que falla, porque hay tres causas posibles y cada una admite soluciones
distintas (y descarta otras):

  (a) NIVEL. La forma de la curva del dia esta bien pero todo el dia esta corrido hacia arriba o
      hacia abajo. Entonces el problema no es intradia y no hay nada que ganar modelando la forma.

  (b) AMPLITUD (aplanamiento). La forma es correcta pero comprimida: el modelo pronostica una curva
      mas plana que la real. Es el sintoma clasico de minimizar error absoluto sobre un objetivo
      incierto -- cubrirse quedandose cerca del promedio. Si es esto, la solucion es recalibrar la
      amplitud, y el modelo de rampas fallido ya daba una pista fuerte (el Delta predicho tenia
      desviacion tipica 41 contra 92 del real, y capturaba el 29% de la magnitud en rampa fuerte).

  (c) DESFASE. La forma y la amplitud estan bien pero llegan una o dos horas tarde o temprano. Si es
      esto, ninguna mejora de magnitud sirve: hay que corregir el momento del quiebre.

Se miden las tres por separado, ademas de:

  (d) ¿ES PREDECIBLE LA FORMA? Se compara la forma pronosticada contra una CLIMATOLOGIA: el perfil
      medio por (hora x dia de semana x mes), que no usa ninguna informacion del dia. Si el ensamble
      no le gana a ese perfil promedio en la parte de forma, entonces el modelo no esta extrayendo
      nada del estado del sistema sobre la forma del dia, y ahi hay margen. Si ya le gana holgado,
      el margen es menor de lo que parece.

DESCOMPOSICION. Para cada dia d con curva real y_d(h) y pronosticada p_d(h):
      nivel_d  = media horaria del dia
      forma_d  = curva menos su media
  error de NIVEL puro  = |media(y_d) - media(p_d)|                (forma perfecta, nivel del modelo)
  error de FORMA puro  = |y_d(h) - (p_d(h) - media(p_d) + media(y_d))|   (nivel perfecto)
Ambos suman aproximadamente el error total y dicen donde esta el dinero.
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
from stacking_24h_v2 import cargar as cargar24, BASE5
from combinador_optimo_mape import qra_cv

PICO = [0, 8, 9, 10, 18, 19]


def main():
    t0 = time.time()
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["real"])
    d["pred"] = qra_cv(d, MEJOR, "smape")
    d["hora"] = d.index.hour

    # El "dia" relevante es la VENTANA de pronostico: 01:00 de un dia a 00:00 del siguiente, que es
    # la subasta. Se agrupa por la fecha del corte, no por la fecha del reloj.
    d["ventana"] = (d.index - pd.Timedelta(hours=1)).normalize()
    completas = d.groupby("ventana")["real"].count()
    d = d[d["ventana"].isin(completas[completas == 24].index)]
    print(f"Ventanas de 24h completas: {d['ventana'].nunique()}  |  filas: {len(d):,}\n")

    Y = d.pivot_table(index="ventana", columns="hora", values="real").to_numpy()
    P = d.pivot_table(index="ventana", columns="hora", values="pred").to_numpy()
    # reordenar columnas al orden de la ventana: horas 1..23 y luego la 0 (medianoche siguiente)
    orden = list(range(1, 24)) + [0]
    Y = d.pivot_table(index="ventana", columns="hora", values="real")[orden].to_numpy()
    P = d.pivot_table(index="ventana", columns="hora", values="pred")[orden].to_numpy()

    ny, npd = Y.mean(1, keepdims=True), P.mean(1, keepdims=True)
    Fy, Fp = Y - ny, P - npd

    mae_total = np.abs(Y - P).mean()
    mae_nivel = np.abs(ny - npd).mean()                 # forma perfecta
    mae_forma = np.abs(Fy - Fp).mean()                  # nivel perfecto
    print("=" * 96)
    print("(a) ¿NIVEL O FORMA? Descomposicion del error del ensamble")
    print("=" * 96)
    print(f"  error total del ensamble                          {mae_total:8.2f}")
    print(f"  si el NIVEL del dia fuera perfecto (solo forma)   {mae_forma:8.2f}   "
          f"({100*mae_forma/mae_total:.0f}% del total)")
    print(f"  si la FORMA fuera perfecta (solo nivel)           {mae_nivel:8.2f}   "
          f"({100*mae_nivel/mae_total:.0f}% del total)")
    print("  Si la parte de FORMA domina, modelar la curva intradia es la via correcta.")

    print("\n" + "=" * 96)
    print("(b) ¿AMPLITUD? ¿Es la curva pronosticada mas plana que la real?")
    print("=" * 96)
    sy, sp = Fy.std(1), Fp.std(1)
    ry, rp = Y.max(1) - Y.min(1), P.max(1) - P.min(1)
    print(f"  desviacion tipica intradia:  real {sy.mean():7.2f}   pronosticada {sp.mean():7.2f}   "
          f"razon {sp.mean()/sy.mean():.3f}")
    print(f"  rango intradia (max-min):    real {ry.mean():7.2f}   pronosticado {rp.mean():7.2f}   "
          f"razon {rp.mean()/ry.mean():.3f}")
    print(f"  dias en que la curva pronosticada es MAS PLANA que la real: "
          f"{100*np.mean(sp < sy):.0f}%")
    # pendiente de la regresion forma_real ~ forma_pronosticada: >1 = el pronostico se queda corto
    b = float(np.sum(Fy * Fp) / np.sum(Fp * Fp))
    print(f"  pendiente de  forma_real ~ forma_pronosticada:  {b:.3f}")
    print(f"    (1.0 = amplitud calibrada;  >1 = el pronostico se queda CORTO y habria que expandir)")

    print("\n" + "=" * 96)
    print("(c) ¿DESFASE? ¿Llega la forma pronosticada a destiempo?")
    print("=" * 96)
    print(f"  {'desplazamiento':>15s} {'correlacion':>12s} {'MAE de forma':>14s}")
    mejor = (0, -9, 9e9)
    for k in range(-3, 4):
        Fp_k = np.roll(Fp, k, axis=1)
        c = float(np.corrcoef(Fy.ravel(), Fp_k.ravel())[0, 1])
        m = float(np.abs(Fy - Fp_k).mean())
        marca = "  <- sin desplazar" if k == 0 else ""
        print(f"  {k:>+12d} h {c:12.4f} {m:14.2f}{marca}")
        if m < mejor[2]:
            mejor = (k, c, m)
    print(f"  Mejor desplazamiento: {mejor[0]:+d} h. Si es 0, NO hay problema de desfase.")

    print("\n" + "=" * 96)
    print("(d) ¿ES PREDECIBLE LA FORMA? El ensamble contra un perfil climatologico")
    print("=" * 96)
    # climatologia: perfil medio por (hora x dia de semana x mes), calculado SOLO con datos < 2026
    hist = cargar_completo()
    hist = hist[hist["fecha_hora"] < "2026-01-01"].copy()
    hist["ventana"] = (hist["fecha_hora"] - pd.Timedelta(hours=1)).dt.normalize()
    hist["hora"] = hist["fecha_hora"].dt.hour
    nivel_dia = hist.groupby("ventana")["precio_bolsa"].transform("mean")
    hist["forma"] = hist["precio_bolsa"] - nivel_dia
    hist["dow"] = hist["ventana"].dt.dayofweek
    hist["mes"] = hist["ventana"].dt.month
    clim = hist.groupby(["hora", "dow", "mes"])["forma"].mean()
    clim_g = hist.groupby("hora")["forma"].mean()

    idx = d.pivot_table(index="ventana", columns="hora", values="real")[orden].index
    C = np.zeros_like(Fy)
    for j, h in enumerate(orden):
        key = pd.MultiIndex.from_arrays([np.full(len(idx), h), idx.dayofweek, idx.month])
        C[:, j] = clim.reindex(key).fillna(clim_g.get(h, 0.0)).to_numpy()
    print(f"  MAE de forma, perfil climatologico (hora x dia x mes, sin mirar el dia)  {np.abs(Fy-C).mean():8.2f}")
    print(f"  MAE de forma, ensamble                                                   {np.abs(Fy-Fp).mean():8.2f}")
    g = 100 * (1 - np.abs(Fy - Fp).mean() / np.abs(Fy - C).mean())
    print(f"  El ensamble le gana a la climatologia en forma por {g:+.1f}%")
    print("  Poco margen => la forma intradia es casi impredecible mas alla del promedio historico.")
    print("  Mucho margen => el ensamble ya extrae informacion; el techo esta mas arriba.")

    print("\n" + "=" * 96)
    print("(e) LO MISMO, SOLO EN LAS 6 HORAS PEORES (0, 8, 9, 10, 18, 19)")
    print("=" * 96)
    col_pico = [orden.index(h) for h in PICO]
    print(f"  error total en esas horas                     {np.abs(Y[:,col_pico]-P[:,col_pico]).mean():8.2f}")
    print(f"  parte de forma                                {np.abs(Fy[:,col_pico]-Fp[:,col_pico]).mean():8.2f}")
    print(f"  parte de nivel                                {np.abs((ny-npd).repeat(len(col_pico),1)).mean():8.2f}")
    print(f"  amplitud: razon pronosticada/real             "
          f"{Fp[:,col_pico].std()/Fy[:,col_pico].std():8.3f}")
    print(f"  climatologia en esas horas                    {np.abs(Fy[:,col_pico]-C[:,col_pico]).mean():8.2f}")

    pd.DataFrame({"ventana": idx, "nivel_real": ny.ravel(), "nivel_pred": npd.ravel(),
                  "std_real": sy, "std_pred": sp, "rango_real": ry, "rango_pred": rp
                  }).to_csv(RES / "diagnostico_forma_intradia.csv", index=False)
    print(f"\nGuardado: diagnostico_forma_intradia.csv  |  LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
