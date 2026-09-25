# -*- coding: utf-8 -*-
"""
LAS MESETAS: horas en que el precio NO se movio y el ensamble aun se equivoca en ~28.5.

POR QUE MIRAR AQUI. La prueba de limite de informacion (`limite_informacion_rampas.py`) mostro que
en terminos relativos el modelo maneja las rampas MEJOR que las mesetas (en una rampa de 543 se
equivoca en 251; donde el precio no se movio nada, en 28.5). Las mesetas son 2.632 horas y el 35%
del error total, y ningun analisis las habia mirado.

HIPOTESIS A DISTINGUIR
  (1) Es error de NIVEL: el dia entero esta corrido. Entonces no es un problema intradia y la
      correccion tendria que venir del pronostico del nivel diario.
  (2) Hay DIAS PLANOS: en Colombia una sola planta hidraulica puede marcar el precio muchas horas
      seguidas. Si esos dias son reconocibles de antemano (p.ej. ayer tambien fue plano), un
      pronostico "plano" ganaria ahi.
  (3) El modelo INVENTA forma donde no la hay: dibuja una curva con picos cuando el dia real es
      plano. Seria el reverso del aplanamiento visto en las rampas.
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from o6_comun import cargar_completo, RES
from stacking_24h_v2 import cargar as cargar24, BASE5
from combinador_optimo_mape import qra_cv, mape

ORDEN = list(range(1, 24)) + [0]


def main():
    d = cargar24().sort_index()
    s = pd.read_csv(RES / "pronostico_GARCH-ged_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    d = d.join(s["pred" if "pred" in s.columns else s.columns[-1]].rename("GARCH-ged"), how="left")
    d["dia"] = d.index.normalize()
    d["g4"] = d.index.hour // 6
    MEJOR = BASE5 + ["GARCH-ged"]
    d = d.dropna(subset=MEJOR + ["real"])
    d["pred"] = qra_cv(d, MEJOR, "smape")
    d["hora"] = d.index.hour
    d["ventana"] = (d.index - pd.Timedelta(hours=1)).normalize()
    comp = d.groupby("ventana")["real"].count()
    d = d[d["ventana"].isin(comp[comp == 24].index)].copy()

    serie = cargar_completo().set_index("fecha_hora")["precio_bolsa"]
    prev = serie.reindex(d.index - pd.Timedelta(hours=1)); prev.index = d.index
    d["rampa"] = (d["real"] - prev).abs()
    d["meseta"] = d["rampa"] < 1e-9

    Y = d.pivot_table(index="ventana", columns="hora", values="real")[ORDEN]
    P = d.pivot_table(index="ventana", columns="hora", values="pred")[ORDEN]
    M = d.pivot_table(index="ventana", columns="hora", values="meseta")[ORDEN].astype(bool)
    y, p, m = Y.to_numpy(), P.to_numpy(), M.to_numpy()
    nivel_y, nivel_p = y.mean(1, keepdims=True), p.mean(1, keepdims=True)

    print("=" * 96)
    print("(1) ¿ES ERROR DE NIVEL?  Descomposicion restringida a las horas de meseta")
    print("=" * 96)
    e_tot = np.abs(y - p)[m].mean()
    e_forma = np.abs((y - nivel_y) - (p - nivel_p))[m].mean()
    e_nivel = np.abs(np.broadcast_to(nivel_y - nivel_p, y.shape))[m].mean()
    print(f"  error en mesetas                     {e_tot:7.2f}")
    print(f"  si el nivel del dia fuera perfecto   {e_forma:7.2f}")
    print(f"  si la forma fuera perfecta           {e_nivel:7.2f}")

    print("\n" + "=" * 96)
    print("(2) ¿HAY DIAS PLANOS?  Horas de meseta por dia, y error segun lo plano que sea el dia")
    print("=" * 96)
    n_plano = m.sum(1)
    rango = y.max(1) - y.min(1)
    print(f"  dias: {len(y)} | horas de meseta por dia: media {n_plano.mean():.1f}, "
          f"mediana {np.median(n_plano):.0f}, max {n_plano.max()}")
    print(f"  dias completamente planos (rango < 1): {int((rango < 1).sum())}")
    print(f"\n  {'horas planas del dia':>22s} {'dias':>6s} {'MAE del dia':>12s} {'rango real':>11s} {'rango pred':>11s}")
    for lo, hi in [(0, 6), (6, 12), (12, 18), (18, 25)]:
        g = (n_plano >= lo) & (n_plano < hi)
        if g.sum() == 0:
            continue
        print(f"  {f'{lo}-{hi-1}':>22s} {int(g.sum()):6d} {np.abs(y[g]-p[g]).mean():12.2f} "
              f"{rango[g].mean():11.1f} {(p[g].max(1)-p[g].min(1)).mean():11.1f}")

    # ¿es anticipable? persistencia del caracter plano de un dia al siguiente
    plano_hoy = n_plano[1:]
    plano_ayer = n_plano[:-1]
    print(f"\n  correlacion (horas planas ayer, horas planas hoy): "
          f"{np.corrcoef(plano_ayer, plano_hoy)[0,1]:+.3f}")

    print("\n" + "=" * 96)
    print("(3) ¿INVENTA FORMA DONDE NO LA HAY?  En tramos planos, cuanto se mueve el pronostico")
    print("=" * 96)
    # tramos de al menos 3 horas consecutivas de meseta dentro de la ventana
    mov_real, mov_pred = [], []
    for i in range(len(y)):
        run = 0
        for j in range(24):
            run = run + 1 if m[i, j] else 0
            if run >= 3 and j >= 1:
                mov_real.append(abs(y[i, j] - y[i, j - 1]))
                mov_pred.append(abs(p[i, j] - p[i, j - 1]))
    mov_real, mov_pred = np.array(mov_real), np.array(mov_pred)
    print(f"  horas dentro de tramos planos de >=3h: {len(mov_real)}")
    print(f"  cambio hora a hora REAL en esos tramos:        {mov_real.mean():6.2f}")
    print(f"  cambio hora a hora PRONOSTICADO en esos tramos: {mov_pred.mean():6.2f}")
    print("  Si el pronostico se mueve mucho donde el real esta quieto, el modelo inventa forma.")

    # cuanto se ganaria si en esos tramos el pronostico se quedara quieto (oraculo de la meseta)
    p_or = p.copy()
    for i in range(len(y)):
        for j in range(1, 24):
            if m[i, j]:
                p_or[i, j] = p_or[i, j - 1]
    print(f"\n  ORACULO 'saber donde hay meseta y aplanar ahi': MAE total {np.abs(y-p).mean():.2f} -> "
          f"{np.abs(y-p_or).mean():.2f}")
    print("  (cota superior de cualquier deteccion de mesetas; no es desplegable)")

    pd.DataFrame({"ventana": Y.index, "horas_planas": n_plano, "rango_real": rango,
                  "mae_dia": np.abs(y - p).mean(1)}).to_csv(RES / "diagnostico_mesetas.csv", index=False)


if __name__ == "__main__":
    main()
