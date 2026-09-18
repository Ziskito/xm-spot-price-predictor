# -*- coding: utf-8 -*-
"""
Verificacion del hallazgo de ensamble_72h_con_lear.py antes de adoptarlo.

El hallazgo: "LEAR + NBX + NBX_exog + NaiveEstac" (4 votantes, sin ningun arbol) gana al ensamble
vigente de 7 votantes en los tres tramos y en los dos modos, con deltas de -1.4 a -3.1 COP/kWh.

Por que hay que verificarlo: el proyecto ya se quemo una vez dando por bueno un efecto de ~2.8
COP/kWh que resulto ser ruido de semilla (el falso hallazgo del ONI, ver verificar_oni_degrada.py).
La regla que quedo: toda diferencia menor a ~2-3 COP/kWh medida con UNA sola configuracion no es
interpretable. Aqui la fuente de aleatoriedad es la ASIGNACION DE PLIEGUES del modo CV (semilla 42
fija dentro de lad_cv). El modo desplegable es determinista (ventana causal, sin pliegues), asi que
si el resultado se sostiene ahi Y en varias semillas de CV, es solido.

Se comprueban tres cosas:
  1) estabilidad del delta frente a 10 semillas de particion en CV;
  2) el modo desplegable, que es determinista y es el que de verdad se despliega;
  3) que la mejora no venga de un puñado de dias raros: distribucion del delta por dia y
     proporcion de dias en que el nuevo ensamble gana.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import pesos_lad
from ensamble_72h_diario import TRAMOS, lad_causal, hac_p, RES

VIGENTE = ["NBX", "NBX_exog", "XGB-directo", "CatB-directo", "CatB-denso", "Ridge-directo", "NaiveEstac"]
NUEVO = ["LEAR", "NBX", "NBX_exog", "NaiveEstac"]
SEMILLAS = list(range(10))


def lad_cv_semilla(sub, cols, semilla):
    X, y = sub[cols].to_numpy(), sub["real"].to_numpy()
    P = np.full(len(sub), np.nan)
    dias = np.sort(sub["dia"].unique())
    for dt in np.array_split(np.random.default_rng(semilla).permutation(dias), 5):
        te = sub["dia"].isin(dt).to_numpy()
        P[te] = X[te] @ pesos_lad(X[~te], y[~te])
    return P


def main():
    v = pd.read_csv(RES / "pronostico_72h_diario_votantes_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    lear = pd.read_csv(RES / "pronostico_lear_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    v = v.merge(lear[["fecha_hora", "cutoff", "paso_horas", "LEAR"]],
                on=["fecha_hora", "cutoff", "paso_horas"], how="inner")
    v["dia"] = v["fecha_hora"].dt.normalize()

    print(f"{'='*94}\n1) ESTABILIDAD FRENTE A 10 SEMILLAS DE PARTICION (modo CV)\n{'='*94}")
    print(f"{'tramo':9s} {'vigente (7)':>22s} {'nuevo (4)':>22s} {'delta':>22s}")
    resumen = {}
    for nom, ini, fin in TRAMOS:
        sub = v[v.paso_horas.between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
        y = sub["real"].to_numpy()
        mv = np.array([np.abs(y - lad_cv_semilla(sub, VIGENTE, s)).mean() for s in SEMILLAS])
        mn = np.array([np.abs(y - lad_cv_semilla(sub, NUEVO, s)).mean() for s in SEMILLAS])
        d = mn - mv
        resumen[nom] = d
        print(f"{nom:9s} {mv.mean():10.2f} +/-{mv.std():5.2f} {mn.mean():10.2f} +/-{mn.std():5.2f} "
              f"{d.mean():10.2f} +/-{d.std():5.2f}")
    print("\nRuido entre semillas (desviacion del MAE del vigente): "
          f"{np.mean([resumen[n].std() for n in resumen]):.3f} COP/kWh")
    for nom in resumen:
        d = resumen[nom]
        veredicto = ("SOLIDO: mismo signo en 10/10 y |efecto| > 2x ruido"
                     if (d < 0).all() and abs(d.mean()) > 2 * d.std() else
                     "revisar: el efecto no supera claramente el ruido")
        print(f"  {nom:9s} delta {d.mean():+.2f} (min {d.min():+.2f}, max {d.max():+.2f}) -> {veredicto}")

    print(f"\n{'='*94}\n2) MODO DESPLEGABLE (determinista, es el que se despliega de verdad)\n{'='*94}")
    for nom, ini, fin in TRAMOS:
        sub = v[v.paso_horas.between(ini, fin)].sort_values(["cutoff", "paso_horas"]).reset_index(drop=True)
        y = sub["real"].to_numpy()
        Pv, _ = lad_causal(sub, VIGENTE)
        Pn, _ = lad_causal(sub, NUEVO)
        ok = ~(np.isnan(Pv) | np.isnan(Pn))
        mv, mn = np.abs(y[ok] - Pv[ok]).mean(), np.abs(y[ok] - Pn[ok]).mean()
        t, pv = hac_p(np.abs(y[ok] - Pv[ok]) - np.abs(y[ok] - Pn[ok]))
        print(f"  {nom:9s} vigente={mv:6.2f}  nuevo={mn:6.2f}  delta={mn-mv:+6.2f} "
              f"({100*(mv-mn)/mv:+5.1f}%)  DM p={pv:.4f}")

        # 3) reparto por dia: la mejora es generalizada o viene de pocos dias?
        dfd = pd.DataFrame({"dia": sub["dia"].to_numpy()[ok],
                            "ev": np.abs(y[ok] - Pv[ok]), "en": np.abs(y[ok] - Pn[ok])})
        g = dfd.groupby("dia").mean()
        gana = (g.en < g.ev).mean() * 100
        print(f"            el nuevo gana en {gana:.0f}% de los {len(g)} dias | "
              f"mediana del delta diario {np.median(g.en - g.ev):+.2f}")


if __name__ == "__main__":
    main()
