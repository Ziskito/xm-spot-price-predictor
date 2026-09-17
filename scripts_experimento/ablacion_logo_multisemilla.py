# -*- coding: utf-8 -*-
"""
Ablacion LOGO con MULTIPLES SEMILLAS -- version defendible del experimento.

Por que hace falta: la primera ablacion (ideas_papers_hidrologia.py, una sola semilla) sugirio que
quitar el ONI MEJORABA el modelo en 2.82 COP/kWh con p<0.0001. Al verificarlo (verificar_oni_degrada.py)
resulto que:
  - el ONI de 2026 esta 100% DENTRO del rango de entrenamiento (no hay extrapolacion),
  - y con 3 semillas el signo del efecto SE INVIERTE (semilla 42: -2.82 a favor de quitarlo;
    semilla 7: +1.69 a favor de dejarlo; semilla 123: -2.96 a favor de quitarlo),
  - mientras el MAE base varia 58.23-60.48 entre semillas (rango 2.25) solo por el azar del ajuste.

Es decir: el "efecto" del ONI era del mismo tamano que el ruido semilla-a-semilla, y el test DM
daba p<0.0001 igual porque con 5,208 horas detecta diferencias sistematicas minusculas (el caso mas
claro: con Ridge, base y sin-ONI dan 59.43 ambos y el DM igual reporta p=0.0000).

Leccion general para el proyecto: cualquier diferencia de MAE menor a ~2-3 COP/kWh medida con UNA
sola semilla no es interpretable. Este script repite la ablacion con 3 semillas y reporta
media +/- desviacion, para separar lo que aporta de verdad de lo que es ruido de ajuste.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from ideas_papers_hidrologia import GRUPOS, construir_datos
from o6_comun import cargar_completo, RES

SEMILLAS = [42, 7, 123]


def main():
    t0 = time.time()
    from catboost import CatBoostRegressor
    df = cargar_completo()
    tr, te, cols = construir_datos(df, [])
    real = te["y"].to_numpy()
    print(f"{len(tr):,} filas de entrenamiento | {len(te):,} de prueba | {len(cols)} variables "
          f"| {len(SEMILLAS)} semillas\n")

    def mae_de(cc, semilla):
        m = CatBoostRegressor(n_estimators=800, max_depth=8, learning_rate=0.05, loss_function="MAE",
                              random_seed=semilla, verbose=0, thread_count=-1)
        m.fit(tr[cc], tr["y"])
        return float(np.abs(real - m.predict(te[cc])).mean())

    base = np.array([mae_de(cols, s) for s in SEMILLAS])
    print(f"LINEA BASE: MAE {base.mean():.2f} +/- {base.std():.2f}  (semillas: {np.round(base, 2)})")
    print(f"El ruido semilla-a-semilla es de +/-{base.std():.2f} COP/kWh -- cualquier efecto menor "
          f"que ~{2*base.std():.1f} no es interpretable.\n")

    filas = [{"grupo": "(linea base)", "mae_media": base.mean(), "mae_std": base.std(),
              "delta_media": 0.0, "delta_std": 0.0, "veredicto": "-"}]
    print(f"{'grupo quitado':35s} {'MAE medio':>10s} {'delta medio':>12s} {'delta por semilla':>26s}  veredicto")
    for nombre, miembros in GRUPOS.items():
        quitar = [c for c in miembros if c in cols]
        if not quitar:
            continue
        cols_sin = [c for c in cols if c not in quitar]
        sin = np.array([mae_de(cols_sin, s) for s in SEMILLAS])
        delta = sin - base  # positivo = quitar el grupo empeora = el grupo aporta
        # veredicto: aporta solo si TODAS las semillas coinciden en signo y el efecto supera el ruido
        mismo_signo = np.all(delta > 0) or np.all(delta < 0)
        supera_ruido = abs(delta.mean()) > 2 * base.std()
        if mismo_signo and supera_ruido:
            veredicto = "APORTA" if delta.mean() > 0 else "ESTORBA"
        elif mismo_signo:
            veredicto = "efecto consistente pero dentro del ruido"
        else:
            veredicto = "ruido (el signo cambia entre semillas)"
        print(f"{nombre:35s} {sin.mean():10.2f} {delta.mean():+12.2f} {str(np.round(delta, 2)):>26s}  {veredicto}")
        filas.append({"grupo": nombre, "mae_media": sin.mean(), "mae_std": sin.std(),
                      "delta_media": delta.mean(), "delta_std": delta.std(), "veredicto": veredicto})

    pd.DataFrame(filas).to_csv(RES / "ablacion_logo_multisemilla.csv", index=False)
    print(f"\nGuardado: ablacion_logo_multisemilla.csv | LISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
