# -*- coding: utf-8 -*-
"""
Version robusta del ensamble NNLS para Origen 6 (2026): en vez de un solo split 50/50, se hace
validacion cruzada de 5 pliegues estratificados por dia. Cada dia de 2026 queda predicho por un
ensamble cuyos pesos se ajustaron SOLO con los otros 4 pliegues -- as: se puede reportar un
MAPE fuera de muestra sobre el 100% de Origen 6, no solo la mitad, comparable directamente con
la tabla de MAPE ~15.4-16% ya documentada.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import nnls

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
MODELOS = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]
K = 5

df = pd.read_csv(RAIZ / "data/processed/resultados/walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
datos_o = df[df["origen"] == "Origen 6"]
pivot = datos_o.pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
real = datos_o.groupby("fecha_hora")["real"].first()
pivot = pivot.join(real).dropna()
pivot["dia"] = pivot.index.normalize()

dias = np.sort(pivot["dia"].unique())
rng = np.random.default_rng(42)
dias_shuffled = rng.permutation(dias)
pliegues = np.array_split(dias_shuffled, K)

predicciones_oos = []
pesos_por_pliegue = []

for k, dias_test in enumerate(pliegues):
    dias_train = np.setdiff1d(dias, dias_test)
    train = pivot[pivot["dia"].isin(dias_train)]
    test = pivot[pivot["dia"].isin(dias_test)]

    X_train, y_train = train[MODELOS].values, train["real"].values
    pesos, _ = nnls(X_train, y_train)
    pesos_norm = pesos / pesos.sum() if pesos.sum() > 0 else pesos
    pesos_por_pliegue.append(pesos_norm)

    X_test = test[MODELOS].values
    pred_test = X_test @ pesos_norm

    for fh, real_val, pred_val in zip(test.index, test["real"].values, pred_test):
        predicciones_oos.append({"fecha_hora": fh, "real": real_val, "pred_ensamble": pred_val})

    print(f"Pliegue {k+1}/{K}: {len(dias_test)} dias test, pesos = " +
          ", ".join(f"{m}={w:.3f}" for m, w in zip(MODELOS, pesos_norm)))

df_oos = pd.DataFrame(predicciones_oos).sort_values("fecha_hora")
error_abs = (df_oos["real"] - df_oos["pred_ensamble"]).abs()
mae = error_abs.mean()
mape = (error_abs / df_oos["real"]).mean() * 100
rmse = np.sqrt(((df_oos["real"] - df_oos["pred_ensamble"]) ** 2).mean())

print(f"\n{'='*70}\nEnsamble NNLS, validacion cruzada 5 pliegues, TODO Origen 6 (2026)\n{'='*70}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

print(f"\nPesos promedio entre pliegues:")
pesos_promedio = np.mean(pesos_por_pliegue, axis=0)
for m, w in zip(MODELOS, pesos_promedio):
    print(f"  {m}: {w:.3f}")

print(f"\nReferencia -- mejores individuales en 2026 (holdout completo, tabla ya documentada):")
print(f"  ARX+GARCH: MAE 55.76 | MAPE 15.43%")
print(f"  N-BEATSx ensamble 5 semillas: MAE 54.39 | MAPE 16.18%")

df_oos.to_csv(RAIZ / "data" / "processed" / "resultados" / "stacking_ensamble_cv_2026.csv", index=False)
print("\nGuardado: stacking_ensamble_cv_2026.csv")
