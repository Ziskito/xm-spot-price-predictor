# -*- coding: utf-8 -*-
"""
Version adaptativa (ventana movil) de la calibracion conforme para las bandas 24-72h.

Misma logica que calibrar_bandas_adaptativo.py de anoche (que si funciono mejor que la version
estatica para el horizonte de 24h), aplicada POR TRAMO de horizonte (1-24h, 25-48h, 49-72h) por
separado, porque cada tramo necesita un margen de tamano distinto (crece con el horizonte).

Para cada dia de prueba, el margen se recalcula usando solo los scores de conformidad de los
ultimos 30 dias ANTERIORES a ese dia (nunca futuros) -- deja que el margen crezca cuando el
mercado se pone mas volatil, en vez de quedarse anclado al nivel de cuando se calibro una vez.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_CRUDO = RAIZ / "data" / "processed" / "resultados" / "pronostico_con_bandas_72h_2026.csv"
RUTA_SALIDA = RAIZ / "data" / "processed" / "resultados" / "pronostico_con_bandas_72h_2026_adaptativo.csv"

NIVEL_OBJETIVO = 0.80
VENTANA_DIAS = 30
DIAS_CALENTAMIENTO = 30

df = pd.read_csv(RUTA_CRUDO, parse_dates=["fecha_hora", "cutoff"]).sort_values("fecha_hora").reset_index(drop=True)
df["score"] = np.maximum(df["q10"] - df["real"], df["real"] - df["q90"])
df["fecha"] = df["fecha_hora"].dt.normalize()

tramos = [("1-24h", 1, 24), ("25-48h", 25, 48), ("49-72h", 49, 72)]
partes = []
resumen = []

for nombre, ini, fin in tramos:
    sub = df[(df["paso_horas"] >= ini) & (df["paso_horas"] <= fin)].copy().sort_values("fecha_hora").reset_index(drop=True)
    fechas_unicas = sorted(sub["fecha"].unique())
    margenes = {}

    for fecha in fechas_unicas:
        ventana_ini = fecha - pd.Timedelta(days=VENTANA_DIAS)
        historia = sub[(sub["fecha"] < fecha) & (sub["fecha"] >= ventana_ini)]
        if len(historia) < 24 * 5:  # menos de ~5 dias de historia -> calentamiento
            margenes[fecha] = sub[sub["fecha"] < fecha]["score"].quantile(0.90) if len(sub[sub["fecha"] < fecha]) > 24 else 0.0
            continue
        n = len(historia)
        nivel_ajustado = min(np.ceil((n + 1) * NIVEL_OBJETIVO) / n, 1.0)
        margenes[fecha] = historia["score"].quantile(nivel_ajustado)

    sub["margen"] = sub["fecha"].map(margenes).fillna(0.0)
    sub["q10_adapt"] = sub["q10"] - sub["margen"]
    sub["q90_adapt"] = sub["q90"] + sub["margen"]

    inicio_eval = fechas_unicas[0] + pd.Timedelta(days=DIAS_CALENTAMIENTO)
    post = sub[sub["fecha"] >= inicio_eval]

    cob_cruda = ((post["real"] >= post["q10"]) & (post["real"] <= post["q90"])).mean() * 100
    cob_adapt = ((post["real"] >= post["q10_adapt"]) & (post["real"] <= post["q90_adapt"])).mean() * 100
    ancho_adapt = (post["q90_adapt"] - post["q10_adapt"]).mean()

    print(f"{nombre:8s}  n_eval={len(post):5d}  cobertura cruda={cob_cruda:5.1f}%  "
          f"-> adaptativa={cob_adapt:5.1f}%  ancho_adapt={ancho_adapt:6.1f}")
    resumen.append({"tramo": nombre, "cobertura_cruda": cob_cruda, "cobertura_adaptativa": cob_adapt,
                     "ancho_adaptativo": ancho_adapt, "n_eval": len(post)})

    salida_tramo = sub[["fecha_hora", "cutoff", "paso_horas", "real", "q50", "margen"]].copy()
    salida_tramo["q10"] = sub["q10_adapt"]
    salida_tramo["q90"] = sub["q90_adapt"]
    partes.append(salida_tramo)

salida = pd.concat(partes, ignore_index=True).sort_values(["cutoff", "paso_horas"])
salida.to_csv(RUTA_SALIDA, index=False)
pd.DataFrame(resumen).to_csv(RAIZ / "data/processed/resultados/bandas_metricas_72h_adaptativo.csv", index=False)
print(f"\nContrato adaptativo 24-72h para OE3 guardado en: {RUTA_SALIDA}")
