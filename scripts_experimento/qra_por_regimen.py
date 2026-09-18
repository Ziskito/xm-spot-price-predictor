# -*- coding: utf-8 -*-
"""
Idea 1 de los 4 papers nuevos: PESOS DEL ENSAMBLE CONDICIONADOS POR REGIMEN HIDROLOGICO.

De donde sale: Albani et al. (2025, Energy Economics 148:108651) observan que su modelo se degrada
exactamente cuando se rompe la correlacion entre hidrologia (ENA) y precio, y proponen en su
discusion apagar el componente hidrologico cuando |corr| < 0.5, ademas de un ensamble cuyos pesos
dependan de esa correlacion ("These weights could be determined, for example, by the correlation
between prices and ENA and some scores based on past predictions. This will be the subject of
future research.").

Nosotros ya sabemos que el MAE se triplica entre celdas de regimen (28.5 vs 97.1 COP/kWh), pero
esas celdas solo se usaban como ETIQUETA DE CONFIANZA. Aqui se usan para algo distinto: agrupar
el ajuste de los pesos del QRA. La hipotesis es que el votante optimo cambia segun el regimen
(p.ej. los arboles mandan en regimen humedo estable y los lineales en seco extremo), y que los
pesos unicos por franja horaria estan promediando dos regimenes que piden pesos distintos.

Anti-fuga: los umbrales de embalse y ONI se calculan SOLO con el periodo de entrenamiento
(pre-2026). La estratificacion descriptiva previa (ideas_papers_hidrologia.py) usaba la mediana
del propio conjunto de prueba, lo cual sirve para describir pero no para ponderar un modelo.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import cargar as cargar24, BASE5
from pesos_por_franja import dm
from validar_qra_24h import qra

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
CORTE = pd.Timestamp("2026-01-01")


def anadir_regimen(d):
    """Agrega columnas de regimen usando umbrales y climatologia SOLO de entrenamiento."""
    tr = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv",
                     parse_dates=["fecha_hora"]).set_index("fecha_hora").sort_index()
    te = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv",
                     parse_dates=["fecha_hora"]).set_index("fecha_hora").sort_index()
    tr = tr[tr.index < CORTE]

    # climatologia de embalse por dia del anio, estimada solo con entrenamiento
    clim = tr.groupby(tr.index.dayofyear)["volumen_embalses"].mean()
    def anom(s):
        return s - s.index.dayofyear.map(clim).to_numpy()
    anom_tr = anom(tr["volumen_embalses"])
    anom_te = anom(te["volumen_embalses"])

    # umbrales: medianas del ENTRENAMIENTO (no del periodo evaluado)
    u_emb = float(anom_tr.median())
    u_oni = float(tr["oni"].median())

    emb = anom_te.reindex(d.index)
    oni = te["oni"].reindex(d.index)
    d = d.copy()

    # (a) regimen ABSOLUTO: umbral fijo de la historia 2019-2025
    d["embalse"] = np.where(emb >= u_emb, "emb_alto", "emb_bajo")
    d["enso"] = np.where(oni >= u_oni, "nino", "neutral")
    d["regimen"] = d["embalse"] + "|" + d["enso"]

    # (b) regimen RELATIVO con ventana movil causal de 90 dias -- mas fiel a Albani et al., que
    # calculan la correlacion hidrologia-precio sobre una ventana movil de 255 dias y no contra un
    # umbral fijo. Resuelve ademas que en 2026 el umbral absoluto nunca marque "embalse bajo".
    # shift(1) para que el umbral de una hora nunca use esa misma hora.
    med_emb = emb.rolling("90D").median().shift(1)
    med_oni = oni.rolling("90D").median().shift(1)
    d["regimen_rel"] = (np.where(emb >= med_emb, "emb_alto_rel", "emb_bajo_rel") + "|" +
                        np.where(oni >= med_oni, "nino_rel", "neutral_rel"))

    d["g4"] = d.index.hour // 6
    d["g4_regimen"] = d["g4"].astype(str) + "|" + d["regimen"]
    d["g4_regimen_rel"] = d["g4"].astype(str) + "|" + d["regimen_rel"]
    print(f"Umbrales de ENTRENAMIENTO -> anomalia de embalse: {u_emb:+.3g} | ONI: {u_oni:+.2f}")
    print("\nHoras de 2026 por celda de regimen ABSOLUTO (umbral historico fijo):")
    print(d["regimen"].value_counts().to_string())
    print("\nHoras de 2026 por celda de regimen RELATIVO (mediana movil causal de 90 dias):")
    print(d["regimen_rel"].value_counts().to_string(), "\n")
    return d


def main():
    d = cargar24().sort_index()
    d["dia"] = d.index.normalize()
    d = anadir_regimen(d)
    if d[["embalse", "enso"]].isna().any().any():
        print("AVISO: hay horas sin regimen asignado")
    real = d["real"].to_numpy()

    resultados = {}
    preds = {}
    for etiqueta, grupo in [("QRA por franja (v4 actual)", "g4"),
                            ("QRA por regimen abs", "regimen"),
                            ("QRA por franja x regimen abs", "g4_regimen"),
                            ("QRA por regimen rel (90d)", "regimen_rel"),
                            ("QRA por franja x regimen rel", "g4_regimen_rel")]:
        np.random.seed(42)
        Q = qra(d, BASE5, grupo, [0.5], modo="cv")
        p = Q[:, 0]
        preds[etiqueta] = p
        n_celdas = d[grupo].nunique()
        resultados[etiqueta] = {"MAE": float(np.abs(real - p).mean()),
                                "celdas": n_celdas,
                                "horas_por_celda": int(len(d) / n_celdas)}

    tabla = pd.DataFrame(resultados).T
    print(tabla.round(3).to_string())

    base = preds["QRA por franja (v4 actual)"]
    print(f"\n{'='*78}\nDM contra la linea base (QRA por franja, el ensamble v4 vigente)\n{'='*78}")
    for etiqueta, p in preds.items():
        if etiqueta == "QRA por franja (v4 actual)":
            continue
        delta = np.abs(real - p).mean() - np.abs(real - base).mean()
        t, pv = dm(real, base, p, maxlags=48)
        signo = "MEJORA" if delta < 0 else "empeora"
        veredicto = "significativo" if pv < 0.05 else "NO significativo"
        print(f"{etiqueta:28s} delta MAE {delta:+7.3f}  ({signo})  DM p={pv:.4f}  {veredicto}")

    tabla.to_csv(RES / "qra_por_regimen.csv")
    print(f"\nGuardado: qra_por_regimen.csv")


if __name__ == "__main__":
    main()
