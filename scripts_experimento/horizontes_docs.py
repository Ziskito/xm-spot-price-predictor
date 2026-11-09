# -*- coding: utf-8 -*-
"""
Cifras de la comparacion entre horizontes (comparacion_horizontes.py) y del walk-forward historico de 72 h
(realista_validacion_historica_72h.py) para los documentos (informe final, funcionamiento, metodologia). Si el historico
aun no existe, `historico()` devuelve None y los documentos lo omiten.
"""
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
MEJ = RAIZ / "data/processed/resultados/escenario_realista/mejoras"
PERIODOS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
ORIGEN = {"Origen 1": "2020-21", "Origen 2": "2021-22", "Origen 3": "2022-23", "Origen 4": "2023-24", "Origen 5": "2024-25"}


def comparacion():
    t = pd.read_csv(MEJ / "comparacion_horizontes.csv")
    r = json.loads((MEJ / "comparacion_horizontes_resumen.json").read_text(encoding="utf-8"))
    return t, r


def historico():
    """Tabla periodo x dia (1, 2, 3) con MAE del pronostico y de la persistencia al mismo corte; None si falta el 72 h."""
    ruta = MEJ / "validacion_historica_72h.csv"
    if not ruta.exists():
        return None
    h = pd.read_csv(ruta)
    v2 = pd.read_csv(MEJ / "validacion_historica_v2.csv")
    b = pd.read_csv(RAIZ / "data/processed/resultados/escenario_realista/benchmark/metricas.csv")
    pers = {ORIGEN[p.split(" (")[0]]: m for p, m in b[b["modelo"] == "Persistencia"].set_index("periodo")["MAE"].items()
            if p.split(" (")[0] in ORIGEN}
    filas = []
    for _, r in v2.iterrows():
        per = ORIGEN[r["origen"]]
        filas.append({"periodo": per, "dia": 1, "MAE": r["MAE_v2"], "MAE_pers": pers[per], "MAE_ref": r["MAE_predespacho"],
                      "cobertura": float("nan")})
    for _, r in h.iterrows():
        filas.append({"periodo": r["periodo"], "dia": int(r["dia"]), "MAE": r["MAE"], "MAE_pers": r["MAE_pers"],
                      "MAE_ref": r["MAE_ref"], "cobertura": r["cobertura"], "DM_t_vs_pers": r["DM_t_vs_pers"]})
    return pd.DataFrame(filas)
