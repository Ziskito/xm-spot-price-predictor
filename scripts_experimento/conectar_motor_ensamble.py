# -*- coding: utf-8 -*-
"""
Conecta el motor de decision al ensamble (en lugar de las bandas de N-BEATSx solas).

Genera dos contratos de pronostico con el formato que espera src/motor_decision.py
(fecha_hora, real, q10, q50, q90 -- una fila por hora):

  * 24 h: mediana = ensamble DESPLEGABLE (pesos solo con dias anteriores; la cifra honesta de
    operacion, MAE 42,89, desde el 15-ene-2026).
  * 72 h: mediana = producto de 72 h del ensamble (puente 24h->72h + combinador por tramo),
    tomando una ventana cada 72 h alineada con las ventanas del contrato vigente de N-BEATSx.

Bandas: la FORMA (ancho hora a hora) sale de los cuantiles crudos de N-BEATSx, recentrados en la
mediana del ensamble; luego se calibran por conformal adaptativo (metodo de calibrar_bandas_adaptativo.py)
con un margen por hora del dia y ventana de 60 dias ("Mondrian"), para que la cobertura sea pareja en
todas las horas. Asi el ancho sigue variando por hora y el filtro de "banda ancha" del motor conserva
su sentido.

No modifica fuentes_pronostico.json: eso lo decide quien corre el script (ver --activar).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "data" / "processed" / "resultados"
NIVEL, VENTANA_DIAS = 0.80, 30


def margen_conforme(scores_previos, minimo=24 * 7):
    n = len(scores_previos)
    if n < minimo:
        return np.nanquantile(scores_previos, 0.90) if n > 24 else 0.0
    nivel = min(np.ceil((n + 1) * NIVEL) / n, 1.0)
    return np.nanquantile(scores_previos, nivel)


def calibrar(df, clave_grupo=None, ventana_dias=None):
    """Ensancha [q10, q90] con el cuantil conforme de los scores de los dias previos (causal)."""
    ventana_dias = ventana_dias or VENTANA_DIAS
    df = df.sort_values("fecha_hora").reset_index(drop=True)
    df["score"] = np.maximum(df["q10_base"] - df["real"], df["real"] - df["q90_base"])
    df["dia"] = df["fecha_hora"].dt.normalize()
    margen = np.zeros(len(df))
    grupos = [None] if clave_grupo is None else df[clave_grupo].unique()
    for g in grupos:
        sel = np.ones(len(df), bool) if g is None else (df[clave_grupo] == g).to_numpy()
        sub = df[sel]
        for dia in sub["dia"].unique():
            hist = sub[(sub["dia"] < dia) & (sub["dia"] >= dia - pd.Timedelta(days=ventana_dias))]
            margen[sel & (df["dia"] == dia).to_numpy()] = margen_conforme(
                hist["score"].to_numpy(), minimo=24 * 7 if clave_grupo in (None, "tramo") else 7)
    df["margen"] = margen
    df["q10"] = df["q10_base"] - margen
    df["q90"] = df["q90_base"] + margen
    return df.drop(columns=["score", "dia"])


def contrato_24h():
    ens = pd.read_csv(RES / "informe_avance" / "ensamble_24h_desplegable.csv", parse_dates=["fecha_hora"])
    nb = pd.read_csv(RES / "pronostico_con_bandas_2026_adaptativo.csv", parse_dates=["fecha_hora"])
    nb["off10"] = (nb["q10"] + nb["margen"]) - nb["q50"]
    nb["off90"] = (nb["q90"] - nb["margen"]) - nb["q50"]
    d = ens.merge(nb[["fecha_hora", "off10", "off90"]], on="fecha_hora", how="inner")
    d["q50"] = d["pred_desplegable"]
    d["q10_base"] = d["q50"] + np.minimum(d["off10"], 0)
    d["q90_base"] = d["q50"] + np.maximum(d["off90"], 0)
    # un margen por hora del dia ("Mondrian"): con un margen unico la cobertura iba de 54 % (00:00)
    # a 94 % (01-04 h); por hora queda entre 77 % y 84 % (calibracion_por_hora_24h.py)
    d["hora"] = d["fecha_hora"].dt.hour
    d = calibrar(d, "hora", ventana_dias=60)
    return d[["fecha_hora", "real", "q50", "q10", "q90", "margen"]]


def contrato_72h():
    prod = pd.read_csv(RES / "informe_avance" / "producto_72h_reconstruido.csv", parse_dates=["fecha_hora", "cutoff"])
    nb = pd.read_csv(RES / "pronostico_con_bandas_72h_2026_adaptativo.csv", parse_dates=["fecha_hora", "cutoff"])
    nb["off10"] = (nb["q10"] + nb["margen"]) - nb["q50"]
    nb["off90"] = (nb["q90"] - nb["margen"]) - nb["q50"]
    # ventana del ensamble que arranca 1 h despues del corte de N-BEATSx: mismo paso, misma forma de banda
    nb["cutoff_ens"] = nb["cutoff"] + pd.Timedelta(hours=1)
    d = prod.merge(nb[["cutoff_ens", "paso_horas", "off10", "off90"]],
                   left_on=["cutoff", "paso_horas"], right_on=["cutoff_ens", "paso_horas"], how="inner")
    d["q50"] = d["pred"]
    d["q10_base"] = d["q50"] + np.minimum(d["off10"], 0)
    d["q90_base"] = d["q50"] + np.maximum(d["off90"], 0)
    # un margen por hora del dia, como a 24 h: por tramo la cobertura iba de 66 % (19 h) a 87 %;
    # por hora queda entre 75 % y 84 % con el mismo ancho medio
    d["hora"] = d["fecha_hora"].dt.hour
    d = calibrar(d, "hora", ventana_dias=60)
    assert d["fecha_hora"].is_unique
    return d[["fecha_hora", "cutoff", "paso_horas", "real", "q50", "q10", "q90", "margen"]]


def resumen(df):
    y = df["real"]
    return {"horas": int(len(df)), "MAE_q50": float((y - df["q50"]).abs().mean()),
            "cobertura_pct": float(((y >= df["q10"]) & (y <= df["q90"])).mean() * 100),
            "ancho_medio": float((df["q90"] - df["q10"]).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--activar", action="store_true",
                    help="apunta fuentes_pronostico.json a los contratos nuevos (guarda copia del anterior)")
    args = ap.parse_args()
    salidas = {"24h": ("pronostico_ensamble_bandas_24h_2026.csv", contrato_24h()),
               "72h": ("pronostico_ensamble_bandas_72h_2026.csv", contrato_72h())}
    for hz, (nombre, df) in salidas.items():
        df.to_csv(RES / nombre, index=False)
        print(hz, nombre, {k: round(v, 2) for k, v in resumen(df).items()})
    if args.activar:
        ruta = RES / "fuentes_pronostico.json"
        conf = json.loads(ruta.read_text(encoding="utf-8"))
        respaldo = RES / "fuentes_pronostico_nbeatsx.json"
        if not respaldo.exists():
            respaldo.write_text(json.dumps(conf, indent=2, ensure_ascii=False), encoding="utf-8")
        modelos = {"24h": "Ensamble de 6 modelos (pesos causales) + bandas conformes adaptativas por hora del dia (60 dias)",
                   "72h": "Ensamble 72 h (puente 24h->72h + combinador por tramo) + bandas conformes adaptativas por hora del dia (60 dias)"}
        for hz, (nombre, df) in salidas.items():
            conf[hz] = {"archivo": nombre, "modelo": modelos[hz], "calibrado": True,
                        "cobertura_objetivo_pct": 80,
                        "cobertura_medida_pct": round(resumen(df)["cobertura_pct"], 1),
                        "generado": pd.Timestamp.now().strftime("%Y-%m-%d"),
                        "notebook_origen": "scripts_experimento/conectar_motor_ensamble.py",
                        "notas": "Forma de banda de N-BEATSx recentrada en el ensamble y calibrada por conformal "
                                 "adaptativo. Contrato anterior en fuentes_pronostico_nbeatsx.json."}
        ruta.write_text(json.dumps(conf, indent=2, ensure_ascii=False), encoding="utf-8")
        print("fuentes_pronostico.json apunta ahora al ensamble (respaldo: fuentes_pronostico_nbeatsx.json)")
