# -*- coding: utf-8 -*-
"""
Verificacion exhaustiva de la explicacion de la tarjeta del Operador (10-nov-2026, a raiz de un «esperar» explicado como
«rango normal» con la aguja en la zona barata).

Para cada hora, rol, horizonte, metodo y filtro de contrato se recalcula la senal paso a paso, como el motor:
  1) umbrales  -> senal cruda (precio bajo / normal / alto)
  2) filtro de banda ancha (solo banda e hibrido)
  3) filtro intradia
  4) filtro de precio de contrato (solo quita la senal favorable)
y se toma como motivo real el PRIMER paso que cambio la senal. Ese motivo se compara con el que elige la tarjeta
(dashboard/app.py, vista_operador): contrato -> banda ancha -> intradia -> frase de la senal. Ademas se comprueba que la
frase por defecto sea coherente (por ejemplo, «rango normal» solo cuando el precio esta de verdad entre los umbrales) y que
la senal final del recalculo sea identica a la del dashboard (serie_con_senal).
Uso: python scripts_experimento/verificar_explicaciones.py   (sale con codigo 1 si encuentra una incoherencia)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import motor_decision as md  # noqa: E402

RES = RAIZ / "data/processed/resultados"
FAV = {"generador": ("vender", "retener"), "comercializador": ("comprar", "evitar_compra")}


def precio_fechado():
    return pd.concat([pd.read_csv(RAIZ / "data/processed" / f, usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])
                      for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]
                     ).drop_duplicates("fecha_hora").set_index("fecha_hora")["precio_bolsa"].sort_index()


def mc(fechas):
    t = pd.read_csv(RAIZ / "data/processed/mc_mensual_xm.csv")
    s = pd.Series(t["mc"].to_numpy(), index=pd.PeriodIndex(t["mes"], freq="M"))
    dia = pd.DatetimeIndex(fechas) - pd.Timedelta(hours=1)
    meses = dia.to_period("M") - np.where(dia.day >= 10, 1, 2)
    return np.array([s.get(min(m, s.index.max()), np.nan) for m in meses])


def motivo_tarjeta(crudo, senal, por_contrato, banda, metodo, intradia):
    """La misma cadena de decision de la tarjeta (app.py)."""
    if por_contrato:
        return "contrato"
    if senal == "esperar" and crudo != "normal" and metodo in ("banda", "hibrido") and banda:
        return "banda"
    if senal == "esperar" and crudo != "normal" and intradia:
        return "intradia"
    return "frase"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    hist = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
    previo = precio_fechado()
    errores, resumen = [], []
    for horizonte, archivo in (("24h", "pronostico_realista_bandas_24h_2026.csv"), ("72h", "pronostico_realista_bandas_72h_2026.csv")):
        df = pd.read_csv(RES / archivo, parse_dates=["fecha_hora"]).drop_duplicates("fecha_hora").sort_values("fecha_hora").reset_index(drop=True)
        ref = mc(df["fecha_hora"])
        ancha = md.banda_ancha(df)
        rango = df["q50"].groupby(df["fecha_hora"].dt.normalize()).rank(pct=True)
        for metodo in md.METODOS:
            for rol in ("generador", "comercializador"):
                fav, des = FAV[rol]
                if metodo in ("rodante", "hibrido"):
                    bajo, alto = md.umbrales_rodantes(df["q50"], 30, previo=md.precio_previo_ventana(df, previo, 30))
                else:
                    u = md.umbrales_fijos(hist); bajo, alto = pd.Series(u["bajo"], index=df.index), pd.Series(u["alto"], index=df.index)
                for intradia in (True, False):
                    for contrato in (True, False):
                        oficial = md.generar_senales(df, metodo, rol, hist, precio_previo=previo, intradia=intradia)
                        q = df["q50"]
                        crudo = np.where(bajo.isna() | alto.isna(), "normal", np.where(q <= bajo, "bajo", np.where(q >= alto, "alto", "normal")))
                        s0 = pd.Series(np.select([crudo == "alto", crudo == "bajo"],
                                                 ["vender", "retener"] if rol == "generador" else ["evitar_compra", "comprar"],
                                                 default="esperar"), index=df.index)
                        s1 = s0.mask(ancha, "esperar") if metodo in ("banda", "hibrido") else s0.copy()
                        s2 = md.filtro_intradia(df, s1, rol) if intradia else s1.copy()
                        if not (s2 == oficial).all():
                            errores.append(f"{horizonte} {metodo} {rol} intradia={intradia}: el recalculo no coincide con el motor "
                                           f"en {(s2 != oficial).sum()} horas")
                        s3 = s2.copy()
                        por_contrato = np.zeros(len(df), bool)
                        if contrato:
                            no = (q.to_numpy() > ref) if rol == "comercializador" else (q.to_numpy() < ref)
                            por_contrato = (s3 == fav).to_numpy() & no & ~np.isnan(ref)
                            s3[por_contrato] = "esperar"
                        # motivo real: el primer paso que cambio la senal
                        real = np.where(s1 != s0, "banda", np.where(s2 != s1, "intradia", np.where(s3 != s2, "contrato", "frase")))
                        tarjeta = np.array([motivo_tarjeta(c, s, pc, a, metodo, intradia)
                                            for c, s, pc, a in zip(crudo, s3, por_contrato, ancha.to_numpy())])
                        mal = real != tarjeta
                        # coherencia de la frase por defecto: «esperar» con frase = precio de verdad en rango normal
                        incoh = (tarjeta == "frase") & (s3.to_numpy() == "esperar") & (crudo != "normal")
                        for i in np.flatnonzero(mal | incoh)[:3]:
                            errores.append(f"{horizonte} {metodo} {rol} intradia={intradia} contrato={contrato} "
                                           f"{df['fecha_hora'][i]:%Y-%m-%d %H}h: crudo={crudo[i]} senal={s3[i]} motivo real={real[i]} "
                                           f"tarjeta={tarjeta[i]}")
                        resumen.append({"horizonte": horizonte, "metodo": metodo, "rol": rol, "intradia": intradia,
                                        "contrato": contrato, "horas": len(df), "incoherencias": int((mal | incoh).sum()),
                                        **{f"motivo_{m}": int((real == m).sum()) for m in ("frase", "banda", "intradia", "contrato")}})
    r = pd.DataFrame(resumen)
    pd.set_option("display.width", 220)
    print(r.groupby(["horizonte", "metodo"])[["horas", "incoherencias", "motivo_frase", "motivo_banda", "motivo_intradia",
                                               "motivo_contrato"]].sum().to_string())
    print(f"\n{len(r)} combinaciones, {int(r['horas'].sum())} casos hora-combinación, incoherencias: {int(r['incoherencias'].sum())}")
    for e in errores[:30]:
        print("  ERROR", e)
    sys.exit(1 if errores else 0)
