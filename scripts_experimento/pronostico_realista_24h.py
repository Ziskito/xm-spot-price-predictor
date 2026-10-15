# -*- coding: utf-8 -*-
"""
Pronostico de 24 h del ESCENARIO REALISTA para el motor y el dashboard (14-oct-2026; experimentos en
resultados/escenario_realista/mejoras/RESUMEN.md y bitacora del 7-oct).

Solo usa lo que XM y NOAA ya publicaron a las 00:00 del dia D (precio, demanda y generacion hasta las 23:00 de D-4;
embalses y aportes de D-2; ONI de M-2; predespacho ideal, costo marginal y demanda del despacho programado de D,
publicados el dia anterior). Ventanas de 01:00 de D a 00:00 de D+1.
  * tres CatBoost (perdida MAE, objetivo log(precio / predespacho)) reentrenados cada mes con lo publicado
    (hasta 73 h antes del corte), con variables distintas: R+P (realistas + predespacho), R+P+F+C (+ forma
    intradia del predespacho + dMAR/dDEM) y R+P+E (+ error reciente del predespacho, ventanas hasta D-5); se promedian
  * combinador causal con el predespacho (QRA, objetivo MAE, franjas de 6 h, margen de 5 dias, previa del
    origen 5 los primeros 14 dias)
  * bandas del 80 %: q50 x exp(+-m), m = cuantil conforme de |log(real / q50)| de la misma hora en los 60 dias
    anteriores con precio ya publicado (realista_motor.bandas)
Salidas (resultados/): ensamble_24h_realista_2026.csv y pronostico_realista_bandas_24h_2026.csv (contrato del motor).
Con --produccion, modo de operacion diaria (ver main; lo llama actualizar_produccion.py).
Con --activar (o --solo-activar, sin reentrenar), fuentes_pronostico.json apunta el horizonte de 24 h a este contrato y a las probabilidades de eventos
realistas (realista_eventos.py --produccion); la configuracion anterior queda en fuentes_pronostico_optimista.json.
"""
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import realista_comun as rc  # noqa: E402
import realista_exp_errorpre as rep  # noqa: E402
import realista_exp_forma as rf  # noqa: E402
from realista_exp_arboles import conjuntos  # noqa: E402
from realista_exp_combinador import causal  # noqa: E402
from realista_exp_votantes import INI_TR, MARGEN, catboost, mensual  # noqa: E402
from realista_motor import bandas  # noqa: E402
from ensamble_con_predespacho_24h import imar, imar_con_respaldo  # noqa: E402
from produccion_datos import hoy  # noqa: E402

RES = rc.RAIZ / "data" / "processed" / "resultados"
PROD = RES / "produccion"
INI, FIN = pd.Timestamp("2026-01-01 01:00"), pd.Timestamp("2026-08-05 00:00")

def previa_24h(d, sets):
    """Previa del combinador (origen 5, 2024-07 a 2025-06): los mismos CatBoost entrenados antes de ese periodo.
    No cambia con los datos nuevos: en produccion se guarda en resultados/produccion/previa_24h.csv."""
    ruta = PROD / "previa_24h.csv"
    if "--produccion" in sys.argv and ruta.exists():
        return pd.read_csv(ruta, parse_dates=["fecha_hora", "dia"], float_precision="round_trip").set_index("fecha_hora")
    from ensamble_24h_enero import origen5
    previa = origen5()
    previa["Predespacho"] = imar().reindex(previa.index).to_numpy()
    trp = d.loc[INI_TR:pd.Timestamp("2024-07-01") - MARGEN].dropna(subset=["precio_bolsa", "pre_obj", "precio_lag24h"])
    tep = d.loc["2024-07-01":"2025-06-30 23:00"].dropna(subset=["pre_obj"])
    previa["Cat_prom"] = np.mean([catboost(trp, tep, cols).reindex(previa.index) for cols in sets.values()], axis=0)
    previa = previa.dropna(subset=["Predespacho", "Cat_prom"])[["real", "Predespacho", "Cat_prom", "dia"]]
    if "--produccion" in sys.argv:
        PROD.mkdir(parents=True, exist_ok=True)
        previa.to_csv(ruta)
    return previa


def main():
    """--produccion: datos hasta hoy (produccion_datos.py) y pronostico hasta la ventana del ultimo predespacho publicado
    (manana, si ya salio). Incremental: las ventanas ya empezadas (dia <= hoy) quedan como se publicaron; solo se
    calculan las nuevas y las futuras, reentrenando los CatBoost de su mes. El precio real se refresca en todas."""
    t0 = time.time()
    prod = "--produccion" in sys.argv
    d = rc.tabla_realista(forzar="--sin-reconstruir" not in sys.argv, produccion=prod).set_index("fecha_hora")
    F = rf.forma(d)
    d = d.join(F)
    C = rf.cm_programado(d)
    d = d.join(C)
    E = rep.errores(d)
    d = d.join(E)
    RP = conjuntos(d)["R+P"]
    sets = {"Cat_RP": RP, "Cat_RPFC": RP + list(F.columns) + list(C.columns), "Cat_RPE": RP + list(E.columns)}
    if prod:
        fin = imar().index.max().normalize() + pd.Timedelta(days=1)       # 00:00 que cierra la ventana del ultimo predespacho
        idx = d.loc[INI:fin].dropna(subset=["pre_obj"]).index
    else:
        fin = FIN
        idx = d.loc[INI:fin].dropna(subset=["precio_bolsa"]).index
    viejo = pd.DataFrame()
    if prod and (RES / "ensamble_24h_realista_2026.csv").exists():
        viejo = pd.read_csv(RES / "ensamble_24h_realista_2026.csv", parse_dates=["fecha_hora", "dia"], float_precision="round_trip").set_index("fecha_hora")
        viejo = viejo[viejo["dia"] <= hoy().normalize()]                       # ventanas ya empezadas: congeladas
    nuevos = idx.difference(viejo.index)
    pan = pd.DataFrame(index=nuevos)
    pan["Predespacho"] = rc.alinear(nuevos, imar_con_respaldo(imar(), idx.max()))
    for k, cols in sets.items():
        pan[k] = mensual(d, cols, nuevos.min(), fin).reindex(nuevos) if len(nuevos) else np.nan
        print(f"  {k} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pan["Cat_prom"] = pan[list(sets)].mean(axis=1)
    pan = pd.concat([viejo.drop(columns=[c for c in ("real", "dia") if c in viejo]), pan]).sort_index().loc[idx]
    pan["real"] = d.loc[idx, "precio_bolsa"]
    pan["dia"] = (pan.index - pd.Timedelta(hours=1)).normalize()
    previa = previa_24h(d, sets)
    for p in (pan, previa):
        p["g4"] = p.index.hour // 6
    comb = pd.Series(causal(pan, previa, ["Predespacho", "Cat_prom"], "g4", objetivo="mae"), index=pan.index)
    pan.loc[nuevos, "pred_realista"] = comb.loc[nuevos]
    falta = pan["pred_realista"].isna()
    pan.loc[falta, "pred_realista"] = pan.loc[falta, "Predespacho"]          # respaldo (no deberia ocurrir)
    pan.index.name = "fecha_hora"
    pan = pan[["real", "Predespacho"] + list(sets) + ["Cat_prom", "dia", "pred_realista"]]
    pan.to_csv(RES / "ensamble_24h_realista_2026.csv")
    con = bandas(pd.DatetimeIndex(idx), pan["real"].to_numpy(), pan["pred_realista"].to_numpy())
    con["margen"] = np.log(con["q90"] / con["q50"])
    con = con[["fecha_hora", "real", "q50", "q10", "q90", "margen"]].set_index("fecha_hora")
    if prod and (RES / "pronostico_realista_bandas_24h_2026.csv").exists():   # bandas ya publicadas: congeladas
        cv = pd.read_csv(RES / "pronostico_realista_bandas_24h_2026.csv", parse_dates=["fecha_hora"], float_precision="round_trip").set_index("fecha_hora")
        quedan = viejo.index.intersection(cv.index)
        con.loc[quedan, ["q50", "q10", "q90", "margen"]] = cv.loc[quedan, ["q50", "q10", "q90", "margen"]]
    con = con.reset_index()
    con.to_csv(RES / "pronostico_realista_bandas_24h_2026.csv", index=False)
    pub = con[con["real"].notna()]
    y, q = pub["real"], pub["q50"]
    ev = pub["fecha_hora"] >= "2026-01-15"
    cob = ((y >= pub["q10"]) & (y <= pub["q90"])).mean() * 100
    pp = pan[pan["real"].notna()]
    resumen = {"horas": int(len(con)), "horas_con_real": int(len(pub)), "nuevas": int(len(nuevos)), "MAE": float(np.abs(q - y).mean()),
               "MAE_desde_15ene": float(np.abs(q - y)[ev].mean()), "MAPE_%": float((np.abs(q - y) / y).mean() * 100),
               "cobertura_%": float(cob), "ancho_medio": float((con["q90"] - con["q10"]).mean()),
               "MAE_predespacho_solo": float(np.abs(pp["Predespacho"] - pp["real"]).mean())}
    print({k: round(v, 3) for k, v in resumen.items()}, "| hasta", con["fecha_hora"].max())
    if "--activar" in sys.argv:
        activar(cob)
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")
    return resumen


def activar(cob):
    if True:
        ruta = RES / "fuentes_pronostico.json"
        conf = json.loads(ruta.read_text(encoding="utf-8"))
        respaldo = RES / "fuentes_pronostico_optimista.json"
        if not respaldo.exists():
            shutil.copy2(ruta, respaldo)
        conf["24h"] = {"archivo": "pronostico_realista_bandas_24h_2026.csv",
                       "modelo": "Escenario realista: predespacho ideal de XM + promedio de 3 CatBoost reentrenados cada mes "
                                 "(combinador causal, objetivo MAE) + bandas conformes por hora en escala log (60 dias)",
                       "eventos": "probabilidades_eventos_24h_realista_2026.csv",
                       "calibrado": True, "cobertura_objetivo_pct": 80, "cobertura_medida_pct": round(cob, 1),
                       "generado": pd.Timestamp.now().strftime("%Y-%m-%d"),
                       "notebook_origen": "scripts_experimento/pronostico_realista_24h.py",
                       "notas": "Solo usa lo que XM y NOAA ya publicaron a las 00:00 del dia (precio con ~3 dias de rezago). "
                                "Configuracion anterior (ensamble de 7, escenario optimista) en fuentes_pronostico_optimista.json."}
        ruta.write_text(json.dumps(conf, indent=2, ensure_ascii=False), encoding="utf-8")
        print("fuentes_pronostico.json: 24h -> contrato realista")


if __name__ == "__main__":
    if "--solo-activar" in sys.argv:          # activa el contrato ya calculado, sin reentrenar
        c = pd.read_csv(RES / "pronostico_realista_bandas_24h_2026.csv")
        activar(((c["real"] >= c["q10"]) & (c["real"] <= c["q90"])).mean() * 100)
    else:
        main()
