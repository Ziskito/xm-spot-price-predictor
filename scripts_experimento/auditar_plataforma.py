# -*- coding: utf-8 -*-
"""
Auditoria funcional del dashboard (10-nov-2026, pedido de Juan: «revisa toda la plataforma, cada funcion»).

Parte A, con streamlit.testing (sin navegador): recorre las tres vistas con todas las combinaciones de la barra lateral
(rol, horizonte, ventana de comparacion, filtro de contrato con MC o precio propio, tema) y varios dias y horas, oprime
los botones de dia y hora y los controles de Analista y Metodologia, y comprueba:
  * que ninguna ejecucion lance una excepcion;
  * que la tarjeta del Operador (datos de #augur-datos, los mismos del reporte PDF) coincida con un recalculo
    independiente del motor: senal, umbrales, q10/q50/q90 y el motivo de la explicacion;
  * que las tarjetas de Analista coincidan con evaluar_backtest sobre la misma senal.
Uso: python scripts_experimento/auditar_plataforma.py   (sale con codigo 1 si encuentra un error)
"""
import html
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import motor_decision as md  # noqa: E402
from verificar_explicaciones import precio_fechado  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(RAIZ / "dashboard" / "app.py")
TITULO = {"comprar": "Comprar en bolsa", "vender": "Despachar y vender", "evitar_compra": "Evitar la bolsa",
          "retener": "Retener generación", "esperar": "Esperar"}
FAV = {"generador": "vender", "comercializador": "comprar"}
HIST = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
PREVIO = precio_fechado()
_MC = pd.read_csv(RAIZ / "data/processed/mc_mensual_xm.csv")
MC = pd.Series(_MC["mc"].to_numpy(), index=pd.PeriodIndex(_MC["mes"], freq="M"))
errores = []


def contrato_ref(fechas, contrato):
    if contrato is None:
        return np.full(len(fechas), np.nan)
    if contrato != "mc":
        return np.full(len(fechas), float(contrato))
    dia = pd.DatetimeIndex(fechas) - pd.Timedelta(hours=1)
    meses = dia.to_period("M") - np.where(dia.day >= 10, 1, 2)
    return np.array([MC.get(min(m, MC.index.max()), np.nan) for m in meses])


_cache = {}


def esperado(horizonte, rol, ventana, contrato):
    clave = (horizonte, rol, ventana, contrato)
    if clave in _cache:
        return _cache[clave]
    b, _ = md.cargar_fuente_pronostico(RAIZ, horizonte)
    t = md.comparar_metodos_estable(b, rol, HIST, precio_previo=PREVIO, intradia=True, ventana_dias=ventana)
    metodo, _ = md.elegir_mejor_metodo_estable(t)
    s = md.generar_senales(b, metodo, rol, HIST, ventana_dias=ventana, precio_previo=PREVIO, intradia=True)
    ref = contrato_ref(b["fecha_hora"], contrato)
    q = b["q50"].to_numpy()
    no = (q > ref) if rol == "comercializador" else (q < ref)
    por_contrato = (s == FAV[rol]).to_numpy() & no & ~np.isnan(ref)
    s = s.copy(); s[por_contrato] = "esperar"
    if metodo in ("rodante", "hibrido"):
        bajo, alto = md.umbrales_rodantes(b["q50"], ventana, previo=md.precio_previo_ventana(b, PREVIO, ventana))
    else:
        u = md.umbrales_fijos(HIST); bajo, alto = pd.Series(u["bajo"], index=b.index), pd.Series(u["alto"], index=b.index)
    out = b.assign(senal=s, por_contrato=por_contrato, bajo=bajo, alto=alto, metodo=metodo,
                   ancha=md.banda_ancha(b), rango=b["q50"].groupby(b["fecha_hora"].dt.normalize()).rank(pct=True))
    _cache[clave] = out
    return out


def datos(at):
    for m in at.markdown:
        g = re.search(r'id="augur-datos"[^>]*data-r="([^"]*)"', m.value)
        if g:
            return json.loads(html.unescape(g.group(1)))
    return None


def revisar_operador(at, etiqueta, horizonte, rol, ventana, contrato):
    if at.exception:
        errores.append(f"{etiqueta}: excepción {at.exception[0].message[:200]}"); return
    d = datos(at)
    if d is None:
        errores.append(f"{etiqueta}: no hay datos de la tarjeta"); return
    e = esperado(horizonte, rol, ventana, contrato)
    dia = pd.Timestamp(pd.to_datetime(d["dia"], dayfirst=True))
    f = e[(e["fecha_hora"].dt.normalize() == dia) & (e["fecha_hora"].dt.hour == d["hora"])]
    if f.empty:
        errores.append(f"{etiqueta}: la hora {d['dia']} {d['hora']} no existe en el pronóstico"); return
    f = f.iloc[0]
    if d["accion"] != TITULO[f["senal"]]:
        errores.append(f"{etiqueta} {d['dia']} {d['hora']}h: tarjeta «{d['accion']}» y motor «{f['senal']}»")
    for k in ("q10", "q50", "q90", "bajo", "alto"):
        if d[k] is not None and not np.isnan(f[k]) and abs(d[k] - f[k]) > 0.05:
            errores.append(f"{etiqueta} {d['dia']} {d['hora']}h: {k} {d[k]} frente a {f[k]:.2f}")
    crudo = "normal" if np.isnan(f["bajo"]) else ("bajo" if f["q50"] <= f["bajo"] else ("alto" if f["q50"] >= f["alto"] else "normal"))
    frase = d["frase"]
    if f["por_contrato"]:
        ok = "precio de contrato" in frase
    elif f["senal"] == "esperar" and crudo != "normal" and f["metodo"] in ("banda", "hibrido") and f["ancha"]:
        ok = "banda ancha" in frase
    elif f["senal"] == "esperar" and crudo != "normal":
        ok = "filtro intradía" in frase
    elif f["senal"] == "esperar":
        ok = "rango normal" in frase or "No hay" in frase
    else:
        ok = "filtro" not in frase and "precio de contrato" not in frase      # «cubrirse con contratos» es la frase correcta
    if not ok:
        errores.append(f"{etiqueta} {d['dia']} {d['hora']}h: frase incoherente (señal {f['senal']}, precio {crudo}): {frase[:120]}")


def app(vista, rol="generador", horizonte="24h", ventana=30, contrato="mc", tema="claro", dia=None, hora=None):
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["vista"] = vista
    at.query_params["tema"] = tema
    if dia:
        at.query_params["dia"] = dia
    if hora is not None:
        at.query_params["hora"] = str(hora)
    at.session_state["ventana_umbral"] = ventana
    at.session_state["usar_contrato"] = contrato is not None
    at.session_state["mi_precio_contrato"] = float(contrato) if contrato not in (None, "mc") else 0.0
    at.run()
    if rol != "generador":
        at.selectbox[0].set_value(rol); at.run()
    if horizonte != "24h":
        at.selectbox[1].set_value(horizonte); at.run()
    return at


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    n = 0
    # ---- Operador: combinaciones de la barra lateral en dias y horas distintos
    dias = ["2026-01-02", "2026-03-23", "2026-06-15", "2026-08-20", "2026-10-23", "2026-11-03"]
    horas = [0, 8, 16, 19, 23]
    combos = [(r, h, v, c) for r in ("generador", "comercializador") for h in ("24h", "72h") for v in (7, 30, 365)
              for c in ("mc", None, 500.0)]
    for i, (rol, hz, v, c) in enumerate(combos):
        dia, hora = dias[i % len(dias)], horas[i % len(horas)]
        at = app("operador", rol, hz, v, c, dia=dia, hora=hora)
        revisar_operador(at, f"op {rol} {hz} v{v} c{c}", hz, rol, v, c); n += 1
        # navegar: dia anterior, siguiente, ultimo y otra hora
        for accion in ("dia_ant", "dia_sig", "dia_ult"):
            boton = at.button(key=accion)
            if boton.disabled:          # p. ej. «Anterior» en el primer dia o «Siguiente» en el ultimo: correcto
                continue
            boton.click(); at.run(); n += 1
            revisar_operador(at, f"op {rol} {hz} v{v} c{c} tras {accion}", hz, rol, v, c)
        at.number_input(key="hora_operador").set_value(12); at.run(); n += 1
        revisar_operador(at, f"op {rol} {hz} v{v} c{c} hora 12", hz, rol, v, c)
        print(f"  operador {i + 1}/{len(combos)} ({(time.time() - t0) / 60:.1f} min, {len(errores)} errores)", flush=True)
    # todas las horas de un dia de operacion, con los dos roles (la explicacion cambia hora a hora)
    for rol in ("generador", "comercializador"):
        for hora in range(24):
            at = app("operador", rol, "24h", 30, "mc", dia="2026-11-10", hora=hora); n += 1
            revisar_operador(at, f"op {rol} 10-nov h{hora}", "24h", rol, 30, "mc")
    # tema oscuro y mapa de calor por precio
    at = app("operador", tema="oscuro"); n += 1
    revisar_operador(at, "op oscuro", "24h", "generador", 30, "mc")
    at.radio(key="modo_calor").set_value("Precio esperado"); at.run(); n += 1
    if at.exception:
        errores.append(f"mapa de calor por precio: {at.exception[0].message[:200]}")
    print(f"  operador listo ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # ---- Analista: controles y metodo manual
    for rol in ("generador", "comercializador"):
        for hz in ("24h", "72h"):
            at = app("analista", rol, hz); n += 1
            if at.exception:
                errores.append(f"analista {rol} {hz}: {at.exception[0].message[:200]}"); continue
            # tarjeta de ventaja frente al backtest del motor con el metodo elegido y todo el rango
            metodo = [s for s in at.selectbox if s.label == "Método de umbral"][0].value
            b_, _ = md.cargar_fuente_pronostico(RAIZ, hz)
            s_ = md.generar_senales(b_, metodo, rol, HIST, ventana_dias=30, precio_previo=PREVIO, intradia=True)
            ref = contrato_ref(b_["fecha_hora"], "mc")
            no = (b_["q50"].to_numpy() > ref) if rol == "comercializador" else (b_["q50"].to_numpy() < ref)
            s_ = s_.copy(); s_[(s_ == FAV[rol]).to_numpy() & no & ~np.isnan(ref)] = "esperar"
            bt = md.evaluar_backtest(b_, s_, rol)
            texto = " ".join(m.value for m in at.markdown)
            g = re.search(r'Ventaja económica.*?class="stat-v">([+−-]?[\d.]+,\d)', texto)
            if not g:
                errores.append(f"analista {rol} {hz}: no encontré la tarjeta de ventaja")
            else:
                v = float(g.group(1).replace("−", "-").replace(".", "").replace(",", "."))
                if abs(v - bt["ventaja_cop_kwh"]) > 0.06:
                    errores.append(f"analista {rol} {hz} ({metodo}): la tarjeta dice {v} y el backtest {bt['ventaja_cop_kwh']:.2f}")
            for m in md.METODOS:
                sb = [s for s in at.selectbox if s.label == "Método de umbral"][0]
                sb.set_value(m); at.run(); n += 1
                if at.exception:
                    errores.append(f"analista {rol} {hz} método {m}: {at.exception[0].message[:200]}")
            at.toggle(key="filtro_intradia_w").set_value(False); at.run(); n += 1
            sl = [s for s in at.slider if s.label.startswith("Percentiles")][0]
            sl.set_value((10, 90)); at.run(); n += 1
            if at.exception:
                errores.append(f"analista {rol} {hz} controles: {at.exception[0].message[:200]}")
    print(f"  analista listo ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # ---- Metodologia: los dos periodos
    for rol in ("generador", "comercializador"):
        at = app("metodologia", rol); n += 1
        if at.exception:
            errores.append(f"metodologia {rol}: {at.exception[0].message[:200]}")
    print(f"\n{n} ejecuciones de la app en {(time.time() - t0) / 60:.1f} min · errores: {len(errores)}")
    for e in errores[:60]:
        print("  ERROR", e)
    sys.exit(1 if errores else 0)
