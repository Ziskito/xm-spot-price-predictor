# -*- coding: utf-8 -*-
"""
Auditoria del dashboard en un navegador real (Chromium, Playwright), 10-nov-2026: lo que streamlit.testing no ve porque
es JavaScript de la pagina. En las tres vistas, temas claro y oscuro, y anchos de escritorio y celular:
  * errores de JavaScript y de consola;
  * cada boton Exportar (JPG y PDF) y los reportes del dia y completo deben descargar un archivo no vacio;
  * Ampliar abre la grafica grande y se cierra; Centrar y el clic en una hora de la franja responden;
  * Metodologia: el video y el cambio a «Todo 2026»; Analista: «Ver datos filtrados».
Uso: python scripts_experimento/auditar_navegador.py [url]   (por defecto http://localhost:8501)
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
SAL = Path(__file__).resolve().parents[1] / "scratch_auditoria"
errores, avisos = [], []


def esperar_app(pg, maximo=90):
    t0 = time.time()
    while time.time() - t0 < maximo:
        listo = pg.evaluate("""() => { const a = document.querySelector('[data-testid="stApp"]');
                               return a && a.getAttribute('data-test-script-state') === 'notRunning'; }""")
        if listo:
            pg.wait_for_timeout(2500)
            return True
        pg.wait_for_timeout(1000)
    return False


def descargas(pg, etiqueta, selector):
    botones = pg.locator(selector)
    for i in range(botones.count()):
        b = botones.nth(i)
        nombre = (b.get_attribute("data-f") or b.get_attribute("data-tipo") or "?")
        try:
            b.scroll_into_view_if_needed()
            with pg.expect_download(timeout=60_000) as d:
                b.click()
            ruta = d.value.path()
            tam = Path(ruta).stat().st_size if ruta else 0
            if tam < 2_000:
                errores.append(f"{etiqueta}: la descarga {selector} #{i} ({nombre}) pesa {tam} bytes")
        except Exception as e:  # noqa: BLE001
            errores.append(f"{etiqueta}: el botón {selector} #{i} ({nombre}) no descargó nada ({str(e)[:80]})")
        pg.wait_for_timeout(500)
    return botones.count()


def revisar(pg, etiqueta, vista, tema, ancho):
    consola = []
    pg.on("pageerror", lambda e: consola.append(f"JS: {e}"))
    pg.on("console", lambda m: consola.append(f"consola: {m.text}") if m.type == "error" else None)
    pg.goto(f"{URL}/?vista={vista}&tema={tema}", timeout=120_000)
    if not esperar_app(pg):
        errores.append(f"{etiqueta}: la app no terminó de cargar"); return
    exc = pg.locator('[data-testid="stException"]')
    if exc.count():
        errores.append(f"{etiqueta}: excepción en pantalla: {exc.first.inner_text()[:200]}")
    if vista == "operador":
        n = descargas(pg, etiqueta, ".exp-bar button")
        if n < 6:
            avisos.append(f"{etiqueta}: solo {n} botones de exportar")
        if ancho > 900:
            descargas(pg, etiqueta, ".augur-rep")
        amp = pg.locator(".st-key-ampliar_dia button")
        if not amp.count():
            errores.append(f"{etiqueta}: no aparece el botón Ampliar")
        else:
            amp.first.click(); pg.wait_for_timeout(2500)
            if not pg.locator('[role="dialog"]').count():
                errores.append(f"{etiqueta}: Ampliar no abrió la gráfica grande")
            else:
                pg.keyboard.press("Escape"); pg.wait_for_timeout(1500)
        cen = pg.locator(".st-key-centrar_dia button")
        if cen.count():
            cen.first.click(); esperar_app(pg, 30)
        h = pg.locator(".st-key-hora_btn_20 button")
        if not h.count():
            errores.append(f"{etiqueta}: no aparece la hora 20 de la franja")
        else:
            h.first.scroll_into_view_if_needed(); h.first.click(); esperar_app(pg, 60)
            v = pg.locator(".st-key-hora_operador input").input_value()
            if v != "20":
                errores.append(f"{etiqueta}: el clic en la hora 20 de la franja dejó la hora en {v}")
    if vista == "metodologia":
        if not pg.locator("iframe[src*='youtube']").count():
            errores.append(f"{etiqueta}: no aparece el video de YouTube")
        b = pg.get_by_text("Todo 2026", exact=True)
        if not b.count():
            errores.append(f"{etiqueta}: no aparece el selector «Todo 2026»")
        else:
            antes = pg.locator(".st-key-sec_desempeno").inner_text()
            b.first.click(); esperar_app(pg, 60)
            if pg.locator(".st-key-sec_desempeno").inner_text() == antes:
                errores.append(f"{etiqueta}: «Todo 2026» no cambió las cifras")
    if vista == "analista":
        e = pg.get_by_text("Ver datos filtrados").first
        if not e.count():
            errores.append(f"{etiqueta}: no aparece «Ver datos filtrados»")
        else:
            e.scroll_into_view_if_needed(); e.click(); pg.wait_for_timeout(2500)
    ancho_doc = pg.evaluate("() => document.documentElement.scrollWidth")
    if ancho_doc > ancho + 4:
        errores.append(f"{etiqueta}: la página se desborda a lo ancho ({ancho_doc} px en una ventana de {ancho})")
    exc = pg.locator('[data-testid="stException"]')
    if exc.count():
        errores.append(f"{etiqueta}: excepción tras interactuar: {exc.first.inner_text()[:200]}")
    for c in consola:
        if "favicon" not in c and "ResizeObserver" not in c:
            errores.append(f"{etiqueta}: {c[:200]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    with sync_playwright() as p:
        nav = p.chromium.launch()
        for vista in ("operador", "analista", "metodologia"):
            for tema in ("claro", "oscuro"):
                for ancho, alto in ((1440, 950), (390, 844)):
                    ctx = nav.new_context(viewport={"width": ancho, "height": alto}, accept_downloads=True)
                    pg = ctx.new_page()
                    etiqueta = f"{vista} {tema} {ancho}px"
                    revisar(pg, etiqueta, vista, tema, ancho)
                    ctx.close()
                    print(f"  {etiqueta} ({(time.time() - t0) / 60:.1f} min, {len(errores)} errores)", flush=True)
        nav.close()
    print(f"\nerrores: {len(errores)} · avisos: {len(avisos)}")
    for e in errores:
        print("  ERROR", e)
    for a in avisos:
        print("  AVISO", a)
    sys.exit(1 if errores else 0)
