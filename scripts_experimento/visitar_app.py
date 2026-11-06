# -*- coding: utf-8 -*-
"""
Visita el dashboard con un navegador sin pantalla para que Streamlit Community Cloud no lo deje dormido ni frio
(6-nov-2026). Lo usan .github/workflows/mantener_despierta.yml (cada hora) y actualizar_pronostico.yml (despues de
subir el pronostico, que redespliega la app: asi el arranque lo espera el robot y no el usuario).
  * si la app esta dormida, oprime «Yes, get this app back up!»
  * espera a que la app termine de cargar (el iframe /~/+/ muestra el dashboard) hasta MAX_S segundos
Uso: python scripts_experimento/visitar_app.py [--esperar-redespliegue SEG]
"""
import sys
import time

from playwright.sync_api import sync_playwright

URL = "https://augurpredictor.streamlit.app/"
MAX_S = 240


def cargada(pagina):
    """La app vive en el iframe /~/+/; cuenta como cargada cuando ese marco ya muestra el dashboard (stApp y Augur)."""
    for marco in pagina.frames:
        if "/~/+/" not in marco.url:
            continue
        try:
            if marco.locator('[data-testid="stApp"]').count() and marco.locator("text=Augur").count():
                return True
        except Exception:                                # noqa: BLE001 (marcos que se recargan durante el arranque)
            pass
    return False


if __name__ == "__main__":
    if "--esperar-redespliegue" in sys.argv:
        espera = int(sys.argv[sys.argv.index("--esperar-redespliegue") + 1])
        print(f"esperando {espera} s a que Streamlit tome el commit", flush=True)
        time.sleep(espera)
    with sync_playwright() as p:
        pagina = p.chromium.launch().new_page()
        pagina.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        pagina.wait_for_timeout(8_000)
        boton = pagina.get_by_role("button", name="Yes, get this app back up!")
        if boton.count():
            print("la app estaba dormida: se despierta", flush=True)
            boton.first.click()
        t0 = time.time()
        while time.time() - t0 < MAX_S and not cargada(pagina):
            pagina.wait_for_timeout(5_000)
        print(f"app {'cargada' if cargada(pagina) else 'SIN cargar'} tras {time.time() - t0:.0f} s", flush=True)
        pagina.wait_for_timeout(15_000)                  # mantiene la sesion abierta un momento
