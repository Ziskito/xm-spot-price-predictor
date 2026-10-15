# -*- coding: utf-8 -*-
"""
Actualizacion diaria del dashboard augur (desde el 14-oct-2026). La corre GitHub Actions cada dia
(.github/workflows/actualizar_pronostico.yml) despues de que XM publica el predespacho ideal (~10:30) y el despacho
programado (~14:30) del dia siguiente; tambien se puede correr a mano: python scripts_experimento/actualizar_produccion.py
  1) produccion_datos.py: datos nuevos de XM (precio, demanda, generacion, embalses, aportes, predespacho, dMAR/dDEM) y
     ONI de NOAA, en data/produccion/ (el dataset oficial del proyecto no se toca)
  2) pronostico_realista_24h.py --produccion --activar: CatBoost del mes reentrenados con todo lo publicado,
     pronostico hasta la ventana de manana (si ya salio su predespacho), combinador y bandas
  3) realista_eventos.py --produccion --operacion: probabilidades de desplome y pico
  4) pronostico_realista_72h.py --produccion --activar: pasos 25-72 y contrato de 72 h
  5) resultados/produccion/estado.json: hora de la corrida y hasta donde llegan los datos y los pronosticos
Lo ya publicado (ventanas y cortes de hoy o antes) no se reescribe; el precio real si se completa a medida que XM lo
publica. Si un paso 2-4 falla, la corrida termina con error y el workflow no publica nada.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import produccion_datos as pdat  # noqa: E402

RES = pdat.PROC / "resultados"


def correr(*args):
    t0 = time.time()
    print(f"== {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, str(AQUI / args[0]), *args[1:]], cwd=AQUI.parent)
    if r.returncode:
        raise SystemExit(f"fallo {args[0]} (codigo {r.returncode})")
    print(f"   {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    inicio = pdat.hoy()
    print("== datos", flush=True)
    fuentes = pdat.actualizar()
    correr("pronostico_realista_24h.py", "--produccion", "--activar")
    correr("realista_eventos.py", "--produccion", "--operacion")
    correr("pronostico_realista_72h.py", "--produccion", "--activar")
    c24 = pd.read_csv(RES / "pronostico_realista_bandas_24h_2026.csv", parse_dates=["fecha_hora"])
    c72 = pd.read_csv(RES / "pronostico_realista_bandas_72h_2026.csv", parse_dates=["fecha_hora", "cutoff"])
    estado = {"actualizado": inicio.strftime("%Y-%m-%d %H:%M"),
              "precio_real_hasta": str(c24.loc[c24["real"].notna(), "fecha_hora"].max()),
              "pronostico_24h_hasta": str(c24["fecha_hora"].max()),
              "pronostico_72h_ultimo_corte": str(c72["cutoff"].max().date()),
              "pronostico_72h_hasta": str(c72["fecha_hora"].max()),
              "fuentes": fuentes}
    (RES / "produccion").mkdir(exist_ok=True)
    (RES / "produccion" / "estado.json").write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(estado, indent=2, ensure_ascii=False))
    print(f"Listo en {(pdat.hoy() - inicio).total_seconds() / 60:.1f} min")
