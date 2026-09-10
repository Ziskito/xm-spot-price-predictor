# -*- coding: utf-8 -*-
"""
Descubre que metricas ofrece el API de XM que todavia no estamos usando, en particular:
  - precio de gas / combustible / costo marginal termico
  - disponibilidad declarada y mantenimientos programados
  - intercambios internacionales (TIE, Ecuador)
  - cualquier variable forward-looking (declaraciones, ofertas, programado vs real)

Guarda el catalogo completo en data/catalogo_metricas_xm.csv para poder consultarlo sin
volver a llamar al API.
"""
import requests
import pandas as pd
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
URL_LISTS = "https://servapibi.xm.com.co/lists"

payload = {"MetricId": "ListadoMetricas", "StartDate": "2026-01-01", "EndDate": "2026-01-01", "Entity": "Sistema"}
r = requests.post(URL_LISTS, json=payload, timeout=120)
r.raise_for_status()
datos = r.json()

filas = []
for item in datos.get("Items", []):
    for entidad in item.get("ListEntities", []):
        valores = entidad.get("Values", {})
        filas.append(valores if isinstance(valores, dict) else {"raw": str(valores)})

df = pd.DataFrame(filas)
print("Columnas del catalogo:", df.columns.tolist())
print("Total de metricas:", len(df))

ruta = RAIZ / "data" / "catalogo_metricas_xm.csv"
df.to_csv(ruta, index=False, encoding="utf-8-sig")
print("Guardado:", ruta)

# --- Busqueda por tema ---
col_nombre = next((c for c in df.columns if "name" in c.lower() or "nombre" in c.lower()), None)
col_id = next((c for c in df.columns if c.lower().endswith("id")), None)
col_desc = next((c for c in df.columns if "desc" in c.lower()), None)
col_ent = next((c for c in df.columns if "entity" in c.lower()), None)
col_url = next((c for c in df.columns if "url" in c.lower()), None)

busquedas = {
    "GAS / COMBUSTIBLE / COSTO": ["gas", "combustib", "fuel", "costo", "cost", "marginal", "arranque", "parada"],
    "DISPONIBILIDAD / MANTENIMIENTO": ["disponib", "mantenim", "indisponib", "declarad"],
    "INTERNACIONAL / ECUADOR": ["intern", "tie", "import", "export", "ecuador", "interconex"],
    "OFERTAS / DECLARACIONES (forward-looking)": ["oferta", "declara", "program", "despacho", "predespacho", "ideal"],
    "PRECIO (otros)": ["precio", "escasez", "contrato"],
    "RESTRICCIONES": ["restric", "reconcil"],
}

texto_busqueda = df.apply(lambda fila: " ".join(str(v) for v in fila.values).lower(), axis=1)

for tema, claves in busquedas.items():
    mascara = texto_busqueda.apply(lambda t: any(k in t for k in claves))
    sub = df[mascara]
    print(f"\n{'='*78}\n### {tema}  ({len(sub)} metricas)\n{'='*78}")
    for _, fila in sub.iterrows():
        partes = []
        if col_id: partes.append(f"{fila[col_id]}")
        if col_nombre: partes.append(f"{fila[col_nombre]}")
        if col_ent: partes.append(f"[{fila[col_ent]}]")
        if col_url: partes.append(f"{str(fila[col_url]).split('/')[-1]}")
        print("  " + " | ".join(str(p) for p in partes))
