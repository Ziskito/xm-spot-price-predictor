# -*- coding: utf-8 -*-
"""Delta ex post del precio de bolsa = PrecBolsNaci - MaxPrecOferNal (API publica de XM), 2018-2026.
XM lo publica con el mismo rezago que el precio (~3 dias). Salida: data/raw/delta_expost_2018_2026.csv"""
import time
import pandas as pd
import requests

def post(met, a, b):
    for i in range(8):
        try:
            return requests.post("https://servapibi.xm.com.co/hourly", timeout=60, json={"MetricId": met, "StartDate": f"{a:%Y-%m-%d}",
                                 "EndDate": f"{b:%Y-%m-%d}", "Entity": "Sistema"}).json()
        except Exception:
            time.sleep(5 * (i + 1))
    raise RuntimeError(met)

def bajar(met, ini, fin):
    out, a = {}, pd.Timestamp(ini)
    while a <= pd.Timestamp(fin):
        b = min(a + pd.Timedelta(days=27), pd.Timestamp(fin))
        for it in post(met, a, b).get("Items", []):
            v = it["HourlyEntities"][0]["Values"]
            for h in range(1, 25):
                x = v.get(f"Hour{h:02d}")
                if x not in (None, ""):
                    out[pd.Timestamp(it["Date"]) + pd.Timedelta(hours=h - 1)] = float(x)
        a = b + pd.Timedelta(days=1)
    return pd.Series(out).sort_index()

if __name__ == "__main__":
    viejo = pd.read_csv("data/raw/delta_expost_2024_2026.csv", parse_dates=[0], index_col=0)
    nuevo = pd.DataFrame({"precio": bajar("PrecBolsNaci", "2018-01-01", "2023-12-31"), "mpo_expost": bajar("MaxPrecOferNal", "2018-01-01", "2023-12-31")}).dropna()
    nuevo["delta_expost"] = nuevo.precio - nuevo.mpo_expost
    t = pd.concat([nuevo, viejo]).sort_index()
    t = t[~t.index.duplicated(keep="last")]
    t.to_csv("data/raw/delta_expost_2018_2026.csv")
    print("guardado", len(t), t.index.min(), t.index.max())
