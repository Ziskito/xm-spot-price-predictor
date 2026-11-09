# -*- coding: utf-8 -*-
"""
Umbrales alternativos del motor de decision, sugeridos por la revision de literatura e industria (6-nov-2026).

Vigente: rodante 30 dias (percentiles 25/75 del pronostico de los 30 dias previos) + filtro intradia.
Variantes:
  R7, R14, R60, R90   rodante con otra ventana (referencia)
  C(a,b)              combinacion de ventanas: promedio de los umbrales de a y b dias (Hubicka, Marcjasz y Weron;
                      Marcjasz, Serafin y Weron 2018: mezclar ventanas cortas y largas)
  PC(T)               ventana que se corta en el ultimo punto de cambio del nivel del precio (Nasiadka, Nitka y
                      Weron): cada dia se busca, en los 120 dias previos del pronostico diario (log), el corte que
                      mas separa las medias (segmentos de 7 dias o mas); si el estadistico t supera T, la ventana
                      va desde ese corte (minimo 7 dias, maximo 30); si no, 30 dias
  MC                  referencia economica: el MC de XM (costo promedio de los contratos del mercado regulado).
                      Comercializador: comprar si el pronostico <= MC; generador: vender si >= MC. MC del mes m-1
                      desde el dia 10 del mes m (antes, el de m-2): XM lo publica en los primeros dias del mes
  R30+MC              vigente, pero la senal favorable solo si ademas cumple la condicion del MC
Todas con el filtro intradia. Periodos: cinco años historicos (15-jul a 30-jun, 2020-21 a 2024-25) y 2026 (15-ene a
5-ago) con el pronostico realista predespacho + CatBoost (benchmark/predicciones_combinadas.csv), y la operacion
(6-ago al ultimo precio publicado, contrato de produccion). Criterio del motor: ventaja en la peor mitad de cada
periodo y frecuencia de accion entre 10 % y 40 % en las dos mitades. Medida economica adicional, frente al MC:
comercializador = promedio por hora de 1[comprar] x (MC - precio real); generador = 1[vender] x (precio real - MC)
(lo que gana por hora del periodo quien cambia contrato por bolsa en las horas que dice el motor).
Salida: resultados/escenario_realista/mejoras/motor_umbrales_alternativos.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import motor_decision as md  # noqa: E402

RES = RAIZ / "data/processed/resultados"
MEJ = RES / "escenario_realista/mejoras"
PERIODOS = [("2020-21", "2020-07-15", "2021-06-30 23:00"), ("2021-22", "2021-07-15", "2022-06-30 23:00"),
            ("2022-23", "2022-07-15", "2023-06-30 23:00"), ("2023-24", "2023-07-15", "2024-06-30 23:00"),
            ("2024-25", "2024-07-15", "2025-06-30 23:00"), ("2026", "2026-01-15", "2026-08-05 00:00")]


def precio_real():
    m = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])
                   for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]
                  + [pd.read_csv(RAIZ / "data/produccion/serie_xm.csv", usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])])
    return m.drop_duplicates("fecha_hora", keep="last").set_index("fecha_hora").sort_index()["precio_bolsa"]


def mc_disponible(fechas):
    mc = pd.read_csv(RAIZ / "data/processed/mc_mensual_xm.csv")
    mc = pd.Series(mc["mc"].to_numpy(), index=pd.PeriodIndex(mc["mes"], freq="M"))
    dia = (pd.DatetimeIndex(fechas) - pd.Timedelta(hours=1))           # dia operativo
    mes = dia.to_period("M")
    rezago = np.where(dia.day >= 10, 1, 2)
    return np.array([mc.get(m - int(k), np.nan) for m, k in zip(mes, rezago)])


def rodante(df, previo, v):
    return md.umbrales_rodantes(df["q50"], v, previo=md.precio_previo_ventana(df, previo, v))


def punto_cambio(df, previo, T, horizonte=120, minimo=7, tope=30):
    """Ventana que arranca en el ultimo punto de cambio detectado (causal, por dia)."""
    serie = pd.concat([previo[previo.index < df["fecha_hora"].min()].iloc[-tope * 24:], df.set_index("fecha_hora")["q50"]])
    diario = np.log(serie.clip(lower=1)).groupby((serie.index - pd.Timedelta(hours=1)).normalize()).mean()
    largo = {}
    dias = diario.index
    for i, d in enumerate(dias):
        h = diario.iloc[max(0, i - horizonte):i].to_numpy()        # dias anteriores a d
        L = tope
        if len(h) >= 2 * minimo:
            mejor_t, mejor_k = 0.0, None
            for k in range(minimo, len(h) - minimo + 1):
                a, b = h[:k], h[k:]
                s = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(h) - 2))
                t = abs(b.mean() - a.mean()) / (s * np.sqrt(1 / len(a) + 1 / len(b)) + 1e-9)
                if t > mejor_t:
                    mejor_t, mejor_k = t, k
            if mejor_t > T:
                L = int(np.clip(len(h) - mejor_k, minimo, tope))
        largo[d] = L
    # umbrales hora a hora con la ventana del dia
    valores = serie.to_numpy()
    pos0 = len(serie) - len(df)
    bajo, alto = np.full(len(df), np.nan), np.full(len(df), np.nan)
    dia_op = (df["fecha_hora"] - pd.Timedelta(hours=1)).dt.normalize().to_numpy()
    for j in range(len(df)):
        L = largo[pd.Timestamp(dia_op[j])] * 24
        p = pos0 + j
        if p - L >= 0:
            w = valores[p - L:p]
            bajo[j], alto[j] = np.nanquantile(w, 0.25), np.nanquantile(w, 0.75)
    return pd.Series(bajo, index=df.index), pd.Series(alto, index=df.index), pd.Series(largo)


def senal(df, bajo, alto, rol):
    p = df["q50"].to_numpy()
    caro, barato = p >= np.asarray(alto), p <= np.asarray(bajo)
    et = ["vender", "retener"] if rol == "generador" else ["evitar_compra", "comprar"]
    return md.filtro_intradia(df, pd.Series(np.select([caro, barato], et, default="esperar"), index=df.index), rol)


def evaluar(df, s, rol, mc):
    corte = df["fecha_hora"].min() + (df["fecha_hora"].max() - df["fecha_hora"].min()) / 2
    m1 = (df["fecha_hora"] < corte).to_numpy()
    accion = "comprar" if rol == "comercializador" else "vender"
    h1, h2 = md.evaluar_backtest(df[m1], s[m1], rol), md.evaluar_backtest(df[~m1], s[~m1], rol)
    act = (s == accion).to_numpy()
    gana = (mc - df["real"].to_numpy()) if rol == "comercializador" else (df["real"].to_numpy() - mc)
    eco = np.where(act, gana, 0.0)
    ok = all(0.10 <= h["frecuencia_accion"] <= 0.40 and h["ventaja_cop_kwh"] >= 0 for h in (h1, h2))
    return {"peor_mitad": min(h1["ventaja_cop_kwh"], h2["ventaja_cop_kwh"]), "estable": ok,
            "frec_min_%": 100 * min(h1["frecuencia_accion"], h2["frecuencia_accion"]),
            "frec_max_%": 100 * max(h1["frecuencia_accion"], h2["frecuencia_accion"]),
            "vs_MC_por_hora": float(np.nanmean(eco)),
            "vs_MC_peor_mitad": float(min(np.nanmean(eco[m1]), np.nanmean(eco[~m1])))}


if __name__ == "__main__":
    real = precio_real()
    comb = pd.read_csv(RES / "escenario_realista/benchmark/predicciones_combinadas.csv", parse_dates=["fecha_hora"])
    prod = pd.read_csv(RES / "pronostico_realista_bandas_24h_2026.csv", parse_dates=["fecha_hora"])
    tramos = []
    for nom, ini, fin in PERIODOS:
        x = comb[(comb["fecha_hora"] >= ini) & (comb["fecha_hora"] <= fin)][["fecha_hora", "real", "Pre + CatBoost"]]
        tramos.append((nom, x.rename(columns={"Pre + CatBoost": "q50"})))
    op = prod[(prod["fecha_hora"] >= "2026-08-06 01:00") & prod["real"].notna()][["fecha_hora", "real", "q50"]]
    tramos.append(("operación ago-nov", op))

    variantes = ["R30 (vigente)", "R7", "R14", "R60", "R90", "C(7,90)", "C(14,60)", "C(30,365)", "PC(T=5)", "PC(T=10)",
                 "MC", "R30+MC"]
    filas, largos = [], []
    for nom, df in tramos:
        df = df.dropna(subset=["q50"]).sort_values("fecha_hora").reset_index(drop=True)
        previo = real[real.index <= df["fecha_hora"].min() - pd.Timedelta(hours=73)]
        mc = mc_disponible(df["fecha_hora"])
        U = {v: rodante(df, previo, v) for v in (7, 14, 30, 60, 90, 365)}
        umb = {"R30 (vigente)": U[30], "R7": U[7], "R14": U[14], "R60": U[60], "R90": U[90]}
        for a, b in ((7, 90), (14, 60), (30, 365)):
            umb[f"C({a},{b})"] = ((U[a][0] + U[b][0]) / 2, (U[a][1] + U[b][1]) / 2)
        for T in (5, 10):
            b_, a_, L = punto_cambio(df, previo, T)
            umb[f"PC(T={T})"] = (b_, a_)
            largos.append({"periodo": nom, "T": T, "dias_con_ventana_corta_%": float((L < 30).mean() * 100),
                           "ventana_media_dias": float(L.mean())})
        mcs = pd.Series(mc, index=df.index)
        for rol in ("generador", "comercializador"):
            for v in variantes:
                if v == "MC":
                    s = senal(df, mcs, mcs, rol)
                elif v == "R30+MC":
                    s = senal(df, U[30][0], U[30][1], rol)
                    fav = "comprar" if rol == "comercializador" else "vender"
                    no_mc = (df["q50"] > mcs) if rol == "comercializador" else (df["q50"] < mcs)
                    s[(s == fav) & no_mc] = "esperar"
                else:
                    s = senal(df, *umb[v], rol)
                filas.append({"periodo": nom, "rol": rol, "variante": v, **evaluar(df, s, rol, mc)})
        print(f"  {nom} listo", flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(MEJ / "motor_umbrales_alternativos.csv", index=False)
    pd.DataFrame(largos).to_csv(MEJ / "motor_umbrales_puntos_cambio.csv", index=False)
    pd.set_option("display.width", 250)
    for rol in ("generador", "comercializador"):
        x = t[t["rol"] == rol]
        print(f"\n=== {rol}: ventaja en la peor mitad (COP/kWh); * = no estable")
        tab = x.assign(v=x["peor_mitad"].round(0).astype("Int64").astype(str) + np.where(x["estable"], "", "*")
                       ).pivot(index="variante", columns="periodo", values="v").reindex(variantes)
        tab["estable en"] = x.groupby("variante")["estable"].sum().reindex(variantes).astype(int).astype(str) + "/7"
        print(tab.to_string())
        print(f"--- {rol}: ganancia frente al contrato (MC), COP/kWh por hora del periodo")
        print(x.pivot(index="variante", columns="periodo", values="vs_MC_por_hora").reindex(variantes).round(1).to_string())
    print(pd.DataFrame(largos).round(1).to_string(index=False))
