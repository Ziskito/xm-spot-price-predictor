# -*- coding: utf-8 -*-
"""
Problema 2, segunda ronda. La primera (forma_del_dia.py) mostro que separar nivel y forma sube el
Corr-f de 0.799 a 0.821 con +1% de MAE, pero el MHD (ubicar el pico y el valle) no mejoro y la
persistencia sigue ganando ahi. Dos cambios:

1) Candidatos de forma basados en el PERFIL SEMANAL del precio: la mediana de la misma hora en los
   ultimos 7 dias, y la misma hora hace una semana. Capturan la forma tipica del dia (cuando cae el
   valle de la madrugada, cuando sube el pico de la noche) con menos ruido que copiar solo ayer.
   Todos se conocen en el corte (00:00): la hora t usa precios de t-24 hacia atras.
2) Version DESPLEGABLE: pesos de forma con ventana expansiva (solo ventanas anteriores) sobre el
   nivel del v4 desplegable. Sin esto no se puede afirmar que la mejora se sostiene en operacion.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stacking_24h_v2 import BASE5, pesos_lad
from metricas_forma_2026 import cargar_24h
from forma_del_dia import por_dia, hac
from o6_comun import cargar_completo, RES


def main():
    df = cargar_24h().rename(columns={"Ensamble v1 (NNLS)": "v1", "Ensamble v4 (QRA franjas)": "v4"}).sort_index()
    v4 = pd.read_csv(RES / "stacking_24h_v4_qra_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    df["v4c"] = v4["pred_causal"].reindex(df.index)
    s = cargar_completo().set_index("fecha_hora")["precio_bolsa"]
    df["perfil7"] = pd.concat([s.shift(24 * k) for k in range(1, 8)], axis=1).median(axis=1).reindex(df.index)
    df["lag168"] = s.shift(168).reindex(df.index)
    df["ventana"] = (df.index - pd.Timedelta(hours=1)).normalize()
    df["dia"] = df.index.normalize()

    todas = BASE5 + ["v4", "v4c", "perfil7", "lag168"]
    medias = df.groupby("ventana")[todas + ["real"]].transform("mean")
    dev = df[todas + ["real"]] - medias
    real = df["real"].to_numpy()
    y = dev["real"].to_numpy()

    def cv(cols):
        X = dev[cols].to_numpy()
        out = np.full(len(df), np.nan)
        dias = np.sort(df["dia"].unique())
        for dt in np.array_split(np.random.default_rng(42).permutation(dias), 5):
            te = df["dia"].isin(dt).to_numpy()
            out[te] = X[te] @ pesos_lad(X[~te], y[~te])
        return out

    def causal(cols, min_ventanas=14):
        X = dev[cols].to_numpy()
        out = np.full(len(df), np.nan)
        v = df["ventana"].to_numpy()
        vs = np.sort(np.unique(v))
        ultimo = None
        for i in range(min_ventanas, len(vs)):
            tr, te = (v < vs[i]) & ~np.isnan(X).any(1), v == vs[i]
            w = pesos_lad(X[tr], y[tr])
            out[te] = X[te] @ w
            ultimo = w
        return out, ultimo

    base_cv = BASE5 + ["v4"]
    sem_cv = BASE5 + ["v4", "perfil7", "lag168"]
    sem_c = BASE5 + ["v4c", "perfil7", "lag168"]
    forma_c, w_c = causal(sem_c)
    variantes_cv = {
        "v4 (referencia)": df["v4"].to_numpy(),
        "nivel v4 + forma ajustada (ronda 1)": medias["v4"].to_numpy() + cv(base_cv),
        "nivel v4 + forma ajustada con perfiles semanales": medias["v4"].to_numpy() + cv(sem_cv),
        "nivel v4 + forma perfil7 puro": medias["v4"].to_numpy() + dev["perfil7"].to_numpy(),
        "Persistencia": df["Persistencia"].to_numpy(),
    }
    variantes_c = {
        "v4 desplegable (referencia)": df["v4c"].to_numpy(),
        "nivel v4 desplegable + forma desplegable con perfiles": medias["v4c"].to_numpy() + forma_c,
    }

    filas = []
    for titulo, variantes, ref_nom in [("VALIDACION CRUZADA", variantes_cv, "v4 (referencia)"),
                                       ("VERSION DESPLEGABLE (tras 14 dias)", variantes_c, "v4 desplegable (referencia)")]:
        ok = ~np.isnan(np.column_stack(list(variantes.values()))).any(1)
        idx, r = df.index[ok], real[ok]
        ref = variantes[ref_nom][ok]
        ref_dia = por_dia(idx, r, ref)
        print(f"\n{titulo} -- {ok.sum():,} horas")
        print(f"{'variante':54s} {'MAE':>6s} {'MAPE':>6s} {'Corr-f':>7s} {'MHD':>5s} {'MPD':>6s}   p(MAE) p(Corr-f) p(MHD) p(MPD)")
        for nom, p in variantes.items():
            p = p[ok]
            e = np.abs(r - p)
            pd_ = por_dia(idx, r, p)
            fila = {"caso": titulo, "variante": nom, "mae": e.mean(), "mape": (e / r).mean() * 100,
                    "corr_f": pd_["corr_f"].mean(), "mhd": pd_["mhd"].mean(), "mpd": pd_["mpd"].mean()}
            ps = ""
            if nom != ref_nom:
                j = ref_dia.join(pd_, rsuffix="_n", how="inner")
                fila["p_mae"] = hac(np.abs(r - ref) - e, 23)[1]
                fila["p_corr_f"] = hac(j["corr_f_n"] - j["corr_f"], 7)[1]
                fila["p_mhd"] = hac(j["mhd"] - j["mhd_n"], 7)[1]
                fila["p_mpd"] = hac(j["mpd"] - j["mpd_n"], 7)[1]
                ps = f"  {fila['p_mae']:.4f} {fila['p_corr_f']:.4f}  {fila['p_mhd']:.4f} {fila['p_mpd']:.4f}"
            print(f"{nom:54s} {fila['mae']:6.2f} {fila['mape']:5.2f}% {fila['corr_f']:7.3f} {fila['mhd']:5.2f} {fila['mpd']:6.2f}{ps}")
            filas.append(fila)
    print("\nPesos de forma desplegables (ultimo dia): " + " | ".join(f"{m}={w:.2f}" for m, w in zip(sem_c, w_c)))
    pd.DataFrame(filas).to_csv(RES / "forma_del_dia_v2_metricas.csv", index=False)
    out = pd.DataFrame({"real": real, **variantes_cv, **variantes_c}, index=df.index)
    out.reset_index().to_csv(RES / "forma_del_dia_v2_2026.csv", index=False)


if __name__ == "__main__":
    main()
