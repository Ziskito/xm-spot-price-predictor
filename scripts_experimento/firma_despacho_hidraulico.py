# -*- coding: utf-8 -*-
"""
Firma de despacho hidraulico (idea de Ssembatya et al. 2026, Energy Reports): el hallazgo del
paper es que las DECISIONES DE PROGRAMACION del despacho hidraulico (como se reparte la
generacion dentro de la semana) pesan mas que la cantidad de agua disponible para explicar el
error de pronostico de precio. Hasta ahora solo usamos NIVELES (volumen_embalses, aportes_hidricos)
-- nunca una variable de "forma" del despacho.

Construccion: para cada dia, se mide que tan CONCENTRADA esta la generacion hidraulica en las
horas de precio alto de ESE MISMO DIA (correlacion de Spearman entre el perfil horario de
generacion hidraulica y el perfil horario de precio). Un valor alto significa que los operadores
estan despachando agua estrategicamente en las horas caras (comportamiento "smart"); un valor
bajo/negativo sugiere generacion mas pareja o incluso invertida.

OJO CON LA FUGA: el score de un dia usa el precio DE ESE MISMO DIA -- no se puede usar tal cual
para pronosticar ese dia. Se usa el PROMEDIO DE LOS ULTIMOS 7 DIAS de ese score (ya conocido al
momento de pronosticar) como la feature real, igual que el resto de variables rezagadas del
pipeline.

Primero se tamiza con correlacion parcial (mismo metodo que las 12 variables nuevas de anoche)
antes de gastar tiempo de entrenamiento de red neuronal en probarla.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")


def correlacion_parcial(x, y, z):
    d = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(d) < 100:
        return np.nan
    def residuo(v):
        A = np.vstack([np.ones(len(d)), d["z"].values]).T
        coef, *_ = np.linalg.lstsq(A, d[v].values, rcond=None)
        return d[v].values - A @ coef
    rx, ry = residuo("x"), residuo("y")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    print("Cargando generacion por tipo (hidraulica) y precio...")
    gen = pd.read_csv(RAIZ / "data" / "generacion_por_tipo_2019_2026.csv", parse_dates=["fecha_hora"])
    a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
    b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
    base = pd.concat([a, b], ignore_index=True)[["fecha_hora", "precio_bolsa"]].sort_values("fecha_hora")

    # precio SIN rezagar (el real de esa hora), necesario para construir el score dia a dia
    df = gen[["fecha_hora", "HIDRAULICA"]].merge(base, on="fecha_hora", how="inner")
    df["fecha"] = df["fecha_hora"].dt.normalize()

    print(f"Filas: {len(df):,} | dias: {df['fecha'].nunique():,}")

    print("\nCalculando score diario de alineacion despacho-precio (Spearman intra-dia)...")
    scores = []
    for fecha, grupo in df.groupby("fecha"):
        if len(grupo) < 20:  # dia incompleto
            continue
        rho, _ = spearmanr(grupo["HIDRAULICA"], grupo["precio_bolsa"])
        scores.append({"fecha": fecha, "alineacion_despacho": rho})

    df_scores = pd.DataFrame(scores).sort_values("fecha").set_index("fecha")
    print(f"Dias con score: {len(df_scores)}")
    print(df_scores["alineacion_despacho"].describe())

    # Rezagar: promedio movil de 7 dias, EXCLUYENDO el dia actual (shift(1) antes del rolling)
    df_scores["alineacion_media_7d"] = df_scores["alineacion_despacho"].shift(1).rolling(7).mean()
    df_scores["alineacion_lag1d"] = df_scores["alineacion_despacho"].shift(1)

    # ANOMALIA: la alineacion esta casi siempre alta (mean 0.90) -- el nivel no varia lo
    # suficiente dia a dia para ser informativo. Prueba refinada: desviacion respecto a la norma
    # reciente (30 dias) -- ¿hoy el despacho fue MAS o MENOS estrategico de lo usual?
    norma_30d = df_scores["alineacion_despacho"].shift(1).rolling(30).mean()
    df_scores["alineacion_anomalia_lag1d"] = df_scores["alineacion_lag1d"] - norma_30d
    df_scores["alineacion_anomalia_media_7d"] = df_scores["alineacion_anomalia_lag1d"].rolling(7).mean()

    # Expandir a horaria y unir con el dataset base para el tamiz
    df_scores_diario = df_scores.reset_index()
    base_h = base.copy()
    base_h["fecha"] = base_h["fecha_hora"].dt.normalize()
    m = base_h.merge(df_scores_diario, on="fecha", how="left").sort_values("fecha_hora")

    m["precio_lag24h_ref"] = m["precio_bolsa"].shift(24)
    ref = m["precio_lag24h_ref"].corr(m["precio_bolsa"])
    print(f"\nREFERENCIA -- corr(precio_t-24h, precio_t) = {ref:.4f}")

    for col in ["alineacion_lag1d", "alineacion_media_7d", "alineacion_anomalia_lag1d", "alineacion_anomalia_media_7d"]:
        corr_cont = m[col].corr(m["precio_bolsa"])
        corr_parc = correlacion_parcial(m[col], m["precio_bolsa"], m["precio_lag24h_ref"])
        veredicto = ("CANDIDATA REAL" if not pd.isna(corr_parc) and abs(corr_parc) >= 0.05
                      else "redundante/debil")
        print(f"  {col:22s} corr_cruda={corr_cont:+.4f}  corr_PARCIAL={corr_parc:+.4f}   {veredicto}")

    # Guardar para uso posterior si pasa el tamiz
    ruta_out = RAIZ / "data" / "firma_despacho_hidraulico.csv"
    df_scores_diario.to_csv(ruta_out, index=False)
    print(f"\nGuardado: {ruta_out}")


if __name__ == "__main__":
    main()
