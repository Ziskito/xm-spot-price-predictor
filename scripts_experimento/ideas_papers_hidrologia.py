# -*- coding: utf-8 -*-
"""
Ideas tomadas de dos papers de mercados hidro-dominados, probadas sobre nuestro modelo (2026-09-17).

PAPER 1 -- "Electricity price forecasting across Norway's five bidding zones in the post-crisis era"
(arXiv 2604.26634). Mercado noruego, >90% hidro, 5 zonas. Su resultado mas fuerte es una ablacion
LOGO (leave-one-group-out): quitar los rezagos de precio cuesta 17-24 EUR/MWh de MAE, quitar el
calendario cuesta 0.04-0.42, y quitar los EMBALSES cuesta ~0.01-0.03 (es decir, nada). Conclusion
del paper: "las exogenas aportan poco a la precision puntual una vez que la estructura
autorregresiva y estacional ya esta representada", y su recomendacion es usarlas para ESTRATIFICAR
EL RIESGO (saber cuando el modelo va a fallar), no para bajar el error.

PAPER 2 -- "How optimistic inflow forecasts distort dispatch, prices, and contracts in
hydro-dominated power systems: evidence from Brazil" (arXiv 2607.00504v2). Muestra que el error
ACUMULADO de pronostico de aportes se traslada al precio con rezago de 1-3 meses, porque el nivel
de embalse "guarda memoria" de esos errores. Su recomendacion explicita para quien modela precio:
incluir el ESTADO DE EMBALSE y la ANOMALIA ACUMULADA de aportes, no solo el aporte puntual del dia,
y segmentar por regimen humedo/seco porque el mecanismo de precio es distinto en cada uno.

QUE SE PRUEBA AQUI (sobre los modelos directos de 24h, que son rapidos):
  A) Ablacion LOGO por grupo de variables -- replicar el experimento noruego con nuestros datos.
     Responde: nuestra hidrologia, .aporta o es decorativa?
  B) Grupo NUEVO de variables de "memoria hidrologica" (idea del paper brasileno): anomalia de
     aportes y de embalse contra su normal estacional (climatologia por dia del ano calculada SOLO
     con entrenamiento), su acumulado a 30/60/90 dias, el deficit acumulado (solo anomalias
     negativas, que es lo que seca un embalse) y la interaccion embalse x ONI.
  C) Estratificacion del error por celdas de regimen (embalse alto/bajo x ONI alto/bajo), que es la
     recomendacion practica del paper noruego y alimenta directamente el requisito de "confianza"
     en los metadatos de las imagenes de decision (OE3, Anexo 1).
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from xgboost_directo_72h import construir_largo, COLS_ESTADO, COLS_FUTURO
from o6_comun import cargar_completo, RES

H = 24
PASO_MUESTREO = 4
CORTE = pd.Timestamp("2026-01-01")

# --- Grupos de variables, para la ablacion LOGO (replica del experimento noruego) ---
GRUPOS = {
    "precio (rezagos e historia)": ["precio_bolsa", "precio_media_24h", "precio_media_7d", "precio_media_30d",
                                     "precio_std_24h", "precio_std_7d", "precio_rango_24h", "ratio_volatilidad",
                                     "precio_lag168h", "precio_mismo_hora_disp"],
    "calendario": ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos", "dia_anio_sin", "dia_anio_cos",
                    "es_pandemia", "paso_horas"],
    "festivos": ["es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h", "festivo_lag168h",
                  "mismatch_festivo_24h", "mismatch_festivo_168h"],
    "hidrologia (actual)": ["volumen_embalses", "volumen_embalses_vs_media30d", "volumen_embalses_delta_1d",
                             "volumen_embalses_delta_7d", "aportes_hidricos", "aportes_hidricos_vs_media30d",
                             "aportes_hidricos_delta_1d", "aportes_hidricos_delta_7d"],
    "demanda y generacion": ["demanda", "demanda_media_24h", "generacion"],
    "ONI": ["oni"],
}
NUEVAS = ["aportes_anom", "aportes_anom_acum30", "aportes_anom_acum90", "deficit_acum90",
          "volumen_anom", "volumen_anom_acum30", "volumen_anom_x_oni"]


def agregar_memoria_hidrologica(df):
    """Idea del paper brasileno: el precio no responde al aporte de HOY sino a la desviacion
    ACUMULADA frente a lo normal, porque es eso lo que vacia o llena el embalse. La climatologia
    (normal por dia del ano) se calcula SOLO con datos anteriores a 2026 -- sin fuga."""
    d = df.copy()
    d["doy"] = d["fecha_hora"].dt.dayofyear
    train = d[d["fecha_hora"] < CORTE]
    for col, base in [("aportes_hidricos", "aportes"), ("volumen_embalses", "volumen")]:
        clim = train.groupby("doy")[col].mean()
        d[f"{base}_anom"] = d[col] - d["doy"].map(clim)
        # el acumulado se calcula sobre la serie horaria; 30d = 720 h
        s = d[f"{base}_anom"]
        d[f"{base}_anom_acum30"] = s.rolling(24 * 30, min_periods=24).mean()
        if base == "aportes":
            d["aportes_anom_acum90"] = s.rolling(24 * 90, min_periods=24).mean()
            # deficit: solo la parte negativa (lo que seca el embalse), acumulada
            d["deficit_acum90"] = s.clip(upper=0).rolling(24 * 90, min_periods=24).mean()
    # interaccion embalse x ONI: un embalse bajo con El Nino fuerte no es lo mismo que con La Nina
    z = (d["volumen_anom"] - train["volumen_embalses"].mean() * 0) / d["volumen_anom"].std()
    d["volumen_anom_x_oni"] = z * d["oni"]
    return d


def construir_datos(df, cols_extra):
    pos = pd.Series(df.index.values, index=df["fecha_hora"])
    idx_fin = int(pos[CORTE])
    cortes_test = pd.date_range("2025-12-31 23:00", df["fecha_hora"].max() - pd.Timedelta(hours=H), freq="24h")
    idx_test = pos.reindex(cortes_test).dropna().astype(int).to_numpy()
    idx_train = np.arange(200 + 24 * 90, idx_fin - H, PASO_MUESTREO)
    pasos = np.arange(1, H + 1)
    estado = COLS_ESTADO + [c for c in cols_extra if c in df.columns]
    import xgboost_directo_72h as xd
    orig = xd.COLS_ESTADO
    xd.COLS_ESTADO = estado  # el constructor lee el estado en el momento del corte
    try:
        tr = construir_largo(df, idx_train, pasos).dropna()
        te = construir_largo(df, idx_test, pasos).dropna()
    finally:
        xd.COLS_ESTADO = orig
    cols = estado + COLS_FUTURO + ["precio_mismo_hora_disp", "paso_horas"]
    return tr, te, cols


def evaluar(tr, te, cols, etiqueta, semilla=42):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(n_estimators=800, max_depth=8, learning_rate=0.05, loss_function="MAE",
                          random_seed=semilla, verbose=0, thread_count=-1)
    m.fit(tr[cols], tr["y"])
    p = m.predict(te[cols])
    e = np.abs(te["y"].to_numpy() - p)
    return p, e.mean(), (e / te["y"].to_numpy()).mean() * 100


def dm(real, p_ref, p_new, maxlags=23):
    d = np.abs(real - p_ref) - np.abs(real - p_new)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def main():
    t0 = time.time()
    df = agregar_memoria_hidrologica(cargar_completo())
    print(f"Variables nuevas creadas: {[c for c in NUEVAS if c in df.columns]}")

    # ---------- B) linea base vs. linea base + memoria hidrologica ----------
    tr_b, te_b, cols_b = construir_datos(df, [])
    print(f"\nDataset: {len(tr_b):,} filas de entrenamiento, {len(te_b):,} de prueba, {len(cols_b)} variables")
    real = te_b["y"].to_numpy()
    p_base, mae_base, mape_base = evaluar(tr_b, te_b, cols_b, "base")
    print(f"\n{'='*78}\nB) MEMORIA HIDROLOGICA (idea del paper brasileno)\n{'='*78}")
    print(f"  {'linea base (variables actuales)':45s} MAE={mae_base:6.2f}  MAPE={mape_base:5.2f}%")

    tr_n, te_n, cols_n = construir_datos(df, NUEVAS)
    p_new, mae_new, mape_new = evaluar(tr_n, te_n, cols_n, "nuevas")
    t, pv = dm(te_n["y"].to_numpy(), p_base, p_new)
    marca = "GANA (sig.)" if pv < 0.05 and t > 0 else ("PIERDE (sig.)" if pv < 0.05 else "sin diferencia")
    print(f"  {'+ memoria hidrologica (7 variables nuevas)':45s} MAE={mae_new:6.2f}  MAPE={mape_new:5.2f}%  "
          f"({100*(mae_base-mae_new)/mae_base:+.1f}%, DM p={pv:.4f} {marca})")

    # ---------- A) ablacion LOGO (replica del experimento noruego) ----------
    print(f"\n{'='*78}\nA) ABLACION LOGO -- cuanto cuesta quitar cada grupo (replica del paper noruego)\n{'='*78}")
    print(f"  {'grupo quitado':35s} {'MAE':>7s} {'delta MAE':>10s} {'DM p':>8s}")
    filas = [{"prueba": "base", "grupo": "(ninguno)", "mae": mae_base, "delta": 0.0, "dm_p": np.nan}]
    for nombre, miembros in GRUPOS.items():
        quitar = [c for c in miembros if c in cols_b]
        if not quitar:
            continue
        cols_sin = [c for c in cols_b if c not in quitar]
        p_sin, mae_sin, _ = evaluar(tr_b, te_b, cols_sin, nombre)
        t, pv = dm(real, p_sin, p_base)  # base vs sin-grupo: positivo = base mejor = el grupo aporta
        print(f"  {nombre:35s} {mae_sin:7.2f} {mae_sin-mae_base:+10.2f} {pv:8.4f}"
              f"  {'aporta (sig.)' if pv < 0.05 and t > 0 else 'no aporta'}")
        filas.append({"prueba": "LOGO", "grupo": nombre, "mae": mae_sin,
                      "delta": mae_sin - mae_base, "dm_p": pv})
    pd.DataFrame(filas).to_csv(RES / "ablacion_logo_24h.csv", index=False)

    # ---------- C) estratificacion del error por celdas de regimen ----------
    print(f"\n{'='*78}\nC) DONDE FALLA EL MODELO -- celdas de regimen (recomendacion del paper noruego)\n{'='*78}")
    te_c = te_b.copy()
    te_c["error"] = np.abs(real - p_base)
    fh = pd.to_datetime(te_c["fecha_hora"])
    ctx = df.set_index("fecha_hora")
    te_c["volumen_anom"] = ctx["volumen_anom"].reindex(fh).to_numpy()
    te_c["oni"] = ctx["oni"].reindex(fh).to_numpy()
    te_c["embalse"] = np.where(te_c["volumen_anom"] >= te_c["volumen_anom"].median(), "alto", "bajo")
    te_c["enso"] = np.where(te_c["oni"] >= te_c["oni"].median(), "El Nino (ONI alto)", "neutral/Nina (ONI bajo)")
    tabla = te_c.pivot_table(index="embalse", columns="enso", values="error", aggfunc="mean").round(1)
    conteo = te_c.pivot_table(index="embalse", columns="enso", values="error", aggfunc="size")
    print("  MAE por celda de regimen:\n" + tabla.to_string())
    print("\n  horas por celda:\n" + conteo.to_string())
    ef_emb = te_c[te_c.embalse == "bajo"]["error"].mean() - te_c[te_c.embalse == "alto"]["error"].mean()
    ef_oni = (te_c[te_c.enso.str.startswith("El Nino")]["error"].mean()
              - te_c[te_c.enso.str.startswith("neutral")]["error"].mean())
    print(f"\n  Efecto embalse bajo: {ef_emb:+.1f} COP/kWh de MAE   |   Efecto El Nino: {ef_oni:+.1f} COP/kWh")
    tabla.to_csv(RES / "estratificacion_regimen_24h.csv")
    print(f"\nLISTO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
