# -*- coding: utf-8 -*-
"""
Motor de decision (OE3): traduce un pronostico con bandas de incertidumbre
[fecha_hora, real, q10, q50, q90] en una senal de comprar/vender/esperar,
distinta segun el rol del usuario (generador vs. comercializador).

Contrato de entrada esperado (mismo que produce la sesion del 2026-09-10 para
N-BEATSx adaptativo, y el que se genera en 12_motor_decision_Rafa.ipynb para
XGBoost a 72h):
    columnas: fecha_hora, real, q50, q10, q90

El motor en si (generar_senales, evaluar_backtest, comparar_metodos) NO sabe
ni le importa que modelo produjo esas columnas -- funciona igual para 24h o
72h, con N-BEATSx, XGBoost o cualquier modelo futuro, siempre que el CSV
cumpla el contrato de columnas.

Que archivo/modelo alimenta cada horizonte se decide en UN SOLO lugar:
data/processed/resultados/fuentes_pronostico.json, leido por
cargar_fuente_pronostico() mas abajo. Cambiar de modelo (ej. cuando exista
un pipeline de N-BEATSx a 72h) es editar ese JSON, no tocar codigo -- ver
docs/motor_decision_guia.md, seccion "Cambiar o agregar un modelo".
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ("generador", "comercializador")
METODOS = ("fijo", "rodante", "banda", "hibrido")
COLUMNAS_CONTRATO = ["fecha_hora", "real", "q10", "q50", "q90"]
RUTA_CONFIG_FUENTES = Path("data") / "processed" / "resultados" / "fuentes_pronostico.json"


def validar_contrato_pronostico(df, nombre_archivo=""):
    """Verifica que un DataFrame de pronostico cumpla el contrato de columnas
    del motor de decision. Uso previsto: correrlo sobre cualquier CSV nuevo
    ANTES de registrarlo en fuentes_pronostico.json (ver guia).

    Las columnas y los NaN son errores duros (el motor no puede funcionar sin
    eso). Los cuantiles cruzados (q10>q50 o q50>q90) son solo advertencia: en
    la practica aparecen como un puñado de filas borde de calibraciones
    adaptativas (ej. 2/5208 filas en las bandas de 24h del 2026-09-10) y no
    vale la pena bloquear todo el pipeline por eso -- pero se reportan para
    que quien registre un modelo nuevo los revise."""
    faltantes = [c for c in COLUMNAS_CONTRATO if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"{nombre_archivo or 'el DataFrame'} no cumple el contrato del motor de decisión, "
            f"faltan columnas: {faltantes}. Se requieren: {COLUMNAS_CONTRATO}"
        )
    if df[["q10", "q50", "q90"]].isna().any().any():
        raise ValueError(f"{nombre_archivo or 'el DataFrame'} tiene NaN en q10/q50/q90 -- revisar antes de usarlo")

    n_cruzados = int((~(df["q10"] <= df["q50"])).sum() + (~(df["q50"] <= df["q90"])).sum())
    if n_cruzados:
        print(
            f"[validar_contrato_pronostico] advertencia: {nombre_archivo or 'el DataFrame'} tiene "
            f"{n_cruzados} fila(s) con cuantiles cruzados (q10>q50 o q50>q90) de {len(df)} totales -- "
            "revisar si crece en un modelo nuevo, probablemente borde de calibración."
        )
    return True


def cargar_config_fuentes(raiz):
    """Lee data/processed/resultados/fuentes_pronostico.json -- el registro de
    que modelo/archivo alimenta cada horizonte. Ver docs/motor_decision_guia.md."""
    path = Path(raiz) / RUTA_CONFIG_FUENTES
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Este archivo es el registro de fuentes del motor de decisión -- "
            "sin él no hay forma de saber qué modelo alimenta cada horizonte."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cargar_fuente_pronostico(raiz, horizonte):
    """Carga el pronóstico con bandas ACTIVO para un horizonte, según
    fuentes_pronostico.json. Este es el único punto de entrada que
    notebooks/12_motor_decision_Rafa.ipynb y dashboard/app.py deberían usar
    para leer datos de pronóstico -- así, cambiar el modelo de un horizonte
    (ej. de XGBoost a N-BEATSx en 72h cuando exista ese pipeline) es editar
    el JSON, no tocar el notebook ni el dashboard.

    Devuelve (df, metadata) donde df tiene las columnas de COLUMNAS_CONTRATO
    y metadata es la entrada cruda del JSON (modelo, calibrado, etc.)."""
    config = cargar_config_fuentes(raiz)
    if horizonte not in config:
        raise KeyError(
            f"No hay fuente configurada para horizonte='{horizonte}' en {RUTA_CONFIG_FUENTES}. "
            f"Horizontes disponibles: {list(config.keys())}"
        )
    metadata = config[horizonte]
    path_csv = Path(raiz) / "data" / "processed" / "resultados" / metadata["archivo"]
    df = pd.read_csv(path_csv, parse_dates=["fecha_hora"])
    validar_contrato_pronostico(df, nombre_archivo=str(path_csv))
    return df[COLUMNAS_CONTRATO], metadata


def umbrales_fijos(precio_historico, p_bajo=25, p_alto=75):
    """Percentiles calculados una sola vez sobre una serie historica (ej. train 2019-2025)."""
    precio_historico = np.asarray(precio_historico, dtype=float)
    return {
        "bajo": float(np.nanpercentile(precio_historico, p_bajo)),
        "alto": float(np.nanpercentile(precio_historico, p_alto)),
    }


def umbrales_rodantes(serie, ventana_dias=30, p_bajo=25, p_alto=75):
    """Percentiles con ventana movil CAUSAL: para la hora t solo usa las
    `ventana_dias*24` horas anteriores a t (shift(1) antes de la ventana, para
    no incluir el propio valor de t -- evita fuga de informacion futura)."""
    horas = ventana_dias * 24
    base = serie.shift(1)
    bajo = base.rolling(horas, min_periods=horas).quantile(p_bajo / 100)
    alto = base.rolling(horas, min_periods=horas).quantile(p_alto / 100)
    return bajo, alto


def generar_senales(df, metodo, rol, precio_historico_train=None,
                     p_bajo=25, p_alto=75, ventana_dias=30, percentil_ancho_filtro=75):
    """
    df: DataFrame con columnas fecha_hora, real, q10, q50, q90 (una fila por hora).
    metodo: "fijo" | "rodante" | "banda" | "hibrido".
        - "fijo": percentiles del historico de entrenamiento (2019-2025).
        - "rodante": percentiles con ventana movil causal de `ventana_dias`.
        - "banda": fijo + "esperar" forzado cuando el ancho de la banda
          [q10,q90] esta en el percentil alto (poca confianza).
        - "hibrido": igual que "banda" pero con el umbral RODANTE en vez del
          fijo -- umbral que se adapta al regimen reciente Y filtro de
          confianza por ancho de banda, combinados. Agregado 2026-09-24 tras
          encontrar que "banda" (fijo) pierde toda capacidad de discriminar
          cuando el precio del periodo evaluado se aleja mucho del historico
          2019-2025 (ver bitacora del README) -- no tiene todavia respaldo de
          literatura especifica, es una combinacion de piezas ya validadas
          por separado (ver seccion de investigacion pendiente en el README).
    rol: "generador" | "comercializador".
    precio_historico_train: requerido para metodo "fijo"/"banda" -- serie de
        precio_bolsa historico (ej. 2019-2025) sobre la que se calculan los
        percentiles fijos.

    Devuelve una Serie de texto con la senal, alineada al indice de df.
    """
    if rol not in ROLES:
        raise ValueError(f"rol debe ser uno de {ROLES}, recibido: {rol}")
    if metodo not in METODOS:
        raise ValueError(f"metodo debe ser uno de {METODOS}, recibido: {metodo}")

    precio = df["q50"]

    if metodo in ("rodante", "hibrido"):
        bajo, alto = umbrales_rodantes(precio, ventana_dias, p_bajo, p_alto)
    else:  # "fijo" y "banda" comparten el mismo umbral fijo sobre el historico
        if precio_historico_train is None:
            raise ValueError(f"metodo='{metodo}' requiere precio_historico_train")
        u = umbrales_fijos(precio_historico_train, p_bajo, p_alto)
        bajo, alto = u["bajo"], u["alto"]

    barato = precio <= bajo
    caro = precio >= alto

    if rol == "comercializador":
        # comprador: precio bajo = oportunidad de comprar; precio alto = evitar bolsa (usar contratos)
        senal = np.select([caro, barato], ["evitar_compra", "comprar"], default="esperar")
    else:
        # generador: precio alto = oportunidad de vender; precio bajo = retener (no despachar si se puede evitar)
        senal = np.select([caro, barato], ["vender", "retener"], default="esperar")

    senal = pd.Series(senal, index=df.index)

    if metodo in ("banda", "hibrido"):
        ancho = df["q90"] - df["q10"]
        umbral_ancho = np.nanpercentile(ancho, percentil_ancho_filtro)
        senal = senal.mask(ancho > umbral_ancho, "esperar")

    return senal


def evaluar_backtest(df, senal, rol):
    """Backtest economico simple: compara el precio real promedio en las horas
    donde el motor dice actuar (comprar/vender) contra el precio real promedio
    general. Positivo = la senal identifica horas mejores que el promedio.

    No es una simulacion de portafolio -- es la comprobacion minima de que la
    regla realmente encuentra horas mejores que el azar antes de confiar en ella.
    """
    accion = "comprar" if rol == "comercializador" else "vender"
    mask = senal == accion
    precio_medio_general = df["real"].mean()

    if not mask.any():
        return {"ventaja_cop_kwh": np.nan, "frecuencia_accion": 0.0, "horas_accion": 0}

    precio_medio_accion = df.loc[mask, "real"].mean()
    if rol == "comercializador":
        ventaja = precio_medio_general - precio_medio_accion  # compra mas barato que el promedio
    else:
        ventaja = precio_medio_accion - precio_medio_general  # vende mas caro que el promedio

    return {
        "ventaja_cop_kwh": float(ventaja),
        "frecuencia_accion": float(mask.mean()),
        "horas_accion": int(mask.sum()),
    }


def comparar_metodos(df, rol, precio_historico_train, p_bajo=25, p_alto=75,
                      ventana_dias=30, percentil_ancho_filtro=75,
                      frecuencia_min=0.10, frecuencia_max=0.40, mascara_evaluacion=None):
    """Corre los 4 metodos para un rol dado y devuelve una tabla comparativa,
    marcando como 'valido' solo los metodos cuya frecuencia de accion cae en
    un rango razonable (por defecto 10%-40% de las horas) -- una regla que
    casi nunca actua o que actua case siempre no es una regla util, aunque su
    ventaja promedio se vea bien en las pocas veces que dispara.

    mascara_evaluacion: booleano opcional alineado al indice de `df`. Si se
        da, la senal se calcula sobre TODO `df` (necesario para que "rodante"/
        "hibrido" tengan continuidad en su ventana movil) pero el backtest se
        mide solo en las filas donde la mascara es True -- permite comparar
        metodos en un rango de fechas mas chico sin romper el calculo de los
        umbrales que dependen de continuidad temporal. Si no se da, se mide
        sobre todo `df` (comportamiento de siempre)."""
    filas = []
    for metodo in METODOS:
        senal = generar_senales(df, metodo, rol, precio_historico_train,
                                 p_bajo, p_alto, ventana_dias, percentil_ancho_filtro)
        if mascara_evaluacion is None:
            resultado = evaluar_backtest(df, senal, rol)
        else:
            resultado = evaluar_backtest(df[mascara_evaluacion], senal[mascara_evaluacion], rol)
        resultado["metodo"] = metodo
        resultado["valido"] = frecuencia_min <= resultado["frecuencia_accion"] <= frecuencia_max
        filas.append(resultado)

    tabla = pd.DataFrame(filas)[["metodo", "ventaja_cop_kwh", "frecuencia_accion", "horas_accion", "valido"]]
    return tabla.sort_values("ventaja_cop_kwh", ascending=False).reset_index(drop=True)


def elegir_mejor_metodo(tabla_comparativa):
    """Dada la tabla de comparar_metodos(), elige el metodo con mayor ventaja
    economica ENTRE los marcados como validos. Si ninguno es valido, elige el
    de mayor ventaja de todas formas pero lo deja claro en el resultado."""
    validos = tabla_comparativa[tabla_comparativa["valido"]]
    if len(validos) > 0:
        ganador = validos.iloc[0]
        return ganador["metodo"], True
    ganador = tabla_comparativa.iloc[0]
    return ganador["metodo"], False


def comparar_metodos_estable(df, rol, precio_historico_train, corte=None, p_bajo=25, p_alto=75,
                              ventana_dias=30, percentil_ancho_filtro=75,
                              frecuencia_min=0.10, frecuencia_max=0.40):
    """Alternativa a comparar_metodos() que NO rankea por la ventaja promedio
    del periodo completo, sino por un criterio maximin: el desempeño en la
    MITAD MAS DEBIL del periodo. Se agrego el 2026-09-24 porque el criterio
    de comparar_metodos() eligio "banda" como ganador (mejor promedio) sin
    detectar que su frecuencia de accion se desplomaba a ~0% en la mitad mas
    barata del año -- un promedio bueno sostenido casi todo por una sola
    mitad, no un desempeño parejo. Es un criterio de decision razonable
    (teoria de decision minimax/maximin, Wald 1950) pero la implementacion
    exacta (dos mitades fijas, no una ventana movil ni un bandit no
    estacionario) es una eleccion de ingenieria propia, sin una prueba de
    sensibilidad ni una cita externa especifica todavia -- ver la nota de
    investigacion pendiente en el README antes de confiar en esto a ciegas.

    corte: fecha que divide el periodo en dos mitades [inicio, corte) y
        [corte, fin]. Si no se da, se usa el punto medio del rango de `df`.

    Devuelve una tabla con, por metodo: ventaja/frecuencia en cada mitad, la
    ventaja de la peor mitad ("peor_mitad", el criterio de ranking) y si el
    metodo es "estable" (frecuencia valida Y ventaja no-negativa en las DOS
    mitades)."""
    if corte is None:
        corte = df["fecha_hora"].min() + (df["fecha_hora"].max() - df["fecha_hora"].min()) / 2
    mitad_1 = df["fecha_hora"] < corte
    mitad_2 = ~mitad_1

    def valido(freq):
        return not np.isnan(freq) and frecuencia_min <= freq <= frecuencia_max

    filas = []
    for metodo in METODOS:
        senal = generar_senales(df, metodo, rol, precio_historico_train,
                                 p_bajo, p_alto, ventana_dias, percentil_ancho_filtro)
        df_con_senal = df.assign(senal=senal)
        h1 = evaluar_backtest(df_con_senal[mitad_1], df_con_senal.loc[mitad_1, "senal"], rol)
        h2 = evaluar_backtest(df_con_senal[mitad_2], df_con_senal.loc[mitad_2, "senal"], rol)

        ambas_conocidas = not (np.isnan(h1["ventaja_cop_kwh"]) or np.isnan(h2["ventaja_cop_kwh"]))
        peor_mitad = min(h1["ventaja_cop_kwh"], h2["ventaja_cop_kwh"]) if ambas_conocidas else np.nan
        estable = (
            ambas_conocidas
            and h1["ventaja_cop_kwh"] >= 0 and h2["ventaja_cop_kwh"] >= 0
            and valido(h1["frecuencia_accion"]) and valido(h2["frecuencia_accion"])
        )
        filas.append({
            "metodo": metodo,
            "ventaja_H1": h1["ventaja_cop_kwh"], "frecuencia_H1": h1["frecuencia_accion"],
            "ventaja_H2": h2["ventaja_cop_kwh"], "frecuencia_H2": h2["frecuencia_accion"],
            "peor_mitad": peor_mitad,
            "estable": estable,
        })

    tabla = pd.DataFrame(filas)
    return tabla.sort_values("peor_mitad", ascending=False, na_position="last").reset_index(drop=True)


def elegir_mejor_metodo_estable(tabla_comparativa_estable):
    """Como elegir_mejor_metodo(), pero sobre comparar_metodos_estable():
    elige el metodo con mejor 'peor_mitad' ENTRE los marcados como estables.
    Si ninguno es estable, elige el de mejor 'peor_mitad' de todas formas
    pero lo deja explicito (segundo valor = False)."""
    estables = tabla_comparativa_estable[tabla_comparativa_estable["estable"]]
    if len(estables) > 0:
        ganador = estables.iloc[0]
        return ganador["metodo"], True
    ganador = tabla_comparativa_estable.iloc[0]
    return ganador["metodo"], False
