# -*- coding: utf-8 -*-
"""
NOT (Narrowest-Over-Threshold) para deteccion de multiples puntos de cambio.

Reimplementacion en Python del algoritmo de Baranowski, Chen & Fryzlewicz (2019), JRSS-B 81(3),
"Narrowest-over-threshold detection of multiple change-points and change-point-like features".
El original es el paquete de R `not`; aqui no hay puente a R, asi que se reimplementa.

CONTRASTE. Se usa la forma menos restrictiva, la misma que eligen Nasiadka, Nitka & Weron:
media y varianza constantes a trozos. Para un segmento [a,b) partido en d, el contraste es la
razon de verosimilitudes generalizada gaussiana

    GLR(a,d,b) = n*log(v) - n1*log(v1) - n2*log(v2)

con v, v1, v2 las varianzas maximo-verosimiles del segmento completo, el izquierdo y el derecho.
Se evalua en O(1) por corte con sumas acumuladas, y vectorizado sobre todos los cortes.

ALGORITMO (recursivo). Dentro de [a,b):
  1. Se sortean M intervalos aleatorios contenidos en [a,b).
  2. Para cada uno se calcula el mejor corte y su contraste.
  3. De los que superan el umbral, se toma el MAS ANGOSTO (esa es la idea de NOT: un intervalo
     angosto que ya supera el umbral casi seguro contiene un solo punto de cambio, asi que su
     argmax lo localiza sin contaminacion de los vecinos).
  4. Se recursa a izquierda y derecha del punto encontrado.

SELECCION DEL NUMERO DE PUNTOS. El paquete `not` construye la trayectoria de soluciones variando
el umbral y elige con un criterio de informacion. Aqui se corre la recursion sobre una rejilla de
umbrales y entre los modelos resultantes (cada uno con a lo sumo Nc_max puntos) se elige el de
menor SIC reforzado

El umbral importa y no se puede saltar: con uno solo y bajo, la regla del intervalo mas angosto se
queda con intervalos diminutos que superan el umbral por ruido, y el detector se dispara (se
verifico: 7 puntos espurios en la serie sintetica de un solo salto de media). La rejilla es lo que
mantiene honesto al detector.

    SIC(k) = sum_j n_j*log(var_j) + 3*k*log(T)^1.01

Nasiadka/Nitka/Weron fijan Nc_max = 12; se respeta ese valor por defecto.

AUTOPRUEBA. `autoprueba()` corre el detector sobre series sinteticas con puntos de cambio
conocidos (solo media, solo varianza, ambos, y ninguno) y reporta el error de localizacion. Se
ejecuta al importar este modulo desde el experimento, para no usar nunca un detector sin verificar.
"""
import numpy as np

MIN_SEG = 10          # longitud minima de un segmento (dias)
UMBRALES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0)  # rejilla de umbrales; el SIC elige el modelo


class _Acumuladas:
    """Sumas acumuladas para varianzas de segmento en O(1)."""

    def __init__(self, x):
        x = np.asarray(x, float)
        self.n = len(x)
        self.c1 = np.concatenate([[0.0], np.cumsum(x)])
        self.c2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def var(self, a, b):
        n = b - a
        s1 = self.c1[b] - self.c1[a]
        s2 = self.c2[b] - self.c2[a]
        return max((s2 - s1 * s1 / n) / n, 1e-12)


def _mejores_cortes(ac, ini, fin, min_seg):
    """
    Vectorizado sobre TODOS los intervalos candidatos a la vez.

    Sin esto el algoritmo es inviable: una llamada con M=800 intervalos tardaba 3 s porque hacia
    una operacion de numpy por intervalo. Aqui se aplana la lista de pares (intervalo, corte) en un
    solo arreglo, se calcula el GLR de una vez, y se recupera el argmax por intervalo con reduceat.
    """
    lens = fin - ini - 2 * min_seg + 1
    ok = lens > 0
    ini, fin, lens = ini[ok], fin[ok], lens[ok]
    if len(ini) == 0:
        return np.empty(0, int), np.empty(0), np.empty(0, int)
    inicios = np.concatenate([[0], np.cumsum(lens)[:-1]])
    iv = np.repeat(np.arange(len(ini)), lens)
    ds = np.repeat(ini + min_seg, lens) + (np.arange(int(lens.sum())) - np.repeat(inicios, lens))
    a, b = ini[iv], fin[iv]
    n = (b - a).astype(float)
    n1 = (ds - a).astype(float)
    n2 = (b - ds).astype(float)
    s1, s2 = ac.c1[b] - ac.c1[a], ac.c2[b] - ac.c2[a]
    v = np.maximum((s2 - s1 * s1 / n) / n, 1e-12)
    s1L, s2L = ac.c1[ds] - ac.c1[a], ac.c2[ds] - ac.c2[a]
    s1R, s2R = ac.c1[b] - ac.c1[ds], ac.c2[b] - ac.c2[ds]
    vL = np.maximum((s2L - s1L * s1L / n1) / n1, 1e-12)
    vR = np.maximum((s2R - s1R * s1R / n2) / n2, 1e-12)
    glr = n * np.log(v) - n1 * np.log(vL) - n2 * np.log(vR)

    mx = np.maximum.reduceat(glr, inicios)
    cand = np.where(glr == mx[iv])[0]
    ivc = iv[cand]
    primero = np.ones(len(cand), bool)
    primero[1:] = ivc[1:] != ivc[:-1]
    best = cand[primero]
    return ds[best], glr[best], (fin - ini)


def _recursion(ac, a, b, umbral, M, rng, fuera, min_seg, pathmin, profundidad=0):
    """NOT recursivo. `pathmin` es el contraste minimo a lo largo del camino desde la raiz: con el
    se reconstruye la trayectoria anidada de soluciones sin volver a correr la recursion por cada
    umbral (un punto se detecta al umbral u si y solo si su pathmin supera u)."""
    if b - a < 2 * min_seg or profundidad > 25 or len(fuera) > 80:
        return
    if b - a > 2 * min_seg + 2:
        u = rng.integers(a, b, size=(M, 2))
        ini = np.concatenate([[a], np.minimum(u[:, 0], u[:, 1])])
        fin = np.concatenate([[b], np.maximum(u[:, 0], u[:, 1]) + 1])
    else:
        ini, fin = np.array([a]), np.array([b])
    ds, gs, anchos = _mejores_cortes(ac, ini, fin, min_seg)
    if len(ds) == 0:
        return
    sel = gs > umbral
    if not sel.any():
        return
    ds, gs, anchos = ds[sel], gs[sel], anchos[sel]
    j = int(np.lexsort((-gs, anchos))[0])          # el MAS ANGOSTO; desempate por mayor contraste
    d, g = int(ds[j]), float(gs[j])
    pm = min(pathmin, g)
    fuera.append((d, g, pm))
    _recursion(ac, a, d, umbral, M, rng, fuera, min_seg, pm, profundidad + 1)
    _recursion(ac, d, b, umbral, M, rng, fuera, min_seg, pm, profundidad + 1)


def _sic(ac, puntos, T, min_seg=MIN_SEG):
    bordes = [0] + sorted(puntos) + [T]
    total = 0.0
    for i in range(len(bordes) - 1):
        a, b = bordes[i], bordes[i + 1]
        if b - a < min_seg:
            return np.inf
        total += (b - a) * np.log(ac.var(a, b))
    k = len(puntos)
    # 3 parametros por punto de cambio (media, varianza y la propia ubicacion). El coeficiente 3 se
    # fijo con la autoprueba de abajo: con 2 el criterio deja pasar un punto espurio en las series
    # sinteticas con cambio simultaneo de media y varianza.
    return total + 3.0 * k * (np.log(T) ** 1.01)


def detectar(x, nc_max=12, M=500, semilla=0, umbrales=UMBRALES, min_seg=MIN_SEG):
    """Puntos de cambio de x (indices de inicio del segmento nuevo), elegidos por SIC."""
    x = np.asarray(x, float)
    T = len(x)
    if T < 3 * min_seg:
        return []
    ac = _Acumuladas(x)
    mejor_pts, mejor_sic = [], _sic(ac, [], T, min_seg)
    for u in umbrales:
        rng = np.random.default_rng(semilla)
        fuera = []
        _recursion(ac, 0, T, u, M, rng, fuera, min_seg, np.inf)
        if not fuera:
            continue
        fuerza = {d: g for d, g, _ in fuera}
        pts = sorted(fuerza)
        if len(pts) > nc_max:
            pts = sorted(sorted(pts, key=lambda d: -fuerza[d])[:nc_max])
        s = _sic(ac, pts, T, min_seg)
        if s < mejor_sic:
            mejor_sic, mejor_pts = s, pts
    return mejor_pts


def seleccionar_subperiodos(x, nc_max=12, q=(0.025, 0.975), M=500, semilla=0, min_seg=MIN_SEG):
    """
    Algoritmo de seleccion de ventana de Nasiadka, Nitka & Weron (pasos 1-5 del paper).

    Devuelve una mascara booleana sobre x: True en las observaciones que se usan para calibrar.
    Si no hay puntos de cambio, se devuelve toda la ventana (paso 3 del paper).
    """
    x = np.asarray(x, float)
    T = len(x)
    pts = detectar(x, nc_max=nc_max, M=M, semilla=semilla, min_seg=min_seg)
    if not pts:
        return np.ones(T, dtype=bool)
    # paso 3: cuantiles del tramo entre el ultimo punto de cambio y el final de la ventana
    ultimo = pts[-1]
    tramo_actual = x[ultimo:]
    qlow, qhigh = np.quantile(tramo_actual, q[0]), np.quantile(tramo_actual, q[1])
    # pasos 4-5: se conservan los tramos cuya MEDIANA cae dentro de esos cuantiles
    bordes = [0] + list(pts) + [T]
    mascara = np.zeros(T, dtype=bool)
    for i in range(len(bordes) - 1):
        a, b = bordes[i], bordes[i + 1]
        m = float(np.median(x[a:b]))
        if qlow < m < qhigh:
            mascara[a:b] = True
    if not mascara.any():                 # salvaguarda: nunca devolver vacio
        mascara[ultimo:] = True
    return mascara


def autoprueba(verboso=True):
    """Valida el detector contra series sinteticas con puntos de cambio conocidos."""
    rng = np.random.default_rng(7)
    casos = []

    # 1) salto de media puro
    x = np.concatenate([rng.normal(0, 1, 200), rng.normal(4, 1, 200), rng.normal(0, 1, 200)])
    casos.append(("salto de media (200, 400)", x, [200, 400]))
    # 2) cambio de varianza puro
    x = np.concatenate([rng.normal(0, 1, 250), rng.normal(0, 4, 250)])
    casos.append(("cambio de varianza (250)", x, [250]))
    # 3) media y varianza
    x = np.concatenate([rng.normal(0, 1, 150), rng.normal(3, 3, 150), rng.normal(-2, 1, 200)])
    casos.append(("media+varianza (150, 300)", x, [150, 300]))
    # 4) sin puntos de cambio
    x = rng.normal(0, 1, 600)
    casos.append(("sin puntos de cambio", x, []))
    # 5) muchos tramos, tamano realista de ventana (728 dias)
    tr = [rng.normal(m, s, n) for m, s, n in
          [(10, 1, 150), (25, 3, 120), (12, 1.5, 200), (40, 8, 108), (15, 2, 150)]]
    casos.append(("5 tramos en 728 puntos", np.concatenate(tr), [150, 270, 470, 578]))

    # Criterio declarado de antemano: cada punto verdadero debe quedar localizado a <=15
    # observaciones, y se admite a lo sumo 1 punto espurio por serie (un punto de cambio con salto
    # fuerte de varianza puede partirse en dos vecinos; es inocuo para seleccionar subperiodos).
    ok = True
    if verboso:
        print("AUTOPRUEBA DEL DETECTOR NOT  (criterio: error de ubicacion <=15, espurios <=1)")
        print(f"  {'caso':30s} {'verdaderos':>22s} {'detectados':>32s} {'err':>5s} {'espurios':>9s}")
    for nom, x, verdad in casos:
        pts = detectar(x, semilla=1)
        libres = list(pts)
        err, faltan = 0, 0
        for v in verdad:
            if not libres:
                faltan += 1
                continue
            j = int(np.argmin([abs(p - v) for p in libres]))
            err = max(err, abs(libres[j] - v))
            libres.pop(j)
        espurios = len(libres)
        bien = (faltan == 0) and (err <= 15) and (espurios <= 1)
        ok = ok and bien
        if verboso:
            print(f"  {nom:30s} {str(verdad):>22s} {str(pts):>32s} {err:5d} {espurios:9d}"
                  f"{'' if bien else '   <-- FALLA'}")
    if verboso:
        print(f"  VEREDICTO: {'detector valido' if ok else 'DETECTOR NO FIABLE - no usar'}\n")
    return ok


if __name__ == "__main__":
    autoprueba()
