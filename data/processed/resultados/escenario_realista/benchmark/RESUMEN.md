# Benchmark y walk-forward del escenario realista (19 y 20-oct-2026)

Pedido de Juan: justificar con el protocolo completo (benchmark, walk-forward historico, Diebold-Mariano) la eleccion de
CatBoost y del diseño "predespacho + correccion". Todo con solo la informacion publica al corte (00:00 de D).

Periodos: cinco historicos (15-jul a 30-jun, 2020-21 a 2024-25, ~8.424 h cada uno) y 2026.

## 1. Correctores del predespacho (realista_benchmark_modelos.py)

Mismas condiciones para todos: objetivo log(precio/predespacho), variables R+P, reentrenamiento mensual con lo
publicado, 1.500 arboles / profundidad 6 / tasa 0,05, perdida absoluta cuando el modelo la admite, sin afinar.
Combinador QRA (MAE, franjas de 6 h, margen 5 dias) que arranca en cada periodo con 60 dias de calentamiento.
2026 = 15-ene a 15-oct (6.575 h).

| Modelo | O1 | O2 | O3 | O4 | O5 | 2026 | media |
|---|---|---|---|---|---|---|---|
| Persistencia (D-4) | 26,6 | 37,0 | 87,5 | 137,6 | 161,4 | 117,9 | 94,7 |
| Predespacho solo | 13,6 | 12,5 | 23,5 | 37,0 | 59,5 | 42,0 | 31,4 |
| CatBoost solo | 11,9 | 11,2 | 24,7 | 45,7 | 61,9 | 40,1 | 32,6 |
| LightGBM solo | 12,0 | 11,4 | 23,6 | 43,7 | 64,7 | 40,6 | 32,7 |
| XGBoost solo | 14,0 | 12,6 | 24,2 | 43,5 | 71,4 | 41,6 | 34,6 |
| RandomForest solo | 12,1 | 11,5 | 22,8 | 48,8 | 64,0 | 44,1 | 33,9 |
| Lasso (tipo LEAR) solo | 12,5 | 11,7 | 24,2 | 53,6 | 74,5 | 44,3 | 36,8 |
| **Pre + CatBoost** | **9,33** | 12,01 | 21,19 | 36,34 | **56,70** | 38,77 | **29,05** |
| Pre + LightGBM | 9,34 | 11,92 | 21,19 | 36,33 | 56,92 | 38,83 | 29,09 |
| Pre + XGBoost | 9,35 | 12,57 | 21,23 | 36,33 | 56,91 | 39,00 | 29,23 |
| Pre + RandomForest | 9,38 | 11,63 | 21,20 | 36,34 | 56,85 | 38,98 | 29,06 |
| Pre + Lasso | 9,45 | 12,05 | 21,26 | 36,40 | 56,87 | 39,40 | 29,24 |
| Pre + promedio de 3 boosting | 9,30 | 12,20 | 21,16 | 36,31 | 56,85 | 38,67 | 29,08 |

DM frente a Pre + CatBoost (t > 1,96 = significativamente peor; < -1,96 = mejor):
- Pre + LightGBM: empate en los 6 periodos (|t| <= 1,5).
- Pre + XGBoost: empate salvo O2 (peor, t = 2,9).
- Pre + RandomForest: empate salvo O2 (mejor, t = -2,9).
- Pre + Lasso: peor en O1 (2,5), O4 (2,1) y 2026 (4,4).
- Predespacho solo: peor en 5 de 6 (empate en O2, t = 1,0).

Lectura: lo que importa es corregir el predespacho con un modelo de arboles no lineal; entre CatBoost, LightGBM,
XGBoost y RandomForest no hay diferencia significativa consistente. CatBoost tiene la mejor media combinada (29,05) y la
mejor media solo (32,58, empatado con LightGBM 32,65), y solo es superado con significancia en un periodo (O2, por
RandomForest). El lineal (Lasso) es significativamente peor en 3 de 6: la no linealidad aporta.

## 2. Modelos que pronostican desde cero (realista_walkforward_individuales.py)

Mismas funciones de escenario_realista_24h.py, modelo fijo entrenado con todo lo anterior al inicio de cada periodo.
2026 = 15-ene a 5-ago (las redes y el ARX usan el maestro oficial, que llega hasta ahi).

| Modelo | O1 | O2 | O3 | O4 | O5 | 2026 | media |
|---|---|---|---|---|---|---|---|
| Pre + CatBoost (diseño) | 9,3 | 12,0 | 21,2 | 36,3 | 56,7 | 35,2 | 28,5 |
| Predespacho solo | 13,6 | 12,5 | 23,5 | 37,0 | 59,5 | 39,2 | 30,9 |
| GARCH-ged (LEAR) | 30,4 | 35,9 | 81,3 | 139,3 | 148,6 | 100,6 | 89,4 |
| Persistencia | 26,6 | 37,0 | 87,5 | 137,6 | 161,4 | 108,8 | 93,1 |
| N-BEATSx | 95,2 | 40,6 | 88,9 | 148,1 | 184,2 | 101,7 | 109,8 |
| XGBoost | 28,1 | 40,2 | 99,1 | 208,2 | 183,0 | 102,7 | 110,2 |
| N-HiTS | 125,3 | 38,0 | 85,2 | 151,5 | 169,0 | 115,7 | 114,1 |
| ARX+GARCH | 221,5 | 37,0 | 104,6 | 147,9 | 157,8 | 105,0 | 129,0 |

DM frente a la persistencia: GARCH-ged la mejora en 3 de 6 (O3, O5, 2026) y pierde en O1; ARX+GARCH, XGBoost,
N-BEATSx y N-HiTS nunca la mejoran con significancia y pierden en 2 a 4 periodos. Todos los "Pre + corrector" la
mejoran en los 6 (t entre -9 y -19).

O1 (2020-21): ARX+GARCH y las redes se degradan mes a mes (ARX: 42 en jul-2020 -> 535 en feb-2021). No es un error
de codigo (XGBoost y GARCH-ged, mismo flujo, dan valores normales): se entrenan con solo 17 meses (feb-2019 a
jul-2020, con los precios altos de inicios de 2020) y quedan fijos todo el año, extrapolando un nivel de ~385 COP/kWh
frente a ~164 real. Mismo patron, menos marcado, que en el walk-forward optimista (ARX+GARCH 50,5 vs persistencia 16,8).

Conclusion: sin el precio de ayer, ningun modelo que pronostica desde cero supera de forma consistente a la
persistencia; el predespacho solo ya los supera a todos por un factor de ~3, y corregirlo con arboles lo mejora en
5 de 6 periodos. Esto justifica el cambio de diseño respecto al ensamble de 7 (escenario optimista).

## 3. Detector de picos y desplomes por horizonte (eventos_72h/, realista_eventos_72h.py)

BSS frente a la climatologia por hora (* = Brier significativamente mejor, DM HAC p < 0,05; ! = significativamente peor):

| Evento / pasos | O1 | O2 | O3 | O4 | O5 | 2026 |
|---|---|---|---|---|---|---|
| desplome 1-24 | +0,46* | +0,57* | +0,42* | +0,33* | +0,31* | +0,34* |
| desplome 25-48 | -0,08 | +0,31* | +0,07 | -0,01 | +0,07 | +0,13* |
| desplome 49-72 | -0,30! | +0,25* | -0,04 | -0,03 | +0,05 | +0,11* |
| pico 1-24 | +0,23* | +0,16* | +0,29* | +0,37* | +0,41* | +0,55* |
| pico 25-48 | -0,01 | -0,09 | +0,01 | +0,19* | +0,19* | +0,33* |
| pico 49-72 | -0,02 | +0,01 | -0,04 | +0,10* | +0,18* | +0,25* |

- El detector de 24 h (pasos 1-24) queda validado historicamente: mejora significativa en los 6 periodos y ambos eventos.
- Picos a 25-72 h: significativo en los tres periodos mas recientes, empate (nunca peor) en 2020-23.
- Desplomes a 25-72 h: inconsistente, y peor que la climatologia en O1 a 49-72 h.
- La regla "predespacho de D (0/1)" es peor que la climatologia a 25-72 h en todos los periodos.

## 4. Reentrenamiento por el cambio de reglas (regimen_2026/)

MAE 15-ene a 15-oct (6.576 h): base 38,43; curva50 38,17 (DM t = -2,65); curva82 38,34; rec365 38,70; rec120 38,93
(t = +2,98, peor). Desde el 9-sep: base 65,35; curva50 64,76 (t = -2,08). Ponderar por recencia no ayuda ni despues
del cambio de reglas. curva50 es el unico candidato.

Validacion historica de curva50 (realista_validacion_historica_curva.py, 20-oct, mismo protocolo que la v2):

| Periodo | base | curva50 | DM t (curva50 vs base) |
|---|---|---|---|
| O1 2020-21 | 9,23 | 9,19 | -2,04 (mejor, p = 0,04) |
| O2 2021-22 | 11,60 | 11,61 | 0,30 (empate) |
| O3 2022-23 | 21,11 | 20,97 | -2,28 (mejor, p = 0,02) |
| O4 2023-24 | 36,21 | 36,28 | 1,34 (empate) |
| O5 2024-25 | 56,58 | 57,02 | 2,28 (peor, p = 0,02) |

Decision: NO se adopta. Gana en 3 de 6 periodos (con 2026), empata en 2 y pierde con significancia en 2024-25; la
media historica queda igual (27,01 frente a 26,95). El cambio de reglas de la CREG 101 127 no se puede anticipar de
forma robusta con esta aproximacion de la curva (embalse agregado, X % supuesto, CERE como OCV).
