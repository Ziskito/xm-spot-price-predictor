# Mejoras del escenario realista (7-oct-2026)

Escenario realista: a las 00:00 de D solo se usa lo publicado por XM a esa hora (precio, demanda y generacion hasta
las 23:00 de D-4; embalses y aportes hasta D-2; ONI con 2 meses; predespacho ideal y despacho programado de D,
publicados el dia anterior). Mismas 4.849 horas (15-ene a 5-ago-2026). Referencia: ensamble 7 realista, MAE 36,47.

## Lo que funciona

| Variante | MAE | sesgo | DM t vs ens7 realista |
|---|---|---|---|
| Ensamble 7 realista (referencia) | 36,47 | -3,4 | - |
| Combinador con objetivo MAE (mismos 7 votantes) | 35,76 | -0,1 | -4,3 |
| CatBoost con el predespacho por dentro, reentrenado cada mes (solo) | 35,94 | -1,4 | -1,2 |
| **Predespacho + CatBoost mensual, combinador MAE, franjas 6 h** | **35,02** | 0,2 | -5,6 |
| Predespacho + CatBoost mensual 3 semillas, combinador MAE | 34,80 | 0,5 | -5,8 |
| Predespacho + CatBoost semanal, combinador MAE | 35,00 | 0,2 | -5,1 |
| Predespacho + CatBoost semanal con despacho programado (G) | 35,04 | 0,1 | -5,6 |
| Candidato afinado (desde 2021, 3000 iter., G, semanal) | 34,90 | -0,2 | -5,3 |

Todas las versiones de "predespacho + CatBoost con combinador MAE" quedan entre 34,8 y 35,1: es una meseta. La version
adoptable es la simple (mensual, sin G). G (despacho programado por tipo de recurso) choca con la limitacion OE1-3
del Anexo 1 y no mejora la combinacion.

CatBoost: perdida MAE, objetivo log(precio / predespacho) (aprende la correccion del predespacho), variables
realistas + predespacho de la hora y del dia (R+P). Lo que importa: la perdida L1 (con RMSE: 45,8) y reentrenar
(fijo hasta dic-2025: 37,5; mensual: 35,9).

## Lo que no funciona

- XGBoost con error cuadratico (42-46) o L1 (38,3); LEAR realista por hora (40,8-42,4).
- N-BEATSx / N-HiTS con el predespacho como variable futura: 117 -> 58, lejos del CatBoost.
- Delta ex post (D) y precio hibrido lleno con el predespacho (H) como variables del CatBoost: empeoran (38,9-40,4).
- Correccion de sesgo con los errores ya publicados (EWMA, 4 dias de rezago): ninguna variante mejora.
- Calibracion de nivel del predespacho y predespacho + delta como votante (bitacora anterior): no mejoran el ensamble.
- MPO y delta del propio predespacho: MPO = costo marginal y delta ~ 0; no aportan.
- Combinador: hora a hora (37,6), franjas de 3 h (37,0), olvido exponencial (36,3-36,4), medias y medianas (62-87).
- Ponderar por antiguedad en el CatBoost (39,9-41,2), profundidad 8 (39,6), historia desde 2023 (38,2).
- Pronostico oficial de demanda (PRON_SIN): semanal y sobrescrito despues de publicado; no auditable para fuga.
- Ofertas iniciales (OFERTAS/INICIAL/OFEImmdd.txt, precio diario y disponibilidad horaria por planta, creadas D-1
  ~08:10): el 100 % de los archivos revisados (mar-2023, mar-2025, mar-2026, jul-2026) se modifico despues del corte
  (XM los reescribe al inicio del mes siguiente) y hay versiones "_ORI", senal de correcciones. No auditable; ademas
  el predespacho ya resume las ofertas. No se descargo.

## Sesgo por mes que queda

Enero y febrero: el predespacho queda 20-30 por debajo con precios bajos; el CatBoost corrige la mitad (sesgo
-7 -> -4). Junio: 11 a 18 de junio el precio real quedo 60-170 por encima del predespacho y todos los modelos lo
siguen; el resto de junio se sigue bien. No hay dato publico previo que lo anticipe.

## Validacion historica (realista_validacion_historica.py)

Mismos 5 periodos de validacion_historica_predespacho.py. CatBoost mensual (R+P, sin G) + predespacho, combinador
MAE sin previa. MAE (predespacho solo -> predespacho + CatBoost; entre parentesis el ensamble 7 OPTIMISTA, que supone
todo el precio hasta el corte y usa ventanas 00-23):

| Periodo | Persistencia D-4 | Predespacho | Pre + CatBoost | cambio | DM t | (ens7 optimista) |
|---|---|---|---|---|---|---|
| 2020-21 | 26,6 | 13,6 | 9,3 | -31,7 % | -20,1 | (8,9) |
| 2021-22 | 37,0 | 12,5 | 11,9 | -5,0 % | -1,3 (p 0,21) | (10,5) |
| 2022-23 | 87,5 | 23,5 | 21,3 | -9,5 % | -6,6 | (19,8) |
| 2023-24 | 137,6 | 37,0 | 36,3 | -1,9 % | -7,6 | (34,0) |
| 2024-25 | 161,4 | 59,5 | 56,6 | -4,9 % | -9,0 | (53,2) |

(Con margen de 5 dias en el combinador; ver "Margen del combinador" abajo. Con margen 4: 9,3 / 11,6 / 21,1 / 36,3 / 56,5.)

Con G (validacion_historica_realista.csv) da practicamente lo mismo (9,3 / 11,9 / 21,0 / 36,3 / 56,6).
En 2026: predespacho 38,97 -> 35,02 (-10 %).

## Margen del combinador (rigor)

Al corte de D el precio llega hasta las 23:00 de D-4, pero la ventana de D-4 termina a las 00:00 de D-3. Con margen 4
el combinador entrenaba con esa hora aun no publicada (1 de 24 horas del dia mas reciente). Con margen 5
(realista_margen5.py): ensamble 7 realista 36,47 -> 36,57; predespacho + CatBoost 35,02 -> 35,06. Las cifras
defendibles son las de margen 5; realista_exp_combinador.causal usa 5 por defecto desde entonces.

## Mas pruebas (7-oct, noche)

- Forma intradia del predespacho (cambio frente a horas vecinas, posicion de la hora, z, hora del maximo y minimo)
  en el CatBoost: solo 35,94 -> 35,84; combinado 35,02 -> 35,08. No aporta.
- Error reciente del predespacho, log(real/predespacho) de ventanas ya publicadas (D-5), como variables: CatBoost
  35,94 -> 35,74; combinado 35,02 -> 34,88 (DM p = 0,13). No significativo; no se adopta.
- Precio preliminar TX1 de la API: mismo rezago de 3 dias que PrecBolsNaci. Archivos de Redespacho (publicados el
  mismo dia D hacia las 22:00): son operativos (seguridad, unidades, areas), sin precios.
- Motor (realista_motor.py, bandas conformes en escala log por hora con 5 dias de rezago, filtro intradia, peor
  mitad en COP/kWh): generador 151 (ens7 optimista) / 121 (ens7 realista) / 116 (predespacho + CatBoost) / 130
  (predespacho solo) / 180 (pronostico perfecto); comercializador 83,8 / 80,6 / 83,0 / 82,5 / 86,9. El
  comercializador casi no pierde; el generador pierde ~30 en el escenario realista y el mejor MAE no lo recupera
  (depende de la forma intradia, no del nivel). Ordenar las horas del dia con el predespacho: +3 a +4 (ruido).

## Costo marginal y demanda del despacho programado (dMAR / dDEM)

Archivos de la carpeta DESPACHO publicados D-1 hacia las 14:30 (descargar_dmar_ddem.py ->
despacho_programado_cm_demanda_xm.csv, 2018-2026). dMAR: costo marginal del despacho programado (con restricciones);
dDEM: demanda pronosticada por subarea, sumada para el sistema. Ambos son del sistema (no desglosan por tipo de
recurso). A tiempo: dDEM 99 %, dMAR 93 % (feb-may 2025 XM reescribio meses completos; se excluyen).
- dMAR solo es mucho peor que el iMAR y sesgado hacia abajo (2026: MAE 68 frente a 37,5; sesgo -64).
- CatBoost con forma (F) + dMAR/dDEM (C): combinado con el predespacho 34,75 (DM p = 0,24 frente a 35,02).
- dMAR como votante extra: MAE igual (35,07) pero MAPE 10,0 -> 9,1 % y mar-abr mucho mejor (24 -> 21, 38 -> 33);
  junio peor y sesgo -11.
- Promedio de 3 CatBoost (R+P, R+P+F+C, R+P+E) + predespacho: 34,71 frente a 35,06 en las mismas condiciones
  (margen 5; DM t = -3,1, p = 0,002). Ganancia pequeña, probablemente por promediar; elegida mirando 2026, asi que
  se valido en 2020-2025 (realista_validacion_historica_v2.py, margen 5):

| Periodo | Predespacho | Simple (Pre + Cat R+P) | v2 (Pre + promedio 3 Cat) | DM p | mejor Cat individual |
|---|---|---|---|---|---|
| 2020-21 | 13,63 | 9,31 | 9,23 | 0,002 | R+P+F+C (10,68) |
| 2021-22 | 12,53 | 11,91 | 11,67 | 0,001 | R+P+F+C (10,83) |
| 2022-23 | 23,50 | 21,27 | 21,05 | <0,001 | R+P+F+C (22,01) |
| 2023-24 | 36,99 | 36,28 | 36,32 | 0,52 | R+P+F+C (41,51) |
| 2024-25 | 59,52 | 56,58 | 56,52 | 0,61 | R+P+F+C (60,56) |

Ganancia pequeña pero robusta: mejora en 3 de 5 años y empata en 2; nunca empeora significativamente.
Por regimen en 2026: normal (ene-abr) 22,8 frente a 30,1 del predespacho; El Niño (may-ago) 47,8 frente a 48,8
(no significativo) y 51,4 del ensamble 7 realista.

## Rampas (rampas_realista.csv)

| Modelo | MAE | MAE en rampas | rampa anticipada | amplitud | R2 intradia |
|---|---|---|---|---|---|
| Ensamble 7 optimista | 32,6 | 127,2 | 0,58 | 0,91 | 0,689 |
| Ensamble 7 realista | 36,5 | 126,0 | 0,66 | 0,93 | 0,647 |
| Predespacho solo | 39,0 | 119,4 | 0,69 | 0,96 | 0,617 |
| Pre + CatBoost | 35,1 | 117,2 | 0,70 | 0,94 | 0,647 |
| Pre + promedio 3 CatBoost | 34,7 | 116,6 | 0,68 | 0,95 | 0,656 |

DM en rampas (Pre + 3 CatBoost frente al ensamble 7 realista): t = -5,5, p < 0,001.

## Detector de eventos realista (eventos_realista.csv)

Variables publicadas al corte (forma y nivel del predespacho, forma del real de D-5, frecuencia de eventos en 7
ventanas publicadas, hidrologia D-2, calendario); entrenado 2019-2025, reentrenado cada semana de 2026.
Desplome: AUC 0,944, BSS 0,35, PR 0,46 (detector actual optimista: 0,939 / 0,31 / 0,43).
Pico: 0,970 / 0,53 / 0,80 (actual: 0,972 / 0,53 / 0,80). No pierde nada con datos publicos.

## Archivos

Scripts en scripts_experimento/: realista_comun.py, realista_exp_arboles.py, realista_exp_neuronales.py,
realista_exp_lineal.py, realista_exp_votantes.py, realista_exp_combinador.py, realista_exp_frecuencia.py,
realista_exp_mezcla.py, realista_exp_programado.py, realista_exp_hiper.py, realista_exp_candidato.py,
realista_validacion_historica.py, realista_atlas.py, descargar_delta_expost.py, descargar_despacho_programado.py.
Atlas por mes: atlas/atlas_realista_nuevo_2026-MM.png (predespacho + CatBoost mensual, combinador MAE).
