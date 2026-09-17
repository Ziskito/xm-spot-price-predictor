#!/bin/bash
# Cola SECUENCIAL de los trabajos pesados del 2026-09-11 (nunca dos entrenamientos a la vez: la
# maquina tiene limite termico). Cada paso escribe su propio log y la cola deja un rastro con hora
# de inicio/fin y codigo de salida en logs_cola.txt.
cd /c/Users/mgdbj/xm-spot-price-predictor || exit 1
source venv/Scripts/activate
RES=data/processed/resultados

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs_cola.txt; }
correr() {
    nombre=$1; shift
    log "INICIO $nombre"
    "$@" > "logs_$nombre.txt" 2>&1
    log "FIN $nombre (codigo $?)"
}

correr tft_lstm      python scripts_experimento/tft_lstm_24h.py 32 256 1000 ambos
correr ventanas      python scripts_experimento/ventanas_calibracion_24h.py
correr recalibracion python scripts_experimento/recalibracion_2026.py 30
correr eval72diaria  python scripts_experimento/evaluacion_72h_diaria.py

# Respaldo de las salidas del walk-forward de 3 meses antes de reejecutar los notebooks
for f in walkforward_predicciones_crudas diebold_mariano walkforward_5origenes; do
    [ -f "$RES/$f.csv" ] && [ ! -f "$RES/${f}_3meses_backup.csv" ] && cp "$RES/$f.csv" "$RES/${f}_3meses_backup.csv"
done
log "Respaldo de salidas de 3 meses listo"

correr nb10 python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 notebooks/10_diebold_mariano_juan.ipynb
correr nb07 python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 notebooks/07_validacion_walkforward_juan.ipynb
log "COLA COMPLETA"
