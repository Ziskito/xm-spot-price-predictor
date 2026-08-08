# xm-spot-price-predictor
Sistema de predicción de precio de bolsa XM con ML y análisis de escenarios
## Dependencia especial: Prophet requiere CmdStan

Prophet necesita CmdStan (motor de inferencia bayesiana en C++), que en Windows
no se pudo compilar desde el código fuente (falla conocida de RTools/mingw32-make).
Se instaló en su lugar como binario precompilado vía conda-forge, en un entorno
separado del venv del proyecto:

    conda create -n cmdstan_env -c conda-forge cmdstan -y

La variable de entorno CMDSTAN debe apuntar a la carpeta resultante, por ejemplo:

    C:\Users\<usuario>\miniconda3\envs\cmdstan_env\Library\bin\cmdstan

En este equipo quedó fijada de forma permanente. En un equipo nuevo, hay que
repetir la instalación de conda y volver a fijar esa variable.