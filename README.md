
# Tarea 3 — Computación de Alto Rendimiento (IIC3533)

Implementación de factorización LU distribuida mediante **MPI4PY, NumPy y SciPy**, desarrollada para la Tarea 3 del curso IIC3533.

El repositorio incluye la versión serial de referencia y una implementación paralela con distribución bloque-cíclica bidimensional.

## 1. Archivos principales

- `lu_serial.py`: implementación serial de referencia.
- `lu_distributed.py`: factorización LU distribuida con MPI.
- `environment-mpi.yml`: dependencias del entorno Python/MPI.
- `docs/OUTPUT_CONTRACT.md`: especificación de la salida JSON y las métricas.
- `tests/`: pruebas de distribución, comunicaciones, correctitud, métricas y robustez.
- `scripts/`: scripts utilizados para los experimentos seriales.

## 2. Preparación del entorno

Se requiere Python con NumPy, SciPy, MPI4PY y una implementación MPI compatible.

Para crear el entorno local con Conda:

```bash
conda env create -f environment-mpi.yml
conda activate tarea3-hpc
```

Es importante utilizar un intérprete Python y un lanzador `mpiexec` compatibles con la misma implementación MPI.

Antes de ejecutar el programa, configurar los hilos de BLAS:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONOPTIMIZE=0
```

En el clúster, estas variables deben ajustarse según los CPUs solicitados por proceso en SLURM.

## 3. Ejecución de LU distribuida

El programa recibe:

| Parámetro | Descripción |
|---|---|
| `--N` | Dimensión de la matriz |
| `--b` | Tamaño de bloque |
| `--pr` | Filas del grid de procesos |
| `--pc` | Columnas del grid de procesos |
| `--check` | Verificación numérica opcional |

Ejemplo con cuatro procesos, grid 2×2 y verificación de correctitud:

```bash
mpiexec -n 4 python -B lu_distributed.py \
  --N 32 --b 8 --pr 2 --pc 2 --check
```

Para guardar la salida de una ejecución:

```bash
mpiexec -n 4 python -B lu_distributed.py \
  --N 256 --b 32 --pr 2 --pc 2 > resultado.json
```

**Condiciones importantes:**

- El número de procesos debe cumplir `p = pr × pc`.
- La dimensión de la matriz debe ser divisible por el bloque: `N % b == 0`.
- `--check` solo está disponible para `N <= 256`.
- No utilizar `python -O`, ya que desactiva comprobaciones de pivoteo.
- Evitar oversubscription: respetar los CPUs disponibles y los hilos por proceso.

## 4. Resultados y métricas

Cada ejecución válida genera un único objeto **JSON desde el rank 0**, con la configuración, información del entorno y tiempos individuales de todos los procesos.

Las principales métricas son:

| Métrica | Descripción |
|---|---|
| `compute_s` | Tiempo de operaciones numéricas locales |
| `communication_s` | Tiempo dentro de las comunicaciones MPI |
| `total_s` | Tiempo total local de factorización |
| `wall_s` | Máximo `total_s` entre todos los procesos |
| `unaccounted_s` | Tiempo no clasificado como cómputo o comunicación |

Todos los tiempos están expresados en **segundos**.

El tiempo de comunicación incluye posibles esperas y sincronización dentro de las colectivas MPI; no representa exclusivamente transferencia de datos por red.

La inicialización, la verificación de correctitud y la recolección final de métricas quedan fuera del tiempo medido.

Para consultar el tiempo global de una ejecución guardada:

```bash
python -c 'import json; print(json.load(open("resultado.json"))["wall_s"])'
```

La definición completa de los campos se encuentra en [`docs/OUTPUT_CONTRACT.md`](docs/OUTPUT_CONTRACT.md).

## 5. Correctitud y validación

La implementación distribuida fue contrastada con `lu_serial.py` mediante comparaciones de factores LU y residuos relativos.

Se probaron diferentes tamaños de matriz, tamaños de bloque y grids de hasta 4×4, incluyendo procesos sin bloques locales.

Las pruebas locales de distribución, paneles, LU completa, métricas y robustez finalizaron satisfactoriamente.

**Importante:** estas verificaciones se realizaron en un entorno local con MPICH. Antes de las campañas oficiales, se debe comprobar el funcionamiento del programa en el clúster, la compatibilidad de MPI, la configuración de hilos y el consumo de memoria.

## 6. Experimentos en el clúster

Para los apartados C–G se debe utilizar la configuración de procesos, hilos y nodos indicada en el enunciado.

Se recomienda:

1. Verificar el entorno Python/MPI del clúster.
2. Ejecutar primero una prueba pequeña.
3. Confirmar que el JSON contiene todos los ranks y tiempos válidos.
4. Configurar los recursos de SLURM según cada experimento.
5. Guardar las salidas JSON para construir las tablas y gráficos.

Los resultados de las pruebas locales no sustituyen las mediciones de rendimiento solicitadas en el clúster.
