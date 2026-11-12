# Contrato de salida LU MPI — esquema 1.0

El apartado B del enunciado exige LU distribuida, tiempos separados de cómputo y
comunicación por proceso y comparación con el serial. Este esquema, el wall-time,
la política de cronómetros, el límite de comprobación y sus tolerancias son
las decisiones metodológicas de Persona 1 para consumir esas mediciones.

Una ejecución válida emite **un objeto JSON** en stdout, exclusivamente desde
rank 0, terminado por newline. Se puede leer con `json.load`. Varias ejecuciones
pueden guardarse externamente como JSON Lines. No hay banners ni progreso.
La ayuda `--help` es una operación de interfaz, sin registro experimental.
Los errores van a stderr. Todos los tiempos son números JSON en **segundos**,
con la precisión de `float` original, sin redondeo ni NaN/Infinity.

## Campos en orden de construcción

| Campo | Tipo | Significado |
|---|---|---|
| `schema_version` | string | Siempre `"1.0"` |
| `run_id` | string UUID | Identificador generado por root antes de medir |
| `started_at` | string ISO 8601 con zona | Fecha UTC de preparación del registro en root, antes de Barrier/LU |
| `code` | object | Identificación del código descrita abajo |
| `environment` | object | Entorno de root descrito abajo |
| `N`, `b`, `pr`, `pc`, `p` | integers positivos | Tamaño, bloque, filas/columnas del grid y procesos; `N%b=0`, `p=pr*pc` |
| `wall_s` | number | `max(r.total_s for r in rank_times)` |
| `rank_times` | array de objects | Exactamente p registros, en orden ascendente de rank |
| `check` | object | Estado y resultados numéricos descritos abajo |

`code.git_commit` es HEAD disponible o null; `code.git_dirty` es boolean o null
si Git no está disponible. HEAD identifica la base y puede tener cambios locales.
`code.sha256` contiene SHA-256 de `lu_distributed.py` y `lu_serial.py`, que identifica
el contenido efectivamente disponible al preparar la corrida. No requiere un
repositorio Git para ejecutar desde un ZIP.

`environment` contiene strings `python`, `executable`, `numpy`, `scipy`, `mpi4py`,
`mpi_library` (versión/configuración de la biblioteca cargada), `platform`,
`thread_policy="external; requested values recorded per rank"`,
`clock="time.perf_counter"` y numbers `clock_resolution_s`, `time_tolerance_s`.
Estos datos describen root; los pools, variables y hostname son individuales.
No se afirma una configuración de clúster ni un número de nodos no observado.

Cada registro de `rank_times` contiene:

| Campo | Tipo | Definición |
|---|---|---|
| `rank`, `row`, `col` | integers | `row=rank//pc`, `col=rank%pc` |
| `compute_s` | number >=0 | Suma de intervalos numéricos locales LU |
| `communication_s` | number >=0 | Suma de intervalos de las Bcast LU efectivamente llamadas |
| `total_s` | number >=0 | Duración local de toda `lu_blocked_distributed` |
| `hostname` | string | Nombre reportado por `MPI.Get_processor_name()` |
| `thread_env` | object | Strings solicitados o null de OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS |
| `blas_pools` | array o null | Consulta previa con threadpoolctl; null si no está instalado. Puede ser vacía si no detecta pools |
| `unaccounted_s` | number | `total_s-compute_s-communication_s`, sin ajuste |

Cada pool es el registro devuelto por threadpoolctl; usualmente incluye
`user_api`, `internal_api`, `num_threads`, `prefix`, `filepath`, `version`,
`threading_layer`, `architecture`. Los campos específicos dependen del backend.
`num_threads` es la configuración reportada por la biblioteca durante la consulta,
no una traza de utilización de CPU. Un valor solicitado no certifica por sí solo
la cantidad efectiva. p procesos y T hilos por proceso son recursos distintos.
El algoritmo nunca fija T internamente.

## Alcance temporal fijo

Todos los intervalos usan el mismo reloj monótono de alta resolución
`time.perf_counter()`. Solo se comparan duraciones locales; no se comparan las
referencias absolutas de relojes entre nodos.

`compute_s` incluye exclusivamente:

1. Llamada a `lu_diag_block` en el propietario de la diagonal.
2. Solve U y escritura al buffer `upper` cuando hay datos.
3. Solve L y escritura al buffer `lower` cuando hay datos.
4. Producto y resta `active -= lower @ upper` cuando la región es no vacía.

La copia inicial de diagonal, su copia final a almacenamiento local, las copias
finales de los paneles a la matriz local, asignación de buffers, slices,
validaciones, contexto de warnings/errstate y comprobaciones de finitud quedan
como overhead. La política es la misma para todos los ranks y configuraciones.
Un rank que no ejecuta operaciones numéricas acumula exactamente cero cómputo.

`communication_s` incluye la duración local de cada `MPI.Bcast`:

| Orden | Difusión | Comunicador | Participantes | Raíz local |
|---|---|---|---|---|
| 1 | Diagonal por fila | row_comm de r | myrow=r | c |
| 2 | Diagonal por columna | col_comm de c | mycol=c | r |
| 3 | Panel L | Cada row_comm | Todos | c |
| 4 | Panel U | Cada col_comm | Todos | r |

Aquí r=k%pr, c=k%pc. Se preservan orden, buffers float64 C/MPI.DOUBLE y
llamadas con count=0. No hay comunicaciones en Schur. El tiempo de las
colectivas **incluye espera y sincronización**, además de transferencia;
no es latencia pura de red ni tiempo exclusivo de transporte.

El flujo de CLI es:

```text
parseo/validación -> grid -> matriz local -> metadatos -> acumulador nuevo
-> COMM_WORLD.Barrier()                         [fuera de tiempos]
-> inicio total_s -> LU completa -> fin total_s [incluye control/overhead]
-> gather métricas -> check opcional -> JSON -> Free [fuera de tiempos]
```

No existe Barrier final. Cada rank detiene su propio total antes de su gather;
root recibe todos los tiempos antes de verificar o construir la salida global.
Los gathers diagnósticos y el bcast del resultado de check tampoco se acumulan.
Inicio del intérprete, imports, generación, setup, barrera, referencia serial,
residuo, reconstrucción, recolección, serialización, impresión y liberación no
entran en las tres métricas. Los intervalos numéricos y MPI son disjuntos.

`unaccounted_s` mide overhead transcurrido sin clasificar, incluida instrumentación;
no es tiempo exclusivo de CPU ni comunicación adicional. No se fuerza igualdad
entre total y las dos sumas. Se rechaza overhead menor que -1e-9 s; ese margen
absoluto de 1 ns corresponde a un tick de la resolución observada y solo admite
redondeo temporal minúsculo. No se truncan ni corrigen los valores aceptados.
La resolución realmente reportada se registra; el margen del esquema 1.0 es fijo.

Son inválidos ranks ausentes/repetidos, coordenadas inconsistentes y tiempos
negativos/no finitos. `wall_s` se calcula después del gather desde los totales
validados. No sumar tiempos de ranks ni sustituir el wall-time por su promedio,
por compute_s máximo o por duración externa de Python.

## Política de --check

Solo admite **N<=256**. Se rechaza un tamaño mayor colectivamente con rc=2,
antes de crear el grid o generar matrices; no se desactiva silenciosamente.

Después de detener LU y reunir métricas se recogen los factores locales en
root, incluyendo ranks con shapes (0,n), (m,0), (0,0). Root deriva ejes de cada
rank, verifica shape, cobertura única y finitud, y reconstruye los factores
empaquetados. Genera A con `matrix_entries`, ejecuta `lu_serial.lu_blocked`
original sobre una copia con el mismo N,b y calcula:

- `relative_residual = ||A-L@U||_F / ||A||_F`.
- `relative_factor_difference = ||LU_MPI-LU_serial||_F / ||LU_serial||_F`.
- `max_absolute_factor_difference = max(abs(LU_MPI-LU_serial))`, diagnóstico.

Se aceptan ambos relativos **<=1e-10**, sin exigir igualdad bit a bit. Shapes y
valores deben ser válidos/finitos. Los factores locales no se modifican.

Sin solicitud: `check={"status":"not_run"}`, sin métricas inventadas.
Con solicitud: `status` es `passed` o `failed`; se agregan los tres números
anteriores, `residual_tolerance=1e-10`, `factor_tolerance=1e-10`, `max_N=256`.
El estado se comunica por bcast fuera de LU para que todos retornen igual.
Un fallo numérico emite JSON inequívoco `failed`, diagnóstico stderr y rc=1;
**esa corrida no es válida para rendimiento**. Una excepción irrecuperable
(incluidas reconstrucción o referencia serial) usa MPI.Abort(1) y diagnóstico,
sin fabricar un JSON de éxito. No se introducen colectivas de recuperación.
Los errores iniciales de argumentos retornan 2; LU local conserva su Abort.

## Invocación y consumo

Desde un entorno MPI compatible (Python y mpiexec del mismo prefijo), en secuencia:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1
mpiexec -n 4 python -B lu_distributed.py --N 32 --b 8 --pr 2 --pc 2 > run.json 2> run.stderr
mpiexec -n 8 python -B lu_distributed.py --N 16 --b 8 --pr 2 --pc 4 --check > check.json 2> check.stderr
python -c 'import json; r=json.load(open("run.json")); print(r["wall_s"])'
```

No usar `-O`: elimina el assert original de pivoteo y el algoritmo lo rechaza.
Dimensionar p*T según CPUs permitidas; no activar oversubscription. Estos ejemplos
son pruebas pequeñas, no campañas C–G. En este repositorio el prefijo existente
es `.local/etapa_02/envs/tarea3-hpc`; no viene incluido en el ZIP de revisión.

Las pruebas se reproducen con `tests/run_metrics.py --output <directorio_nuevo>`;
los tres runners anteriores también siguen disponibles. La guía con comandos
y evidencia está en `.local/etapa_05/REPRODUCIR.md` del paquete de revisión.

## Ejemplo real

Registro íntegro de la prueba `tests_01/cli_0_check` (N=16,b=16,grid1×1), obtenido
por el runner; únicamente se añade indentación para lectura. Los tiempos son
pruebas funcionales de metrología, sin interpretación como rendimiento oficial.

```json
{
  "schema_version": "1.0",
  "run_id": "2fe5d3bc-f598-4d78-b60f-f99cdfa1eb17",
  "started_at": "2026-11-12T18:33:19.141601+00:00",
  "code": {
    "git_commit": "6e16fc53eb11967d818e338fb9273132d9323284",
    "git_dirty": true,
    "sha256": {
      "lu_distributed.py": "f1238b1be83954b615ecb88ae1f2290893158f36c9154f8400c28c611e6a5805",
      "lu_serial.py": "92d40dde1c02d54b92c9d757f2d028bb9da1365f1b49441e324520c219c5727c"
    }
  },
  "environment": {
    "python": "3.13.18 | packaged by conda-forge | (main, Oct 26 2026, 11:38:18) [GCC 15.3.0]",
    "executable": "/home/ppvergara/U/Compu/Tarea3-HPC/.local/etapa_02/envs/tarea3-hpc/bin/python",
    "numpy": "2.5.4",
    "scipy": "1.18.1",
    "mpi4py": "4.1.2",
    "mpi_library": "MPICH Version:      5.0.2\nMPICH Release date: Fri Sep 25 20:45:26 UTC 2026\nMPICH ABI:          18:2:6\nMPICH Device:       ch4:ucx ofi\nMPICH configure:    --prefix=/home/ppvergara/U/Compu/Tarea3-HPC/.local/etapa_02/envs/tarea3-hpc --disable-doc --disable-dependency-tracking --enable-cxx --enable-fortran --enable-f08 --enable-mpi-abi --with-wrapper-dl-type=none --disable-static --with-device=ch4:ucx,ofi --with-ucx=/home/ppvergara/U/Compu/Tarea3-HPC/.local/etapa_02/envs/tarea3-hpc --with-libfabric=/home/ppvergara/U/Compu/Tarea3-HPC/.local/etapa_02/envs/tarea3-hpc --with-hwloc=/home/ppvergara/U/Compu/Tarea3-HPC/.local/etapa_02/envs/tarea3-hpc --with-yaksa=embedded --with-pm=hydra:gforker\nMPICH CC:           x86_64-conda-linux-gnu-cc   \nMPICH CXX:          x86_64-conda-linux-gnu-c++   \nMPICH F77:          x86_64-conda-linux-gnu-gfortran   \nMPICH FC:           x86_64-conda-linux-gnu-gfortran   \nMPICH features:     threadcomm\n",
    "platform": "Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39",
    "thread_policy": "external; requested values recorded per rank",
    "clock": "time.perf_counter",
    "clock_resolution_s": 1e-09,
    "time_tolerance_s": 1e-09
  },
  "N": 16,
  "b": 16,
  "pr": 1,
  "pc": 1,
  "p": 1,
  "wall_s": 0.0005008429998270003,
  "rank_times": [
    {
      "rank": 0,
      "row": 0,
      "col": 0,
      "compute_s": 0.00020422200032044202,
      "communication_s": 3.237999771954492e-05,
      "total_s": 0.0005008429998270003,
      "hostname": "PEDRO",
      "thread_env": {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1"
      },
      "blas_pools": [
        {
          "user_api": "blas",
          "internal_api": "openblas",
          "num_threads": 1,
          "prefix": "libopenblas",
          "filepath": "/home/ppvergara/U/Compu/Tarea3-HPC/.local/etapa_02/envs/tarea3-hpc/lib/libopenblasp-r0.3.34.so",
          "version": "0.3.34",
          "threading_layer": "pthreads",
          "architecture": "Haswell"
        }
      ],
      "unaccounted_s": 0.0002642410017870134
    }
  ],
  "check": {
    "status": "passed",
    "relative_residual": 1.2896192303615033e-16,
    "relative_factor_difference": 0.0,
    "max_absolute_factor_difference": 0.0,
    "residual_tolerance": 1e-10,
    "factor_tolerance": 1e-10,
    "max_N": 256
  }
}
```
