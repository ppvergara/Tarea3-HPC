# Traspaso técnico A/B — LU distribuida

## I. Objetivo y alcance

`lu_distributed.py` factoriza la matriz diagonal dominante del enunciado mediante
LU sin pivoteo, distribución bloque-cíclica 2D, paneles MPI y actualización local
con NumPy/SciPy. Genera únicamente las entradas locales; reconstruye factores
solo al solicitar una comprobación pequeña. No requiere editar el algoritmo al
cambiar el grid o los hilos dentro del alcance declarado.

Persona 2 utiliza B para C (bloques en el clúster) y D (configuraciones 0–3).
Persona 3 utiliza B y esos registros para E (uno/dos nodos, configuraciones 4–5),
F (bloques extremos en 2×2) y G (modelo de costo). Persona 2 debe conservar y
compartir los registros por rank de C/D para evaluar su reutilización en E/F.
Esta guía no realiza esas campañas ni sus análisis.

Está validada la correctitud local contra el serial, incluidos grids hasta 4×4
y ranks sin bloques. Las pruebas de entrega usan un entorno MPI existente:
no demuestran una instalación nueva desde cero. Antes de campañas grandes hay
que comprobar MPI, colocación, hilos, RAM y ejecución pequeña **en el clúster**.

## II. Preparación del entorno

Desde la raíz de una copia del repositorio, en Linux/WSL con Conda:

```bash
conda env create -f environment-mpi.yml
conda activate tarea3-hpc
```

Si el entorno ya funciona, basta activarlo. La especificación incluye Python
3.13, NumPy, SciPy, mpi4py, MPICH y threadpoolctl desde conda-forge. No fija
versiones exactas de todos los paquetes: registrar las realmente instaladas.
En el clúster usar el entorno y lanzador compatibles con su integración SLURM;
no suponer que un `mpirun` del sistema sirve con un mpi4py de otro proveedor.
La coincidencia de prefijo es una comprobación práctica, no una prueba de ABI.

```bash
command -v python mpiexec
python -c 'import sys,numpy,scipy,mpi4py; print(sys.executable); print(sys.version); print(numpy.__version__,scipy.__version__,mpi4py.__version__)'
mpiexec --version
python -c 'from mpi4py import MPI; print(MPI.get_vendor()); print(MPI.Get_library_version())'
```

Controlar hilos **antes de iniciar Python**, manteniendo las aserciones:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1
python -c 'import numpy,scipy.linalg,threadpoolctl; print(threadpoolctl.threadpool_info())'
mpiexec -n 2 python -c 'from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.rank,c.size,MPI.Get_processor_name(),flush=True)'
```

El último comando es diagnóstico y produce líneas por rank; la CLI LU produce
un único JSON. La consulta de pools indica configuración de bibliotecas, no
actividad instantánea de hilos. Revisar también `blas_pools` y `thread_env` de
**cada rank** en el JSON. En el entorno local comprobado: Python 3.13.18,
NumPy 2.5.4, SciPy 1.18.1, mpi4py 4.1.2, MPICH/Hydra 5.0.2,
OpenBLAS 0.3.34 (pthreads), threadpoolctl 3.8.0.

## III. Interfaz de ejecución

| Flag | Significado | Restricción / valor por defecto |
|---|---|---|
| `--N` | Dimensión de la matriz | Entero positivo; defecto 4096 |
| `--b` | Lado del bloque | Entero positivo; defecto 256; `N % b == 0` |
| `--pr` | Filas del grid | Entero positivo, obligatorio |
| `--pc` | Columnas del grid | Entero positivo, obligatorio |
| `--check` | Contraste con serial y residuo | Opcional; solo `N <= 256` |

El número de procesos debe ser `p=pr*pc`. Todos los ranks deben recibir los
mismos argumentos, incluido `--check`. Se conserva literalmente la forma del
grid: 2×4 y 4×2 son configuraciones distintas. No usar `python -O` ni un
`PYTHONOPTIMIZE` distinto de 0. Los hilos se configuran externamente.

Prueba pequeña de correctitud (requiere cuatro CPUs disponibles con T=1):

```bash
mpiexec -n 4 python -B lu_distributed.py --N 32 --b 8 --pr 2 --pc 2 --check
```

Ejecución normal del README, con JSON y stderr separados:

```bash
mpiexec -n 4 python -B lu_distributed.py --N 256 --b 32 --pr 2 --pc 2 > resultado.json 2> resultado.stderr
```

Ambos ejemplos se ejecutaron desde una copia limpia del commit indicado en IX.
Los tiempos de estas pruebas no son resultados oficiales de rendimiento.

## IV. Configuración MPI y BLAS

`p` cuenta procesos MPI; `T` cuenta hilos BLAS solicitados por proceso. En cada
nodo debe cumplirse `procesos_en_nodo*T <= CPUs_asignadas_en_nodo`. Dimensionar
además la memoria agregada de sus ranks y del runtime. No confundir 16 CPUs
lógicas de WSL con 16 cores físicos, ni con una asignación del clúster.

Para un smoke 2×2 con T=2 y al menos ocho CPUs permitidas:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mpiexec -n 4 python -B lu_distributed.py --N 32 --b 8 --pr 2 --pc 2 --check > t2.json 2> t2.stderr
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Se comprobó localmente ese smoke: las tres variables y OpenBLAS reportaron 2
para los cuatro ranks; el check pasó. No certifica rendimiento, afinidad,
T=4/8 ni operación del clúster con T>1.

El [enunciado, pp. 2–3](tarea03.pdf) exige estos parámetros de job:

| Parámetro | Valor o relación que debe configurar el responsable |
|---|---|
| `--qos`, `--partition` | `hpc-iic3533` en ambos |
| `--account` | `default-account` |
| `--ntasks` | `p=pr*pc`; el PDF usa `SLURM_NTASKS` para el lanzador |
| `--cpus-per-task` | `T` |
| `--nodes`, `--ntasks-per-node` | Según configuración, verificar colocación real |
| Variables BLAS | Las tres iguales a `SLURM_CPUS_PER_TASK` antes de Python |

Límites del enunciado por grupo: **8 CPUs por nodo, 4 GB de RAM, 15 minutos
por job, máximo 4 jobs en cola y 1 en ejecución**. Nodos listados: `ventress`
y `yodaxico`. Deben confirmar allí si los 4 GB se aplican por job/nodo/grupo y
qué unidad/política efectiva utiliza SLURM; esta guía no resuelve esa ambigüedad.

| Configuración | Grid | p | T | Nodos | Procesos/nodo | CPUs totales |
|---|---|---:|---|---:|---:|---|
| 0 | 1×1 | 1 | 1,2,4,8 | 1 | 1 | 1,2,4,8 |
| 1 | 1×2 o 2×1 | 2 | 1,2,4 | 1 | 2 | 2,4,8 |
| 2 | 2×2 | 4 | 1,2 | 1 | 4 | 4,8 |
| 3 | 2×4 o 4×2 | 8 | 1 | 1 | 8 | 8 |
| 4 | 2×4 | 8 | 1 | 2 | 4 | 8 |
| 5 | 4×4 | 16 | 1 | 2 | 8 | 16 |

Los scripts SLURM de campañas corresponden a Personas 2–3. Antes de lanzarlos:

1. Confirmar Python, mpi4py y biblioteca MPI cargada compatibles.
2. Confirmar el lanzador compatible con esa biblioteca y SLURM.
3. Comprobar `size=pr*pc` en el diagnóstico MPI.
4. Comprobar las tres variables y pools BLAS frente a `--cpus-per-task`.
5. Comprobar hostnames por rank, ranks por nodo y afinidad/asignación efectiva.
6. Ejecutar la prueba pequeña con `--check` en la asignación real.
7. Recoger y cargar el JSON completo, sin banners del lanzador en stdout.
8. Revisar CPUs, RAM, tiempo y límites de cola; medir memoria pico en un piloto autorizado.
9. Conservar stderr y código de salida, también ante errores.
10. Registrar commit, hashes y configuración de job utilizados.

## V. Salida JSON

Referencia normativa: [OUTPUT_CONTRACT.md](OUTPUT_CONTRACT.md), esquema **1.0**.
Todos los tiempos están en **segundos**, sin redondeo prematuro.

| Campo | Interpretación |
|---|---|
| `wall_s` | Máximo de `total_s` entre todos los ranks; no sumar ni promediar |
| `compute_s` | LU diagonal, solves y producto/resta Schur efectivamente ejecutados |
| `communication_s` | Intervalos dentro de Bcast LU, con posibles esperas y sincronización; no solo red |
| `total_s` | Fase LU local completa, incluido overhead; barrera inicial fuera, sin barrera final |
| `rank_times` | Exactamente p registros ordenados, con rank, row, col, tiempos, hostname e hilos |
| `check` | `not_run`, `passed` o `failed`; consultar los criterios numéricos |

Generación, inicialización, gathering final, referencia serial, reconstrucción,
check e impresión quedan fuera de LU. `unaccounted_s=total_s-compute_s-communication_s`
no se ajusta artificialmente; el contrato permite hasta -1e-9 s de redondeo.

Lectura rápida, independiente de MPI:

```bash
python -c 'import json; r=json.load(open("resultado.json")); print(r["wall_s"]); print([(q["rank"],q["compute_s"],q["communication_s"],q["total_s"]) for q in r["rank_times"]])'
```

Extracto **real** del smoke 2×2/T=1 de entrega, limitado a estos campos para
lectura (no es un documento completo del contrato):

```json
{
  "run_id": "ed787a87-02ff-4ff4-a2d2-6bf87e498a1b",
  "N": 32, "b": 8, "pr": 2, "pc": 2, "p": 4,
  "wall_s": 0.006420576999516925,
  "check": {
    "status": "passed",
    "relative_residual": 8.334633652212967e-17,
    "relative_factor_difference": 0.0,
    "max_absolute_factor_difference": 0.0,
    "residual_tolerance": 1e-10,
    "factor_tolerance": 1e-10,
    "max_N": 256
  }
}
```

El contrato enlazado contiene además un ejemplo real completo. Los registros
originales de entrega se conservan en el paquete de revisión técnica.

## VI. Recolección de resultados

Crear un directorio nuevo por corrida evita sobrescrituras. Con el entorno
y T previamente configurados, por ejemplo:

```bash
RUN_DIR="logs/$(date -u +%Y%m%dT%H%M%SZ)_N32_b8_2x2_T1"
mkdir -p logs
mkdir "$RUN_DIR"
(
  set -o noclobber
  printf '%s\n' '32b54f34aa429826de4d792479397ee383f8478e' > "$RUN_DIR/code-base.txt"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$RUN_DIR/commit.txt"
    git status --short > "$RUN_DIR/git-status.txt"
  fi
  sha256sum lu_distributed.py lu_serial.py docs/OUTPUT_CONTRACT.md > "$RUN_DIR/SHA256SUMS.txt"
  mpiexec -n 4 python -B lu_distributed.py --N 32 --b 8 --pr 2 --pc 2 --check > "$RUN_DIR/result.json" 2> "$RUN_DIR/stderr"
  RUN_RC=$?
  printf '%s\n' "$RUN_RC" > "$RUN_DIR/exit-code.txt"
  exit "$RUN_RC"
)
```

Si `mkdir` falla porque el nombre existe, elegir otro antes de ejecutar el bloque.
`noclobber` protege archivos de salida existentes. Registrar también N,b,pr,pc,p,T,
nodos, launcher, versiones Python/MPI/NumPy/SciPy/BLAS, identificador externo de
corrida y copia del script/configuración SLURM. El JSON incluye `run_id`, hashes,
fecha UTC y entorno, pero no conserva todo el script SLURM ni acredita su asignación.

En una copia de `git archive`/ZIP sin `.git`, `code.git_commit` y `git_dirty` son
null: conservar por separado el SHA de origen y comprobar `code.sha256`.
No rellenar el JSON original para simular que había un checkout Git.

Guardar stdout y stderr originales antes de agregar. Descartar del conjunto
válido, conservando evidencia, una corrida con retorno no cero, timeout,
`check.status=failed`, JSON incompleto, parámetros inesperados, hilos equivocados,
recursos excedidos o versión no identificable. `not_run` es normal cuando no
se pidió check; no significa correctitud numérica comprobada en esa corrida.

Agrupar después por versión, entorno, N,b,pr,pc,p,T,nodos y política de recursos.
No mezclar versiones o entornos sin identificarlos. Las tablas/gráficos pueden
construirse posteriormente sin cambiar la LU.

## VII. Prueba mínima antes de campañas

Ejecutar el ejemplo N32/b8/2×2 con `--check`, T=1 y cuatro CPUs asignadas.
Resultado válido: retorno 0, un JSON 1.0 único, cuatro ranks 0–3 con coordenadas
(0,0),(0,1),(1,0),(1,1), tiempos finitos, `wall_s=max(total_s)` y check `passed`
con ambos relativos <=1e-10; revisar stderr. En la entrega los stderr fueron vacíos.

Este consumidor de biblioteca estándar comprueba las invariantes básicas y
muestra los tiempos por rank; no modifica el documento:

```bash
python - "$RUN_DIR/result.json" <<'PY'
import json, math, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
assert r['schema_version'] == '1.0'
assert r['p'] == r['pr'] * r['pc'] and r['N'] % r['b'] == 0
assert len(r['rank_times']) == r['p']
for rank, q in enumerate(r['rank_times']):
    assert q['rank'] == rank
    assert (q['row'], q['col']) == divmod(rank, r['pc'])
    for key in ('compute_s', 'communication_s', 'total_s'):
        assert type(q[key]) in (int, float) and math.isfinite(q[key]) and q[key] >= 0
    assert q['unaccounted_s'] == q['total_s'] - q['compute_s'] - q['communication_s']
    assert q['unaccounted_s'] >= -1e-9
    print(rank, q['compute_s'], q['communication_s'], q['total_s'])
assert r['wall_s'] == max(q['total_s'] for q in r['rank_times'])
assert r['check']['status'] in ('passed', 'not_run')
print('wall_s:', r['wall_s'], 'check:', r['check']['status'])
PY
```

Para un chequeo estricto adicional con **T=1 y pools detectados**, se reutiliza
el consumidor existente (Python estándar, no lanza MPI):

```bash
python -B tests/run_robustness.py --validate-json "$RUN_DIR/result.json"
```

Este último compara también hashes con el código local y exige T=1; no usarlo
como validador de T=2. Para T>1 revisar variables y `num_threads` de cada pool.

| Síntoma | Identificación / acción |
|---|---|
| `size=... debe ser pr*pc=...` | Error de grid, retorno 2; corregir lanzamiento |
| `ModuleNotFoundError` | Intérprete incorrecto o dependencia ausente; revisar `sys.executable` |
| Error MPI_Init/UCX/PMI, `libmpi` ausente, ranks size=1 inesperados | Revisar entorno, ABI y lanzador antes de culpar a LU |
| Demasiados slots o p*T mayor que asignación | Oversubscription; ajustar recursos, no forzarla |
| `MPI Abort(1)`, rank/k/fase y traceback | Fallo local; guardar configuración, hashes y ambos streams para reproducir |
| `check.status=failed`, retorno 1 | Corrida numéricamente inválida; conservar y reportar |
| `JSONDecodeError`, ranks faltantes, stdout vacío/intercalado | Corrida incompleta; revisar stderr, retorno y salida del lanzador |

Con recursos suficientes, comprobar también N32/b8 en 2×4 y 4×2 (8 CPUs/T=1)
y N16/b8 en 4×4 (16 CPUs/T=1; 12 ranks sin bloques). El caso 4×4 local fue
mononodo WSL; en el clúster corresponde distribuir sus 16 procesos en dos nodos.

## VIII. Limitaciones conocidas y memoria

No hay bloques parciales: N%b debe ser cero. La LU no incluye intercambios de
filas y rechaza pivoteo inesperado; utiliza la matriz diagonal dominante del
problema. `--check` se limita a N<=256 y reconstruye en root fuera del cronómetro.
Correctitud local validada no certifica clúster ni multinodo. T=2 cuenta solo
con el smoke local descrito; T=4/8 y todo T>1 remoto siguen pendientes.

Memoria local de datos float64: `8*m*n` bytes, con m y n entradas de los ejes
locales. En un paso, L ocupa `8*m_k*b`, U `8*b*n_k`, diagonal `8*b*b` y producto
Schur `8*m_k*n_k`. Añadir copias NumPy/SciPy, temporales de generación/solves,
máscaras de finitud, paneles de pasos coexistentes, buffers internos y runtime
Python/MPI/BLAS. Incluso un rank sin bloques puede recibir diagonal/paneles.

Estimaciones analíticas previas, **no memoria pico medida ni garantía de RSS**:

| N / b / grid | Máximo arreglo local/rank | Datos locales agregados | Presupuesto agregado de arreglos LU |
|---|---:|---:|---:|
| 8192 / 256 / 2×2 | 128 MiB | 512 MiB | 1214.0625 MiB |
| 8192 / 4096 / 2×2 | 128 MiB | 512 MiB | 3648 MiB |
| 8192 / 4096 / 4×4 | 128 MiB | 512 MiB | 8448 MiB |
| 8192 / 512 / 4×4 | 32 MiB | 512 MiB | 1720.25 MiB |

Cada copia de diagonal b=4096 ocupa 128 MiB. Los presupuestos suman máximos
conservadores por rank, no necesariamente simultáneos; excluyen runtime,
workspaces y fragmentación y deben contrastarse además con la generación.
Sumar por **nodo según colocación real**, no dividir automáticamente por nodos.
Por ejemplo, agrupar hipotéticamente ranks 0–7 y 8–15 del caso 4×4/b4096 da
6016 y 2432 MiB de ese presupuesto: no prueba OOM, pero exige revisar RAM y
colocación. Tampoco el caso 2×2/b4096 certifica cabida bajo un límite de 4 GB.
La memoria pico real sigue sin medirse. No lanzar campañas grandes basándose
solo en estas estimaciones ni ocultar un problema de RAM cambiando la matemática.

## IX. Versión entregada y respaldo de correctitud

Commit integrado de código: **`32b54f34aa429826de4d792479397ee383f8478e`**,
con README y merge de la LU/pruebas. Comprobado contra `master` remoto mediante
consulta de solo lectura el 12/11/2026. Esta guía y los CSV son material nuevo
sin commit; el SHA identifica el código base, no una publicación de esta guía.

| Archivo protegido | SHA-256 |
|---|---|
| `lu_distributed.py` | `f1238b1be83954b615ecb88ae1f2290893158f36c9154f8400c28c611e6a5805` |
| `lu_serial.py` | `92d40dde1c02d54b92c9d757f2d028bb9da1365f1b49441e324520c219c5727c` |
| `OUTPUT_CONTRACT.md` | `f4f5131e49450aa995b0c6da4e6a3050ee2e370a024f7b7a4793c788009fcdf2` |

El código y contrato permanecen idénticos a la versión local previamente
aprobada: 137/137 pruebas integrales y 132/132 regresiones, sin timeouts.
Las 29 ejecuciones numéricas cubren 28 configuraciones y una repetición,
N=1,8,16,32,40,48,64,72,96,128,256, b=1,4,8,16,24,32,128 y grids
1×1,1×2,2×1,2×2,2×3,3×2,1×4,4×1,2×4,4×2,4×4 (no todo el producto cartesiano).
Se contrastaron factores con `lu_serial.lu_blocked`, allclose atol=rtol=1e-10,
Frobenius relativo <=1e-10 y residuos MPI/serial <=1e-10. Máximos observados:
residuo MPI `1.6789673676683307e-16`, diferencia absoluta `1.3877787807814457e-17`
y relativa `6.101361561887498e-19`. Se observaron pivotes identidad y hasta
15 ranks sin bloques. Los casos de errores inducidos acreditan detección,
no factorizaciones exitosas. Los tests/runners públicos permiten reproducir
esas comprobaciones en otro entorno con recursos compatibles.

La comprobación adicional de entrega desde `git archive` pasó seis casos:
los cuatro grids indicados en VII con check/T=1, N256/b32/2×2 sin check y
N32/b8/2×2 con T=2. Todos con JSON completo, retorno 0, stderr vacío y hashes
idénticos; los cinco checks pasaron. Los consumidores independientes verificaron
tiempos/coordenadas/metadatos y hilos 1/2, incluidos los 12 ranks vacíos de 4×4.
Se reutilizó el entorno MPI existente, sin copiarlo al árbol limpio ni reinstalar.

## X. Checklist de recepción

- [ ] Preparar/activar entorno y registrar versiones e intérprete.
- [ ] Lanzar el diagnóstico MPI, comprobar size y nodos.
- [ ] Ejecutar LU pequeña con grid explícito y check `passed`.
- [ ] Obtener un único JSON 1.0 con todos los ranks y retorno 0.
- [ ] Leer tiempos por rank y comprobar `wall_s=max(total_s)`.
- [ ] Configurar hilos externamente, verificar pools y CPUs por nodo sin sobreasignar.
- [ ] Guardar JSON/stderr originales, retorno, commit/hashes, run_id y job/configuración.
- [ ] Reconocer y excluir corridas inválidas sin borrar sus registros.

## XI. Resultado serial A disponible para el grupo

**b_A=256 aprobado**, N=4096, un hilo BLAS; tres repeticiones independientes
por cada uno de los siete bloques. Menor mediana observada: **2.114 s**.
Los tiempos son los impresos por el serial con resolución 0.001 s; excluyen
generación, checksum y verificación. Pilotos y diagnósticos no se mezclaron.

| b | N/b | t1 (s) | t2 (s) | t3 (s) | Mediana (s) | Mín–máx (s) | Rango (s) |
|---:|---:|---:|---:|---:|---:|---|---:|
| 64 | 64 | 3.188 | 2.992 | 2.831 | 2.992 | 2.831–3.188 | 0.357 |
| 128 | 32 | 2.615 | 2.263 | 2.486 | 2.486 | 2.263–2.615 | 0.352 |
| **256** | 16 | 2.402 | 2.110 | 2.114 | **2.114** | 2.110–2.402 | 0.292 |
| 512 | 8 | 2.252 | 2.141 | 2.283 | 2.252 | 2.141–2.283 | 0.142 |
| 1024 | 4 | 2.318 | 2.170 | 2.310 | 2.310 | 2.170–2.318 | 0.148 |
| 2048 | 2 | 2.639 | 2.445 | 2.473 | 2.473 | 2.445–2.639 | 0.194 |
| 4096 | 1 | 3.073 | 2.781 | 2.954 | 2.954 | 2.781–3.073 | 0.292 |

La elección sigue el criterio previamente aprobado de menor mediana.
Frente a b512 (2.252 s), la reducción mediana es 0.138 s, aproximadamente
6.13 %. Los rangos se solapan: tres observaciones no establecen superioridad
estadística. **b_A no determina automáticamente el mejor b del clúster**.
No hace falta repetir A para utilizar este resultado.

Entorno original de A (distinto del MPI de B): Intel Core i7-13620H,
Ubuntu 24.04.5/WSL2, Python 3.13.15, NumPy 2.5.3, SciPy 1.18.1,
threadpoolctl 3.5.0, Intel MKL 2025.0-Product. Las tres variables se fijaron
a 1 antes de cada hijo; MKL/OpenMP reportaron 1 en diagnósticos separados.
Referencia: [lu_serial.py](../lu_serial.py), SHA-256 de IX, sin modificaciones.

Datos reutilizables:

- [a_serial_runs.csv](../results/oficiales/a_serial_runs.csv): 21 filas crudas,
  valores originales, IDs/orden/repetición, estado y hashes de streams.
- [a_serial_summary.csv](../results/oficiales/a_serial_summary.csv): siete filas
  agregadas, copia exacta del resumen aprobado, contrastada con stdout originales.

En el CSV crudo se conservaron `run_id,type,N,b,repetition,order,factorization_s,
returncode,status,stdout_sha256,stderr_sha256`. Se omitieron exactamente
`started,ended,residual,stdout,stderr,error,command,external_process_s`:
comandos contienen rutas personales, las rutas de streams son privadas,
residual no se midió en oficiales, error estaba vacío y los otros campos no
son necesarios para recalcular la tabla. No se alteraron tiempos ni originales.
El resumen conserva todas sus columnas, incluido GFLOP/s del modelo 2N³/3,
que no es un contador de hardware. Los hashes permiten cotejar streams conservados
en la evidencia de revisión; no implican que esos logs estén en el repo público.

Recálculo sin MPI, sin dependencias privadas y sin repetir experimentos:

```bash
python - <<'PY'
import csv, statistics
from decimal import Decimal
with open('results/oficiales/a_serial_runs.csv') as f:
    rows = list(csv.DictReader(f))
with open('results/oficiales/a_serial_summary.csv') as f:
    for s in csv.DictReader(f):
        times = [Decimal(r['factorization_s']) for r in rows if r['b'] == s['b']]
        assert len(times) == 3
        assert statistics.median(times) == Decimal(s['median_s'])
        assert max(times) - min(times) == Decimal(s['range_s'])
        print(s['b'], times, s['median_s'], s['range_s'])
PY
```

Los scripts históricos [run_a_serial.py](../scripts/run_a_serial.py) y
[summarize_a.py](../scripts/summarize_a.py) documentan el protocolo original.
El primero fija un intérprete personal y consulta documentación privada;
no es un runner portable ni un paso de instalación de B. El segundo espera
la campaña original completa con metadatos/logs: para los CSV compartidos usar
el recálculo anterior. Sus textos históricos de «propuesta pendiente» no
sustituyen la decisión aprobada b_A=256 aquí registrada.
