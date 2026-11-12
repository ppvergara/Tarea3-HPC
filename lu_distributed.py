"""LU distribuida bloque-cíclica con métricas por rank y salida JSON 1.0.

Ejemplo: mpiexec -n 4 python lu_distributed.py --N 32 --b 8 --pr 2 --pc 2
La CLI factoriza in-place mediante diagonal/paneles y Schur local.
Hilos BLAS configurados externamente. --check admite N <= 256.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid
import warnings

import mpi4py
from mpi4py import MPI
import numpy as np
import scipy
from scipy.linalg import LinAlgWarning, solve_triangular

from lu_serial import lu_blocked, lu_diag_block, matrix_entries, residual


CHECK_MAX_N = 256
CHECK_TOLERANCE = 1e-10
TIME_TOLERANCE_S = 1e-9  # Un tick del reloj observado; no se corrigen los datos.
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


@dataclass
class RankTimers:
    """Acumulador nuevo por factorización; todos los intervalos usan perf_counter."""
    compute_s: float = 0.0
    communication_s: float = 0.0
    total_s: float = 0.0


@contextmanager
def _measure(timers, field):
    if timers is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        setattr(timers, field, getattr(timers, field) + elapsed)


@dataclass(frozen=True)
class Parameters:
    N: int
    b: int
    pr: int
    pc: int
    check: bool = False


def validate_parameters(comm, params):
    """Validación colectiva previa a Split; todos aceptan o lanzan ValueError.

    Todos los miembros de comm deben llamar, incluso si sus parámetros difieren.
    """
    configurations = comm.allgather(params)
    if any(p != configurations[0] for p in configurations):
        raise ValueError('Los argumentos deben ser idénticos en todos los ranks.')
    for name in ('N', 'b', 'pr', 'pc'):
        if getattr(params, name) <= 0:
            raise ValueError(f'{name} debe ser > 0.')
    if params.N % params.b:
        raise ValueError('N debe ser divisible por b (N % b == 0).')
    if comm.Get_size() != params.pr * params.pc:
        raise ValueError(f'COMM_WORLD size={comm.Get_size()} debe ser pr*pc='
                         f'{params.pr * params.pc}.')
    if params.check and params.N > CHECK_MAX_N:
        raise ValueError(f'--check requiere N <= {CHECK_MAX_N}; recibido N={params.N}.')


class _ParseExit(Exception):
    def __init__(self, status):
        self.status = status


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)

    def exit(self, status=0, message=None):
        raise _ParseExit(status)


def parse_parameters(comm, argv=None):
    """Parsea en cada rank y coordina incluso errores de sintaxis y --help."""
    parser = _Parser(description=__doc__)
    parser.add_argument('--N', type=int, default=4096)
    parser.add_argument('--b', type=int, default=256)
    parser.add_argument('--pr', type=int, required=True)
    parser.add_argument('--pc', type=int, required=True)
    parser.add_argument('--check', action='store_true',
                        help=f'verifica factores y residuo frente al serial (N <= {CHECK_MAX_N})')
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            args = parser.parse_args(argv)
        outcome = ('parameters', Parameters(**vars(args)))
    except ValueError as exc:
        outcome = ('error', str(exc))
    except _ParseExit as exc:
        outcome = ('help' if exc.status == 0 else 'error', captured.getvalue())
    outcomes = comm.allgather(outcome)
    errors = [(r, value) for r, (kind, value) in enumerate(outcomes) if kind == 'error']
    if errors:
        rank, message = errors[0]
        raise ValueError(f'Argumentos inválidos en rank {rank}: {message}')
    if any(o[0] != outcomes[0][0] for o in outcomes):
        raise ValueError('Los argumentos deben ser idénticos en todos los ranks.')
    if outcome[0] == 'help':
        if comm.Get_rank() == 0:
            print(captured.getvalue(), end='')
        return None
    validate_parameters(comm, outcome[1])
    return outcome[1]


@dataclass
class ProcessGrid:
    comm: MPI.Comm
    params: Parameters
    rank: int
    myrow: int
    mycol: int
    row_comm: MPI.Comm
    col_comm: MPI.Comm

    @classmethod
    def create(cls, comm, params):
        validate_parameters(comm, params)
        rank = comm.Get_rank()
        myrow, mycol = divmod(rank, params.pc)
        row_comm = comm.Split(myrow, mycol)
        col_comm = comm.Split(mycol, myrow)
        return cls(comm, params, rank, myrow, mycol, row_comm, col_comm)

    def free(self):
        """Liberar colectivamente al terminar; no libera el comunicador padre."""
        self.row_comm.Free()
        self.col_comm.Free()


class BlockCyclicLayout:
    """Metadatos locales; bloques y entradas en orden global ascendente.

    Requiere parámetros validados y coordenadas del grid. Los mapeos de un
    bloque/entrada ajeno devuelven None; índices fuera del dominio se rechazan.
    """

    def __init__(self, grid):
        self.params = grid.params
        self.myrow, self.mycol = grid.myrow, grid.mycol
        p = self.params
        self.nb = p.N // p.b
        self.block_rows = np.arange(self.myrow, self.nb, p.pr, dtype=np.int64)
        self.block_cols = np.arange(self.mycol, self.nb, p.pc, dtype=np.int64)
        offsets = np.arange(p.b, dtype=np.int64)
        self.rows = (self.block_rows[:, None] * p.b + offsets).reshape(-1)
        self.cols = (self.block_cols[:, None] * p.b + offsets).reshape(-1)
        self.shape = (len(self.rows), len(self.cols))

    @property
    def num_blocks(self):
        return len(self.block_rows) * len(self.block_cols)

    def _check_block(self, I, J):
        if not (0 <= I < self.nb and 0 <= J < self.nb):
            raise IndexError('Índice de bloque global fuera de rango.')

    def owner(self, I, J):
        self._check_block(I, J)
        return I % self.params.pr, J % self.params.pc

    def global_block_to_local(self, I, J):
        self._check_block(I, J)
        if self.owner(I, J) != (self.myrow, self.mycol):
            return None
        return I // self.params.pr, J // self.params.pc

    def local_block_to_global(self, i, j):
        if not (0 <= i < len(self.block_rows) and 0 <= j < len(self.block_cols)):
            raise IndexError('Índice de bloque local fuera de rango.')
        return int(self.block_rows[i]), int(self.block_cols[j])

    def block_slices(self, I, J):
        local = self.global_block_to_local(I, J)
        if local is None:
            raise KeyError(f'El bloque ({I}, {J}) pertenece a otro proceso.')
        i, j = local
        b = self.params.b
        return slice(i*b, (i+1)*b), slice(j*b, (j+1)*b)

    def global_entry_to_local(self, row, col):
        if not (0 <= row < self.params.N and 0 <= col < self.params.N):
            raise IndexError('Índice de entrada global fuera de rango.')
        b = self.params.b
        local = self.global_block_to_local(row // b, col // b)
        if local is None:
            return None
        i, j = local
        return i*b + row % b, j*b + col % b

    def local_entry_to_global(self, row, col):
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            raise IndexError('Índice de entrada local fuera de rango.')
        return int(self.rows[row]), int(self.cols[col])

    def active_slices(self, k):
        """Región rectangular con bloques I,J>k; k=-1 incluye todo.

        Son slices básicos (vista), pero la región puede tener strides que
        impiden usarla directamente como buffer MPI contiguo. No realiza LU.
        """
        if not -1 <= k < self.nb:
            raise IndexError('k debe estar entre -1 y nb-1.')
        r = int(np.searchsorted(self.block_rows, k, side='right')) * self.params.b
        c = int(np.searchsorted(self.block_cols, k, side='right')) * self.params.b
        return slice(r, self.shape[0]), slice(c, self.shape[1])


@dataclass
class LocalMatrix:
    layout: BlockCyclicLayout
    values: np.ndarray

    @classmethod
    def initialize(cls, grid):
        layout = BlockCyclicLayout(grid)
        # La función original recibe SOLO los índices locales globales. No se
        # invoca ninguna rutina LU del serial ni se genera una réplica global.
        values = matrix_entries(layout.rows, layout.cols, layout.params.N)
        return cls(layout, values)


@dataclass(frozen=True)
class StepPanels:
    """Paneles recibidos, en el orden de los ejes locales con bloques > k.

    lower: (filas activas locales, b); upper: (b, columnas activas locales).
    row_slice/col_slice delimitan esa misma región de LocalMatrix.values.
    diagonal contiene LU empaquetado solo en la fila/columna propietarias;
    fuera de esa unión es None. Los buffers son float64 C independientes
    del almacenamiento local, incluso en los emisores. No se aplica Schur.
    """
    k: int
    block_rows: np.ndarray
    block_cols: np.ndarray
    row_slice: slice
    col_slice: slice
    lower: np.ndarray
    upper: np.ndarray
    diagonal: np.ndarray | None


def diagonal_panel_step(grid, local, k, timers=None):
    """Fase colectiva de UN paso k: diagonal, paneles y sus difusiones.

    Precondición: todos los ranks llaman con el mismo k y con el estado
    correcto al inicio de ese paso (Schur anteriores ya aplicados por el
    llamador). Repetir esta función sin Schur NO produce una LU correcta.

    Orden local compatible: diagonal por fila r (root=c), diagonal por
    columna c (root=r), resolver U, resolver L, L por cada fila (root=c),
    U por cada columna (root=r). Solo la unión de fila r y columna c recibe
    diagonal; TODOS participan en las dos difusiones de paneles, aun vacíos.
    No requiere barreras ni colectivas de control entre fases.

    Fallos locales, incluidos ValueError/AssertionError y advertencias de
    singularidad de LAPACK, abortan el comunicador padre con diagnóstico.
    No se intenta una colectiva de recuperación que otros ranks no esperan.
    """
    phase = 'validación local'
    try:
        p, layout, a = grid.params, local.layout, local.values
        if sys.flags.optimize:
            raise RuntimeError('No usar -O: desactiva la comprobación de pivoteo del serial.')
        if not isinstance(k, (int, np.integer)) or not 0 <= k < layout.nb:
            raise ValueError('k debe ser un índice de bloque entre 0 y nb-1.')
        if (layout.params != p or
                (layout.myrow, layout.mycol) != (grid.myrow, grid.mycol)):
            raise ValueError('Layout y grid incompatibles.')
        if (a.shape != layout.shape or a.dtype != np.float64 or
                not a.flags.c_contiguous or not a.flags.writeable):
            raise ValueError('La matriz local debe ser compacta C, float64, escribible y de shape layout.shape.')
        r, c = k % p.pr, k % p.pc
        rs, cs = layout.active_slices(k)
        block_rows = layout.block_rows[rs.start // p.b:]
        block_cols = layout.block_cols[cs.start // p.b:]
        lower = np.empty((rs.stop - rs.start, p.b), dtype=np.float64)
        upper = np.empty((p.b, cs.stop - cs.start), dtype=np.float64)
        diagonal = None
        if grid.myrow == r or grid.mycol == c:
            diagonal = np.empty((p.b, p.b), dtype=np.float64)

        phase = 'factorización diagonal'
        if (grid.myrow, grid.mycol) == (r, c):
            diagonal[:] = a[layout.block_slices(k, k)]
            if not np.isfinite(diagonal).all():
                raise ValueError('Diagonal con NaN/Inf.')
            with warnings.catch_warnings():
                warnings.simplefilter('error', LinAlgWarning)
                with _measure(timers, 'compute_s'):
                    lu_diag_block(diagonal)  # Mantiene assert de pivotes identidad.
            if not np.isfinite(diagonal).all():
                raise FloatingPointError('LU diagonal con NaN/Inf.')
            a[layout.block_slices(k, k)] = diagonal

        phase = 'Bcast diagonal por fila'
        if grid.myrow == r:
            with _measure(timers, 'communication_s'):
                grid.row_comm.Bcast([diagonal, MPI.DOUBLE], root=c)
        phase = 'Bcast diagonal por columna'
        if grid.mycol == c:
            with _measure(timers, 'communication_s'):
                grid.col_comm.Bcast([diagonal, MPI.DOUBLE], root=r)

        phase = 'resolución panel U'
        if grid.myrow == r and upper.size:
            kr = slice((k // p.pr) * p.b, (k // p.pr + 1) * p.b)
            with _measure(timers, 'compute_s'):
                upper[:] = solve_triangular(diagonal, a[kr, cs], lower=True,
                                            unit_diagonal=True, check_finite=False)
            if not np.isfinite(upper).all():
                raise FloatingPointError('Panel U con NaN/Inf.')
            a[kr, cs] = upper

        phase = 'resolución panel L'
        if grid.mycol == c and lower.size:
            kc = slice((k // p.pc) * p.b, (k // p.pc + 1) * p.b)
            with _measure(timers, 'compute_s'):
                lower[:] = solve_triangular(diagonal, a[rs, kc].T, trans='T',
                                            lower=False, check_finite=False).T
            if not np.isfinite(lower).all():
                raise FloatingPointError('Panel L con NaN/Inf.')
            a[rs, kc] = lower

        phase = 'Bcast panel L por fila'
        with _measure(timers, 'communication_s'):
            grid.row_comm.Bcast([lower, MPI.DOUBLE], root=c)
        phase = 'Bcast panel U por columna'
        with _measure(timers, 'communication_s'):
            grid.col_comm.Bcast([upper, MPI.DOUBLE], root=r)
        return StepPanels(k, block_rows, block_cols, rs, cs, lower, upper, diagonal)
    except Exception:
        print(f'Error en diagonal_panel_step: rank={grid.rank} '
              f'coords=({grid.myrow},{grid.mycol}) k={k} fase={phase}; MPI Abort(1)',
              file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        grid.comm.Abort(1)
        raise  # Solo alcanzable si un comunicador de prueba retorna de Abort.


def schur_update(local, panels, timers=None):
    """Actualizar in-place SOLO los bloques i,j > panels.k, sin comunicación.

    Consume StepPanels del paso actual: lower (m,b), upper (b,n) y los
    slices básicos de la región activa (m,n). La vista puede tener strides;
    no se copia ni se exige contigüidad C de esa región. Una sola operación
    @ por región no vacía; con m=0 o n=0 no se multiplica.

    Las excepciones son locales: el llamador colectivo debe abortar el
    trabajo MPI, como hace lu_blocked_distributed, sin intentar recuperarlo
    mediante colectivas que otros ranks podrían no estar esperando.
    """
    layout, a = local.layout, local.values
    if not isinstance(panels.k, (int, np.integer)) or not 0 <= panels.k < layout.nb:
        raise ValueError('k de Schur debe estar entre 0 y nb-1.')
    if (a.shape != layout.shape or a.dtype != np.float64 or not a.flags.writeable):
        raise ValueError('Schur requiere matriz local float64 escribible de shape layout.shape.')
    if (panels.row_slice, panels.col_slice) != layout.active_slices(panels.k):
        raise ValueError('Slices de Schur incompatibles con la región i,j > k.')
    active = a[panels.row_slice, panels.col_slice]
    m, n = active.shape
    b = layout.params.b
    if (panels.lower.shape != (m, b) or panels.upper.shape != (b, n) or
            panels.lower.dtype != np.float64 or panels.upper.dtype != np.float64):
        raise ValueError('Schur requiere lower (m,b), upper (b,n) float64 y active (m,n).')
    if not m or not n:
        return
    # NumPy escribe sobre la vista original aun cuando no sea C contigua.
    # El único temporal grande es el producto (m,n), sin copia previa de active.
    with np.errstate(over='raise', invalid='raise'):
        with _measure(timers, 'compute_s'):
            active -= panels.lower @ panels.upper
    if not np.isfinite(active).all():
        raise FloatingPointError('Actualización de Schur con NaN/Inf.')


def lu_blocked_distributed(grid, local, timers=None):
    """LU completa in-place; todos los ranks recorren k=0,...,nb-1.

    Requiere matriz original local, grid/layout compatibles y parámetros
    colectivamente validados. Cada rank completa su Schur antes de llamar
    al paso siguiente. Las Bcast bloqueantes entregan los paneles necesarios;
    no se añaden barreras ni otras comunicaciones. Conserva local.values.
    """
    k, phase = None, 'validación local'
    try:
        if local.layout.params != grid.params:
            raise ValueError('Layout y grid incompatibles para LU completa.')
        for k in range(grid.params.N // grid.params.b):
            phase = 'diagonal/paneles'
            panels = (diagonal_panel_step(grid, local, k) if timers is None else
                      diagonal_panel_step(grid, local, k, timers=timers))
            phase = 'actualización de Schur'
            if timers is None:
                schur_update(local, panels)
            else:
                schur_update(local, panels, timers=timers)
    except Exception:
        print(f'Error en lu_blocked_distributed: rank={grid.rank} '
              f'coords=({grid.myrow},{grid.mycol}) k={k} fase={phase}; MPI Abort(1)',
              file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        grid.comm.Abort(1)
        raise


def measure_factorization(grid, local):
    """Único intervalo total: Barrier de preparación fuera; sin Barrier final."""
    timers = RankTimers()
    grid.comm.Barrier()
    start = time.perf_counter()
    lu_blocked_distributed(grid, local, timers=timers)
    timers.total_s = time.perf_counter() - start
    return timers


def rank_environment():
    """Consulta previa a Barrier; no impone hilos ni requiere threadpoolctl."""
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    return dict(hostname=MPI.Get_processor_name(),
                thread_env={key: os.environ.get(key) for key in THREAD_VARIABLES},
                blas_pools=pools)


def execution_metadata():
    """Identidad de root fuera de LU; hash distingue cambios sin commit."""
    directory = Path(__file__).resolve().parent
    def git(*args):
        try:
            result = subprocess.run(['git', '-C', str(directory), *args],
                                    capture_output=True, text=True, timeout=2)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    status = git('status', '--porcelain=v1')
    return dict(run_id=str(uuid.uuid4()),
                started_at=datetime.now(timezone.utc).isoformat(),
                code=dict(git_commit=git('rev-parse', 'HEAD'),
                          git_dirty=None if status is None else bool(status),
                          sha256={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                  for name in ('lu_distributed.py', 'lu_serial.py')}),
                environment=dict(python=sys.version, executable=sys.executable,
                                 numpy=np.__version__, scipy=scipy.__version__,
                                 mpi4py=mpi4py.__version__, mpi_library=MPI.Get_library_version(),
                                 platform=platform.platform(),
                                 thread_policy='external; requested values recorded per rank',
                                 clock='time.perf_counter',
                                 clock_resolution_s=time.get_clock_info('perf_counter').resolution,
                                 time_tolerance_s=TIME_TOLERANCE_S))


def validate_rank_times(params, rank_times):
    """Validación en root después de gather; conserva floats y ordena ranks."""
    p = params.pr * params.pc
    if len(rank_times) != p:
        raise ValueError('Métricas: falta algún rank.')
    ranks = [r['rank'] for r in rank_times]
    if any(type(r) is not int for r in ranks) or sorted(ranks) != list(range(p)):
        raise ValueError('Métricas: ranks duplicados o fuera de rango.')
    ordered = sorted(rank_times, key=lambda r: r['rank'])
    for record in ordered:
        row, col = divmod(record['rank'], params.pc)
        if (type(record['row']) is not int or type(record['col']) is not int or
                (record['row'], record['col']) != (row, col)):
            raise ValueError('Métricas: coordenadas incompatibles con rank.')
        for field in ('compute_s', 'communication_s', 'total_s'):
            value = record[field]
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Métricas: {field} debe ser finito y no negativo.')
        overhead = record['total_s'] - record['compute_s'] - record['communication_s']
        if overhead < -TIME_TOLERANCE_S:
            raise ValueError('Métricas: cómputo + comunicación exceden total_s.')
        record['unaccounted_s'] = overhead
    return ordered


def collect_rank_times(grid, timers, environment):
    record = dict(rank=grid.rank, row=grid.myrow, col=grid.mycol,
                  compute_s=timers.compute_s, communication_s=timers.communication_s,
                  total_s=timers.total_s, **environment)
    records = grid.comm.gather(record, root=0)
    return validate_rank_times(grid.params, records) if grid.rank == 0 else None


def reconstruct_factors(params, payload):
    """Diagnóstico pequeño exclusivo de root, con ejes derivados de rank."""
    if params.N > CHECK_MAX_N:
        raise ValueError(f'Reconstrucción limitada a N <= {CHECK_MAX_N}.')
    if len(payload) != params.pr * params.pc:
        raise ValueError('Reconstrucción: faltan ranks.')
    factors = np.full((params.N, params.N), np.nan)
    coverage = np.zeros((params.N, params.N), dtype=np.uint8)
    indices = np.arange(params.N)
    for rank, data in enumerate(payload):
        row, col = divmod(rank, params.pc)
        rows = indices[indices // params.b % params.pr == row]
        cols = indices[indices // params.b % params.pc == col]
        if data.shape != (len(rows), len(cols)) or not np.isfinite(data).all():
            raise ValueError('Reconstrucción: shape o finitud local inválida.')
        factors[np.ix_(rows, cols)] = data
        coverage[np.ix_(rows, cols)] += 1
    if not np.all(coverage == 1) or not np.isfinite(factors).all():
        raise ValueError('Reconstrucción: cobertura o finitud inválida.')
    return factors


def check_numerical(params, factors):
    """No modifica factors. Criterios fijados: ambos relativos <= 1e-10."""
    if params.N > CHECK_MAX_N:
        raise ValueError(f'Comprobación limitada a N <= {CHECK_MAX_N}.')
    if factors.shape != (params.N, params.N) or not np.isfinite(factors).all():
        raise ValueError('Comprobación: shape o finitud global inválida.')
    original = matrix_entries(range(params.N), range(params.N), params.N)
    reference = original.copy()
    lu_blocked(reference, params.b)
    if reference.shape != factors.shape or not np.isfinite(reference).all():
        raise ValueError('Comprobación: referencia serial inválida.')
    delta = factors - reference
    norm = float(np.linalg.norm(reference))
    if not norm:
        raise ValueError('Comprobación: norma serial cero.')
    rd = float(residual(original, factors))
    difference = float(np.linalg.norm(delta)) / norm
    maximum = float(np.max(np.abs(delta)))
    if not all(math.isfinite(x) for x in (rd, difference, maximum)):
        raise ValueError('Comprobación: métricas numéricas NaN/Inf.')
    passed = rd <= CHECK_TOLERANCE and difference <= CHECK_TOLERANCE
    return dict(status='passed' if passed else 'failed', relative_residual=rd,
                relative_factor_difference=difference, max_absolute_factor_difference=maximum,
                residual_tolerance=CHECK_TOLERANCE, factor_tolerance=CHECK_TOLERANCE,
                max_N=CHECK_MAX_N)


def verify_factorization(grid, local):
    """Gather diagnóstico y bcast de estado, siempre posteriores al total."""
    if not grid.params.check:
        return dict(status='not_run')
    if grid.params.N > CHECK_MAX_N:
        raise ValueError(f'--check requiere N <= {CHECK_MAX_N}.')
    payload = grid.comm.gather(local.values, root=0)
    result = None
    if grid.rank == 0:
        # Excepciones inesperadas siguen la política Abort de main.
        result = check_numerical(grid.params, reconstruct_factors(grid.params, payload))
    return grid.comm.bcast(result, root=0)


def build_output(params, metadata, rank_times, check):
    ordered = validate_rank_times(params, rank_times)
    return dict(schema_version='1.0', **metadata,
                N=params.N, b=params.b, pr=params.pr, pc=params.pc, p=params.pr*params.pc,
                wall_s=max(r['total_s'] for r in ordered), rank_times=ordered, check=check)


def main(argv=None):
    comm = MPI.COMM_WORLD
    try:
        # Solo los errores colectivos iniciales admiten una salida normal 2.
        # Un ValueError posterior es local y debe seguir la rama Abort.
        try:
            params = parse_parameters(comm, argv)
        except ValueError as exc:
            if comm.Get_rank() == 0:
                print(f'Error: {exc}', file=sys.stderr)
            return 2
        if params is None:
            return 0
        grid = ProcessGrid.create(comm, params)
        local = LocalMatrix.initialize(grid)
        environment = rank_environment()
        metadata = execution_metadata() if grid.rank == 0 else None
        timers = measure_factorization(grid, local)
        rank_times = collect_rank_times(grid, timers, environment)
        check = verify_factorization(grid, local)
        if grid.rank == 0:
            result = build_output(params, metadata, rank_times, check)
            print(json.dumps(result, allow_nan=False))
            if check['status'] == 'failed':
                print('Error: --check failed; corrida no válida para rendimiento.',
                      file=sys.stderr, flush=True)
        grid.free()
        return 1 if check['status'] == 'failed' else 0
    except Exception as exc:
        print(f'Error inesperado en rank {comm.Get_rank()}: {exc}; MPI Abort(1)', file=sys.stderr,
              flush=True)
        comm.Abort(1)  # Un fallo local no deja a otros ranks en una colectiva.


if __name__ == '__main__':
    raise SystemExit(main())
