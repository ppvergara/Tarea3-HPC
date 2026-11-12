"""Grid bloque-cíclico y fase diagonal/paneles MPI (Etapa 3, sin Schur).

Ejemplo: mpiexec -n 4 python lu_distributed.py --N 32 --b 8 --pr 2 --pc 2
La CLI solo inicializa; diagonal_panel_step es la interfaz numérica reutilizable.
Hilos BLAS configurados externamente. Las pruebas están en tests/. No hay LU completa.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import sys
import traceback
import warnings

from mpi4py import MPI
import numpy as np
from scipy.linalg import LinAlgWarning, solve_triangular

from lu_serial import lu_diag_block, matrix_entries


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
    if params.check:
        raise ValueError('--check está reservado para verificar LU; '
                         'la Etapa 3 todavía no implementa LU completa. '
                         'Use tests/test_distribution.py para la distribución.')


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
                        help='reservado para LU completa; se rechaza en Etapa 3')
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


def diagonal_panel_step(grid, local, k):
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
                lu_diag_block(diagonal)  # Mantiene assert de pivotes identidad.
            if not np.isfinite(diagonal).all():
                raise FloatingPointError('LU diagonal con NaN/Inf.')
            a[layout.block_slices(k, k)] = diagonal

        phase = 'Bcast diagonal por fila'
        if grid.myrow == r:
            grid.row_comm.Bcast([diagonal, MPI.DOUBLE], root=c)
        phase = 'Bcast diagonal por columna'
        if grid.mycol == c:
            grid.col_comm.Bcast([diagonal, MPI.DOUBLE], root=r)

        phase = 'resolución panel U'
        if grid.myrow == r and upper.size:
            kr = slice((k // p.pr) * p.b, (k // p.pr + 1) * p.b)
            upper[:] = solve_triangular(diagonal, a[kr, cs], lower=True,
                                        unit_diagonal=True, check_finite=False)
            if not np.isfinite(upper).all():
                raise FloatingPointError('Panel U con NaN/Inf.')
            a[kr, cs] = upper

        phase = 'resolución panel L'
        if grid.mycol == c and lower.size:
            kc = slice((k // p.pc) * p.b, (k // p.pc + 1) * p.b)
            lower[:] = solve_triangular(diagonal, a[rs, kc].T, trans='T',
                                        lower=False, check_finite=False).T
            if not np.isfinite(lower).all():
                raise FloatingPointError('Panel L con NaN/Inf.')
            a[rs, kc] = lower

        phase = 'Bcast panel L por fila'
        grid.row_comm.Bcast([lower, MPI.DOUBLE], root=c)
        phase = 'Bcast panel U por columna'
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
        if comm.Get_rank() == 0:
            print(f'Etapa 3: inicialización local N={params.N} b={params.b} '
                  f'grid={params.pr}x{params.pc}; LU no implementada.')
        # Mantener el almacenamiento disponible para las siguientes etapas.
        del local
        grid.free()
        return 0
    except Exception as exc:
        print(f'Error inesperado en rank {comm.Get_rank()}: {exc}; MPI Abort(1)', file=sys.stderr,
              flush=True)
        comm.Abort(1)  # Un fallo local no deja a otros ranks en una colectiva.


if __name__ == '__main__':
    raise SystemExit(main())
