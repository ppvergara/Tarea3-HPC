"""Etapa 2: grid MPI e inicialización bloque-cíclica 2D. Aún no implementa LU.

Ejemplo: mpiexec -n 4 python lu_distributed.py --N 32 --b 8 --pr 2 --pc 2
Hilos BLAS configurados externamente. Las pruebas están en tests/.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import sys

from mpi4py import MPI
import numpy as np

from lu_serial import matrix_entries


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
                         'la Etapa 2 todavía no implementa LU. '
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
                        help='reservado para LU futura; se rechaza en Etapa 2')
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


def main(argv=None):
    comm = MPI.COMM_WORLD
    try:
        params = parse_parameters(comm, argv)
        if params is None:
            return 0
        grid = ProcessGrid.create(comm, params)
        local = LocalMatrix.initialize(grid)
        if comm.Get_rank() == 0:
            print(f'Etapa 2: inicialización local N={params.N} b={params.b} '
                  f'grid={params.pr}x{params.pc}; LU no implementada.')
        # Mantener el almacenamiento disponible para las siguientes etapas.
        del local
        grid.free()
        return 0
    except ValueError as exc:
        if comm.Get_rank() == 0:
            print(f'Error: {exc}', file=sys.stderr)
        return 2
    except Exception as exc:
        print(f'Error inesperado en rank {comm.Get_rank()}: {exc}', file=sys.stderr,
              flush=True)
        comm.Abort(1)  # Un fallo local no deja a otros ranks en una colectiva.


if __name__ == '__main__':
    raise SystemExit(main())
