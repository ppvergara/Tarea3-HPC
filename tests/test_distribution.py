"""Diagnóstico MPI de Etapa 2 (sin LU), para matrices pequeñas.

mpiexec -n 6 python tests/test_distribution.py --N 48 --b 8 --pr 2 --pc 3
Cada rank verifica su almacenamiento; solo root reconstruye la matriz global.
"""

import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpi4py import MPI
import numpy as np
import lu_distributed as distributed
from lu_serial import matrix_entries

# Inicialización evalúa la misma expresión sin reducción. Tolerancia fijada
# antes de ejecutar: margen de redondeo float64 para N<=128, no criterio de LU.
ATOL = 1e-14
RTOL = 1e-14
MAX_N = 128


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect(exception, function, *args):
    try:
        function(*args)
    except exception:
        return
    raise AssertionError(f'Falta {exception.__name__}: {function.__name__}{args}')


def check_local(grid, local, calls, row_members, col_members):
    p, a, layout = grid.params, local.values, local.layout
    rank = grid.comm.Get_rank()
    r, c = rank // p.pc, rank % p.pc
    br = [I for I in range(p.N // p.b) if I % p.pr == r]
    bc = [J for J in range(p.N // p.b) if J % p.pc == c]
    rows = [i for I in br for i in range(I*p.b, (I+1)*p.b)]
    cols = [j for J in bc for j in range(J*p.b, (J+1)*p.b)]
    require((grid.rank, grid.myrow, grid.mycol) == (rank, r, c), 'Coordenadas')
    require(grid.row_comm.Get_rank() == c, 'Rank local fila')
    require(grid.col_comm.Get_rank() == r, 'Rank local columna')
    require(grid.row_comm.Get_size() == p.pc, 'Tamaño fila')
    require(grid.col_comm.Get_size() == p.pr, 'Tamaño columna')
    require(row_members == [r*p.pc+j for j in range(p.pc)], 'Membresía/orden fila')
    require(col_members == [i*p.pc+c for i in range(p.pr)], 'Membresía/orden columna')
    require(layout.block_rows.tolist() == br, 'Filas de bloques/orden')
    require(layout.block_cols.tolist() == bc, 'Columnas de bloques/orden')
    require(layout.rows.tolist() == rows and layout.cols.tolist() == cols, 'Entradas/orden')
    require(a.ndim == 2 and a.shape == (len(rows), len(cols)) == layout.shape, 'Forma')
    require(a.dtype == np.float64 and a.flags.c_contiguous, 'Arreglo compacto float64')
    require(a.nbytes == len(rows)*len(cols)*8, 'Memoria local')
    require(layout.num_blocks == len(br)*len(bc), 'Número de bloques')
    require(len(calls) == 1, 'Generador invocado una sola vez')
    require(calls[0] == (rows, cols, p.N), 'Generación exclusivamente local')
    expected = matrix_entries(rows, cols, p.N)
    np.testing.assert_allclose(a, expected, rtol=RTOL, atol=ATOL)

    blocks = []
    for I in range(layout.nb):
        for J in range(layout.nb):
            owner = (I % p.pr, J % p.pc)
            require(layout.owner(I, J) == owner, 'Regla módulo')
            position = layout.global_block_to_local(I, J)
            if owner != (r, c):
                require(position is None, 'Bloque ajeno')
                expect(KeyError, layout.block_slices, I, J)
                continue
            i, j = br.index(I), bc.index(J)
            require(position == (i, j), 'Offset local de bloque')
            require(layout.local_block_to_global(i, j) == (I, J), 'Inverso bloque')
            view = a[layout.block_slices(I, J)]
            require(view.shape == (p.b, p.b), 'Forma de bloque')
            np.testing.assert_allclose(view, matrix_entries(
                range(I*p.b, (I+1)*p.b), range(J*p.b, (J+1)*p.b), p.N),
                rtol=RTOL, atol=ATOL)
            blocks.append([I, J])

    for i in range(p.N):
        for j in range(p.N):
            position = layout.global_entry_to_local(i, j)
            if (i//p.b % p.pr, j//p.b % p.pc) == (r, c):
                require(position == (rows.index(i), cols.index(j)), 'Offset entrada')
                require(layout.local_entry_to_global(*position) == (i, j), 'Inverso entrada')
            else:
                require(position is None, 'Entrada ajena')

    active = []
    for k in range(-1, layout.nb):
        rs, cs = layout.active_slices(k)
        view = a[rs, cs]
        ri = [i for i, global_row in enumerate(rows) if global_row//p.b > k]
        ci = [j for j, global_col in enumerate(cols) if global_col//p.b > k]
        require(rs.start == len(rows)-len(ri) and cs.start == len(cols)-len(ci),
                'Inicio de región activa')
        require(rs.stop == len(rows) and cs.stop == len(cols), 'Fin de región activa')
        require(rs.step is None and cs.step is None, 'Slices rectangulares básicos')
        require(view.shape == (len(ri), len(ci)), 'Forma activa incluso vacía')
        np.testing.assert_array_equal(view, a[np.ix_(ri, ci)])
        if view.size:
            require(np.shares_memory(view, a), 'Región activa debe ser vista')
        active.append(dict(k=k, shape=list(view.shape), strides=list(view.strides),
                           c_contiguous=bool(view.flags.c_contiguous)))
    np.testing.assert_array_equal(a, expected)  # Las consultas no alteran datos.
    for I, J in [(-1, 0), (layout.nb, 0), (0, layout.nb)]:
        expect(IndexError, layout.global_block_to_local, I, J)
    expect(IndexError, layout.local_block_to_global, len(br), 0)
    expect(IndexError, layout.local_block_to_global, 0, len(bc))
    expect(IndexError, layout.global_entry_to_local, -1, 0)
    expect(IndexError, layout.global_entry_to_local, p.N, 0)
    expect(IndexError, layout.local_entry_to_global, len(rows), 0)
    expect(IndexError, layout.local_entry_to_global, 0, len(cols))
    expect(IndexError, layout.active_slices, -2)
    expect(IndexError, layout.active_slices, layout.nb)

    figure1 = None
    if (p.N//p.b, p.pr, p.pc, r, c) == (6, 2, 3, 0, 1):
        require(br == [0, 2, 4] and bc == [1, 4], 'Ejemplo Figura 1: ejes')
        require(blocks == [[0,1], [0,4], [2,1], [2,4], [4,1], [4,4]],
                'Ejemplo Figura 1: bloques')
        figure1 = True
    return dict(rank=rank, coordinates=[r, c], block_rows=br, block_cols=bc,
                blocks=blocks, shape=list(a.shape), num_blocks=len(blocks),
                row_members=row_members, col_members=col_members,
                empty_rows=not rows, empty_cols=not cols, empty_matrix=a.size == 0,
                generated_shape=[len(rows), len(cols)], active=active,
                figure1_reference_passed=figure1)


def check_reconstruction(p, payload):
    rebuilt = np.full((p.N, p.N), np.nan)
    entries = np.zeros((p.N, p.N), dtype=np.int64)
    coverage = np.zeros((p.N//p.b, p.N//p.b), dtype=np.int64)
    owners = np.full(coverage.shape, -1, dtype=np.int64)
    for metadata, data in payload:
        rows = [i for I in metadata['block_rows'] for i in range(I*p.b, (I+1)*p.b)]
        cols = [j for J in metadata['block_cols'] for j in range(J*p.b, (J+1)*p.b)]
        rebuilt[np.ix_(rows, cols)] = data
        entries[np.ix_(rows, cols)] += 1
        for I, J in metadata['blocks']:
            require(metadata['rank'] == (I % p.pr)*p.pc + J % p.pc, 'Propietario')
            coverage[I, J] += 1
            owners[I, J] = metadata['rank']
    require(np.all(coverage == 1), 'Cobertura de bloques exactamente uno')
    require(np.all(entries == 1), 'Cobertura de entradas exactamente uno')
    require(np.isfinite(rebuilt).all(), 'Reconstrucción finita, sin faltantes')
    reference = matrix_entries(range(p.N), range(p.N), p.N)
    require(rebuilt.shape == reference.shape, 'Dimensiones reconstruidas')
    np.testing.assert_allclose(rebuilt, reference, rtol=RTOL, atol=ATOL)
    return dict(shape=list(rebuilt.shape), shape_matches=True,
                max_absolute_difference=float(np.max(np.abs(rebuilt-reference))),
                blocks_expected=int(coverage.size), blocks_received=int(coverage.sum()),
                coverage_min=int(coverage.min()), coverage_max=int(coverage.max()),
                duplicates=int(np.sum(coverage > 1)), missing=int(np.sum(coverage == 0)),
                entries_coverage_min=int(entries.min()), entries_coverage_max=int(entries.max()),
                block_owners=owners.tolist(), comparison_passed=True)


def main():
    comm = MPI.COMM_WORLD
    p = distributed.parse_parameters(comm)
    if p is None:
        return 0
    if p.N > MAX_N:
        raise ValueError(f'El diagnóstico admite N <= {MAX_N}.')
    grid = distributed.ProcessGrid.create(comm, p)
    calls = []

    def local_generator(rows, cols, N):
        calls.append((list(map(int, rows)), list(map(int, cols)), N))
        return matrix_entries(rows, cols, N)

    original = distributed.matrix_entries
    try:
        distributed.matrix_entries = local_generator
        local = distributed.LocalMatrix.initialize(grid)
    finally:
        distributed.matrix_entries = original
    # Todas las colectivas de comunicador se ejecutan antes de las aserciones.
    row_members = grid.row_comm.allgather(comm.Get_rank())
    col_members = grid.col_comm.allgather(comm.Get_rank())
    error, metadata = None, None
    try:
        metadata = check_local(grid, local, calls, row_members, col_members)
    except Exception:
        error = traceback.format_exc()
    errors = comm.allgather(error)
    result = dict(kind='etapa_02_distribution_test', N=p.N, b=p.b, pr=p.pr,
                  pc=p.pc, processes=comm.Get_size(), atol=ATOL, rtol=RTOL,
                  passed=not any(errors), errors=errors)
    if result['passed']:
        payload = comm.gather((metadata, local.values), root=0)
        if comm.Get_rank() == 0:
            try:
                result['reconstruction'] = check_reconstruction(p, payload)
                result['ranks'] = [m for m, a in payload]
            except Exception:
                result.update(passed=False, reconstruction_error=traceback.format_exc())
        result['passed'] = comm.bcast(result['passed'], root=0)
    grid.free()
    if comm.Get_rank() == 0:
        print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
