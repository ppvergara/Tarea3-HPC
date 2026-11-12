"""Worker MPI de Etapa 3. Referencia global pequeña SOLO en rank 0.

Cada k empieza desde un fixture serial nuevo con Schur previos; la operación
distribuida se detiene antes de Schur. --scenario altera solo este test.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpi4py import MPI
import numpy as np
from scipy.linalg import solve_triangular
import lu_distributed as distributed
from lu_serial import lu_diag_block, matrix_entries

ATOL = RTOL = 1e-10
MAX_N = 128


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(actual, expected):
    require(actual.shape == expected.shape, 'Shape numérico distinto')
    require(np.isfinite(actual).all(), 'Resultado no finito')
    delta = actual - expected
    maximum = float(np.max(np.abs(delta), initial=0))
    norm = float(np.linalg.norm(expected))
    relative = float(np.linalg.norm(delta)) / (norm if norm else 1.0)
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
    require(relative <= RTOL, f'Error relativo Frobenius {relative} > {RTOL}')
    return dict(max_absolute=maximum, relative_frobenius=relative,
                elements=int(actual.size), passed=True)


def serial_phase(a, b, k, schur=False):
    """Transcripción del paso de lu_blocked del profesor, sobre matriz global."""
    s, e = k*b, (k+1)*b
    lu_diag_block(a[s:e, s:e])
    diagonal = a[s:e, s:e]
    if e < len(a):
        a[s:e, e:] = solve_triangular(diagonal, a[s:e, e:], lower=True,
                                     unit_diagonal=True, check_finite=False)
        a[e:, s:e] = solve_triangular(diagonal, a[e:, s:e].T, trans='T',
                                     lower=False, check_finite=False).T
        if schur:
            a[e:, e:] -= a[e:, s:e] @ a[s:e, e:]


def scalar_lu(block):
    """Oráculo adicional Doolittle escalar; no llama SciPy ni producción."""
    result = block.copy()
    for h in range(len(block)):
        require(result[h, h] != 0, 'Pivote escalar cero')
        for i in range(h+1, len(block)):
            result[i, h] /= result[h, h]
            for j in range(h+1, len(block)):
                result[i, j] -= result[i, h] * result[h, j]
    return result


def fixture(p, k):
    state = matrix_entries(range(p.N), range(p.N), p.N)
    for previous in range(k):
        serial_phase(state, p.b, previous, schur=True)
    expected = state.copy()
    serial_phase(expected, p.b, k)  # Sin Schur actual.
    s, e = k*p.b, (k+1)*p.b
    diagonal = expected[s:e, s:e]
    oracle = compare(diagonal, scalar_lu(state[s:e, s:e]))
    lower, upper = np.tril(diagonal, -1)+np.eye(p.b), np.triu(diagonal)
    oracle['factor_equation'] = compare(lower @ upper, state[s:e, s:e])
    packets = []
    for rank in range(p.pr*p.pc):
        r, c = divmod(rank, p.pc)
        # Ejes calculados independientemente de BlockCyclicLayout.
        rows = [i for i in range(p.N) if i//p.b % p.pr == r]
        cols = [j for j in range(p.N) if j//p.b % p.pc == c]
        active_rows = [i for i in rows if i >= e]
        active_cols = [j for j in cols if j >= e]
        packets.append(dict(before=state[np.ix_(rows, cols)],
                            after=expected[np.ix_(rows, cols)], diagonal=diagonal,
                            lower=expected[np.ix_(active_rows, range(s,e))],
                            upper=expected[np.ix_(range(s,e), active_cols)],
                            lower_rhs=state[np.ix_(active_rows, range(s,e))],
                            upper_rhs=state[np.ix_(range(s,e), active_cols)]))
    return packets, expected, oracle


def expected_events(p, rank, k):
    r, c = k % p.pr, k % p.pc
    myrow, mycol = divmod(rank, p.pc)
    br = [i for i in range(k+1, p.N//p.b) if i % p.pr == myrow]
    bc = [j for j in range(k+1, p.N//p.b) if j % p.pc == mycol]
    events = []

    def add(operation, axis, group, root, shape, blocks):
        members = ([group*p.pc+j for j in range(p.pc)] if axis == 'row_comm'
                   else [i*p.pc+group for i in range(p.pr)])
        events.append(dict(operation=operation, communicator=axis, group=group,
                           root=root, shape=list(shape), count=int(np.prod(shape)),
                           blocks=blocks, members=members))
    if myrow == r:
        add('diagonal_row', 'row_comm', myrow, c, (p.b,p.b), [[k,k]])
    if mycol == c:
        add('diagonal_col', 'col_comm', mycol, r, (p.b,p.b), [[k,k]])
    add('panel_l', 'row_comm', myrow, c, (len(br)*p.b,p.b), [[i,k] for i in br])
    add('panel_u', 'col_comm', mycol, r, (p.b,len(bc)*p.b), [[k,j] for j in bc])
    return events


class ObservedComm:
    """Intercepta las llamadas reales a Bcast; no añade colectivas."""
    def __init__(self, real, axis, events, log, packet, k, rank, pc):
        self.real, self.axis, self.events, self.log = real, axis, events, log
        self.packet, self.k, self.rank, self.pc = packet, k, rank, pc

    def __getattr__(self, name):
        return getattr(self.real, name)

    def Bcast(self, specification, root=0):
        require(bool(self.events), 'Colectiva extra')
        expected = self.events.pop(0)
        require(expected['communicator'] == self.axis, 'Orden/comunicador incorrecto')
        require(root == expected['root'], 'Raíz local incorrecta')
        require(isinstance(specification, list) and len(specification) == 2,
                'Se espera buffer explícito [array, MPI.DOUBLE]')
        buffer, datatype = specification
        require(datatype == MPI.DOUBLE, 'Tipo MPI')
        require(buffer.dtype == np.float64 and buffer.flags.c_contiguous, 'Buffer float64 C')
        require(list(buffer.shape) == expected['shape'] and buffer.size == expected['count'],
                'Shape/recuento de mensaje')
        self.real.Bcast(specification, root=root)
        reference = self.packet[{'diagonal_row':'diagonal', 'diagonal_col':'diagonal',
                                 'panel_l':'lower', 'panel_u':'upper'}[expected['operation']]]
        metrics = compare(buffer, reference)
        self.log.append(dict(**expected, k=self.k, rank=self.rank,
                             coordinates=list(divmod(self.rank, self.pc)),
                             sequence=len(self.log), participated=True,
                             dtype=str(buffer.dtype), mpi_type='MPI.DOUBLE',
                             c_contiguous=bool(buffer.flags.c_contiguous),
                             strides=list(buffer.strides), metrics=metrics,
                             received=buffer.tolist()))


def check_local(p, grid, local, k, packet, panels, before, original, calls, events, log):
    rank, a = grid.rank, local.values
    r, c = divmod(rank, p.pc)
    br = [i for i in range(k+1, p.N//p.b) if i % p.pr == r]
    bc = [j for j in range(k+1, p.N//p.b) if j % p.pc == c]
    require(not events, 'Faltan colectivas')
    require(a is original and a.flags.c_contiguous, 'Almacenamiento in-place compacto')
    require(panels.k == k and panels.block_rows.tolist() == br and
            panels.block_cols.tolist() == bc, 'Identidad/orden global de paneles')
    rs, cs = local.layout.active_slices(k)
    require((panels.row_slice, panels.col_slice) == (rs, cs), 'Slices para región activa')
    require(a[rs, cs].shape == (len(br)*p.b, len(bc)*p.b), 'Dimensiones futuras')
    require(calls['diagonal'] == int(rank == (k%p.pr)*p.pc+k%p.pc),
            'Solo el propietario factoriza, exactamente una vez')
    expected_solves = []
    if r == k%p.pr and bc:
        expected_solves.append(dict(kind='U', rhs_shape=[p.b,len(bc)*p.b],
                                    lower=True, unit_diagonal=True, check_finite=False))
    if c == k%p.pc and br:
        expected_solves.append(dict(kind='L', rhs_shape=[p.b,len(br)*p.b],
                                    trans='T', lower=False, check_finite=False))
    require(calls['solves'] == expected_solves, 'Participantes/semántica de solves')
    metrics = dict(local=compare(a, packet['after']),
                   lower=compare(panels.lower, packet['lower']),
                   upper=compare(panels.upper, packet['upper']))
    if r == k%p.pr or c == k%p.pc:
        metrics['diagonal'] = compare(panels.diagonal, packet['diagonal'])
    else:
        require(panels.diagonal is None, 'Diagonal solo en fila/columna propietarias')
    for buffer in (panels.lower, panels.upper):
        require(buffer.dtype == np.float64 and buffer.flags.c_contiguous, 'Panel compacto')
        require(not np.shares_memory(buffer, a), 'Buffer independiente de vista con strides')
    # Ecuaciones y oráculo denso independientes de la resolución triangular MPI.
    lu = packet['diagonal']
    lkk, ukk = np.tril(lu, -1)+np.eye(p.b), np.triu(lu)
    metrics['lower_equation'] = compare(panels.lower @ ukk, packet['lower_rhs'])
    metrics['upper_equation'] = compare(lkk @ panels.upper, packet['upper_rhs'])
    metrics['lower_dense'] = compare(panels.lower,
                                    np.linalg.solve(ukk.T, packet['lower_rhs'].T).T)
    metrics['upper_dense'] = compare(panels.upper,
                                    np.linalg.solve(lkk, packet['upper_rhs']))
    # Prueba explícita de sensibilidad a confundir U^-T con U^-1 en L.
    separation = None
    if p.b > 1 and panels.lower.size:
        wrong = np.linalg.solve(ukk, packet['lower_rhs'].T).T
        separation = float(np.max(np.abs(wrong-packet['lower'])))
        require(separation > 100*ATOL, 'Fixture no detecta transposición incorrecta de L')
    rows, cols = local.layout.rows, local.layout.cols
    I, J = rows[:, None]//p.b, cols[None, :]//p.b
    changed = ((I == k) & (J >= k)) | ((J == k) & (I >= k))
    np.testing.assert_array_equal(a[~changed], before[~changed])
    np.testing.assert_array_equal(a[rs, cs], before[rs, cs])  # No Schur.
    source_views = []
    def describe(name, view):
        source_views.append(dict(name=name, shape=list(view.shape),
                                 strides=list(view.strides),
                                 c_contiguous=bool(view.flags.c_contiguous)))
    if rank == (k%p.pr)*p.pc+k%p.pc:
        describe('diagonal',before[local.layout.block_slices(k,k)])
    if r == k%p.pr:
        describe('U',before[(k//p.pr)*p.b:(k//p.pr+1)*p.b,cs])
    if c == k%p.pc:
        describe('L',before[rs,(k//p.pc)*p.b:(k//p.pc+1)*p.b])
    blocks = []
    for i, bi in enumerate(local.layout.block_rows):
        for j, bj in enumerate(local.layout.block_cols):
            s = (slice(i*p.b,(i+1)*p.b), slice(j*p.b,(j+1)*p.b))
            blocks.append(dict(block=[int(bi),int(bj)], owner=rank,
                               metrics=compare(a[s],packet['after'][s])))
    return dict(rank=rank, coordinates=[r,c], shape=list(a.shape),
                block_rows=local.layout.block_rows.tolist(), block_cols=local.layout.block_cols.tolist(),
                active_rows=br, active_cols=bc, empty_matrix=a.size == 0,
                metrics=metrics, blocks=blocks, calls=calls, communications=log,
                untouched_exact=True, schur_untouched_exact=True, in_place=True,
                source_views=source_views,
                wrong_l_transpose_separation=separation)


def check_global(p, k, payload, reference):
    rebuilt = np.full((p.N,p.N),np.nan)
    coverage = np.zeros((p.N,p.N),dtype=int)
    groups = {}
    for metadata, a in payload:
        rows = [i for I in metadata['block_rows'] for i in range(I*p.b,(I+1)*p.b)]
        cols = [j for J in metadata['block_cols'] for j in range(J*p.b,(J+1)*p.b)]
        rebuilt[np.ix_(rows,cols)] = a
        coverage[np.ix_(rows,cols)] += 1
        for event in metadata['communications']:
            key = (event['operation'],event['communicator'],event['group'])
            groups.setdefault(key,[]).append(event)
    require(np.all(coverage == 1), 'Reconstrucción sin duplicados/faltantes')
    communications = []
    for key, members in groups.items():
        first = members[0]
        require([m['rank'] for m in members] == first['members'], 'Participación colectiva')
        for member in members:
            for field in ('root','shape','count','blocks','members'):
                require(member[field] == first[field], f'Mensaje incompatible: {field}')
        communications.append(dict(operation=key[0], communicator=key[1], group=key[2],
                                   members=first['members'], root=first['root'],
                                   shape=first['shape'], count=first['count'], passed=True))
    require(len(groups) == 2+p.pr+p.pc, 'Número global de colectivas lógicas')
    return dict(metrics=compare(rebuilt,reference), communications=communications,
                reconstructed=rebuilt.tolist(), passed=True)


def failure_scenario(grid, local, scenario):
    """Fallos SOLO del worker. La estrategia de terminación es de producción."""
    rank, b = grid.rank, grid.params.b
    if scenario == 'init_value':
        original = distributed.LocalMatrix.initialize.__func__
        def initialize(cls, current_grid):
            if current_grid.rank == 1:
                raise ValueError('fallo local controlado de inicialización')
            return original(cls, current_grid)
        distributed.LocalMatrix.initialize = classmethod(initialize)
        return distributed.main(['--N','32','--b','8','--pr','2','--pc','2'])
    k = 1
    owner = (k%grid.params.pr)*grid.params.pc+k%grid.params.pc
    if rank == owner and scenario in ('pivot','singular'):
        block = local.values[local.layout.block_slices(k,k)]
        block[:] = np.eye(b) if scenario == 'pivot' else 0
        if scenario == 'pivot':
            block[0,0], block[0,1], block[1,0] = 0, 1, 1
    if rank == 2 and scenario == 'shape':
        local.values = local.values[:-1,:]
    if rank == 2 and scenario in ('scipy_value','scipy_linalg'):
        def broken_solve(a, rhs, **kwargs):
            if scenario == 'scipy_value':
                return solve_triangular(a, np.ones((b+1,1)))
            return solve_triangular(np.zeros((b,b)), np.ones((b,1)))
        distributed.solve_triangular = broken_solve
    distributed.diagonal_panel_step(grid, local, -1 if rank == 2 and scenario == 'invalid_k' else k)
    raise AssertionError(f'El fallo {scenario} no terminó por Abort')


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--ks', default='0')
    extra.add_argument('--scenario', default='normal')
    args, remaining = extra.parse_known_args()
    comm = MPI.COMM_WORLD
    p = distributed.parse_parameters(comm, remaining)
    require(p is not None and p.N <= MAX_N, f'Worker limitado a N <= {MAX_N}')
    grid = distributed.ProcessGrid.create(comm, p)
    local = distributed.LocalMatrix.initialize(grid)
    if args.scenario not in ('normal','wrong_l'):
        return failure_scenario(grid, local, args.scenario)
    ks = [int(k) for k in args.ks.split(',')]
    require(all(0 <= k < p.N//p.b for k in ks), 'k del fixture inválido')
    result = dict(kind='etapa_03_panels_test', N=p.N, b=p.b, pr=p.pr, pc=p.pc,
                  ks=ks, scenario=args.scenario, atol=ATOL, rtol=RTOL,
                  relative_criterion='Frobenius <= 1e-10; denominador 1 si referencia cero',
                  passed=True, steps=[])
    for k in ks:
        packets, expected, oracle = fixture(p,k) if grid.rank == 0 else (None,None,None)
        packet = comm.scatter(packets, root=0)
        if k == 0:
            # Generación original local, sin usar la matriz del fixture como entrada.
            local.values[:] = matrix_entries(local.layout.rows,local.layout.cols,p.N)
            np.testing.assert_array_equal(local.values,packet['before'])
        else:
            local.values[:] = packet['before']
        before, original_array = local.values.copy(), local.values
        events, log = expected_events(p,grid.rank,k), []
        row, col = grid.row_comm, grid.col_comm
        grid.row_comm = ObservedComm(row,'row_comm',events,log,packet,k,grid.rank,p.pc)
        grid.col_comm = ObservedComm(col,'col_comm',events,log,packet,k,grid.rank,p.pc)
        original_diag, original_solve = distributed.lu_diag_block, distributed.solve_triangular
        calls = dict(diagonal=0,solves=[])
        def observed_diag(a):
            calls['diagonal'] += 1
            return original_diag(a)
        def observed_solve(a, rhs, **kwargs):
            calls['solves'].append(dict(kind='U' if kwargs.get('lower') else 'L',
                                        rhs_shape=list(rhs.shape), **kwargs))
            if args.scenario == 'wrong_l' and kwargs.get('trans') == 'T':
                kwargs['trans'] = 'N'  # Mutación deliberada, solo en test.
            return original_solve(a,rhs,**kwargs)
        distributed.lu_diag_block, distributed.solve_triangular = observed_diag, observed_solve
        try:
            panels = distributed.diagonal_panel_step(grid,local,k)
        finally:
            grid.row_comm, grid.col_comm = row, col
            distributed.lu_diag_block, distributed.solve_triangular = original_diag, original_solve
        error, metadata = None, None
        try:
            metadata = check_local(p,grid,local,k,packet,panels,before,original_array,calls,events,log)
        except Exception:
            error = traceback.format_exc()
        errors = comm.allgather(error)
        step = dict(k=k, owner=(k%p.pr)*p.pc+k%p.pc, passed=not any(errors), errors=errors)
        if step['passed']:
            payload = comm.gather((metadata,local.values),root=0)
            if grid.rank == 0:
                try:
                    step.update(global_check=check_global(p,k,payload,expected),
                                scalar_diagonal_oracle=oracle, ranks=[m for m,a in payload])
                except Exception:
                    step.update(passed=False, global_error=traceback.format_exc())
        step['passed'] = comm.bcast(step['passed'],root=0)
        result['steps'].append(step)
        result['passed'] &= step['passed']
        if not step['passed']:
            break
    grid.free()
    if grid.rank == 0:
        print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
