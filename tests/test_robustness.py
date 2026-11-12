"""Etapa 6: main real, reconstrucción independiente pequeña y fallos solo de test.

No es un benchmark. Los observadores no añaden colectivas dentro de LU.
El JSON productivo capturado se conserva literalmente, separado del reporte.
"""
import argparse
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpi4py import MPI
import numpy as np
import lu_distributed as d
import lu_serial as serial
from test_panels import compare, expected_events, require
from test_lu_complete import CountedLower


def axes(p, rank):
    row, col = divmod(rank, p.pc)
    return (np.array([i for i in range(p.N) if i//p.b % p.pr == row], dtype=int),
            np.array([j for j in range(p.N) if j//p.b % p.pc == col], dtype=int))


def reconstruct(p, payload, field):
    """No usa reconstruct_factors ni índices de BlockCyclicLayout."""
    require(len(payload) == p.pr*p.pc, 'Cantidad de ranks de reconstrucción')
    a = np.full((p.N, p.N), np.nan)
    coverage = np.zeros((p.N, p.N), dtype=int)
    blocks = np.zeros((p.N//p.b, p.N//p.b), dtype=int)
    for rank, record in enumerate(payload):
        require(record['rank'] == rank, 'Ranks únicos y ordenados')
        rows, cols = axes(p, rank)
        require(record[field].shape == (len(rows), len(cols)), 'Dimensiones locales independientes')
        require(record['block_rows'] == sorted(set(rows//p.b)), 'Filas bloque reales')
        require(record['block_cols'] == sorted(set(cols//p.b)), 'Columnas bloque reales')
        a[np.ix_(rows, cols)] = record[field]
        coverage[np.ix_(rows, cols)] += 1
        for i in record['block_rows']:
            for j in record['block_cols']:
                require(rank == (i % p.pr)*p.pc+j % p.pc, 'Propietario real')
                blocks[i, j] += 1
    require(np.all(coverage == 1) and np.all(blocks == 1), 'Cobertura exacta entradas/bloques')
    require(np.isfinite(a).all(), 'Reconstrucción finita')
    return a


def normal(p):
    require(p.N <= 256 and not sys.flags.optimize, 'Prueba pequeña sin -O')
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.rank
    state = dict(steps=[], events=[], pivots=[], current_k=None, solves=[])
    measure, step, schur = d.measure_factorization, d.diagonal_panel_step, d.schur_update
    factor, solve = serial.lu_factor, d.solve_triangular

    class ObserveComm:
        def __init__(self, real, axis): self.real, self.axis = real, axis
        def __getattr__(self, name): return getattr(self.real, name)
        def Bcast(self, specification, root=0):
            a, datatype = specification
            require(datatype == MPI.DOUBLE and a.dtype == np.float64 and a.flags.c_contiguous,
                    'Buffer MPI float64 C')
            state['events'].append(dict(axis=self.axis, root=root, shape=list(a.shape), count=a.size))
            return self.real.Bcast(specification, root=root)

    def diagonal(a, **kwargs):
        lu, piv = factor(a, **kwargs)
        require(np.array_equal(piv, np.arange(len(piv))), 'Pivotes identidad observados')
        k = state['current_k']
        require(rank == (k % p.pr)*p.pc+k % p.pc, 'Solo propietario factoriza')
        state['pivots'].append(dict(k=k, rank=rank, pivots=piv.tolist()))
        return lu, piv

    def triangular(a, rhs, **kwargs):
        require(rhs.size > 0, 'Solve vacío inesperado')
        k = state['current_k']
        row, col = divmod(rank, p.pc)
        if kwargs.get('lower'):
            require(row == k % p.pr and kwargs == dict(lower=True, unit_diagonal=True, check_finite=False),
                    'Semántica U')
        else:
            require(col == k % p.pc and kwargs == dict(trans='T', lower=False, check_finite=False),
                    'Semántica L')
        state['solves'].append(dict(k=k, rhs_shape=list(rhs.shape), kwargs=kwargs))
        return solve(a, rhs, **kwargs)

    def panel(g, local, k, timers=None):
        state['current_k'] = k
        state['before'] = local.values.copy()
        return step(g, local, k, timers=timers)

    def update(local, panels, timers=None):
        k = panels.k
        rows, cols = axes(p, rank)
        active = (rows[:, None]//p.b > k) & (cols[None, :]//p.b > k)
        finalized = (rows[:, None]//p.b < k) | (cols[None, :]//p.b < k)
        np.testing.assert_array_equal(local.values[active], state['before'][active])
        np.testing.assert_array_equal(local.values[finalized], state['before'][finalized])
        after_panels = local.values.copy()
        view = local.values[panels.row_slice, panels.col_slice]
        calls = []
        schur(local, replace(panels, lower=CountedLower(panels.lower, calls)), timers=timers)
        require(len(calls) == int(bool(view.size)), 'Una GEMM no vacía; ninguna vacía')
        np.testing.assert_array_equal(local.values[~active], after_panels[~active])
        np.testing.assert_array_equal(local.values[finalized], state['before'][finalized])
        state['steps'].append(dict(k=k, lower_shape=list(panels.lower.shape),
                                   upper_shape=list(panels.upper.shape), active_shape=list(view.shape),
                                   active_c_contiguous=bool(view.flags.c_contiguous),
                                   matmul_calls=len(calls), finalized_preserved=True))

    def measured(g, local):
        rows, cols = axes(p, rank)
        np.testing.assert_array_equal(local.layout.rows, rows)
        np.testing.assert_array_equal(local.layout.cols, cols)
        np.testing.assert_array_equal(local.values, serial.matrix_entries(rows, cols, p.N))
        state.update(input=local.values.copy(), local=local, array=local.values,
                     block_rows=local.layout.block_rows.tolist(), block_cols=local.layout.block_cols.tolist())
        row, col = g.row_comm, g.col_comm
        g.row_comm, g.col_comm = ObserveComm(row, 'row_comm'), ObserveComm(col, 'col_comm')
        try:
            with patch.object(serial, 'lu_factor', diagonal):
                timers = measure(g, local)
        finally:
            g.row_comm, g.col_comm = row, col
        state['factors'] = local.values.copy()
        if not local.values.size:
            require(timers.compute_s == 0, 'Rank sin bloques inventa cómputo')
        return timers

    flags = ['--N',str(p.N),'--b',str(p.b),'--pr',str(p.pr),'--pc',str(p.pc),'--check']
    stream = io.StringIO()
    with patch.object(d, 'measure_factorization', measured), patch.object(d, 'diagonal_panel_step', panel), \
         patch.object(d, 'schur_update', update), patch.object(d, 'solve_triangular', triangular), \
         redirect_stdout(stream):
        rc = d.main(flags)
    require(rc == 0, 'main real falló')
    require(state['local'].values is state['array'], 'Almacenamiento in-place')
    np.testing.assert_array_equal(state['local'].values, state['factors'])
    expected = [dict(axis=e['communicator'], root=e['root'], shape=e['shape'], count=e['count'])
                for k in range(p.N//p.b) for e in expected_events(p, rank, k)]
    require(state['events'] == expected, 'Orden/raíces/formas/count Bcast')
    require(len(state['steps']) == p.N//p.b, 'Todos los pasos')
    packet = {k:state[k] for k in ('input','factors','steps','events','pivots','solves','block_rows','block_cols')}
    packet.update(rank=rank, raw_stdout=stream.getvalue(), shape=list(state['factors'].shape))
    payload = comm.gather(packet, root=0)
    if rank == 0:
        require(payload[0]['raw_stdout'].count('\n') == 1, 'Un solo JSON en root')
        require(all(not r['raw_stdout'] for r in payload[1:]), 'Solo root imprime JSON')
        output = json.loads(payload[0]['raw_stdout'])
        original = serial.matrix_entries(range(p.N), range(p.N), p.N)
        rebuilt_input = reconstruct(p, payload, 'input')
        np.testing.assert_array_equal(rebuilt_input, original)
        factors = reconstruct(p, payload, 'factors')
        reference = original.copy()
        serial.lu_blocked(reference, p.b)
        metrics = compare(factors, reference)
        # Fórmula explícita e independiente del cálculo del check productivo.
        lower, upper = np.tril(factors, -1)+np.eye(p.N), np.triu(factors)
        rd = float(np.linalg.norm(original-lower@upper, 'fro')/np.linalg.norm(original, 'fro'))
        rs = float(serial.residual(original, reference))
        require(np.isfinite([rd,rs]).all() and max(rd,rs) <= 1e-10, 'Residuos <=1e-10')
        check = output['check']
        require(check['status'] == 'passed', 'Check real')
        for value, expected_value in [(check['relative_residual'],rd),
                                      (check['relative_factor_difference'],metrics['relative_frobenius']),
                                      (check['max_absolute_factor_difference'],metrics['max_absolute'])]:
            require(np.isclose(value, expected_value, rtol=1e-12, atol=1e-25), 'Métrica check recalculada')
        pivots = sorted((e for r in payload for e in r['pivots']), key=lambda e:e['k'])
        require([e['k'] for e in pivots] == list(range(p.N//p.b)), 'Diagonal única cada k')
        for record, timing in zip(payload, output['rank_times']):
            require(timing['rank'] == record['rank'], 'Todos los ranks en rank_times')
            if not np.prod(record['shape']): require(timing['compute_s'] == 0, 'Cómputo vacío cero')
        print(json.dumps(dict(kind='etapa_06_robustness', passed=True, N=p.N,b=p.b,pr=p.pr,pc=p.pc,
                              metrics=dict(**metrics, distributed_residual=rd, serial_residual=rs,
                                           shape_matches=factors.shape == reference.shape, finite=True),
                              input=original.tolist(), factors=factors.tolist(), serial_factors=reference.tolist(),
                              input_exact=True, coverage_exact=True, pivots=pivots,
                              output=output, raw_stdout=payload[0]['raw_stdout'],
                              ranks=[{k:v for k,v in r.items() if k not in ('input','factors','raw_stdout')}
                                     for r in payload]), allow_nan=False))
    return 0


def failure(p, scenario):
    """Cada mutación vive en este proceso de test; main decide cómo terminar MPI."""
    rank, current = MPI.COMM_WORLD.rank, dict(k=None)
    step, solve, diag = d.diagonal_panel_step, d.solve_triangular, d.lu_diag_block
    schur, measure, rebuild, validate = d.schur_update, d.measure_factorization, d.reconstruct_factors, d.validate_rank_times
    def panel(g, local, k, timers=None):
        current['k'] = k
        if k == 1 and rank == 3 and scenario in ('late_pivot','late_singular'):
            a = local.values[local.layout.block_slices(k,k)]
            a[:] = np.eye(p.b) if scenario == 'late_pivot' else 0
            if scenario == 'late_pivot': a[0,0],a[0,1],a[1,0] = 0,1,1
        if k == 2 and rank == 0 and scenario == 'late_diagonal_nan':
            local.values[local.layout.block_slices(k,k)][0,0] = np.nan
        return step(g, local, k, timers=timers)
    def triangular(a, rhs, **kwargs):
        out = solve(a, rhs, **kwargs)
        if current['k'] == 2:
            if scenario == 'late_upper_inf' and rank == 1 and kwargs.get('lower'):
                out.flat[0] = np.inf
            if scenario == 'late_lower_nan' and rank == 2 and kwargs.get('trans') == 'T':
                out.flat[0] = np.nan
        return out
    def diagonal(a):
        diag(a)
        if scenario == 'late_diagonal_inf_output' and rank == 3 and current['k'] == 1:
            a[0,0] = np.inf
    def update(local, panels, timers=None):
        if scenario == 'late_schur_overflow' and rank == 3 and panels.k == 1:
            panels = replace(panels, lower=np.full_like(panels.lower,1e308), upper=np.full_like(panels.upper,1e308))
        return schur(local, panels, timers=timers)
    def measured(g, local):
        out = measure(g, local)
        values = dict(time_nan=np.nan, time_inf=np.inf, time_negative=-1.0, time_bool=True)
        if rank == 1 and scenario in values: out.compute_s = values[scenario]
        if rank == 0 and scenario == 'check_failed': local.values[0,0] += 1
        return out
    def reconstruct(params, payload):
        if scenario == 'reconstruct_missing': payload = payload[:-1]
        if scenario == 'reconstruct_nan': payload[0][0,0] = np.nan
        return rebuild(params, payload)
    def records(params, payload):
        if scenario == 'metrics_missing': payload = payload[:-1]
        if scenario == 'metrics_duplicate':
            payload = [dict(r) for r in payload]; payload[-1]['rank'] = 0
        return validate(params, payload)
    flags = ['--N',str(p.N),'--b',str(p.b),'--pr',str(p.pr),'--pc',str(p.pc),'--check']
    with ExitStack() as stack:
        for name, replacement in [('diagonal_panel_step',panel),('solve_triangular',triangular),
                                  ('lu_diag_block',diagonal),('schur_update',update),
                                  ('measure_factorization',measured),('reconstruct_factors',reconstruct),
                                  ('validate_rank_times',records)]:
            stack.enter_context(patch.object(d, name, replacement))
        if scenario == 'bound_guard':
            flags[1] = '272'
            stack.enter_context(patch.object(d.ProcessGrid, 'create', side_effect=AssertionError('Se creó grid antes del rechazo')))
        return d.main(flags)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--scenario', default='normal')
    args, flags = parser.parse_known_args()
    p = d.parse_parameters(MPI.COMM_WORLD, flags)
    require(p is not None and p.N <= 256, 'Worker solo pequeño')
    return normal(p) if args.scenario == 'normal' else failure(p, args.scenario)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
