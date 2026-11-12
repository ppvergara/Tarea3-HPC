"""MPI verification for Stage 5. Virtual clocks and faults exist only here."""
import argparse
import builtins
import copy
from contextlib import redirect_stdout
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
from lu_serial import lu_blocked, matrix_entries, residual
from test_panels import compare, require, expected_events
from test_lu_complete import reconstruct


class Clock:
    def __init__(self):
        self.value = 0.0
    def __call__(self):
        return self.value
    def advance(self, seconds):
        self.value += seconds


class BroadcastObserver:
    def __init__(self, real, axis, events, clock=None):
        self.real, self.axis, self.events, self.clock = real, axis, events, clock
    def __getattr__(self, name):
        return getattr(self.real, name)
    def Bcast(self, spec, root=0):
        a, datatype = spec
        self.events.append(dict(axis=self.axis, root=root, shape=list(a.shape),
                                count=a.size, dtype=str(a.dtype), contiguous=a.flags.c_contiguous,
                                double=datatype == MPI.DOUBLE))
        if self.clock:
            self.clock.advance(7)
        return self.real.Bcast(spec, root=root)


class ClockLower(np.ndarray):
    def __new__(cls, a, clock):
        result = np.asarray(a).view(cls)
        result.clock = clock
        return result
    def __array_finalize__(self, source):
        self.clock = getattr(source, 'clock', None)
    def __matmul__(self, other):
        self.clock.advance(5)
        return np.asarray(self) @ other


def observed_lu(grid, local, clock=None):
    events, counts = [], dict(diagonal=0, solves=0, schur=0, overhead_calls=0)
    row, col = grid.row_comm, grid.col_comm
    grid.row_comm = BroadcastObserver(row, 'row_comm', events, clock)
    grid.col_comm = BroadcastObserver(col, 'col_comm', events, clock)
    diag, solve, update = d.lu_diag_block, d.solve_triangular, d.schur_update
    active_slices = local.layout.active_slices
    def diagonal(a):
        counts['diagonal'] += 1
        if clock:
            clock.advance(2)
        return diag(a)
    def triangular(a, rhs, **kwargs):
        counts['solves'] += 1
        if clock:
            clock.advance(3)
        return solve(a, rhs, **kwargs)
    def schur(current, panels, timers=None):
        if panels.lower.shape[0] and panels.upper.shape[1]:
            counts['schur'] += 1
        if clock:
            panels = replace(panels, lower=ClockLower(panels.lower, clock))
        return update(current, panels, timers=timers)
    def slices(k):
        counts['overhead_calls'] += 1
        if clock:
            clock.advance(11)  # Indexing is total-only, outside numerical timers.
        return active_slices(k)
    try:
        with patch.object(d, 'lu_diag_block', diagonal), patch.object(d, 'solve_triangular', triangular), patch.object(d, 'schur_update', schur), patch.object(local.layout, 'active_slices', slices):
            if clock:
                with patch.object(d.time, 'perf_counter', clock):
                    timers = d.measure_factorization(grid, local)
            else:
                timers = d.measure_factorization(grid, local)
    finally:
        grid.row_comm, grid.col_comm = row, col
    expected = []
    for k in range(grid.params.N//grid.params.b):
        for e in expected_events(grid.params, grid.rank, k):
            expected.append(dict(axis=e['communicator'], root=e['root'], shape=e['shape'],
                                 count=e['count'], dtype='float64', contiguous=True, double=True))
    require(events == expected, 'Bcast order/communicator/root/shape/count changed')
    if clock:
        require(timers.compute_s == 2*counts['diagonal'] + 3*counts['solves'] + 5*counts['schur'], 'Numerical time missing or double counted')
        require(timers.communication_s == 7*len(events), 'Bcast time missing or double counted, including zero count')
        require(timers.total_s == timers.compute_s + timers.communication_s + 11*counts['overhead_calls'], 'Total-only overhead or nested timing incorrect')
    return timers, dict(events=events, counts=counts, virtual=clock is not None,
                        compute_s=timers.compute_s, communication_s=timers.communication_s,
                        total_s=timers.total_s)


def normal(p):
    grid = d.ProcessGrid.create(MPI.COMM_WORLD, p)
    baseline = d.LocalMatrix.initialize(grid)
    # Uninstrumented observer uses the exact same Bcasts without any timer.
    events = []
    row, col = grid.row_comm, grid.col_comm
    grid.row_comm = BroadcastObserver(row, 'row_comm', events)
    grid.col_comm = BroadcastObserver(col, 'col_comm', events)
    with patch.object(d.time, 'perf_counter', side_effect=AssertionError('Clock read on uninstrumented path')):
        d.lu_blocked_distributed(grid, baseline)
    grid.row_comm, grid.col_comm = row, col
    measured = d.LocalMatrix.initialize(grid)
    timers, real = observed_lu(grid, measured)
    require(real['events'] == events, 'Instrumented/uninstrumented communication trace differs')
    metrics = compare(measured.values, baseline.values)
    before_check = measured.values.copy()
    check = d.verify_factorization(replace(grid, params=replace(p, check=True)), measured)
    require(check['status'] == 'passed', 'Real numerical check failed')
    np.testing.assert_array_equal(before_check, measured.values)
    virtual = d.LocalMatrix.initialize(grid)
    _, scope = observed_lu(grid, virtual, Clock())
    compare(virtual.values, baseline.values)
    times = d.collect_rank_times(grid, timers, d.rank_environment())
    factors = grid.comm.gather(measured.values, root=0)
    ranks = grid.comm.gather(dict(rank=grid.rank, comparison=metrics, check_preserved_exact=True,
                                 real=real, clock_scope=scope), root=0)
    if grid.rank == 0:
        rebuilt = reconstruct(p, factors)  # Independent test reconstruction.
        original = matrix_entries(range(p.N), range(p.N), p.N)
        serial = original.copy(); lu_blocked(serial, p.b)
        numerical = compare(rebuilt, serial)
        rd = float(residual(original, rebuilt))
        require(rd <= 1e-10, 'Independent residual')
        result = dict(passed=True, N=p.N, b=p.b, pr=p.pr, pc=p.pc,
                      rank_times=times, ranks=ranks, check=check, numerical=numerical,
                      independent_residual=rd, factors=rebuilt.tolist(), serial_factors=serial.tolist())
        print(json.dumps(result, allow_nan=False))
    grid.free()
    return 0


def main_scope(p):
    """Run the real main with additive delays on excluded phases, no sleeps."""
    clock, trace, snapshots = Clock(), [], {}
    initialize = d.LocalMatrix.initialize.__func__
    create = d.ProcessGrid.create.__func__
    collect, verify, metadata = d.collect_rank_times, d.verify_factorization, d.execution_metadata
    measure, free = d.measure_factorization, d.ProcessGrid.free
    dumps = d.json.dumps
    class Comm:
        def __init__(self, real): self.real = real
        def __getattr__(self, name): return getattr(self.real, name)
        def Barrier(self):
            trace.append('barrier'); clock.advance(2000)
            return self.real.Barrier()
    def make(cls, comm, params):
        grid = create(cls, comm, params); grid.comm = Comm(comm)
        return grid
    def init(cls, grid):
        trace.append('initialize'); clock.advance(1000)
        return initialize(cls, grid)
    def env():
        clock.advance(3000)
        return dict(hostname='virtual clock test', thread_env={}, blas_pools=None)
    def meta():
        clock.advance(4000)
        return metadata()
    def lu(grid, local):
        # The measured fake LU is an allowed harness for main boundaries.
        original_lu = d.lu_blocked_distributed
        def operation(g, a, timers=None):
            trace.append('LU'); clock.advance(17)
            original_lu(g, a)  # No inner clocks: tested separately by observed_lu.
        with patch.object(d, 'lu_blocked_distributed', operation):
            result = measure(grid, local)
        snapshots['total_s'] = result.total_s
        require(result.total_s == 17, 'Setup/barrier leaked into total')
        return result
    def gathering(grid, timers, environment):
        trace.append('gather_metrics'); clock.advance(5000)
        require(timers.total_s == 17, 'Total changed before gather')
        return collect(grid, timers, environment)
    def checking(grid, local):
        trace.append('check'); clock.advance(6000)
        return verify(grid, local)
    def serializing(*args, **kwargs):
        trace.append('serialize'); clock.advance(9000)
        return dumps(*args, **kwargs)
    def printing(*args, **kwargs):
        trace.append('print'); clock.advance(7000)
        return builtins.print(*args, **kwargs)
    def freeing(grid):
        trace.append('free'); clock.advance(8000)
        return free(grid)
    stream = io.StringIO()
    flags = ['--N',str(p.N),'--b',str(p.b),'--pr',str(p.pr),'--pc',str(p.pc),'--check']
    with patch.object(d.time, 'perf_counter', clock), patch.object(d.ProcessGrid, 'create', classmethod(make)), patch.object(d.LocalMatrix, 'initialize', classmethod(init)), patch.object(d, 'rank_environment', env), patch.object(d, 'execution_metadata', meta), patch.object(d, 'measure_factorization', lu), patch.object(d, 'collect_rank_times', gathering), patch.object(d, 'verify_factorization', checking), patch.object(d.json, 'dumps', serializing), patch.object(d, 'print', printing, create=True), patch.object(d.ProcessGrid, 'free', freeing), redirect_stdout(stream):
        rc = d.main(flags)
    require(rc == 0 and snapshots['total_s'] == 17, 'Main scope failed')
    traces = MPI.COMM_WORLD.gather(trace, root=0)
    if MPI.COMM_WORLD.rank == 0:
        output = json.loads(stream.getvalue())
        require(all(r['total_s'] == 17 for r in output['rank_times']), 'Post-LU activity leaked into totals')
        require(output['wall_s'] == 17 and output['check']['status'] == 'passed', 'Main check/output')
        require(all(t[:5] == ['initialize','barrier','LU','gather_metrics','check'] for t in traces), 'Main phase order')
        print(json.dumps(dict(passed=True, total_s=17, traces=traces,
                              excluded_delays=[1000,2000,3000,4000,5000,6000,7000,8000,9000])))
    return 0


def failure(p, scenario):
    measure, rebuild, check = d.measure_factorization, d.reconstruct_factors, d.check_numerical
    def corrupt(grid, local):
        result = measure(grid, local)
        if scenario == 'invalid_time' and grid.rank == 1:
            result.compute_s = result.total_s + 1
        if scenario == 'corrupt_factors' and grid.rank == 0:
            local.values[0,0] += 1.0
        return result
    def broken_rebuild(params, payload):
        if scenario == 'reconstruct_error':
            payload[0] = payload[0][:-1,:]
        return rebuild(params, payload)
    def broken_check(params, factors):
        if scenario == 'check_error':
            raise ValueError('fallo controlado de comprobación')
        return check(params, factors)
    with patch.object(d, 'measure_factorization', corrupt), patch.object(d, 'reconstruct_factors', broken_rebuild), patch.object(d, 'check_numerical', broken_check):
        return d.main(['--N',str(p.N),'--b',str(p.b),'--pr',str(p.pr),'--pc',str(p.pc),'--check'])


def validators(p):
    """Production validator adversarial cases, independent of JSON consumer."""
    records = [dict(rank=i, row=i//p.pc, col=i%p.pc, compute_s=1.0,
                    communication_s=2.0, total_s=4.0) for i in range(p.pr*p.pc)]
    accepted = d.validate_rank_times(p, copy.deepcopy(records[::-1]))
    require([r['rank'] for r in accepted] == list(range(p.pr*p.pc)), 'Sorting')
    cases = [records[:-1], records+[records[0]]]
    for field, value in [('rank',True),('rank',1),('row',True),('row',999),('col',999),
                         ('compute_s',-1),('communication_s',float('nan')),
                         ('total_s',float('inf')),('total_s','1'),('compute_s',True),
                         ('compute_s',5)]:
        bad = copy.deepcopy(records); bad[0][field] = value; cases.append(bad)
    for bad in cases:
        try: d.validate_rank_times(p, bad)
        except ValueError: pass
        else: raise AssertionError('Production validator accepted invalid times/ranks')
    near = copy.deepcopy(records)
    near[0]['total_s'] = 3.0-5e-10
    require(d.validate_rank_times(p, near)[0]['unaccounted_s'] < 0, 'Resolution tolerance must preserve raw value')
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(dict(passed=True, rejected_cases=len(cases), resolution_tolerance_preserved=True)))
    return 0


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--scenario', default='normal')
    args, remaining = parser.parse_known_args()
    p = d.parse_parameters(MPI.COMM_WORLD, remaining)
    require(p.N <= 256, 'Small test only')
    if args.scenario == 'normal': return normal(p)
    if args.scenario == 'scope': return main_scope(p)
    if args.scenario == 'validators': return validators(p)
    return failure(p, args.scenario)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
