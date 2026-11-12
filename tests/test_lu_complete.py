"""Correctitud MPI de Etapa 4, N<=128; matrices globales SOLO en rank 0.

Primero ejecuta LU productiva sin observadores. Una segunda ejecución observa
cada paso sin añadir colectivas entre iteraciones; guarda estados para cotejar
con el serial global y cuenta @ sobre lower. Los fallos son exclusivos del test.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpi4py import MPI
import numpy as np
import lu_distributed as distributed
from lu_serial import lu_blocked, matrix_entries, residual
from test_panels import compare, require, serial_phase

ATOL = RTOL = 1e-10
MAX_N = 128


def axes(p, rank):
    r, c = divmod(rank, p.pc)
    return (np.array([i for i in range(p.N) if i//p.b % p.pr == r], dtype=int),
            np.array([j for j in range(p.N) if j//p.b % p.pc == c], dtype=int))


def reconstruct(p, payload):
    """Ejes independientes del layout; cobertura exactamente uno."""
    rebuilt = np.full((p.N, p.N), np.nan)
    coverage = np.zeros((p.N, p.N), dtype=int)
    for rank, data in enumerate(payload):
        rows, cols = axes(p, rank)
        require(data.shape == (len(rows), len(cols)), 'Shape reconstrucción')
        rebuilt[np.ix_(rows, cols)] = data
        coverage[np.ix_(rows, cols)] += 1
    require(np.all(coverage == 1), 'Cobertura sin duplicados ni faltantes')
    require(np.isfinite(rebuilt).all(), 'Factores globales NaN/Inf')
    return rebuilt


def final_metrics(original, factors, reference):
    metrics = compare(factors, reference)
    rd, rs = float(residual(original, factors)), float(residual(original, reference))
    require(np.isfinite([rd, rs]).all() and max(rd, rs) <= RTOL, 'Residuos LU')
    return dict(**metrics, distributed_residual=rd, serial_residual=rs,
                shape_matches=factors.shape == reference.shape, finite=True,
                pivoting='identidad exigida por assert en cada diagonal; sin -O')


class CountedLower(np.ndarray):
    """Cuenta las operaciones @ reales; delega el producto a ndarray normal."""
    def __new__(cls, array, calls):
        result = np.asarray(array).view(cls)
        result.calls = calls
        return result

    def __array_finalize__(self, source):
        self.calls = getattr(source, 'calls', None)

    def __matmul__(self, upper):
        product = np.asarray(self) @ upper
        self.calls.append(dict(lower_shape=list(self.shape), upper_shape=list(upper.shape),
                               product_shape=list(product.shape)))
        return product


def check_invariants(local, panels, update):
    """Entradas inválidas deben fallar ANTES de modificar el arreglo."""
    before = local.values.copy()
    variants = [replace(panels, k=-1), replace(panels, row_slice=slice(0, 0)),
                replace(panels, lower=panels.lower[:, :-1]),
                replace(panels, upper=panels.upper[:-1, :]),
                replace(panels, lower=panels.lower.astype(np.float32))]
    for bad in variants:
        try:
            update(local, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Schur aceptó invariantes inválidas')
        np.testing.assert_array_equal(local.values, before)
    return len(variants)


def observe_sequence(grid, local):
    p, comm = grid.params, grid.comm
    rows, cols = axes(p, grid.rank)
    np.testing.assert_array_equal(rows, local.layout.rows)
    np.testing.assert_array_equal(cols, local.layout.cols)
    array, address, strides = local.values, local.values.ctypes.data, local.values.strides
    original_step, original_update = distributed.diagonal_panel_step, distributed.schur_update
    records = []
    pending = {}

    def step(current_grid, current_local, k):
        require(k == len(records), 'k consecutivos e idénticos al bucle esperado')
        pending['before'] = current_local.values.copy()
        return original_step(current_grid, current_local, k)

    def update(current_local, panels):
        k = panels.k
        before = pending.pop('before')
        after_panels = current_local.values.copy()
        active = (rows[:, None]//p.b > k) & (cols[None, :]//p.b > k)
        finalized = (rows[:, None]//p.b < k) | (cols[None, :]//p.b < k)
        np.testing.assert_array_equal(after_panels[active], before[active])
        np.testing.assert_array_equal(after_panels[finalized], before[finalized])
        lower, upper = panels.lower.copy(), panels.upper.copy()
        calls = []
        original_update(current_local, replace(panels, lower=CountedLower(panels.lower, calls)))
        a = current_local.values
        require(a is array and a.ctypes.data == address and a.strides == strides,
                'Identidad/buffer/organización local in-place')
        require(a.shape == local.layout.shape and np.isfinite(a).all(), 'Shape y finitud local')
        require(len(calls) == int(np.any(active)), 'Una @ por región activa no vacía; cero si vacía')
        np.testing.assert_array_equal(a[~active], after_panels[~active])
        np.testing.assert_array_equal(a[finalized], before[finalized])
        np.testing.assert_array_equal(panels.lower, lower)
        np.testing.assert_array_equal(panels.upper, upper)
        view = a[panels.row_slice, panels.col_slice]
        m, n = sum(rows//p.b > k), sum(cols//p.b > k)
        require(view.shape == (m, n) and lower.shape == (m, p.b) and upper.shape == (p.b, n),
                'Formas de región y paneles')
        if view.size:
            require(np.shares_memory(view, a), 'Vista activa comparte almacenamiento')
            require(calls[0]['product_shape'] == [int(m), int(n)], 'Forma producto @')
        checked = check_invariants(current_local, panels, original_update) if k == 0 and view.size else 0
        records.append(dict(k=k, before=before, after_panels=after_panels, after_schur=a.copy(),
                            lower=lower, upper=upper, matmul_calls=calls, invariant_negatives=checked,
                            active_shape=list(view.shape), active_strides=list(view.strides),
                            active_c_contiguous=bool(view.flags.c_contiguous),
                            row_slice=[panels.row_slice.start, panels.row_slice.stop],
                            col_slice=[panels.col_slice.start, panels.col_slice.stop],
                            active_empty=not bool(view.size), empty_local=not bool(a.size),
                            finalized_exact=True, outside_schur_exact=True, panels_exact=True,
                            in_place=True))
    distributed.diagonal_panel_step, distributed.schur_update = step, update
    try:
        distributed.lu_blocked_distributed(grid, local)
    finally:
        distributed.diagonal_panel_step, distributed.schur_update = original_step, original_update
    return comm.gather(records, root=0)


def global_steps(p, original, reference, payload):
    """Serial global por cortes contiguos; jamás usa slices locales MPI."""
    state = original.copy()
    steps = []
    for k in range(p.N//p.b):
        before = state.copy()
        serial_phase(state, p.b, k, schur=False)
        expected_panels = state.copy()
        s, e = k*p.b, (k+1)*p.b
        # Referencia adicional: suma de productos exteriores, sin GEMM por bloques.
        expected_schur = state.copy()
        for h in range(s, e):
            expected_schur[e:, e:] -= np.outer(state[e:, h], state[h, e:])
        state[e:, e:] -= state[e:, s:e] @ state[s:e, e:]
        reconstructed = {name: reconstruct(p, [r[k][name] for r in payload])
                         for name in ('before', 'after_panels', 'after_schur')}
        metrics = dict(before=compare(reconstructed['before'], before),
                       after_panels=compare(reconstructed['after_panels'], expected_panels),
                       after_schur=compare(reconstructed['after_schur'], state),
                       outer_product_equation=compare(reconstructed['after_schur'][e:, e:],
                                                     expected_schur[e:, e:]))
        # Región fuera de Schur se conserva EXACTAMENTE en la reconstrucción.
        mask = np.ones((p.N, p.N), dtype=bool); mask[e:, e:] = False
        np.testing.assert_array_equal(reconstructed['after_schur'][mask],
                                      reconstructed['after_panels'][mask])
        ranks = []
        for rank, records in enumerate(payload):
            record = records[k]
            rows, cols = axes(p, rank)
            ar, ac = rows[rows >= e], cols[cols >= e]
            panel_metrics = dict(lower=compare(record['lower'], expected_panels[np.ix_(ar, range(s, e))]),
                                 upper=compare(record['upper'], expected_panels[np.ix_(range(s, e), ac)]))
            ranks.append(dict(rank=rank, **{key:value for key,value in record.items()
                          if key not in ('before','after_panels','after_schur','lower','upper')},
                              panel_metrics=panel_metrics))
        steps.append(dict(k=k, metrics=metrics, ranks=ranks,
                          serial_before=before.tolist(), serial_after_panels=expected_panels.tolist(),
                          serial_after_schur=state.tolist(),
                          **{name:value.tolist() for name,value in reconstructed.items()}, passed=True))
    # La transcripción por paso debe acabar en la LU ORIGINAL del profesor.
    compare(state, reference)
    return steps


def failure_scenario(grid, local, scenario):
    """Inyectar errores locales en Schur; Abort pertenece a producción."""
    original = distributed.schur_update
    def broken(current_local, panels):
        if grid.rank == 2 and panels.k == 0:
            if scenario == 'schur_shape':
                panels = replace(panels, lower=panels.lower[:-1, :])
            elif scenario == 'schur_slices':
                panels = replace(panels, col_slice=slice(0, current_local.values.shape[1]))
            elif scenario == 'schur_value':
                raise ValueError('fallo local controlado durante Schur')
            elif scenario == 'schur_overflow':
                panels = replace(panels, lower=np.full_like(panels.lower, 1e308),
                                 upper=np.full_like(panels.upper, 1e308))
            elif scenario == 'schur_nan':
                panels.lower[0, 0] = np.nan
        return original(current_local, panels)
    distributed.schur_update = broken
    if scenario == 'main_schur':
        def fail(current_local, panels, timers=None):
            if grid.rank == 2:
                raise ValueError('fallo Schur desde main')
            return original(current_local, panels, timers=timers)
        distributed.schur_update = fail
        return distributed.main(['--N','32','--b','8','--pr','2','--pc','2'])
    distributed.lu_blocked_distributed(grid, local)
    raise AssertionError(f'El fallo {scenario} no terminó mediante MPI Abort')


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--scenario', default='normal', choices=['normal','schur_shape','schur_slices',
                       'schur_value','schur_overflow','schur_nan','main_schur'])
    args, remaining = extra.parse_known_args()
    comm = MPI.COMM_WORLD
    p = distributed.parse_parameters(comm, remaining)
    require(p is not None and p.N <= MAX_N, f'Worker limitado a N <= {MAX_N}')
    require(not sys.flags.optimize, 'No usar -O')
    grid = distributed.ProcessGrid.create(comm, p)
    local = distributed.LocalMatrix.initialize(grid)
    if args.scenario != 'normal':
        return failure_scenario(grid, local, args.scenario)
    original = matrix_entries(range(p.N), range(p.N), p.N) if grid.rank == 0 else None
    reference = original.copy() if grid.rank == 0 else None
    if grid.rank == 0:
        lu_blocked(reference, p.b)  # Oráculo principal ORIGINAL, intacto.
    array, address, shape, strides = local.values, local.values.ctypes.data, local.values.shape, local.values.strides
    distributed.lu_blocked_distributed(grid, local)  # Ejecución sin observadores ni colectivas de test.
    require(local.values is array and local.values.ctypes.data == address and
            local.values.shape == shape and local.values.strides == strides, 'LU in-place')
    payload = comm.gather(local.values, root=0)
    result = dict(kind='etapa_04_lu_test', N=p.N, b=p.b, pr=p.pr, pc=p.pc,
                  atol=ATOL, rtol=RTOL, residual_limit=RTOL, passed=True,
                  original_lu_blocked_reference=True, assertions_enabled=not sys.flags.optimize)
    if grid.rank == 0:
        factors = reconstruct(p, payload)
        result.update(final=final_metrics(original, factors, reference),
                      original=original.tolist(), serial_factors=reference.tolist(),
                      distributed_factors=factors.tolist())
    local.values[:] = matrix_entries(local.layout.rows, local.layout.cols, p.N)
    observed = observe_sequence(grid, local)
    if grid.rank == 0:
        result['steps'] = global_steps(p, original, reference, observed)
        result['observed_final'] = final_metrics(original, reconstruct(p, [r[-1]['after_schur'] for r in observed]), reference)
    grid.free()
    if grid.rank == 0:
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
