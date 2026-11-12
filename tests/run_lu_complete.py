"""Batería secuencial de correctitud Etapa 4; no es instrumentación de B.

Usar el Python del prefijo MPI. --output debe ser nuevo. Los JSON son evidencia
de tests, sin esquema experimental ni tiempos numéricos por rank.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from run_distribution import ROOT, THREAD_ENV, sha

CASES = [(16,16,1,1), (32,8,1,1), (32,8,1,2), (32,8,2,1), (32,8,2,2),
         (48,8,2,3), (48,8,3,2), (32,4,2,2), (40,8,2,3),
         (16,8,1,4), (16,8,4,1), (16,8,2,4), (16,8,4,2), (16,16,2,2),
         (16,8,4,4), (64,16,4,4), (8,1,2,2)]
FAILURES = [('schur_shape','Schur requiere lower'),
            ('schur_slices','Slices de Schur incompatibles'),
            ('schur_value','fallo local controlado durante Schur'),
            ('schur_overflow','FloatingPointError'),
            ('schur_nan','Schur con NaN/Inf'), ('main_schur','fallo Schur desde main')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, default=Path(sys.executable).parent/'mpiexec')
    parser.add_argument('--max-processes', type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument('--timeout', type=float, default=40)
    args = parser.parse_args()
    capacity = len(os.sched_getaffinity(0))
    for name in ['/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
                 '/sys/fs/cgroup/cpu/init.scope/cpu.cfs_quota_us']:
        path = Path(name)
        if not path.exists():
            continue
        words = path.read_text().split()
        if name.endswith('cpu.max'):
            if words[0] != 'max':
                capacity = min(capacity, int(words[0])//int(words[1]))
        elif int(words[0]) > 0:
            capacity = min(capacity, int(words[0])//int(path.with_name('cpu.cfs_period_us').read_text()))
    if not 1 <= args.max_processes <= capacity:
        parser.error(f'--max-processes entre 1 y {capacity}; sin oversubscription')
    if not 0 < args.timeout <= 120:
        parser.error('--timeout entre 0 y 120 segundos')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy(); env.update(THREAD_ENV)
    python, launcher = str(Path(sys.executable).absolute()), str(args.mpiexec.resolve())
    hashes = {p:sha(ROOT/p) for p in ['lu_serial.py','lu_distributed.py','tests/test_lu_complete.py',
              'tests/run_lu_complete.py','tests/test_panels.py','tests/run_distribution.py',
              'tests/test_distribution.py','tests/run_panels.py','environment-mpi.yml']}
    summary = dict(kind='etapa_04_test_suite', command=[sys.executable,*sys.argv], cwd=str(ROOT),
                   python=python, python_version=sys.version, mpiexec=launcher, environment=THREAD_ENV,
                   capacity=capacity, max_processes=args.max_processes,
                   affinity=sorted(os.sched_getaffinity(0)), hashes=hashes, cases=[])

    def save():
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    def run(name, config, scenario='normal', cli=False, expected_error=None):
        N,b,pr,pc = config
        flags = ['--N',str(N),'--b',str(b),'--pr',str(pr),'--pc',str(pc)]
        cmd = [launcher,'-n',str(pr*pc),python,'-B',
               'lu_distributed.py' if cli else 'tests/test_lu_complete.py',*flags]
        if not cli:
            cmd += ['--scenario',scenario]
        record = dict(name=name, configuration=config, scenario=scenario, cli=cli, processes=pr*pc,
                      command=cmd, shell_command=shlex.join(['env',*[f'{k}={v}' for k,v in THREAD_ENV.items()],*cmd]),
                      cwd=str(ROOT), timeout_s=args.timeout, expected_error=expected_error, timed_out=False,
                      returncode=None, status='pending', started=datetime.now(ZoneInfo('America/Santiago')).isoformat())
        summary['cases'].append(record)
        if pr*pc > args.max_processes:
            record.update(status='not_run', reason='Procesos exceden CPUs autorizadas'); save(); return
        directory = out/name; directory.mkdir()
        for path,digest in hashes.items():
            if sha(ROOT/path) != digest:
                raise RuntimeError('Código cambió durante batería')
        save(); start = time.monotonic()
        with (directory/'stdout').open('xb') as stdout, (directory/'stderr').open('xb') as stderr:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                record['timed_out'] = True; os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        record.update(returncode=proc.returncode, duration_s=time.monotonic()-start,
                      stdout_sha256=sha(directory/'stdout'), stderr_sha256=sha(directory/'stderr'))
        stdout, stderr = (directory/'stdout').read_text(), (directory/'stderr').read_text()
        try:
            if record['timed_out']:
                raise ValueError('Timeout siempre es fallo')
            if expected_error:
                if not proc.returncode or expected_error not in stderr or 'MPI Abort(1)' not in stderr:
                    raise ValueError('Falta error específico, Abort o salida no cero')
                if 'rank=2' not in stderr:
                    raise ValueError('Falta diagnóstico rank=2')
                record['expected_abort_observed'] = True
            else:
                if proc.returncode != 0 or stderr:
                    raise ValueError('Salida no cero o stderr inesperado')
                if cli:
                    if 'LU distribuida finalizada' not in stdout or f'grid={pr}x{pc}' not in stdout:
                        raise ValueError('CLI no informa finalización esperada')
                else:
                    result = json.loads(stdout)
                    if not result['passed'] or [result[k] for k in ('N','b','pr','pc')] != config:
                        raise ValueError('Correctitud/configuración incorrecta')
                    if len(result['steps']) != N//b or any(not s['passed'] for s in result['steps']):
                        raise ValueError('Faltan pasos numéricos correctos')
                    record['metrics'] = result['final']; record['steps_verified'] = len(result['steps'])
                    ranks = [r for step in result['steps'] for r in step['ranks']]
                    record.update(matmul_calls=sum(len(r['matmul_calls']) for r in ranks),
                                  empty_active_regions=sum(r['active_empty'] for r in ranks),
                                  noncontiguous_active_regions=sum(not r['active_empty'] and not r['active_c_contiguous'] for r in ranks),
                                  empty_ranks=sum(r['empty_local'] for r in result['steps'][0]['ranks']),
                                  invariant_negatives=sum(r['invariant_negatives'] for r in ranks))
            record['status'] = 'passed'
        except Exception as exc:
            record.update(status='failed', error=str(exc))
        (directory/'record.json').write_text(json.dumps(record, indent=2)+'\n'); save()
        print(f'{name}: {record["status"]}, rc={proc.returncode}', flush=True)

    for case in CASES:
        N,b,pr,pc = case
        run(f'N{N}_b{b}_{pr}x{pc}', list(case))
    for scenario,message in FAILURES:
        run('failure_'+scenario, [32,8,2,2], scenario, expected_error=message)
    run('cli_2x2', [32,8,2,2], cli=True)
    run('cli_empty_2x4', [16,8,2,4], cli=True)
    summary['counts'] = {s:sum(r['status']==s for r in summary['cases']) for s in ('passed','failed','not_run')}
    save(); print(json.dumps(summary['counts']))
    return 1 if summary['counts']['failed'] or summary['counts']['not_run'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
