"""Batería de distribución de Etapa 2; incluye smoke de la CLI actual.

Usar el Python del entorno MPI; --output debe ser un directorio nuevo.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CASES = [(16,8,1,1), (32,8,1,2), (32,8,2,1), (32,8,2,2),
         (48,8,2,3), (48,8,3,2), (16,8,1,4), (16,8,4,1),
         (16,8,2,4), (16,8,4,2), (16,8,4,4),
         (32,4,2,2), (16,16,2,2), (40,8,2,3), (8,1,2,2)]
THREAD_ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path,
                        default=Path(sys.executable).parent/'mpiexec')
    parser.add_argument('--max-processes', type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    capacity = len(os.sched_getaffinity(0))
    # Respetar también cuotas numéricas de los controladores habituales.
    for path in ['/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
                 '/sys/fs/cgroup/cpu/init.scope/cpu.cfs_quota_us']:
        f = Path(path)
        if not f.exists():
            continue
        words = f.read_text().split()
        if path.endswith('cpu.max'):
            if words[0] != 'max':
                capacity = min(capacity, int(words[0])//int(words[1]))
        elif int(words[0]) > 0:
            period = int(f.with_name('cpu.cfs_period_us').read_text())
            capacity = min(capacity, int(words[0])//period)
    if not 1 <= args.max_processes <= capacity:
        parser.error(f'--max-processes debe estar entre 1 y {capacity}; sin oversubscription')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(THREAD_ENV)
    launcher = str(args.mpiexec.resolve())
    python = str(Path(sys.executable).absolute())
    worker = [python, '-B', 'tests/test_distribution.py']
    program = [python, '-B', 'lu_distributed.py']
    summary = dict(kind='etapa_02_test_suite', python=python,
                   python_version=sys.version, mpiexec=launcher,
                   affinity=sorted(os.sched_getaffinity(0)), capacity=capacity,
                   max_processes=args.max_processes, environment=THREAD_ENV,
                   hashes={p:sha(ROOT/p) for p in ['lu_distributed.py', 'lu_serial.py',
                           'tests/test_distribution.py', 'tests/run_distribution.py',
                           'environment-mpi.yml']}, cases=[])

    def save():
        (out/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    def run(name, cmd, p, category, config=None, expected_error=None):
        record = dict(name=name, category=category, configuration=config, processes=p,
                      command=cmd, shell_command=shlex.join(
                          ['env', *(f'{k}={v}' for k,v in THREAD_ENV.items()), *cmd]),
                      cwd=str(ROOT), timeout_s=args.timeout,
                      started=datetime.now(ZoneInfo('America/Santiago')).isoformat(),
                      timed_out=False, status='pending', returncode=None)
        summary['cases'].append(record)
        if p > args.max_processes:
            record.update(status='not_run', reason='Procesos exceden recursos autorizados')
            save()
            return
        directory = out/name
        directory.mkdir()
        for rel in ['lu_distributed.py', 'tests/test_distribution.py']:
            require_hash = summary['hashes'][rel]
            if sha(ROOT/rel) != require_hash:
                raise RuntimeError('Código cambió durante la batería')
        save()
        start = time.monotonic()
        with (directory/'stdout').open('xb') as stdout, (directory/'stderr').open('xb') as stderr:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                    start_new_session=True)
            try:
                record['returncode'] = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                record['timed_out'] = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                record['returncode'] = proc.returncode
        record['duration_s'] = time.monotonic()-start
        stdout = (directory/'stdout').read_text()
        stderr = (directory/'stderr').read_text()
        record.update(stdout_sha256=sha(directory/'stdout'), stderr_sha256=sha(directory/'stderr'))
        try:
            if record['timed_out']:
                raise ValueError('Timeout: nunca se considera aprobado')
            if category == 'negative':
                if record['returncode'] != 2 or 'Error:' not in stderr or expected_error not in stderr:
                    raise ValueError('No se observó el error esperado con código 2')
                if stdout.strip():
                    raise ValueError('Argumentos inválidos generaron salida aparentemente válida')
            else:
                if record['returncode'] != 0 or stderr:
                    raise ValueError('Salida no cero o stderr no vacío')
                if category == 'distribution':
                    result = json.loads(stdout)
                    if not result['passed']:
                        raise ValueError('Prueba distribuida fallida')
                    if [result[k] for k in ['N','b','pr','pc']] != config:
                        raise ValueError('Configuración distinta a la solicitada')
                    record['reconstruction'] = result['reconstruction']
                    record['empty_ranks'] = sum(r['empty_matrix'] for r in result['ranks'])
                    record['noncontiguous_active_views'] = sum(
                        not region['c_contiguous'] and all(region['shape'])
                        for r in result['ranks'] for region in r['active'])
                elif category == 'cli' and 'LU distribuida finalizada' not in stdout:
                    raise ValueError('Falta diagnóstico de finalización de LU')
                elif category == 'help' and '--pr' not in stdout:
                    raise ValueError('Falta ayuda')
            record['status'] = 'passed'
        except Exception as exc:
            record.update(status='failed', error=str(exc))
        (directory/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        save()
        print(f'{name}: {record["status"]}, rc={record["returncode"]}', flush=True)

    for N,b,pr,pc in CASES:
        config = [N,b,pr,pc]
        flags = ['--N',str(N),'--b',str(b),'--pr',str(pr),'--pc',str(pc)]
        run(f'N{N}_b{b}_{pr}x{pc}', [launcher,'-n',str(pr*pc),*worker,*flags],
            pr*pc, 'distribution', config)
    base = ['--N','16','--b','8','--pr','1','--pc','2']
    for flag, index in [('N',1), ('b',3), ('pr',5), ('pc',7)]:
        for value in ['0','-1']:
            flags = base.copy()
            flags[index] = value
            run(f'invalid_{flag}_{value}', [launcher,'-n','2',*program,*flags],
                2, 'negative', expected_error=f'{flag} debe ser > 0')
    for name, flags, message in [
        ('nondivisible', ['--N','17',*base[2:]], 'N debe ser divisible'),
        ('size_mismatch', [*base[:5],'2',*base[6:]], 'size=2 debe ser pr*pc=4'),
        ('check_reserved', [*base,'--check'], '--check está reservado'),
        ('unknown', [*base,'--unknown'], 'unrecognized arguments'),
        ('integer', ['--N','abc',*base[2:]], 'invalid int value'),
        ('missing_grid', ['--N','16','--b','8'], 'required')]:
        run('invalid_'+name, [launcher,'-n','2',*program,*flags], 2, 'negative',
            expected_error=message)
    # MPMD: los ranks reciben argumentos distintos, incluso un error de parseo
    # exclusivo de rank 0. Comprueba que ningún rank entra solo en Split.
    for name, left, message in [
        ('different', ['--N','32',*base[2:]], 'idénticos'),
        ('one_invalid', ['--N','0',*base[2:]], 'idénticos'),
        ('one_syntax', [*base,'--unknown'], 'Argumentos inválidos'),
        ('one_help', ['--help'], 'idénticos')]:
        run('mpmd_'+name, [launcher,'-n','1',*program,*left,':','-n','1',*program,*base],
            2, 'negative', expected_error=message)
    run('cli_2x2', [launcher,'-n','4',*program,'--N','32','--b','8','--pr','2','--pc','2'],
        4, 'cli')
    run('help', [launcher,'-n','2',*program,'--help'], 2, 'help')
    summary['counts'] = {s:sum(r['status']==s for r in summary['cases'])
                         for s in ['passed','failed','not_run']}
    save()
    print(json.dumps(summary['counts']))
    return 1 if summary['counts']['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
