"""Batería secuencial pequeña de Etapa 3; no mide rendimiento ni ejecuta LU MPI."""
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

# Un lanzamiento por geometría; cada k reinicia un fixture matemáticamente válido.
CASES = [(16,8,1,1,[0,1]), (32,8,1,2,[0,1]), (32,8,2,1,[0,1]),
         (32,8,2,2,[0,1,3]), (48,8,2,3,[0,1,2,5]), (48,8,3,2,[0,1,2,5]),
         (16,8,1,4,[0,1]), (16,8,4,1,[0,1]), (16,8,2,4,[0,1]),
         (16,8,4,2,[0,1]), (16,8,4,4,[0,1]), (16,16,2,2,[0]),
         (32,4,2,2,[0,1,3,7]), (40,8,2,3,[0,1,4]), (8,1,2,2,[0,1,7])]
FAILURES = [('pivot','Hay pivoteo',3), ('singular','LinAlgWarning',3),
            ('shape','matriz local debe ser',2), ('scipy_value','ValueError',2),
            ('scipy_linalg','LinAlgError',2), ('invalid_k','k debe ser',2),
            ('init_value','fallo local controlado de inicialización',1),
            ('wrong_l','Not equal to tolerance',None)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,default=Path(sys.executable).parent/'mpiexec')
    parser.add_argument('--max-processes',type=int,default=len(os.sched_getaffinity(0)))
    parser.add_argument('--timeout',type=float,default=40)
    args = parser.parse_args()
    capacity = len(os.sched_getaffinity(0))
    for name in ['/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
                 '/sys/fs/cgroup/cpu/init.scope/cpu.cfs_quota_us']:
        path = Path(name)
        if not path.exists():
            continue
        words = path.read_text().split()
        if name.endswith('cpu.max'):
            if words[0] != 'max':
                capacity = min(capacity,int(words[0])//int(words[1]))
        elif int(words[0]) > 0:
            period = int(path.with_name('cpu.cfs_period_us').read_text())
            capacity = min(capacity,int(words[0])//period)
    if not 1 <= args.max_processes <= capacity:
        parser.error(f'--max-processes entre 1 y {capacity}; sin oversubscription')
    if not 0 < args.timeout <= 120:
        parser.error('--timeout debe estar entre 0 y 120 segundos')
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    env = os.environ.copy()
    env.update(THREAD_ENV)
    python, launcher = str(Path(sys.executable).absolute()), str(args.mpiexec.resolve())
    hashes = {p:sha(ROOT/p) for p in ['lu_serial.py','lu_distributed.py','tests/test_panels.py',
                                     'tests/run_panels.py','tests/run_distribution.py',
                                     'tests/test_distribution.py','environment-mpi.yml']}
    summary = dict(kind='etapa_03_test_suite', command=[sys.executable,*sys.argv],
                   cwd=str(ROOT), python=python, python_version=sys.version, mpiexec=launcher,
                   environment=THREAD_ENV, affinity=sorted(os.sched_getaffinity(0)),
                   capacity=capacity, max_processes=args.max_processes,
                   hashes=hashes, cases=[])

    def save():
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    def run(name, config, ks, scenario='normal', expected_error=None, failure_rank=None):
        N,b,pr,pc = config
        count = pr*pc
        cmd = [launcher,'-n',str(count),python,'-B','tests/test_panels.py',
               '--N',str(N),'--b',str(b),'--pr',str(pr),'--pc',str(pc),
               '--ks',','.join(map(str,ks)),'--scenario',scenario]
        record = dict(name=name, configuration=config, ks=ks, scenario=scenario, processes=count,
                      command=cmd, shell_command=shlex.join(
                          ['env',*[f'{k}={v}' for k,v in THREAD_ENV.items()],*cmd]),
                      cwd=str(ROOT), timeout_s=args.timeout, expected_error=expected_error,
                      failure_rank=failure_rank, timed_out=False, status='pending', returncode=None,
                      started=datetime.now(ZoneInfo('America/Santiago')).isoformat())
        summary['cases'].append(record)
        if count > args.max_processes:
            record.update(status='not_run',reason='Procesos exceden CPUs autorizadas')
            save()
            return
        directory = out/name
        directory.mkdir()
        for p,digest in hashes.items():
            if sha(ROOT/p) != digest:
                raise RuntimeError('El código cambió durante la batería')
        save()
        start = time.monotonic()
        with (directory/'stdout').open('xb') as stdout, (directory/'stderr').open('xb') as stderr:
            proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                                    start_new_session=True)
            try:
                proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                record['timed_out'] = True
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
        record.update(returncode=proc.returncode,duration_s=time.monotonic()-start,
                      stdout_sha256=sha(directory/'stdout'),stderr_sha256=sha(directory/'stderr'))
        stdout, stderr = (directory/'stdout').read_text(), (directory/'stderr').read_text()
        try:
            if record['timed_out']:
                raise ValueError('Timeout; nunca se acepta como terminación propia de MPI')
            if expected_error:
                if not proc.returncode or expected_error not in stderr or 'Abort(' not in stderr:
                    raise ValueError('No se observó excepción esperada y MPI Abort con salida no cero')
                if failure_rank is not None and f'rank {failure_rank}' not in stderr and f'rank={failure_rank}' not in stderr:
                    raise ValueError('Falta diagnóstico del rank que falla')
                record['expected_abort_observed'] = True
            else:
                if proc.returncode != 0 or stderr:
                    raise ValueError('Salida no cero o stderr inesperado')
                result = json.loads(stdout)
                if not result['passed'] or result['ks'] != ks or len(result['steps']) != len(ks):
                    raise ValueError('Falló la verificación matemática/protocolo o faltan pasos')
                if [result[k] for k in ['N','b','pr','pc']] != config:
                    raise ValueError('Configuración distinta de la pedida')
                record['steps_verified'] = len(ks)
                record['max_absolute'] = max(s['global_check']['metrics']['max_absolute'] for s in result['steps'])
                record['max_relative_frobenius'] = max(s['global_check']['metrics']['relative_frobenius'] for s in result['steps'])
                record['collective_calls'] = sum(len(r['communications']) for s in result['steps'] for r in s['ranks'])
                record['zero_count_calls'] = sum(e['count']==0 for s in result['steps'] for r in s['ranks'] for e in r['communications'])
                record['empty_ranks'] = max(sum(r['empty_matrix'] for r in s['ranks']) for s in result['steps'])
            record['status'] = 'passed'
        except Exception as exc:
            record.update(status='failed',error=str(exc))
        (directory/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        save()
        print(f'{name}: {record["status"]}, rc={proc.returncode}',flush=True)

    for N,b,pr,pc,ks in CASES:
        run(f'N{N}_b{b}_{pr}x{pc}',[N,b,pr,pc],ks)
    for scenario,message,rank in FAILURES:
        run('failure_'+scenario,[32,8,2,2],[0] if scenario=='wrong_l' else [1],
            scenario,message,rank)
    summary['counts'] = {s:sum(r['status']==s for r in summary['cases'])
                         for s in ['passed','failed','not_run']}
    save()
    print(json.dumps(summary['counts']))
    return 1 if summary['counts']['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
