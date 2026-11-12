#!/usr/bin/env python3
"""Campaña A: procesos independientes, un hilo, sin modificar lu_serial.py.

Ejecutar desde cualquier directorio con el Python aprobado. Cada --output debe
ser nuevo. Ver --help; el procesamiento posterior se hace con summarize_a.py.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/home/ppvergara/miniconda3/envs/tarea1-hpc/bin/python')
BLOCKS = [4096, 2048, 1024, 512, 256, 128, 64]
# Orden fijado antes de medir: descendente, ascendente y rotación de tres.
ROUNDS = [BLOCKS, BLOCKS[::-1], BLOCKS[3:] + BLOCKS[:3]]
ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0')
LINE = re.compile(r'^N=(\d+) b=(\d+) tiempo=(\d+\.\d{3}) s '
                  r'GFLOP/s=(\S+) checksum=(\S+)$', re.MULTILINE)
RESIDUAL = re.compile(r'^residuo relativo=(\S+)$', re.MULTILINE)
CHECK_LIMIT = 1e-10  # Criterio diagnóstico, definido antes de ejecutar.
DIAGNOSTIC = r'''
import contextlib, io, json, os, platform, sys
import numpy as np
import scipy
from scipy.linalg import lu_factor, solve_triangular
import threadpoolctl
a = np.array([[4., 1.], [1., 4.]])
product = a @ a
lu, piv = lu_factor(a)
solve_triangular(lu, a)
np_config, sp_config = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(np_config): np.show_config()
with contextlib.redirect_stdout(sp_config): scipy.show_config()
pools = threadpoolctl.threadpool_info()
print(json.dumps(dict(executable=sys.executable, python=sys.version,
    numpy=np.__version__, scipy=scipy.__version__,
    threadpoolctl=threadpoolctl.__version__, platform=platform.platform(),
    uname=list(platform.uname()), affinity=sorted(os.sched_getaffinity(0)),
    environment={k: os.environ.get(k) for k in (
        'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'BLIS_NUM_THREADS','OMP_DYNAMIC','MKL_DYNAMIC','OMP_PROC_BIND',
        'OMP_PLACES','PYTHONOPTIMIZE','PYTHONDONTWRITEBYTECODE')},
    threadpools=pools, numpy_config=np_config.getvalue(),
    scipy_config=sp_config.getvalue(),
    single_thread_confirmed=bool(pools) and all(p['num_threads']==1 for p in pools)),
    indent=2))
'''


def now():
    return datetime.now(ZoneInfo('America/Santiago')).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def snapshot():
    return dict(time=now(), loadavg=os.getloadavg(),
                meminfo=Path('/proc/meminfo').read_text(),
                cpu_stat=Path('/proc/stat').read_text().splitlines()[0])


def parse_output(stdout, n, b, check):
    matches = list(LINE.finditer(stdout))
    if len(matches) != 1:
        raise ValueError('Se esperaba exactamente una línea de tiempo serial')
    m = matches[0]
    if (int(m[1]), int(m[2])) != (n, b):
        raise ValueError('N/b de stdout no coinciden con el comando')
    t = float(m[3])
    if not all(math.isfinite(float(x)) for x in (m[3], m[4], m[5])):
        raise ValueError('Salida no finita')
    residual = None
    if check:
        residues = RESIDUAL.findall(stdout)
        if len(residues) != 1:
            raise ValueError('Falta residuo diagnóstico')
        residual = float(residues[0])
        if not math.isfinite(residual) or not 0 <= residual <= CHECK_LIMIT:
            raise ValueError('Residuo fuera del criterio diagnóstico')
    elif t <= 0:
        raise ValueError('Tiempo de benchmark sin resolución suficiente')
    return t, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True,
                        choices=['diagnostico', 'piloto', 'oficial'])
    parser.add_argument('--output', required=True, type=Path,
                        help='Directorio NUEVO; nunca se sobrescribe')
    parser.add_argument('--pilot', type=Path,
                        help='Piloto válido requerido para fase oficial')
    args = parser.parse_args()
    if Path(sys.executable).absolute() != PYTHON:
        parser.error(f'Usar el intérprete aprobado: {PYTHON}')
    if args.phase == 'oficial':
        if not args.pilot:
            parser.error('--pilot es obligatorio para mediciones oficiales')
        pm = json.loads((args.pilot / 'metadata.json').read_text())
        with (args.pilot / 'raw.csv').open() as f:
            pr = list(csv.DictReader(f))
        if (pm['phase'] != 'piloto' or not pm['diagnostic']['single_thread_confirmed']
                or len(pr) != 7 or {int(r['b']) for r in pr} != set(BLOCKS)
                or any(r['status'] != 'valido' for r in pr)
                or pm['hashes']['lu_serial.py'] != sha(ROOT / 'lu_serial.py')
                or pm['hashes']['scripts/run_a_serial.py'] != sha(__file__)):
            parser.error('Piloto incompleto, inválido o de código distinto')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'logs').mkdir()
    child_env = os.environ.copy()
    child_env.update(ENV)  # Fijado ANTES de iniciar cada proceso Python.
    diagnostic_cmd = [str(PYTHON), '-B', '-c', DIAGNOSTIC]
    d = subprocess.run(diagnostic_cmd, env=child_env, cwd=ROOT,
                       capture_output=True, timeout=60)
    (out / 'threadcheck.stdout').write_bytes(d.stdout)
    (out / 'threadcheck.stderr').write_bytes(d.stderr)
    diagnostic = json.loads(d.stdout) if d.returncode == 0 else {}
    hashes = {p: sha(ROOT / p) for p in [
        'lu_serial.py', 'scripts/run_a_serial.py', 'docs/tarea03.pdf',
        '.gitignore', '.local/ROADMAP_PERSONA1_T3.md', '.local/ETAPA_00_CIERRE.md']}
    metadata = dict(phase=args.phase, started=now(), cwd=str(ROOT),
                    invocation=sys.argv, thread_environment=ENV, hashes=hashes,
                    diagnostic_command=diagnostic_cmd,
                    diagnostic_returncode=d.returncode, diagnostic=diagnostic,
                    git={k: subprocess.check_output(['git', *v], cwd=ROOT,
                         text=True).strip() for k, v in dict(
                             branch=['branch', '--show-current'],
                             commit=['rev-parse', 'HEAD'],
                             status=['status', '--short', '--untracked-files=all']).items()},
                    lscpu=subprocess.check_output(['lscpu'], text=True),
                    os_release=Path('/etc/os-release').read_text(),
                    initial_system=snapshot(), check_limit=CHECK_LIMIT,
                    pilot=str(args.pilot.resolve()) if args.pilot else None,
                    timing_source='lu_serial.py stdout: perf_counter alrededor de lu_blocked; .3f')
    write_json(out / 'metadata.json', metadata)
    if d.returncode or d.stderr or not diagnostic.get('single_thread_confirmed'):
        raise SystemExit('No se confirmó un hilo: no se ejecuta la campaña')
    if args.phase == 'diagnostico':
        schedule = [(1, 64, 16), (2, 128, 32)]
    elif args.phase == 'piloto':
        schedule = [(1, 4096, b) for b in BLOCKS]
    else:
        schedule = [(rep, 4096, b) for rep, order in enumerate(ROUNDS, 1) for b in order]
    metadata['schedule'] = schedule
    write_json(out / 'metadata.json', metadata)
    fields = ['run_id', 'type', 'N', 'b', 'repetition', 'order', 'started', 'ended',
              'factorization_s', 'residual', 'returncode', 'stdout', 'stderr',
              'stdout_sha256', 'stderr_sha256', 'status', 'error', 'command',
              'external_process_s']
    all_valid = True
    with (out / 'raw.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()
        for order, (rep, n, b) in enumerate(schedule, 1):
            rid = f'{args.phase}_{order:02d}_r{rep}_N{n}_b{b}'
            cmd = [str(PYTHON), '-B', str(ROOT / 'lu_serial.py'), '--N', str(n), '--b', str(b)]
            check = args.phase == 'diagnostico'
            if check:
                cmd.append('--check')
            row = dict(run_id=rid, type=args.phase, N=n, b=b, repetition=rep,
                       order=order, started=now(), status='incompleto',
                       stdout=f'logs/{rid}.stdout', stderr=f'logs/{rid}.stderr',
                       command=shlex.join(['env', *(f'{k}={v}' for k, v in ENV.items()), *cmd]))
            record_path = out / 'logs' / f'{rid}.json'
            before = snapshot()
            write_json(record_path, dict(row=row, before=before))
            print(f'Inicio {rid}', flush=True)
            start = time.perf_counter()
            interrupted = False
            with (out / row['stdout']).open('xb') as stdout, (out / row['stderr']).open('xb') as stderr:
                try:
                    process = subprocess.run(cmd, env=child_env, cwd=ROOT,
                                             stdout=stdout, stderr=stderr, timeout=900)
                    row['returncode'] = process.returncode
                except (subprocess.TimeoutExpired, KeyboardInterrupt, OSError) as e:
                    row['returncode'] = None
                    row['error'] = str(e) or type(e).__name__
                    interrupted = isinstance(e, KeyboardInterrupt)
            row.update(ended=now(), external_process_s=time.perf_counter() - start,
                       stdout_sha256=sha(out / row['stdout']),
                       stderr_sha256=sha(out / row['stderr']), status='fallido')
            try:
                if row['returncode'] != 0 or (out / row['stderr']).stat().st_size:
                    raise ValueError('Salida no cero o stderr no vacío')
                row['factorization_s'], row['residual'] = parse_output(
                    (out / row['stdout']).read_text(), n, b, check)
                row['status'] = 'valido'
            except ValueError as e:
                row['error'] = row.get('error', str(e))
            write_json(record_path, dict(row=row, before=before, after=snapshot()))
            writer.writerow(row)
            f.flush()
            all_valid &= row['status'] == 'valido'
            print(f"Fin {rid}: {row['status']}, LU={row.get('factorization_s')} s", flush=True)
            if interrupted:
                break
    metadata.update(ended=now(), final_system=snapshot(),
                    all_valid=all_valid and not interrupted)
    write_json(out / 'metadata.json', metadata)
    if not metadata['all_valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
