"""Etapa 6: batería secuencial reproducible; --output siempre debe ser nuevo.

Incluye la CLI real, worker observable, negativos, repetibilidad y pilotos
opcionales. No ejecuta campañas C–G. Un timeout siempre falla.
"""
import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from run_distribution import ROOT, THREAD_ENV, sha
from run_metrics import capacity, require, strict_json, validate_output

BASE = [(16,16,1,1),(32,8,1,1),(32,8,1,2),(32,8,2,1),(32,8,2,2),
        (48,8,2,3),(48,8,3,2),(16,8,1,4),(16,8,4,1),(16,8,2,4),
        (16,8,4,2),(16,8,4,4),(64,16,4,4),(40,8,2,3),(32,4,2,2),(8,1,2,2)]
ADDITIONAL = [(16,16,2,2),(1,1,4,4),(72,8,2,3),(72,8,3,2),(40,8,3,2),
              (96,8,2,4),(96,8,4,2),(96,24,2,2),(128,32,2,2),(128,16,2,2),
              (256,128,2,2),(256,16,2,2)]
FAILURES = [('late_pivot','Hay pivoteo'),('late_singular','LinAlgWarning'),
            ('late_diagonal_nan','Diagonal con NaN/Inf'),('late_upper_inf','Panel U con NaN/Inf'),
            ('late_lower_nan','Panel L con NaN/Inf'),('late_diagonal_inf_output','LU diagonal con NaN/Inf'),
            ('late_schur_overflow','FloatingPointError'),('reconstruct_missing','Reconstrucción: faltan ranks'),
            ('reconstruct_nan','Reconstrucción: shape o finitud'),
            ('time_nan','compute_s debe ser finito'),('time_inf','compute_s debe ser finito'),
            ('time_negative','compute_s debe ser finito'),('time_bool','compute_s debe ser finito'),
            ('metrics_missing','Métricas: falta algún rank'),('metrics_duplicate','Métricas: ranks duplicados')]


def validate_contract(result, config=None, status=None):
    """Consumidor independiente de los validadores de producción; esquema 1.0."""
    validate_output(result, config, status)
    code, env = result['code'], result['environment']
    commit = code['git_commit']
    require(commit is None or (type(commit) is str and len(commit)==40 and
                              all(c in '0123456789abcdef' for c in commit)), 'Tipo git_commit')
    for key in ('python','executable','numpy','scipy','mpi4py','mpi_library','platform','thread_policy','clock'):
        require(type(env[key]) is str and bool(env[key]), 'Tipo metadato '+key)
    require(type(env['clock_resolution_s']) in (int,float) and
            math.isfinite(env['clock_resolution_s']) and env['clock_resolution_s']>0, 'Resolución reloj')
    require(type(env['time_tolerance_s']) in (int,float), 'Tipo tolerancia temporal')
    for r in result['rank_times']:
        require(type(r['thread_env']) is dict and type(r['blas_pools']) is list, 'Tipos de entorno rank')
        for pool in r['blas_pools']:
            require(type(pool['num_threads']) is int and pool['num_threads']==1, 'Tipo hilos efectivos')
    return result


def invalid_documents(good):
    """Entradas para procesos consumidores separados, con rc=2 y diagnóstico."""
    cases = [('syntax','{'),('extra_document','{} {}'),('duplicate_key','{"x":1,"x":2}'),
             ('nan_literal','{"x":NaN}'),('infinity_literal','{"x":Infinity}')]
    mutations = [(('schema_version',),'2.0'),(('p',),99),(('N',),True),(('b',),0),
                 (('wall_s',),0),(('rank_times',),[]),
                 (('rank_times',0,'compute_s'),True),(('rank_times',0,'communication_s'),-1),
                 (('rank_times',0,'total_s'),'0.0'),(('rank_times',0,'unaccounted_s'),-1),
                 (('rank_times',0,'row'),99),(('rank_times',0,'col'),True),
                 (('rank_times',0,'rank'),1),(('rank_times',1,'rank'),0),
                 (('code','git_commit'),123),(('code','git_dirty'),'false'),
                 (('code','sha256','lu_serial.py'),'0'*64),(('environment','numpy'),False),
                 (('environment','clock_resolution_s'),True),(('environment','time_tolerance_s'),'1e-9'),
                 (('rank_times',0,'blas_pools',0,'num_threads'),True),
                 (('check','status'),'passed')]
    for index,(path,value) in enumerate(mutations):
        bad=copy.deepcopy(good); target=bad
        for key in path[:-1]: target=target[key]
        target[path[-1]]=value
        cases.append((f'mutation_{index:02d}', json.dumps(bad,allow_nan=False)))
    for key in good:
        bad=copy.deepcopy(good); bad.pop(key)
        cases.append(('missing_'+key,json.dumps(bad,allow_nan=False)))
    return cases


def cross_comparisons(reports):
    import numpy as np
    comparisons=[]
    grouped={}
    for name,r in reports.items(): grouped.setdefault((r['N'],r['b']),[]).append((name,r))
    for key,items in grouped.items():
        first_name,first=items[0]
        for name,r in items[1:]:
            a,b=np.array(first['factors']),np.array(r['factors'])
            require(np.array_equal(first['input'],r['input']), 'Entradas distintas entre grids/repeticiones')
            np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)
            rel=float(np.linalg.norm(a-b)/np.linalg.norm(a))
            require(rel<=1e-10,'Frobenius entre grids/repeticiones')
            comparisons.append(dict(left=first_name,right=name,N=key[0],b=key[1],
                                    left_grid=[first['pr'],first['pc']],right_grid=[r['pr'],r['pc']],
                                    input_exact=True,max_absolute=float(np.max(np.abs(a-b))),
                                    relative_frobenius=rel,
                                    residual_difference=abs(first['metrics']['distributed_residual']-
                                                            r['metrics']['distributed_residual']),passed=True))
    return comparisons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--validate-json',type=Path)
    parser.add_argument('--max-processes',type=int,default=capacity())
    parser.add_argument('--mpiexec',type=Path,default=Path(sys.executable).parent/'mpiexec')
    parser.add_argument('--timeout',type=float,default=40)
    parser.add_argument('--pilots',action='store_true',help='512/64 y 1024/128, grid 2x2, sin check')
    args=parser.parse_args()
    if args.validate_json:
        try: validate_contract(strict_json(args.validate_json.read_text()))
        except Exception as exc:
            print('Error de contrato JSON: '+str(exc),file=sys.stderr); return 2
        return 0
    if args.output is None: parser.error('--output requerido')
    if not 1<=args.max_processes<=capacity(): parser.error('Sin oversubscription')
    if not 0<args.timeout<=120: parser.error('Timeout entre 0 y 120 segundos')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy(); env.update(THREAD_ENV)
    python,launcher=str(Path(sys.executable).absolute()),str(args.mpiexec.resolve())
    require(Path(launcher).parent == Path(python).parent, 'Python y MPI deben usar mismo prefijo')
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'lu_serial.py',ROOT/'lu_distributed.py',
             ROOT/'docs/OUTPUT_CONTRACT.md',*sorted((ROOT/'tests').glob('*.py'))]}
    summary=dict(kind='etapa_06_suite',command=[sys.executable,*sys.argv],capacity=capacity(),
                 affinity=sorted(os.sched_getaffinity(0)),environment=THREAD_ENV,hashes=hashes,cases=[])
    reports,cli_checks={},{}
    def save(): (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    def run(name,cmd,processes,category,config=None,rc=0,error=None,abort=False,origin=None):
        record=dict(name=name,category=category,configuration=config,processes=processes,origin=origin,
                    command=cmd,shell_command=shlex.join(['env',*[f'{k}={v}' for k,v in THREAD_ENV.items()],*cmd]),
                    cwd=str(ROOT),started=datetime.now(ZoneInfo('America/Santiago')).isoformat(),
                    timeout_s=args.timeout,expected_returncode=rc,expected_error=error,
                    timed_out=False,status='pending')
        summary['cases'].append(record)
        if processes>args.max_processes:
            record.update(status='not_run',reason='Procesos exceden CPUs permitidas');save();return
        directory=out/name;directory.mkdir()
        require(all(sha(ROOT/p)==h for p,h in hashes.items()), 'Código cambió durante batería')
        save();start=time.monotonic()
        with (directory/'stdout').open('xb') as stdout,(directory/'stderr').open('xb') as stderr:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            try:proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                record['timed_out']=True;os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        record.update(returncode=proc.returncode,duration_s=time.monotonic()-start,
                      stdout_sha256=sha(directory/'stdout'),stderr_sha256=sha(directory/'stderr'))
        stdout,stderr=(directory/'stdout').read_text(),(directory/'stderr').read_text()
        result=None
        try:
            require(not record['timed_out'],'Timeout siempre falla')
            require(proc.returncode==rc,'Código de salida inesperado')
            if error:
                require(error in stderr,'Falta diagnóstico esperado')
                if abort:require('MPI Abort(1)' in stderr,'Falta terminación MPI propia')
                if category=='check_failed': validate_contract(strict_json(stdout),config,'failed')
                else:require(not stdout.strip(),'Negativa imprimió resultado aparentemente válido')
            else:
                require(not stderr,'stderr inesperado')
                result=strict_json(stdout)
                if category=='worker':
                    require(result['passed'] and [result[k] for k in ('N','b','pr','pc')]==config,'Worker/config')
                    validate_contract(result['output'],config,'passed')
                    validate_contract(strict_json(result['raw_stdout']),config,'passed')
                    reports[name]=result;record['metrics']=result['metrics']
                    (directory/'result.json').write_text(result['raw_stdout'])
                    record.update(empty_ranks=sum(not math.prod(r['shape']) for r in result['ranks']),
                                  zero_count_calls=sum(e['count']==0 for r in result['ranks'] for e in r['events']),
                                  empty_lower=sum(s['lower_shape'][0]==0 for r in result['ranks'] for s in r['steps']),
                                  empty_upper=sum(s['upper_shape'][1]==0 for r in result['ranks'] for s in r['steps']))
                else:
                    validate_contract(result,config,'passed' if category=='cli_check' else 'not_run')
                    record.update(wall_s=result['wall_s'],check=result['check'])
                    if category=='cli_check': cli_checks[name]=result
            record['status']='passed'
        except Exception as exc:record.update(status='failed',error=str(exc))
        (directory/'record.json').write_text(json.dumps(record,indent=2)+'\n');save()
        print(f'{name}: {record["status"]}, rc={proc.returncode}',flush=True)
        return result
    def mpi(config,worker=False,scenario=None,check=False,actual_p=None):
        N,b,pr,pc=config
        cmd=[launcher,'-n',str(actual_p or pr*pc),python,'-B',
             'tests/test_robustness.py' if worker else 'lu_distributed.py',
             '--N',str(N),'--b',str(b),'--pr',str(pr),'--pc',str(pc)]
        if scenario:cmd+=['--scenario',scenario]
        if check:cmd+=['--check']
        return cmd
    matrix=[]
    for config_tuple in BASE+ADDITIONAL:
        config=list(config_tuple);N,b,pr,pc=config;name=f'N{N}_b{b}_{pr}x{pc}'
        origin='matriz base; reejecución Etapa 6' if config_tuple in BASE else 'adicional Etapa 6'
        if config_tuple==(16,16,2,2):origin+='; ya cubierto en Etapas 2–5'
        matrix.append(dict(name=name,N=N,b=b,pr=pr,pc=pc,p=pr*pc,origin=origin))
        run('worker_'+name,mpi(config,worker=True),pr*pc,'worker',config,origin=origin)
        run('cli_'+name,mpi(config,check=True),pr*pc,'cli_check',config,origin=origin)
    config=[72,8,2,3]
    run('repeat_N72_b8_2x3',mpi(config,worker=True),6,'worker',config,origin='repetición en lanzamiento separado')
    for config in ([16,16,1,1],[32,8,2,2],[16,8,4,4]):
        run('plain_'+'_'.join(map(str,config)),mpi(config),config[2]*config[3],'cli_plain',config)
    for scenario,message in FAILURES:
        run('failure_'+scenario,mpi([32,8,2,2],worker=True,scenario=scenario),4,'negative_local',
            [32,8,2,2],rc=1,error=message,abort=True)
    run('check_failed',mpi([32,8,2,2],worker=True,scenario='check_failed'),4,'check_failed',
        [32,8,2,2],rc=1,error='--check failed')
    run('bound_guard',mpi([32,8,2,2],worker=True,scenario='bound_guard'),4,'negative_parameters',
        [32,8,2,2],rc=2,error='--check requiere N <= 256')
    for config in ([10,4,1,1],[10,4,2,2]):
        run('nondivisible_'+str(config[2]),mpi(config),config[2]*config[3],'negative_parameters',
            config,rc=2,error='N debe ser divisible')
    for i,key in enumerate(('N','b','pr','pc')):
        for value in (0,-1):
            config=[16,8,1,2];config[i]=value
            run(f'invalid_{key}_{value}',mpi(config,actual_p=2),2,'negative_parameters',config,rc=2,error=key+' debe ser > 0')
    run('size_mismatch',mpi([16,8,2,2],actual_p=2),2,'negative_parameters',[16,8,2,2],rc=2,error='size=2 debe ser pr*pc=4')
    base=mpi([16,8,1,2])
    for name,tail,message in [('unknown',['--unknown'],'unrecognized arguments'),('incomplete',['--b'],'expected one argument'),
                              ('integer',['--N','abc'],'invalid int value')]:
        run(name,base+tail,2,'negative_parameters',rc=2,error=message)
    for index,change in enumerate((['--N','32'],['--b','4'],['--pr','2','--pc','1'],['--check'])):
        flags=['--N','16','--b','8','--pr','1','--pc','2']
        program=[python,'-B','lu_distributed.py']
        cmd=[launcher,'-n','1',*program,*flags,*change,':','-n','1',*program,*flags]
        run('mpmd_'+str(index),cmd,2,'negative_parameters',rc=2,error='idénticos')
    # Negative JSON cases are independent single-process consumers, not MPI algorithms.
    source=out/'plain_32_8_2_2/stdout'
    if source.exists() and next(r for r in summary['cases'] if r['name']=='plain_32_8_2_2')['status']=='passed':
        good=strict_json(source.read_text());fixtures=out/'invalid_json';fixtures.mkdir()
        for name,stream in invalid_documents(good):
            file=fixtures/(name+'.json');file.write_text(stream)
            run('json_'+name,[python,'-B','tests/run_robustness.py','--validate-json',str(file)],1,
                'negative_json',rc=2,error='Error de contrato JSON')
    if args.pilots:
        for config in ([512,64,2,2],[1024,128,2,2]):
            # Array/buffers are tiny relative to this conservative free-memory guard.
            available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                               if line.startswith('MemAvailable:')))*1024
            require(available>=512*1024**2,'Menos de 512 MiB disponibles: no lanzar piloto')
            run('pilot_N'+str(config[0]),mpi(config),4,'pilot',config,origin='funcional sin check; no acredita correctitud numérica')
    # Compare saved matrices, not hashes, checksums or execution times.
    summary['cross_comparisons']=cross_comparisons(reports)
    summary['cli_independent_checks']=[]
    for name,result in cli_checks.items():
        worker=reports.get('worker_'+name.removeprefix('cli_'))
        if worker is None:continue
        for field,key in [('relative_residual','distributed_residual'),('relative_factor_difference','relative_frobenius'),
                          ('max_absolute_factor_difference','max_absolute')]:
            a,b=result['check'][field],worker['metrics'][key]
            require(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-25),'Check CLI difiere del contraste independiente')
        summary['cli_independent_checks'].append(dict(name=name,passed=True))
    (out/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    summary['counts']={s:sum(r['status']==s for r in summary['cases']) for s in ('passed','failed','not_run')}
    save();print(json.dumps(summary['counts']))
    return int(bool(summary['counts']['failed'] or summary['counts']['not_run']))


if __name__=='__main__':raise SystemExit(main())
