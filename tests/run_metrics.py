"""Sequential Stage 5 suite; stdout/stderr and commands preserved per attempt."""
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
import uuid

from run_distribution import ROOT, THREAD_ENV, sha

CASES = [(16,16,1,1),(32,8,1,1),(32,8,1,2),(32,8,2,1),(32,8,2,2),
         (48,8,2,3),(48,8,3,2),(16,8,2,4),(16,8,4,2),(16,8,4,4),
         (16,16,2,2),(32,4,2,2),(8,1,2,2)]


def require(condition, message):
    if not condition: raise ValueError(message)


def strict_json(stream):
    def bad(value): raise ValueError('Non-standard JSON number: '+value)
    def pairs(items):
        result = {}
        for key,value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(stream, parse_constant=bad, object_pairs_hook=pairs)


def validate_output(result, config=None, check_status=None):
    """Independent consumer, intentionally does not call production validators."""
    fields = ('schema_version','run_id','started_at','code','environment',
              'N','b','pr','pc','p','wall_s','rank_times','check')
    require(type(result) is dict and all(k in result for k in fields), 'Missing schema fields')
    require(result['schema_version'] == '1.0', 'Schema version')
    uuid.UUID(result['run_id'])
    require(datetime.fromisoformat(result['started_at']).tzinfo is not None, 'Timestamp timezone')
    for k in ('N','b','pr','pc','p'):
        require(type(result[k]) is int and result[k]>0, 'Positive integer '+k)
    N,b,pr,pc,p = [result[k] for k in ('N','b','pr','pc','p')]
    require(p==pr*pc and N%b==0, 'Configuration/product')
    if config is not None: require([N,b,pr,pc]==list(config), 'Requested configuration')
    for name in ('lu_distributed.py','lu_serial.py'):
        digest = result['code']['sha256'][name]
        require(type(digest) is str and len(digest)==64 and all(c in '0123456789abcdef' for c in digest), 'Source hash')
        require(digest==sha(ROOT/name), 'Source hash differs from tested source')
    require(type(result['code']['git_dirty']) in (bool,type(None)), 'git_dirty type')
    for k in ('python','executable','numpy','scipy','mpi4py','mpi_library','platform','thread_policy','clock','clock_resolution_s','time_tolerance_s'):
        require(k in result['environment'], 'Missing environment '+k)
    require(result['environment']['clock']=='time.perf_counter', 'Clock')
    require(result['environment']['time_tolerance_s']==1e-9, 'Timing tolerance')
    records = result['rank_times']
    require(type(records) is list and len(records)==p, 'Rank count')
    for rank,r in enumerate(records):
        require(type(r['rank']) is int and r['rank']==rank, 'Unique ascending ranks')
        require(type(r['row']) is int and type(r['col']) is int and (r['row'],r['col'])==divmod(rank,pc), 'Coordinates')
        for k in ('compute_s','communication_s','total_s','unaccounted_s'):
            require(type(r[k]) in (float,int) and math.isfinite(r[k]), 'Finite numeric '+k)
            require(r[k]>=(-1e-9 if k=='unaccounted_s' else 0), 'Negative '+k)
        overhead = r['total_s']-r['compute_s']-r['communication_s']
        require(overhead>=-1e-9 and r['unaccounted_s']==overhead, 'Timing partition')
        require(type(r['hostname']) is str and r['hostname'], 'Hostname')
        require(r['thread_env']=={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}, 'Requested threads')
        require(r['blas_pools'] and all(pool['num_threads']==1 for pool in r['blas_pools']), 'Observed BLAS threads')
    require(type(result['wall_s']) in (float,int) and math.isfinite(result['wall_s']) and result['wall_s']==max(r['total_s'] for r in records), 'Wall maximum')
    require(any(r['total_s'] != round(r['total_s'],3) for r in records), 'Premature rounding')
    check = result['check']
    require(check['status'] in ('not_run','passed','failed'), 'Check state')
    if check_status: require(check['status']==check_status, 'Expected check state')
    if check['status']=='not_run': require(check==dict(status='not_run'), 'Invented check metrics')
    else:
        require(N<=256 and check['max_N']==256, 'Check bound')
        for k in ('relative_residual','relative_factor_difference','max_absolute_factor_difference','residual_tolerance','factor_tolerance'):
            require(type(check[k]) in (float,int) and math.isfinite(check[k]) and check[k]>=0, 'Check numeric '+k)
        require(check['residual_tolerance']==check['factor_tolerance']==1e-10, 'Numerical tolerances')
        passed = check['relative_residual']<=1e-10 and check['relative_factor_difference']<=1e-10
        require(passed==(check['status']=='passed'), 'Check criteria/state mismatch')
    return result


def negative_schema_tests(good):
    tested = []
    def reject(name, function):
        try: function()
        except (ValueError,KeyError,TypeError,AttributeError): tested.append(name); return
        raise ValueError('Invalid output accepted: '+name)
    for stream in ('{} {}','{','{"x":NaN}','{"x":Infinity}','{"x":1,"x":2}'):
        reject('JSON '+stream, lambda: strict_json(stream))
    for key in good:
        bad=copy.deepcopy(good); bad.pop(key)
        reject('missing '+key, lambda: validate_output(bad))
    for field,value in [('compute_s',-1.0),('communication_s',float('nan')),('total_s',float('inf')),
                        ('compute_s',True),('total_s','0.1'),('compute_s',1e6),('rank',99),('row',99),('unaccounted_s',-1.0)]:
        bad=copy.deepcopy(good); bad['rank_times'][0][field]=value
        reject(f'rank {field}={value}', lambda: validate_output(bad))
    for key,value in [('wall_s',0),('p',99),('schema_version','0'),('rank_times',[])]:
        bad=copy.deepcopy(good); bad[key]=value
        reject('top '+key, lambda: validate_output(bad))
    return tested


def capacity():
    count=len(os.sched_getaffinity(0))
    for name in ('/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu/cpu.cfs_quota_us','/sys/fs/cgroup/cpu/init.scope/cpu.cfs_quota_us'):
        path=Path(name)
        if not path.exists(): continue
        words=path.read_text().split()
        if name.endswith('cpu.max'):
            if words[0]!='max': count=min(count,int(words[0])//int(words[1]))
        elif int(words[0])>0:
            count=min(count,int(words[0])//int(path.with_name('cpu.cfs_period_us').read_text()))
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-processes',type=int,default=capacity())
    parser.add_argument('--timeout',type=float,default=40)
    parser.add_argument('--mpiexec',type=Path,default=Path(sys.executable).parent/'mpiexec')
    args=parser.parse_args()
    if not 1<=args.max_processes<=capacity(): parser.error('Sin oversubscription')
    if not 0<args.timeout<=120: parser.error('Timeout entre 0 y 120')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy(); env.update(THREAD_ENV)
    launcher=str(args.mpiexec.resolve()); python=str(Path(sys.executable).absolute())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'lu_serial.py',ROOT/'lu_distributed.py',*sorted((ROOT/'tests').glob('*.py'))]}
    summary=dict(kind='etapa_05_metrics_suite', command=[sys.executable,*sys.argv], environment=THREAD_ENV,
                 capacity=capacity(), affinity=sorted(os.sched_getaffinity(0)), hashes=hashes, cases=[])
    def save(): (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    def run(name,config,worker=False,scenario=None,check=False,rc=0,error=None,extra_cmd=None):
        N,b,pr,pc=config
        flags=['--N',str(N),'--b',str(b),'--pr',str(pr),'--pc',str(pc)]
        cmd=extra_cmd or [launcher,'-n',str(pr*pc),python,'-B','tests/test_metrics.py' if worker else 'lu_distributed.py',*flags]
        if not extra_cmd:
            if scenario: cmd+=['--scenario',scenario]
            if check: cmd+=['--check']
        record=dict(name=name, configuration=list(config),command=cmd,cwd=str(ROOT),
                    shell_command=shlex.join(['env',*[f'{k}={v}' for k,v in THREAD_ENV.items()],*cmd]),
                    started=datetime.now().astimezone().isoformat(),timeout_s=args.timeout,timed_out=False,status='pending')
        summary['cases'].append(record)
        if pr*pc>args.max_processes:
            record.update(status='not_run'); save(); return
        directory=out/name; directory.mkdir()
        require(all(sha(ROOT/p)==h for p,h in hashes.items()), 'Code changed during suite')
        save(); start=time.monotonic()
        with (directory/'stdout').open('xb') as stdout,(directory/'stderr').open('xb') as stderr:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            try: proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                record['timed_out']=True; os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        record.update(returncode=proc.returncode,duration_s=time.monotonic()-start,
                      stdout_sha256=sha(directory/'stdout'),stderr_sha256=sha(directory/'stderr'))
        stdout,stderr=(directory/'stdout').read_text(),(directory/'stderr').read_text()
        try:
            require(not record['timed_out'], 'Timeout always fails')
            require(proc.returncode==rc, 'Unexpected returncode')
            if error:
                require(error in stderr, 'Expected diagnostic missing')
                if scenario=='corrupt_factors': validate_output(strict_json(stdout),config,'failed')
                else: require(not stdout.strip(), 'Invalid execution emitted result')
            else:
                require(not stderr, 'Unexpected stderr')
                result=strict_json(stdout)
                if worker:
                    require(result['passed'], 'Worker failed')
                    if scenario not in ('scope','validators'):
                        require([result[k] for k in ('N','b','pr','pc')]==list(config), 'Worker config')
                        record.update(check=result['check'], ranks=len(result['ranks']),
                                      bcast_calls=sum(len(r['real']['events']) for r in result['ranks']),
                                      zero_count_calls=sum(e['count']==0 for r in result['ranks'] for e in r['real']['events']))
                else:
                    validate_output(result,config,'passed' if check else 'not_run')
                    record['wall_s']=result['wall_s']; record['check']=result['check']
                    (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                    if name=='cli_0_plain':
                        summary['schema_negative_tests']=negative_schema_tests(result)
            record['status']='passed'
        except Exception as exc: record.update(status='failed',error=str(exc))
        (directory/'record.json').write_text(json.dumps(record,indent=2)+'\n'); save()
        print(f'{name}: {record["status"]}, rc={proc.returncode}',flush=True)
    for index,config in enumerate(CASES):
        run(f'worker_{index}',config,worker=True)
        run(f'cli_{index}_plain',config)
        run(f'cli_{index}_check',config,check=True)
    run('check_at_limit',(256,16,2,2),check=True)
    run('scope_main',(32,8,2,2),worker=True,scenario='scope')
    run('production_validators',(16,8,1,2),worker=True,scenario='validators')
    for scenario,error in [('corrupt_factors','--check failed'),('reconstruct_error','Reconstrucción: shape'),
                           ('check_error','fallo controlado de comprobación'),('invalid_time','exceden total_s')]:
        run(scenario,(32,8,2,2),worker=True,scenario=scenario,rc=1,error=error)
    run('check_over_limit',(272,16,1,2),check=True,rc=2,error='--check requiere N <= 256')
    # A mismatch involving check must fail before any matrix allocation.
    program=[python,'-B','lu_distributed.py']
    flags=['--N','16','--b','8','--pr','1','--pc','2']
    run('mpmd_check',(16,8,1,2),rc=2,error='idénticos',extra_cmd=[launcher,'-n','1',*program,*flags,'--check',':','-n','1',*program,*flags])
    summary['counts']={s:sum(r['status']==s for r in summary['cases']) for s in ('passed','failed','not_run')}
    save(); print(json.dumps(summary['counts']))
    return int(bool(summary['counts']['failed'] or summary['counts']['not_run']))


if __name__=='__main__': raise SystemExit(main())
