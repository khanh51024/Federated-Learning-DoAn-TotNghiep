"""Independent team workers, immutable 117-job protocol, durable per-quota-cycle usage.
No credentials are handled by this module. Each member runs their own assigned worker.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
os.environ['MPLBACKEND'] = 'Agg'
import yaml
from fl_training.budget_state import BudgetSession, atomic_json, apply_usage_observations, quota_remaining_from_state
from fl_training.config import load_training_config
from fl_training.kaggle_budget import ROOT, MODES, acquire_lock, release_lock, run_job, collect_jobs, source_hash
from fl_training.research_reporting import validate_research_spec
from fl_training.sweep import _condition_name


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def coordinator_hash():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def finite(value,name,maximum=None):
    if isinstance(value,bool): raise ValueError(f'{name} must be numeric')
    value=float(value)
    if not math.isfinite(value) or value<0 or (maximum is not None and value>maximum): raise ValueError(f'Invalid {name}')
    return value


def identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}',value): raise ValueError('Invalid worker/cycle id')
    return value


def create_job(base,condition,seed,mode,rounds,output,dataset_root=None,partition_index=None):
    raw=copy.deepcopy(base); raw['data'].update(alpha=None,quantity_alpha=None,feature_skew='none'); raw['data'].update(condition)
    if dataset_root: raw['data']['dataset_root']=str(Path(dataset_root).resolve())
    if partition_index: raw['data']['partition_index']=str(Path(partition_index).resolve())
    raw['training']['seed']=seed; raw['federation']['max_rounds']=rounds; raw['early_stopping']['enabled']=False
    ident=f'main/{_condition_name(condition)}__seed-{seed}/{mode}'
    directory=Path(output)/'main'/f'{_condition_name(condition)}__seed-{seed}'/mode.replace('-','_')
    raw['output'].update(root=str(directory.resolve()),progress=False,evaluate_test_after_train=True)
    config=Path(output)/'configs'/(ident.replace('/','__')+'.yaml'); config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(yaml.safe_dump(raw,sort_keys=False),encoding='utf-8')
    cfg=load_training_config(config,base_dir=ROOT,mode='train' if mode=='fedavg' else mode)
    return dict(id=ident,config=config,directory=directory,cfg=cfg,condition=condition,seed=seed,mode=mode)


def freeze_project(spec_path,calibration_path,workers,output,*,dataset_root=None,partition_index=None):
    output=Path(output)
    if (output/'project_plan.json').exists(): raise ValueError('Frozen plan exists; refusing overwrite')
    if not workers or len(set(workers))!=len(workers): raise ValueError('Unique workers required')
    workers=[identifier(w) for w in workers]
    spec=yaml.safe_load(Path(spec_path).read_text(encoding='utf-8')); validate_research_spec(spec)
    base=yaml.safe_load((ROOT/spec['base_config']).read_text(encoding='utf-8'))
    for section,values in spec.get('overrides',{}).items(): base[section].update(values)
    measured=json.loads(Path(calibration_path).read_text(encoding='utf-8')); timings=measured.get('calibration',{})
    if set(timings)!=set(MODES) or timings['local-only'].get('client_models_measured')!=10: raise ValueError('Three complete calibration methods required')
    entries=measured.get('jobs',{})
    if measured.get('rounds') is not None or len(entries)!=3 or any(not k.startswith('calibration/') or v.get('status')!='completed' for k,v in entries.items()): raise ValueError('Expected committed calibration-only ledger')
    for item in timings.values():
        if item.get('test_used_for_selection') is not False or finite(item['seconds_per_round'],'timing')<=0: raise ValueError('Invalid timing-only calibration')
        finite(item['fixed_seconds'],'overhead')
    output.mkdir(parents=True,exist_ok=True)
    probe=[create_job(base,c,s,m,spec['budget']['max_rounds'],output/'validation',dataset_root,partition_index) for c in spec['conditions'] for s in spec['seeds'] for m in MODES]
    keys=['total_hours','reserve_hours','session_hours','session_reserve_minutes','calibration_rounds','min_rounds','max_rounds','safety_factor']
    budget={k:spec['budget'][k] for k in sorted(keys)}; budget['session_hours']=7
    legacy={'jobs':[{'id':j['id'],'semantic':j['cfg'].semantic_config_hash,'rounds':spec['budget']['max_rounds']} for j in probe], 'budget':budget,'source':source_hash()}
    identity=hashlib.sha256(json.dumps(legacy,sort_keys=True).encode()).hexdigest()
    if identity!=measured['identity']: raise ValueError('Calibration source/config/protocol mismatch')
    jobs=[]; loads={w:0.0 for w in workers}
    for old in probe:
        j=create_job(base,old['condition'],old['seed'],old['mode'],10,output/'validation',dataset_root,partition_index); t=timings[j['mode']]
        jobs.append({'id':j['id'],'mode':j['mode'],'condition':j['condition'],'seed':j['seed'],'semantic_hash':j['cfg'].semantic_config_hash,'estimated_seconds':1.5*(t['fixed_seconds']+10*t['seconds_per_round'])})
    for job in sorted(jobs,key=lambda j:(-j['estimated_seconds'],j['id'])):
        worker=min(workers,key=lambda w:(loads[w],w)); job['worker']=worker; loads[worker]+=job['estimated_seconds']
    plan={'schema_version':1,'rounds':10,'workers':workers,'jobs':jobs,'base_config':base,'trainer_source_hash':source_hash(),'coordinator_hash':coordinator_hash(),'calibration_ledger_sha256':hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),'calibration_identity':identity,'calibration':timings,'estimated_main_hours':sum(loads.values())/3600,'worker_estimated_hours':{w:t/3600 for w,t in loads.items()},'budget_scope':'30h per account/quota cycle, 3h reserve; cumulative usage never resets','session_hours':7,'stop_new_work_minutes':15,'scientific_stage2_complete':False}
    plan['identity']=digest(plan); atomic_json(output/'project_plan.json',plan); return plan


def read_plan(path):
    plan=json.loads(Path(path).read_text(encoding='utf-8')); identity=plan.pop('identity',None)
    if digest(plan)!=identity: raise ValueError('Project plan corrupt or changed')
    plan['identity']=identity
    if plan['trainer_source_hash']!=source_hash() or plan['coordinator_hash']!=coordinator_hash(): raise ValueError('Trainer/coordinator source mismatch')
    if plan['rounds']!=10 or len(plan['jobs'])!=117 or len({j['id'] for j in plan['jobs']})!=117 or any(j['worker'] not in plan['workers'] for j in plan['jobs']): raise ValueError('Invalid research assignments')
    return plan


def worker_jobs(plan,worker,root,dataset_root=None,partition_index=None):
    if worker not in plan['workers']: raise ValueError('Unknown worker')
    jobs=[]
    for item in plan['jobs']:
        if item['worker']!=worker: continue
        job=create_job(plan['base_config'],item['condition'],item['seed'],item['mode'],plan['rounds'],root,dataset_root,partition_index)
        if job['id']!=item['id'] or job['cfg'].semantic_config_hash!=item['semantic_hash']: raise ValueError('Job data/config/protocol/budget mismatch')
        jobs.append(job)
    return jobs


def validate_state(state,plan,worker):
    if state.get('project_identity')!=plan['identity'] or state.get('worker')!=worker: raise ValueError('Wrong project/worker ledger')
    allowed={j['id'] for j in plan['jobs'] if j['worker']==worker}
    if set(state.get('jobs',{}))-allowed: raise ValueError('Ledger contains another worker job')
    for cycle in state.get('cycles',{}).values(): finite(cycle.get('used_seconds',0),'usage')
    return state


def save_state(path,state):
    state['cumulative_used_seconds']=sum(finite(c.get('used_seconds',0),'usage') for c in state['cycles'].values())
    atomic_json(path,state)


def open_cycle(state,cycle_id,quota,*,new_cycle=False,recover=False,save=lambda:None):
    identifier(cycle_id); finite(quota,'remaining quota',30)
    cycles=state.setdefault('cycles',{}); active=state.get('active_cycle')
    if active and active!=cycle_id:
        if not new_cycle: raise ValueError('Quota renewal requires explicit --new-cycle')
        if cycle_id in cycles: raise ValueError('Cannot reopen a closed cycle')
        old=cycles[active]
        if old.get('active_lease'):
            if not recover: raise ValueError('Interrupted lease requires recovery')
            BudgetSession(old,save,30,3,7,15,quota_remaining_hours=quota_remaining_from_state(old))
        old['closed']=True
    elif active==cycle_id and new_cycle: raise ValueError('Cannot reset active cycle')
    if cycle_id not in cycles: cycles[cycle_id]={'used_seconds':0.0,'external_charges':{}}
    state['active_cycle']=cycle_id
    return cycles[cycle_id]


def run_worker(plan,worker,output,cycle_id,quota,*,new_cycle=False,recover=False,charge_hours=0,charge_id=None,already_used_hours=None,dataset_root=None,partition_index=None):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    token=acquire_lock(output/'runner.lock',recover); state_path=output/'team_state.json'; session=None; state=None; jobs=[]
    try:
        if state_path.exists(): state=validate_state(json.loads(state_path.read_text()),plan,worker)
        else:
            if (output/'main').exists(): raise ValueError('Artifacts without ledger; restore complete output')
            state={'schema_version':1,'project_identity':plan['identity'],'worker':worker,'jobs':{},'cycles':{}}
        jobs=worker_jobs(plan,worker,output,dataset_root,partition_index)
        save=lambda:save_state(state_path,state)
        cycle=open_cycle(state,cycle_id,quota,new_cycle=new_cycle,recover=recover,save=save)
        apply_usage_observations(cycle,already_used_hours=already_used_hours,quota_remaining_hours=quota,charge_hours=charge_hours,charge_id=charge_id)
        save(); session=BudgetSession(cycle,save,30,3,7,15,quota_remaining_hours=quota_remaining_from_state(cycle))
        for job in jobs:
            print('Running/resuming',worker,job['id'],flush=True)
            if run_job(job,session,state,save) is None: state['status']='paused_quota_or_session'; save(); return 75
            collect_jobs(jobs,output/'comparison')
        state['status']='assigned_jobs_completed'; save(); return 0
    finally:
        try:
            if jobs and state is not None: collect_jobs(jobs,output/'comparison')
        finally:
            try:
                if session: session.tick()
            finally: release_lock(output/'runner.lock',token)


def _collect_team(plan,roots,output,*,dataset_root=None,partition_index=None):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    if set(roots)-set(plan['workers']): raise ValueError('Unknown worker mapping')
    all_jobs=[]; usage={}
    for worker in plan['workers']:
        root=Path(roots.get(worker,output/'missing_workers'/worker)).resolve(); state_path=root/'team_state.json'
        if state_path.exists():
            state=validate_state(json.loads(state_path.read_text()),plan,worker)
            usage[worker]=sum(finite(c.get('used_seconds',0),'usage') for c in state['cycles'].values())/3600
        elif (root/'main').exists(): raise ValueError(f'{worker}: artifacts without ledger')
        all_jobs.extend(worker_jobs(plan,worker,root,dataset_root,partition_index))
    status=collect_jobs(all_jobs,output)
    atomic_json(output/'team_usage.json',{'worker_hours':usage,'cumulative_worker_hours':sum(usage.values()),'missing_worker_ledgers':[w for w in plan['workers'] if w not in usage],'scientific_stage2_complete':False})
    return status



def collect_team(plan,roots,output,*,dataset_root=None,partition_index=None):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    # Never leave old charts visible when validation of a restored worker fails.
    collect_jobs([],output)
    try:
        return _collect_team(plan,roots,output,dataset_root=dataset_root,partition_index=partition_index)
    except Exception as exc:
        atomic_json(output/'collection_status.json',{'expected_jobs':len(plan['jobs']),'completed_jobs':0,
            'invalid':len(plan['jobs']),'matrix_complete':False,'scientific_stage2_complete':False,
            'error':str(exc),'jobs':[{'job':j['id'],'status':'invalid','reason':'Worker collection rejected; no results accepted'} for j in plan['jobs']]})
        atomic_json(output/'team_usage.json',{'status':'invalid','error':str(exc),'scientific_stage2_complete':False})
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['freeze','plan-worker','run','collect'])
    p.add_argument('--project'); p.add_argument('--config',default='configs/stage2_research.yaml'); p.add_argument('--calibration-ledger'); p.add_argument('--workers',nargs='+'); p.add_argument('--worker'); p.add_argument('--output-root',required=True)
    p.add_argument('--dataset-root'); p.add_argument('--partition-index'); p.add_argument('--quota-cycle'); p.add_argument('--quota-remaining-hours',type=float); p.add_argument('--already-used-hours',type=float)
    p.add_argument('--new-cycle',action='store_true'); p.add_argument('--recover',action='store_true'); p.add_argument('--charge-hours',type=float,default=0); p.add_argument('--charge-id'); p.add_argument('--worker-root',action='append',default=[])
    a=p.parse_args(argv); paths={'dataset_root':a.dataset_root,'partition_index':a.partition_index}
    if a.action=='freeze':
        if not a.calibration_ledger or not a.workers:p.error('freeze requires --calibration-ledger and --workers')
        plan=freeze_project(a.config,a.calibration_ledger,a.workers,a.output_root,**paths); print(json.dumps({'identity':plan['identity'],'jobs':len(plan['jobs']),'rounds':plan['rounds'],'worker_estimated_hours':plan['worker_estimated_hours']},indent=2));return 0
    if not a.project:p.error('--project required')
    plan=read_plan(a.project)
    if a.action=='collect':
        roots={}
        for item in a.worker_root:
            w,sep,path=item.partition('=')
            if not sep or not path or w in roots:raise ValueError('Invalid/duplicate worker root')
            roots[w]=path
        print(json.dumps(collect_team(plan,roots,a.output_root,**paths),indent=2));return 0
    if not a.worker:p.error('--worker required')
    if a.action=='plan-worker':
        jobs=worker_jobs(plan,a.worker,Path(a.output_root).resolve(),**paths);print(f'Validated {len(jobs)} assigned jobs at {plan["rounds"]} rounds; no training started');return 0
    if not a.quota_cycle or a.quota_remaining_hours is None:p.error('run requires --quota-cycle and --quota-remaining-hours')
    return run_worker(plan,a.worker,a.output_root,a.quota_cycle,a.quota_remaining_hours,new_cycle=a.new_cycle,recover=a.recover,charge_hours=a.charge_hours,charge_id=a.charge_id,already_used_hours=a.already_used_hours,**paths)

if __name__=='__main__':raise SystemExit(main())
