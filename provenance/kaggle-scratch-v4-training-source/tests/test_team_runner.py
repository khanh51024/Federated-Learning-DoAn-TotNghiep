from pathlib import Path
import copy,json,sys,hashlib,yaml
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import team_runner as team
from fl_training.budget_state import BudgetSession,apply_usage_observations

@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    root=tmp_path_factory.mktemp('team-project')
    # Synthetic scheduling fixture: historic timings are NOT a calibration of
    # the current trainer. Keep the measured ledger immutable and explicitly
    # bind only this temporary test fixture to the source under test.
    spec=yaml.safe_load((ROOT/'configs/stage2_research.yaml').read_text())
    base=yaml.safe_load((ROOT/spec['base_config']).read_text())
    for section,values in spec.get('overrides',{}).items(): base[section].update(values)
    probe=[team.create_job(base,c,s,m,spec['budget']['max_rounds'],root/'probe')
           for c in spec['conditions'] for s in spec['seeds'] for m in team.MODES]
    keys=['total_hours','reserve_hours','session_hours','session_reserve_minutes','calibration_rounds','min_rounds','max_rounds','safety_factor']
    budget={k:spec['budget'][k] for k in sorted(keys)}; budget['session_hours']=7
    identity={'jobs':[{'id':j['id'],'semantic':j['cfg'].semantic_config_hash,'rounds':spec['budget']['max_rounds']} for j in probe],
              'budget':budget,'source':team.source_hash()}
    measured=json.loads((ROOT/'tests/fixtures/kaggle_calibration_20260914.json').read_text())
    measured.update(identity=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(), fixture_scope='synthetic_test_only')
    fixture=root/'synthetic_calibration.json'; fixture.write_text(json.dumps(measured))
    team.freeze_project(ROOT/'configs/stage2_research.yaml',fixture,['nguynhongnamkhnh','nifi1109','khanh51204','nifi51024'],root)
    return team.read_plan(root/'project_plan.json')

def test_historic_calibration_rejected_after_trainer_changes(tmp_path):
    with pytest.raises(ValueError,match='Calibration source/config/protocol mismatch'):
        team.freeze_project(ROOT/'configs/stage2_research.yaml',ROOT/'tests/fixtures/kaggle_calibration_20260914.json',['worker1'],tmp_path/'stale')

def test_static_complete_assignment(plan):
    assert len(plan['jobs'])==117 and len({j['id'] for j in plan['jobs']})==117
    assert plan['rounds']==10
    assert {j['seed'] for j in plan['jobs']}=={42,123,2026}
    assert set(plan['workers'])=={'nguynhongnamkhnh','nifi1109','khanh51204','nifi51024'}
    assert max(plan['worker_estimated_hours'].values())-min(plan['worker_estimated_hours'].values())<1
    assert plan['estimated_main_hours']==pytest.approx(164.61342754853982)

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1,31,True])
def test_invalid_quota_rejected(bad):
    with pytest.raises(ValueError):team.open_cycle({},'cycle1',bad)

def test_cycles_cumulative_no_reset_or_double_charge(tmp_path):
    state={'cycles':{}}; path=tmp_path/'state.json'; save=lambda:team.save_state(path,state)
    c=team.open_cycle(state,'q1',30)
    apply_usage_observations(c,charge_hours=1,charge_id='setup')
    apply_usage_observations(c,charge_hours=1,charge_id='setup')
    c['used_seconds']+=25*3600;save()
    with pytest.raises(ValueError):team.open_cycle(state,'q1',30,new_cycle=True)
    with pytest.raises(ValueError):team.open_cycle(state,'q2',30)
    new=team.open_cycle(state,'q2',30,new_cycle=True);new['used_seconds']=3600;save()
    assert state['cumulative_used_seconds']==27*3600 and c['closed']
    with pytest.raises(ValueError):team.open_cycle(state,'q1',30,new_cycle=True)
    assert BudgetSession(new,save,30,3,7,15,quota_remaining_hours=30).total_remaining()==26*3600

def test_interrupted_lease_requires_recovery(tmp_path):
    state={'active_cycle':'q1','cycles':{'q1':{'used_seconds':10,'active_lease':{'lease_id':'old','reserved_seconds':30,'charged_seconds':5,'heartbeat_unix':0}}}}
    with pytest.raises(ValueError):team.open_cycle(state,'q2',30,new_cycle=True)
    team.open_cycle(state,'q2',30,new_cycle=True,recover=True)
    assert state['cycles']['q1']['used_seconds']==35
    assert 'active_lease' not in state['cycles']['q1']

def test_plan_tamper_rejected(plan,tmp_path):
    bad=copy.deepcopy(plan);bad['rounds']=20;p=tmp_path/'bad.json';p.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='corrupt'):team.read_plan(p)

def test_wrong_worker_state_rejected(plan):
    state={'project_identity':plan['identity'],'worker':'nifi1109','jobs':{},'cycles':{}}
    with pytest.raises(ValueError):team.validate_state(state,plan,'khanh51204')
    other=next(j['id'] for j in plan['jobs'] if j['worker']!='nifi1109')
    state['jobs'][other]={}
    with pytest.raises(ValueError):team.validate_state(state,plan,'nifi1109')

def test_real_relocated_configs_identity_and_local_paths(plan,tmp_path):
    worker='nifi1109';jobs=team.worker_jobs(plan,worker,tmp_path,ROOT.parent/'PlantVillage-Dataset/raw/color',ROOT/'data/partitions_train_v3_research/index.json')
    assert all(j['cfg'].federation.max_rounds==10 for j in jobs)
    assert all(j['directory'].name=='local_only' for j in jobs if j['mode']=='local-only')
    changed=copy.deepcopy(plan);changed['jobs'][0]['semantic_hash']='bad'
    with pytest.raises(ValueError,match='mismatch'):team.worker_jobs(changed,changed['jobs'][0]['worker'],tmp_path/'bad')

def test_collect_cpu_missing_and_wrong_ledger_clears_stale(plan,tmp_path,monkeypatch):
    monkeypatch.setattr(team,'run_job',lambda *a,**k:pytest.fail('collector started training'))
    output=tmp_path/'comparison'; result=team.collect_team(plan,{},output)
    assert result['expected_jobs']==117 and result['missing']==117 and result['completed_jobs']==0
    (output/'comparison.json').write_text('[{"stale":true}]')
    wrong=tmp_path/'wrong';wrong.mkdir();(wrong/'team_state.json').write_text(json.dumps({'worker':'wrong','project_identity':'bad'}))
    with pytest.raises(ValueError):team.collect_team(plan,{'nifi1109':wrong},output)
    assert json.loads((output/'comparison.json').read_text())==[]
    assert json.loads((output/'collection_status.json').read_text())['invalid']==117

def test_resume_cycle_keeps_committed_job_state(plan,tmp_path,monkeypatch):
    calls=[]
    def run(job,session,state,save):
        calls.append(job['id'])
        entry=state['jobs'].setdefault(job['id'],{'attempts':0})
        if len(calls)==1:entry['status']='completed';save();return job['directory']
        if len(calls)==2:return None
        assert state['jobs'][calls[0]]['status']=='completed'
        return None
    monkeypatch.setattr(team,'run_job',run)
    monkeypatch.setattr(team,'collect_jobs',lambda *a,**k:{})
    assert team.run_worker(plan,'nifi1109',tmp_path,'q1',30,charge_hours=.1,charge_id='s1')==75
    before=json.loads((tmp_path/'team_state.json').read_text())['cumulative_used_seconds']
    assert team.run_worker(plan,'nifi1109',tmp_path,'q1',29.9,charge_hours=.1,charge_id='s1')==75
    after=json.loads((tmp_path/'team_state.json').read_text())
    assert before<=after['cumulative_used_seconds']<before+60
    assert after['jobs'][calls[0]]['status']=='completed'
    assert not (tmp_path/'runner.lock').exists()
