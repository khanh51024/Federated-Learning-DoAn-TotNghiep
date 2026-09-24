"""Check the actual scratch entrypoint: new weights, no silent reuse, exact resume."""
from dataclasses import replace
import json

import pytest
import torch

from stage2_scratch import experiment as scratch
from stage1_compat.integrity import atomic_json
from stage1_compat.checkpoint import load_checkpoint
from tests.test_stage1_compat_runner import actual_runner_fixture


def test_real_model_random_init_never_downloads_and_has_independent_storage(monkeypatch):
    import stage1_compat.models as models
    def forbidden(*args, **kwargs):
        raise AssertionError('Scratch attempted to load pretrained weights')
    monkeypatch.setattr(models.MobileNet_V3_Small_Weights, 'get_state_dict', forbidden)
    torch.manual_seed(42)
    a = models.create_mobilenetv3_stage1(pretrained=False)
    torch.manual_seed(42)
    b = models.create_mobilenetv3_stage1(pretrained=False)
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))
    with torch.no_grad():
        next(a.parameters()).add_(100)
    assert not torch.equal(next(a.parameters()),next(b.parameters()))
    torch.manual_seed(123)
    c = models.create_mobilenetv3_stage1(pretrained=False)
    assert not torch.equal(next(b.parameters()),next(c.parameters()))


@pytest.fixture
def scratch_runner(actual_runner_fixture, monkeypatch, tmp_path):
    runner, factory, train, val, parts, job = actual_runner_fixture
    original = scratch.make_job
    monkeypatch.setattr(scratch, 'make_job', lambda n, s, smoke=False, *a, **kw:
                        replace(original(n, s, smoke, *a, **kw), num_classes=4, batch_size=4, rounds=3))
    def checked_factory(num_classes=4,pretrained=False):
        assert pretrained is False
        return factory(num_classes,False)
    monkeypatch.setattr(scratch,'create_mobilenetv3_stage1',checked_factory)
    monkeypatch.setattr(runner,'create_mobilenetv3_stage1',checked_factory)
    import stage2_scratch.runner as scratch_runner_mod
    monkeypatch.setattr(scratch_runner_mod, 'create_mobilenetv3_stage1', checked_factory)
    suite = {'smoke':False,'common':{'train':'train','val':'val','test':'test'},
             'class_names':[str(i) for i in range(4)],
             'conditions':{n:{'manifest_sha256':n} for n in ('label100','label01')}}
    for name in suite['conditions']:
        atomic_json(tmp_path/'suite'/name/'partition_config.json',
                    {'domain_profiles':[],'domain_kind':'none','domain_assignment_sha256':'same'})
    rows = [{'client_id':i//8} for i in range(40)]
    monkeypatch.setattr(scratch,'read_rows',lambda p:rows if p.name=='centralized_train.csv' else [])
    monkeypatch.setattr(scratch,'preflight',lambda *a:(suite,{}))
    # Return independent dataset objects, as real ManifestDataset does per job.
    import copy
    monkeypatch.setattr(scratch,'ManifestDataset',lambda rows,root,training=False,*a:
                        copy.deepcopy(train if training else val))
    def run(path,resume=False,conditions=None):
        return scratch.run(tmp_path/'suite',tmp_path,path,conditions or ['label100','label01'],
                           [42],device='cpu',resume=resume)
    return run,suite


def test_fresh_runs_reset_between_conditions_and_reject_existing_results(scratch_runner,tmp_path):
    run,suite=scratch_runner
    output=tmp_path/'new'
    assert all(r['status']=='COMPLETED' for r in run(output))
    report=scratch.collect(output,tmp_path/'suite')
    assert len(report['completed_jobs'])==2 and len(report['paired_deltas'])==1
    metrics=[json.loads((output/f'{n}_seed42/fedavg_metrics.json').read_text())
             for n in ('label100','label01')]
    assert metrics[0]['identity']['context']['initialization_sha256']==metrics[1]['identity']['context']['initialization_sha256']
    assert metrics[0]['checkpoint_sha256']!=metrics[0]['identity']['context']['initialization_sha256']
    assert metrics[0]['history']==metrics[1]['history']  # same synthetic data, fresh initialization
    before=(output/'label100_seed42/fedavg_metrics.json').read_bytes()
    with pytest.raises(FileExistsError,match='NEW output'):
        run(output)
    assert (output/'label100_seed42/fedavg_metrics.json').read_bytes()==before
    assert all(r['status']=='COMPLETED' for r in run(output,resume=True))
    assert (output/'label100_seed42/fedavg_metrics.json').read_bytes()==before


def test_explicit_scratch_resume_matches_uninterrupted(scratch_runner,tmp_path,monkeypatch):
    run,_=scratch_runner
    full,partial=tmp_path/'full',tmp_path/'partial'
    run(full,conditions=['label100'])
    original=scratch.BudgetLedger.can_continue
    calls=[]
    def once(self,**kwargs):
        calls.append(1)
        return len(calls)==1,'test pause'
    monkeypatch.setattr(scratch.BudgetLedger,'can_continue',once)
    assert run(partial,conditions=['label100'])[0]['status']=='PAUSED_QUOTA'
    monkeypatch.setattr(scratch.BudgetLedger,'can_continue',original)
    with pytest.raises(FileExistsError):
        run(partial,conditions=['label100'])
    run(partial,resume=True,conditions=['label100'])
    def checkpoint(root):
        return load_checkpoint(root/'label100_seed42/checkpoints/label100_seed42_checkpoint.pth')
    a,b=checkpoint(full),checkpoint(partial)
    assert a['history']==b['history']
    for key in a['global_model_state']:
        assert torch.equal(a['global_model_state'][key],b['global_model_state'][key])


def test_reject_old_protocol_and_changed_code_before_resume(scratch_runner,tmp_path):
    run,_=scratch_runner
    old=tmp_path/'old'; old.mkdir()
    atomic_json(old/'matched_protocol.json',{'pretrained':True})
    with pytest.raises(ValueError,match='same scratch protocol'):
        run(old,resume=True)
    output=tmp_path/'new'
    run(output,conditions=['label100'])
    p=output/'scratch_protocol.json'
    data=json.loads(p.read_text()); data['source_sha256']='changed'
    atomic_json(p,data)
    with pytest.raises(ValueError,match='same scratch protocol'):
        run(output,resume=True)


def test_scratch_collector_rejects_historical_pretrained():
    from stage1_compat.config import JobConfig
    job = JobConfig(job_id="label100_seed42", seed=42, rounds=10, pretrained=True)
    fake_result = {
        "resolved_config": job.to_dict(),
        "test_metrics": {"accuracy": 0.99, "macro_f1": 0.99},
        "identity": {
            "seed": 42, "runtime": {"torch": "test"},
            "context": {
                "scope": "stage2_matched_clean_v5", "smoke": False, "condition": "label100",
                "common": {"test": "same"}, "matched_source_sha256": "source",
                "matched_suite_sha256": "suite", "initialization_sha256": "initialization",
                "train_device": "cpu", "evaluation_device": "cpu"
            }
        }
    }
    with pytest.raises(ValueError, match='scratch results'):
        scratch.comparison_key(fake_result)
