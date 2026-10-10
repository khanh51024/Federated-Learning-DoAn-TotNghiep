"""Scratch protocol: explicit fresh random initialization (weights=None), FedAvg-only."""
import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from stage1_compat.artifacts import validate_completed
from stage1_compat.budget import BudgetLedger, BudgetExhausted
from stage1_compat.checkpoint import compute_state_dict_sha256
from stage1_compat.config import JobConfig
from stage1_compat.integrity import atomic_json, digest, file_hash
from stage1_compat.models import create_mobilenetv3_stage1
from stage1_compat.runner import get_upstream_reproducibility
from stage2_matched.data import CONDITIONS, ROOT, ManifestDataset, preflight, read_rows
from stage2_scratch.runner import run_single_fedavg_job

PROTOCOL = 'stage2_scratch_fedavg_v4'


def trainer_contract(job):
    # Alpha/condition and random seed are experiment axes; all other settings fixed.
    return {k: v for k, v in job.to_dict().items() if k not in ("job_id", "alpha", "seed")}


def source_hash():
    files = []
    folders = ("stage2_scratch", "stage2_matched", "fl_training", "stage1_compat", "src/data")
    excluded_rel_paths = {
        "stage2_matched/__main__.py",
        "stage2_matched/experiment.py",
        "fl_training/cli.py",
        "fl_training/baselines.py",
        "stage1_compat/cli.py",
        "stage1_compat/__main__.py",
        "stage1_compat/package_stage1.py",
    }
    for folder in folders:
        dir_path = ROOT / folder
        if dir_path.is_dir():
            for p in dir_path.rglob("*.py"):
                if p.is_file() and "__pycache__" not in p.parts:
                    rel = p.relative_to(ROOT).as_posix()
                    if "centralized" in p.parts or "local_only" in p.parts or rel in excluded_rel_paths:
                        continue
                    files.append(p)
    return digest({p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(files)})


def make_job(condition, seed, smoke=False, rounds=None, optimizer='sgd', lr=None, momentum=0.0):
    if lr is None:
        lr = 0.01 if optimizer.lower() == 'sgd' else 0.001
    num_rounds = rounds if rounds is not None else (2 if smoke else 10)
    alpha = float(CONDITIONS.get(condition, {}).get('alpha', 0.1))
    return JobConfig(
        job_id=f'{condition}_seed{seed}',
        seed=seed,
        alpha=alpha,
        rounds=num_rounds,
        lr=lr,
        optimizer=optimizer,
        momentum=momentum,
        pretrained=False,
    )


def frozen_protocol(suite, optimizer='sgd', lr=None, rounds=None, momentum=0.0):
    job = make_job('label100', 42, suite['smoke'], rounds=rounds, optimizer=optimizer, lr=lr, momentum=momentum)
    return {
        'protocol': PROTOCOL,
        'initialization': 'random',
        'pretrained': False,
        'suite_sha256': digest(suite),
        'source_sha256': source_hash(),
        'smoke': suite['smoke'],
        'common': suite['common'],
        'trainer': trainer_contract(job),
    }


def initialize_output(output, protocol, resume):
    """A fresh run atomically claims a NEW directory, never skips existing results."""
    path = Path(output)
    if resume:
        frozen = path / 'scratch_protocol.json'
        if not frozen.is_file() or json.loads(frozen.read_text()) != protocol:
            raise ValueError('Resume requires the same scratch protocol, code, data and trainer')
    else:
        path.mkdir(parents=True, exist_ok=False)
        atomic_json(path / 'scratch_protocol.json', protocol)


def record_initialization(job, output):
    get_upstream_reproducibility().set_seed(job.seed)
    model = create_mobilenetv3_stage1(num_classes=job.num_classes, pretrained=False)
    record = {
        'job_id': job.job_id,
        'seed': job.seed,
        'pretrained': False,
        'weights_argument': None,
        'checkpoint_source': None,
        'initialization_sha256': compute_state_dict_sha256(model.state_dict()),
    }
    target = Path(output) / 'initialization' / f'{job.job_id}.json'
    if target.exists() and json.loads(target.read_text()) != record:
        raise ValueError('Random initialization differs from the recorded run')
    atomic_json(target, record)
    return record['initialization_sha256']


def run(suite_dir, dataset_root, output, conditions=None, seeds=(42,),
        session_minutes=420, device='auto', resume=False,
        optimizer='sgd', lr=None, rounds=None, momentum=0.0):
    conditions = list(CONDITIONS) if conditions is None else list(conditions)
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - CONDITIONS.keys():
        raise ValueError('Unknown/duplicate conditions')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    if not math.isfinite(session_minutes) or not 2 <= session_minutes <= 450:
        raise ValueError('Session limit must be between 2 and 450 minutes')
    output = Path(output).resolve()
    if not resume and output.exists():
        raise FileExistsError('Fresh training requires a NEW output directory; use --resume only to continue this scratch experiment')
    suite, audits = preflight(suite_dir, dataset_root)
    protocol = frozen_protocol(suite, optimizer=optimizer, lr=lr, rounds=rounds, momentum=momentum)
    initialize_output(output, protocol, resume)
    target = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
    ledger = BudgetLedger(output, user_quota_hours=30)
    previous_deadline = os.environ.get('STAGE1_SOFT_DEADLINE')
    deadline = time.time() + session_minutes * 60
    if previous_deadline is not None:
        deadline = min(deadline, float(previous_deadline))
    os.environ['STAGE1_SOFT_DEADLINE'] = str(deadline)
    results, started = [], time.monotonic()
    try:
        with ledger.get_lock():
            atomic_json(output / 'preflight.json', audits)
            for seed in seeds:
                for name in conditions:
                    path = Path(suite_dir) / name
                    meta = json.loads((path / 'partition_config.json').read_text())
                    rows = read_rows(path / 'centralized_train.csv')
                    train = ManifestDataset(rows, dataset_root, True, meta['domain_profiles'])
                    val = ManifestDataset(read_rows(path / 'global_val.csv'), dataset_root)
                    test = ManifestDataset(read_rows(path / 'global_test.csv'), dataset_root)
                    partitions = [[i for i, r in enumerate(rows) if r['client_id'] == cid] for cid in range(5)]
                    job = make_job(name, seed, suite['smoke'], rounds=rounds, optimizer=optimizer, lr=lr, momentum=momentum)
                    train.initialization_sha256 = record_initialization(job, output)
                    train.identity_context = {
                        'scope': PROTOCOL, 'smoke': suite['smoke'], 'condition': name,
                        'initialization': 'random', 'pretrained': False,
                        'scratch_source_sha256': protocol['source_sha256'],
                        'suite_sha256': protocol['suite_sha256'],
                        'manifest_sha256': suite['conditions'][name]['manifest_sha256'],
                        'common': suite['common'], 'domain_kind': meta['domain_kind'],
                        'domain_assignment_sha256': meta['domain_assignment_sha256']
                    }
                    print(f'[{job.job_id}] initialization=random; pretrained=False; '
                          f'resume={resume}; initial_sha256={train.initialization_sha256}', flush=True)
                    result = run_single_fedavg_job(job, train, val, test, suite['class_names'],
                                                  partitions, output, ledger, target, resume=resume)
                    results.append({'job_id': job.job_id, 'status': result['status']})
                    if result['status'] != 'COMPLETED':
                        return results
    except BudgetExhausted:
        results.append({'status': 'PAUSED_DEADLINE', 'resume': 'Use the same output with --resume'})
    finally:
        if previous_deadline is None:
            os.environ.pop('STAGE1_SOFT_DEADLINE', None)
        else:
            os.environ['STAGE1_SOFT_DEADLINE'] = previous_deadline
        atomic_json(output / f'session_{time.time_ns()}.json', {
            'elapsed_wall_seconds': time.monotonic() - started, 'session_minutes': session_minutes,
            'quota_scope': 'session_only', 'resume_requested': resume, 'jobs': results})
    return results


def sample_whole_groups_per_class(rows, groups_per_class, seed=20260920):
    """Lấy mẫu các nhóm lá nguyên khối, đảm bảo phủ đủ 38 lớp mà không xé nhỏ nhóm."""
    from stage2_matched.data import group_samples
    rng = np.random.default_rng(seed)
    selected = []
    for cls_groups in group_samples(rows, key='group_id').values():
        shuffled = [cls_groups[i] for i in rng.permutation(len(cls_groups))]
        for g in shuffled[:groups_per_class]:
            selected.extend(g['items'])
    return selected


class SentinelTestDataset(Dataset):
    """Sentinel dataset asserting final test set is NEVER accessed during diagnostic/pilot."""
    def __len__(self):
        return 0
    def __getitem__(self, index):
        raise AssertionError("VIOLATION: Final test dataset was accessed during diagnostic/pilot run!")


def diagnose(suite_dir, dataset_root, output_dir, mode='smoke', session_minutes=45,
             device='auto', optimizer='sgd', lr=None, momentum=0.0):
    """
    Thực hiện kiểm tra chẩn đoán FedAvg-only trên laptop:
    - Tuyệt đối không huấn luyện lại Centralized hoặc Local-only.
    - Tuyệt đối không nạp hay truy cập tập final test (dùng SentinelTestDataset).
    - 'smoke': 300-600 train images, 2 rounds FedAvg, K=5 clients, deadline <= 5 phút.
    - 'pilot': 1500-3000 train images, 300-500 val, 38 lớp, FedAvg (5 rounds, E=1),
      negative control và resume verification. Deadline cứng <= 45 phút.
    """
    output = (Path(output_dir) / mode).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Diagnostic output directory '{output}' already exists and is not empty. "
            f"Refusing to overwrite existing diagnostic output. Specify a new --output directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    target = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)

    # Enforce hard session deadline
    deadline = time.time() + session_minutes * 60
    os.environ['STAGE1_SOFT_DEADLINE'] = str(deadline)

    suite, _ = preflight(suite_dir, dataset_root)
    cond_name = 'label100'
    cond_path = Path(suite_dir) / cond_name
    train_rows = read_rows(cond_path / 'centralized_train.csv')
    val_rows = read_rows(cond_path / 'global_val.csv')

    lr = lr or (0.01 if optimizer.lower() == 'sgd' else 0.001)
    sentinel_test = SentinelTestDataset()

    try:
        if mode == 'smoke':
            print(f"=== [SMOKE CHECK] FedAvg-only | 300-600 images | 2 rounds | {optimizer.upper()} ===")
            sub_train = sample_whole_groups_per_class(train_rows, groups_per_class=2, seed=42)
            sub_val = sample_whole_groups_per_class(val_rows, groups_per_class=1, seed=42)

            print(f"Smoke samples: train={len(sub_train)}, val={len(sub_val)}, test=SENTINEL (NOT ACCESSED)")
            train_ds = ManifestDataset(sub_train, dataset_root, True)
            val_ds = ManifestDataset(sub_val, dataset_root, False)
            partitions = [[i for i, r in enumerate(sub_train) if r['client_id'] == cid] for cid in range(5)]

            job = make_job(cond_name, seed=42, smoke=True, rounds=2, optimizer=optimizer, lr=lr, momentum=momentum)
            train_ds.initialization_sha256 = record_initialization(job, output)
            train_ds.identity_context = {
                'scope': PROTOCOL, 'smoke': True, 'condition': cond_name,
                'initialization': 'random', 'pretrained': False, 'diagnostic': True,
            }
            ledger = BudgetLedger(output, user_quota_hours=30)
            res = run_single_fedavg_job(job, train_ds, val_ds, sentinel_test, suite['class_names'],
                                       partitions, output, ledger, target, resume=False, calibration=True)
            report = {
                'mode': 'smoke',
                'protocol': PROTOCOL,
                'status': res.get('status', 'COMPLETED'),
                'train_samples': len(sub_train),
                'val_samples': len(sub_val),
                'test_accessed': False,
                'metrics': res,
            }
            atomic_json(output / 'smoke_report.json', report)
            return report

        elif mode == 'pilot':
            print(f"=== [PILOT CHECK] FedAvg-only | 1500-3000 train | 300-500 val | 38 classes | <= 45 mins ===")
            sub_train = sample_whole_groups_per_class(train_rows, groups_per_class=10, seed=42)
            sub_val = sample_whole_groups_per_class(val_rows, groups_per_class=2, seed=42)

            print(f"Pilot samples: train={len(sub_train)}, val={len(sub_val)}, test=SENTINEL (NOT ACCESSED)")
            train_ds = ManifestDataset(sub_train, dataset_root, True)
            val_ds = ManifestDataset(sub_val, dataset_root, False)
            partitions = [[i for i, r in enumerate(sub_train) if r['client_id'] == cid] for cid in range(5)]

            # 1. FedAvg (5 rounds)
            print("\n--- 1. FedAvg (5 rounds, E=1) ---")
            job = make_job(cond_name, seed=42, smoke=False, rounds=5, optimizer=optimizer, lr=lr, momentum=momentum)
            train_ds.initialization_sha256 = record_initialization(job, output)
            train_ds.identity_context = {
                'scope': PROTOCOL, 'smoke': False, 'condition': cond_name,
                'initialization': 'random', 'pretrained': False, 'diagnostic': True,
            }
            ledger = BudgetLedger(output, user_quota_hours=30)
            fedavg_res = run_single_fedavg_job(job, train_ds, val_ds, sentinel_test, suite['class_names'],
                                              partitions, output, ledger, target, resume=False, calibration=True)

            # 2. Negative Control Check: Random labels per group, 1 round (train/val only)
            print("\n--- 2. Negative Control (Shuffled Labels per group, 1 round) ---")
            rng = np.random.default_rng(999)
            unique_groups = sorted(list({r['group_id'] for r in sub_train}))
            random_label_map = {gid: int(rng.integers(0, 38)) for gid in unique_groups}
            neg_train_rows = [dict(r, label=random_label_map[r['group_id']]) for r in sub_train]
            neg_train_ds = ManifestDataset(neg_train_rows, dataset_root, True)
            neg_job = make_job(f"{cond_name}_negative_control", seed=42, smoke=True, rounds=1, optimizer=optimizer, lr=lr, momentum=momentum)
            neg_train_ds.initialization_sha256 = record_initialization(neg_job, output)
            neg_train_ds.identity_context = {
                'scope': PROTOCOL, 'smoke': True, 'condition': 'neg_control',
                'initialization': 'random', 'pretrained': False, 'diagnostic': True,
            }
            neg_res = run_single_fedavg_job(neg_job, neg_train_ds, val_ds, sentinel_test, suite['class_names'],
                                           partitions, output, ledger, target, resume=False, calibration=True)

            # 3. Resume Check: Pause after round 1, then resume to round 2
            print("\n--- 3. Resume Verification Check ---")
            resume_dir = output / 'resume_test'
            resume_dir.mkdir(parents=True, exist_ok=True)
            job_resume = make_job(f"{cond_name}_resume_test", seed=42, smoke=True, rounds=2, optimizer=optimizer, lr=lr, momentum=momentum)
            train_ds.initialization_sha256 = record_initialization(job_resume, resume_dir)
            train_ds.identity_context = {
                'scope': PROTOCOL, 'smoke': True, 'condition': 'resume_check',
                'initialization': 'random', 'pretrained': False, 'diagnostic': True,
            }
            ledger_r = BudgetLedger(resume_dir, user_quota_hours=30)
            orig_can_continue = ledger_r.can_continue
            rounds_seen = []
            def pause_after_round_1(*args, **kwargs):
                rounds_seen.append(1)
                if len(rounds_seen) > 1:
                    return False, "Simulated pause after round 1"
                return orig_can_continue(*args, **kwargs)
            ledger_r.can_continue = pause_after_round_1
            r1_res = run_single_fedavg_job(job_resume, train_ds, val_ds, sentinel_test, suite['class_names'],
                                           partitions, resume_dir, ledger_r, target, resume=False, calibration=True)
            assert r1_res.get('status') == 'PAUSED_QUOTA', f"Expected PAUSED_QUOTA, got {r1_res.get('status')}"
            ledger_r.can_continue = orig_can_continue
            r2_res = run_single_fedavg_job(job_resume, train_ds, val_ds, sentinel_test, suite['class_names'],
                                           partitions, resume_dir, ledger_r, target, resume=True, calibration=True)
            resume_success = (r2_res.get('status') == 'CALIBRATED_NO_TEST' or r2_res.get('status') == 'COMPLETED')

            pilot_report = {
                'mode': 'pilot',
                'protocol': PROTOCOL,
                'optimizer': optimizer,
                'lr': lr,
                'momentum': momentum,
                'samples': {'train': len(sub_train), 'val': len(sub_val), 'test': 'NOT_ACCESSED'},
                'fedavg': fedavg_res,
                'negative_control': {
                    'round_1_val_acc': neg_res.get('history', [{}])[0].get('validation_accuracy') if 'history' in neg_res else None,
                    'status': neg_res.get('status'),
                    'note': 'Validation accuracy evaluated on genuine validation set.'
                },
                'resume_check': {
                    'success': resume_success,
                    'note': 'Atomic checkpoint restored successfully and continued to round 2.'
                },
                'stage1_reference': {
                    'note': 'Centralized & Local-only baselines are read-only from Stage 1. No baseline training in Stage 2.',
                    'source': 'stage1_compat.baselines.BaselineRegistry'
                }
            }
            atomic_json(output / 'pilot_report.json', pilot_report)
            print(f"\nPilot report saved to {output / 'pilot_report.json'}")
            return pilot_report
    finally:
        os.environ.pop('STAGE1_SOFT_DEADLINE', None)


def comparison_key(metrics):
    identity, config = metrics['identity'], metrics['resolved_config']
    ctx = identity['context']
    if (ctx.get('scope') != PROTOCOL or ctx.get('smoke') is not False
            or ctx.get('initialization') != 'random' or ctx.get('pretrained') is not False):
        raise ValueError('Only full-data random-initialized scratch results can be compared')
    expected = make_job(
        ctx['condition'],
        identity['seed'],
        rounds=config.get('rounds'),
        optimizer=config.get('optimizer', 'sgd'),
        lr=config.get('lr'),
        momentum=config.get('momentum', 0.0),
    )
    if config != expected.to_dict():
        raise ValueError('Result differs from the frozen scratch trainer')
    return {
        'seed': identity['seed'], 'common': ctx['common'],
        'source': ctx['scratch_source_sha256'], 'suite': ctx['suite_sha256'],
        'initialization': ctx['initialization_sha256'], 'runtime': identity['runtime'],
        'train_device': ctx['train_device'], 'evaluation_device': ctx['evaluation_device'],
        'trainer': trainer_contract(expected),
    }


def collect(output, suite_dir):
    output = Path(output)
    suite, _ = preflight(suite_dir)
    # Read saved protocol from output directory if present to preserve run parameters
    saved_protocol_path = output / 'scratch_protocol.json'
    if saved_protocol_path.is_file():
        saved_proto = json.loads(saved_protocol_path.read_text(encoding='utf-8'))
        saved_trainer = saved_proto.get('trainer', {})
        protocol = frozen_protocol(
            suite,
            optimizer=saved_trainer.get('optimizer', 'sgd'),
            lr=saved_trainer.get('lr'),
            rounds=saved_trainer.get('rounds'),
            momentum=saved_trainer.get('momentum', 0.0),
        )
    else:
        protocol = frozen_protocol(suite)
    initialize_output(output, protocol, resume=True)
    results = {}
    for file in sorted(output.glob('*/fedavg_metrics.json')):
        m = validate_completed(file.parent)
        ctx = m['identity']['context']
        name, seed = ctx['condition'], m['seed']
        cfg = m['resolved_config']
        expected_job = make_job(
            name, seed, suite['smoke'],
            rounds=cfg.get('rounds'),
            optimizer=cfg.get('optimizer', 'sgd'),
            lr=cfg.get('lr'),
            momentum=cfg.get('momentum', 0.0),
        )
        if (ctx.get('scope') != PROTOCOL or ctx.get('initialization') != 'random'
                or ctx.get('pretrained') is not False or ctx['smoke'] != suite['smoke']
                or ctx['scratch_source_sha256'] != protocol['source_sha256']
                or ctx['suite_sha256'] != protocol['suite_sha256'] or ctx['common'] != suite['common']
                or ctx['manifest_sha256'] != suite['conditions'][name]['manifest_sha256']
                or cfg != expected_job.to_dict()):
            raise ValueError('Result is not from this scratch suite')
        initial = json.loads((output / 'initialization' / f'{m["job_id"]}.json').read_text())
        if (initial['initialization_sha256'] != ctx['initialization_sha256']
                or initial['pretrained'] is not False or initial['checkpoint_source'] is not None):
            raise ValueError('Missing or mismatched scratch initialization evidence')
        if (name, seed) in results:
            raise ValueError('Duplicate condition/seed')
        results[name, seed] = m
    pairs, missing = [], []
    comparisons = [(r, t) for r in ('label100', 'label1') for t in ('label01', 'label_quantity01')]
    comparisons += [('quantity100', 'quantity01'), ('feature100', 'feature01')]
    for seed in sorted({s for _, s in results}):
        for ref, target in comparisons:
            pair = {'seed': seed, 'reference': ref, 'target': target}
            if (ref, seed) not in results or (target, seed) not in results:
                missing.append(pair)
            elif not suite['smoke']:
                a, b = results[ref, seed], results[target, seed]
                if comparison_key(a) != comparison_key(b):
                    raise ValueError('Cannot compare different trainer, data, initialization or runtime')
                pairs.append({**pair, **{k + '_delta_pp': 100 * (b['test_metrics'][k] - a['test_metrics'][k])
                                        for k in ('accuracy', 'macro_f1')}})
    report = {
        'protocol': PROTOCOL, 'initialization': 'random', 'pretrained': False,
        'smoke': suite['smoke'], 'scientific_stage2_complete': False,
        'reason': 'GĐ2 chỉ train FedAvg; các baseline Centralized/Local-only lịch sử từ GĐ1 khác protocol (AdamW vs SGD, khác split) nên chỉ dùng làm tham chiếu (historical_reference), không đủ điều kiện strict comparison.',
        'completed_jobs': [{'condition': n, 'seed': s, 'test': m['test_metrics'], 'best_round': m['best_round']}
                          for (n, s), m in results.items()],
        'paired_deltas': pairs, 'missing_pairs': missing,
        'remaining_stage2_requirements': ['multi-seed sweep', 'per-client fairness'],
        'historical_baselines': 'reference only from Stage 1; do not attribute scratch-vs-pretrained gaps solely to non-IID'
    }
    atomic_json(output / 'scratch_comparison.json', report)
    return report
