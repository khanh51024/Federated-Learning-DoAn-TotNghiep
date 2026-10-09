"""Bounded real-image CPU checks of v4. Smoke metrics are not research results."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fl_training.prepare import create_smoke_bundle, audit_manifest_directory


def main():
    output = ROOT.parent / 'output/local-validation-20260915' / time.strftime('smoke_%H%M%S')
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    for key, directory in [('TEMP', output / 'temp'), ('TMP', output / 'temp'),
                           ('RAY_TMPDIR', ROOT.parent / 'output/ray-local'), ('TORCH_HOME', output / 'torch')]:
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    index = json.loads((ROOT / 'data/partitions_train_v4_content_aware/index.json').read_text())
    records = []
    print(f'OUTPUT={output}', flush=True)
    for scenario, feature in [('iid', 'none'), ('label_skew', 'moderate')]:
        condition = output / f'{scenario}_{feature}'
        source = ROOT / next(p['relative_dir'] for p in index['partitions'].values()
                             if p['scenario'] == scenario and p['feature_skew'] == feature
                             and (scenario == 'iid' or p['alpha'] == .1))
        bundle = create_smoke_bundle(source, condition / 'bundle', target_client_images=96, target_val_images=76)
        audit = audit_manifest_directory(bundle, require_all_classes=False, dataset_root=ROOT / index['dataset_path'])
        (condition / 'content_audit.json').write_text(json.dumps(audit, indent=2))
        raw = yaml.safe_load((ROOT / 'configs/train_smoke.yaml').read_text())
        raw['data'].update(bundle_path=str(bundle), partition_index='data/partitions_train_v4_content_aware/index.json',
                           scenario=scenario, feature_skew=feature)
        raw['output'].update(root=str(condition / 'runs/fedavg'), evaluate_test_after_train=True, progress=False)
        raw['runtime']['startup_timeout_seconds'] = 180
        raw['federation']['timeout_seconds'] = 600
        config = condition / 'train.yaml'
        config.write_text(yaml.safe_dump(raw), encoding='utf-8')
        for mode in ['centralized', 'local-only', 'fedavg']:
            args = ['train'] if mode == 'fedavg' else ['baseline', '--mode', mode]
            cmd = [sys.executable, '-m', 'fl_training.cli', *args, '--config', str(config)]
            start = time.monotonic()
            with (condition / f'{mode}.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=720)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                    raise
            record = dict(scenario=scenario, feature=feature, mode=mode, exit_code=code,
                          seconds=round(time.monotonic() - start, 2))
            records.append(record)
            (output / 'commands.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
            if code:
                raise RuntimeError(f'Failed {condition / (mode + ".log")}')
        import torch
        for run in (condition / 'runs').glob('*/*'):
            targets = sorted(run.glob('client_*')) if run.parent.name == 'local_only' else [run]
            for target in targets:
                metrics = json.loads((target / 'test_metrics.json').read_text())
                with (bundle / 'global_test.csv').open(encoding='utf-8') as stream:
                    count = len(list(csv.DictReader(stream)))
                assert metrics['total_samples'] == count
                assert sum(map(sum, metrics['confusion_matrix'])) == count
                load = lambda name: torch.load(target / name, map_location='cpu', weights_only=False)
                final, best = load('model_final.pt'), load('best.pt')
                assert all(torch.equal(v, best['model_state_dict'][k]) for k, v in final['model_state_dict'].items())
        (condition / 'verification.json').write_text(json.dumps(dict(
            all_final_weights_equal_best=True, confusion_matrix_counts_match_test=True,
            scientific_stage2_complete=False, pretrained=False, rounds=2, clients=2), indent=2))
    print('All bounded training and evaluation checks passed', flush=True)


if __name__ == '__main__':
    main()
