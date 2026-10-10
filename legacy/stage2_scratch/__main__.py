"""python -m stage2_scratch: FedAvg-only execution from random weights (weights=None).

Centralized và Local-only không được huấn luyện lại trong GĐ2;
chỉ đọc từ kết quả GĐ1 làm tham chiếu lịch sử qua action compare-stage1.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='FedAvg from random weights; pretrained is always disabled; FedAvg-only')
    parser.add_argument('action', choices=['preflight', 'run', 'collect', 'smoke', 'pilot', 'flower_verify', 'compare-stage1'])
    parser.add_argument('--suite', default='data/partitions_stage2_scratch_v3')
    parser.add_argument('--dataset', default='../PlantVillage-Dataset/raw/color')
    parser.add_argument('--output', default='runs/stage2_scratch_v3')
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--session-minutes', type=float, default=420)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--optimizer', choices=['sgd', 'adamw'], default='sgd')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--resume', action='store_true', help='Explicitly continue ONLY an existing matching scratch experiment')
    args = parser.parse_args()
    if args.resume and args.action != 'run':
        parser.error('--resume only applies to run')
    runtime = Path(__file__).resolve().parents[1] / '.stage2_scratch_runtime'
    for key in ('TEMP', 'TMP', 'TMPDIR', 'TORCH_HOME'):
        path = runtime / ('torch' if key == 'TORCH_HOME' else 'tmp')
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    from .experiment import run, collect, frozen_protocol, diagnose
    from .baselines import load_historical_baselines
    from stage2_matched.data import preflight
    if args.action == 'preflight':
        suite, audits = preflight(args.suite, args.dataset)
        result = {'protocol': frozen_protocol(suite, optimizer=args.optimizer, lr=args.lr, rounds=args.rounds), 'counts': suite['counts'], 'audits': audits}
    elif args.action == 'collect':
        result = collect(args.output, args.suite)
    elif args.action == 'compare-stage1':
        result = load_historical_baselines()
        target_path = Path(args.output) / 'stage1_comparison.json'
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    elif args.action in ('smoke', 'pilot'):
        result = diagnose(args.suite, args.dataset, args.output, mode=args.action,
                          session_minutes=args.session_minutes, device=args.device,
                          optimizer=args.optimizer, lr=args.lr)
    elif args.action == 'flower_verify':
        # Single-round integration check of Flower vs Sequential oracle (diagnostic only, sentinel test)
        from .flower_adapter import run_flower_fedavg
        from .experiment import make_job, SentinelTestDataset
        from stage2_matched.data import read_rows, ManifestDataset
        path = Path(args.suite) / (args.conditions[0] if args.conditions else 'label100')
        meta = json.loads((path / 'partition_config.json').read_text())
        rows = read_rows(path / 'centralized_train.csv')[:300]
        train = ManifestDataset(rows, args.dataset, True, meta.get('domain_profiles', []))
        val = ManifestDataset(read_rows(path / 'global_val.csv')[:100], args.dataset)
        sentinel_test = SentinelTestDataset()
        partitions = [[i for i, r in enumerate(rows) if r['client_id'] == cid] for cid in range(5)]
        suite, _ = preflight(args.suite, args.dataset)
        job = make_job('label100', 42, smoke=True, rounds=1, optimizer=args.optimizer, lr=args.lr)
        result = run_flower_fedavg(job, train, val, sentinel_test, suite['class_names'], partitions, args.output, torch.device('cpu'), optimizer_name=args.optimizer)
    else:
        result = run(args.suite, args.dataset, args.output, args.conditions, args.seeds,
                     args.session_minutes, args.device, resume=args.resume,
                     optimizer=args.optimizer, lr=args.lr, rounds=args.rounds)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
