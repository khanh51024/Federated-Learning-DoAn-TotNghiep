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
    parser.add_argument('--suite', default=None, help='Path to partitions suite (defaults to candidate data/ or workspace training-data/)')
    parser.add_argument('--dataset', default=None, help='Path to PlantVillage color dataset root')
    parser.add_argument('--output', default='runs/stage2_scratch_v3')
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--session-minutes', type=float, default=420)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--optimizer', choices=['sgd', 'adamw'], default='sgd')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--resume', action='store_true', help='Explicitly continue ONLY an existing matching scratch experiment')
    parser.add_argument('--runtime-dir', default=None, help='Directory for temporary and torch cache files')
    args = parser.parse_args()
    if args.resume and args.action != 'run':
        parser.error('--resume only applies to run')

    repo_root = Path(__file__).resolve().parents[1]

    # Resolve runtime directory: respect STAGE2_RUNTIME_DIR, --runtime-dir, or fallback to repo runtime
    runtime_env = os.environ.get('STAGE2_RUNTIME_DIR') or args.runtime_dir
    if runtime_env:
        runtime = Path(runtime_env).resolve()
        for key in ('TEMP', 'TMP', 'TMPDIR', 'TORCH_HOME'):
            path = runtime / ('torch' if key == 'TORCH_HOME' else 'tmp')
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
    elif not os.environ.get('TORCH_HOME'):
        runtime = repo_root / '.stage2_scratch_runtime'
        for key in ('TEMP', 'TMP', 'TMPDIR', 'TORCH_HOME'):
            path = runtime / ('torch' if key == 'TORCH_HOME' else 'tmp')
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)

    ACTIONS_REQUIRING_DATASET = {'preflight', 'run', 'smoke', 'pilot', 'flower_verify'}

    # Resolve suite path (F02: never silently fallback on explicit path; R2-01: anchored precedence)
    if args.suite is not None:
        explicit_suite = Path(args.suite).resolve()
        if not explicit_suite.is_dir():
            raise FileNotFoundError(f"Specified --suite path does not exist or is not a directory: {args.suite}")
        suite_path = explicit_suite
    else:
        # Default resolution anchored to STAGE2_SUITE_PATH, repo_root and workspace ancestors
        suite_candidates = []
        if os.environ.get('STAGE2_SUITE_PATH'):
            suite_candidates.append(Path(os.environ['STAGE2_SUITE_PATH']).resolve())
        suite_candidates.append(repo_root / 'data' / 'partitions_stage2_scratch_v3')
        for ancestor in [repo_root] + list(repo_root.parents):
            suite_candidates.append(ancestor / 'training-data' / 'stage2' / 'partitions_stage2_scratch_v3')

        suite_path = None
        for cand in suite_candidates:
            if cand.is_dir() and (cand / 'suite.json').is_file():
                suite_path = cand
                break
        if suite_path is None:
            raise FileNotFoundError(
                "Default partitions suite not found. Please provide explicit --suite <path> "
                "or set STAGE2_SUITE_PATH."
            )
    suite_arg = str(suite_path)

    # Resolve dataset path: only mandatory for actions requiring image dataset (R2-01)
    dataset_path = None
    if args.dataset is not None:
        explicit_dataset = Path(args.dataset).resolve()
        if not explicit_dataset.is_dir():
            raise FileNotFoundError(f"Specified --dataset path does not exist or is not a directory: {args.dataset}")
        dataset_path = explicit_dataset
    elif args.action in ACTIONS_REQUIRING_DATASET:
        # Default resolution anchored to STAGE2_DATASET_PATH, repo_root and workspace ancestors
        dataset_candidates = []
        if os.environ.get('STAGE2_DATASET_PATH'):
            dataset_candidates.append(Path(os.environ['STAGE2_DATASET_PATH']).resolve())
        dataset_candidates.append(repo_root / 'data' / 'PlantVillage-Dataset' / 'raw' / 'color')
        for ancestor in [repo_root] + list(repo_root.parents):
            dataset_candidates.append(ancestor / 'training-data' / 'stage2' / 'PlantVillage-Dataset' / 'raw' / 'color')
            dataset_candidates.append(ancestor / 'train-gd-2' / 'PlantVillage-Dataset' / 'raw' / 'color')
            dataset_candidates.append(ancestor / 'PlantVillage-Dataset' / 'raw' / 'color')

        for cand in dataset_candidates:
            if cand.is_dir():
                dataset_path = cand
                break
        if dataset_path is None:
            raise FileNotFoundError(
                "Default PlantVillage dataset not found. Please provide explicit --dataset <path> "
                "or set STAGE2_DATASET_PATH."
            )
    dataset_arg = str(dataset_path) if dataset_path is not None else None

    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    from .experiment import run, collect, frozen_protocol, diagnose
    from .baselines import load_historical_baselines
    from stage2_matched.data import preflight
    if args.action == 'preflight':
        suite, audits = preflight(suite_arg, dataset_arg)
        result = {'protocol': frozen_protocol(suite, optimizer=args.optimizer, lr=args.lr, rounds=args.rounds), 'counts': suite['counts'], 'audits': audits}
    elif args.action == 'collect':
        result = collect(args.output, suite_arg)
    elif args.action == 'compare-stage1':
        result = load_historical_baselines(suite_dir=suite_arg)
        target_path = Path(args.output) / 'stage1_comparison.json'
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    elif args.action in ('smoke', 'pilot'):
        result = diagnose(suite_arg, dataset_arg, args.output, mode=args.action,
                          session_minutes=args.session_minutes, device=args.device,
                          optimizer=args.optimizer, lr=args.lr)
    elif args.action == 'flower_verify':
        # Single-round integration check of Flower vs Sequential oracle (diagnostic only, sentinel test)
        from .flower_adapter import run_flower_fedavg, HAS_FLWR
        if not HAS_FLWR:
            raise ImportError(
                "Flower (flwr) is not installed. To run flower_verify, install the 'flwr' extra/dependency: pip install .[flower]"
            )
        from .experiment import make_job, SentinelTestDataset
        from stage2_matched.data import read_rows, ManifestDataset
        path = Path(suite_arg) / (args.conditions[0] if args.conditions else 'label100')
        meta = json.loads((path / 'partition_config.json').read_text())
        rows = read_rows(path / 'centralized_train.csv')[:300]
        train = ManifestDataset(rows, dataset_arg, True, meta.get('domain_profiles', []))
        val = ManifestDataset(read_rows(path / 'global_val.csv')[:100], dataset_arg)
        sentinel_test = SentinelTestDataset()
        partitions = [[i for i, r in enumerate(rows) if r['client_id'] == cid] for cid in range(5)]
        suite, _ = preflight(suite_arg, dataset_arg)
        job = make_job('label100', 42, smoke=True, rounds=1, optimizer=args.optimizer, lr=args.lr)
        result = run_flower_fedavg(job, train, val, sentinel_test, suite['class_names'], partitions, args.output, torch.device('cpu'), optimizer_name=args.optimizer)
    else:
        result = run(suite_arg, dataset_arg, args.output, args.conditions, args.seeds,
                     args.session_minutes, args.device, resume=args.resume,
                     optimizer=args.optimizer, lr=args.lr, rounds=args.rounds)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
