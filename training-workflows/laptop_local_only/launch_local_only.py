"""Launcher for Local-Only baseline training on laptop.

Design Principles:
1. Local-Only Paradigm: Each client trains an independent model on its local shard.
   Zero weight aggregation, zero server synchronization (unlike FedAvg).
   All client models initialize from the exact same pinned W0 state (MobileNetV3-Small,
   seed 42) from the FedAvg spec v4 to ensure rigorous, scientifically fair comparison.
2. Zero Copy / Shared Dataset: Directly mounts and consumes D:\\university\\do-an-tot-nghiep\\dataset
   (release mixed/pv_pd_v3). Never copies images, manifests, or shards; never creates ad-hoc splits.
3. Audited Partition Schemes: Dynamically reads partition_spec_v4.json at runtime,
   allowing selection of audited scenarios (status == PASS) or uniform 'iid'.
4. Flexible Granularity: Supports training a single client (--client-id client_00)
   or all clients (--client-id all).
5. Safe Defaults: Default mode is a fast, safe 'smoke' run (1 round, 2 batches).
   Full training (60 rounds, full batches) strictly requires an explicit flag (--full).
6. Non-Destructive Dry-Run: --dry-run rigorously validates paths, release manifest SHA256,
   pretrained weights, W0 state, partition audit, and output directories without training.
7. Reuses Existing Runner: Imports and executes train-gd-2/.../local_only_mixed_runner.py
   directly, preserving fail-closed invariants and evaluation protocols.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.data_selection import inspect_dataset  # noqa: E402


def resolve_workspace_root() -> Path:
    """Resolve the repository workspace root dynamically."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "dataset").is_dir() and (candidate / "training-artifacts").is_dir():
        return candidate.resolve()
    env_root = os.environ.get("WORKSPACE_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root).resolve()
    cwd = Path.cwd().resolve()
    for parent in [cwd, cwd.parent, cwd.parent.parent]:
        if (parent / "dataset").is_dir() and (parent / "training-artifacts").is_dir():
            return parent.resolve()
    return candidate.resolve()


def configure_sys_path(workspace_root: Path) -> Tuple[Path, Path]:
    """Configure sys.path to discover existing runner and plant_data_contract."""
    campaign_dir = workspace_root / "train-gd-2" / "kaggle-workspace" / "campaign-mixed-pv-pd-v1"
    contract_dir = workspace_root / "plant_data_contract"

    if campaign_dir.is_dir() and str(campaign_dir) not in sys.path:
        sys.path.insert(0, str(campaign_dir))
    if contract_dir.is_dir() and str(contract_dir) not in sys.path:
        sys.path.insert(0, str(contract_dir))
    if str(workspace_root) not in sys.path:
        sys.path.append(str(workspace_root))

    return campaign_dir, contract_dir


def compute_sha256(file_path: Path, max_bytes: Optional[int] = None) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        if max_bytes is None:
            while chunk := f.read(65536):
                h.update(chunk)
        else:
            remaining = max_bytes
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
    return h.hexdigest()


def get_audited_partition_schemes(release_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Query partition_spec_v4.json for all audited, verified partition schemes."""
    schemes: Dict[str, Dict[str, Any]] = {
        "iid": {
            "status": "PASS",
            "scenario": "iid",
            "skew_type": "uniform_iid",
            "description": "Uniform random split (verified by union completeness)",
            "num_clients": 5,
        }
    }
    spec_path = release_dir / "partition_spec_v4.json"
    if not spec_path.is_file():
        return schemes

    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
        scenarios = data.get("scenarios", {})
        for name, sc in scenarios.items():
            if isinstance(sc, dict) and sc.get("status") == "PASS":
                meta = sc.get("metadata", {})
                schemes[name] = {
                    "status": "PASS",
                    "scenario": name,
                    "skew_type": meta.get("skew_type", "non_iid"),
                    "description": f"Audited non-IID partition (skew={meta.get('skew_type', 'non_iid')})",
                    "num_clients": sc.get("num_clients", 5),
                    "mean_entropy": sc.get("mean_entropy"),
                    "mean_js_divergence": sc.get("mean_js_divergence"),
                }
    except Exception as err:
        schemes["_error"] = {"status": "ERROR", "message": str(err)}

    return schemes


class PreflightChecker:
    """Validates paths, hashes, integrity invariants, and runner dependencies."""

    def __init__(
        self,
        workspace_root: Path,
        dataset_root: Path,
        spec_file: Path,
        pretrained_weights: Path,
        partition_scheme: str,
        client_id: Optional[str],
        output_dir: Path,
        mode: str,
        rounds: int,
        max_batches: Optional[int],
        plantdoc_sampling_ratio: str,
        seed: int,
        resume: bool = False,
    ):
        self.workspace_root = workspace_root
        self.dataset_root = dataset_root
        self.spec_file = spec_file
        self.pretrained_weights = pretrained_weights
        self.partition_scheme = partition_scheme
        self.client_id = client_id
        self.output_dir = output_dir
        self.mode = mode
        self.rounds = rounds
        self.max_batches = max_batches
        self.plantdoc_sampling_ratio = plantdoc_sampling_ratio
        self.seed = seed
        self.resume = resume

        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}

    def run_checks(self) -> bool:
        # 1. Dataset Release Checks
        release_dir = self.dataset_root / "mixed" / "pv_pd_v3"
        self.info["release_dir"] = str(release_dir)
        if not release_dir.is_dir():
            self.errors.append(f"Dataset release directory missing: {release_dir}")
            return False

        release_manifest = release_dir / "release_manifest.json"
        if not release_manifest.is_file():
            self.errors.append(f"release_manifest.json missing at: {release_manifest}")
        else:
            calc_sha = compute_sha256(release_manifest)
            self.info["computed_release_sha256"] = calc_sha

        train_manifest = release_dir / "manifests" / "train.jsonl"
        val_manifest = release_dir / "manifests" / "val.jsonl"
        if not train_manifest.is_file():
            self.errors.append(f"train.jsonl manifest missing at: {train_manifest}")
        if not val_manifest.is_file():
            self.errors.append(f"val.jsonl manifest missing at: {val_manifest}")

        # 2. Spec Checks
        if not self.spec_file.is_file():
            self.errors.append(f"FedAvg spec file missing at: {self.spec_file}")
            return False

        try:
            spec_data = json.loads(self.spec_file.read_text(encoding="utf-8"))
            self.info["spec_data"] = spec_data
            expected_release_sha = spec_data.get("release_manifest_sha256")
            if "computed_release_sha256" in self.info:
                if self.info["computed_release_sha256"] != expected_release_sha:
                    self.errors.append(
                        f"Release manifest SHA256 mismatch!\n"
                        f"  Computed: {self.info['computed_release_sha256']}\n"
                        f"  Spec:     {expected_release_sha}"
                    )
        except Exception as err:
            self.errors.append(f"Failed to parse spec file JSON: {err}")
            return False

        # 3. Model Weights & W0 State Checks
        if not self.pretrained_weights.is_file():
            self.errors.append(f"Pretrained weights missing at: {self.pretrained_weights}")
        else:
            self.info["pretrained_weights"] = str(self.pretrained_weights)
            expected_weights_sha = spec_data.get("pretrained_weights_sha256")
            if expected_weights_sha and compute_sha256(self.pretrained_weights) != expected_weights_sha:
                self.errors.append("Pretrained weights SHA256 differs from pinned spec")

        w0_name = spec_data.get("pretrained_w0_state_file", "mobilenet_v3_small_38_seed42_w0.pt")
        w0_path = self.pretrained_weights.with_name(w0_name)
        self.info["w0_path"] = str(w0_path)
        if not w0_path.is_file():
            self.errors.append(f"Initial W0 state file missing at: {w0_path}")
        else:
            w0_sha = compute_sha256(w0_path)
            expected_w0_sha = spec_data.get("pretrained_w0_state_sha256")
            if expected_w0_sha and w0_sha != expected_w0_sha:
                self.errors.append(
                    f"W0 state SHA256 mismatch!\n"
                    f"  Computed: {w0_sha}\n"
                    f"  Spec:     {expected_w0_sha}"
                )
            self.info["w0_sha256"] = w0_sha

        # 4. Partition Scheme & Audited Scenarios
        audited = get_audited_partition_schemes(release_dir)
        self.info["audited_schemes"] = list(audited.keys())
        if self.partition_scheme not in audited:
            valid_list = ", ".join(k for k in audited.keys() if not k.startswith("_"))
            self.errors.append(
                f"Partition scheme '{self.partition_scheme}' is NOT an audited scenario!\n"
                f"Available audited schemes: [{valid_list}]"
            )
        else:
            if self.mode == "full":
                approved_jobs = spec_data.get("kaggle_fedavg_jobs", [])
                if spec_data.get("status") != "READY_FULL" or not any(
                    job.get("partition_scheme") == self.partition_scheme
                    and job.get("status") == "READY_FULL" for job in approved_jobs
                ):
                    self.errors.append("Full local-only partition is not approved in the selected READY_FULL spec")
            scheme_dir = release_dir / "partitions" / self.partition_scheme
            if not scheme_dir.is_dir():
                self.errors.append(f"Partition directory missing: {scheme_dir}")
            else:
                for i in range(5):
                    shard_file = scheme_dir / f"client_{i:02d}.jsonl"
                    if not shard_file.is_file():
                        self.errors.append(f"Client shard missing: {shard_file}")
            if scheme_dir.is_dir():
                try:
                    view = inspect_dataset(self.dataset_root, self.partition_scheme)
                    self.info["partition_train_count"] = view["train_count"]
                except (ValueError, FileNotFoundError, KeyError) as err:
                    self.errors.append(f"Partition integrity check failed: {err}")

        # 5. Client Selection
        hparams = spec_data.get("hyperparameters", {}).get("fedavg", {})
        client_count = int(hparams.get("clients_per_round", 5))
        valid_client_ids = [f"client_{i:02d}" for i in range(client_count)]
        self.info["valid_client_ids"] = valid_client_ids

        effective_client = None if self.client_id in (None, "all", "", "ALL") else self.client_id
        if effective_client is not None and effective_client not in valid_client_ids:
            self.errors.append(
                f"Invalid client ID '{self.client_id}'. Must be one of {valid_client_ids} or 'all'."
            )
        self.info["effective_client"] = effective_client
        self.info["selected_clients"] = [effective_client] if effective_client else valid_client_ids

        # 6. Output Directory Invariant
        if self.resume:
            if effective_client is None:
                self.errors.append("Resume requires one explicit client ID")
            elif not (self.output_dir / effective_client / "checkpoint_last.pt").is_file():
                self.errors.append("Resume checkpoint_last.pt is missing")
        elif self.output_dir.exists() and any(self.output_dir.iterdir()):
            self.errors.append(
                f"Output directory must be empty or non-existent to avoid collisions: {self.output_dir}"
            )

        # 7. Execution Budget Checks
        if self.rounds < 1 or self.rounds > 60:
            self.errors.append(f"Rounds must be between 1 and 60, got: {self.rounds}")
        if self.max_batches is not None and self.max_batches <= 0:
            self.errors.append(f"max_batches must be positive, got: {self.max_batches}")

        # 8. Runner Discovery Check
        campaign_dir, _ = configure_sys_path(self.workspace_root)
        runner_file = campaign_dir / "local_only_mixed_runner.py"
        if not runner_file.is_file():
            self.errors.append(f"Existing runner script missing at: {runner_file}")
        else:
            self.info["runner_file"] = str(runner_file)

        return len(self.errors) == 0

    def print_dry_run_report(self) -> None:
        """Print a clear, structured preflight audit report."""
        print("=" * 76)
        print("   LOCAL-ONLY LAPTOP LAUNCHER — PREFLIGHT & DRY-RUN AUDIT REPORT")
        print("=" * 76)

        status_tag = "[PASS] PREFLIGHT OK" if not self.errors else "[FAIL] PREFLIGHT FAILED"
        print(f"Overall Status:        {status_tag}")
        print(f"Mode:                  {self.mode.upper()}")
        print(f"Workspace Root:        {self.workspace_root}")
        print(f"Shared Dataset Root:   {self.dataset_root}")
        print(f"Dataset Release:       {self.info.get('release_dir', 'N/A')}")
        print(f"Release Manifest SHA:  {self.info.get('computed_release_sha256', 'N/A')}")
        print(f"FedAvg Spec File:      {self.spec_file}")
        print(f"Pretrained Weights:    {self.info.get('pretrained_weights', 'N/A')}")
        print(f"Initial W0 State:      {self.info.get('w0_path', 'N/A')}")
        print(f"W0 SHA256:             {self.info.get('w0_sha256', 'N/A')}")
        print(f"Partition Scheme:      {self.partition_scheme} (Audited: Yes)")
        print(f"Target Client(s):      {self.info.get('selected_clients', 'N/A')}")
        print(f"Epoch/Round Budget:    rounds={self.rounds}, max_batches={self.max_batches}")
        print(f"PlantDoc Sampling:     {self.plantdoc_sampling_ratio}")
        print(f"Random Seed:           {self.seed}")
        print(f"Output Directory:      {self.output_dir}")
        print(f"Reused Runner:         {self.info.get('runner_file', 'N/A')}")

        if self.warnings:
            print("-" * 76)
            print("WARNINGS:")
            for w in self.warnings:
                print(f"  [!] {w}")

        if self.errors:
            print("-" * 76)
            print("ERRORS DETECTED:")
            for e in self.errors:
                print(f"  [X] {e}")
            print("=" * 76)
            return

        print("-" * 76)
        print("Target Runner Invocation (Equivalent CLI command):")
        cid_flag = f"--client-id {self.info['effective_client']}" if self.info.get("effective_client") else ""
        mb_flag = f"--max-batches {self.max_batches}" if self.max_batches else ""
        print(
            f"  python {self.info.get('runner_file')}\n"
            f"    --dataset-root \"{self.dataset_root}\"\n"
            f"    --spec-file \"{self.spec_file}\"\n"
            f"    --pretrained-weights \"{self.pretrained_weights}\"\n"
            f"    --partition-scheme \"{self.partition_scheme}\"\n"
            f"    --output-dir \"{self.output_dir}\"\n"
            f"    --rounds {self.rounds} \\\n"
            f"    --seed {self.seed} \\\n"
            f"    --plantdoc-sampling-ratio \"{self.plantdoc_sampling_ratio}\" \\\n"
            f"    {cid_flag} {mb_flag}".strip()
        )
        print("=" * 76)
        print("[DRY-RUN] Preflight verified successfully. No training was executed.")
        print("=" * 76)


def generate_default_output_dir(
    workspace_root: Path, partition_scheme: str, client_id: Optional[str], mode: str
) -> Path:
    """Generate a clean timestamped output directory under laptop_local_only/runs."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cid_str = client_id if client_id and client_id != "all" else "all_clients"
    dir_name = f"local_only_{partition_scheme}_{cid_str}_{mode}_{ts}"
    return workspace_root / "training-workflows" / "laptop_local_only" / "runs" / dir_name


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laptop Local-Only Baseline Launcher (Independent Client Models, No Weight Aggregation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core execution configuration
    parser.add_argument(
        "--partition-scheme",
        default="iid",
        help="Audited partition scheme to train on (e.g., 'iid', 'label_alpha_0_1', 'label_alpha_0_5', 'quantity_alpha_0_3', etc.). Default: 'iid'",
    )
    parser.add_argument(
        "--client-id",
        default=None,
        help="Target client to train ('client_00' through 'client_04') or 'all' to train all clients. Default: 'all'",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Execution mode: 'smoke' (safe short run: 1 round, 2 batches) or 'full' (60 rounds, full batches). Default: 'smoke'",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Explicit flag to run FULL training (shortcut for --mode full: 60 rounds, full batches). Default: False",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform preflight validation of all paths, hashes, partition audit, and configuration without training.",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Continue one client from checkpoint_last.pt in --output-dir")

    # Overrides and paths
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Separate output directory for metrics and checkpoints. Defaults to a fresh timestamped directory in runs/.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to shared dataset directory. Defaults to D:\\university\\do-an-tot-nghiep\\dataset.",
    )
    parser.add_argument(
        "--spec-file",
        type=Path,
        default=None,
        help="Path to FedAvg spec file v4. Defaults to preflight_extract/quality_spec_v4_FEDAVG_FULL.json.",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        help="Path to pretrained MobileNetV3 weights (.pth). Defaults to preflight_extract/mobilenet_v3_small-047dcff4.pth.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model and client sampling. Default: 42.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto",
        help="auto selects laptop CUDA when available; cuda fails if PyTorch CUDA is unavailable.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override total rounds budget. (Default: 1 for smoke, 60 for full).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Override max batches per epoch/eval. (Default: 2 for smoke, None for full).",
    )
    parser.add_argument(
        "--plantdoc-sampling-ratio",
        default="0.25",
        help="PlantDoc sampling ratio; default 0.25 matches the FedAvg baseline.",
    )
    parser.add_argument(
        "--list-schemes",
        action="store_true",
        help="List all audited partition schemes verified in partition_spec_v4.json and exit.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but this Python environment has no usable CUDA device")
    workspace_root = resolve_workspace_root()

    # Default paths resolution
    dataset_root = args.dataset_root or (workspace_root / "dataset")
    release_dir = dataset_root / "mixed" / "pv_pd_v3"

    # List schemes quick action
    if args.list_schemes:
        schemes = get_audited_partition_schemes(release_dir)
        print("=" * 70)
        print("AUDITED PARTITION SCHEMES (from dataset/mixed/pv_pd_v3/partition_spec_v4.json)")
        print("=" * 70)
        for name, details in sorted(schemes.items()):
            if name.startswith("_"):
                continue
            desc = details.get("description", "")
            skew = details.get("skew_type", "N/A")
            entropy = details.get("mean_entropy")
            ent_str = f", mean_entropy={entropy:.4f}" if entropy is not None else ""
            print(f"  • {name:<22} [{skew}] {desc}{ent_str}")
        print("=" * 70)
        return 0

    # Determine execution mode: --full flag takes precedence
    mode = "full" if args.full or args.mode == "full" else "smoke"

    # Mode-dependent defaults:
    # Smoke: safe short run (1 round, 2 batches)
    # Full: full campaign budget (60 rounds, full batches)
    default_rounds = 60 if mode == "full" else 1
    default_max_batches = None if mode == "full" else 2

    rounds = args.rounds if args.rounds is not None else default_rounds
    max_batches = args.max_batches if args.max_batches is not None else default_max_batches
    if mode == "full" and (rounds != 60 or max_batches is not None):
        raise ValueError("Full local-only requires all 60 rounds and uncapped batches; use smoke for capped runs")

    # Default spec and weights
    primary_spec = (
        workspace_root
        / "training-artifacts"
        / "cloud-v4-full-ready-20261001"
        / "preflight_extract"
        / "quality_spec_v4_FEDAVG_FULL.json"
    )
    spec_file = args.spec_file or primary_spec

    primary_weights = (
        workspace_root
        / "training-artifacts"
        / "cloud-v4-full-ready-20261001"
        / "preflight_extract"
        / "mobilenet_v3_small-047dcff4.pth"
    )
    if not primary_weights.is_file():
        primary_weights = (
            workspace_root
            / "training-artifacts"
            / "pretrained_weights"
            / "mobilenet_v3_small-047dcff4.pth"
        )
    pretrained_weights = args.pretrained_weights or primary_weights

    # Output directory
    output_dir = args.output_dir or generate_default_output_dir(
        workspace_root, args.partition_scheme, args.client_id, mode
    )

    # Perform preflight audit
    checker = PreflightChecker(
        workspace_root=workspace_root,
        dataset_root=dataset_root,
        spec_file=spec_file,
        pretrained_weights=pretrained_weights,
        partition_scheme=args.partition_scheme,
        client_id=args.client_id,
        output_dir=output_dir,
        mode=mode,
        rounds=rounds,
        max_batches=max_batches,
        plantdoc_sampling_ratio=args.plantdoc_sampling_ratio,
        seed=args.seed,
        resume=args.resume,
    )

    ok = checker.run_checks()

    if args.dry_run:
        checker.print_dry_run_report()
        return 0 if ok else 1

    if not ok:
        checker.print_dry_run_report()
        print("\n[ABORT] Preflight checks failed. Please fix the errors listed above.")
        return 1

    # Warning for laptop full training
    if mode == "full":
        print("\n" + "!" * 76)
        print("  CAUTION: FULL TRAINING MODE INITIATED ON LAPTOP")
        print("  This run will execute 60 rounds over all local batches.")
        print("  On CPU/laptop GPU this may require several hours.")
        print("!" * 76 + "\n")

    # Configure paths for runner reuse
    configure_sys_path(workspace_root)

    try:
        import local_only_mixed_runner
    except ImportError as err:
        print(f"[FATAL] Unable to import existing runner 'local_only_mixed_runner': {err}")
        print("Ensure 'train-gd-2/kaggle-workspace/campaign-mixed-pv-pd-v1' and dependencies exist.")
        return 1

    # Prepare execution arguments for existing runner
    effective_client = None if args.client_id in (None, "all", "", "ALL") else args.client_id
    runner_args = argparse.Namespace(
        dataset_root=dataset_root,
        spec_file=spec_file,
        pretrained_weights=pretrained_weights,
        partition_scheme=args.partition_scheme,
        output_dir=output_dir,
        client_id=effective_client,
        seed=args.seed,
        rounds=rounds,
        max_batches=max_batches,
        plantdoc_sampling_ratio=args.plantdoc_sampling_ratio,
        device=args.device,
        resume=args.resume,
    )

    print(f"Starting local-only execution in mode: {mode.upper()}...")
    print(f"Output directory: {output_dir}")

    # local_only_mixed_runner.run() creates output_dir and handles client loops
    summary = local_only_mixed_runner.run(runner_args)

    # Save launcher run metadata alongside runner summary
    run_meta = {
        "launcher": "laptop_local_only",
        "version": "1.1.0",
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "partition_scheme": args.partition_scheme,
        "client_id": effective_client or "all",
        "rounds": rounds,
        "max_batches": max_batches,
        "seed": args.seed,
        "device": summary.get("device", args.device),
        "resumed": args.resume,
        "dataset_root": str(dataset_root),
        "spec_file": str(spec_file),
        "output_dir": str(output_dir),
    }
    meta_path = output_dir / "launcher_run_meta.json"
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print("LOCAL-ONLY TRAINING COMPLETED SUCCESSFULLY")
    print(f"Summary metrics: {output_dir / 'summary.json'}")
    print(f"Launcher meta:   {meta_path}")
    print("=" * 76)

    return 0


if __name__ == "__main__":
    sys.exit(main())
