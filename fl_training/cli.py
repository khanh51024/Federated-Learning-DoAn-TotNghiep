"""
fl_training.cli: Command line interface for preflight, prepare-data, smoke, train, and evaluate.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from .checkpoint import load_checkpoint, verify_checkpoint_compatibility
from .config import load_training_config
from .data import build_evaluation_loader
from .model import create_mobilenet_v3_small, set_model_state_dict
from .prepare import audit_manifest_directory, create_smoke_bundle, prepare_data
from .progress import EventLogger, ProgressRenderer
from .task import evaluate_model
from .evaluation_artifacts import bind_evaluation, validate_evaluation_partition, write_evaluation


def run_preflight(config_path: str | Path, base_dir: Optional[str | Path] = None) -> int:
    """Check environment, dependency imports, hardware, data partition, and index."""
    print("=" * 70)
    print("FL_TRAINING PREFLIGHT CHECK")
    print("=" * 70)

    pkg_root = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parents[1]

    # 1. Environment & Hardware
    py_ver = platform.python_version()
    print(f"[*] Python Version: {py_ver}")
    cpu_count = os.cpu_count() or 1
    print(f"[*] Logical CPUs available: {cpu_count}")
    if cpu_count < 2:
        print("[!] Warning: Less than 2 logical CPUs detected. Ray actor concurrency may be constrained.")

    # 2. PyTorch & CUDA
    cuda_avail = torch.cuda.is_available()
    cuda_dev_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    print(f"[*] PyTorch: {torch.__version__} | CUDA Available: {cuda_avail} ({cuda_dev_name})")

    # 3. Flower & Ray
    try:
        import flwr
        import ray
        print(f"[*] Flower: {flwr.__version__} | Ray: {ray.__version__}")
    except ImportError as e:
        print(f"[X] Missing dependency: {e}")
        return 1

    # 4. Config & Partition Validation
    try:
        cfg = load_training_config(config_path, base_dir=pkg_root)
        print(f"[*] Configuration valid: schema_version={cfg.schema_version}, model={cfg.model.name}")
        print(f"[*] Training: optimizer={cfg.training.optimizer}, lr={cfg.training.lr}, batch={cfg.training.batch_size}")
        print(f"[*] Federation: clients={cfg.federation.num_clients}, max_rounds={cfg.federation.max_rounds}")
        print(f"[*] Runtime: client_device={cfg.runtime.client_device}, server_device={cfg.runtime.server_device}")
    except FileNotFoundError as e:
        print(f"[X] Config resolution failed: {e}")
        print("    If partition index is missing, please run:")
        print("    python -m fl_training.cli prepare-data --config configs/partition_training.yaml")
        return 1
    except Exception as e:
        print(f"[X] Configuration error: {e}")
        return 1

    # 5. Partition Directory and Manifests Audit
    part_dir = cfg.data.partition_dir
    if not part_dir.exists():
        print(f"[X] Partition directory not found: {part_dir}")
        print("    Please run: python -m fl_training.cli prepare-data --config configs/partition_training.yaml")
        return 1

    try:
        audit_res = audit_manifest_directory(part_dir, require_all_classes=cfg.data.bundle_path is None, dataset_root=cfg.data.dataset_root)
        if cfg.data.bundle_path is not None:
            print("[!] Smoke bundle: class coverage is not a benchmark acceptance criterion")
        print(f"[*] Partition Audit: PASSED (train={audit_res['train_samples']}, val={audit_res['val_samples']}, test={audit_res['test_samples']})")
        print(f"[*] Validation Classes Present: {audit_res['val_classes_present']}/38")
    except Exception as e:
        print(f"[X] Partition manifest audit failed: {e}")
        return 1

    print("=" * 70)
    print("PREFLIGHT STATUS: READY FOR SIMULATION / TRAINING")
    print("=" * 70)
    return 0


def run_prepare_data_cmd(config_path: str | Path, strict: bool = False) -> int:
    print(f"[*] Running prepare-data with config: {config_path}")
    try:
        res = prepare_data(config_path, strict_cache=strict)
        print(f"[*] Successfully generated {len(res['generated_partitions'])} partitions.")
        print(f"[*] Index written to: {res['index_path']}")
        print(f"[*] Test SHA-256: {res['test_paths_sha256']}")
        print(f"[*] Val SHA-256:  {res['val_paths_sha256']}")
        return 0
    except Exception as e:
        print(f"[X] prepare-data failed: {e}")
        return 1


def _run_simulation_subprocess(
    config_path: Path,
    run_dir: Path,
    num_supernodes: int,
    client_device: str,
    cpus_per_client: int,
    max_concurrent_clients: int,
    run_id: str,
    resume_checkpoint: Optional[Path] = None,
    force_resume_stopped: bool = False,
    object_store_memory_mb: int = 256,
) -> subprocess.Popen:
    """
    Launch Flower run_simulation in a separate Python subprocess.
    The parent process monitors events and manages the single progress bar.
    """
    py_executable = sys.executable
    code = (
        "import os, sys; "
        "from flwr.simulation import run_simulation; "
        "from fl_training.server_app import app as server_app; "
        "from fl_training.client_app import app as client_app; "
        f"num_nodes = {num_supernodes}; "
        f"cpus = {cpus_per_client}; "
        f"gpus = 1 if '{client_device}' == 'cuda' else 0; "
        f"max_concurrent = {max_concurrent_clients}; "
        "backend_cfg = {'client_resources': {'num_cpus': cpus, 'num_gpus': gpus}, "
        f"'init_args': {{'num_cpus': cpus * max_concurrent, 'object_store_memory': {object_store_memory_mb} * 1024 * 1024}}}}; "
        "run_simulation(server_app=server_app, client_app=client_app, num_supernodes=num_nodes, "
        "backend_name='ray', backend_config=backend_cfg)"
    )

    env = os.environ.copy()
    env["FL_TRAINING_CONFIG_PATH"] = str(config_path)
    env["FL_TRAINING_RUN_DIR"] = str(run_dir)
    env["FL_TRAINING_RUN_ID"] = run_id
    env["FL_TRAINING_ATTEMPT_ID"] = "1"
    env.pop("FL_TRAINING_RESUME_CHECKPOINT", None)
    env.pop("FL_TRAINING_FORCE_RESUME_STOPPED", None)
    if resume_checkpoint:
        env["FL_TRAINING_RESUME_CHECKPOINT"] = str(resume_checkpoint)
    if force_resume_stopped:
        env["FL_TRAINING_FORCE_RESUME_STOPPED"] = "1"

    # Subprocess execution
    proc = subprocess.Popen(
        [py_executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Stop only the subprocess tree launched for this run."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def run_simulation_launcher(
    config_path: str | Path,
    mode: str = "train",
    resume_checkpoint: Optional[str | Path] = None,
    no_progress: bool = False,
    force_resume_stopped: bool = False,
) -> int:
    """
    Launcher orchestrating the run directory, spawning the Flower simulation subprocess,
    and rendering a single terminal progress bar from events.jsonl.
    """
    cfg_file = Path(config_path).resolve()
    pkg_root = Path(__file__).resolve().parents[1]
    if mode == "smoke":
        data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))["data"]
        bundle = pkg_root / data["bundle_path"]
        if not bundle.exists():
            index = json.loads((pkg_root / data["partition_index"]).read_text(encoding="utf-8"))
            matches = [p for p in index["partitions"].values() if all(
                data.get(k) is None or p.get(k) == data[k]
                for k in ("scenario", "alpha", "quantity_alpha", "feature_skew", "split_seed"))]
            if len(matches) != 1:
                raise ValueError("Smoke source must resolve to exactly one partition")
            create_smoke_bundle(pkg_root / matches[0]["relative_dir"], bundle)
    cfg = load_training_config(cfg_file, base_dir=pkg_root, mode=mode)
    audit_manifest_directory(cfg.data.partition_dir, require_all_classes=cfg.data.bundle_path is None, dataset_root=cfg.data.dataset_root)

    run_dir = cfg.output.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved configuration
    cfg.save_yaml(run_dir / "resolved_config.yaml")

    resume_path = Path(resume_checkpoint).resolve() if resume_checkpoint else None
    if resume_path:
        print(f"[*] Resuming training from checkpoint: {resume_path}")

    events_file = run_dir / "events.jsonl"
    log_file = run_dir / "runtime.log"
    EventLogger(events_file, cfg.run_id, attempt_id=1).log_phase(0, "prepare")

    print("=" * 70)
    print(f"LAUNCHING FLOWER SIMULATION ({mode.upper()})")
    print(f"[*] Run ID:     {cfg.run_id}")
    print(f"[*] Output Dir: {run_dir}")
    print(f"[*] Clients:    {cfg.federation.num_clients} | Max Rounds: {cfg.federation.max_rounds}")
    print(f"[*] Device:     Client: {cfg.runtime.client_device} | Server: {cfg.runtime.server_device}")
    print("=" * 70)

    num_supernodes = cfg.federation.num_clients
    proc = _run_simulation_subprocess(
        config_path=cfg_file,
        run_dir=run_dir,
        num_supernodes=num_supernodes,
        client_device=cfg.runtime.client_device,
        cpus_per_client=cfg.runtime.cpus_per_client,
        max_concurrent_clients=cfg.runtime.max_concurrent_clients,
        run_id=cfg.run_id,
        resume_checkpoint=resume_path,
        force_resume_stopped=force_resume_stopped,
        object_store_memory_mb=cfg.runtime.object_store_memory_mb,
    )

    renderer = ProgressRenderer(
        event_log_path=events_file,
        max_rounds=cfg.federation.max_rounds,
        description=f"FL {mode.capitalize()}",
        enabled=cfg.output.progress and not no_progress,
        client_logs_dir=run_dir / "client_logs",
    )

    log_fh = open(log_file, "w", encoding="utf-8")
    status = None
    exit_code = 0
    fatal_runtime = threading.Event()
    started_at = time.monotonic()

    def copy_stdout() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            if "Simulation Runtime crashed." in line or "An error was encountered. Ending simulation." in line:
                fatal_runtime.set()

    reader = threading.Thread(target=copy_stdout, name="fl-runtime-log-reader", daemon=True)
    reader.start()

    try:
        while proc.poll() is None:
            evt_status = renderer.poll()
            if evt_status in ("stopped", "failed"):
                status = evt_status
            if fatal_runtime.is_set() or (
                time.monotonic() - started_at > cfg.runtime.startup_timeout_seconds
                and not (run_dir / "server_started.json").exists()
            ):
                status = "failed"
                _terminate_process_tree(proc)
                break
            time.sleep(0.1)

        reader.join(timeout=5)

        final_status = renderer.poll()
        if final_status == "failed":
            status = final_status
        proc_exit = proc.returncode
        if proc_exit != 0:
            print(f"[X] Simulation process exited with code {proc_exit}. See {log_file}")
            exit_code = proc_exit
        elif status == "failed":
            print(f"[X] Server emitted a failed event. See {log_file}")
            exit_code = 1
        else:
            print("[*] Simulation completed successfully (code 0).")
    except KeyboardInterrupt:
        print("\n[!] Interrupt signal received! Terminating simulation process...")
        _terminate_process_tree(proc)
        summary_path = run_dir / "summary.json"
        previous = {}
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as file:
                previous = json.load(file)
        previous.update({
            "schema_version": 2, "status": "interrupted", "run_id": cfg.run_id,
            "stop_reason": "KeyboardInterrupt", "scientific_stage2_complete": False,
        })
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(previous, file, indent=2)
        exit_code = 130
    finally:
        reader.join(timeout=5)
        log_fh.close()

    if exit_code != 0:
        failure_logger = EventLogger(events_file, cfg.run_id)
        failure_logger.log_failed(renderer.completed_rounds, "Runtime failed or interrupted; inspect runtime.log")
        summary_path = run_dir / "summary.json"
        previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        previous.update(status="interrupted" if exit_code == 130 else "failed", run_id=cfg.run_id,
                        scientific_stage2_complete=False, last_completed_round=renderer.completed_rounds)
        summary_path.write_text(json.dumps(previous, indent=2), encoding="utf-8")

    if exit_code == 0 and log_file.exists():
        runtime_text = log_file.read_text(encoding="utf-8", errors="replace").lower()
        warnings = []
        if "access violation" in runtime_text:
            warnings.append("Windows/Ray shutdown access violation was observed after artifacts were committed")
        if "deprecated" in runtime_text and "run_simulation" in runtime_text:
            warnings.append("Flower run_simulation API emitted a deprecation warning")
        summary_path = run_dir / "summary.json"
        if warnings and summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as file:
                summary = json.load(file)
            if summary.get("status") == "completed":
                summary["status"] = "completed_with_warnings"
            summary["runtime_warnings"] = warnings
            with open(summary_path, "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2)
            print(f"[!] Run completed with {len(warnings)} runtime warning(s); see summary.json")

    # Post-train evaluation if enabled and exited successfully
    launcher_logger = EventLogger(events_file, cfg.run_id, attempt_id=1)
    def postprocessing_failed(error: str) -> int:
        launcher_logger.log_failed(renderer.completed_rounds, error)
        summary_path = run_dir / "summary.json"
        previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        previous.update(training_status=previous.get("status"), status="postprocessing_failed", error=error)
        summary_path.write_text(json.dumps(previous, indent=2), encoding="utf-8")
        renderer.close()
        return 1

    if exit_code == 0 and cfg.output.evaluate_test_after_train and mode != "smoke":
        best_pt = run_dir / "best.pt"
        if not best_pt.exists():
            return postprocessing_failed("Missing best.pt after completed training")
        if best_pt.exists():
            print("\n[*] Evaluating best model on global test set...")
            launcher_logger.log_phase(cfg.federation.max_rounds, "evaluate")
            renderer.poll()
            try:
                eval_code = run_evaluation(best_pt, config_path=cfg_file, device=cfg.runtime.server_device)
            except Exception as exc:
                return postprocessing_failed(f"Evaluation failed: {exc}")
            if eval_code != 0:
                return postprocessing_failed("Evaluation/report returned a failure")

    if exit_code == 0 and cfg.output.evaluate_test_after_train:
        try:
            from .reporting import generate_report

            launcher_logger.log_phase(cfg.federation.max_rounds, "report/plot")
            renderer.poll()
            print(f"[*] Report saved under {generate_report(run_dir)['run_id']}: {run_dir / 'report'}")
        except Exception as exc:
            print(f"[X] Report generation failed: {exc}")
            return postprocessing_failed(f"Report generation failed: {exc}")

    if exit_code == 0 and not cfg.output.evaluate_test_after_train:
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(status="calibration_completed", artifact_scope="calibration_no_test",
                           evaluation_pending=True, scientific_stage2_complete=False)
            from .budget_state import atomic_json
            atomic_json(summary_path, summary)
    renderer.close()
    return exit_code


def run_evaluation(
    checkpoint_path: str | Path,
    config_path: Optional[str | Path] = None,
    device: str = "cpu",
    output_dir: Optional[str | Path] = None,
    dataset_root_override: Optional[str | Path] = None,
    partition_dir_override: Optional[str | Path] = None,
) -> int:
    """Evaluate a saved checkpoint on the global test set."""
    cp_file = Path(checkpoint_path).resolve()
    if not cp_file.exists():
        print(f"[X] Checkpoint file not found: {cp_file}")
        return 1

    payload = load_checkpoint(cp_file)
    cp_cfg = payload.get("config", {})

    if config_path:
        config_file = Path(config_path).resolve()
        cfg = load_training_config(config_file, base_dir=Path(__file__).resolve().parents[1])
        partition_dir = cfg.data.partition_dir
        dataset_root = cfg.data.dataset_root
        batch_size = cfg.training.eval_batch_size
    elif "data" in cp_cfg:
        partition_dir = Path(cp_cfg["data"]["partition_dir"])
        dataset_root = Path(cp_cfg["data"]["dataset_root"])
        batch_size = int(cp_cfg.get("training", {}).get("eval_batch_size", 32))
    else:
        print("[X] Could not determine partition directory from checkpoint or config")
        return 1

    if partition_dir_override:
        partition_dir = Path(partition_dir_override).resolve()
    if dataset_root_override:
        dataset_root = Path(dataset_root_override).resolve()

    # Relocate version-1 checkpoints after the package directory rename by
    # looking up their immutable partition hash in the current index.
    package_root = Path(__file__).resolve().parents[1]
    if not partition_dir.exists() and cp_cfg.get("data", {}).get("partition_config_hash"):
        index_path = package_root / "data" / "partitions_train_v1" / "index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as file:
                index = json.load(file)
            part_hash = cp_cfg["data"]["partition_config_hash"]
            if part_hash in index.get("partitions", {}):
                partition_dir = package_root / index["partitions"][part_hash]["relative_dir"]
    if not partition_dir.exists():
        stale_parts = list(partition_dir.parts)
        if "data" in stale_parts:
            candidate = package_root.joinpath(*stale_parts[stale_parts.index("data"):])
            if candidate.exists():
                partition_dir = candidate
    if not dataset_root.exists():
        relocated_dataset = package_root.parent / "PlantVillage-Dataset" / "raw" / "color"
        if relocated_dataset.exists():
            dataset_root = relocated_dataset

    test_csv = partition_dir / "global_test.csv"
    if not test_csv.exists():
        print(f"[X] Global test manifest not found at {test_csv}")
        return 1

    protocol_fingerprint = validate_evaluation_partition(payload, partition_dir)
    from .content_audit import audit_image_content
    audit_image_content(partition_dir, dataset_root)

    print(f"[*] Loading test data from: {test_csv}")
    test_loader = build_evaluation_loader(
        manifest_path=test_csv,
        dataset_root=dataset_root,
        batch_size=batch_size,
    )

    model = create_mobilenet_v3_small(num_classes=38, weights=None)
    model_sd = payload.get("model_state_dict") or payload.get("best_model_state_dict")
    from .model import check_parameters_finite
    check_parameters_finite(model_sd)
    set_model_state_dict(model, model_sd, strict=True)

    print(f"[*] Evaluating on device: {device}...")
    eval_res = evaluate_model(model, test_loader, device=device, num_classes=38)
    class_names = (
        cfg.data.class_names if config_path
        else list(cp_cfg.get("data", {}).get("class_names", []))
    )
    if class_names:
        eval_res["class_names"] = class_names
        for row in eval_res["per_class"]:
            row["class_name"] = class_names[int(row["class_id"])]

    print("=" * 70)
    print("GLOBAL TEST EVALUATION REPORT")
    print("=" * 70)
    print(f"[*] Evaluated Samples: {eval_res['total_samples']}")
    print(f"[*] Test Loss:         {eval_res['loss']:.4f}")
    print(f"[*] Top-1 Accuracy:    {eval_res['accuracy'] * 100:.2f}%")
    print(f"[*] Macro-F1:          {eval_res['macro_f1']:.4f}")
    print("=" * 70)

    out_path = Path(output_dir).resolve() if output_dir else cp_file.parent
    bind_evaluation(eval_res, payload, cp_file, test_csv, protocol_fingerprint)
    write_evaluation(eval_res, out_path)
    print(f"[*] Metrics saved to {out_path / 'test_metrics.json'}")
    try:
        from .reporting import generate_report

        generate_report(out_path)
    except Exception as exc:
        print(f"[X] Evaluation succeeded but report generation failed: {exc}")
        return 1
    return 0


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m fl_training.cli",
        description="Federated Learning CLI for PlantVillage MobileNetV3 with Flower and FedAvg",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. prepare-data
    p_prep = subparsers.add_parser("prepare-data", help="Generate train/val/test partitions and index")
    p_prep.add_argument("--config", type=str, default="configs/partition_content_aware_v4.yaml", help="Content-aware Stage-2 partition config")
    p_prep.add_argument("--strict", action="store_true", help="Rehash all source images strictly")

    # 2. preflight
    p_pref = subparsers.add_parser("preflight", help="Check hardware, environment, data partitions, and config")
    p_pref.add_argument("--config", type=str, default="configs/train_fedavg_v4.yaml", help="Content-aware FedAvg config")

    # 3. smoke
    p_smoke = subparsers.add_parser("smoke", help="Run 2-client 2-round Flower integration smoke test")
    p_smoke.add_argument("--config", type=str, default="configs/train_smoke.yaml", help="Path to train_smoke.yaml")
    p_smoke.add_argument("--no-progress", action="store_true", help="Disable terminal tqdm progress bar")

    # 4. train
    p_train = subparsers.add_parser("train", help="Run Flower simulation training")
    p_train.add_argument("--config", type=str, default="configs/train_fedavg_v4.yaml", help="Content-aware FedAvg config")
    p_train.add_argument("--resume", type=str, default=None, help="Path to last.pt checkpoint to resume from")
    p_train.add_argument("--no-progress", action="store_true", help="Disable terminal tqdm progress bar")
    p_train.add_argument("--force-resume-stopped", action="store_true", help="Explicitly continue an early-stopped checkpoint")

    # 5. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate checkpoint on test set")
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    p_eval.add_argument("--config", type=str, default=None, help="Optional train config path")
    p_eval.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Evaluation device")
    p_eval.add_argument("--output", type=str, default=None, help="Output directory for test metrics")
    p_eval.add_argument("--dataset-root", type=str, default=None, help="Relocate a checkpoint to this dataset root")
    p_eval.add_argument("--partition-dir", type=str, default=None, help="Relocate a checkpoint to this partition directory")

    # 6. report
    p_report = subparsers.add_parser("report", help="Regenerate plots and diagnostics from run artifacts")
    p_report.add_argument("--run-dir", type=str, required=True, help="Run directory containing history/evaluation artifacts")

    # 7. baseline
    p_baseline = subparsers.add_parser("baseline", help="Train a Centralized or Local-only baseline")
    p_baseline.add_argument("--config", type=str, default="configs/train_fedavg_v4.yaml")
    p_baseline.add_argument("--mode", required=True, choices=["centralized", "local-only"])
    p_baseline.add_argument("--resume-dir", help="Resume a baseline run directory")

    # 8. sweep
    p_sweep = subparsers.add_parser("sweep", help="Plan or execute the controlled Stage-2 experiment matrix")
    p_sweep.add_argument("--config", type=str, default="configs/stage2_fedavg_main_v4.yaml")
    p_sweep.add_argument("--execute", action="store_true", help="Actually train; default is a safe dry run")
    p_sweep.add_argument("--collect", action="store_true", help="Rebuild comparison from existing run artifacts without training")

    args = parser.parse_args()

    if args.command == "prepare-data":
        code = run_prepare_data_cmd(args.config, strict=args.strict)
    elif args.command == "preflight":
        code = run_preflight(args.config)
    elif args.command == "smoke":
        code = run_simulation_launcher(args.config, mode="smoke", no_progress=args.no_progress)
    elif args.command == "train":
        code = run_simulation_launcher(
            args.config, mode="train", resume_checkpoint=args.resume,
            no_progress=args.no_progress, force_resume_stopped=args.force_resume_stopped,
        )
    elif args.command == "evaluate":
        code = run_evaluation(
            args.checkpoint, config_path=args.config, device=args.device,
            output_dir=args.output, dataset_root_override=args.dataset_root,
            partition_dir_override=args.partition_dir,
        )
    elif args.command == "report":
        from .reporting import generate_report

        result = generate_report(args.run_dir)
        print(f"[*] Report generated for {result['run_id']}: {Path(args.run_dir).resolve() / 'report'}")
        code = 0
    elif args.command == "baseline":
        from .baselines import run_baseline

        output = run_baseline(args.config, args.mode, resume_dir=args.resume_dir)
        print(f"[*] Baseline artifacts: {output}")
        code = 0
    elif args.command == "sweep":
        from .sweep import run_sweep

        output = run_sweep(args.config, execute=args.execute, collect=args.collect)
        action = "executed" if args.execute else ("collected" if args.collect else "planned only")
        print(f"[*] Sweep {action}: {output}")
        code = 0
    else:
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main_cli()
