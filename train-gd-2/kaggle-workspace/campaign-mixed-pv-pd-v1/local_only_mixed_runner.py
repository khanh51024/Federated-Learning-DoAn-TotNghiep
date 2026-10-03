"""Pilot local-only baseline: one independent model per audited client shard.

Each model starts from the same pinned W0 and gets the same local epoch budget
as one client in the comparable FedAvg run. No weights cross client boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from fedavg_mixed_runner import (
    eval_collate_fn, evaluate_metrics, model_from_global_state, set_seed, train_client,
)
from plant_data_contract.dataset import CanonicalClassificationDataset, build_domain_balanced_sampler
from plant_data_contract.integrity import verify_release_integrity
from plant_data_contract.models import create_model
from plant_data_contract.partitions import verify_partition_audit, verify_partition_directory
from plant_data_contract.training_progress import TrainingProgress
from plant_data_contract.transforms import get_transforms


def collate(batch):
    return (torch.stack([row[0] for row in batch]),
            torch.tensor([row[1] for row in batch], dtype=torch.long),
            [row[2] for row in batch])


def run(args):
    root = args.dataset_root.resolve()
    release = root / "mixed/pv_pd_v3"
    spec = json.loads(args.spec_file.read_text(encoding="utf-8"))
    hparams = spec["hyperparameters"]["fedavg"]
    count = int(hparams["clients_per_round"])
    ids = [f"client_{i:02d}" for i in range(count)]
    release_sha = verify_release_integrity(release)["release_manifest_sha256"]
    if release_sha != spec["release_manifest_sha256"]:
        raise ValueError("Local-only release differs from the pinned FedAvg spec")
    partitions = verify_partition_directory(release, args.partition_scheme, ids)
    verify_partition_audit(release, args.partition_scheme, ids, release_sha)
    if args.client_id is not None and args.client_id not in ids:
        raise ValueError("Unknown client ID")
    selected = [args.client_id] if args.client_id is not None else ids
    rounds = args.rounds if args.rounds is not None else int(hparams["rounds"])
    if rounds < 1 or rounds > 60:
        raise ValueError("Local-only rounds must be in [1, 60]")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    resume = getattr(args, "resume", False)
    if resume and len(selected) != 1:
        raise ValueError("Resume requires exactly one --client-id")
    if not resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}")

    set_seed(args.seed)
    w0_path = args.pretrained_weights.with_name(spec["pretrained_w0_state_file"])
    model, fingerprint, _ = create_model(
        num_classes=38, seed=args.seed, init="imagenet_v1",
        pretrained_weights_path=args.pretrained_weights, w0_state_path=w0_path,
        expected_w0_state_sha256=spec["pretrained_w0_state_sha256"],
    )
    if fingerprint != spec["expected_w0_fingerprint"]:
        raise ValueError("Local-only W0 fingerprint differs from the FedAvg campaign")
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    transform = spec.get("preprocessing", "canonical_v1")
    val = CanonicalClassificationDataset.from_manifest(
        release / "manifests/val.jsonl", dataset_root=root,
        transform=get_transforms(transform, is_train=False), return_metadata=True,
    )
    val_loader = DataLoader(val, batch_size=int(hparams["client_batch_size"]),
                            shuffle=False, num_workers=0, collate_fn=eval_collate_fn)
    requested_device = getattr(args, "device", "auto")
    if requested_device not in ("auto", "cuda", "cpu"):
        raise ValueError("--device must be auto, cuda, or cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this Python environment has no usable CUDA device")
    device = torch.device("cuda:0" if requested_device == "cuda" or
                          (requested_device == "auto" and torch.cuda.is_available()) else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"backend": "local_only", "seed": args.seed,
               "device": str(device),
               "partition_scheme": args.partition_scheme,
               "partial_batches": args.max_batches,
               "plantdoc_sampling_ratio": args.plantdoc_sampling_ratio,
               "release_manifest_sha256": release_sha, "w0_fingerprint": fingerprint,
               "rounds": rounds, "clients": {}}

    batch_size = int(hparams["client_batch_size"])
    local_epochs = int(hparams["client_epochs"])
    steps_per_round = sum(
        min((partitions["client_metrics"][cid]["n_k"] + batch_size - 1) // batch_size
            * local_epochs, args.max_batches)
        if args.max_batches else
        (partitions["client_metrics"][cid]["n_k"] + batch_size - 1) // batch_size
        * local_epochs
        for cid in selected
    )
    progress = TrainingProgress(rounds * steps_per_round)
    for cid in selected:
        # Re-seed each independent client; W0 remains byte-identical.
        set_seed(args.seed)
        shard = release / "partitions" / args.partition_scheme / f"{cid}.jsonl"
        dataset = CanonicalClassificationDataset.from_manifest(
            shard, dataset_root=root,
            transform=get_transforms(transform, is_train=True), return_metadata=True,
        )
        n_k = partitions["client_metrics"][cid]["n_k"]
        train_labels = {int(row["class_id"]) for row in dataset.samples}
        val_rows = val.samples
        def seen_label_fraction(rows):
            return (sum(int(row["class_id"]) in train_labels for row in rows) / len(rows)
                    if rows else 0.0)
        class_coverage = {
            "train_label_count": len(train_labels),
            "missing_class_ids": sorted(set(range(38)) - train_labels),
            "validation_seen_label_fraction": seen_label_fraction(val_rows),
            "plantdoc_validation_seen_label_fraction": seen_label_fraction(
                [row for row in val_rows if row.get("source_domain") == "plantdoc"]),
            "plantvillage_validation_seen_label_fraction": seen_label_fraction(
                [row for row in val_rows if row.get("source_domain") == "plantvillage"]),
        }
        has_pd = any(row.get("source_domain") == "plantdoc" for row in dataset.samples)
        ratio = str(args.plantdoc_sampling_ratio).lower()
        use_sampling = ratio not in ("none", "0", "0.0", "", "false")
        sampler = (build_domain_balanced_sampler(dataset, plantdoc_weight=float(ratio))
                   if use_sampling and has_pd else None)
        loader = DataLoader(dataset, batch_size=int(hparams["client_batch_size"]),
                            shuffle=sampler is None, sampler=sampler,
                            num_workers=0, collate_fn=collate)
        state = {key: value.clone() for key, value in initial.items()}
        scaler = GradScaler("cuda", enabled=bool(hparams.get("amp", False) and torch.cuda.is_available()),
                            init_scale=2048.0)
        best = {"plantdoc_macro_f1_supported": -1.0}
        client_dir = args.output_dir / cid
        client_dir.mkdir(exist_ok=resume)
        start_round = 1
        if resume:
            checkpoint = torch.load(client_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            expected = {"client_id": cid, "partition_scheme": args.partition_scheme,
                        "release_manifest_sha256": release_sha, "w0_fingerprint": fingerprint,
                        "seed": args.seed, "rounds": rounds,
                        "plantdoc_sampling_ratio": str(args.plantdoc_sampling_ratio)}
            for key, value in expected.items():
                if checkpoint.get(key) != value:
                    raise ValueError(f"Local-only resume mismatch for {key}")
            state = checkpoint["model_state_dict"]
            best = checkpoint["best_validation"]
            start_round = int(checkpoint["round"]) + 1
            if not 1 <= start_round <= rounds + 1:
                raise ValueError("Invalid last completed round in local-only checkpoint")
            torch.set_rng_state(checkpoint["torch_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            random.setstate(checkpoint["python_rng_state"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
            progress.completed += (start_round - 1) * ((n_k + batch_size - 1) // batch_size) * local_epochs
            progress.summary(f"Resuming {cid} at round {start_round}/{rounds}")
        for round_no in range(start_round, rounds + 1):
            if sampler is not None:
                sampler.set_epoch((round_no - 1) * local_epochs)
            state, telemetry = train_client(
                client_id=cid, global_state=state, loader=loader, device=device,
                epochs=int(hparams["client_epochs"]), lr=float(hparams["client_lr"]),
                momentum=float(hparams.get("client_momentum", 0.0)),
                weight_decay=float(hparams["client_weight_decay"]),
                amp=bool(hparams.get("amp", False)), scaler=scaler,
                raw_sample_count=n_k, max_batches=args.max_batches,
                progress_callback=lambda loss, cid=cid, round_no=round_no: progress.update(
                    increment=1, phase="train", position=f"{cid} round {round_no}/{rounds}",
                    loss=loss, lr=float(hparams["client_lr"])),
            )
            eval_model = model_from_global_state(state).to(device)
            result = evaluate_metrics(eval_model, val_loader, device,
                                      max_batches=args.max_batches)
            pd = result.get("per_source", {}).get("plantdoc", {})
            pv = result.get("per_source", {}).get("plantvillage", {})
            pd_f1 = float(pd.get("raw_macro_f1_supported", 0.0))
            if pd_f1 > best["plantdoc_macro_f1_supported"]:
                best = {"round": round_no, "plantdoc_macro_f1_supported": pd_f1,
                        "plantvillage_macro_f1_supported": float(pv.get("raw_macro_f1_supported", 0.0)),
                        "accuracy": float(result["accuracy"])}
                best_path = client_dir / "checkpoint_best.pt"
                best_tmp = best_path.with_suffix(".pt.tmp")
                torch.save({"round": round_no, "model_state_dict": state,
                            "w0_fingerprint": fingerprint,
                            "release_manifest_sha256": release_sha}, best_tmp)
                os.replace(best_tmp, best_path)
            last_path = client_dir / "checkpoint_last.pt"
            last_tmp = last_path.with_suffix(".pt.tmp")
            torch.save({"round": round_no, "model_state_dict": state,
                        "best_validation": best, "client_id": cid,
                        "partition_scheme": args.partition_scheme,
                        "release_manifest_sha256": release_sha,
                        "w0_fingerprint": fingerprint, "seed": args.seed,
                        "rounds": rounds,
                        "plantdoc_sampling_ratio": str(args.plantdoc_sampling_ratio),
                        "torch_rng_state": torch.get_rng_state(),
                        "numpy_rng_state": np.random.get_state(),
                        "python_rng_state": random.getstate(),
                        "scaler_state_dict": scaler.state_dict(),
                        "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, last_tmp)
            os.replace(last_tmp, last_path)
            progress.summary(f"{cid} round {round_no}/{rounds}: loss={telemetry['loss']:.4f} "
                             f"PlantDoc F1={pd_f1:.4f}")
        summary["clients"][cid] = {
            "n_k": n_k, "class_coverage": class_coverage, "best_validation": best,
        }

    progress.close()

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--spec-file", required=True, type=Path)
    parser.add_argument("--pretrained-weights", required=True, type=Path)
    parser.add_argument("--partition-scheme", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--client-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--plantdoc-sampling-ratio", default="none")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
