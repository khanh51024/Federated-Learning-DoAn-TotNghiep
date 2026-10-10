import argparse
import sys
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import create_model
from utils.config import DEFAULT_BATCH_SIZE, DEFAULT_NUM_CLIENTS, RESULTS_DIR, SEED, partition_path
from utils.data_partition import class_distribution, dirichlet_partition, load_partition, partition_summary, save_partition
from utils.data_utils import load_datasets, make_loader, save_json
from utils.evaluate import device, evaluate, train_one_epoch
from utils.plotting import plot_class_distribution
from utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-only lower-bound baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--partition-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(); set_seed(args.seed)
    train_set, _, test_set, class_names = load_datasets(args.seed)
    partition_file = args.partition_file or partition_path(args.alpha, args.seed, args.clients)
    if partition_file.exists():
        partitions = load_partition(partition_file, train_set.targets, expected_clients=args.clients)
    else:
        partitions = dirichlet_partition(train_set.targets, args.clients, args.alpha, args.seed)
        save_partition(partition_file, train_set.targets, partitions, args.alpha, args.seed)
    test_loader = make_loader(test_set, args.batch_size)
    target_device, criterion = device(), nn.CrossEntropyLoss()
    client_results = []
    print(f"Training Local-only | {args.clients} clients | {args.epochs} epochs/client | alpha={args.alpha}")

    for client_id, indices in enumerate(partitions):
        set_seed(args.seed)  # mọi client bắt đầu từ cùng khởi tạo
        model = create_model(len(class_names), not args.no_pretrained).to(target_device)
        loader = make_loader(Subset(train_set, indices), args.batch_size, shuffle=True)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, loader, optimizer, criterion, target_device, f"Client {client_id + 1}/{args.clients} | epoch {epoch}/{args.epochs}")
        metrics = evaluate(model, test_loader, criterion, target_device)
        metrics["client_id"] = client_id; metrics["train_samples"] = len(indices)
        client_results.append(metrics)
        print(f"Client {client_id}: acc={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}")

    accuracies = [row["accuracy"] for row in client_results]
    f1_scores = [row["macro_f1"] for row in client_results]
    payload = {"experiment": "local_only", "seed": args.seed, "alpha": args.alpha, "epochs": args.epochs,
               "partition_file": str(partition_file),
               "partition": partition_summary(train_set.targets, partitions), "clients": client_results,
               "class_distribution": class_distribution(train_set.targets, partitions),
               "average_accuracy": sum(accuracies) / len(accuracies),
               "accuracy_std": float(torch.tensor(accuracies).std(unbiased=False)),
               "accuracy_min": min(accuracies), "accuracy_max": max(accuracies),
               "average_macro_f1": sum(f1_scores) / len(f1_scores),
               "macro_f1_std": float(torch.tensor(f1_scores).std(unbiased=False)),
               "macro_f1_min": min(f1_scores), "macro_f1_max": max(f1_scores)}
    save_json("local_only_metrics.json", payload)
    plot_class_distribution(payload["class_distribution"], class_names, RESULTS_DIR / "local_only_class_distribution.png",
                            f"Local-only class distribution (alpha={args.alpha})")
    print("Local-only training hoàn thành.")


if __name__ == "__main__":
    main()
