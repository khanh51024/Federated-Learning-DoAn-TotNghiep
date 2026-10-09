import argparse
import copy
import sys
from pathlib import Path

import torch
from torch import nn, optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import create_model
from utils.config import DEFAULT_BATCH_SIZE, RESULTS_DIR, SEED
from utils.data_utils import load_datasets, make_loader, save_json
from utils.evaluate import device, evaluate, train_one_epoch
from utils.plotting import plot_history
from utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized upper-bound baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    train_set, validation_set, test_set, class_names = load_datasets(args.seed)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True)
    validation_loader = make_loader(validation_set, args.batch_size)
    test_loader = make_loader(test_set, args.batch_size)
    target_device = device()
    model = create_model(len(class_names), not args.no_pretrained).to(target_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_accuracy, best_epoch, best_state, history = -1.0, 0, None, []
    print(f"Training Centralized | device={target_device} | epochs={args.epochs}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, target_device, f"Epoch {epoch}/{args.epochs}")
        metrics = evaluate(model, validation_loader, criterion, target_device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": metrics["loss"], "validation_accuracy": metrics["accuracy"], "validation_macro_f1": metrics["macro_f1"]})
        print(f"Epoch {epoch}/{args.epochs}: val_acc={metrics['accuracy']:.4f}, val_macro-F1={metrics['macro_f1']:.4f}")
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = metrics["accuracy"], epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, target_device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "class_names": class_names, "epoch": best_epoch,
                "validation_accuracy": best_accuracy, "test_metrics": test_metrics}, RESULTS_DIR / "centralized_best_model.pth")
    payload = {"experiment": "centralized", "seed": args.seed, "epochs": args.epochs,
               "split": {"train": len(train_set), "validation": len(validation_set), "test": len(test_set)},
               "selection_metric": "validation_accuracy", "best_epoch": best_epoch,
               "best_validation_accuracy": best_accuracy, "test_metrics": test_metrics, "history": history}
    save_json("centralized_metrics.json", payload)
    plot_history(history, RESULTS_DIR / "centralized_history.png", "Centralized baseline (validation)")
    print("Centralized training hoàn thành.")


if __name__ == "__main__":
    main()
