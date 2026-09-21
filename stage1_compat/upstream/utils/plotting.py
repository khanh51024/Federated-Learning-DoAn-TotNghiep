from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: list[dict], path: Path, title: str) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    evaluation_prefix = "validation" if "validation_loss" in history[0] else "test"
    if f"{evaluation_prefix}_loss" in history[0]:
        axes[0].plot(epochs, [row[f"{evaluation_prefix}_loss"] for row in history], label=f"{evaluation_prefix.title()} loss")
    axes[1].plot(epochs, [row[f"{evaluation_prefix}_accuracy"] for row in history], label="Accuracy")
    axes[1].plot(epochs, [row[f"{evaluation_prefix}_macro_f1"] for row in history], label="Macro-F1")
    axes[0].set_title("Loss"); axes[1].set_title(f"{evaluation_prefix.title()} metrics")
    for axis in axes:
        axis.set_xlabel("Epoch / round"); axis.legend(); axis.grid(alpha=0.25)
    fig.suptitle(title); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_class_distribution(distribution: list[list[int]], class_names: list[str], path: Path, title: str) -> None:
    """Save a client-by-class heatmap that documents the non-IID partition."""
    values = np.asarray(distribution)
    fig_height = max(6, len(class_names) * 0.24)
    fig, axis = plt.subplots(figsize=(8, fig_height))
    image = axis.imshow(values.T, aspect="auto", cmap="YlOrRd")
    axis.set_xticks(range(len(distribution)), [f"Client {index}" for index in range(len(distribution))])
    axis.set_yticks(range(len(class_names)), class_names)
    axis.tick_params(axis="y", labelsize=7)
    axis.set_xlabel("Client"); axis.set_ylabel("Class"); axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Training samples")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
