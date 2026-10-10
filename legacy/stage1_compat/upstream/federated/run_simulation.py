"""FedAvg mô phỏng cục bộ với validation theo round và test cuối cùng."""

import argparse
from collections import OrderedDict
import sys
from pathlib import Path

import flwr as fl
import torch
from torch import nn, optim
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import create_model
from utils.config import DEFAULT_BATCH_SIZE, DEFAULT_NUM_CLIENTS, RESULTS_DIR, SEED, partition_path
from utils.data_partition import class_distribution, dirichlet_partition, load_partition, partition_summary, save_partition
from utils.data_utils import load_datasets, make_loader, save_json
from utils.evaluate import device, evaluate, train_one_epoch
from utils.plotting import plot_class_distribution, plot_history
from utils.reproducibility import set_seed


def get_parameters(model):
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def set_parameters(model, parameters) -> None:
    state = OrderedDict((key, torch.tensor(value)) for key, value in zip(model.state_dict().keys(), parameters))
    model.load_state_dict(state, strict=True)


def weighted_metrics(metrics):
    """Tổng hợp accuracy/Macro-F1 theo số mẫu đánh giá của client."""
    total = sum(num_examples for num_examples, _ in metrics)
    return {
        name: sum(num_examples * values.get(name, 0.0) for num_examples, values in metrics) / total
        for name in ("accuracy", "macro_f1")
    }


def weighted_parameters(client_parameters, client_samples):
    total_samples = sum(client_samples)
    return [
        sum(samples * parameters[index] for parameters, samples in zip(client_parameters, client_samples)) / total_samples
        for index in range(len(client_parameters[0]))
    ]


class ProgressFedAvg(fl.server.strategy.FedAvg):
    """FedAvg in metric sau mỗi server round trực tiếp trong terminal."""

    def __init__(self, total_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.total_rounds = total_rounds
        self.history_rows = []
        self.last_train_loss = 0.0
        self.current_parameters = None
        self.best_parameters = None
        self.best_validation_accuracy = -1.0
        self.best_round = 0

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None and aggregated[0] is not None:
            parameters, _ = aggregated
            self.current_parameters = [value.copy() for value in fl.common.parameters_to_ndarrays(parameters)]
            total = sum(fit_res.num_examples for _, fit_res in results)
            self.last_train_loss = sum(
                fit_res.num_examples * float(fit_res.metrics.get("train_loss", 0.0))
                for _, fit_res in results
            ) / max(total, 1)
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is not None and aggregated[0] is not None:
            loss, metrics = aggregated
            validation_accuracy = metrics.get("accuracy", 0.0)
            row = {"epoch": server_round, "train_loss": self.last_train_loss,
                   "validation_loss": loss, "validation_accuracy": validation_accuracy,
                   "validation_macro_f1": metrics.get("macro_f1", 0.0)}
            self.history_rows.append(row)
            if validation_accuracy > self.best_validation_accuracy and self.current_parameters is not None:
                self.best_validation_accuracy = validation_accuracy
                self.best_round = server_round
                self.best_parameters = [value.copy() for value in self.current_parameters]
            print(f"[FedAvg round {server_round}/{self.total_rounds}] val_loss={loss:.4f} | "
                  f"val_accuracy={validation_accuracy:.4f} | macro-F1={row['validation_macro_f1']:.4f}")
        return aggregated


class PlantClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_set, validation_set, class_count, args):
        self.client_id = client_id
        self.train_loader = make_loader(train_set, args.batch_size, shuffle=True)
        self.validation_loader = make_loader(validation_set, args.batch_size)
        # Client nhận global weights ngay trước khi fit/evaluate; không cần nạp lại ImageNet weights.
        self.model = create_model(class_count, pretrained=False).to(device())
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.client_id}] fit started: samples={len(self.train_loader.dataset)}", flush=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        losses = [train_one_epoch(self.model, self.train_loader, optimizer, self.criterion, device())
                  for _ in range(self.args.local_epochs)]
        print(f"[Client {self.client_id}] fit finished", flush=True)
        return get_parameters(self.model), len(self.train_loader.dataset), {"train_loss": sum(losses) / len(losses)}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        metrics = evaluate(self.model, self.validation_loader, self.criterion, device())
        print(f"[Client {self.client_id}] evaluation finished", flush=True)
        return float(metrics["loss"]), len(self.validation_loader.dataset), {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}


def run_sequential_fedavg(global_model, train_set, validation_set, partitions, class_count, args):
    """Run the same FedAvg update locally without Flower's Ray simulator."""
    clients = [
        PlantClient(client_id, Subset(train_set, indices), validation_set, class_count, args)
        for client_id, indices in enumerate(partitions)
    ]
    current_parameters = get_parameters(global_model)
    history_rows = []
    best_parameters = None
    best_validation_accuracy = -1.0
    best_round = 0

    for server_round in range(1, args.rounds + 1):
        fitted_parameters, sample_counts, train_losses = [], [], []
        for client in clients:
            parameters, samples, metrics = client.fit(current_parameters, {})
            fitted_parameters.append(parameters)
            sample_counts.append(samples)
            train_losses.append(metrics["train_loss"])
        current_parameters = weighted_parameters(fitted_parameters, sample_counts)
        validation_results = []
        for client in clients:
            loss, samples, metrics = client.evaluate(current_parameters, {})
            validation_results.append((samples, {"loss": loss, **metrics}))
        total_validation = sum(samples for samples, _ in validation_results)
        validation_loss = sum(samples * values["loss"] for samples, values in validation_results) / total_validation
        aggregated = weighted_metrics([(samples, values) for samples, values in validation_results])
        row = {"epoch": server_round, "train_loss": sum(loss * samples for loss, samples in zip(train_losses, sample_counts)) / sum(sample_counts),
               "validation_loss": validation_loss, "validation_accuracy": aggregated["accuracy"],
               "validation_macro_f1": aggregated["macro_f1"]}
        history_rows.append(row)
        if row["validation_accuracy"] > best_validation_accuracy:
            best_validation_accuracy = row["validation_accuracy"]
            best_round = server_round
            best_parameters = [value.copy() for value in current_parameters]
        print(f"[FedAvg round {server_round}/{args.rounds}] val_loss={validation_loss:.4f} | "
              f"val_accuracy={row['validation_accuracy']:.4f} | macro-F1={row['validation_macro_f1']:.4f}")
    return history_rows, best_parameters, best_round, best_validation_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower FedAvg baseline")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--partition-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--backend", choices=("sequential", "flower"), default="sequential")
    args = parser.parse_args(); set_seed(args.seed)
    train_set, validation_set, test_set, class_names = load_datasets(args.seed)
    partition_file = args.partition_file or partition_path(args.alpha, args.seed, args.clients)
    if partition_file.exists():
        partitions = load_partition(partition_file, train_set.targets, expected_clients=args.clients)
    else:
        partitions = dirichlet_partition(train_set.targets, args.clients, args.alpha, args.seed)
        save_partition(partition_file, train_set.targets, partitions, args.alpha, args.seed)
    global_model = create_model(len(class_names), not args.no_pretrained)

    print(f"Training FedAvg | {args.clients} clients | {args.rounds} rounds | local epochs={args.local_epochs} | alpha={args.alpha}")
    if args.backend == "sequential":
        rounds, best_parameters, best_round, best_validation_accuracy = run_sequential_fedavg(
            global_model, train_set, validation_set, partitions, len(class_names), args)
    else:
        def client_fn(context):
            client_id = int(context.node_config["partition-id"])
            return PlantClient(client_id, Subset(train_set, partitions[client_id]), validation_set, len(class_names), args).to_client()

        strategy = ProgressFedAvg(
            total_rounds=args.rounds,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=args.clients, min_evaluate_clients=args.clients, min_available_clients=args.clients,
            initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(global_model)),
            evaluate_metrics_aggregation_fn=weighted_metrics,
        )
        client_resources = {"num_cpus": 1, "num_gpus": 1.0 if torch.cuda.is_available() else 0.0}
        print(f"Flower client resources: {client_resources}")
        fl.simulation.start_simulation(
            client_fn=client_fn, num_clients=args.clients,
            client_resources=client_resources,
            config=fl.server.ServerConfig(num_rounds=args.rounds), strategy=strategy,
        )
        rounds = strategy.history_rows
        best_parameters = strategy.best_parameters
        best_round = strategy.best_round
        best_validation_accuracy = strategy.best_validation_accuracy
    if best_parameters is None:
        raise RuntimeError("Không có checkpoint FedAvg để đánh giá test.")
    set_parameters(global_model, best_parameters)
    test_metrics = evaluate(global_model.to(device()), make_loader(test_set, args.batch_size), nn.CrossEntropyLoss(), device())
    payload = {"experiment": "fedavg", "seed": args.seed, "alpha": args.alpha, "rounds": args.rounds,
               "local_epochs": args.local_epochs,
               "split": {"train": len(train_set), "validation": len(validation_set), "test": len(test_set)},
               "selection_metric": "validation_accuracy", "best_round": best_round,
               "best_validation_accuracy": best_validation_accuracy, "test_metrics": test_metrics,
               "partition_file": str(partition_file),
               "partition": partition_summary(train_set.targets, partitions),
               "class_distribution": class_distribution(train_set.targets, partitions), "history": rounds}
    save_json("fedavg_metrics.json", payload)
    plot_history(rounds, RESULTS_DIR / "fedavg_history.png", f"FedAvg (alpha={args.alpha})")
    plot_class_distribution(payload["class_distribution"], class_names, RESULTS_DIR / "fedavg_class_distribution.png",
                            f"FedAvg class distribution (alpha={args.alpha})")
    if rounds:
        print(f"FedAvg test (best validation round {best_round}): "
              f"acc={test_metrics['accuracy']:.4f}, macro-F1={test_metrics['macro_f1']:.4f}")
    print("FedAvg training hoàn thành.")


if __name__ == "__main__":
    main()
