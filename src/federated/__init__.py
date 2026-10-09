"""Federated-learning package.

Keep package import lightweight so ``python -m src.federated.server_app`` and
``python -m src.federated.client_app`` do not pre-import the module being run.
Heavy client/server symbols are exposed lazily for backward compatibility.
"""

from .strategy import aggregate_fedavg, fedavg_weighted_average
from .telemetry import ClientRoundTelemetry, ServerRoundTelemetry, TelemetryManager
from .weight_inspector import export_weight_stats_csv, inspect_layer_weights, inspect_model_state
from .faulty_client import apply_fault_mode

__all__ = [
    "aggregate_fedavg",
    "fedavg_weighted_average",
    "ClientRoundTelemetry",
    "ServerRoundTelemetry",
    "TelemetryManager",
    "inspect_layer_weights",
    "inspect_model_state",
    "export_weight_stats_csv",
    "apply_fault_mode",
    "PlantVillageClient",
    "start_client_process",
    "run_federated_simulation",
    "start_flower_server",
]


def __getattr__(name):
    if name in {"PlantVillageClient", "start_client_process"}:
        from .client_app import PlantVillageClient, start_client_process
        return {"PlantVillageClient": PlantVillageClient, "start_client_process": start_client_process}[name]
    if name in {"run_federated_simulation", "start_flower_server"}:
        from .server_app import run_federated_simulation, start_flower_server
        return {
            "run_federated_simulation": run_federated_simulation,
            "start_flower_server": start_flower_server,
        }[name]
    raise AttributeError(name)
