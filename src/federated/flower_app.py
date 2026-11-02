"""Chạy học liên kết của scripts/train_v4.py trên Flower: ServerApp, ClientApp, mô phỏng bằng Ray (N1.2, N3.1 của yêu cầu đề tài).

Máy chủ dùng ĐÚNG vòng lặp của train_v4.py (đánh giá val, nhật ký, checkpoint, chọn mô hình tốt nhất); chỉ khác cách chạy một vòng: `make_round_fn` gửi trọng số xuống
qua tin nhắn Flower và nhận về, thay cho `simulate_round`. Client dùng đúng `train_one_client` của đường một tiến trình và đúng các lớp trong src/federated/algorithms.py:

  máy chủ giữ   mô hình chung, biến điều khiển chung c của SCAFFOLD, vận tốc của FedAvgM
  client giữ    c_i của SCAFFOLD, BatchNorm riêng của FedBN, mô hình vòng trước của MOON (`client_state`, lưu trên đĩa của client, không gửi đi)
  tin nhắn      trọng số và phần thêm của thuật toán (`download`, `upload`). Byte ghi vào nhật ký là kích thước thật của phần mảng trong tin nhắn, chưa tính khung gRPC.

Khác đường một tiến trình một chút về số ngẫu nhiên: dropout trong client lấy từ bộ sinh của tiến trình client, nên hai đường không trùng từng bit;
N1.2 kiểm chứng bằng chênh lệch độ chính xác so với độ lệch giữa các hạt giống. Chạy tiếp sau khi bị ngắt thì bộ nạp ảnh của client bắt đầu lại, nên run đó không tái lập từng bit.

`flwr` chỉ được nạp trong hàm: bộ train một tiến trình không cần cài Flower (tests/test_train_requirements.py).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

ROUND_MESSAGE = "train"
_CLIENT_CACHE: dict[int, dict[str, Any]] = {}     # theo từng tiến trình client (actor Ray): bộ nạp ảnh giữ worker sống qua các vòng, như persistent_workers của đường một tiến trình
LOADER_BUILDS = {"count": 0}


@dataclasses.dataclass(frozen=True)
class ClientPlan:
    """Công thức máy chủ gửi cho client; đủ để client tự dựng dữ liệu và thuật toán của mình."""
    recipe: dict[str, Any]
    release_dir: str
    release_sha: str
    scheme: str
    limit: int
    algorithm: str
    algorithm_options: dict[str, float]
    state_dir: str
    device: str


def state_to_record(state: dict[str, Any]) -> Any:
    from flwr.app import ArrayRecord

    return ArrayRecord(torch_state_dict=dict(state)) if state else ArrayRecord()


def record_to_state(record: Any) -> dict[str, Any]:
    return dict(record.to_torch_state_dict()) if len(record) else {}


def record_bytes(record: Any) -> int:
    """Kích thước thật của phần mảng trong tin nhắn (đã tuần tự hóa, có cả đầu mục của từng mảng)."""
    return sum(len(array.data) for array in record.values())


# --------------------------------------------------------------------------- phía client
def state_path(plan: ClientPlan, client: int, step: int) -> Path:
    return Path(plan.state_dir) / f"client_{client:02d}.step{step:04d}.pt"


def _client_env(plan: ClientPlan, client: int) -> dict[str, Any]:
    """Dữ liệu, mô hình khung và thuật toán của client; dựng một lần trong mỗi tiến trình."""
    if client in _CLIENT_CACHE:
        return _CLIENT_CACHE[client]
    import torch

    from src.federated.algorithms import make_algorithm
    from src.model.mobilenetv3 import create_model
    from src.training.client_round import Recipe, prepare_client, train_transform
    from src.training.data import NUM_CLIENTS, load_release, load_shard, subset

    recipe = Recipe(**plan.recipe)
    device = torch.device(plan.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("client được giao chạy trên GPU nhưng tiến trình này không thấy GPU; kiểm tra client_resources của Ray")
    release_dir = Path(plan.release_dir)
    release = load_release(release_dir, plan.release_sha)
    rows = subset(load_shard(release_dir, plan.scheme, client, release), plan.limit // NUM_CLIENTS if plan.limit else 0, recipe.seed + client)
    data = prepare_client(rows, client, recipe, train_transform(recipe), device, persistent=True)
    LOADER_BUILDS["count"] += 1
    template = create_model(num_classes=38, pretrained=False).to(device)
    if recipe.channels_last:
        template = template.to(memory_format=torch.channels_last)
    options = plan.algorithm_options
    algo = make_algorithm(plan.algorithm, template, NUM_CLIENTS, mu=options["mu"], moon_mu=options["moon_mu"], moon_tau=options["moon_tau"],
                          server_momentum=options["server_momentum"], server_lr=options["server_lr"], momentum=options["momentum"])
    _CLIENT_CACHE[client] = {"recipe": recipe, "device": device, "data": data, "template": template, "algo": algo}
    return _CLIENT_CACHE[client]


def client_step(plan: ClientPlan, client: int, step: int, lr: float, parent: dict[str, Any], server_extra: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Việc của client trong một vòng: nạp trạng thái riêng của vòng trước, học, lưu trạng thái riêng của vòng này. Trả (trọng số, phần thêm gửi lên, số liệu)."""
    import copy

    import torch

    from src.model.mobilenetv3 import set_parameters
    from src.training.client_round import train_one_client

    env = _client_env(plan, client)
    algo = env["algo"]
    algo.load_download(server_extra)
    if step > 0:
        previous = state_path(plan, client, step - 1)
        if previous.is_file():
            algo.load_client_state(client, torch.load(previous, map_location="cpu", weights_only=False))
    local = copy.deepcopy(env["template"])
    set_parameters(local, parent)
    state, stats = train_one_client(local, client, parent, algo, env["data"], lr, env["recipe"], env["device"])
    extra_up = algo.upload(client)
    private = algo.client_state(client)
    if private:
        target = state_path(plan, client, step)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        torch.save(private, tmp)
        tmp.replace(target)
        older = state_path(plan, client, step - 2)      # giữ hai đời: máy chủ có thể bị ngắt trước khi chốt vòng này
        if older.is_file():
            older.unlink()
    return state, extra_up, stats


def make_client_app() -> Any:
    from flwr.app import ConfigRecord, Message, RecordDict
    from flwr.clientapp import ClientApp

    app = ClientApp()

    @app.query()
    def identify(message: Message, context: Any) -> Message:
        """Máy chủ hỏi nút này giữ shard nào."""
        return Message(RecordDict({"identity": ConfigRecord({"partition_id": int(context.node_config["partition-id"])})}), reply_to=message)

    @app.train()
    def train(message: Message, context: Any) -> Message:
        config = message.content["config"]
        plan = ClientPlan(**json.loads(str(config["plan"])))
        client = int(context.node_config["partition-id"])
        state, extra_up, stats = client_step(plan, client, int(config["step"]), float(config["lr"]), record_to_state(message.content["arrays"]), record_to_state(message.content["extra"]))
        arrays, extra = state_to_record(state), state_to_record(extra_up)
        reply_stats = {**stats, "num_examples": _CLIENT_CACHE[client]["data"]["num_examples"], "bytes_up": record_bytes(arrays) + record_bytes(extra)}
        return Message(RecordDict({"arrays": arrays, "extra": extra, "report": ConfigRecord({"stats": json.dumps(reply_stats)})}), reply_to=message)

    return app


# --------------------------------------------------------------------------- phía máy chủ
def _replies_or_raise(replies: Any, expected: int, what: str) -> list[Any]:
    got = list(replies)
    errors = [reply.error.reason for reply in got if reply.has_error()]
    if errors:
        raise RuntimeError(f"client báo lỗi khi {what}: {errors}")
    if len(got) != expected:
        raise RuntimeError(f"chỉ {len(got)}/{expected} client trả lời khi {what}; dừng run, không gộp thiếu client để giữ đúng công thức")
    return got


def locate_clients(grid: Any, num_clients: int, timeout: float) -> dict[int, int]:
    """{số thứ tự shard: mã nút Flower}. Mỗi shard phải có đúng một nút."""
    from flwr.app import ConfigRecord, Message, MessageType, RecordDict

    nodes = list(grid.get_node_ids())
    if len(nodes) != num_clients:
        raise RuntimeError(f"có {len(nodes)} nút Flower, cần {num_clients}")
    asks = [Message(RecordDict({"ask": ConfigRecord({"what": "partition_id"})}), dst_node_id=node, message_type=MessageType.QUERY) for node in nodes]
    replies = _replies_or_raise(grid.send_and_receive(asks, timeout=timeout), num_clients, "hỏi shard")
    located = {int(reply.content["identity"]["partition_id"]): int(reply.metadata.src_node_id) for reply in replies}
    if sorted(located) != list(range(num_clients)):
        raise RuntimeError(f"các nút giữ shard {sorted(located)}, cần đủ 0..{num_clients - 1}")
    return located


def make_round_fn(grid: Any, algo: Any, plan: ClientPlan, located: dict[int, int], timeout: float) -> Callable[[int, float, dict[str, Any]], tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    from flwr.app import ConfigRecord, Message, MessageType, RecordDict

    plan_json = json.dumps(dataclasses.asdict(plan))

    def run_round(step: int, lr: float, parent: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        arrays, extra = state_to_record(parent), state_to_record(algo.download())
        bytes_down = record_bytes(arrays) + record_bytes(extra)
        config = ConfigRecord({"plan": plan_json, "step": step, "lr": lr})
        messages = [Message(RecordDict({"arrays": arrays, "extra": extra, "config": config}), dst_node_id=located[k], message_type=MessageType.TRAIN) for k in sorted(located)]
        replies = _replies_or_raise(grid.send_and_receive(messages, timeout=timeout), len(messages), f"học ở vòng {step}")
        updates, client_stats = [], []
        for reply in replies:
            stats = json.loads(str(reply.content["report"]["stats"]))
            weights = record_to_state(reply.content["arrays"])
            num_examples = int(stats.pop("num_examples"))
            algo.absorb_upload(stats["client"], record_to_state(reply.content["extra"]), weights)
            updates.append({"weights": weights, "num_examples": num_examples, "client_id": stats["client"]})
            client_stats.append({**stats, "bytes_down": bytes_down})
        order = sorted(range(len(updates)), key=lambda i: updates[i]["client_id"])     # thứ tự trả lời không cố định; gộp và nhật ký theo số thứ tự client
        return [updates[i] for i in order], [client_stats[i] for i in order]

    return run_round


def make_server_app(training_loop: Callable[[Any], None], algo: Any, plan: ClientPlan, num_clients: int, round_timeout: float) -> Any:
    from flwr.serverapp import ServerApp

    app = ServerApp()

    @app.main()
    def main(grid: Any, context: Any) -> None:
        located = locate_clients(grid, num_clients, round_timeout)
        training_loop(make_round_fn(grid, algo, plan, located, round_timeout))

    return app


def run_on_flower(training_loop: Callable[[Any], None], algo: Any, plan: ClientPlan, *, client_cpus: int, round_timeout: float, use_gpu: bool, num_clients: int = 5) -> None:
    """Mỗi client giữ trọn GPU khi có GPU nên chúng học lần lượt như đường một tiến trình; ServerApp chạy trong tiến trình này. Lỗi ở server thì run_simulation ném ra, không nuốt."""
    from flwr.simulation import run_simulation

    backend = {"client_resources": {"num_cpus": client_cpus, "num_gpus": 1.0 if use_gpu else 0.0},
               "init_args": {"num_cpus": client_cpus, "num_gpus": 1 if use_gpu else 0, "include_dashboard": False, "log_to_driver": True}}
    failure: list[BaseException] = []
    finished = {"done": False}

    def tracked(run_round: Any) -> None:
        # run_simulation chạy ServerApp trong luồng riêng và có thể chỉ ghi log khi luồng lỗi: giữ lại lỗi để ném ở luồng chính, không để run giả vờ xong
        try:
            training_loop(run_round)
            finished["done"] = True
        except BaseException as error:  # noqa: BLE001 - ghi lại để ném lại ở dưới
            failure.append(error)
            raise

    run_simulation(server_app=make_server_app(tracked, algo, plan, num_clients, round_timeout), client_app=make_client_app(), num_supernodes=num_clients, backend_name="ray", backend_config=backend)
    if failure:
        raise failure[0]
    if not finished["done"]:
        raise RuntimeError("Flower kết thúc mà vòng lặp huấn luyện chưa chạy xong; xem log Ray ở trên")
