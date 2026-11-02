"""Đường Flower của bộ train (src/federated/flower_app.py): tách client và máy chủ phải cho đúng kết quả như một tiến trình; lỗi client không được bị nuốt.

Hầu hết test không cần Ray: gọi đúng các hàm hai phía và cho trọng số đi qua ArrayRecord thật. Test có Ray chạy thật một run nhỏ, chỉ bật khi đặt RUN_FLOWER_RAY_TEST=1 vì tốn RAM."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flwr")

from src.federated import flower_app as fa  # noqa: E402
from src.federated.algorithms import make_algorithm  # noqa: E402
from src.model.mobilenetv3 import create_model, get_parameters, set_parameters  # noqa: E402
from src.training.client_round import Recipe, train_one_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALGOS = ("fedavg", "fedprox", "fedbn", "scaffold", "fedavgm", "moon")
RECIPE = Recipe(dataset_root="-", seed=1, batch_size=2, momentum=0.9, weight_decay=1e-4, label_smoothing=0.1, workers=0, amp="fp16")
DEVICE = torch.device("cpu")


def _loader(offset: int) -> torch.utils.data.DataLoader:
    class Data(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            generator = torch.Generator().manual_seed(offset * 100 + i)
            return torch.randn(3, 64, 64, generator=generator), (offset + i) % 38, i
    return torch.utils.data.DataLoader(Data(), batch_size=2, shuffle=False)


def _data(k: int) -> dict:
    return {"loader": _loader(k), "weights": torch.ones(4), "adjust": None, "augment": None, "generator": None, "num_examples": 4 + 2 * k}


def _plan(tmp_path: Path, name: str) -> fa.ClientPlan:
    return fa.ClientPlan(recipe={}, release_dir="-", release_sha="-", scheme="-", limit=0, algorithm=name, algorithm_options={}, state_dir=str(tmp_path / "state"), device="cpu")


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return create_model(num_classes=38, pretrained=False)


OPTIONS = {"mu": 0.5, "moon_mu": 1.0, "moon_tau": 0.5, "server_momentum": 0.5, "server_lr": 1.0, "momentum": 0.9}


@pytest.mark.parametrize("name", ALGOS)
def test_two_sides_match_a_single_process(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Cùng hạt giống ở mỗi client mỗi vòng: tách hai phía phải ra cùng mô hình, cùng trạng thái, tức là đổi đường chạy không đổi công thức."""
    rounds, clients, lr = 3, 2, 0.05

    single = _model()
    algo = make_algorithm(name, single, clients, **OPTIONS)
    for step in range(rounds):
        parent = get_parameters(single)
        updates = []
        for k in range(clients):
            torch.manual_seed(1000 * step + k)
            state, _ = train_one_client(copy.deepcopy(single), k, parent, algo, _data(k), lr, RECIPE, DEVICE)
            updates.append({"weights": state, "num_examples": 4 + 2 * k, "client_id": k})
        set_parameters(single, algo.aggregate(updates, parent))

    server_model = _model()
    server = make_algorithm(name, server_model, clients, **OPTIONS)
    cache = {k: {"recipe": RECIPE, "device": DEVICE, "data": _data(k), "template": copy.deepcopy(server_model), "algo": make_algorithm(name, _model(), clients, **OPTIONS)} for k in range(clients)}
    monkeypatch.setattr(fa, "_CLIENT_CACHE", cache)
    plan = _plan(tmp_path, name)
    for step in range(rounds):
        parent = get_parameters(server_model)
        arrays, extra = fa.state_to_record(parent), fa.state_to_record(server.download())
        updates = []
        for k in range(clients):
            torch.manual_seed(1000 * step + k)
            state, extra_up, _ = fa.client_step(plan, k, step, lr, fa.record_to_state(arrays), fa.record_to_state(extra))
            weights = fa.record_to_state(fa.state_to_record(state))
            server.absorb_upload(k, fa.record_to_state(fa.state_to_record(extra_up)), weights)
            updates.append({"weights": weights, "num_examples": 4 + 2 * k, "client_id": k})
        set_parameters(server_model, server.aggregate(updates, parent))

    expected, got = get_parameters(single), get_parameters(server_model)
    assert expected.keys() == got.keys()
    for key in expected:
        if name == "scaffold" and expected[key].is_floating_point():
            # Δc đi qua tin nhắn rồi cộng lại: sai khác làm tròn float32 của (c_i + Δ) − c_i, không phải khác công thức
            assert torch.allclose(expected[key], got[key], atol=1e-4, rtol=1e-4), key
        else:
            assert torch.equal(expected[key], got[key]), key

    if name == "scaffold":
        for key in algo.c:
            assert torch.allclose(algo.c[key], server.c[key], atol=1e-4, rtol=1e-4)
    if name == "fedbn":
        for k in range(clients):
            for key, value in algo.client_bn[k].items():
                assert torch.equal(value, server.client_bn[k][key])          # máy chủ có đủ BN để chấm mô hình cá nhân hóa ở bước tốt nhất
    if name == "moon":
        assert not server.previous                                           # mô hình vòng trước chỉ nằm ở client


@pytest.mark.parametrize("name,has_state", [("fedavg", False), ("fedprox", False), ("fedavgm", False), ("scaffold", True), ("fedbn", True), ("moon", True)])
def test_private_state_stays_on_client_disk_and_is_rotated(name: str, has_state: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    template = _model()
    monkeypatch.setattr(fa, "_CLIENT_CACHE", {0: {"recipe": RECIPE, "device": DEVICE, "data": _data(0), "template": template, "algo": make_algorithm(name, _model(), 1, **OPTIONS)}})
    plan = _plan(tmp_path, name)
    state_dir = Path(plan.state_dir)
    parent = get_parameters(template)
    server_extra = make_algorithm(name, template, 1, **OPTIONS).download()
    for step in range(3):
        fa.client_step(plan, 0, step, 0.01, parent, server_extra)
    files = sorted(p.name for p in state_dir.glob("*.pt")) if state_dir.exists() else []
    # giữ hai đời gần nhất: máy chủ có thể bị ngắt trước khi chốt vòng mới nhất
    assert files == (["client_00.step0001.pt", "client_00.step0002.pt"] if has_state else [])


def test_client_resumes_from_the_state_on_disk_not_from_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Tiến trình client khởi động lại giữa chừng (mất bộ nhớ) vẫn phải tiếp tục đúng c_i đã lưu."""
    template = _model()
    plan = _plan(tmp_path, "scaffold")
    first = make_algorithm("scaffold", _model(), 1, **OPTIONS)
    monkeypatch.setattr(fa, "_CLIENT_CACHE", {0: {"recipe": RECIPE, "device": DEVICE, "data": _data(0), "template": template, "algo": first}})
    parent = get_parameters(template)
    fa.client_step(plan, 0, 0, 0.05, parent, make_algorithm("scaffold", template, 1, **OPTIONS).download())
    saved = {k: v.clone() for k, v in first.ci[0].items()}

    restarted = make_algorithm("scaffold", _model(), 1, **OPTIONS)               # c_i của đối tượng mới bằng 0
    assert all(float(v.abs().sum()) == 0 for v in restarted.ci[0].values())
    seen = {}
    original = restarted.make_hooks

    def spy(client, local, parent_state, lr, device):
        seen.update({k: v.clone() for k, v in restarted.ci[client].items()})
        return original(client, local, parent_state, lr, device)

    restarted.make_hooks = spy
    monkeypatch.setattr(fa, "_CLIENT_CACHE", {0: {"recipe": RECIPE, "device": DEVICE, "data": _data(0), "template": template, "algo": restarted}})
    fa.client_step(plan, 0, 1, 0.05, parent, make_algorithm("scaffold", template, 1, **OPTIONS).download())
    for key, value in saved.items():
        assert torch.equal(seen[key], value)


def test_record_keeps_integer_buffers_and_handles_empty():
    model = _model()
    state = get_parameters(model)
    back = fa.record_to_state(fa.state_to_record(state))
    assert back.keys() == state.keys()
    for key, value in state.items():
        assert back[key].dtype == value.dtype and torch.equal(back[key], value)
    assert any(not v.is_floating_point() for v in back.values())                  # num_batches_tracked của BatchNorm
    assert fa.record_to_state(fa.state_to_record({})) == {}
    assert fa.record_bytes(fa.state_to_record({})) == 0
    # kích thước thật lớn hơn kích thước tensor một chút: có đầu mục của từng mảng
    raw = sum(t.numel() * t.element_size() for t in state.values())
    assert raw < fa.record_bytes(fa.state_to_record(state)) < raw + 300 * len(state)


# --------------------------------------------------------------------------- vòng phía máy chủ, qua lưới giả
class FakeGrid:
    """Trả lời như các ClientApp: đúng dạng nội dung, cho phép thứ tự đảo, thiếu client hoặc báo lỗi."""

    def __init__(self, respond):
        self.respond = respond
        self.sent = []

    def get_node_ids(self):
        return [101, 102]

    def send_and_receive(self, messages, *, timeout=None):
        self.sent.append(list(messages))
        return self.respond(list(messages))


def _reply(message, client: int, value: float, n: int):
    from flwr.app import ConfigRecord, RecordDict

    weights = {"w": torch.full((2,), value), "num_batches_tracked": torch.tensor(3)}
    extra = {"d": torch.ones(2) * client}
    arrays, extra_record = fa.state_to_record(weights), fa.state_to_record(extra)
    stats = {"client": client, "loss": 1.0, "steps": 2, "delta_l2": 0.1, "delta_l2_without_bn": 0.1, "num_examples": n, "bytes_up": fa.record_bytes(arrays) + fa.record_bytes(extra_record)}
    content = RecordDict({"arrays": arrays, "extra": extra_record, "report": ConfigRecord({"stats": json.dumps(stats)})})
    return SimpleNamespace(content=content, has_error=lambda: False, metadata=SimpleNamespace(src_node_id=message.metadata.dst_node_id))


class RecordingAlgo:
    def __init__(self):
        self.absorbed = []

    def download(self):
        return {"c": torch.zeros(3)}

    def absorb_upload(self, client, payload, weights):
        self.absorbed.append((client, sorted(payload), sorted(weights)))


def _round(respond, tmp_path: Path):
    grid = FakeGrid(respond)
    algo = RecordingAlgo()
    run = fa.make_round_fn(grid, algo, _plan(tmp_path, "scaffold"), {0: 101, 1: 102}, timeout=5.0)
    return grid, algo, run


def test_round_sends_real_payload_and_orders_replies_by_client(tmp_path: Path):
    grid, algo, run = _round(lambda messages: [_reply(messages[1], 1, 3.0, 30), _reply(messages[0], 0, 1.0, 10)], tmp_path)        # trả lời đảo thứ tự
    parent = {"w": torch.zeros(2), "num_batches_tracked": torch.tensor(0)}
    updates, stats = run(4, 0.02, parent)

    assert [u["client_id"] for u in updates] == [0, 1] and [s["client"] for s in stats] == [0, 1]
    assert [u["num_examples"] for u in updates] == [10, 30]
    assert torch.equal(updates[1]["weights"]["w"], torch.full((2,), 3.0))
    assert "num_examples" not in stats[0]
    assert algo.absorbed[0][1] == ["d"] and algo.absorbed[0][2] == ["num_batches_tracked", "w"]

    sent = grid.sent[0]
    assert sorted(m.metadata.dst_node_id for m in sent) == [101, 102]
    assert all(m.metadata.message_type == "train" for m in sent)
    config = sent[0].content["config"]
    assert int(config["step"]) == 4 and float(config["lr"]) == pytest.approx(0.02)
    assert json.loads(str(config["plan"]))["algorithm"] == "scaffold"
    assert stats[0]["bytes_down"] == fa.record_bytes(sent[0].content["arrays"]) + fa.record_bytes(sent[0].content["extra"])
    assert stats[0]["bytes_up"] > 0


def test_round_stops_when_a_client_errors_or_is_missing(tmp_path: Path):
    parent = {"w": torch.zeros(2)}
    failing = SimpleNamespace(has_error=lambda: True, error=SimpleNamespace(reason="hết bộ nhớ GPU"))
    _, _, run = _round(lambda messages: [_reply(messages[0], 0, 1.0, 10), failing], tmp_path)
    with pytest.raises(RuntimeError, match="hết bộ nhớ GPU"):
        run(0, 0.01, parent)
    _, _, run = _round(lambda messages: [_reply(messages[0], 0, 1.0, 10)], tmp_path)
    with pytest.raises(RuntimeError, match="1/2 client"):
        run(0, 0.01, parent)


def test_locate_clients_requires_every_shard_exactly_once():
    from flwr.app import ConfigRecord, RecordDict

    def answer(mapping):
        def respond(messages):
            return [SimpleNamespace(content=RecordDict({"identity": ConfigRecord({"partition_id": mapping[m.metadata.dst_node_id]})}), has_error=lambda: False,
                                    metadata=SimpleNamespace(src_node_id=m.metadata.dst_node_id)) for m in messages]
        return respond

    assert fa.locate_clients(FakeGrid(answer({101: 1, 102: 0})), 2, 5.0) == {1: 101, 0: 102}
    with pytest.raises(RuntimeError, match="giữ shard"):
        fa.locate_clients(FakeGrid(answer({101: 0, 102: 0})), 2, 5.0)
    with pytest.raises(RuntimeError, match="có 2 nút"):
        fa.locate_clients(FakeGrid(answer({})), 5, 0.01)

    # các nút đăng ký bất đồng bộ: lúc đầu chưa có nút nào, máy chủ phải chờ chứ không báo lỗi
    late = FakeGrid(answer({101: 0, 102: 1}))
    calls = {"n": 0}

    def appearing():
        calls["n"] += 1
        return [] if calls["n"] == 1 else [101, 102]

    late.get_node_ids = appearing
    assert fa.locate_clients(late, 2, 5.0) == {0: 101, 1: 102}
    assert calls["n"] == 2


def test_run_on_flower_raises_when_the_loop_failed_or_never_finished(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """run_simulation chạy ServerApp trong luồng riêng; lỗi ở đó phải về luồng chính, không để train_v4 chạy tiếp như đã xong."""
    import flwr.simulation

    def loop_raises(_):
        raise ValueError("vòng lặp hỏng")

    calls = {}

    def fake_run(server_app, client_app, num_supernodes, backend_name, backend_config):
        calls.update(num=num_supernodes, backend=backend_name, config=backend_config)
        with pytest.raises(ValueError):
            server_app._main(SimpleNamespace(get_node_ids=lambda: [], send_and_receive=lambda *a, **k: []), SimpleNamespace())

    monkeypatch.setattr(flwr.simulation, "run_simulation", fake_run)
    monkeypatch.setattr(fa, "locate_clients", lambda grid, n, timeout: {})
    monkeypatch.setattr(fa, "make_round_fn", lambda *a, **k: None)
    with pytest.raises(ValueError, match="vòng lặp hỏng"):
        fa.run_on_flower(loop_raises, RecordingAlgo(), _plan(tmp_path, "fedavg"), client_cpus=2, round_timeout=5.0, use_gpu=False)
    assert calls["num"] == 5 and calls["backend"] == "ray"
    assert calls["config"]["client_resources"] == {"num_cpus": 2, "num_gpus": 0.0}

    monkeypatch.setattr(flwr.simulation, "run_simulation", lambda **kwargs: None)       # không gọi ServerApp: không được coi là xong
    with pytest.raises(RuntimeError, match="chưa chạy xong"):
        fa.run_on_flower(lambda _: None, RecordingAlgo(), _plan(tmp_path, "fedavg"), client_cpus=2, round_timeout=5.0, use_gpu=True)


def test_trainer_rejects_flower_and_adamw_outside_their_scope():
    def run(*args):
        return subprocess.run([sys.executable, "-X", "utf8", "scripts/train_v4.py", "--seed", "1", "--limit", "10", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")

    bad = run("--method", "centralized", "--runtime", "flower")
    assert bad.returncode == 2 and "--runtime flower" in bad.stderr
    bad = run("--method", "fedavg", "--optimizer", "adamw")
    assert bad.returncode == 2 and "--optimizer adamw" in bad.stderr


@pytest.mark.skipif(os.environ.get("RUN_FLOWER_RAY_TEST") != "1", reason="chạy Ray thật tốn RAM; đặt RUN_FLOWER_RAY_TEST=1 để bật")
def test_real_ray_run_end_to_end(tmp_path: Path):
    pytest.importorskip("ray")
    out = tmp_path / "runs"
    command = [sys.executable, "-X", "utf8", "scripts/train_v4.py", "--method", "fedavg", "--runtime", "flower", "--algo", "scaffold", "--seed", "7", "--limit", "60", "--budget", "2",
               "--workers", "0", "--device", "cpu", "--batch-size", "8", "--out", str(out), "--tag", "ray"]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=900)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    run = out / "fedavg_s7_ray"
    assert (run / "DONE").is_file()
    assert json.loads((run / "config.json").read_text(encoding="utf-8"))["runtime"] == "flower"
    rows = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in rows] == [0, 1]
    assert [c["client"] for c in rows[0]["train"]["clients"]] == [0, 1, 2, 3, 4]
    assert all(c["bytes_up"] > 6_000_000 and c["bytes_down"] > 6_000_000 for c in rows[0]["train"]["clients"])
    # bộ nhớ đệm của client sống qua các vòng: vòng hai không dựng lại dữ liệu
    assert max(c["client_env_builds"] for c in rows[1]["train"]["clients"]) <= max(c["client_env_builds"] for c in rows[0]["train"]["clients"])
    assert sorted(p.name for p in (run / "client_state").glob("client_00.*")) == ["client_00.step0000.pt", "client_00.step0001.pt"]


def test_missing_client_state_is_an_error_not_a_silent_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Thiếu tệp trạng thái mà đi tiếp thì c_i về 0 và thuật toán sai mà không ai biết."""
    template = _model()
    monkeypatch.setattr(fa, "_CLIENT_CACHE", {0: {"recipe": RECIPE, "device": DEVICE, "data": _data(0), "template": template, "algo": make_algorithm("scaffold", _model(), 1, **OPTIONS)}})
    with pytest.raises(RuntimeError, match="thiếu trạng thái riêng"):
        fa.client_step(_plan(tmp_path, "scaffold"), 0, 3, 0.01, get_parameters(template), make_algorithm("scaffold", template, 1, **OPTIONS).download())
    # thuật toán không có trạng thái riêng thì không cần tệp
    monkeypatch.setattr(fa, "_CLIENT_CACHE", {0: {"recipe": RECIPE, "device": DEVICE, "data": _data(0), "template": template, "algo": make_algorithm("fedavg", _model(), 1, **OPTIONS)}})
    fa.client_step(_plan(tmp_path, "fedavg"), 0, 3, 0.01, get_parameters(template), {})
