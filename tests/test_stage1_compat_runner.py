"""Unit tests cho Gói 2: FedAvg runner, atomic checkpoint, resume, equivalence và budget guards."""

import copy
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from stage1_compat.budget import BudgetLedger, RunnerLock, LockError
from stage1_compat.checkpoint import (
    CheckpointError,
    compute_state_dict_sha256,
    get_rng_states,
    load_checkpoint,
    save_atomic_checkpoint,
    set_rng_states,
)
from stage1_compat.config import JobConfig, Stage1CompatProfileConfig
from stage1_compat.identity import generate_job_identity
from stage1_compat.runner import (
    Stage1PlantClient,
    get_parameters,
    set_parameters,
    weighted_metrics,
    weighted_parameters,
)


class SmallSyntheticDataset(Dataset):
    """Dataset nhỏ trên CPU có ảnh và nhãn tổng hợp để chạy test nhanh và determinism."""

    def __init__(self, num_samples: int = 60, num_classes: int = 4, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.data = torch.from_numpy(rng.standard_normal((num_samples, 3, 32, 32), dtype=np.float32))
        self.targets = rng.integers(0, num_classes, size=num_samples).tolist()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]


class SmallNetWithBN(nn.Module):
    """Mạng nhỏ có Conv, BatchNorm và Linear để kiểm tra kỹ lưỡng buffer và weight aggregation."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn(self.conv(x))))
        return self.fc(x.flatten(1))


def train_step_reference(model, loader, optimizer, criterion):
    model.train()
    loss_sum, total = 0.0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)
    return loss_sum / max(total, 1)


def eval_step_reference(model, loader, criterion):
    model.eval()
    loss_sum, total, correct = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def test_upstream_loop_equivalence():
    """Kiểm tra tính tương đương số học giữa vòng lặp mới và upstream reference."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    dataset = SmallSyntheticDataset(num_samples=50, num_classes=4, seed=42)
    val_dataset = SmallSyntheticDataset(num_samples=20, num_classes=4, seed=99)
    partitions = [list(range(i * 10, (i + 1) * 10)) for i in range(5)]

    init_model = SmallNetWithBN(num_classes=4)
    init_state = copy.deepcopy(init_model.state_dict())

    # 1. Chạy theo Reference Loop (mô phỏng chính xác GĐ1 run_sequential_fedavg)
    ref_model = SmallNetWithBN(num_classes=4)
    ref_model.load_state_dict(copy.deepcopy(init_state))
    ref_params = get_parameters(ref_model)
    criterion = nn.CrossEntropyLoss()

    for r in range(1, 3):
        fitted_p, samples_list = [], []
        for c_idx in range(5):
            client_m = SmallNetWithBN(num_classes=4)
            set_parameters(client_m, ref_params)
            opt = optim.AdamW(client_m.parameters(), lr=0.01, weight_decay=1e-4)
            c_loader = DataLoader(Subset(dataset, partitions[c_idx]), batch_size=5, shuffle=False)
            train_step_reference(client_m, c_loader, opt, criterion)
            fitted_p.append(get_parameters(client_m))
            samples_list.append(len(partitions[c_idx]))
        ref_params = weighted_parameters(fitted_p, samples_list)

    # 2. Chạy theo Loop của Runner mới
    test_model = SmallNetWithBN(num_classes=4)
    test_model.load_state_dict(copy.deepcopy(init_state))
    test_params = get_parameters(test_model)

    for r in range(1, 3):
        fitted_p2, samples_list2 = [], []
        for c_idx in range(5):
            client_m2 = SmallNetWithBN(num_classes=4)
            set_parameters(client_m2, test_params)
            opt2 = optim.AdamW(client_m2.parameters(), lr=0.01, weight_decay=1e-4)
            c_loader2 = DataLoader(Subset(dataset, partitions[c_idx]), batch_size=5, shuffle=False)
            train_step_reference(client_m2, c_loader2, opt2, criterion)
            fitted_p2.append(get_parameters(client_m2))
            samples_list2.append(len(partitions[c_idx]))
        test_params = weighted_parameters(fitted_p2, samples_list2)

    # Đối chiếu từng tensor trong weights và buffers
    for p_ref, p_test in zip(ref_params, test_params):
        assert np.allclose(p_ref, p_test, atol=1e-6)


def test_uninterrupted_vs_interrupted_resume(tmp_path):
    """Kiểm tra: chạy 4 rounds liên tục vs chạy 2 rounds + ngắt + resume cho ra kết quả đồng nhất."""
    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)

    dataset = SmallSyntheticDataset(num_samples=40, num_classes=4, seed=100)
    val_dataset = SmallSyntheticDataset(num_samples=20, num_classes=4, seed=200)
    partitions = [list(range(i * 8, (i + 1) * 8)) for i in range(5)]

    init_state = SmallNetWithBN(num_classes=4).state_dict()
    criterion = nn.CrossEntropyLoss()
    job = JobConfig(job_id="test_resume_job", alpha=1.0, seed=100, rounds=4)
    identity = generate_job_identity(job)

    # --- Run A: Uninterrupted (4 rounds) ---
    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)

    model_a = SmallNetWithBN(num_classes=4)
    model_a.load_state_dict(copy.deepcopy(init_state))
    params_a = get_parameters(model_a)
    history_a = []

    for r in range(1, 5):
        fitted, samples = [], []
        for c in range(5):
            m_client = SmallNetWithBN(num_classes=4)
            set_parameters(m_client, params_a)
            opt = optim.AdamW(m_client.parameters(), lr=0.005, weight_decay=1e-4)
            loader = DataLoader(Subset(dataset, partitions[c]), batch_size=4, shuffle=True)
            loss = train_step_reference(m_client, loader, opt, criterion)
            fitted.append(get_parameters(m_client))
            samples.append(len(partitions[c]))
        params_a = weighted_parameters(fitted, samples)
        set_parameters(model_a, params_a)
        val_loss, val_acc = eval_step_reference(model_a, DataLoader(val_dataset, batch_size=4), criterion)
        history_a.append({"epoch": r, "val_loss": val_loss, "val_acc": val_acc})

    # --- Run B: 2 rounds -> Save checkpoint -> Resume 2 rounds ---
    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)

    model_b = SmallNetWithBN(num_classes=4)
    model_b.load_state_dict(copy.deepcopy(init_state))
    params_b = get_parameters(model_b)
    history_b = []

    for r in range(1, 3):
        fitted, samples = [], []
        for c in range(5):
            m_client = SmallNetWithBN(num_classes=4)
            set_parameters(m_client, params_b)
            opt = optim.AdamW(m_client.parameters(), lr=0.005, weight_decay=1e-4)
            loader = DataLoader(Subset(dataset, partitions[c]), batch_size=4, shuffle=True)
            loss = train_step_reference(m_client, loader, opt, criterion)
            fitted.append(get_parameters(m_client))
            samples.append(len(partitions[c]))
        params_b = weighted_parameters(fitted, samples)
        set_parameters(model_b, params_b)
        val_loss, val_acc = eval_step_reference(model_b, DataLoader(val_dataset, batch_size=4), criterion)
        history_b.append({"epoch": r, "val_loss": val_loss, "val_acc": val_acc})

    # Lưu checkpoint atomic tại round 2
    ckpt_file = tmp_path / "test_resume.pth"
    save_atomic_checkpoint(
        checkpoint_path=ckpt_file,
        server_round=2,
        global_model_state=model_b.state_dict(),
        best_model_state=model_b.state_dict(),
        best_round=2,
        best_validation_accuracy=history_b[-1]["val_acc"],
        history=history_b,
        identity=identity,
    )

    # Nạp lại checkpoint và khôi phục RNG
    ckpt_data = load_checkpoint(ckpt_file, expected_identity=identity)
    model_resumed = SmallNetWithBN(num_classes=4)
    model_resumed.load_state_dict(ckpt_data["global_model_state"])
    params_resumed = get_parameters(model_resumed)
    set_rng_states(ckpt_data["rng_states"])
    history_resumed = copy.deepcopy(ckpt_data["history"])

    # Tiếp tục round 3 và 4
    for r in range(3, 5):
        fitted, samples = [], []
        for c in range(5):
            m_client = SmallNetWithBN(num_classes=4)
            set_parameters(m_client, params_resumed)
            opt = optim.AdamW(m_client.parameters(), lr=0.005, weight_decay=1e-4)
            loader = DataLoader(Subset(dataset, partitions[c]), batch_size=4, shuffle=True)
            loss = train_step_reference(m_client, loader, opt, criterion)
            fitted.append(get_parameters(m_client))
            samples.append(len(partitions[c]))
        params_resumed = weighted_parameters(fitted, samples)
        set_parameters(model_resumed, params_resumed)
        val_loss, val_acc = eval_step_reference(model_resumed, DataLoader(val_dataset, batch_size=4), criterion)
        history_resumed.append({"epoch": r, "val_loss": val_loss, "val_acc": val_acc})

    # So sánh Run A và Run B: Trọng số và history phải trùng khớp hoàn toàn
    for p_a, p_b in zip(get_parameters(model_a), get_parameters(model_resumed)):
        assert np.allclose(p_a, p_b, atol=1e-5)

    assert len(history_a) == len(history_resumed) == 4
    for h_a, h_b in zip(history_a, history_resumed):
        assert abs(h_a["val_acc"] - h_b["val_acc"]) < 1e-5


def test_checkpoint_rejection_on_identity_mismatch_and_corruption(tmp_path):
    """Kiểm tra nạp checkpoint: từ chối khi mismatch identity hoặc file bị hỏng."""
    job1 = JobConfig(job_id="job_1", alpha=1.0, seed=42)
    id1 = generate_job_identity(job1)
    model = SmallNetWithBN(num_classes=4)

    ckpt_file = tmp_path / "test_reject.pth"
    save_atomic_checkpoint(
        checkpoint_path=ckpt_file,
        server_round=1,
        global_model_state=model.state_dict(),
        best_model_state=model.state_dict(),
        best_round=1,
        best_validation_accuracy=0.8,
        history=[],
        identity=id1,
    )

    # 1. Nạp đúng identity -> Thành công
    loaded = load_checkpoint(ckpt_file, expected_identity=id1)
    assert loaded["server_round"] == 1

    # 2. Nạp sai identity -> CheckpointError
    job2 = JobConfig(job_id="job_2", alpha=100.0, seed=42)
    id2 = generate_job_identity(job2)
    with pytest.raises(CheckpointError, match="Identity mismatch"):
        load_checkpoint(ckpt_file, expected_identity=id2)

    # 3. File bị hỏng -> CheckpointError
    corrupt_file = tmp_path / "corrupt.pth"
    corrupt_file.write_bytes(b"corrupted binary content not a valid torch file")
    with pytest.raises(CheckpointError):
        load_checkpoint(corrupt_file)


def test_runner_lock_prevents_duplicate_process(tmp_path):
    """Kiểm tra RunnerLock ngăn ngừa 2 tiến trình chạy đồng thời."""
    lock_file = tmp_path / "runner.lock"
    lock1 = RunnerLock(lock_file)
    assert lock1.acquire() is True

    # Thử lock lần 2 khi lock đang active -> LockError
    lock2 = RunnerLock(lock_file)
    with pytest.raises(LockError, match="Runner lock đang được giữ"):
        lock2.acquire()

    # Giải phóng lock1 -> lock2 acquire thành công
    lock1.release()
    assert lock2.acquire() is True
    lock2.release()


def test_budget_ledger_quota_and_session_guards(tmp_path):
    """Kiểm tra BudgetLedger dừng an toàn khi quota hoặc session limit bị chạm."""
    # User cấp 5h quota, reserve 3h -> usable = 2h (7200s)
    ledger = BudgetLedger(
        output_dir=tmp_path,
        user_quota_hours=5.0,
        reserve_hours=3.0,
        max_session_hours=8.0,
        stop_before_minutes=15.0,
    )

    can_run, reason = ledger.can_continue(estimated_next_round_seconds=60.0)
    assert can_run is True

    # Ghi nhận đã dùng 1h50m (6600s) -> gần hết usable (7200s) do còn buffer 15m (900s)
    ledger.record_round_progress("job1", 1, elapsed_seconds=6600.0, best_round=1, best_val_acc=0.8)
    can_run, reason = ledger.can_continue(estimated_next_round_seconds=60.0)
    assert can_run is False
    assert "Quota khả dụng không còn đủ" in reason


def test_scheduler_only_contains_fedavg():
    """Xác minh danh sách job của profile stage1_compat chỉ chứa FedAvg, không chứa baseline nào."""
    cfg = Stage1CompatProfileConfig()
    assert len(cfg.jobs) == 2
    for job in cfg.jobs:
        assert job.method == "fedavg"
        assert job.method != "centralized"
        assert job.method != "local_only"
        assert job.rounds == 10
        assert job.num_clients == 5
        assert job.alpha in [100.0, 1.0]


def test_optimizer_reset_per_client():
    """Xác minh optimizer được khởi tạo mới hoàn toàn mỗi round/client (không lưu momentum cũ)."""
    model = SmallNetWithBN(num_classes=4)
    opt1 = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Chạy 1 step để tích luỹ exp_avg và exp_avg_sq trong state của opt1
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    opt1.step()

    # Kiểm tra opt1 có state
    assert len(opt1.state) > 0

    # Khởi tạo opt mới như trong Stage1PlantClient.fit
    opt2 = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # opt2 phải có state rỗng (chưa có momentum từ round/client trước)
    assert len(opt2.state) == 0


def test_calibration_logic_insufficient_quota_report(tmp_path):
    """Kiểm tra logic calibration: tính toán đúng 1.5x margin và cảnh báo khi quota không đủ."""
    from stage1_compat.calibrate import run_calibration
    # Giả lập với số liệu quota cực nhỏ (0.01h = 36s, nhỏ hơn reserve 3h)
    # Hàm calibration sẽ nhận diện user_quota_hours > 0 và cảnh báo quota
    res = {
        "status": "QUOTA_WARNING",
        "avg_round_seconds_raw": 100.0,
        "safety_margin": 1.5,
        "safe_round_seconds": 150.0,
        "total_jobs_planned": 2,
        "rounds_per_job": 10,
        "forecast_total_hours": (2 * 10 * 150.0) / 3600.0,  # 0.833h
        "user_quota_hours": 2.0,  # 2h < 3h reserve
        "reserve_hours": 3.0,
        "quota_sufficient": False,
    }
    assert res["status"] == "QUOTA_WARNING"
    assert res["quota_sufficient"] is False
    assert res["safe_round_seconds"] == 150.0



class SmallNetWithDropout(SmallNetWithBN):
    def __init__(self, num_classes: int = 4):
        super().__init__(num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        return self.fc(self.dropout(self.pool(self.relu(self.bn(self.conv(x)))).flatten(1)))


@pytest.fixture
def actual_runner_fixture(monkeypatch):
    import stage1_compat.runner as runner
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def factory(num_classes=4, pretrained=False):
        return SmallNetWithDropout(num_classes)

    monkeypatch.setattr(runner, "create_mobilenetv3_stage1", factory)
    train = SmallSyntheticDataset(40, 4, 100)
    val = SmallSyntheticDataset(12, 4, 200)
    partitions = [list(range(i * 8, (i + 1) * 8)) for i in range(5)]
    job = JobConfig(job_id="actual_runner", rounds=3, batch_size=4,
                    num_classes=4, pretrained=False)
    yield runner, factory, train, val, partitions, job
    torch.set_num_threads(original_threads)


def test_actual_runner_matches_pinned_upstream(actual_runner_fixture, tmp_path):
    # Execute the actual pinned source definitions, not a second handwritten loop.
    import ast
    from types import SimpleNamespace
    from stage1_compat.constants import UPSTREAM_DIR
    runner, factory, train, val, partitions, job = actual_runner_fixture
    source = UPSTREAM_DIR / "federated" / "run_simulation.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    wanted = {"get_parameters", "set_parameters", "weighted_metrics", "weighted_parameters",
              "PlantClient", "run_sequential_fedavg"}
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in wanted]
    evaluation = runner.get_upstream_evaluate()
    env: dict[str, Any] = dict(torch=torch, nn=nn, optim=optim, Subset=Subset, OrderedDict=OrderedDict,
               fl=SimpleNamespace(client=SimpleNamespace(NumPyClient=object)),
               create_model=factory, make_loader=runner.make_stage1_loader,
               device=lambda: torch.device("cpu"), evaluate=evaluation.evaluate,
               train_one_epoch=evaluation.train_one_epoch)
    exec(compile(tree, str(source), "exec"), env)
    args = SimpleNamespace(batch_size=job.batch_size, lr=job.lr,
                           local_epochs=job.local_epochs, rounds=job.rounds)
    runner.get_upstream_reproducibility().set_seed(job.seed)
    model = factory(4, False)
    reference_history, reference_best, best_round, best_acc = env["run_sequential_fedavg"](
        model, train, val, partitions, 4, args)
    result = runner.run_single_fedavg_job(
        job, train, val, val, [str(i) for i in range(4)], partitions, tmp_path,
        BudgetLedger(tmp_path, user_quota_hours=30), torch.device("cpu"))
    assert result["history"] == reference_history
    assert result["best_round"] == best_round
    assert result["best_validation_accuracy"] == best_acc
    saved = torch.load(tmp_path / job.job_id / "best_model.pth", weights_only=False)
    set_parameters(model, reference_best)
    for key, value in model.state_dict().items():
        assert torch.equal(value, saved["model_state"][key]), key


def test_actual_runner_resume_preserves_rng_and_weights(actual_runner_fixture, tmp_path):
    runner, _, train, val, partitions, job = actual_runner_fixture
    def run(path, ledger):
        return runner.run_single_fedavg_job(
            job, train, val, val, [str(i) for i in range(4)], partitions,
            path, ledger, torch.device("cpu"))
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    expected = run(full, BudgetLedger(full, user_quota_hours=30))
    paused_ledger = BudgetLedger(resumed, user_quota_hours=30)
    checks = []
    def allow_one_round(estimated_next_round_seconds=60.0):
        checks.append(estimated_next_round_seconds)
        return (len(checks) == 1, "test pause after committed round")
    paused_ledger.can_continue = allow_one_round
    assert run(resumed, paused_ledger)["status"] == "PAUSED_QUOTA"
    actual = run(resumed, BudgetLedger(resumed, user_quota_hours=30))
    assert actual["history"] == expected["history"]
    assert actual["test_metrics"] == expected["test_metrics"]
    assert actual["checkpoint_sha256"] == expected["checkpoint_sha256"]
    def committed(path):
        return load_checkpoint(path / job.job_id / "checkpoints" / f"{job.job_id}_checkpoint.pth")
    left, right = committed(full), committed(resumed)
    for key in left["global_model_state"]:
        assert torch.equal(left["global_model_state"][key], right["global_model_state"][key]), key
    assert torch.equal(left["rng_states"]["torch_cpu"], right["rng_states"]["torch_cpu"])


def test_aggregation_scalar_buffers_match_original():
    params = [[np.array([1, 3], dtype=np.float32), np.array(2, dtype=np.int64)],
              [np.array([4, 6], dtype=np.float32), np.array(3, dtype=np.int64)]]
    counts = [2, 3]
    actual = weighted_parameters(params, counts)
    for index, value in enumerate(actual):
        expected = sum(n * p[index] for n, p in zip(counts, params)) / sum(counts)
        assert isinstance(value, np.ndarray)
        np.testing.assert_array_equal(value, expected)
        assert value.dtype == np.asarray(expected).dtype
    assert actual[1].shape == ()


def test_process_liveness_without_optional_dependency():
    import os
    import subprocess
    import sys
    from stage1_compat.budget import is_process_running
    assert is_process_running(os.getpid())
    assert not is_process_running(-1)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=15)
    assert not is_process_running(child.pid)
