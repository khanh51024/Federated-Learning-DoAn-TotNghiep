"""Phần thuần của bộ huấn luyện v4 (src/training): trọng số cân bằng miền, chỉ số theo nguồn, lịch tốc độ học, kiểm SHA release."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.training.data import domain_balanced_weights, load_release
from src.training.loop import cosine_lr
from src.training.metrics import macro_f1, source_key, summarize


def test_domain_weights_give_lab_and_field_images_equal_total_weight_with_mean_one():
    rows = [{"source_domain": "plantvillage"}] * 9 + [{"source_domain": "plantdoc"}] * 3
    w = np.array(domain_balanced_weights(rows))
    assert w[:9].sum() == pytest.approx(w[9:].sum()) and w.mean() == pytest.approx(1.0)
    assert domain_balanced_weights([{"source_domain": "plantdoc"}] * 4) == [1.0] * 4


def test_macro_f1_uses_only_classes_present_in_the_targets():
    assert macro_f1(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0
    assert macro_f1(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])) == pytest.approx((2 / 3 + 0.8) / 2)
    assert macro_f1(np.array([0, 0]), np.array([5, 5])) == 0.0


def test_summary_weights_every_source_equally_in_the_selection_score():
    rows = ([{"source_domain": "plantvillage", "source_dataset": "PlantVillage", "class_id": 0}] * 40
            + [{"source_domain": "plantdoc", "source_dataset": "PlantDoc", "class_id": 1}] * 30
            + [{"source_domain": "plantdoc", "source_dataset": "mendeley_x", "class_id": 2}] * 5)
    logits = np.zeros((len(rows), 38), np.float32)
    logits[:40, 0] = 1          # PlantVillage đúng hết
    logits[40:70, 3] = 1        # PlantDoc sai hết
    logits[70:, 2] = 1          # nguồn nhỏ đúng hết nhưng < 30 ảnh: không vào điểm chọn
    s = summarize(logits, rows)
    assert s["selection_sources"] == ["PlantDoc", "PlantVillage"] and s["selection_score"] == pytest.approx(0.5)
    assert s["plantvillage_top1"] == 1.0 and s["field_top1"] == pytest.approx(5 / 35, abs=1e-4)
    assert s["by_source"]["mendeley_x"] == {"n": 5, "top1": 1.0, "macro_f1": 1.0}
    assert source_key({"source_domain": "plantdoc", "source_dataset": "plantdoc"}) == "PlantDoc"


def test_cosine_lr_warms_up_then_decays_to_zero():
    assert cosine_lr(0.1, 0, 10, warmup=2) == pytest.approx(0.05)
    assert cosine_lr(0.1, 1, 10, warmup=2) == pytest.approx(0.1)
    assert cosine_lr(0.1, 2, 10, warmup=2) == pytest.approx(0.1)
    assert cosine_lr(0.1, 10, 10, warmup=2) == pytest.approx(0.0)
    assert cosine_lr(0.1, 6, 10, warmup=2) == pytest.approx(0.05)


def test_load_release_refuses_a_different_release_or_an_edited_manifest(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    hashes = {}
    for name in ("train", "val", "calibration", "legacy_diagnostic_test"):
        data = json.dumps({"sample_id": name}) + "\n"
        (manifests / f"{name}.jsonl").write_bytes(data.encode())   # ghi byte: tránh đổi xuống dòng trên Windows
        hashes[name] = hashlib.sha256(data.encode()).hexdigest()
    (tmp_path / "release_manifest.json").write_text(json.dumps({"manifest_hashes": hashes}), encoding="utf-8")
    sha = hashlib.sha256((tmp_path / "release_manifest.json").read_bytes()).hexdigest()
    release = load_release(tmp_path, sha)
    assert release["sha256"] == sha and release["train"] == [{"sample_id": "train"}]
    with pytest.raises(SystemExit):
        load_release(tmp_path, "0" * 64)
    (manifests / "val.jsonl").write_text(json.dumps({"sample_id": "changed"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_release(tmp_path, sha)


def _rows(spec):
    """spec: [(miền, lớp, số ảnh), ...] -> các dòng manifest tối thiểu."""
    return [{"source_domain": d, "class_id": c} for d, c, n in spec for _ in range(n)]


def test_balanced_sampling_with_power_one_matches_the_domain_weights():
    from src.training.data import balanced_sampling_weights

    rows = _rows([("plantvillage", 0, 60), ("plantvillage", 1, 30), ("plantdoc", 0, 8), ("plantdoc", 2, 2)])
    p = np.array(balanced_sampling_weights(rows, 1.0))
    w = np.array(domain_balanced_weights(rows))
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(p, w / w.sum())                               # cùng phân bố với trọng số mất mát của run mốc
    assert p[:90].sum() == pytest.approx(0.5)


def test_square_root_class_power_lifts_rare_classes_inside_each_domain():
    from src.training.data import balanced_sampling_weights

    rows = _rows([("plantdoc", 0, 90), ("plantdoc", 1, 10)])
    p1 = np.array(balanced_sampling_weights(rows, 1.0))
    ph = np.array(balanced_sampling_weights(rows, 0.5))
    assert p1[90:].sum() == pytest.approx(0.1)
    assert ph[90:].sum() == pytest.approx(np.sqrt(10) / (np.sqrt(90) + np.sqrt(10)))   # 0,25 > 0,1
    assert np.array(balanced_sampling_weights(rows, 0.0))[90:].sum() == pytest.approx(0.5)


def test_class_prior_follows_the_sampling_weights():
    from src.training.data import class_prior

    rows = _rows([("plantdoc", 0, 3), ("plantdoc", 2, 1)])
    prior = class_prior(rows, [1, 1, 1, 3], num_classes=3)
    assert prior == pytest.approx([0.5, 0.0, 0.5])


def test_logit_adjustment_is_off_at_tau_zero_and_finite_for_absent_classes():
    from src.training.loop import logit_adjustment

    assert logit_adjustment([0.5, 0.5, 0.0], 0.0) is None
    adj = logit_adjustment([0.5, 0.5, 0.0], 1.0)
    assert np.isfinite(adj.numpy()).all() and float(adj[2]) == pytest.approx(np.log(1e-4))


def test_weighted_loader_draws_exactly_len_dataset_samples_per_epoch():
    from src.training.loop import make_loader

    data = list(range(50))
    loader = make_loader(data, batch_size=10, shuffle=True, workers=0, seed=3, sampling_weights=[0.0] * 25 + [1.0] * 25)
    drawn = [int(x) for b in loader for x in b]
    assert len(drawn) == 50 and min(drawn) >= 25                    # số bước mỗi epoch không đổi, chỉ đổi ảnh được chọn


def test_train_epoch_counts_on_device_and_still_refuses_a_non_finite_loss():
    import torch

    from src.training.loop import train_epoch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 4 * 4, 38)

        def forward(self, x):
            return self.fc(x.flatten(1))

    batches = [(torch.randn(5, 3, 4, 4), torch.tensor([0, 1, 2, 3, 0]), torch.arange(5)) for _ in range(3)]
    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    out = train_epoch(model, batches, opt, scaler, torch.device("cpu"), torch.ones(5), 0.1)
    assert out["steps"] == 3 and out["images"] == 15 and np.isfinite(out["loss"]) and 0 <= out["train_top1"] <= 1
    with torch.no_grad():
        model.fc.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        train_epoch(model, batches, opt, scaler, torch.device("cpu"), torch.ones(5), 0.1)
