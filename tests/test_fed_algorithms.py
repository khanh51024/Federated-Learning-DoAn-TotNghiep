"""src/federated/algorithms.py: FedProx, FedBN, SCAFFOLD, FedAvgM, MOON. Mọi test chạy trên CPU, không cần dữ liệu ảnh thật."""
from __future__ import annotations

import math

import pytest
import torch

from src.federated import algorithms as alg
from src.federated.strategy import aggregate_fedavg
from src.model.mobilenetv3 import create_model, get_parameters


class Tiny(torch.nn.Module):
    """Một lớp tuyến tính + BatchNorm1d: đủ tham số học được, bộ đệm thống kê và bộ đếm num_batches_tracked."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2, bias=False)
        self.bn = torch.nn.BatchNorm1d(2)

    def forward(self, x):
        return self.bn(self.fc(x))


def tiny_state(model, value, bn_mean=0.0, tracked=0):
    state = get_parameters(model)
    state["fc.weight"] = torch.full_like(state["fc.weight"], value)
    state["bn.running_mean"] = torch.full_like(state["bn.running_mean"], bn_mean)
    state["bn.num_batches_tracked"] = torch.tensor(tracked)
    return state


def updates_of(states, sizes):
    return [{"weights": s, "num_examples": n, "client_id": k} for k, (s, n) in enumerate(zip(states, sizes))]


def test_bn_state_keys_cover_only_batchnorm_layers_of_the_real_model():
    model = create_model(num_classes=38, pretrained=False)
    keys = alg.bn_state_keys(model)
    modules = sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    assert modules > 10 and len(keys) == 5 * modules
    assert all(k.rsplit(".", 1)[1] in alg._BN_SUFFIXES for k in keys)
    assert "classifier.3.weight" not in keys


def test_plain_fedavg_algorithm_is_exactly_aggregate_fedavg():
    model = Tiny()
    parent = tiny_state(model, 0.0)
    ups = updates_of([tiny_state(model, 1.0, 2.0, 5), tiny_state(model, 3.0, 4.0, 7)], [10, 30])
    algo = alg.make_algorithm("fedavg", model, 2)
    assert algo.start_state(0, parent) is parent and algo.make_hooks(0, model, parent, 0.1, "cpu") is None
    expected = aggregate_fedavg(ups, parent_weights=parent)
    got = algo.aggregate(ups, parent)
    assert all(torch.equal(got[k], expected[k]) for k in expected)


def test_update_norms_split_batchnorm_part():
    model = Tiny()
    parent = tiny_state(model, 0.0)
    local = tiny_state(model, 1.0, bn_mean=2.0)
    total, without_bn = alg.update_norms(local, parent, alg.FedAlgorithm(model, 1).bn_keys)
    assert without_bn == pytest.approx(math.sqrt(6.0))            # fc.weight: 6 phần tử, mỗi phần tử lệch 1
    assert total == pytest.approx(math.sqrt(6.0 + 2 * 4.0))        # thêm running_mean: 2 phần tử lệch 2
    assert alg.state_bytes(parent) > 0


def test_fedprox_penalty_matches_formula_and_vanishes_at_anchor():
    model = Tiny()
    anchor = {name: p.detach().clone() for name, p in model.named_parameters()}
    hooks = alg.ProxHooks(anchor, mu=0.5)
    assert float(hooks.extra_loss(model, None, None)) == 0.0
    with torch.no_grad():
        model.fc.weight.add_(1.0)                                  # 6 phần tử lệch 1
    assert float(hooks.extra_loss(model, None, None)) == pytest.approx(0.5 * 0.5 * 6)
    assert alg.ProxHooks(anchor, mu=0.0).extra_loss(model, None, None) is None


def test_fedprox_algorithm_builds_hooks_with_mu():
    model = Tiny()
    algo = alg.make_algorithm("fedprox", model, 2, mu=0.2)
    hooks = algo.make_hooks(0, model, get_parameters(model), 0.1, torch.device("cpu"))
    assert isinstance(hooks, alg.ProxHooks) and hooks.mu == 0.2


def test_scaffold_adds_correction_to_gradients():
    model = Tiny()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    hooks = alg.ScaffoldHooks({name: torch.full_like(p, 2.0) for name, p in model.named_parameters()})
    hooks.adjust_gradients(model)
    assert all(torch.equal(p.grad, torch.full_like(p, 3.0)) for p in model.parameters())


def test_scaffold_control_variates_follow_the_paper_update():
    model = Tiny()
    algo = alg.make_algorithm("scaffold", model, 2, momentum=0.0)
    parent = tiny_state(model, 0.0)
    lr, steps = 0.1, 4
    locals_ = []
    for k, value in enumerate((-0.4, -0.8)):                          # client k đi từ x = 0 tới y_k, x − y_k = 0,4 hoặc 0,8
        local = Tiny()
        local.load_state_dict(tiny_state(model, value))
        algo.after_local(k, local, parent, lr, steps, 10 * (k + 1))
        locals_.append(get_parameters(local))
    # c_i⁺ = c_i − c + (x − y_i)/(K·η) với c = c_i = 0
    assert torch.allclose(algo._pending[0]["fc.weight"], torch.full((2, 3), 0.4 / (steps * lr)))
    assert torch.allclose(algo._pending[1]["fc.weight"], torch.full((2, 3), 0.8 / (steps * lr)))
    algo.aggregate(updates_of(locals_, [10, 20]), parent)
    # c = Σ p_k c_k với p = (1/3, 2/3); c_i được thay bằng giá trị mới; hàng chờ trống
    expected_c = (1 / 3) * (0.4 / 0.4) + (2 / 3) * (0.8 / 0.4)
    assert torch.allclose(algo.c["fc.weight"], torch.full((2, 3), expected_c))
    assert torch.allclose(algo.ci[1]["fc.weight"], torch.full((2, 3), 2.0)) and algo._pending == {}
    # vòng sau, hiệu chỉnh của client = c − c_i
    hooks = algo.make_hooks(0, model, parent, lr, torch.device("cpu"))
    assert torch.allclose(hooks.correction["fc.weight"], torch.full((2, 3), expected_c - 1.0))


def test_scaffold_with_momentum_uses_effective_learning_rate():
    model = Tiny()
    algo = alg.make_algorithm("scaffold", model, 1, momentum=0.9)
    parent = tiny_state(model, 0.0)
    local = Tiny()
    local.load_state_dict(tiny_state(model, -1.0))
    algo.after_local(0, local, parent, 0.1, 2, 5)
    assert torch.allclose(algo._pending[0]["fc.weight"], torch.full((2, 3), 1.0 / (2 * 0.1 / 0.1)))     # η_hiệu_dụng = 0,1/(1−0,9) = 1


def test_scaffold_extra_bytes_cover_one_copy_of_trainable_parameters_each_way():
    model = Tiny()
    algo = alg.make_algorithm("scaffold", model, 2)
    up, down = algo.extra_bytes(get_parameters(model))
    assert up == down == sum(p.numel() * 4 for p in model.parameters())


def test_fedbn_keeps_client_batchnorm_but_not_the_counter():
    model = Tiny()
    algo = alg.make_algorithm("fedbn", model, 2)
    assert "bn.num_batches_tracked" not in algo.personal_keys and "bn.running_mean" in algo.personal_keys
    parent = tiny_state(model, 0.0, bn_mean=0.0, tracked=3)
    assert algo.start_state(0, parent) is parent                      # vòng đầu: chưa có BN riêng
    local = Tiny()
    local.load_state_dict(tiny_state(model, 1.0, bn_mean=7.0, tracked=9))
    algo.after_local(0, local, parent, 0.1, 1, 10)
    nxt = tiny_state(model, 5.0, bn_mean=1.0, tracked=20)
    start = algo.start_state(0, nxt)
    assert torch.equal(start["bn.running_mean"], torch.full((2,), 7.0))      # BN riêng giữ nguyên
    assert torch.equal(start["fc.weight"], nxt["fc.weight"])                 # trọng số khác lấy từ mô hình toàn cục
    assert int(start["bn.num_batches_tracked"]) == 20                         # bộ đếm theo toàn cục để cổng gộp không từ chối
    assert algo.start_state(1, nxt) is nxt
    assert torch.equal(algo.personalized_state(0, nxt)["bn.running_mean"], torch.full((2,), 7.0))


def test_fedbn_global_eval_model_is_plain_fedavg():
    model = Tiny()
    algo = alg.make_algorithm("fedbn", model, 2)
    parent = tiny_state(model, 0.0, tracked=1)
    ups = updates_of([tiny_state(model, 1.0, 2.0, 2), tiny_state(model, 3.0, 4.0, 3)], [10, 30])
    expected = aggregate_fedavg(ups, parent_weights=parent)
    got = algo.aggregate(ups, parent)
    assert all(torch.equal(got[k], expected[k]) for k in expected)


def test_fedavgm_without_momentum_equals_fedavg():
    model = Tiny()
    parent = tiny_state(model, 0.0, tracked=1)
    ups = updates_of([tiny_state(model, 1.0, 2.0, 2), tiny_state(model, 3.0, 4.0, 3)], [10, 30])
    expected = aggregate_fedavg(ups, parent_weights=parent)
    got = alg.make_algorithm("fedavgm", model, 2, server_momentum=0.0, server_lr=1.0).aggregate(ups, parent)
    assert all(torch.allclose(got[k].float(), expected[k].float(), atol=1e-6) for k in expected)


def test_fedavgm_accumulates_velocity_on_trainable_parameters_only():
    model = Tiny()
    algo = alg.make_algorithm("fedavgm", model, 2, server_momentum=0.9, server_lr=0.1)
    p0 = tiny_state(model, 0.0, tracked=0)
    round1 = algo.aggregate(updates_of([tiny_state(model, 1.0, 5.0, 1)] * 2, [10, 10]), p0)         # Δ1 = 1
    assert torch.allclose(round1["fc.weight"], torch.full((2, 3), 0.1))                              # 0 + 0,1·1
    assert torch.allclose(round1["bn.running_mean"], torch.full((2,), 5.0))                          # bộ đệm: trung bình thường, không động lượng
    p1 = round1
    merged2 = tiny_state(model, 0.1 + 1.0, 5.0, 2)                                                    # Δ2 = 1
    round2 = algo.aggregate(updates_of([merged2] * 2, [10, 10]), p1)
    assert torch.allclose(round2["fc.weight"], torch.full((2, 3), 0.1 + 0.1 * (0.9 * 1.0 + 1.0)))   # v2 = β·v1 + Δ2


def test_moon_loss_is_log2_when_previous_equals_global_and_removes_its_hook():
    torch.manual_seed(0)
    local = create_model(num_classes=38, pretrained=False).eval()
    global_model = create_model(num_classes=38, pretrained=False).eval()
    global_model.load_state_dict(local.state_dict())
    previous = create_model(num_classes=38, pretrained=False).eval()
    previous.load_state_dict(local.state_dict())
    hooks = alg.MoonHooks(local, global_model, previous, mu=2.0, tau=0.5)
    images = torch.randn(2, 3, 224, 224)
    logits = local(images)
    assert float(hooks.extra_loss(local, images, logits)) == pytest.approx(2.0 * math.log(2), abs=1e-5)
    hooks.close()
    assert len(local.classifier[1]._forward_hooks) == 0


def test_moon_has_no_extra_loss_in_first_round():
    local = create_model(num_classes=38, pretrained=False).eval()
    algo = alg.make_algorithm("moon", local, 2, moon_mu=1.0, moon_tau=0.5)
    hooks = algo.make_hooks(0, local, get_parameters(local), 0.1, torch.device("cpu"))
    images = torch.randn(1, 3, 224, 224)
    assert hooks.extra_loss(local, images, local(images)) is None
    hooks.close()
    algo.after_local(0, local, get_parameters(local), 0.1, 1, 5)
    second = algo.make_hooks(0, local, get_parameters(local), 0.1, torch.device("cpu"))
    assert second.previous is not None and not any(p.requires_grad for p in second.previous.parameters())
    second.close()


@pytest.mark.parametrize("name", ["scaffold", "fedbn", "fedavgm", "moon"])
def test_algorithm_state_roundtrip(name):
    model = Tiny() if name != "moon" else create_model(num_classes=38, pretrained=False)
    algo = alg.make_algorithm(name, model, 2)
    state = get_parameters(model)
    local = Tiny() if name != "moon" else create_model(num_classes=38, pretrained=False)
    algo.after_local(0, local, state, 0.1, 2, 5)
    saved = algo.state_dict()
    fresh = alg.make_algorithm(name, model, 2)
    fresh.load_state_dict(saved)
    assert set(fresh.state_dict()) == set(saved)


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError):
        alg.make_algorithm("fedmagic", Tiny(), 2)


def _toy_loader(n=8, batch=4):
    class Data(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return torch.randn(3, 224, 224, generator=g), i % 38, i
    return torch.utils.data.DataLoader(Data(), batch_size=batch, shuffle=False)


def _distance(a, b):
    return math.sqrt(sum(float(((a[k].float() - b[k].float()) ** 2).sum()) for k in a if a[k].is_floating_point()))


def test_train_epoch_with_proximal_hook_stays_closer_to_the_anchor():
    from src.training.loop import make_optimizer, train_epoch

    results = {}
    for mu in (0.0, 5.0):
        torch.manual_seed(1)
        model = create_model(num_classes=38, pretrained=False)
        anchor = get_parameters(model)
        hooks = alg.ProxHooks({n: p.detach().clone() for n, p in model.named_parameters()}, mu)
        stats = train_epoch(model, _toy_loader(), make_optimizer(model, 0.05, 0.9, 0.0), torch.amp.GradScaler("cpu", enabled=False),
                            torch.device("cpu"), torch.ones(8), 0.0, hooks=hooks)
        assert stats["steps"] == 2 and math.isfinite(stats["loss"])
        results[mu] = _distance(get_parameters(model), anchor)
    assert results[5.0] < results[0.0]


def test_train_epoch_runs_with_scaffold_and_moon_hooks():
    from src.training.loop import make_optimizer, train_epoch

    torch.manual_seed(2)
    model = create_model(num_classes=38, pretrained=False)
    scaffold = alg.make_algorithm("scaffold", model, 2, momentum=0.9)
    hooks = scaffold.make_hooks(0, model, get_parameters(model), 0.05, torch.device("cpu"))
    train_epoch(model, _toy_loader(), make_optimizer(model, 0.05, 0.9, 0.0), torch.amp.GradScaler("cpu", enabled=False),
                torch.device("cpu"), torch.ones(8), 0.0, hooks=hooks)
    moon = alg.make_algorithm("moon", model, 2, moon_mu=1.0, moon_tau=0.5)
    moon.after_local(0, model, get_parameters(model), 0.05, 2, 8)                   # có mô hình vòng trước -> có số hạng đối chiếu
    local = create_model(num_classes=38, pretrained=False)
    local.load_state_dict(model.state_dict())
    moon_hooks = moon.make_hooks(0, local, get_parameters(local), 0.05, torch.device("cpu"))
    stats = train_epoch(local, _toy_loader(), make_optimizer(local, 0.05, 0.9, 0.0), torch.amp.GradScaler("cpu", enabled=False),
                        torch.device("cpu"), torch.ones(8), 0.0, hooks=moon_hooks)
    moon_hooks.close()
    assert math.isfinite(stats["loss"])
