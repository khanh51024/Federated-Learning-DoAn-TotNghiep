"""Thuật toán học liên kết chịu non-IID (GĐ3) cho mô phỏng một tiến trình của scripts/train_v4.py.

Mỗi thuật toán là một lớp con của `FedAlgorithm` với bốn điểm móc quanh vòng FedAvg:

  start_state   trạng thái client bắt đầu vòng (FedBN giữ BatchNorm riêng của client)
  make_hooks    mất mát phụ / chỉnh gradient khi client học (FedProx, SCAFFOLD, MOON); None = học như FedAvg
  after_local   cập nhật trạng thái riêng của client sau khi học xong (biến điều khiển SCAFFOLD, mô hình vòng trước của MOON, BN của FedBN)
  aggregate     máy chủ gộp (FedAvg có trọng số n_k; FedAvgM thêm động lượng phía máy chủ; SCAFFOLD cập nhật biến điều khiển chung)

`fedavg` đi đúng đường cũ: không móc nào, gộp bằng `aggregate_fedavg`, nên run FedAvg của đợt v4b tái lập được.

Quy ước chung: trạng thái là `dict[str, Tensor]` trên CPU (như `get_parameters`); chỉ tham số học được (`named_parameters`) chịu hiệu chỉnh
của thuật toán, bộ đệm BatchNorm (running_mean/var) luôn gộp theo FedAvg trừ khi thuật toán nói khác (FedBN).

Tài liệu gốc: FedProx (Li và cộng sự, MLSys 2020), FedBN (Li và cộng sự, ICLR 2021), SCAFFOLD (Karimireddy và cộng sự, ICML 2020),
FedAvgM (Hsu và cộng sự, 2019), MOON (Li, He, Song, CVPR 2021).
"""
from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F

from src.federated.strategy import aggregate_fedavg

ALGORITHMS = ("fedavg", "fedprox", "fedbn", "scaffold", "fedavgm", "moon")
_BN_SUFFIXES = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")


# ---------------------------------------------------------------------------
# Tiện ích dùng chung
# ---------------------------------------------------------------------------
def bn_state_keys(model: torch.nn.Module) -> list[str]:
    """Khóa state_dict thuộc các lớp BatchNorm (tham số, bộ đệm thống kê và bộ đếm)."""
    state = model.state_dict()
    keys: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            prefix = f"{name}." if name else ""
            keys += [prefix + s for s in _BN_SUFFIXES if prefix + s in state]
    return keys


def state_bytes(state: dict[str, torch.Tensor]) -> int:
    """Số byte nếu gửi nguyên trạng thái (dạng thô, chưa nén)."""
    return sum(t.numel() * t.element_size() for t in state.values())


def update_norms(local: dict[str, torch.Tensor], parent: dict[str, torch.Tensor], bn_keys: set[str]) -> tuple[float, float]:
    """(‖w_k − w‖₂ trên mọi tensor số thực, cũng đại lượng đó nhưng bỏ phần BatchNorm)."""
    total = without_bn = 0.0
    for key, base in parent.items():
        if not base.is_floating_point():
            continue
        diff = local[key].double() - base.double()
        square = float((diff * diff).sum())
        total += square
        if key not in bn_keys:
            without_bn += square
    return math.sqrt(total), math.sqrt(without_bn)


def _trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


# ---------------------------------------------------------------------------
# Móc vào vòng học của client (src/training/loop.py::train_epoch)
# ---------------------------------------------------------------------------
class LocalHooks:
    """Mặc định không làm gì. `extra_loss` được cộng vào mất mát; `adjust_gradients` chạy sau backward (đã bỏ thang GradScaler), trước bước tối ưu."""

    def extra_loss(self, model: torch.nn.Module, images: torch.Tensor, logits: torch.Tensor) -> torch.Tensor | None:
        return None

    def adjust_gradients(self, model: torch.nn.Module) -> None:
        return None

    def close(self) -> None:
        return None


class ProxHooks(LocalHooks):
    """FedProx: thêm (μ/2)·‖w − w_toàn_cục‖² vào mất mát của client."""

    def __init__(self, anchor: dict[str, torch.Tensor], mu: float):
        self.anchor, self.mu = anchor, mu

    def extra_loss(self, model, images, logits):
        if not self.mu:
            return None
        total = sum(((p.float() - self.anchor[name]) ** 2).sum() for name, p in model.named_parameters())
        return 0.5 * self.mu * total


class ScaffoldHooks(LocalHooks):
    """SCAFFOLD: cộng (c − c_i) vào gradient mỗi bước, kéo hướng học của client về hướng trung bình toàn hệ thống."""

    def __init__(self, correction: dict[str, torch.Tensor]):
        self.correction = correction

    def adjust_gradients(self, model):
        for name, p in model.named_parameters():
            if p.grad is not None:
                p.grad.add_(self.correction[name])


def representation(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Đặc trưng 1024 chiều của MobileNetV3-Small trước lớp dropout và lớp phân loại (đầu ra Hardswish của classifier[1])."""
    x = model.avgpool(model.features(images))
    return model.classifier[1](model.classifier[0](torch.flatten(x, 1)))


class MoonHooks(LocalHooks):
    """MOON: mất mát đối chiếu kéo đặc trưng của mô hình đang học về đặc trưng mô hình toàn cục và đẩy khỏi đặc trưng mô hình cục bộ vòng trước.
    Vòng đầu (chưa có mô hình vòng trước) không có số hạng phụ."""

    def __init__(self, local: torch.nn.Module, global_model: torch.nn.Module, previous: torch.nn.Module | None, mu: float, tau: float):
        self.global_model, self.previous, self.mu, self.tau = global_model, previous, mu, tau
        self._z: torch.Tensor | None = None
        self._handle = local.classifier[1].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._z = output

    def extra_loss(self, model, images, logits):
        if self.previous is None or not self.mu or self._z is None:
            return None
        z = self._z.float()
        with torch.no_grad():
            z_global = representation(self.global_model, images).float()
            z_previous = representation(self.previous, images).float()
        positive = F.cosine_similarity(z, z_global, dim=1)
        negative = F.cosine_similarity(z, z_previous, dim=1)
        contrast = torch.stack([positive, negative], dim=1) / self.tau
        target = torch.zeros(len(z), dtype=torch.long, device=z.device)
        return self.mu * F.cross_entropy(contrast, target)

    def close(self):
        self._handle.remove()


# ---------------------------------------------------------------------------
# Thuật toán
# ---------------------------------------------------------------------------
class FedAlgorithm:
    name = "fedavg"

    def __init__(self, model: torch.nn.Module, num_clients: int, **options: Any):
        self.num_clients = num_clients
        self.options = options
        self.bn_keys = set(bn_state_keys(model))

    # --- điểm móc ---------------------------------------------------------------------------------------------
    def start_state(self, client: int, parent: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return parent

    def make_hooks(self, client: int, local: torch.nn.Module, parent: dict[str, torch.Tensor], lr: float, device: Any) -> LocalHooks | None:
        return None

    def after_local(self, client: int, local: torch.nn.Module, parent: dict[str, torch.Tensor], lr: float, steps: int, num_examples: int) -> None:
        return None

    def aggregate(self, updates: list[dict[str, Any]], parent: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return aggregate_fedavg(updates, parent_weights=parent)

    # --- ghi chú cho báo cáo -------------------------------------------------------------------------------
    def extra_bytes(self, model_state: dict[str, torch.Tensor]) -> tuple[int, int]:
        """Byte phụ mỗi client mỗi vòng ngoài trọng số mô hình: (lên máy chủ, xuống client)."""
        return 0, 0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, **self.options}

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


class FedProx(FedAlgorithm):
    name = "fedprox"

    def make_hooks(self, client, local, parent, lr, device):
        mu = float(self.options["mu"])
        anchor = {name: parent[name].to(device=device, dtype=torch.float32) for name, _ in local.named_parameters()}
        return ProxHooks(anchor, mu)


class FedBN(FedAlgorithm):
    """BatchNorm riêng cho từng client: tham số và thống kê BN của client không bị ghi đè bởi mô hình toàn cục.

    Mô hình toàn cục dùng để chọn checkpoint và đánh giá vẫn gộp BN theo FedAvg (cần một mô hình duy nhất để so cùng các cách khác); đây KHÔNG phải
    mô hình cá nhân hóa của FedBN. Mô hình cá nhân hóa của client k = trọng số toàn cục + BN riêng `client_bn[k]` (xem `personalized_state`)."""
    name = "fedbn"

    def __init__(self, model, num_clients, **options):
        super().__init__(model, num_clients, **options)
        # bộ đếm num_batches_tracked luôn lấy theo mô hình toàn cục: bộ đếm riêng sẽ nhỏ hơn bộ đếm cha và bị cổng gộp từ chối
        self.personal_keys = sorted(k for k in self.bn_keys if not k.endswith("num_batches_tracked"))
        self.client_bn: dict[int, dict[str, torch.Tensor]] = {}

    def start_state(self, client, parent):
        if client not in self.client_bn:
            return parent
        state = dict(parent)
        state.update(self.client_bn[client])
        return state

    def after_local(self, client, local, parent, lr, steps, num_examples):
        state = local.state_dict()
        self.client_bn[client] = {key: state[key].detach().cpu().clone() for key in self.personal_keys}

    def personalized_state(self, client: int, global_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.start_state(client, global_state)

    def state_dict(self):
        return {"client_bn": self.client_bn}

    def load_state_dict(self, state):
        self.client_bn = {int(k): v for k, v in (state.get("client_bn") or {}).items()}


class Scaffold(FedAlgorithm):
    """SCAFFOLD, phiên bản II (c_i⁺ = c_i − c + (x − y_i)/(K·η)). Với động lượng m, độ dời mỗi bước xấp xỉ η/(1−m)·g nên dùng η_hiệu_dụng = η/(1−m).
    Đây là xấp xỉ: bản gốc phân tích cho SGD thuần. Trung bình biến điều khiển theo n_k, cùng trọng số với phép gộp mô hình."""
    name = "scaffold"

    def __init__(self, model, num_clients, **options):
        super().__init__(model, num_clients, **options)
        zeros = {name: torch.zeros_like(p, device="cpu") for name, p in model.named_parameters()}
        self.c = {k: v.clone() for k, v in zeros.items()}
        self.ci = {i: {k: v.clone() for k, v in zeros.items()} for i in range(num_clients)}
        self._pending: dict[int, dict[str, torch.Tensor]] = {}
        self.momentum = float(options.get("momentum", 0.0))

    def make_hooks(self, client, local, parent, lr, device):
        correction = {k: (self.c[k] - self.ci[client][k]).to(device) for k in self.c}
        return ScaffoldHooks(correction)

    def after_local(self, client, local, parent, lr, steps, num_examples):
        effective_lr = lr / (1.0 - self.momentum) if self.momentum else lr
        y = _trainable(local)
        updated = {}
        for name in self.c:
            x = parent[name].float()
            updated[name] = self.ci[client][name] - self.c[name] + (x - y[name]) / (max(1, steps) * effective_lr)
        self._pending[client] = updated

    def aggregate(self, updates, parent):
        merged = aggregate_fedavg(updates, parent_weights=parent)
        total = sum(int(u["num_examples"]) for u in updates)
        for update in updates:
            k, weight = update["client_id"], int(update["num_examples"]) / total
            for name in self.c:
                self.c[name] += weight * (self._pending[k][name] - self.ci[k][name])
            self.ci[k] = self._pending[k]
        self._pending = {}
        return merged

    def extra_bytes(self, model_state):
        trainable = sum(t.numel() * t.element_size() for name, t in self.c.items())
        return trainable, trainable          # lên: Δc_i; xuống: c

    def state_dict(self):
        return {"c": self.c, "ci": self.ci}

    def load_state_dict(self, state):
        self.c = state["c"]
        self.ci = {int(k): v for k, v in state["ci"].items()}


class FedAvgM(FedAlgorithm):
    """FedAvg với động lượng phía máy chủ: v ← β·v + (w_gộp − w); w ← w + η_s·v. Chỉ áp cho tham số học được; bộ đệm BatchNorm lấy trung bình thường.
    β = 0 và η_s = 1 cho đúng FedAvg."""
    name = "fedavgm"

    def __init__(self, model, num_clients, **options):
        super().__init__(model, num_clients, **options)
        self.trainable = {name for name, _ in model.named_parameters()}
        self.velocity: dict[str, torch.Tensor] = {}

    def aggregate(self, updates, parent):
        merged = aggregate_fedavg(updates, parent_weights=parent)
        beta, server_lr = float(self.options["beta"]), float(self.options["server_lr"])
        for name in self.trainable:
            delta = merged[name].double() - parent[name].double()
            velocity = beta * self.velocity.get(name, torch.zeros_like(delta)) + delta
            self.velocity[name] = velocity
            merged[name] = (parent[name].double() + server_lr * velocity).to(parent[name].dtype)
        return merged

    def state_dict(self):
        return {"velocity": self.velocity}

    def load_state_dict(self, state):
        self.velocity = state.get("velocity") or {}


class Moon(FedAlgorithm):
    name = "moon"

    def __init__(self, model, num_clients, **options):
        super().__init__(model, num_clients, **options)
        self.previous: dict[int, dict[str, torch.Tensor]] = {}

    def make_hooks(self, client, local, parent, lr, device):
        global_model = copy.deepcopy(local).eval()          # `local` đang giữ đúng trạng thái toàn cục lúc bắt đầu vòng
        previous = None
        if client in self.previous:
            previous = copy.deepcopy(local).eval()
            previous.load_state_dict(self.previous[client])
        for model in (global_model, previous):
            if model is not None:
                for p in model.parameters():
                    p.requires_grad_(False)
        return MoonHooks(local, global_model, previous, float(self.options["mu"]), float(self.options["tau"]))

    def after_local(self, client, local, parent, lr, steps, num_examples):
        self.previous[client] = {k: v.detach().cpu().clone() for k, v in local.state_dict().items()}

    def state_dict(self):
        return {"previous": self.previous}

    def load_state_dict(self, state):
        self.previous = {int(k): v for k, v in (state.get("previous") or {}).items()}


def make_algorithm(name: str, model: torch.nn.Module, num_clients: int, *, mu: float = 0.01, moon_mu: float = 1.0, moon_tau: float = 0.5,
                   server_momentum: float = 0.9, server_lr: float = 0.1, momentum: float = 0.9) -> FedAlgorithm:
    """`mu` cho FedProx; `moon_mu`/`moon_tau` cho MOON; `server_momentum`/`server_lr` cho FedAvgM; `momentum` là động lượng SGD của client (SCAFFOLD cần)."""
    if name == "fedavg":
        return FedAlgorithm(model, num_clients)
    if name == "fedprox":
        return FedProx(model, num_clients, mu=mu)
    if name == "fedbn":
        return FedBN(model, num_clients)
    if name == "scaffold":
        return Scaffold(model, num_clients, momentum=momentum)
    if name == "fedavgm":
        return FedAvgM(model, num_clients, beta=server_momentum, server_lr=server_lr)
    if name == "moon":
        return Moon(model, num_clients, mu=moon_mu, tau=moon_tau)
    raise ValueError(f"thuật toán không có: {name} (có: {', '.join(ALGORITHMS)})")
