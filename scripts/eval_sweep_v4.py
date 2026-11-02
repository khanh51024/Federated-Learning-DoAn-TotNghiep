"""Gộp mọi run huấn luyện v4 thành bảng và biểu đồ theo cách chia client × thuật toán × hạt giống (Đợt 2, 3, 5 của yêu cầu đề tài).

Đầu vào: một hoặc nhiều thư mục run của scripts/train_v4.py (mỗi run có config.json, history.jsonl, logits_best.npz, final.json, DONE).
Script chỉ ĐỌC logits đã lưu, không chạy lại mô hình; riêng `--plantwild` tính logit PlantWild trên máy này khi chưa có cache (như scripts/eval_trained_v4.py).

Mỗi ô (cách chia, biến thể) có:
  parts          top-1 và macro-F1 trung bình ± độ lệch chuẩn qua hạt giống, tách PlantVillage / PlantDoc giữ riêng / ngoài đồng cùng nguồn / toàn tập thử
                 (Local-only: trung bình 5 client trong một hạt giống trước)
  gap            khoảng cách đến Centralized cùng hạt giống, tính bằng điểm phần trăm
  gap_closure    (thuật toán − FedAvg) / (Centralized − FedAvg) trên trung bình, đúng công thức N3.4
  vs_fedavg      so với FedAvg cùng cách chia, ghép cặp theo hạt giống và theo ảnh; xem `compare_to_fedavg` về tiêu chí "có ý nghĩa"
  communication  số vòng đến hội tụ, tổng MB đến hội tụ và cả ngân sách, so với gửi toàn bộ ảnh về trung tâm (1 MB = 10^6 byte)
  fairness       độ chính xác theo phân phối nhãn của từng client (N2.3), độ lệch chuẩn giữa client, chênh lệch mạnh–yếu, client nhỏ nhất
Run trùng ô (chạy lại) không bị lấy bừa: thư mục đứng sau trong --runs thắng, chênh lệch ghi ở `reruns` làm bằng chứng tái lập (N1.3).

Run khác công thức với Centralized gốc bị loại khỏi bảng và ghi lý do ở `excluded` (nguyên tắc "khác biệt duy nhất là cách dùng dữ liệu").
Ghi vào artifacts/sweep_v4/ (results.json, table_main.csv, table_fairness.csv, curves_by_alpha.png, gap_by_alpha.png); không đụng artifacts/train_v4_results.json.

Chạy:
    python scripts/eval_sweep_v4.py --runs runs/v4b runs/sweep
    python scripts/eval_sweep_v4.py --runs runs/v4b --plantwild
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_trained_v4 import V3_DIR, jsonl, part_masks, plantwild_logits  # noqa: E402
from scripts.report_client_fairness import client_accuracy, domain_class_ids, fairness_summary, group_recall  # noqa: E402
from scripts.report_statistics import _cluster_of, cluster_bootstrap_ci, mcnemar_exact  # noqa: E402
from src import registry as reg  # noqa: E402
from src.training.data import NUM_CLIENTS, load_release, load_shard  # noqa: E402
from src.training.metrics import NUM_CLASSES, macro_f1  # noqa: E402

BOOTSTRAP_SEED = 20261102
CONVERGENCE_TOLERANCE = 0.01          # 1 điểm macro-F1 của điểm chọn checkpoint (val)
MIN_SEEDS = 3                         # yêu cầu đề tài: mọi số chính trên tối thiểu 3 hạt giống
TEST_PARTS = ("plantvillage_test", "plantdoc_holdout_v3", "field_same_source_test", "all_test")
PLANTWILD_PART = "plantwild_strict"
PART_LABELS = {"plantvillage_test": "PlantVillage", "plantdoc_holdout_v3": "PlantDoc giữ riêng", "field_same_source_test": "Ngoài đồng cùng nguồn",
               "all_test": "Toàn tập thử", PLANTWILD_PART: "PlantWild"}
PLOT_PARTS = ("plantvillage_test", "field_same_source_test", "plantdoc_holdout_v3")
RECIPE_KEYS = ("budget", "local_epochs", "batch_size", "lr", "momentum", "weight_decay", "label_smoothing", "warmup", "train_transform", "train_size",
               "eval_transform", "loss_weighting", "channels_last", "amp", "selection")
FAIRNESS_SPLITS = ("val", "legacy_diagnostic_test")
DOMAINS = ("all", "plantvillage", "field")
RUNTIME_SUFFIX = "+flower"
_ALPHA_RE = re.compile(r"^label_alpha_(\d+)_(\d+)(?:_p(\d+))?$")


# --------------------------------------------------------------------------- hàm thuần (có test)
def scheme_alpha(scheme: str) -> tuple[float | None, int | None]:
    """(α, hạt giống chia khác nếu có). `iid` là α = vô cùng; kịch bản không theo Dirichlet nhãn trả (None, None)."""
    if scheme == "iid":
        return math.inf, None
    match = _ALPHA_RE.match(scheme)
    if not match:
        return None, None
    return float(f"{match.group(1)}.{match.group(2)}"), int(match.group(3)) if match.group(3) else None


def variant_of(cfg: dict[str, Any]) -> str:
    """Tên biến thể của một run: thuật toán + tham số + các cờ làm nó khác FedAvg thường; Centralized/Local có thể mang tên bộ tối ưu."""
    method = cfg["method"]
    if method != "fedavg":
        optimizer = cfg.get("optimizer", "sgd")
        parts = [method] if optimizer == "sgd" else [method, optimizer]
    else:
        algorithm = cfg.get("algorithm") or {"name": "fedavg"}
        params = ",".join(f"{k}={algorithm[k]}" for k in sorted(algorithm) if k != "name" and not (algorithm["name"] == "scaffold" and k == "momentum"))
        parts = [f"{algorithm['name']}({params})" if params else algorithm["name"]]
    if cfg.get("logit_adjust_tau"):
        parts.append(f"la={cfg['logit_adjust_tau']}")
    if cfg.get("personalization"):
        parts.append(f"tinh_chinh={cfg['personalization']['kind']}")
    if cfg.get("update_compression"):
        parts.append(f"nen={cfg['update_compression']}")
    if cfg.get("server_dp"):
        parts.append(f"dp={cfg['server_dp']['noise_multiplier']}/{cfg['server_dp']['clip']}")
    if cfg.get("runtime"):
        parts.append(str(cfg["runtime"]))
    return "+".join(parts)


def recipe_of(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: cfg.get(key) for key in RECIPE_KEYS}


def spread(per_seed: dict[int, float]) -> dict[str, Any]:
    """Trung bình ± độ lệch chuẩn mẫu qua hạt giống."""
    values = [per_seed[s] for s in sorted(per_seed)]
    return {"mean": float(np.mean(values)) if values else None, "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "per_seed": {int(s): float(per_seed[s]) for s in sorted(per_seed)}, "n_seeds": len(values)}


def gap_closure(algorithm: float | None, fedavg: float | None, centralized: float | None) -> float | None:
    """Tỷ lệ thu hẹp khoảng cách (N3.4). None khi FedAvg không thua Centralized: công thức khi đó không có nghĩa."""
    if algorithm is None or fedavg is None or centralized is None or centralized - fedavg <= 1e-9:
        return None
    return (algorithm - fedavg) / (centralized - fedavg)


def rounds_to_convergence(scores: list[float | None], tolerance: float = CONVERGENCE_TOLERANCE) -> int | None:
    """Số vòng ít nhất mà từ đó về sau điểm val không còn tụt dưới (tốt nhất − tolerance). None nếu vòng cuối vẫn tụt dưới mức đó."""
    known = [s for s in scores if s is not None]
    if not known:
        return None
    threshold = max(known) - tolerance
    below = [i for i, s in enumerate(scores) if s is None or s < threshold]
    last_below = below[-1] if below else -1
    return None if last_below == len(scores) - 1 else last_below + 2


def round_bytes(row: dict[str, Any], model_bytes: int) -> tuple[int, int, bool]:
    """(byte lên, byte xuống, có số trong nhật ký hay không) của một vòng; run cũ không ghi byte thì tính từ kích thước mô hình gửi thô."""
    clients = row["train"].get("clients", [])
    if clients and all("bytes_up" in c for c in clients):
        return sum(c["bytes_up"] for c in clients), sum(c["bytes_down"] for c in clients), True
    return len(clients) * model_bytes, len(clients) * model_bytes, False


def communication_of(history: list[dict[str, Any]], model_bytes: int) -> dict[str, Any]:
    scores = [row["val"]["selection_score"] for row in history]
    converged = rounds_to_convergence(scores)
    per_round = [round_bytes(row, model_bytes) for row in history]
    up = [r[0] for r in per_round]
    down = [r[1] for r in per_round]
    upto = converged if converged is not None else len(history)
    return {"rounds_done": len(history), "rounds_to_convergence": converged, "mb_up_total": sum(up) / 1e6, "mb_down_total": sum(down) / 1e6,
            "mb_total": (sum(up) + sum(down)) / 1e6, "mb_up_to_convergence": sum(up[:upto]) / 1e6, "mb_total_to_convergence": (sum(up[:upto]) + sum(down[:upto])) / 1e6,
            "bytes_from_history": all(r[2] for r in per_round), "train_seconds": float(sum(row.get("seconds", 0.0) for row in history))}


def compare_to_fedavg(correct_algo: dict[int, np.ndarray], correct_fedavg: dict[int, np.ndarray], clusters: list[str], rng: np.random.Generator) -> dict[str, Any] | None:
    """So một mô hình với mô hình đối chứng trên cùng ảnh, ghép theo hạt giống (mặc định là thuật toán so với FedAvg). Giá trị đúng/sai có thể là tỉ lệ khi một phía là trung bình nhiều client.

    Chênh lệch từng ảnh = (tỉ lệ đúng của thuật toán qua các hạt giống) − (của FedAvg); khoảng tin cậy 95% bằng bootstrap theo cụm ảnh.
    Tiêu chí "vượt có ý nghĩa" đặt trước khi có số: cận dưới khoảng tin cậy > 0 VÀ chênh lệch dương ở MỌI hạt giống. Với 3 hạt giống không dùng
    kiểm định t theo hạt giống vì chỉ có 2 bậc tự do. McNemar từng hạt giống ghi kèm để đối chiếu, không dùng để quyết định."""
    seeds = sorted(set(correct_algo) & set(correct_fedavg))
    if not seeds:
        return None
    algo = np.stack([correct_algo[s] for s in seeds]).astype(np.float64)
    base = np.stack([correct_fedavg[s] for s in seeds]).astype(np.float64)
    interval = cluster_bootstrap_ci(clusters, (algo - base).mean(axis=0), rng)
    by_seed = {int(s): float((algo[i] - base[i]).mean()) for i, s in enumerate(seeds)}
    low = interval["low"]
    return {"seeds": [int(s) for s in seeds], "diff_points": 100 * interval["estimate"], "ci95_points": None if low is None else [100 * low, 100 * interval["high"]],
            "diff_points_by_seed": {s: 100 * d for s, d in by_seed.items()}, "clusters": interval["clusters"],
            # McNemar chỉ có nghĩa khi mỗi ảnh là đúng/sai của một mô hình; Local-only là trung bình 5 client nên bỏ qua
            "mcnemar_p_by_seed": {int(s): mcnemar_exact(correct_algo[s], correct_fedavg[s])["p_value"] for s in seeds if correct_algo[s].dtype == bool and correct_fedavg[s].dtype == bool},
            "better_significant": bool(low is not None and low > 0 and all(d > 0 for d in by_seed.values()))}


def part_score(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    return {"top1": float(np.mean(preds == targets)), "macro_f1": macro_f1(targets, preds)}


# --------------------------------------------------------------------------- đọc run
def load_runs(run_dirs: list[Path], release: dict[str, Any], include_limited: bool, wild: dict[str, Any] | None, threads: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runs, excluded = [], []
    for base in run_dirs:
        for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            if not (run_dir / "DONE").is_file():
                if (run_dir / "config.json").is_file():
                    excluded.append({"run": str(run_dir), "reason": "chưa chạy xong (không có DONE)"})
                continue
            final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
            cfg = final["config"]
            if cfg.get("limit") and not include_limited:
                excluded.append({"run": str(run_dir), "reason": "run thử trên tập nhỏ (--limit)"})
                continue
            if cfg.get("release_manifest_sha256") != release["sha256"]:
                excluded.append({"run": str(run_dir), "reason": f"release khác: {cfg.get('release_manifest_sha256')}"})
                continue
            data = np.load(run_dir / "logits_best.npz", allow_pickle=False)
            preds = {name: data[name].argmax(axis=1).astype(np.int16) for name in FAIRNESS_SPLITS}
            ids = {name: data[f"{name}_sample_ids"].tolist() for name in FAIRNESS_SPLITS}
            personal = None
            personal_path = run_dir / "logits_personalized.npz"
            if personal_path.is_file():
                saved = np.load(personal_path, allow_pickle=False)
                personal = {name: [saved[f"{name}_client_{k}"].argmax(axis=1).astype(np.int16) for k in range(NUM_CLIENTS)] for name in FAIRNESS_SPLITS}
            wild_preds = None
            if wild is not None:
                wild_preds = plantwild_logits(run_dir, wild["items"], threads)[wild["strict"]].argmax(axis=1).astype(np.int16)
            # chết máy giữa lúc ghi nhật ký và lưu checkpoint làm một bước xuất hiện hai lần khi chạy tiếp: giữ dòng sau cùng của mỗi bước, nếu không số vòng và tổng MB bị đếm dư
            by_step: dict[Any, dict[str, Any]] = {}
            for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    by_step[row.get("step", len(by_step))] = row
            history = [by_step[step] for step in sorted(by_step)]
            runs.append({"name": run_dir.name, "dir": run_dir, "cfg": cfg, "method": cfg["method"], "seed": int(cfg["seed"]), "client": cfg.get("client"),
                         "scheme": cfg.get("scheme"), "variant": variant_of(cfg), "preds": preds, "ids": ids, "personal": personal, "wild": wild_preds,
                         "history": history, "best_step": final.get("best_step"), "selection_score": final.get("selection_score")})
    return runs, excluded


def model_size_bytes(run: dict[str, Any]) -> int:
    """Số byte gửi thô một mô hình, lấy từ checkpoint của run (như `state_bytes` trong src/federated/algorithms.py)."""
    import torch

    state = torch.load(run["dir"] / "checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state"]
    return sum(t.numel() * t.element_size() for t in state.values())


def raw_image_bytes(release: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    """Tổng dung lượng file ảnh của tập train: lượng phải gửi nếu mọi client đưa toàn bộ ảnh về trung tâm."""
    total, missing = 0, 0
    for row in release["train"]:
        path = dataset_root / row["relative_path"]
        if path.is_file():
            total += path.stat().st_size
        else:
            missing += 1
    if missing:
        raise SystemExit(f"thiếu {missing} ảnh train dưới {dataset_root}: không tính được dung lượng ảnh thô")
    return {"images": len(release["train"]), "mb": total / 1e6}


# --------------------------------------------------------------------------- gộp
def build_report(run_dirs: list[Path], release_dir: Path, dataset_root: Path, v3_test_ids: set[str], include_limited: bool = False,
                 wild: dict[str, Any] | None = None, threads: int = 8) -> dict[str, Any]:
    release = load_release(release_dir, None)
    runs, excluded = load_runs(run_dirs, release, include_limited, wild, threads)
    if not runs:
        raise SystemExit("không có run nào dùng được")
    rows_by_id = {name: {r["sample_id"]: r for r in release[name]} for name in FAIRNESS_SPLITS}

    baseline = next((r for r in runs if r["variant"] == "centralized"), None)
    if baseline is None:
        raise SystemExit("thiếu run Centralized SGD: không có mốc để tính khoảng cách")
    base_recipe = recipe_of(baseline["cfg"])
    kept = []
    for run in runs:
        changed = {k: (base_recipe[k], v) for k, v in recipe_of(run["cfg"]).items() if v != base_recipe[k]}
        if changed and run["variant"] == "centralized+adamw":
            run["recipe_differences"] = changed            # đối chứng N1.1: khác bộ tối ưu là chủ đích, ghi rõ
        elif changed:
            excluded.append({"run": run["name"], "reason": f"khác công thức với Centralized gốc: {changed}"})
            continue
        kept.append(run)
    runs = kept

    def seed_scores(run: dict[str, Any]) -> dict[str, dict[str, float]]:
        ids = run["ids"]["legacy_diagnostic_test"]
        rows = [rows_by_id["legacy_diagnostic_test"][s] for s in ids]
        targets = np.array([int(r["class_id"]) for r in rows])
        masks = {**part_masks(rows, v3_test_ids), "all_test": np.ones(len(rows), dtype=bool)}
        out = {part: part_score(run["preds"]["legacy_diagnostic_test"][masks[part]], targets[masks[part]]) for part in TEST_PARTS if masks[part].any()}
        run["correct"] = {part: run["preds"]["legacy_diagnostic_test"][masks[part]] == targets[masks[part]] for part in TEST_PARTS if masks[part].any()}
        run["part_clusters"] = {part: [_cluster_of(r) for r, m in zip(rows, masks[part]) if m] for part in TEST_PARTS if masks[part].any()}
        run["part_n"] = {part: int(masks[part].sum()) for part in TEST_PARTS if masks[part].any()}
        if wild is not None:
            targets_wild = wild["targets"][wild["strict"]]
            out[PLANTWILD_PART] = part_score(run["wild"], targets_wild)
            run["correct"][PLANTWILD_PART] = run["wild"] == targets_wild
            run["part_clusters"][PLANTWILD_PART] = wild["clusters"]
            run["part_n"][PLANTWILD_PART] = int(len(targets_wild))
        return out

    for run in runs:
        run["scores"] = seed_scores(run)

    # Cùng ô, cùng hạt giống, cùng client xuất hiện hơn một lần (chạy lại): thư mục đứng sau trong --runs thắng, chênh lệch được ghi lại làm bằng chứng tái lập (N1.3)
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    reruns: list[dict[str, Any]] = []
    for run in runs:
        key = (run["scheme"] if run["method"] != "centralized" else None, run["variant"], run["seed"], run["client"])
        if key in latest:
            old = latest[key]
            reruns.append({"variant": run["variant"], "scheme": key[0], "seed": run["seed"], "client": run["client"], "kept": str(run["dir"]), "replaced": str(old["dir"]),
                           "top1_diff_points": {part: 100 * (run["scores"][part]["top1"] - old["scores"][part]["top1"]) for part in run["scores"] if part in old["scores"]}})
        latest[key] = run
    runs = list(latest.values())

    # gom theo ô; Centralized không phụ thuộc cách chia
    cells: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        cells[(run["scheme"] if run["method"] != "centralized" else None, run["variant"])].append(run)

    def seed_part_values(group: list[dict[str, Any]], part: str, metric: str) -> dict[int, float]:
        """Giá trị theo hạt giống; Local-only: trung bình các client (chỉ tính hạt giống đủ 5 client)."""
        by_seed: dict[int, list[float]] = defaultdict(list)
        for run in group:
            if part in run["scores"]:
                by_seed[run["seed"]].append(run["scores"][part][metric])
        if group[0]["method"] == "local":
            return {s: float(np.mean(v)) for s, v in by_seed.items() if len(v) == NUM_CLIENTS}
        return {s: float(v[0]) for s, v in by_seed.items()}

    central = cells.get((None, "centralized"), [])
    parts_used = list(TEST_PARTS) + ([PLANTWILD_PART] if wild is not None else [])
    central_values = {part: {m: seed_part_values(central, part, m) for m in ("top1", "macro_f1")} for part in parts_used}
    size = model_size_bytes(baseline)
    raw = raw_image_bytes(release, dataset_root)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    report_cells: list[dict[str, Any]] = []
    shard_cache: dict[str, list[list[dict[str, Any]]]] = {}

    def shards_of(scheme: str) -> list[list[dict[str, Any]]]:
        if scheme not in shard_cache:
            shard_cache[scheme] = [load_shard(release_dir, scheme, k, release) for k in range(NUM_CLIENTS)]
        return shard_cache[scheme]

    for (scheme, variant), group in sorted(cells.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])):
        alpha, partition_seed = scheme_alpha(scheme) if scheme else (None, None)
        method = group[0]["method"]
        cell: dict[str, Any] = {"scheme": scheme, "alpha": None if alpha is None else ("iid" if math.isinf(alpha) else alpha), "partition_seed": partition_seed,
                                "method": method, "variant": variant, "seeds": sorted({r["seed"] for r in group}), "runs": sorted(r["name"] for r in group)}
        cell["below_min_seeds"] = len(cell["seeds"]) < MIN_SEEDS
        if method != "local":
            cell["val_selection_score"] = spread({r["seed"]: r["selection_score"] for r in group})
        if group[0].get("recipe_differences"):
            cell["recipe_differences"] = group[0]["recipe_differences"]
        cell["parts"] = {}
        # thuật toán chạy trên Flower so với FedAvg cũng trên Flower; bản thân FedAvg trên Flower so với FedAvg một tiến trình, đó là phép kiểm chứng tương đương N1.2
        on_flower = variant.endswith(RUNTIME_SUFFIX) and variant != f"fedavg{RUNTIME_SUFFIX}" and (scheme, f"fedavg{RUNTIME_SUFFIX}") in cells
        fedavg_group = cells.get((scheme, f"fedavg{RUNTIME_SUFFIX}" if on_flower else "fedavg"))
        fedavg_values = {part: {m: seed_part_values(fedavg_group, part, m) for m in ("top1", "macro_f1")} for part in parts_used} if fedavg_group else None
        for part in parts_used:
            values = {m: seed_part_values(group, part, m) for m in ("top1", "macro_f1")}
            if not values["top1"]:
                continue
            example = next(r for r in group if part in r["scores"])
            block: dict[str, Any] = {"n": example["part_n"][part], "top1": spread(values["top1"]), "macro_f1": spread(values["macro_f1"])}
            if method != "centralized" or variant != "centralized":
                for metric in ("top1", "macro_f1"):
                    paired = {s: 100 * (central_values[part][metric][s] - v) for s, v in values[metric].items() if s in central_values[part][metric]}
                    if paired:
                        block[f"gap_{metric}_points"] = spread(paired)
            if method == "fedavg" and variant != "fedavg" and fedavg_values:
                common = sorted(set(values["top1"]) & set(fedavg_values[part]["top1"]) & set(central_values[part]["top1"]))
                if common:
                    block["gap_closure_top1"] = gap_closure(float(np.mean([values["top1"][s] for s in common])), float(np.mean([fedavg_values[part]["top1"][s] for s in common])),
                                                             float(np.mean([central_values[part]["top1"][s] for s in common])))
                    block["fedavg_gap_points"] = 100 * float(np.mean([central_values[part]["top1"][s] - fedavg_values[part]["top1"][s] for s in common]))
                    block["vs_fedavg"] = compare_to_fedavg({r["seed"]: r["correct"][part] for r in group}, {r["seed"]: r["correct"][part] for r in fedavg_group},
                                                           example["part_clusters"][part], rng)
            if method == "fedavg":
                # Tập trung hơn ô này có ý nghĩa không; FedAvg có hơn Local-only trung bình 5 client không (cần cho tập nhỏ như PlantDoc 172 ảnh)
                block["centralized_vs_this"] = compare_to_fedavg({r["seed"]: r["correct"][part] for r in central if part in r["correct"]}, {r["seed"]: r["correct"][part] for r in group},
                                                                 example["part_clusters"][part], rng)
                local_group = cells.get((scheme, "local"))
                if local_group and variant == "fedavg":
                    by_seed: dict[int, list[np.ndarray]] = defaultdict(list)
                    for run in local_group:
                        by_seed[run["seed"]].append(run["correct"][part])
                    local_mean = {s: np.mean(v, axis=0) for s, v in by_seed.items() if len(v) == NUM_CLIENTS}
                    block["this_vs_local"] = compare_to_fedavg({r["seed"]: r["correct"][part] for r in group}, local_mean, example["part_clusters"][part], rng)
            cell["parts"][part] = block
        if method == "fedavg":
            per_seed = {r["seed"]: communication_of(r["history"], size) for r in group}
            cell["communication"] = {
                "bytes_from_history": all(c["bytes_from_history"] for c in per_seed.values()),
                "model_mb_per_direction": size / 1e6,
                **{key: spread({s: c[key] for s, c in per_seed.items() if c[key] is not None})
                   for key in ("rounds_to_convergence", "mb_up_total", "mb_total", "mb_up_to_convergence", "mb_total_to_convergence", "train_seconds")},
                "raw_images_mb": raw["mb"],
            }
            cell["communication"]["mb_up_total_vs_raw"] = cell["communication"]["mb_up_total"]["mean"] / raw["mb"]
            cell["communication"]["mb_total_vs_raw"] = cell["communication"]["mb_total"]["mean"] / raw["mb"]
        if method in ("fedavg", "local") and scheme:
            cell["fairness"] = fairness_of_cell(group, method, shards_of(scheme), rows_by_id)
            if any(r["personal"] for r in group):
                cell["fairness_personalized"] = fairness_of_cell(group, "personalized", shards_of(scheme), rows_by_id)
                cell["parts_personalized"] = personalized_parts(group, rows_by_id, v3_test_ids)
        report_cells.append(cell)

    return {"evidence": "gộp từ logits và nhật ký đã lưu của từng run; không chạy lại mô hình", "release_manifest_sha256": release["sha256"],
            "definitions": {"gap_points": "Centralized − ô này, cùng hạt giống, điểm phần trăm", "gap_closure_top1": "(thuật toán − FedAvg) / (Centralized − FedAvg) trên trung bình",
                            "rounds_to_convergence": f"số vòng ít nhất mà từ đó điểm val không tụt dưới (tốt nhất − {CONVERGENCE_TOLERANCE})",
                            "mb": "1 MB = 10^6 byte; đường một tiến trình tính từ kích thước tensor gửi thô; đường Flower là phần mảng trong tin nhắn, lớn hơn khoảng 0,5% do đầu mục của từng mảng; đều chưa tính khung truyền trên mạng",
                            "fairness": "độ chính xác = tổng theo cặp (miền, lớp) của (tỉ lệ cặp trong shard client × tỉ lệ đúng của cặp đó trên tập đo); cặp không có ảnh trong tập đo bị bỏ và chuẩn hóa lại, phần bị bỏ ghi ở coverage_min",
                            "gap_closure_note": "không đọc được khi fedavg_gap_points cùng cỡ với độ lệch chuẩn của khoảng cách: mẫu số quá nhỏ",
                            "fairness_std": "độ lệch chuẩn tổng thể giữa 5 client trong một hạt giống, rồi lấy trung bình qua hạt giống",
                            "better_significant": "cận dưới khoảng tin cậy 95% bootstrap theo cụm ảnh > 0 và chênh lệch dương ở mọi hạt giống"},
            "centralized_recipe": base_recipe, "raw_images": raw, "cells": report_cells, "excluded": excluded, "reruns": reruns}


def personalized_parts(group: list[dict[str, Any]], rows_by_id: dict[str, dict[str, dict[str, Any]]], v3_test_ids: set[str]) -> dict[str, Any]:
    """Mô hình cá nhân hóa trên tập thử CHUNG (đủ 38 lớp): trung bình 5 client trong một hạt giống. Cho thấy cái giá của việc cá nhân hóa: giỏi hơn trên phân phối riêng, có thể kém đi trên lớp client không có."""
    split = "legacy_diagnostic_test"
    per_seed: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: {"top1": {}, "macro_f1": {}})
    for run in group:
        if not run["personal"]:
            continue
        rows = [rows_by_id[split][s] for s in run["ids"][split]]
        targets = np.array([int(r["class_id"]) for r in rows])
        masks = {**part_masks(rows, v3_test_ids), "all_test": np.ones(len(rows), dtype=bool)}
        for part in TEST_PARTS:
            if not masks[part].any():
                continue
            scores = [part_score(run["personal"][split][k][masks[part]], targets[masks[part]]) for k in range(NUM_CLIENTS)]
            for metric in ("top1", "macro_f1"):
                per_seed[part][metric][run["seed"]] = float(np.mean([s[metric] for s in scores]))
    return {part: {metric: spread(values) for metric, values in block.items()} for part, block in per_seed.items()}


def fairness_of_cell(group: list[dict[str, Any]], kind: str, shards: list[list[dict[str, Any]]], rows_by_id: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Độ chính xác của mô hình trên phân phối nhãn từng client. Toàn cục: một mô hình đo cho cả 5 client. Local-only: mô hình của chính client đó.
    `personalized`: FedBN, trọng số toàn cục + BatchNorm riêng của client đó."""
    num_groups = 2 * NUM_CLASSES
    counts = np.stack([np.bincount(domain_class_ids(rows, NUM_CLASSES), minlength=num_groups).astype(np.float64) for rows in shards])
    domain_groups = {"all": np.ones(num_groups, dtype=bool), "plantvillage": np.arange(num_groups) < NUM_CLASSES, "field": np.arange(num_groups) >= NUM_CLASSES}
    n_k = [len(rows) for rows in shards]
    seeds = sorted({r["seed"] for r in group})
    out: dict[str, Any] = {"n_k": n_k}
    for split in FAIRNESS_SPLITS:
        out[split] = {}
        for domain in DOMAINS:
            per_seed: dict[int, dict[str, Any]] = {}
            for seed in seeds:
                members = [r for r in group if r["seed"] == seed]
                if kind == "local" and len(members) != NUM_CLIENTS:
                    continue
                accuracies: list[float | None] = []
                coverages: list[float] = []
                for k in range(NUM_CLIENTS):
                    if kind == "local":
                        run = next(r for r in members if r["client"] == k)
                        preds = run["preds"][split]
                    elif kind == "personalized":
                        run = members[0]
                        if not run["personal"]:
                            continue
                        preds = run["personal"][split][k]
                    else:
                        run = members[0]
                        preds = run["preds"][split]
                    rows = [rows_by_id[split][s] for s in run["ids"][split]]
                    targets = np.array([int(r["class_id"]) for r in rows])
                    groups = domain_class_ids(rows, NUM_CLASSES)
                    result = client_accuracy(group_recall(groups, preds == targets, num_groups), counts[k] * domain_groups[domain])
                    accuracies.append(result["accuracy"])
                    coverages.append(result["coverage"])
                if len(accuracies) == NUM_CLIENTS and any(a is not None for a in accuracies):    # miền không có ảnh trong tập đo thì không có khối số
                    per_seed[seed] = {**fairness_summary(accuracies, n_k), "coverage_min": min(coverages), "by_client": accuracies}
            if not per_seed:
                continue
            block = {key: spread({s: v[key] for s, v in per_seed.items() if v[key] is not None})
                     for key in ("mean", "std_between_clients", "best", "worst", "gap_best_worst", "smallest_client_accuracy")}
            block["coverage_min"] = min(v["coverage_min"] for v in per_seed.values())
            client_values = [[v["by_client"][k] for v in per_seed.values() if v["by_client"][k] is not None] for k in range(NUM_CLIENTS)]
            block["by_client_mean"] = [float(np.mean(values)) if values else None for values in client_values]
            out[split][domain] = block
    return out


# --------------------------------------------------------------------------- bảng
def write_tables(report: dict[str, Any], out_dir: Path) -> None:
    fmt = lambda value, digits=2: "" if value is None else f"{value:.{digits}f}"   # noqa: E731
    with (out_dir / "table_main.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scheme", "alpha", "variant", "seeds", "part", "n", "top1_mean_%", "top1_std_%", "macro_f1_mean_%", "macro_f1_std_%", "gap_top1_points", "gap_macro_f1_points",
                         "gap_closure_top1", "diff_vs_fedavg_points", "ci95_low", "ci95_high", "better_significant", "rounds_to_convergence", "mb_total_to_convergence", "mb_total", "mb_total_vs_raw_images"])
        for cell in report["cells"]:
            comm = cell.get("communication", {})
            for part, block in cell["parts"].items():
                versus = block.get("vs_fedavg") or {}
                ci = versus.get("ci95_points") or [None, None]
                pct = lambda s: None if s is None or s["mean"] is None else 100 * s["mean"]   # noqa: E731
                std = lambda s: None if s is None or s["std"] is None else 100 * s["std"]     # noqa: E731
                writer.writerow([cell["scheme"] or "", cell["alpha"] if cell["alpha"] is not None else "", cell["variant"], len(cell["seeds"]), part, block["n"],
                                 fmt(pct(block["top1"])), fmt(std(block["top1"])), fmt(pct(block["macro_f1"])), fmt(std(block["macro_f1"])),
                                 fmt(block.get("gap_top1_points", {}).get("mean")), fmt(block.get("gap_macro_f1_points", {}).get("mean")), fmt(block.get("gap_closure_top1"), 3),
                                 fmt(versus.get("diff_points")), fmt(ci[0]), fmt(ci[1]), versus.get("better_significant", ""),
                                 fmt((comm.get("rounds_to_convergence") or {}).get("mean"), 1), fmt((comm.get("mb_total_to_convergence") or {}).get("mean"), 0),
                                 fmt((comm.get("mb_total") or {}).get("mean"), 0), fmt(comm.get("mb_total_vs_raw"), 3)])
    with (out_dir / "table_fairness.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scheme", "alpha", "variant", "model", "split", "domain", "seeds", "mean_%", "std_between_clients_%", "best_%", "worst_%", "gap_best_worst_points", "smallest_client_%", "coverage_min"])
        for cell in report["cells"]:
            for key in ("fairness", "fairness_personalized"):
                block = cell.get(key)
                if not block:
                    continue
                for split in FAIRNESS_SPLITS:
                    for domain, data in block[split].items():
                        pct = lambda name: fmt(None if data[name]["mean"] is None else 100 * data[name]["mean"])  # noqa: E731,B023
                        writer.writerow([cell["scheme"], cell["alpha"] if cell["alpha"] is not None else "", cell["variant"], "ca_nhan_hoa" if key.endswith("personalized") else "chinh",
                                         split, domain, len(cell["seeds"]), pct("mean"), pct("std_between_clients"), pct("best"), pct("worst"), pct("gap_best_worst"),
                                         pct("smallest_client_accuracy"), fmt(data["coverage_min"], 3)])


# --------------------------------------------------------------------------- biểu đồ (màu theo thực thể, không đổi giữa các hình; đã qua validate_palette.js)
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES_COLORS = {"centralized": "#2a78d6", "fedavg": "#eb6834", "local": "#1baf7a", "fedprox": "#eda100", "fedbn": "#e87ba4", "scaffold": "#008300", "moon": "#6250d6", "fedbnprox": "#e34948"}
SERIES_ORDER = ("centralized", "fedavg", "local", "fedprox", "fedbn", "scaffold", "moon", "fedbnprox")
SERIES_MARKERS = {"fedavg": "o", "local": "s", "fedprox": "^", "fedbn": "D", "scaffold": "v", "moon": "p", "fedbnprox": "h"}   # mã hóa phụ: in đen trắng vẫn phân biệt được
SERIES_LABELS = {"centralized": "Tập trung", "fedavg": "FedAvg", "local": "Cục bộ", "fedprox": "FedProx", "fedbn": "FedBN", "scaffold": "SCAFFOLD", "moon": "MOON", "fedbnprox": "FedBN+FedProx"}


def alpha_positions(cells: list[dict[str, Any]]) -> list[Any]:
    """Các mức α có mặt, từ lệch mạnh đến gần đồng nhất, `iid` cuối cùng; bỏ các cách chia dùng hạt giống chia khác (đo dao động, không phải mức lệch)."""
    values = {c["alpha"] for c in cells if c["alpha"] is not None and c["partition_seed"] is None}
    return sorted((v for v in values if v != "iid")) + (["iid"] if "iid" in values else [])


def plot_series(report: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Các ô ứng viên của mỗi họ thuật toán theo từng α; biến thể có thêm cờ (dấu +) không vào đường chính. `plot_candidate` chọn một ô để vẽ."""
    series: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for cell in report["cells"]:
        variant = cell["variant"].removesuffix(RUNTIME_SUFFIX)      # cùng thuật toán, khác đường chạy: vẫn thuộc đường chính
        if cell["alpha"] is None or cell["partition_seed"] is not None or "+" in variant:
            continue
        family = "local" if cell["method"] == "local" else variant.split("(")[0]
        if family in SERIES_COLORS:
            series[family][str(cell["alpha"])].append(cell)
    return series


def plot_candidate(series: dict[str, dict[str, list[dict[str, Any]]]], family: str, position: Any, part: str) -> dict[str, Any] | None:
    """Ô để vẽ: đủ số hạt giống và có điểm val cao nhất trong họ. Chọn theo val, không theo tập thử (N3.2); run dò tham số một hạt giống không vào hình."""
    candidates = [c for c in series.get(family, {}).get(str(position), []) if not c["below_min_seeds"] and part in c["parts"]]
    if not candidates:
        return None
    plain = [c for c in candidates if c.get("variant") == family]      # cùng một thuật toán có cả hai đường chạy: giữ một đường cho cả đồ thị, không chọn đường nào có val cao hơn
    return max(plain or candidates, key=lambda c: (c.get("val_selection_score") or {}).get("mean") or 0.0)


def _style_axis(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)


def plot_figures(report: dict[str, Any], out_dir: Path) -> list[str]:
    """Hai hình theo α, cần ít nhất 2 mức α; chỉ vẽ ô đủ tối thiểu 3 hạt giống để không ngầm báo cáo số một lần chạy.

    curves_by_alpha: ba mốc Tập trung, FedAvg, Cục bộ (N2.4). gap_by_alpha: khoảng cách đến Tập trung của FedAvg và các thuật toán (N3); tách riêng vì
    Cục bộ thấp hơn hàng chục điểm, vẽ chung thì các thuật toán dính vào nhau. Trong mỗi hàng các ô cùng thang trục để so được giữa các tập."""
    cells = report["cells"]
    positions = alpha_positions(cells)
    if len(positions) < 2:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["IID" if p == "iid" else f"{p:g}" for p in positions]
    xs = list(range(len(positions)))
    series = plot_series(report)
    central = next((c for c in cells if c["variant"] == "centralized"), None)
    written = []

    def values_at(family: str, position: Any, part: str, metric: str) -> tuple[float, float] | None:
        cell = plot_candidate(series, family, position, part)
        if cell is None:
            return None
        block = cell["parts"][part][metric]
        return block["mean"], block["std"] or 0.0

    fig, axes = plt.subplots(2, len(PLOT_PARTS), figsize=(13, 7.2), sharex=True, sharey="row", facecolor=SURFACE)
    for row, (metric, metric_label) in enumerate((("top1", "Top-1 (%)"), ("macro_f1", "Macro-F1 (%)"))):
        for col, part in enumerate(PLOT_PARTS):
            ax = axes[row][col]
            _style_axis(ax)
            if central and part in central["parts"]:
                mean, std = central["parts"][part][metric]["mean"], central["parts"][part][metric]["std"] or 0.0
                ax.axhline(100 * mean, color=SERIES_COLORS["centralized"], linewidth=2, linestyle=(0, (5, 3)), label=SERIES_LABELS["centralized"], zorder=2)
                ax.axhspan(100 * (mean - std), 100 * (mean + std), color=SERIES_COLORS["centralized"], alpha=0.14, linewidth=0, zorder=1)
            for family in ("fedavg", "local"):
                points = [(x, values_at(family, p, part, metric)) for x, p in zip(xs, positions)]
                points = [(x, v) for x, v in points if v is not None]
                if not points:
                    continue
                px = [x for x, _ in points]
                mean = np.array([100 * v[0] for _, v in points])
                std = np.array([100 * v[1] for _, v in points])
                ax.fill_between(px, mean - std, mean + std, color=SERIES_COLORS[family], alpha=0.16, linewidth=0, zorder=2)
                ax.plot(px, mean, color=SERIES_COLORS[family], linewidth=2, marker=SERIES_MARKERS[family], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=SERIES_LABELS[family], zorder=3)
            ax.set_xticks(xs, labels)
            if row == 0:
                ax.set_title(PART_LABELS[part], color=INK, fontsize=11, loc="left", pad=8)
            if col == 0:
                ax.set_ylabel(metric_label, color=INK_SOFT, fontsize=10)
            if row == 1:
                ax.set_xlabel("Mức α (nhỏ = lệch mạnh)", color=INK_SOFT, fontsize=10)
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(names), frameon=False, labelcolor=INK_SOFT, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "curves_by_alpha.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)
    written.append("curves_by_alpha.png")

    fig, axes = plt.subplots(1, len(PLOT_PARTS), figsize=(13, 4), sharex=True, sharey=True, facecolor=SURFACE)
    for col, part in enumerate(PLOT_PARTS):
        ax = axes[col]
        _style_axis(ax)
        ax.axhline(0, color=INK_SOFT, linewidth=1, zorder=1)
        for family in SERIES_ORDER[1:]:
            if family == "local":
                continue
            points = []
            for x, position in zip(xs, positions):
                cell = plot_candidate(series, family, position, part)
                if cell is not None and "gap_top1_points" in cell["parts"][part]:
                    block = cell["parts"][part]["gap_top1_points"]
                    points.append((x, block["mean"], block["std"] or 0.0))
            if not points:
                continue
            px = [p[0] for p in points]
            mean = np.array([p[1] for p in points])
            std = np.array([p[2] for p in points])
            ax.fill_between(px, mean - std, mean + std, color=SERIES_COLORS[family], alpha=0.10, linewidth=0, zorder=2)
            ax.plot(px, mean, color=SERIES_COLORS[family], linewidth=2, marker=SERIES_MARKERS[family], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1, label=SERIES_LABELS[family], zorder=3)
        ax.set_xticks(xs, labels)
        ax.set_title(PART_LABELS[part], color=INK, fontsize=11, loc="left", pad=8)
        ax.set_xlabel("Mức α (nhỏ = lệch mạnh)", color=INK_SOFT, fontsize=10)
        if col == 0:
            ax.set_ylabel("Khoảng cách đến Tập trung (điểm %)", color=INK_SOFT, fontsize=10)
    handles, names = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, loc="upper center", ncol=len(names), frameon=False, labelcolor=INK_SOFT, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_dir / "gap_by_alpha.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)
    written.append("gap_by_alpha.png")
    return written


# --------------------------------------------------------------------------- chạy
def load_plantwild() -> dict[str, Any]:
    from scripts.build_external_eval_set import EXTERNAL_DIR, MANIFEST

    rows = json.loads(reg.abs_path(MANIFEST).read_text(encoding="utf-8"))["samples"]
    base = reg.abs_path(EXTERNAL_DIR)
    strict = np.array([r["tier"] == "strict" for r in rows])
    return {"items": [{"key": r["byte_sha256"], "path": base / r["relative_path"]} for r in rows], "targets": np.array([int(r["class_id"]) for r in rows]),
            "strict": strict, "clusters": [r["byte_sha256"] for r, s in zip(rows, strict) if s]}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "dataset_sources")
    parser.add_argument("--release-dir", type=Path, default=None, help="mặc định <dataset-root>/mixed/pv_pd_v4")
    parser.add_argument("--out", type=Path, default=ROOT / reg.SWEEP_V4_DIR)
    parser.add_argument("--plantwild", action="store_true", help="thêm tập PlantWild (dùng cache plantwild_logits.npz của run, thiếu thì tính trên máy này)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--include-limited", action="store_true", help="tính cả run thử --limit, chỉ để kiểm script; bắt buộc đi kèm --out khác mặc định")
    args = parser.parse_args()
    if args.include_limited and args.out == ROOT / reg.SWEEP_V4_DIR:
        parser.error("--include-limited không được ghi vào thư mục kết quả chính thức; dùng --out tmp/...")
    release_dir = args.release_dir or args.dataset_root / "mixed" / "pv_pd_v4"
    v3_test_ids = {r["sample_id"] for r in jsonl(reg.abs_path(V3_DIR) / "manifests" / "legacy_diagnostic_test.jsonl")}
    report = build_report(args.runs, release_dir, args.dataset_root, v3_test_ids, args.include_limited, load_plantwild() if args.plantwild else None, args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_tables(report, args.out)
    figures = plot_figures(report, args.out)
    print(f"[sweep] {len(report['cells'])} ô, loại {len(report['excluded'])} run; ghi {args.out}; biểu đồ: {figures or 'chưa đủ 2 mức α'}")
    for item in report["excluded"]:
        print(f"[sweep] loại {item['run']}: {item['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
