"""Thí nghiệm mô phỏng học phân tán cộng đồng: nhiều người tham gia, nhiều vòng, đo từng phiên bản trên tập test giữ riêng.

Mỗi vòng: K người mô phỏng, mỗi người 30-100 ảnh của kho calibration (không giao train/validation), học lớp cuối bằng đúng SGD của
trình duyệt (`src/federated/community_sim.py`, đã kiểm khớp JS), máy chủ kiểm, cắt chuẩn, gộp FedAvg có trần và qua cổng giám sát đúng
như `web/services/community_store.py`. Số báo cáo là độ chính xác / macro-recall / PlantDoc của phiên bản hiện hành trên test giữ riêng
(11.015 ảnh, chưa dùng để chọn mô hình); cổng chạy trên tập giám sát validation như production.

Chạy (sau scripts/build_community_sim_features.py):
    python scripts/simulate_community.py              # đủ kịch bản, ghi artifacts/community_fl/simulation_report.json
    python scripts/simulate_community.py --quick      # 1 seed, 5 vòng, không ghi
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402
from src.federated import community as core  # noqa: E402
from src.federated import community_sim as sim  # noqa: E402

FEATURES = "data/community_sim/features.npz"
REPORT = "artifacts/community_fl/simulation_report.json"
KS = (3, 10, 25)
SEEDS = (11, 12, 13, 14, 15)
ROUNDS = 20
METRICS = ("accuracy", "macro_recall", "plantdoc_accuracy", "plantvillage_accuracy")


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Gộp các seed: đường trung bình theo vòng, số cuối cùng của từng seed, tỉ lệ vòng được nhận, chuẩn cập nhật."""
    curves = {m: np.array([[point[m] for point in run["curve"]] for run in runs], dtype=np.float64) for m in METRICS}
    finals = {m: curves[m][:, -1] for m in METRICS}
    base = {m: float(curves[m][0, 0]) for m in METRICS}
    logs = [r for run in runs for r in run["rounds"]]
    users = [u for r in logs for u in r["users"]]
    return {
        "seeds": len(runs),
        "rounds": len(runs[0]["rounds"]),
        "mean_curve": {m: [round(float(v), 5) for v in curves[m].mean(axis=0)] for m in METRICS},
        "final_mean": {m: round(float(finals[m].mean()), 5) for m in METRICS},
        "final_std": {m: round(float(finals[m].std(ddof=1)) if len(runs) > 1 else 0.0, 5) for m in METRICS},
        "final_delta_points": {m: [round(float((v - base[m]) * 100), 3) for v in finals[m]] for m in METRICS},
        "final_delta_mean_points": {m: round(float((finals[m].mean() - base[m]) * 100), 3) for m in METRICS},
        "accepted_round_share": round(float(np.mean([r["accepted"] for r in logs])), 4),
        "rejected_by_gate": int(sum(1 for r in logs if r.get("reasons"))),
        "mean_update_norm": round(float(np.mean([u["norm"] for u in users])), 4) if users else None,
        "clipped_share": round(float(np.mean([u["clipped"] for u in users])), 4) if users else None,
        "mean_delta_norm_per_round": round(float(np.mean([r["delta_norm"] for r in logs if "delta_norm" in r])), 5)
        if any("delta_norm" in r for r in logs) else None,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="1 seed, 5 vòng, K=10; không ghi báo cáo")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    path = reg.abs_path(FEATURES)
    if not path.is_file():
        print(f"[sim] chưa có {FEATURES} — chạy scripts/build_community_sim_features.py trước", file=sys.stderr)
        return 1
    with np.load(path, allow_pickle=False) as data:
        pool = sim.Pool(data["cal_x"].astype(np.float32), data["cal_y"], data["cal_s"])
        test_set = core.EvalSet(data["test_x"].astype(np.float32), data["test_y"], data["test_s"])
        features_fingerprint = str(data["fingerprint"])
        test_manifest_sha = str(data["test_manifest_sha256"])
    assets_dir = reg.abs_path(reg.COMMUNITY_FL_DIR)
    manifest = json.loads((assets_dir / "community_manifest.json").read_text(encoding="utf-8"))
    with np.load(assets_dir / manifest["base_head"]["file"], allow_pickle=False) as data:
        base = core.as_head(data["weight"], data["bias"])
    gate_set = core.load_eval_set(assets_dir / manifest["eval_features"]["file"], manifest["eval_features"]["sha256"])

    policy = core.Policy()
    config = sim.LocalConfig(epochs=policy.local_epochs, lr=policy.local_lr, batch=policy.local_batch, holdout_fraction=policy.holdout_fraction)
    seeds = SEEDS[:1] if args.quick else SEEDS
    ks = (10,) if args.quick else KS
    rounds = 5 if args.quick else args.rounds
    base_test = core.evaluate_head(base, test_set)
    print(f"[sim] kho {len(pool.labels)} ảnh ({len(pool.classes)} nhóm), test {len(test_set.labels)} ảnh; "
          f"base test: chính xác {base_test['accuracy']:.4f}, macro-recall {base_test['macro_recall']:.4f}, PlantDoc {base_test['plantdoc_accuracy']:.4f}")

    results: dict[str, Any] = {}
    started = time.time()
    plan = [(scenario, k, True) for scenario in sim.SCENARIOS for k in ks]
    plan += [(scenario, 10, False) for scenario in ("iid", "three_groups", "plantdoc_only")]  # đối chứng: không có cổng giám sát
    for scenario, k, gate in plan:
        runs = [sim.run_scenario(pool, gate_set, test_set, base, policy, config, scenario=scenario, k=k, rounds=rounds,
                                 seed=seed * 1000 + k, gate_enabled=gate) for seed in seeds]
        key = f"{scenario}|K={k}|{'gate' if gate else 'no_gate'}"
        results[key] = {"scenario": scenario, "k": k, "gate": gate, **_summary(runs)}
        delta = results[key]["final_delta_mean_points"]
        print(f"[sim] {key:34} Δ chính xác {delta['accuracy']:+.2f} · macro {delta['macro_recall']:+.2f} · PlantDoc {delta['plantdoc_accuracy']:+.2f} điểm"
              f" · nhận {results[key]['accepted_round_share']:.0%} vòng  ({time.time() - started:.0f}s)", flush=True)
    if args.quick:
        print("[sim] --quick: không ghi báo cáo")
        return 0

    report = {
        "evidence": ("mô phỏng bằng đúng chính sách máy chủ và SGD của trình duyệt; kho người tham gia = calibration (không giao "
                     "train/validation); số báo cáo trên test giữ riêng; cổng chạy trên tập giám sát validation như production"),
        "policy": policy.to_dict(),
        "local_config": {"epochs": config.epochs, "lr": config.lr, "batch": config.batch, "holdout_fraction": config.holdout_fraction,
                         "images_per_user": [config.n_min_images, config.n_max_images]},
        "seeds": list(seeds), "rounds": rounds, "k_values": list(ks),
        "scenarios": {
            "iid": "mỗi người lấy ảnh bất kỳ trong kho",
            "three_groups": "mỗi người chỉ có ảnh của 3 nhóm bệnh ngẫu nhiên",
            "one_group": "mỗi người chỉ có ảnh của 1 nhóm bệnh",
            "plantdoc_only": "mỗi người chỉ có ảnh chụp ngoài đồng (PlantDoc)",
        },
        "base_test": {m: base_test[m] for m in (*METRICS, "count", "plantdoc_count", "plantvillage_count")},
        "pool": {"images": int(len(pool.labels)), "classes": len(pool.classes), "plantdoc": int(len(pool.plantdoc))},
        "test": {"images": int(len(test_set.labels)), "manifest_sha256": test_manifest_sha},
        "features_fingerprint": features_fingerprint,
        "runs": results,
        "caveat": ("Mô phỏng, không phải người dùng thật: ảnh của người mô phỏng là ảnh của đồ án. Test giữ riêng chỉ dùng để báo cáo. "
                   "PlantDoc test chỉ 172 ảnh nên số PlantDoc dao động mạnh giữa các seed."),
        "report_sha256_of_runs": hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest(),
    }
    out = reg.abs_path(REPORT)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[sim] wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
