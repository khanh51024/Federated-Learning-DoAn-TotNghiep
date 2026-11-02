"""Generate the nine-question report from measured artifacts, without sealed evaluation."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_distributed_run import audit_run
from src import registry as reg

SEEDS = (42, 123, 2026)
SCHEMES = ("label_alpha_100_0", "label_alpha_10_0", "label_alpha_1_0", "label_alpha_0_5")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |", "|" + "---|" * len(headers),
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def number(value, digits=2):
    return "chưa đo" if value is None else f"{value:.{digits}f}"


def spread(value):
    if not value or value.get("mean") is None:
        return "chưa đo"
    return f"{100 * value['mean']:.2f} ± {100 * (value.get('std') or 0):.2f}"


def source_link(path, symbol):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol)
    return f"[{path}:{node.lineno}](../{path}#L{node.lineno})"


def snippet(path, marker, count):
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if marker in line)
    return f"[{path}:{start+1}](../{path}#L{start+1})\n\n```python\n" + "\n".join(lines[start:start+count]) + "\n```\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/v4_sweep")
    parser.add_argument("--network-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/KIEM_CHUNG_PHAN_TAN_TAP_TRUNG.md")
    args = parser.parse_args()
    args.network_summary = args.network_summary.resolve()
    evidence_dir = ROOT / reg.DISTRIBUTED_VERIFICATION_DIR
    expected = [(f"fedavg_s{seed}_{scheme}_flower", "flower", scheme, seed)
                for scheme in ("label_alpha_0_1", "iid") for seed in SEEDS]
    expected += [(f"fedavg_s{seed}_{scheme}", "simulation", scheme, seed) for seed in SEEDS for scheme in SCHEMES]
    audits = [audit_run(args.runs / name) for name, _, _, _ in expected]
    if not all(a["passed"] for a in audits):
        raise SystemExit("Not all 18 runs passed: " + "; ".join(a["run"] for a in audits if not a["passed"]))
    for (_, runtime, scheme, seed), audit in zip(expected, audits):
        if (audit['runtime'], audit['scheme'], audit['seed']) != (runtime, scheme, seed):
            raise SystemExit(f"Run identity does not match the planned matrix: {audit['run']}")
    (evidence_dir / "batch_audit.json").write_text(json.dumps({"passed": True, "runs": audits}, indent=2) + "\n", encoding="utf-8")
    sweep = read(ROOT / reg.SWEEP_V4_DIR / "results.json")
    comparisons = read(evidence_dir / "weight_comparison.json")
    planned_names = {name for name, _, _, _ in expected}
    measured_names = {name for cell in sweep['cells'] for name in cell['runs']}
    compared_names = {Path(item['run']).name for item in comparisons['comparisons']}
    if not planned_names <= measured_names or not planned_names <= compared_names:
        raise SystemExit('Sweep or weight comparison is missing planned runs')
    equivalence = read(ROOT / reg.FEDSGD_EQUIVALENCE_JSON)
    preflight = read(evidence_dir / "preflight_passed.json")
    environment = read(evidence_dir / "environment.json")
    baseline_config = read(ROOT / 'runs/v4b/fedavg_s42/config.json')
    flower_config = read(args.runs / expected[0][0] / 'config.json')
    network = read(args.network_summary)
    net = network["runs"]["wan"]
    parity_rows, outside_seed_scale = [], []
    cells = {(c['scheme'], c['variant']): c for c in sweep['cells']}
    for candidate in sweep['cells']:
        if candidate['variant'] != 'fedavg+flower':
            continue
        reference = cells.get((candidate['scheme'], 'fedavg'))
        for part in ('all_test', 'plantvillage_test', 'field_same_source_test', 'plantdoc_holdout_v3'):
            comparison = candidate['parts'][part].get('vs_fedavg')
            if reference is None or comparison is None:
                parity_rows.append([candidate['scheme'], part, 'chưa có mốc mô phỏng cùng cách chia', '—', '—'])
                continue
            interval = comparison.get('ci95_points')
            seed_scale = 100 * max(reference['parts'][part]['top1']['std'], candidate['parts'][part]['top1']['std'])
            if abs(comparison['diff_points']) > seed_scale:
                outside_seed_scale.append(f"{candidate['scheme']}/{part}")
            parity_rows.append([candidate['scheme'], part, number(comparison['diff_points']),
                                'chưa tính' if interval is None else f"[{interval[0]:.2f}; {interval[1]:.2f}]",
                                spread(reference['parts'][part]['top1'])])
    round_rows, client_rows = [], []
    for audit in audits:
        by_step = {}
        for line in (args.runs / audit["run"] / "history.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_step[row["step"]] = row
        for step, row in sorted(by_step.items()):
            train = row["train"]
            round_rows.append({"run": audit["run"], "round": step + 1, "runtime": audit["runtime"], "lr": row.get("lr"),
                               "val_top1": row["val"].get("top1"), "val_selection_score": row["val"].get("selection_score"),
                               "aggregation_max_abs_diff": train["aggregation_check"]["max_abs_diff"],
                               "round_wall_seconds": train.get("round_wall_seconds"), "seconds_including_val": row["seconds"]})
            for client in train["clients"]:
                client_rows.append({"run": audit["run"], "round": step + 1, "runtime": audit["runtime"], **client})
    for name, rows in (("round_metrics.csv", round_rows), ("client_metrics.csv", client_rows)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (evidence_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    flower = [a for a in audits if a["runtime"] == "flower"]
    scaled_audits = [a for a in audits if a['aggregation_scaled_rounds']]
    maximum_scaled_units = max((a['aggregation_max_scaled_roundoff_units'] for a in scaled_audits), default=None)
    sections = ["# Kiểm chứng huấn luyện phân tán và tập trung\n",
                "Báo cáo này trả lời chín yêu cầu bằng checkpoint, nhật ký và phép đo thật. "
                "Các run chất lượng dùng năm client mô phỏng Flower/Ray trên một GPU; phép đo mạng dùng hai client thật trên hai máy. "
                "Không đánh giá PlantWild cho run mới và không mở độ chính xác tập niêm phong.\n",
                "## 1. Trọng số có sai lệch khi truyền không\n",
                f"Đối chiếu {sum(a['transport_comparisons'] for a in flower)} lượt SHA-256 hai chiều trên "
                f"{len(flower)} run Flower, mỗi run {flower[0]['rounds']} vòng và năm client: tất cả đạt. "
                "Client băm đúng trọng số nhận/gửi, server băm trọng số gửi/nhận; bất kỳ lệch nào làm run dừng. "
                f"Preflight riêng đạt {preflight['run']['transport_comparisons']} lượt. "
                "Byte Flower là payload mảng đã tuần tự hóa, không gồm toàn bộ khung truyền mạng.\n",
                table(["Run", "Lượt hash", "Gửi lên MB", "Gửi xuống MB", "Sai lệch gộp lớn nhất"],
                      [[a['run'], a['transport_comparisons'], number(a['bytes_up']/1e6), number(a['bytes_down']/1e6), f"{a['aggregation_max_abs_diff']:.3e}"] for a in flower]),
                "\n## 2. Sai lệch so với huấn luyện tập trung\n",
                "Phép A dùng GD toàn batch, không momentum/dropout, BatchNorm đóng băng. Đây là phép kiểm công thức, "
                "không phải toàn bộ công thức SGD minibatch của các run chất lượng. Ngưỡng chốt trước: "
                "max_abs ≤ 1e-10 và L2 tương đối ≤ 1e-9 cho float64.\n",
                table(["Phép", "Cách chia", "Kiểu số", "Bước cục bộ", "L2", "Max abs", "Dự đoán trùng"],
                      [[c['check'], c['scheme'], c['dtype'], c['local_steps'], f"{c['l2']:.3e}", f"{c['max_abs']:.3e}", f"{100*c['prediction_agreement']:.2f}%"] for c in equivalence['checks']]),
                "\nKhoảng cách giữa checkpoint đã train được tách tham số học được và thống kê BatchNorm. "
                "Hash W0 của run mới phải khớp W0 dựng lại theo seed. V4b không ghi hash lúc train: dựng lại công thức "
                "không chứng minh được trạng thái khởi tạo thực tế của lần chạy cũ.\n",
                table(["Run", "L2 tham số", "L2 thống kê BN", "Dự đoán trùng tập thử"],
                      [[Path(c['run']).name, number(c['weight_distance']['learnable:all']['l2'],4), number(c['weight_distance']['bn_stats:all']['l2'],4), f"{100*c['prediction_agreement']['legacy_diagnostic_test']['agreement']:.2f}%"] for c in comparisons['comparisons']]),
                "\n## 3. Phương pháp kiểm tra\n",
                "Kiểm SHA gói trước giải nén, hash release và từng shard; kiểm W0; hash trao đổi từng vòng; "
                "tính lại phép gộp bằng NumPy float64 độc lập; so checkpoint và ghép dự đoán theo sample_id. "
                "Các test cố ý đổi một bit, gộp đều thay vì theo n_k, thiếu client/vòng và NaN phải bị phát hiện. "
                "Phép thử GPU dùng đúng gpu_v2, channels-last, batch 128, lr 0,02, bf16, workers 8, limit 3000, budget 3.\n",
                "Bằng chứng: [preflight](../artifacts/distributed_verification/preflight_passed.json), "
                "[log test](../artifacts/distributed_verification/preflight_tests.log), "
                "[kiểm toán từng run](../artifacts/distributed_verification/batch_audit.json), "
                "[checkpoint](../artifacts/distributed_verification/weight_comparison.json).\n",
                "## 4. Chạy nhiều client\n",
                table(["Cách chia", "Client 0", "Client 1", "Client 2", "Client 3", "Client 4"],
                      [[scheme, *sizes] for scheme, sizes in environment['train_shard_sizes'].items()]),
                "\nNăm shard là phân hoạch dữ liệu train, không phải năm máy vật lý. Phép đo mục 8 dùng hai máy vật lý.\n",
                "## 5. Thông số Flower theo vòng\n",
                f"Môi trường đo: PyTorch {environment['torch']}, Flower {environment['flower']}, Ray {environment['ray']}, {environment['gpu']}. "
                "Mọi run chính dùng 40 vòng, E=1, SGD momentum 0,9, weight decay 1e-4, lr 0,02 cosine, "
                "batch 128, label smoothing 0,1, gpu_v2 và bf16. "
                "[Bảng từng vòng](../artifacts/distributed_verification/round_metrics.csv) chứa lr, val, kiểm gộp và thời gian; "
                "[bảng từng client](../artifacts/distributed_verification/client_metrics.csv) chứa loss, delta_l2, khoảng cách đến mô hình gộp, byte và thời gian. "
                "Val tham gia chọn checkpoint; số chất lượng chính bên dưới lấy trên tập thử.\n",
                "## 6. Mô hình gộp khác mô hình client thế nào\n",
                f"Sai lệch phép gộp lớn nhất qua {sum(a['rounds'] for a in audits)} vòng là "
                f"{max(a['aggregation_max_abs_diff'] for a in audits):.3e}. Sáu run A/B dùng cổng tuyệt đối 1e-6. "
                f"Các run khối C bổ sung đơn vị làm tròn theo tensor; giá trị lớn nhất {number(maximum_scaled_units,6)}, giới hạn 8. "
                "Một đơn vị bằng epsilon của dtype nhân max(1, trị tuyệt đối lớn nhất của tensor tham chiếu và tensor đã gộp). "
                "Đây là chuẩn theo thang tensor, không phải số ULP từng phần tử. "
                "Đây là chênh lệch giữa hai cách tính số học của phép gộp, không phải phép so bit. "
                "Cổng ban đầu 1e-7 đã được sửa thành 1e-6 trong quá trình chạy sau khi gặp đúng epsilon float32 tại thang 1. "
                "Phép tính lại trên năm checkpoint Local-only thật cho thấy thứ tự nhân/chia có thể làm hai kết quả float32 "
                "lệch một ULP; xem [kiểm tra làm tròn](../artifacts/distributed_verification/aggregation_roundoff_analysis.json). "
                "Đây là điều chỉnh ngưỡng kỹ thuật được công khai sau quan sát, không phải ngưỡng đặt trước; "
                "ngưỡng FedSGD A không đổi. Nhật ký chưa ghi ULP từng tensor nên không khẳng định mọi vòng trùng một ULP. "
                "l2_to_merged là khoảng cách thật giữa mô hình client sau học "
                "và mô hình chung sau gộp, không phải lỗi truyền. Tính công bằng được tổng hợp trong "
                "[table_fairness.csv](../artifacts/sweep_v4/table_fairness.csv): dùng độ chính xác mô phỏng phân phối "
                "miền–lớp của shard trên tập đo, có báo độ phủ; không phải tập thử cục bộ mới thu thập ở từng cơ sở.\n",
                table(["Cách chia", "Biến thể", "Accuracy client (%)", "SD giữa client (điểm %)", "Mạnh − yếu (điểm %)", "Client nhỏ nhất (%)", "Độ phủ tối thiểu (%)"],
                      [[c['scheme'], c['variant'], spread(f['mean']), spread(f['std_between_clients']), spread(f['gap_best_worst']),
                        spread(f['smallest_client_accuracy']), number(100*f['coverage_min'])]
                       for c in sweep['cells'] if c.get('fairness')
                       for f in [c['fairness']['legacy_diagnostic_test']['all']]]),
                "## 7. Code backend tập trung và phân tán\n",
                table(["Vai trò", "Mã nguồn"], [
                    ["Chuẩn bị ba mốc và vòng lặp train/val/checkpoint", source_link('scripts/train_v4.py','main')],
                    ["Một vòng mô phỏng một tiến trình", source_link('scripts/train_v4.py','simulate_round')],
                    ["Vòng lặp chung, nhánh Centralized/Local và FedAvg", source_link('scripts/train_v4.py','training_loop')],
                    ["Gửi/nhận Flower và kiểm toàn vẹn", source_link('src/federated/flower_app.py','make_round_fn') + '; ' + source_link('src/federated/flower_app.py','check_transport')],
                    ["Client Flower gọi bộ train dùng chung", source_link('src/federated/flower_app.py','client_step')],
                    ["Huấn luyện một client", source_link('src/training/client_round.py','train_one_client')],
                    ["Gộp trung bình theo n_k", source_link('src/federated/strategy.py','aggregate_fedavg')],
                    ["Server deployment cho phép đo WAN", source_link('src/federated/server_app.py','start_flower_server')],
                    ["Client deployment cho phép đo WAN", '[src/federated/client_app.py](../src/federated/client_app.py)']]),
                "\nNhánh tập trung chuẩn bị toàn bộ tập train, rồi dùng train_epoch trong vòng lặp chung:\n",
                snippet('scripts/train_v4.py', 'if args.method == "centralized":', 7),
                snippet('scripts/train_v4.py', 'for group in optimizer.param_groups:', 6),
                "Đường phân tán gửi trọng số cho từng client, nhận cập nhật và kiểm hash trước gộp:\n",
                snippet('src/federated/flower_app.py', 'def run_round(step:', 16),
                "Phần gộp tích lũy float64 theo số ảnh của client rồi đưa về kiểu số gốc:\n",
                snippet('src/federated/strategy.py', 'n_k = update["num_examples"]', 7),
                "\n## 8. Tốc độ trên hai máy\n",
                network['notes'] + "\n",
                "Đường WAN dùng backend deployment gRPC và shard PlantVillage nhỏ; không thay thế thí nghiệm chất lượng v4 trên Flower/Ray.\n",
                table(["Máy", "Train cục bộ trung vị (s)", "Ảnh thực xử lý/s", "RTT trung vị (ms)", "Lên MiB/s", "Xuống MiB/s"],
                      [[name, number(v['local_train_time_s']['median'],3), number(v['processed_train_images_per_s']['median']), number(v['probe_rtt_persistent_ms']['median'],3), number(v['probe_up_mb_per_s']['median'],3), number(v['probe_down_mb_per_s']['median'],3)] for name,v in net['by_machine'].items()]),
                f"\nBa vòng đều đủ hai client thật, hash đúng và không bị từ chối. Trung vị thời gian fit một vòng: "
                f"{net['round_fit_elapsed_ms']['median']/1000:.3f} s. Đồng hồ đo ở chính máy thực hiện tác vụ. "
                "Thời gian fit có cả probe RTT/băng thông chủ động, không đại diện vòng train không gắn probe. "
                "Không dùng thời gian các đợt v4b/Flower hoặc khối C chạy song song để suy riêng chênh lệch tốc độ runtime. "
                "Ảnh thực xử lý/s tính cả hai epoch; chỉ số cũ train_images_per_s chỉ lấy số ảnh duy nhất trong shard. "
                "Băng thông dùng MiB=2^20 byte; bảng truyền thông sweep dùng MB=10^6 byte. "
                f"[History và summary]({('../' + args.network_summary.parent.relative_to(ROOT).as_posix())}/summary.json).\n",
                "## 9. Khác biệt và giới hạn\n",
                table(["Cách chia", "Biến thể", "Toàn tập thử top-1 (%)", "PlantVillage (%)", "Ngoài đồng cùng nguồn (%)", "PlantDoc (%)"],
                      [[c['scheme'] or 'toàn bộ', c['variant'], *[spread(c['parts'].get(p,{}).get('top1')) for p in ('all_test','plantvillage_test','field_same_source_test','plantdoc_holdout_v3')]] for c in sweep['cells']]),
                "\n![Chất lượng theo mức alpha](../artifacts/sweep_v4/curves_by_alpha.png)\n",
                "![Khoảng cách đến mô hình tập trung](../artifacts/sweep_v4/gap_by_alpha.png)\n",
                "\nSố liệu là trung bình ± độ lệch chuẩn mẫu qua ba seed. Chi tiết macro-F1, khoảng tin cậy, "
                "chênh lệch ghép seed, vòng hội tụ và byte ở [table_main.csv](../artifacts/sweep_v4/table_main.csv) "
                "và [results.json](../artifacts/sweep_v4/results.json). Không dùng việc kiểm định không bác bỏ khác biệt "
                "để khẳng định tương đương. Cần đọc chênh lệch Flower–mô phỏng bên cạnh độ dao động seed.\n",
                table(["Cách chia", "Tập đo", "Flower − mô phỏng (điểm %)", "CI95 theo cụm", "Mốc mô phỏng trung bình ± SD (%)"], parity_rows),
                ("\nChênh lệch lớn hơn SD của cả hai bên tại: " + ', '.join(outside_seed_scale) +
                 ". Vì vậy không kết luận hai pipeline tương đương trên mọi nhóm dữ liệu. Đây là so sánh thang dao động mô tả, không phải kiểm định tương đương.\n")
                if outside_seed_scale else "\nChênh lệch trung bình nằm trong thang SD lớn hơn của hai bên trên các nhóm có đối chứng; đây không phải kiểm định tương đương.\n",
                f"Môi trường V4b: {baseline_config['gpu']}, torch {baseline_config['torch']}, Python {baseline_config['python']}, "
                f"mã {baseline_config['git_commit'][:9]}; Flower: {flower_config['gpu']}, torch {flower_config['torch']}, "
                f"Python {flower_config['python']}, mã {flower_config['git_commit'][:9]}. Dropout của Flower lấy RNG trong tiến trình client. "
                "Không quy riêng mọi chênh lệch quan sát được cho đường truyền hoặc runtime.\n",
                f"Tổng ảnh train để gửi một lần về trung tâm: {number(sweep['raw_images']['mb'])} MB "
                f"cho {sweep['raw_images']['images']} ảnh đã mã hóa.\n",
                table(["Cách chia", "Biến thể", "MB gửi lên", "MB hai chiều", "Hai chiều / gửi ảnh", "Vòng hội tụ seed 42/123/2026"],
                      [[c['scheme'], c['variant'], number(comm['mb_up_total']['mean']), number(comm['mb_total']['mean']),
                        number(comm['mb_total_vs_raw'],3) + '×',
                        '/'.join(str(int(comm['rounds_to_convergence']['per_seed'][str(s)])) if str(s) in comm['rounds_to_convergence']['per_seed'] else 'chưa đạt' for s in SEEDS)]
                       for c in sweep['cells'] if c.get('communication') for comm in [c['communication']]]),
                "\nByte của mô phỏng v4b được suy từ tensor; byte Flower được ghi từ payload mảng, đều chưa gồm mọi khung giao thức. "
                "So sánh MB hai chiều với ảnh gửi một lần không khẳng định FL luôn tiết kiệm băng thông. "
                "Hội tụ ở đây là chỉ số val giữ trong một điểm macro-F1 so với mức tốt nhất đến hết run. "
                "Seed chưa đạt được ghi rõ, không lấy trung bình hai seed rồi trình bày như đủ ba seed. "
                "Chi phí đến hội tụ trong CSV dùng hết ngân sách khi seed chưa đạt; cần đọc cùng cờ này.\n",
                "\nIID chỉ chạy Flower trong đợt này; không có đối chứng IID một tiến trình để suy ra tương đương runtime ở IID. "
                "Đường theo α có năm mức FedAvg và mốc Centralized; Local-only chỉ được đo ở α=0,1, không nội suy thành số đo ở các α còn lại.\n",
                "Khoảng cách production còn chịu tác động của nhiều bước minibatch cục bộ, SGD momentum được làm mới mỗi vòng client, "
                "BatchNorm, dropout/tăng cường ngẫu nhiên và cách chuẩn hóa mất mát theo shard/batch. B/C chứng minh "
                "một số cơ chế tạo lệch, không phân rã định lượng toàn bộ khoảng cách đã train. "
                "Test PlantDoc nhỏ; độ chính xác mô hình toàn cục và độ chính xác theo phân phối client là hai phép đo khác nhau.\n",
                "Trong mô phỏng Ray, các shard nằm cùng máy và dùng chung hệ thống tệp. Việc chỉ truyền trọng số trong "
                "giao thức không chứng minh cô lập dữ liệu giữa các tổ chức; đợt này không kiểm chứng DP hay secure aggregation.\n",
                "## Các lượt đã train và phạm vi còn thiếu\n",
                table(["Run mới", "Seed", "Runtime", "Vòng", "Tổng thời gian vòng (s)"],
                      [[a['run'],a['seed'],a['runtime'],a['rounds'],number(a['round_seconds_sum'])] for a in audits]),
                "\nĐã bổ sung 6 run Flower và 12 run quét α; dùng lại 21 run v4b. Preflight và phép đo mạng "
                "không tính vào 18 run chất lượng. Không train thêm FedProx/FedBN/SCAFFOLD/MOON, DP, Local-only theo α "
                "hoặc cá nhân hóa; không mở tập niêm phong.\n",
                "Theo file Word, đợt này bổ sung bằng chứng cho N1.2, N1.3, N2.3 và demo hai client thật của N4.4. "
                "N2.1/N2.4 còn thiếu Local-only ở các α khác; N2.2 có phân hoạch nhưng chưa train lệch số lượng/nguồn. "
                "Chưa hoàn thành thí nghiệm thuật toán và dò siêu tham số Đợt 3, kiểm thử niêm phong N5.3 hay quyển báo cáo/bản thảo N5.5. "
                "Không suy ra mức 60%, 85% hoặc 100% chỉ từ 18 run này.\n"]
    timing_path = evidence_dir / 'batch_timing.json'
    if timing_path.is_file():
        timing = read(timing_path)
        sections += ["## Thời gian và chi phí có căn cứ\n",
                     f"Batch bắt đầu {timing['started_utc']}, kết thúc {timing['ended_utc']}; "
                     f"thời gian trôi qua {timing['wall_seconds']/3600:.3f} giờ. "
                     f"Theo đơn giá trong bàn giao {timing['price_vnd_per_hour_from_handoff']:,} VND/giờ, "
                     f"chi phí ước tính cho riêng khoảng batch là {timing['estimated_vnd']:.0f} VND. "
                     "Chưa có hóa đơn để xác nhận tiền bị trừ thực tế; số này không gồm thời gian thuê trước batch "
                     "hoặc sau batch. Tổng thời gian vòng ở bảng run có thể chồng lấp ở khối C, không cộng thành thời gian thuê.\n"]
    sections += ["## Đối chiếu từng nhiệm vụ trong file Word\n", table(["Nhiệm vụ", "Bằng chứng và phần còn thiếu"], [
        ['N1.1', 'V4b có Centralized SGD cùng công thức; chưa bổ sung đối chứng AdamW cùng release trong đợt này.'],
        ['N1.2', 'Đã chạy Flower ba seed và đo chênh lệch với mô phỏng ở α=0,1; kết luận theo bảng chênh lệch, không mặc định đạt tương đương.'],
        ['N1.3', 'Run FedAvg mới có nhật ký đủ vòng, hash, delta và thời gian; chưa chạy lại Local-only để bổ sung trường nhật ký cũ còn thiếu.'],
        ['N1.4', 'Giữ thiết kế tập niêm phong hiện có; không mở độ chính xác trong đợt này.'],
        ['N2.1', 'FedAvg đủ năm α và ba seed; Local-only còn thiếu ở các α khác.'],
        ['N2.2', 'Đã có phân hoạch lệch số lượng/nguồn; chưa train hai dạng này trong đợt hiện tại.'],
        ['N2.3', 'Có accuracy mô phỏng phân phối client và độ phủ trong bảng fairness.'],
        ['N2.4', 'Có đường Centralized/FedAvg theo α; Local-only chỉ có điểm α=0,1.'],
        ['N3.1', 'Mã thuật toán có test; chưa có bộ thí nghiệm Flower đầy đủ cho bốn thuật toán.'],
        ['N3.2', 'Chưa dò siêu tham số các thuật toán trong đợt này.'],
        ['N3.3', 'Chưa thực nghiệm cải tiến mới đủ ba seed trong đợt này.'],
        ['N3.4', 'Chưa có so sánh cải tiến để nghiệm thu thu hẹp khoảng cách.'],
        ['N4.1', 'Có artifact lượng tử hóa lịch sử; đợt này không đo lại trên thiết bị đích.'],
        ['N4.2', 'Đã đo payload và hội tụ; chưa train thêm kỹ thuật nén cập nhật để đo đánh đổi.'],
        ['N4.3', 'Không chạy DP theo phạm vi đã chốt; chưa có đường epsilon–accuracy.'],
        ['N4.4', 'Đã chạy hai client thật trên hai máy; server đồng máy client GPU.'],
        ['N5.1', 'Có bảng ba mốc và FedAvg nhiều α; còn thiếu thuật toán cải tiến và Local-only nhiều α.'],
        ['N5.2', 'Có chất lượng, non-IID, truyền thông và fairness; riêng tư mới phân tích giới hạn, không có bảo đảm DP.'],
        ['N5.3', 'Chưa chạy tập niêm phong, đúng ràng buộc của đợt này.'],
        ['N5.4', 'Có script, cấu hình, mã băm, bảng/biểu đồ và hướng dẫn tái lập bên dưới.'],
        ['N5.5', 'Có báo cáo kiểm chứng Markdown; chưa tạo quyển tốt nghiệp Word hoặc bản thảo bài báo.']]),
        "\n## Tái lập\n",
        "Mã train gốc được ghim ở ed88947eb; khối C có thêm trường log làm tròn theo thang tensor, không đổi phép gộp hay optimizer. "
        "Commit và code_fingerprint của từng run nằm trong batch_audit.json; script kiểm chứng và kết quả nằm trên nhánh codex/continue-flower-verification. "
        "Gói đầu vào có manifest SHA riêng, không dùng bundle.json cũ. Trên máy thuê, chạy preflight trong tmux, "
        "sau khi kiểm dữ liệu; chỉ chạy train khi cả preflight_passed.json và network_speedtest_passed.json đạt.\n",
        "```bash\nbash scripts/remote/run_verification_batch.sh preflight /root/cropfed\n"
        "bash scripts/remote/run_verification_batch.sh train /root/cropfed\n```\n",
        "Sau khi kéo đủ run đã kiểm SHA, tại gốc repo:\n",
        "```bash\npython scripts/eval_sweep_v4.py --runs runs/v4b runs/v4_sweep --threads 2\n"
        "python scripts/compare_v4_checkpoints.py --baseline-runs runs/v4b --runs runs/v4b runs/v4_sweep "
        "--out artifacts/distributed_verification/weight_comparison.json\n"
        f"python scripts/report_distributed_verification.py --network-summary {args.network_summary.relative_to(ROOT).as_posix()}\n```\n"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
