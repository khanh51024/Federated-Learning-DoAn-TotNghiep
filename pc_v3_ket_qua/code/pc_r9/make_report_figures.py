"""Report figures F1-F7 (TU VAN PC 2026-11-10). Reads only files inside <pc_v3_ket_qua>/results (histories/per_round.csv, histories/payload_bytes.csv, e1/E1_per_client.csv,
e5/E5_RESULT.json, e2_e3/E3_RESULT.json, p6/P6_RESULT.json) and writes results/figures/F<k>_*.png (150 dpi) + the source CSV of every figure.
No model is run. Mean +- SD (ddof 1) over seeds everywhere; a figure shows a band or error bars, never a bare mean. Validation numbers unless stated (F7 = P6 test).
Usage: python -I -B make_report_figures.py <pc_v3_ket_qua dir>
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(sys.argv[1]) / "results"
OUT = R / "figures"
assert not OUT.exists(), "create-once"
OUT.mkdir()
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

rows = list(csv.DictReader(open(R / "histories/per_round.csv", encoding="utf-8")))
for r in rows:
    r["round"] = int(r["round"])
    for k in ("val_macro_f1_raw", "val_accuracy", "plantdoc_macro_f1"):
        r[k] = float(r[k]) if r[k] != "" else None
    r["alpha"] = float(r["alpha"]) if r["alpha"] != "" else None

runs = defaultdict(list)
for r in rows:
    runs[(r["run"], r["client"])].append(r)


def endpoint(rs, key):
    mx = max(x["round"] for x in rs)
    v = [x[key] for x in rs if x["round"] >= mx - 2 and x[key] is not None]
    return st.mean(v)


def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def write_csv(name, header, data):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


def fam_runs(method, alpha=None, family=None, scheme=""):
    out = {}
    for (run, cl), rs in runs.items():
        r0 = rs[0]
        if r0["method"] == method and r0["alpha"] == alpha and r0["scheme"] == scheme and (family is None or r0["family"] == family):
            out.setdefault(run, []).append(rs)
    return out


def ep_by_seed(method, alpha, key="val_macro_f1_raw", family=None):
    """seed -> endpoint (clients of local-only averaged first)."""
    d = {}
    for run, lst in fam_runs(method, alpha, family).items():
        d[lst[0][0]["seed"]] = st.mean(endpoint(rs, key) for rs in lst)
    return d


ALPHAS = [0.1, 0.5, 1.0, 10.0, 100.0]
# central
cen = {}
for opt in ("adamw", "sgd"):
    d = {}
    for (run, cl), rs in runs.items():
        if rs[0]["method"] == f"central_{opt}":
            d[rs[0]["seed"]] = endpoint(rs, "val_macro_f1_raw")
    cen[opt] = ms(list(d.values()))

# ---------------- F1
METH = [("fedavg", "FedAvg", "o-", "tab:blue"), ("scaffold", "SCAFFOLD", "s", "tab:red"), ("fedprox", "FedProx", "^", "tab:green"), ("fedbn", "FedBN (mô hình server)", "v", "tab:purple"),
        ("moon", "MOON", "D", "tab:brown"), ("local_only", "Local-only (TB 5 client)", "x--", "tab:gray")]
fig, (ax, axl) = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
data = []
for opt, col, lab in (("adamw", "black", "Tập trung AdamW"), ("sgd", "dimgray", "Tập trung SGD")):
    m, s = cen[opt]
    ax.axhline(m, color=col, lw=1, ls="-" if opt == "adamw" else ":", label=lab + " (n=5)")
    ax.axhspan(m - s, m + s, color=col, alpha=0.10)
    data.append([lab, "", m, s, 5])
for meth, lab, sty, col in METH:
    xs, ms_, ss, ns = [], [], [], []
    for a in ALPHAS:
        d = ep_by_seed(meth, a, family="local_only" if meth == "local_only" else None)
        if len(d) >= 3:
            m, s = ms(list(d.values()))
            xs.append(a); ms_.append(m); ss.append(s); ns.append(len(d))
            data.append([lab, a, m, s, len(d)])
    if xs:
        tgt = axl if meth == "local_only" else ax
        tgt.errorbar(xs, ms_, yerr=ss, marker={"fedavg": "o", "scaffold": "s", "fedprox": "^", "fedbn": "v", "moon": "D", "local_only": "x"}[meth], color=col, label=lab + (f" (n={min(ns)})" if min(ns) == max(ns) else f" (n={min(ns)}–{max(ns)})"), capsize=2, lw=1.2, ms=4,
                     linestyle="-" if meth in ("fedavg", "local_only") else "none")
ax.set_xscale("log")
axl.set_xlabel("α (Dirichlet, nhỏ = lệch nhãn mạnh)")
ax.set_ylabel("macro-F1 validation\n(TB 3 vòng/epoch cuối)")
axl.set_ylabel("Local-only")
ax.set_title("F1. Macro-F1 validation theo mức non-IID (TB ± SD qua seed)")
ax.legend(fontsize=7, loc="lower right")
axl.legend(fontsize=7, loc="upper left")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.text(0.01, 0.01, "n = số seed (S5 gộp seed 42–46 ở α 0,1 và 0,5). Giá trị chọn giải pháp (G, 3 seed 42–44) xem results/noniid_deploy_select_20261109.json; F1 là mô tả. FedBN ở α=100 là R-DIAG (runner q18).", fontsize=6, ha="left")
fig.savefig(OUT / "F1_macroF1_theo_alpha.png"); plt.close(fig)
write_csv("F1_macroF1_theo_alpha.csv", ["nhom", "alpha", "mean", "sd", "n_seed"], data)

# ---------------- F2
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
data = []
for ax, a in zip(axs, (0.1, 0.5)):
    for meth, lab, sty, col in METH[:5]:
        by = defaultdict(list)
        for run, lst in fam_runs(meth, a).items():
            for x in lst[0]:
                by[x["round"]].append(x["val_macro_f1_raw"])
        if not by:
            continue
        rr = sorted(by)
        m = [st.mean(by[r]) for r in rr]; s = [st.stdev(by[r]) if len(by[r]) > 1 else 0 for r in rr]
        ax.plot(rr, m, color=col, label=f"{lab} (n={len(by[rr[0]])})", lw=1.1)
        ax.fill_between(rr, [a_ - b for a_, b in zip(m, s)], [a_ + b for a_, b in zip(m, s)], color=col, alpha=0.15)
        data += [[lab, a, r, m_, s_, len(by[r])] for r, m_, s_ in zip(rr, m, s)]
    ax.set_title(f"α = {a}"); ax.set_xlabel("vòng")
    ax.set_ylim(0.5, 1.0)
axs[0].set_ylabel("macro-F1 validation"); axs[0].legend(fontsize=7, loc="lower right")
fig.suptitle("F2. Đường hội tụ theo vòng (TB ± SD qua seed)")
fig.tight_layout(); fig.savefig(OUT / "F2_hoi_tu_theo_vong.png"); plt.close(fig)
write_csv("F2_hoi_tu_theo_vong.csv", ["phuong_phap", "alpha", "vong", "mean", "sd", "n_seed"], data)

# ---------------- F3
fig, ax = plt.subplots(figsize=(6, 3.8))
data = []
fa = {a: ms(list(ep_by_seed("fedavg", a).values())) for a in ALPHAS}
for opt, col, lab in (("adamw", "black", "Tập trung AdamW − FedAvg"), ("sgd", "dimgray", "Tập trung SGD − FedAvg")):
    g = [cen[opt][0] - fa[a][0] for a in ALPHAS]
    e = [(cen[opt][1] ** 2 + fa[a][1] ** 2) ** 0.5 for a in ALPHAS]
    ax.errorbar(ALPHAS, [100 * x for x in g], yerr=[100 * x for x in e], color=col, marker="o", capsize=2, label=lab, ls="-" if opt == "adamw" else ":")
    data += [[lab, a, gg, ee] for a, gg, ee in zip(ALPHAS, g, e)]
ax.set_xscale("log"); ax.set_xlabel("α"); ax.set_ylabel("khoảng cách macro-F1 (điểm %)")
ax.set_title("F3. Khoảng cách Tập trung − FedAvg theo α\n(SD = căn(SD_tập trung² + SD_FedAvg²), hai nhóm độc lập)")
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "F3_khoang_cach_theo_alpha.png"); plt.close(fig)
write_csv("F3_khoang_cach_theo_alpha.csv", ["nhom", "alpha", "gap", "sd"], data)

# ---------------- F4
comm = json.load(open(R / "e2_e3/E3_RESULT.json", encoding="utf-8"))["E3_2_comm"]
raw_mb = comm["raw_train_images"]["MB"]
pb = list(csv.DictReader(open(R / "histories/payload_bytes.csv", encoding="utf-8")))
def mb_round(run):
    x = next(r for r in pb if r["run"] == run)
    return 5 * (int(x["payload_down_bytes"]) + int(x["payload_up_bytes"])) / 1e6
MBR = {"fedavg": mb_round("fedavg_a01_s42"), "fedavg_fp16_uplink": mb_round("fedavg_fp16_uplink_a01_s42"), "scaffold": comm["scaffold_per_run_40_rounds"]["MB"] / 40}
fig, ax = plt.subplots(figsize=(6.6, 4.2))
data = []
for meth, lab, col in (("fedavg", "FedAvg", "tab:blue"), ("fedavg_fp16_uplink", "FedAvg + fp16 uplink", "tab:orange"), ("scaffold", "SCAFFOLD", "tab:red")):
    for a in ((0.1, 0.5) if meth != "fedavg_fp16_uplink" else (0.1,)):
        pts = {"R95": [], "R99": []}; accs = []
        for run, lst in fam_runs(meth, a).items():
            rs = lst[0]; e = endpoint(rs, "val_macro_f1_raw")
            by = {x["round"]: x["val_macro_f1_raw"] for x in rs}
            for tag, f in (("R95", 0.95), ("R99", 0.99)):
                hit = [r for r in sorted(by) if by[r] >= f * e]
                if hit:
                    pts[tag].append(hit[0] * MBR[meth])
            accs.append(endpoint(rs, "val_accuracy"))
        ya, _ = ms(accs)
        for tag, mk in (("R95", "o"), ("R99", "s")):
            if len(pts[tag]) >= 2:
                xm, xs = ms(pts[tag])
                ax.errorbar([xm], [ya], xerr=[xs], fmt=mk, color=col, capsize=2, ms=5, label=f"{lab}, α={a}, {tag}" if True else None, alpha=1.0 if a == 0.1 else 0.55)
                data.append([lab, a, tag, xm, xs, ya, len(pts[tag]), MBR[meth]])
ax.axvline(raw_mb, color="k", ls="--", lw=1)
ax.text(raw_mb, ax.get_ylim()[1], f" ảnh thô {raw_mb:.0f} MB", rotation=90, va="top", ha="left", fontsize=7)
ax.set_xlabel("tổng MB truyền (5 client, hai chiều) tới khi đạt R95 / R99 (TB ± SD)")
ax.set_ylabel("accuracy validation (TB 3 vòng cuối)")
ax.set_title("F4. Truyền thông tới R95 / R99 và accuracy\n(số liệu mô phỏng trong một tiến trình, không phải đo mạng)")
ax.legend(fontsize=6, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "F4_truyen_thong.png"); plt.close(fig)
write_csv("F4_truyen_thong.csv", ["phuong_phap", "alpha", "moc", "MB_tb", "MB_sd", "accuracy_endpoint_tb", "n_seed", "MB_moi_vong_5_client"], data)

# ---------------- F5
e5 = json.load(open(R / "e5/E5_RESULT.json", encoding="utf-8"))["models"]
names = ["centralized_adamw", "centralized_sgd", "fedavg", "scaffold"]
def lat1(lst):
    return next(x["median_ms"] for x in lst if x["threads"] == 1)
data = []
for n in names:
    m = e5[n]
    data.append([n, m["fp32"]["torchscript_file_bytes"] / 1e6, m["dynamic_linear"]["torchscript_file_bytes"] / 1e6, lat1(m["fp32"]["latency_torchscript"]), lat1(m["dynamic_linear"]["latency_torchscript"]),
                 m["dynamic_linear"]["delta_minus_fp32"]["accuracy"], m["dynamic_linear"]["delta_minus_fp32"]["macro_f1"]])
write_csv("F5_int8_so_fp32.csv", ["mo_hinh", "MB_fp32", "MB_int8", "ms_fp32_1thread", "ms_int8_1thread", "delta_accuracy", "delta_macroF1"], data)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
x = range(len(names))
axs[0].bar([i - 0.2 for i in x], [d[1] for d in data], 0.4, label="fp32"); axs[0].bar([i + 0.2 for i in x], [d[2] for d in data], 0.4, label="int8 (Linear)")
axs[0].set_ylabel("MB (TorchScript)"); axs[0].set_title("Kích thước")
axs[1].bar([i - 0.2 for i in x], [d[3] for d in data], 0.4, label="fp32"); axs[1].bar([i + 0.2 for i in x], [d[4] for d in data], 0.4, label="int8 (Linear)")
axs[1].set_ylabel("ms / ảnh (CPU, 1 luồng, trung vị)"); axs[1].set_title("Độ trễ")
axs[2].bar(list(x), [d[5] for d in data], 0.5, color="tab:green"); axs[2].axhline(0, color="k", lw=0.8)
axs[2].set_ylabel("Δ accuracy (int8 − fp32), validation"); axs[2].set_title("Chênh accuracy")
for a in axs:
    a.set_xticks(list(x)); a.set_xticklabels(["AdamW", "SGD", "FedAvg", "SCAFFOLD"], fontsize=7)
axs[0].legend(fontsize=7, loc="lower center")
axs[0].set_ylim(0, 8)
fig.suptitle("F5. int8 động (chỉ lớp Linear) so với fp32; 4 mô hình, một seed (42), chưa có SD")
fig.tight_layout(); fig.savefig(OUT / "F5_int8_so_fp32.png"); plt.close(fig)

# ---------------- F6
pc = list(csv.DictReader(open(R / "e1/E1_per_client.csv", encoding="utf-8")))
fig, ax = plt.subplots(figsize=(6.4, 4))
data = []
for cl in sorted({r["client"] for r in pc}):
    xs, m_, s_ = [], [], []
    for a in ALPHAS:
        v = [float(r["acc_k"]) for r in pc if r["method"] == "fedavg" and float(r["alpha"]) == a and r["client"] == cl and r["valid"] == "True"]
        if len(v) >= 2:
            mm, ss = ms(v); xs.append(a); m_.append(mm); s_.append(ss); data.append([cl, a, mm, ss, len(v)])
    ax.errorbar(xs, m_, yerr=s_, marker="o", capsize=2, lw=1, ms=3, label=cl)
ax.set_xscale("log"); ax.set_xlabel("α"); ax.set_ylabel("accuracy theo client (tập mô phỏng\nphân phối từng client)")
ax.set_title("F6. Công bằng giữa client (E1), FedAvg, TB ± SD qua seed"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "F6_cong_bang_client.png"); plt.close(fig)
write_csv("F6_cong_bang_client.csv", ["client", "alpha", "acc_tb", "acc_sd", "n_seed"], data)

# ---------------- F7
g = json.load(open(R / "p6/P6_RESULT.json", encoding="utf-8"))["analysis"]["groups_mean_sd"]
GR = [("central|adamw|None", "Tập trung AdamW"), ("central|sgd|None", "Tập trung SGD"), ("fl|fedavg|0.1", "FedAvg α=0,1"), ("fl|fedavg|0.5", "FedAvg α=0,5"), ("fl|fedavg|100.0", "FedAvg α=100"),
      ("fl|scaffold|0.1", "SCAFFOLD α=0,1"), ("fl|scaffold|0.5", "SCAFFOLD α=0,5")]
SETS = [("PD172", "PD-172 (legacy, đã lộ)"), ("PD463", "PD-463 (legacy, đã lộ)"), ("PVL", "PV-legacy (đã lộ)"), ("EXT", "EXT-FINAL (ngoài miền)")]
fig, ax = plt.subplots(figsize=(9, 4.2))
data = []
w = 0.12
for i, (k, lab) in enumerate(GR):
    ys = [g[k]["sets"][s]["accuracy"]["mean"] for s, _ in SETS]
    es = [g[k]["sets"][s]["accuracy"]["sd"] or 0 for s, _ in SETS]
    ax.bar([j + (i - 3) * w for j in range(4)], ys, w, yerr=es, capsize=1.5, label=f"{lab} (n={len(g[k]['members'])})")
    data += [[lab, s, y, e, len(g[k]["members"])] for (s, _), y, e in zip(SETS, ys, es)]
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in SETS], fontsize=8)
ax.set_ylabel("accuracy trên test (TB ± SD qua seed)")
ax.set_title("F7. Test cuối P6 (chạy một lần): EXT-FINAL là ảnh hiện trường ngoài miền, 7 lớp; các tập legacy đã dùng 3 lần")
ax.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); fig.savefig(OUT / "F7_P6_theo_tap.png"); plt.close(fig)
write_csv("F7_P6_theo_tap.csv", ["nhom", "tap", "accuracy_tb", "accuracy_sd", "n_seed"], data)
print("figures", sorted(p.name for p in OUT.glob("*.png")))
