"""Builder of P6_CHECKPOINTS_PC.json for P6_FINAL_TEST_SPEC r2 (pc_r9/P6_SPEC_r2_PC.md). Create-once. No evaluation, no test file read.
Every entry: name, family, platform PC, pc_path (relative to <WORKSPACE_ROOT>), selection, sha256 (full), format, validity evidence.
FL runs: <dir>/<run>_out/<run>/run/checkpoint_last.pt, validity = RESULT.json valid is True (else excluded and listed).
Central: run/checkpoint_last.pt, validity = summary.json status in OK and 35 (AdamW) / 40 (SGD) history epochs.
Local-only: run/client_0k/checkpoint_last.pt for k 0-4, validity = run/summary.json present + ledger exit valid True.
Deploy PC: r3data_pc / bg_pc / r3c_pc checkpoint_best.pt (QC), as in r1 s2.3. E5 int8: dynamic_linear TorchScript of the 4 web models.
R-DIAG: included only if its RESULT.json exists and is valid when this builder runs (descriptive family).
Usage: python -I -B build_p6_list.py <out_json>
"""
import hashlib
import json
import sys
from pathlib import Path

WS = Path(r"<WORKSPACE_ROOT>")
E = WS / "experiments/train-improve-20261015"
OK = ("MAX_BUDGET_REACHED", "CONVERGED_EARLY", "QUALITY_GATE_FAILED")
SEEDS3, SEEDS5 = (42, 43, 44), (42, 43, 44, 45, 46)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ledger_valid():
    v = {}
    for l in open(E / "pc_runs_ledger.jsonl", encoding="utf-8"):
        e = json.loads(l)
        if e.get("valid") is True and e.get("run"):
            v[e["run"]] = True
    return v


ENTRIES, EXCLUDED = [], []


def add(name, family, path, selection, evidence, valid, fmt="state_dict_model_state", **meta):
    path = Path(path)
    if not valid or not path.is_file():
        EXCLUDED.append({"name": name, "family": family, "pc_path": path.relative_to(WS).as_posix(), "reason": evidence if path.is_file() else "missing"})
        return
    ENTRIES.append({"name": name, "family": family, "platform": "PC", "pc_path": path.relative_to(WS).as_posix(), "selection": selection,
                    "sha256": sha(path), "format": fmt, "validity": evidence, **meta})


def fl(name, family, d, run, **meta):
    rd = E / d / f"{run}_out" / run
    res = rd / "RESULT.json"
    ok = res.is_file() and json.load(open(res, encoding="utf-8")).get("valid") is True
    add(name, family, rd / "run/checkpoint_last.pt", "checkpoint_last (round 40)", f"{d}/{run}_out/{run}/RESULT.json valid={ok}", ok, **meta)


def central(name, d, opt, seed):
    rd = E / d / "run"
    s = json.load(open(rd / "summary.json", encoding="utf-8"))
    n = len(s["history"])
    ok = s.get("status") in OK and n == (35 if opt == "adamw" else 40)
    add(name, "central_p40", rd / "checkpoint_last.pt", f"checkpoint_last (epoch {n})", f"summary status={s.get('status')} epochs={n}", ok,
        optimizer=opt, seed=seed)


def main():
    out = Path(sys.argv[1])
    assert not out.exists(), out
    lv = ledger_valid()
    for s in SEEDS5:
        central(f"pc_central_adamw_s{s}", "r3l/ctrl_out" if s == 42 else f"n11l/adamw_s{s}_out", "adamw", s)
        central(f"pc_central_sgd_s{s}", f"n11l/sgd_s{s}_out", "sgd", s)
    for s in SEEDS3:
        fl(f"pc_fedavg_a01_s{s}", "fl_alpha_curve", "q15l", f"fedavg_s{s}", alpha=0.1, method="fedavg", seed=s)
        fl(f"pc_fedavg_a05_s{s}", "fl_alpha_curve", "q16l", f"fedavg_a05_s{s}", alpha=0.5, method="fedavg", seed=s)
        for a, tag in ((1.0, "1"), (10.0, "10"), (100.0, "100")):
            fl(f"pc_fedavg_a{tag}_s{s}", "fl_alpha_curve", "q17l", f"fedavg_a{tag}_s{s}", alpha=a, method="fedavg", seed=s)
        fl(f"pc_scaffold_a01_s{s}", "fl_methods", "q15l", f"scaffold_s{s}", alpha=0.1, method="scaffold", seed=s)
        fl(f"pc_scaffold_a05_s{s}", "fl_methods", "q16l", f"scaffold_a05_s{s}", alpha=0.5, method="scaffold", seed=s)
        for m in ("fedprox", "moon", "fedbn"):
            for a, tag in ((0.1, "a01"), (0.5, "a05")):
                fl(f"pc_{m}_{tag}_s{s}", "fl_methods", "q19l", f"c_m_{m}_{tag}_s{s}", alpha=a, method=m, seed=s,
                   **({"note": "server model (averaged BN); personalised local-BN protocol per PHASE_C_SPEC_PC"} if m == "fedbn" else {}))
        for a, tag in ((0.1, "a01"), (0.5, "a05")):
            fl(f"pc_scaffold_bn_{tag}_s{s}", "fl_methods_a1", "q22l", f"a1_scaffold_bn_{tag}_s{s}", alpha=a, method="scaffold_bn", seed=s,
               note="server model; proposed by the group (A1 not shown as winner)")
        fl(f"pc_fedavg_fp16_a01_s{s}", "fl_a2_fp16", "q22l", f"a2_fedavg_fp16_a01_s{s}", alpha=0.1, method="fedavg_fp16_uplink", seed=s)
        for q in ("qs", "ss"):
            fl(f"pc_fedavg_{q}_s{s}", "fl_phase_d", "q20l", f"fedavg_{q}_s{s}", scheme=q, method="fedavg", seed=s)
        fl(f"pc_rdiag_fedavg_pdnat_a100_s{s}", "rdiag_descriptive", "rdiagl", f"rd_fedavg_pdnat_a100_s{s}", alpha=100.0, method="fedavg_pd_natural", seed=s)
        fl(f"pc_rdiag_fedbn_a100_s{s}", "rdiag_descriptive", "rdiagl", f"rd_fedbn_a100_s{s}", alpha=100.0, method="fedbn", seed=s,
           note="runner q18 vs baseline runner q17")
    for s in (45, 46):
        for m in ("fedavg", "scaffold", "fedprox", "moon", "fedbn"):
            for a, tag in ((0.1, "a01"), (0.5, "a05")):
                fl(f"pc_{m}_{tag}_s{s}", "fl_methods" if m != "fedavg" else "fl_s5_fedavg", "q21l", f"s5_{m}_{tag}_s{s}", alpha=a, method=m, seed=s)
    for s in (41,):
        for m in ("fedavg", "scaffold", "p1", "p2", "p3"):
            fl(f"pc_d4_{m}_s41", "d4_descriptive", "q18l", f"d4_{m}_s41", alpha=0.1, method=m, seed=41)
    for tag in ("01", "05", "1", "10", "100"):
        for s in SEEDS3:
            run = f"pc_lo17_lo_a{tag}_s{s}"
            for k in range(5):
                add(f"pc_local_a{tag}_s{s}_client_{k:02d}", "local_only", E / f"lo17l/lo_a{tag}_s{s}_out/run/client_{k:02d}/checkpoint_last.pt",
                    "checkpoint_last (epoch 40)", f"ledger {run} valid={lv.get(run, False)}", lv.get(run, False), seed=s, client=k)
    for q in ("qs", "ss"):
        for s in SEEDS3:
            run = f"pc_lo20_lo_{q}_s{s}"
            for k in range(5):
                add(f"pc_local_{q}_s{s}_client_{k:02d}", "local_only_phase_d", E / f"lo20l/lo_{q}_s{s}_out/run/client_{k:02d}/checkpoint_last.pt",
                    "checkpoint_last (epoch 40)", f"ledger {run} valid={lv.get(run, False)}", lv.get(run, False), seed=s, client=k)
    for name, d, note in (("pc_deploy_r3data", "r3l/data_out", "EXT-SEL winner a274bb9e"), ("pc_deploy_bg", "r3xl/bg_out", ""),
                          ("pc_deploy_r3c", "r3xl/r3c_out", "guardrail FAIL: evaluated, not selectable")):
        add(name, "deploy_candidate", E / d / "run/checkpoint_best.pt", "checkpoint_best (QC)", "r1 s2.3 list", True, note=note)
    for role in ("centralized_adamw", "centralized_sgd", "fedavg", "scaffold"):
        add(f"pc_int8_dynamic_{role}", "e5_int8_dynamic", E / f"e5/{role}/model_dynamic_linear_torchscript.pt", "E5 dynamic_linear",
            "E5_RESULT d8f983d3, web rule (a) met", True, fmt="torchscript_int8_dynamic_linear",
            note="int8 dong chi lop Linear; conv van fp32")
    r3 = [x for x in ENTRIES if x["name"] == "pc_deploy_r3data"]
    assert r3 and r3[0]["sha256"].startswith("a274bb9e"), "r3data_pc sha"
    doc = {"stage": "P6 r2 PC checkpoint list (pc_r9/P6_SPEC_r2_PC.md)", "builder_sha256": sha(Path(__file__)), "root_on_pc": str(WS),
           "n_entries": len(ENTRIES), "checkpoints": ENTRIES, "excluded": EXCLUDED}
    s = json.dumps(doc, ensure_ascii=False, indent=1)
    out.with_suffix(".tmp").write_text(s, encoding="utf-8")
    out.with_suffix(".tmp").rename(out)
    print(json.dumps({"n": len(ENTRIES), "excluded": EXCLUDED, "sha256": sha(out)})[:3000])


if __name__ == "__main__":
    main()
