"""Builder for R-DIAG (pc_r9/RDIAG_SPEC.md r1). Create-once everywhere, exact anchors (count 1 each).
1. E/rdiag/specs/fedavg_spec_rdiag_pdnat.json = byte copy of q17/specs/fedavg_spec_q17.json (1f8a3958) with ONLY
   "selected_variant": "pretrained_pd25" -> "pretrained" (the runner's READY_FULL gate then expects plantdoc ratio 0 = natural sampling).
2. pc_r9/rdiag_laptop_kaggle.py from pc_r9/q22_laptop_kaggle.py (d40e5c5f): + runner "q17" (q17 sandbox, Q17 lock), RUNS = the 6 R-DIAG
   runs, digest registry + label_alpha_100_0 (c77a0030, the q17 alpha-100 anchors), per-run spec file and plantdoc ratio in fed_cmd
   (default = the q22 values, so nothing else changes). Validity unchanged.
3. pc_r9/rdiag_laptop_driver.py from pc_r9/q22_laptop_driver.py (5dc21278): kaggle pin -> rdiag kaggle, + Q17 lock and R-DIAG lock
   checks, runner key check q17/q18, output rdiagl.
4. E/rdiag/RDIAG_STAGE_LOCK.json: shas of the variant spec, the kaggle, the driver, this builder's sources.
Usage: python -I -B build_rdiag.py
"""
import hashlib
import json
from pathlib import Path

WS = Path(r"<WORKSPACE_ROOT>")
PL = WS / "plans/cai-thien-train-20261015/pc_r9"
E = WS / "experiments/train-improve-20261015"
RD = E / "rdiag"
SPEC_SRC = (E / "q17/specs/fedavg_spec_q17.json", "1f8a3958fa2ab1ff930f8fa83e2fcb4cba8cbbb40d8090727889126fbc957de4")
K_SRC = (PL / "q22_laptop_kaggle.py", "d40e5c5f1873e3d622db7e32011af78658b00768e672734568805db027777f3a")
D_SRC = (PL / "q22_laptop_driver.py", "5dc2127889c435328b2b0c6d06a26ab7a5c2ddfdb25209b97c13911e7313728d")
Q17_LOCK_SHA = "2f71e6ff33064480a361992e396143229980129429934c1701ab1d51e44aff71"
SPEC_OUT = RD / "specs/fedavg_spec_rdiag_pdnat.json"
K_OUT = PL / "rdiag_laptop_kaggle.py"
D_OUT = PL / "rdiag_laptop_driver.py"
LOCK_OUT = RD / "RDIAG_STAGE_LOCK.json"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def edit(s, edits, name):
    for a, c in edits:
        assert s.count(a) == 1, ("anchor", name, a[:90], s.count(a))
        s = s.replace(a, c)
    return s


def read(src):
    b = src[0].read_bytes()
    assert sha(b) == src[1], ("source sha", src[0])
    return b.decode("utf-8")


K_EDITS = [
    ('"""Q22 (A1A2_SPEC r1 5824f9de) built',
     '"""R-DIAG (pc_r9/RDIAG_SPEC.md r1) built by pc_r9/build_rdiag.py from pc_r9/q22_laptop_kaggle.py d40e5c5f with ONLY: + runner "q17",\n'
     'RUNS = the 6 R-DIAG runs, digest + label_alpha_100_0, per-run spec file / plantdoc ratio in fed_cmd. Validity unchanged.\n'
     'Q22 (A1A2_SPEC r1 5824f9de) built'),
    ('RUNS = Q22_RUNS  # Q22: only the 9 A1 / A2 runs are runnable through this file\n',
     'RUNS = {}  # R-DIAG r1: (b) FedAvg alpha 100, PlantDoc natural sampling (q17 runner, variant spec); (c) FedBN alpha 100 (q18 runner)\n'
     'RD_SPEC = {}\n'
     'RD_RATIO = {}\n'
     'for _s in (42, 43, 44):\n'
     '    RUNS[f"rd_fedavg_pdnat_a100_s{_s}"] = (f"fed_mixed_label_a100_p40_s{_s}", "label_alpha_100_0", "q17", ["--fl-method", "fedavg"])\n'
     '    RD_SPEC[f"rd_fedavg_pdnat_a100_s{_s}"] = E / "rdiag/specs/fedavg_spec_rdiag_pdnat.json"\n'
     '    RD_RATIO[f"rd_fedavg_pdnat_a100_s{_s}"] = "none"\n'
     '    RUNS[f"rd_fedbn_a100_s{_s}"] = (f"fed_mixed_label_a100_p40_s{_s}", "label_alpha_100_0", "q18", ["--fl-method", "fedbn"])\n'),
    ('          "label_alpha_0_5": "d5a1e3dfffe96e6a783e8f5b1fb5c9581c680d48cb004e837e95a2a3da9b7fd4"}',
     '          "label_alpha_0_5": "d5a1e3dfffe96e6a783e8f5b1fb5c9581c680d48cb004e837e95a2a3da9b7fd4",\n'
     '          "label_alpha_100_0": "c77a0030d7bec24eae78515f341c1fbf97df947aa8b9fc6e53a2adc68274e7c5"}'),
    ('RUNNER = {"q18": Q18_ROOT / "sandbox", "q22": WS / "experiments/train-improve-20261015/q22/sandbox"}',
     'RUNNER = {"q18": Q18_ROOT / "sandbox", "q22": WS / "experiments/train-improve-20261015/q22/sandbox",\n'
     '          "q17": WS / "experiments/train-improve-20261015/q17/sandbox"}'),
    ('"--spec-file", str(SPEC), "--dataset-root"', '"--spec-file", str(RD_SPEC.get(run, SPEC)), "--dataset-root"'),
    ('"--plantdoc-sampling-ratio", "0.25", "--sampler', '"--plantdoc-sampling-ratio", RD_RATIO.get(run, "0.25"), "--sampler'),
]

D_EDITS = [
    ('"""Q22 PC driver', '"""R-DIAG PC driver (RDIAG_SPEC r1), built by pc_r9/build_rdiag.py from pc_r9/q22_laptop_driver.py 5dc21278 with ONLY:\n'
     'kaggle pin -> pc_r9/rdiag_laptop_kaggle.py, + Q17 lock and R-DIAG lock checks, runner key q17/q18, output rdiagl.\n'
     'Q22 PC driver'),
    ('Q14K = (ROOT / "plans/cai-thien-train-20261015/pc_r9/q22_laptop_kaggle.py", "d40e5c5f1873e3d622db7e32011af78658b00768e672734568805db027777f3a")',
     'Q14K = (ROOT / "plans/cai-thien-train-20261015/pc_r9/rdiag_laptop_kaggle.py", "@KSHA@")\n'
     'Q17_LOCK = (E / "q17/Q17_STAGE_LOCK.json", "' + Q17_LOCK_SHA + '")\n'
     'RD_LOCK = E / "rdiag/RDIAG_STAGE_LOCK.json"'),
    ('RUNS = tuple([f"a1_scaffold_bn_{a}_s{s}" for s in (42, 43, 44) for a in ("a01", "a05")]\n'
     '             + [f"a2_fedavg_fp16_a01_s{s}" for s in (42, 43, 44)])',
     'RUNS = tuple([f"rd_fedavg_pdnat_a100_s{s}" for s in (42, 43, 44)] + [f"rd_fedbn_a100_s{s}" for s in (42, 43, 44)])'),
    ('    assert sha(Q14K[0]) == Q14K[1], "q15_laptop_kaggle sha"\n',
     '    assert sha(Q14K[0]) == Q14K[1], "q15_laptop_kaggle sha"\n'
     '    assert sha(Q17_LOCK[0]) == Q17_LOCK[1], "Q17 lock sha"\n'
     '    lock17 = json.loads(Q17_LOCK[0].read_text(encoding="utf-8"))\n'
     '    bad17 = [k_ for k_, v_ in lock17["files"].items() if sha(E / "q17" / k_) != v_]\n'
     '    assert not bad17, ("Q17 lock mismatch", bad17[:3])\n'
     '    rdl = json.loads(RD_LOCK.read_text(encoding="utf-8"))\n'
     '    badrd = [k_ for k_, v_ in rdl["files"].items() if sha(ROOT / k_) != v_]\n'
     '    assert not badrd and rdl["files"]["plans/cai-thien-train-20261015/pc_r9/rdiag_laptop_kaggle.py"] == Q14K[1], ("R-DIAG lock", badrd)\n'),
    ('    assert k.RUNS[run][2] == "q22" and k.Q15_ROOT', '    assert k.RUNS[run][2] == ("q17" if "pdnat" in run else "q18") and k.Q15_ROOT'),
    ('E / "q22l" / f"{run}_out"', 'E / "rdiagl" / f"{run}_out"'),
    ('json.dumps({"driver": "q22l"', 'json.dumps({"driver": "rdiagl"'),
]


def write_once(p, b):
    assert not p.exists(), ("create-once", p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    assert not tmp.exists()
    tmp.write_bytes(b)
    tmp.rename(p)
    assert sha(p.read_bytes()) == sha(b)


def main():
    spec = read(SPEC_SRC)
    spec2 = edit(spec, [('"selected_variant": "pretrained_pd25"', '"selected_variant": "pretrained"')], "spec")
    json.loads(spec2)
    k = edit(read(K_SRC), K_EDITS, "kaggle")
    ksha = sha(k.encode("utf-8"))
    d = edit(read(D_SRC), D_EDITS, "driver").replace("@KSHA@", ksha)
    assert d.count(ksha) == 1
    for p, b in ((SPEC_OUT, spec2.encode("utf-8")), (K_OUT, k.encode("utf-8")), (D_OUT, d.encode("utf-8"))):
        write_once(p, b)
    rel = lambda p: p.relative_to(WS).as_posix()
    files = {rel(p): sha(p.read_bytes()) for p in (SPEC_OUT, K_OUT, D_OUT)}
    lock = {"stage": "R-DIAG r1 (pc_r9/RDIAG_SPEC.md)", "files": files,
            "sources": {rel(SPEC_SRC[0]): SPEC_SRC[1], rel(K_SRC[0]): K_SRC[1], rel(D_SRC[0]): D_SRC[1]},
            "q17_stage_lock_sha256": Q17_LOCK_SHA, "builder_sha256": sha(Path(__file__).read_bytes())}
    write_once(LOCK_OUT, json.dumps(lock, ensure_ascii=False, indent=1).encode("utf-8"))
    print(json.dumps({"files": files, "lock_sha256": sha(LOCK_OUT.read_bytes())}))


if __name__ == "__main__":
    main()
