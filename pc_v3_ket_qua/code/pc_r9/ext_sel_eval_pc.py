"""PC build (pc_r9/build_ext_sel_eval_pc.py from r8_draft/ext_sel_eval.py b38020ad), DEPLOY_RUNS_SPEC r3 PC: candidates + baseline r3data_pc.
EXT-SEL evaluation (DEPLOY_RUNS_SPEC r2 sections 3 and 5). ONE call evaluates every registered classifier candidate ONCE on
EXT-SEL (split_v1, lock 15601109) and applies the registered selection rule. Reads EXT_SEL_MANIFEST.jsonl only (never EXT_FINAL).
CPU only, 2 threads, deterministic algorithms; images are untrusted downloads -> run with python -I.
Image path (same as training eval): open -> EXIF transpose (plant_data_contract normalize_exif_orientation) -> crop crop_xyxy when set
(FieldPlant; FP0 boxes are post-EXIF) -> to_rgb_on_black -> canonical_v1 eval transform -> mobilenet_v3_small, 38-class head, strict
load of model_state. Every image byte sha is checked against the manifest before use.
Score S = unweighted mean of per-class recall over the EXT-SEL classes (exact fractions). Secondary (reported, not used): mean recall
over the unseen-field classes (Potato healthy, Apple rust, Apple healthy; r3c: Potato healthy only, its Apple classes are in-source).
Selection: winner = argmax S among selectable candidates with guardrail PASS (gates file); it must beat R3-data's S by >= 0.02,
otherwise the winner is R3-data. Ties -> earlier in CANDIDATES order (R3-data first). Q5 Central = reference, never selectable.
Usage: python -I -B ext_sel_eval.py <deploy_gates_json> <gates_sha256>      (writes ext_test/sel_eval_v1, create-once)
Selftest (no EXT data): env EXT_SEL_SELFTEST_MANIFEST=<jsonl outside ext_test> and EXT_SEL_SELFTEST_OUT=<dir outside ext_test>;
same code, checkpoint paths taken from the gates file, result tagged SELFTEST.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import csv
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

WS = Path(r"<WORKSPACE_ROOT>")
E = WS / "experiments/train-improve-20261015"
SPLIT = E / "ext_test/split_v1"
LOCK_SHA = "15601109da68b39961386a85ee1661503bc2c9947278c0003f8dde258ad9a56f"
SEL_SHA = "9a3d0e3ff95f66fe4cc7562cd22bb647bc1462481d53e7b57606a53aa990f6f2"
REAL_OUT = E / "ext_test/sel_eval_pc_v1"
IMAGE_ROOTS = (WS / "dataset", E / "r3b/download", E / "r3c/download")  # FieldPlant | Potato (Mendeley) | PP2020 test (EXT seal v2)
CANDIDATES = [  # DEPLOY_RUNS_SPEC r3 PC section 1: (label, checkpoint path, pinned sha, selectable); platform in PLATFORM
    ("r3data_pc", "experiments/train-improve-20261015/r3l/data_out/run/checkpoint_best.pt", "a274bb9e713728ab5578198661b93419d4ff8d5050cf643416346736997f58a4", True),
    ("bg_pc", "experiments/train-improve-20261015/r3xl/bg_out/run/checkpoint_best.pt", "fad07c39666acb2dee20e7627adc0f5a3443c6b4f4adc7c516e5a8f17aac0d33", True),
    ("r3c_pc", "experiments/train-improve-20261015/r3xl/r3c_out/run/checkpoint_best.pt", "8751d8f10cab78f2e7394c95cb3091e4643cada5c31b2b371942d0f65e75a344", True),
    ("r3data_lap", "experiments/train-improve-20261015/laptop_runs/r3l/data_out/run/checkpoint_best.pt", "905043dde38f14901148a342e886195f9a7f48eca1fddfd826712ca90648f450", True),
    ("aug2_lap", "experiments/train-improve-20261015/laptop_runs/r3xl/aug2_out/run/checkpoint_best.pt", "0463eb3eefc1811a18ea2186472c172b439098f2c4ae90e283167a79fe54c308", True),
    ("bal_lap", "experiments/train-improve-20261015/laptop_runs/r3xl/bal_out/run/checkpoint_best.pt", "8f9336dadbc8a9e2a6fef80b9d12166a807ebb0cde2c8a996dc866cb1e46bdc9", True),
    ("q5central", "experiments/train-improve-20261015/q5/colab/received_20261021/checkpoint_best.pt",
     "c45fa9937b636d815ef99f4bd21f078cad86521cf67a5b118f7c64e0a7809998", False),
]
PLATFORM = {"r3data_pc": "PC", "bg_pc": "PC", "r3c_pc": "PC", "r3data_lap": "laptop", "aug2_lap": "laptop", "bal_lap": "laptop",
            "q5central": "Colab T4"}
UNSEEN = ("Potato___healthy", "Apple___Cedar_apple_rust", "Apple___healthy")
IN_SOURCE = {"r3c_pc": ("Apple___Cedar_apple_rust", "Apple___healthy")}
MARGIN = Fraction(2, 100)
Z = 1.959963984540054


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_rows(manifest):
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [r["ext_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate ext_id"
    for r in rows:
        p = (WS / r["path"]).resolve()
        assert any(p.is_relative_to(x.resolve()) for x in IMAGE_ROOTS) and p.is_file(), ("image outside the EXT roots or missing", r["path"])
        assert 0 <= int(r["class_id"]) < 38
    return rows


def main(argv):
    selftest_m = os.environ.get("EXT_SEL_SELFTEST_MANIFEST")
    if selftest_m:
        manifest = Path(selftest_m).resolve()
        out = Path(os.environ["EXT_SEL_SELFTEST_OUT"]).resolve()
        for p in (manifest, out):
            assert not p.is_relative_to((E / "ext_test").resolve()), ("selftest must stay outside ext_test", p)
        man_sha = sha(manifest)
    else:
        assert sha(SPLIT / "EXT_SPLIT_LOCK.json") == LOCK_SHA, "split lock sha"
        lock = json.loads((SPLIT / "EXT_SPLIT_LOCK.json").read_text(encoding="utf-8"))
        assert lock["parts"]["EXT_SEL_MANIFEST.jsonl"]["sha256"] == SEL_SHA
        manifest, out = SPLIT / "EXT_SEL_MANIFEST.jsonl", REAL_OUT
        man_sha = sha(manifest)
        assert man_sha == SEL_SHA, "EXT-SEL manifest sha"
    assert len(argv) == 2, "usage: <deploy_gates_json> <gates_sha256>"
    gates_p, gates_sha = Path(argv[0]).resolve(), argv[1]
    assert sha(gates_p) == gates_sha, "gates file sha"
    gates = json.loads(gates_p.read_text(encoding="utf-8"))["candidates"]
    assert not out.exists(), ("create-once: EXT-SEL is evaluated once", out)
    tmp = out.with_name(out.name + ".tmp")
    assert not tmp.exists(), ("stale tmp dir", tmp)

    cand = []
    for label, rel, pinned, selectable in CANDIDATES:
        g = gates[label]
        ck = (WS / g["checkpoint"]).resolve() if selftest_m else (WS / rel).resolve()
        if not selftest_m:
            assert g["checkpoint"] == rel, ("gates checkpoint path", label)
        assert g["valid"] is True, ("candidate not valid", label)
        assert isinstance(g["guardrail_pass"], bool)
        want = g["checkpoint_sha256"] if (pinned is None or selftest_m) else pinned
        assert g["checkpoint_sha256"] == want, ("gates sha vs pinned", label)
        assert sha(ck) == want, ("checkpoint sha", label)
        cand.append((label, ck, want, selectable, g["guardrail_pass"]))

    rows = load_rows(manifest)
    bad = [r["ext_id"] for r in rows if sha(WS / r["path"]) != r["byte_sha256"]]
    assert not bad, ("image byte sha mismatch", bad[:3])

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision.models import mobilenet_v3_small
    sys.path.insert(0, str(WS / "plant_data_contract"))
    from plant_data_contract.crop_geometry import normalize_exif_orientation
    from plant_data_contract.image_rgb import to_rgb_on_black
    from plant_data_contract.transforms import get_transforms
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    tf = get_transforms("canonical_v1", is_train=False)

    xs, sizes = [], []
    for r in rows:
        with Image.open(WS / r["path"]) as img:
            im = normalize_exif_orientation(img)
            if r.get("crop_xyxy") is not None:
                x0, y0, x1, y1 = r["crop_xyxy"]
                assert 0 <= x0 < x1 <= im.size[0] and 0 <= y0 < y1 <= im.size[1], ("crop outside frame", r["ext_id"])
                im = im.crop((x0, y0, x1, y1))
            sizes.append(list(im.size))
            xs.append(tf(to_rgb_on_black(im)))
    X = torch.stack(xs)
    y = [int(r["class_id"]) for r in rows]
    classes = sorted({(int(r["class_id"]), r["target_class"]) for r in rows})
    sources = sorted({r["source"] for r in rows})

    tmp.mkdir(parents=True)
    res = {}
    for label, ck, ck_sha, selectable, gpass in cand:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 38)
        model.load_state_dict(state["model_state"], strict=True)
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(X[i:i + 32]) for i in range(0, len(X), 32)])
        prob = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).tolist()
        per_class, recalls = {}, {}
        for cid, name in classes:
            idx = [i for i, t in enumerate(y) if t == cid]
            k = sum(pred[i] == cid for i in idx)
            lo, hi = wilson(k, len(idx))
            recalls[name] = Fraction(k, len(idx))
            per_class[name] = {"class_id": cid, "n": len(idx), "correct": k, "recall": k / len(idx), "wilson95": [lo, hi]}
        S = sum(recalls.values()) / len(recalls)
        sec_cls = [c for c in UNSEEN if c in recalls and c not in IN_SOURCE.get(label, ())]
        if not selftest_m:
            assert set(UNSEEN) <= set(recalls), "unseen-field classes missing from EXT-SEL"
        sec = sum(recalls[c] for c in sec_cls) / len(sec_cls) if sec_cls else None
        per_source = {}
        for s in sources:
            idx = [i for i, r in enumerate(rows) if r["source"] == s]
            per_source[s] = {"n": len(idx), "accuracy": sum(pred[i] == y[i] for i in idx) / len(idx)}
        csv_p = tmp / f"{label}_predictions.csv"
        with open(csv_p, "w", encoding="utf-8", newline="\n") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["ext_id", "source", "target_id", "target_class", "predicted_id", "confidence", "prob_target", "eval_w", "eval_h"])
            for i, r in enumerate(rows):
                w.writerow([r["ext_id"], r["source"], y[i], r["target_class"], pred[i], f"{prob[i, pred[i]].item():.6f}",
                            f"{prob[i, y[i]].item():.6f}", sizes[i][0], sizes[i][1]])
        np.savez_compressed(tmp / f"{label}_logits.npz", logits=logits.numpy(), ext_id=np.array([r["ext_id"] for r in rows]))
        res[label] = {"checkpoint": str(ck.relative_to(WS)).replace("\\", "/"), "checkpoint_sha256": ck_sha,
                      "checkpoint_epoch": state.get("epoch"), "selectable": selectable, "guardrail_pass": gpass, "platform": PLATFORM[label],
                      "S": float(S), "S_fraction": f"{S.numerator}/{S.denominator}", "per_class": per_class, "per_source": per_source,
                      "secondary_unseen_field": {"classes": sec_cls, "mean_recall": None if sec is None else float(sec)},
                      "in_source_flag": list(IN_SOURCE.get(label, ())), "predictions_csv_sha256": sha(csv_p),
                      "logits_npz_sha256": sha(tmp / f"{label}_logits.npz")}
        res[label]["_S"] = S
    order = [c[0] for c in CANDIDATES]
    elig = [l for l in order if res[l]["selectable"] and res[l]["guardrail_pass"]]
    assert "r3data_pc" in elig, "baseline must be eligible"
    best = max(elig, key=lambda l: (res[l]["_S"], -order.index(l)))
    delta = res[best]["_S"] - res["r3data_pc"]["_S"]
    winner = best if (best != "r3data_pc" and delta >= MARGIN) else "r3data_pc"
    for l in res:
        res[l].pop("_S")
    result = {"stage": "EXT-SEL deploy selection (DEPLOY_RUNS_SPEC r3 PC section 3)", "selftest": bool(selftest_m),
              "manifest": str(manifest), "manifest_sha256": man_sha, "split_lock_sha256": None if selftest_m else LOCK_SHA,
              "gates_file": str(gates_p), "gates_sha256": gates_sha, "n_items": len(rows), "classes": [c[1] for c in classes],
              "eval": "CPU, 4 threads, deterministic; EXIF transpose -> crop_xyxy -> to_rgb_on_black -> canonical_v1 eval",
              "torch": torch.__version__, "candidates": res,
              "selection": {"rule": "argmax S over selectable + guardrail PASS (all platforms); must beat r3data_pc S by >= 0.02, else r3data_pc; ties -> order",
                            "eligible": elig, "best_eligible": best, "delta_vs_r3data_pc": float(delta),
                            "delta_fraction": f"{delta.numerator}/{delta.denominator}", "winner": winner},
              "labels": ["selection, biased", "model_visual_check_claude (not human QA) for visually checked EXT items",
                         "EXT-FINAL later gives the unbiased estimate of the chosen model"]}
    (tmp / "EXT_SEL_RESULT.json").write_bytes(json.dumps(result, ensure_ascii=False, indent=1).encode("utf-8"))
    os.replace(tmp, out)
    print(json.dumps({"out": str(out), "winner": winner, "best_eligible": best, "delta": float(delta),
                      "S": {l: round(res[l]["S"], 4) for l in order}, "result_sha256": sha(out / "EXT_SEL_RESULT.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
