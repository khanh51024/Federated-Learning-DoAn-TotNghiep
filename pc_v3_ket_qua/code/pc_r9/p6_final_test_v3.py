"""P6 final test (pc_r9/P6_SPEC_r2_PC.md r2 f7717d8f, TU VAN PASS 2026-11-09 with conditions 1-4; review of b6b22263 points 1-5).
ONE call evaluates every checkpoint of the PC list (P6_CHECKPOINTS_PC.json) and of the laptop list (P6_CHECKPOINTS_LAPTOP_r1.json e118897a)
once on every set. Order inside the call: forward + R4 -> p6/EVAL_RAW.json + logits + p6/sets/<set>.json written atomically FIRST ->
analysis -> P6_RESULT.json. Images are untrusted downloads -> python -I.
Modes (positional: mode out_dir cache_dir pc_list pc_list_sha):
  dryrun         val.jsonl ONLY (PIL.Image.open guarded by a whitelist of the val images). Forwards ALL entries (PC + laptop, not the
                 detector) on VAL, then runs the full analysis on proxy sets (PD172 / PD463 / EXT := the PD rows of VAL, PVL := the PV rows
                 of VAL); checks 6 models against known val numbers (2e-3), EXT pipeline == dataset pipeline (argmax, 269 PD val rows, those
                 6 models), R4 plumbing on 8 PD val images, int8 delta vs CPU parent. Output outside p6/.
  final          PD-172, PD-463, PV-legacy, EXT-FINAL (440), R4 per-leaf (EXT-FINAL FieldPlant). Output = experiments/.../p6 (create-once).
  analysis_only  re-runs ONLY the analysis from p6/EVAL_RAW.json + p6/logits + p6/sets (no forward, no image opened; for a crash of
                 the analysis after the forward; ledger INCIDENT; not a re-run of the test). Writes P6_RESULT.json if absent.
Pipelines: dataset sets through the q15 runner's CanonicalClassificationDataset (sandbox plant_data_contract; crop_geometry / image_rgb /
transforms / dataset byte-equal to the workspace copy, asserted) with canonical_v1 eval split in two: Resize+ToTensor cached as uint8 (exact,
asserted) in <cache_dir> (outside p6/, never copied); Normalize at forward time. EXT: open -> EXIF transpose -> crop crop_xyxy ->
to_rgb_on_black -> same transform (ext_sel_eval_pc). R4: r4_sel_eval_pc rule. Prediction = argmax of the 38 raw logits. int8 TorchScript
on CPU (engine x86); delta vs its fp32 parent evaluated on CPU in the same call. fp32 device: cuda if available (--cpu forces CPU).
Holm: reject iff p_holm <= 0.05 (as S5 / A1).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import importlib.util
import json
import math
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

WS = Path(r"<WORKSPACE_ROOT>")
E = WS / "experiments/train-improve-20261015"
REL = WS / "dataset/mixed/pv_pd_v3"
RUNNER = (E / "q15/sandbox/fedavg_mixed_runner.py", "7c29f6f6a3b164743231be4ef78075cda328d6b2b2cbb960d72a529741d0d462")
LAPTOP_LIST = (WS / "plans/cai-thien-train-20261015/r8_draft/p6/P6_CHECKPOINTS_LAPTOP_r1.json", "e118897a")
SPLIT = E / "ext_test/split_v1"
EXT_LOCK_SHA = "15601109da68b39961386a85ee1661503bc2c9947278c0003f8dde258ad9a56f"
EXT_FINAL_SHA = "81271f2b2b8d1d8e1b46a38d5e102eedbdc2c765d052fd5ea1429f8615cd27d8"
FP0 = (E / "fp0/FP0_MANIFEST.json", "9d05270a68ddb8fcc417193abb88e547b3e4e1e2e2a829ddb596770f2e4023d0")
DET_DIR = E / "laptop_runs/detl/det_out/run"
E5_RESULT = (E / "e5/E5_RESULT.json", "d8f983d3d699b83e9f495c13b41a87e9f430d6a2e7323a953ab49177e0acbe66")
EXT_ROOTS = (WS / "dataset", E / "r3b/download", E / "r3c/download")
SCORE, MARGIN = 0.40, 0.08
Z = 1.959963984540054
BATCH = 128
ALPHA_HOLM = 0.05
CONTRACT_FILES = ("crop_geometry.py", "image_rgb.py", "transforms.py", "dataset.py")
WEB_INT8 = {"pc_int8_dynamic_centralized_adamw": "pc_central_adamw_s42", "pc_int8_dynamic_centralized_sgd": "pc_central_sgd_s42",
            "pc_int8_dynamic_fedavg": "pc_fedavg_a01_s42", "pc_int8_dynamic_scaffold": "pc_scaffold_a01_s42"}
DRY_MODELS = ("pc_central_adamw_s42", "pc_fedavg_a01_s42", "pc_scaffold_a01_s42", "pc_local_a01_s42_client_00", "pc_deploy_r3data",
              "pc_int8_dynamic_fedavg")
DEPLOY_FAMILY = (("r3data_pc_vs_q5_reference", "ref_q5_central"), ("r3data_pc_vs_r4_per_leaf", "R4"),
                 ("r3data_pc_vs_aug2_lap", "lap_deploy_aug2"), ("r3data_pc_vs_bal_lap", "lap_deploy_bal"))
ASETS = ("PD172", "PD463", "PVL", "EXT")
GROUP_N = {"central|adamw|None": 5, "central|sgd|None": 5, "fl|fedavg|0.1": 5, "fl|fedavg|0.5": 5, "fl|fedavg|1.0": 3, "fl|fedavg|10.0": 3,
           "fl|fedavg|100.0": 3, "fl|scaffold_bn|0.1": 3, "fl|scaffold_bn|0.5": 3, "fl|fedavg_fp16_uplink|0.1": 3,
           **{f"fl|{m}|{a}": 5 for m in ("scaffold", "fedprox", "moon", "fedbn") for a in (0.1, 0.5)}}
OPENED = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic(p, obj):
    p = Path(p)
    p.with_suffix(".tmp").write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(p.with_suffix(".tmp"), p)
    return sha(p)


def wilson(k, n):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - j) * ps[i]))
        out[i] = run
    return out


def metrics(logits, t, sources):
    pr = logits.argmax(1)
    n = len(t)
    k = int((pr == t).sum())
    f38, fpres, per_class = [], [], {}
    present = sorted(set(t.tolist()))
    for c in range(38):
        tp = int(((pr == c) & (t == c)).sum()); fp = int(((pr == c) & (t != c)).sum()); fn = int(((pr != c) & (t == c)).sum())
        p_ = tp / (tp + fp) if tp + fp else 0.0; r_ = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        f38.append(f)
        if c in present:
            fpres.append(f)
            per_class[str(c)] = {"n": tp + fn, "correct": tp, "recall": r_, "wilson95": wilson(tp, tp + fn)}
    src = defaultdict(list)
    for i, s in enumerate(sources):
        src[s].append(i)
    return {"n": n, "correct": k, "accuracy": k / n, "accuracy_wilson95": wilson(k, n), "macro_f1_38": float(np.mean(f38)),
            "macro_f1_present": float(np.mean(fpres)), "per_class": per_class,
            "per_source": {s: {"n": len(ix), "accuracy": float((pr[ix] == t[ix]).mean()), "wilson95": wilson(int((pr[ix] == t[ix]).sum()), len(ix))}
                           for s, ix in src.items()}}


def unit_tests():
    assert abs(mcnemar(10, 2) - 0.03857421875) < 1e-12 and mcnemar(0, 0) == 1.0 and mcnemar(3, 3) == 1.0
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    lo, hi = wilson(8, 10)
    assert abs(lo - 0.4901624) < 1e-6 and abs(hi - 0.9433178) < 1e-6, (lo, hi)
    t, p = np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 0])
    m = metrics(np.eye(38)[p] * 5.0, t, ["a"] * 5)
    assert abs(m["accuracy"] - 0.6) < 1e-12 and abs(m["macro_f1_present"] - (0.5 + 0.8 + 0.0) / 3) < 1e-12, m
    return "PASS"


def msd(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": st.mean(xs) if xs else None, "sd": st.stdev(xs) if len(xs) > 1 else None, "n": len(xs)}


def load_runner():
    assert sha(RUNNER[0]) == RUNNER[1], "runner sha"
    sys.path.insert(0, str(RUNNER[0].parent))
    spec = importlib.util.spec_from_file_location("q15_runner", RUNNER[0])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    pdc = Path(sys.modules["plant_data_contract"].__file__).parent
    cf = {}
    for f in CONTRACT_FILES:
        a_, b_ = sha(pdc / f), sha(WS / "plant_data_contract/plant_data_contract" / f)
        assert a_ == b_, ("contract differs from workspace", f)
        cf[f] = a_
    return m, cf


def read_lists(pc_list, pc_sha):
    assert sha(pc_list) == pc_sha, "PC list sha"
    assert sha(LAPTOP_LIST[0]).startswith(LAPTOP_LIST[1]), "laptop list sha"
    pc = json.load(open(pc_list, encoding="utf-8"))["checkpoints"]
    lap = json.load(open(LAPTOP_LIST[0], encoding="utf-8"))["checkpoints"]
    ents = [dict(e, platform="PC") for e in pc] + [dict(e, platform="laptop") for e in lap]
    names = [e["name"] for e in ents]
    assert len(set(names)) == len(names), "duplicate names"
    bad = [e["name"] for e in ents if not (WS / e["pc_path"]).is_file() or sha(WS / e["pc_path"]) != e["sha256"]]
    assert not bad, ("missing / sha mismatch -> stop before any evaluation", bad[:5])
    return {e["name"]: e for e in ents}


class Cache:
    """uint8 N x 3 x 224 x 224 memmap per set (outside p6/) + targets / sources / ids."""

    def __init__(self, root, name):
        self.dir = Path(root) / name

    def build(self, items, n):
        import torch
        assert not self.dir.exists(), ("cache exists (create-once)", self.dir)
        self.dir.mkdir(parents=True)
        X = np.lib.format.open_memmap(self.dir / "x_u8.npy", mode="w+", dtype=np.uint8, shape=(n, 3, 224, 224))
        t, s, ids, extra = [], [], [], []
        for i, (x, y, src, iid, ex) in enumerate(items):
            u = (x * 255.0).round()
            assert float((u / 255.0 - x).abs().max()) == 0.0, "uint8 cache not exact"
            X[i] = u.to(dtype=torch.uint8).numpy()
            t.append(int(y)); s.append(src); ids.append(iid); extra.append(ex)
        assert len(t) == n, (len(t), n)
        X.flush()
        self.t, self.sources, self.ids, self.extra = np.array(t, dtype=np.int64), s, ids, extra
        self.X = np.load(self.dir / "x_u8.npy", mmap_mode="r")
        return self

    def meta(self):
        return {"n": len(self.t), "ids": self.ids, "targets": self.t.tolist(), "sources": self.sources, "extra": self.extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dryrun", "final", "analysis_only"])
    ap.add_argument("out_dir"); ap.add_argument("cache_dir"); ap.add_argument("pc_list"); ap.add_argument("pc_list_sha")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir).resolve()
    ut = unit_tests()
    ents = read_lists(Path(a.pc_list), a.pc_list_sha)
    if a.mode == "analysis_only":
        return analysis_only(out, ents)
    cache_root = Path(a.cache_dir).resolve()
    dry = a.mode == "dryrun"
    if dry:
        assert not out.is_relative_to((E / "p6").resolve()), "dry-run output must stay outside p6/ (condition 4)"
    else:
        assert out == (E / "p6").resolve(), "final output = experiments/train-improve-20261015/p6"
    assert not cache_root.is_relative_to(out) and not out.exists(), ("create-once / cache outside out", out)
    import torch
    import torch.nn as nn
    import PIL.Image
    from torchvision.models import mobilenet_v3_small
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.quantized.engine = "x86"
    torch.set_num_threads(4)
    R, contract = load_runner()
    from plant_data_contract.crop_geometry import crop_bounding_box, normalize_exif_orientation
    from plant_data_contract.image_rgb import to_rgb_on_black
    tf = R.get_transforms("canonical_v1", is_train=False)
    assert [type(x).__name__ for x in tf.transforms] == ["Resize", "ToTensor", "Normalize"], tf
    import torchvision.transforms as T
    pre, norm = T.Compose(tf.transforms[:2]), tf.transforms[2]
    dev = torch.device("cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu")

    guard = {"allowed": None}
    orig_open = PIL.Image.open

    def guarded_open(fp, *args, **kw):
        p = str(Path(fp).resolve()) if isinstance(fp, (str, os.PathLike)) else repr(fp)
        if guard["allowed"] is not None:
            assert p in guard["allowed"], ("dry run opened a non-val file", p)
        OPENED.append(p)
        return orig_open(fp, *args, **kw)
    PIL.Image.open = guarded_open

    def ds_items(manifest, keep):
        OPENED.append(str(manifest))
        ds = R.CanonicalClassificationDataset.from_manifest(manifest, dataset_root=WS / "dataset", transform=pre, return_metadata=True)
        return ds, [i for i, s_ in enumerate(ds.samples) if keep(s_)]

    def gen_ds(ds, idx):
        for i in idx:
            x, y, m = ds[i]
            yield x, y, m.get("source_domain", "unknown"), m.get("sample_id") or m.get("annotation_id"), None

    def ext_tensor(path, crop):
        with PIL.Image.open(WS / path) as img:
            im = normalize_exif_orientation(img)
            if crop is not None:
                x0, y0, x1, y1 = crop
                assert 0 <= x0 < x1 <= im.size[0] and 0 <= y0 < y1 <= im.size[1], ("crop outside frame", path)
                im = im.crop((x0, y0, x1, y1))
            return pre(to_rgb_on_black(im))

    inputs = {"script_sha256": sha(__file__), "mode": a.mode, "unit_tests": ut, "pc_list": str(a.pc_list), "pc_list_sha256": a.pc_list_sha,
              "laptop_list_sha256": sha(LAPTOP_LIST[0]), "runner_sha256": RUNNER[1], "contract_files_byte_equal_workspace": contract,
              "det_result_sha256": sha(DET_DIR / "DET_RESULT.json"), "e5_result_sha256": sha(E5_RESULT[0]),
              "checkpoints": {n: {"pc_path": e["pc_path"], "sha256": e["sha256"], "platform": e["platform"]} for n, e in ents.items()},
              "device_fp32": str(dev), "torch": torch.__version__, "manifests": {}}
    assert inputs["e5_result_sha256"] == E5_RESULT[1], "E5_RESULT sha"
    sets = {}
    if dry:
        valm = REL / "manifests/val.jsonl"
        inputs["manifests"]["val.jsonl"] = sha(valm)
        ds, idx = ds_items(valm, lambda s_: True)
        guard["allowed"] = {str(ds.resolver.resolve(s_["relative_path"], must_exist=True).resolve()) for s_ in ds.samples}
        assert len(idx) == 4613
        sets["VAL"] = Cache(cache_root, "VAL").build(gen_ds(ds, idx), len(idx))
        pd_rows = [s_ for s_ in ds.samples if s_.get("source_domain") == "plantdoc"]
        rel = lambda s_: ds.resolver.resolve(s_["relative_path"], must_exist=True).resolve().relative_to(WS)
        sets["VAL_PD_EXTPIPE"] = Cache(cache_root, "VAL_PD_EXTPIPE").build(
            ((ext_tensor(rel(s_), None), s_["class_id"], "plantdoc", s_.get("annotation_id"), None) for s_ in pd_rows), len(pd_rows))
        r4_rows = [{"path": str(rel(s_)), "class_id": s_["class_id"]} for s_ in pd_rows[:8]]
    else:
        for f in ("legacy_diagnostic_test.jsonl", "calibration.jsonl"):
            inputs["manifests"][f] = sha(REL / "manifests" / f)
        dl, il = ds_items(REL / "manifests/legacy_diagnostic_test.jsonl", lambda s_: True)
        pd172 = [i for i in il if dl.samples[i].get("source_domain") == "plantdoc"]
        pvl = [i for i in il if dl.samples[i].get("source_domain") == "plantvillage"]
        dc, ic = ds_items(REL / "manifests/calibration.jsonl", lambda s_: s_.get("source_domain") == "plantdoc")
        assert (len(pd172), len(pvl), len(ic)) == (172, 10843, 291), (len(pd172), len(pvl), len(ic))
        sets["PD172"] = Cache(cache_root, "PD172").build(gen_ds(dl, pd172), 172)
        sets["CAL_PD291"] = Cache(cache_root, "CAL_PD291").build(gen_ds(dc, ic), 291)
        sets["PVL"] = Cache(cache_root, "PVL").build(gen_ds(dl, pvl), 10843)
        assert sha(SPLIT / "EXT_SPLIT_LOCK.json") == EXT_LOCK_SHA
        assert json.load(open(SPLIT / "EXT_SPLIT_LOCK.json", encoding="utf-8"))["parts"]["EXT_FINAL_MANIFEST.jsonl"]["sha256"] == EXT_FINAL_SHA
        em = SPLIT / "EXT_FINAL_MANIFEST.jsonl"
        assert sha(em) == EXT_FINAL_SHA, "EXT-FINAL manifest sha"
        inputs["manifests"]["EXT_FINAL_MANIFEST.jsonl"] = EXT_FINAL_SHA
        OPENED.append(str(em))
        rows = [json.loads(l) for l in em.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 440 and len({r["ext_id"] for r in rows}) == 440
        for r in rows:
            p = (WS / r["path"]).resolve()
            assert any(p.is_relative_to(x.resolve()) for x in EXT_ROOTS) and sha(p) == r["byte_sha256"], ("EXT image", r["ext_id"])
        sets["EXT"] = Cache(cache_root, "EXT").build(((ext_tensor(r["path"], r.get("crop_xyxy")), r["class_id"], r["source"], r["ext_id"],
                                                       {"path": r["path"], "crop_xyxy": r.get("crop_xyxy")}) for r in rows), 440)
        assert sha(FP0[0]) == FP0[1]
        inputs["fp0_manifest_sha256"] = FP0[1]
        crops = {c["crop_id"]: c for c in json.load(open(FP0[0], encoding="utf-8"))["crops"]}
        r4_rows = []
        for r in rows:
            if r["source"] == "fieldplant":
                x, y, w, h = crops[r["ext_id"]]["bbox_xywh"]
                r4_rows.append({"path": r["path"], "class_id": r["class_id"], "ext_id": r["ext_id"], "fp0_bbox_xyxy": [x, y, x + w, y + h]})
    run_names = [n for n, e in ents.items() if e["family"] != "detector"]
    (out / "logits").mkdir(parents=True)
    (out / "sets").mkdir()
    for sn, c in sets.items():
        write_atomic(out / "sets" / f"{sn}.json", c.meta())

    def build_model(e):
        if e.get("format", "").startswith("torchscript"):
            return torch.jit.load(str(WS / e["pc_path"]), map_location="cpu").eval(), "cpu", "torchscript"
        ck = torch.load(WS / e["pc_path"], map_location="cpu", weights_only=False)
        key = "model_state" if "model_state" in ck else "model_state_dict"
        m = mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 38)
        m.load_state_dict(ck[key], strict=True)
        return m.eval(), None, key

    def forward(model, device, c, bs=BATCH):
        if not isinstance(model, torch.jit.ScriptModule):
            model = model.to(device)
        outl = []
        with torch.inference_mode():
            for i in range(0, len(c.t), bs):
                x = norm(torch.from_numpy(np.ascontiguousarray(c.X[i:i + bs])).float() / 255.0)
                outl.append(model(x.to(device)).float().cpu().numpy())
        return np.concatenate(outl)

    inputs["int8_batch"] = 1
    raw = {"stage": "P6 r2 " + a.mode, "models": {}, "int8_batch": 1, "fp32_batch": BATCH}
    LG = {}
    jobs = [(n, None) for n in run_names] + [(p, "cpu") for i8, p in WEB_INT8.items() if i8 in run_names]
    for n, force in jobs:
        e = ents[n]
        t0 = time.time()
        model, fixed, key = build_model(e)
        device = torch.device(force or fixed or dev.type)
        tag = n if force is None else f"{n}__cpu_parent"
        row = {"family": e["family"], "platform": e["platform"], "device": str(device), "state_key": key, "sets": {}}
        for sn in sets:
            if sn == "VAL_PD_EXTPIPE" and n not in DRY_MODELS and n not in WEB_INT8.values():
                continue
            # v3: dynamic int8 computes activation scales per batch -> evaluated one image at a time (= web); fp32 keeps BATCH
            lo = forward(model, device, sets[sn], 1 if e.get("format", "").startswith("torchscript_int8") else BATCH)
            LG[(tag, sn)] = lo
            np.savez_compressed(out / "logits" / f"{tag}__{sn}.npz", logits=lo)
            if sn != "CAL_PD291":
                row["sets"][sn] = metrics(lo, sets[sn].t, sets[sn].sources)
        if not dry:
            lo = np.concatenate([LG[(tag, "PD172")], LG[(tag, "CAL_PD291")]])
            LG[(tag, "PD463")] = lo
            row["sets"]["PD463"] = metrics(lo, np.concatenate([sets["PD172"].t, sets["CAL_PD291"].t]),
                                           sets["PD172"].sources + sets["CAL_PD291"].sources)
        row["wall_s"] = time.time() - t0
        raw["models"][tag] = row
        del model
        print(tag, json.dumps({sn: round(v["accuracy"], 4) for sn, v in row["sets"].items()}), round(row["wall_s"], 1), flush=True)

    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.transforms.functional import to_tensor
    det_res = json.load(open(DET_DIR / "DET_RESULT.json", encoding="utf-8"))
    assert det_res["status"] == "DONE" and sha(DET_DIR / "checkpoint_last.pt") == det_res["checkpoint_sha256"]
    assert ents["lap_detector"]["pc_path"] == (DET_DIR / "checkpoint_last.pt").relative_to(WS).as_posix()
    dck = torch.load(DET_DIR / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    det = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=2, min_size=dck["config"]["min_size"],
                                            max_size=dck["config"]["max_size"])
    det.load_state_dict(dck["model_state_dict"], strict=True)
    det.roi_heads.detections_per_img = 100
    det.eval()
    clf = build_model(ents["pc_deploy_r3data"])[0].cpu()

    def classify(img):
        with torch.inference_mode():
            p = torch.softmax(clf(norm(pre(to_rgb_on_black(img))).unsqueeze(0)), dim=1)[0]
        k = int(p.argmax())
        return k, float(p[k])

    images = defaultdict(list)
    for r in r4_rows:
        images[r["path"]].append(r)
    img_rows = []
    for path in sorted(images):
        rs = images[path]
        with PIL.Image.open(WS / path) as rawimg:
            im = normalize_exif_orientation(rawimg).convert("RGB")
        w, h = im.size
        with torch.inference_mode():
            o = det([to_tensor(im)])[0]
        boxes = [[float(v) for v in b] for b in o["boxes"][:100]]
        scores = [float(s_) for s_ in o["scores"][:100]]
        keep = [i for i, s_ in enumerate(scores) if s_ >= SCORE]
        det_cls = {i: classify(im.crop(tuple(crop_bounding_box(w, h, boxes[i], MARGIN)))) for i in keep}
        whole = classify(im)
        if det_cls:
            best = max(det_cls, key=lambda i: (det_cls[i][1], -i))
            rule, fb = det_cls[best][0], False
        else:
            rule, fb = whole[0], True
        cls = sorted({int(r["class_id"]) for r in rs})
        img_rows.append({"path": path, "target_ids": cls, "single_class": len(cls) == 1, "n_det": len(keep), "rule_pred": rule,
                         "fallback": fb, "whole_pred": whole[0]})
    single = [r for r in img_rows if r["single_class"]]
    r_ok = [r["rule_pred"] == r["target_ids"][0] for r in single]
    w_ok = [r["whole_pred"] == r["target_ids"][0] for r in single]
    raw["R4"] = {"n_images": len(img_rows), "n_single_class": len(single), "rule_accuracy": (sum(r_ok) / len(single)) if single else None,
                 "rule_wilson95": wilson(sum(r_ok), len(single)), "whole_accuracy": (sum(w_ok) / len(single)) if single else None,
                 "whole_wilson95": wilson(sum(w_ok), len(single)), "fallback_rate": (sum(r["fallback"] for r in img_rows) / len(img_rows)),
                 "images": img_rows}
    if dry:
        (out / "OPENED_FILES.json").write_text(json.dumps({"n": len(OPENED), "unique": sorted(set(OPENED))}, indent=0), encoding="utf-8")
        inputs["opened_files_sha256"] = sha(out / "OPENED_FILES.json")
        assert not any(("ext_test" in p) or ("legacy_diagnostic" in p) or ("calibration.jsonl" in p) for p in OPENED), "dry run touched a test file"
    inputs["eval_raw_sha256"] = write_atomic(out / "EVAL_RAW.json", raw)   # FIRST, before any analysis (review point 1a)
    write_atomic(out / "INPUTS.json", inputs)
    print("WROTE EVAL_RAW", inputs["eval_raw_sha256"], flush=True)

    if dry:
        T_, LGa = proxy_sets(raw, LG, sets)
        res = {"stage": raw["stage"], "dry_checks": dry_checks(raw, LG, sets, ents), "analysis_on_proxy_sets": analysis(raw, LGa, T_, ents, n_ext=None)}
        p = out / "DRYRUN_RESULT.json"
    else:
        T_ = {"PD172": sets["PD172"].t, "PD463": np.concatenate([sets["PD172"].t, sets["CAL_PD291"].t]), "PVL": sets["PVL"].t, "EXT": sets["EXT"].t}
        res = {"stage": raw["stage"], "eval_raw_sha256": inputs["eval_raw_sha256"], "analysis": analysis(raw, LG, T_, ents, n_ext=440)}
        p = out / "P6_RESULT.json"
    res["analysis_script_sha256"] = sha(__file__)
    print("WROTE", p, write_atomic(p, res), flush=True)


def proxy_sets(raw, LG, sets):
    """Dry run only: PD172 / PD463 / EXT := the PD rows of VAL, PVL := the PV rows of VAL (review point 1c). Adds the proxy metrics."""
    v = sets["VAL"]
    pdm = np.array([s_ == "plantdoc" for s_ in v.sources]); pvm = ~pdm
    T_ = {"PD172": v.t[pdm], "PD463": v.t[pdm], "EXT": v.t[pdm], "PVL": v.t[pvm]}
    LGa = {}
    for (tag, sn), lo in list(LG.items()):
        if sn != "VAL":
            continue
        for k, m in (("PD172", pdm), ("PD463", pdm), ("EXT", pdm), ("PVL", pvm)):
            LGa[(tag, k)] = lo[m]
            raw["models"][tag]["sets"][k] = metrics(lo[m], T_[k], [s_ for s_, keep in zip(v.sources, m) if keep])
    return T_, LGa


def dry_checks(raw, LG, sets, ents):
    known = {}
    for n in DRY_MODELS:
        e = ents[n]
        if n.startswith("pc_int8"):
            known[n] = json.load(open(E5_RESULT[0], encoding="utf-8"))["models"]["fedavg"]["dynamic_linear"]["metrics"]["macro_f1"]
        elif n == "pc_deploy_r3data":
            known[n] = json.load(open(E / "r3l/data_out/run/summary.json", encoding="utf-8"))["best_checkpoint_validation"]["raw_macro_f1"]
        elif e["family"] == "local_only":
            known[n] = json.load(open((WS / e["pc_path"]).parent / "history.json", encoding="utf-8"))[-1]["val_macro_f1_raw"]
        else:
            known[n] = json.load(open((WS / e["pc_path"]).parent / "summary.json", encoding="utf-8"))["history"][-1]["val_macro_f1_raw"]
    chk = {n: {"known": known[n], "dry": raw["models"][n]["sets"]["VAL"]["macro_f1_38"],
               "abs_d": abs(known[n] - raw["models"][n]["sets"]["VAL"]["macro_f1_38"])} for n in DRY_MODELS}
    pdm = np.array([s_ == "plantdoc" for s_ in sets["VAL"].sources])
    pipe = {n: bool(np.array_equal(LG[(n, "VAL")][pdm].argmax(1), LG[(n, "VAL_PD_EXTPIPE")].argmax(1))) for n in DRY_MODELS}
    return {"val_vs_known": chk, "all_within_2e-3": all(v["abs_d"] <= 2e-3 for v in chk.values()),
            "ext_pipeline_argmax_equal_dataset_pipeline": pipe, "all_pipe_equal": all(pipe.values()),
            "int8_cpu_parent_delta_val_macro_f1_38": raw["models"]["pc_int8_dynamic_fedavg"]["sets"]["VAL"]["macro_f1_38"]
            - raw["models"]["pc_fedavg_a01_s42__cpu_parent"]["sets"]["VAL"]["macro_f1_38"],
            "R4_plumbing": {k: raw["R4"][k] for k in ("n_images", "n_single_class", "rule_accuracy", "whole_accuracy", "fallback_rate")}}


def analysis_only(out, ents):
    assert out == (E / "p6").resolve() and not (out / "P6_RESULT.json").exists()
    raw = json.load(open(out / "EVAL_RAW.json", encoding="utf-8"))
    inp = json.load(open(out / "INPUTS.json", encoding="utf-8"))
    assert inp["eval_raw_sha256"] == sha(out / "EVAL_RAW.json"), "EVAL_RAW sha"
    meta = {sn: json.load(open(out / "sets" / f"{sn}.json", encoding="utf-8")) for sn in ("PD172", "CAL_PD291", "PVL", "EXT")}
    T_ = {"PD172": np.array(meta["PD172"]["targets"]), "PD463": np.array(meta["PD172"]["targets"] + meta["CAL_PD291"]["targets"]),
          "PVL": np.array(meta["PVL"]["targets"]), "EXT": np.array(meta["EXT"]["targets"])}
    LG = {}
    for tag in raw["models"]:
        for sn in ("PD172", "CAL_PD291", "PVL", "EXT"):
            LG[(tag, sn)] = np.load(out / "logits" / f"{tag}__{sn}.npz")["logits"]
        LG[(tag, "PD463")] = np.concatenate([LG[(tag, "PD172")], LG[(tag, "CAL_PD291")]])
    res = {"stage": raw["stage"], "eval_raw_sha256": inp["eval_raw_sha256"], "analysis": analysis(raw, LG, T_, ents, n_ext=440),
           "analysis_only_rerun": "analysis recomputed from saved logits (no forward, no image opened); not a re-run of the test",
           "analysis_script_sha256": sha(__file__)}
    print("WROTE", out / "P6_RESULT.json", write_atomic(out / "P6_RESULT.json", res))


def correct(LG, tag, sn, t):
    return LG[(tag, sn)].argmax(1) == t


def analysis(raw, LG, T, ents, n_ext):
    M = raw["models"]
    A = {"labels": {"legacy": "exposed: legacy test, used 3 times before (Q5 / Q6 / Q11b)",
                    "ranking": "lua chon dua tren validation; tren test <ket qua> (condition 2)",
                    "mcnemar_fl": "descriptive; only the count of seeds with p < 0.05 out of the total is reported (condition 3)",
                    "holm_rule": "reject iff p_holm <= 0.05"}}
    tests = []
    ok_r3 = correct(LG, "pc_deploy_r3data", "EXT", T["EXT"])
    for lab, other in DEPLOY_FAMILY:
        if other == "R4":
            single = [r for r in raw["R4"]["images"] if r["single_class"]]
            x = np.array([r["whole_pred"] == r["target_ids"][0] for r in single]); y = np.array([r["rule_pred"] == r["target_ids"][0] for r in single])
        else:
            x, y = ok_r3, correct(LG, other, "EXT", T["EXT"])
            if n_ext is not None:
                assert len(x) == n_ext, (lab, len(x))
        b, c = int((x & ~y).sum()), int((~x & y).sum())
        tests.append({"test": lab, "n": int(len(x)), "acc_r3data_pc": float(x.mean()), "acc_other": float(y.mean()), "b_r3data_only": b,
                      "c_other_only": c, "p": mcnemar(b, c), "platform_other": "PC" if other == "R4" else ents[other]["platform"]})
    assert len(tests) == 4
    for t_, ph in zip(tests, holm([t_["p"] for t_ in tests])):
        t_["p_holm"] = ph
        t_["reject"] = ph <= ALPHA_HOLM
    A["deploy_family_holm4"] = tests
    pcs = {n: e for n, e in ents.items() if e["platform"] == "PC"}
    groups = defaultdict(list)
    for n, e in pcs.items():
        f = e["family"]
        if f.startswith("fl_") or f in ("rdiag_descriptive", "d4_descriptive"):
            fam = f if f in ("fl_phase_d", "d4_descriptive", "rdiag_descriptive") else "fl"
            groups[f"{fam}|{e.get('method')}|{e.get('alpha', e.get('scheme'))}"].append(n)
        elif f == "central_p40":
            groups[f"central|{e['optimizer']}|None"].append(n)
    for k, nn_ in GROUP_N.items():
        assert len(groups.get(k, [])) == nn_, ("group size", k, len(groups.get(k, [])), nn_)
    G = {k: {"members": sorted(ns), "sets": {sn: {m: msd([M[n]["sets"][sn][m] for n in ns]) for m in ("accuracy", "macro_f1_present")}
                                               for sn in T}} for k, ns in sorted(groups.items())}
    A["groups_mean_sd"] = G
    pairs, skipped = [], []
    for n, e in sorted(pcs.items()):
        m, al, s = e.get("method"), e.get("alpha"), e.get("seed")
        if not e["family"].startswith(("fl_methods", "fl_a2", "rdiag", "d4")) or m in (None, "fedavg"):
            continue
        if e["family"] == "d4_descriptive":
            cand = ["pc_d4_fedavg_s41"]
        else:
            cand = [k for k, v in pcs.items() if v.get("method") == "fedavg" and v.get("alpha") == al and v.get("seed") == s
                    and v["family"] in ("fl_alpha_curve", "fl_s5_fedavg")]
        if len(cand) != 1:
            skipped.append({"model": n, "candidates": cand})
            continue
        for sn in T:
            x, y = correct(LG, cand[0], sn, T[sn]), correct(LG, n, sn, T[sn])
            b, c = int((x & ~y).sum()), int((~x & y).sum())
            pairs.append({"method": m, "alpha": al, "seed": s, "set": sn, "fedavg": cand[0], "model": n, "b_fedavg_only": b,
                          "c_method_only": c, "p": mcnemar(b, c)})
    cnt = defaultdict(lambda: [0, 0])
    for p_ in pairs:
        k = f"{p_['method']}|{p_['alpha']}|{p_['set']}"
        cnt[k][1] += 1
        cnt[k][0] += p_["p"] < 0.05
    A["mcnemar_per_seed"] = {"pairs": pairs, "skipped_pairs": skipped,
                             "count_p_lt_0_05_over_total": {k: f"{v[0]}/{v[1]}" for k, v in sorted(cnt.items())}}
    loc = defaultdict(list)
    for n, e in pcs.items():
        if e["family"].startswith("local_only"):
            loc[n.rsplit("_client_", 1)[0]].append(n)
    L = {}
    for run, ns in sorted(loc.items()):
        assert len(ns) == 5, (run, ns)
        summ = json.load(open((WS / ents[ns[0]]["pc_path"]).parent.parent / "summary.json", encoding="utf-8"))["clients"]
        nk = {n: summ[n[-9:]]["n_k"] for n in ns}
        small = min(ns, key=lambda n: nk[n])
        L[run] = {sn: {"client_mean_accuracy": st.mean(M[n]["sets"][sn]["accuracy"] for n in ns),
                       "client_mean_macro_f1_present": st.mean(M[n]["sets"][sn]["macro_f1_present"] for n in ns),
                       "per_client_accuracy": {n[-9:]: M[n]["sets"][sn]["accuracy"] for n in sorted(ns)},
                       "smallest_client": {"name": small[-9:], "n_k": nk[small], "accuracy": M[small]["sets"][sn]["accuracy"]}}
                  for sn in T}
    for tag in ("01", "05", "1", "10", "100", "qs", "ss"):
        k = f"pc_local_a{tag}" if tag not in ("qs", "ss") else f"pc_local_{tag}"
        assert sum(1 for r in L if r.rsplit("_s", 1)[0] == k) == 3, ("local runs", k)
    A["local_only"] = L
    gap = {}
    for sn in T:
        row = {"central_adamw": G["central|adamw|None"]["sets"][sn], "central_sgd": G["central|sgd|None"]["sets"][sn]}
        for al, tag in ((0.1, "01"), (0.5, "05"), (1.0, "1"), (10.0, "10"), (100.0, "100")):
            row[f"fedavg_a{tag}"] = G[f"fl|fedavg|{al}"]["sets"][sn]
            row[f"local_a{tag}_client_mean_accuracy"] = msd([L[f"pc_local_a{tag}_s{s}"][sn]["client_mean_accuracy"] for s in (42, 43, 44)])
        assert all((v["mean"] is not None) if "mean" in v else all(x["mean"] is not None for x in v.values()) for v in row.values()),             ("empty gap cell", sn)
        gap[sn] = row
    A["gap_table"] = gap
    A["int8_vs_cpu_fp32_parent"] = {i8: {sn: {"d_accuracy": M[i8]["sets"][sn]["accuracy"] - M[f"{p}__cpu_parent"]["sets"][sn]["accuracy"],
                                             "d_macro_f1_present": M[i8]["sets"][sn]["macro_f1_present"] - M[f"{p}__cpu_parent"]["sets"][sn]["macro_f1_present"]}
                                        for sn in T} for i8, p in WEB_INT8.items()}
    A["laptop_rows"] = {n: {sn: {k: M[n]["sets"][sn][k] for k in ("n", "accuracy", "accuracy_wilson95", "macro_f1_present")} for sn in T}
                        for n, e in ents.items() if e["platform"] == "laptop" and n in M}
    return A


if __name__ == "__main__":
    main()
