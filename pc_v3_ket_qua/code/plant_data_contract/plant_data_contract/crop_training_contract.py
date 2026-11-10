"""CPU-only opt-in validation for paired crop arms; no trainer entry point."""
from collections import Counter
import math

MODES = ("Centralized", "FedAvg", "Local-only")
IDENTITY_KEYS = ("manifest_sha256", "class_map_sha256", "w0_sha256", "split_sha256",
                 "draw_index_sha256", "sampler_sha256", "optimizer_steps", "transform_sha256", "seed")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_paired_jobs(jobs):
    """Every mode has a GT/predicted pair with shared locked original-sample identity."""
    require(len(jobs) == 6, "Need six jobs: three modes x two crop arms")
    require({(j["mode"], j["crop_arm"]) for j in jobs} ==
            {(m, a) for m in MODES for a in ("gt", "predicted")}, "Mode/arm mismatch")
    reference = {k: jobs[0].get(k) for k in IDENTITY_KEYS}
    require(all(v is not None for v in reference.values()), "Missing locked job identity")
    require(type(reference["optimizer_steps"]) is int and reference["optimizer_steps"] > 0,
            "Invalid optimizer step budget")
    for key in IDENTITY_KEYS:
        if key.endswith("sha256"):
            value = reference[key]
            require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "Invalid identity hash")
    for job in jobs:
        require({k: job.get(k) for k in IDENTITY_KEYS} == reference, "Paired job identity mismatch")
        require(job.get("class_count") == 38, "Class map must contain 38 classes")
        require(job.get("cutmix", False) is False and job.get("mixup", False) is False,
                "Mixing requires a separate locked train-only ablation")
    return reference


def validate_samples(samples):
    """Reject original sample duplication, owner changes, and cross-split image/groups."""
    ids, image_split, group_split, byte_split = set(), {}, {}, {}
    counts = Counter()
    for s in samples:
        require(s["sample_id"] not in ids, "Duplicate original sample")
        ids.add(s["sample_id"])
        require(s["split"] in ("train", "validation", "test"), "Invalid split")
        require(isinstance(s["owner"], str) and bool(s["owner"]), "Original owner missing")
        require(type(s["class_id"]) is int and 0 <= s["class_id"] < 38, "Invalid class ID")
        for field, seen in (("image_id", image_split), ("scene_group_id", group_split), ("image_sha256", byte_split)):
            key = s[field]
            require(isinstance(key, str) and bool(key), "Missing source/split identity")
            require(key not in seen or seen[key][0] == s["split"], "Image/source group crosses split")
            require(key not in seen or seen[key][1] == s["owner"], "Image/source group crosses owners")
            seen[key] = (s["split"], s["owner"])
        require(s["source"] in ("PlantVillage", "PlantDoc", "FieldPlant"), "Unknown source")
        if s["source"] == "FieldPlant":
            require(s.get("taxonomy_verified") is True and s.get("provenance_verified") is True
                    and s.get("human_qa_verified") is True and bool(s.get("mapping_evidence")),
                    "FieldPlant taxonomy/provenance/QA blocked")
        if s["split"] == "train":
            counts[s["owner"]] += 1
    return dict(counts)


def validate_arm_ownership(original, derived):
    """Crops/augmentation are views of originals, not extra n_k contributions."""
    counts = validate_samples(original)
    lookup = {s["sample_id"]: s for s in original}
    for view in derived:
        require(view.get("original_sample_id") in lookup, "Unmatched view has no original label")
        base = lookup[view["original_sample_id"]]
        for k in ("owner", "split", "class_id", "source", "image_id", "image_sha256", "scene_group_id"):
            require(view.get(k) == base[k], "Derived view changes owner/label/source/split")
    require({v["original_sample_id"] for v in derived} == set(lookup), "Missing original in crop arm")
    return counts


def validate_predicted_matches(matches, gt_ids, pred_ids, split, allow_gt_fallback=False):
    """Only explicit one-to-one QA matches carry existing labels; no auto pseudo-labels."""
    require(split in ("train", "validation", "test", "inference"), "Invalid match split")
    require(not allow_gt_fallback or split == "train", "GT fallback is train-only")
    if split == "inference":
        require(not matches, "Inference has no GT matching or fallback")
        return []
    used_gt, used_pred = set(), set()
    for m in matches:
        require(m.get("human_qa_verified") is True, "GT match requires human QA")
        g, p = m["gt_id"], m["pred_id"]
        require(g in gt_ids and p in pred_ids, "Unknown GT/predicted object")
        require(g not in used_gt and p not in used_pred, "Match is not one-to-one")
        require(type(m.get("iou")) in (int, float) and math.isfinite(m["iou"]) and 0 <= m["iou"] <= 1, "Invalid match IoU")
        used_gt.add(g); used_pred.add(p)
    return list(matches)
