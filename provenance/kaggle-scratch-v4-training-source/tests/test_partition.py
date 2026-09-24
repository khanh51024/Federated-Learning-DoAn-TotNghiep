"""
Tests for the GĐ2 non-IID partition data layer.

Two tiers:
  * unit tests on synthetic fixtures -- always run, no dataset needed
  * integration tests on the real PlantVillage tree -- skipped automatically
    when the dataset is absent

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    FedAvgPartition,
    PlantVillageDataset,
    build_loader,
    group_samples,
    partition,
)
from src.data.dataset import load_manifest, quick_stats  # noqa: E402
from src.data.dirichlet_split import _allocate_counts, _repair_min_size  # noqa: E402
from scripts.sweep_alpha import name_for  # noqa: E402
from scripts.verify_integrity import audit  # noqa: E402
from src.data.leaf_groups import resolve_group_id, stem_to_leaf_key  # noqa: E402
from src.data.metrics import (  # noqa: E402
    audit_group_integrity,
    audit_sample_integrity,
    calculate_partition_metrics,
)
from src.data.partitioner import DatasetPartitioner  # noqa: E402
from src.data.transforms import (  # noqa: E402
    BaseTransform,
    ClientDomainTransform,
    ClientFeatureProfile,
    create_deterministic_client_profile,
    get_client_transform,
    get_default_transform,
)

REAL_DATASET = (ROOT.parent / "PlantVillage-Dataset" / "raw" / "color")
REAL_LEAF_MAP = ROOT.parent / "PlantVillage-Dataset" / "leaf-map.json"
requires_real_dataset = pytest.mark.skipif(
    not REAL_DATASET.exists(), reason=f"PlantVillage not found at {REAL_DATASET}"
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def make_sample(path: str, label: int, class_name: str, group_id: str) -> dict:
    return {"relative_path": path, "label": label, "class_name": class_name, "group_id": group_id}


@pytest.fixture
def synthetic_pool():
    """
    4 classes; class 0 and 1 have 3-image leaf groups, class 2 and 3 have 1.
    Group ids are shared across images of the same leaf on purpose.
    """
    pool = []
    specs = [
        ("ClsA", 0, 12, 4),   # 12 images in 4 leaves of 3
        ("ClsB", 1, 9, 3),    # 9 images in 3 leaves of 3
        ("ClsC", 2, 8, 8),    # 8 singleton leaves
        ("ClsD", 3, 5, 5),    # 5 singleton leaves
    ]
    for name, label, n_images, n_groups in specs:
        per = n_images // n_groups
        for g in range(n_groups):
            for j in range(per):
                idx = g * per + j
                pool.append(make_sample(
                    f"data/color/{name}/img_{idx:04d}.png", label, name, f"{name}:::{g}"
                ))
        # any remainder becomes singletons
        for idx in range(n_groups * per, n_images):
            pool.append(make_sample(
                f"data/color/{name}/img_{idx:04d}.png", label, name, f"{name}:::single{idx}"
            ))
    return pool


@pytest.fixture
def synthetic_dataset(tmp_path):
    """A tiny on-disk dataset with a leaf map, for end-to-end partitioner tests."""
    color = tmp_path / "ds" / "color"
    rng = np.random.default_rng(0)
    n_classes, n_leaves, per_leaf = 4, 6, 3
    leaf_entries = {}
    for c in range(n_classes):
        name = f"Crop___class{c}"
        d = color / name
        d.mkdir(parents=True)
        for leaf in range(n_leaves):
            for k in range(per_leaf):
                # <uuid>___<original stem>.png  -- real PlantVillage filename shape
                orig = f"PFX_C{c} {leaf * 10 + k:04d}"
                fname = f"{rng.integers(10**8)}___{orig}.png"
                Image.fromarray(
                    rng.integers(0, 255, (8, 8, 3), dtype=np.uint8), mode="RGB"
                ).save(d / fname)
                leaf_entries[f"{name}/{fname}"] = f"{name}:::{leaf}"
    leaf_map_path = tmp_path / "ds" / "leaf-map.json"

    # Build the leaf map keyed the way aggregate_map.py does: lowercase stem.
    lm = {}
    for rel, gid in leaf_entries.items():
        cls, leaf = gid.split(":::")
        stem = rel.rsplit("/", 1)[-1]
        stem = stem.split("___", 1)[1]
        lm[Path(stem).stem.lower().strip()] = [f"{cls}:::{leaf}.0"]
    leaf_map_path.write_text(json.dumps(lm), encoding="utf-8")

    return {"root": tmp_path, "color": color, "leaf_map": leaf_map_path, "n_classes": n_classes}


# --------------------------------------------------------------------------
# integer allocation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("total", [0, 1, 7, 100, 4344])
def test_allocate_counts_is_exact(total):
    rng = np.random.default_rng(total)
    for _ in range(20):
        p = rng.dirichlet(np.full(10, 0.1))
        counts = _allocate_counts(total, p)
        assert counts.sum() == total
        assert (counts >= 0).all()


def test_allocate_counts_handles_degenerate_proportions():
    assert _allocate_counts(10, np.zeros(5)).sum() == 10
    assert _allocate_counts(0, np.array([0.5, 0.5])).sum() == 0


# --------------------------------------------------------------------------
# leaf group resolution
# --------------------------------------------------------------------------
def test_stem_to_leaf_key_strips_uuid_and_extension():
    assert stem_to_leaf_key("001187a0-57ab___RS_Early.B 8178.JPG") == "rs_early.b 8178"
    assert stem_to_leaf_key("plain_name.png") == "plain_name"


def test_resolve_group_id_prefers_class_match():
    lm = {"rs_hl 1": ["Peach___healthy:::1", "Soybean___healthy:::1"]}
    gid, ok = resolve_group_id("u___RS_HL 1.JPG", "Soybean___healthy", "p", lm)
    assert gid == "Soybean___healthy:::1" and ok
    gid, ok = resolve_group_id("u___RS_HL 1.JPG", "Peach___healthy", "p", lm)
    assert gid == "Peach___healthy:::1" and ok


def test_resolve_group_id_singleton_fallback():
    gid, ok = resolve_group_id("u___Unknown 9.JPG", "ClsA", "a/b.png", {})
    assert not ok and gid == "img::a/b.png"


def test_leaf_map_class_mismatch_is_not_trusted():
    gid, ok = resolve_group_id("u___same.JPG", "ClassA", "ClassA/u.jpg",
                               {"same": ["ClassB:::1"]})
    assert not ok and gid.startswith("img::")


# --------------------------------------------------------------------------
# partition scenarios
# --------------------------------------------------------------------------
def test_group_samples_keeps_leaves_intact(synthetic_pool):
    groups = group_samples(synthetic_pool)
    for class_id, gs in groups.items():
        for g in gs:
            assert len({s["group_id"] for s in g["items"]}) == 1


@pytest.mark.parametrize("scenario,kwargs", [
    ("iid", {}),
    ("label_skew", {"alpha": 0.1}),
    ("label_skew", {"alpha": 100.0}),
    ("quantity_skew", {"quantity_alpha": 0.1}),
    ("label_quantity_skew", {"alpha": 0.1, "quantity_alpha": 0.1}),
])
def test_partition_conserves_and_never_splits_a_leaf(synthetic_pool, scenario, kwargs):
    groups = group_samples(synthetic_pool)
    client_samples, diag = partition(
        groups, scenario=scenario, num_clients=4, seed=7,
        min_samples_per_client=1, **kwargs,
    )
    flat = [s for v in client_samples.values() for s in v]

    assert len(flat) == len(synthetic_pool)
    assert diag["conserved"] is True
    assert diag["output_items"] == diag["input_items"]

    paths = [s["relative_path"] for s in flat]
    assert len(paths) == len(set(paths)), "an image landed in two clients"

    # group atomicity: every leaf group belongs to exactly one client
    owner = {}
    for cid, items in client_samples.items():
        for s in items:
            assert owner.setdefault(s["group_id"], cid) == cid, "leaf group split across clients"


def test_partition_is_deterministic(synthetic_pool):
    groups = group_samples(synthetic_pool)
    a, _ = partition(groups, scenario="label_skew", num_clients=4, alpha=0.1, seed=42,
                     min_samples_per_client=1)
    b, _ = partition(groups, scenario="label_skew", num_clients=4, alpha=0.1, seed=42,
                     min_samples_per_client=1)
    for c in a:
        assert [s["relative_path"] for s in a[c]] == [s["relative_path"] for s in b[c]]


def test_smaller_alpha_produces_stronger_label_skew(synthetic_pool):
    groups = group_samples(synthetic_pool)
    entropies = {}
    for alpha in (0.05, 100.0):
        cs, _ = partition(groups, scenario="label_skew", num_clients=4, alpha=alpha,
                          seed=3, min_samples_per_client=1)
        entropies[alpha] = calculate_partition_metrics(cs, total_classes=4)["shannon_entropy_mean"]
    assert entropies[0.05] < entropies[100.0]


def test_quantity_skew_changes_volume_not_label_mix(synthetic_pool):
    groups = group_samples(synthetic_pool)
    flat_cs, _ = partition(groups, scenario="quantity_skew", num_clients=4, quantity_alpha=100.0,
                           seed=5, min_samples_per_client=1)
    skewed_cs, _ = partition(groups, scenario="quantity_skew", num_clients=4, quantity_alpha=0.05,
                             seed=5, min_samples_per_client=1)
    flat_sizes = sorted(len(v) for v in flat_cs.values())
    skewed_sizes = sorted(len(v) for v in skewed_cs.values())
    assert max(skewed_sizes) - min(skewed_sizes) > max(flat_sizes) - min(flat_sizes)

    m_flat = calculate_partition_metrics(flat_cs, total_classes=4)
    m_skew = calculate_partition_metrics(skewed_cs, total_classes=4)
    assert m_skew["client_size_gini"] >= m_flat["client_size_gini"]


def test_min_size_repair_kicks_in(synthetic_pool):
    """A demanding threshold must be met (or reported honestly), never violated silently."""
    groups = group_samples(synthetic_pool)
    total = sum(len(g["items"]) for gs in groups.values() for g in gs)
    threshold = total // 4  # forces every client to be near the mean size
    cs, diag = partition(groups, scenario="label_skew", num_clients=4, alpha=0.01,
                         seed=11, min_samples_per_client=threshold, max_retries=5)
    sizes = [len(v) for v in cs.values()]
    assert sum(sizes) == total, "repair must not lose or duplicate samples"
    assert diag["min_size_satisfied"] == (min(sizes) >= threshold)


def test_min_size_repair_moves_whole_groups():
    rows_a = [make_sample(f"a{i}", 0, "A", "leaf_a") for i in range(8)]
    rows_b = [make_sample(f"b{i}", 0, "A", "leaf_b") for i in range(8)]
    repaired, moved = _repair_min_size({0: rows_a + rows_b, 1: []}, 5)
    owners = {}
    for client_id, rows in repaired.items():
        for row in rows:
            assert owners.setdefault(row["group_id"], client_id) == client_id
    assert moved == 1 and len(repaired[1]) == 8


def test_impossible_minimum_raises(synthetic_pool):
    groups = group_samples(synthetic_pool)
    total = len(synthetic_pool)
    with pytest.raises(RuntimeError, match="No partition was written"):
        partition(groups, scenario="iid", num_clients=4, seed=1,
                  min_samples_per_client=total, max_retries=2)


def test_partition_name_includes_seed_and_effective_parameters():
    base = {"scenario": "label_quantity_skew", "alpha": 0.1,
            "quantity_alpha": 0.1, "feature_skew": "none", "group_aware": True}
    assert name_for({**base, "seed": 42}) != name_for({**base, "seed": 43})
    assert "qalpha_0_1" in name_for({**base, "seed": 42})


def test_unknown_scenario_raises(synthetic_pool):
    with pytest.raises(ValueError, match="Unknown scenario"):
        partition(group_samples(synthetic_pool), scenario="nope", num_clients=2, seed=1)


def test_invalid_alpha_raises(synthetic_pool):
    with pytest.raises(ValueError):
        partition(group_samples(synthetic_pool), scenario="label_skew",
                  num_clients=2, alpha=0.0, seed=1)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_metrics_record_empty_clients():
    """Regression: an empty client used to vanish from statistics.csv."""
    cs = {
        0: [make_sample("a", 0, "A", "g0"), make_sample("b", 1, "B", "g1")],
        1: [],
    }
    m = calculate_partition_metrics(cs, total_classes=3)
    assert len(m["client_details"]) == 2
    empty = [r for r in m["client_details"] if r["client_id"] == 1][0]
    assert empty["sample_count"] == 0 and empty["num_classes"] == 0
    assert m["empty_clients"] == 1
    assert m["tvd_max"] == 1.0


def test_metrics_iid_is_near_zero_divergence():
    cs = {c: [make_sample(f"x{c}_{i}", i % 4, "C", f"g{c}_{i}") for i in range(200)]
          for c in range(5)}
    m = calculate_partition_metrics(cs, total_classes=4)
    assert m["tvd_mean"] < 0.05
    assert m["client_size_gini"] < 0.01
    assert m["entropy_norm_mean"] > 0.95


def test_sample_and_group_audits():
    pool = [make_sample(f"p{i}", i % 2, "C", f"g{i}") for i in range(10)]
    cs = {0: pool[:5], 1: pool[5:]}
    test = [make_sample("t0", 0, "C", "gt0")]
    assert audit_sample_integrity(cs, pool, test)["conserved"] is True
    assert audit_group_integrity(cs, test)["leakage_free"] is True

    # break it: same group in two clients
    bad = {0: pool[:6], 1: pool[4:]}
    assert audit_sample_integrity(bad, pool, test)["conserved"] is False
    assert audit_group_integrity(bad, test)["leakage_free"] is False


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
def test_base_transform_mobilenetv3_contract():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
    out = BaseTransform()(img, rng_seed=1)
    assert out.shape == (3, 224, 224) and out.dtype == np.float32
    # normalised range for ImageNet mean/std on [0,1] inputs
    assert out.min() > -3.0 and out.max() < 3.0


def test_base_transform_is_deterministic_given_seed():
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    t = BaseTransform(augment=True)
    a = t(img, rng_seed=123)
    b = t(img, rng_seed=123)
    c = t(img, rng_seed=124)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_feature_skew_levels_are_ordered_and_reproducible():
    def total_shift(level):
        p = create_deterministic_client_profile(3, seed=42, level=level)
        return (abs(p.brightness_factor - 1) + abs(p.contrast_factor - 1)
                + abs(p.saturation_factor - 1) + p.blur_radius + p.noise_std)

    assert total_shift("none") == 0.0
    assert total_shift("mild") <= total_shift("moderate") <= total_shift("strong")

    p1 = create_deterministic_client_profile(2, seed=42, level="strong")
    p2 = create_deterministic_client_profile(2, seed=42, level="strong")
    assert p1.to_dict() == p2.to_dict()
    assert create_deterministic_client_profile(2, seed=42, level="strong").to_dict() != \
        create_deterministic_client_profile(3, seed=42, level="strong").to_dict()


def test_client_transform_changes_pixels_and_is_seeded():
    arr = np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    clean = get_default_transform()(img, rng_seed=5)
    skewed = get_client_transform(1, seed=42, level="strong")(img, rng_seed=5)
    assert clean.shape == skewed.shape == (3, 224, 224)
    assert not np.allclose(clean, skewed, atol=1e-4)

    again = get_client_transform(1, seed=42, level="strong")(img, rng_seed=5)
    assert np.allclose(skewed, again, atol=1e-6), "noise must be seed-derived, not global RNG"


def test_noise_is_sample_dependent():
    """Same client, different sample index -> different noise realisation.

    The profile is built explicitly so the test always exercises the noise
    path (a randomly drawn profile may legitimately have noise_std == 0).
    """
    arr = np.random.default_rng(3).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    profile = ClientFeatureProfile(client_id=0, noise_std=6.0, level="strong")
    t = ClientDomainTransform(profile=profile)
    a = t(img, rng_seed=1)
    b = t(img, rng_seed=2)
    assert not np.allclose(a, b, atol=1e-4)
    assert np.allclose(a, t(img, rng_seed=1), atol=1e-6), "same seed must repeat exactly"


# --------------------------------------------------------------------------
# dataset / loader
# --------------------------------------------------------------------------
def write_manifest(path: Path, rows, base: Path) -> None:
    fields = ["relative_path", "label", "class_name", "client_id", "split",
              "group_id", "scenario", "alpha", "quantity_alpha", "feature_skew", "seed"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


@pytest.fixture
def tiny_manifest(tmp_path):
    rng = np.random.default_rng(9)
    rows = []
    for i in range(6):
        rel = f"imgs/c{i % 2}/{i:03d}.png"
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(p)
        rows.append({
            "relative_path": rel, "label": i % 2, "class_name": f"c{i % 2}",
            "client_id": 0, "split": "train", "group_id": f"g{i}",
            "scenario": "label_skew", "alpha": 0.1, "quantity_alpha": 0.1,
            "feature_skew": "moderate", "seed": 42,
        })
    m = tmp_path / "manifest.csv"
    write_manifest(m, rows, tmp_path)
    return tmp_path, m


def test_dataset_and_loader_shapes(tiny_manifest):
    tmp_path, m = tiny_manifest
    ds = PlantVillageDataset(m, base_dir=tmp_path, transform=get_default_transform((224, 224)))
    assert len(ds) == 6
    img, lab = ds[0]
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    assert arr.shape == (3, 224, 224)
    assert int(lab) in (0, 1)

    loader = build_loader(ds, batch_size=4, shuffle=True, seed=1, num_workers=0)
    batches = list(loader)
    assert sum(len(b[1]) for b in batches) == 6


def test_loader_shuffle_is_epoch_reproducible(tiny_manifest):
    from src.data.dataset import DataLoaderSimple

    tmp_path, m = tiny_manifest
    ds = PlantVillageDataset(m, base_dir=tmp_path, transform=get_default_transform())
    l1 = DataLoaderSimple(ds, batch_size=2, shuffle=True, seed=5)
    l2 = DataLoaderSimple(ds, batch_size=2, shuffle=True, seed=5)
    l1.set_epoch(3)
    l2.set_epoch(3)
    a = [np.asarray(b[1]) for b in l1]
    b = [np.asarray(x[1]) for x in l2]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))

    l1.set_epoch(4)
    c = [np.asarray(x[1]) for x in l1]
    assert not all(np.array_equal(x, y) for x, y in zip(a, c)), "epoch must change the order"


def test_epoch_changes_augmentation_but_is_reproducible(tiny_manifest):
    tmp_path, m = tiny_manifest
    ds = PlantVillageDataset(m, base_dir=tmp_path, transform=get_default_transform(augment=True))
    ds.set_epoch(2)
    a = np.asarray(ds[0][0])
    ds.set_epoch(3)
    b = np.asarray(ds[0][0])
    ds.set_epoch(2)
    c = np.asarray(ds[0][0])
    assert not np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_empty_manifest_raises(tmp_path):
    m = tmp_path / "empty.csv"
    write_manifest(m, [], tmp_path)
    with pytest.raises(ValueError, match="empty"):
        PlantVillageDataset(m, base_dir=tmp_path)


# --------------------------------------------------------------------------
# end-to-end on the synthetic on-disk dataset
# --------------------------------------------------------------------------
def test_partitioner_end_to_end(synthetic_dataset):
    info = synthetic_dataset
    p = DatasetPartitioner(
        dataset_path="ds/color",
        leaf_map_path="ds/leaf-map.json",
        base_dir=str(info["root"]),
        test_ratio=0.25,
        num_clients=3,
        alpha=0.5,
        seed=13,
        scenario="label_skew",
        feature_skew="moderate",
        min_samples_per_client=1,
        group_aware=True,
    )
    p.scan_dataset()
    out = info["root"] / "out" / "alpha_0_5"
    cfg = p.partition(out)

    assert cfg["total_images"] == info["n_classes"] * 6 * 3
    assert cfg["group_integrity"]["leakage_free"] is True
    assert cfg["sample_integrity"]["conserved"] is True
    assert cfg["leaf_group_audit"]["leaf_map_coverage"] == 1.0

    # every produced file exists
    for rel in ["global_test.csv", "centralized_train.csv", "partition_config.json",
                "fedavg_meta.json", "statistics.csv", "client_class_matrix.csv"]:
        assert (out / rel).exists(), rel
    for c in range(3):
        assert (out / "clients" / f"client_{c:02d}.csv").exists()

    # CSVs are byte-stable for the same seed
    first = (out / "clients" / "client_00.csv").read_bytes()
    p2 = DatasetPartitioner(
        dataset_path="ds/color", leaf_map_path="ds/leaf-map.json", base_dir=str(info["root"]),
        test_ratio=0.25, num_clients=3, alpha=0.5, seed=13, scenario="label_skew",
        feature_skew="moderate", min_samples_per_client=1, group_aware=True,
    )
    out2 = info["root"] / "out2" / "alpha_0_5"
    p2.scan_dataset()
    p2.partition(out2)
    assert (out2 / "clients" / "client_00.csv").read_bytes() == first


def test_group_aware_split_beats_image_level_on_leakage(synthetic_dataset):
    """The whole point of group_aware: image-level splitting leaks leaf groups."""
    info = synthetic_dataset
    results = {}
    for ga in (True, False):
        p = DatasetPartitioner(
            dataset_path="ds/color", leaf_map_path="ds/leaf-map.json", base_dir=str(info["root"]),
            test_ratio=0.25, num_clients=3, alpha=0.5, seed=13, scenario="label_skew",
            min_samples_per_client=1, group_aware=ga,
        )
        p.scan_dataset()
        out = info["root"] / ("ga" if ga else "img")
        cfg = p.partition(out)
        results[ga] = cfg["group_integrity"]

    assert results[True]["groups_split_across_clients"] == 0
    assert results[True]["groups_in_both_train_and_test"] == 0
    assert results[False]["groups_split_across_clients"] > 0 or \
        results[False]["groups_in_both_train_and_test"] > 0


def test_local_validation_is_audited_and_stale_files_are_removed(synthetic_dataset):
    info = synthetic_dataset
    p = DatasetPartitioner(
        dataset_path="ds/color", leaf_map_path="ds/leaf-map.json", base_dir=str(info["root"]),
        test_ratio=0.25, client_val_ratio=0.25, num_clients=3, alpha=1.0, seed=13,
        scenario="label_skew", feature_skew="none", min_samples_per_client=1,
    )
    p.scan_dataset()
    out = info["root"] / "val-audit"
    p.partition(out)
    assert any((out / "clients").glob("client_*_val.csv"))
    assert not audit(out).failures

    p.client_val_ratio = 0.0
    p.partition(out)
    assert not any((out / "clients").glob("client_*_val.csv"))
    assert not audit(out).failures


def test_fedavg_partition_weights_and_loaders(synthetic_dataset):
    info = synthetic_dataset
    p = DatasetPartitioner(
        dataset_path="ds/color", leaf_map_path="ds/leaf-map.json", base_dir=str(info["root"]),
        test_ratio=0.25, num_clients=3, alpha=1.0, seed=13, scenario="label_skew",
        feature_skew="moderate", min_samples_per_client=1,
    )
    p.scan_dataset()
    out = info["root"] / "fa"
    p.partition(out)

    part = FedAvgPartition(out)
    w = part.aggregation_weights
    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert np.array_equal(part.n_samples.sum() * w, part.n_samples.astype(float))
    assert part.num_classes == info["n_classes"]

    rep = part.readiness_report(check_images=2)
    assert rep["ready"] is True, rep["problems"]
    assert rep["image_probe"]["client_00"]["shapes"] == [(3, 224, 224)]

    ds_c = part.client_dataset(0)
    assert len(ds_c) == int(part.n_samples[0])
    assert len(part.centralized_dataset()) == part.total_train_samples
    assert len(part.global_test_dataset()) > 0

    subset = part.aggregation_weights_for([0, 2])
    assert abs(sum(subset.values()) - 1.0) < 1e-12

    override = FedAvgPartition(out, dataset_root=info["root"] / "ds" / "color")
    assert override.client_dataset(0).records[0]["full_path"].exists()

    part_none = FedAvgPartition(out, feature_skew="none", augment_train=True)
    assert part_none.client_dataset(0).transform.augment is True
    part_strong = FedAvgPartition(out, feature_skew="strong", augment_train=False)
    assert all(profile.level == "strong" for profile in part_strong.profiles)

    fair = FedAvgPartition(out, augment_train=False)
    local = fair.client_dataset(0)
    central = fair.centralized_dataset()
    path = local.records[0]["relative_path"]
    central_index = next(i for i, row in enumerate(central.records) if row["relative_path"] == path)
    local_image = local[0][0]
    central_image = central[central_index][0]
    local_image = local_image.numpy() if hasattr(local_image, "numpy") else np.asarray(local_image)
    central_image = central_image.numpy() if hasattr(central_image, "numpy") else np.asarray(central_image)
    assert np.allclose(local_image, central_image), "centralized must replay the source client profile"

    # feature skew ablation must change the pixels
    part_off = FedAvgPartition(out, feature_skew="none")
    a = np.asarray(ds_c[0][0] if not hasattr(ds_c[0][0], "numpy") else ds_c[0][0].numpy())
    b = np.asarray(part_off.client_dataset(0)[0][0])
    b = b.numpy() if hasattr(b, "numpy") else b
    assert not np.allclose(a, b, atol=1e-4)


# --------------------------------------------------------------------------
# real dataset (skipped when PlantVillage is absent)
# --------------------------------------------------------------------------
@requires_real_dataset
def test_real_dataset_leaf_map_resolves():
    p = DatasetPartitioner(
        dataset_path=str(REAL_DATASET), leaf_map_path=str(REAL_LEAF_MAP),
        base_dir=str(ROOT), test_ratio=0.2, num_clients=10, alpha=0.1, seed=42,
    )
    p.scan_dataset()
    la = p.leaf_audit
    assert la["total_images"] == 54305
    assert la["classes_total"] == 38
    # the official leaf map covers roughly three quarters of the corpus
    assert 0.6 < la["leaf_map_coverage"] < 0.9
    assert la["distinct_groups"] > 5000
    assert la["images_per_group_mean"] > 1.2


def test_load_manifest_and_quick_stats_regression(tmp_path):
    """Regression test for AttributeError on PlantVillageDataset.dataset_root."""
    manifest_csv = tmp_path / "test_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "label", "class_name", "client_id", "split", "group_id"])
        writer.writerow(["sample/img1.jpg", "0", "Apple___Apple_scab", "0", "train", "g0"])
        writer.writerow(["sample/img2.jpg", "1", "Apple___Black_rot", "0", "train", "g1"])

    records = load_manifest(manifest_csv)
    assert len(records) == 2
    assert records[0]["relative_path"] == "sample/img1.jpg"
    assert records[0]["label"] == 0

    stats = quick_stats(manifest_csv, num_classes=2)
    assert stats["images"] == 2
    assert stats["classes_present"] == 2
    assert stats["leaf_groups"] == 2
