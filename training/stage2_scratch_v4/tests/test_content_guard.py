import csv
import hashlib
import json
from pathlib import Path

import pytest

from fl_training.content_audit import audit_image_content
from fl_training.prepare import create_smoke_bundle
from src.data.leaf_groups import attach_group_ids


def test_grouping_rejects_missing_and_conflicting_hashes():
    rows = [{'relative_path': 'a/x.jpg', 'class_name': 'a'},
            {'relative_path': 'b/y.jpg', 'class_name': 'b'}]
    with pytest.raises(ValueError, match='complete'):
        attach_group_ids(rows, {}, {})
    with pytest.raises(ValueError, match='Missing'):
        attach_group_ids(rows, {}, {'a/x.jpg': 'abc'})
    with pytest.raises(ValueError, match='conflicting labels'):
        attach_group_ids(rows, {}, {'a/x.jpg': 'abc', 'b/y.jpg': 'abc'})


def make_bundle(tmp_path):
    bundle, dataset = tmp_path / 'bundle', tmp_path / 'images'
    (bundle / 'clients').mkdir(parents=True)
    classes = [f'class{i:02d}' for i in range(38)]
    (bundle / 'fedavg_meta.json').write_text(json.dumps({'class_names': classes, 'content_aware': True}))
    inventory = {}
    for i, manifest in enumerate(['clients/client_00.csv', 'global_val.csv', 'global_test.csv']):
        rel = f'{classes[0]}/{i}.jpg'
        image = dataset / rel
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f'image {i}'.encode())
        inventory[rel] = hashlib.sha256(image.read_bytes()).hexdigest()
        with (bundle / manifest).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['relative_path', 'class_name', 'label', 'group_id'])
            writer.writeheader()
            writer.writerow(dict(relative_path=rel, class_name=classes[0], label=0, group_id=str(i)))
    (bundle / 'image_content.json').write_text(json.dumps(inventory))
    return bundle, dataset


def test_pinned_content_detects_changed_real_bytes(tmp_path):
    bundle, dataset = make_bundle(tmp_path)
    assert audit_image_content(bundle, dataset)['verified_images'] == 3
    (dataset / 'class00/1.jpg').write_bytes(b'changed image')
    with pytest.raises(ValueError, match='pinned inventory'):
        audit_image_content(bundle, dataset)


def test_byte_overlap_not_hidden_by_different_group_ids(tmp_path):
    bundle, dataset = make_bundle(tmp_path)
    (dataset / 'class00/1.jpg').write_bytes((dataset / 'class00/0.jpg').read_bytes())
    inventory = json.loads((bundle / 'image_content.json').read_text())
    inventory['class00/1.jpg'] = inventory['class00/0.jpg']
    (bundle / 'image_content.json').write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match='across splits/clients'):
        audit_image_content(bundle, dataset)


def test_content_aware_requires_inventory(tmp_path):
    bundle, dataset = make_bundle(tmp_path)
    (bundle / 'image_content.json').unlink()
    with pytest.raises(ValueError, match='requires image_content'):
        audit_image_content(bundle, dataset)


def test_real_smoke_preserves_classes_groups_and_feature_profiles(tmp_path):
    root = Path(__file__).resolve().parents[1]
    index_file = root / 'data/partitions_train_v4_content_aware/index.json'
    if not index_file.exists():
        pytest.skip("data/partitions_train_v4_content_aware/index.json not present in candidate")
    index = json.loads(index_file.read_text())
    source = root / next(p['relative_dir'] for p in index['partitions'].values()
                         if p['scenario'] == 'label_skew' and p['feature_skew'] == 'moderate')
    bundle = create_smoke_bundle(source, tmp_path / 'smoke', target_client_images=40, target_val_images=40)
    second = create_smoke_bundle(source, tmp_path / 'same', target_client_images=40, target_val_images=40)
    for name in ['global_val.csv', 'global_test.csv']:
        read = lambda path: list(csv.DictReader(path.open(encoding='utf-8')))
        selected, full = read(bundle / name), read(source / name)
        assert len({r['label'] for r in selected}) == 38
        groups = {r['group_id'] for r in selected}
        assert len(selected) == sum(r['group_id'] in groups for r in full)
        assert (bundle / name).read_bytes() == (second / name).read_bytes()
    meta = json.loads((bundle / 'fedavg_meta.json').read_text())
    assert meta['feature_skew'] == 'moderate'
    assert len(meta['client_profiles']) == 2
    config = json.loads((bundle / 'partition_config.json').read_text())
    assert config['client_profiles'] == json.loads((source / 'partition_config.json').read_text())['client_profiles'][:2]
    assert audit_image_content(bundle, root / index['dataset_path'])['pinned_content_verified']
    iid = root / next(p['relative_dir'] for p in index['partitions'].values()
                      if p['scenario'] == 'iid' and p['feature_skew'] == 'none')
    iid_bundle = create_smoke_bundle(iid, tmp_path / 'iid', target_client_images=40, target_val_images=40)
    for name in ['global_val.csv', 'global_test.csv']:
        paths = lambda p: {r['relative_path'] for r in csv.DictReader(p.open(encoding='utf-8'))}
        assert paths(bundle / name) == paths(iid_bundle / name)
