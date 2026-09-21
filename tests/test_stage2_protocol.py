import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from fl_training.prepare import compute_relative_paths_hash, compute_class_mapping_hash
from fl_training.stage2_protocol import validate_stage2_sweep
from fl_training.task import evaluate_model


@pytest.fixture
def protocol(tmp_path):
    names = [f'c{i:02d}' for i in range(38)]
    fields = ['relative_path', 'label', 'class_name', 'client_id', 'group_id']
    splits = {s: [dict(relative_path=f'{n}/{s}.jpg', label=str(i), class_name=n,
                      client_id=str(i % 2) if s == 'train' else '-1', group_id=f'{s}-{i}')
                   for i, n in enumerate(names)] for s in ('train', 'val', 'test')}
    def write(path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    partitions = {}
    for scenario in ('iid', 'label_skew'):
        part = tmp_path / scenario
        (part / 'clients').mkdir(parents=True)
        for i in range(2):
            write(part / 'clients' / f'client_{i:02d}.csv', [r for r in splits['train'] if r['client_id'] == str(i)])
        write(part / 'centralized_train.csv', splits['train'])
        for split in ('val', 'test'):
            write(part / f'global_{split}.csv', splits[split])
        meta = dict(group_aware=True, content_aware=True, num_clients=2, class_names=names,
                    scenario=scenario, alpha=.1, feature_skew='none', seed=42)
        (part / 'partition_config.json').write_text(json.dumps(meta))
        (part / 'image_content.json').write_text(json.dumps({r['relative_path']:r['group_id'] for rows in splits.values() for r in rows}))
        partitions[scenario] = dict(scenario=scenario, alpha=.1, feature_skew='none', split_seed=42, relative_dir=scenario)
    index = {'partitions': partitions, 'total_source_images':114,
             'class_mapping_sha256':compute_class_mapping_hash(splits['val']),
             **{f'{s}_paths_sha256':compute_relative_paths_hash([r['relative_path'] for r in splits[s]]) for s in ('val', 'test')}}
    path = tmp_path / 'index.json'; path.write_text(json.dumps(index))
    cfg = SimpleNamespace(data=SimpleNamespace(bundle_path=None, partition_index=path, split_seed=42),
                          federation=SimpleNamespace(num_clients=2), early_stopping=SimpleNamespace(enabled=False))
    spec = dict(protocol='content_aware_v4', primary_mode='fedavg', role='main', modes=['fedavg'],
                conditions=[dict(scenario=s, alpha=.1, feature_skew='none', split_seed=42) for s in partitions])
    return spec, cfg, tmp_path


def test_accepts_fedavg_and_paired_controls(protocol):
    spec, cfg, root = protocol
    assert validate_stage2_sweep(spec, cfg, root)['same_holdout_across_conditions']
    spec.update(role='controls', modes=['centralized', 'local-only'])
    assert validate_stage2_sweep(spec, cfg, root)['primary_method'] == 'FedAvg'


@pytest.mark.parametrize('mutation,match', [
    ('legacy', 'group-aware'), ('smoke', 'Smoke bundles'), ('holdout', 'locked index'),
    ('inventory', 'image inventory'), ('algorithm', 'primary method'), ('axis', 'partition axes'),
    ('mislabelled_alpha', 'non-IID axis')])
def test_rejects_invalid_protocol_before_training(protocol, mutation, match):
    spec, cfg, root = protocol
    if mutation == 'legacy':
        path = root / 'iid/partition_config.json'; meta = json.loads(path.read_text())
        meta['content_aware'] = False; path.write_text(json.dumps(meta))
    elif mutation == 'smoke': cfg.data.bundle_path = root
    elif mutation == 'holdout':
        path = root / 'label_skew/global_test.csv'
        path.write_text(path.read_text().replace('c00/test.jpg', 'c00/changed.jpg'))
    elif mutation == 'inventory':
        path = root / 'label_skew/image_content.json'; meta = json.loads(path.read_text())
        meta['c00/train.jpg'] = 'different'; path.write_text(json.dumps(meta))
    elif mutation == 'algorithm': spec['modes'] = ['fedprox']
    elif mutation == 'axis': spec['conditions'][0]['dataset_root'] = 'elsewhere'
    elif mutation == 'mislabelled_alpha':
        path = root / 'label_skew/partition_config.json'; meta = json.loads(path.read_text())
        meta['alpha'] = 100; path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=match): validate_stage2_sweep(spec, cfg, root)


def test_perfect_predictions_are_reported_without_accuracy_cap():
    labels = torch.arange(38)
    logits = torch.nn.functional.one_hot(labels, num_classes=38).float() * 10
    metrics = evaluate_model(torch.nn.Identity(), DataLoader(TensorDataset(logits, labels), batch_size=19), num_classes=38)
    assert metrics['accuracy'] == 1.0
    assert metrics['macro_f1'] == 1.0


def test_main_and_controls_use_the_same_experiment_matrix():
    root = Path(__file__).resolve().parents[1]
    specs = [yaml.safe_load((root / 'configs' / name).read_text(encoding='utf-8')) for name in
             ('stage2_fedavg_main_v4.yaml', 'stage2_controls_v4.yaml', 'sweep_content_aware_v4.yaml')]
    for field in ('base_config', 'conditions', 'seeds', 'output_root', 'sweep_id'):
        assert specs[0][field] == specs[1][field] == specs[2][field]
    assert specs[0]['modes'] == ['fedavg']
    assert set(specs[1]['modes']) == {'centralized', 'local-only'}
