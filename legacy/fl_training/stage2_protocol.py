"""Fail closed on mixed/legacy data protocols before a v4 FedAvg sweep starts."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

from .prepare import audit_manifest_directory, compute_relative_paths_hash, compute_class_mapping_hash


def validate_stage2_sweep(spec, cfg, package_root: Path):
    if spec.get('protocol') != 'content_aware_v4':
        raise ValueError('Unknown Stage-2 sweep protocol')
    roles = {'main': {'fedavg'}, 'controls': {'centralized', 'local-only'},
             'comparison': {'fedavg', 'centralized', 'local-only'}}
    if spec.get('primary_mode') != 'fedavg' or set(spec.get('modes', [])) != roles.get(spec.get('role', 'main')):
        raise ValueError('Stage-2 v4 requires FedAvg as the primary method')
    if cfg.data.bundle_path is not None:
        raise ValueError('Smoke bundles cannot be used as Stage-2 research results')
    if cfg.early_stopping.enabled:
        raise ValueError('Stage-2 comparison requires a fixed common round budget')
    index = json.loads(cfg.data.partition_index.read_text(encoding='utf-8'))
    if not spec.get('conditions') or not index.get('partitions'):
        raise ValueError('Stage-2 requires a non-empty experiment matrix and partition index')
    reference = None
    results = []
    for condition in spec['conditions']:
        if set(condition) - {'scenario', 'alpha', 'quantity_alpha', 'feature_skew', 'split_seed'}:
            raise ValueError('Conditions may only vary partition axes; keep the trainer fixed')
        selector = {'feature_skew': 'none', 'split_seed': cfg.data.split_seed, **condition}
        matches = [p for p in index['partitions'].values() if all(
            value is None or p.get(key) == value for key, value in selector.items())]
        if len(matches) != 1:
            raise ValueError(f'Condition must resolve to exactly one partition: {condition}')
        part = package_root / matches[0]['relative_dir']
        meta = json.loads((part / 'partition_config.json').read_text(encoding='utf-8'))
        for key, value in selector.items():
            stored_key = 'seed' if key == 'split_seed' else key
            if value is not None and meta.get(stored_key) != value:
                raise ValueError(f'Partition metadata differs from requested non-IID axis: {key}')
        if not meta.get('group_aware') or not meta.get('content_aware') or meta.get('smoke'):
            raise ValueError('Stage-2 v4 requires group-aware and content-aware full partitions')
        if meta['num_clients'] != cfg.federation.num_clients:
            raise ValueError('Client counts differ across the experiment protocol')
        inventory = json.loads((part / 'image_content.json').read_text(encoding='utf-8'))
        audit = audit_manifest_directory(part, total_source_images=index['total_source_images'])
        if audit['num_clients'] != cfg.federation.num_clients:
            raise ValueError('Actual client manifests differ from the declared client count')
        signature = {'classes': meta['class_names'],
                     'image_inventory': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()}
        for split in ('val', 'test'):
            with (part / f'global_{split}.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            paths_hash = compute_relative_paths_hash([row['relative_path'] for row in rows])
            if paths_hash != index[f'{split}_paths_sha256']:
                raise ValueError(f'Global {split} differs from the locked index')
            if compute_class_mapping_hash(rows) != index['class_mapping_sha256']:
                raise ValueError('Held-out class mapping differs from the index')
            canonical = sorted((r['relative_path'], r['label'], r['class_name'], r['group_id']) for r in rows)
            signature[split] = hashlib.sha256(json.dumps(canonical).encode()).hexdigest()
        if reference is not None and signature != reference:
            raise ValueError('Conditions must share validation/test, class mapping and image inventory')
        reference = signature
        results.append({'condition': condition, 'partition': str(part), 'audit': audit})
    return {'primary_method': 'FedAvg', 'algorithm_extension_methods': [],
            'same_holdout_across_conditions': True, 'conditions': results,
            'stage1_historical_comparison': 'reference_only_until_protocols_match',
            'accuracy_cap': None, 'scientific_stage2_complete': False,
            'scope': 'Full manifest preflight; actual image bytes are checked again before each training run'}
