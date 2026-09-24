"""Verify real image bytes, labels and ownership, independently of group IDs/cache."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def audit_image_content(
    partition_dir: Path,
    dataset_root: Path,
    observed_hashes: dict[str, str] | None = None,
    expected_hashes: dict[str, str] | None = None,
    hashes: dict[str, str] | None = None,
):
    dataset_root = Path(dataset_root)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {dataset_root}")

    if expected_hashes is None and hashes is not None:
        expected_hashes = hashes

    def rows(path):
        with path.open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))

    metadata = next((partition_dir / name for name in ('partition_config.json', 'fedavg_meta.json')
                     if (partition_dir / name).exists()), None)
    classes = json.loads(metadata.read_text(encoding='utf-8')).get('class_names', []) if metadata else []
    if len(classes) != 38 or len(set(classes)) != 38:
        raise ValueError('Content audit requires the complete unique 38-class mapping')
    owners = {}
    seen_paths = set()
    expected_file = partition_dir / 'image_content.json'
    expected = expected_hashes
    if expected is None and expected_file.exists():
        expected = json.loads(expected_file.read_text(encoding='utf-8'))
    if metadata and json.loads(metadata.read_text(encoding='utf-8')).get('content_aware') and expected is None:
        raise ValueError('Content-aware training requires image_content.json; rerun prepare-data --strict')
    actual = {}
    manifests = [(p.stem, p) for p in sorted((partition_dir / 'clients').glob('client_*.csv'))]
    manifests += [(s, partition_dir / f'global_{s}.csv') for s in ('val', 'test')]
    resolved_root = dataset_root.resolve()
    for owner, manifest in manifests:
        for row in rows(manifest):
            rel = '/'.join(row['relative_path'].replace('\\', '/').split('/')[-2:])
            if rel in seen_paths:
                raise ValueError(f'Duplicate canonical image path: {rel}')
            seen_paths.add(rel)
            label = int(row['label'])
            if not 0 <= label < len(classes) or classes[label] != row['class_name'] or rel.split('/')[0] != row['class_name']:
                raise ValueError(f'Image label/class mapping mismatch: {rel}')
            if observed_hashes is not None:
                if rel not in observed_hashes:
                    raise FileNotFoundError(f'Image file not found in observed dataset: {rel}')
                digest = observed_hashes[rel]
            else:
                if ".." in rel:
                    raise ValueError(f'Image path outside dataset: {rel}')
                path = resolved_root / rel
                try:
                    data = path.read_bytes()
                except FileNotFoundError:
                    raise FileNotFoundError(f'Image file not found on disk: {path}')
                digest = hashlib.sha256(data).hexdigest()
            actual[rel] = digest
            if expected is not None and expected.get(rel) != digest:
                raise ValueError(f'Image bytes differ from pinned inventory: {rel}')
            previous = owners.setdefault(digest, (owner, label))
            if previous != (owner, label):
                raise ValueError(f'Identical image bytes across splits/clients or labels: {rel}: {previous} vs {(owner, label)}')
    if expected is not None and set(expected) != seen_paths:
        raise ValueError('Pinned inventory does not match manifest coverage')
    return {'verified_images': len(seen_paths), 'unique_byte_hashes': len(owners),
            'cross_owner_byte_duplicates': 0, 'pinned_content_verified': expected is not None,
            'content_sha256': hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(),
            'scope': 'Exact bytes and labels; not perceptual similarity or unknown leaf identities'}
