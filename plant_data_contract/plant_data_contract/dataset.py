"""PyTorch Dataset implementation for Canonical Classification manifests."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from plant_data_contract.path_resolver import PathResolver
from plant_data_contract.crop_geometry import normalize_exif_orientation


class CanonicalClassificationDataset(Dataset):
    """Canonical Classification Dataset reading from release manifests.

    Supports both PlantVillage (whole leaf) and PlantDoc (materialized crop).
    Guarantees strict path resolution and source_domain isolation.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dataset_root: Union[str, Path],
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        self.samples = samples
        self.resolver = PathResolver(dataset_root)
        self.transform = transform
        self.return_metadata = return_metadata

        self.targets = [s["class_id"] for s in self.samples]
        self.domains = [s.get("source_domain", "plantvillage") for s in self.samples]
        self.group_ids = [s.get("group_id", "") for s in self.samples]
        self.sample_ids = [s.get("sample_id", "") for s in self.samples]

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        dataset_root: Union[str, Path],
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
        supervised_only: bool = True,
    ) -> "CanonicalClassificationDataset":
        """Load dataset from a JSON or JSONL manifest."""
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {path}")

        samples: List[Dict[str, Any]] = []
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        s = json.loads(line)
                        if not supervised_only or s.get("supervised_eligible", True):
                            samples.append(s)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw_samples = data.get("samples", data) if isinstance(data, dict) else data
            for s in raw_samples:
                if not supervised_only or s.get("supervised_eligible", True):
                    samples.append(s)

        return cls(
            samples=samples,
            dataset_root=dataset_root,
            transform=transform,
            return_metadata=return_metadata,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, Dict[str, Any]]]:
        sample = self.samples[index]
        resolved_path = self.resolver.resolve(sample["relative_path"], must_exist=True)

        with Image.open(resolved_path) as img:
            transposed = normalize_exif_orientation(img)
            decoded_width, decoded_height = transposed.size
            if transposed.mode in ("RGBA", "LA") or (transposed.mode == "P" and "transparency" in transposed.info):
                rgba = transposed.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (0, 0, 0))
                bg.paste(rgba, mask=rgba.split()[3])
                rgb_img = bg
            else:
                rgb_img = transposed.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(rgb_img)
        else:
            tensor = rgb_img

        class_id = sample["class_id"]
        if self.return_metadata:
            metadata = dict(sample)
            metadata["decoded_width"] = decoded_width
            metadata["decoded_height"] = decoded_height
            return tensor, class_id, metadata
        return tensor, class_id


class AuditedDomainBalancedSampler(Sampler[int]):
    """Deterministic domain sampler with per-epoch draw telemetry.

    ``with_replacement`` preserves the historical weighted-random policy.
    ``cycle_without_replacement`` is an experimental lower-repeat option that
    cycles through shuffled domain members before drawing any member again.
    """

    VALID_REPEAT_POLICIES = {"with_replacement", "cycle_without_replacement"}

    def __init__(
        self,
        dataset: CanonicalClassificationDataset,
        plantdoc_weight: float = 0.5,
        seed: int = 42,
        repeat_policy: str = "with_replacement",
    ):
        if not len(dataset):
            raise ValueError("Cannot build a sampler for an empty dataset")
        if not 0.0 <= float(plantdoc_weight) <= 1.0:
            raise ValueError("plantdoc_weight must be between 0 and 1")
        if repeat_policy not in self.VALID_REPEAT_POLICIES:
            raise ValueError(f"Unknown repeat policy: {repeat_policy}")

        self.dataset = dataset
        self.plantdoc_weight = float(plantdoc_weight)
        self.seed = int(seed)
        self.repeat_policy = repeat_policy
        self.num_samples = len(dataset)
        self.epoch = 0
        self.last_statistics: Optional[Dict[str, Any]] = None
        self._last_indices: Optional[List[int]] = None

        self._pv_indices = [i for i, domain in enumerate(dataset.domains) if domain == "plantvillage"]
        self._pd_indices = [i for i, domain in enumerate(dataset.domains) if domain == "plantdoc"]
        if not self._pv_indices or not self._pd_indices:
            self._weights = torch.full((self.num_samples,), 1.0 / self.num_samples, dtype=torch.double)
        else:
            pv_weight = (1.0 - self.plantdoc_weight) / len(self._pv_indices)
            pd_weight = self.plantdoc_weight / len(self._pd_indices)
            self._weights = torch.as_tensor(
                [pd_weight if domain == "plantdoc" else pv_weight for domain in dataset.domains],
                dtype=torch.double,
            )

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("Sampler epoch must be non-negative")
        self.epoch = int(epoch)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "epoch": self.epoch,
            "plantdoc_weight": self.plantdoc_weight,
            "repeat_policy": self.repeat_policy,
            "num_samples": self.num_samples,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        expected = {
            "seed": self.seed,
            "plantdoc_weight": self.plantdoc_weight,
            "repeat_policy": self.repeat_policy,
            "num_samples": self.num_samples,
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(f"Sampler resume mismatch for {field}: {state.get(field)!r} != {value!r}")
        self.set_epoch(int(state.get("epoch", 0)))

    @staticmethod
    def _cycle(indices: List[int], count: int, generator: torch.Generator) -> List[int]:
        draws: List[int] = []
        domain_indices = torch.as_tensor(indices, dtype=torch.long)
        while len(draws) < count:
            order = torch.randperm(len(indices), generator=generator)
            remaining = count - len(draws)
            draws.extend(domain_indices[order[:remaining]].tolist())
        return draws

    def _draw_cycle_without_replacement(self, generator: torch.Generator) -> List[int]:
        if not self._pv_indices or not self._pd_indices:
            return self._cycle(list(range(self.num_samples)), self.num_samples, generator)

        plantdoc_count = int(round(self.num_samples * self.plantdoc_weight))
        plantvillage_count = self.num_samples - plantdoc_count
        draws = (
            self._cycle(self._pd_indices, plantdoc_count, generator)
            + self._cycle(self._pv_indices, plantvillage_count, generator)
        )
        order = torch.randperm(len(draws), generator=generator).tolist()
        return [draws[index] for index in order]

    def _summarize(self, indices: List[int]) -> Dict[str, Any]:
        sample_counts = Counter(indices)
        domain_counts = Counter(self.dataset.domains[index] for index in indices)
        class_counts = Counter(str(self.dataset.targets[index]) for index in indices)
        scene_counts = Counter(
            self.dataset.group_ids[index] or f"__missing_group__:{index}"
            for index in indices
        )
        repeat_histogram = Counter(str(count) for count in sample_counts.values())
        return {
            "schema_version": 1,
            "epoch": self.epoch,
            "seed": self.seed,
            "repeat_policy": self.repeat_policy,
            "requested_plantdoc_weight": self.plantdoc_weight,
            "total_draws": len(indices),
            "draws_by_domain": dict(sorted(domain_counts.items())),
            "draws_by_class_id": dict(sorted(class_counts.items(), key=lambda item: int(item[0]))),
            "draws_by_scene": dict(sorted(scene_counts.items())),
            "sample_repeat_summary": {
                "unique_samples_drawn": len(sample_counts),
                "repeated_draws": len(indices) - len(sample_counts),
                "max_draws_per_sample": max(sample_counts.values(), default=0),
                "draw_count_histogram": dict(sorted(repeat_histogram.items(), key=lambda item: int(item[0]))),
            },
            "scene_repeat_summary": {
                "unique_scenes_drawn": len(scene_counts),
                "max_draws_per_scene": max(scene_counts.values(), default=0),
            },
        }

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.repeat_policy == "with_replacement":
            indices = torch.multinomial(
                self._weights,
                self.num_samples,
                replacement=True,
                generator=generator,
            ).tolist()
        else:
            indices = self._draw_cycle_without_replacement(generator)
        self._last_indices = indices
        self.last_statistics = self._summarize(indices)
        return iter(indices)

    def statistics_for_draw_count(self, draw_count: int) -> Dict[str, Any]:
        if self._last_indices is None:
            raise RuntimeError("Sampler statistics are unavailable before iteration")
        if not 0 <= int(draw_count) <= len(self._last_indices):
            raise ValueError("draw_count is outside the generated epoch sequence")
        return self._summarize(self._last_indices[:int(draw_count)])


def build_domain_balanced_sampler(
    dataset: CanonicalClassificationDataset,
    plantdoc_weight: float = 0.5,
    seed: int = 42,
    repeat_policy: str = "with_replacement",
) -> AuditedDomainBalancedSampler:
    """Create the historical weighted sampler with deterministic audit support."""
    return AuditedDomainBalancedSampler(
        dataset=dataset,
        plantdoc_weight=plantdoc_weight,
        seed=seed,
        repeat_policy=repeat_policy,
    )
