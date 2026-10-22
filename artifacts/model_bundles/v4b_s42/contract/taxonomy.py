"""Taxonomy definitions and loader for PlantVillage and PlantDoc."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PLANTVILLAGE_38_CLASSES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

PLANTDOC_DETECTION_29_CLASSES: List[str] = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf",
    "Bell_pepper leaf spot",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
    "Tomato two spotted spider mites leaf",
    "grape leaf",
    "grape leaf black rot",
]

# 10 PlantVillage classes with zero real-field samples in PlantDoc
UNSUPPORTED_PLANTDOC_PV_CLASSES: List[str] = [
    "Apple___Black_rot",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___healthy",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Potato___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Target_Spot",
]


class TaxonomyContract:
    """Manages class mappings, indexing, and hierarchy."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        mappings: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.classes = list(classes or PLANTVILLAGE_38_CLASSES)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class: Dict[int, str] = {i: c for i, c in enumerate(self.classes)}
        self.mappings = mappings or {}

        # Derived crop hierarchy
        self.crops = sorted(list({c.split("___", 1)[0] for c in self.classes}))
        self.crop_to_idx = {crop: i for i, crop in enumerate(self.crops)}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def get_class_index(self, class_name: str) -> int:
        if class_name not in self.class_to_idx:
            raise KeyError(f"Class '{class_name}' not in taxonomy classes ({self.num_classes} classes).")
        return self.class_to_idx[class_name]

    def get_class_name(self, index: int) -> str:
        if index not in self.idx_to_class:
            raise IndexError(f"Index {index} out of bounds for taxonomy of size {self.num_classes}.")
        return self.idx_to_class[index]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "classes": self.classes,
            "crops": self.crops,
            "class_to_idx": self.class_to_idx,
            "unsupported_in_plantdoc": UNSUPPORTED_PLANTDOC_PV_CLASSES,
            "mappings": self.mappings,
        }


def load_taxonomy(taxonomy_json_path: Path | str) -> TaxonomyContract:
    path = Path(taxonomy_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Taxonomy JSON file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return TaxonomyContract(
        classes=data.get("classes", PLANTVILLAGE_38_CLASSES),
        mappings=data.get("mappings", {}),
    )
