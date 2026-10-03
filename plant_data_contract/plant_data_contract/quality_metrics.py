"""Source-aware error metrics for the fixed 38-class plant taxonomy."""

import numpy as np

from plant_data_contract.taxonomy import PLANTVILLAGE_38_CLASSES


def summarize_quality(targets, predictions, domains, confidence, high_confidence_threshold=0.9):
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    domains = np.asarray(domains)
    confidence = np.asarray(confidence, dtype=np.float64)
    if not (len(targets) == len(predictions) == len(domains) == len(confidence)):
        raise ValueError("Prediction, target, domain and confidence lengths differ")
    if not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Invalid prediction confidence")
    if np.any((targets < 0) | (targets >= 38) | (predictions < 0) | (predictions >= 38)):
        raise ValueError("Prediction or target class outside taxonomy")

    crops = np.asarray([name.split("___", 1)[0] for name in PLANTVILLAGE_38_CLASSES])
    crop_correct = crops[targets] == crops[predictions]
    joint_correct = targets == predictions
    by_source = {}
    for source in ("plantdoc", "plantvillage"):
        mask = domains == source
        count = int(mask.sum())
        if count == 0:
            by_source[source] = {"count": 0, "crop_accuracy": None, "wrong_crop_count": 0,
                                 "right_crop_wrong_disease_count": 0, "high_confidence_wrong_rate": None}
            continue
        wrong = mask & ~joint_correct
        recalls = {}
        for class_id, name in enumerate(PLANTVILLAGE_38_CLASSES):
            support_mask = mask & (targets == class_id)
            support = int(support_mask.sum())
            if support:
                recalls[name] = {"support": support, "recall": float((predictions[support_mask] == class_id).mean())}
        by_source[source] = {
            "count": count,
            "crop_accuracy": float(crop_correct[mask].mean()),
            "wrong_crop_count": int((mask & ~crop_correct).sum()),
            "right_crop_wrong_disease_count": int((mask & crop_correct & ~joint_correct).sum()),
            "high_confidence_wrong_rate": float((wrong & (confidence >= high_confidence_threshold)).sum() / count),
            "high_confidence_wrong_count": int((wrong & (confidence >= high_confidence_threshold)).sum()),
            "per_class_recall": recalls,
        }
    return {"high_confidence_threshold": high_confidence_threshold, "by_source": by_source}
