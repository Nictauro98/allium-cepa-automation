from allium_cepa_classifier.config import AlliumCepaConfig, ProductionConfig
from allium_cepa_classifier.providers.factory import get_storage


def ensure_production_weights(cfg: ProductionConfig) -> AlliumCepaConfig:
    """Pull prod weights from object storage into the local cache, return an inference config."""
    storage = get_storage()
    cfg.weights_cache_dir.mkdir(parents=True, exist_ok=True)

    det = cfg.weights_cache_dir / "object_detection.pt"
    clf = cfg.weights_cache_dir / "classifier_calibrated.pt"
    cal = cfg.weights_cache_dir / "yolo_isotonic_calibrator.pkl"

    for key, dest in [
        (cfg.detection_key, det),
        (cfg.classifier_key, clf),
        (cfg.calibrator_key, cal),
    ]:
        if not dest.exists():
            storage.get_file(key, dest)

    return AlliumCepaConfig(
        detection_weights_path=det,
        classification_weights_path=clf,
        detection_calibrator_path=cal,
        use_cpu=True,  # HF Spaces free tier has no GPU
    )
