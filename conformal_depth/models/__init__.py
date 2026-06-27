def __getattr__(name):
    """Lazy-import all model classes so that transformers/peft are only
    required when a model is actually instantiated, not at import time.
    This keeps the test suite fast and dependency-free for pure-logic tests."""
    _map = {
        "ConformalDepthModel": ("conformal_depth.models.conformal_depth", "ConformalDepthModel"),
        "DepthAnythingZeroShot": ("conformal_depth.models.baselines", "DepthAnythingZeroShot"),
        "ZoeDepthWrapper":       ("conformal_depth.models.baselines", "ZoeDepthWrapper"),
        "MonoViTWrapper":        ("conformal_depth.models.baselines", "MonoViTWrapper"),
        "MCDropoutWrapper":      ("conformal_depth.models.baselines", "MCDropoutWrapper"),
        "DeepEnsemble":          ("conformal_depth.models.baselines", "DeepEnsemble"),
        "DVLoRAAdapter":         ("conformal_depth.models.lora",      "DVLoRAAdapter"),
    }
    if name in _map:
        import importlib
        module_path, attr = _map[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'conformal_depth.models' has no attribute {name!r}")


__all__ = [
    "ConformalDepthModel",
    "DepthAnythingZeroShot",
    "ZoeDepthWrapper",
    "MonoViTWrapper",
    "MCDropoutWrapper",
    "DeepEnsemble",
    "DVLoRAAdapter",
]
