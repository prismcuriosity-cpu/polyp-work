from .losses import TotalLoss, PinballLoss, SILogLoss, IlluminationDeclineLoss


def get_trainer():
    """Lazy import to avoid pulling transformers into test environments."""
    from .trainer import ConformalDepthTrainer
    return ConformalDepthTrainer


__all__ = [
    "TotalLoss", "PinballLoss", "SILogLoss", "IlluminationDeclineLoss",
    "get_trainer",
]
