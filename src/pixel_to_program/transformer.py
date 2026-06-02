"""Public attention-system API; optional torch execution stays lazy."""

from .attention_model import ModelConfig, build_model, token_loss
from .attention_training import FORMAT_VERSION, dataset_fingerprint, load_checkpoint, train
from .attention_inference import evaluate, generate, predict
from .experiments import ablation

__all__ = [
    "ModelConfig",
    "build_model",
    "token_loss",
    "FORMAT_VERSION",
    "dataset_fingerprint",
    "load_checkpoint",
    "train",
    "evaluate",
    "generate",
    "predict",
    "ablation",
]
