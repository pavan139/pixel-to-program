"""Paired architecture/search experiments on fixed synthetic data splits."""

from __future__ import annotations
import json
from pathlib import Path
from .attention_model import ModelConfig
from .attention_training import dataset_fingerprint, train
from .attention_inference import evaluate


def ablation(
    directory: Path,
    output: Path,
    *,
    epochs: int = 1,
    seeds: tuple[int, ...] = (17, 23),
    d_model: int = 32,
    beam_sizes: tuple[int, ...] = (1, 4),
) -> dict:
    if not seeds or len(seeds) > 5 or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("ablation requires 1..5 unique integer seeds")
    if (
        not beam_sizes
        or len(beam_sizes) > 4
        or len(set(beam_sizes)) != len(beam_sizes)
        or any(type(value) is not int or not 1 <= value <= 16 for value in beam_sizes)
    ):
        raise ValueError("beam sizes must be 1..4 unique integers in [1, 16]")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        for architecture in ("cnn_gru", "patch_transformer"):
            checkpoint = output / f"{architecture}-{seed}.pt"
            config = ModelConfig(
                architecture=architecture,
                d_model=d_model,
                heads=2,
                encoder_layers=1,
                decoder_layers=1,
                feedforward=2 * d_model,
                dropout=0.1,
            )
            training = train(directory, checkpoint, config=config, epochs=epochs, batch_size=4, seed=seed)
            for beam_size in beam_sizes:
                for constrained in (False, True):
                    test = evaluate(
                        Path(training["best_checkpoint"]), directory, beam_size=beam_size, constrained=constrained
                    )
                    results.append(
                        {
                            "seed": seed,
                            "architecture": architecture,
                            "parameters": training["parameters"],
                            "global_step": training["global_step"],
                            "dev_token_loss": training["best_dev_token_loss"],
                            "beam_size": beam_size,
                            "constrained": constrained,
                            "test": test,
                        }
                    )
    report = {
        "experiment": "paired_architecture_and_search_ablation",
        "dataset_sha256": dataset_fingerprint(directory),
        "seeds": list(seeds),
        "epochs": epochs,
        "same_examples_optimizer_schedule_and_decoding": True,
        "parameter_counts_matched": False,
        "regularization_matched": False,
        "comparability_notes": [
            "Same data, optimizer plan, token loss and decoder search; parameter counts differ.",
            "Transformer dropout is configurable; the legacy single-layer GRU baseline has no dropout.",
            "Short paired runs are descriptive pipeline evidence, not causal architecture superiority.",
        ],
        "scope": "controlled_synthetic_data; short runs do not establish useful learned accuracy",
        "results": results,
    }
    (output / "ablation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
