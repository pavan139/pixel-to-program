"""Image-memory reuse, batched likelihood search and held-out evaluation."""

from __future__ import annotations
from dataclasses import asdict
import hashlib
from pathlib import Path
from .attention_model import ModelConfig
from .attention_training import dataset_fingerprint, load_checkpoint
from .data import HELDOUT, load_rows
from .decoding import allowed_program_tokens, primitive_metrics
from .model import MAX_TOKENS, TOKEN_TO_ID, VOCABULARY, _torch, decode, image_tensor
from .scene import Scene, parse_program, read_ppm, reconstruction_metrics, render_rgb
from .search import beam_search


def generate(
    model,
    config: ModelConfig,
    image,
    *,
    beam_size: int = 4,
    max_tokens: int = MAX_TOKENS,
    length_penalty: float = 0.0,
    constrained: bool = True,
) -> dict:
    torch = _torch()
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS:
        raise ValueError("model generation budget must be 1..32 tokens")
    model.eval()
    with torch.inference_mode():
        memory = model.encode_images(image) if config.architecture == "patch_transformer" else model.encoder(image)
    if image.shape[0] != 1:
        raise ValueError("beam generation accepts one image at a time")

    def probabilities(prefixes):
        ids = torch.tensor(
            [[TOKEN_TO_ID["<bos>"], *prefix] for prefix in prefixes], dtype=torch.long, device=image.device
        )
        with torch.inference_mode():
            if config.architecture == "patch_transformer":
                logits = model.decode_prefix(memory.expand(len(prefixes), -1, -1), ids)[:, -1]
            else:
                hidden = memory.expand(len(prefixes), -1).unsqueeze(0).contiguous()
                sequence, _ = model.decoder(model.embedding(ids), hidden)
                logits = model.output(sequence[:, -1])
            return torch.log_softmax(logits.float(), dim=-1).cpu().tolist()

    def allowed(prefix):
        return [TOKEN_TO_ID[token] for token in allowed_program_tokens([VOCABULARY[index] for index in prefix])]

    result = beam_search(
        probabilities,
        eos_id=TOKEN_TO_ID["<eos>"],
        beam_size=beam_size,
        max_tokens=max_tokens,
        length_penalty=length_penalty,
        allowed_tokens=allowed if constrained else None,
    )
    result["image_encoder_calls"] = 1
    result["visual_tokens"] = (32 // config.patch_size) ** 2 if config.architecture == "patch_transformer" else 1
    result["constrained"] = constrained
    return result


def predict(
    checkpoint: Path,
    image: Path,
    *,
    device: str = "cpu",
    beam_size: int = 4,
    max_tokens: int = MAX_TOKENS,
    length_penalty: float = 0.0,
    constrained: bool = True,
) -> dict:
    model, config, saved = load_checkpoint(checkpoint, device, selection="best")
    result = generate(
        model,
        config,
        image_tensor(read_ppm(image)).unsqueeze(0).to(device),
        beam_size=beam_size,
        max_tokens=max_tokens,
        length_penalty=length_penalty,
        constrained=constrained,
    )
    try:
        program = decode(result["tokens"])
        parse_program(program)
        valid, error = True, None
    except ValueError as exception:
        program, valid, error = None, False, str(exception)
    return {
        "architecture": config.architecture,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "selected_epoch": saved["best_epoch"],
        "checkpoint_selection": "best",
        "prediction": program,
        "valid_program": valid,
        "validation_error": error,
        "search": result,
        "learned_model_evaluated": True,
    }


def evaluate(
    checkpoint: Path,
    directory: Path,
    *,
    split: str = "test",
    device: str = "cpu",
    beam_size: int = 4,
    max_tokens: int = MAX_TOKENS,
    length_penalty: float = 0.0,
    constrained: bool = True,
    allow_dataset_shift: bool = False,
) -> dict:
    model, config, saved = load_checkpoint(checkpoint, device, selection="best")
    rows = load_rows(directory, split)
    fingerprint = dataset_fingerprint(directory)
    if fingerprint != saved["dataset_sha256"] and not allow_dataset_shift:
        raise ValueError("evaluation dataset changed; explicitly allow and report dataset shift")
    if split == "test" and {row["id"] for row in rows} & (set(saved["train_ids"]) | set(saved["dev_ids"])):
        raise ValueError("held-out evaluation overlaps training/development scenes")
    cases = []
    for row in rows:
        image = read_ppm(directory / row["image"])
        result = generate(
            model,
            config,
            image_tensor(image).unsqueeze(0).to(device),
            beam_size=beam_size,
            max_tokens=max_tokens,
            length_penalty=length_penalty,
            constrained=constrained,
        )
        try:
            program = decode(result["tokens"])
            scene = parse_program(program)
            metrics = reconstruction_metrics(image, render_rgb(scene))
            valid, error = True, None
        except ValueError as exception:
            scene, program, valid, error = Scene(()), None, False, str(exception)
            metrics = {"exact_raster_match": False, "foreground_iou": 0.0, "pixel_agreement": 0.0, "rgb_mae": 1.0}
        cases.append(
            {
                "id": row["id"],
                "prediction": program,
                "expected_program": row["program"],
                "valid_program": valid,
                "validation_error": error,
                "search": result,
                **metrics,
                **primitive_metrics(parse_program(row["program"]), scene, HELDOUT.get(split)),
            }
        )
    target_shapes = sum(len(parse_program(row["program"]).shapes) for row in rows)
    heldout_count = sum(case.get("heldout_expected_count", 0) for case in cases)
    true_positive = sum(case["primitive_true_positive"] for case in cases)
    false_positive = sum(case["primitive_false_positive"] for case in cases)
    false_negative = sum(case["primitive_false_negative"] for case in cases)
    return {
        "architecture": config.architecture,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "selected_epoch": saved["best_epoch"],
        "checkpoint_selection": "best",
        "config": asdict(config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "learned_model_evaluated": True,
        "scope": "synthetic_geometric_scenes",
        "dataset_sha256": fingerprint,
        "training_dataset_sha256": saved["dataset_sha256"],
        "dataset_shift": fingerprint != saved["dataset_sha256"],
        "held_out": split == "test",
        "split": split,
        "denominator": len(cases),
        "beam_size": beam_size,
        "length_penalty": length_penalty,
        "constrained": constrained,
        "valid_program_rate": sum(case["valid_program"] for case in cases) / len(cases),
        "exact_raster_rate": sum(case["exact_raster_match"] for case in cases) / len(cases),
        "mean_foreground_iou": sum(case["foreground_iou"] for case in cases) / len(cases),
        "mean_rgb_mae": sum(case["rgb_mae"] for case in cases) / len(cases),
        "primitive_precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0,
        "primitive_recall": true_positive / target_shapes,
        "primitive_f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative),
        "heldout_exact_recall": sum(case.get("heldout_exact_matches", 0) for case in cases) / heldout_count
        if heldout_count
        else None,
        "cases": cases,
    }
