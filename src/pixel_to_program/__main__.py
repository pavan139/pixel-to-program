from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .data import generate_scene, write_dataset
from .scene import (
    Scene,
    parse_program,
    ppm_bytes,
    reconstruction_metrics,
    render_rgb,
    render_svg,
    scene_hash,
    serialize,
)


def evaluate(seed: int = 17, cases_per_split: int = 12) -> dict:
    if not 1 <= cases_per_split <= 10000:
        raise ValueError("cases_per_split must be in [1, 10000]")
    cases = []
    for split_index, split in enumerate(("train", "dev", "test")):
        for index in range(cases_per_split):
            scene = generate_scene(split, seed + split_index * 1000000 + index)
            program = serialize(scene)
            reconstructed = parse_program(program)
            expected = render_rgb(scene)
            cases.append(
                {
                    "id": scene_hash(scene),
                    "split": split,
                    "program": program,
                    "oracle_roundtrip": reconstruction_metrics(expected, render_rgb(reconstructed)),
                    "blank_image_baseline": reconstruction_metrics(expected, render_rgb(Scene(()))),
                }
            )
    return {
        "schema_version": 1,
        "experiment": "synthetic_codec_and_renderer_validation",
        "backend": "ground_truth_program_roundtrip",
        "learned_model_evaluated": False,
        "seed": seed,
        "denominator": len(cases),
        "cases_per_split": cases_per_split,
        "oracle_exact_raster_matches": sum(case["oracle_roundtrip"]["exact_raster_match"] for case in cases),
        "cases": cases,
        "limitations": [
            "Ground-truth programs are supplied to the codec. This is not image-to-program prediction accuracy.",
            "Blank backgrounds inflate raw pixel agreement; foreground IoU is reported separately.",
            "Training and learned-model evaluation require the optional torch dependency and an actual checkpoint.",
        ],
    }


def search_demo() -> dict:
    from .search import beam_search

    def backend(prefixes):
        rows = []
        for prefix in prefixes:
            values = (
                (0.6, 0.4, 0)
                if not prefix
                else (0, 0.5, 0.5)
                if prefix == (0,)
                else (0.01, 0, 0.99)
                if prefix == (1,)
                else (0, 0, 1)
            )
            rows.append([math.log(value) if value else -math.inf for value in values])
        return rows

    return {
        "experiment": "likelihood_beam_search",
        "backend": "synthetic_probability_fixture",
        "learned_model_evaluated": False,
        "vocabulary": ["A", "B", "EOS"],
        "exhaustive_best_completed_probability": 0.396,
        "searches": [
            {"beam_size": width, **beam_search(backend, eos_id=2, beam_size=width, max_tokens=3)} for width in (1, 2, 4)
        ],
        "limitations": ["This verifies search mechanics, not image-to-program quality."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic image-to-program dataset and optional CNN/GRU training")
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search-demo", help="verify likelihood search on a small exact probability fixture")
    search.add_argument("--output", type=Path)
    for name in ("demo", "evaluate", "dataset", "train", "predict", "evaluate-model"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path)
        if name in ("demo", "evaluate", "dataset", "train"):
            command.add_argument("--seed", type=int, default=17)
        if name == "demo":
            command.add_argument("--directory", type=Path, default=Path("artifacts/demo"))
        if name == "evaluate":
            command.add_argument("--cases-per-split", type=int, default=12)
        if name == "dataset":
            command.add_argument("--directory", type=Path, default=Path("data/generated"))
            command.add_argument("--train", type=int, default=256)
            command.add_argument("--dev", type=int, default=64)
            command.add_argument("--test", type=int, default=64)
        if name in ("train", "evaluate-model"):
            command.add_argument("--data-dir", type=Path, default=Path("data/generated"))
        if name in ("train", "predict", "evaluate-model"):
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--device", default="cpu")
        if name in ("predict", "evaluate-model"):
            command.add_argument(
                "--constrained", action="store_true", help="mask tokens using the executable scene grammar"
            )
            command.add_argument("--max-tokens", type=int, default=32)
        if name == "train":
            command.add_argument("--epochs", type=int, default=10)
            command.add_argument("--batch-size", type=int, default=32)
            command.add_argument("--hidden", type=int, default=128)
        if name == "predict":
            command.add_argument("--image", type=Path, required=True)
        if name == "evaluate-model":
            command.add_argument("--split", choices=("train", "dev", "test"), default="test")
    for name in ("train-system", "predict-system", "evaluate-system", "ablation"):
        command = sub.add_parser(name, help="attention/GRU experiments with shared optimization and decoding")
        command.add_argument("--output", type=Path)
        if name in ("train-system", "evaluate-system", "ablation"):
            command.add_argument("--data-dir", type=Path, default=Path("data/generated"))
        if name != "ablation":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--device", default="cpu")
        if name in ("predict-system", "evaluate-system"):
            command.add_argument("--beam-size", type=int, default=4)
            command.add_argument("--max-tokens", type=int, default=32)
            command.add_argument("--length-penalty", type=float, default=0.0)
            command.add_argument("--unconstrained", action="store_true")
        if name == "predict-system":
            command.add_argument("--image", type=Path, required=True)
        if name == "evaluate-system":
            command.add_argument("--split", choices=("train", "dev", "test"), default="test")
            command.add_argument("--allow-dataset-shift", action="store_true")
        if name in ("train-system", "ablation"):
            command.add_argument("--epochs", type=int, default=10 if name == "train-system" else 1)
            command.add_argument("--d-model", type=int, default=64 if name == "train-system" else 32)
        if name == "train-system":
            command.add_argument(
                "--architecture", choices=("patch_transformer", "cnn_gru"), default="patch_transformer"
            )
            command.add_argument("--heads", type=int, default=4)
            command.add_argument("--encoder-layers", type=int, default=1)
            command.add_argument("--decoder-layers", type=int, default=2)
            command.add_argument("--feedforward", type=int, default=128)
            command.add_argument("--patch-size", type=int, default=4)
            command.add_argument("--dropout", type=float, default=0.1)
            command.add_argument("--batch-size", type=int, default=16)
            command.add_argument("--learning-rate", type=float, default=0.001)
            command.add_argument("--weight-decay", type=float, default=0.01)
            command.add_argument("--smoothing", type=float, default=0.0)
            command.add_argument("--warmup-steps", type=int, default=0)
            command.add_argument("--seed", type=int, default=17)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--stop-after-epochs", type=int)
        if name == "ablation":
            command.add_argument("--run-dir", type=Path, default=Path("artifacts/attention-ablation"))
            command.add_argument("--seeds", type=int, nargs="+", default=[17, 23])
            command.add_argument("--beam-sizes", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    if args.command == "search-demo":
        result = search_demo()
    elif args.command == "demo":
        scene = generate_scene("test", args.seed)
        args.directory.mkdir(parents=True, exist_ok=True)
        (args.directory / "scene.ppm").write_bytes(ppm_bytes(scene))
        (args.directory / "scene.svg").write_text(render_svg(scene), encoding="utf-8")
        (args.directory / "scene.txt").write_text(serialize(scene) + "\n", encoding="utf-8")
        result = {
            "backend": "synthetic_ground_truth",
            "learned_model_evaluated": False,
            "directory": str(args.directory),
            "program": serialize(scene),
            "roundtrip": reconstruction_metrics(render_rgb(scene), render_rgb(parse_program(serialize(scene)))),
        }
    elif args.command == "evaluate":
        result = evaluate(args.seed, args.cases_per_split)
    elif args.command == "dataset":
        result = write_dataset(args.directory, train=args.train, dev=args.dev, test=args.test, seed=args.seed)
    elif args.command in {"train-system", "predict-system", "evaluate-system", "ablation"}:
        from . import transformer

        try:
            if args.command == "train-system":
                config = transformer.ModelConfig(
                    architecture=args.architecture,
                    d_model=args.d_model,
                    heads=args.heads,
                    encoder_layers=args.encoder_layers,
                    decoder_layers=args.decoder_layers,
                    feedforward=args.feedforward,
                    patch_size=args.patch_size,
                    dropout=args.dropout,
                )
                result = transformer.train(
                    args.data_dir,
                    args.checkpoint,
                    config=config,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    smoothing=args.smoothing,
                    warmup_steps=args.warmup_steps,
                    seed=args.seed,
                    device=args.device,
                    resume=args.resume,
                    stop_after_epochs=args.stop_after_epochs,
                )
            elif args.command == "predict-system":
                result = transformer.predict(
                    args.checkpoint,
                    args.image,
                    device=args.device,
                    beam_size=args.beam_size,
                    max_tokens=args.max_tokens,
                    length_penalty=args.length_penalty,
                    constrained=not args.unconstrained,
                )
            elif args.command == "evaluate-system":
                result = transformer.evaluate(
                    args.checkpoint,
                    args.data_dir,
                    split=args.split,
                    device=args.device,
                    beam_size=args.beam_size,
                    max_tokens=args.max_tokens,
                    length_penalty=args.length_penalty,
                    constrained=not args.unconstrained,
                    allow_dataset_shift=args.allow_dataset_shift,
                )
            else:
                result = transformer.ablation(
                    args.data_dir,
                    args.run_dir,
                    epochs=args.epochs,
                    seeds=tuple(args.seeds),
                    d_model=args.d_model,
                    beam_sizes=tuple(args.beam_sizes),
                )
        except (ValueError, RuntimeError, OSError) as error:
            parser.error(str(error))
    else:
        from .model import evaluate_model, predict, train

        if args.command == "train":
            result = train(
                args.data_dir,
                args.checkpoint,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden=args.hidden,
                seed=args.seed,
                device=args.device,
            )
        elif args.command == "predict":
            result = predict(
                args.checkpoint,
                args.image,
                device=args.device,
                constrained=args.constrained,
                max_tokens=args.max_tokens,
            )
        else:
            result = evaluate_model(
                args.checkpoint,
                args.data_dir,
                split=args.split,
                device=args.device,
                constrained=args.constrained,
                max_tokens=args.max_tokens,
            )
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
