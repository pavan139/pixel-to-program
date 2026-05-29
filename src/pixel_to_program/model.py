"""Optional real CNN/GRU training. All torch imports occur after opt-in calls."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from .data import HELDOUT, load_rows
from .decoding import allowed_program_tokens, primitive_metrics
from .scene import Scene, parse_program, read_ppm, reconstruction_metrics, render_rgb

VOCABULARY = ["<pad>", "<bos>", "<eos>", "SCENE", "END", "RECT", "ELLIPSE", "RED", "BLUE", "GREEN"] + [
    str(i) for i in range(33)
]
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCABULARY)}
MAX_TOKENS = 32


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install optional training dependencies with pip install -e '.[train]'") from error
    return torch


def encode(program: str) -> list[int]:
    parse_program(program)
    result = [TOKEN_TO_ID["<bos>"]] + [TOKEN_TO_ID[token] for token in program.split()] + [TOKEN_TO_ID["<eos>"]]
    if len(result) > MAX_TOKENS:
        raise ValueError("encoded program exceeds model token budget")
    return result


def decode(ids: list[int], *, require_eos: bool = True) -> str:
    if not isinstance(ids, list) or len(ids) > MAX_TOKENS + 1:
        raise ValueError("token sequence exceeds the bounded decoder contract")
    tokens = []
    for index, token_id in enumerate(ids):
        if type(token_id) is not int or not 0 <= token_id < len(VOCABULARY):
            raise ValueError("model emitted an unknown token ID")
        token = VOCABULARY[token_id]
        if token == "<eos>":
            return " ".join(tokens)
        if token == "<bos>" and index == 0:
            continue
        if token in ("<bos>", "<pad>"):
            raise ValueError("BOS/padding inside an unfinished generated program")
        tokens.append(token)
    if require_eos:
        raise ValueError("generation exhausted its token budget without EOS")
    return " ".join(tokens)


def _build_model(hidden: int = 128):
    torch = _torch()
    nn = torch.nn

    class EncoderDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, hidden),
                nn.Tanh(),
            )
            self.embedding = nn.Embedding(len(VOCABULARY), hidden, padding_idx=0)
            self.decoder = nn.GRU(hidden, hidden, batch_first=True)
            self.output = nn.Linear(hidden, len(VOCABULARY))

        def forward(self, images, input_ids):
            state = self.encoder(images).unsqueeze(0)
            sequence, _ = self.decoder(self.embedding(input_ids), state)
            return self.output(sequence)

        def generate(self, images, *, constrained: bool = False, max_tokens: int = MAX_TOKENS):
            if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS:
                raise ValueError(f"generation token budget must be in [1, {MAX_TOKENS}]")
            state = self.encoder(images).unsqueeze(0)
            token = torch.full((images.shape[0], 1), TOKEN_TO_ID["<bos>"], device=images.device, dtype=torch.long)
            results = []
            histories = [[] for _ in range(images.shape[0])]
            finished = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
            for _ in range(max_tokens):
                sequence, state = self.decoder(self.embedding(token), state)
                logits = self.output(sequence[:, -1:])
                if constrained:
                    for row, history in enumerate(histories):
                        allowed = [TOKEN_TO_ID[value] for value in allowed_program_tokens(history)]
                        mask = torch.full_like(logits[row], float("-inf"))
                        mask[:, allowed] = logits[row, :, allowed]
                        logits[row] = mask
                token = logits.argmax(dim=-1)
                token[finished] = TOKEN_TO_ID["<pad>"]
                results.append(token)
                for row, value in enumerate(token[:, 0].tolist()):
                    histories[row].append(VOCABULARY[value])
                finished |= token[:, 0] == TOKEN_TO_ID["<eos>"]
                if bool(finished.all()):
                    break
            return torch.cat(results, dim=1)

    return EncoderDecoder()


def image_tensor(rgb: bytes):
    torch = _torch()
    if len(rgb) != 32 * 32 * 3:
        raise ValueError("expected 32x32 RGB")
    return torch.tensor(list(rgb), dtype=torch.float32).reshape(32, 32, 3).permute(2, 0, 1) / 255.0


def _batch(directory: Path, rows: list[dict], device: str):
    torch = _torch()
    images = torch.stack([image_tensor(read_ppm(directory / row["image"])) for row in rows]).to(device)
    encoded = [encode(row["program"]) for row in rows]
    length = max(map(len, encoded))
    tokens = torch.tensor([ids + [0] * (length - len(ids)) for ids in encoded], dtype=torch.long, device=device)
    return images, tokens


def train(
    directory: Path,
    checkpoint: Path,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    hidden: int = 128,
    seed: int = 17,
    device: str = "cpu",
) -> dict:
    if (
        any(type(value) is not int for value in (epochs, batch_size, hidden, seed))
        or not 1 <= epochs <= 1000
        or not 1 <= batch_size <= 512
        or not 16 <= hidden <= 1024
    ):
        raise ValueError("training settings exceed explicit budgets")
    torch = _torch()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    rows = load_rows(directory, "train")
    dev = load_rows(directory, "dev")
    model = _build_model(hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    history = []
    best_dev = float("inf")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256((directory / "manifest.jsonl").read_bytes()).hexdigest()
    for epoch in range(epochs):
        model.train()
        order = rows[:]
        rng.shuffle(order)
        training_loss_sum, training_tokens = 0.0, 0
        for start in range(0, len(order), batch_size):
            images, tokens = _batch(directory, order[start : start + batch_size], device)
            optimizer.zero_grad()
            logits = model(images, tokens[:, :-1])
            token_count = int((tokens[:, 1:] != 0).sum())
            loss_sum = loss_fn(logits.reshape(-1, len(VOCABULARY)), tokens[:, 1:].reshape(-1))
            loss = loss_sum / token_count
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_loss_sum += loss_sum.item()
            training_tokens += token_count
        model.eval()
        dev_loss_sum, dev_tokens = 0.0, 0
        with torch.inference_mode():
            for start in range(0, len(dev), batch_size):
                images, tokens = _batch(directory, dev[start : start + batch_size], device)
                logits = model(images, tokens[:, :-1])
                dev_loss_sum += loss_fn(logits.reshape(-1, len(VOCABULARY)), tokens[:, 1:].reshape(-1)).item()
                dev_tokens += int((tokens[:, 1:] != 0).sum())
        dev_loss = dev_loss_sum / dev_tokens
        if not math.isfinite(dev_loss):
            raise ValueError("nonfinite development loss")
        history.append(
            {
                "epoch": epoch + 1,
                "train_token_mean_loss": training_loss_sum / training_tokens,
                "dev_token_mean_loss": dev_loss,
                "train_target_tokens": training_tokens,
                "dev_target_tokens": dev_tokens,
            }
        )
        if dev_loss < best_dev:
            best_dev = dev_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hidden": hidden,
                    "vocabulary": VOCABULARY,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "manifest_sha256": manifest_hash,
                },
                checkpoint,
            )
    result = {
        "backend": "trained_cnn_gru",
        "train_examples": len(rows),
        "dev_examples": len(dev),
        "checkpoint": str(checkpoint),
        "selected_by": "minimum_dev_teacher_forced_loss",
        "history": history,
        "test_evaluated": False,
        "manifest_sha256": manifest_hash,
    }
    checkpoint.with_suffix(".training.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _load(checkpoint: Path, device: str):
    torch = _torch()
    if not checkpoint.is_file() or checkpoint.stat().st_size > 256_000_000:
        raise ValueError("checkpoint must be an existing file no larger than 256 MB")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if (
        not isinstance(saved, dict)
        or saved.get("vocabulary") != VOCABULARY
        or type(saved.get("hidden")) is not int
        or not 16 <= saved["hidden"] <= 1024
        or not isinstance(saved.get("state_dict"), dict)
    ):
        raise ValueError("checkpoint vocabulary or architecture is incompatible")
    model = _build_model(saved["hidden"]).to(device)
    try:
        model.load_state_dict(saved["state_dict"])
    except (RuntimeError, TypeError) as error:
        raise ValueError("checkpoint weights do not match the declared architecture") from error
    model.eval()
    return model


def predict(
    checkpoint: Path, image: Path, *, device: str = "cpu", constrained: bool = False, max_tokens: int = MAX_TOKENS
) -> dict:
    torch = _torch()
    model = _load(checkpoint, device)
    with torch.inference_mode():
        ids = model.generate(
            image_tensor(read_ppm(image)).unsqueeze(0).to(device), constrained=constrained, max_tokens=max_tokens
        )[0].tolist()
    program = None
    try:
        program = decode(ids)
        parse_program(program)
        error = None
    except ValueError as exc:
        error = str(exc)
    return {
        "backend": "trained_cnn_gru",
        "constrained": constrained,
        "program": program,
        "raw_token_ids": ids,
        "eos_emitted": TOKEN_TO_ID["<eos>"] in ids,
        "valid_program": error is None,
        "validation_error": error,
    }


def evaluate_model(
    checkpoint: Path,
    directory: Path,
    *,
    split: str = "test",
    device: str = "cpu",
    constrained: bool = False,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    torch = _torch()
    model = _load(checkpoint, device)
    rows = load_rows(directory, split)
    cases = []
    with torch.inference_mode():
        for row in rows:
            image = read_ppm(directory / row["image"])
            ids = model.generate(
                image_tensor(image).unsqueeze(0).to(device), constrained=constrained, max_tokens=max_tokens
            )[0].tolist()
            expected = parse_program(row["program"])
            scene = Scene(())
            program = None
            error = None
            try:
                program = decode(ids)
                scene = parse_program(program)
                metrics = reconstruction_metrics(image, render_rgb(scene))
                valid = True
            except ValueError as exception:
                error = str(exception)
                metrics = {"exact_raster_match": False, "foreground_iou": 0.0, "pixel_agreement": 0.0, "rgb_mae": 1.0}
                valid = False
            cases.append(
                {
                    "id": row["id"],
                    "prediction": program,
                    "raw_token_ids": ids,
                    "eos_emitted": TOKEN_TO_ID["<eos>"] in ids,
                    "validation_error": error,
                    "expected_program": row["program"],
                    "valid_program": valid,
                    **metrics,
                    **primitive_metrics(expected, scene, HELDOUT.get(split)),
                }
            )
    count = len(cases)
    heldout_expected = sum(case.get("heldout_expected_count", 0) for case in cases)
    return {
        "backend": "trained_cnn_gru",
        "learned_model_evaluated": True,
        "split": split,
        "constrained": constrained,
        "max_tokens": max_tokens,
        "denominator": count,
        "valid_program_rate": sum(case["valid_program"] for case in cases) / count,
        "exact_raster_rate": sum(case["exact_raster_match"] for case in cases) / count,
        "mean_foreground_iou": sum(case["foreground_iou"] for case in cases) / count,
        "primitive_micro_recall": sum(case["primitive_true_positive"] for case in cases)
        / sum(len(parse_program(row["program"]).shapes) for row in rows),
        "heldout_exact_recall": sum(case.get("heldout_exact_matches", 0) for case in cases) / heldout_expected
        if heldout_expected
        else None,
        "invalid_prediction_policy": "count as zero reconstruction score, retain raw output",
        "cases": cases,
    }
