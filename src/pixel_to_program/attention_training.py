"""Atomic latest/best snapshots and reproducible shared training engine."""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from .attention_model import ModelConfig, build_model, token_loss
from .data import load_rows
from .model import VOCABULARY, _batch, _torch

FORMAT_VERSION = 2


def dataset_fingerprint(directory: Path) -> str:
    digest = hashlib.sha256()
    for filename in ("manifest.jsonl", "dataset.json"):
        path = directory / filename
        if not path.is_file() or path.stat().st_size > 100_000_000:
            raise ValueError("dataset manifest missing or over budget")
        digest.update(filename.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_torch_save(value: dict, destination: Path) -> None:
    torch = _torch()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, destination)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(destination.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_checkpoint(checkpoint: Path, device: str = "cpu", *, selection: str = "latest"):
    torch = _torch()
    if selection not in {"latest", "best"}:
        raise ValueError("checkpoint selection must be latest or best")
    if not checkpoint.is_file() or checkpoint.stat().st_size > 256_000_000:
        raise ValueError("checkpoint must be a local file under 256 MB")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    required = {
        "format_version",
        "config",
        "vocabulary",
        "model_state",
        "optimizer_state",
        "run_config",
        "run_config_sha256",
        "dataset_sha256",
        "train_ids",
        "dev_ids",
        "epoch",
        "global_step",
        "best_dev_loss",
        "best_model_state",
        "best_epoch",
        "best_global_step",
        "history",
        "torch_rng",
        "cuda_rng",
        "python_rng",
    }
    if (
        not isinstance(saved, dict)
        or set(saved) != required
        or saved["format_version"] != FORMAT_VERSION
        or saved["vocabulary"] != VOCABULARY
    ):
        raise ValueError("checkpoint schema/version/vocabulary mismatch")
    try:
        config = ModelConfig(**saved["config"])
        expected = hashlib.sha256(json.dumps(saved["run_config"], sort_keys=True).encode()).hexdigest()
        if expected != saved["run_config_sha256"] or saved["run_config"].get("model") != saved["config"]:
            raise ValueError("checkpoint run configuration hash mismatch")
        if (
            type(saved["epoch"]) is not int
            or type(saved["global_step"]) is not int
            or saved["epoch"] < 1
            or saved["global_step"] < 1
        ):
            raise ValueError("invalid checkpoint progress")
        if len(saved["dataset_sha256"]) != 64 or not math.isfinite(saved["best_dev_loss"]):
            raise ValueError("invalid checkpoint identity or loss")
        if (
            type(saved["best_epoch"]) is not int
            or not 1 <= saved["best_epoch"] <= saved["epoch"]
            or type(saved["best_global_step"]) is not int
            or not 1 <= saved["best_global_step"] <= saved["global_step"]
        ):
            raise ValueError("invalid best-checkpoint selection metadata")
        if len(set(saved["train_ids"])) != len(saved["train_ids"]) or set(saved["train_ids"]) & set(saved["dev_ids"]):
            raise ValueError("checkpoint split overlap")
        model = build_model(config).to(device)
        for key in ("model_state", "best_model_state"):
            model.load_state_dict(saved[key], strict=True)
            if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
                raise ValueError("nonfinite checkpoint parameters")
        model.load_state_dict(saved["best_model_state" if selection == "best" else "model_state"])
    except (TypeError, KeyError, RuntimeError, AttributeError) as error:
        raise ValueError("malformed checkpoint architecture or state") from error
    model.eval()
    return model, config, saved


def train(
    directory: Path,
    checkpoint: Path,
    *,
    config: ModelConfig | None = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    smoothing: float = 0.0,
    warmup_steps: int = 0,
    seed: int = 17,
    device: str = "cpu",
    resume: bool = False,
    stop_after_epochs: int | None = None,
) -> dict:
    config = ModelConfig() if config is None else config
    if not isinstance(config, ModelConfig):
        raise ValueError("config must be a validated ModelConfig")
    if (
        any(type(value) is not int for value in (epochs, batch_size, warmup_steps, seed))
        or not 1 <= epochs <= 200
        or not 1 <= batch_size <= 256
        or warmup_steps < 0
        or not -(2**31) <= seed < 2**31
    ):
        raise ValueError("training dimensions exceed explicit budgets")
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or not 1 <= stop_after_epochs <= epochs):
        raise ValueError("stop_after_epochs must be within the planned epoch budget")
    if (
        any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (learning_rate, weight_decay, smoothing)
        )
        or not 0 < learning_rate <= 0.05
        or not 0 <= weight_decay <= 1
        or not 0 <= smoothing <= 0.2
    ):
        raise ValueError("optimizer settings must be finite and within documented bounds")
    if not resume and checkpoint.exists():
        raise ValueError("checkpoint exists; choose a new path or explicit --resume")
    torch = _torch()
    rows, dev = load_rows(directory, "train"), load_rows(directory, "dev")
    batches_per_epoch = math.ceil(len(rows) / batch_size)
    total_steps = batches_per_epoch * epochs
    if total_steps > 100000 or warmup_steps >= total_steps:
        raise ValueError("training step/warmup budget exceeded")
    fingerprint = dataset_fingerprint(directory)
    run_config = {
        "model": asdict(config),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "label_smoothing": smoothing,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "total_steps": total_steps,
        "optimizer": "AdamW",
        "scheduler": "linear_warmup_then_cosine",
        "device": device,
    }
    run_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    start_epoch, global_step, best_dev, history = 0, 0, math.inf, []
    best_state, best_epoch, best_global_step = None, 0, 0
    if resume:
        model, loaded_config, saved = load_checkpoint(checkpoint, device)
        if loaded_config != config or saved["run_config_sha256"] != run_hash or saved["dataset_sha256"] != fingerprint:
            raise ValueError("resume requires identical model, optimizer plan, and dataset")
        if saved["train_ids"] != [row["id"] for row in rows] or saved["dev_ids"] != [row["id"] for row in dev]:
            raise ValueError("resume split membership changed")
        start_epoch, global_step = saved["epoch"], saved["global_step"]
        best_dev, history = saved["best_dev_loss"], saved["history"]
        best_state, best_epoch, best_global_step = (
            saved["best_model_state"],
            saved["best_epoch"],
            saved["best_global_step"],
        )
    else:
        model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if resume:
        optimizer.load_state_dict(saved["optimizer_state"])
        torch.set_rng_state(saved["torch_rng"])
        rng.setstate(saved["python_rng"])
        if device.startswith("cuda"):
            if len(saved["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("CUDA RNG/device topology changed during resume")
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    finish_epoch = min(epochs, start_epoch + stop_after_epochs) if stop_after_epochs else epochs
    for epoch in range(start_epoch, finish_epoch):
        order = rows[:]
        rng.shuffle(order)
        model.train()
        training_sum, training_count = 0.0, 0
        for offset in range(0, len(order), batch_size):
            images, tokens = _batch(directory, order[offset : offset + batch_size], device)
            step = global_step + 1
            if warmup_steps and step <= warmup_steps:
                multiplier = step / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                multiplier = 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * multiplier
            optimizer.zero_grad(set_to_none=True)
            summed, count = token_loss(model(images, tokens[:, :-1]), tokens[:, 1:], smoothing=smoothing)
            (summed / count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            training_sum += summed.item()
            training_count += count
        model.eval()
        dev_sum, dev_count = 0.0, 0
        with torch.inference_mode():
            for offset in range(0, len(dev), batch_size):
                images, tokens = _batch(directory, dev[offset : offset + batch_size], device)
                summed, count = token_loss(model(images, tokens[:, :-1]), tokens[:, 1:])
                dev_sum += summed.item()
                dev_count += count
        dev_loss = dev_sum / dev_count
        improved = dev_loss < best_dev
        best_dev = min(best_dev, dev_loss)
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch, best_global_step = epoch + 1, global_step
        history.append(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_token_loss": training_sum / training_count,
                "dev_token_loss": dev_loss,
                "train_target_tokens": training_count,
                "dev_target_tokens": dev_count,
            }
        )
        payload = {
            "format_version": FORMAT_VERSION,
            "config": asdict(config),
            "vocabulary": VOCABULARY,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "run_config": run_config,
            "run_config_sha256": run_hash,
            "dataset_sha256": fingerprint,
            "train_ids": [row["id"] for row in rows],
            "dev_ids": [row["id"] for row in dev],
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_dev_loss": best_dev,
            "best_model_state": best_state,
            "best_epoch": best_epoch,
            "best_global_step": best_global_step,
            "history": history,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
            "python_rng": rng.getstate(),
        }
        _atomic_torch_save(payload, checkpoint)
    report = {
        "architecture": config.architecture,
        "config": asdict(config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dataset_sha256": fingerprint,
        "run_config_sha256": run_hash,
        "seed": seed,
        "epochs_completed": finish_epoch,
        "epochs_planned": epochs,
        "global_step": global_step,
        "resumed": resume,
        "latest_checkpoint": str(checkpoint),
        "best_checkpoint": str(checkpoint),
        "best_selection": "embedded_best_model_state",
        "best_epoch": best_epoch,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "best_dev_token_loss": best_dev,
        "history": history,
        "torch_version": str(torch.__version__),
        "test_evaluated": False,
    }
    checkpoint.with_suffix(".training.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
