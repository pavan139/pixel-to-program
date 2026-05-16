from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .scene import Scene, Shape, parse_program, ppm_bytes, read_ppm, render_rgb, scene_hash, serialize

TRAIN_PAIRS = (("RECT", "RED"), ("RECT", "BLUE"), ("ELLIPSE", "RED"), ("ELLIPSE", "GREEN"))
HELDOUT = {"dev": ("RECT", "GREEN"), "test": ("ELLIPSE", "BLUE")}


def heldout_visibility(scene: Scene, pair: tuple[str, str]) -> float:
    """Fraction of the heldout primitive that changes the final rendered image."""
    indices = [index for index, shape in enumerate(scene.shapes) if (shape.kind, shape.color) == pair]
    if len(indices) != 1:
        raise ValueError("heldout scenes require exactly one designated primitive")
    index = indices[0]
    full = render_rgb(scene)
    without = render_rgb(Scene(scene.shapes[:index] + scene.shapes[index + 1 :]))
    intrinsic = render_rgb(Scene((scene.shapes[index],)))
    changed = sum(full[offset : offset + 3] != without[offset : offset + 3] for offset in range(0, len(full), 3))
    foreground = sum(intrinsic[offset : offset + 3] != b"\xff\xff\xff" for offset in range(0, len(intrinsic), 3))
    return changed / foreground


def generate_scene(split: str, seed: int) -> Scene:
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev, or test")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    for _attempt in range(100):
        count = rng.randint(1, 3)
        shapes = []
        for index in range(count):
            pair = HELDOUT[split] if index == count - 1 and split in HELDOUT else rng.choice(TRAIN_PAIRS)
            width, height = rng.randint(4, 14), rng.randint(4, 14)
            shapes.append(Shape(*pair, rng.randint(0, 32 - width), rng.randint(0, 32 - height), width, height))
        scene = Scene(tuple(shapes))
        if split == "train":
            return scene
        if heldout_visibility(scene, HELDOUT[split]) >= 0.5:
            return scene
    raise RuntimeError("could not generate a visibly identifiable heldout primitive within 100 attempts")


def write_dataset(directory: Path, *, train: int = 256, dev: int = 64, test: int = 64, seed: int = 17) -> dict:
    counts = {"train": train, "dev": dev, "test": test}
    if any(type(count) is not int or not 1 <= count <= 100000 for count in counts.values()):
        raise ValueError("each split size must be in [1, 100000]")
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("manifest.jsonl", "dataset.json"):
        if not (directory / filename).resolve().is_relative_to(directory.resolve()):
            raise ValueError("dataset metadata path escapes output directory")
    rows = []
    ids = set()
    raster_ids = set()
    for split_index, (split, count) in enumerate(counts.items()):
        generated = 0
        attempt = 0
        while generated < count:
            if attempt >= max(100, count * 50):
                raise RuntimeError("dataset deduplication exhausted its bounded generation attempts")
            scene_seed = seed + split_index * 1000000 + attempt
            scene = generate_scene(split, scene_seed)
            attempt += 1
            identifier = scene_hash(scene)
            raster_identifier = hashlib.sha256(render_rgb(scene)).hexdigest()
            if identifier in ids or raster_identifier in raster_ids:
                continue
            ids.add(identifier)
            raster_ids.add(raster_identifier)
            relative = Path("images") / split / f"{generated:06d}.ppm"
            target = directory / relative
            if not target.resolve().is_relative_to(directory.resolve()):
                raise ValueError("dataset image path escapes output directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(ppm_bytes(scene))
            rows.append(
                {
                    "id": identifier,
                    "split": split,
                    "seed": scene_seed,
                    "image": relative.as_posix(),
                    "raster_sha256": raster_identifier,
                    "program": serialize(scene),
                    "provenance": "synthetic_scene_generator_v1",
                }
            )
            generated += 1
    (directory / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    description = {
        "schema_version": 1,
        "seed": seed,
        "counts": counts,
        "train_shape_color_pairs": TRAIN_PAIRS,
        "heldout_pairs": HELDOUT,
        "split_policy": "dev/test each contain a visible designated heldout pair; training never contains either heldout pair; rendered images deduplicated globally",
    }
    (directory / "dataset.json").write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
    return description


def load_rows(directory: Path, split: str) -> list[dict]:
    if split not in ("train", "dev", "test"):
        raise ValueError("unknown split")
    manifest = directory / "manifest.jsonl"
    if not manifest.resolve().is_relative_to(directory.resolve()) or not manifest.is_file():
        raise ValueError("manifest must be a regular file inside the dataset directory")
    if manifest.stat().st_size > 100_000_000:
        raise ValueError("manifest exceeds 100 MB input budget")
    rows = []
    seen = set()
    seen_rasters = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if len(line) > 16_384:
            raise ValueError("manifest line exceeds 16 KB input budget")

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key in manifest")
                result[key] = value
            return result

        try:
            row = json.loads(line, object_pairs_hook=unique_keys)
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError(f"invalid JSON on manifest line {line_number}") from error
        fields = {"id", "split", "seed", "image", "raster_sha256", "program", "provenance"}
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("manifest row must contain exactly the documented fields")
        if type(row["seed"]) is not int or not isinstance(row["provenance"], str) or len(row["provenance"]) > 100:
            raise ValueError("manifest seed/provenance has an invalid type or size")
        if any(not isinstance(row[key], str) for key in ("id", "image", "raster_sha256", "program")):
            raise ValueError("manifest identifiers, paths and programs must be strings")
        if row.get("split") not in ("train", "dev", "test"):
            raise ValueError(f"invalid split on manifest line {line_number}")
        scene = parse_program(row["program"])
        if not 1 <= len(scene.shapes) <= 3:
            raise ValueError("dataset scenes require one to three primitives")
        if row.get("id") != scene_hash(scene) or row["id"] in seen:
            raise ValueError("manifest has a wrong ID or duplicate scene")
        seen.add(row["id"])
        pairs = {(shape.kind, shape.color) for shape in scene.shapes}
        if row["split"] == "train" and not pairs <= set(TRAIN_PAIRS):
            raise ValueError("heldout composition leaked into training")
        if row["split"] in HELDOUT and HELDOUT[row["split"]] not in pairs:
            raise ValueError("heldout example lacks its designated composition")
        if row["split"] == "dev" and HELDOUT["test"] in pairs:
            raise ValueError("test composition leaked into development")
        if row["split"] in HELDOUT:
            if not pairs <= set(TRAIN_PAIRS) | {HELDOUT[row["split"]]}:
                raise ValueError("heldout split contains another split's reserved composition")
            if heldout_visibility(scene, HELDOUT[row["split"]]) < 0.5:
                raise ValueError("heldout primitive is not sufficiently visible")
        relative = Path(row["image"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or "\x00" in row["image"]
            or len(row["image"]) > 1024
            or any(len(part) > 255 for part in relative.parts)
        ):
            raise ValueError("image path must be relative without parent traversal")
        image = directory / row["image"]
        if not image.resolve().is_relative_to(directory.resolve()):
            raise ValueError("image path escapes dataset directory")
        if not image.is_file():
            raise ValueError("manifest image must be an existing regular file")
        rgb = read_ppm(image)
        raster_hash = hashlib.sha256(rgb).hexdigest()
        if rgb != render_rgb(scene) or row.get("raster_sha256") != raster_hash:
            raise ValueError("manifest image does not match its program and raster hash")
        if raster_hash in seen_rasters:
            raise ValueError("duplicate rendered image in dataset")
        seen_rasters.add(raster_hash)
        if row["split"] == split:
            rows.append(row)
    if not rows:
        raise ValueError("requested split is empty")
    return rows
