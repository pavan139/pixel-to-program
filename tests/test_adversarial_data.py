import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pixel_to_program.data import generate_scene, heldout_visibility, load_rows, write_dataset
from pixel_to_program.scene import Scene, Shape, ppm_bytes, render_rgb, scene_hash, serialize


class ManifestContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        write_dataset(self.data, train=2, dev=2, test=2)
        self.manifest = self.data / "manifest.jsonl"
        self.original = self.manifest.read_text()

    def tearDown(self):
        self.temporary.cleanup()

    def alter(self, operation):
        rows = [json.loads(line) for line in self.original.splitlines()]
        operation(rows)
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_nonobject_missing_extra_and_wrong_types_fail_as_values(self):
        mutations = [
            lambda rows: rows.__setitem__(0, []),
            lambda rows: rows[0].pop("image"),
            lambda rows: rows[0].update(extra="ignored?"),
            lambda rows: rows[0].update(seed=True),
            lambda rows: rows[0].update(image=[]),
            lambda rows: rows[0].update(split=[]),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.alter(mutation)
                with self.assertRaises(ValueError):
                    load_rows(self.data, "train")

    def test_duplicate_json_keys_rejected(self):
        self.manifest.write_text(self.original.replace('"split": "train"', '"split": "test", "split": "train"', 1))
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            load_rows(self.data, "train")

    def test_absolute_parent_null_and_missing_paths_rejected(self):
        for image in (
            str(self.data / "images/train/000000.ppm"),
            "images/../images/train/000000.ppm",
            "missing.ppm",
            "bad\x00.ppm",
            "x" * 300,
        ):
            with self.subTest(image=image):
                self.alter(lambda rows: rows[0].update(image=image))
                with self.assertRaises(ValueError):
                    load_rows(self.data, "train")

    def test_image_and_manifest_symlink_escape(self):
        outside = self.root / "outside.ppm"
        outside.write_bytes((self.data / "images/train/000000.ppm").read_bytes())
        image = self.data / "images/train/000000.ppm"
        image.unlink()
        image.symlink_to(outside)
        with self.assertRaises(ValueError):
            load_rows(self.data, "train")
        outside_manifest = self.root / "outside.jsonl"
        outside_manifest.write_text(self.original)
        self.manifest.unlink()
        self.manifest.symlink_to(outside_manifest)
        with self.assertRaisesRegex(ValueError, "manifest"):
            load_rows(self.data, "train")

    def test_write_output_symlink_escape(self):
        destination = self.root / "new"
        destination.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (destination / "images").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            write_dataset(destination, train=1, dev=1, test=1)
        self.assertEqual(list(outside.iterdir()), [])

    def test_occluded_heldout_scene_rejected_even_with_matching_image(self):
        scene = Scene((Shape("ELLIPSE", "BLUE", 0, 0, 4, 4), Shape("RECT", "RED", 0, 0, 8, 8)))
        self.assertEqual(heldout_visibility(scene, ("ELLIPSE", "BLUE")), 0)

        def mutation(rows):
            row = next(row for row in rows if row["split"] == "test")
            row.update(
                program=serialize(scene),
                id=scene_hash(scene),
                raster_sha256=hashlib.sha256(render_rgb(scene)).hexdigest(),
            )
            (self.data / row["image"]).write_bytes(ppm_bytes(scene))

        self.alter(mutation)
        with self.assertRaisesRegex(ValueError, "visible"):
            load_rows(self.data, "test")

    def test_generation_guarantees_visible_holdout_for_many_seeds(self):
        for split, pair in (("dev", ("RECT", "GREEN")), ("test", ("ELLIPSE", "BLUE"))):
            for seed in range(100):
                self.assertGreaterEqual(heldout_visibility(generate_scene(split, seed), pair), 0.5)

    def test_duplicate_generation_is_bounded(self):
        scene = generate_scene("train", 1)
        with patch("pixel_to_program.data.generate_scene", return_value=scene):
            with self.assertRaisesRegex(RuntimeError, "bounded"):
                write_dataset(self.root / "duplicate", train=2, dev=1, test=1)


if __name__ == "__main__":
    unittest.main()
