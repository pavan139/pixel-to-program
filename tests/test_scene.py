import json
import tempfile
import unittest
from pathlib import Path

from pixel_to_program.__main__ import evaluate
from pixel_to_program.data import HELDOUT, TRAIN_PAIRS, generate_scene, load_rows, write_dataset
from pixel_to_program.model import decode, encode
from pixel_to_program.scene import (
    Scene,
    Shape,
    parse_program,
    ppm_bytes,
    read_ppm,
    reconstruction_metrics,
    render_rgb,
    render_svg,
    serialize,
)


class SceneTests(unittest.TestCase):
    def test_program_roundtrip(self):
        scene = generate_scene("train", 17)
        self.assertEqual(parse_program(serialize(scene)), scene)

    def test_unsupported_commands_rejected(self):
        for source in ("SCENE EXEC RED 0 0 3 3 END", "SCENE RECT BLACK 0 0 3 3 END", "SCENE <script> END"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_program(source)

    def test_empty_scene_allowed(self):
        self.assertEqual(render_rgb(parse_program("SCENE END")), bytes([255] * 3072))

    def test_incomplete_program_rejected(self):
        for source in ("", "SCENE", "SCENE RECT RED 0 0 3 END", "RECT RED 0 0 3 3 END", "SCENE END trailing"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_program(source)

    def test_canonical_coordinates_required(self):
        for value in ("-1", "1.5", "01", "+1", "999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_program(f"SCENE RECT RED {value} 0 3 3 END")

    def test_out_of_bounds_rejected(self):
        for source in ("SCENE RECT RED 31 0 2 3 END", "SCENE RECT RED 0 0 0 3 END", "SCENE RECT RED 0 30 3 3 END"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_program(source)

    def test_scene_budget(self):
        with self.assertRaises(ValueError):
            parse_program("SCENE " + "RECT RED 0 0 3 3 " * 5 + "END")
        with self.assertRaises(ValueError):
            parse_program(" " * 4097)

    def test_boolean_coordinate_not_integer(self):
        with self.assertRaises(ValueError):
            Shape("RECT", "RED", True, 0, 2, 2)

    def test_later_shapes_occlude_earlier(self):
        scene = parse_program("SCENE RECT RED 0 0 3 3 RECT BLUE 0 0 1 1 END")
        image = render_rgb(scene)
        self.assertEqual(image[:3], bytes((40, 95, 225)))
        self.assertEqual(image[3:6], bytes((230, 45, 45)))

    def test_ellipse_has_unfilled_corners(self):
        image = render_rgb(parse_program("SCENE ELLIPSE RED 0 0 8 8 END"))
        self.assertEqual(image[:3], bytes((255, 255, 255)))
        self.assertNotEqual(image[(4 * 32 + 4) * 3 : (4 * 32 + 4) * 3 + 3], bytes((255, 255, 255)))

    def test_svg_only_allowlisted_primitives(self):
        source = render_svg(generate_scene("test", 1))
        self.assertTrue(source.startswith("<svg xmlns="))
        self.assertIn("<ellipse", source)
        self.assertNotIn("script", source)

    def test_metrics_dont_reward_blank_foreground(self):
        scene = generate_scene("test", 2)
        metrics = reconstruction_metrics(render_rgb(scene), render_rgb(Scene(())))
        self.assertEqual(metrics["foreground_iou"], 0)
        self.assertFalse(metrics["exact_raster_match"])
        with self.assertRaises(ValueError):
            reconstruction_metrics(b"", b"")

    def test_ppm_roundtrip_and_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.ppm"
            scene = generate_scene("test", 3)
            path.write_bytes(ppm_bytes(scene))
            self.assertEqual(read_ppm(path), render_rgb(scene))
            path.write_bytes(b"P6\n999 999\n255\n")
            with self.assertRaises(ValueError):
                read_ppm(path)

    def test_composition_splits(self):
        for seed in range(60):
            train_pairs = {(shape.kind, shape.color) for shape in generate_scene("train", seed).shapes}
            self.assertTrue(train_pairs <= set(TRAIN_PAIRS))
            self.assertNotIn(HELDOUT["dev"], train_pairs)
            self.assertNotIn(HELDOUT["test"], train_pairs)
            for split in ("dev", "test"):
                self.assertIn(
                    HELDOUT[split], {(shape.kind, shape.color) for shape in generate_scene(split, seed).shapes}
                )

    def test_heldout_composition_is_visually_observable(self):
        # Regression: seed325 previously hid the designated first primitive entirely.
        for split in ("dev", "test"):
            for seed in (0, 1, 17, 325, 1002):
                scene = generate_scene(split, seed)
                self.assertEqual((scene.shapes[-1].kind, scene.shapes[-1].color), HELDOUT[split])
                self.assertNotEqual(render_rgb(scene), render_rgb(Scene(scene.shapes[:-1])))

    def test_dataset_manifest_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_dataset(path, train=12, dev=12, test=12)
            ids = []
            raster_ids = []
            for split in ("train", "dev", "test"):
                rows = load_rows(path, split)
                self.assertEqual(len(rows), 12)
                ids.extend(row["id"] for row in rows)
                raster_ids.extend(row["raster_sha256"] for row in rows)
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(raster_ids), len(set(raster_ids)))

    def test_manifest_image_program_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_dataset(path, train=1, dev=1, test=1)
            (path / "images/train/000000.ppm").write_bytes(ppm_bytes(Scene(())))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_rows(path, "train")

    def test_manifest_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_dataset(path, train=1, dev=1, test=1)
            manifest = path / "manifest.jsonl"
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            rows[0]["image"] = "../outside.ppm"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaises(ValueError):
                load_rows(path, "train")

    def test_tokenizer_roundtrip_without_torch(self):
        program = serialize(generate_scene("train", 8))
        self.assertEqual(decode(encode(program)), program)
        with self.assertRaises(ValueError):
            decode([999])

    def test_reference_report_does_not_claim_model_accuracy(self):
        result = evaluate()
        self.assertEqual(result["denominator"], 36)
        self.assertEqual(result["oracle_exact_raster_matches"], 36)
        self.assertFalse(result["learned_model_evaluated"])


if __name__ == "__main__":
    unittest.main()
