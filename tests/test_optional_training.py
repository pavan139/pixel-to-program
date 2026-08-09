"""Small integration check that executes only when torch is installed."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from pixel_to_program.data import load_rows, write_dataset
from pixel_to_program.model import TOKEN_TO_ID, VOCABULARY, _batch, _build_model, _load, evaluate_model, predict, train
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional torch dependency is not installed")
class OptionalTrainingTests(unittest.TestCase):
    def test_train_save_load_predict_and_evaluate(self):
        import torch

        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                data = path / "data"
                checkpoint = path / "model.pt"
                write_dataset(data, train=4, dev=2, test=2)
                report = train(data, checkpoint, epochs=1, batch_size=2, hidden=16)
                self.assertTrue(checkpoint.exists())
                self.assertFalse(report["test_evaluated"])
                result = predict(checkpoint, data / "images/test/000000.ppm")
                self.assertIn("valid_program", result)
                # One epoch need not yield a valid program; errors must stay visible.
                self.assertEqual(result["valid_program"], result["validation_error"] is None)
                evaluated = evaluate_model(checkpoint, data)
                self.assertEqual(evaluated["denominator"], 2)
                self.assertTrue(evaluated["learned_model_evaluated"])
                self.assertEqual(len(evaluated["cases"]), 2)
                model = _load(checkpoint, "cpu")
                images, tokens = _batch(data, load_rows(data, "dev"), "cpu")
                with torch.inference_mode():
                    logits = model(images, tokens[:, :-1])
                    summed = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, len(VOCABULARY)), tokens[:, 1:].reshape(-1), ignore_index=0, reduction="sum"
                    )
                actual_dev = summed.item() / int((tokens[:, 1:] != 0).sum())
                self.assertAlmostEqual(report["history"][0]["dev_token_mean_loss"], actual_dev, places=5)
        finally:
            torch.set_num_threads(previous)

    def test_actual_neural_masking_raw_failures_and_short_budget(self):
        import torch

        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                data, checkpoint = path / "data", path / "fixture.pt"
                write_dataset(data, train=2, dev=1, test=1)
                model = _build_model(16)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                    for token, value in (
                        ("<pad>", 1000),
                        ("RECT", 100),
                        ("GREEN", 99),
                        ("31", 98),
                        ("1", 97),
                        ("END", 96),
                        ("<eos>", 95),
                    ):
                        model.output.bias[TOKEN_TO_ID[token]] = value
                torch.save({"hidden": 16, "vocabulary": VOCABULARY, "state_dict": model.state_dict()}, checkpoint)
                image = data / "images/test/000000.ppm"
                raw = predict(checkpoint, image)
                self.assertFalse(raw["valid_program"])
                self.assertIn("padding", raw["validation_error"])
                masked = predict(checkpoint, image, constrained=True)
                self.assertTrue(masked["valid_program"])
                self.assertTrue(masked["eos_emitted"])
                self.assertEqual(len(masked["raw_token_ids"]), 27)
                short = predict(checkpoint, image, constrained=True, max_tokens=4)
                self.assertFalse(short["valid_program"])
                self.assertIn("without EOS", short["validation_error"])
                # Evaluation must use autoregressive generation, never teacher-forced forward.
                model.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("teacher forcing leaked into evaluation")
                )
                with patch("pixel_to_program.model._load", return_value=model):
                    result = evaluate_model(checkpoint, data, constrained=True)
                self.assertEqual(result["valid_program_rate"], 1)
                self.assertEqual(result["heldout_exact_recall"], 0)
                self.assertEqual(result["primitive_micro_recall"], 0)
        finally:
            torch.set_num_threads(previous)

    def test_checkpoint_architecture_and_budget_types_fail_explicitly(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            for saved in (
                [1, 2],
                {"hidden": 16.0, "vocabulary": VOCABULARY, "state_dict": {}},
                {"hidden": 16, "vocabulary": VOCABULARY, "state_dict": {}},
            ):
                torch.save(saved, path)
                with self.subTest(saved=type(saved)), self.assertRaises(ValueError):
                    _load(path, "cpu")
            with self.assertRaises(ValueError):
                train(Path(directory), path, epochs=True)


if __name__ == "__main__":
    unittest.main()
