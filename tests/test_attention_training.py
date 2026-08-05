import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from pixel_to_program.data import write_dataset
from pixel_to_program.model import TOKEN_TO_ID, VOCABULARY, decode
from pixel_to_program.scene import parse_program
from pixel_to_program.transformer import (
    ModelConfig,
    ablation,
    build_model,
    evaluate,
    generate,
    load_checkpoint,
    token_loss,
    train,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency not installed")
class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch

        torch.set_num_threads(cls.previous_threads)

    def config(self, **kwargs):
        return ModelConfig(
            d_model=16, heads=2, encoder_layers=1, decoder_layers=1, feedforward=32, patch_size=8, **kwargs
        )

    def test_causal_decoder_does_not_read_future_tokens(self):
        import torch

        torch.manual_seed(2)
        model = build_model(self.config(dropout=0)).eval()
        image = torch.rand(1, 3, 32, 32)
        first = torch.tensor([[1, TOKEN_TO_ID["SCENE"], TOKEN_TO_ID["RECT"], TOKEN_TO_ID["RED"]]])
        second = first.clone()
        second[:, 2:] = torch.tensor([[TOKEN_TO_ID["ELLIPSE"], TOKEN_TO_ID["BLUE"]]])
        with torch.inference_mode():
            left, right = model(image, first), model(image, second)
        torch.testing.assert_close(left[:, :2], right[:, :2], atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(left[:, 2:], right[:, 2:]))

    def test_visual_memory_conditions_output_and_receives_gradients(self):
        import torch

        torch.manual_seed(3)
        model = build_model(self.config(dropout=0))
        images = torch.rand(2, 3, 32, 32)
        ids = torch.tensor([[1, 3], [1, 3]])
        logits = model(images, ids)
        self.assertEqual(tuple(model.encode_images(images).shape), (2, 16, 16))
        self.assertFalse(torch.allclose(logits[0], logits[1]))
        targets = torch.tensor([[3, 4], [3, 4]])
        summed, count = token_loss(logits, targets)
        (summed / count).backward()
        self.assertGreater(model.patch_embedding.weight.grad.abs().sum().item(), 0)
        self.assertTrue(bool(torch.isfinite(model.patch_embedding.weight.grad).all()))

    def test_padding_loss_is_token_weighted_and_padding_invariant(self):
        import torch

        logits = torch.zeros(1, 4, len(VOCABULARY))
        targets = torch.tensor([[3, 4, 2, 0]])
        before, count = token_loss(logits, targets)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(before.item(), 3 * math.log(len(VOCABULARY)), places=5)
        logits[:, -1, :] = 10000
        after, _ = token_loss(logits, targets)
        torch.testing.assert_close(before, after)
        with self.assertRaises(ValueError):
            token_loss(logits, torch.zeros_like(targets))

    def test_batched_beam_search_encodes_the_image_only_once(self):
        import torch
        from unittest.mock import patch

        torch.manual_seed(4)
        config = self.config(dropout=0)
        model = build_model(config).eval()
        with patch.object(model, "encode_images", wraps=model.encode_images) as encoder:
            result = generate(model, config, torch.rand(1, 3, 32, 32), beam_size=4)
        self.assertEqual(encoder.call_count, 1)
        self.assertEqual(result["image_encoder_calls"], 1)
        self.assertTrue(result["eos_emitted"])
        parse_program(decode(result["tokens"]))

    def test_training_resume_is_identical_with_dropout_optimizer_and_rng(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            write_dataset(data, train=4, dev=2, test=2)
            config = self.config(dropout=0.1)
            full, interrupted = root / "full.pt", root / "interrupted.pt"
            baseline = train(data, full, config=config, epochs=2, batch_size=2, seed=21, warmup_steps=1)
            first = train(
                data, interrupted, config=config, epochs=2, batch_size=2, seed=21, warmup_steps=1, stop_after_epochs=1
            )
            self.assertEqual(first["global_step"], 2)
            resumed = train(
                data, interrupted, config=config, epochs=2, batch_size=2, seed=21, warmup_steps=1, resume=True
            )
            self.assertEqual(baseline["history"], resumed["history"])
            a, _, _ = load_checkpoint(full)
            b, _, _ = load_checkpoint(interrupted)
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                train(
                    data,
                    interrupted,
                    config=config,
                    epochs=2,
                    batch_size=2,
                    seed=21,
                    warmup_steps=1,
                    resume=True,
                    learning_rate=0.002,
                )
            prediction = evaluate(Path(resumed["best_checkpoint"]), data, beam_size=1)
            self.assertEqual(prediction["denominator"], 2)
            self.assertTrue(prediction["held_out"])
            self.assertFalse(prediction["dataset_shift"])
            with self.assertRaises(ValueError):
                train(data, interrupted, config=config, epochs=2)

    def test_shared_engine_runs_gru_baseline_and_paired_ablation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            write_dataset(data, train=2, dev=1, test=1)
            result = ablation(data, root / "runs", epochs=1, seeds=(7,), d_model=16, beam_sizes=(1,))
            self.assertEqual(len(result["results"]), 4)
            self.assertEqual({row["architecture"] for row in result["results"]}, {"cnn_gru", "patch_transformer"})
            self.assertEqual(len({row["test"]["dataset_sha256"] for row in result["results"]}), 1)
            self.assertFalse(result["parameter_counts_matched"])
            self.assertTrue(all(row["global_step"] == 1 for row in result["results"]))
            self.assertTrue(all(math.isfinite(row["dev_token_loss"]) for row in result["results"]))

    def test_best_and_latest_weights_are_one_atomic_checkpoint_with_explicit_selection(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, checkpoint = root / "data", root / "model.pt"
            write_dataset(data, train=2, dev=1, test=1)
            report = train(data, checkpoint, config=self.config(), epochs=1, batch_size=2)
            self.assertEqual(report["latest_checkpoint"], report["best_checkpoint"])
            self.assertEqual(len(list(root.glob("*.pt"))), 1)
            saved = torch.load(checkpoint, weights_only=True)
            name = next(iter(saved["model_state"]))
            original_best = saved["best_model_state"][name].clone()
            saved["model_state"][name].add_(0.25)
            self.assertTrue(torch.equal(saved["best_model_state"][name], original_best))
            torch.save(saved, checkpoint)
            latest, _, _ = load_checkpoint(checkpoint, selection="latest")
            best, _, _ = load_checkpoint(checkpoint, selection="best")
            torch.testing.assert_close(best.state_dict()[name], original_best)
            self.assertFalse(torch.equal(latest.state_dict()[name], best.state_dict()[name]))
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint, selection="unknown")

    def test_checkpoint_binding_rejects_dataset_changes_and_malformed_models(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, checkpoint = root / "data", root / "model.pt"
            write_dataset(data, train=2, dev=1, test=1)
            train(data, checkpoint, config=self.config(), epochs=1, batch_size=2)
            write_dataset(data, train=2, dev=1, test=1, seed=44)
            with self.assertRaises(ValueError):
                evaluate(checkpoint, data)
            with self.assertRaises(ValueError):
                train(data, checkpoint, config=self.config(), epochs=1, batch_size=2, resume=True)
            saved = torch.load(checkpoint, weights_only=True)
            saved["vocabulary"] = ["wrong"]
            torch.save(saved, root / "bad.pt")
            with self.assertRaises(ValueError):
                load_checkpoint(root / "bad.pt")


if __name__ == "__main__":
    unittest.main()
