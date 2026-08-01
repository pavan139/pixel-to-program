import itertools
import math
import unittest

from pixel_to_program.search import Hypothesis, beam_search
from pixel_to_program.transformer import ModelConfig


def logs(values):
    return [math.log(value) if value else -math.inf for value in values]


class BeamSearchTests(unittest.TestCase):
    def test_beam_recovers_jointly_better_completion_than_greedy(self):
        def backend(prefixes):
            return [
                logs((0.6, 0.4, 0))
                if not prefix
                else logs((0, 0.5, 0.5))
                if prefix == (0,)
                else logs((0.01, 0, 0.99))
                if prefix == (1,)
                else logs((0, 0, 1))
                for prefix in prefixes
            ]

        greedy = beam_search(backend, eos_id=2, beam_size=1, max_tokens=3)
        searched = beam_search(backend, eos_id=2, beam_size=2, max_tokens=3)
        self.assertEqual(greedy["tokens"], [0, 1, 2])
        self.assertEqual(searched["tokens"], [1, 2])
        self.assertGreater(searched["log_probability"], greedy["log_probability"])
        self.assertAlmostEqual(math.exp(searched["log_probability"]), 0.396)

    def test_sufficient_width_matches_exhaustive_constrained_joint_search(self):
        def distribution(prefix):
            return (0.5, 0.3, 0.2) if not prefix else (0.1, 0.7, 0.2) if prefix[-1] == 0 else (0.6, 0.1, 0.3)

        def backend(prefixes):
            return [logs(distribution(prefix)) for prefix in prefixes]

        allowed = lambda prefix: (0, 1) if len(prefix) < 2 else (2,)
        candidates = []
        for prefix in itertools.product((0, 1), repeat=2):
            ids = prefix + (2,)
            score = sum(math.log(distribution(ids[:index])[token]) for index, token in enumerate(ids))
            candidates.append((score, ids))
        expected_score, expected = max(candidates)
        result = beam_search(backend, eos_id=2, beam_size=4, max_tokens=3, allowed_tokens=allowed)
        self.assertEqual(result["tokens"], list(expected))
        self.assertAlmostEqual(result["log_probability"], expected_score)

    def test_constraints_preserve_original_model_scores_without_renormalizing(self):
        result = beam_search(
            lambda prefixes: [logs((0.6, 0.3, 0.1)) if not prefix else logs((0.4, 0.1, 0.5)) for prefix in prefixes],
            eos_id=2,
            beam_size=1,
            max_tokens=2,
            allowed_tokens=lambda prefix: (1,) if not prefix else (2,),
        )
        self.assertEqual(result["tokens"], [1, 2])
        self.assertAlmostEqual(result["log_probability"], math.log(0.15))

    def test_completed_hypotheses_are_never_expanded(self):
        def backend(prefixes):
            self.assertTrue(all(2 not in prefix for prefix in prefixes))
            return [logs((0.1, 0.1, 0.8)) for _ in prefixes]

        result = beam_search(backend, eos_id=2, beam_size=2, max_tokens=3)
        self.assertEqual(result["tokens"], [2])
        self.assertTrue(result["eos_emitted"])

    def test_retained_completion_survives_eviction_by_unfinished_prefixes(self):
        def backend(prefixes):
            return [logs((0.7, 0.1, 0.2)) if not prefix else logs((0.5, 0.5, 0)) for prefix in prefixes]

        result = beam_search(backend, eos_id=2, beam_size=2, max_tokens=3)
        self.assertEqual(result["tokens"], [2])
        self.assertAlmostEqual(math.exp(result["log_probability"]), 0.2)
        self.assertTrue(result["eos_emitted"])
        # Width-one greed never admitted EOS to its beam and must stay greedy.
        greedy = beam_search(backend, eos_id=2, beam_size=1, max_tokens=3)
        self.assertEqual(greedy["tokens"], [0, 0, 0])
        self.assertFalse(greedy["eos_emitted"])

    def test_incomplete_and_dead_end_outputs_remain_explicit(self):
        backend = lambda prefixes: [logs((0.5, 0.5)) for _ in prefixes]
        partial = beam_search(backend, eos_id=1, beam_size=1, max_tokens=2, allowed_tokens=lambda _: (0,))
        self.assertEqual(partial["status"], "token_budget_exhausted")
        self.assertFalse(partial["eos_emitted"])
        dead = beam_search(backend, eos_id=1, allowed_tokens=lambda _: ())
        self.assertEqual(dead["status"], "no_legal_continuation")

    def test_backend_and_mask_contract_errors(self):
        bad = ([0.0, 0.0], [math.nan, 0.0], [math.inf, -math.inf], [-math.inf, -math.inf])
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ValueError):
                beam_search(lambda prefixes: [values for _ in prefixes], eos_id=1)
        for allowed in ((0, 0), (True,), (3,), ([0],)):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                beam_search(
                    lambda prefixes: [logs((0.5, 0.5)) for _ in prefixes], eos_id=1, allowed_tokens=lambda _: allowed
                )
        with self.assertRaises(ValueError):
            beam_search(lambda _: [], eos_id=0)
        for kwargs in (
            {"beam_size": True},
            {"beam_size": 17},
            {"max_tokens": 0},
            {"length_penalty": math.nan},
            {"length_penalty": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                beam_search(lambda prefixes: [[0.0] for _ in prefixes], eos_id=0, **kwargs)

    def test_length_penalty_is_explicit_and_ties_are_deterministic(self):
        hypothesis = Hypothesis((0, 0, 1), -4.0, True)
        self.assertEqual(hypothesis.score(0), -4.0)
        self.assertAlmostEqual(hypothesis.score(1), -3.0)
        backend = lambda prefixes: [logs((0.5, 0.5)) for _ in prefixes]
        self.assertEqual(beam_search(backend, eos_id=1, beam_size=2), beam_search(backend, eos_id=1, beam_size=2))


class ConfigurationTests(unittest.TestCase):
    def test_invalid_attention_and_training_dimensions(self):
        for values in (
            {"d_model": 17},
            {"heads": 0},
            {"patch_size": 2},
            {"dropout": True},
            {"dropout": math.inf},
            {"encoder_layers": 0},
            {"architecture": "unknown"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ModelConfig(**values)


if __name__ == "__main__":
    unittest.main()
