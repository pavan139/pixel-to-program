import unittest

from pixel_to_program.decoding import allowed_program_tokens, primitive_metrics
from pixel_to_program.model import TOKEN_TO_ID, decode
from pixel_to_program.scene import Scene, Shape


class GrammarAndExtractionTests(unittest.TestCase):
    def test_geometry_aware_width_and_height(self):
        self.assertEqual(allowed_program_tokens([]), ("SCENE",))
        self.assertEqual(allowed_program_tokens(["SCENE", "RECT", "RED", "31", "30"]), ("1",))
        self.assertEqual(allowed_program_tokens(["SCENE", "RECT", "RED", "31", "30", "1"]), ("1", "2"))

    def test_shape_budget_and_completion(self):
        prefix = ["SCENE"] + ["RECT", "RED", "0", "0", "1", "1"] * 4
        self.assertEqual(allowed_program_tokens(prefix), ("END",))
        self.assertEqual(allowed_program_tokens(prefix + ["END"]), ("<eos>",))
        self.assertEqual(allowed_program_tokens(prefix + ["END", "<eos>"]), ("<pad>",))
        with self.assertRaises(ValueError):
            allowed_program_tokens(prefix + ["RECT"])

    def test_bad_prefixes_and_internal_special_tokens(self):
        for prefix in (["<bos>"], ["SCENE", "END", "RECT"], ["SCENE", "RECT", "BLACK"], ["SCENE", "RECT", "RED", "32"]):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                allowed_program_tokens(prefix)
        for values in (["SCENE", "<pad>", "END", "<eos>"], ["SCENE", "<bos>", "END", "<eos>"], ["SCENE", "END"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                decode([TOKEN_TO_ID[token] for token in values])

    def test_eos_is_required_and_right_padding_is_ignored(self):
        self.assertEqual(
            decode([TOKEN_TO_ID[token] for token in ("<bos>", "SCENE", "END", "<eos>", "<pad>")]), "SCENE END"
        )
        with self.assertRaises(ValueError):
            decode([TOKEN_TO_ID["SCENE"]] * 34)
        with self.assertRaises(ValueError):
            allowed_program_tokens(["SCENE"] * 33)

    def test_primitive_metrics_do_not_reward_wrong_color_or_geometry(self):
        blue = Shape("ELLIPSE", "BLUE", 1, 1, 4, 4)
        wrong_color = Shape("ELLIPSE", "RED", 1, 1, 4, 4)
        wrong_geometry = Shape("ELLIPSE", "BLUE", 2, 1, 4, 4)
        for prediction in (wrong_color, wrong_geometry):
            result = primitive_metrics(Scene((blue,)), Scene((prediction,)), ("ELLIPSE", "BLUE"))
            self.assertEqual(result["primitive_recall"], 0)
            self.assertEqual(result["heldout_exact_matches"], 0)
        self.assertTrue(
            primitive_metrics(Scene((blue,)), Scene((wrong_geometry,)), ("ELLIPSE", "BLUE"))["heldout_pair_present"]
        )

    def test_multiset_counts_duplicates_and_ignores_order(self):
        a = Shape("RECT", "RED", 0, 0, 4, 4)
        b = Shape("RECT", "BLUE", 5, 5, 4, 4)
        result = primitive_metrics(Scene((a, b)), Scene((b, a)))
        self.assertEqual(result["primitive_f1"], 1)
        result = primitive_metrics(Scene((a, b)), Scene((a, a)))
        self.assertEqual(result["primitive_true_positive"], 1)
        self.assertEqual(result["primitive_false_positive"], 1)


if __name__ == "__main__":
    unittest.main()
