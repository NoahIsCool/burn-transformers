"""Check the committed scientific protocol without weights, downloads, or training.

    python3 -m unittest discover -s examples/mechanistic/grokking_training -p 'test_*.py' -v
"""

import cmath
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("grokking_training_plot", ROOT / "plot.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def observation(step=0, train=0.1, test=0.1):
    return dict(step=step, train_loss=2.0, test_loss=3.0,
                train_accuracy=train, test_accuracy=test,
                embedding_fourier_power=[1 / 56] * 56,
                readout_fourier_power=[1 / 56] * 56)


class FrozenProtocolTests(unittest.TestCase):
    def test_exact_author_split_and_pinned_bytes(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        split_bytes = (ROOT / "train_split.json").read_bytes()
        fixture = json.loads(split_bytes)
        pairs = [(a, b, 113) for a in range(113) for b in range(113)]
        random.Random(0).shuffle(pairs)
        train = fixture["train_indices"]
        self.assertEqual(train, [a * 113 + b for a, b, _ in pairs[:3830]])
        self.assertEqual(len(set(train)), 3830)
        self.assertTrue(all(0 <= pair < 113 * 113 for pair in train))
        self.assertEqual(len(set(range(113 * 113)) - set(train)), 8939)
        self.assertEqual(hashlib.sha256(split_bytes).hexdigest(), manifest["data"]["train_split_sha256"])

    def test_mainline_configuration_preserves_training_output_class(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        architecture = manifest["architecture"]
        self.assertEqual([architecture[key] for key in ("prime", "hidden_width", "heads", "head_width", "mlp_width", "layers")],
                         [113, 128, 4, 32, 512, 1])
        self.assertEqual(architecture["training_output_vocabulary"], 114)
        self.assertEqual(architecture["analysis_output_vocabulary"], 113)
        self.assertEqual(manifest["initialization"]["standard_deviation"]["mlp_output"], "1/sqrt(128)")
        self.assertEqual(manifest["initialization"]["standard_deviation"]["unembedding"], "1/sqrt(114)")
        optimizer = manifest["optimizer"]
        self.assertEqual([optimizer[key] for key in ("learning_rate", "weight_decay", "beta1", "beta2", "epsilon")],
                         [0.001, 1.0, 0.9, 0.98, 1e-8])
        self.assertEqual(optimizer["optimizer_updates"], 40000)
        self.assertFalse(optimizer["early_stopping"])

    def test_measurement_schedule_includes_baseline_and_full_run(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        expected = [0, *range(100, 1001, 100), *range(1500, 40001, 500)]
        self.assertEqual(manifest["measurement"]["progress_update_counts"], expected)
        criterion = manifest["acceptance"]["criterion"]
        for threshold in ("> 0.99", "< 0.50", "strictly later"):
            self.assertIn(threshold, criterion)

    def test_criterion_is_strict_and_requires_later_generalization(self):
        self.assertEqual(plot.delayed_generalization([observation(0, 0.99, 0.1), observation(100, 1, 1)]), (None, None))
        self.assertEqual(plot.delayed_generalization([observation(0, 1, 0.5), observation(100, 1, 1)]), (None, None))
        self.assertEqual(plot.delayed_generalization([observation(0, 1, 0.1), observation(100, 1, 0.99)]), (0, None))
        self.assertEqual(plot.delayed_generalization([observation(0, 1, 1), observation(100, 1, 0.1), observation(200, 1, 1)]),
                         (100, 200))

    def test_frequency_indices_are_one_based_with_explicit_tie_order(self):
        self.assertEqual(plot.strongest_frequencies([0.1, 0.3, 0.3, 0.1, 0.2], count=3), [2, 3, 5])

    def test_fourier_pair_selection_matches_an_independent_complex_dft(self):
        # cos(7x) has amplitude 2 and sin(19x) amplitude 1: powers are 4/5, 1/5.
        prime = 113
        signal = [2 * math.cos(math.tau * 7 * x / prime) + math.sin(math.tau * 19 * x / prime)
                  for x in range(prime)]
        coefficients = [sum(value * cmath.exp(-1j * math.tau * k * x / prime)
                            for x, value in enumerate(signal)) for k in range(prime)]
        powers = [abs(coefficients[k]) ** 2 + abs(coefficients[prime - k]) ** 2 for k in range(1, 57)]
        total = math.fsum(powers)
        powers = [value / total for value in powers]
        self.assertAlmostEqual(powers[6], 0.8, places=12)
        self.assertAlmostEqual(powers[18], 0.2, places=12)
        self.assertEqual(plot.strongest_frequencies(powers, count=2), [7, 19])

    def test_affine_sinusoid_variance_explained_separates_constant_offset(self):
        # An independent least-squares calculation with explicit centering.
        # On the full modular grid, x and an unrelated sine are orthogonal.
        prime = 113
        x = [math.cos(math.tau * 7 * (a + b) / prime)
             for a in range(prime) for b in range(prime)]
        noise = [0.5 * math.sin(math.tau * 19 * (a + b) / prime)
                 for a in range(prime) for b in range(prime)]
        count = len(x)
        mean_x = math.fsum(x) / count
        self.assertAlmostEqual(mean_x, 0, places=12)
        for offset in (0, 3, 10):
            y = [offset + 2 * value + residual for value, residual in zip(x, noise)]
            mean_y = math.fsum(y) / count
            variance_x = math.fsum((value - mean_x) ** 2 for value in x)
            variance_y = math.fsum((value - mean_y) ** 2 for value in y)
            covariance = math.fsum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
            slope = covariance / variance_x
            fitted = [mean_y + slope * (value - mean_x) for value in x]
            residual_sum_squares = math.fsum((a - b) ** 2 for a, b in zip(y, fitted))
            ordinary_r_squared = 1 - residual_sum_squares / variance_y
            # Signal variance is 2, noise variance is 1/8, so R² is 16/17.
            self.assertAlmostEqual(ordinary_r_squared, 16 / 17, places=12)
            x2 = math.fsum(value * value for value in x)
            xy = math.fsum(a * b for a, b in zip(x, y))
            simplified_affine = (xy * xy / x2) / variance_y
            self.assertAlmostEqual(simplified_affine, ordinary_r_squared, places=12)
            legacy = 1 - (math.fsum(value * value for value in y) - xy * xy / x2) / variance_y
            self.assertAlmostEqual(legacy, (16 - 8 * offset * offset) / 17, places=11)
            if offset:
                self.assertLess(legacy, 0)

    def test_plot_input_rejects_misordered_steps_nonfinite_and_bad_power(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.json"
            valid = [observation(0), observation(100)]
            path.write_text(json.dumps(valid))
            self.assertEqual(plot.load_progress(path), valid)
            invalid_cases = [[], list(reversed(valid)), [observation(0), observation(0)]]
            for field, value in (("train_accuracy", 99), ("test_loss", float("nan")),
                                 ("readout_fourier_power", [1] * 56), ("embedding_fourier_power", [1])):
                wrong = copy.deepcopy(valid)
                wrong[-1][field] = value
                invalid_cases.append(wrong)
            for invalid in invalid_cases:
                with self.subTest(invalid=invalid):
                    path.write_text(json.dumps(invalid))
                    with self.assertRaises(ValueError):
                        plot.load_progress(path)


if __name__ == "__main__":
    unittest.main()
