"""Independent analytic checks for training diagnostics (requires NumPy)."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("clock_training_analysis", Path(__file__).with_name("analyze.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class Diagnostics(unittest.TestCase):
    def fixture(self, depend_on_difference):
        p = analysis.P
        tokens = np.arange(p)
        embedding = np.zeros((p, analysis.HIDDEN))
        for pair, frequency in enumerate([3, 7, 11]):
            embedding[:, pair * 2] = (3 - pair) * np.cos(2 * np.pi * frequency * tokens / p)
            embedding[:, pair * 2 + 1] = (3 - pair) * np.sin(2 * np.pi * frequency * tokens / p)
        a, b = tokens[:, None], tokens[None, :]
        phase = a - b if depend_on_difference else a + b
        correct = np.cos(2 * np.pi * phase / p)
        logits = np.zeros((p, p, p))
        logits[a, b, (a + b) % p] = correct
        return {"embedding": embedding, "logits": logits}

    def test_pure_circles_and_difference_dependent_logits(self):
        report, _, _ = analysis.geometry(self.fixture(True))
        self.assertAlmostEqual(report["circularity_first_four_pcs"], 1, places=12)
        self.assertLess(report["distance_irrelevance"], 1e-14)
        self.assertEqual(report["dominant_frequency_first_six_pcs"], [3, 3, 7, 7, 11, 11])

    def test_sum_dependent_logits_have_unit_distance_irrelevance(self):
        report, _, _ = analysis.geometry(self.fixture(False))
        self.assertAlmostEqual(report["distance_irrelevance"], 1, places=12)


if __name__ == "__main__":
    unittest.main()
