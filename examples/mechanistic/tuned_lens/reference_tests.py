"""Analytical metric checks; no checkpoint, network, model fitting, or study run."""

import math
import copy
import unittest

import torch

from reference import Totals, token_cross_entropy, token_kl, validate_fixture


class MetricProtocolTests(unittest.TestCase):
    def test_kl_direction_is_final_to_lens(self):
        final = torch.tensor([[[0.9, 0.1]]]).log()
        lens = torch.tensor([[[0.6, 0.4]]]).log()
        actual = token_kl(final, lens).item()
        expected = 0.9 * math.log(0.9 / 0.6) + 0.1 * math.log(0.1 / 0.4)
        reversed_kl = 0.6 * math.log(0.6 / 0.9) + 0.4 * math.log(0.4 / 0.1)
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertGreater(abs(actual - reversed_kl), 0.05)

    def test_identical_distributions_have_zero_kl(self):
        probabilities = torch.tensor([[[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]]])
        actual = token_kl(probabilities.log(), probabilities.log())
        torch.testing.assert_close(actual, torch.zeros(1, 2, dtype=torch.float64), atol=1e-7, rtol=0)

    def test_ce_predicts_the_next_token_and_omits_the_final_position(self):
        logits = torch.tensor([[[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]]]).log()
        ids = torch.tensor([[0, 1, 0]])
        losses = token_cross_entropy(logits, ids)
        self.assertEqual(losses.numel(), 2)
        expected = torch.tensor([-math.log(0.2), -math.log(0.3)], dtype=torch.float64)
        torch.testing.assert_close(losses, expected, atol=1e-6, rtol=0)
        logits[:, -1] = torch.tensor([1000.0, -1000.0])
        torch.testing.assert_close(token_cross_entropy(logits, ids), losses)

    def test_aggregation_weights_actual_token_counts(self):
        totals = Totals()
        totals.add(torch.tensor([1.0, 1.0]), torch.tensor([2.0]))
        totals.add(torch.tensor([3.0] * 4), torch.tensor([4.0] * 3))
        summary = totals.means()
        self.assertEqual(summary["kl_tokens"], 6)
        self.assertEqual(summary["ce_tokens"], 4)
        self.assertAlmostEqual(summary["kl_nats_per_token"], 14 / 6)
        self.assertAlmostEqual(summary["ce_nats_per_token"], 14 / 4)

    def test_metrics_reject_accidental_broadcasting_and_empty_records(self):
        with self.assertRaises(ValueError):
            token_kl(torch.zeros(1, 2, 3), torch.zeros(1, 1, 3))
        with self.assertRaises(ValueError):
            token_cross_entropy(torch.zeros(1, 1, 3), torch.zeros(1, 1, dtype=torch.long))
        with self.assertRaises(ValueError):
            Totals().means()

    def test_long_vocabulary_tail_matches_analytical_kl(self):
        vocabulary = 50304
        final_log_probs = torch.full((1, 1, vocabulary), -math.log(vocabulary), dtype=torch.float64)
        candidate_logits = torch.full((1, 1, vocabulary), -20.0, dtype=torch.float32)
        candidate_logits[..., 0] = 0.0
        log_normalizer = math.log1p((vocabulary - 1) * math.exp(-20.0))
        expected = -math.log(vocabulary) + 20 * (vocabulary - 1) / vocabulary + log_normalizer
        self.assertAlmostEqual(token_kl(final_log_probs, candidate_logits).item(), expected, places=11)

    def test_fixture_counts_and_source_order_are_enforced(self):
        protocol = {"records": 4, "max_input_tokens": 128,
                    "reference_positions": 8, "eos_token_id": 0}
        fixture = {
            "records": [{"record_index": i, "input_ids": list(range(8)),
                         "input_tokens": 8, "ce_tokens": 7} for i in range(4)],
            "totals": {"input_tokens": 32, "ce_tokens": 28},
        }
        validate_fixture(fixture, protocol, vocabulary=10)
        wrong_order = copy.deepcopy(fixture)
        wrong_order["records"].reverse()
        with self.assertRaises(ValueError):
            validate_fixture(wrong_order, protocol, vocabulary=10)
        wrong_counts = copy.deepcopy(fixture)
        wrong_counts["records"][0]["ce_tokens"] = 8
        with self.assertRaises(ValueError):
            validate_fixture(wrong_counts, protocol, vocabulary=10)
        wrong_total = copy.deepcopy(fixture)
        wrong_total["totals"]["input_tokens"] = 33
        with self.assertRaises(ValueError):
            validate_fixture(wrong_total, protocol, vocabulary=10)


if __name__ == "__main__":
    unittest.main()
