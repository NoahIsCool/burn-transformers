"""Pure threshold checks for reference-only Qwen prompt selection."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import select_qwen35_08b_reference_prompt as selection


class SelectionTests(unittest.TestCase):
    """No model load or Burn output is involved in these policy checks."""

    def test_fixed_candidate_set_is_bounded_and_distinct(self):
        self.assertEqual(len(selection.CANDIDATES), 12)
        self.assertEqual(len(set(selection.CANDIDATES)), 12)

    def test_each_independent_gate_rejects_a_candidate(self):
        record = {
            "token_ids": [3] * 32, "teacher_token_ids": [3] * 32,
            "decision_stability": [{"stable_under_budget": True}] * 32,
            "prompt_cached_cosine": 0.999, "cached_teacher_cosines": [0.999] * 32,
        }
        self.assertTrue(selection.evidence_passes(record))
        for field, value in [
            ("token_ids", [3] * 31),
            ("teacher_token_ids", [4] * 32),
            ("token_ids", [248_044] * 32),
            ("decision_stability", [{"stable_under_budget": False}] * 32),
            ("prompt_cached_cosine", 0.9989),
            ("cached_teacher_cosines", [0.9989] * 32),
            ("cached_teacher_cosines", [float("nan")] * 32),
        ]:
            changed = copy.deepcopy(record)
            changed[field] = value
            self.assertFalse(selection.evidence_passes(changed), field)


if __name__ == "__main__":
    unittest.main()
