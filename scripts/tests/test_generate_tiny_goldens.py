"""Pure tests for the tiny architecture golden generator and its committed fixtures."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "generate_tiny_goldens.py"
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tiny_goldens"
SPEC = importlib.util.spec_from_file_location("generate_tiny_goldens", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GOLDENS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOLDENS)


class TinyGoldenTests(unittest.TestCase):
    """Check generator invariants and fixture integrity without importing PyTorch."""

    def test_cases_override_only_known_shared_dimensions_consistently(self) -> None:
        """Every case keeps the shared vocabulary and padding token used by the inputs."""

        for name, (model_type, overrides, purpose) in GOLDENS.CASES.items():
            with self.subTest(case=name):
                self.assertTrue(model_type)
                self.assertTrue(purpose.endswith("."))
                self.assertNotIn("vocab_size", overrides)
                self.assertNotIn("pad_token_id", overrides)

    def test_committed_fixtures_match_the_case_inventory(self) -> None:
        """Each case has exactly a config, a checkpoint, and a golden with consistent shapes."""

        self.assertEqual(sorted(path.name for path in FIXTURES.iterdir()), sorted(GOLDENS.CASES))
        for name in GOLDENS.CASES:
            with self.subTest(case=name):
                directory = FIXTURES / name
                self.assertEqual(
                    sorted(path.name for path in directory.iterdir()),
                    ["config.json", "golden.json", "model.safetensors"],
                )
                golden = json.loads((directory / "golden.json").read_text())
                self.assertEqual(golden["case"], name)
                self.assertEqual(golden["checkpoint_sha256"], GOLDENS.file_sha256(directory / "model.safetensors"))
                batch, sequence, vocabulary = golden["logits_shape"]
                self.assertEqual((batch, sequence, vocabulary), (2, GOLDENS.SEQUENCE_LENGTH, GOLDENS.VOCAB_SIZE))
                self.assertEqual(len(golden["logits"]), batch * sequence * vocabulary)
                layers, positions, hidden = golden["hidden_states_row0_shape"]
                self.assertEqual(len(golden["hidden_states_row0"]), layers * positions * hidden)
                self.assertEqual(len(golden["greedy_tokens"]), GOLDENS.GREEDY_NEW_TOKENS)

    def test_left_padded_row_positions_restart_at_zero(self) -> None:
        """The padded row's first valid token has position zero, matching generate()."""

        golden = json.loads((FIXTURES / "llama3" / "golden.json").read_text())
        mask = golden["attention_mask"][1]
        positions = golden["position_ids"][1]
        self.assertEqual(mask, [0] * GOLDENS.PADDING + [1] * (GOLDENS.SEQUENCE_LENGTH - GOLDENS.PADDING))
        self.assertEqual(positions[GOLDENS.PADDING:], list(range(GOLDENS.SEQUENCE_LENGTH - GOLDENS.PADDING)))
        self.assertEqual(golden["position_ids"][0], list(range(GOLDENS.SEQUENCE_LENGTH)))


if __name__ == "__main__":
    unittest.main()
