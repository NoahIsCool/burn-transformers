"""Pure tests for the DiffLlama attention golden generator and the committed constant."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "generate_diffllama_reference.py"
RUST_TESTS = Path(__file__).resolve().parents[2] / "src" / "attention" / "differential_tests.rs"
SPEC = importlib.util.spec_from_file_location("generate_diffllama_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class DiffLlamaReferenceTests(unittest.TestCase):
    """Check the generator's pure helpers without importing PyTorch or Transformers."""

    def test_fixture_values_follow_the_rust_helper(self) -> None:
        """Entries are `(((37 i + 11 seed) mod 23) 0.1 - 1.1) scale / 0.1`, rounded to F32."""

        values = GENERATOR.fixture_values(3, 9, 0.1)
        # (99 mod 23) = 7, (136 mod 23) = 21, (173 mod 23) = 12.
        expected = [GENERATOR.to_f32(base * 0.1 - 1.1) for base in (7, 21, 12)]
        self.assertEqual(values, expected)
        self.assertEqual(GENERATOR.to_f32(0.1), 0.10000000149011612)
        self.assertEqual(len(GENERATOR.fixture_values(24, 1, 0.05)), 24)

    def test_rust_float_matches_the_committed_literal_style(self) -> None:
        """Nine significant digits with fraction digits grouped in threes."""

        for value, literal in [
            (0.226941466, "0.226_941_466"),
            (-0.81052202, "-0.810_522_02"),
            (0.0763680786, "0.076_368_078_6"),
            (-0.00463789701, "-0.004_637_897_01"),
            (0.4194327, "0.419_432_7"),
            (1.0, "1.0"),
        ]:
            with self.subTest(value=value):
                self.assertEqual(GENERATOR.rust_float(value), literal)

    def test_committed_constant_parses_into_the_documented_layouts(self) -> None:
        """The Rust file holds 24 outputs for each of the kv = 2, 1, 4 layouts."""

        committed = GENERATOR.committed_constant(RUST_TESTS.read_text(encoding="utf-8"))
        references = GENERATOR.parse_constant(committed)
        self.assertEqual([heads for heads, _ in references], list(GENERATOR.KEY_VALUE_HEADS))
        for _, values in references:
            self.assertEqual(len(values), GENERATOR.HIDDEN_SIZE * GENERATOR.TOKENS)
        rendered = GENERATOR.rust_constant(references)
        self.assertEqual(GENERATOR.f32_mismatches(references, GENERATOR.parse_constant(rendered)), [])

    def test_f32_comparison_ignores_spelling_but_not_values(self) -> None:
        """Two literals of one F32 agree; a different F32 or layout is reported."""

        committed = [(2, [-0.0404148551, 0.5])]
        self.assertEqual(GENERATOR.f32_mismatches(committed, [(2, [-0.0404148549, 0.5])]), [])
        problems = GENERATOR.f32_mismatches(committed, [(2, [-0.0404148600, 0.5])])
        self.assertEqual(len(problems), 1)
        self.assertIn("kv=2[0]", problems[0])
        self.assertEqual(
            GENERATOR.f32_mismatches(committed, [(1, [0.0, 0.5])]),
            ["key/value head layouts differ"],
        )
        self.assertIn("1 regenerated", GENERATOR.f32_mismatches(committed, [(2, [0.5])])[0])

    def test_versions_must_be_the_pinned_releases(self) -> None:
        """Pinned releases (with or without a local build suffix) pass; others stop the run."""

        GENERATOR.check_versions({"transformers": "5.2.0", "torch": "2.10.0+cpu"})
        with self.assertRaises(SystemExit):
            GENERATOR.check_versions({"transformers": "5.3.0", "torch": "2.10.0"})
        with self.assertRaises(SystemExit):
            GENERATOR.check_versions({"transformers": "5.2.0"})


if __name__ == "__main__":
    unittest.main()
