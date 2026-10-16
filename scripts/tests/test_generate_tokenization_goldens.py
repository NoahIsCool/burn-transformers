"""Pure tests for the tokenizer and chat-template golden generator."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve().parents[1] / "generate_tokenization_goldens.py"
RUST_TESTS = Path(__file__).resolve().parents[2] / "src" / "tokenization_tests.rs"
SPEC = importlib.util.spec_from_file_location("generate_tokenization_goldens", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class TokenizationGoldenTests(unittest.TestCase):
    """Check the generator's inputs and checker without importing Transformers."""

    def setUp(self) -> None:
        """Reads the Rust tests whose goldens the generator regenerates."""

        self.rust = RUST_TESTS.read_text(encoding="utf-8")

    def test_inline_templates_are_the_rust_tests_templates(self) -> None:
        """Every inline template and the FLAN-T5 text appear verbatim in the Rust tests."""

        for template in [
            GENERATOR.UNSET_FIELDS_TEMPLATE,
            GENERATOR.CONTEXT_TEMPLATE,
            GENERATOR.CONTINUE_TEMPLATE,
            GENERATOR.KEY_ORDER_TEMPLATE,
            GENERATOR.FLAN_T5_TEXT,
        ]:
            with self.subTest(template=template[:40]):
                self.assertIn(GENERATOR.rust_string_literal(template), self.rust)
        for token in GENERATOR.FIXTURE_SPECIAL_TOKENS.values():
            self.assertIn(f'"{token}"', self.rust)

    def test_inline_cases_pass_each_tests_options(self) -> None:
        """The renderer sees each case's conversation, template, and switches."""

        calls: list[tuple[Any, dict[str, Any]]] = []

        def render(conversations: Any, **options: Any) -> str:
            """Records one call and returns a placeholder rendering."""

            calls.append((conversations, options))
            return "rendered"

        cases = GENERATOR.inline_cases(render)
        self.assertEqual(set(cases.values()), {"rendered"})
        self.assertEqual(len(calls), len(cases))
        options = {name: call[1] for name, call in zip(cases, calls)}
        self.assertTrue(options["continue_final_message"]["continue_final_message"])
        self.assertEqual(
            options["context_variables_with_special_tokens"]["bos_token"], "<bos>"
        )
        switches = options["context_variables_with_switches"]
        self.assertEqual(switches["documents"], [{"text": "d", "title": "T"}])
        self.assertIs(switches["enable_thinking"], False)
        self.assertEqual(options["map_key_order_python"]["tools"], GENERATOR.KEY_ORDER_TOOLS)

    def test_first_rendering_unwraps_the_reference_result(self) -> None:
        """`render_jinja_template` returns `(renderings, indices)`; the adapter keeps the string."""

        render = GENERATOR.first_rendering(lambda conversations, **_: (["a", "b"], []))
        self.assertEqual(render([[]]), "a")

    def test_conversations_are_key_sorted_like_the_rust_fixtures(self) -> None:
        """Sorted dictionaries make the goldens independent of map ordering."""

        for build in [GENERATOR.tool_use_conversation, GENERATOR.typed_tool_conversation]:
            messages, options = build()
            for tool_call in messages[1]["tool_calls"]:
                self.assertEqual(
                    json.dumps(tool_call), json.dumps(tool_call, sort_keys=True)
                )
            for tool in options["tools"]:
                self.assertEqual(json.dumps(tool), json.dumps(tool, sort_keys=True))
            self.assertTrue(options["add_generation_prompt"])

    def test_every_pinned_tokenizer_variable_gates_a_rust_test(self) -> None:
        """Each environment variable is the one an ignored Rust test reads."""

        self.assertEqual(set(GENERATOR.TOKENIZER_CASES), set(GENERATOR.PINNED_TOKENIZERS))
        for variable in GENERATOR.TOKENIZER_CASES:
            self.assertIn(f'"{variable}"', self.rust)

    def test_checker_finds_strings_and_rustfmt_id_arrays(self) -> None:
        """Strings match as escaped literals; ID arrays ignore whitespace and trailing commas."""

        source = 'let a = "x\\n\\"q\\"";\nlet b = [\n    1, 2,\n    3,\n];'
        goldens = {"text": 'x\n"q"', "ids": [1, 2, 3], "nested": {"more": [1, 2, 3]}}
        self.assertEqual(GENERATOR.missing_goldens(goldens, source), [])
        self.assertEqual(
            GENERATOR.missing_goldens({"text": "y", "ids": [1, 2], "nested": {"z": "q"}}, source),
            ["text", "ids", "nested.z"],
        )
        self.assertEqual(GENERATOR.rust_string_literal("a\\b\tc\r"), '"a\\\\b\\tc\\r"')
        self.assertEqual(GENERATOR.rust_id_array([4, 5]), "[4,5]")

    def test_versions_must_be_the_pinned_releases(self) -> None:
        """Pinned releases pass; any other version stops the run."""

        GENERATOR.check_versions({"transformers": "5.2.0", "tokenizers": "0.22.2"})
        with self.assertRaises(SystemExit):
            GENERATOR.check_versions({"transformers": "5.2.0", "tokenizers": "0.22.1"})


if __name__ == "__main__":
    unittest.main()
