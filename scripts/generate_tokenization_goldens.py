#!/usr/bin/env python3
"""Regenerate the Hugging Face tokenizer and chat-template goldens of the tokenization tests.

``src/tokenization_tests.rs`` pins strings and token IDs that the official
``transformers`` implementation produces. This offline maintainer tool
recomputes every one of them, so the goldens can be audited and regenerated
from the tree:

* **inline templates** need no artifact: they render a template string from
  the Rust test with ``transformers.utils.chat_template_utils.render_jinja_template``
  (unset message fields, the template context variables,
  ``continue_final_message``, and Python dictionary key order);
* **pinned tokenizers** need a local checkpoint directory named by the same
  environment variable the matching ignored Rust test reads
  (``TRANSFORMERS_FLAN_T5_SMALL_DIR``, ``TRANSFORMERS_QWEN35_DIR``,
  ``TRANSFORMERS_NEMOTRON_H_DIR``, ``TRANSFORMERS_MUSE_GLIMMER_DIR``): FLAN-T5
  truncation IDs and the official chat templates of Qwen3.5, Nemotron-H, and
  Muse Glimmer on a tool-use and a typed-argument conversation. A case whose
  directory is not set is skipped and reported.

Tokenizers load with ``local_files_only=True`` and the Hub disabled, so
nothing is downloaded. Usage::

    python3 -I scripts/generate_tokenization_goldens.py [--check src/tokenization_tests.rs]

Without ``--check`` the goldens are printed as JSON. With it, every golden
must appear in the named Rust file, strings as Rust string literals and ID
lists as arrays (whitespace-insensitive); the tool exits nonzero when one is
missing. The run is refused unless the installed ``transformers`` and
``tokenizers`` are the pinned releases.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Tokenizers come from local directories only; never contact a hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#: Releases the committed goldens were generated with.
PINNED_VERSIONS = {"transformers": "5.2.0", "tokenizers": "0.22.2"}

#: Pinned tokenizer revisions named by the Rust tests, keyed by environment variable.
PINNED_TOKENIZERS = {
    "TRANSFORMERS_FLAN_T5_SMALL_DIR": "google/flan-t5-small 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
    "TRANSFORMERS_QWEN35_DIR": "Qwen/Qwen3.5-0.8B 2fc06364715b967f1860aea9cf38778875588b17",
    "TRANSFORMERS_NEMOTRON_H_DIR": (
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
    ),
    "TRANSFORMERS_MUSE_GLIMMER_DIR": "Muse-Glimmer-30B a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
}

# ---------------------------------------------------------------------------
# Inline-template cases (Rust: chat_* tests in src/tokenization_tests.rs)
# ---------------------------------------------------------------------------

#: `chat_messages_omit_unset_fields_like_python_dicts`: a message without a field lacks the key.
UNSET_FIELDS_TEMPLATE = (
    "{%- for message in messages %}\n"
    "    {%- if 'tool_calls' in message %}\n"
    "        {%- if not message.tool_calls|length == 1 %}{{ raise_exception('one tool call') }}{%- endif %}\n"
    "        {%- set call = message.tool_calls[0].function %}\n"
    "        {{- '<|call|>' + call.name + ' ' + call.arguments | tojson + '<|end|>' }}\n"
    "    {%- elif message.role == 'tool' %}\n"
    "        {{- '<|tool ' + message.tool_call_id + '|>' + message.content + '<|end|>' }}\n"
    "    {%- else %}\n"
    "        {{- '<|' + message.role + '|>' + message.content + '<|end|>' }}\n"
    "    {%- endif %}\n"
    "{%- endfor %}"
)
UNSET_FIELDS_MESSAGES = [
    {"role": "user", "content": "weather?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Zürich", "unit": "c"}},
            }
        ],
    },
    {"role": "tool", "content": "21", "tool_call_id": "call_1"},
]

#: `chat_context_matches_hugging_face_variables`: the variables `apply_chat_template` defines.
CONTEXT_TEMPLATE = (
    "{{ bos_token }}|{{ eos_token }}|{{ unk_token }}|{{ pad_token }}|{{ sep_token }}|"
    "{{ cls_token }}|{{ mask_token }}|{{ add_generation_prompt }}|{{ tools }}|{{ documents }}|"
    "{{ enable_thinking is defined }}|{{ add_vision_id is defined }}"
)
#: Special tokens of the Rust fixture tokenizer (`base_config`).
FIXTURE_SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
}

#: `continue_final_message_leaves_the_final_turn_open`: the prompt ends inside the last turn.
CONTINUE_TEMPLATE = "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>\n{% endfor %}"
CONTINUE_MESSAGES = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "The answer is"},
]

#: `chat_map_key_order_is_sorted_until_order_is_preserved`: Python keeps insertion order.
KEY_ORDER_TEMPLATE = (
    "{{ messages[0] }}|{% for k, v in messages[0].items() %}{{ k }};{% endfor %}|{{ tools[0] | tojson }}"
)
KEY_ORDER_TOOLS = [{"type": "function", "function": {"name": "f", "description": "d"}}]


def first_rendering(render_jinja_template: Callable[..., Any]) -> Callable[..., str]:
    """Adapts `render_jinja_template`, which returns `(renderings, generation_indices)`.

    The returned function renders one conversation and returns its string.
    """

    def render(conversations: list[list[dict[str, Any]]], **options: Any) -> str:
        renderings, _ = render_jinja_template(conversations, **options)
        return renderings[0]

    return render


def inline_cases(render: Callable[..., str]) -> dict[str, str]:
    """Renders every inline-template golden; `render` maps a conversation list to one string."""

    return {
        "messages_omit_unset_fields": render(
            [UNSET_FIELDS_MESSAGES], chat_template=UNSET_FIELDS_TEMPLATE
        ),
        "context_variables_with_special_tokens": render(
            [[{"role": "user", "content": "x"}]],
            chat_template=CONTEXT_TEMPLATE,
            **FIXTURE_SPECIAL_TOKENS,
        ),
        "context_variables_with_switches": render(
            [[{"role": "user", "content": "x"}]],
            chat_template=CONTEXT_TEMPLATE,
            add_generation_prompt=True,
            tools=[{"name": "t"}],
            documents=[{"text": "d", "title": "T"}],
            enable_thinking=False,
            add_vision_id=True,
        ),
        "continue_final_message": render(
            [CONTINUE_MESSAGES], chat_template=CONTINUE_TEMPLATE, continue_final_message=True
        ),
        "map_key_order_python": render(
            [[{"role": "user", "content": "hi"}]],
            chat_template=KEY_ORDER_TEMPLATE,
            tools=KEY_ORDER_TOOLS,
        ),
    }


# ---------------------------------------------------------------------------
# Pinned-tokenizer cases (Rust: the ignored *_matches_hugging_face* tests)
# ---------------------------------------------------------------------------

#: `flan_t5_truncation_matches_hugging_face_golden_ids`.
FLAN_T5_TEXT = "translate English to German: the quick brown fox jumps over the lazy dog"
#: `(max_length, truncation side)` pairs of the FLAN-T5 test, in order.
FLAN_T5_TRUNCATIONS = ((8, "right"), (8, "left"), (2, "right"), (2, "left"), (1, "right"), (1, "left"))


def tool_use_conversation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The Rust `tool_use_conversation`: messages and `apply_chat_template` keyword arguments.

    Dictionaries are key-sorted, as in the Rust fixture, so the goldens do not
    depend on whether JSON maps preserve insertion order.
    """

    messages = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"arguments": {"city": "Paris"}, "name": "get_weather"}, "type": "function"}
            ],
        },
        {"role": "tool", "content": "18C", "name": "get_weather", "tool_call_id": "call_0"},
    ]
    options = {
        "add_generation_prompt": True,
        "tools": [
            {
                "function": {
                    "description": "Get the weather for a city.",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {"city": {"description": "City name", "type": "string"}},
                        "required": ["city"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
        "current_date": "2026-09-05",
        "reasoning_strength": "low",
    }
    return messages, options


def typed_tool_conversation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The Rust `typed_tool_conversation`: booleans, null, a float, and a list in arguments and schema."""

    arguments = {"count": 3, "flag": True, "items": [1, 2], "name": "x", "none_val": None, "ratio": 1.5}
    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"arguments": arguments, "name": "f"}, "type": "function"}],
        },
        {"role": "tool", "content": "ok"},
    ]
    options = {
        "add_generation_prompt": True,
        "tools": [
            {
                "function": {
                    "description": "d",
                    "name": "f",
                    "parameters": {
                        "properties": {
                            "flag": {
                                "default": False,
                                "description": "a flag",
                                "nullable": True,
                                "type": "boolean",
                            }
                        },
                        "required": ["flag"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ],
    }
    return messages, options


def flan_t5_cases(tokenizer: Any) -> dict[str, list[int]]:
    """Token IDs of the FLAN-T5 truncation test, keyed by case."""

    cases = {"flan_t5_full": tokenizer(FLAN_T5_TEXT)["input_ids"]}
    for max_length, side in FLAN_T5_TRUNCATIONS:
        tokenizer.truncation_side = side
        cases[f"flan_t5_{max_length}_{side}"] = tokenizer(
            FLAN_T5_TEXT, truncation=True, max_length=max_length
        )["input_ids"]
    tokenizer.truncation_side = "right"
    cases["flan_t5_4_right_without_special_tokens"] = tokenizer(
        FLAN_T5_TEXT, add_special_tokens=False, truncation=True, max_length=4
    )["input_ids"]
    return cases


def template_case(tokenizer: Any, conversation: tuple[list[dict[str, Any]], dict[str, Any]]) -> str:
    """Renders one conversation with a tokenizer's official template, untokenized."""

    messages, options = conversation
    return tokenizer.apply_chat_template(messages, tokenize=False, **options)


#: Pinned-tokenizer cases: environment variable -> case name -> golden builder.
TOKENIZER_CASES: Mapping[str, Mapping[str, Callable[[Any], Any]]] = {
    "TRANSFORMERS_FLAN_T5_SMALL_DIR": {"flan_t5": flan_t5_cases},
    "TRANSFORMERS_QWEN35_DIR": {
        "qwen35_tool_prompt": lambda tokenizer: template_case(tokenizer, tool_use_conversation()),
        "qwen35_typed_tool_arguments": lambda tokenizer: template_case(
            tokenizer, typed_tool_conversation()
        ),
    },
    "TRANSFORMERS_NEMOTRON_H_DIR": {
        "nemotron_plain_prompt": lambda tokenizer: template_case(
            tokenizer, ([{"role": "user", "content": "Hello"}], {"add_generation_prompt": True})
        ),
        "nemotron_typed_tool_schema_and_arguments": lambda tokenizer: template_case(
            tokenizer, typed_tool_conversation()
        ),
    },
    "TRANSFORMERS_MUSE_GLIMMER_DIR": {
        "muse_glimmer_tool_prompt": lambda tokenizer: template_case(tokenizer, tool_use_conversation()),
    },
}

# ---------------------------------------------------------------------------
# Checking against the Rust test file
# ---------------------------------------------------------------------------


def rust_string_literal(value: str) -> str:
    """Renders `value` as the Rust string literal the tests commit.

    Backslashes, double quotes, newlines, carriage returns, and tabs are
    escaped; every other character, non-ASCII included, is kept verbatim.
    """

    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def rust_id_array(ids: Sequence[int]) -> str:
    """Renders token IDs as a whitespace-free Rust array, `[1,2,3]`."""

    return "[" + ",".join(str(int(token)) for token in ids) + "]"


def without_whitespace(source: str) -> str:
    """Removes whitespace and trailing commas before `]`, so rustfmt line breaks do not matter."""

    return re.sub(r",\]", "]", re.sub(r"\s+", "", source))


def missing_goldens(goldens: Mapping[str, Any], rust_source: str) -> list[str]:
    """Names every golden that does not appear in `rust_source`.

    Strings must appear as their exact Rust literal; ID lists as arrays,
    ignoring whitespace and trailing commas (rustfmt's layout).
    """

    compact = without_whitespace(rust_source)
    missing = []
    for name, value in goldens.items():
        if isinstance(value, str):
            if rust_string_literal(value) not in rust_source:
                missing.append(name)
        elif isinstance(value, Mapping):
            missing.extend(f"{name}.{inner}" for inner in missing_goldens(value, rust_source))
        elif rust_id_array(value) not in compact:
            missing.append(name)
    return missing


def check_versions(installed: Mapping[str, str]) -> None:
    """Refuses to run against reference releases other than the pinned ones."""

    for package, pinned in PINNED_VERSIONS.items():
        version = installed.get(package)
        if version is None or version.split("+")[0] != pinned:
            raise SystemExit(f"{package} {version} is installed; the goldens are pinned to {pinned}")


def generate(environment: Mapping[str, str]) -> tuple[dict[str, Any], list[str]]:
    """Computes every available golden; returns them and the skipped environment variables."""

    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    from transformers.utils.chat_template_utils import render_jinja_template

    check_versions({"transformers": transformers.__version__, "tokenizers": tokenizers.__version__})
    goldens: dict[str, Any] = dict(inline_cases(first_rendering(render_jinja_template)))
    skipped = []
    for variable, cases in TOKENIZER_CASES.items():
        directory = environment.get(variable)
        if not directory:
            skipped.append(variable)
            continue
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        for name, build in cases.items():
            golden = build(tokenizer)
            if isinstance(golden, dict):
                goldens.update(golden)
            else:
                goldens[name] = golden
    return goldens, skipped


def main(argv: Sequence[str] | None = None) -> int:
    """Prints the goldens as JSON, or checks that a Rust test file commits each of them."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--check", type=Path, help="Rust test file that must contain every golden")
    arguments = parser.parse_args(argv)
    goldens, skipped = generate(os.environ)
    for variable in skipped:
        print(f"skipped {variable} ({PINNED_TOKENIZERS[variable]}): not set", file=sys.stderr)
    if arguments.check is None:
        print(json.dumps(goldens, ensure_ascii=False, indent=2))
        return 0
    missing = missing_goldens(goldens, arguments.check.read_text(encoding="utf-8"))
    if missing:
        print(f"{arguments.check}: goldens not found: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"{arguments.check}: all {len(goldens)} regenerated goldens are committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
