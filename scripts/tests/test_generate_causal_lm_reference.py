"""Tests for the generic causal-LM reference generator.

`ReferenceGeneratorTests` is pure (no PyTorch import). `ModelLoadingTests`
needs PyTorch and Transformers and is skipped without them; it loads the
committed tiny fixtures, BF16-stored copies of them, and a tiny random MXFP4
gpt-oss checkpoint written to a temporary directory, so it runs offline in a
few seconds.
"""

from __future__ import annotations

import importlib.util
import math
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "generate_causal_lm_reference.py"
SPEC = importlib.util.spec_from_file_location("generate_causal_lm_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)

#: Whether the model-loading tests can run.
HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))
#: Tiny seeded gpt-oss checkpoint (F32 weights, YaRN RoPE) committed for the Rust goldens.
GPT_OSS_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tiny_goldens" / "gpt-oss"
#: Tiny Qwen2 checkpoint (QKV biases, untied head) for dense F64 evaluation.
QWEN2_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tiny_goldens" / "qwen2"
#: Tiny Gemma3 text checkpoint, whose `embed_scale` buffer takes the default dtype.
GEMMA3_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tiny_goldens" / "gemma3-text"
#: Prompt evaluated by the model-loading comparisons.
TOKENS = [1, 5, 9, 14, 3, 27, 40, 8]


def bfloat16_copy(source: Path, destination: Path) -> Path:
    """Writes `source`'s configuration and weights to `destination`, every floating tensor stored as BF16.

    The copy is what a BF16 release of the fixture would be, so a plain F32
    load of it is the truth that `--bf16-resident` must reproduce.
    """

    import torch
    from safetensors.torch import load_file, save_file

    destination.mkdir(parents=True)
    (destination / "config.json").write_bytes((source / "config.json").read_bytes())
    for path in REFERENCE.weight_files(source):
        tensors = load_file(path)
        save_file(
            {name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
             for name, tensor in tensors.items()},
            destination / path.name,
            metadata={"format": "pt"},
        )
    return destination


def write_mxfp4_gpt_oss(destination: Path) -> Path:
    """Writes a tiny random gpt-oss checkpoint laid out like openai/gpt-oss-20b at 6cee5e81.

    The configuration keeps that checkpoint's keys (legacy ``rope_scaling``
    YaRN with a top-level ``rope_theta``, ``swiglu_limit``, and the MXFP4
    ``quantization_config``) with tiny dimensions. Experts are stored as U8
    MXFP4 blocks and scales and everything else as BF16, the dtypes of the
    real shards, so Transformers decodes the experts to BF16 on the CPU.
    """

    import torch
    from safetensors.torch import save_file

    hidden, intermediate, experts, heads, kv_heads, head_dim, vocabulary, layers = 64, 64, 4, 4, 2, 16, 128, 2
    config = {
        "architectures": ["GptOssForCausalLM"],
        "attention_bias": True,
        "attention_dropout": 0.0,
        "eos_token_id": 2,
        "experts_per_token": 2,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": hidden,
        "initial_context_length": 4096,
        "initializer_range": 0.02,
        "intermediate_size": intermediate,
        "layer_types": ["sliding_attention", "full_attention"],
        "max_position_embeddings": 131072,
        "model_type": "gpt_oss",
        "num_attention_heads": heads,
        "num_experts_per_tok": 2,
        "num_hidden_layers": layers,
        "num_key_value_heads": kv_heads,
        "num_local_experts": experts,
        "output_router_logits": False,
        "pad_token_id": vocabulary - 1,
        "quantization_config": {
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head",
            ],
            "quant_method": "mxfp4",
        },
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "yarn",
            "truncate": False,
        },
        "rope_theta": 150000,
        "router_aux_loss_coef": 0.9,
        "sliding_window": 128,
        "swiglu_limit": 7.0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": vocabulary,
    }
    generator = torch.Generator().manual_seed(0)

    def bf16(*shape: int, scale: float = 0.05, offset: float = 0.0):
        """Returns seeded normal BF16 values."""

        return (offset + scale * torch.randn(*shape, generator=generator)).to(torch.bfloat16)

    def u8(*shape: int, low: int = 0, high: int = 256):
        """Returns seeded uniform bytes in `[low, high)`."""

        return torch.randint(low, high, shape, generator=generator, dtype=torch.uint8)

    tensors = {
        "model.embed_tokens.weight": bf16(vocabulary, hidden),
        "lm_head.weight": bf16(vocabulary, hidden),
        "model.norm.weight": bf16(hidden, offset=1.0),
    }
    projections = {
        "q_proj": (heads * head_dim, hidden),
        "k_proj": (kv_heads * head_dim, hidden),
        "v_proj": (kv_heads * head_dim, hidden),
        "o_proj": (hidden, heads * head_dim),
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        tensors[prefix + "input_layernorm.weight"] = bf16(hidden, offset=1.0)
        tensors[prefix + "post_attention_layernorm.weight"] = bf16(hidden, offset=1.0)
        tensors[prefix + "mlp.router.weight"] = bf16(experts, hidden)
        tensors[prefix + "mlp.router.bias"] = bf16(experts)
        # MXFP4: 32 values per block, two E2M1 nibbles per byte, one biased E8M0 scale per block.
        tensors[prefix + "mlp.experts.gate_up_proj_blocks"] = u8(experts, 2 * intermediate, hidden // 32, 16)
        tensors[prefix + "mlp.experts.gate_up_proj_scales"] = u8(experts, 2 * intermediate, hidden // 32, low=118, high=124)
        tensors[prefix + "mlp.experts.gate_up_proj_bias"] = bf16(experts, 2 * intermediate)
        tensors[prefix + "mlp.experts.down_proj_blocks"] = u8(experts, hidden, intermediate // 32, 16)
        tensors[prefix + "mlp.experts.down_proj_scales"] = u8(experts, hidden, intermediate // 32, low=118, high=124)
        tensors[prefix + "mlp.experts.down_proj_bias"] = bf16(experts, hidden)
        for name, shape in projections.items():
            tensors[prefix + f"self_attn.{name}.weight"] = bf16(*shape)
            tensors[prefix + f"self_attn.{name}.bias"] = bf16(shape[0])
        tensors[prefix + "self_attn.sinks"] = bf16(heads, scale=1.0)
    destination.mkdir(parents=True)
    (destination / "config.json").write_text(REFERENCE.json.dumps(config, indent=1))
    save_file(tensors, destination / "model.safetensors", metadata={"format": "pt"})
    return destination


class FakeTensor:
    """Stands in for a contiguous CPU tensor whose `numpy()` returns an array."""

    def __init__(self, array):
        """Wraps an array."""

        self.array = array

    def contiguous(self) -> "FakeTensor":
        """Returns itself, as a contiguous tensor would."""

        return self

    def numpy(self):
        """Returns the wrapped array."""

        return self.array


class ReferenceGeneratorTests(unittest.TestCase):
    """Check the format helpers, argument handling, and safe publication."""

    def test_offline_controls_are_installed_at_import(self) -> None:
        """Importing the script switches every library to offline mode."""

        import os

        for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            self.assertEqual(os.environ[variable], "1")

    def test_f32le_round_trips_and_is_little_endian(self) -> None:
        """Values survive encoding exactly and use the documented byte order."""

        values = [0.0, -1.5, 3.25, 2.0**-100, float("inf")]
        payload = REFERENCE.encode_f32le(values)
        self.assertEqual(payload[:8], struct.pack("<2f", 0.0, -1.5))
        self.assertEqual(REFERENCE.decode_f32le(payload), values)
        with self.assertRaises(ValueError):
            REFERENCE.decode_f32le(b"\x00\x00\x00")

    def test_tensor_serialization_matches_the_reference_encoding(self) -> None:
        """The fast tensor path writes the same bytes as the pure encoder."""

        try:
            import numpy
        except ImportError:  # pragma: no cover - numpy ships with PyTorch.
            self.skipTest("numpy is unavailable")
        array = numpy.array([[1.0, -2.0], [0.5, 7.25]], dtype=numpy.float32)
        self.assertEqual(
            REFERENCE.tensor_f32le(FakeTensor(array)),
            REFERENCE.encode_f32le([1.0, -2.0, 0.5, 7.25]),
        )

    def test_top_two_margin_prefers_the_lowest_index_on_ties(self) -> None:
        """Margins are the gap to the runner-up; ties have a zero margin."""

        self.assertEqual(REFERENCE.top_two_margin([0.0, 3.0, 1.0]), (1, 2.0))
        token, margin = REFERENCE.top_two_margin([2.0, 5.0, 5.0])
        self.assertEqual((token, margin), (1, 0.0))
        self.assertTrue(math.isfinite(REFERENCE.top_two_margin([-1.0, -2.0])[1]))
        with self.assertRaises(ValueError):
            REFERENCE.top_two_margin([1.0])

    def test_arguments_default_to_the_fixed_prompts(self) -> None:
        """Without prompts the three fixed prompts and 32 greedy tokens are used."""

        arguments = REFERENCE.parse_arguments(["--model-dir", "m", "--output", "o"])
        self.assertEqual(arguments.prompts, list(REFERENCE.DEFAULT_PROMPTS))
        self.assertEqual(arguments.greedy_tokens, 32)
        self.assertTrue(arguments.add_special_tokens)
        custom = REFERENCE.parse_arguments(
            ["--model-dir", "m", "--output", "o", "--prompt", "a", "--prompt", "b",
             "--greedy-tokens", "4", "--no-special-tokens", "--revision", "abc"]
        )
        self.assertEqual(custom.prompts, ["a", "b"])
        self.assertEqual((custom.greedy_tokens, custom.add_special_tokens, custom.revision), (4, False, "abc"))
        with self.assertRaises(SystemExit):
            REFERENCE.parse_arguments(["--model-dir", "m", "--output", "o", "--greedy-tokens", "0"])

    def test_dtype_defaults_to_float32_and_accepts_float64_and_both_half_precisions(self) -> None:
        """The evaluation dtype is F32 unless F64, BF16, or F16 is requested; others are refused."""

        base = ["--model-dir", "m", "--output", "o"]
        self.assertEqual(REFERENCE.parse_arguments(base).dtype, "float32")
        for dtype in ("float64", "bfloat16", "float16"):
            self.assertEqual(REFERENCE.parse_arguments(base + ["--dtype", dtype]).dtype, dtype)
        self.assertEqual(REFERENCE.DTYPES, ("float32", "float64", "bfloat16", "float16"))
        self.assertEqual(REFERENCE.HALF_DTYPES, ("bfloat16", "float16"))
        for dtype in ("half", "float8", "fp16"):
            with self.subTest(dtype=dtype):
                with self.assertRaises(SystemExit):
                    REFERENCE.parse_arguments(base + ["--dtype", dtype])
                with self.assertRaises(ValueError):
                    REFERENCE.load_model(Path("m"), dtype)

    def test_truncation_keeps_the_first_layers(self) -> None:
        """Truncation cuts the depth and the layer schedule; impossible depths are refused."""

        from types import SimpleNamespace

        config = SimpleNamespace(num_hidden_layers=4, layer_types=["s", "f", "s", "f"])
        REFERENCE.truncate_config(config, 2)
        self.assertEqual((config.num_hidden_layers, config.layer_types), (2, ["s", "f"]))
        plain = SimpleNamespace(num_hidden_layers=3)
        REFERENCE.truncate_config(plain, 3)
        self.assertEqual(plain.num_hidden_layers, 3)
        for layers in (0, 4):
            with self.assertRaises(ValueError):
                REFERENCE.truncate_config(plain, layers)
        base = ["--model-dir", "m", "--output", "o"]
        self.assertIsNone(REFERENCE.parse_arguments(base).num_hidden_layers)
        self.assertEqual(REFERENCE.parse_arguments(base + ["--num-hidden-layers", "1"]).num_hidden_layers, 1)
        with self.assertRaises(SystemExit):
            REFERENCE.parse_arguments(base + ["--num-hidden-layers", "0"])

    def test_publication_never_replaces_a_bundle(self) -> None:
        """A finished staging directory is renamed once; an existing bundle is kept."""

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            staging = root / ".bundle.tmp"
            staging.mkdir()
            (staging / "reference.json").write_text("{}")
            destination = root / "bundle"
            REFERENCE.publish(staging, destination)
            self.assertTrue((destination / "reference.json").is_file())
            again = root / ".bundle.again"
            again.mkdir()
            with self.assertRaises(FileExistsError):
                REFERENCE.publish(again, destination)
            with self.assertRaises(FileExistsError):
                REFERENCE.main(["--model-dir", str(root), "--output", str(destination)])

    def test_failed_generation_removes_its_staging_directory(self) -> None:
        """A failure before publication leaves no hidden staging directory behind."""

        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            destination = root / "bundle"

            def fail(arguments, staging):
                (staging / "logits-0.f32le").write_bytes(b"partial")
                raise MemoryError("simulated out-of-memory")

            with mock.patch.object(REFERENCE, "build_bundle", side_effect=fail):
                with self.assertRaises(MemoryError):
                    REFERENCE.main(["--model-dir", str(root), "--output", str(destination)])
            self.assertEqual(list(root.iterdir()), [])

    def test_weight_files_are_listed_or_reported_missing(self) -> None:
        """Weight discovery is sorted and refuses an empty checkpoint."""

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with self.assertRaises(FileNotFoundError):
                REFERENCE.weight_files(root)
            for name in ("b.safetensors", "a.safetensors", "c.bin"):
                (root / name).write_bytes(b"x")
            self.assertEqual([path.name for path in REFERENCE.weight_files(root)], ["a.safetensors", "b.safetensors"])
            self.assertEqual(
                REFERENCE.sha256_file(root / "a.safetensors"),
                "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881",
            )


    def test_bf16_resident_mode_accepts_only_bfloat16_floating_tensors(self) -> None:
        """Header inspection ignores integer tensors and refuses any wider floating weight."""

        def write(path: Path, dtypes: dict[str, str]) -> None:
            """Writes a SafeTensors header naming one two-byte tensor per entry."""

            header, offset = {}, 0
            for name, dtype in dtypes.items():
                header[name] = {"dtype": dtype, "shape": [1], "data_offsets": [offset, offset + 2]}
                offset += 2
            encoded = REFERENCE.json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            write(root / "a.safetensors", {"w": "BF16", "blocks": "U8"})
            write(root / "b.safetensors", {"bias": "BF16", "scales": "U8"})
            self.assertEqual(REFERENCE.checkpoint_floating_dtypes(root), {"BF16"})
            REFERENCE.require_bfloat16_checkpoint(root)
            write(root / "c.safetensors", {"correction": "F32"})
            self.assertEqual(REFERENCE.checkpoint_floating_dtypes(root), {"BF16", "F32"})
            with self.assertRaisesRegex(ValueError, "all BF16"):
                REFERENCE.require_bfloat16_checkpoint(root)

    def test_bf16_resident_flag_is_recorded(self) -> None:
        """The command line exposes the mode and defaults it off."""

        base = ["--model-dir", "m", "--output", "o"]
        self.assertFalse(REFERENCE.parse_arguments(base).bf16_resident)
        self.assertTrue(REFERENCE.parse_arguments(base + ["--bf16-resident"]).bf16_resident)

    def test_bf16_resident_dtype_is_checked_before_the_checkpoint(self) -> None:
        """A BF16 or unknown target is refused before any file is read."""

        missing = Path("no-such-checkpoint")
        for dtype in ("bfloat16", "float16"):
            with self.assertRaisesRegex(ValueError, "wider"):
                REFERENCE.require_bf16_resident_inputs(missing, dtype)
        with self.assertRaisesRegex(ValueError, "dtype must be one of"):
            REFERENCE.require_bf16_resident_inputs(missing, "float8")
        with self.assertRaises(FileNotFoundError):
            REFERENCE.require_bf16_resident_inputs(missing, "float64")


@unittest.skipUnless(HAS_TORCH, "needs PyTorch and Transformers")
class ModelLoadingTests(unittest.TestCase):
    """Check that loaded references compute what plain Transformers inference computes."""

    def test_parameter_cast_leaves_buffers_and_integers_alone(self) -> None:
        """Floating parameters change dtype exactly; buffers and integer parameters keep theirs."""

        import torch

        module = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
        original = module.weight.detach().clone()
        frequencies = torch.tensor([1.0, 1.0 / 3.0], dtype=torch.float32)
        module.register_buffer("inv_freq", frequencies.clone())
        module.count = torch.nn.Parameter(torch.tensor([3]), requires_grad=False)
        self.assertEqual(REFERENCE.cast_floating_parameters(module, torch.float32), 2)
        self.assertEqual((module.weight.dtype, module.bias.dtype), (torch.float32, torch.float32))
        self.assertTrue(torch.equal(module.weight, original.to(torch.float32)))
        self.assertEqual(module.inv_freq.dtype, torch.float32)
        self.assertTrue(torch.equal(module.inv_freq, frequencies))
        self.assertEqual(module.count.dtype, torch.int64)
        self.assertEqual(REFERENCE.cast_floating_parameters(module, torch.float32), 0)
        # Downcasting must not touch the F32 buffer either (`Module.to` would).
        self.assertEqual(REFERENCE.cast_floating_parameters(module, torch.bfloat16), 2)
        self.assertEqual(module.inv_freq.dtype, torch.float32)

    def test_bfloat16_references_keep_f32_rotary_tables(self) -> None:
        """A BF16 reference equals plain BF16 Transformers inference, rotary tables included.

        The tiny fixture uses YaRN; rounding its ``inv_freq`` to BF16 changes
        the logits by about 1.2e-2, so this exact comparison detects any
        buffer cast.
        """

        import torch
        import transformers

        model = REFERENCE.load_causal_lm(GPT_OSS_FIXTURE, "bfloat16")
        tables = {name: buffer.dtype for name, buffer in model.named_buffers() if name.endswith("inv_freq")}
        self.assertTrue(tables)
        self.assertEqual(set(tables.values()), {torch.float32}, tables)
        self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.bfloat16})
        plain = transformers.AutoModelForCausalLM.from_pretrained(
            GPT_OSS_FIXTURE, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
        ).eval()
        tokens = torch.tensor([[1, 5, 9, 14, 3, 27, 40, 8]])
        with torch.no_grad():
            self.assertTrue(torch.equal(model(input_ids=tokens).logits, plain(input_ids=tokens).logits))

    def test_float32_references_are_entirely_f32_and_truncate(self) -> None:
        """An F32 reference holds only F32 parameters and buffers; `layers` keeps the first blocks."""

        import torch

        model = REFERENCE.load_causal_lm(GPT_OSS_FIXTURE, "float32")
        self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.float32})
        floating = {buffer.dtype for buffer in model.buffers() if buffer.is_floating_point()}
        self.assertEqual(floating, {torch.float32})
        self.assertEqual(len(model.model.layers), 4)
        truncated = REFERENCE.load_causal_lm(GPT_OSS_FIXTURE, "float32", layers=2)
        self.assertEqual(len(truncated.model.layers), 2)
        self.assertEqual(truncated.config.layer_types, ["sliding_attention", "full_attention"])
        with self.assertRaises(ValueError):
            REFERENCE.load_causal_lm(GPT_OSS_FIXTURE, "float8")

    def test_float64_references_hold_f64_parameters_and_agree_with_f32(self) -> None:
        """An F64 reference casts every parameter to F64 and stays within F32 rounding of F32 logits."""

        import torch

        model = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float64")
        self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.float64})
        single = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32")
        tokens = [1, 5, 9, 14, 3, 27, 40, 8]
        double_logits = REFERENCE.prompt_logits(model, tokens)
        single_logits = REFERENCE.prompt_logits(single, tokens)
        # Both are written as F32; the tiny fixture's logits are O(1), so F32
        # accumulation error is far below this bound while a wrong dtype path
        # (for example BF16 weights) would exceed it.
        self.assertEqual(double_logits.dtype, torch.float32)
        self.assertLess(float((double_logits - single_logits).abs().max()), 1e-4)


    def test_bf16_resident_blocks_compute_exactly_what_a_plain_f32_load_computes(self) -> None:
        """Streaming a BF16-stored checkpoint equals a true F32 load of the same file, bit for bit.

        The truth is ``load_causal_lm(copy, "float32")``, built under the F32
        default dtype like any plain F32 reference, so a buffer that the BF16
        build rounds would show up here. After a forward pass every block
        holds its BF16 parameters again; truncation streams the kept blocks.
        """

        import torch

        with tempfile.TemporaryDirectory() as root:
            copy = bfloat16_copy(GPT_OSS_FIXTURE, Path(root) / "gpt-oss-bf16")
            streamed = REFERENCE.load_bf16_resident(copy, "float32")
            truth = REFERENCE.load_causal_lm(copy, "float32")
            tokens = torch.tensor([TOKENS])
            with torch.no_grad():
                streamed_logits = streamed(input_ids=tokens).logits
                truth_logits = truth(input_ids=tokens).logits
            self.assertEqual(streamed_logits.dtype, torch.float32)
            self.assertTrue(torch.equal(streamed_logits, truth_logits))
            blocks = REFERENCE.decoder_blocks(streamed)
            self.assertEqual(len(blocks), 4)
            self.assertEqual({parameter.dtype for parameter in blocks.parameters()}, {torch.bfloat16})
            inside = {id(parameter) for parameter in blocks.parameters()}
            self.assertEqual(
                {parameter.dtype for parameter in streamed.parameters() if id(parameter) not in inside},
                {torch.float32},
            )
            # The cached path runs the same hooks and agrees with the plain model too.
            self.assertEqual(
                REFERENCE.greedy_continuation(streamed, [1, 5, 9], 4),
                REFERENCE.greedy_continuation(truth, [1, 5, 9], 4),
            )
            truncated = REFERENCE.load_bf16_resident(copy, "float32", layers=2)
            truncated_truth = REFERENCE.load_causal_lm(copy, "float32", layers=2)
            self.assertEqual(len(REFERENCE.decoder_blocks(truncated)), 2)
            with torch.no_grad():
                self.assertTrue(
                    torch.equal(truncated(input_ids=tokens).logits, truncated_truth(input_ids=tokens).logits)
                )

    def test_instantiation_buffers_equal_those_of_plain_loads(self) -> None:
        """The parameter-free instantiation reproduces a plain load's buffers in dtype and value.

        Covers YaRN rotary tables (gpt-oss), Gemma3's per-attention-type
        tables and its default-dtype ``embed_scale``, and an F64 target.
        """

        import torch

        for fixture, dtype in ((GPT_OSS_FIXTURE, "float32"), (GEMMA3_FIXTURE, "float32"), (QWEN2_FIXTURE, "float64")):
            with self.subTest(fixture=fixture.name, dtype=dtype):
                expected, persistent = REFERENCE.instantiation_buffers(fixture, dtype)
                plain = REFERENCE.load_causal_lm(fixture, dtype)
                actual = dict(plain.named_buffers())
                self.assertTrue(expected)
                self.assertEqual(set(expected), set(actual))
                for name, buffer in actual.items():
                    self.assertEqual(expected[name].dtype, buffer.dtype, name)
                    self.assertTrue(torch.equal(expected[name], buffer), name)
                self.assertEqual(REFERENCE.buffer_mismatches(plain, expected, persistent), [])
                self.assertFalse(persistent & set(actual))
        truncated, _persistent = REFERENCE.instantiation_buffers(GPT_OSS_FIXTURE, "float32", layers=1)
        self.assertEqual(set(truncated), {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"})
        with self.assertRaises(ValueError):
            REFERENCE.instantiation_buffers(GPT_OSS_FIXTURE, "float8")

    def test_buffer_mismatches_report_every_difference(self) -> None:
        """Dtype, value, shape, one-sided, and persistent floating buffers are reported; integer buffers are not."""

        import torch

        def module(**buffers):
            """Returns a module holding the given buffers; names ending in `_kept` are persistent."""

            holder = torch.nn.Module()
            for name, value in buffers.items():
                holder.register_buffer(name, value, persistent=name.endswith("_kept"))
            return holder

        scale = torch.tensor(32.0**0.5)
        frequencies = torch.tensor([1.0, 0.25])
        expected = {
            "same": frequencies.clone(),
            "rounded": scale.clone(),
            "moved": frequencies.clone(),
            "resized": frequencies.clone(),
            "only_expected": frequencies.clone(),
            "table_kept": frequencies.clone(),
            "steps": torch.tensor([3]),
            "nan": torch.tensor([float("nan")]),
        }
        loaded = module(
            same=frequencies.clone(),
            rounded=scale.to(torch.bfloat16),
            moved=frequencies + 1e-7,
            resized=frequencies[:1].clone(),
            only_loaded=frequencies.clone(),
            table_kept=frequencies.clone(),
            steps=torch.tensor([4]),
            nan=torch.tensor([float("nan")]),
        )
        self.assertEqual(
            REFERENCE.buffer_mismatches(loaded, expected, frozenset({"table_kept"})),
            [
                "moved (values differ)",
                "nan (values differ)",
                "only_expected (not a floating buffer of the BF16 load)",
                "only_loaded (not a floating buffer of a plain load)",
                "resized (values differ)",
                "rounded (torch.bfloat16 instead of torch.float32)",
                "table_kept (persistent; its value comes from the checkpoint)",
            ],
        )
        self.assertEqual(REFERENCE.buffer_mismatches(module(same=frequencies.clone()), {"same": frequencies}, frozenset()), [])

    def test_bf16_resident_mode_refuses_gemma3_default_dtype_buffers(self) -> None:
        """Gemma3's ``embed_scale`` is built in BF16 by a BF16 load, so the mode is refused.

        Without the guard the streamed model would differ from a true F32
        load of the same BF16-stored file; the rotary tables, which
        Transformers keeps in F32, are not reported.
        """

        import torch

        with tempfile.TemporaryDirectory() as root:
            copy = bfloat16_copy(GEMMA3_FIXTURE, Path(root) / "gemma3-bf16")
            REFERENCE.require_bf16_resident_inputs(copy, "float32")
            with self.assertRaisesRegex(
                ValueError, r"cannot reproduce a float32 model: .*model\.embed_tokens\.embed_scale"
            ):
                REFERENCE.load_bf16_resident(copy, "float32")
            unguarded = REFERENCE.load_causal_lm(copy, "bfloat16")
            expected, persistent = REFERENCE.instantiation_buffers(copy, "float32")
            self.assertEqual(
                REFERENCE.buffer_mismatches(unguarded, expected, persistent),
                ["model.embed_tokens.embed_scale (torch.bfloat16 instead of torch.float32)"],
            )
            self.assertEqual(float(unguarded.model.embed_tokens.embed_scale), 5.65625)
            REFERENCE.stream_blocks_in_dtype(unguarded, torch.float32)
            truth = REFERENCE.load_causal_lm(copy, "float32")
            tokens = torch.tensor([TOKENS])
            with torch.no_grad():
                self.assertFalse(torch.equal(unguarded(input_ids=tokens).logits, truth(input_ids=tokens).logits))

    def test_bf16_resident_mode_reproduces_an_mxfp4_gpt_oss_checkpoint(self) -> None:
        """A checkpoint laid out like gpt-oss-20b passes the guard and streams bit-identically to F32.

        Transformers decodes the MXFP4 experts to BF16 on the CPU in both
        loads; the only floating buffers are the F32 YaRN rotary tables.
        """

        import torch

        with tempfile.TemporaryDirectory() as root:
            checkpoint = write_mxfp4_gpt_oss(Path(root) / "gpt-oss-mxfp4")
            self.assertEqual(REFERENCE.checkpoint_floating_dtypes(checkpoint), {"BF16"})
            streamed = REFERENCE.load_bf16_resident(checkpoint, "float32")
            self.assertEqual(
                {name: buffer.dtype for name, buffer in streamed.named_buffers()},
                {"model.rotary_emb.inv_freq": torch.float32, "model.rotary_emb.original_inv_freq": torch.float32},
            )
            truth = REFERENCE.load_causal_lm(checkpoint, "float32")
            tokens = torch.tensor([TOKENS + [100, 64, 77, 12]])
            with torch.no_grad():
                self.assertTrue(torch.equal(streamed(input_ids=tokens).logits, truth(input_ids=tokens).logits))
            self.assertEqual(
                REFERENCE.greedy_continuation(streamed, [1, 5, 9], 8),
                REFERENCE.greedy_continuation(truth, [1, 5, 9], 8),
            )

    def test_bf16_resident_loading_refuses_the_f32_fixture(self) -> None:
        """The fixture's F32 weights would be rounded by a BF16 load, so the mode is refused."""

        for load in (
            lambda dtype: REFERENCE.load_model(GPT_OSS_FIXTURE, dtype, bf16_resident=True),
            lambda dtype: REFERENCE.load_bf16_resident(GPT_OSS_FIXTURE, dtype),
        ):
            with self.assertRaisesRegex(ValueError, "all BF16"):
                load("float32")
            with self.assertRaisesRegex(ValueError, "wider"):
                load("bfloat16")


def write_truth_manifest(root: Path, prompt_ids: list, tokens: list, prompt_index: int = 0) -> Path:
    """Writes the fields of a truth manifest that `teacher_tokens` reads: prompt 0 and its greedy tokens."""

    import json

    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "prompts": [{"token_ids": prompt_ids}],
        "greedy": {"prompt_index": prompt_index, "steps": [{"token": token, "margin": 1.0} for token in tokens]},
        "provenance": {"dtype": "float32"},
    }
    (root / "reference.json").write_text(json.dumps(manifest))
    return root


def read_rows(path: Path, columns: int) -> list:
    """Reads an f32le file as rows of `columns` values."""

    values = REFERENCE.decode_f32le(path.read_bytes())
    return [values[start:start + columns] for start in range(0, len(values), columns)]


class GraphAndDecodeArgumentTests(unittest.TestCase):
    """Check the graph options and the decode self-consistency helpers without PyTorch."""

    def test_graph_defaults_to_the_eager_cpu_graph_and_accepts_mps_and_sdpa(self) -> None:
        """Without options the eager CPU graph runs; MPS, SDPA, and a truth bundle can be chosen."""

        base = ["--model-dir", "m", "--output", "o"]
        defaults = REFERENCE.parse_arguments(base)
        self.assertEqual(
            (defaults.device, defaults.attn_implementation, defaults.decode_tokens_from), ("cpu", "eager", None)
        )
        chosen = REFERENCE.parse_arguments(
            base + ["--device", "mps", "--attn-implementation", "sdpa", "--dtype", "bfloat16",
                    "--decode-tokens-from", "truth"]
        )
        self.assertEqual(
            (chosen.device, chosen.attn_implementation, chosen.decode_tokens_from), ("mps", "sdpa", Path("truth"))
        )
        self.assertEqual(REFERENCE.DEVICES, ("cpu", "mps"))
        self.assertEqual(REFERENCE.ATTENTION_IMPLEMENTATIONS, ("eager", "sdpa"))
        for extra in (
            ["--device", "cuda"],
            ["--attn-implementation", "flash_attention_2"],
            ["--device", "mps", "--dtype", "float64"],
            ["--bf16-resident", "--device", "mps"],
            ["--bf16-resident", "--attn-implementation", "sdpa"],
        ):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                REFERENCE.parse_arguments(base + extra)

    def test_graph_requirements_are_checked_before_anything_loads(self) -> None:
        """Unknown graphs, F64 on MPS, and BF16-resident streaming off the eager CPU graph are refused first."""

        REFERENCE.require_graph("float64", "eager", "cpu")
        REFERENCE.require_graph("bfloat16", "sdpa", "mps")
        missing = Path("no-such-checkpoint")
        for (dtype, attention, device), message in (
            (("float32", "flash", "cpu"), "attention must be one of"),
            (("float32", "eager", "cuda"), "device must be one of"),
            (("float64", "eager", "mps"), "no float64"),
        ):
            with self.subTest(attention=attention, device=device):
                with self.assertRaisesRegex(ValueError, message):
                    REFERENCE.require_graph(dtype, attention, device)
                with self.assertRaisesRegex(ValueError, message):
                    REFERENCE.load_model(missing, dtype, attention=attention, device=device)
                with self.assertRaisesRegex(ValueError, message):
                    REFERENCE.load_causal_lm(missing, dtype, attention=attention, device=device)
        for attention, device in (("sdpa", "cpu"), ("eager", "mps")):
            with self.assertRaisesRegex(ValueError, "eager CPU graph only"):
                REFERENCE.load_model(missing, "float32", bf16_resident=True, attention=attention, device=device)

    def test_step_gaps_are_each_steps_largest_absolute_difference(self) -> None:
        """Gaps are per-step maxima over the vocabulary; a non-finite step has none, misshapen logits are refused."""

        self.assertEqual(
            REFERENCE.step_gaps([[1.0, 2.0, 3.0], [0.0, -1.0, 5.0]], [[1.0, 2.5, 2.0], [0.0, -1.0, 5.0]]),
            [1.0, 0.0],
        )
        # The difference is taken in F64: 1 + 2^-23 and 1 are F32 neighbours.
        self.assertEqual(REFERENCE.step_gaps([[1.0 + 2.0**-23]], [[1.0]]), [2.0**-23])
        # A non-finite value on either side leaves its step without a gap; the other steps keep theirs.
        self.assertEqual(
            REFERENCE.step_gaps([[float("nan"), 0.0], [1.0, 2.0], [0.0, 0.0]],
                                [[0.0, 0.0], [1.5, 2.0], [0.0, float("-inf")]]),
            [None, 0.5, None],
        )
        for cached, stateless, message in (
            ([[0.0], [0.0]], [[0.0]], "different numbers of steps"),
            ([[0.0, 1.0]], [[0.0]], "differ in length"),
            ([[]], [[]], "differ in length or are empty"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                REFERENCE.step_gaps(cached, stateless)

    def test_teacher_tokens_come_only_from_a_matching_truth_bundle(self) -> None:
        """The truth's greedy tokens are returned with its identity; any mismatch is refused."""

        with tempfile.TemporaryDirectory() as root:
            truth = write_truth_manifest(Path(root) / "truth", [1, 5, 9], [4, 7, 2])
            tokens, source = REFERENCE.teacher_tokens(truth, [1, 5, 9], 10)
            self.assertEqual(tokens, [4, 7, 2])
            self.assertEqual(
                source, {"manifest_sha256": REFERENCE.sha256_file(truth / "reference.json"), "dtype": "float32"}
            )
            for name, steps, index, ours, message in (
                ("other prompt", [4], 0, [1, 5], "other token IDs"),
                ("other prompt index", [4], 1, [1, 5, 9], "does not extend prompt 0"),
                ("empty", [], 0, [1, 5, 9], "is empty"),
                ("beyond the vocabulary", [4, 10], 0, [1, 5, 9], "not an integer below 10"),
                ("negative", [-1], 0, [1, 5, 9], "not an integer below 10"),
                ("boolean", [True], 0, [1, 5, 9], "not an integer below 10"),
                ("float", [4.0], 0, [1, 5, 9], "not an integer below 10"),
            ):
                with self.subTest(name=name):
                    bad = write_truth_manifest(Path(root) / name, [1, 5, 9], steps, index)
                    with self.assertRaisesRegex(ValueError, message):
                        REFERENCE.teacher_tokens(bad, ours, 10)


@unittest.skipUnless(HAS_TORCH, "needs PyTorch and Transformers")
class GraphAndDecodeModelTests(unittest.TestCase):
    """Check other graphs and the decode self-consistency record on the tiny fixtures."""

    def test_sdpa_graph_is_selected_and_agrees_with_eager_within_f32_rounding(self) -> None:
        """`attention` reaches Transformers; on the CPU in F32 both graphs agree to rounding."""

        eager = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32")
        sdpa = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32", attention="sdpa")
        self.assertEqual((eager.config._attn_implementation, sdpa.config._attn_implementation), ("eager", "sdpa"))
        gap = (REFERENCE.prompt_logits(eager, TOKENS) - REFERENCE.prompt_logits(sdpa, TOKENS)).abs().max()
        self.assertLess(float(gap), 1e-5)

    def test_mps_graph_runs_on_the_device_and_returns_cpu_logits(self) -> None:
        """An MPS model keeps its dtypes (F32 rotary tables included) and agrees with the CPU in F32."""

        import torch

        if not torch.backends.mps.is_available():
            self.skipTest("needs an MPS device")
        cpu = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32")
        mps = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32", device="mps")
        self.assertEqual({parameter.device.type for parameter in mps.parameters()}, {"mps"})
        logits = REFERENCE.prompt_logits(mps, TOKENS)
        self.assertEqual((logits.device.type, logits.dtype), ("cpu", torch.float32))
        self.assertLess(float((logits - REFERENCE.prompt_logits(cpu, TOKENS)).abs().max()), 1e-4)
        self.assertEqual(len(REFERENCE.greedy_continuation(mps, [1, 5, 9], 3)), 3)
        half = REFERENCE.load_causal_lm(GPT_OSS_FIXTURE, "bfloat16", device="mps")
        self.assertEqual({parameter.dtype for parameter in half.parameters()}, {torch.bfloat16})
        tables = {buffer.dtype for name, buffer in half.named_buffers() if name.endswith("inv_freq")}
        self.assertEqual(tables, {torch.float32})

    def test_decode_record_compares_each_cached_step_with_one_stateless_pass(self) -> None:
        """Cached rows are the model's own cached decode, stateless rows one uncached pass, gaps their F32 drift."""

        import torch

        model = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32")
        vocabulary = model.config.vocab_size
        prompt, tokens = TOKENS[:4], TOKENS[4:]
        source = {"manifest_sha256": "0" * 64, "dtype": "float32"}
        with tempfile.TemporaryDirectory() as root:
            staging = Path(root)
            record = REFERENCE.decode_record(model, prompt, tokens, source, staging)
            cached = read_rows(staging / REFERENCE.DECODE_CACHED_FILE, vocabulary)
            stateless = read_rows(staging / REFERENCE.DECODE_STATELESS_FILE, vocabulary)
            self.assertEqual(record["cached_logits_sha256"], REFERENCE.sha256_file(staging / record["cached_logits_file"]))
            self.assertEqual(
                record["stateless_logits_sha256"], REFERENCE.sha256_file(staging / record["stateless_logits_file"])
            )
        self.assertEqual(record["logits_shape"], [len(tokens), vocabulary])
        self.assertEqual((record["prompt_index"], record["first_position"], record["tokens_source"]), (0, 4, source))
        self.assertEqual([step["position"] for step in record["steps"]], [4, 5, 6, 7])
        self.assertEqual([step["fed_token"] for step in record["steps"]], tokens)
        full = REFERENCE.prompt_logits(model, prompt + tokens)
        self.assertEqual(stateless, full[len(prompt):].tolist())
        manual = []
        with torch.no_grad():
            output = model(input_ids=torch.tensor([prompt]), use_cache=True)
            for token in tokens:
                output = model(input_ids=torch.tensor([[token]]), past_key_values=output.past_key_values, use_cache=True)
                manual.append(output.logits[0, -1].tolist())
        self.assertEqual(cached, manual)
        gaps = [step["max_abs_gap"] for step in record["steps"]]
        self.assertEqual(gaps, REFERENCE.step_gaps(cached, stateless))
        self.assertEqual(record["max_abs_gap"], max(gaps))
        self.assertLess(record["max_abs_gap"], 1e-5)

    def test_decode_record_exposes_a_decode_position_off_by_one(self) -> None:
        """A cached decode at the wrong rotary position, a typical cache defect, opens a gap far above rounding."""

        class ShiftedDecode:
            """Feeds every cached decode step one position too far, as an off-by-one cache offset would."""

            def __init__(self, model):
                """Wraps a loaded model."""

                self.model = model
                self.device = model.device

            def __call__(self, **inputs):
                """Runs the model, shifting the position of single-token cached steps by one."""

                import torch

                past = inputs.get("past_key_values")
                if past is not None:
                    position = past.get_seq_length() + 1
                    inputs["position_ids"] = torch.tensor([[position]], device=self.device)
                return self.model(**inputs)

        model = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32")
        with tempfile.TemporaryDirectory() as honest_root, tempfile.TemporaryDirectory() as shifted_root:
            honest = REFERENCE.decode_record(model, TOKENS[:4], TOKENS[4:], {}, Path(honest_root))
            shifted = REFERENCE.decode_record(ShiftedDecode(model), TOKENS[:4], TOKENS[4:], {}, Path(shifted_root))
        self.assertLess(honest["max_abs_gap"], 1e-5)
        self.assertGreater(shifted["max_abs_gap"], 1e-2)

    def test_bundle_records_the_graph_and_the_decode_record(self) -> None:
        """`build_bundle` passes the graph to the loader, records it, and adds decode files only on request."""

        import json
        from unittest import mock

        class Tokenizer:
            """Tokenizes the two test prompts to fixed IDs, as the fixtures carry no tokenizer."""

            def __call__(self, text, add_special_tokens=True):
                """Returns the fixed IDs of `text`."""

                return {"input_ids": {"a": TOKENS[:4], "b": [2, 4]}[text]}

        model = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32", attention="sdpa")
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            truth = write_truth_manifest(root / "truth", TOKENS[:4], TOKENS[4:])
            base = ["--model-dir", str(QWEN2_FIXTURE), "--output", str(root / "out"), "--prompt", "a",
                    "--prompt", "b", "--greedy-tokens", "2", "--attn-implementation", "sdpa"]
            for extra, files in (
                ([], ["logits-0.f32le", "logits-1.f32le"]),
                (["--decode-tokens-from", str(truth)],
                 ["decode-cached.f32le", "decode-stateless.f32le", "logits-0.f32le", "logits-1.f32le"]),
            ):
                with self.subTest(extra=extra):
                    staging = Path(tempfile.mkdtemp(dir=root))
                    with mock.patch.object(REFERENCE, "load_model", return_value=(Tokenizer(), model)) as load:
                        manifest = REFERENCE.build_bundle(REFERENCE.parse_arguments(base + extra), staging)
                    load.assert_called_once_with(QWEN2_FIXTURE.resolve(), "float32", None, False, "sdpa", "cpu")
                    provenance = manifest["provenance"]
                    self.assertEqual((provenance["device"], provenance["attention_implementation"]), ("cpu", "sdpa"))
                    self.assertEqual(sorted(path.name for path in staging.iterdir()), files)
                    self.assertEqual("decode_consistency" in manifest, bool(extra))
                    json.dumps(manifest, allow_nan=False)
            record = manifest["decode_consistency"]
            self.assertEqual([step["fed_token"] for step in record["steps"]], TOKENS[4:])
            self.assertEqual(record["tokens_source"]["manifest_sha256"], REFERENCE.sha256_file(truth / "reference.json"))



class FixedTokenizer:
    """Tokenizes the test prompts to fixed IDs, as the tiny fixtures carry no tokenizer."""

    #: Token IDs of each test prompt; ``"b"`` is the longest.
    IDS = {"a": TOKENS[:4], "b": TOKENS, "c": [2, 4]}

    def __call__(self, text, add_special_tokens=True):
        """Returns the fixed IDs of `text`."""

        return {"input_ids": list(self.IDS[text])}


def scaled_head_copy(source: Path, destination: Path, factor: float) -> Path:
    """Writes `source` to `destination` with its untied output head multiplied by `factor`.

    A large enough factor drives the logits beyond F16's finite range (65,504)
    while F32 evaluation stays finite, which is the official F16 overflow that
    a bundle must record as data.
    """

    from safetensors.torch import load_file, save_file

    destination.mkdir(parents=True)
    (destination / "config.json").write_bytes((source / "config.json").read_bytes())
    for path in REFERENCE.weight_files(source):
        tensors = load_file(path)
        tensors["lm_head.weight"] = tensors["lm_head.weight"] * factor
        save_file(tensors, destination / path.name, metadata={"format": "pt"})
    return destination


def build_with(model, arguments: list, staging: Path) -> dict:
    """Runs `build_bundle` on `arguments` with `model` and the fixed tokenizer in place of a load."""

    from unittest import mock

    with mock.patch.object(REFERENCE, "load_model", return_value=(FixedTokenizer(), model)):
        return REFERENCE.build_bundle(REFERENCE.parse_arguments(arguments), staging)


class HalfPrecisionAndPromptDecodeArgumentTests(unittest.TestCase):
    """Check the float16 dtype, the prompt-decode option, and their refusals without PyTorch."""

    def test_half_precision_dtypes_are_refused_with_bf16_resident(self) -> None:
        """`--bf16-resident` widens BF16 weights, so it is refused with either half-precision dtype."""

        base = ["--model-dir", "m", "--output", "o", "--bf16-resident"]
        self.assertTrue(REFERENCE.parse_arguments(base).bf16_resident)
        self.assertEqual(REFERENCE.parse_arguments(base + ["--dtype", "float64"]).dtype, "float64")
        for dtype in REFERENCE.HALF_DTYPES:
            with self.subTest(dtype=dtype), self.assertRaises(SystemExit):
                REFERENCE.parse_arguments(base + ["--dtype", dtype])
        # MPS runs F16 and BF16 but not F64.
        mps = ["--model-dir", "m", "--output", "o", "--device", "mps"]
        self.assertEqual(REFERENCE.parse_arguments(mps + ["--dtype", "float16"]).dtype, "float16")
        REFERENCE.require_graph("float16", "sdpa", "mps")
        with self.assertRaises(SystemExit):
            REFERENCE.parse_arguments(mps + ["--dtype", "float64"])

    def test_prompt_decode_is_off_by_default(self) -> None:
        """Only `--prompt-decode` adds the quarter-prefill record."""

        base = ["--model-dir", "m", "--output", "o"]
        self.assertFalse(REFERENCE.parse_arguments(base).prompt_decode)
        self.assertTrue(REFERENCE.parse_arguments(base + ["--prompt-decode"]).prompt_decode)
        self.assertEqual(REFERENCE.PROMPT_DECODE_FILE, "decode-prompt.f32le")

    def test_longest_prompt_and_quarter_prefill_follow_the_gate_protocol(self) -> None:
        """The longest prompt is decoded (the first of equal ones) after `max(T // 4, 1)` prefill tokens."""

        prompts = [{"token_ids": [1] * 5}, {"token_ids": [1] * 13}, {"token_ids": [1] * 13}, {"token_ids": [1] * 8}]
        self.assertEqual(REFERENCE.longest_prompt(prompts), 1)
        self.assertEqual(REFERENCE.longest_prompt(prompts[:1]), 0)
        with self.assertRaisesRegex(ValueError, "no prompt"):
            REFERENCE.longest_prompt([])
        self.assertEqual(
            [REFERENCE.quarter_prefill(tokens) for tokens in (1, 3, 4, 7, 8, 13, 1237)],
            [1, 1, 1, 1, 2, 3, 309],
        )
        with self.assertRaisesRegex(ValueError, "at least one token"):
            REFERENCE.quarter_prefill(0)


@unittest.skipUnless(HAS_TORCH, "needs PyTorch and Transformers")
class HalfPrecisionAndPromptDecodeModelTests(unittest.TestCase):
    """Check float16 evaluations, prompt-decode records, non-finite records, and unchanged default outputs."""

    def test_float16_references_hold_f16_parameters_and_stay_within_f16_rounding(self) -> None:
        """F16 runs on the CPU (and MPS when present) keep F32 rotary tables and stay near F32.

        The tiny fixture's logits are below 1.3 in magnitude, and F16's unit
        roundoff ``2^-11`` accumulated through four blocks moved them by
        ``3.1e-3`` on both devices when this test was written. The bound
        ``8e-3`` lies between that and the BF16 evaluation's ``1.7e-2``, so a
        graph that silently ran in BF16 fails, and a zero deviation (a graph
        that silently ran in F32) fails too.
        """

        import torch

        single = REFERENCE.prompt_logits(REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32"), TOKENS)
        bfloat = REFERENCE.prompt_logits(REFERENCE.load_causal_lm(QWEN2_FIXTURE, "bfloat16"), TOKENS)
        self.assertGreater(float((bfloat - single).abs().max()), 8e-3)
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float16", device=device)
                self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.float16})
                tables = {buffer.dtype for name, buffer in model.named_buffers() if name.endswith("inv_freq")}
                self.assertEqual(tables, {torch.float32})
                logits = REFERENCE.prompt_logits(model, TOKENS)
                self.assertEqual((logits.device.type, logits.dtype), ("cpu", torch.float32))
                deviation = float((logits - single).abs().max())
                self.assertGreater(deviation, 0.0)
                self.assertLess(deviation, 8e-3)

    def test_prompt_decode_equals_a_hand_written_quarter_prefill_decode(self) -> None:
        """The record's rows are the prefill's own logits followed by one cached decode per later token."""

        import torch

        model = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32")
        prefill = REFERENCE.quarter_prefill(len(TOKENS))
        self.assertEqual(prefill, 2)
        rows = []
        with torch.no_grad():
            output = model(input_ids=torch.tensor([TOKENS[:prefill]]), use_cache=True)
            rows.extend(output.logits[0].tolist())
            for token in TOKENS[prefill:]:
                output = model(input_ids=torch.tensor([[token]]), past_key_values=output.past_key_values, use_cache=True)
                rows.append(output.logits[0, -1].tolist())
        decoded = REFERENCE.prompt_decode_logits(model, TOKENS, prefill)
        self.assertEqual(decoded.tolist(), rows)
        # Cached decoding agrees with the stateless pass to F32 rounding, not bit for bit.
        self.assertLess(float((decoded - REFERENCE.prompt_logits(model, TOKENS)).abs().max()), 1e-5)
        for bad in (0, len(TOKENS) + 1):
            with self.assertRaisesRegex(ValueError, "cannot prefill"):
                REFERENCE.prompt_decode_logits(model, TOKENS, bad)
        vocabulary = model.config.vocab_size
        with tempfile.TemporaryDirectory() as root:
            staging = Path(root)
            manifest = build_with(
                model,
                ["--model-dir", str(QWEN2_FIXTURE), "--output", str(staging / "out"), "--prompt", "a",
                 "--prompt", "b", "--prompt", "c", "--greedy-tokens", "2", "--prompt-decode"],
                staging,
            )
            record = manifest["prompt_decode"]
            self.assertEqual(
                {key: record[key] for key in ("prompt_index", "prefill_tokens", "logits_file", "logits_shape", "finite")},
                {"prompt_index": 1, "prefill_tokens": 2, "logits_file": "decode-prompt.f32le",
                 "logits_shape": [len(TOKENS), vocabulary], "finite": True},
            )
            self.assertNotIn("first_non_finite", record)
            self.assertEqual(record["logits_sha256"], REFERENCE.sha256_file(staging / record["logits_file"]))
            self.assertEqual(read_rows(staging / record["logits_file"], vocabulary), rows)
            REFERENCE.json.dumps(manifest, allow_nan=False)

    def test_an_overflowing_evaluation_is_recorded_as_non_finite_data(self) -> None:
        """F16 logits beyond 65,504 become infinite: prompts, greedy, and decode records say so in strict JSON."""

        import torch

        plain = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32")
        largest = float(REFERENCE.prompt_logits(plain, TOKENS).abs().max())
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            # Scale the head so that F32 logits reach about 1e6, far beyond F16's range.
            overflowing = scaled_head_copy(QWEN2_FIXTURE, root / "overflowing", 1e6 / largest)
            wide = REFERENCE.prompt_logits(REFERENCE.load_causal_lm(overflowing, "float32"), TOKENS)
            self.assertTrue(bool(torch.isfinite(wide).all()))
            self.assertGreater(float(wide.abs().max()), 65504.0)
            model = REFERENCE.load_causal_lm(overflowing, "float16")
            truth = write_truth_manifest(root / "truth", TOKENS[:4], TOKENS[4:])
            staging = root / "staging"
            staging.mkdir()
            manifest = build_with(
                model,
                ["--model-dir", str(overflowing), "--output", str(root / "out"), "--prompt", "a", "--prompt", "b",
                 "--greedy-tokens", "3", "--dtype", "float16", "--decode-tokens-from", str(truth), "--prompt-decode"],
                staging,
            )
            text = REFERENCE.json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False)
            self.assertEqual(REFERENCE.json.loads(text), manifest)
            logits = REFERENCE.prompt_logits(model, TOKENS[:4])
            expected = REFERENCE.first_non_finite(logits)
            self.assertIsNotNone(expected)
            prompt = manifest["prompts"][0]
            self.assertEqual((prompt["finite"], prompt["first_non_finite"]), (False, expected))
            flat = expected["position"] * logits.shape[1] + expected["token"]
            self.assertFalse(math.isfinite(float(logits.reshape(-1)[flat])))
            self.assertTrue(bool(torch.isfinite(logits.reshape(-1)[:flat]).all()))
            if not REFERENCE.all_finite(logits[-1]):
                self.assertEqual((prompt["last_top1"], prompt["last_margin"]), (None, None))
            self.assertEqual((manifest["greedy"]["steps"], manifest["greedy"]["stopped"]), ([], "non_finite"))
            decode = manifest["decode_consistency"]
            self.assertFalse(decode["finite"])
            self.assertIsNone(decode["max_abs_gap"])
            self.assertIn(decode["first_non_finite"]["logits"], ("cached", "stateless"))
            self.assertEqual(decode["first_non_finite"]["position"], 4 + decode["first_non_finite"]["step"])
            self.assertFalse(manifest["prompt_decode"]["finite"])
            self.assertIn("first_non_finite", manifest["prompt_decode"])
            # The non-finite logits themselves are written, byte for byte.
            self.assertEqual((staging / "logits-0.f32le").read_bytes(), REFERENCE.tensor_f32le(logits))

    def test_greedy_stops_before_the_first_non_finite_step(self) -> None:
        """A continuation that turns non-finite mid-way keeps its finite steps and records why it stopped."""

        import torch

        class TurnsNonFinite:
            """Returns the wrapped model's logits, made NaN from the third cached call on."""

            def __init__(self, model):
                """Wraps a loaded model."""

                self.model, self.device, self.calls = model, model.device, 0
                self.config = model.config

            def __call__(self, **inputs):
                """Runs the model and poisons its logits once two calls have returned."""

                output = self.model(**inputs)
                self.calls += 1
                if self.calls > 2 and inputs.get("use_cache"):
                    output.logits = torch.full_like(output.logits, float("nan"))
                return output

        model = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32")
        honest = REFERENCE.greedy_continuation(model, TOKENS[:4], 4)
        poisoned = REFERENCE.greedy_continuation(TurnsNonFinite(model), TOKENS[:4], 4)
        self.assertEqual(len(honest), 4)
        self.assertEqual(poisoned, honest[:2])
        with tempfile.TemporaryDirectory() as root:
            staging = Path(root)
            manifest = build_with(
                TurnsNonFinite(model),
                ["--model-dir", str(QWEN2_FIXTURE), "--output", str(staging / "out"), "--prompt", "a",
                 "--greedy-tokens", "4"],
                staging,
            )
        # The prompt pass is the first call, so the cached prefill and one decode stay finite.
        self.assertEqual(manifest["greedy"]["stopped"], "non_finite")
        self.assertEqual(manifest["greedy"]["steps"], honest[:1])
        self.assertNotIn("finite", manifest["prompts"][0])

    def test_default_outputs_are_those_of_the_previous_protocol(self) -> None:
        """Finite float32 and bfloat16 bundles hold exactly what the generator wrote before these options.

        The previous protocol is restated here by hand: one uncached pass per
        prompt, a cached greedy continuation of prompt 0 margin by margin, and
        cached against stateless decode rows. Every logits file must equal it
        byte for byte, and the manifest records must carry no new key except
        the decode record's ``finite`` flag.
        """

        import torch

        def previous(model, prompts, steps, tokens):
            """Returns the logits bytes, greedy steps, and decode bytes of the previous protocol."""

            files = {}
            with torch.no_grad():
                for index, ids in enumerate(prompts):
                    logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].to(torch.float32)
                    files[f"logits-{index}.f32le"] = logits.contiguous().numpy().astype("<f4").tobytes()
                greedy = []
                output = model(input_ids=torch.tensor([prompts[0]]), use_cache=True)
                for _ in range(steps):
                    token, margin = REFERENCE.top_two_margin(output.logits[0, -1].to(torch.float32).tolist())
                    greedy.append({"token": token, "margin": margin})
                    output = model(input_ids=torch.tensor([[token]]), past_key_values=output.past_key_values, use_cache=True)
                cached = []
                output = model(input_ids=torch.tensor([prompts[0]]), use_cache=True)
                for token in tokens:
                    output = model(input_ids=torch.tensor([[token]]), past_key_values=output.past_key_values, use_cache=True)
                    cached.append(output.logits[0, -1].to(torch.float32))
                stateless = model(input_ids=torch.tensor([prompts[0] + tokens]), use_cache=False).logits
                stateless = stateless[0, len(prompts[0]):].to(torch.float32).contiguous()
            files["decode-cached.f32le"] = torch.stack(cached).numpy().astype("<f4").tobytes()
            files["decode-stateless.f32le"] = stateless.numpy().astype("<f4").tobytes()
            return files, greedy

        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                model = REFERENCE.load_causal_lm(QWEN2_FIXTURE, dtype)
                truth = write_truth_manifest(root / "truth", TOKENS[:4], TOKENS[4:])
                staging = root / "staging"
                staging.mkdir()
                manifest = build_with(
                    model,
                    ["--model-dir", str(QWEN2_FIXTURE), "--output", str(root / "out"), "--prompt", "a",
                     "--prompt", "c", "--greedy-tokens", "3", "--dtype", dtype, "--decode-tokens-from", str(truth)],
                    staging,
                )
                files, greedy = previous(model, [TOKENS[:4], [2, 4]], 3, TOKENS[4:])
                self.assertEqual(sorted(path.name for path in staging.iterdir()), sorted(files))
                for name, payload in files.items():
                    self.assertEqual((staging / name).read_bytes(), payload, name)
                self.assertEqual(manifest["greedy"], {"prompt_index": 0, "steps": greedy})
                for prompt in manifest["prompts"]:
                    self.assertEqual(
                        set(prompt),
                        {"text", "token_ids", "logits_file", "logits_shape", "logits_sha256", "last_top1", "last_margin"},
                    )
                decode = manifest["decode_consistency"]
                self.assertIs(decode["finite"], True)
                self.assertNotIn("first_non_finite", decode)
                self.assertNotIn("prompt_decode", manifest)


if __name__ == "__main__":
    unittest.main()
