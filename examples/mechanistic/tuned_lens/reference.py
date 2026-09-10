#!/usr/bin/env python3
"""Independent CPU reference for the bounded Tuned Lens experiment.

After prepare.py, run this file from the repository root. It uses PyTorch
equations matching Hugging Face GPT-NeoX 4.28.1, including its cached F32 rotary
tables, rather than importing the Rust implementation or a newer HF model.
Model/lens computation and exported logits stay F32; loss analysis uses F64.
No model or translator is fitted. Outputs remain under ignored target/.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time

import safetensors
from safetensors.torch import load_file, save
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F

from prepare import fingerprint, strict_json, verified_output_cache


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[2] / "target/mechanistic/tuned_lens"
UPSTREAM = (
    "https://github.com/huggingface/transformers/blob/v4.28.1/"
    "src/transformers/models/gpt_neox/modeling_gpt_neox.py"
)


class PythiaReference:
    """The published graph, with packed head-major QKV kept in author layout."""

    def __init__(self, weights, config):
        self.weights = weights
        self.config = config
        expected = {
            "model_type": "gpt_neox", "hidden_size": 768,
            "num_hidden_layers": 12, "num_attention_heads": 12,
            "intermediate_size": 3072, "vocab_size": 50304,
            "rotary_pct": 0.25, "rotary_emb_base": 10000,
            "layer_norm_eps": 1e-5, "hidden_act": "gelu",
            "use_parallel_residual": True, "tie_word_embeddings": False,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"Unexpected paired-checkpoint configuration: {key}")
        self.heads = config["num_attention_heads"]
        self.head_width = config["hidden_size"] // self.heads
        self.rotary_width = int(self.head_width * config["rotary_pct"])

        # HF 4.28.1 constructs these F32 tables at initialization. Recompute
        # derived buffers; the SafeTensors input contains learned weights only.
        powers = torch.arange(0, self.rotary_width, 2).float() / self.rotary_width
        inverse_frequency = 1.0 / (config["rotary_emb_base"] ** powers)
        positions = torch.arange(config["max_position_embeddings"]).float()
        angles = torch.outer(positions, inverse_frequency)
        duplicated = torch.cat([angles, angles], dim=-1)
        self.cosine = duplicated.cos()[None, None]
        self.sine = duplicated.sin()[None, None]
        self.scale = (torch.tensor(1.0) / torch.tensor(self.head_width).float().sqrt()).item()

    def linear(self, hidden, name):
        return F.linear(hidden, self.weights[name + ".weight"], self.weights.get(name + ".bias"))

    def normalize(self, hidden, name):
        return F.layer_norm(
            hidden, (self.config["hidden_size"],),
            self.weights[name + ".weight"], self.weights[name + ".bias"],
            self.config["layer_norm_eps"],
        )

    def rotate(self, projected):
        length = projected.shape[2]
        prefix, suffix = projected.split(
            [self.rotary_width, self.head_width - self.rotary_width], dim=-1
        )
        real, imaginary = prefix.chunk(2, dim=-1)
        quarter_turn = torch.cat([-imaginary, real], dim=-1)
        rotated = prefix * self.cosine[:, :, :length] + quarter_turn * self.sine[:, :, :length]
        return torch.cat([rotated, suffix], dim=-1)

    def attention(self, hidden, name):
        batch, length, _ = hidden.shape
        packed = self.linear(hidden, name + ".query_key_value")
        packed = packed.reshape(batch, length, self.heads, 3 * self.head_width)
        query, key, value = [
            part.permute(0, 2, 1, 3) for part in packed.chunk(3, dim=-1)
        ]
        query, key = self.rotate(query), self.rotate(key)
        query = query.reshape(batch * self.heads, length, self.head_width)
        key = key.reshape(batch * self.heads, length, self.head_width)
        # Retain HF's baddbmm scaling location instead of pre-scaling Q.
        scores = torch.baddbmm(
            torch.zeros(batch * self.heads, length, length), query,
            key.transpose(1, 2), beta=1.0, alpha=self.scale,
        ).reshape(batch, self.heads, length, length)
        causal = torch.ones(length, length, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, torch.finfo(torch.float32).min)
        context = scores.softmax(dim=-1) @ value
        context = context.permute(0, 2, 1, 3).reshape(batch, length, -1)
        return self.linear(context, name + ".dense")

    def readout(self, hidden):
        normalized = self.normalize(hidden, "gpt_neox.final_layer_norm")
        return self.linear(normalized, "embed_out")

    def forward(self, input_ids):
        hidden = F.embedding(input_ids, self.weights["gpt_neox.embed_in.weight"])
        lens_inputs = []
        for index in range(self.config["num_hidden_layers"]):
            # Lens 0 sees embeddings; lens 1 sees block 0's output; and so on.
            lens_inputs.append(hidden)
            name = f"gpt_neox.layers.{index}"
            attention = self.attention(self.normalize(hidden, name + ".input_layernorm"), name + ".attention")
            mlp = self.normalize(hidden, name + ".post_attention_layernorm")
            mlp = F.gelu(self.linear(mlp, name + ".mlp.dense_h_to_4h"), approximate="none")
            mlp = self.linear(mlp, name + ".mlp.dense_4h_to_h")
            hidden = mlp + attention + hidden
        return self.readout(hidden), lens_inputs


def token_kl(final_log_probs, candidate_logits):
    """KL(final || candidate), one F64 value per input position, in nats.

    Supply final log probabilities computed in F64 from the unchanged F32
    logits. Wide-vocabulary F32 normalizers can otherwise measurably bias KL.
    """
    if final_log_probs.shape != candidate_logits.shape or final_log_probs.ndim != 3:
        raise ValueError("KL requires matching [batch, token, vocabulary] tensors")
    candidate_log_probs = F.log_softmax(candidate_logits.double(), dim=-1)
    final_log_probs = final_log_probs.double()
    return (final_log_probs.exp() * (final_log_probs - candidate_log_probs)).sum(dim=-1)


def token_cross_entropy(logits, input_ids):
    """Position t predicts input token t+1; the final position has no CE label."""
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape or input_ids.shape[1] < 2:
        raise ValueError("CE requires matching inputs with at least two positions")
    predictions = logits[:, :-1].reshape(-1, logits.shape[-1]).double()
    labels = input_ids[:, 1:].reshape(-1)
    return F.cross_entropy(predictions, labels, reduction="none")


@dataclass
class Totals:
    """Pool token losses, so short records receive proportionally less weight."""

    kl_sum: float = 0.0
    kl_tokens: int = 0
    ce_sum: float = 0.0
    ce_tokens: int = 0

    def add(self, kl, ce):
        # F64 analysis and population aggregation avoid an accidental
        # dependence on vocabulary width, record length, or reduction grouping.
        self.kl_sum += kl.double().sum().item()
        self.kl_tokens += kl.numel()
        self.ce_sum += ce.double().sum().item()
        self.ce_tokens += ce.numel()

    def means(self):
        if not self.kl_tokens or not self.ce_tokens:
            raise ValueError("Cannot summarize an empty study")
        return {
            "kl_nats_per_token": self.kl_sum / self.kl_tokens,
            "ce_nats_per_token": self.ce_sum / self.ce_tokens,
            "kl_tokens": self.kl_tokens, "ce_tokens": self.ce_tokens,
        }


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_fixture(fixture, protocol, vocabulary):
    records = fixture["records"]
    if [record["record_index"] for record in records] != list(range(protocol["records"])):
        raise ValueError("The study requires the first four records in source order")
    input_total, ce_total = 0, 0
    for record in records:
        ids = record["input_ids"]
        if not 2 <= len(ids) <= protocol["max_input_tokens"] or ids[0] != protocol["eos_token_id"]:
            raise ValueError("Invalid bounded input length or missing prepended EOS")
        if any(type(token) is not int or token < 0 or token >= vocabulary for token in ids):
            raise ValueError("Token ID outside the model vocabulary")
        if record["input_tokens"] != len(ids) or record["ce_tokens"] != len(ids) - 1:
            raise ValueError("Recorded token counts do not match the fixture")
        input_total += len(ids)
        ce_total += len(ids) - 1
    if fixture["totals"] != {"input_tokens": input_total, "ce_tokens": ce_total}:
        raise ValueError("Fixture totals disagree with actual record lengths")
    if len(records[0]["input_ids"]) < protocol["reference_positions"]:
        raise ValueError("The first record is shorter than the reference-logit window")


def atomic_write(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def top_predictions(logits, tokenizer):
    return [
        {"token_id": token, "text": tokenizer.decode([token], skip_special_tokens=False)}
        for token in logits[0].argmax(dim=-1).tolist()
    ]


@torch.inference_mode()
def run_study(model, translators, fixture, protocol, tokenizer):
    layers = model.config["num_hidden_layers"]
    totals = {kind: [Totals() for _ in range(layers)] for kind in ("raw", "tuned")}
    final_totals = Totals()
    exported, predictions = {}, {"raw": [], "tuned": []}
    positions = protocol["reference_positions"]
    for record in fixture["records"]:
        ids = torch.tensor([record["input_ids"]], dtype=torch.long)
        final_logits, hidden_states = model.forward(ids)
        final_log_probs = F.log_softmax(final_logits.double(), dim=-1)
        final_totals.add(torch.zeros(ids.shape), token_cross_entropy(final_logits, ids))
        first_record = record["record_index"] == 0
        if first_record:
            exported["reference.final"] = final_logits[:, :positions].clone().contiguous()
            predictions["final"] = top_predictions(exported["reference.final"], tokenizer)
        for layer, hidden in enumerate(hidden_states):
            translated = hidden + F.linear(
                hidden, translators[f"translators.{layer}.weight"],
                translators[f"translators.{layer}.bias"],
            )
            for kind, state in (("raw", hidden), ("tuned", translated)):
                logits = model.readout(state)
                totals[kind][layer].add(
                    token_kl(final_log_probs, logits), token_cross_entropy(logits, ids)
                )
                if first_record:
                    key = f"reference.{kind}.{layer}"
                    exported[key] = logits[:, :positions].clone().contiguous()
                    predictions[kind].append(top_predictions(exported[key], tokenizer))
        print(f"Reference record {record['record_index']}: {ids.numel()} KL positions, {ids.numel()-1} CE labels", flush=True)
    final = final_totals.means()
    measurements = {
        "input_tokens": final["kl_tokens"],
        "ce_tokens": final["ce_tokens"],
        "final_ce": final["ce_nats_per_token"],
        "layers": [
            {"index": layer,
             "point": "decoder.embedding" if layer == 0 else f"decoder.layers.{layer-1}.residual.post",
             "raw_kl": totals["raw"][layer].means()["kl_nats_per_token"],
             "tuned_kl": totals["tuned"][layer].means()["kl_nats_per_token"],
             "raw_ce": totals["raw"][layer].means()["ce_nats_per_token"],
             "tuned_ce": totals["tuned"][layer].means()["ce_nats_per_token"]}
            for layer in range(layers)
        ],
    }
    return exported, measurements, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads (default: 1)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    preparation_path = OUTPUT / "preparation.json"
    manifest_path = HERE / "manifest.json"
    manifest = strict_json(manifest_path.read_bytes())
    # Check preparation code/runtime identity and EVERY generated artifact,
    # including the proof that historical and later checkpoint tensors match.
    if not verified_output_cache(OUTPUT, fingerprint(manifest_path)):
        raise ValueError("Preparation is absent or stale; rerun prepare.py")
    preparation = strict_json(preparation_path.read_bytes())
    protocol = manifest["protocol"]
    paths = {name: OUTPUT / item["path"] for name, item in preparation["outputs"].items()}
    fixture = strict_json(paths["fixture"].read_bytes())
    config = strict_json(paths["config"].read_bytes())
    validate_fixture(fixture, protocol, config["vocab_size"])
    model = PythiaReference(load_file(paths["model"]), config)
    translators = load_file(paths["lens"])
    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    started = time.perf_counter()
    tensors, measurements, predictions = run_study(model, translators, fixture, protocol, tokenizer)
    duration = time.perf_counter() - started
    serialized = save(tensors)
    metadata = {
        "schema_version": 1,
        "implementation": {"kind": "independent PyTorch equations", "upstream": UPSTREAM,
                           "torch": str(torch.__version__), "safetensors": safetensors.__version__,
                           "python": platform.python_version(), "dtype": "F32", "threads": args.threads,
                           "metric_precision": "F64 log-softmax, KL, cross-entropy, and token aggregation on unchanged F32 model/lens logits",
                           "numerical_correction": "Original CPU F32 vocabulary reductions biased KL. Independent PyTorch and NumPy F64 calculations agreed; model computation, exported logits, metric definitions, data, and predeclared tolerances are unchanged.",
                           "note": "Cached F32 rotary tables, packed QKV, exact GELU; no training or fitting."},
        "provenance": {"manifest_sha256": file_hash(manifest_path),
                       "preparation_sha256": file_hash(preparation_path),
                       "reference_script_sha256": file_hash(Path(__file__)),
                       "inputs": {name: preparation["outputs"][name] for name in paths}},
        "protocol": protocol,
        "parity": {"sample_tokens": protocol["reference_positions"], **manifest["parity"]},
        "metrics": measurements,
        "top_predictions": {"record_index": 0, "input_ids": fixture["records"][0]["input_ids"][:protocol["reference_positions"]],
                            "positions": list(range(protocol["reference_positions"])), **predictions},
        "outputs": {"reference": {"path": "reference.safetensors", "bytes": len(serialized),
                                  "sha256": hashlib.sha256(serialized).hexdigest(),
                                  "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()}}},
        "reference_forward_seconds": duration,
    }
    encoded_metadata = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode()
    atomic_write(OUTPUT / "reference.safetensors", serialized)
    atomic_write(OUTPUT / "reference.json", encoded_metadata)
    print(f"Wrote reference logits and token-weighted metrics in {duration:.2f}s to {OUTPUT}")


if __name__ == "__main__":
    main()
