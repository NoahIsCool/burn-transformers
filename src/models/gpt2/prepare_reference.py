"""Regenerate GPT-2 parity fixtures with PyTorch/Transformers, without training.

Run normally for the committed tiny fixture. Add --checkpoint /local/gpt2
--output /tmp/gpt2-reference.json to measure existing pretrained weights.
No network access or downloads occur in this helper.
"""

import argparse
import hashlib
import json
from pathlib import Path

import safetensors.torch
import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel


def synthetic_parameters(state_dict):
    """Small deterministic test data; no parameter payload is checked in."""
    weights = {}
    mask = (1 << 64) - 1
    for tensor_index, (name, tensor) in enumerate(sorted(state_dict.items())):
        state = (20260910 ^ ((tensor_index + 1) * 0x9E3779B97F4A7C15)) & mask
        values = []
        is_scale = name.endswith(".weight") and (".ln_" in name or name.startswith("ln_f."))
        for _ in range(tensor.numel()):
            state ^= (state << 13) & mask
            state ^= state >> 7
            state ^= (state << 17) & mask
            centered = (state >> 40) - (1 << 23)
            # Nontrivial MLP inputs make exact and tanh GELU distinguishable
            # beyond the regression tolerance, even in this tiny graph.
            shift = 32 if name.endswith(".bias") else 23 if name.endswith("mlp.c_fc.weight") else 29
            value = centered / (1 << shift)
            values.append(1.0 + value if is_scale else value)
        weights[name] = torch.tensor(values, dtype=torch.float32).reshape(tensor.shape)
    return weights


def measure(model, token_ids, token_mask=None):
    """Keep full last-position logits and the actual component boundaries."""
    captured = {}
    handles = []
    for index, block in enumerate(model.transformer.h):
        def capture(_module, _inputs, output, index=index):
            captured[f"mlp_{index}"] = output.detach().cpu().tolist()

        handles.append(block.mlp.register_forward_hook(capture))
    with torch.no_grad():
        output = model(
            token_ids,
            attention_mask=token_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    for handle in handles:
        handle.remove()
    return {
        "logits": output.logits.detach().cpu().tolist(),
        "hidden_states": [x.detach().cpu().tolist() for x in output.hidden_states],
        "mlp_outputs": captured,
    }


def main():
    transformers.utils.logging.disable_progress_bar()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--activation", choices=["gelu", "gelu_new"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20260910)
    versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    if args.checkpoint:
        config = GPT2Config.from_pretrained(args.checkpoint, local_files_only=True)
        if args.activation:
            config.activation_function = args.activation
        model = GPT2LMHeadModel.from_pretrained(
            args.checkpoint, config=config, local_files_only=True,
            attn_implementation="eager",
        ).float().eval()
        ids = torch.tensor([
            [50256, 383, 1175, 15436, 422, 262, 614, 1596, 2624, 284, 262, 614, 1596],
            [50256, 383, 1175, 15436, 422, 262, 614, 1596, 486, 284, 262, 614, 1596],
        ])
        measured = measure(model, ids)
        # Readout parity needs every vocabulary logit, but only the last token.
        measured["logits"] = [[row[-1]] for row in measured["logits"]]
        result = {
            "versions": versions, "config": model.config.to_dict(),
            "input_ids": ids.tolist(), "reference": measured,
        }
        target = args.output or Path("/tmp/gpt2-reference.json")
    else:
        target_dir = Path(__file__).parent / "fixtures"
        target_dir.mkdir(exist_ok=True)
        config = GPT2Config(
            vocab_size=19, n_positions=16, n_embd=12, n_layer=2, n_head=3,
            n_inner=17, activation_function="gelu_new",
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).float().eval()
        weights = synthetic_parameters(model.transformer.state_dict())
        model.transformer.load_state_dict(weights)
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        references = {"gelu_new": measure(model, ids, mask)}
        config.activation_function = "gelu"
        exact_model = GPT2LMHeadModel(config).float().eval()
        exact_model.transformer.load_state_dict(weights)
        references["gelu"] = measure(exact_model, ids, mask)
        result = {
            "versions": versions,
            "seed": 20260910,
            "synthetic_parameter_generator": "xorshift64-v2 (see synthetic_parameters)",
            "parameter_shapes": {name: list(tensor.shape) for name, tensor in weights.items()},
            "checkpoint_sha256": hashlib.sha256(safetensors.torch.save(weights)).hexdigest(),
            "config": config.to_dict() | {"activation_function": "gelu_new"},
            "input_ids": ids.tolist(), "token_mask": mask.tolist(), "references": references,
        }
        target = args.output or target_dir / "tiny_reference.json"
    target.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(target)


if __name__ == "__main__":
    main()
