"""Scratch diagnostic: the official implementation's block 0 with PyTorch's SiLU and with Burn's composed SiLU.

Usage: python3 -I torch_block0.py MODEL_DIR TOKENS_FILE OUT_DIR [VARIANT...]

Loads only the embedding and decoder layer 0 of a local Llama-family checkpoint
(Transformers AutoModelForCausalLM with num_hidden_layers=1, local files only),
runs every prompt of TOKENS_FILE (one comma-separated line per prompt) through it,
and writes raw F32 files under OUT_DIR/<variant>/prompt<i>/: embedding.bin,
layer0.bin (residual after block 0), mlp0_pre.bin (the act_fn input, the gate
projection) and mlp0_act.bin (its activation). Variants are
<device>-<dtype>-eager[-burnsilu]; "-burnsilu" replaces act_fn by
x * torch.sigmoid(x), the operation order of Burn's activation::silu.
"""

import json
import sys
from pathlib import Path

import torch
import transformers

VARIANTS = {}
for device in ("cpu", "mps"):
    for dtype in ("bfloat16", "float16"):
        VARIANTS[f"{device}-{dtype}-eager"] = (device, dtype, False)
        VARIANTS[f"{device}-{dtype}-eager-burnsilu"] = (device, dtype, True)


class RoundedSigmoidSilu(torch.nn.Module):
    """SiLU as x * sigmoid(x) with the sigmoid rounded to the activation dtype first."""

    def forward(self, x):
        return x * torch.sigmoid(x)


def one_layer_config(model_dir):
    """Returns the checkpoint's config truncated to its first decoder layer."""
    config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
    for name in ("layer_types", "no_rope_layers", "sliding_window_pattern_layers"):
        value = getattr(config, name, None)
        if isinstance(value, list):
            setattr(config, name, value[:1])
    config.num_hidden_layers = 1
    return config


def main():
    model_dir, tokens_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    chosen = sys.argv[4:] or list(VARIANTS)
    prompts = [[int(t) for t in line.split(",")] for line in tokens_file.read_text().splitlines() if line.strip()]
    torch.use_deterministic_algorithms(True)
    transformers.utils.logging.disable_progress_bar()
    transformers.utils.logging.set_verbosity_error()
    for name in chosen:
        device, dtype, burnsilu = VARIANTS[name]
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir, config=one_layer_config(model_dir), local_files_only=True,
            dtype=getattr(torch, dtype), attn_implementation="eager",
        )
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.data = parameter.data.to(getattr(torch, dtype))
        layer = model.model.layers[0]
        if burnsilu:
            layer.mlp.act_fn = RoundedSigmoidSilu()
        model.eval().to(device)
        captured = {}

        def keep(label, tensor):
            captured[label] = tensor[0].to(torch.float32).cpu().numpy()

        def activation_hook(_module, inputs, output):
            # A forward hook that returns a value replaces the output, so return nothing.
            keep("mlp0_pre", inputs[0])
            keep("mlp0_act", output)

        handles = [
            model.model.embed_tokens.register_forward_hook(lambda m, i, o: keep("embedding", o)),
            layer.register_forward_hook(lambda m, i, o: keep("layer0", o[0] if isinstance(o, tuple) else o)),
            layer.mlp.act_fn.register_forward_hook(activation_hook),
        ]
        for index, tokens in enumerate(prompts):
            captured.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([tokens], device=device), use_cache=False)
            target = out / name / f"prompt{index}"
            target.mkdir(parents=True, exist_ok=True)
            for label, value in captured.items():
                value.astype("<f4").tofile(target / f"{label}.bin")
        for handle in handles:
            handle.remove()
        print(f"{name}: {len(prompts)} prompts written", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    print(f"torch {torch.__version__} transformers {transformers.__version__}")
    (out / "versions.json").write_text(json.dumps({"torch": torch.__version__, "transformers": transformers.__version__}))


if __name__ == "__main__":
    main()
