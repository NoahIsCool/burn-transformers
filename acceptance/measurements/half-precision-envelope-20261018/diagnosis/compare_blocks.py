"""Scratch (half2 S9 diagnosis): per-block F16-against-F32 comparison of Qwen2-7B, official and Burn (not committed).

Usage: python3 -I compare_blocks.py DIAG_DIR

Reads DIAG_DIR/torch-<dtype>-<device>-<attention>.npz (torch_blocks.py) and
DIAG_DIR/burn2-metal-<dtype>/prompt<i>/ (block-dump: the default graph's
points, and the eager graph's with an `.eager` suffix plus its scaled scores),
and prints for every
graph and prompt, per block: the largest |value| of the residual entering the
block, of the attention output, of the MLP's down_proj input (torch only), of
the MLP output and of the residual after it, split into position 0 and
positions >= 1; the first non-finite point; for eager torch graphs the
largest |q k^T| in the model dtype; and each F16 graph's max |Δ| from the
same implementation's F32 graph (official F32 CPU eager for torch, Burn
Metal F32 on the same graph for Burn) at position 0 and at positions >= 1,
where both are finite.
"""

import sys
from pathlib import Path

import numpy as np

HIDDEN = 3584
LAYERS = 28
PROMPTS = [5, 13, 8]
F16_MAX = 65504.0


def torch_graph(path):
    """Per-prompt, per-block activations of one torch graph."""
    data = np.load(path)
    out = []
    for p in range(len(PROMPTS)):
        blocks = []
        for l in range(LAYERS):
            block = {name: data[f"p{p}.l{l}.{name}"] for name in ("pre", "attn", "act", "mlp", "post")}
            if f"p{p}.l{l}.scores_raw_max" in data:
                block["scores_raw_max"] = float(data[f"p{p}.l{l}.scores_raw_max"])
                block["scores_nonfinite"] = int(data[f"p{p}.l{l}.scores_nonfinite"])
                block["scores_argmax"] = data[f"p{p}.l{l}.scores_finite_argmax"].tolist()
            blocks.append(block)
        out.append(blocks)
    return out


def burn_graph(root, suffix):
    """Per-prompt, per-block activations of one Burn dump's default (`""`) or eager (`".eager"`) graph."""
    out = []
    for p, tokens in enumerate(PROMPTS):
        blocks = []
        for l in range(LAYERS):
            block = {}
            for name in ("pre", "attn", "mlp", "post"):
                block[name] = np.fromfile(root / f"prompt{p}/l{l}.{name}{suffix}.bin", dtype="<f4").reshape(tokens, HIDDEN)
            if suffix:
                scores = np.fromfile(root / f"prompt{p}/l{l}.scores.bin", dtype="<f4")
                finite = scores[np.isfinite(scores)]
                block["scores_max"] = float(np.abs(finite).max()) if finite.size else float("nan")
                block["scores_nonfinite"] = int((~np.isfinite(scores)).sum())
            blocks.append(block)
        out.append(blocks)
    return out


def split_max(array):
    """Largest |value| at position 0 and at positions >= 1 (inf when non-finite)."""
    def largest(part):
        if part.size == 0:
            return float("nan")
        return float("inf") if not np.isfinite(part).all() else float(np.abs(part).max())
    return largest(array[:1]), largest(array[1:])


def split_delta(a, b):
    """max |a - b| at position 0 and at positions >= 1, over entries finite in both."""
    def delta(x, y):
        keep = np.isfinite(x) & np.isfinite(y)
        return float(np.abs(x[keep].astype(np.float64) - y[keep]).max()) if keep.any() else float("nan")
    return delta(a[:1], b[:1]), delta(a[1:], b[1:])


def first_nonfinite(blocks):
    """(block, point, position) of the first non-finite activation in execution order."""
    for l, block in enumerate(blocks):
        for name in ("pre", "attn", "act", "mlp", "post"):
            if name in block and not np.isfinite(block[name]).all():
                rows = np.where(~np.isfinite(block[name]).all(axis=1) | ~np.isfinite(block[name]).any(axis=1))[0]
                bad = np.where(~np.isfinite(block[name]))
                return l, name, int(bad[0].min()), int(len(set(bad[0].tolist())))
    return None


def fmt(pair):
    """Two numbers for a table cell."""
    return " / ".join("nan" if np.isnan(v) else ("inf" if np.isinf(v) else f"{v:.4g}") for v in pair)


def main():
    root = Path(sys.argv[1])
    graphs = {}
    for path in sorted(root.glob("torch-*.npz")):
        graphs[path.stem] = torch_graph(path)
    for path in sorted(root.glob("burn2-metal-*")):
        if path.is_dir():
            graphs[path.name] = burn_graph(path, "")
            graphs[path.name + "-eager"] = burn_graph(path, ".eager")
    pairs = {"torch-float16-cpu-eager": "torch-float32-cpu-eager", "torch-float16-cpu-sdpa": "torch-float32-cpu-eager",
             "torch-float16-mps-eager": "torch-float32-cpu-eager", "burn2-metal-f16": "burn2-metal-f32",
             "burn2-metal-f16-eager": "burn2-metal-f32-eager", "burn2-metal-f32": "torch-float32-cpu-eager"}
    print(f"== graphs: {', '.join(graphs)}")
    for path in sorted(root.glob("burn2-metal-*")):
        for name, label in (("logits_plain.bin", "default"), ("logits_eager.bin", "eager")):
            rows = []
            for p, tokens in enumerate(PROMPTS):
                logits = np.fromfile(path / f"prompt{p}/{name}", dtype="<f4").reshape(tokens, -1)
                rows.append("".join("F" if np.isfinite(row).all() else "x" for row in logits))
            print(f"== {path.name} {label} graph logits finite per position (F finite, x not), prompts 0-2: {' '.join(rows)}")
    for path in sorted(root.glob("burn2-metal-*")):
        for p, tokens in enumerate(PROMPTS):
            for l in range(LAYERS):
                scores = np.fromfile(path / f"prompt{p}/l{l}.scores.bin", dtype="<f4")
                scores = scores.reshape(-1, tokens, tokens)
                if np.isfinite(scores).all():
                    continue
                allowed = np.tril(np.ones((tokens, tokens), bool))
                bad = ~np.isfinite(scores)
                signs = sorted({float(v) for v in np.sign(scores[bad])})
                dead = [(h, q) for h in range(scores.shape[0]) for q in range(tokens)
                        if np.all(np.isneginf(scores[h, q, :q + 1]))]
                print(f"== {path.name} eager scores prompt {p} block {l}: {int(bad.sum())} non-finite "
                      f"({int((bad & allowed).sum())} at allowed keys), signs {signs}; (head, query) rows whose every "
                      f"allowed score is -inf: {len(dead)} {dead}")
    for name in sorted(graphs):
        if name.startswith("torch-"):
            data = np.load(root / f"{name}.npz")
            rows = ["".join("F" if ok else "x" for ok in data[f"p{p}.logits_finite_rows"]) for p in range(len(PROMPTS))]
            print(f"== {name} logits finite per position (F finite, x not), prompts 0-2: {' '.join(rows)}")
    for name, blocks_by_prompt in graphs.items():
        for p, blocks in enumerate(blocks_by_prompt):
            print(f"\n== {name} prompt {p} ({PROMPTS[p]} tokens): first non-finite {first_nonfinite(blocks)}")
            header = "block | max|pre| t0 / t>=1 | max|attn| | max|act| | max|mlp| | max|post|"
            reference = graphs.get(pairs.get(name, ""), None)
            if reference:
                header += f" | max|Δ post| vs {pairs[name]} t0 / t>=1"
            header += " | scores"
            print(header)
            for l, block in enumerate(blocks):
                row = [str(l), fmt(split_max(block["pre"])), fmt(split_max(block["attn"])),
                       fmt(split_max(block["act"])) if "act" in block else "-", fmt(split_max(block["mlp"])),
                       fmt(split_max(block["post"]))]
                if reference:
                    row.append(fmt(split_delta(block["post"], reference[p][l]["post"])))
                if "scores_raw_max" in block:
                    row.append(f"|qk| {block['scores_raw_max']:.6g} nonfinite {block['scores_nonfinite']} at {block['scores_argmax'][:3]}")
                elif "scores_max" in block:
                    row.append(f"|scaled| {block['scores_max']:.6g} nonfinite {block['scores_nonfinite']}")
                else:
                    row.append("-")
                print(" | ".join(row))


if __name__ == "__main__":
    main()
