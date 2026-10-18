"""Scratch (half2 S9 diagnosis): Qwen3-4B Flex/BF16 eager and fused dumps against the gate's envelope (not committed).

Usage: python3 -I compare_dumps.py DIAG_DIR GATE_LOG_DIR

Reads the burn-dump outputs DIAG_DIR/data/qwen3-4b-flex-bf16-{fused,eager}-decode/
(prompt 0 plus the truth's 32 greedy tokens, prefilled for 5 tokens and then
decoded one token at a time) and DIAG_DIR/data/qwen3-4b-flex-bf16-eager-prompt{1,2}/,
measures them the way tests/support/half_precision_calibration.rs does (max and
mean |Δ| over every logit, in F64) against the F32 truth's prompt logits and the
F32 CPU-eager decode truth's stateless rows, and prints each statistic with the
bound the f9f78b5 gate printed for the same statistic (read from the committed
Qwen3-4B Flex/BF16 gate logs). The fused rows must reproduce the gate's own
printed statistics, which checks this measurement.
"""

import re
import sys
from pathlib import Path

import numpy as np

REFERENCES = Path("/Volumes/T9/transformers-references")
TRUTH = REFERENCES / "Qwen3-4B-531c80e2"
DECODE_TRUTH = REFERENCES / "Qwen3-4B-531c80e2-f32-cpu-eager"
VOCABULARY = 151936
PROMPTS = [5, 13, 8]
STEPS = 32


def rows(path, count):
    """`count` rows of raw little-endian F32 logits."""
    values = np.fromfile(path, dtype="<f4")
    return values.reshape(-1, VOCABULARY)[:count] if count else values.reshape(-1, VOCABULARY)


def measure(actual, truth):
    """(max |Δ|, mean |Δ|) over every logit, in F64; None when a value is non-finite."""
    if not np.isfinite(actual).all():
        return None
    gap = np.abs(actual.astype(np.float64) - truth.astype(np.float64))
    return float(gap.max()), float(gap.mean())


def bounds(log_dir):
    """The gate's printed bounds: per prompt (max, mean), and decode rows (max, mean)."""
    prompt = (log_dir / "Qwen3-4B.bf16.flex.prompt_logits_match_the_reference.log").read_text()
    cached = (log_dir / "Qwen3-4B.bf16.flex.cached_greedy_matches_the_reference.log").read_text()
    number = r"([0-9.e+-]+)"
    per_prompt = {int(m.group(1)): (float(m.group(4)), float(m.group(6))) for m in re.finditer(
        rf"^prompt (\d+): \d+ tokens, max \|Δ\| = {number} at position (\d+) \(bound {number}\), mean \|Δ\| = {number} \(bound {number}\)",
        prompt, re.M)}
    assert sorted(per_prompt) == [0, 1, 2], per_prompt
    decode = re.search(rf"^cached decode \(cached rows\): .* \(bound {number}\), mean \|Δ\| = {number} \(bound {number}\)", cached, re.M)
    gate = {"cached": re.search(rf"^cached decode \(cached rows\): .*max \|Δ\| = {number} at step", cached, re.M).group(1),
            "stateless": re.search(rf"^cached decode \(stateless rows\): .*max \|Δ\| = {number} at step", cached, re.M).group(1),
            "prompt0": re.search(rf"^prompt 0: .*max \|Δ\| = {number} at position", prompt, re.M).group(1)}
    return per_prompt, (float(decode.group(1)), float(decode.group(3))), gate


def show(label, statistic, bound):
    """One line: the statistic, its bound, and the ratio."""
    if statistic is None:
        print(f"{label}: non-finite")
        return
    verdict = "within" if statistic[0] <= bound[0] and statistic[1] <= bound[1] else "OUTSIDE"
    print(f"{label}: max |Δ| = {statistic[0]:.6g} (bound {bound[0]:.6g}, {statistic[0] / bound[0]:.3f} of it), "
          f"mean |Δ| = {statistic[1]:.6g} (bound {bound[1]:.6g}, {statistic[1] / bound[1]:.3f} of it): {verdict} the envelope")


def main():
    diag, logs = Path(sys.argv[1]), Path(sys.argv[2])
    per_prompt, decode_bound, gate = bounds(logs)
    truth_rows = [rows(TRUTH / f"logits-{i}.f32le", 0) for i in range(3)]
    decode_truth = rows(DECODE_TRUTH / "decode-stateless.f32le", 0)
    assert decode_truth.shape[0] == STEPS
    print(f"== gate f9f78b5 Flex/BF16 printed: prompt 0 max {gate['prompt0']}, cached rows max {gate['cached']}, "
          f"stateless rows max {gate['stateless']}")
    for attention in ("fused", "eager"):
        root = diag / f"data/qwen3-4b-flex-bf16-{attention}-decode"
        plain = rows(root / "logits_plain.bin", 0)
        cached = rows(root / "decode_logits.bin", 0)
        assert plain.shape[0] == PROMPTS[0] + STEPS and cached.shape[0] == STEPS
        print(f"\n== Flex/BF16 {attention} attention")
        show(f"{attention} prompt 0", measure(plain[:PROMPTS[0]], truth_rows[0]), per_prompt[0])
        show(f"{attention} cached decode rows", measure(cached, decode_truth), decode_bound)
        show(f"{attention} stateless decode rows", measure(plain[PROMPTS[0]:], decode_truth), decode_bound)
        gap = float(np.abs(cached.astype(np.float64) - plain[PROMPTS[0]:]).max())
        print(f"{attention} cached-versus-stateless gap max |Δ| = {gap:.6g}")
        if attention == "eager":
            for index in (1, 2):
                prompt = rows(diag / f"data/qwen3-4b-flex-bf16-eager-prompt{index}/logits_plain.bin", 0)
                show(f"eager prompt {index}", measure(prompt, truth_rows[index]), per_prompt[index])


if __name__ == "__main__":
    main()
