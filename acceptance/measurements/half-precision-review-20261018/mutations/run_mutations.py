"""Review round: shows that each new test fails when the call site or rounding point it pins is reverted.

Usage: python3 -I run_mutations.py CLONE LOGS

CLONE is a scratch Git clone of the worktree checked out at the round's code
head (clean); LOGS receives one log per mutation and a baseline log. Each
mutation replaces exactly one source fragment, runs the named test with
`cargo test --locked` (debug profile, CARGO_TARGET_DIR under
/Volumes/bag of holding/cargo-targets/), records the diff and the output, and
restores the file from Git. A mutation must make its test fail; the baseline
(no mutation) must pass.
"""

import os
import subprocess
import sys
from datetime import datetime, timezone

TARGET = "/Volumes/bag of holding/cargo-targets/tfm-half-followups2-mutate"
EAGER = ("--lib", "attention::dispatch::tests::half_precision_eager_attention_rounds_like_transformers_eager")
#: (name, file, original fragment, mutated fragment, test selector, description)
MUTATIONS = [
    ("eager-scores-native-matmul", "src/attention/dispatch.rs",
     "f32_accumulated_matmul(fold_groups(query, key_value_heads), key.swap_dims(2, 3))",
     "fold_groups(query, key_value_heads).matmul(key.swap_dims(2, 3))", EAGER,
     "scaled_scores forms Q K^T with Burn's own matmul (Flex F16 partial sums)"),
    ("eager-mix-native-matmul", "src/attention/dispatch.rs",
     "f32_accumulated_matmul(fold_groups(probabilities, key_value_heads), value)",
     "fold_groups(probabilities, key_value_heads).matmul(value)", EAGER,
     "grouped_mix forms P V with Burn's own matmul (Flex F16 partial sums)"),
    ("eager-softmax-in-dtype", "src/attention/dispatch.rs",
     "            .mask_fill(unsupported_row.clone(), 0.0)\n            .cast(DType::F32),",
     "            .mask_fill(unsupported_row.clone(), 0.0),", EAGER,
     "fp32_softmax normalizes in the activation dtype"),
    ("eager-rounded-scale", "src/attention/dispatch.rs",
     "crate::feed_forward::evaluate_in_f32(product, |product| product * scale)",
     "(product * scale)", EAGER,
     "scaled_scores multiplies by the dtype-rounded scale (Burn's tensor * scalar)"),
    ("mamba2-composed-silu", "src/mamba2.rs",
     "crate::feed_forward::silu(convolved).swap_dims(1, 2)",
     "activation::silu(convolved).swap_dims(1, 2)",
     ("--lib", "mamba2::tests::convolution_silu_is_rounded_once_in_half_precision"),
     "the Mamba-2 convolution SiLU call site reverted to activation::silu"),
    ("delta-net-composed-silu", "src/delta_net.rs",
     "crate::feed_forward::silu(convolution.output).swap_dims(1, 2)",
     "activation::silu(convolution.output).swap_dims(1, 2)",
     ("--lib", "delta_net::tests::convolution_silu_is_rounded_once_in_half_precision"),
     "the Gated DeltaNet convolution SiLU call site reverted to activation::silu"),
    ("gemma-f16-reads-bf16-graphs", "tests/gemma_artifacts.rs",
     "    if dtype == TensorDType::F16 {\n        variables.official_f16",
     "    if false {\n        variables.official_f16",
     ("--test", "gemma_artifacts", "official_graphs_follow_the_run_dtype"),
     "official_variable gives an F16 run the official BF16 graphs"),
]


def run(cmd, cwd, env=None):
    """Runs ``cmd`` and returns (exit status, combined output)."""
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def stamp():
    """The current UTC time in ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def test_command(selector):
    """The cargo command running one test exactly."""
    *target, name = selector
    return ["cargo", "test", "--locked", *target, name, "--", "--exact"]


def write_log(path, lines):
    """Writes a log file."""
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    """Runs the baseline and every mutation; returns non-zero if any outcome is unexpected."""
    clone, logs = sys.argv[1], sys.argv[2]
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ, CARGO_TARGET_DIR=TARGET)
    head = run(["git", "rev-parse", "HEAD"], clone)[1].strip()
    unexpected = 0
    for selector in sorted({m[4] for m in MUTATIONS}):
        status, output = run(test_command(selector), clone, env)
        name = "baseline." + selector[-1].replace("::tests::", ".").replace("::", ".")
        write_log(os.path.join(logs, name + ".log"),
                  [f"== source {head} dirty={len(run(['git', 'status', '--porcelain'], clone)[1].splitlines())}",
                   f"== mutation none (baseline) {stamp()}", f"== cmd {' '.join(test_command(selector))}",
                   *[line for line in output.splitlines() if line.startswith(("test ", "running", "thread"))],
                   f"== exit={status} expected=0"])
        unexpected += status != 0
    for name, path, original, mutated, selector, description in MUTATIONS:
        full = os.path.join(clone, path)
        text = open(full).read()
        if text.count(original) != 1:
            raise SystemExit(f"{name}: the original fragment occurs {text.count(original)} times")
        open(full, "w").write(text.replace(original, mutated))
        diff = run(["git", "diff", "--", path], clone)[1]
        status, output = run(test_command(selector), clone, env)
        open(full, "w").write(text)
        clean = run(["git", "status", "--porcelain"], clone)[1].strip()
        lines = [f"== source {head} with one mutation: {description}", f"== start {stamp()}",
                 f"== cmd {' '.join(test_command(selector))}", "== diff", *diff.splitlines(), "== output",
                 *[line for line in output.splitlines()
                   if line.startswith(("test ", "running", "thread", "assertion", "  left", " right"))
                   or "failed:" in line],
                 f"== exit={status} expected=failure", f"== restored, git status: {clean or 'clean'}"]
        write_log(os.path.join(logs, f"mutation.{name}.log"), lines)
        unexpected += status == 0 or bool(clean)
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
