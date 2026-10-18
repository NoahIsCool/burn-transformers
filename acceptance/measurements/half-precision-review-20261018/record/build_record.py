"""Review round: writes acceptance/measurements/half-precision-review-20261018.json from its campaign directory.

Usage: python3 -I build_record.py WORKTREE

Reads the campaign directory ``acceptance/measurements/half-precision-review-20261018/``
(gate and operation-level run logs, the mutation logs, the torch probe, the
benchmark comparison, the preconditions, and the source validation), and
writes the record with every file's SHA-256. Prose is the round's account;
every status, digest, and quote is read from the logs (a whole line, or for
the Metal BF16 load error the error itself out of the load report's
one-line dump), so the record's Python test can recompute them.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

CAMPAIGN = "half-precision-review-20261018"
RECORD = f"{CAMPAIGN}.json"
SOURCE = "1e38eecf989a1ae568b67b2e9fcdd9b2ebdbadb8"
BASE = "8e6f85b51b3a0f06feb36a91e093f78a810cf4d1"
HISTORICAL = ["half-precision-calibration-20261017.json", "mac-validation-20261017.json",
              "half-precision-envelope-20261018.json", "half-precision-followups-20261017.json"]


def sha256(data):
    """Hexadecimal SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def line(text, pattern):
    """The first line of ``text`` containing ``pattern`` (which must occur)."""
    for candidate in text.splitlines():
        if pattern in candidate:
            return candidate
    raise SystemExit(f"{pattern!r} not found")


def main():
    """Builds the record."""
    worktree = Path(sys.argv[1])
    measurements = worktree / "acceptance/measurements"
    campaign = measurements / CAMPAIGN

    def text(relative):
        """Text of a campaign file."""
        return (campaign / relative).read_text()

    def evidence(relative, *patterns):
        """A {log, quotes} entry quoting the first line holding each pattern."""
        body = text(relative)
        return {"log": relative, "quotes": [line(body, pattern) for pattern in patterns]}

    build = text("gates/logs/build.log")
    binaries = [{"path": path, "sha256": digest} for path, digest in
                re.findall(r"^== binary (.+) sha256 ([0-9a-f]{64})$", build, re.M)]
    runs = []
    for raw in text("gates/logs/runs.jsonl").splitlines():
        run = json.loads(raw)
        name = run["name"]
        expected = "fail" if name.startswith(("uncalibrated.", "refusal.")) else "pass"
        log = f"gates/logs/{name}.log"
        entry = {"name": name, "kind": run["kind"], "test": run["test"], "exit": run["exit"], "result": run["result"],
                 "expected": expected, "wall_seconds": run["wall_seconds"],
                 "peak_memory_footprint_bytes": run["peak_memory_footprint_bytes"], "log": log}
        body = text(log)
        if name.startswith("uncalibrated."):
            entry["quotes"] = [line(body, "uncalibrated: official ")]
        elif name.startswith("refusal."):
            entry["quotes"] = [re.search(r"cannot install Gemma parameter `[^`]+`: backend cannot store checkpoint "
                                         r"dtype BF16", body).group(0)]
        elif name.startswith("operations.metal"):
            entry["quotes"] = [line(body, "Metal F16")]
        runs.append(entry)
    tallies = {"runs": len(runs), "pass": sum(r["result"] == "pass" for r in runs),
               "fail": sum(r["result"] == "fail" for r in runs),
               "as_expected": sum(r["result"] == r["expected"] for r in runs)}
    mutation_logs = sorted(p.name for p in (campaign / "mutations/logs").iterdir())
    mutations = []
    for name in mutation_logs:
        body = text(f"mutations/logs/{name}")
        status = int(re.search(r"^== exit=(-?\d+)", body, re.M).group(1))
        entry = {"log": f"mutations/logs/{name}", "exit": status,
                 "expected": "pass" if name.startswith("baseline.") else "fail"}
        if name.startswith("mutation."):
            entry["mutation"] = line(body, "== source").split("with one mutation: ", 1)[1]
            entry["quotes"] = [line(body, "panicked at"), line(body, "test result")]
        mutations.append(entry)
    validation = text("validation/summary.log")
    checks = [{"name": m.group(1), "exit": int(m.group(2)), "wall_seconds": int(m.group(3)),
               "log": f"validation/{m.group(1)}.log"}
              for m in re.finditer(r"^(\S+) exit=(\d+) wall=(\d+)s", validation, re.M)]
    commands = {}
    for check in checks:
        commands[check["name"]] = line(text(check["log"]), "== cmd ").removeprefix("== cmd ")
        check["command"] = commands[check["name"]]

    record = {
        "schema_version": 1,
        "scope": "half_precision_review_round_not_a_support_declaration",
        "date_utc": "2026-10-18",
        "advertised_supported": False,
        "passed": False,
        "passed_note": ("Every run of this round has its expected outcome, but the round qualifies nothing: the "
                        "Nemotron-H official-precision failure at f9f78b5 stands (not rerun), and the Qwen3.5-0.8B "
                        "cold-first-token breach of the approved September 10 baseline awaits the maintainer."),
        "source_commit": SOURCE,
        "base_commit": BASE,
        "checkout": "/Users/nover/repos/transformers-worktrees/tfm-half-followups2 (branch claude/tfm-half-followups2)",
        "head_note": ("Every gate, operation-level, and mutation binary was built from the code head "
                      f"{SOURCE[:7]}: the gate and library test binaries in the worktree itself (clean, verified "
                      "before and after every run), the mutations in a scratch clone at the same commit. The commits "
                      "after the code head (63647b5, the innovation-matrix sentence, and the record commit) change "
                      "only documentation, this record, its campaign directory, and its Python test; no Rust "
                      "source, test, or Cargo file changes after the code head."),
        "sources": [
            {"commit": "90f6152df3e31c1867a7abb33ae366714eb1ed27",
             "subject": "fix(transformers): sum eager half-precision attention products in F32"},
            {"commit": "f7b8d99b54cbe4e342c17e85f1b5a528267a1cac",
             "subject": "test(transformers): pin the once-rounded SiLU at both convolution sites"},
            {"commit": "e2f72797804396c688b2228ba6a3c6920a1a5639",
             "subject": "test(transformers): judge Gemma F16 runs as uncalibrated until decision (d)"},
            {"commit": "1a60aeb975e296ab2e844d46d5eca24c8af4a691",
             "subject": "test(transformers): pin Metal's eager F16 attention to Transformers' eager graph"},
            {"commit": SOURCE,
             "subject": "docs(transformers): state the Nemotron-H failure and the eager graph's users"},
            {"commit": "63647b56712084822814f81a373cb4ee7c73b7a9",
             "subject": "docs(transformers): state the eager graph's rounding points in the innovation matrix",
             "note": "documentation only, after the code head"},
        ],
        "cargo_lock_sha256": sha256((worktree / "Cargo.lock").read_bytes()),
        "toolchain": {
            "rustc": "1.94.0-nightly (4f14395c3 2025-12-22), aarch64-apple-darwin",
            "cargo": "1.94.0-nightly (3861f60f6 2025-12-19)",
            "clippy_toolchains": ["nightly 1.94.0 (4f14395c3 2025-12-22)", "1.99.1 (58cf8f9c3 2026-10-13)"],
            "msrv_check": "1.92.0 (ded5c06cf 2025-12-08)",
            "burn": "0.21.0",
            "profile": "release for the gate and operation-level binaries; debug for the mutation checks",
            "python": "3.11.9 (system python3), torch 2.10.0, transformers 5.2.0, numpy 2.4.3",
        },
        "host": "Apple M3 Max, 96 GiB unified memory, macOS 27.0.1 (26A434), Metal; shared with desktop applications",
        "preconditions": evidence("preconditions/preconditions.log", "on /Volumes/T9 ", "on /Volumes/bag of holding "),
        "harness": ("gates/harness/run_review.py, a copy of the follow-ups round's rerun runner: each test in its own "
                    "process from the prebuilt release binary (BINARY [--ignored] --exact --nocapture --test-threads 1 "
                    "TEST) under /usr/bin/time -l and the earlier rounds' memory watchdog, from the clean worktree at "
                    "the code head. Each log opens with == source <commit> dirty=<n>, the binary and its SHA-256, the "
                    "test, every TRANSFORMERS_* variable, and the start time, load, and free memory, and ends with == "
                    "exit= and the watchdog's figures. Every group but the bundle-only one held "
                    "/Users/nover/repos/transformers-worktrees/.heavy-model.lock (gates/logs/driver.log); the torch "
                    "probe ran under the same lock."),
        "binaries": binaries,
        "review": [
            {"id": "benchmark-breach", "file": "acceptance/measurements/half-precision-envelope-20261018.json",
             "verdict": "confirmed, with one correction",
             "finding": ("Decision (a) makes re-baselining a maintainer decision if the benchmark rerun breaches the "
                         "approved 5%/10% thresholds, yet the envelope record's verdict was 'No regression' and its "
                         "awaiting list had no benchmark item. The comparison it made, head against the base binary, "
                         "shows only that the slowdown predates this branch. Against the approved September 10 "
                         "baseline the breach is confirmed in the baseline benchmark, for both binaries (all six "
                         "controlled cold samples exceed 1.10x). Correction: in the public benchmark the gate run's "
                         "single head sample (+12.2%) and the base binary's rerun median (+11.0%, 2 of 3 samples) "
                         "breach, but the head's three rerun samples are below the approved baseline (-16.1% to "
                         "-36.2%), so there the head's breach is not confirmed."),
             "resolution": ("benchmark_breach below and an awaiting_maintainer item; README.md, CHANGELOG.md, "
                            "RELEASE_GATES.md, and ARCHITECTURE_COMPATIBILITY.md (whose Qwen3.5 row repeated 'finds no "
                            "regression') now say the breach is not attributable to this branch and awaits the "
                            "maintainer. The envelope record stays byte-identical."),
             "commits": ["the record commit"]},
            {"id": "nemotron-docs", "file": "README.md", "verdict": "confirmed",
             "finding": ("README.md listed official base-model parity and NEMOTRON_H_ACCEPTANCE.md said the gate "
                         "passes on 9a9c730, although it fails at f9f78b5 (cached decision 22 at 0.998558 against "
                         "the 0.998635 floor) and the failure stands at 8181605."),
             "resolution": ("Both say the gate passed at 9a9c730 and fails at f9f78b5 (and so at 8181605), attributed "
                            "to the once-rounded Mamba-2 convolution SiLU, pending review of the floor rule, and link "
                            "the envelope record; NEMOTRON_H_ACCEPTANCE.md's comparison and remaining-gates sections "
                            "too."),
             "commits": [SOURCE]},
            {"id": "mixer-silu-tests", "file": "src/mamba2_tests.rs", "verdict": "confirmed",
             "finding": ("No test reached the Mamba-2 or Gated DeltaNet convolution SiLU call site; putting "
                         "activation::silu back in either mixer failed nothing."),
             "resolution": ("convolution_silu_is_rounded_once_in_half_precision in src/mamba2_tests.rs and "
                            "src/delta_net_tests.rs read the post-convolution activation exactly through the F32 "
                            "recurrent state (identity convolutions, exact unit factors), in BF16, F16 (equal to the "
                            "F32 SiLU rounded once, unequal to the composed graph on separating probes; BF16 2.8125 "
                            "gives 2.65625, not 2.640625) and F32 (bit-identical to activation::silu), in both scan "
                            "modes. Reverting each call site fails its test (mutations)."),
             "commits": ["f7b8d99b54cbe4e342c17e85f1b5a528267a1cac"]},
            {"id": "eager-rounding-points", "file": "src/attention/dispatch_tests.rs",
             "verdict": "confirmed; the test exposed a graph defect, fixed",
             "finding": ("Only the eager score scale and the fused output were pinned. The requested emulation "
                         "(Q K^T summed in F32 and rounded, the unrounded scale in F32, the mask and bias add in the "
                         "dtype, an F32 softmax rounded once, P V summed in F32 and rounded) matched Burn's eager "
                         "graph in BF16 but not in F16 on Flex: 10 of 120 probabilities and 45 of 96 outputs "
                         "differed, because Burn Flex 0.21's F16 GEMM keeps F16 partial sums."),
             "resolution": ("The graph, not a tolerance, was fixed: on inference Flex the eager graph forms its two "
                            "F16 products from F32 operands and rounds once (f32_accumulated_matmul). The new test "
                            "half_precision_eager_attention_rounds_like_transformers_eager compares probabilities "
                            "and outputs with the emulation bit for bit on grouped heads with a causal mask, a "
                            "fully masked row, and a bias, and the outputs with Transformers 5.2.0's own "
                            "eager_attention_forward (pinned bits); four variants show the grid detects each "
                            "alternative rounding. Metal's F16 products are F32 sums rounded once and Metal's "
                            "eager graph equals Transformers' eager on the grid (ignored Metal tests), so Metal is "
                            "unchanged."),
             "commits": ["90f6152df3e31c1867a7abb33ae366714eb1ed27", "1a60aeb975e296ab2e844d46d5eca24c8af4a691"]},
            {"id": "eager-scale-users", "file": "CHANGELOG.md", "verdict": "confirmed",
             "finding": ("muse_attention_precision selects the eager graph for F16 and BF16, so the eager-scale "
                         "change altered Muse-Glimmer's default half-precision numerics (128-wide heads); GPT-2 also "
                         "defaults to eager, where 64-wide heads make the scale an exact 1/8. CHANGELOG.md said 'the "
                         "fused default path is unaffected', and ARCHITECTURE_COMPATIBILITY.md attributed Muse's "
                         "f9f78b5 statistics to the SiLU change alone. The envelope record's not_run entry for "
                         "gpt-oss also says the eager graph is one 'which no real-checkpoint gate selects', which is "
                         "wrong for Muse-Glimmer-30B's official gate."),
             "resolution": ("CHANGELOG.md and RELEASE_GATES.md name Muse-Glimmer's half-precision default and GPT-2 "
                            "and GPT-NeoX (a no-op where the scale is a power of two) as eager-graph users; the Muse "
                            "rows attribute 0.999931849 (against 0.999943377 at c15f24e) to both changes, which no "
                            "control separates. The envelope record stays byte-identical; this entry corrects it."),
             "commits": [SOURCE]},
            {"id": "gemma-f16-refusal", "file": "tests/gemma_artifacts.rs", "verdict": "confirmed",
             "finding": ("The gate-level refusal of TRANSFORMERS_GEMMA_DTYPE=f16 and the dropped Metal/F16 profile "
                         "belong to stage S7, held until the maintainer confirms decision (d); it hid the measured "
                         "outcome of decision (b, F16): an uncalibrated envelope."),
             "resolution": ("load_bundle no longer refuses f16; an F16 run reads TRANSFORMERS_GEMMA_F16_OFFICIAL_DIRS "
                            "and _LONG_F16_OFFICIAL_DIRS and fails as uncalibrated before loading (all five Metal "
                            "runs here). The docs say Gemma F16 is uncalibrated (all official F16 graphs overflow) "
                            "and leave the refusal to decision (d); S7 can reinstate it."),
             "commits": ["e2f72797804396c688b2228ba6a3c6920a1a5639"]},
            {"id": "pytorch-scale-evidence", "file": "src/attention/dispatch.rs", "verdict": "confirmed",
             "finding": ("No committed log measured PyTorch's scalar multiply; probe_half.log measured only Burn "
                         "against an emulation labelled PyTorch."),
             "resolution": ("probes/probe_torch_half.log: on the CPU, x * (1/sqrt(128)) with a Python float equals "
                            "the F32 product rounded once on all 65,280 finite BF16 and 63,488 finite F16 values; on "
                            "MPS all F16 agree and 1,120 BF16 values differ, each an F32 product in the subnormal "
                            "range flushed to zero (1,118 would round to a BF16 subnormal, 2 to the smallest "
                            "normal). The docstring cites the log and names the convention as the CPU's."),
             "commits": ["90f6152df3e31c1867a7abb33ae366714eb1ed27"]},
        ],
        "benchmark_breach": {
            "policy": ("BENCHMARKS.md: performance changes above 5% alert; a greater-than-10% regression fails only "
                       "after a controlled rerun confirms it on the same benchmark series. The approved baseline is "
                       "the September 10 controlled measurements (v0_1-release-20260910.json, "
                       "user_authorized_rebaseline)."),
            "decision_source": {"record": "half-precision-followups-20261017.json", "decision": "a",
                                "quote": ("If the benchmark rerun breaches the approved 5%/10% thresholds, "
                                          "re-baselining becomes a maintainer decision.")},
            "script": "benchmark/breach.py",
            **evidence("benchmark/breach.log",
                       "== baseline benchmark: approved cold first token",
                       "   gate run (f9f78b5): 0.432347",
                       "   controlled rerun base (7bd057e): 0.7597",
                       "   controlled rerun head (f9f78b5): 0.5278",
                       "   controlled rerun, both binaries: 6 of 6",
                       "== public benchmark: approved cold first token",
                       "   gate run (f9f78b5): 0.410578",
                       "   controlled rerun base (7bd057e): 0.4446",
                       "   controlled rerun head (f9f78b5): 0.3021",
                       "   controlled rerun, both binaries: 2 of 6"),
            "measured": ("Cold first token (a single sample per run, including lazy kernel compilation) against the "
                         "approved baseline: baseline benchmark 0.167520 s, gate run 0.432 s (+158.1%), controlled "
                         "rerun base 7bd057e 0.322 to 0.760 s and head f9f78b5 0.228 to 0.528 s, all six above 1.10x; "
                         "public benchmark 0.365949 s, gate run 0.411 s (+12.2%), base 0.294 to 0.445 s (median "
                         "+11.0%, 2 of 3 above 1.10x), head 0.234 to 0.307 s (all below the baseline). Every warm and "
                         "natural-loop median is faster than the baseline and every token and prediction hash is "
                         "unchanged (envelope record)."),
            "verdict": ("The cold first token breaches the approved baseline in both benchmarks' gate runs; the "
                        "controlled rerun confirms the breach in the baseline benchmark for both binaries, and in the "
                        "public benchmark for the base binary, not the head. The base binary 7bd057e shows it too, so "
                        "the slowdown predates this branch and is not attributable to it (the head's medians are "
                        "below the base's, ratios 0.813 and 0.743). Whether to re-baseline is a maintainer decision; "
                        "nothing is re-baselined or waived here."),
            "supersedes": ("the envelope record's benchmarks.verdict ('No regression'), which judged the head against "
                           "the base binary only; that record stays byte-identical"),
        },
        "awaiting_maintainer": [
            {"decision": "a (benchmark)", "item": (
                "Re-baselining Qwen3.5-0.8B's cold first token: it breaches the approved September 10 baseline in "
                "both benchmarks, in the base binary 7bd057e as well as the head (benchmark_breach); the slowdown "
                "predates this branch. Until then the breach stands and no baseline changes.")},
            {"decision": "d", "item": (
                "The Gemma 3 F16 policy: a library-level default refusal with an explicit opt-in (a family-declared "
                "range hazard checked by a generic LoadOptions policy), and the gate changes the decision lists "
                "(refuse TRANSFORMERS_GEMMA_DTYPE=f16, drop a Metal/F16 Gemma profile). This round withdrew the "
                "gate-level refusal; until the decision, Gemma f16 runs fail as uncalibrated.")},
            {"decision": "d-metal-bf16", "item": (
                "A Burn >= 0.22 upgrade (MSRV 1.92 -> at least 1.95, the backend-erased Tensor API), the only route "
                "found to Metal BF16; Burn 0.21 keeps the contextual refusal.")},
            {"decision": None, "item": (
                "A reviewed rule for the Nemotron-H official-precision floors (each path's worst official F16 row), "
                "which the once-rounded Mamba-2 SiLU misses at cached decision 22; nothing was tuned.")},
            {"decision": None, "item": "A non-finite guard in generation, and a post-attention-add hook (v0.2 interface)."},
            {"decision": None, "item": "Gemma 3 1B/270m and Gemma 2 checkpoints for F16 measurements."},
        ],
        "runs": runs,
        "tallies": tallies,
        "mutations": {
            "script": "mutations/run_mutations.py",
            "clone": f"a scratch Git clone of the worktree at {SOURCE[:7]}, CARGO_TARGET_DIR=/Volumes/bag of holding/cargo-targets/tfm-half-followups2-mutate",
            "runs": mutations,
        },
        "probes": {
            "torch": {"script": "probes/probe_torch_half.py",
                      **evidence("probes/probe_torch_half.log",
                                 "   cpu torch.bfloat16: 0 of 65280",
                                 "   mps torch.bfloat16: 1120 of 65280",
                                 "   cpu torch.float16: 0 of 63488",
                                 "   mps torch.float16: 0 of 63488",
                                 "would move 1508", "would move 8722",
                                 "   cpu torch.float16: 1 + 4096",
                                 "   mps torch.bfloat16: 1 + 4096",
                                 "torch.bfloat16: eager_attention_forward against the emulation",
                                 "torch.float16: eager_attention_forward against the emulation"),
                      "note": ("Run at 90f6152 from the worktree under the heavy-model lock; it reads no crate code. "
                               "Its output bits are pinned in src/attention/dispatch_tests.rs "
                               "(TRANSFORMERS_EAGER_BF16, TRANSFORMERS_EAGER_F16).")},
        },
        "findings": [
            {"id": "eager-f16-flex-partial-sums", "status": "fixed (90f6152)",
             "summary": ("On inference Flex the eager graph's F16 products are F32 sums rounded once; before, Flex's "
                         "F16 GEMM moved 10 of 120 probabilities and 45 of 96 outputs of the emulation grid. No gate "
                         "profile uses F16 on Flex, so no gate changes."),
             "evidence": [{"log": "gates/logs/operations.eager-emulation.log",
                           "quotes": [line(text("gates/logs/operations.eager-emulation.log"), "test result")]}]},
            {"id": "metal-f16-matmul", "status": "measured",
             "summary": "Burn Metal's F16 products are F32 sums rounded once (the eager graph keeps them).",
             "evidence": [evidence("gates/logs/operations.metal-f16-matmul.log", "Metal F16 1 + 4096",
                                   "Metal F16 [16, 256]")]},
            {"id": "metal-eager-matches-transformers", "status": "measured on the grid",
             "summary": ("Metal's F16 eager graph, Muse-Glimmer's half-precision default, equals Transformers' "
                         "eager_attention_forward bit for bit on every supported output of the emulation grid."),
             "evidence": [evidence("gates/logs/operations.metal-eager.log", "Metal F16 eager attention")]},
            {"id": "pytorch-products", "status": "measured",
             "summary": ("PyTorch 2.10.0 sums F16 and BF16 matrix products in F32 and rounds once on the CPU and MPS "
                         "(the probe's long sum and grid product); its CPU scalar multiply is the F32 product rounded "
                         "once; MPS flushes BF16 subnormal F32 products to zero."),
             "evidence": [evidence("probes/probe_torch_half.log", "   cpu torch.bfloat16: 1 + 4096",
                                   "   mps torch.float16: 1 + 4096")]},
            {"id": "eager-attention-excess", "status": "open",
             "summary": ("The envelope round's finding (Qwen3-4B Flex/BF16 eager cached decode mean at 1.565x its "
                         "bound) now has an operation-level test behind it: Flex/BF16 eager attention rounds where "
                         "Transformers' eager attention does and matches it bit for bit on the grid, so the excess is "
                         "not a rounding-point mismatch of the attention operation at those shapes. Flex/BF16 is "
                         "unchanged by this round and the diagnosis was not rerun."),
             "evidence": [evidence("gates/logs/operations.eager-emulation.log", "test result")]},
            {"id": "burn-flex-pins", "status": "open upstream",
             "summary": ("Flex's F16 GEMM still keeps F16 partial sums outside the eager graph's two products "
                         "(projections and every other F16 product on Flex; no gate profile uses F16 on Flex), its "
                         "F32-to-BF16 cast ignores the three lowest F32 bits, and tanh GELU overflows for BF16 inputs "
                         "of at least 2^127; the pin tests are unchanged."),
             "evidence": [evidence("gates/logs/operations.flex-f16-matmul.log", "test result")]},
            {"id": "envelope-record-corrections", "status": "recorded here; the envelope record stays byte-identical",
             "summary": ("Two statements of the envelope record are wrong: benchmarks.verdict 'No regression' "
                         "(benchmark_breach), and not_run's 'the eager-scale change reaches only the eager graph ..., "
                         "which no real-checkpoint gate selects': Muse-Glimmer-30B's official gate selects it by "
                         "default in F16, so its f9f78b5 statistics reflect both the SiLU and the eager-scale "
                         "changes. Its decision (d) entry lists a gate-level refusal as implemented that anticipated "
                         "the maintainer; this round withdrew it."),
             "evidence": []},
            {"id": "unaudited-eager-variants", "status": "open",
             "summary": ("Multi-head latent attention (src/mla.rs) forms its own scores and value products, and "
                         "other sites multiply by scalars rounded to the tensor dtype; neither was audited against "
                         "the reference's rounding points in this round."),
             "evidence": []},
        ],
        "not_run": [
            {"gates": "Llama-family gates (F32, Flex/BF16, Metal/F16) and their bundle-only tests",
             "reason": ("no change of this round reaches them: they use the fused attention graph in every dtype, the "
                        "eager-product change takes effect only for F16 on inference Flex, and their test binary's "
                        "sources are unchanged; their last runs are at 8181605 (follow-ups record)")},
            {"gates": "Muse-Glimmer-30B, Qwen3.8-27B, Qwen3.5-0.8B parity and memory, gpt-oss, DeepSeek-V2-Lite, "
                      "Qwen3-30B-A3B, and FLAN-T5 gates, smokes, and audits",
             "reason": ("they run on Metal or in F32/BF16 on Flex, where the eager graph keeps the backend's own "
                        "products; the ignored Metal tests show Metal's F16 products are F32 sums and its eager "
                        "graph equals Transformers' eager on the grid (inferred unchanged, not rerun)")},
            {"gates": "the Nemotron-H Metal smoke and official-precision gate",
             "reason": "this round corrects only its documentation; the failure measured at f9f78b5 stands"},
            {"gates": "the Qwen3.5-0.8B benchmarks",
             "reason": ("the breach concerns the envelope round's measurements of the approved baseline; a new "
                        "baseline is the maintainer's decision")},
            {"gates": "Gemma F16 on Flex",
             "reason": "not a gate profile; it would fail as uncalibrated exactly as the Metal runs did"},
            {"gates": "the library-level Gemma 3 F16 policy (plan S7)", "reason": "decision (d) awaits the maintainer"},
            {"gates": "CUDA and ROCm builds and every CUDA gate", "reason": "no CUDA or ROCm host"},
            {"gates": "coverage qualification (95% shared/core, 90% crate-wide)", "reason": "not run in this round"},
            {"gates": "python3 scripts/check_release.py --gate package",
             "reason": "builds into the checkout's target directory on the local disk; not run"},
            {"gates": "F16 measurements of Gemma 3 1B/270m and Gemma 2", "reason": "no local checkpoints; downloads are not allowed"},
        ],
        "source_validation": {
            "commit": SOURCE,
            "tree": ("a clean Git clone of the worktree checked out at the code head, CARGO_TARGET_DIR=/Volumes/bag "
                     "of holding/cargo-targets/tfm-half-followups2"),
            "script": "validation/source_gates.sh",
            "summary": evidence("validation/summary.log", "== source", "release-source exit="),
            "checks": checks,
            "passed": all(check["exit"] == 0 for check in checks),
        },
        "historical_records": {name: sha256((measurements / name).read_bytes()) for name in HISTORICAL},
        "campaign_directory": CAMPAIGN,
    }
    files = sorted(p for p in campaign.rglob("*") if p.is_file())
    record["files"] = {p.relative_to(campaign).as_posix(): sha256(p.read_bytes()) for p in files}
    (measurements / RECORD).write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {RECORD}: {tallies}, {len(mutations)} mutation logs, {len(checks)} checks, {len(files)} files")


if __name__ == "__main__":
    main()
