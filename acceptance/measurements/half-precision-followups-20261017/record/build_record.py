"""Scratch: builds acceptance/measurements/half-precision-followups-20261017.json from its committed campaign files.

Usage: python3 -I build_record.py CRATE_ROOT

Every run entry, quote, digest, comparison, and tally is derived from the
files already copied into the campaign directory and from the committed
envelope and Mac validation records; the narrative fields (notes, findings,
not_run) are written here by hand and cite those files.
"""

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(sys.argv[1]).resolve()
MEASUREMENTS = ROOT / "acceptance/measurements"
NAME = "half-precision-followups-20261017"
CAMPAIGN = MEASUREMENTS / NAME
RERUN = f"{NAME}/rerun"
LOGS = f"{RERUN}/logs"
ENVELOPE_NAME = "half-precision-envelope-20261018"
ENVELOPE = json.loads((MEASUREMENTS / f"{ENVELOPE_NAME}.json").read_text())
MAC = json.loads((MEASUREMENTS / "mac-validation-20261017.json").read_text())
HEAD = "818160545802c1374f7607d26ce20cc86d2d758d"
CHECKOUT = "/Users/nover/repos/transformers-worktrees/tfm-half-followups2"
BRANCH = "claude/tfm-half-followups2"
MASK_THREAD = re.compile(r"thread '([^']*)' \(\d+\)")
MASK_TIME = re.compile(r"finished in [\d.]+s")
#: The runner's real-checkpoint groups, in run order.
CHECKPOINT_GROUPS = ("SmolLM-135M", "Qwen3-4B", "SmolLM3-3B", "Llama-3.1-8B", "Qwen2-7B-f64", "gemma")
EXPECTED = {
    "uncalibrated: official cpu-eager float16 non-finite at prompt 0 position 1": "b-f16",
    "TRANSFORMERS_GEMMA_DTYPE=f16 is not a gate profile": "d",
    "backend cannot store checkpoint dtype BF16": "d-metal-bf16",
}


def sha256(data):
    """Hexadecimal SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def read(relative):
    """Text of a file named relative to acceptance/measurements."""
    return (MEASUREMENTS / relative).read_text(errors="replace")


def body(text):
    """Test output from `running 1 test` to `test result`, thread ids and timings masked (as compare_runs.py)."""
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith("running "):
            inside = True
        if inside:
            line = MASK_THREAD.sub(r"thread '\1' (TID)", line)
            line = MASK_TIME.sub("finished in Xs", line)
            lines.append(line)
        if line.startswith("test result"):
            break
    return lines


def exit_status(text):
    """The `== exit=` status of a run log."""
    return int(re.search(r"(?m)^== exit=(-?\d+)", text).group(1))


def stat_lines(lines):
    """Statistic lines (`|Δ|`) without the first line's `test NAME ... ` prefix."""
    return [re.sub(r"^test \S+ \.\.\. ", "", line) for line in lines if "|Δ|" in line]


def greedy_counts(lines):
    """(label, matched, total) of each greedy summary line, in either round's wording."""
    out = []
    for line in lines:
        line = re.sub(r"^test \S+ \.\.\. ", "", line)
        match = re.match(r"^((?:long )?greedy): (\d+) of (\d+) (?:greedy )?tokens", line)
        if match:
            out.append([match.group(1), int(match.group(2)), int(match.group(3))])
    return out


def quotes_of(text):
    """The measurement lines of a run log's test output, verbatim (prefixes and long load reports cut)."""
    out, inside = [], False
    for raw in text.splitlines():
        if raw.startswith("running "):
            inside = True
            continue
        if not inside:
            continue
        line = re.sub(r"^test \S+ \.\.\. ", "", raw)
        if raw.startswith("test result"):
            out.append(raw)
            break
        if "is a tie at tolerance" in line or re.match(r"^(official \S+ (BF16|F16): |decode truth: reference.json|/Volumes/\S+ \(Float)", line):
            continue
        if re.match(r"^prompt \d+ \(\d+ tokens\): official |^decode rows \((cached|stateless)\): official |^prompt \d+ decoded after a \d+-token prefill: official ", line):
            continue
        if "backend cannot store checkpoint dtype BF16" in line:
            start = line.index("strict load: ")
            end = line.index("dtype BF16", start) + len("dtype BF16")
            out.append(line[start:end])
            continue
        if ("|Δ|" in line or "bounds of" in line or "envelope max" in line or "leave-one-out" in line
                or re.match(r"^(long )?greedy: (\d+ of \d+|threshold)", line) or "uncalibrated:" in line
                or "is not a gate profile" in line or "panicked at" in line or "matrix products use" in line):
            out.append(line)
    return out


def files_digests():
    """SHA-256 of every file in the campaign directory, keyed relative to acceptance/measurements."""
    return {path.relative_to(MEASUREMENTS).as_posix(): sha256(path.read_bytes())
            for path in sorted(CAMPAIGN.rglob("*")) if path.is_file() and path.name != ".DS_Store"}


def group_of(name):
    """The runner group a run belonged to (run_rerun.py's plan)."""
    if name.startswith(("calibration.", "gemma.f16.refusal-unit.")):
        return "calibration"
    if name.startswith("refusal."):
        return "refusals"
    return next(group for group in CHECKPOINT_GROUPS if name.startswith(group + "."))


def role_of(kind):
    """What a run can establish."""
    if kind.startswith("bundles/"):
        return "bundle-only calibration (reads official bundles; no checkpoint, no Burn model; not a gate pass)"
    if kind == "unit":
        return "unit test (ordinary, not ignored; reads no artifact)"
    if "(refused)" in kind:
        return "refusal check of a profile the round did not keep (expected to fail before or at load)"
    return "gate"


def envelope_runs():
    """The envelope record's runs by name."""
    return {run["name"]: run for run in ENVELOPE["runs"]}


def mac_runs():
    """The Mac validation record's runs by log path."""
    return {run["log"]: run for run in MAC["runs"]}


def pre_silu_log(name):
    """The Mac validation (pre-SiLU) log of an F32 run, relative to acceptance/measurements, or None."""
    if ".f32." not in name:
        return None
    if name.startswith("gemma."):
        return f"mac-validation-20261017/gemma/c2936fe/{name.replace('gemma.', 'gemma-3-4b-it.', 1)}.log"
    return f"mac-validation-20261017/llama-family/{name}.log"


def build_runs():
    """One entry per process in runs.jsonl, with its comparisons."""
    jsonl = [json.loads(line) for line in read(f"{LOGS}/runs.jsonl").splitlines()]
    envelope, mac = envelope_runs(), mac_runs()
    runs = []
    for entry in jsonl:
        name = entry["name"]
        log = f"{LOGS}/{name}.log"
        text = read(log)
        kind_test = re.search(r"(?m)^== (\S+(?: \S+)?(?: \(refused\))?) (\S+)\n== (?:env|cwd)", text)
        binary = re.search(r"(?m)^== binary (.+) sha256 ([0-9a-f]{64})$", text)
        env = dict(re.findall(r"(?m)^== env ([A-Z0-9_]+)=(.*)$", text))
        start = re.search(r"(?m)^== start (\S+) ", text).group(1)
        wall = float(re.search(r"(?m)^== exit=-?\d+ wall=([\d.]+)s ", text).group(1))
        peak = int(re.findall(r"(?m)^\s*(\d+)  peak memory footprint$", text)[-1])
        rss = int(re.findall(r"(?m)^\s*(\d+)  maximum resident set size$", text)[-1])
        code = exit_status(text)
        passed = "\ntest result: ok. 1 passed; 0 failed;" in text
        expected = [message for message in EXPECTED if message in text]
        if passed and code == 0:
            status = "pass"
        elif code == 101 and len(expected) == 1 and "\ntest result: FAILED. 0 passed; 1 failed;" in text:
            status = "expected_fail"
        else:
            status = "fail"
        assert entry["kind"] == kind_test.group(1) and entry["test"] == kind_test.group(2), name
        assert entry["exit"] == code and entry["wall_seconds"] == wall and entry["peak_memory_footprint_bytes"] == peak, name
        assert entry["log_sha256"] == sha256((MEASUREMENTS / log).read_bytes()), name
        run = {
            "name": name,
            "group": group_of(name),
            "role": role_of(entry["kind"]),
            "kind": entry["kind"],
            "test": entry["test"],
            "ignored": " --ignored " in re.search(r"(?m)^== cmd (.*)$", text).group(1) + " ",
            "binary_sha256": binary.group(2),
            "env": env,
            "start_utc": start,
            "exit": code,
            "status": status,
            "wall_seconds": wall,
            "peak_memory_footprint_bytes": peak,
            "maximum_resident_set_bytes": rss,
            "log": log,
            "log_sha256": entry["log_sha256"],
            "quotes": quotes_of(text),
        }
        if status == "expected_fail":
            run["expected_failure"] = expected[0]
            run["decision"] = EXPECTED[expected[0]]
        old = envelope.get(name)
        if old is not None:
            old_log = f"{ENVELOPE_NAME}/{old['log']}"
            old_text = read(old_log)
            same = body(old_text) == body(text) and exit_status(old_text) == code
            assert same, name
            run["f9f78b5"] = {
                "record": f"{ENVELOPE_NAME}.json",
                "log": old_log,
                "status": old["status"],
                "identical_output": same,
                "output_lines": len(body(text)),
                "peak_memory_footprint_bytes": int(re.findall(r"(?m)^\s*(\d+)  peak memory footprint$", old_text)[-1]),
            }
        else:
            run["f9f78b5"] = None
        prior = pre_silu_log(name)
        if prior is not None:
            old_text = read(prior)
            old_stats, new_stats = stat_lines(body(old_text)), stat_lines(body(text))
            assert all(line in new_stats for line in old_stats), name
            assert greedy_counts(body(old_text)) == greedy_counts(body(text)), name
            assert exit_status(old_text) == code
            run["pre_silu"] = {
                "record": "mac-validation-20261017.json",
                "log": prior,
                "source_commit": mac[prior]["source_commit"],
                "statistic_lines_repeated": len(old_stats),
                "greedy": greedy_counts(body(text)),
                "new_statistic_lines": [line for line in new_stats if line not in old_stats],
            }
        runs.append(run)
    return runs


def tallies(runs):
    """Status counts per group and overall."""
    out = {}
    for run in runs:
        group = out.setdefault(run["group"], {"pass": 0, "expected_fail": 0, "fail": 0})
        group[run["status"]] += 1
    total = {status: sum(group[status] for group in out.values()) for status in ("pass", "expected_fail", "fail")}
    return {"groups": out, "total": total}


def bundle_table(runs):
    """Every bundle the runs named, with its manifest digest and the earlier record that pins it."""
    identities = json.loads(read(f"{NAME}/identities/input-identities.json"))
    envelope_digests = json.loads(read(f"{ENVELOPE_NAME}/bundles/bundle-digests.json"))
    round2 = {entry["bundle"]: entry for entry in json.loads(read(f"{ENVELOPE_NAME}/bundles/round2-bundle-digests.json"))}
    mac = {entry["path"]: entry for entry in MAC["bundles"]}
    rows = []
    for path, entry in sorted(identities["bundles"].items()):
        name = path.rsplit("/", 1)[1]
        if name in envelope_digests:
            pinned = {"record": f"{ENVELOPE_NAME}.json", "file": f"{ENVELOPE_NAME}/bundles/bundle-digests.json"}
            assert envelope_digests[name]["reference.json"]["sha256"] == entry["reference_json_sha256"]
            role = "official graph" if re.search(r"-(bf16|f16)-(cpu|mps)-(eager|sdpa)-7a50e44$", name) else "decode truth (F64, generated this round)"
        elif path in round2:
            pinned = {"record": f"{ENVELOPE_NAME}.json", "file": f"{ENVELOPE_NAME}/bundles/round2-bundle-digests.json"}
            assert round2[path]["reference_json_sha256"] == entry["reference_json_sha256"]
            role = "decode truth (round-2 F32 CPU-eager bundle)"
        else:
            pinned = {"record": "mac-validation-20261017.json", "file": None}
            assert mac[path]["manifest_sha256"] == entry["reference_json_sha256"]
            role = "truth"
        rows.append({
            "path": path,
            "role": role,
            "dtype": entry["dtype"],
            "device": entry["device"],
            "attention_implementation": entry["attention_implementation"],
            "revision": entry["revision"],
            "config_sha256": entry["config_sha256"],
            "script_sha256": entry["script_sha256"],
            "reference_json_sha256": entry["reference_json_sha256"],
            "pinned_by": pinned,
        })
    named = set()
    for run in runs:
        for key, value in run["env"].items():
            if key.endswith(("_REFERENCE_DIR", "_DECODE_TRUTH_DIR")):
                named.add(value)
            elif key.endswith("_OFFICIAL_DIRS"):
                named.update(value.split(":"))
    assert named == {row["path"] for row in rows}
    return rows


def checkpoint_table():
    """Every checkpoint directory the runs named, tied to the Mac validation record's identities."""
    identities = json.loads(read(f"{NAME}/identities/input-identities.json"))
    mac = {entry["path"]: entry for entry in MAC["checkpoints"]}
    rows = []
    for path, entry in sorted(identities["checkpoints"].items()):
        pinned = mac[path]
        assert pinned["config_sha256"] == entry["config_sha256"]
        rows.append({"key": pinned["key"], "repository": pinned["repository"], "revision": pinned["revision"],
                     "path": path, "config_sha256": entry["config_sha256"], "pinned_by": "mac-validation-20261017.json"})
    return rows


def main():
    """Writes the record."""
    runs = build_runs()
    tally = tallies(runs)
    counterpart = [run for run in runs if run["f9f78b5"]]
    peaks = [(run["peak_memory_footprint_bytes"] / run["f9f78b5"]["peak_memory_footprint_bytes"] - 1, run["name"])
             for run in counterpart]
    gate_peaks = [(change, name) for change, name in peaks
                  if next(run for run in counterpart if run["name"] == name)["role"] == "gate"]
    largest = max(runs, key=lambda run: run["peak_memory_footprint_bytes"])
    decide = json.loads(read(f"{NAME}/stage-reports/decide.json"))
    ids = ["a", "b", "b-f16", "c", "d", "d-metal-bf16"]
    implementation = {
        "a": {"status": "implemented", "commits": ["c10974ef9b0df07c8fa669e74e3a46349dd0381d", "1b1dd748a55e4a50f8a92a5832ccd919c4b4eeea"],
              "evidence": "Landed after the envelope harness (78978b6), so no head judged it under the 1.2/0.5 rules. The envelope record holds the Qwen3.5-0.8B requalification (parity, benchmarks with a controlled rerun, memory) and the Nemotron-H failure attributed to the Mamba-2 convolution SiLU; this rerun shows every printed F32 statistic of the pre-SiLU round unchanged (comparisons.pre_silu_f32)."},
        "b": {"status": "implemented", "commits": ["c2c7165a79f33bcbcc8503e174bb83f57855d8b2", "7a50e44fb6f952c80d546c6eb8ab3a87e1830825", "78978b68a50186bfee3137cb4e828aa0fcc4500d", "edf4cf0f0e3295092bbd6c59c4b6c15f6df229a0", "3cf845fc970e6b2af362da64c5671e44b003c659"],
              "evidence": "R4 with the derived slacks BF16 1.5/1.3 (envelope record, derivation); the operation-level rounding-point tests (3cf845f) found and fixed the eager attention scale (edf4cf0). This rerun's bundle-only leave-one-out tests pass for every calibrated truth at 8181605."},
        "b-f16": {"status": "implemented", "commits": ["7a50e44fb6f952c80d546c6eb8ab3a87e1830825", "78978b68a50186bfee3137cb4e828aa0fcc4500d"],
                  "evidence": "F16 slacks 1.7/1.7 from the official F16 graphs (envelope record); Qwen2-7B fails as uncalibrated in F16 here too (3 gates, 2 calibration tests), and gemma-3-4b-it has no F16 envelope."},
        "c": {"status": "implemented", "commits": ["c2c7165a79f33bcbcc8503e174bb83f57855d8b2", "7a50e44fb6f952c80d546c6eb8ab3a87e1830825", "78978b68a50186bfee3137cb4e828aa0fcc4500d"],
              "evidence": "Truth-anchored decode rows judge every half-precision cached greedy gate of this rerun; the F32 runs print their new decode-truth rows (pre_silu.new_statistic_lines); 0.5 remains only as the provisional tolerance of cached prefills, batched rows, and Gemma's long chunk ends."},
        "d": {"status": "least invasive variant only (gate-level refusal); the library-level policy awaits the maintainer",
              "commits": ["78978b68a50186bfee3137cb4e828aa0fcc4500d", "490d949fae8d9855ee0834da8063bcd4ac6479db"],
              "evidence": "The gates refuse TRANSFORMERS_GEMMA_DTYPE=f16 (all five Metal Gemma gates in this rerun); the plan's S7 (RangeHazardPolicy in LoadOptions, a family-declared hazard, and its tests) was not implemented, because the decision is flagged for user review. The library still evaluates the faithful, overflowing F16 graph."},
        "d-metal-bf16": {"status": "kept the existing contextual refusal; no CubeCL patch and no Burn upgrade", "commits": [],
                         "evidence": "This rerun measured the refusal in the gates themselves: SmolLM-135M and gemma-3-4b-it Metal BF16 fail at load at model.embed_tokens.weight. The Burn >= 0.22 upgrade awaits the maintainer."},
    }
    decisions = []
    for identifier, decision in zip(ids, decide["decisions"], strict=True):
        decisions.append({"id": identifier, **decision, "implementation": implementation[identifier]})

    def run_quote(name, fragment):
        """A highlight: the run's single log line holding ``fragment``, verbatim."""
        run = next(run for run in runs if run["name"] == name)
        text = read(run["log"])
        line = next(line for line in text.splitlines() if fragment in line)
        return {"run": name, "status": run["status"], "quote": re.sub(r"^test \S+ \.\.\. ", "", line)}

    highlights = [
        dict(run_quote("gemma.bf16.flex.cached_greedy_matches_the_reference", "cached decode (cached rows)"),
             note="the nearest margin of every half-precision gate (0.9447 of the maximum bound), unchanged from f9f78b5"),
        dict(run_quote("SmolLM-135M.f16.metal.prompt_logits_match_the_reference", "prompt 0: 5 tokens"),
             note="the nearest Llama-family margin (mean at 0.9335 of its bound), unchanged"),
        dict(run_quote("SmolLM3-3B.bf16.flex.prompt_logits_match_the_reference", "prompt 2: 8 tokens"),
             note="the nearest Llama-family Flex/BF16 mean (0.7881 of its bound)"),
        dict(run_quote("SmolLM-135M.bf16.flex.prompt_logits_match_the_reference", "prompt 2: 8 tokens"),
             note="the nearest Llama-family Flex/BF16 maximum (0.7879 of its bound)"),
        dict(run_quote("Llama-3.1-8B.bf16.flex.cached_greedy_matches_the_reference", "decoded after a 3-token prefill: 14 tokens"),
             note="the quarter-prefill decoded prompt against the official graphs' own cached decode"),
        dict(run_quote("Qwen3-4B.bf16.flex.cached_greedy_matches_the_reference", "cached decode (cached rows)"),
             note="truth-anchored decode rows of the default fused graph, within the envelope"),
        dict(run_quote("gemma.bf16.flex.long_context_logits_cross_the_sliding_window", "unwindowed: max"),
             note="negative control: the run without the sliding window must fail the rule, and does"),
        dict(run_quote("Qwen3-4B.bf16.flex.cached_greedy_matches_the_reference", "greedy: 26 of 32"),
             note="decisive greedy steps under the envelope tolerance"),
        dict(run_quote("Qwen2-7B-f64.bf16.flex.cached_greedy_matches_the_reference", "greedy: 0 of 32"),
             note="no decisive greedy step; the cached test rests on the decode rows (MIN_GREEDY=0)"),
        dict(run_quote("Qwen2-7B-f64.f16.metal.prompt_logits_match_the_reference", "uncalibrated:"),
             note="decision (b, F16): no envelope where an official F16 graph is non-finite"),
        dict(run_quote("refusal.gemma.f16.metal.prompt_logits_match_the_reference", "is not a gate profile"),
             note="decision (d): the gate refuses F16 before loading"),
        dict(run_quote("refusal.SmolLM-135M.bf16.metal.prompt_logits_match_the_reference", "backend cannot store checkpoint dtype BF16"),
             note="decision (d, Metal BF16): the contextual refusal of Burn 0.21, measured in the gate"),
        dict(run_quote("Llama-3.1-8B.f32.metal.cached_greedy_matches_the_reference", "cached decode: 32 decode rows"),
             note="the F32 decode-truth row check added by decision (c), which has no pre-SiLU counterpart"),
    ]
    for highlight in highlights:
        line = highlight["quote"]
        if "backend cannot store" in line:
            highlight["quote"] = line[line.index("strict load: "):line.index("dtype BF16") + len("dtype BF16")]
    margins_text = read(f"{LOGS}/margins.log")
    nearest = margins_text[margins_text.index("== nearest 15"):].splitlines()[1:]
    margin_quotes = [line.strip()[:200] for line in nearest[:7]]
    compare_text = read(f"{LOGS}/compare.log")
    compare_tally = compare_text.strip().splitlines()[-1]
    record = {
        "schema_version": 1,
        "scope": "half_precision_followups_round_not_a_support_declaration",
        "date_utc": "2026-10-18",
        "name_note": "The record keeps the name the follow-ups round was planned under, dated by the round's start on 2026-10-17 (its decision measurements, recorded under the envelope record's prior/ directory). The implementation's gate runs (envelope record) and this rerun ran on 2026-10-18 UTC.",
        "advertised_supported": False,
        "passed": False,
        "rerun_passed": tally["total"]["fail"] == 0,
        "passed_note": "The rerun at 8181605 has no unexpected failure (rerun_passed). The round as a whole does not pass: the Nemotron-H official-precision gate failed at f9f78b5 (failures), was not rerun here, and 8181605 compiles the same library code.",
        "source_commit": HEAD,
        "checkout": {"path": CHECKOUT, "branch": BRANCH, "state": "clean: every rerun log opens with `== source <HEAD> dirty=0` and closes with `== source after run <HEAD> dirty=0`, and the runner stops on any change"},
        "head_note": "8181605 differs from the envelope round's gate head f9f78b5 only by a /// comment on a cfg(test)-only field (1816d73), Markdown documentation (490d949), and the envelope record with its campaign directory and Python test (8181605). The rerun's binaries, compiled in the worktree itself at 8181605, are byte-identical to the f9f78b5 gate binaries (build). The commit that adds this record changes only this record, its campaign directory, its Python test, and Markdown documentation.",
        "sources": [
            {"commit": "7bd057ecc56d60df2776639befd30058720d3c09", "role": "base", "description": "claude/tfm-half-followups2 at the start of the round, a copy of claude/tfm-polish with the shared-RNG test sweep"},
            {"commit": "c2c7165a79f33bcbcc8503e174bb83f57855d8b2", "role": "implementation", "description": "C1: the round-2 generator patch (generator SHA-256 58ab9f55...), so the 35 round-2 bundles name committed bytes"},
            {"commit": "7a50e44fb6f952c80d546c6eb8ab3a87e1830825", "role": "implementation", "description": "C2: float16 references, quarter-prefill decode rows, non-finite results recorded as data, and the derivation script (generator SHA-256 a4ed0db4...)"},
            {"commit": "78978b68a50186bfee3137cb4e828aa0fcc4500d", "role": "implementation", "description": "C3: the official-graph envelope rule, truth-anchored decode rows, fail-closed four-graph pairing, and Gemma's F16 refusal in the gates"},
            {"commit": "c10974ef9b0df07c8fa669e74e3a46349dd0381d", "role": "implementation", "description": "C4: half-precision SiLU evaluated in F32 and rounded once at every F.silu site"},
            {"commit": "1b1dd748a55e4a50f8a92a5832ccd919c4b4eeea", "role": "documentation", "description": "C4b: the once-rounded SiLU convention in TRANSFORMER_INNOVATIONS.md"},
            {"commit": "edf4cf0f0e3295092bbd6c59c4b6c15f6df229a0", "role": "implementation", "description": "C5: eager half-precision attention scores multiplied by the unrounded F32 scale"},
            {"commit": "3cf845fc970e6b2af362da64c5671e44b003c659", "role": "tests", "description": "C6: operation-level rounding-point tests and the three pinned Burn Flex deviations"},
            {"commit": "f9f78b5fc7cca9d1a8afaf29f8f9df04fe7c3008", "role": "envelope round gate head", "description": "the source of every gate binary of the envelope record; its runs are this rerun's comparison"},
            {"commit": "1816d734e9f575b36c36fe47073f420230b0764a", "role": "documentation", "description": "a /// comment on a cfg(test)-only field; the envelope record's source checks ran here"},
            {"commit": "490d949fae8d9855ee0834da8063bcd4ac6479db", "role": "documentation", "description": "the official-envelope gate documentation"},
            {"commit": HEAD, "role": "rerun head", "description": "the envelope record; the integrated head every run of this record ran at, from the worktree itself"},
            {"commit": "53f167b9f18c201e4392c708133b988198f12116", "role": "comparison (pre-SiLU)", "description": "the Mac validation round's Llama-family F32 runs (mac-validation-20261017.json)"},
            {"commit": "c2936fecf26cca91cd4e4c6e1e7f74fb570aa9b5", "role": "comparison (pre-SiLU)", "description": "the Mac validation round's second Gemma pass (mac-validation-20261017.json)"},
        ],
        "cargo_lock_sha256": "9734d5bc2d0a0c6c2d6db45671d4dbddad46424eea878bff6254b06be54827e1",
        "toolchain": {
            "log": f"{LOGS}/preflight.log",
            "quotes": ["cargo 1.94.0-nightly (3861f60f6 2025-12-19)", "rustc 1.94.0-nightly (4f14395c3 2025-12-22)", "Python 3.11.9",
                       "== Cargo.lock sha256 9734d5bc2d0a0c6c2d6db45671d4dbddad46424eea878bff6254b06be54827e1"],
            "burn": "0.21.0",
            "profile": "release",
            "cargo_arguments": "--locked --release --features metal,tokenizers --test llama_family_artifacts --test gemma_artifacts --no-run",
            "cargo_target_dir": "/Volumes/bag of holding/cargo-targets/tfm-half-followups2-gates",
        },
        "host": "The envelope round's host, an Apple M3 Max (Mac15,11) with 96 GiB of unified memory, shared with desktop applications and other sessions' processes; at recording time sw_vers reports macOS 27.0.1 (26A434) and sysctl hw.memsize 103079215104 bytes. Local free space was 23 GiB at the preflight, above the 20 GiB AGENTS.md threshold; builds still used the envelope round's external CARGO_TARGET_DIR, so the rebuilt binaries could be compared with its binaries in place.",
        "preconditions": {
            "log": f"{LOGS}/preflight.log",
            "quotes": [
                "== source 818160545802c1374f7607d26ce20cc86d2d758d dirty=0",
                "/dev/disk5s1 on /Volumes/T9 (apfs, local, nodev, nosuid, journaled, noowners)",
                "/dev/disk7s1 on /Volumes/bag of holding (apfs, local, nodev, nosuid, journaled, noowners)",
                "/dev/disk3s1s1   460Gi    13Gi    23Gi    36%    484k  239M    0%   /",
                "== lock present: no",
            ],
            "note": "run_rerun.py checks both volumes before every group and the checkout's HEAD and cleanliness before and after every process; no run stopped on either check (driver.log ends with `done`).",
        },
        "harness": "Each test ran in its own process from the prebuilt release test binary (BINARY [--ignored] --exact --nocapture --test-threads 1 TEST) under /usr/bin/time -l, with the working directory the worktree checkout itself, through rerun/harness/run_rerun.py (a copy of the envelope round's run_gates.py that runs from the checkout, writes the checkout's HEAD and cleanliness at the start and end of every log, and adds the refusal group). Each log opens with == source, == checkout, == binary <path> sha256 <digest>, the profile and test, every TRANSFORMERS_* variable, the working directory and command, and the start time, load average, and free memory, and ends with == exit=, the watchdog's figures, and == source after run. compare_runs.py and margins.py read the logs only.",
        "lock": {
            "path": "/Users/nover/repos/transformers-worktrees/.heavy-model.lock",
            "policy": "mkdir, retried every 60 s, taken for each real-checkpoint group (SmolLM-135M, Qwen3-4B, SmolLM3-3B, Llama-3.1-8B, Qwen2-7B-f64, gemma, refusals) and released when the group ended; the calibration group (bundle-only tests and the ordinary unit test, no checkpoint) ran without it, as the plan says",
            "log": f"{LOGS}/driver.log",
            "quotes": ["2026-10-18T05:06:48+00:00 lock acquired (SmolLM-135M)", "2026-10-18T05:24:25+00:00 lock released (refusals)",
                       "2026-10-18T05:24:25+00:00 done calibration SmolLM-135M Qwen3-4B SmolLM3-3B Llama-3.1-8B Qwen2-7B-f64 gemma refusals"],
        },
        "watchdog": {
            "policy": ENVELOPE["watchdog"]["policy"],
            "outcome": "never fired: no log carries a watchdog= field; the lowest free memory any run saw is min_free_pct below, with no swap growth",
            "min_free_pct": min(int(re.search(r"min_free_pct=(\d+)", read(run["log"])).group(1)) for run in runs),
            "max_swap_growth_mib": max(int(re.search(r"max_swap_growth_mib=(\d+)", read(run["log"])).group(1)) for run in runs),
        },
        "build": {
            "noop": {"log": f"{LOGS}/build-0-noop.log",
                     "quotes": ["    Finished `release` profile [optimized] target(s) in 0.64s",
                                "== source 818160545802c1374f7607d26ce20cc86d2d758d dirty=0"]},
            "rebuild": {"log": f"{LOGS}/build.log",
                        "quotes": ["== touched src/lib.rs (mtime only); sha256 after 038a29f78cab81e6f8dab9488f91287bf96fcfa983a54be4ab597136143fa925; git blob b621be3ad4d840832770481fe8e918a33be60900 HEAD blob b621be3ad4d840832770481fe8e918a33be60900",
                                   "   Compiling transformers v0.1.0 (/Users/nover/repos/transformers-worktrees/tfm-half-followups2)",
                                   "    Finished `release` profile [optimized] target(s) in 1m 00s",
                                   "== source after build 818160545802c1374f7607d26ce20cc86d2d758d dirty=0"]},
            "note": "The first build compiled nothing: cargo judged the binaries in the shared CARGO_TARGET_DIR, built by the envelope round from a git-archive export of f9f78b5, fresh for the worktree. Copies of those binaries were kept (rerun/prior-binaries/SHA256SUMS lists their digests; the 106 MB copies stay in the scratchpad), src/lib.rs's modification time was updated without changing its bytes or Git blob, and cargo recompiled the crate from the worktree. The recompiled binaries are byte-identical to the copies and to the envelope record's binaries.",
        },
        "binaries": [
            {"target": "llama_family_artifacts", "path": "/Volumes/bag of holding/cargo-targets/tfm-half-followups2-gates/release/deps/llama_family_artifacts-693e11c29d0eb894",
             "sha256": "72a204e1c37262448b47e772e980a72cba33fc8ae1618af1385ad521009db738", "source_commit": HEAD, "built_from": CHECKOUT,
             "envelope_record_label": "A", "retained": True},
            {"target": "gemma_artifacts", "path": "/Volumes/bag of holding/cargo-targets/tfm-half-followups2-gates/release/deps/gemma_artifacts-7e9fb558e3797b0a",
             "sha256": "533d5d9c242bcc783dea609917c57471e30f92fd01b010316e1dde57993b2e17", "source_commit": HEAD, "built_from": CHECKOUT,
             "envelope_record_label": "A", "retained": True},
        ],
        "identities": {
            "file": f"{NAME}/identities/input-identities.json",
            "script": f"{NAME}/identities/digest_inputs.py",
            "note": "Written while recording, after the rerun, by reading the external volumes only: each bundle's reference.json (which names every logits file by SHA-256; the gates verify each file they read against it, and each checkpoint's config.json against the bundle's provenance), each checkpoint's config.json, and the two test binaries, which still hold the digests the logs recorded.",
        },
        "checkpoints": checkpoint_table(),
        "bundles": bundle_table(runs),
        "decisions": decisions,
        "decisions_source": {
            "file": f"{NAME}/stage-reports/decide.json",
            "note": "The decision stage's output as the workflow passed it to the implementation stage (workflow journal key v2:f60f2bdc...); the decisions above repeat its question, decision, rationale, alternatives_rejected, and user_review fields verbatim, and its implementation_plan stays in the file.",
        },
        "awaiting_maintainer": [
            {"decision": "d", "item": "A library-level Gemma 3 F16 policy: a default refusal of F16 loads of gemma3/gemma3_text with an explicit opt-in that runs the faithful, overflowing graph, as a family-declared range hazard checked by a generic LoadOptions policy, and its scope (family-wide or measured sizes) and API. Until then only the gates refuse f16."},
            {"decision": "d-metal-bf16", "item": "A Burn >= 0.22 upgrade (MSRV 1.92 -> at least 1.95, the backend-erased Tensor API), the only route found to Metal BF16; Burn 0.21 keeps the contextual refusal."},
            {"decision": None, "item": "A reviewed rule for the Nemotron-H official-precision floors (each path's worst official F16 row), which the once-rounded Mamba-2 SiLU now misses at cached decision 22; nothing was tuned."},
            {"decision": None, "item": "A non-finite guard in generation, and a post-attention-add hook (v0.2 interface)."},
            {"decision": None, "item": "Gemma 3 1B/270m and Gemma 2 checkpoints for F16 measurements."},
        ],
        "implementation": {
            "record": f"{ENVELOPE_NAME}.json",
            "record_sha256": sha256((MEASUREMENTS / f"{ENVELOPE_NAME}.json").read_bytes()),
            "stage_report": f"{NAME}/stage-reports/implement.json",
            "summary": "The implementation stage's own record, left byte-identical: the 57 official bundles by file digest, the leave-one-out derivation of the slacks, 119 single-process runs at f9f78b5 (113 pass, 5 expected_fail, 1 fail), the Qwen3.5-0.8B requalification, the larger checkpoints, diagnoses, probes, and the source checks at 1816d73.",
            "tallies": ENVELOPE["tallies"],
        },
        "runs": runs,
        "tallies": tally,
        "total_wall_seconds": round(sum(run["wall_seconds"] for run in runs), 1),
        "comparisons": {
            "f9f78b5": {
                "rule": "the whole test output, from `running 1 test` to `test result`, with thread ids and the `finished in` time masked, and the exit status, equal to the envelope record's run of the same name",
                "identical": len(counterpart),
                "without_counterpart": sorted(run["name"] for run in runs if not run["f9f78b5"]),
                "log": f"{LOGS}/compare.log",
                "quotes": [compare_tally],
                "peak_memory_change": {
                    "note": "each run's peak memory footprint over its envelope-round counterpart's, minus one",
                    "all": {"runs": len(peaks), "min": round(min(peaks)[0], 4), "min_run": min(peaks)[1],
                            "max": round(max(peaks)[0], 4), "max_run": max(peaks)[1]},
                    "gates": {"runs": len(gate_peaks), "min": round(min(gate_peaks)[0], 4), "min_run": min(gate_peaks)[1],
                              "max": round(max(gate_peaks)[0], 4), "max_run": max(gate_peaks)[1]},
                },
            },
            "pre_silu_f32": {
                "rule": "every statistic line (one containing |Δ|) of the Mac validation round's F32 log occurs verbatim in this run's output, the greedy counts are equal, and so is the exit status; lines new since then are listed per run (new_statistic_lines), not judged",
                "runs": sum(1 for run in runs if "pre_silu" in run),
                "caveat": "This compares figures at the logs' printed precision, not logits bit for bit, and the earlier F32 cached-greedy logs printed only token counts, so the F32 decode-truth rows added by decision (c) have no pre-SiLU counterpart.",
            },
        },
        "largest_peak": {"run": largest["name"], "peak_memory_footprint_bytes": largest["peak_memory_footprint_bytes"], "log": largest["log"],
                         "quotes": [f"{largest['peak_memory_footprint_bytes']}  peak memory footprint"]},
        "margins": {
            "log": f"{LOGS}/margins.log",
            "quotes": margin_quotes,
            "note": "margins.py prints, from the logs alone, each statistic's ratio to its bound; the two largest ratios are the Gemma long negative control (the run without the sliding window), which must exceed its bound.",
        },
        "highlights": highlights,
        "final_head_checks": {
            "source_commit": HEAD,
            "summary": {"log": f"{NAME}/final/summary.log",
                        "quotes": [f"head {HEAD} dirty=0", "python-tests exit=0", "doc-audit exit=0", "fmt exit=0"]},
            "checks": [
                {"name": "python-tests", "command": "python3 -m unittest discover -s scripts/tests", "log": f"{NAME}/final/python-tests.log",
                 "quotes": ["Ran 560 tests in 4.676s", "OK (skipped=14)"]},
                {"name": "doc-audit", "command": "python3 scripts/check_documentation.py --standards --classifier <the coverage-layout classifier>", "log": f"{NAME}/final/doc-audit.log",
                 "quotes": ["integration_tests: 0/3297 authored sites missing required line docs", "models: 0/8499 authored sites missing required line docs",
                            "shared: 0/7568 authored sites missing required line docs"]},
                {"name": "fmt", "command": "cargo fmt --check", "log": f"{NAME}/final/fmt.log", "bytes": 0},
            ],
            "note": "Run by the implementation stage on the worktree after its last commit (8181605); the logs carry no command lines, so the commands are that stage's account (stage-reports/implement.json). The other source checks ran at 1816d73 (envelope record, source_validation); no Rust source changed after 1816d73.",
        },
        "failures": [
            {"run": "nemotron-h.f16.metal.official-precision", "record": f"{ENVELOPE_NAME}.json", "source_commit": "f9f78b5fc7cca9d1a8afaf29f8f9df04fe7c3008",
             "status": "fail", "rerun": False, "log": f"{ENVELOPE_NAME}/gates/logs/nemotron-h.f16.metal.official-precision.log",
             "quotes": ["Nemotron recurrent cached row cosine 0.9985579844887273 against the F32 truth is below the official F16 floor 0.9986350472017235 at decision 22"],
             "applies_to_head": "yes: 8181605 compiles the same library code as f9f78b5, and the rerun did not run this gate",
             "finding": "nemotron-floor-failure"},
        ],
        "findings": [
            {"id": "integrated-head-reproduces", "status": "measured",
             "summary": "Every run with an envelope-round counterpart (99 of 106) reproduces it exactly at the integrated head: the same test output once thread ids and timings are masked, the same exit status, from binaries rebuilt in the worktree that are byte-identical to the f9f78b5 gate binaries. The other 7 runs are the refusal checks, which the envelope round did not run.",
             "evidence": [{"log": f"{LOGS}/compare.log", "quotes": [compare_tally]}]},
            {"id": "f32-unchanged-by-silu", "status": "measured at printed precision",
             "summary": "All 40 F32 gates (Llama family and gemma-3-4b-it, Flex and Metal) pass and repeat every printed |Δ| statistic line and every greedy count of the last pre-SiLU round (Llama family at 53f167b, Gemma at c2936fe) verbatim, as decision (a) predicts for F32 (bit-identical SiLU). The F32 decode-truth rows are new and have no earlier counterpart.",
             "evidence": [{"log": f"{LOGS}/compare.log", "quotes": ["3 statistic lines all present verbatim; greedy then [], now []; exit then 0 -> EQUAL",
                                                                    "11 statistic lines all present verbatim"]}]},
            {"id": "gemma-metal-f16-refused", "status": "measured (decision d, gate level)",
             "summary": "All five Metal gemma-3-4b-it gates with TRANSFORMERS_GEMMA_DTYPE=f16 fail before loading with the refusal that cites the measured overflow (exit 101); the ordinary unit test of that refusal and the bundle-only test that all eight official F16 graphs are non-finite pass.",
             "evidence": [{"log": f"{LOGS}/refusal.gemma.f16.metal.long_context_cached_decode_wraps_the_rings.log",
                           "quotes": ["panicked at tests/gemma_artifacts.rs:405:5:", "block 5's post-MLP residual add at BOS reaches 73,509.6 > 65,504"]},
                          {"log": f"{LOGS}/calibration.gemma.f16.official_f16_graphs_overflow_so_gemma_has_no_f16_envelope.log",
                           "quotes": ["/Volumes/T9/transformers-references/gemma-3-4b-it-093f9f38-long: uncalibrated: official cpu-eager float16 non-finite at prompt 0 position 0"]}]},
            {"id": "metal-bf16-refused-in-gates", "status": "measured (decision d, Metal BF16)",
             "summary": "On Burn 0.21 a Metal BF16 gate fails at load with the contextual error, at the first parameter, on both families; no logits are produced.",
             "evidence": [{"log": f"{LOGS}/refusal.SmolLM-135M.bf16.metal.prompt_logits_match_the_reference.log",
                           "quotes": ["cannot install Llama-family parameter `model.embed_tokens.weight`: backend cannot store checkpoint dtype BF16"]},
                          {"log": f"{LOGS}/refusal.gemma.bf16.metal.prompt_logits_match_the_reference.log",
                           "quotes": ["cannot install Gemma parameter `model.embed_tokens.weight`: backend cannot store checkpoint dtype BF16"]}]},
            {"id": "qwen2-7b-f16-uncalibrated", "status": "open: no F16 claim possible",
             "summary": "Qwen2-7B's three Metal/F16 gates and its two F16 calibration tests fail before loading as uncalibrated, exactly as at f9f78b5; an F16 envelope would need a rule change (for example SDPA graphs only).",
             "evidence": [{"log": f"{LOGS}/Qwen2-7B-f64.f16.metal.cached_greedy_matches_the_reference.log",
                           "quotes": ["uncalibrated: official cpu-eager float16 non-finite at prompt 0 position 1"]}]},
            {"id": "gemma-step-13-margin", "status": "open: deserves a targeted look",
             "summary": "gemma-3-4b-it's Flex/BF16 cached decode row at step 13 reproduces exactly, at 0.9447 of its maximum bound, the nearest margin of every half-precision gate; the stateless row at the same step is far lower.",
             "evidence": [{"log": f"{LOGS}/gemma.bf16.flex.cached_greedy_matches_the_reference.log",
                           "quotes": ["cached decode (cached rows): 32 rows from position 6, max |Δ| = 2.1528186798095703e0 at step 13 (bound 2.278956413269043e0)",
                                      "cached decode (stateless rows): 32 rows from position 6, max |Δ| = 9.293746948242188e-1 at step 13 (bound 2.278956413269043e0)"]}]},
            {"id": "no-decisive-greedy", "status": "recorded",
             "summary": "Under the envelope tolerance no greedy step of SmolLM-135M, SmolLM3-3B, or Qwen2-7B in Flex/BF16, nor of Gemma's long prompt, is decisive; those cached tests run with MIN_GREEDY=0 and rest on the decode rows, and the 0.5 self-consistency tolerance of cached prefills, batched rows, and chunk ends stays provisional.",
             "evidence": [{"log": f"{LOGS}/SmolLM3-3B.bf16.flex.cached_greedy_matches_the_reference.log", "quotes": ["greedy: 0 of 32 greedy tokens decisive and matching (smallest decisive margin inf), 32 ties skipped"]},
                          {"log": f"{LOGS}/gemma.bf16.flex.long_context_cached_decode_wraps_the_rings.log", "quotes": ["long greedy: 0 of 8 greedy tokens decisive and matching (smallest decisive margin inf), 8 ties skipped"]}]},
            {"id": "shared-target-dir-reuse", "status": "process finding",
             "summary": "In a shared CARGO_TARGET_DIR cargo reused, as fresh, binaries built from another copy of the same tree (the f9f78b5 export). Here the forced rebuild from the worktree produced byte-identical binaries, but a stage that switches source trees must force or verify a rebuild before it attributes a run to a checkout.",
             "evidence": [{"log": f"{LOGS}/build-0-noop.log", "quotes": ["    Finished `release` profile [optimized] target(s) in 0.64s"]},
                          {"log": f"{LOGS}/build.log", "quotes": ["   Compiling transformers v0.1.0 (/Users/nover/repos/transformers-worktrees/tfm-half-followups2)"]}]},
        ],
        "open_findings_carried": ["nemotron-floor-failure", "qwen2-7b-f16-uncalibrated", "gemma-f16-overflow", "metal-bf16-unavailable",
                                  "gemma-step-13-margin", "eager-attention-excess", "burn-flex-pins", "scalar-rounding", "muse-q8-peak",
                                  "provisional-self-consistency"],
        "not_run": [
            {"gates": "Nemotron-H Metal parity and official-precision gates, Qwen3.5-0.8B parity, benchmark and memory gates, Qwen3.8-27B, Muse-Glimmer-30B, DeepSeek-V2-Lite, and Qwen3-30B-A3B gates, smokes, and audits",
             "reason": "outside this rerun's scope; last run at f9f78b5 in the envelope record (the Nemotron-H official-precision gate failing), and 8181605 compiles the same library code"},
            {"gates": "gpt-oss and FLAN-T5 (small and XXL) gates", "reason": "ClampedSwiGlu and the F32 gated-GELU paths are unchanged by this round (envelope record, not_run); not in this rerun's scope"},
            {"gates": "CUDA and ROCm builds and every CUDA gate", "reason": "no CUDA or ROCm host"},
            {"gates": "coverage qualification (95% shared/core, 90% crate-wide)", "reason": "not run in this round"},
            {"gates": "python3 scripts/check_release.py --gate package", "reason": "it builds into the worktree's target directory on the local disk; only the package listing and a size estimate were checked (envelope round)"},
            {"gates": "clippy, cargo test, MSRV, rustdoc, and check_release --gate source at 8181605", "reason": "they ran at 1816d73 (envelope record); no Rust source changed after it. The Python tests, the documentation audit, and cargo fmt --check ran at 8181605 (final_head_checks)"},
            {"gates": "F16 of Gemma 3 1B/270m and Gemma 2", "reason": "no local checkpoints; downloads are not allowed"},
            {"gates": "official batched and chunked rows, and the position-0-split envelope", "reason": "the generator does not record batched or chunked rows yet; the split envelope has not been derived"},
            {"gates": "the library-level Gemma 3 F16 policy (plan S7) and its heavy opt-in test", "reason": "decision (d) awaits the maintainer (user_review true)"},
        ],
        "excluded": [
            {"path": "/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/followups/rerun/ (the files directly under it, dated 2026-10-17)",
             "reason": "an earlier rerun attempt at fd90c6c, made while both external volumes were missing (its preflight.log lists them as MISSING); it belongs to no round recorded here and is neither copied nor relied on"},
            {"path": "/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/followups/rerun/8181605/prior-binaries/",
             "reason": "the 106 MB copies of the binaries present before the forced rebuild; rerun/prior-binaries/SHA256SUMS records their digests, which equal the rebuilt binaries'"},
        ],
        "stage_reports": [
            {"path": f"{NAME}/stage-reports/decide.json", "stage": "decide:half-precision", "workflow_key": "v2:f60f2bdc673f71bff5a5602d64724908adee98649a1067481d389ff60176cc75"},
            {"path": f"{NAME}/stage-reports/implement.json", "stage": "impl:half-followups", "workflow_key": "v2:ebda08d6a44a2939dd4cb1cb7952107a64efe0138dfde762b4880b64ada44516"},
            {"path": f"{NAME}/stage-reports/rerun.json", "stage": "rerun:half-gates", "workflow_key": "v2:749033e55d84d74713a251d6a6f6f40866612a9a0d1a77086d942e1346360f6e"},
        ],
        "stage_reports_note": "The workflow stages' own accounts, copied from the workflow journal as the stages returned them. They are claims, not evidence: every figure this record relies on is quoted from a log, and only the decisions are taken from decide.json.",
        "run_time_digests": {
            "file": f"{RERUN}/SHA256SUMS",
            "prior_binaries": f"{RERUN}/prior-binaries/SHA256SUMS",
            "note": "Written by the rerun stage after its last run; it lists every log and harness file (relative to rerun/) and the two prior-binary copies. The copies of the logs and harness here have exactly those digests.",
        },
        "historical_records": {name: sha256((MEASUREMENTS / name).read_bytes())
                               for name in ("half-precision-calibration-20261017.json", "mac-validation-20261017.json", f"{ENVELOPE_NAME}.json")},
        "campaign_directory": NAME,
        "builder": {"script": f"{NAME}/record/build_record.py",
                    "note": "Derives every run entry, quote, comparison, identity row, and tally of this record from the committed campaign files and the envelope and Mac validation records, asserting each comparison as it goes; the narrative fields are written in it by hand."},
    }
    record["files"] = files_digests()
    out = MEASUREMENTS / f"{NAME}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{out}: {out.stat().st_size} bytes, {len(runs)} runs, tallies {tally['total']}, files {len(record['files'])}")


if __name__ == "__main__":
    main()
