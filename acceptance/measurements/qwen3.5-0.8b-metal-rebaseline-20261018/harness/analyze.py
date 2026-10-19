"""Print the re-baseline campaign's steady state, statistics, and comparisons.

Usage: python3 -I analyze.py PROTOCOL_SCRIPT CAMPAIGN_SCRIPT CAMPAIGN_DIR SEPTEMBER_DIR

PROTOCOL_SCRIPT is the crate's scripts/qwen35_benchmark_protocol.py, whose
rules the record uses; CAMPAIGN_SCRIPT is the copy of the script that ran the
campaign (harness/qwen35_benchmark_protocol.py), whose original steady-state
rule (the earliest window of at least six final rounds whose half-medians
agree within 2.5%) this prints for comparison, start by start. Reads the
campaign with the protocol script and prints: every candidate block of the
steady-state rule with the series beyond the 2.5% tolerance, their noise and
z; the steady state after each earlier whole block of rounds; the window's
per-position counts, and the original window's; both binaries' identities;
every gated metric's round-to-round noise over the window and its
steady-state median and range for both binaries with the head/September-
rebuild ratio and both binaries' ratios to the approved September 10 values;
the head's medians over the original rule's window beside the corrected
ones; the cold-first-token diagnostic (median, range, and rank
correlations, with and without the samples that followed discarded attempts);
and every contaminated attempt the runner set aside, with its shader-cache
growth and the clean rerun that followed it. Writes nothing.

The record's first version, under the original rule, quoted
analysis/analysis.log, which an earlier version of this script wrote with the
crate's script of commit 12296ff; this version's output is
analysis/analysis-20261019.log.
"""

import importlib.util
import json
import sys
from pathlib import Path

SUMMARIES = {"baseline": "qwen-controlled-current-summary.json", "public": "qwen-public-summary.json"}


def load(path, name):
    """Import a protocol script under ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_rule(protocol, original, samples):
    """Per-start drifting series under the campaign script's half-median rule, and its earliest steady start."""
    rounds = max(sample["round"] for sample in samples)
    cells = {}
    for sample in samples:
        cells.setdefault((sample["label"], sample["benchmark"]), []).append(sample)
    values = {}
    for (label, kind), runs in cells.items():
        runs = sorted(runs, key=lambda run: run["round"])
        for metric in protocol.gated_metric_names(kind):
            values[f"{label}/{kind}/{metric}"] = [run["metrics"]["gated"][metric]["median"] for run in runs]
    starts = {}
    for first in range(1, rounds - original.STEADY_WINDOW_ROUNDS + 2):
        drifting = {}
        for name, series in values.items():
            window = series[first - 1:]
            half = len(window) // 2
            drift = protocol.median(window[-half:]) / protocol.median(window[:half]) - 1.0
            if abs(drift) > original.STEADY_DRIFT:
                drifting[name] = drift
        starts[first] = drifting
    chosen = original.summarize(samples)["steady_state"]["first_round"]
    earliest = next((first for first, drifting in starts.items() if not drifting), None)
    if chosen != earliest:
        raise SystemExit(f"original rule mismatch: script {chosen}, recomputed {earliest}")
    return chosen, starts


def print_assessments(protocol, steady):
    """Every candidate block of a steady-state result, with the series beyond the tolerance."""
    for assessment in steady["assessments"]:
        beyond = {name: item for name, item in assessment["series"].items()
                  if abs(item["drift"]) > protocol.STEADY_DRIFT}
        print(f"   block from round {assessment['first_round']} against rounds {assessment['later_rounds'][0]} to "
              f"{assessment['later_rounds'][-1]}: {len(assessment['drifting'])} drifting, {len(beyond)} beyond tolerance")
        for name, item in sorted(beyond.items(), key=lambda pair: -abs(pair[1]["drift"])):
            z = "none (zero noise)" if item["z"] is None else f"{item['z']:.2f}"
            print(f"      {name} drift {item['drift']:+.4f} noise {item['noise']:.4f} z {z} "
                  f"{'DRIFTING' if item['drifting'] else 'within noise'}")


def main():
    protocol, original = load(sys.argv[1], "protocol"), load(sys.argv[2], "original")
    root, september = Path(sys.argv[3]), Path(sys.argv[4])
    samples = protocol.load_campaign(root)
    summary = protocol.summarize(samples)
    steady = summary["steady_state"]
    print(f"== rounds {summary['rounds']}; steady state from round {steady['first_round']} (rounds {steady['rounds']}); "
          f"limiting block from round {steady['limiting_block']}; tolerance {protocol.STEADY_DRIFT}, "
          f"z limit {protocol.STEADY_Z}, window at least {protocol.STEADY_WINDOW_ROUNDS} rounds in whole blocks")
    print_assessments(protocol, steady)
    for rounds in range(protocol.MINIMUM_ROUNDS, summary["rounds"], protocol.BLOCK_ROUNDS):
        early = protocol.summarize([sample for sample in samples if sample["round"] <= rounds])["steady_state"]
        print(f"== after {rounds} rounds: steady state from round {early['first_round']}; limiting block from round "
              f"{early['limiting_block']} with {len(early['drifting'])} drifting series")
        print_assessments(protocol, early)
    chosen, starts = original_rule(protocol, original, samples)
    print(f"== original rule (campaign script, half-medians within {original.STEADY_DRIFT} over at least "
          f"{original.STEADY_WINDOW_ROUNDS} final rounds): earliest steady start {chosen}")
    for first, drifting in starts.items():
        listed = ", ".join(f"{name} {drift:+.4f}" for name, drift in drifting.items()) or "none"
        print(f"   start {first}: {'steady' if not drifting else 'drifting'}: {listed}")
    print(f"== window position counts {json.dumps(steady['position_counts'], sort_keys=True)}")
    for label, identity in summary["binaries"].items():
        print(f"== binary {label}: sha256 {identity['sha256']} commit {identity['source_commit']} status "
              f"{identity['worktree_status']!r} toolchain {identity['rustup_toolchain']} features "
              f"{identity['cargo_features']} profile {identity['profile']!r}")
    counts = {}
    for sample in samples:
        if sample["round"] >= chosen:
            key = f"{sample['label']}/{sample['benchmark']}"
            counts.setdefault(key, {}).setdefault(str(sample["position"]), 0)
            counts[key][str(sample["position"])] += 1
    print(f"== original window position counts {json.dumps(counts, sort_keys=True)}")
    window = steady["rounds"]
    old_window = list(range(chosen, summary["rounds"] + 1))
    for kind in protocol.BENCHMARKS:
        approved = protocol.run_metrics(json.loads((september / SUMMARIES[kind]).read_text())["benchmark"], kind)
        print(f"== {kind} benchmark (steady-state medians in s over rounds {window[0]} to {window[-1]}; approved "
              f"September 10 values from {SUMMARIES[kind]})")
        head, base = summary["cells"][f"head/{kind}"], summary["cells"][f"september/{kind}"]
        runs = sorted((sample for sample in samples if (sample["label"], sample["benchmark"]) == ("head", kind)),
                      key=lambda run: run["round"])
        for metric in protocol.gated_metric_names(kind):
            h, b = head["gated"][metric], base["gated"][metric]
            a = approved["gated"][metric]["median"]
            print(f"   {metric} round-to-round noise over the window: head {h['noise']:.4f}, september {b['noise']:.4f}")
            previous = protocol.median([run["metrics"]["gated"][metric]["median"] for run in runs
                                        if run["round"] in old_window])
            print(f"   {metric}: head {h['median']:.6f} [{h['minimum']:.6f}, {h['maximum']:.6f}] | september "
                  f"{b['median']:.6f} [{b['minimum']:.6f}, {b['maximum']:.6f}] | head/september "
                  f"{h['median'] / b['median']:.3f} | approved {a:.6f}: head {h['median'] / a - 1:+.1%}, "
                  f"september {b['median'] / a - 1:+.1%} | head over rounds {old_window[0]} to {old_window[-1]} "
                  f"{previous:.6f} ({h['median'] / previous - 1:+.2%})")
        a = approved["diagnostic"]["cold_first_token"]
        for label, cell in (("head", head), ("september", base)):
            cold = cell["diagnostic"]["cold_first_token"]
            s, every = cold["steady_state"], cold["all_rounds"]
            without = cold["without_rounds_after_discarded_attempts"]
            print(f"   cold first token (diagnostic) {label}: steady median {s['median']:.4f} range "
                  f"[{s['minimum']:.4f}, {s['maximum']:.4f}] | all rounds median {every['median']:.4f} range "
                  f"[{every['minimum']:.4f}, {every['maximum']:.4f}] | approved {a:.6f} ({s['median'] / a - 1:+.1%}) | "
                  f"spearman with shader-cache growth {cold['spearman_with_metal_cache_growth']:+.3f}, "
                  f"with round {cold['spearman_with_round']:+.3f}")
            print(f"      per round cold {[round(v, 4) for v in every['values']]} growth {cold['metal_cache_growth_bytes']}")
            print(f"      after discarded attempts: rounds {cold['rounds_after_discarded_attempts']}, attempts' growth "
                  f"{cold['discarded_attempts_metal_cache_growth_bytes']}; spearman with growth including the "
                  f"attempts' {cold['spearman_with_metal_cache_growth_including_discarded_attempts']:+.3f}")
            ws, wa = without["steady_state"], without["all_rounds"]
            print(f"      without those rounds: steady median {ws['median']:.4f} range [{ws['minimum']:.4f}, "
                  f"{ws['maximum']:.4f}] ({len(ws['values'])} rounds) | all rounds median {wa['median']:.4f} range "
                  f"[{wa['minimum']:.4f}, {wa['maximum']:.4f}] ({len(wa['values'])} rounds) | spearman with growth "
                  f"{without['spearman_with_metal_cache_growth']:+.3f}, with round {without['spearman_with_round']:+.3f}")
        if head["tokens"] == base["tokens"] == protocol.token_identity(
                json.loads((september / SUMMARIES[kind]).read_text())["benchmark"], kind):
            print("   token and prediction hashes: identical in every run of both binaries and to September 10")
        else:
            print("   token and prediction hashes: DIFFER")
    for sample in samples:
        for attempt in sample["discarded_attempts"]:
            record = json.loads((root / attempt["host"]).read_text())
            report = json.loads((root / attempt["host"].replace("host/", "bench/", 1)
                                 .replace(".host.json", ".json")).read_text())
            print(f"== {attempt['kind']} attempt {Path(attempt['host']).name}: admitted at load "
                  f"{record['admission']['admitting_load_1m']}/{record['admission']['admitting_load_5m']}, in-run "
                  f"maximum {record['in_run_load_1m_maximum']}, wall {record['wall_seconds']} s, shader-cache growth "
                  f"{attempt['metal_cache_growth_bytes']} bytes, cold {report['cold_first_token']['seconds']:.4f} s; "
                  f"clean rerun: growth {sample['metal_cache_growth_bytes']} bytes, cold "
                  f"{sample['metrics']['diagnostic']['cold_first_token']:.4f} s")


if __name__ == "__main__":
    main()
