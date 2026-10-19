"""Print the re-baseline campaign's steady state, statistics, and comparisons.

Usage: python3 -I analyze.py PROTOCOL_SCRIPT CAMPAIGN_DIR SEPTEMBER_DIR

Reads the campaign with the protocol script, prints the steady-state window
and drifts (and the steady state after each earlier whole block of rounds), every gated metric's steady-state median and range for both
binaries with the head/September-rebuild ratio and both binaries' ratios to
the approved September 10 values, the cold-first-token diagnostic (median,
range, and rank correlations), and every contaminated attempt the runner set
aside. Writes nothing.
"""

import importlib.util
import json
import sys
from pathlib import Path

SUMMARIES = {"baseline": "qwen-controlled-current-summary.json", "public": "qwen-public-summary.json"}


def main():
    spec = importlib.util.spec_from_file_location("protocol", sys.argv[1])
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    root, september = Path(sys.argv[2]), Path(sys.argv[3])
    samples = protocol.load_campaign(root)
    summary = protocol.summarize(samples)
    steady = summary["steady_state"]
    print(f"== rounds {summary['rounds']}; steady state from round {steady['first_round']} "
          f"(rounds {steady['rounds']}); maximum |drift| {steady['maximum_absolute_drift']:.4f} "
          f"(limit {protocol.STEADY_DRIFT})")
    for name, drift in sorted(steady["drift"].items(), key=lambda item: -abs(item[1]))[:6]:
        print(f"   largest drift {name} {drift:+.4f}")
    for rounds in range(protocol.MINIMUM_ROUNDS, summary["rounds"], protocol.BLOCK_ROUNDS):
        early = protocol.summarize([sample for sample in samples if sample["round"] <= rounds])["steady_state"]
        print(f"== after {rounds} rounds: steady state from round {early['first_round']}; drifting at the "
              f"{'chosen' if early['first_round'] else 'whole'} window: {early['drifting']}")
    for kind in protocol.BENCHMARKS:
        approved = protocol.run_metrics(json.loads((september / SUMMARIES[kind]).read_text())["benchmark"], kind)
        print(f"== {kind} benchmark (steady-state medians in s; approved September 10 values from {SUMMARIES[kind]})")
        head, base = summary["cells"][f"head/{kind}"], summary["cells"][f"september/{kind}"]
        for metric in protocol.gated_metric_names(kind):
            h, b = head["gated"][metric], base["gated"][metric]
            a = approved["gated"][metric]["median"]
            print(f"   {metric}: head {h['median']:.6f} [{h['minimum']:.6f}, {h['maximum']:.6f}] | september "
                  f"{b['median']:.6f} [{b['minimum']:.6f}, {b['maximum']:.6f}] | head/september "
                  f"{h['median'] / b['median']:.3f} | approved {a:.6f}: head {h['median'] / a - 1:+.1%}, "
                  f"september {b['median'] / a - 1:+.1%}")
        a = approved["diagnostic"]["cold_first_token"]
        for label, cell in (("head", head), ("september", base)):
            cold = cell["diagnostic"]["cold_first_token"]
            s, every = cold["steady_state"], cold["all_rounds"]
            print(f"   cold first token (diagnostic) {label}: steady median {s['median']:.4f} range "
                  f"[{s['minimum']:.4f}, {s['maximum']:.4f}] | all rounds median {every['median']:.4f} range "
                  f"[{every['minimum']:.4f}, {every['maximum']:.4f}] | approved {a:.6f} ({s['median'] / a - 1:+.1%}) | "
                  f"spearman with shader-cache growth {cold['spearman_with_metal_cache_growth']:+.3f}, "
                  f"with round {cold['spearman_with_round']:+.3f}")
            print(f"      per round cold {[round(v, 4) for v in every['values']]} growth {cold['metal_cache_growth_bytes']}")
        if head["tokens"] == base["tokens"] == protocol.token_identity(
                json.loads((september / SUMMARIES[kind]).read_text())["benchmark"], kind):
            print("   token and prediction hashes: identical in every run of both binaries and to September 10")
        else:
            print("   token and prediction hashes: DIFFER")
    for path in sorted((root / "host").glob("*.contaminated-*.host.json")):
        record = json.loads(path.read_text())
        print(f"== contaminated attempt {path.name}: admitted at load {record['admission']['admitting_load_1m']}"
              f"/{record['admission']['admitting_load_5m']}, in-run maximum {record['in_run_load_1m_maximum']}, "
              f"wall {record['wall_seconds']} s")


if __name__ == "__main__":
    main()
