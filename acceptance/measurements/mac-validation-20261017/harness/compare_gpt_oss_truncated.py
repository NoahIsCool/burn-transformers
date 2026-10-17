"""Compares the truncated gpt-oss rerun logs with the f4c56c4 replay log (scratch, not committed).

Usage: python3 -I compare_gpt_oss_truncated.py REPLAY_GATES_LOG RUN_DIR

For every test of the 1-layer and 4-layer bundles, prints pass/fail, wall time,
the `/usr/bin/time -l` peak memory footprint and maximum RSS, the measurement
lines (deviation, greedy and cached-decode lines, with the load timing
stripped), whether those lines are byte-identical to the f4c56c4 replay, and
the peak-footprint change against it.
"""

import json
import os
import re
import sys

KEEP = re.compile(r"(prompt \d+:|greedy:|batched row|cached decode|max \|Δ\||mean \|Δ\||cosine)")
LOADED = re.compile(r"loaded (\d+) blocks as (\w+) in ([\d.]+)s")


def replay_sections(path):
    """Returns {(bundle short name, test): {"lines": [...], "peak": int}} from the replay log."""
    out, bundle, test = {}, None, None
    in_gpt = False
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if line.startswith("== gpt-oss-20b truncated"):
            in_gpt = True
            continue
        if in_gpt and line.startswith("== "):
            break
        if not in_gpt:
            continue
        m = re.match(r"-- gpt-oss-20b-6cee5e81-(\dlayer)-f32-reference.bundle", line)
        if m:
            bundle = m.group(1)
            continue
        m = re.match(r"test (\S+) \.\.\. (.*)", line)
        if m:
            test = m.group(1)
            out[(bundle, test)] = {"lines": [], "peak": None}
            rest = LOADED.sub("", m.group(2)).strip()
            if rest:
                out[(bundle, test)]["lines"].append(rest)
            continue
        m = re.match(r"\s*(\d+)\s+peak memory footprint", line)
        if m and test:
            out[(bundle, test)]["peak"] = int(m.group(1))
            continue
        if test and KEEP.search(line):
            out[(bundle, test)]["lines"].append(line)
    return out


def parse_run(path):
    """Returns the figures of one per-test rerun log."""
    text = open(path, encoding="utf-8").read()
    lines, loaded = [], None
    for line in text.splitlines():
        m = re.match(r"test (\S+) \.\.\. (.*)", line)
        if m:
            lm = LOADED.search(m.group(2))
            if lm:
                loaded = {"blocks": int(lm.group(1)), "dtype": lm.group(2), "seconds": float(lm.group(3))}
            rest = LOADED.sub("", m.group(2)).strip()
            if rest and rest not in ("ok", "FAILED"):
                lines.append(rest)
            continue
        if line.startswith("== ") or line.startswith("test result"):
            continue
        if KEEP.search(line):
            lines.append(line)
    get = lambda pat, cast=str: (lambda m: cast(m.group(1)) if m else None)(re.search(pat, text))
    return {
        "source": get(r"== source (\S+)"),
        "binary_sha256": get(r"sha256 ([0-9a-f]{64})"),
        "start": get(r"== start (\S+)"),
        "result": get(r"test result: (\w+)\."),
        "passed": get(r"(\d+) passed;", int),
        "failed": get(r"(\d+) failed;", int),
        "exit": get(r"== exit=(-?\d+)", int),
        "wall_s": get(r"wall=([\d.]+)s", float),
        "real_s": get(r"([\d.]+) real", float),
        "test_finished_s": get(r"finished in ([\d.]+)s", float),
        "peak_footprint_bytes": get(r"(\d+)\s+peak memory footprint", int),
        "max_rss_bytes": get(r"(\d+)\s+maximum resident set size", int),
        "min_free_pct": get(r"min_free_pct=(\d+)", int),
        "panicked": "panicked" in text,
        "loaded": loaded,
        "lines": lines,
    }


def main():
    """Prints the comparison and writes results.json into the run directory."""
    replay = replay_sections(sys.argv[1])
    run_dir = sys.argv[2]
    results = []
    for bundle in ("1layer", "4layer"):
        for backend, prefix in (("flex", "flex_"), ("metal", "metal::metal_")):
            for test in ("prompt_logits_match_the_reference", "cached_greedy_matches_the_reference",
                         "batched_prefill_matches_the_reference", "cached_long_prompt_matches_the_reference"):
                name = prefix + test
                path = os.path.join(run_dir, f"gpt-oss-20b-{bundle}.f32.{backend}.{test}.log")
                if not os.path.exists(path):
                    print(f"{bundle} {name}: NOT RUN")
                    continue
                run = parse_run(path)
                old = replay.get((bundle, name), {"lines": [], "peak": None})
                identical = run["lines"] == old["lines"]
                delta = (run["peak_footprint_bytes"] - old["peak"]) if (old["peak"] and run["peak_footprint_bytes"]) else None
                run.update(bundle=bundle, test=name, backend=backend.capitalize(), dtype="F32",
                           identical_to_f4c56c4=identical, f4c56c4_lines=old["lines"],
                           f4c56c4_peak_footprint_bytes=old["peak"], peak_delta_bytes=delta,
                           log=path)
                results.append(run)
                print(f"{bundle} {backend.capitalize()}/F32 {name}: {run['result']} exit={run['exit']} "
                      f"wall={run['wall_s']}s load={run['loaded']['seconds'] if run['loaded'] else None}s "
                      f"peak={run['peak_footprint_bytes'] / 1e9 if run['peak_footprint_bytes'] else None:.3f} GB "
                      f"(f4c56c4 {old['peak'] / 1e9 if old['peak'] else float('nan'):.3f} GB, "
                      f"delta {delta / 1e6 if delta is not None else float('nan'):+.1f} MB) "
                      f"identical_lines={identical}")
                for line in run["lines"]:
                    print("    " + line)
                if not identical:
                    print("    f4c56c4:")
                    for line in old["lines"]:
                        print("    " + line)
    json.dump(results, open(os.path.join(run_dir, "results.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
