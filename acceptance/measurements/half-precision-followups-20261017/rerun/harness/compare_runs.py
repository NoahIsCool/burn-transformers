"""Scratch (rerun at 8181605): compares each rerun log's test output with earlier rounds (not committed).

Usage: python3 -I compare_runs.py > logs/compare.log

For every run in logs/runs.jsonl:
  * against the official-envelope round at f9f78b5 (half2/gates/logs, same run
    name): the whole test output, from `running 1 test` to `test result`, must be
    identical once thread ids and the `finished in` time are masked, and so must
    the exit status;
  * for F32 runs, also against the last pre-SiLU round (Mac validation round:
    Llama family at 53f167b, Gemma at c2936fe): every statistic line of the
    earlier output (one containing `|Δ|`) must occur verbatim in the rerun's
    output. Lines that only changed format (the greedy summary, the new
    decode-truth rows, the F32 matmul banner, the filtered-out count) are listed,
    not judged.
"""

import json
import os
import re

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
SP = "/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad"
F9 = f"{SP}/half2/gates/logs"
MAC_LLAMA = f"{SP}/mac-validation/llama-family"
MAC_GEMMA = f"{SP}/mac-validation/gemma/c2936fe"


def body(path):
    """Test output lines from `running 1 test` to `test result`, with run-specific details masked."""
    lines, inside = [], False
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("running "):
            inside = True
        if inside:
            line = re.sub(r"thread '([^']*)' \(\d+\)", r"thread '\1' (TID)", line)
            line = re.sub(r"finished in [\d.]+s", "finished in Xs", line)
            lines.append(line)
        if line.startswith("test result"):
            break
    return lines


def source(path):
    """The commit named by a log's first line."""
    first = open(path, errors="replace").readline().split()
    return first[2] if len(first) > 2 else "?"


def exit_status(path):
    """The `== exit=` status of a log."""
    match = re.search(r"^== exit=(-?\d+)", open(path, errors="replace").read(), re.M)
    return int(match.group(1)) if match else None


def stat_lines(lines):
    """Statistic lines (`|Δ|`), without the `test NAME ... ` prefix of the first output line."""
    out = []
    for line in lines:
        line = re.sub(r"^test \S+ \.\.\. ", "", line)
        if "|Δ|" in line:
            out.append(line)
    return out


def greedy_counts(lines):
    """(label, matched, total) of each greedy summary line, in either round's wording."""
    out = []
    for line in lines:
        line = re.sub(r"^test \S+ \.\.\. ", "", line)
        match = re.match(r"^((?:long )?greedy): (\d+) of (\d+) (?:greedy )?tokens", line)
        if match:
            out.append((match.group(1), int(match.group(2)), int(match.group(3))))
    return out


def pre_silu(name):
    """The last pre-SiLU F32 log of a run, or None."""
    if ".f32." not in name:
        return None
    if name.startswith("gemma."):
        path = f"{MAC_GEMMA}/{name.replace('gemma.', 'gemma-3-4b-it.', 1)}.log"
    else:
        path = f"{MAC_LLAMA}/{name}.log"
    return path if os.path.exists(path) else None


def main():
    """Prints one verdict per run and a tally."""
    runs = [json.loads(line) for line in open(os.path.join(ROOT, "logs", "runs.jsonl"))]
    tally = {"identical_to_f9f78b5": 0, "differs_from_f9f78b5": 0, "no_f9f78b5_run": 0,
             "f32_stats_equal_pre_silu": 0, "f32_stats_differ_pre_silu": 0}
    for run in runs:
        name, new = run["name"], run["log"]
        new_body = body(new)
        old = f"{F9}/{name}.log"
        print(f"== {name}: exit {run['exit']} result {run['result']}")
        if os.path.exists(old):
            old_body = body(old)
            same = old_body == new_body and exit_status(old) == run["exit"]
            tally["identical_to_f9f78b5" if same else "differs_from_f9f78b5"] += 1
            print(f"   vs {source(old)[:7]} {old}: {'IDENTICAL' if same else 'DIFFERENT'} "
                  f"({len(new_body)} output lines; exit {exit_status(old)} then, {run['exit']} now)")
            if not same:
                for line in old_body:
                    if line not in new_body:
                        print(f"   - {line}")
                for line in new_body:
                    if line not in old_body:
                        print(f"   + {line}")
        else:
            tally["no_f9f78b5_run"] += 1
            print("   vs f9f78b5: no run of this name in the official-envelope round")
        prior = pre_silu(name)
        if prior:
            prior_body = body(prior)
            old_stats, new_stats = stat_lines(prior_body), stat_lines(new_body)
            missing = [line for line in old_stats if line not in new_stats]
            added = [line for line in new_stats if line not in old_stats]
            old_greedy, new_greedy = greedy_counts(prior_body), greedy_counts(new_body)
            ok = (not missing) and (old_stats or old_greedy) and old_greedy == new_greedy \
                and exit_status(prior) == run["exit"]
            tally["f32_stats_equal_pre_silu" if ok else "f32_stats_differ_pre_silu"] += 1
            print(f"   vs pre-SiLU {source(prior)[:7]} {prior}: {len(old_stats)} statistic lines "
                  f"{'all present verbatim' if not missing else f'({len(missing)} MISSING)'}; "
                  f"greedy then {old_greedy}, now {new_greedy}; exit then {exit_status(prior)} -> "
                  f"{'EQUAL' if ok else 'DIFFERENT'}")
            for line in missing:
                print(f"   missing: {line}")
            for line in added:
                print(f"   new in rerun (no earlier counterpart): {line}")
    print("== tally " + json.dumps(tally))


if __name__ == "__main__":
    main()
