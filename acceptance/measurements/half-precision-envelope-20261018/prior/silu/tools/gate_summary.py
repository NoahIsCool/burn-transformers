"""Scratch: tabulates the prompt-logit gate logs of the SiLU experiment (not committed).

Usage: python3 -I gate_summary.py LOG_DIR

Reads <checkpoint>.<dtype>.<backend>.prompt_logits.<label>.log files and prints,
per checkpoint, profile and prompt, the base and patched max and mean |delta|
against the bound (as value / bound), the gate verdict, the peak memory
footprint and the exit line. Prompts the gate did not reach are shown as "--".
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

PROMPT = re.compile(r"prompt (\d+): (\d+) tokens, max \|Δ\| = ([0-9.e+-]+) at position (\d+) "
                    r"\(bound ([0-9.e+-]+)\), mean \|Δ\| = ([0-9.e+-]+) \(bound ([0-9.e+-]+)\)")


def parse(path):
    """Returns the per-prompt rows, verdict, failure message, peak footprint and exit line of one log."""
    text = path.read_text()
    rows = {int(m[1]): (float(m[3]), int(m[4]), float(m[5]), float(m[6]), float(m[7])) for m in PROMPT.finditer(text)}
    verdict = "PASS" if "test result: ok. 1 passed" in text else "FAIL"
    failure = next((line for line in text.splitlines() if "exceeds the half-precision bound" in line
                    or "non-finite" in line), "")
    peak = re.search(r"(\d+)\s+peak memory footprint", text)
    exit_line = next((line for line in text.splitlines() if line.startswith("== exit=")), "")
    return rows, verdict, failure, int(peak[1]) if peak else None, exit_line


def main():
    root = Path(sys.argv[1])
    table = defaultdict(dict)
    for path in sorted(root.glob("*.prompt_logits.*.log")):
        name, dtype, backend, _, label, _ = path.name.rsplit(".", 5)
        table[(name, f"{backend}/{dtype}")][label] = parse(path)
    for (name, profile), labels in table.items():
        print(f"## {name} {profile}")
        for label in ("base", "patched"):
            if label not in labels:
                continue
            rows, verdict, failure, peak, exit_line = labels[label]
            cells = []
            for prompt in range(3):
                if prompt in rows:
                    maximum, position, bound_max, mean, bound_mean = rows[prompt]
                    cells.append(f"p{prompt} max {maximum:.4f}@{position}/{bound_max:.4f} ({maximum / bound_max:.2f})"
                                 f" mean {mean:.5f}/{bound_mean:.5f} ({mean / bound_mean:.2f})")
                else:
                    cells.append(f"p{prompt} --")
            peak_text = f"{peak / 1e9:.2f} GB" if peak else "?"
            print(f"  {label:8s} {verdict} peak {peak_text}; " + "; ".join(cells))
            if failure:
                print(f"           {failure.strip()}")
            print(f"           {exit_line}")


if __name__ == "__main__":
    main()
