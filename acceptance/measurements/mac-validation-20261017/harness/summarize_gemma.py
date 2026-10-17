"""Scratch helper: digests per-test gate logs into one line of key figures each (not committed).

Usage: python3 -I summarize_gemma.py LOG_DIR...
For every *.log written by run_gemma_gates.py, prints the source, binary SHA-256 prefix, label,
test, verdict, the deviation/greedy/consistency lines (abbreviated), panic message, peak memory
footprint, and wall time.
"""

import re
import sys
from pathlib import Path

KEYS = re.compile(r"max \||greedy:|long greedy|decode steps from|cached:|panicked|exceeds|non-finite|unwindowed: max")


def digest(path):
    """Returns the digest lines of one log."""
    text = path.read_text(errors="replace").splitlines()
    head = {line.split(" ", 2)[1]: line.split(" ", 2)[2] if line.count(" ") >= 2 else ""
            for line in text if line.startswith("== ") and line.count(" ") >= 2}
    source = next((line.split()[2] for line in text if line.startswith("== source")), "?")
    binary = next((line.rsplit(" ", 1)[1][:12] for line in text if line.startswith("== binary")), "?")
    label = next((line[3:] for line in text if line.startswith("== ") and ("/" in line.split()[1])
                  and not line.startswith("== env") and not line.startswith("== cmd")
                  and not line.startswith("== binary") and not line.startswith("== cwd")), "?")
    result = next((line for line in text if line.startswith("test result")), "no result")
    verdict = "PASS" if "test result: ok. 1 passed" in result else "FAIL"
    peak = next((int(line.split()[0]) for line in text if line.strip().endswith("peak memory footprint")), 0)
    wall = next((line.split("wall=")[1].split()[0] for line in text if line.startswith("== exit=")), "?")
    lines = []
    for line in text:
        if KEYS.search(line) and not line.startswith("== ") and "bounds of prompt" not in line:
            lines.append(line.replace("test ", "", 1) if line.startswith("test ") else line)
    return [f"## {path.name}: {verdict} | source {source[:7]} | binary {binary} | {label} | "
            f"peak {peak / 1e9:.2f} GB | wall {wall}"] + ["    " + line for line in lines]


def main():
    for directory in map(Path, sys.argv[1:]):
        for path in sorted(directory.glob("*.log")):
            if path.name in ("summary.log", "heavy.log", "build.log", "rebuild.log"):
                continue
            print("\n".join(digest(path)))


if __name__ == "__main__":
    main()
