"""Summarizes the quantized group's per-test logs (scratch helper, not committed).

Usage: python3 -I digest_quantized.py OUT_DIR

Prints, for each log, the source line, binary hash, test result, every line
the test printed with figures, the exit line, and the `/usr/bin/time -l`
peak memory footprint and maximum resident set size.
"""

import glob
import os
import re
import sys


def main():
    """Prints a digest of every *.log in OUT_DIR except summary/build logs."""
    out_dir = sys.argv[1]
    for path in sorted(glob.glob(os.path.join(out_dir, "*.log"))):
        name = os.path.basename(path)
        if name in ("summary.log", "build.log"):
            continue
        text = open(path, errors="replace").read()
        print(f"#### {name}")
        for line in text.splitlines():
            s = line.strip()
            if (s.startswith("== source") or s.startswith("== binary") or s.startswith("== exit")
                    or s.startswith("== artifact") or s.startswith("== authenticated")
                    or s.startswith("== start") or s.startswith("test result")
                    or "panicked" in s or "Burnpack:" in s or "smoke:" in s or "Metal:" in s
                    or "audit:" in s or "conversion:" in s or "launches=" in s
                    or s.startswith("thread ") or "assertion" in s or s.startswith("left")
                    or s.startswith("right")):
                print("  " + s[:1200])
        for label in ("peak memory footprint", "maximum resident set size"):
            match = re.search(r"(\d+)\s+" + label, text)
            if match:
                print(f"  {label}: {int(match.group(1))} bytes ({int(match.group(1)) / 1e9:.3f} GB)")
        match = re.search(r"([\d.]+) real", text)
        if match:
            print(f"  time real: {match.group(1)} s")


if __name__ == "__main__":
    main()
