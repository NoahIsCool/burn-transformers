"""Review round: compares each rerun gate's test output with its run of the same name at 8181605.

Usage: python3 -I compare_runs.py WORKTREE LOGS

For every ``gemma.*.log`` and ``calibration.gemma.bf16.*.log`` in LOGS, reads
the follow-ups round's log of the same name
(``half-precision-followups-20261017/rerun/logs/``), authenticated by the
digest that record's ``files`` lists, and compares the test output from
``running 1 test`` to ``test result`` with thread ids and timings masked, and
the exit status. Prints one line per run and a tally.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

MASK_THREAD = re.compile(r"thread '([^']*)' \(\d+\)")
MASK_TIME = re.compile(r"finished in [\d.]+s")


def body(text):
    """The test output from ``running 1 test`` to ``test result``, thread ids and timings masked."""
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith("running "):
            inside = True
        if inside:
            lines.append(MASK_TIME.sub("finished in Xs", MASK_THREAD.sub(r"thread '\1' (TID)", line)))
        if line.startswith("test result"):
            break
    return lines


def exit_status(text):
    """The ``== exit=`` status of a run log."""
    return int(re.search(r"(?m)^== exit=(-?\d+)", text).group(1))


def main():
    """Compares and tallies."""
    measurements = Path(sys.argv[1]) / "acceptance/measurements"
    followups = json.loads((measurements / "half-precision-followups-20261017.json").read_text())
    logs = Path(sys.argv[2])
    tally = {"identical": 0, "different": 0}
    for path in sorted(logs.glob("*.log")):
        if not (path.name.startswith("gemma.") or path.name.startswith("calibration.gemma.bf16.")):
            continue
        relative = f"half-precision-followups-20261017/rerun/logs/{path.name}"
        data = (measurements / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == followups["files"][relative], relative
        before, now = data.decode(), path.read_text()
        same = body(before) == body(now) and exit_status(before) == exit_status(now)
        tally["identical" if same else "different"] += 1
        print(f"{path.name}: {len(body(now))} output lines, exit {exit_status(now)} -> {'IDENTICAL' if same else 'DIFFERENT'} to 8181605")
    print(f"== tally {json.dumps(tally)}")


if __name__ == "__main__":
    main()
