"""Scratch helper: compact per-test result entries (name, status, detail) from the Llama-family gate logs."""

import json
import re
import sys
from pathlib import Path

ORDER = ["SmolLM-135M", "Qwen3-4B", "SmolLM3-3B", "Qwen2-7B-f64", "Llama-3.1-8B"]
TESTS = ["prompt_logits", "cached_greedy", "batched_prefill"]


def short(value):
    """Formats a float with four significant digits."""
    return f"{float(value):.4g}"


root = Path(sys.argv[1])
entries = []
for name in ORDER:
    for precision, backend in [("f32", "flex"), ("f32", "metal"), ("bf16", "flex"), ("f16", "metal")]:
        for test in TESTS:
            log = next(root.glob(f"{name}.{precision}.{backend}.{test}_*.log"))
            text = log.read_text()
            label, test_name = re.search(r"^== (\S+/\S+) (\S+)$", text, re.M).groups()
            code, wall = re.search(r"^== exit=(\d+) wall=([\d.]+)s", text, re.M).groups()
            peak = int(re.search(r"(\d+)\s+peak memory footprint", text).group(1)) / 1e9
            figures = []
            for line in text.splitlines():
                line = re.sub(r"^test \S+ \.\.\. ", "", line)
                m = re.match(r"prompt (\d+): (\d+) tokens, max \|Δ\| = (\S+)$", line)
                if m:
                    figures.append(f"p{m[1]} max {short(m[3])}")
                m = re.match(r"prompt (\d+): \d+ tokens, max \|Δ\| = (\S+) at position (\d+) \(bound (\S+)\), mean \|Δ\| = (\S+) \(bound (\S+)\)", line)
                if m:
                    figures.append(f"p{m[1]} max {short(m[2])}@{m[3]}/{short(m[4])} mean {short(m[5])}/{short(m[6])}")
                m = re.match(r"batched row (\d+): max \|Δ\| = (\S+)$", line)
                if m:
                    figures.append(f"row{m[1]} max {short(m[2])}")
                m = re.match(r"batched row (\d+): max \|Δ\| = (\S+) to the reference \(bound (\S+)\)", line)
                if m:
                    figures.append(f"row{m[1]} max {short(m[2])}/{short(m[3])}")
                m = re.match(r"batched row (\d+): max \|Δ\| = (\S+) to the stateless", line)
                if m:
                    figures.append(f"row{m[1]} vs stateless {short(m[2])}")
                m = re.match(r"cached prefill: max \|Δ\| = (\S+) to the (reference|stateless)", line)
                if m:
                    figures.append(f"prefill vs {m[2]} {short(m[1])}")
                m = re.match(r"greedy: (\d+) of (\d+) tokens match", line)
                if m:
                    figures.append(f"greedy {m[1]}/{m[2]} tokens match" if precision == "f32" else f"greedy {m[1]}/{m[2]} decisive steps compared, all match")
                m = re.match(r"cached decode: (\d+) decode steps from position (\d+): max \|Δ\| = (\S+) to the stateless pass, at step (\d+)", line)
                if m:
                    figures.append(f"{m[1]} decode steps vs stateless max {short(m[3])} (step {m[4]}, tolerance 0.5)")
                m = re.match(r"prompt (\d+) decoded after a (\d+)-token prefill: \d+ tokens, max \|Δ\| = (\S+) at position (\d+) \(bound (\S+)\), mean \|Δ\| = (\S+) \(bound (\S+)\)", line)
                if m:
                    figures.append(f"p{m[1]} decoded after {m[2]}-token prefill max {short(m[3])}@{m[4]}/{short(m[5])} mean {short(m[6])}/{short(m[7])}")
            panic = re.search(r"panicked at [^\n]*\n([^\n]+)", text)
            detail = (f"{label}; " + "; ".join(figures)
                      + (f"; FAILED: {panic.group(1)}" if panic else "")
                      + f"; peak {peak:.2f} GB; wall {wall} s; log {log.name}")
            entries.append({"name": f"{name} {label} {test_name}", "status": "pass" if code == "0" else "fail",
                            "detail": detail})
json.dump(entries, sys.stdout, ensure_ascii=False, indent=0)
