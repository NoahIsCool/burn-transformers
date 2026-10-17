"""Scratch helper: one-line summaries of every Llama-family gate log (not committed)."""

import re
import sys
from pathlib import Path

ORDER = ["SmolLM-135M", "Qwen3-4B", "SmolLM3-3B", "Qwen2-7B-f64", "Llama-3.1-8B"]
root = Path(sys.argv[1])
for name in ORDER:
    for precision in ["f32", "bf16", "f16"]:
        for log in sorted(root.glob(f"{name}.{precision}.*.log")):
            text = log.read_text()
            label = re.search(r"^== (\S+/\S+) (\S+)$", text, re.M)
            exit_line = re.search(r"^== exit=(\d+) wall=([\d.]+)s", text, re.M)
            peak = re.search(r"(\d+)\s+peak memory footprint", text)
            parts = []
            for line in text.splitlines():
                line = re.sub(r"^test \S+ \.\.\. ", "", line)
                if re.match(r"(prompt \d+:|prompt \d+ decoded|batched row|cached prefill|cached decode:|greedy:)", line):
                    line = re.sub(r"(\d\.\d{4})\d*e", r"\1e", line)
                    parts.append(line)
                if "panicked" in line:
                    pass
            panic = re.search(r"panicked at [^\n]*\n([^\n]+)", text)
            status = "PASS" if exit_line and exit_line.group(1) == "0" else "FAIL"
            print(f"{name} {label.group(1)} {label.group(2)}: {status}; wall {exit_line.group(2)} s; "
                  f"peak {int(peak.group(1)) / 1e9:.2f} GB; " + " | ".join(parts)
                  + (f" | PANIC: {panic.group(1)}" if panic else ""))
