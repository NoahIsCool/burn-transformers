"""Scratch (round 2): official BF16 graph envelope, Burn's position in it, and candidate rules (not committed).

Usage: python3 -I analyze.py OUT_PREFIX

Reads, for each checkpoint, the truth bundle (F32; F64 for Qwen2-7B), the four
official BF16 graph bundles <truth>-bf16-{cpu,mps}-{eager,sdpa} and the F32
CPU-eager bundle <truth>-f32-cpu-eager written by the extended generator, the
Burn observations collected by collect_burn.py, the previous round's
defect-proxy logits (official graphs with Burn's twice-rounded SiLU or GELU),
and decode-defect sensitivity results. Every statistic is computed here from
the files: differences of two F32 values in F64, means summed in F64 (as
tests/support/half_precision_calibration.rs does). Writes OUT_PREFIX.json and the rendered tables
OUT_PREFIX.tables.txt.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_bundles  # noqa: E402

REFERENCES = make_bundles.REFERENCES
DIAG = Path("/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/mac-validation")
CALIBRATION = Path("/Users/nover/repos/transformers-worktrees/tfm-half-followups2/acceptance/measurements/half-precision-calibration-20261017.json")
SLACK = 1.2
GRAPHS = ["bf16-cpu-eager", "bf16-cpu-sdpa", "bf16-mps-eager", "bf16-mps-sdpa"]
SHORT = {"bf16-cpu-eager": "cpu-eager", "bf16-cpu-sdpa": "cpu-sdpa", "bf16-mps-eager": "mps-eager", "bf16-mps-sdpa": "mps-sdpa"}
KEYS = ["SmolLM-135M", "Qwen3-4B", "SmolLM3-3B", "Qwen2-7B", "Llama-3.1-8B", "gemma-3-4b-it", "gemma-long"]
#: Defect proxies of the previous round: key -> {label: directory holding prompt<i>/logits.bin}
DEFECT_PROXIES = {
    "SmolLM-135M": {f"official {g} + Burn SiLU": DIAG / "llama-family/diagnostics/data/SmolLM-135M-official-burnsilu" / f"{g}-burnsilu"
                    for g in ("cpu-bfloat16-eager", "cpu-bfloat16-sdpa", "mps-bfloat16-eager")},
    "SmolLM3-3B": {f"official {g} + Burn SiLU": DIAG / "llama-family/diagnostics/data/SmolLM3-3B-official" / f"{g}-burnsilu"
                   for g in ("cpu-bfloat16-eager", "cpu-bfloat16-sdpa", "mps-bfloat16-eager", "mps-bfloat16-sdpa")},
    "gemma-3-4b-it": {f"official {g} + Burn GELU": DIAG / "gemma/diagnostics/data/official" / f"{g}-burngelu"
                      for g in ("cpu-bfloat16-eager", "mps-bfloat16-eager")},
}
#: Burn prompt subjects per key, in collect_burn.py's labels.
BURN_PROMPT = {
    "gemma-3-4b-it": ["flex-bf16@53f167b", "flex-bf16@c2936fe", "metal-f16@53f167b", "metal-f16@c2936fe"],
    "gemma-long": ["flex-bf16@53f167b", "flex-bf16@c2936fe", "metal-f16@c2936fe"],
}
DEFAULT_BURN_PROMPT = ["flex-bf16@53f167b", "metal-f16@53f167b", "flex-bf16-f32silu@53f167b+scratch"]


def load(name):
    """Manifest and prompt logits of a bundle, as read-only F32 memory maps `[tokens, vocabulary]`."""
    root = REFERENCES / name
    manifest = json.loads((root / "reference.json").read_text())
    vocabulary = manifest["vocabulary_size"]
    rows = [np.memmap(root / p["logits_file"], dtype="<f4", mode="r", shape=(len(p["token_ids"]), vocabulary))
            for p in manifest["prompts"]]
    return manifest, rows


def row_maxima(actual, truth, chunk=64):
    """Per-row max |Δ| (F64 differences), in row chunks to bound memory."""
    out = []
    for start in range(0, truth.shape[0], chunk):
        a = np.asarray(actual[start:start + chunk], dtype=np.float64)
        t = np.asarray(truth[start:start + chunk], dtype=np.float64)
        out.append(np.abs(a - t).max(axis=1))
    return np.concatenate(out)


def decode_rows(name):
    """Cached and stateless decode rows (F64) and the record of a bundle."""
    root = REFERENCES / name
    manifest = json.loads((root / "reference.json").read_text())
    record = manifest["decode_consistency"]
    shape = record["logits_shape"]
    cached = np.fromfile(root / record["cached_logits_file"], dtype="<f4").astype(np.float64).reshape(shape)
    stateless = np.fromfile(root / record["stateless_logits_file"], dtype="<f4").astype(np.float64).reshape(shape)
    return manifest, record, cached, stateless


def measure(actual, truth, chunk=64):
    """Max |Δ| (first position and entry of the maximum) and mean |Δ| of one prompt; F64 differences and sum, in row chunks."""
    best, where, total = -1.0, (0, 0), 0.0
    for start in range(0, truth.shape[0], chunk):
        delta = np.abs(np.asarray(actual[start:start + chunk], dtype=np.float64) - np.asarray(truth[start:start + chunk], dtype=np.float64))
        if not np.isfinite(delta).all():
            return {"non_finite": True}
        flat = int(delta.argmax())
        if delta.flat[flat] > best:
            best, where = float(delta.flat[flat]), (start + flat // truth.shape[1], flat % truth.shape[1])
        total += float(delta.sum())
    return {"max": best, "position": where[0], "entry": where[1], "mean": total / (truth.shape[0] * truth.shape[1])}


def round_up(value, digits=2):
    """Rounds up to `digits` significant digits."""
    if value <= 0:
        return value
    exponent = math.floor(math.log10(value)) - digits + 1
    return math.ceil(value / 10**exponent - 1e-9) * 10**exponent


def verdict(observed, bound):
    """pass/FAIL text for a {max, mean} observation against a {max, mean} bound (inclusive)."""
    if observed is None:
        return "n/a"
    if observed.get("non_finite"):
        return "FAIL(non-finite)"
    failed = [stat for stat in ("max", "mean") if stat in bound and observed[stat] > bound[stat]]
    return "pass" if not failed else "FAIL(" + "+".join(failed) + ")"


def fmt(value):
    """Compact number."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if value == 0:
        return "0"
    return f"{value:.4g}"


def prompt_analysis(burn):
    """Per checkpoint and prompt: official graph deviations, envelope, Burn, defect proxies."""
    out = {}
    calibration = {Path(b["bundle"]).name: b for b in json.loads(CALIBRATION.read_text())["bundles"]}
    for key in KEYS:
        truth_name, _ = make_bundles.CHECKPOINTS[key]
        truth_manifest, truth = load(truth_name)
        graphs = {}
        for graph in GRAPHS:
            _, rows = load(f"{truth_name}-{graph}")
            graphs[SHORT[graph]] = [measure(o, t) for o, t in zip(rows, truth)]
        # Authentication: the CPU-eager statistics equal the committed calibration record's.
        record = calibration[f"{truth_name}-bf16"]
        calib_ok = all(graphs["cpu-eager"][i]["max"] == p["official_max_abs_delta"]
                       and abs(graphs["cpu-eager"][i]["mean"] - p["official_mean_abs_delta"]) <= 1e-12 * p["official_mean_abs_delta"]
                       for i, p in enumerate(record["prompts"]))
        prompts = []
        for index in range(len(truth)):
            per_graph = {g: graphs[g][index] for g in graphs}
            envelope = {stat: max(v[stat] for v in per_graph.values()) for stat in ("max", "mean")}
            loo = {g: {stat: max(per_graph[h][stat] for h in per_graph if h != g) for stat in ("max", "mean")} for g in per_graph}
            prompts.append({"tokens": len(truth_manifest["prompts"][index]["token_ids"]), "graphs": per_graph,
                            "envelope": envelope, "leave_one_out_envelope": loo})
        subjects = {}
        for label in BURN_PROMPT.get(key, DEFAULT_BURN_PROMPT):
            rows = burn["prompts"].get(key, {}).get(label)
            if rows is None:
                continue
            if rows.get("non_finite") and "gate" in rows:
                subjects[f"burn {label}"] = [{"non_finite": True} for _ in truth]
                continue
            subjects[f"burn {label}"] = [rows.get(str(i)) if str(i) in rows else None for i in range(len(truth))]
        for label, directory in DEFECT_PROXIES.get(key, {}).items():
            subjects[f"defect proxy: {label}"] = [
                measure(np.fromfile(directory / f"prompt{i}" / "logits.bin", dtype="<f4").reshape(t.shape), t)
                if (directory / f"prompt{i}" / "logits.bin").exists() else None for i, t in enumerate(truth)]
        out[key] = {"truth": truth_name, "truth_dtype": truth_manifest["provenance"]["dtype"], "prompts": prompts,
                    "subjects": subjects, "cpu_eager_equals_calibration_record": calib_ok}
    return out


def slack_statistics(prompts):
    """Ratios each official graph needs: over its leave-one-out envelope, and over CPU eager."""
    loo, over_ce = [], []
    for key, data in prompts.items():
        for index, prompt in enumerate(data["prompts"]):
            for graph, stats in prompt["graphs"].items():
                for stat in ("max", "mean"):
                    loo.append((stats[stat] / prompt["leave_one_out_envelope"][graph][stat], key, index, stat, graph))
                    over_ce.append((stats[stat] / prompt["graphs"]["cpu-eager"][stat], key, index, stat, graph))
    loo.sort(reverse=True)
    over_ce.sort(reverse=True)
    return loo, over_ce


def prompt_rules(prompts, s_loo, s_split):
    """Bounds and decisions of each candidate prompt rule."""
    split_name = f"R4 ({s_split['max']} max, {s_split['mean']} mean) x envelope"
    rules = {"R0 1.2 x cpu-eager (current)": lambda p, g=None: {k: SLACK * p["graphs"]["cpu-eager"][k] for k in ("max", "mean")},
             "R1 envelope": lambda p, g=None: dict(p["leave_one_out_envelope"][g] if g else p["envelope"]),
             "R2 1.2 x envelope": lambda p, g=None: {k: SLACK * v for k, v in (p["leave_one_out_envelope"][g] if g else p["envelope"]).items()},
             f"R3 {s_loo} x envelope": lambda p, g=None: {k: s_loo * v for k, v in (p["leave_one_out_envelope"][g] if g else p["envelope"]).items()},
             split_name: lambda p, g=None: {k: s_split[k] * v for k, v in (p["leave_one_out_envelope"][g] if g else p["envelope"]).items()}}
    table = []
    for key, data in prompts.items():
        for index, prompt in enumerate(data["prompts"]):
            rows = [(f"official {g} (leave-one-out)", g, prompt["graphs"][g]) for g in prompt["graphs"]]
            rows += [(label, None, values[index]) for label, values in data["subjects"].items()]
            for label, graph, observed in rows:
                decisions = {}
                for name, rule in rules.items():
                    if name.startswith("R0"):
                        decisions[name] = verdict(observed, rule(prompt))
                    else:
                        decisions[name] = verdict(observed, rule(prompt, graph))
                table.append({"key": key, "prompt": index, "subject": label, "observed": observed, "decisions": decisions})
    return list(rules), table


def decode_analysis(burn, defects):
    """Per checkpoint: official graphs' cached-vs-stateless gaps, their deviation from the F32 decode rows, Burn, rules."""
    out = {}
    for key in KEYS:
        truth_name, _ = make_bundles.CHECKPOINTS[key]
        truth_manifest, truth_prompts = load(truth_name)
        f32_manifest, f32_record, f32_cached, f32_stateless = decode_rows(f"{truth_name}-f32-cpu-eager")
        reference = f32_stateless  # F32 CPU eager stateless rows at the decode positions (the truth's code path)
        graphs = {}
        greedy = truth_manifest["greedy"]["steps"]
        bf16 = None
        for graph in GRAPHS:
            manifest, record, cached, stateless = decode_rows(f"{truth_name}-{graph}")
            gaps = np.abs(cached - stateless).max(axis=1)
            graphs[SHORT[graph]] = {
                "gap_per_step": [float(g) for g in gaps], "gap_max": float(gaps.max()), "gap_argmax": int(gaps.argmax()),
                "steps_above_0.5": int((gaps > 0.5).sum()),
                "stateless_dev_per_step": [float(v) for v in np.abs(stateless - reference).max(axis=1)],
                "cached_dev_per_step": [float(v) for v in np.abs(cached - reference).max(axis=1)],
                "stateless_dev_mean": float(np.abs(stateless - reference).sum() / reference.size),
                "cached_dev_mean": float(np.abs(cached - reference).sum() / reference.size),
                "recorded_max_abs_gap": record["max_abs_gap"],
            }
            graphs[SHORT[graph]]["dev_max"] = max(max(graphs[SHORT[graph]]["stateless_dev_per_step"]),
                                                  max(graphs[SHORT[graph]]["cached_dev_per_step"]))
            # Teacher-forced greedy agreement at the gate's decisive steps (margin > 2 s M0 of the CPU-eager twin).
            if graph == "bf16-cpu-eager":
                bf16 = manifest
            _, prompt_rows = load(f"{truth_name}-{graph}")
            graphs[SHORT[graph]]["predictions"] = {
                "stateless": [int(np.asarray(prompt_rows[0][-1]).argmax())] + [int(r.argmax()) for r in stateless[:-1]],
                "cached": [int(np.asarray(prompt_rows[0][-1]).argmax())] + [int(r.argmax()) for r in cached[:-1]]}
        prompt_ce = load(f"{truth_name}-bf16-cpu-eager")[1]
        dev_rows = [row_maxima(o, t) for o, t in zip(prompt_ce, truth_prompts)]
        m0 = float(dev_rows[0].max())
        m_all = float(max(r.max() for r in dev_rows))
        m_after0 = float(max(r[1:].max() for r in dev_rows if len(r) > 1))
        threshold = 2 * SLACK * m0
        decisive = [i for i, s in enumerate(greedy) if s["margin"] > threshold]
        for name, g in graphs.items():
            g["decisive_mismatches"] = {side: [i for i in decisive if g["predictions"][side][i] != greedy[i]["token"]]
                                        for side in ("stateless", "cached")}
        # Margin needed at each step for an official graph's argmax to be determined: twice its deviation there.
        W = max(g["gap_max"] for g in graphs.values())
        E = max(g["dev_max"] for g in graphs.values())
        E_ce = graphs["cpu-eager"]["dev_max"]
        f32_gaps = np.abs(f32_cached - f32_stateless).max(axis=1)
        f32_vs_truth_prompts = None
        if truth_manifest["provenance"]["dtype"] == "float32":
            f32_vs_truth_prompts = 0.0
        # Burn.
        burn_entry = burn["decode"].get(key, {})
        burn_subjects = {}
        for label, value in burn_entry.items():
            if label == "dumps" or value is None:
                continue
            if value.get("non_finite"):
                burn_subjects[f"burn gate {label}"] = {"gap": None, "non_finite": True, "kind": value["kind"]}
            else:
                burn_subjects[f"burn gate {label}"] = {"gap": value["value"], "kind": value["kind"], "step": value["step"]}
        dump_rows = {}
        for config, value in burn_entry.get("dumps", {}).items():
            dump_dir = Path(value["dump_dir"])
            vocabulary = truth_manifest["vocabulary_size"]
            first = len(truth_manifest["prompts"][0]["token_ids"])
            steps = len(greedy)
            plain = np.fromfile(dump_dir / "logits_plain.bin", dtype="<f4").astype(np.float64).reshape(first + steps, vocabulary)[first:]
            decoded = np.fromfile(dump_dir / "decode_logits.bin", dtype="<f4").astype(np.float64).reshape(-1, vocabulary)
            dump_rows[config] = {
                "gap_per_step": value["per_step"], "gap_max": value["max"],
                "stateless_dev_per_step": [float(v) for v in np.abs(plain - reference).max(axis=1)],
                "cached_dev_per_step": [float(v) for v in np.abs(decoded - reference).max(axis=1)],
                "stateless_dev_mean": float(np.abs(plain - reference).sum() / reference.size),
                "cached_dev_mean": float(np.abs(decoded - reference).sum() / reference.size),
                "source": value.get("dump_source")}
            dump_rows[config]["dev_max"] = max(max(dump_rows[config]["stateless_dev_per_step"]), max(dump_rows[config]["cached_dev_per_step"]))
            rerun = " (round-2 rerun of the same binary)" if "/round2/" in value["dump_dir"] else ""
            dump_rows[config]["rerun"] = bool(rerun)
            burn_subjects[f"burn dump {config}@{(value.get('dump_source') or '')[:7]}{rerun}"] = {"gap": value["max"], "kind": "max over steps (dump)"}
        out[key] = {"truth": truth_name, "steps": len(greedy), "first_position": f32_record["first_position"],
                    "graphs": graphs, "W": W, "E": E, "E_ce": E_ce, "M0_ce": m0, "M_all_ce": m_all, "M_after0_ce": m_after0,
                    "greedy_threshold": threshold, "decisive_steps": decisive,
                    "f32_gap_max": float(f32_gaps.max()), "f32_prompt_logits_equal_truth": f32_vs_truth_prompts,
                    "burn": burn_subjects, "burn_dumps": dump_rows, "defects": defects.get(key)}
    return out


def decode_rules(decode, s=SLACK):
    """Candidate decode self-consistency bounds and decisions."""
    names = ["D0 0.5 (current)", "D1 W", "D2 1.2W", "D3 2sE_ce", "D4 2sM'_ce", "D5 2sM0_ce"]
    table = []
    bounds = {}
    for key, d in decode.items():
        b = {"D0 0.5 (current)": 0.5, "D1 W": d["W"], "D2 1.2W": SLACK * d["W"], "D3 2sE_ce": 2 * s * d["E_ce"],
             "D4 2sM'_ce": 2 * s * d["M_after0_ce"], "D5 2sM0_ce": 2 * s * d["M0_ce"]}
        bounds[key] = b
        rows = []
        for graph, g in d["graphs"].items():
            loo_w = max(h["gap_max"] for name, h in d["graphs"].items() if name != graph)
            rows.append((f"official {graph}", g["gap_max"], {"D1 W": loo_w, "D2 1.2W": SLACK * loo_w}))
        for label, value in d["burn"].items():
            rows.append((label, value.get("gap"), {}))
        for defect_graph, defects in (d["defects"] or {}).items():
            for defect, value in defects.items():
                if defect != "none":
                    rows.append((f"defect {defect} on official {defect_graph}", value["gap_max"], {}))
        for label, gap, override in rows:
            decisions = {}
            for name in names:
                bound = override.get(name, b[name])
                decisions[name] = "n/a" if gap is None else ("pass" if gap <= bound else f"FAIL")
            table.append({"key": key, "subject": label, "gap": gap, "decisions": decisions})
    return names, bounds, table


def decode_row_rules(decode, split, s=SLACK):
    """Truth-anchored decode rows: each subject's cached and stateless rows at the decode positions against the F32 rows.

    Bounds: s times the official BF16 twin's (CPU eager) own max and mean deviation over the same rows
    ("T1 s x twin"), or s times the envelope of the official graphs (leave-one-out for an official graph, "T2 s x envelope").
    """
    table = []
    for key, d in decode.items():
        graphs = d["graphs"]

        def stats(g):
            """(max, mean) over both passes' decode rows."""
            return (max(max(g["stateless_dev_per_step"]), max(g["cached_dev_per_step"])),
                    max(g["stateless_dev_mean"], g["cached_dev_mean"]))

        twin = stats(graphs["cpu-eager"])
        subjects = [(f"official {name}", name, stats(g)) for name, g in graphs.items()]
        subjects += [(f"burn dump {config}@{(g.get('source') or '')[:7]}{' (round-2 rerun)' if g.get('rerun') else ''}", None, stats(g))
                     for config, g in d["burn_dumps"].items()]
        for label, name, (mx, mn) in subjects:
            others = [stats(g) for other, g in graphs.items() if other != name]
            env = (max(o[0] for o in others), max(o[1] for o in others))
            t1 = "pass" if mx <= s * twin[0] and mn <= s * twin[1] else "FAIL(" + "+".join(
                k for k, ok in (("max", mx <= s * twin[0]), ("mean", mn <= s * twin[1])) if not ok) + ")"
            t2 = "pass" if mx <= s * env[0] and mn <= s * env[1] else "FAIL(" + "+".join(
                k for k, ok in (("max", mx <= s * env[0]), ("mean", mn <= s * env[1])) if not ok) + ")"
            t3 = "pass" if mx <= split["max"] * env[0] and mn <= split["mean"] * env[1] else "FAIL(" + "+".join(
                k for k, ok in (("max", mx <= split["max"] * env[0]), ("mean", mn <= split["mean"] * env[1])) if not ok) + ")"
            table.append({"key": key, "subject": label, "max": mx, "mean": mn, "twin": twin, "envelope": env,
                          "T1 s x twin": t1, "T2 s x envelope": t2, "T3 split x envelope": t3})
    return table


def decode_row_slack(decode):
    """Leave-one-out ratios of the official graphs' decode-row statistics (max, mean over both passes)."""
    out = []
    for key, d in decode.items():
        stats = {name: (max(max(g["stateless_dev_per_step"]), max(g["cached_dev_per_step"])),
                        max(g["stateless_dev_mean"], g["cached_dev_mean"])) for name, g in d["graphs"].items()}
        for name, (mx, mn) in stats.items():
            others = [v for other, v in stats.items() if other != name]
            out.append((mx / max(o[0] for o in others), key, "decode rows", "max", name))
            out.append((mn / max(o[1] for o in others), key, "decode rows", "mean", name))
    return sorted(out, reverse=True)


def prompt_defect_table(prompts, s_values):
    """Injected prompt defects (prompt_defects.py) against the envelope rules and the current rule."""
    rows = []
    for path in sorted(HERE.glob("prompt-defects-*.json")):
        key = path.stem.replace("prompt-defects-", "")
        data = json.loads(path.read_text())
        if key not in prompts:
            continue
        for defect, values in data["defects"].items():
            for index, observed in enumerate(values):
                prompt = prompts[key]["prompts"][index]
                ce = prompt["graphs"]["cpu-eager"]
                env = prompt["envelope"]
                ratio = {stat: observed[stat] / env[stat] for stat in ("max", "mean")}
                decisions = {"R0 1.2 x cpu-eager (current)": verdict(observed, {k: SLACK * ce[k] for k in ("max", "mean")})}
                for s in s_values:
                    if isinstance(s, dict):
                        decisions[f"({s['max']} max, {s['mean']} mean) x envelope"] = verdict(observed, {k: s[k] * env[k] for k in ("max", "mean")})
                    else:
                        decisions[f"{s} x envelope"] = verdict(observed, {k: s * env[k] for k in ("max", "mean")})
                same = None
                if defect == "none":
                    same = observed["max"] == ce["max"] and abs(observed["mean"] - ce["mean"]) <= 1e-12 * ce["mean"]
                rows.append({"key": key, "prompt": index, "defect": defect, "observed": observed, "ratio_to_envelope": ratio,
                             "decisions": decisions, "none_equals_cpu_eager_bundle": same,
                             "script_sha256": data.get("script_sha256")})
    return rows


DECODED = __import__("re").compile(r"prompt (\d+) decoded after a (\d+)-token prefill: (\d+) tokens, max \|Δ\| = ([0-9.e+-]+) at position (\d+) .*mean \|Δ\| = ([0-9.e+-]+)")


def decoded_prompt_table(prompts, burn, decode, s_values):
    """The gate's second decode check (longest prompt, quarter prefill, then token by token) for official graphs and Burn."""
    rows = []
    for key, d in decode.items():
        if key not in prompts:
            continue
        subjects = []
        for graph, defects in (d["defects"] or {}).items():
            if "none" in defects:
                subjects.append((f"official {graph}", {"max": defects["none"]["decoded_prompt_max"], "mean": defects["none"]["decoded_prompt_mean"]}))
        for label, value in burn["decode"].get(key, {}).items():
            if label == "dumps" or not value or "log" not in value:
                continue
            match = DECODED.search(Path(value["log"]).read_text(errors="replace"))
            if match:
                subjects.append((f"burn gate {label}", {"max": float(match[4]), "mean": float(match[6]), "prompt": int(match[1])}))
        if not subjects:
            continue
        lengths = [p["tokens"] for p in prompts[key]["prompts"]]
        index = max(range(len(lengths)), key=lambda i: lengths[i])
        prompt = prompts[key]["prompts"][index]
        ce, env = prompt["graphs"]["cpu-eager"], prompt["envelope"]
        for label, observed in subjects:
            decisions = {"R0 1.2 x cpu-eager (current)": verdict(observed, {k: SLACK * ce[k] for k in ("max", "mean")})}
            for s in s_values:
                if isinstance(s, dict):
                    decisions[f"({s['max']} max, {s['mean']} mean) x envelope"] = verdict(observed, {k: s[k] * env[k] for k in ("max", "mean")})
                else:
                    decisions[f"{s} x envelope"] = verdict(observed, {k: s * env[k] for k in ("max", "mean")})
            rows.append({"key": key, "prompt": index, "subject": label, "observed": observed, "decisions": decisions})
    return rows


def main():
    prefix = Path(sys.argv[1])
    burn = json.loads((HERE / "burn_observations.json").read_text())
    defects = {}
    # Round 1's SmolLM-135M results first, so that round 2's (truth-bundle tokens) take precedence on shared graphs.
    for path in [HERE.parent / "decode-defects-SmolLM-135M.json"] + sorted(HERE.glob("decode-defects-*.json")):
        if path.exists():
            data = json.loads(path.read_text())
            key = path.stem.replace("decode-defects-", "")
            defects.setdefault(key, {}).update(data["graphs"])
    prompts = prompt_analysis(burn)
    loo, over_ce = slack_statistics(prompts)
    s_loo = round_up(loo[0][0])
    s_split = {"max": round_up(max(r[0] for r in loo if r[3] == "max")), "mean": round_up(max(r[0] for r in loo if r[3] == "mean"))}
    rule_names, prompt_table = prompt_rules(prompts, s_loo, s_split)
    decode = decode_analysis(burn, defects)
    decode_names, decode_bounds, decode_table = decode_rules(decode)
    row_table = decode_row_rules(decode, s_split)
    row_loo = decode_row_slack(decode)
    loo_max = [r for r in loo if r[3] == "max"]
    loo_mean = [r for r in loo if r[3] == "mean"]
    defect_rows = prompt_defect_table(prompts, [SLACK, s_loo, s_split])
    decoded_rows = decoded_prompt_table(prompts, burn, decode, [SLACK, s_loo, s_split])
    # Qwen2-7B: how far the F32 decode-row proxy can be from the F64 truth, measured on the prompts.
    f32_vs_truth = {}
    for key in KEYS:
        truth_name, _ = make_bundles.CHECKPOINTS[key]
        _, truth_rows = load(truth_name)
        _, f32_rows = load(f"{truth_name}-f32-cpu-eager")
        f32_vs_truth[key] = [measure(a, b) for a, b in zip(f32_rows, truth_rows)]
    result = {"prompts": prompts, "slack_leave_one_out": loo[:12], "slack_over_cpu_eager": over_ce[:12],
              "s_loo": s_loo, "s_split": s_split, "prompt_rules": rule_names, "prompt_decisions": prompt_table,
              "decode": decode, "decode_rules": decode_names, "decode_bounds": decode_bounds, "decode_decisions": decode_table,
              "decode_row_decisions": row_table, "decode_row_slack_leave_one_out": row_loo[:12],
              "slack_leave_one_out_max": loo_max[:6], "slack_leave_one_out_mean": loo_mean[:6],
              "prompt_defects": defect_rows, "decoded_prompt_decisions": decoded_rows, "f32_bundle_vs_truth": f32_vs_truth}
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=1, default=float))
    md = []
    md.append("# Official BF16 graph envelope (round 2)\n")
    md.append(f"Leave-one-out slack: largest ratio of an official graph's statistic to the maximum of the other three graphs = {loo[0][0]:.4f} "
              f"({loo[0][1]} prompt {loo[0][2]} {loo[0][3]}, {loo[0][4]}); rounded up to two digits: {s_loo}.")
    md.append("Next: " + "; ".join(f"{r:.3f} {k} p{i} {st} {g}" for r, k, i, st, g in loo[1:8]) + "\n")
    md.append("Ratio of an official graph's statistic to CPU eager's (what the current 1.2 rule must absorb): "
              + "; ".join(f"{r:.3f} {k} p{i} {st} {g}" for r, k, i, st, g in over_ce[:8]) + "\n")
    md.append("Leave-one-out by statistic: max " + "; ".join(f"{r:.3f} {k} p{i} {g}" for r, k, i, st, g in loo_max[:4])
              + " | mean " + "; ".join(f"{r:.3f} {k} p{i} {g}" for r, k, i, st, g in loo_mean[:4]) + "\n")
    md.append("Leave-one-out on the truth-anchored decode rows: " + "; ".join(f"{r:.3f} {k} {st} {g}" for r, k, _, st, g in row_loo[:6]) + "\n")
    md.append("F32 CPU-eager bundle vs truth on the prompts (0 means byte-identical; Qwen2-7B is against F64): "
              + "; ".join(f"{k} " + "/".join(fmt(v["max"]) for v in vals) for k, vals in f32_vs_truth.items()) + "\n")
    md.append("## Prompt statistics\n")
    md.append("| checkpoint | p | stat | cpu-eager | cpu-sdpa | mps-eager | mps-sdpa | envelope E | Burn Flex/BF16 | /E | Burn Metal/F16 | /E | 1.2 cpu-eager | E | 1.2E | " + f"{s_loo}E |")
    md.append("|" + "---|" * 16)
    for key, data in prompts.items():
        flex_label = "burn flex-bf16@c2936fe" if key.startswith("gemma") else "burn flex-bf16@53f167b"
        metal_label = next((l for l in data["subjects"] if l.startswith("burn metal")), None)
        for index, prompt in enumerate(data["prompts"]):
            for stat in ("max", "mean"):
                flex = data["subjects"].get(flex_label, [None] * 3)[index]
                metal = data["subjects"].get(metal_label, [None] * 3)[index] if metal_label else None
                fv = None if flex is None else ("NaN" if flex.get("non_finite") else flex[stat])
                mv = None if metal is None else ("NaN" if metal.get("non_finite") else metal[stat])
                env = prompt["envelope"][stat]
                md.append(f"| {key} | {index} | {stat} | " + " | ".join(fmt(prompt["graphs"][g][stat]) for g in ("cpu-eager", "cpu-sdpa", "mps-eager", "mps-sdpa"))
                          + f" | {fmt(env)} | {fmt(fv)} | {fmt(fv / env) if isinstance(fv, float) else '-'} | {fmt(mv)} | {fmt(mv / env) if isinstance(mv, float) else '-'}"
                          + f" | {fmt(SLACK * prompt['graphs']['cpu-eager'][stat])} | {fmt(env)} | {fmt(SLACK * env)} | {fmt(s_loo * env)} |")
    md.append("\n## Prompt-rule decisions (official graphs judged leave-one-out for envelope rules)\n")
    md.append("| checkpoint | p | subject | max | mean | " + " | ".join(rule_names) + " |")
    md.append("|" + "---|" * (5 + len(rule_names)))
    for row in prompt_table:
        o = row["observed"]
        mx = "-" if o is None else ("NaN" if o.get("non_finite") else fmt(o["max"]))
        mn = "-" if o is None else ("NaN" if o.get("non_finite") else fmt(o["mean"]))
        md.append(f"| {row['key']} | {row['prompt']} | {row['subject']} | {mx} | {mn} | " + " | ".join(row["decisions"][n] for n in rule_names) + " |")
    md.append("\n## Injected prompt defects on the official CPU-eager BF16 graph\n")
    md.append("| checkpoint | p | defect | max | mean | max/E | mean/E | " + " | ".join(defect_rows[0]["decisions"]) + " |" if defect_rows else "(no defect runs)")
    if defect_rows:
        md.append("|" + "---|" * (7 + len(defect_rows[0]["decisions"])))
        for row in defect_rows:
            o = row["observed"]
            md.append(f"| {row['key']} | {row['prompt']} | {row['defect']}{' (= bundle)' if row['none_equals_cpu_eager_bundle'] else ''} | {fmt(o['max'])} | {fmt(o['mean'])} | "
                      f"{row['ratio_to_envelope']['max']:.3f} | {row['ratio_to_envelope']['mean']:.3f} | " + " | ".join(row["decisions"].values()) + " |")
    md.append("\n## Decode self-consistency (max over steps of |cached - stateless|)\n")
    md.append("| checkpoint | steps | F32 cpu-eager | cpu-eager | cpu-sdpa | mps-eager | mps-sdpa | W | E (dev. from F32 rows) | E_ce | M0_ce | M'_ce (t>=1) | " + " | ".join(decode_names) + " |")
    md.append("|" + "---|" * (12 + len(decode_names)))
    for key, d in decode.items():
        md.append(f"| {key} | {d['steps']} | {fmt(d['f32_gap_max'])} | " + " | ".join(fmt(d["graphs"][g]["gap_max"]) for g in ("cpu-eager", "cpu-sdpa", "mps-eager", "mps-sdpa"))
                  + f" | {fmt(d['W'])} | {fmt(d['E'])} | {fmt(d['E_ce'])} | {fmt(d['M0_ce'])} | {fmt(d['M_after0_ce'])} | "
                  + " | ".join(fmt(decode_bounds[key][n]) for n in decode_names) + " |")
    md.append("\n## Decode-rule decisions (official graphs leave-one-out for D1/D2)\n")
    md.append("| checkpoint | subject | gap | " + " | ".join(decode_names) + " |")
    md.append("|" + "---|" * (3 + len(decode_names)))
    for row in decode_table:
        md.append(f"| {row['key']} | {row['subject']} | {fmt(row['gap'])} | " + " | ".join(row["decisions"][n] for n in decode_names) + " |")
    md.append("\n## Decoded longest prompt (quarter prefill, then token by token) against its prompt's bounds\n")
    if decoded_rows:
        names = list(decoded_rows[0]["decisions"])
        md.append("| checkpoint | p | subject | max | mean | " + " | ".join(names) + " |")
        md.append("|" + "---|" * (5 + len(names)))
        for row in decoded_rows:
            md.append(f"| {row['key']} | {row['prompt']} | {row['subject']} | {fmt(row['observed']['max'])} | {fmt(row['observed']['mean'])} | "
                      + " | ".join(row["decisions"][n] for n in names) + " |")
    md.append("\n## Truth-anchored decode rows (cached and stateless rows vs the F32 rows; s = 1.2)\n")
    md.append(f"| checkpoint | subject | max | mean | twin max | twin mean | T1 1.2 x twin | env max | env mean | T2 1.2 x envelope | T3 ({s_split['max']} max, {s_split['mean']} mean) x envelope |")
    md.append("|" + "---|" * 11)
    for row in row_table:
        md.append(f"| {row['key']} | {row['subject']} | {fmt(row['max'])} | {fmt(row['mean'])} | {fmt(row['twin'][0])} | {fmt(row['twin'][1])} | "
                  f"{row['T1 s x twin']} | {fmt(row['envelope'][0])} | {fmt(row['envelope'][1])} | {row['T2 s x envelope']} | {row['T3 split x envelope']} |")
    md.append("\n## Per-step gaps and deviations from the F32 decode rows\n")
    for key, d in decode.items():
        md.append(f"### {key} (decode from position {d['first_position']}; greedy threshold 2sM0 = {fmt(d['greedy_threshold'])}, decisive steps {len(d['decisive_steps'])})")
        for graph, g in d["graphs"].items():
            md.append(f"- {graph} gap: " + " ".join(f"{v:.3f}" for v in g["gap_per_step"]))
            md.append(f"- {graph} stateless dev: " + " ".join(f"{v:.3f}" for v in g["stateless_dev_per_step"]))
            md.append(f"- {graph} decisive-step argmax mismatches: stateless {g['decisive_mismatches']['stateless']}, cached {g['decisive_mismatches']['cached']}")
        for config, g in d["burn_dumps"].items():
            md.append(f"- burn {config} gap: " + " ".join(f"{v:.3f}" for v in g["gap_per_step"]))
            md.append(f"- burn {config} stateless dev: " + " ".join(f"{v:.3f}" for v in g["stateless_dev_per_step"]) + f" (max {g['dev_max']:.4f})")
        md.append("")
    prefix.with_suffix(".tables.txt").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
