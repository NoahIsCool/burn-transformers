"""Scratch (half2 S2): verify the new bundles against their truths and the round-2 bundles (not committed).

Usage: python3 -I verify_bundles.py KEY...

For each truth: every new graph bundle names the truth as its token source, carries
the frozen tool SHA-256, holds a prompt_decode record exactly for short truths,
and every new BF16 graph's prompt logits, greedy steps, and decode files are
byte-identical to the round-2 bundle of the same graph. For Qwen2-7B the F64
decode truth's prompt logits must equal the F64 truth's byte for byte. F16
finiteness is reported, not judged.
"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_bundles as mb
R = mb.REFERENCES
def sha(path):
    return mb.sha256(path)
failures = 0
def check(ok, message):
    global failures
    print(("ok   " if ok else "FAIL ") + message)
    failures += 0 if ok else 1
for key in sys.argv[1:]:
    truth_name, _, short = mb.CHECKPOINTS[key]
    truth_path = R / truth_name / "reference.json"
    truth = json.loads(truth_path.read_text()); truth_sha = sha(truth_path)
    print(f"== {key}: truth {truth_name} sha256 {truth_sha}")
    variants = [v for v in mb.VARIANTS if v != "f64-decode-truth"] + (["f64-decode-truth"] if key == "Qwen2-7B" else [])
    for variant in variants:
        root = R / mb.output_name(truth_name, variant)
        if not (root / "reference.json").exists():
            check(False, f"{root.name}: missing"); continue
        m = json.loads((root / "reference.json").read_text())
        p = m["provenance"]
        dtype, device, attention = mb.VARIANTS[variant]
        check((p["dtype"], p["device"], p["attention_implementation"], p["script_sha256"]) == (dtype, device, attention, mb.TOOL_SHA256),
              f"{root.name}: provenance {p['dtype']}/{p['device']}/{p['attention_implementation']} script {p['script_sha256'][:8]}")
        check([q["token_ids"] for q in m["prompts"]] == [q["token_ids"] for q in truth["prompts"]], f"{root.name}: token IDs equal the truth's")
        d = m["decode_consistency"]
        check(d["tokens_source"]["manifest_sha256"] == truth_sha, f"{root.name}: decode tokens_source is the truth")
        check([s["fed_token"] for s in d["steps"]] == [s["token"] for s in truth["greedy"]["steps"]], f"{root.name}: decode feeds the truth's tokens")
        has_pd = "prompt_decode" in m
        check(has_pd == (short and variant != "f64-decode-truth"), f"{root.name}: prompt_decode present={has_pd}")
        finite = all(q.get("finite", True) for q in m["prompts"]) and d.get("finite", True) and (not has_pd or m["prompt_decode"]["finite"])
        stopped = m["greedy"].get("stopped")
        print(f"     {root.name}: finite={finite} greedy={len(m['greedy']['steps'])}{' stopped=' + stopped if stopped else ''} decode max gap={d['max_abs_gap']}"
              + ("" if finite else f" first non-finite: " + json.dumps([q.get('first_non_finite') for q in m['prompts']] + [d.get('first_non_finite'), m.get('prompt_decode', {}).get('first_non_finite')])))
        if variant.startswith("bf16"):
            old_root = R / f"{truth_name}-{variant}"
            old = json.loads((old_root / "reference.json").read_text())
            same = all(q["logits_sha256"] == o["logits_sha256"] for q, o in zip(m["prompts"], old["prompts"]))
            same = same and all(sha(root / q["logits_file"]) == sha(old_root / o["logits_file"]) for q, o in zip(m["prompts"], old["prompts"]))
            check(same, f"{root.name}: prompt logits byte-identical to round 2 {old_root.name}")
            check(m["greedy"]["steps"] == old["greedy"]["steps"], f"{root.name}: greedy steps identical to round 2")
            od = old["decode_consistency"]
            check(sha(root / d["cached_logits_file"]) == sha(old_root / od["cached_logits_file"]) == od["cached_logits_sha256"] == d["cached_logits_sha256"],
                  f"{root.name}: decode-cached byte-identical to round 2")
            check(sha(root / d["stateless_logits_file"]) == sha(old_root / od["stateless_logits_file"]) == od["stateless_logits_sha256"] == d["stateless_logits_sha256"],
                  f"{root.name}: decode-stateless byte-identical to round 2")
        if variant == "f64-decode-truth":
            check(all(q["logits_sha256"] == t["logits_sha256"] for q, t in zip(m["prompts"], truth["prompts"])),
                  f"{root.name}: prompt logits byte-identical to the F64 truth")
        for name in [q["logits_file"] for q in m["prompts"]] + [d["cached_logits_file"], d["stateless_logits_file"]] + ([m["prompt_decode"]["logits_file"]] if has_pd else []):
            pass
        recorded = {q["logits_file"]: q["logits_sha256"] for q in m["prompts"]}
        recorded.update({d["cached_logits_file"]: d["cached_logits_sha256"], d["stateless_logits_file"]: d["stateless_logits_sha256"]})
        if has_pd:
            recorded[m["prompt_decode"]["logits_file"]] = m["prompt_decode"]["logits_sha256"]
        check(all(sha(root / f) == h for f, h in recorded.items()), f"{root.name}: every logits file matches its recorded digest")
print(f"== {failures} failures")
sys.exit(1 if failures else 0)
