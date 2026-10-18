"""One-off: the C2 generator writes the same bytes as C1 (c2c7165) on the tiny fixtures for default options."""
import importlib.util, json, sys, tempfile
from pathlib import Path
from unittest import mock
WT = Path(sys.argv[1])
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
c1 = load(Path(__file__).parent / "c1_generator.py", "c1")
c2 = load(WT / "scripts/generate_causal_lm_reference.py", "c2")
T = [1, 5, 9, 14, 3, 27, 40, 8]
class Tok:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": {"a": T[:4], "b": T, "c": [2, 4]}[text]}
results = []
for fixture in ("qwen2", "gpt-oss", "gemma3-text"):
    F = WT / "tests/fixtures/tiny_goldens" / fixture
    for dtype in ("float32", "bfloat16"):
        for graph in (("cpu", "eager"), ("cpu", "sdpa"), ("mps", "eager"), ("mps", "sdpa")):
            if fixture == "gpt-oss" and graph != ("cpu", "eager"):
                continue  # Transformers 5.2.0 runs gpt-oss on the eager CPU graph only (no SDPA; no MPS histc for Int)
            outs = []
            with tempfile.TemporaryDirectory() as root:
                root = Path(root)
                truth = root / "truth"; truth.mkdir()
                (truth / "reference.json").write_text(json.dumps({"prompts": [{"token_ids": T[:4]}], "greedy": {"prompt_index": 0, "steps": [{"token": t, "margin": 1.0} for t in T[4:]]}, "provenance": {"dtype": "float32"}}))
                for mod in (c1, c2):
                    model = mod.load_causal_lm(F, dtype, attention=graph[1], device=graph[0])
                    staging = root / mod.__name__; staging.mkdir()
                    args = ["--model-dir", str(F), "--output", str(root / "o"), "--prompt", "a", "--prompt", "b", "--prompt", "c", "--greedy-tokens", "6", "--dtype", dtype, "--device", graph[0], "--attn-implementation", graph[1], "--decode-tokens-from", str(truth)]
                    with mock.patch.object(mod, "load_model", return_value=(Tok(), model)):
                        manifest = mod.build_bundle(mod.parse_arguments(args), staging)
                    files = {p.name: p.read_bytes() for p in staging.iterdir()}
                    for key in ("provenance",):
                        manifest.pop(key)
                    manifest["decode_consistency"].pop("finite", None)
                    outs.append((files, manifest))
            same = outs[0][0] == outs[1][0] and outs[0][1] == outs[1][1]
            results.append((fixture, dtype, graph, same, sorted(outs[0][0])))
            print(fixture, dtype, graph, "IDENTICAL" if same else "DIFFERENT", len(outs[0][0]), "files")
print("ALL IDENTICAL" if all(r[3] for r in results) else "SOME DIFFER")
