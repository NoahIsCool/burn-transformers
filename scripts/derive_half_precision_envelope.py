#!/usr/bin/env python3
"""Derive the official-graph half-precision envelope and its leave-one-out slacks from reference bundles.

The half-precision gates (``tests/support/half_precision_calibration.rs``)
judge a Burn F16 or BF16 run against an F32 (or F64) *truth* bundle with
bounds taken from the official implementation itself: the four graphs that
Transformers runs in the run's dtype on the host, written by
``scripts/generate_causal_lm_reference.py``

* ``--device cpu --attn-implementation eager``,
* ``--device cpu --attn-implementation sdpa``,
* ``--device mps --attn-implementation eager``, and
* ``--device mps --attn-implementation sdpa``,

each with ``--decode-tokens-from TRUTH`` (decode rows at the truth's
teacher-forced greedy positions) and, for the short truths,
``--prompt-decode`` (the official model's own cached decode of the longest
prompt after a quarter-prompt prefill). This tool computes what the gate
reads, from the bundles alone, and the slacks the gate compiles as constants.

Statistics
----------

For logits ``a`` judged against reference logits ``r`` of the same shape
``[rows, vocabulary]`` the deviation is

``max = max |a - r|``  and  ``mean = (1 / (rows * vocabulary)) * sum |a - r|``,

with each difference of two F32 values taken in F64 (exact unless their
exponents differ by more than 29) and the sum accumulated in F64, as the Rust
``Deviation::measure`` does. Three kinds of rows are measured per graph:

``prompt``
    every prompt's logits against the truth's prompt logits;
``decode``
    the graph's cached and stateless decode rows against the *decode truth*:
    the stateless rows of an F32 (F64 for an F64 truth) CPU-eager bundle at
    the same positions, whose prompt logits equal the truth's byte for byte.
    A graph's decode statistic is the larger of its two passes', since the
    gate holds Burn's cached and stateless rows to the same bound;
``prompt_decode``
    the graph's quarter-prefill decode of the longest prompt against the
    truth's logits of that prompt.

The envelope ``E`` of a statistic is its largest value over the four graphs.

Slacks
------

A gate passes a run whose maximum is at most ``s_max E_max`` and whose mean
is at most ``s_mean E_mean``. The slacks are derived from the official
graphs only, never from Burn: each graph's *leave-one-out* ratio is its
statistic over the largest of the other three graphs' (how far the envelope
of three official graphs must stretch to admit the fourth), and ``s_max`` and
``s_mean`` are the largest such ratios of maxima and of means, pooled over
every prompt, decode, and prompt-decode statistic of every calibrated case
of one dtype, rounded up to two significant digits (1.4206 becomes 1.5,
1.2988 becomes 1.3, 1.3 stays 1.3). Rounding works on the shortest decimal
that round-trips the F64 ratio, so a ratio that prints as 1.3 is not pushed
to 1.4 by its binary representation. The slacks must be re-derived, never
tuned, whenever a graph, a version, a prompt, or a checkpoint changes.

Uncalibrated cases
------------------

If any official graph of a case holds a non-finite logit in any of its
measured rows, that case has no envelope in this dtype: an official
evaluation that overflows marks the checkpoint as range-marginal there, and a
finite Burn result would reflect kernel luck rather than fidelity. The case
is reported as uncalibrated, with the first non-finite entry of each such
graph, and is left out of the pooled slacks; the gate fails such a run as
uncalibrated and never passes it.

Pairing
-------

Every case is checked fail-closed before any logit is read, as the Rust
``validate_envelope`` does: the truth is a ``float32`` or ``float64``
evaluation; each graph is a plain (not ``--bf16-resident``) evaluation in the
requested dtype; the four graphs are exactly the four ``(device,
attention_implementation)`` pairs, each once; they share one generator
script SHA-256; each agrees with the truth on model type and class,
configuration and weight digests, repository and revision, Transformers and
PyTorch versions, vocabulary, layer truncation, every prompt's text and
token IDs, and the greedy length (shorter only for a graph whose
continuation stopped on a non-finite step); each decode record feeds the
truth's greedy tokens at positions ``T, ..., T + steps - 1`` and names the
truth's exact manifest digest as its token source; and the decode truth is a
``float32`` or ``float64`` CPU-eager evaluation whose prompt-logit digests
equal the truth's. Prompt-decode records, when present, must be present in
every graph, name the longest prompt (the first of equal ones), and prefill
``max(T // 4, 1)`` tokens. Every logits file must match its recorded digest
and size.

Usage::

    python3 scripts/derive_half_precision_envelope.py --dtype bfloat16 \\
        --cases cases.json --output envelope-bf16.json

where ``cases.json`` lists ``{"name", "truth", "decode_truth", "graphs"}``
objects (directories; ``graphs`` holds four).
"""

from __future__ import annotations

import argparse
import decimal
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Bundle format version the generator writes.
SCHEMA_VERSION = 1
#: Run dtypes whose official graphs this tool calibrates.
RUN_DTYPES = ("bfloat16", "float16")
#: Dtypes a truth or decode truth may be evaluated in.
TRUTH_DTYPES = ("float32", "float64")
#: The four official graphs, as (device, attention implementation).
GRAPHS = (("cpu", "eager"), ("cpu", "sdpa"), ("mps", "eager"), ("mps", "sdpa"))
#: Statistics every kind of row reports.
STATISTICS = ("max", "mean")
#: Row kinds measured per graph.
KINDS = ("prompt", "decode", "prompt_decode")
#: Rows read at once, which bounds the memory of one comparison.
CHUNK_ROWS = 64


class PairingError(ValueError):
    """A bundle that cannot be part of the envelope it was named for."""


def sha256_file(path: Path) -> str:
    """Hashes a file's exact bytes in 16 MiB blocks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def round_up(value: float, digits: int = 2) -> float:
    """Rounds a positive `value` up to `digits` significant digits, on its shortest round-trip decimal.

    ``repr`` gives the shortest decimal that reads back as `value`, so a ratio
    that is 1.3 to F64 precision rounds to 1.3, while 1.30000001 rounds to 1.4.
    """

    if not (value > 0 and math.isfinite(value)):
        raise ValueError(f"a slack must be positive and finite, not {value!r}")
    exact = decimal.Decimal(repr(value))
    exponent = exact.adjusted() - digits + 1
    quantum = decimal.Decimal(1).scaleb(exponent)
    return float(exact.quantize(quantum, rounding=decimal.ROUND_CEILING))


class Bundle:
    """One reference bundle: its directory, parsed manifest, and the manifest's SHA-256."""

    def __init__(self, root: Path):
        """Reads ``root/reference.json`` and checks its schema."""

        self.root = Path(root)
        path = self.root / "reference.json"
        payload = path.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        self.manifest = json.loads(payload)
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise PairingError(f"{path}: unsupported bundle schema {self.manifest.get('schema_version')!r}")

    @property
    def provenance(self) -> dict[str, Any]:
        """The manifest's provenance record."""

        return self.manifest["provenance"]

    @property
    def dtype(self) -> str:
        """The evaluation dtype label."""

        return self.provenance["dtype"]

    @property
    def graph(self) -> tuple[str, str]:
        """The `(device, attention implementation)` the bundle was evaluated with (CPU eager when unrecorded)."""

        return (self.provenance.get("device", "cpu"), self.provenance.get("attention_implementation", "eager"))

    @property
    def vocabulary(self) -> int:
        """The vocabulary size."""

        return int(self.manifest["vocabulary_size"])

    @property
    def prompts(self) -> list[dict[str, Any]]:
        """The prompt records."""

        return self.manifest["prompts"]

    @property
    def greedy(self) -> dict[str, Any]:
        """The greedy continuation record."""

        return self.manifest["greedy"]

    def name(self) -> str:
        """The bundle's directory name, for messages."""

        return self.root.name

    def rows(self, file: str, digest: str, shape: Sequence[int]) -> Any:
        """Returns a read-only F32 memory map of a logits file after checking its digest and size."""

        import numpy

        path = self.root / file
        if Path(file).name != file or file in ("", ".", ".."):
            raise PairingError(f"{self.name()}: {file!r} is not a plain file name")
        expected = int(shape[0]) * int(shape[1]) * 4
        if path.stat().st_size != expected:
            raise PairingError(f"{path} must hold {list(shape)} F32 logits")
        if sha256_file(path) != digest:
            raise PairingError(f"{path} is corrupted")
        return numpy.memmap(path, dtype="<f4", mode="r", shape=(int(shape[0]), int(shape[1])))

    def prompt_rows(self, index: int) -> Any:
        """Returns prompt `index`'s `[tokens, vocabulary]` logits."""

        prompt = self.prompts[index]
        return self.rows(prompt["logits_file"], prompt["logits_sha256"], prompt["logits_shape"])


def deviation(actual: Any, reference: Any) -> dict[str, Any]:
    """Measures `actual` against `reference` (equal `[rows, vocabulary]` F32 arrays) in F64.

    Returns ``max`` with its first ``position`` (row) and ``token`` and
    ``mean``; or ``{"finite": False, "first_non_finite": {...}}`` when
    `actual` holds a NaN or an infinity, naming its first such entry in
    row-major order. A non-finite `reference` raises ``PairingError``: truths
    must be finite.
    """

    import numpy

    if tuple(actual.shape) != tuple(reference.shape) or actual.size == 0:
        raise PairingError(f"compared logits have shapes {tuple(actual.shape)} and {tuple(reference.shape)}")
    best, where, total = -1.0, (0, 0), 0.0
    columns = reference.shape[1]
    for start in range(0, reference.shape[0], CHUNK_ROWS):
        left = numpy.asarray(actual[start:start + CHUNK_ROWS], dtype=numpy.float64)
        right = numpy.asarray(reference[start:start + CHUNK_ROWS], dtype=numpy.float64)
        if not numpy.isfinite(right).all():
            raise PairingError("the reference logits hold a non-finite value")
        finite = numpy.isfinite(left)
        if not finite.all():
            flat = int(numpy.flatnonzero(~finite)[0])
            return {"finite": False,
                    "first_non_finite": {"position": start + flat // columns, "token": flat % columns}}
        delta = numpy.abs(left - right)
        flat = int(delta.argmax())
        if float(delta.flat[flat]) > best:
            best, where = float(delta.flat[flat]), (start + flat // columns, flat % columns)
        total += float(delta.sum())
    return {"max": best, "position": where[0], "token": where[1], "mean": total / actual.size}


def is_finite(statistic: dict[str, Any]) -> bool:
    """Whether a measured statistic came from finite logits."""

    return statistic.get("finite", True)


def require(condition: bool, message: str) -> None:
    """Raises ``PairingError(message)`` unless `condition` holds."""

    if not condition:
        raise PairingError(message)


def validate_truth(truth: Bundle) -> None:
    """Requires a plain float32 or float64 truth with finite-by-construction records."""

    require(truth.dtype in TRUTH_DTYPES, f"{truth.name()}: the truth must be float32 or float64, not {truth.dtype}")
    require(truth.greedy.get("prompt_index") == 0, f"{truth.name()}: the greedy continuation must extend prompt 0")
    require(not truth.greedy.get("stopped"), f"{truth.name()}: the truth's greedy continuation stopped early")


def same_identity(truth: Bundle, other: Bundle) -> None:
    """Requires `other` to evaluate the truth's checkpoint, versions, prompts, and options."""

    t, o = truth.provenance, other.provenance
    pairs = [
        ("model type", truth.manifest["model_type"], other.manifest["model_type"]),
        ("model class", truth.manifest["model_class"], other.manifest["model_class"]),
        ("config.json SHA-256", t["config_sha256"], o["config_sha256"]),
        ("weight files", t["weights_sha256"], o["weights_sha256"]),
        ("repository", t.get("repository"), o.get("repository")),
        ("revision", t.get("revision"), o.get("revision")),
        ("Transformers version", t["transformers"], o["transformers"]),
        ("PyTorch version", t["torch"], o["torch"]),
        ("vocabulary size", truth.vocabulary, other.vocabulary),
        ("layer truncation", t.get("num_hidden_layers"), o.get("num_hidden_layers")),
        ("prompt count", len(truth.prompts), len(other.prompts)),
    ]
    for what, expected, actual in pairs:
        require(expected == actual, f"{other.name()} and the truth {truth.name()} differ in {what}")
    for index, (expected, actual) in enumerate(zip(truth.prompts, other.prompts)):
        require(expected["text"] == actual["text"], f"{other.name()}: prompt {index} text differs from the truth's")
        require(expected["token_ids"] == actual["token_ids"],
                f"{other.name()}: prompt {index} token IDs differ from the truth's")


def validate_decode_record(truth: Bundle, bundle: Bundle) -> dict[str, Any]:
    """Requires `bundle`'s decode record to feed the truth's greedy tokens at the truth's decode positions."""

    record = bundle.manifest.get("decode_consistency")
    require(record is not None, f"{bundle.name()}: no decode record (generate it with --decode-tokens-from)")
    tokens = [step["token"] for step in truth.greedy["steps"]]
    first = len(truth.prompts[0]["token_ids"])
    require(record["prompt_index"] == 0 and record["first_position"] == first,
            f"{bundle.name()}: decode rows must start at position {first} after prompt 0")
    require([step["position"] for step in record["steps"]] == list(range(first, first + len(tokens))),
            f"{bundle.name()}: decode positions must be {first}..{first + len(tokens) - 1}")
    require([step["fed_token"] for step in record["steps"]] == tokens,
            f"{bundle.name()}: decode rows must feed the truth's greedy tokens")
    require(record["tokens_source"]["manifest_sha256"] == truth.sha256,
            f"{bundle.name()}: decode tokens_source must be the truth's manifest {truth.sha256}")
    require(list(record["logits_shape"]) == [len(tokens), truth.vocabulary],
            f"{bundle.name()}: decode rows must be [{len(tokens)}, {truth.vocabulary}]")
    return record


def validate_decode_truth(truth: Bundle, decode_truth: Bundle) -> None:
    """Requires the decode truth to be the truth's own graph in a truth dtype, with equal prompt logits."""

    require(decode_truth.dtype in TRUTH_DTYPES,
            f"{decode_truth.name()}: the decode truth must be float32 or float64, not {decode_truth.dtype}")
    require(decode_truth.graph == ("cpu", "eager"), f"{decode_truth.name()}: the decode truth must be CPU eager")
    require(not decode_truth.provenance.get("bf16_resident"), f"{decode_truth.name()}: not a plain evaluation")
    same_identity(truth, decode_truth)
    for index, (expected, actual) in enumerate(zip(truth.prompts, decode_truth.prompts)):
        require(expected["logits_sha256"] == actual["logits_sha256"],
                f"{decode_truth.name()}: prompt {index} logits differ from the truth's, so it is not the truth's graph")
    validate_decode_record(truth, decode_truth)


def validate_envelope(truth: Bundle, decode_truth: Bundle, graphs: Sequence[Bundle], dtype: str) -> None:
    """Checks the pairing described in the module documentation; raises ``PairingError`` on the first problem."""

    require(dtype in RUN_DTYPES, f"run dtype must be one of {RUN_DTYPES}, not {dtype!r}")
    validate_truth(truth)
    validate_decode_truth(truth, decode_truth)
    require(len(graphs) == len(GRAPHS), f"an envelope needs exactly {len(GRAPHS)} official graphs, not {len(graphs)}")
    seen = [bundle.graph for bundle in graphs]
    for graph in GRAPHS:
        count = seen.count(graph)
        require(count == 1, f"official graph {graph[0]}/{graph[1]} appears {count} times; each must appear once")
    scripts = {bundle.provenance.get("script_sha256") for bundle in graphs}
    require(len(scripts) == 1, f"the official graphs come from {len(scripts)} generator scripts; they must share one")
    decodes = []
    for bundle in graphs:
        require(bundle.dtype == dtype, f"{bundle.name()}: a {bundle.dtype} evaluation cannot calibrate a {dtype} run")
        require(not bundle.provenance.get("bf16_resident"), f"{bundle.name()}: must not be --bf16-resident")
        same_identity(truth, bundle)
        steps, expected = len(bundle.greedy["steps"]), len(truth.greedy["steps"])
        stopped = bundle.greedy.get("stopped") == "non_finite"
        require(steps == expected or (stopped and steps < expected),
                f"{bundle.name()}: greedy length {steps} differs from the truth's {expected}")
        validate_decode_record(truth, bundle)
        decodes.append(bundle.manifest.get("prompt_decode"))
    if any(record is not None for record in decodes):
        lengths = [len(prompt["token_ids"]) for prompt in truth.prompts]
        longest = max(range(len(lengths)), key=lambda index: (lengths[index], -index))
        for bundle, record in zip(graphs, decodes):
            require(record is not None, f"{bundle.name()}: no prompt_decode record, which the other graphs hold")
            require(record["prompt_index"] == longest,
                    f"{bundle.name()}: prompt_decode must decode the longest prompt {longest}")
            require(record["prefill_tokens"] == max(lengths[longest] // 4, 1),
                    f"{bundle.name()}: prompt_decode must prefill max(T // 4, 1) tokens")
            require(list(record["logits_shape"]) == [lengths[longest], truth.vocabulary],
                    f"{bundle.name()}: prompt_decode rows must cover prompt {longest}")


def decode_rows(bundle: Bundle) -> tuple[Any, Any]:
    """Returns a bundle's cached and stateless decode rows."""

    record = bundle.manifest["decode_consistency"]
    shape = record["logits_shape"]
    return (bundle.rows(record["cached_logits_file"], record["cached_logits_sha256"], shape),
            bundle.rows(record["stateless_logits_file"], record["stateless_logits_sha256"], shape))


def larger(statistics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Combines several passes' statistics: the first non-finite one, or the largest max and mean."""

    statistics = list(statistics)
    for statistic in statistics:
        if not is_finite(statistic):
            return statistic
    return {stat: max(statistic[stat] for statistic in statistics) for stat in STATISTICS}


def measure_case(truth: Bundle, decode_truth: Bundle, graphs: Sequence[Bundle]) -> dict[str, Any]:
    """Measures every graph's prompt, decode, and prompt-decode rows; returns the per-graph statistics."""

    measured: dict[str, dict[str, Any]] = {}
    _, reference_decode = decode_rows(decode_truth)
    for bundle in graphs:
        label = f"{bundle.graph[0]}-{bundle.graph[1]}"
        entry: dict[str, Any] = {"bundle": str(bundle.root), "reference_json_sha256": bundle.sha256}
        entry["prompt"] = [deviation(bundle.prompt_rows(index), truth.prompt_rows(index))
                           for index in range(len(truth.prompts))]
        cached, stateless = decode_rows(bundle)
        passes = {"cached": deviation(cached, reference_decode), "stateless": deviation(stateless, reference_decode)}
        entry["decode_passes"] = passes
        entry["decode"] = larger(passes.values())
        record = bundle.manifest.get("prompt_decode")
        if record is not None:
            rows = bundle.rows(record["logits_file"], record["logits_sha256"], record["logits_shape"])
            entry["prompt_decode"] = deviation(rows, truth.prompt_rows(record["prompt_index"]))
        measured[label] = entry
    return measured


def statistics_of(measured: dict[str, Any]) -> list[tuple[str, int | None, dict[str, dict[str, Any]]]]:
    """Lists `(kind, prompt index or None, {graph: statistic})` for every statistic a case defines."""

    graphs = list(measured)
    out = []
    for index in range(len(measured[graphs[0]]["prompt"])):
        out.append(("prompt", index, {graph: measured[graph]["prompt"][index] for graph in graphs}))
    out.append(("decode", None, {graph: measured[graph]["decode"] for graph in graphs}))
    if all("prompt_decode" in measured[graph] for graph in graphs):
        out.append(("prompt_decode", None, {graph: measured[graph]["prompt_decode"] for graph in graphs}))
    return out


def envelope_of(per_graph: dict[str, dict[str, Any]]) -> dict[str, float]:
    """The envelope of finite statistics: the largest max and mean over the graphs."""

    return {stat: max(statistic[stat] for statistic in per_graph.values()) for stat in STATISTICS}


def leave_one_out(per_graph: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Each graph's statistic over the largest of the other graphs' statistics, per statistic.

    A zero in the other graphs' maximum is refused: three official graphs equal
    to the truth would not be half-precision evaluations.
    """

    ratios = {}
    for graph, statistic in per_graph.items():
        ratios[graph] = {}
        for stat in STATISTICS:
            others = max(other[stat] for name, other in per_graph.items() if name != graph)
            require(others > 0, f"the official graphs other than {graph} have zero {stat} deviation")
            ratios[graph][stat] = statistic[stat] / others
    return ratios


def analyze_case(name: str, truth: Bundle, decode_truth: Bundle, graphs: Sequence[Bundle], dtype: str) -> dict[str, Any]:
    """Validates and measures one case; returns its envelopes, leave-one-out ratios, or uncalibrated report."""

    validate_envelope(truth, decode_truth, graphs, dtype)
    measured = measure_case(truth, decode_truth, graphs)
    report: dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "truth": {"bundle": str(truth.root), "dtype": truth.dtype, "reference_json_sha256": truth.sha256},
        "decode_truth": {"bundle": str(decode_truth.root), "dtype": decode_truth.dtype,
                         "reference_json_sha256": decode_truth.sha256},
        "script_sha256": graphs[0].provenance.get("script_sha256"),
        "graphs": measured,
    }
    uncalibrated = []
    for kind, index, per_graph in statistics_of(measured):
        for graph, statistic in per_graph.items():
            if not is_finite(statistic):
                uncalibrated.append({"graph": graph, "kind": kind, "prompt": index,
                                     "first_non_finite": statistic["first_non_finite"]})
    if uncalibrated:
        report["calibrated"] = False
        report["uncalibrated"] = uncalibrated
        return report
    report["calibrated"] = True
    report["statistics"] = [
        {"kind": kind, "prompt": index, "envelope": envelope_of(per_graph), "leave_one_out": leave_one_out(per_graph)}
        for kind, index, per_graph in statistics_of(measured)
    ]
    return report


def pooled_slacks(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pools the leave-one-out ratios of every calibrated case into ranked lists and rounded slacks."""

    ratios = {stat: [] for stat in STATISTICS}
    for case in cases:
        if not case["calibrated"]:
            continue
        for statistic in case["statistics"]:
            for graph, values in statistic["leave_one_out"].items():
                for stat in STATISTICS:
                    ratios[stat].append({"ratio": values[stat], "case": case["name"], "kind": statistic["kind"],
                                         "prompt": statistic["prompt"], "graph": graph})
    result: dict[str, Any] = {}
    for stat in STATISTICS:
        ranked = sorted(ratios[stat], key=lambda entry: entry["ratio"], reverse=True)
        require(bool(ranked), "no calibrated case to derive a slack from")
        by_kind = {kind: max((entry["ratio"] for entry in ranked if entry["kind"] == kind), default=None)
                   for kind in KINDS}
        result[stat] = {"largest": ranked[:8], "count": len(ranked), "largest_by_kind": by_kind,
                        "slack": round_up(ranked[0]["ratio"])}
    return result


def derive(cases: Sequence[dict[str, Any]], dtype: str) -> dict[str, Any]:
    """Analyzes every case of one dtype and derives its slacks."""

    reports = [
        analyze_case(case["name"], Bundle(Path(case["truth"])), Bundle(Path(case["decode_truth"])),
                     [Bundle(Path(graph)) for graph in case["graphs"]], dtype)
        for case in cases
    ]
    calibrated = [report for report in reports if report["calibrated"]]
    return {
        "dtype": dtype,
        "cases": reports,
        "uncalibrated": [report["name"] for report in reports if not report["calibrated"]],
        "slacks": pooled_slacks(calibrated) if calibrated else None,
    }


def label(kind: str, prompt: int | None) -> str:
    """Names a statistic in rendered tables: ``prompt 1``, ``decode rows``, or ``prompt-decode rows``."""

    return f"prompt {prompt}" if kind == "prompt" else kind.replace("_", "-") + " rows"


def render(result: dict[str, Any]) -> str:
    """Renders the derivation as plain-text tables."""

    lines = [f"# Official {result['dtype']} graph envelope"]
    for case in result["cases"]:
        lines.append(f"\n## {case['name']} ({'calibrated' if case['calibrated'] else 'UNCALIBRATED'})")
        if not case["calibrated"]:
            for entry in case["uncalibrated"]:
                where = entry["first_non_finite"]
                lines.append(f"- official {entry['graph']} {result['dtype']} non-finite in {label(entry['kind'], entry['prompt'])} "
                             f"at position {where['position']}, token {where['token']}")
            continue
        for statistic in case["statistics"]:
            envelope = statistic["envelope"]
            loo = "; ".join(f"{graph} {values['max']:.4f}/{values['mean']:.4f}"
                            for graph, values in statistic["leave_one_out"].items())
            lines.append(f"- {label(statistic['kind'], statistic['prompt'])}: E max {envelope['max']:.6g}, mean {envelope['mean']:.6g}; "
                         f"leave-one-out max/mean: {loo}")
    if result["slacks"]:
        for stat in STATISTICS:
            slack = result["slacks"][stat]
            top = slack["largest"][0]
            by_kind = ", ".join(f"{kind} {value:.4f}" for kind, value in slack["largest_by_kind"].items()
                                if value is not None)
            lines.append(f"\n{stat} slack {slack['slack']} from {top['ratio']:.4f} ({top['case']} {label(top['kind'], top['prompt'])}, "
                         f"{top['graph']}); by kind: {by_kind}; {slack['count']} ratios pooled")
    if result["uncalibrated"]:
        lines.append("\nuncalibrated: " + ", ".join(result["uncalibrated"]))
    return "\n".join(lines) + "\n"


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", choices=RUN_DTYPES, required=True, help="run dtype of the official graphs")
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of {name, truth, decode_truth, graphs}")
    parser.add_argument("--output", type=Path, required=True, help="JSON result to write")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Derives the envelope of the listed cases, writes the JSON result, and prints the tables."""

    arguments = parse_arguments(argv)
    cases = json.loads(arguments.cases.read_text())
    result = derive(cases, arguments.dtype)
    arguments.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(render(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
