#!/usr/bin/env python3
"""Derive the Nemotron-H official F16 envelope and its per-path leave-one-out slacks from the official graphs alone.

The Nemotron-H official-precision gate (``tests/nemotron_h_artifacts.rs``)
judges each Burn F16 logit row by its cosine to the official class's F32 row
of the same path and decision (the precision bundle), against a per-path
floor: the worst cosine of the two official F16 evaluations it is
calibrated on (the MPS/F16 oracle and the precision bundle's CPU/F16 rows),
with no slack. This tool applies the half-precision envelope round's
procedure (``scripts/derive_half_precision_envelope.py``) to the gate's row
statistic over four official graphs, by the method registered before any new
data in
``acceptance/measurements/nemotron-h-envelope-20261018/derivation/method.txt``.
The maintainer did not adopt the resulting rule (decision of 2026-10-18,
``acceptance/measurements/nemotron-h-envelope-decision-20261018.json``):
every largest leave-one-out ratio is at most ``1.0046``, so every slack is
``1.1`` only by rounding. Its output is a diagnostic; the gate prints the
same envelope beside its floor verdict and never applies it.

Graphs
------

The four ``(device, attention implementation)`` graphs of the envelope round,
each the pinned official class in float16 on the oracle's prompt and 32
greedy tokens:

* ``cpu-eager``: the precision bundle's ``float16`` rows;
* ``mps-eager``: the authenticated MPS/F16 oracle;
* ``cpu-sdpa`` and ``mps-sdpa``: bundles of
  ``scripts/generate_nemotron_h_f16_graph_reference.py``.

Statistic
---------

For a row ``a`` and its F32 truth row ``r``, with the cosine ``c`` accumulated
in F64, the *normalized deviation* is

``delta(a, r) = | a / |a| - r / |r| |_2 = sqrt(2 (1 - c))``,

the L2 distance between the unit-normalized rows (a distance like the
envelope round's ``|a - r|`` statistics, and a monotone function of the
cosine the gate reads). For each path ``k`` (``prompt``: 1 row; ``teacher``:
32 teacher-forced rows of one stateless chunk; ``cached``: prefill plus 31
one-token decodes) and graph ``g``, ``D[g, k]`` is the largest ``delta`` over
the path's rows (the gated statistic) and ``m[g, k]`` the mean (reported).

Envelope and slack
------------------

``E[k] = max_g D[g, k]``. Graph ``g``'s leave-one-out ratio is
``rho[g, k] = D[g, k] / max_{h != g} D[h, k]``, how far the envelope of the
other graphs must stretch to admit it, and the path's slack is
``s[k] = round_up(max_g rho[g, k])``, the envelope round's ceiling to two
significant digits of the shortest round-trip decimal. The registered (not
adopted) rule would pass a Burn row of path ``k`` when
``delta <= s[k] E[k]``, the cosine floor ``1 - (s[k] E[k])^2 / 2``, which the
method requires to stay at or above ``0.99``. For
transparency the result also reports the pooled slack, the ratios of the
cosine deficit ``1 - c`` (which scales as ``delta^2``), and the
leave-one-out ratios of the means; none of them is adopted.

Pairing
-------

Before any row is compared: the precision bundle's manifest and the oracle's
four files must match their pinned digests (the graph tool's and the
precision tool's constants), and every precision row file the digest its
manifest records; each graph bundle must be schema 1, a float16 evaluation of
a new graph (``cpu/sdpa`` or ``mps/sdpa``) resolving the expected attention
and experts implementations, on the oracle's prompt and tokens, against the
pinned precision manifest and oracle, with every row file matching its
recorded digest and size; the graph bundles must share one generator script;
and the four graphs must cover each ``(device, attention)`` pair once. At
least three graphs are required, since leave-one-out over two is weak.

Usage::

    python3 scripts/derive_nemotron_h_envelope.py \\
        --precision-dir .../nemotron-3.5-lightning-30b-a3b-a9904d24-precision.bundle \\
        --oracle-dir .../nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle \\
        --graph .../f16-graphs/f16-cpu-sdpa.bundle --graph .../f16-graphs/f16-mps-sdpa.bundle \\
        --output envelope.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent


def load(name: str) -> Any:
    """Imports a sibling script as a module."""

    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRAPH_TOOL = load("generate_nemotron_h_f16_graph_reference")
PRECISION = GRAPH_TOOL.PRECISION
ENVELOPE = load("derive_half_precision_envelope")

#: Row paths, in gate order, with their row counts.
PATHS = tuple(PRECISION.KINDS)
#: Graphs the existing bundles provide, as (device, attention).
EXISTING = {("cpu", "eager"): "precision bundle float16", ("mps", "eager"): "MPS/F16 oracle"}
#: Graphs the graph tool provides.
NEW = (("cpu", "sdpa"), ("mps", "sdpa"))
#: Fewest graphs whose leave-one-out ratios the registered method may use.
MINIMUM_GRAPHS = 3
#: Lowest implied cosine floor accepted (the value of the gate's MINIMUM_HALF_PRECISION_FLOOR).
MINIMUM_FLOOR = 0.99


class PairingError(ValueError):
    """A bundle that cannot be part of the envelope."""


def require(condition: bool, message: str) -> None:
    """Raises ``PairingError(message)`` unless `condition` holds."""

    if not condition:
        raise PairingError(message)


def label(graph: tuple[str, str]) -> str:
    """``device-attention``."""

    return f"{graph[0]}-{graph[1]}"


def read_graph_bundle(root: Path, oracle: dict[str, Any], precision_sha256: str,
                      allowed: Sequence[tuple[str, str]] = NEW) -> dict[str, Any]:
    """Validates one graph-tool bundle against the oracle and the precision manifest; returns it with its payloads.

    `allowed` names the graphs the bundle may evaluate: the new graphs for the
    envelope, any of the four for a reported control.
    """

    manifest_bytes = (root / "reference.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    name = root.name
    require(manifest.get("schema_version") == GRAPH_TOOL.SCHEMA_VERSION, f"{name}: unsupported schema")
    graph = manifest["graph"]
    pair = (graph["device"], graph["attention_implementation"])
    require(pair in allowed, f"{name}: {label(pair)} is not one of {[label(p) for p in allowed]}")
    require(graph["dtype"] == "float16", f"{name}: a {graph['dtype']} evaluation is not an F16 graph")
    require(graph["resolved"] == {"attention_implementation": pair[1],
                                  "experts_implementation": GRAPH_TOOL.EXPERTS_RESOLVED[pair[0]]},
            f"{name}: resolved {graph['resolved']}, not the requested graph")
    require(manifest["prompt_ids"] == oracle["prompt_ids"], f"{name}: prompt differs from the oracle's")
    require(manifest["generated_ids"] == oracle["generated_ids"], f"{name}: tokens differ from the oracle's")
    provenance = manifest["provenance"]
    require(provenance["precision_manifest_sha256"] == precision_sha256,
            f"{name}: compared against another precision manifest")
    require(provenance["oracle_sha256"] == oracle["digests"], f"{name}: read another oracle")
    require(provenance["config_sha256"] == PRECISION.CONFIG_SHA256, f"{name}: another checkpoint config")
    require(provenance["transformers_checkout"]["revision"] == PRECISION.TRANSFORMERS_REVISION
            and provenance["transformers_checkout"]["clean"] is True, f"{name}: not the clean pinned checkout")
    require(provenance["precision_tool_sha256"] == GRAPH_TOOL.PRECISION_SCRIPT_SHA256,
            f"{name}: reused another precision tool")
    payloads = {}
    for kind, count in PATHS:
        file = f"float16-{kind}-logits.f32le"
        data = (root / file).read_bytes()
        require(len(data) == count * PRECISION.VOCABULARY * 4, f"{name}: {file} must hold {count} rows")
        require(hashlib.sha256(data).hexdigest() == manifest["files_sha256"][file], f"{name}: {file} is corrupted")
        payloads[kind] = data
    return {"graph": pair, "bundle": str(root), "reference_json_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "script_sha256": provenance["script_sha256"], "payloads": payloads}


def matrices(payloads: dict[str, bytes]) -> dict[str, Any]:
    """Decodes payloads into read-only F32 arrays of shape ``[rows, vocabulary]``."""

    import numpy

    return {kind: numpy.frombuffer(payloads[kind], dtype="<f4").reshape(count, -1) for kind, count in PATHS}


def row_cosines(rows: Any, truth: Any) -> list[float]:
    """Each row's cosine with its truth row, products and sums in F64; rows must be finite and nonzero."""

    import numpy

    left, right = numpy.asarray(rows, dtype=numpy.float64), numpy.asarray(truth, dtype=numpy.float64)
    require(left.shape == right.shape and left.ndim == 2 and left.size > 0, "compared rows differ in shape")
    require(bool(numpy.isfinite(left).all() and numpy.isfinite(right).all()), "compared rows must be finite")
    norms = numpy.sqrt((left * left).sum(axis=1)) * numpy.sqrt((right * right).sum(axis=1))
    require(bool((norms > 0).all()), "compared rows must be nonzero")
    return [float(value) for value in (left * right).sum(axis=1) / norms]


def normalized_deviation(cosine: float) -> float:
    """``sqrt(2 (1 - c))``, with a cosine rounded above 1 read as 1."""

    return math.sqrt(2.0 * max(0.0, 1.0 - cosine))


def path_statistics(cosines: Sequence[float]) -> dict[str, Any]:
    """The largest normalized deviation (first on ties), its decision, the mean, and the cosine deficits."""

    deviations = [normalized_deviation(value) for value in cosines]
    worst = max(range(len(deviations)), key=lambda index: (deviations[index], -index))
    return {"cosines": list(cosines), "deviations": deviations, "max": deviations[worst], "decision": worst,
            "mean": math.fsum(deviations) / len(deviations), "max_deficit": 1.0 - min(cosines),
            "minimum_cosine": min(cosines)}


def leave_one_out(values: dict[str, float]) -> dict[str, float]:
    """Each graph's value over the largest of the other graphs' values; a zero denominator is refused."""

    ratios = {}
    for graph, value in values.items():
        others = max(other for name, other in values.items() if name != graph)
        require(others > 0, f"the official graphs other than {graph} have zero deviation")
        ratios[graph] = value / others
    return ratios


def derive_paths(graphs: dict[str, dict[str, Any]], truth: dict[str, Any]) -> dict[str, Any]:
    """Measures every graph on every path and derives the envelopes, ratios, and slacks."""

    require(len(graphs) >= MINIMUM_GRAPHS,
            f"{len(graphs)} official graphs: leave-one-out needs at least {MINIMUM_GRAPHS}")
    paths = {}
    for kind, count in PATHS:
        measured = {name: path_statistics(row_cosines(rows[kind], truth[kind])) for name, rows in graphs.items()}
        for name, statistic in measured.items():
            require(len(statistic["cosines"]) == count, f"{name} {kind} must have {count} rows")
        maxima = {name: statistic["max"] for name, statistic in measured.items()}
        ratios = leave_one_out(maxima)
        largest = max(ratios, key=lambda name: (ratios[name], name))
        slack = ENVELOPE.round_up(ratios[largest])
        envelope_graph = max(maxima, key=lambda name: (maxima[name], name))
        envelope = maxima[envelope_graph]
        bound = slack * envelope
        floor = 1.0 - bound * bound / 2.0
        require(floor >= MINIMUM_FLOOR, f"{kind}: implied cosine floor {floor} is below {MINIMUM_FLOOR}")
        paths[kind] = {
            "rows": count,
            "graphs": {name: {key: statistic[key] for key in
                              ("max", "decision", "mean", "minimum_cosine", "max_deficit", "cosines")}
                       for name, statistic in measured.items()},
            "envelope": envelope,
            "envelope_graph": envelope_graph,
            "envelope_decision": measured[envelope_graph]["decision"],
            "leave_one_out": ratios,
            "largest_ratio": ratios[largest],
            "largest_ratio_graph": largest,
            "slack": slack,
            "bound": bound,
            "cosine_floor": floor,
            "deficit_leave_one_out": leave_one_out({n: s["max_deficit"] for n, s in measured.items()}),
            "mean_leave_one_out": leave_one_out({n: s["mean"] for n, s in measured.items()}),
        }
    pooled = max((path["largest_ratio"], kind) for kind, path in paths.items())
    deficit = max(max(path["deficit_leave_one_out"].values()) for path in paths.values())
    return {
        "graphs": sorted(graphs),
        "graph_count": len(graphs),
        "paths": paths,
        "slacks": {kind: path["slack"] for kind, path in paths.items()},
        "reported_not_adopted": {
            "pooled_slack": ENVELOPE.round_up(pooled[0]),
            "pooled_largest_ratio": pooled[0],
            "pooled_largest_path": pooled[1],
            "deficit_slacks": {kind: ENVELOPE.round_up(max(path["deficit_leave_one_out"].values()))
                               for kind, path in paths.items()},
            "largest_deficit_ratio": deficit,
            "mean_slacks": {kind: ENVELOPE.round_up(max(path["mean_leave_one_out"].values()))
                            for kind, path in paths.items()},
        },
    }


def control_report(graphs: dict[str, dict[str, Any]], truth: dict[str, Any], name: str,
                   rows: dict[str, Any]) -> dict[str, Any]:
    """Measures a control evaluation of one graph; reports, without adopting, the envelope with it.

    A control re-evaluates one of the four graphs. Its rows are compared with
    those of the graph it repeats, and the leave-one-out derivation is
    repeated twice: with the control in place of that graph, and with the
    control as a fifth graph.
    """

    report: dict[str, Any] = {"repeats": name, "paths": {}}
    for kind, _ in PATHS:
        repeated = row_cosines(rows[kind], graphs[name][kind])
        report["paths"][kind] = {"control": path_statistics(row_cosines(rows[kind], truth[kind])),
                                 "minimum_cosine_to_the_repeated_graph": min(repeated),
                                 "bitwise_equal_to_the_repeated_graph": bool((rows[kind] == graphs[name][kind]).all())}
        for statistic in report["paths"][kind].values():
            if isinstance(statistic, dict):
                statistic.pop("cosines", None)
                statistic.pop("deviations", None)
    replaced = derive_paths(dict(graphs, **{name: rows}), truth)
    added = derive_paths(dict(graphs, **{f"{name}-control": rows}), truth)
    report["replacing"] = {"slacks": replaced["slacks"],
                           "bounds": {kind: path["bound"] for kind, path in replaced["paths"].items()},
                           "cosine_floors": {kind: path["cosine_floor"] for kind, path in replaced["paths"].items()}}
    report["adding"] = {"slacks": added["slacks"],
                        "bounds": {kind: path["bound"] for kind, path in added["paths"].items()},
                        "cosine_floors": {kind: path["cosine_floor"] for kind, path in added["paths"].items()}}
    return report


def derive(precision_dir: Path, oracle_dir: Path, graph_dirs: Sequence[Path],
           control_dirs: Sequence[Path] = ()) -> dict[str, Any]:
    """Authenticates every bundle, pairs the graphs, and derives the envelope; reports any controls."""

    oracle = PRECISION.read_oracle(oracle_dir)
    precision = GRAPH_TOOL.read_precision_bundle(precision_dir)
    GRAPH_TOOL.check_inputs_agree(oracle, precision)
    bundles = [read_graph_bundle(Path(root), oracle, precision["manifest_sha256"]) for root in graph_dirs]
    pairs = [bundle["graph"] for bundle in bundles]
    require(len(set(pairs)) == len(pairs), f"a new graph appears twice: {[label(p) for p in pairs]}")
    scripts = {bundle["script_sha256"] for bundle in bundles}
    require(len(scripts) <= 1, f"the new graphs come from {len(scripts)} generator scripts; they must share one")
    graphs = {"cpu-eager": matrices(precision["payloads"]["float16"]), "mps-eager": matrices(oracle["payloads"])}
    for bundle in bundles:
        graphs[label(bundle["graph"])] = matrices(bundle["payloads"])
    truth = matrices(precision["payloads"]["float32"])
    result = derive_paths(graphs, truth)
    controls = [read_graph_bundle(Path(root), oracle, precision["manifest_sha256"], allowed=tuple(EXISTING) + NEW)
                for root in control_dirs]
    for control in controls:
        require(not scripts or control["script_sha256"] in scripts,
                f"{control['bundle']}: comes from another generator script than the new graphs")
        require(label(control["graph"]) in graphs, f"{control['bundle']}: repeats a graph outside the envelope")
    result["controls_reported_not_adopted"] = [
        dict(control_report(graphs, truth, label(control["graph"]), matrices(control["payloads"])),
             bundle=control["bundle"], reference_json_sha256=control["reference_json_sha256"])
        for control in controls]
    result["inputs"] = {
        "precision": {"bundle": str(precision_dir), "reference_json_sha256": precision["manifest_sha256"]},
        "oracle": {"bundle": str(oracle_dir), "sha256": oracle["digests"]},
        "graphs": [{key: bundle[key] for key in ("bundle", "reference_json_sha256", "script_sha256")}
                   | {"graph": label(bundle["graph"])} for bundle in bundles],
    }
    return result


def render(result: dict[str, Any]) -> str:
    """Renders the derivation as plain-text tables."""

    lines = [f"# Nemotron-H official F16 envelope over {result['graph_count']} graphs: {', '.join(result['graphs'])}"]
    for kind, path in result["paths"].items():
        lines.append(f"\n## {kind} ({path['rows']} rows)")
        for name, statistic in sorted(path["graphs"].items()):
            lines.append(f"- {name}: max delta {statistic['max']:.6e} at decision {statistic['decision']} "
                         f"(cosine {statistic['minimum_cosine']:.9f}), mean delta {statistic['mean']:.6e}; "
                         f"leave-one-out {path['leave_one_out'][name]:.4f} "
                         f"(deficit {path['deficit_leave_one_out'][name]:.4f}, mean {path['mean_leave_one_out'][name]:.4f})")
        lines.append(f"- envelope E {path['envelope']:.6e} ({path['envelope_graph']} decision {path['envelope_decision']}); "
                     f"slack {path['slack']} from {path['largest_ratio']:.4f} ({path['largest_ratio_graph']}); "
                     f"bound {path['bound']:.6e}, cosine floor {path['cosine_floor']:.9f}")
    other = result["reported_not_adopted"]
    lines.append(f"\nreported, not adopted: pooled slack {other['pooled_slack']} from {other['pooled_largest_ratio']:.4f} "
                 f"({other['pooled_largest_path']}); deficit slacks {other['deficit_slacks']}; "
                 f"mean slacks {other['mean_slacks']}")
    for control in result.get("controls_reported_not_adopted", []):
        lines.append(f"\ncontrol repeating {control['repeats']} (reported, not adopted): {control['bundle']}")
        for kind, path in control["paths"].items():
            statistic = path["control"]
            lines.append(f"- {kind}: max delta {statistic['max']:.6e} at decision {statistic['decision']} "
                         f"(cosine {statistic['minimum_cosine']:.9f}); minimum cosine to {control['repeats']} "
                         f"{path['minimum_cosine_to_the_repeated_graph']:.9f}, bitwise equal "
                         f"{path['bitwise_equal_to_the_repeated_graph']}")
        for mode in ("replacing", "adding"):
            floors = ", ".join(f"{kind} {value:.9f}" for kind, value in control[mode]["cosine_floors"].items())
            lines.append(f"- {mode} it: slacks {control[mode]['slacks']}, cosine floors {floors}")
    return "\n".join(lines) + "\n"


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precision-dir", type=Path, required=True, help="authenticated precision bundle")
    parser.add_argument("--oracle-dir", type=Path, required=True, help="authenticated MPS/F16 oracle bundle")
    parser.add_argument("--graph", type=Path, action="append", default=[], help="graph-tool bundle (repeatable)")
    parser.add_argument("--control", type=Path, action="append", default=[],
                        help="graph-tool bundle repeating one of the four graphs: reported, not adopted (repeatable)")
    parser.add_argument("--output", type=Path, required=True, help="JSON result to write")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Derives the envelope, writes the JSON result, and prints the tables."""

    arguments = parse_arguments(argv)
    result = derive(arguments.precision_dir, arguments.oracle_dir, arguments.graph, arguments.control)
    arguments.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(render(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
