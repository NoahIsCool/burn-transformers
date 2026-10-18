"""Pure tests for the Nemotron-H envelope derivation (synthetic rows; no PyTorch or checkpoint)."""

from __future__ import annotations

import contextlib
import copy
import functools
import io
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy


SCRIPT = Path(__file__).resolve().parents[1] / "derive_nemotron_h_envelope.py"
SPEC = importlib.util.spec_from_file_location("derive_nemotron_h_envelope", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DERIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DERIVE)
PRECISION = DERIVE.PRECISION
GRAPH_TOOL = DERIVE.GRAPH_TOOL

#: Synthetic vocabulary width.
WIDTH = 4
#: Rows per path.
COUNTS = dict(DERIVE.PATHS)


def rows_at(angle: float, count: int, worst: int | None = None, worst_angle: float | None = None) -> numpy.ndarray:
    """Unit rows at `angle` radians from ``e0`` in the ``e0, e1`` plane; row `worst` at `worst_angle`."""

    angles = numpy.full(count, angle)
    if worst is not None:
        angles[worst] = worst_angle
    rows = numpy.zeros((count, WIDTH), dtype="<f4")
    rows[:, 0], rows[:, 1] = numpy.cos(angles), numpy.sin(angles)
    return rows


def truth() -> dict[str, numpy.ndarray]:
    """Truth rows along ``e0``."""

    return {kind: rows_at(0.0, count) for kind, count in COUNTS.items()}


def graph(angle: float, worst: int | None = None, worst_angle: float | None = None) -> dict[str, numpy.ndarray]:
    """A graph whose rows sit at `angle` from the truth on every path, one row further off."""

    return {kind: rows_at(angle, count, worst if count > 1 else None, worst_angle) for kind, count in COUNTS.items()}


def chord(angle: float) -> float:
    """The normalized deviation of two unit rows `angle` apart."""

    return 2.0 * math.sin(angle / 2.0)


class StatisticTests(unittest.TestCase):
    """The row statistic, its path summary, and leave-one-out ratios."""

    def test_normalized_deviation_is_the_unit_row_distance(self) -> None:
        """``sqrt(2 (1 - c))`` is 0 for equal rows, sqrt(2) for orthogonal ones, and never NaN above 1."""

        self.assertEqual(DERIVE.normalized_deviation(1.0), 0.0)
        self.assertEqual(DERIVE.normalized_deviation(1.0 + 2e-16), 0.0)
        self.assertAlmostEqual(DERIVE.normalized_deviation(0.0), math.sqrt(2.0), places=15)
        self.assertAlmostEqual(DERIVE.normalized_deviation(math.cos(0.1)), chord(0.1), places=12)

    def test_row_cosines_accumulate_in_f64_and_refuse_bad_rows(self) -> None:
        """Cosines are per row; shape changes, non-finite values, and zero rows are refused."""

        rows = numpy.array([[1.0, 0.0], [1.0, 1.0]], dtype="<f4")
        reference = numpy.array([[1.0, 0.0], [1.0, 0.0]], dtype="<f4")
        cosines = DERIVE.row_cosines(rows, reference)
        self.assertEqual(cosines[0], 1.0)
        self.assertAlmostEqual(cosines[1], 1 / math.sqrt(2), places=15)
        for left, right in ((rows[:1], reference), (numpy.array([[numpy.nan, 0.0], [1.0, 0.0]]), reference),
                            (numpy.zeros((2, 2)), reference), (numpy.zeros((0, 2)), numpy.zeros((0, 2)))):
            with self.assertRaises(DERIVE.PairingError):
                DERIVE.row_cosines(left, right)

    def test_path_statistics_take_the_first_worst_row(self) -> None:
        """The maximum is the first on ties; the mean and the deficit come from the same rows."""

        cosines = [1.0, math.cos(0.2), math.cos(0.2), math.cos(0.1)]
        statistic = DERIVE.path_statistics(cosines)
        self.assertEqual(statistic["decision"], 1)
        self.assertAlmostEqual(statistic["max"], chord(0.2), places=12)
        self.assertAlmostEqual(statistic["mean"], (2 * chord(0.2) + chord(0.1)) / 4, places=12)
        self.assertAlmostEqual(statistic["max_deficit"], 1 - math.cos(0.2), places=15)
        self.assertEqual(statistic["minimum_cosine"], math.cos(0.2))

    def test_leave_one_out_divides_by_the_largest_other_graph(self) -> None:
        """Each ratio is the graph over the largest of the others; zero others are refused."""

        ratios = DERIVE.leave_one_out({"a": 2.0, "b": 1.0, "c": 4.0})
        self.assertEqual(ratios, {"a": 0.5, "b": 0.25, "c": 2.0})
        with self.assertRaises(DERIVE.PairingError):
            DERIVE.leave_one_out({"a": 1.0, "b": 0.0, "c": 0.0})


class DerivationTests(unittest.TestCase):
    """Envelopes, slacks, and floors from synthetic graphs with known geometry."""

    def test_slack_is_the_rounded_largest_ratio_of_each_path(self) -> None:
        """Per path the envelope is the worst graph and the slack the ceiling of its leave-one-out ratio."""

        graphs = {"cpu-eager": graph(0.010), "cpu-sdpa": graph(0.011),
                  "mps-eager": graph(0.010, worst=5, worst_angle=0.030), "mps-sdpa": graph(0.012)}
        result = DERIVE.derive_paths(graphs, truth())
        self.assertEqual(result["graph_count"], 4)
        teacher = result["paths"]["teacher"]
        self.assertEqual((teacher["envelope_graph"], teacher["envelope_decision"]), ("mps-eager", 5))
        self.assertAlmostEqual(teacher["envelope"], chord(0.030), places=7)
        self.assertAlmostEqual(teacher["largest_ratio"], chord(0.030) / chord(0.012), places=5)
        self.assertEqual(teacher["slack"], 2.5)
        self.assertAlmostEqual(teacher["bound"], 2.5 * teacher["envelope"], places=12)
        self.assertAlmostEqual(teacher["cosine_floor"], 1 - teacher["bound"] ** 2 / 2, places=15)
        prompt = result["paths"]["prompt"]
        self.assertEqual(prompt["largest_ratio_graph"], "mps-sdpa")
        self.assertEqual(prompt["slack"], 1.1)
        self.assertEqual(result["slacks"], {kind: path["slack"] for kind, path in result["paths"].items()})
        other = result["reported_not_adopted"]
        self.assertEqual(other["pooled_slack"], 2.5)
        self.assertEqual(other["pooled_largest_path"], "teacher")
        self.assertEqual(other["deficit_slacks"]["teacher"], 6.3)

    def test_three_graphs_suffice_and_two_are_refused(self) -> None:
        """Leave-one-out needs at least three graphs."""

        graphs = {"cpu-eager": graph(0.010), "cpu-sdpa": graph(0.011), "mps-eager": graph(0.012)}
        self.assertEqual(DERIVE.derive_paths(graphs, truth())["graph_count"], 3)
        graphs.pop("cpu-sdpa")
        with self.assertRaisesRegex(DERIVE.PairingError, "at least 3"):
            DERIVE.derive_paths(graphs, truth())

    def test_an_envelope_too_wide_to_gate_on_is_refused(self) -> None:
        """An implied cosine floor below 0.99 fails the derivation."""

        graphs = {"cpu-eager": graph(0.05), "cpu-sdpa": graph(0.05), "mps-eager": graph(0.2)}
        with self.assertRaisesRegex(DERIVE.PairingError, "cosine floor"):
            DERIVE.derive_paths(graphs, truth())

    def test_rows_must_cover_each_path(self) -> None:
        """A graph with a missing row is refused before any ratio."""

        short = graph(0.01)
        short["cached"] = short["cached"][:-1]
        graphs = {"cpu-eager": graph(0.01), "cpu-sdpa": short, "mps-eager": graph(0.02)}
        with self.assertRaises(DERIVE.PairingError):
            DERIVE.derive_paths(graphs, truth())


class BundleTests(unittest.TestCase):
    """Fail-closed pairing of graph-tool bundles against the oracle and the precision manifest."""

    ORACLE = {"prompt_ids": [1, 2, 3], "generated_ids": list(range(PRECISION.DECISIONS)),
              "digests": {"oracle.json": "a" * 64}}
    PRECISION_SHA = "b" * 64

    def write_bundle(self, root: Path, device: str = "cpu", attention: str = "sdpa", angle: float = 0.01,
                     **changes) -> Path:
        """Writes a synthetic graph-tool bundle; `changes` override manifest fields by dotted path."""

        bundle = root / f"f16-{device}-{attention}.bundle"
        bundle.mkdir()
        files = {}
        for kind, rows in graph(angle).items():
            name = f"float16-{kind}-logits.f32le"
            (bundle / name).write_bytes(rows.tobytes())
            files[name] = hashlib.sha256(rows.tobytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "prompt_ids": self.ORACLE["prompt_ids"],
            "generated_ids": self.ORACLE["generated_ids"],
            "graph": {"dtype": "float16", "device": device, "attention_implementation": attention,
                      "resolved": {"attention_implementation": attention,
                                   "experts_implementation": GRAPH_TOOL.EXPERTS_RESOLVED[device]}},
            "files_sha256": files,
            "provenance": {"precision_manifest_sha256": self.PRECISION_SHA, "oracle_sha256": self.ORACLE["digests"],
                           "config_sha256": PRECISION.CONFIG_SHA256, "script_sha256": "c" * 64,
                           "precision_tool_sha256": GRAPH_TOOL.PRECISION_SCRIPT_SHA256,
                           "transformers_checkout": {"revision": PRECISION.TRANSFORMERS_REVISION, "clean": True}},
        }
        for path, value in changes.items():
            node = manifest
            *parents, leaf = path.split(".")
            for key in parents:
                node = node[key]
            node[leaf] = value
        (bundle / "reference.json").write_text(json.dumps(manifest))
        return bundle

    def read(self, bundle: Path):
        """Reads a synthetic bundle at the synthetic vocabulary."""

        with mock.patch.object(PRECISION, "VOCABULARY", WIDTH):
            return DERIVE.read_graph_bundle(bundle, self.ORACLE, self.PRECISION_SHA)

    def test_a_valid_bundle_reads_with_its_identity(self) -> None:
        """A complete new-graph bundle yields its graph, digest, script, and rows."""

        with tempfile.TemporaryDirectory() as root:
            bundle = self.write_bundle(Path(root))
            read = self.read(bundle)
            self.assertEqual(read["graph"], ("cpu", "sdpa"))
            self.assertEqual(read["reference_json_sha256"],
                             hashlib.sha256((bundle / "reference.json").read_bytes()).hexdigest())
            self.assertEqual(DERIVE.matrices(read["payloads"])["teacher"].shape, (PRECISION.DECISIONS, WIDTH))

    def test_every_identity_and_payload_is_checked(self) -> None:
        """Each changed identity, an existing graph, and a corrupted or short file are refused."""

        cases = {
            "schema_version": 2, "graph.dtype": "bfloat16", "graph.device": "mps",
            "graph.resolved": {"attention_implementation": "eager", "experts_implementation": "grouped_mm"},
            "prompt_ids": [1, 2], "generated_ids": [0], "provenance.precision_manifest_sha256": "d" * 64,
            "provenance.oracle_sha256": {}, "provenance.config_sha256": "e" * 64,
            "provenance.precision_tool_sha256": "f" * 64,
            "provenance.transformers_checkout": {"revision": PRECISION.TRANSFORMERS_REVISION, "clean": False},
        }
        for path, value in cases.items():
            with self.subTest(field=path), tempfile.TemporaryDirectory() as root:
                with self.assertRaises(DERIVE.PairingError):
                    self.read(self.write_bundle(Path(root), **{path: value}))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(DERIVE.PairingError, "not a new graph"):
                self.read(self.write_bundle(Path(root), attention="eager"))
        with tempfile.TemporaryDirectory() as root:
            bundle = self.write_bundle(Path(root))
            data = bytearray((bundle / "float16-cached-logits.f32le").read_bytes())
            data[0] ^= 1
            (bundle / "float16-cached-logits.f32le").write_bytes(bytes(data))
            with self.assertRaisesRegex(DERIVE.PairingError, "corrupted"):
                self.read(bundle)
            (bundle / "float16-cached-logits.f32le").write_bytes(bytes(data[:-4]))
            with self.assertRaisesRegex(DERIVE.PairingError, "must hold"):
                self.read(bundle)

    def derive_synthetic(self, root: Path, bundles: list[Path]):
        """Runs `derive` with synthetic oracle and precision readers."""

        oracle = copy.deepcopy(self.ORACLE)
        oracle["payloads"] = {kind: rows.tobytes() for kind, rows in graph(0.02, worst=3, worst_angle=0.03).items()}
        precision = {"manifest": {"prompt_ids": oracle["prompt_ids"], "generated_ids": oracle["generated_ids"]},
                     "manifest_sha256": self.PRECISION_SHA,
                     "payloads": {"float32": {k: r.tobytes() for k, r in truth().items()},
                                  "float16": {k: r.tobytes() for k, r in graph(0.015).items()}}}
        with mock.patch.object(PRECISION, "VOCABULARY", WIDTH), \
                mock.patch.object(PRECISION, "read_oracle", lambda _: oracle), \
                mock.patch.object(GRAPH_TOOL, "read_precision_bundle", lambda _: precision):
            return DERIVE.derive(root / "precision", root / "oracle", bundles)

    def test_derive_pairs_the_four_graphs_and_records_their_identities(self) -> None:
        """The existing two graphs and the new two are measured together; inputs are recorded."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundles = [self.write_bundle(root, "cpu", "sdpa", 0.016), self.write_bundle(root, "mps", "sdpa", 0.021)]
            result = self.derive_synthetic(root, bundles)
            self.assertEqual(result["graphs"], ["cpu-eager", "cpu-sdpa", "mps-eager", "mps-sdpa"])
            self.assertEqual([entry["graph"] for entry in result["inputs"]["graphs"]], ["cpu-sdpa", "mps-sdpa"])
            teacher = result["paths"]["teacher"]
            self.assertEqual(teacher["envelope_graph"], "mps-eager")
            self.assertAlmostEqual(teacher["largest_ratio"], chord(0.03) / chord(0.021), places=5)
            self.assertIn("slack", DERIVE.render(result))

    def test_derive_refuses_a_repeated_graph_or_mixed_generators(self) -> None:
        """Two bundles of one graph, or bundles from two generator scripts, are refused."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_bundle(root, "cpu", "sdpa")
            (root / "again").mkdir()
            again = self.write_bundle(root / "again", "cpu", "sdpa")
            with self.assertRaisesRegex(DERIVE.PairingError, "twice"):
                self.derive_synthetic(root, [first, again])
            other = self.write_bundle(root, "mps", "sdpa", **{"provenance.script_sha256": "9" * 64})
            with self.assertRaisesRegex(DERIVE.PairingError, "generator scripts"):
                self.derive_synthetic(root, [first, other])

    def test_command_line_writes_json_and_prints_tables(self) -> None:
        """`main` writes the derivation as strict JSON and renders its tables."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundles = [self.write_bundle(root, "cpu", "sdpa", 0.016), self.write_bundle(root, "mps", "sdpa", 0.021)]
            result = functools.partial(self.derive_synthetic, root, bundles)()
            printed = io.StringIO()
            with mock.patch.object(DERIVE, "derive", lambda *arguments: result), contextlib.redirect_stdout(printed):
                output = root / "envelope.json"
                self.assertEqual(DERIVE.main(["--precision-dir", "p", "--oracle-dir", "o", "--graph", str(bundles[0]),
                                              "--graph", str(bundles[1]), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["slacks"], result["slacks"])
            self.assertEqual(printed.getvalue(), DERIVE.render(result))


if __name__ == "__main__":
    unittest.main()
