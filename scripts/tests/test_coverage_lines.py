"""LLVM physical-line semantics and malformed-input tests; no compilers or GPU."""

import importlib.util
from itertools import islice
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "coverage_lines.py"
SPEC = importlib.util.spec_from_file_location("coverage_lines", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LINES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LINES)


def segment(line=1, column=1, count=0, has_count=True, entry=True, gap=False):
    """Build exactly the six fields emitted by LLVM JSON renderSegment."""

    return [line, column, count, has_count, entry, gap]


def end(line=1, column=9):
    """End the preceding mapped interval without beginning a skipped region."""

    return segment(line, column, has_count=False, entry=False)


def report(*files):
    """Wrap file objects in the supported LLVM JSON export envelope."""

    return {"type": LINES.EXPORT_TYPE, "version": LINES.EXPORT_VERSION,
            "data": [{"files": list(files)}]}


class PhysicalLineTests(unittest.TestCase):
    """Pin LLVM wrapping/count rules independently of production source scope."""

    def assert_lines(self, segments, expected):
        """Check exact line counts, totals, ordering and normalized run storage."""

        result = LINES.line_coverage(segments)
        self.assertEqual(list(result.iter_lines()), expected)
        self.assertEqual(result.mapped_lines, len(expected))
        self.assertEqual(result.covered_lines, sum(count > 0 for _, count in expected))
        for previous, current in zip(result.runs, result.runs[1:]):
            self.assertLessEqual(previous.stop, current.start)
            self.assertFalse(previous.stop == current.start and previous.count == current.count)
        return result

    def test_empty_and_unmapped_files(self):
        """No regions means no mapped lines, not a 100 percent coverage claim."""

        self.assert_lines([], [])
        self.assert_lines([end()], [])
        self.assert_lines([segment(has_count=False), end(4)], [])

    def test_wrapped_segments_include_final_line(self):
        """Counts carry across lines without entries, including LLVM's end line."""

        result = self.assert_lines([segment(2, 4, 7), end(5, 1)], [(n, 7) for n in range(2, 6)])
        self.assertEqual(result.runs, (LINES.LineRun(2, 6, 7),))

    def test_zero_count_is_mapped_but_uncovered(self):
        """Unexecuted functions keep their physical lines in the denominator."""

        self.assert_lines([segment(1), end(3)], [(1, 0), (2, 0), (3, 0)])

    def test_same_line_regions_use_maximum_not_sum(self):
        """Overlapping closure/function regions contribute one physical line."""

        self.assert_lines([segment(1, 1, 2), segment(1, 4, 7), segment(1, 6, 0), end()], [(1, 7)])

    def test_wrapped_count_can_dominate_new_region(self):
        """The preceding line's active count participates in this line's max."""

        self.assert_lines([segment(1, 1, 9), segment(2, 1, 2), end(3)], [(1, 9), (2, 9), (3, 2)])

    def test_nonentry_count_changes_apply_on_the_next_line(self):
        """A resumed enclosing region is not independently a region entry."""

        self.assert_lines([segment(1, 1, 2), segment(2, 3, 9, entry=False), end(3)],
                          [(1, 2), (2, 2), (3, 9)])

    def test_skipped_start_suppresses_wrapped_mapping(self):
        """An explicit skipped-region entry overrides a positive wrapped count."""

        self.assert_lines([segment(1, 1, 6), segment(2, 3, has_count=False), end(4)], [(1, 6)])

    def test_later_counted_entry_restores_skipped_line(self):
        """A later entry restores mapping and keeps LLVM's wrapped maximum."""

        self.assert_lines([segment(1, 1, 6), segment(2, 1, has_count=False),
                           segment(2, 5, 2), end(3)], [(1, 6), (2, 6), (3, 2)])

    def test_gap_counts_do_not_dominate_entry_line(self):
        """Gap segments carry counts forward but are not non-gap region starts."""

        self.assert_lines([segment(1, 1, 3), segment(2, 1, 99, entry=False, gap=True),
                           segment(2, 4, 5), end(3)], [(1, 3), (2, 5), (3, 5)])

    def test_leading_gap_and_gap_entry_match_distinct_llvm_rules(self):
        """Gap starts without entry map only later lines; counted entries map now."""

        self.assert_lines([segment(1, 1, 8, entry=False, gap=True), end(3)], [(2, 8), (3, 8)])
        self.assert_lines([segment(1, 1, 8, gap=True), end(2)], [(1, 0), (2, 8)])

    def test_gap_entry_restores_skipped_mapping_without_its_own_count(self):
        """The any-counted-entry rule includes gaps, unlike the count maximum."""

        self.assert_lines([segment(1, 1, 4), segment(2, 1, has_count=False),
                           segment(2, 4, 90, gap=True), end(3)], [(1, 4), (2, 4), (3, 90)])

    def test_same_position_skipped_marker_can_resume_enclosing_count(self):
        """LLVM permits a duplicate location when the earlier segment has no count."""

        self.assert_lines([segment(1, 1, 2), segment(2, 1, has_count=False),
                           segment(2, 1, 3, entry=False), end(3)], [(1, 2), (3, 3)])

    def test_separated_functions_leave_unmapped_gaps(self):
        """Ending one function never fills empty source lines before another."""

        self.assert_lines([segment(1, 1, 3), end(2), segment(6, 1, 3), end(7)],
                          [(1, 3), (2, 3), (6, 3), (7, 3)])

    def test_skipped_end_column_one_does_not_map_end_line(self):
        """Region-entry status, not column alone, distinguishes skipped endpoints."""

        self.assert_lines([segment(1, 1, 3), segment(2, 1, has_count=False)], [(1, 3)])

    def test_huge_coordinates_are_measured_without_dense_allocation(self):
        """A valid uint32 source extent occupies one run, not billions of entries."""

        maximum = LINES.MAX_COORDINATE
        result = LINES.line_coverage([segment(count=LINES.MAX_COUNT), end(maximum, maximum)])
        self.assertEqual(result.runs, (LINES.LineRun(1, maximum + 1, LINES.MAX_COUNT),))
        self.assertEqual(result.mapped_lines, maximum)
        self.assertEqual(result.covered_lines, maximum)
        self.assertEqual(list(islice(result.iter_lines(), 2)), [(1, LINES.MAX_COUNT), (2, LINES.MAX_COUNT)])


class ValidationTests(unittest.TestCase):
    """Malformed artifacts fail contextually instead of adjusting measurement."""

    def test_segment_array_and_arity_are_strict(self):
        """Reject missing arrays, tuples, dictionaries and extra segment fields."""

        for invalid in (None, {}, (), "segments", [None], [{}], [[1, 2]], [segment() + [0]]):
            with self.subTest(invalid=invalid), self.assertRaises(LINES.CoverageLinesError):
                LINES.line_coverage(invalid)

    def test_coordinates_reject_nonintegers_and_out_of_range_values(self):
        """Do not accept bool-as-int, zero coordinates, or uint32 overflow."""

        for field in (0, 1):
            for value in (0, -1, True, 1.5, "1", None, LINES.MAX_COORDINATE + 1):
                invalid = segment()
                invalid[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(LINES.CoverageLinesError, "coordinate"):
                    LINES.line_coverage([invalid, end()])

    def test_execution_count_is_a_clamped_nonnegative_json_integer(self):
        """Reject counter overflow, negative values, floats, strings and booleans."""

        for count in (-1, True, 0.5, "0", None, LINES.MAX_COUNT + 1):
            with self.subTest(count=count), self.assertRaisesRegex(LINES.CoverageLinesError, "execution count"):
                LINES.line_coverage([segment(count=count), end()])

    def test_flags_are_boolean_and_uncounted_segments_are_zero(self):
        """No-count markers cannot smuggle positive counters or gap semantics."""

        for field in (3, 4, 5):
            for value in (0, 1, None, "false"):
                invalid = segment()
                invalid[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(LINES.CoverageLinesError, "booleans"):
                    LINES.line_coverage([invalid, end()])
        for invalid in (segment(count=1, has_count=False), segment(has_count=False, gap=True)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(LINES.CoverageLinesError, "uncounted"):
                LINES.line_coverage([invalid])

    def test_positions_cannot_reverse_or_overlap_counted_segments(self):
        """Preserve LLVM's ordered boundary interpretation, including columns."""

        for invalid in ([segment(2), end(1)], [segment(1, 5), end(1, 2)]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(LINES.CoverageLinesError, "unsorted"):
                LINES.line_coverage(invalid)
        for count in (0, 9):
            with self.subTest(count=count), self.assertRaisesRegex(LINES.CoverageLinesError, "overlapping"):
                LINES.line_coverage([segment(count=count), segment(), end()])

    def test_unterminated_counted_segment_is_not_silently_clipped(self):
        """Without its endpoint, the extent of an active count is unknowable."""

        with self.assertRaisesRegex(LINES.CoverageLinesError, "unterminated"):
            LINES.line_coverage([segment(2, 1, 3)])

    def test_only_supported_export_envelopes_are_accepted(self):
        """Require exact type/version and nonempty data blocks, not tool guesses."""

        malformed = [None, [], {}, {**report(), "type": "other"}, {**report(), "version": "3.0.2"},
                     {**report(), "data": []}, {**report(), "data": {}},
                     {**report(), "data": [None]}, {**report(), "data": [{}]},
                     {**report(), "data": [{"files": {}}]}]
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(LINES.CoverageLinesError):
                LINES.report_line_coverage(value)

    def test_file_metadata_requires_names_and_full_segment_exports(self):
        """Missing segments are not interchangeable with a genuinely empty file."""

        for invalid in (None, [], {}, {"filename": ""}, {"filename": 1}, {"filename": "bad\0name"}):
            with self.subTest(invalid=invalid), self.assertRaises(LINES.CoverageLinesError):
                LINES.report_line_coverage(report(invalid))
        with self.assertRaisesRegex(LINES.CoverageLinesError, "summary-only"):
            LINES.report_line_coverage(report({"filename": "a.rs", "summary": {}}))
        with self.assertRaisesRegex(LINES.CoverageLinesError, r"files\[0\].*a.rs.*segment 0"):
            LINES.report_line_coverage(report({"filename": "a.rs", "segments": [[1]]}))

    def test_duplicate_files_across_blocks_are_not_merged(self):
        """Merging uncorrelated coverage maps could silently double denominators."""

        entry = {"filename": "a.rs", "segments": []}
        for value in (report(entry, entry), {**report(entry), "data": [{"files": [entry]}, {"files": [entry]}]}):
            with self.subTest(value=value), self.assertRaisesRegex(LINES.CoverageLinesError, "duplicate filename"):
                LINES.report_line_coverage(value)

    def test_all_files_are_retained_without_trusting_summary_counts(self):
        """This primitive cannot waive tests, generated code, models or dependencies."""

        names = ("src/lib.rs", "src/tests.rs", "src/models/test.rs", "target/generated.rs", "/dep/lib.rs")
        entries = [{"filename": name, "segments": [segment(count=1), end()],
                    "summary": {"lines": {"count": 999, "covered": 0}}} for name in names]
        measured = LINES.report_line_coverage(report(*entries, {"filename": "empty.rs", "segments": []}))
        self.assertEqual(set(measured), {*names, "empty.rs"})
        self.assertTrue(all(measured[name].mapped_lines == measured[name].covered_lines == 1 for name in names))
        self.assertEqual(measured["empty.rs"].mapped_lines, 0)
        self.assertEqual(LINES.report_line_coverage(report()), {})


if __name__ == "__main__":
    unittest.main()
