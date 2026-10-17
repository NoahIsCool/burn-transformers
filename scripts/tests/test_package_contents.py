"""Pin the published/repository-only evidence split and the record tests' archive skips."""

import importlib.util
from pathlib import Path, PurePosixPath
import tempfile
import unittest

import repository_evidence


SCRIPT = Path(__file__).resolve().parents[1] / "package_contents.py"
SPEC = importlib.util.spec_from_file_location("package_contents", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONTENTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTENTS)


class EvidencePolicyTests(unittest.TestCase):
    """Classify every kind of acceptance path and mechanistic-study file."""

    def test_support_records_and_campaign_summaries_are_published(self):
        """Top-level records, campaign summaries and the named compact set ship."""

        for path in ("acceptance/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json",
                     "acceptance/measurements/v0_1-release-20260910.json",
                     "acceptance/measurements/README.md",
                     "acceptance/measurements/small-model-performance-20260910/manifest.json",
                     "acceptance/measurements/small-model-performance-20260910/nested/summary.json"):
            with self.subTest(path=path):
                self.assertTrue(CONTENTS.is_published_evidence(PurePosixPath(path)))

    def test_campaign_directories_and_stray_files_are_repository_only(self):
        """Raw logs, coverage exports and anything outside the two levels stay in Git."""

        for path in ("acceptance/measurements/large-checkpoint-replay-20261016/gates.log",
                     "acceptance/measurements/coding-standards-coverage-20260910/final/coverage.json",
                     "acceptance/measurements/small-model-performance-20260910.json/extra.json",
                     "acceptance/measurements/notes.txt",
                     "acceptance/README.md",
                     "acceptance/history/record.json"):
            with self.subTest(path=path):
                self.assertFalse(CONTENTS.is_published_evidence(PurePosixPath(path)))

    def test_paths_outside_acceptance_are_not_classified(self):
        """The evidence policy refuses to guess about unrelated paths."""

        for path in ("src/lib.rs", "acceptance", "", "measurements/acceptance/record.json"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "not an acceptance evidence path"):
                CONTENTS.is_published_evidence(PurePosixPath(path))

    def test_mechanistic_outputs_are_figures_results_and_captured_runs(self):
        """Programs, protocols and fixtures ship; recorded outputs do not."""

        outputs = ("examples/mechanistic/greater_than/result.json",
                   "examples/mechanistic/v0_2_examples.result.json",
                   "examples/mechanistic/tuned_lens/prediction_quality.png",
                   "examples/mechanistic/grokking_training/training.svg",
                   "examples/mechanistic/grokking_training/evidence/run.json",
                   "examples/mechanistic/clock_and_pizza_training/results/alpha0-paper/protocol.json")
        for path in outputs:
            with self.subTest(path=path):
                self.assertTrue(CONTENTS.is_example_output(PurePosixPath(path)))
        inputs = ("examples/mechanistic/greater_than/manifest.json",
                  "examples/mechanistic/greater_than/fixture.json",
                  "examples/mechanistic/grokking_reference_logits.json",
                  "examples/mechanistic/grokking_training/study.rs",
                  "examples/mechanistic/evidence.md",
                  "examples/mechanistic/grokking_training/results.md",
                  "examples/result.json",
                  "examples/README.md",
                  "src/models/gpt2/fixtures/figure.png")
        for path in inputs:
            with self.subTest(path=path):
                self.assertFalse(CONTENTS.is_example_output(PurePosixPath(path)))

    def test_archive_is_recognized_by_cargo_original_manifest(self):
        """Only an extracted archive carries Cargo.toml.orig beside its manifest."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text("[package]\n")
            self.assertFalse(CONTENTS.is_packaged_archive(root))
            (root / "Cargo.toml.orig").mkdir()
            self.assertFalse(CONTENTS.is_packaged_archive(root))
            (root / "Cargo.toml.orig").rmdir()
            (root / "Cargo.toml.orig").write_text("[package]\n")
            self.assertTrue(CONTENTS.is_packaged_archive(root))

    def test_this_checkout_classifies_its_own_evidence(self):
        """The repository has both kinds of evidence, and only Git holds the second."""

        root = repository_evidence.CRATE
        files = [path.relative_to(root) for path in (root / "acceptance").rglob("*") if path.is_file()]
        published = {path for path in files if CONTENTS.is_published_evidence(PurePosixPath(path.as_posix()))}
        self.assertIn(Path("acceptance/measurements/v0_1-release-20260910.json"), published)
        if CONTENTS.is_packaged_archive(root):
            self.assertEqual(set(files), published)
        else:
            self.assertLess(len(published), len(files))


class RepositoryEvidenceTests(unittest.TestCase):
    """Skip only for omitted repository-only evidence inside an extracted archive."""

    def fixture(self, archive):
        """Create a crate root with one published record and one campaign log."""

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "acceptance/measurements/campaign").mkdir(parents=True)
        (root / "acceptance/measurements/campaign.json").write_text("{}\n")
        (root / "acceptance/measurements/campaign/present.log").write_text("log\n")
        if archive:
            (root / "Cargo.toml.orig").write_text("[package]\n")
        return root

    def test_checkout_returns_every_repository_only_path(self):
        """A missing campaign file in a checkout is returned, so the read still fails."""

        root = self.fixture(archive=False)
        for name in ("present.log", "missing.log"):
            path = root / "acceptance/measurements/campaign" / name
            with self.subTest(name=name):
                self.assertEqual(repository_evidence.repository_evidence(path, root), path)
        with self.assertRaises(FileNotFoundError):
            repository_evidence.repository_evidence(root / "acceptance/measurements/campaign/missing.log", root).read_text()

    def test_archive_skips_only_an_omitted_campaign_file(self):
        """The skip names the omitted file; a present one is still read."""

        root = self.fixture(archive=True)
        present = root / "acceptance/measurements/campaign/present.log"
        self.assertEqual(repository_evidence.repository_evidence(present, root), present)
        with self.assertRaisesRegex(unittest.SkipTest, "acceptance/measurements/campaign/missing.log is repository-only"):
            repository_evidence.repository_evidence(root / "acceptance/measurements/campaign/missing.log", root)

    def test_published_records_are_never_skipped(self):
        """Asking to skip a published record is a programming error in either tree."""

        for archive in (False, True):
            root = self.fixture(archive)
            for name in ("campaign.json", "absent.json"):
                with self.subTest(archive=archive, name=name), self.assertRaisesRegex(ValueError, "published with the crate"):
                    repository_evidence.repository_evidence(root / "acceptance/measurements" / name, root)

    def test_paths_outside_the_crate_or_acceptance_are_rejected(self):
        """The helper only guards evidence below the crate's acceptance directory."""

        root = self.fixture(archive=True)
        with self.assertRaises(ValueError):
            repository_evidence.repository_evidence(root.parent / "elsewhere.log", root)
        with self.assertRaisesRegex(ValueError, "not an acceptance evidence path"):
            repository_evidence.repository_evidence(root / "src/lib.rs", root)


if __name__ == "__main__":
    unittest.main()
