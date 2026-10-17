"""Read repository-only acceptance evidence, skipping cleanly in the published crate.

The published archive omits most measurement campaign directories (see
``scripts/package_contents.py``). A record test that reads one of them calls
:func:`repository_evidence` first: in the extracted archive the test is skipped
with a message naming the omitted file; in a Git checkout the path is returned
unchanged, so a missing file still fails the test. Published records are never
skipped: asking for one raises ``ValueError``, so it must be read directly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


CRATE = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("package_contents", CRATE / "scripts" / "package_contents.py")
assert _SPEC is not None and _SPEC.loader is not None
PACKAGE_CONTENTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(PACKAGE_CONTENTS)


def repository_evidence(path: Path, root: Path = CRATE) -> Path:
    """Return ``path``, or skip the calling test when the published crate omits it.

    ``path`` must lie below ``root``'s ``acceptance/`` directory and must not
    be published evidence. The calling test (or class, from ``setUpClass``) is
    skipped only when ``root`` is an extracted package archive and the file is
    absent there.
    """

    relative = path.resolve().relative_to(root.resolve())
    if PACKAGE_CONTENTS.is_published_evidence(relative):
        raise ValueError(f"{relative.as_posix()} is published with the crate; read it directly")
    if PACKAGE_CONTENTS.is_packaged_archive(root) and not path.exists():
        raise unittest.SkipTest(
            f"{relative.as_posix()} is repository-only evidence that the published crate omits; "
            "run this test from a Git checkout of the repository"
        )
    return path
