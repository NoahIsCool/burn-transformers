"""Checks that an exported directory equals a commit's tree, read-only (scratch, not committed).

Usage: python3 -I verify_export.py REPO COMMIT EXPORT_DIR

Lists the commit's blobs with `git ls-tree -r`, hashes every exported file with
`git hash-object --stdin-paths` (which writes nothing to the object store), and
prints `dirty=<n>`: the number of paths whose blob differs, is missing, or is
present in the export but not in the commit (ignoring Cargo's `target`).
"""

import os
import subprocess
import sys


def main():
    """Prints the mismatch count and each mismatching path."""
    repo, commit, export = sys.argv[1:4]
    listing = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "-z", commit], capture_output=True,
                             check=True).stdout.split(b"\0")
    expected = {}
    for entry in filter(None, listing):
        meta, path = entry.split(b"\t", 1)
        mode, kind, sha = meta.split()
        if kind == b"blob":
            expected[path.decode()] = sha.decode()
    present = []
    for root, dirs, files in os.walk(export):
        dirs[:] = [d for d in dirs if not (root == export and d == "target")]
        for name in files:
            present.append(os.path.relpath(os.path.join(root, name), export))
    paths = sorted(set(expected) & set(present))
    hashes = subprocess.run(["git", "-C", repo, "hash-object", "--no-filters", "--stdin-paths"],
                            input="\n".join(os.path.join(export, p) for p in paths).encode(),
                            capture_output=True, check=True).stdout.decode().split()
    bad = [p for p, h in zip(paths, hashes) if expected[p] != h]
    bad += sorted(set(expected) - set(present))
    bad += sorted(set(present) - set(expected))
    print(f"commit={commit} files={len(expected)} dirty={len(bad)}")
    for path in bad:
        print("  " + path)


if __name__ == "__main__":
    main()
