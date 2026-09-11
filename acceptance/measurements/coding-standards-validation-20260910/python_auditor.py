"""Run one offline unit-test suite and audit actual repository reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--start', required=True)
parser.add_argument('--pattern', default='test*.py')
parser.add_argument('--audit', required=True)
args = parser.parse_args()
root = Path(args.root).resolve()
reads = {}
busy = False

def digest(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError):
        return None

def audit(event, values):
    global busy
    if busy or event != 'open' or not values or not isinstance(values[0], (str, bytes, os.PathLike)):
        return
    mode = values[1] if len(values) > 1 else None
    flags = values[2] if len(values) > 2 else 0
    if isinstance(mode, str) and any(c in mode for c in 'wax+'):
        return
    if isinstance(flags, int) and flags & os.O_ACCMODE == os.O_WRONLY:
        return
    busy = True
    try:
        path = Path(os.fsdecode(values[0])).resolve()
        relative = path.relative_to(root).as_posix()
        if relative.split('/')[0] in {'target', '.git', 'models'} or '__pycache__' in path.parts:
            return
        if relative.startswith('acceptance/measurements/v0_3-training-validation-'):
            return
        value = digest(path)
        if value is not None:
            reads.setdefault(relative, value)
    except (OSError, ValueError):
        pass
    finally:
        busy = False

sys.addaudithook(audit)
start = (root / args.start).resolve()
sys.path.insert(0, str(start))
suite = unittest.defaultTestLoader.discover(str(start), pattern=args.pattern, top_level_dir=str(start))
result = unittest.TextTestRunner(verbosity=2).run(suite)
busy = True
after = {name: digest(root / name) for name in reads}
changed = [name for name, value in reads.items() if after[name] != value]
Path(args.audit).write_text(json.dumps({
    'scope': 'Actual read opens in this Python process beneath the repository; nested subprocesses additionally covered by the runner source/fixture inventory. Build outputs, models, git internals, bytecode caches and this validation output are excluded.',
    'inputs_before_first_read': reads, 'inputs_after': after,
    'changed_inputs': changed, 'tests_run': result.testsRun,
    'failures': len(result.failures), 'errors': len(result.errors),
    'skipped': len(result.skipped),
}, indent=2) + '\n')
sys.exit(0 if result.wasSuccessful() and not changed else 1)
