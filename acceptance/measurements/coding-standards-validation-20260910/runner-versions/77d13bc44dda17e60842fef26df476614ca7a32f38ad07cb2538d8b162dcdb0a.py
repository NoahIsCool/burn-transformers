"""Auditable offline v0.3 source checks; no scientific training/replay jobs."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = Path('/Users/nover/repos/transformers')
OUTPUT = ROOT / 'acceptance/measurements/coding-standards-validation-20260910'
FEATURES = 'std,flex,autodiff,train,tokenizers,chat-templates'
PY_AUDITOR = Path('/tmp/v0_3_python_validation.py')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def files(roots):
    roots = [str(p) for p in roots if p.exists()]
    if not roots:
        return set()
    result = subprocess.run(['rg', '--files', '--hidden', '--no-ignore', '-g', '!target/**', '-g', '!**/__pycache__/**', *roots], cwd=ROOT, text=True, capture_output=True, check=True)
    return {Path(name).resolve() for name in result.stdout.splitlines() if Path(name).is_file()}


def base_inputs():
    names = ('Cargo.toml', 'Cargo.lock', 'build.rs', 'rust-toolchain', 'rust-toolchain.toml', 'rustfmt.toml', '.rustfmt.toml', 'clippy.toml', '.clippy.toml')
    return {ROOT / name for name in names if (ROOT / name).is_file()} | files([ROOT / '.cargo'])


def includes(paths):
    selected = set(paths)
    pending = list(paths)
    while pending:
        path = pending.pop()
        if path.suffix != '.rs':
            continue
        body = path.read_text()
        local = re.findall(r'include_(?:str|bytes)!\(\s*"([^"\n]+)"', body)
        targets = [(path.parent / name).resolve() for name in local]
        for name in re.findall(r'include_(?:str|bytes)!\(\s*concat!\(\s*env!\("CARGO_MANIFEST_DIR"\)\s*,\s*"([^"\n]+)"', body):
            targets.append(ROOT / name.lstrip('/'))
        for target in targets:
            if not target.is_file():
                raise RuntimeError(f'missing include input from {path}: {target}')
            if target not in selected:
                selected.add(target)
                pending.append(target)
    return selected


def scope_paths(scope):
    if scope == 'layout':
        return files([ROOT / 'scripts/coverage-layout']) | base_inputs()
    if scope == 'python':
        selected = files([ROOT / 'scripts'])
        selected |= {p for p in files([ROOT / 'examples']) if p.suffix == '.py'}
        # Runtime subprocess compiler fixtures and test inputs, independent of live study outputs.
        selected |= files([ROOT / 'tests'])
        return selected | base_inputs()
    selected = files([ROOT / 'src', ROOT / 'tests', ROOT / 'benches'])
    selected |= {p for p in files([ROOT / 'examples']) if p.suffix == '.rs'}
    return includes(selected | base_inputs())


def snapshot(scope):
    result = {}
    for path in sorted(scope_paths(scope)):
        try:
            name = path.relative_to(ROOT).as_posix()
        except ValueError:
            name = str(path)
        data = path.read_bytes()
        result[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return result


def changes(before, after):
    return [name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]


def plan(run_root):
    def cargo(toolchain, action, *args):
        return ['cargo', '+' + toolchain, action, '--release', '--locked', '--offline', *args]
    profile = ['--no-default-features', '--features', FEATURES]
    rows = []
    def add(name, command, scope='rust', **extra):
        rows.append({'id': name, 'command': command, 'scope': scope, **extra})
    add('rust-fmt', ['cargo', '+nightly', 'fmt', '--check'])
    add('rust-all-targets', cargo('nightly', 'test', *profile, '--all-targets'))
    add('rust-doctests', cargo('nightly', 'test', *profile, '--doc'))
    add('rust-all-feature-clippy', cargo('nightly', 'clippy', '--all-targets', '--all-features', '--', '-D', 'warnings'))
    add('rust-minimal', cargo('nightly', 'check', '--lib', '--no-default-features'))
    add('rust-std-only', cargo('nightly', 'check', '--lib', '--no-default-features', '--features', 'std'))
    add('rust-autodiff-no-backend', cargo('nightly', 'check', '--lib', '--no-default-features', '--features', 'std,autodiff'))
    add('rust-public-docs', cargo('nightly', 'doc', *profile, '--no-deps'))
    add('msrv-all-targets', cargo('1.92.0', 'test', *profile, '--all-targets'))
    add('msrv-doctests', cargo('1.92.0', 'test', *profile, '--doc'))
    add('stable-build', cargo('stable', 'check', *profile, '--lib'))
    for name, directory, pattern in [
        ('python-tooling', 'scripts/tests', 'test*.py'),
        ('python-grokking-protocol', 'examples/mechanistic/grokking_training', 'test_*.py'),
        ('python-clock-diagnostics', 'examples/mechanistic/clock_and_pizza_training', '*_tests.py'),
        ('python-greater-than-mechanics', 'examples/mechanistic/greater_than', 'test_*.py'),
        ('python-tuned-lens-metrics', 'examples/mechanistic/tuned_lens', '*_tests.py'),
    ]:
        add(name, [sys.executable, str(PY_AUDITOR), '--root', str(ROOT), '--start', directory, '--pattern', pattern], 'python', python_audit=True)
    layout = ['--manifest-path', str(ROOT / 'scripts/coverage-layout/Cargo.toml')]
    target = ['--target-dir', str(ROOT / 'target')]
    add('layout-fmt', ['cargo', '+nightly', 'fmt', *layout, '--check'], 'layout')
    add('layout-clippy', cargo('nightly', 'clippy', *layout, *target, '--all-targets', '--', '-D', 'warnings'), 'layout')
    add('layout-tests', cargo('nightly', 'test', *layout, *target), 'layout')
    add('msrv-layout-tests', cargo('1.92.0', 'test', *layout, *target), 'layout')
    add('authored-documentation', [sys.executable, str(ROOT / 'scripts/check_documentation.py'), '--classifier', str(ROOT / 'target/release/transformers-coverage-layout'), '--output', str(run_root / 'documentation.json')], 'python', report=str(run_root / 'documentation.json'))
    for name, arguments in [
        ('activation_workbench', []), ('attention_head_attribution', []),
        ('custom_composition', []), ('t5_classifier', []),
        ('mla_encoder_decoder', []), ('qwen_prefix', ['--random']),
    ]:
        add('demo-' + name, cargo('nightly', 'run', *profile, '--example', name, *(['--', *arguments] if arguments else [])))
    for name in ['t5_classifier', 'custom_composition', 'mla_encoder_decoder']:
        add('train-' + name, cargo('nightly', 'run', *profile, '--example', name, '--', '--train'), report=str(ROOT / 'target/training' / name / 'result.json'))
    qwen = run_root / 'qwen-adapter'
    add('train-random-qwen-adapter', cargo('nightly', 'run', *profile, '--example', 'train_qwen_adapter', '--', '--output-dir', str(qwen)), report=str(qwen / 'report.json'))
    learner = run_root / 'burn-learner'
    add('train-native-burn-learner', cargo('nightly', 'run', *profile, '--example', 'burn_training', '--', str(learner)), report=str(learner / 'result.json'))
    for backend in ['metal', 'cuda', 'rocm']:
        add('compile-' + backend, cargo('nightly', 'check', '--no-default-features', '--features', 'std,' + backend + ',tokenizers,chat-templates'))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')
    run_root = ROOT / 'target/coding-standards-validation' / stamp
    commands = plan(run_root)
    if args.only:
        commands = [row for row in commands if row['id'] in args.only]
    if not args.run:
        print(json.dumps(commands, indent=2))
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    old_index = OUTPUT / 'manifest.json'
    if old_index.exists():
        old_manifest = json.loads(old_index.read_text())
        old_hash = old_manifest['runner_sha256']
        retained = OUTPUT / 'runner-versions' / (old_hash + '.py')
        retained.parent.mkdir(exist_ok=True)
        if not retained.exists():
            shutil.copy2(OUTPUT / 'runner.py', retained)
        for old_step in old_manifest['steps']:
            old_step.setdefault('runner_sha256', old_hash)
        write(old_index, old_manifest)
    current_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    version = OUTPUT / 'runner-versions' / (current_hash + '.py')
    version.parent.mkdir(exist_ok=True)
    shutil.copy2(__file__, version)
    shutil.copy2(__file__, OUTPUT / 'runner.py')
    shutil.copy2(PY_AUDITOR, OUTPUT / 'python_auditor.py')
    env = os.environ.copy()
    env.update({'CARGO_NET_OFFLINE': 'true', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'CARGO_TERM_COLOR': 'never'})
    versions = {}
    for command in [['rustc', '+nightly', '-vV'], ['cargo', '+nightly', '-V'], ['rustc', '+1.92.0', '-vV'], ['cargo', '+1.92.0', '-V'], ['rustc', '+stable', '-vV'], [sys.executable, '--version']]:
        versions[' '.join(command)] = subprocess.check_output(command, cwd=ROOT, env=env, text=True).strip()
    if 'release: 1.92.0\n' not in versions['rustc +1.92.0 -vV'] + '\n':
        raise RuntimeError('exact MSRV identity did not match 1.92.0')
    index = OUTPUT / 'manifest.json'
    manifest = json.loads(index.read_text()) if index.exists() else {'schema_version': 1, 'steps': []}
    manifest.update({
        'claim': 'Current coding-standards source, documentation, tooling and cheap native-training validation. No coverage measurement, hardware numerical acceptance, package promotion, long scientific training, or pretrained study replay.',
        'scope_policy': {
            'rust': 'All files beneath src/tests/benches; all examples/**/*.rs; Cargo/build/toolchain/lint configuration; recursive literal include_str/include_bytes inputs including CARGO_MANIFEST_DIR concat paths. Includes untracked files. Measurement outputs and unrelated Markdown are excluded unless directly included by Rust.',
            'python': 'All scripts and test fixtures plus Python example sources; actual repository read opens recorded separately for this process. Historical acceptance records read by tests are bound, unrelated evolving scientific output JSON/plots are not.',
            'layout': 'Standalone coverage-layout tool source and lock/config; tool unit tests only, no coverage run.',
        },
        'tool_versions': versions,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python_auditor_sha256': hashlib.sha256(PY_AUDITOR.read_bytes()).hexdigest(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status_before': subprocess.check_output(['git', 'status', '--short', '--untracked-files=all'], cwd=ROOT, text=True).splitlines(),
        'started_utc': manifest.get('started_utc', utc()),
        'environment_overrides': {key: env[key] for key in ['CARGO_NET_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TOKENIZERS_PARALLELISM', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE', 'CARGO_TERM_COLOR']},
        'build_environment': {key: env.get(key) for key in ['RUSTFLAGS', 'RUSTDOCFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'CARGO_TARGET_DIR', 'CARGO_BUILD_TARGET', 'RUSTC_BOOTSTRAP']},
    })
    write(index, manifest)
    for row in commands:
        name = row['id']
        attempt = 1 + sum(entry['id'] == name for entry in manifest['steps'])
        directory = OUTPUT / f'{name}-attempt-{attempt:02}'
        directory.mkdir()
        before = snapshot(row['scope'])
        write(directory / 'inputs-before.json', before)
        command = row['command'].copy()
        if row.get('python_audit'):
            command += ['--audit', str(directory / 'python-reads.json')]
        print(f'BEGIN {name} attempt={attempt}: {" ".join(command)}', flush=True)
        started = utc()
        clock = time.monotonic()
        preflight = {'path': str(ROOT), 'free_bytes': shutil.disk_usage(ROOT).free, 'minimum_bytes': 20*1024**3, 'normal_target': str(ROOT / 'target')}
        write(directory / 'storage-preflight.json', preflight)
        if preflight['free_bytes'] < preflight['minimum_bytes']:
            raise RuntimeError('Local storage preflight requires at least 20 GiB')
        with (directory / 'output.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        after = snapshot(row['scope'])
        write(directory / 'inputs-after.json', after)
        changed = changes(before, after)
        record = {
            'id': name, 'attempt': attempt, 'command': command, 'cwd': str(ROOT), 'scope': row['scope'],
            'runner_sha256': current_hash,
            'started_utc': started, 'finished_utc': utc(), 'seconds': time.monotonic() - clock,
            'exit_code': result.returncode, 'changed_inputs': changed,
            'inputs_before': str((directory / 'inputs-before.json').relative_to(OUTPUT)),
            'inputs_after': str((directory / 'inputs-after.json').relative_to(OUTPUT)),
            'output': str((directory / 'output.log').relative_to(OUTPUT)),
            'output_sha256': hashlib.sha256((directory / 'output.log').read_bytes()).hexdigest(),
            'passed': result.returncode == 0 and not changed,
        }
        if row.get('python_audit'):
            audit_path = directory / 'python-reads.json'
            record['python_read_audit'] = str(audit_path.relative_to(OUTPUT)) if audit_path.exists() else None
            if audit_path.exists():
                record['passed'] &= not json.loads(audit_path.read_text())['changed_inputs']
        if row.get('report') and Path(row['report']).is_file():
            target = directory / 'replay-report.json'
            shutil.copy2(row['report'], target)
            record['replay_report'] = str(target.relative_to(OUTPUT))
            record['replay_report_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        write(directory / 'execution.json', record)
        manifest['steps'].append(record)
        manifest['updated_utc'] = utc()
        write(index, manifest)
        print(f'END {name}: exit={result.returncode}, changed_inputs={len(changed)}, seconds={record["seconds"]:.2f}', flush=True)
        if not record['passed']:
            print((directory / 'output.log').read_text()[-14000:], flush=True)
            if changed:
                print('CHANGED: ' + ', '.join(changed), flush=True)
    latest = {entry['id']: entry for entry in manifest['steps']}
    for entry in latest.values():
        before = json.loads((OUTPUT / entry['inputs_before']).read_text())
        entry['changed_since_validation'] = changes(before, snapshot(entry['scope']))
    manifest['latest_checks_passed'] = all(entry['passed'] and not entry['changed_since_validation'] for entry in latest.values())
    manifest['finished_utc'] = utc()
    manifest['git_status_after'] = subprocess.check_output(['git', 'status', '--short', '--untracked-files=all'], cwd=ROOT, text=True).splitlines()
    write(index, manifest)
    print(f'COMPLETE: {len(latest)} checks; latest_checks_passed={manifest["latest_checks_passed"]}; {index}', flush=True)


if __name__ == '__main__':
    main()
