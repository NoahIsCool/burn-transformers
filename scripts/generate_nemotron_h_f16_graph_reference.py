#!/usr/bin/env python3
"""Evaluate one official Nemotron-H float16 graph on the oracle's inputs, for the precision envelope.

The Nemotron-H official-precision gate (``tests/nemotron_h_artifacts.rs``)
judges Burn's F16 logit rows against the official class's F32 rows of the
precision bundle (``generate_nemotron_h_precision_reference.py``). Its bound
is the envelope of the official float16 graphs, the half-precision envelope
round's four ``(device, attention implementation)`` pairs:

=========  ===========  ===================================================
device     attention    rows
=========  ===========  ===================================================
``cpu``    ``eager``    the precision bundle's ``float16`` evaluation
``cpu``    ``sdpa``     this tool
``mps``    ``eager``    the authenticated MPS/F16 oracle
``mps``    ``sdpa``     this tool
=========  ===========  ===================================================

This offline maintainer tool evaluates one graph of the pinned official class
from the clean Transformers checkout ``cfa309cfa34b5497a32fd50af49aa97aac94c054``
in float16 on one device with one attention implementation, on the oracle's
prompt and its 32 recorded greedy tokens (never its own argmax), and records
the stateless prompt row, the 32 teacher-forced rows of one stateless chunk,
and the 32 cached rows of prefill plus 31 one-token decodes, called exactly as
the oracle and the precision tool call them. A new graph differs from its
device's existing graph only in the attention implementation: the experts
implementation is the existing graph's (the precision tool's default request
on the CPU, ``eager`` like the oracle on MPS), and the manifest records the
requested and resolved implementations. Router-correction buffers stay F32.

The precision tool's helpers (oracle reader, row comparison, block budget)
are reused byte for byte: the tool refuses to run unless that script's
SHA-256 is the one that produced the precision bundle. Inputs are
authenticated before any value is used: the oracle's files and the
precision bundle's manifest by pinned SHA-256, and every precision row file
by the digest its authenticated manifest records; the config and the two
official sources by SHA-256 in a clean checkout at the pinned revision.

Before loading the checkpoint, a self-check on a tiny random official model
proves on the CPU that this tool's device-aware calls equal the precision
tool's calls bit for bit, and on the requested graph that the requested
implementations resolve and (for a CPU budget) that blocks offloaded to disk
compute the same rows as resident blocks.

The manifest compares every row with the F32 truth row of the same path
(cosine accumulated in F64), with the MPS/F16 oracle, and with the CPU/F16
eager rows, and records per-phase timings each with a host-load sample.
Outputs are staged in a hidden sibling directory and published by one rename
that never replaces an existing bundle.

Usage::

    python3 -I scripts/generate_nemotron_h_f16_graph_reference.py \\
        --model-dir /path/to/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \\
        --transformers-dir /path/to/transformers-cfa309 \\
        --oracle-dir /path/to/nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle \\
        --precision-dir /path/to/nemotron-3.5-lightning-30b-a3b-a9904d24-precision.bundle \\
        --device cpu --attn-implementation sdpa \\
        --max-resident-gib 16 --offload-dir /path/to/offload \\
        --output /path/to/f16-graphs/f16-cpu-sdpa.bundle

On the CPU memory is the F16 model (about 63 GB) unless ``--max-resident-gib``
with ``--offload-dir`` keeps only the leading blocks resident, as in the
precision tool. On MPS the whole F16 model (about 65 GB) must fit in unified
memory; the system volume's free space is checked before loading and after
every forward, because MPS may write temporary graph files there.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

# Every official run keeps all MPS operations on MPS, as the oracle does, and
# stays offline; both are set before any ML package can be imported.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

#: The precision tool whose helpers this tool reuses.
PRECISION_SCRIPT = Path(__file__).resolve().parent / "generate_nemotron_h_precision_reference.py"
#: SHA-256 of the precision tool that produced the pinned precision bundle.
PRECISION_SCRIPT_SHA256 = "7d846c4b51cd31acedbdd6f6ac37467f803ccbce99145ec676e1a59d897b759f"


def load_precision_tool(path: Path = PRECISION_SCRIPT) -> Any:
    """Imports the precision tool as a module (it installs the offline controls at import)."""

    spec = importlib.util.spec_from_file_location("generate_nemotron_h_precision_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRECISION = load_precision_tool()

#: Bundle format version, bumped on any incompatible change.
SCHEMA_VERSION = 1
#: Directory name of the precision bundle whose F32 rows are the truth.
PRECISION_BUNDLE = "nemotron-3.5-lightning-30b-a3b-a9904d24-precision.bundle"
#: Pinned SHA-256 of the precision bundle's manifest.
PRECISION_MANIFEST_SHA256 = "5da4427df4c882d3c4d42b5c2d37c2c61a4837597c944b9c67a23903a7ce73f9"
#: Graphs this tool can evaluate, as (device, attention implementation).
GRAPHS = (("cpu", "eager"), ("cpu", "sdpa"), ("mps", "eager"), ("mps", "sdpa"))
#: Experts implementation requested on each device: that device's existing graph's.
EXPERTS_REQUEST = {"cpu": None, "mps": "eager"}
#: Experts implementation each request must resolve to in the pinned checkout.
EXPERTS_RESOLVED = {"cpu": "grouped_mm", "mps": "eager"}
#: The only evaluation dtype.
DTYPE = "float16"
#: Name of the bundle's manifest.
MANIFEST = "reference.json"
#: Free bytes the system volume must keep before an MPS load.
SYSTEM_PREFLIGHT_FREE_BYTES = 15 * 2**30
#: Free bytes the system volume must keep after every MPS forward.
SYSTEM_RUNTIME_FREE_BYTES = 10 * 2**30


def sha256_file(path: Path) -> str:
    """Hashes a file's exact bytes in 16 MiB blocks."""

    return PRECISION.sha256_file(path)


def verify_precision_tool(path: Path = PRECISION_SCRIPT, expected: str = PRECISION_SCRIPT_SHA256) -> str:
    """Requires the reused precision tool to be the exact script that produced the precision bundle."""

    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path.name} has SHA-256 {actual}, expected {expected}")
    return actual


def read_precision_bundle(
    precision_dir: Path, pinned: str = PRECISION_MANIFEST_SHA256, name: str = PRECISION_BUNDLE
) -> dict[str, Any]:
    """Authenticates the precision bundle and returns its manifest and the payloads of its F32 and F16 rows.

    The manifest is checked against its pinned digest; every row file against
    the digest that manifest records. Only the ``float32`` (truth) and
    ``float16`` (CPU/eager) evaluations are read.
    """

    if precision_dir.name != name:
        raise ValueError(f"the precision bundle must be named {name}, not {precision_dir.name}")
    manifest_bytes = (precision_dir / MANIFEST).read_bytes()
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    if actual != pinned:
        raise ValueError(f"precision manifest has SHA-256 {actual}, expected {pinned}")
    manifest = json.loads(manifest_bytes)
    payloads: dict[str, dict[str, bytes]] = {}
    for label in ("float32", "float16"):
        payloads[label] = {}
        for kind, count in PRECISION.KINDS:
            file = f"{label}-{kind}-logits.f32le"
            data = (precision_dir / file).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != manifest["files_sha256"][file]:
                raise ValueError(f"precision file {file} has SHA-256 {digest}, not its manifest's")
            if len(data) != count * PRECISION.VOCABULARY * 4:
                raise ValueError(f"precision file {file} must hold {count} full-vocabulary rows")
            payloads[label][kind] = data
    return {"manifest": manifest, "manifest_sha256": actual, "payloads": payloads}


def check_inputs_agree(oracle: dict[str, Any], precision: dict[str, Any]) -> None:
    """Requires the precision bundle to describe the oracle's prompt and tokens."""

    manifest = precision["manifest"]
    if manifest["prompt_ids"] != oracle["prompt_ids"] or manifest["generated_ids"] != oracle["generated_ids"]:
        raise ValueError("the precision bundle and the oracle describe different inputs")


def resolved_experts(device: str) -> str:
    """The experts implementation the device's request must resolve to."""

    return EXPERTS_RESOLVED[device]


def check_graph(device: str, attention: str) -> None:
    """Refuses a graph outside the envelope's four."""

    if (device, attention) not in GRAPHS:
        raise ValueError(f"unsupported graph {device}/{attention}; expected one of {GRAPHS}")


def load_options(device: str, attention: str) -> dict[str, Any]:
    """The `from_pretrained` keywords of a graph, beyond dtype and placement."""

    check_graph(device, attention)
    options: dict[str, Any] = {"attn_implementation": attention}
    if EXPERTS_REQUEST[device] is not None:
        options["experts_implementation"] = EXPERTS_REQUEST[device]
    return options


def check_loaded(torch: Any, model: Any, device: str, attention: str) -> dict[str, Any]:
    """Requires the loaded model to run the requested graph; checks the F32 router corrections.

    Returns the resolved implementations. On MPS every parameter and buffer
    must live on the device; on the CPU, parameters offloaded to disk live on
    the meta device until Accelerate loads them.
    """

    attn = model.config._attn_implementation
    experts = model.config._experts_implementation
    if attn != attention or experts != resolved_experts(device):
        raise ValueError(f"loaded {attn}/{experts}, expected {attention}/{resolved_experts(device)}")
    corrections = [buffer for name, buffer in model.named_buffers() if name.endswith("e_score_correction_bias")]
    if len(corrections) != 23 or {buffer.dtype for buffer in corrections} != {torch.float32}:
        raise ValueError("the official class must keep its 23 router-correction buffers in F32")
    if device == "mps":
        placed = {tensor.device.type for tensor in list(model.parameters()) + list(model.buffers())}
        if placed != {"mps"}:
            raise ValueError(f"MPS graph has tensors on {sorted(placed)}")
    return {"attention_implementation": attn, "experts_implementation": experts}


def evaluate_on(torch: Any, model: Any, prompt_ids: list[int], generated_ids: list[int], device: str) -> dict[str, Any]:
    """Runs the precision tool's stateless prompt, teacher-forced, and cached calls with inputs on `device`.

    The calls, their keywords, and the row selection are the precision tool's
    ``evaluate``; only the input tensors are created on `device` and the rows
    returned to the CPU (F16 to F32 widening is exact on either side).
    """

    def tokens(values: list[int]) -> Any:
        """One row of token IDs on the device."""

        return torch.tensor([values], dtype=torch.long, device=device)

    def ones(length: int) -> Any:
        """An all-valid attention mask for one row of `length` tokens on the device."""

        return torch.ones((1, length), dtype=torch.long, device=device)

    teacher_ids = prompt_ids + generated_ids[:-1]
    first = len(prompt_ids) - 1
    decisions = len(generated_ids)
    with torch.no_grad():
        stateless = model(
            input_ids=tokens(prompt_ids), attention_mask=ones(len(prompt_ids)), use_cache=False, logits_to_keep=1,
            return_dict=True,
        ).logits[0, -1:].float().cpu()
        teacher = model(
            input_ids=tokens(teacher_ids),
            attention_mask=ones(len(teacher_ids)),
            use_cache=False,
            logits_to_keep=0,
            return_dict=True,
        ).logits[0, first : first + decisions].float().cpu()
        output = model(
            input_ids=tokens(prompt_ids), attention_mask=ones(len(prompt_ids)), use_cache=True, logits_to_keep=1,
            return_dict=True,
        )
        cached = [output.logits[0, -1].float().cpu()]
        for decision in range(1, decisions):
            logical_length = len(prompt_ids) + decision - 1
            output = model(
                input_ids=tokens([generated_ids[decision - 1]]),
                attention_mask=ones(logical_length + 1),
                past_key_values=output.past_key_values,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            cached.append(output.logits[0, -1].float().cpu())
    return {"prompt": stateless, "teacher": teacher, "cached": torch.stack(cached)}


def system_free(minimum: int, purpose: str, path: Path = Path("/")) -> dict[str, Any]:
    """Samples the system volume's free space and refuses to continue below `minimum` bytes."""

    usage = shutil.disk_usage(path)
    sample = {"purpose": purpose, "path": str(path), "free_bytes": usage.free, "minimum_free_bytes": minimum}
    if usage.free < minimum:
        raise RuntimeError(f"{purpose}: only {usage.free} bytes free on {path}, below {minimum}")
    return sample


def host_sample(label: str, processes: int = 8) -> dict[str, Any]:
    """Records the time, the load averages, and the processes using the most CPU right now."""

    top: list[dict[str, Any]] = []
    try:
        listing = subprocess.run(
            ["ps", "-Ao", "pid=,pcpu=,rss=,comm=", "-r"], check=True, capture_output=True, text=True, timeout=20
        ).stdout
        for line in listing.splitlines()[:processes]:
            pid, cpu, rss, command = line.split(None, 3)
            top.append({"pid": int(pid), "cpu_percent": float(cpu), "rss_kib": int(rss), "command": command})
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        top.append({"error": str(error)})
    return {
        "label": label,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "load_average": [round(value, 2) for value in os.getloadavg()],
        "top_cpu": top,
    }


def timed(samples: list[dict[str, Any]], label: str, action: Callable[[], Any]) -> Any:
    """Runs `action`, appending its wall time and host samples taken before and after it."""

    before = host_sample(f"before {label}")
    started = time.perf_counter()
    result = action()
    seconds = time.perf_counter() - started
    samples.append({"phase": label, "seconds": round(seconds, 3), "before": before, "after": host_sample(f"after {label}")})
    return result


def tiny_checkpoint(torch: Any, transformers: Any, model_dir: Path, root: Path) -> Path:
    """Saves the precision tool's tiny random official model (BF16 weights, F32 router corrections)."""

    config_values = json.loads((model_dir / "config.json").read_text())
    for key in ("architectures", "auto_map", "hybrid_override_pattern"):
        config_values.pop(key, None)
    config_values.update(PRECISION.SELF_CHECK_OVERRIDES)
    torch.manual_seed(0)
    tiny = transformers.NemotronHForCausalLM(transformers.NemotronHConfig(**config_values)).eval()
    with torch.no_grad():
        for parameter in tiny.parameters():
            parameter.data = (torch.randn_like(parameter.data) * 0.2).to(torch.bfloat16)
        for name, buffer in tiny.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.copy_(torch.randn_like(buffer) * 0.05)
    path = root / "checkpoint"
    tiny.save_pretrained(path)
    return path


def self_check(torch: Any, transformers: Any, model_dir: Path, device: str, attention: str, offload: bool) -> dict[str, Any]:
    """Checks the harness and the requested graph on a tiny random official model.

    1. On the CPU in F32 with eager attention, `evaluate_on` equals the
       precision tool's ``evaluate`` bit for bit.
    2. In F16 on the requested graph, the requested implementations resolve
       and every row is finite.
    3. With `offload` (CPU only), every other block offloaded to disk gives
       the same rows as all blocks resident, bit for bit.
    """

    prompt, generated = [5, 9, 17, 3, 44, 60], [11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = tiny_checkpoint(torch, transformers, model_dir, root)

        def load(dtype: Any, placement: str, attn: str, **options: Any) -> Any:
            """Loads the tiny checkpoint as the real run loads the checkpoint."""

            if "device_map" not in options:
                options["device_map"] = {"": placement}
            return transformers.NemotronHForCausalLM.from_pretrained(
                checkpoint, local_files_only=True, dtype=dtype, attn_implementation=attn, **options
            ).eval()

        reference = PRECISION.evaluate(torch, load(torch.float32, "cpu", "eager"), prompt, generated)
        mirrored = evaluate_on(torch, load(torch.float32, "cpu", "eager"), prompt, generated, "cpu")
        harness = all(torch.equal(mirrored[kind], reference[kind]) for kind, _ in PRECISION.KINDS)
        extra = {key: value for key, value in load_options(device, attention).items() if key != "attn_implementation"}
        graph_model = load(torch.float16, device, attention, **extra)
        resolved = check_tiny_loaded(graph_model, device, attention)
        rows = evaluate_on(torch, graph_model, prompt, generated, device)
        finite = all(bool(torch.isfinite(rows[kind]).all()) for kind, _ in PRECISION.KINDS)
        offloaded_equal = None
        if offload:
            device_map = {f"model.{name}": "cpu" for name, _ in graph_model.model.named_children() if name != "layers"}
            device_map["lm_head"] = "cpu"
            for index in range(len(graph_model.model.layers)):
                device_map[f"model.layers.{index}"] = "disk" if index % 2 else "cpu"
            offloaded_model = load(
                torch.float16, device, attention, device_map=device_map, offload_folder=str(root / "offload"), **extra
            )
            offloaded = evaluate_on(torch, offloaded_model, prompt, generated, device)
            offloaded_equal = all(torch.equal(offloaded[kind], rows[kind]) for kind, _ in PRECISION.KINDS)
    if not harness or not finite or offloaded_equal is False:
        raise ValueError(f"self-check failed: harness={harness}, finite={finite}, offloaded_equal={offloaded_equal}")
    return {"harness_bitwise_equal": harness, "graph": resolved, "finite": finite, "offloaded_bitwise_equal": offloaded_equal}


def check_tiny_loaded(model: Any, device: str, attention: str) -> dict[str, str]:
    """Requires the tiny model to resolve the requested implementations (it has no 23 router buffers)."""

    attn = model.config._attn_implementation
    experts = model.config._experts_implementation
    if attn != attention or experts != resolved_experts(device):
        raise ValueError(f"tiny model loaded {attn}/{experts}, expected {attention}/{resolved_experts(device)}")
    return {"attention_implementation": attn, "experts_implementation": experts}


def rows_from(torch: Any, payloads: dict[str, bytes]) -> dict[str, Any]:
    """Decodes authenticated F32LE payloads into `[rows, vocabulary]` tensors."""

    return {
        kind: torch.frombuffer(bytearray(payloads[kind]), dtype=torch.float32).reshape(count, PRECISION.VOCABULARY)
        for kind, count in PRECISION.KINDS
    }


def publish(staging: Path, destination: Path) -> None:
    """Renames a finished staging directory into place without replacing anything."""

    PRECISION.publish(staging, destination)


def build_bundle(arguments: argparse.Namespace, staging: Path) -> dict[str, Any]:
    """Evaluates the requested graph, writes its rows into `staging`, and returns the manifest."""

    device, attention = arguments.device, arguments.attn_implementation
    check_graph(device, attention)
    samples: list[dict[str, Any]] = [{"phase": "start", "host": host_sample("start")}]
    precision_tool = verify_precision_tool()
    model_dir = arguments.model_dir.resolve(strict=True)
    config_sha256 = sha256_file(model_dir / "config.json")
    if config_sha256 != PRECISION.CONFIG_SHA256:
        raise ValueError(f"config.json has SHA-256 {config_sha256}, expected {PRECISION.CONFIG_SHA256}")
    checkout = PRECISION.verify_checkout(arguments.transformers_dir.resolve(strict=True))
    oracle = PRECISION.read_oracle(arguments.oracle_dir.resolve(strict=True))
    precision = read_precision_bundle(arguments.precision_dir.resolve(strict=True))
    check_inputs_agree(oracle, precision)
    torch, transformers = PRECISION.import_official(arguments.transformers_dir.resolve(strict=True))
    torch.use_deterministic_algorithms(True)
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("the MPS backend is unavailable")
    offload = arguments.max_resident_gib is not None
    widening = timed(samples, "self-check", lambda: self_check(torch, transformers, model_dir, device, attention, offload))
    torch.manual_seed(0)

    system: list[dict[str, Any]] = []
    if device == "mps":
        system.append(system_free(SYSTEM_PREFLIGHT_FREE_BYTES, "system volume before the MPS load"))
    options = load_options(device, attention)
    offload_root, offloaded_blocks = None, 0
    if offload:
        budget = int(arguments.max_resident_gib * 2**30)
        options["device_map"] = PRECISION.block_device_map(torch, transformers, model_dir, DTYPE, budget)
        offload_root = Path(tempfile.mkdtemp(prefix=".offload.", dir=arguments.offload_dir.resolve(strict=True)))
        options["offload_folder"] = str(offload_root)
        offloaded_blocks = sum(value == "disk" for value in options["device_map"].values())
    else:
        options["device_map"] = {"": device}
    try:
        model = timed(samples, "load", lambda: transformers.NemotronHForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, dtype=torch.float16, **options).eval())
        resolved = check_loaded(torch, model, device, attention)

        def run() -> dict[str, Any]:
            """Evaluates the three row families and synchronizes the device."""

            result = evaluate_on(torch, model, oracle["prompt_ids"], oracle["generated_ids"], device)
            if device == "mps":
                torch.mps.synchronize()
            return result

        result = timed(samples, "evaluate", run)
        if device == "mps":
            system.append(system_free(SYSTEM_RUNTIME_FREE_BYTES, "system volume after the evaluation"))
    finally:
        model = None
        gc.collect()
        if offload_root is not None:
            shutil.rmtree(offload_root, ignore_errors=True)

    files = {}
    for kind, count in PRECISION.KINDS:
        tensor = result[kind]
        if list(tensor.shape) != [count, PRECISION.VOCABULARY] or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{device}/{attention} {kind} rows are malformed")
        name = f"{DTYPE}-{kind}-logits.f32le"
        (staging / name).write_bytes(PRECISION.tensor_f32le(tensor))
        files[name] = sha256_file(staging / name)
        result[kind] = tensor.contiguous()
    references = {
        "float32": rows_from(torch, precision["payloads"]["float32"]),
        "float16-mps-oracle": rows_from(torch, oracle["payloads"]),
        "float16-cpu-eager": rows_from(torch, precision["payloads"]["float16"]),
    }
    comparisons = {
        label: {kind: PRECISION.summarize_tensors(result[kind], rows[kind]) for kind, _ in PRECISION.KINDS}
        for label, rows in references.items()
    }
    samples.append({"phase": "end", "host": host_sample("end")})
    return {
        "schema_version": SCHEMA_VERSION,
        "repository": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "revision": PRECISION.CHECKPOINT_REVISION,
        "prompt_ids": oracle["prompt_ids"],
        "generated_ids": oracle["generated_ids"],
        "graph": {
            "dtype": DTYPE,
            "device": device,
            "attention_implementation": attention,
            "experts_implementation_requested": EXPERTS_REQUEST[device],
            "resolved": resolved,
        },
        "files_sha256": files,
        "comparisons": comparisons,
        "provenance": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "script": "scripts/generate_nemotron_h_f16_graph_reference.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "precision_tool_sha256": precision_tool,
            "transformers": transformers.__version__,
            "transformers_checkout": checkout,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "environment": {key: os.environ.get(key) for key in (
                "PYTORCH_ENABLE_MPS_FALLBACK", "HF_DEACTIVATE_ASYNC_LOAD", "HF_HUB_OFFLINE", "USE_HUB_KERNELS")},
            "model_dir": str(model_dir),
            "config_sha256": config_sha256,
            "oracle_dir": str(arguments.oracle_dir.resolve()),
            "oracle_sha256": oracle["digests"],
            "precision_dir": str(arguments.precision_dir.resolve()),
            "precision_manifest_sha256": precision["manifest_sha256"],
            "self_check": widening,
            "max_resident_gib": arguments.max_resident_gib,
            "offloaded_blocks": offloaded_blocks,
            "system_volume": system,
            "timings": samples,
        },
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True, help="pinned local checkpoint")
    parser.add_argument("--transformers-dir", type=Path, required=True, help="clean checkout at the pinned revision")
    parser.add_argument("--oracle-dir", type=Path, required=True, help="authenticated MPS/F16 oracle bundle")
    parser.add_argument("--precision-dir", type=Path, required=True, help="authenticated precision bundle")
    parser.add_argument("--device", choices=("cpu", "mps"), required=True, help="device of the official graph")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), required=True,
                        help="attention implementation of the official graph")
    parser.add_argument("--output", type=Path, required=True, help="bundle directory to create")
    parser.add_argument(
        "--max-resident-gib",
        type=float,
        help="CPU only: keep at most this many GiB of decoder blocks in memory and offload the rest to --offload-dir",
    )
    parser.add_argument("--offload-dir", type=Path, help="directory for temporary offloaded blocks")
    arguments = parser.parse_args(argv)
    if (arguments.max_resident_gib is None) != (arguments.offload_dir is None):
        parser.error("--max-resident-gib and --offload-dir go together")
    if arguments.device == "mps" and arguments.max_resident_gib is not None:
        parser.error("an MPS graph keeps the whole model on the device; --max-resident-gib is CPU only")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    """Builds the bundle in a hidden staging directory and publishes it, or removes the staging."""

    arguments = parse_arguments(argv)
    destination = arguments.output.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing bundle {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        manifest = build_bundle(arguments, staging)
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        publish(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    summary = {
        label: {kind: [round(item["minimum"], 9), item["minimum_decision"], item["top1_agreements"]]
                for kind, item in kinds.items()}
        for label, kinds in manifest["comparisons"].items()
    }
    print(json.dumps({"bundle": str(destination), "graph": manifest["graph"], "minimum_cosines": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
