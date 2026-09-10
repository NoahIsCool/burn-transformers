#!/usr/bin/env python3
"""Independently check each real first Burn gradient/update with CPU PyTorch.

This never trains on or selects against the validation documents. The exported
batch contains only the first training document, frozen backbone outputs and
weights, and the first Burn update. All probability analysis uses F64 over the
unchanged F32 logits; the transformer and affine translator computation is F32.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def tensor(path, shape):
    values = np.fromfile(path, dtype="<f4")
    if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError(f"{path}: expected finite F32 tensor of shape {shape}")
    return torch.from_numpy(values.reshape(shape))


def error(left, right):
    return float((left.detach().double() - right.detach().double()).abs().max())


def reference_step(hidden, teacher, unembedding, gamma, beta, lr):
    """Exactly the declared identity + affine, norm, unembedding, and SGD graph."""
    width = hidden.shape[-1]
    correction = torch.nn.Linear(width, width, bias=True)
    with torch.no_grad():
        correction.weight.zero_()
        correction.bias.zero_()
    optimizer = torch.optim.SGD(correction.parameters(), lr=lr, momentum=0.9,
                               dampening=0.0, nesterov=True, weight_decay=0.001)
    translated = hidden + correction(hidden)
    logits = F.layer_norm(translated, [width], gamma, beta, eps=1e-5) @ unembedding
    teacher_logp = teacher.double().log_softmax(-1)
    student_logp = logits.double().log_softmax(-1)
    loss = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1).mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(correction.parameters(), 1.0)
    gradients = [correction.weight.grad.detach().T.clone(), correction.bias.grad.detach().clone()]
    optimizer.step()
    return {"loss": float(loss.detach()), "gradient_norm_before_clipping": float(grad_norm),
            "gradient_weight": gradients[0], "gradient_bias": gradients[1],
            "updated_weight": correction.weight.detach().T, "updated_bias": correction.bias.detach()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=HERE.parents[2] / "target/mechanistic/tuned_lens_training")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((HERE / "manifest.json").read_text())
    tol = manifest["tolerances"]
    results = []
    for stage in manifest["protocol"]["stages"]:
        root = args.root / f"stage_{stage}_first_step"
        rust = json.loads((root / "metrics.json").read_text())
        rows, width, vocab = rust["shape"]
        measured = reference_step(tensor(root / "hidden.f32", (rows, width)),
                                  tensor(root / "teacher.f32", (rows, vocab)),
                                  tensor(root / "unembedding.f32", (width, vocab)),
                                  tensor(root / "norm_weight.f32", (width,)),
                                  tensor(root / "norm_bias.f32", (width,)),
                                  manifest["protocol"]["optimizer"]["base_learning_rate"])
        deviations = {"loss": abs(measured["loss"] - rust["loss"])}
        for name in ("gradient_weight", "gradient_bias", "updated_weight", "updated_bias"):
            expected = tensor(root / f"{name}.f32", tuple(measured[name].shape))
            deviations[name] = error(measured[name], expected)
        passed = deviations["loss"] <= tol["reference_loss_abs"] and all(
            value <= tol["reference_gradient_max_abs"] for name, value in deviations.items() if name != "loss")
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.iterdir()) if path.is_file()}
        results.append({"stage": stage, "passed": passed, "rust_loss": rust["loss"],
                        "torch_loss": measured["loss"], "torch_gradient_norm": measured["gradient_norm_before_clipping"],
                        "max_abs_errors": deviations, "inputs_sha256": hashes})
        print(f"stage {stage}: {deviations}; passed={passed}", flush=True)
    report = {"status": "passed" if all(item["passed"] for item in results) else "failed",
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "manifest_sha256": hashlib.sha256((HERE / "manifest.json").read_bytes()).hexdigest(),
              "reference_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "stages": results}
    (args.root / "reference.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["status"] != "passed":
        raise SystemExit("independent gradient/update check failed; reference.json retained")


if __name__ == "__main__":
    main()
