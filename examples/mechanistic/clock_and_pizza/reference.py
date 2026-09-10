"""Small, independent CPU reference for the Clock and Pizza checkpoints.

This translates the author's equations directly; it never imports their code.
The Rust example is the experiment. This file supplies its numerical control.
"""

import math

import numpy as np
import torch


def forward(weights, tokens, attention_coefficient, embedding=None):
    """Return last-token logits; no normalization, masking, or dropout is used."""
    if embedding is None:
        embedding = weights["embed.W_E"].T
    hidden = embedding[tokens] + weights["pos_embed.W_pos"]
    attention = "blocks.0.attn."
    query, key, value = [
        torch.einsum("hdk,bpk->bhpd", weights[attention + name], hidden)
        for name in ("W_Q", "W_K", "W_V")
    ]
    scores = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    probabilities = scores.softmax(dim=-1)
    # J is all ones, not a row-normalized uniform distribution. The checkpoint
    # contains a triangular mask buffer, but the author's forward never uses it.
    probabilities = (
        attention_coefficient * probabilities + (1 - attention_coefficient)
    )
    attended = (probabilities @ value).transpose(1, 2).flatten(2)
    hidden = hidden + attended @ weights[attention + "W_O"].T
    mlp = "blocks.0.mlp."
    intermediate = (hidden @ weights[mlp + "W_in"].T + weights[mlp + "b_in"]).relu()
    hidden = hidden + intermediate @ weights[mlp + "W_out"].T + weights[mlp + "b_out"]
    return (hidden @ weights["unembed.W_U"])[:, -1, :].contiguous()


def centered_pca(embedding):
    """Match PCA(20)'s centered full SVD, without a scikit-learn dependency.

    Components have shape [hidden, 20]. Reconstruction includes the mean:
    mean + scores[:, selected] @ components[:, selected].T.
    """
    values = embedding.numpy()
    mean = values.mean(axis=0, keepdims=True)
    left, singular, right = np.linalg.svd(values - mean, full_matrices=False)
    # Fix arbitrary SVD signs for readable, repeatable exported coordinates.
    signs = np.sign(right[np.arange(len(right)), np.abs(right).argmax(axis=1)])
    right *= signs[:, None]
    left *= signs[None, :]
    return {
        "pca.mean": torch.from_numpy(mean.copy()),
        "pca.components": torch.from_numpy(right[:20].T.copy()),
        "pca.scores": torch.from_numpy((left[:, :20] * singular[:20]).copy()),
    }


def reconstruct(pca, selected):
    return (
        pca["pca.mean"]
        + pca["pca.scores"][:, selected] @ pca["pca.components"][:, selected].T
    )


def fraction_variance_explained(observed, predicted):
    """Paper Table 2: separately standardize the complete tensors, then 1-var(diff)."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = (observed - observed.mean()) / observed.std(ddof=0)
    predicted = (predicted - predicted.mean()) / predicted.std(ddof=0)
    return float(1 - (observed - predicted).var(ddof=0))


def metrics(logits, tokens):
    """Evaluate every input pair; there is no sampled split or fitted threshold."""
    modulus = logits.shape[-1]
    a, b = tokens.T.numpy()
    values = logits.numpy().astype(np.float64)
    labels = (a + b) % modulus
    correct_logits = values[np.arange(len(values)), labels]
    by_sum_and_difference = np.empty((modulus, modulus), dtype=np.float64)
    by_sum_and_difference[labels, (a - b) % modulus] = correct_logits
    q = by_sum_and_difference.std(axis=0, ddof=0).mean() / correct_logits.std(ddof=0)
    return {
        "accuracy": float(np.mean(values.argmax(axis=-1) == labels)),
        "correct_predictions": int(np.sum(values.argmax(axis=-1) == labels)),
        "input_pairs": len(values),
        "distance_irrelevance_q": float(q),
    }


def circle_metrics(logits, frequency):
    modulus = logits.shape[-1]
    a, b, c = np.ogrid[:modulus, :modulus, :modulus]
    clock = np.cos(2 * math.pi * frequency * (a + b - c) / modulus)
    pizza = np.abs(np.cos(math.pi * frequency * (a - b) / modulus)) * clock
    observed = logits.numpy().reshape(modulus, modulus, modulus)
    return {
        "frequency": frequency,
        "clock_fve": fraction_variance_explained(observed, clock),
        "pizza_fve": fraction_variance_explained(observed, pizza),
    }


@torch.inference_mode()
def calculate(weights, model, protocol):
    """Export PCA coordinates and five complete reference forward passes."""
    modulus = protocol["modulus"]
    tokens = torch.cartesian_prod(torch.arange(modulus), torch.arange(modulus))
    pca = centered_pca(weights["embed.W_E"].T)
    tensors, measurements = dict(pca), {}
    for name, selected in protocol["interventions"].items():
        embedding = None if selected is None else reconstruct(pca, selected)
        logits = forward(weights, tokens, model["attention_coefficient"], embedding)
        tensors[f"reference.{name}"] = logits
        measurements[name] = metrics(logits, tokens)
    # These published frequencies identify Model A's circles. Applying them to
    # Model B's different embedding basis would not be a corresponding control.
    if model["attention_coefficient"] == 0:
        for name, frequency in protocol["circle_frequencies"].items():
            measurements[name].update(circle_metrics(tensors[f"reference.{name}"], frequency))
    return tensors, measurements
