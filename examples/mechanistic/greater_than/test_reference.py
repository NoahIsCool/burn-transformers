"""Small mechanics checks; no checkpoint or study run is required."""

import copy
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from prepare import strict_json, validate_manifest, HERE
from reference import analytic_metrics, capture_readout, forward, interventions, probability_metrics, run_condition


class ReferenceMechanicsTests(unittest.TestCase):
    def test_full_vocabulary_mass_and_cutoff_are_not_numeric_renormalization(self):
        # Most mass is on a non-numeric token. The numeric subset must retain it.
        logits = torch.tensor([[0.0, 1.0, -1.0, 7.0, 0.5]], dtype=torch.float32)
        actual = probability_metrics(logits, [0, 1, 2], [1])[0]
        independent = analytic_metrics(logits[0].numpy(), [0, 1, 2], 1)
        self.assertLess(actual["numeric_mass"], 0.01)
        self.assertLess(actual["pd"], 0.0)
        self.assertAlmostEqual(actual["pd"], independent["pd"], places=15)
        self.assertAlmostEqual(actual["sharpness"], actual["probabilities"][2] - actual["probabilities"][0], places=15)
        for a, b in zip(actual["probabilities"], independent["probabilities"]):
            self.assertAlmostEqual(a, b, places=15)
        shifted = probability_metrics(logits + 10000.0, [0, 1, 2], [1])[0]
        self.assertEqual(actual, shifted)

    def test_callbacks_edit_only_selected_token_restore_in_order_and_clean_up(self):
        torch.manual_seed(4)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=11, n_positions=8, n_embd=12, n_layer=2, n_head=3,
            activation_function="gelu", attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
            attn_implementation="eager",
        )).eval()
        ids = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            original_outputs = []
            handle = model.transformer.h[0].mlp.register_forward_hook(
                lambda _module, _inputs, output: original_outputs.append(output.clone()))
            try:
                with capture_readout(model, 2) as capture:
                    baseline = forward(model, ids)
            finally:
                handle.remove()
            observed = []
            with interventions(model, [(0, None)], 2):
                handle = model.transformer.h[0].mlp.register_forward_hook(
                    lambda _module, _inputs, output: observed.append(output.clone()))
                try:
                    zeroed = forward(model, ids)
                finally:
                    handle.remove()
            self.assertTrue(torch.equal(observed[0][:, :2], original_outputs[0][:, :2]))
            self.assertEqual(torch.count_nonzero(observed[0][:, 2]).item(), 0)
            self.assertFalse(torch.equal(baseline, zeroed))
            with interventions(model, [(0, None), (0, capture[0])], 2):
                restored = forward(model, ids)
            self.assertTrue(torch.equal(baseline, restored))
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with interventions(model, [(0, None)], 2):
                    raise RuntimeError("intentional failure")
            self.assertTrue(torch.equal(baseline, forward(model, ids)))
            self.assertEqual(len(model.transformer.h[0].mlp._forward_hooks), 0)

    def test_condition_dispatch_preserves_patch_direction_and_direct_readout(self):
        torch.manual_seed(5)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=11, n_positions=8, n_embd=12, n_layer=2, n_head=3,
            activation_function="gelu", attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
            attn_implementation="eager",
        )).eval()
        clean_ids, corrupt_ids = torch.tensor([[1, 2, 3]]), torch.tensor([[1, 4, 3]])
        manifest = {"readout_position": 2}
        with torch.inference_mode():
            with capture_readout(model, 2) as clean:
                forward(model, clean_ids)
            with capture_readout(model, 2) as corrupt:
                forward(model, corrupt_ids)
            for condition, destination, source in (
                ("node_patch_0", corrupt_ids, clean),
                ("node_reverse_0", clean_ids, corrupt),
            ):
                def paired_replacement(_module, _inputs, output):
                    replaced = output.clone()
                    replaced[:, 2:3] = source[0]
                    return replaced

                handle = model.transformer.h[0].mlp.register_forward_hook(paired_replacement)
                try:
                    expected = forward(model, destination)
                finally:
                    handle.remove()
                actual = run_condition(model, condition, clean_ids, corrupt_ids, clean, corrupt, manifest)
                self.assertTrue(torch.equal(actual, expected), condition)

            # A direct contribution must use captured residuals without running
            # any transformer block again; node patching does run those blocks.
            def reject_block_execution(_module, _inputs):
                raise AssertionError("direct readout recomputed a transformer block")

            handles = [block.register_forward_pre_hook(reject_block_execution)
                       for block in model.transformer.h]
            try:
                direct = run_condition(model, "direct_0", clean_ids, corrupt_ids, clean, corrupt, manifest)
                residual = clean["residual"] - clean[0] + corrupt[0]
                expected = model.lm_head(model.transformer.ln_f(residual))[:, 0]
                self.assertTrue(torch.equal(direct, expected))
            finally:
                for handle in handles:
                    handle.remove()

    def test_protocol_rejects_missing_condition_and_wrong_corruption_position(self):
        manifest = strict_json((HERE / "manifest.json").read_bytes())
        validate_manifest(manifest)
        missing = copy.deepcopy(manifest)
        missing["conditions"].pop()
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_manifest(missing)
        changed = copy.deepcopy(manifest)
        changed["changed_position"] = 7
        with self.assertRaisesRegex(ValueError, "dimensions"):
            validate_manifest(changed)

    def test_non_finite_logits_are_rejected_before_analysis(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            probability_metrics(torch.tensor([[0.0, float("nan"), 2.0]]), [0, 1, 2], [1])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
