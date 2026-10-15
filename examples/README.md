# Transformer experiments: train, inspect, and change

Start with a short Rust example, then change its architecture, activation
selection, or inserted module. Models and learned additions remain ordinary
Burn `Module` values. Read the [activation guide](../EXPERIMENTS.md) and
[training guide](../TRAINING.md) for the shared interface and training contracts.

## Start small

These examples require no downloads and train on Flex CPU. The tasks are
synthetic integration checks, not pretrained-model quality benchmarks.

| Example | Change you can make | Run with |
| --- | --- | --- |
| [T5 classifier](t5_classifier.rs) | Per-layer heads/MLP widths, an inserted adapter, and a classification head. | `--features autodiff --example t5_classifier -- --train` |
| [Custom composition](custom_composition.rs) | Replace attention with a sequence mixer and connect different widths. | `--features autodiff --example custom_composition -- --train` |
| [MLA encoder-decoder](mla_encoder_decoder.rs) | Edit compressed cross-attention activations and use bidirectional inputs. | `--features autodiff --example mla_encoder_decoder -- --train` |
| [Frozen Qwen adapter](train_qwen_adapter.rs) | Train inserted processing through later frozen blocks; add an intermediate loss. | `--features autodiff --example train_qwen_adapter` |
| [Native Burn Learner](burn_training.rs) | Use Burn's own training framework, dropout, metrics, and checkpointer. | `--features train --example burn_training` |

Prefix each row's arguments with `cargo run --release --locked`, for example:

```sh
cargo run --release --locked --features autodiff --example t5_classifier -- --train
```

The T5, composition, and MLA examples also retain their original untrained
forward demonstrations: omit `-- --train`. Their [training guide](training/README.md)
describes the held-out tasks, gradients, and reload controls. The
[Qwen guide](training/qwen_adapter.md) also covers loading a real local prefix;
its eight-example fixture measures training accuracy only.

For an introduction to tensor selection and intervention without training:

```sh
cargo run --release --locked --example activation_workbench
cargo run --release --locked --example qwen_prefix -- --random
```

[activation_workbench.rs](activation_workbench.rs) captures selected neurons,
ablates and restores them, inserts an adapter, and allows future tokens through
attention. It then runs one family-generic experiment, written once against
the `HookedModel` and `StagedModel` traits (an activation cache of every point,
a residual ablation, and staged clean-to-corrupted patching), on a modular
decoder, GPT-2, and a Llama-family model selected by `AutoCausalLm`.
[qwen_prefix.rs](qwen_prefix.rs) demonstrates block selection and optional
local weight loading. Their random task outputs are not learned results.

## 1. Grokking: train, then find the circuit

**Paper:** Nanda et al., [Progress measures for grokking via mechanistic
interpretability](https://arxiv.org/abs/2301.05217).

[grokking_train.rs](grokking_train.rs) trains a one-block transformer from random
weights on modular addition. The full fixed protocol runs 40,000 AdamW updates,
records training and held-out curves, saves model/optimizer checkpoints, and
then preserves, removes, and restores the learned Fourier components through
public hooks. It needs no model or dataset download.

```sh
# Ten CPU updates check execution; they do not establish grokking.
cargo run --release --locked --features autodiff --example grokking_train -- --steps 10 --output target/mechanistic/grokking-smoke

# Full experiment on Apple Silicon; omit metal/backend for the slower CPU run.
cargo run --release --locked --features autodiff,metal --example grokking_train -- --backend metal --output target/mechanistic/grokking-training-metal-seed0
```

The measured full run took about 6.6 minutes after compilation. It showed a
prolonged train/test gap before both accuracies reached 100%. Final full-domain
accuracy was 100% with the selected subspace, 9.60% after removing it, and 100%
after restoration. This is one seeded realization, with learned frequencies
and diagnostic differences reported explicitly.

Read the [training protocol, learning curve, and results](mechanistic/grokking_training/README.md).
The separate [published-checkpoint analysis](mechanistic/grokking.md) retains its
reproduction of Table 1's ten numbers; it is a numerical reference, not evidence
that the repository trained those published weights.

## 2. Clock and Pizza: change attention itself

**Paper:** Zhong et al., [The Clock and the Pizza](https://arxiv.org/abs/2306.17844v2).

[clock_and_pizza_train.rs](clock_and_pizza_train.rs) trains addition modulo 59
using unmasked attention with `A′ = alpha*A + (1-alpha)*J`, where `J` contains
ones. Train both `alpha=0` and `alpha=1` from random weights, then inspect
embedding circles, distance dependence, and gradient symmetry.

```sh
cargo run --release --locked --features autodiff,metal --example clock_and_pizza_train -- --paper --backend metal --alpha 0 --output target/mechanistic/clock_and_pizza_training/alpha0-paper
cargo run --release --locked --features autodiff,metal --example clock_and_pizza_train -- --paper --backend metal --alpha 1 --output target/mechanistic/clock_and_pizza_training/alpha1-paper
```

Each condition uses 20,000 full-batch updates. A five-update `--smoke` mode checks
execution only. The [training guide and measured geometry](mechanistic/clock_and_pizza_training/README.md)
report the actual learned algorithms; an attention setting alone does not prove
that a particular mechanism emerged. Both measured runs reached 100% held-out
accuracy in about four minutes each, but neither passed the paper's strict
circularity threshold. These runs establish task learning and diagnostic access,
not reproduction of the representative circular mechanisms. The
[released-model analysis](mechanistic/clock_and_pizza/README.md)
retains Table 2 parity and its documented published accuracy discrepancy.

## 3. Tuned Lens: learn intermediate readouts

**Paper:** Belrose et al., [Eliciting Latent Predictions from Transformers with
the Tuned Lens](https://arxiv.org/abs/2303.08112v4).

[tuned_lens_train.rs](tuned_lens_train.rs) initializes new affine translators at
identity and trains them against frozen Pythia-160M predictions. It uses three
preselected stages, disjoint training/held-out documents, the full vocabulary,
and native SGD with momentum. The pretrained readout remains frozen while
propagating gradients to each translator.

```sh
python3 examples/mechanistic/tuned_lens/prepare.py
cargo run --release --locked --features autodiff,tokenizers --example tuned_lens_train
```

All three newly trained translators improved held-out KL and next-token loss
under the fixed small protocol; training took about a minute on the recorded
CPU host. Preparation downloads roughly 781 MB once. Read the
[training guide and results](mechanistic/tuned_lens_training/README.md) for its
short-context/data-budget adaptations and independent gradient/update checks.
[tuned_lens.rs](tuned_lens.rs) remains an analysis of the released translators.

## 4. Greater-Than: learn a replacement inside GPT-2

**Paper:** Hanna et al., [How does GPT-2 compute greater-than?](https://arxiv.org/abs/2305.00586).

The original paper analyzes an already trained model. [greater_than_train.rs](greater_than_train.rs)
adds a **new distillation extension**: train a small module to replace a late
MLP activation while later GPT-2 blocks and the readout remain frozen. Clean
intermediate features explicitly supply the year information removed from the
corrupted prompt. This is not GPT-2 pretraining or the paper's own training method.

```sh
python3 examples/mechanistic/greater_than/prepare.py --checkpoint /local/gpt2
cargo run --release --locked --features autodiff --example greater_than_train
```

The [training protocol and results](mechanistic/greater_than_training/README.md)
define the development/held-out split, teacher, optimizer budget, controls, and
recovery metrics. The [original activation-patching study](mechanistic/greater_than/README.md)
remains available as the published-mechanism comparison on 194 fixed prompt pairs.
The new fixed run reduced held-out teacher-to-student KL from 0.5622 to 0.0500
nats, about 91%, with unchanged backbone weights and an exact resumed update.
It recovered much of the teacher patch's effect, without matching the clean
model's full greater-than behavior.

## An exact attention identity

[attention_head_attribution.rs](attention_head_attribution.rs) numerically
reproduces the independent/additive-head decomposition in Elhage et al.'s
[Transformer Circuits framework](https://transformer-circuits.pub/2021/framework/index.html#architecture-attn-independent).
It checks reconstruction, ablation, restoration, and future-token visibility:

```sh
cargo run --release --locked --example attention_head_attribution
```

This algebraic identity needs no training. Fixed weights make its numerical
answer transparent; it does not claim discovery of a learned circuit.

## Evidence and inference references

Each study preserves its protocol, successful and failed controls, source/artifact
hashes, and numerical claim boundaries. Short smoke runs check code; full training
and held-out measurements establish the separately stated empirical results.
The [historical v0.2 validation](../acceptance/measurements/v0_2-interface-validation-20260910-retry/README.md)
records the ten original examples at their earlier source snapshot.
The [v0.3 validation](../acceptance/measurements/v0_3-training-validation-20260910/README.md)
adds native training checks, the eleven inexpensive example executions, and
both compiler profiles. Longer scientific runs retain their individual evidence.

## Load a checkpoint and generate

These programs read a local checkpoint directory; none downloads anything.

| Example | Checkpoints | Run with |
| --- | --- | --- |
| [llama_generate.rs](llama_generate.rs) | Any `llama`, `mistral`, `qwen2`, `qwen3`, `smollm3`, or `olmo2` directory | `--features tokenizers --example llama_generate -- MODEL_DIR ["PROMPT"] [MAX_NEW_TOKENS]` |
| [gemma_generate.rs](gemma_generate.rs) | Gemma 2, Gemma 3 text, or the text tower of multimodal Gemma 3 | `--features tokenizers --example gemma_generate -- MODEL_DIR ["PROMPT"] [MAX_NEW_TOKENS]` |
| [flan_t5_small.rs](flan_t5_small.rs) | The pinned FLAN-T5-small revision; cache-aware four-beam translation | `--no-default-features --features std,flex,tokenizers --example flan_t5_small -- MODEL_DIR "PROMPT"` |
| [qwen35_chat.rs](qwen35_chat.rs) | The pinned Qwen3.5-0.8B revision on Metal; chat-template rendering | see [the Qwen profile](../QWEN3_5_0_8B_RELEASE.md) |

Prefix each row with `cargo run --release --locked`. The Llama and Gemma
examples require the `tokenizers` feature, cast weights to F32, and run on the
Flex CPU backend; an F32 Gemma 3 4B needs about 16 GiB of memory. Follow each
example's feature and artifact requirements. Training compatibility does not extend checkpoint
support claims; the [model status page](../ARCHITECTURE_COMPATIBILITY.md)
records which checkpoints are validated.
