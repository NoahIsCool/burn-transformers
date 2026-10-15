# Nemotron-H 3.5 Lightning acceptance protocol

This document defines the artifact and numerical gates for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at immutable revision
`a9904d24bcc1d289a1950fa9d2b978c47cf903b9`. The exact local artifact has passed
full-payload authentication, strict source-ordered F16 application, and a short
Burn Metal base-model smoke. That smoke is internal-consistency evidence; the
separately pinned official oracle has not yet been compared with Burn. A
path-safe ignored Burn-versus-official gate is now implemented and compiles,
but it has deliberately not been executed as part of that smoke. Real-weight
MTP parity and the remaining release gates are also incomplete, so the model is
not advertised as supported. The committed machine-readable state remains
[`acceptance/nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json`](acceptance/nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json).
An external, authenticated seven-file base-model oracle now proves the official
Transformers graph can load and execute the complete language tower. It does
not execute the 270 physical `mtp.*` tensors and does not make the checkpoint
advertised as supported. Its independent Rust reader pins every external file
and keeps the separate MTP evidence path fail-closed.

## Pinned evidence

The exact 2,352-byte `config.json` has SHA-256
`a3827a0f5e311547b40943dc081e3ff2f8a277466e8c1a3df2291e8db8a7617c`.
The exact 612,207-byte `model.safetensors.index.json` has SHA-256
`67f21da80ce245a3e24967f54eef3fa10e67a63eba16ef19a2d0569f06103f50`.
Committed prefix-plus-header fixtures cover all fourteen SafeTensors shards.
Ordinary tests parse these bytes without tensor allocation and validate every
name, shard assignment, dtype, shape, contiguous byte range, header length, and
file length.

The physical inventory contains 6,513 tensors: 6,242 under `backbone.*`, one
LM head, and 270 under `mtp.*`. It contains 6,489 BF16 tensors and 24 FP32
router-correction buffers. Tensor payloads total 65,826,538,624 bytes; complete
shards including headers total 65,827,374,264 bytes. The canonical inventory
anchors are:

- index name/shard assignments:
  `0e3b013820bc7ef347c3daf15f3886a2f3fa8e6e3c38e1859662b624601eda09`;
- physical name/shard/dtype/shape/range inventory:
  `e255cec4fe6d43dfddaed52b3a5ef719c111d6c9821ce7ff53219080b50762c4`;
- corrected-metadata F16 load plan:
  `a47056d8d8e5ad814f79f4b147c3d7b340362013c31f126b7578e25efb51c906`;
- adversarial component oracle:
  `3b1549cede5872148a452eb8428952e6ed1f4cf56c2dbbd264b56d4f536d08bc`.

The corrected-metadata plan remains a test-only independent proof: after
substituting the physical byte total, the family contract maps all 6,513 source
tensors, retains the 24 FP32 correction buffers, casts 6,489 BF16 tensors to
F16, and has no aliases or unexplained keys. Production loading does not mutate
that metadata. It uses the provenance-bound accounting policy described below,
then executes the otherwise identical strict plan.

The index's `metadata.total_parameters = 31,577,937,344` is internally
reconcilable: the base tower plus LM head contain 31,577,940,288 physical
elements before MTP, and the 2,944-element difference is exactly the 23
128-element base router correction buffers. The pinned Transformers source
declares these as non-trainable buffers. The separate MTP body contains
1,335,325,952 physical elements and is not included in that reported base
parameter count.

## Upstream `total_size` defect

The upstream index reports `metadata.total_size = 65,842,365,568`, while the
exact physical headers sum to 65,826,538,624 tensor bytes. The difference is
15,826,944 bytes. This is not an approximate reconciliation:

```text
23 base MoE routers * 128 experts * 2,688 hidden units * 2 BF16 bytes
= 15,826,944 bytes
```

All 23 `backbone.layers.*.mixer.gate.weight` matrices occur exactly once in the
index and exactly once in the physical headers. Each has shape `[128, 2688]`
and occupies 688,128 bytes. The separate
`mtp.layers.1.mixer.gate.weight` is also physical and occupies 688,128 bytes,
but including it would not match the discrepancy. Thus the producer appears to
have counted only the 23 base routers twice in `metadata.total_size`; it did not
duplicate their keys or payloads.

The generic checkpoint loader continues to reject this mismatch. The
implemented Nemotron-only resolution is a narrow accounting policy bound to the
exact repository revision, raw config SHA-256, raw index SHA-256, both declared
and physical totals, the exact 23 router names discovered as BF16
`[128, 2688]`, and the separately required MTP router. It interprets the
reported total as physical bytes plus those exact base routers for accounting
validation only. Adversarial tests reject near-miss labels, hashes, totals,
names, shapes, dtypes, missing routers, and an incorrectly included MTP router.
The policy does not change the physical inventory, mutate the checkpoint, or
weaken generic exact-size validation.

## Burn Metal implementation smoke

Commit `7c1f816f520bb05c33c22774452cae434e305784` was executed from a clean
worktree with repository tree `dd342dc91ff8782cb4d5c97142cf685861f24465`
and transformer subtree `c37026d9bdf5694a50a7d294c7b52930eb97abd9`.
The release build used `std,metal,tokenizers` with Rust
`1.94.0-nightly (4f14395c3 2025-12-22)` on `aarch64-apple-darwin`.

The gate reauthenticated all 65,827,374,264 artifact bytes before loading. It
then applied all 6,513 tensors, including the 270 MTP tensors, with no alias,
allowlisted omission, or unexpected key. It converted all 6,489 BF16 tensors
to F16 and retained all 24 FP32 correction buffers. Pass two read each source
once, delivered 65,826,538,624 bytes, opened the fourteen shards exactly in
`00001` through `00014` order, kept at most one shard open, and observed an
actual peak executor-owned host staging allocation of 704,643,072 bytes. The
clean run loaded and synchronized parameters in 102.101 seconds.

The base smoke used the fixed six-token prompt `[22177, 4304, 1044, 57424,
59417, 1046]`. Stateless and cached-prefill logits had cosine `1.0`, while all
54 semantic last-token hidden rows were bit-identical; their little-endian F32
hidden-trace SHA-256 was
`80249a14decaa09f775b5e6bd469d17a5515e6c3cfc54863ef9bb93801cd2f5c`.
The first greedy token was `1256`. One recurrent decode step produced hidden
trace SHA-256
`eac085e4be9796ba5f828ccd27e8caeda7294228333911beaf6bf33b74e4fc3c`
and matched a stateless teacher-forced prefix at full-vocabulary cosine
`0.999376514` with identical top-1. Cache inspection proved six F16 GQA layers,
23 Mamba layers with FP32 recurrent matrices and F16 convolution history, and
the exact cursor transition `0 -> 6 -> 7`.

The first real run also found a checkpoint-specific runtime defect: Lightning
sets `n_group = topk_group = 1`, but Burn rejects a one-hot tensor with only one
class. A sole retained router group makes group pruning mathematically the
identity, so the production path now bypasses one-hot construction and sends
all corrected scores directly to expert top-k. A deterministic regression test
proves that path matches a multi-group configuration in which every group is
retained.

None of this is official-reference parity. That real-weight smoke installed the
MTP tensors but did not execute them. The exact one-stage body, dedicated cache,
and speculative verification protocol now execute in tiny Flex tests only. No
real MTP result, 32-token generation, quantization, or 32K-context result is
implied.

## Split numerical reference

No single pinned upstream runtime is authoritative for both halves of this
checkpoint.

The target backbone is pinned to Hugging Face Transformers commit
[`cfa309cfa34b5497a32fd50af49aa97aac94c054`](https://github.com/huggingface/transformers/tree/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/nemotron_h).
Its `NemotronHForCausalLM` implementation provides the backbone equations, but
explicitly ignores `mtp.*` as unexpected load keys. Its exact modeling source
is 50,797 bytes with SHA-256
`d42067429e3268d1163579f5d257fe0baa9a2b02edba5fcb175fb8d42c9b6af4`;
its configuration source is 13,031 bytes with SHA-256
`00a4c60bd302d796b25c6ae25c608a5ee7e27f9a0adea92911b0c45470232052`.
The base-reference workflow must independently count all 270 raw `mtp.*`
checkpoint keys, then require Transformers to report zero of them after its
explicit unexpected-key ignore filter; the reference record carries both
counts so filtering cannot be mistaken for execution.

Transformers' generic `MtpModel` at the same commit is not a viable fallback.
Its 30,974-byte `src/transformers/modeling_layers.py` (SHA-256
`81469b99e3f933ad78eaf5c90d401a95a02a5f569732c6ed2b7020daaed31953`)
requires a `num_mtp_layers` config field, whereas the exact checkpoint and
Nemotron-H config expose `num_nextn_predict_layers`. Its generic construction
also instantiates one decoder block per prediction stage by copying the final
base-layer class, while this checkpoint stores an ordered two-block
attention-plus-MoE stage. Renaming the count field alone would therefore not
make that path an authoritative execution of these 270 tensors.

The checkpoint card pins vLLM v0.27.1 for serving, and that exact release
contains a dedicated executable Nemotron-H MTP graph. Its commit is
[`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`](https://github.com/vllm-project/vllm/tree/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac).
The pinned `nemotron_h_mtp.py` is 18,008 bytes with SHA-256
`0a799030f5bda1b564b749aaefe6cc3f479c90ff28bd449070b4bb7a21a10176`;
its imported `nemotron_h.py` is 32,582 bytes with SHA-256
`4a0d76f7b7b8c9902adb65b0b2700e2924786cf2ba337f3ff0506fe75ff1ae67`;
and its configuration adapter is 13,414 bytes with SHA-256
`7cbf03b9d0c1b8734074101d9178cfd5785b1a9191148396f7ca971f7cc709d7`.
The card's explicit SGLang MTP example names a mutable `dev` container rather
than an immutable source revision, so it cannot satisfy this oracle's source
identity gate without an independently pinned image digest and source tree.

The physical MTP body is independently testable: all 270 keys are isolated in
shard 14, which contains no base keys. Eight keys implement its full-attention
block and 262 its MoE block. The executable ordering is normalized shifted
token embedding followed by normalized target hidden state, concatenation and
projection, attention residual, MoE residual, then final normalization. The
MTP graph shares the target embedding and LM head.

## Authenticated official base-model oracle

[`scripts/generate_nemotron_h_base_reference.py`](scripts/generate_nemotron_h_base_reference.py)
at Git commit `0f222f6213ebf320a5fd91221021d5499622987d`
(139,608 bytes, SHA-256
`10ab5c7a6aea744ba28120a2b5415ff91c2260e9f01da69c583a1cc4f060b94a`)
generated the external directory
`nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle`. The generator
authenticates the exact config, index, fourteen shards, pinned Transformers
checkout and sources, and eleven primary distribution payload manifests before
importing the model. Their combined artifact and distribution manifest SHA-256
values are `cb72063ee5b215548d9adbc67052de5f54871f1e28b80a0e0db9782afd4ad759`
and `6eca7252cebb0750fa344a7d05419a4511f3ecf985e6f893dadf3e0e94c2b2a6`.
It enforces and records the CPython runtime but explicitly treats interpreter,
standard-library, system-library, and transitive-distribution bytes as trusted
environment inputs rather than claiming to authenticate them. It repeats all
content checks after inference, blocks network, remote code and external
kernels, requires a clean committed copy of itself, and writes through a hidden
sibling directory followed by one atomic no-replace rename. Every final file is
create-only; failed runs leave no partial final bundle.

The official F16 MPS eager graph loaded 31,577,937,344 parameter elements. Its
401 state entries include 23 FP32 router-correction buffers totaling 2,944
elements. Its strict loading report contains zero missing, unexpected,
mismatched, or error entries. Before
loading, the independent header parser counted all 6,513 physical tensors,
including exactly 270 `mtp.*` tensors with name-manifest SHA-256
`9dceb828e3bb078ebb0dfe24b1e7257ba6bb2f9c49c0be14539caad770078e48`.
The pinned class's sole unexpected-key filter is exactly `mtp.*`; after that
filter, no unexplained load anomaly remains. The record explicitly says those
MTP tensors were not instantiated or executed.

The literal prompt `Hello world, cached decoding.` tokenizes without added
special tokens to `[22177, 4304, 1044, 57424, 59417, 1046]`. Model config
EOS/PAD remain 2/0, the tokenizer maps both EOS and PAD to 11, and generation
accepts EOS `[2, 11]`. Its binary16-stable greedy continuation is the pair
`[1256, 1010]` repeated sixteen times. The authenticated tokenizer maps those
IDs to `['ĠĠ', 'Ċ']`, yielding two spaces followed by a newline on every cycle.
This whitespace loop is retained as evidence rather than replaced with a more
favorable prompt.

The bundle preserves both official numerical paths: one full-vocabulary prompt
row, all 32 cached rows, all 32 teacher-forced rows, the 54-boundary stateless
prompt hidden trace, cached-prefill hidden trace, and first recurrent-decode
hidden trace. Stateless and cached prompt hidden payloads are byte-identical;
their logit cosine is `1.0000000000000009` under the recorded F64 reduction,
and the minimum corresponding hidden-boundary cosine is
`0.9999999999999998`. Cached and teacher-forced top-1 IDs agree for all 32
decisions, but the two official Mamba algorithms are not numerically
interchangeable: recurrent cached versus chunked teacher logits reach a minimum
cosine of `0.9215047947571176` at decision 22. At that row the dot product is
`401545.8810470642` and the two norms are `637.4856846878333` and
`683.5450448546004`. The largest absolute error is `3.314453125` at vocabulary
index 6507 (cached `5.6171875`, teacher `2.302734375`); the largest relative
error is `17301.86108714409` at index 67673 under denominator
`max(abs(teacher), 1e-12)`. Cached top two are token 1256 at `17.4375` and token
8129 at `11.6484375`, a `5.7890625` margin; teacher top two are token 1256 at
`17.03125` and token 1032 at `12.0390625`, a `4.9921875` margin. Thus the
crossed paths differ substantially while retaining the same stable top-1.

Per-layer localization first falls below 0.999 at decision 1 after MoE layer 27
(`0.9919212313408887`). The worst hidden boundary is decision 14 after MoE layer 1
(`0.4418906779817166`): the identical embedding first falls to
`0.8920338531579629` after Mamba layer 0, then drops another
`0.4501431751762463` at the following MoE boundary. This proves the recurrent
versus chunk-scan difference begins in Mamba and is amplified across the next
MoE block; expert-ID hooks were not captured, so it does not by itself prove
that different experts were selected.

This source self-divergence is diagnostic, not a relaxed parity threshold. A
Burn cached run must match the official cached rows, and a Burn stateless run
must match the official teacher rows, each at cosine at least 0.999. Crossing
the two official paths is gated by exact top-1 equality and preserved as two
separate oracle matrices, not mislabeled as a 0.999 comparison.

The ignored Rust parity gate encodes this distinction in its types. It first
compares a dedicated Burn stateless prompt row and all 32 positions from one
Burn teacher-forced chunk with the official stateless prompt/teacher payloads.
Separately, it compares Burn cached prefill plus 31 recurrent updates with all
32 official cached rows. Every full-vocabulary row requires cosine at least
0.999 and exact top-1, in contiguous decision order. Each of the 54 boundaries
in the stateless-prompt, cached-prefill, and first recurrent-decode hidden traces
also requires row-wise cosine at least 0.999 at the same semantic point. A
stateless/cached path or hidden-phase mismatch fails before numerical
comparison. This gate has passed compile-only and pure contract tests; until its
ignored 30B Metal run completes, it is executable policy rather than acceptance
evidence.

The exact official cache has physical batch one while `DynamicCache.batch_size`
reports its `-1` sentinel. Each of 23 Mamba layers stores an F16 convolution
tail `[1, 6144, 4]` and FP32 recurrent state `[1, 64, 64, 128]`; each of six
NoPE-GQA layers stores F16 key/value tensors `[1, 2, L, 128]`, growing from
`L=6` to `L=37`; the 23 MoE schedule slots remain uninitialized cache
placeholders. The generator validates every layer at every decision.

The published file anchors are:

- record JSON, 98,220 bytes:
  `bd3abb417f2fee82d1bef8bb654ca846fa4816351b1d9d7d0d057105f92de196`;
- prompt logits, 524,288 bytes:
  `f3c014bb043e9ffa3a114a4296ca0e0c2165c4abd63b942df6d910a4383a1dff`;
- stateless hidden trace, 580,608 bytes:
  `56033b284b0c926a40fd50e51eeb232412d3d77ae97282782dc39b9645ff0be2`;
- cached logits, 16,777,216 bytes:
  `b3a1ae948e55e6db409e0f9973c41c8767c874c6d64b1d351cfff7bbafa72480`;
- teacher logits, 16,777,216 bytes:
  `c1878c5e1396ddf01f3dfa85729ea58aa879ece28507399bdaf7addf4a65601b`;
- cached-prefill hidden trace, 580,608 bytes:
  `56033b284b0c926a40fd50e51eeb232412d3d77ae97282782dc39b9645ff0be2`;
- first cached-decode hidden trace, 580,608 bytes:
  `46c65b5cc91b53c3cdb2c3b6df9acaf4fbb216e91ae09f007c8d1b5571d93b79`.

[`tests/support/nemotron_h_base_reference.rs`](tests/support/nemotron_h_base_reference.rs)
independently pins those exact names, lengths, and digests. It rejects a
symlinked bundle or member, any missing or extra member, traversal, a changed
closed JSON schema, source/provenance drift, non-finite F32LE payload values,
descriptor mismatch, and changed numerical summaries. It also recomputes the
prompt ranking, all 64 cached/teacher argmaxes, and the minimum crossed-path
cosine from the payloads. Its accessors attach a stateless or cached path to
every returned row and an exact semantic point to every hidden trace.
[`tests/support/nemotron_h_parity.rs`](tests/support/nemotron_h_parity.rs)
enforces those identities, full-vector shape and finiteness, nonzero norms,
contiguous coverage, the 0.999 threshold, and exact stable top-1. The record
cannot authorize itself through hashes inside its own JSON.

The generator requires 40 GiB of free system-volume space before MPS load,
samples a 10 GiB runtime reserve after expensive forwards, and separately
requires 1 GiB on the external output volume; this closes the temporary
MPS-graph disk failure observed during an earlier concurrent diagnostic.

The committed synthetic fixture probes F16 RMSNorm rounding order, grouped
gated RMSNorm, correction-biased router selection with unbiased route weights,
and the MTP topology contract. A separate combined base-plus-MTP reader remains
intentionally fail-closed: its record and six payload anchors, including MTP
draft logits and an MTP hidden trace, are absent until a pinned vLLM execution
is independently reviewed. The accepted base-only bundle cannot satisfy or
weaken that MTP gate.

## Reproduction commands

The metadata, mapping, accounting-defect, MTP-inventory, and fail-closed oracle
tests require no weight payloads:

```sh
cargo test --manifest-path Cargo.toml \
  --no-default-features --features std \
  --test nemotron_h_artifacts
```

The generator's 34 pure tests exercise its closed schemas, integrity checks,
disk preflights, MTP filtering, cache validation, numerical diagnostics, and
atomic create-only publication without importing PyTorch or loading weights:

```sh
python3 -m unittest \
  scripts/tests/test_generate_nemotron_h_base_reference.py
```

After an external pinned download, authenticate all seven controls and every
byte of all fourteen weight shards without allocating tensors:

```sh
TRANSFORMERS_NEMOTRON_H_DIR=/absolute/path/to/pinned-checkpoint \
cargo test --release --manifest-path Cargo.toml \
  --no-default-features --features std,tokenizers \
  --test nemotron_h_artifacts \
  validates_pinned_nemotron_control_and_weight_payloads \
  -- --ignored --exact --nocapture
```

Strictly apply the complete F16 graph and run the bounded base Metal smoke with:

```sh
TRANSFORMERS_NEMOTRON_H_DIR=/absolute/path/to/pinned-checkpoint \
cargo test --release --manifest-path Cargo.toml \
  --no-default-features --features std,metal,tokenizers \
  --test nemotron_h_artifacts \
  loads_pinned_nemotron_f16_and_smokes_base_inference \
  -- --ignored --exact --nocapture --test-threads=1
```

Run the heavier corresponding-path Burn/official comparison only with both
authenticated local inputs and serialized Metal execution:

```sh
TRANSFORMERS_NEMOTRON_H_DIR=/absolute/path/to/pinned-checkpoint \
TRANSFORMERS_NEMOTRON_H_REFERENCE_DIR=/absolute/path/to/nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle \
cargo test --release --manifest-path Cargo.toml \
  --no-default-features --features std,metal,tokenizers \
  --test nemotron_h_artifacts \
  compares_pinned_nemotron_f16_with_official_base_reference \
  -- --ignored --exact --nocapture --test-threads=1
```

That command reauthenticates all checkpoint and oracle bytes, strictly reloads
the complete F16 graph, and checks corresponding streams only. It never treats
the official cached-versus-teacher cosine as a Burn parity criterion.

Generate a new immutable base oracle only from generator commit `0f222f6`, the
pinned CPython 3.11.9 environment, the exact clean Transformers checkout, and an
output directory in which the fixed bundle name does not already exist:

```sh
/absolute/path/to/pinned/python3.11 \
  scripts/generate_nemotron_h_base_reference.py \
  --model-dir /absolute/path/to/pinned-checkpoint \
  --transformers-dir /absolute/path/to/transformers-cfa309c \
  --output-dir /absolute/path/on/external-volume
```

Authenticate the independently pinned base-only bundle without loading a model:

```sh
TRANSFORMERS_NEMOTRON_H_REFERENCE_DIR=/absolute/path/to/nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle \
cargo test --manifest-path Cargo.toml \
  --no-default-features --features std \
  --test nemotron_h_artifacts \
  validates_pinned_nemotron_official_base_reference_bundle \
  -- --ignored --exact --nocapture
```

The future combined MTP reader remains a different ignored test and fails before
reading any bundle because none of its independent anchors is populated.

## Remaining release gates

This base oracle satisfies only the official-reference side of the numerical
contract. The recorded Metal smoke also predates later runtime changes: the
shared Mamba-2 mixer's vectorized chunk scan, PyTorch's thresholded `dt`
softplus, batched cached execution with padded queries pinned to their row's
last readable position, and one-pass MTP verification. Its stateless,
cached-prefill, and one-token decode gates must therefore be replayed on the
integrated source. Before this checkpoint can be advertised as supported:

1. Execute the implemented ignored Metal gate and record Burn cached execution
   against the official cached rows and Burn stateless execution against the
   official stateless/teacher-forced rows at cosine at least 0.999, with exact
   top-1 throughout. The gate already keeps the two official Mamba paths
   distinct; compile-only validation is not a numerical pass.
2. Generate the MTP evidence with pinned vLLM and independently review and pin
   every combined-bundle hash; do not reuse the base loader as MTP evidence.
3. Exercise real-weight MTP draft/verification state handling, padding,
   context boundaries, native Q8/Q4 quality and packed kernels, the 32K stress
   gate, and standardized memory/latency/throughput reporting.
