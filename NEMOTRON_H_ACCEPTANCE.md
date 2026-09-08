# Nemotron-H 3.5 Lightning acceptance protocol

This document defines the artifact and numerical gates for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at immutable revision
`a9904d24bcc1d289a1950fa9d2b978c47cf903b9`. The exact local artifact has now
passed full-payload authentication, strict source-ordered F16 application, and
a short Burn Metal base-model smoke. That smoke is internal-consistency
evidence, not comparison with an official numerical oracle; real-weight MTP
parity and the remaining release gates are still incomplete, so the model is not
advertised as supported. The machine-readable state is
[`acceptance/nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json`](acceptance/nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json).

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

The committed synthetic fixture probes F16 RMSNorm rounding order, grouped
gated RMSNorm, correction-biased router selection with unbiased route weights,
and the MTP topology contract. A real reference must additionally provide six
little-endian F32 payloads: prompt logits, target hidden trace, 32 cached logits,
32 teacher-forced logits, MTP draft logits, and MTP hidden trace. Its strict
reader rejects unknown fields, non-regular files, symlinks, nested or traversing
paths, wrong shapes, non-finite declarations, changed sources, and extra bundle
members. All seven independent trust anchors (record plus six payloads) are
intentionally absent, so every real bundle fails closed until review pins them.

## Reproduction commands

The metadata, mapping, accounting-defect, MTP-inventory, and fail-closed oracle
tests require no weight payloads:

```sh
cargo test --manifest-path crates/transformers/Cargo.toml \
  --no-default-features --features std \
  --test nemotron_h_artifacts
```

After an external pinned download, authenticate all seven controls and every
byte of all fourteen weight shards without allocating tensors:

```sh
TRANSFORMERS_NEMOTRON_H_DIR=/absolute/path/to/pinned-checkpoint \
cargo test --release --manifest-path crates/transformers/Cargo.toml \
  --no-default-features --features std,tokenizers \
  --test nemotron_h_artifacts \
  validates_pinned_nemotron_control_and_weight_payloads \
  -- --ignored --exact --nocapture
```

Strictly apply the complete F16 graph and run the bounded base Metal smoke with:

```sh
TRANSFORMERS_NEMOTRON_H_DIR=/absolute/path/to/pinned-checkpoint \
cargo test --release --manifest-path crates/transformers/Cargo.toml \
  --no-default-features --features std,metal,tokenizers \
  --test nemotron_h_artifacts \
  loads_pinned_nemotron_f16_and_smokes_base_inference \
  -- --ignored --exact --nocapture --test-threads=1
```

After a dual-runtime reference bundle has been independently reviewed and its
seven hashes have been pinned in the reader, authenticate it with:

```sh
TRANSFORMERS_NEMOTRON_H_REFERENCE_DIR=/absolute/path/to/nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle \
cargo test --release --manifest-path crates/transformers/Cargo.toml \
  --no-default-features --features std \
  --test nemotron_h_artifacts \
  validates_pinned_nemotron_official_reference_bundle \
  -- --ignored --exact --nocapture
```

Running the last command before hash review is expected to fail; the bundle may
not authorize itself through hashes stored inside its JSON.

## Remaining release gates

Before this checkpoint can be advertised as supported:

1. Generate the target evidence with pinned Transformers and the MTP evidence
   with pinned vLLM; independently review and pin every bundle hash.
2. Match full-vocabulary prompt logits and hidden traces, then match 32-token
   cached generation and stateless top-1 at every prefix.
3. Exercise official-parity recurrent decode, MTP draft/verification state
   handling, padding, context boundaries, native Q8/Q4 quality and packed
   kernels, the 32K stress gate, and standardized
   memory/latency/throughput reporting.
