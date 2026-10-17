# Model status and compatibility

This page records, for every model family, what is implemented, what has been
validated against a reference implementation, and what is advertised as
supported. These are different claims and are kept apart on purpose:

| Claim | What it means | Recorded in |
| --- | --- | --- |
| Configuration compatible | The crate deserializes the official text topology, including wrapper and legacy aliases, and rejects contradictory tensor shapes before allocating a model. | [Configuration compatibility](#configuration-compatibility) |
| Implemented | Stateless and cached inference run and pass synthetic tests, and a strict loader owns an explicit tensor contract that rejects unexplained keys. | Family module documentation and tests |
| Experimentally validated | A pinned checkpoint, or a tiny checkpoint generated with the reference library, matched the reference implementation within a stated tolerance on a named backend and dtype. | [Real-checkpoint evidence](#real-checkpoint-evidence), [tiny goldens](#tiny-reference-goldens) |
| Advertised supported | A committed `acceptance/` record sets `advertised_supported: true` for an exact revision, backend, and dtype. | [SUPPORT.md](SUPPORT.md) |

Loading a checkpoint successfully establishes neither of the last two claims.
Version 0.1 advertises only the bounded Qwen3.5-0.8B Metal/F16 base-text and
FLAN-T5-small Flex/F32 / Metal/F32 profiles named in
[the release record](acceptance/measurements/v0_1-release-20260910.json).
Everything else on this page is experimental.

## Status at a glance

| Family | `model_type` | Implemented | Experimentally validated | Advertised supported |
| --- | --- | --- | --- | --- |
| Qwen3.5 / Qwen3.8 hybrid | `qwen3_5_text` | Gated DeltaNet and gated full attention, partial MRoPE, external recurrent/KV caches, batched cached decoding, batch-one MTP speculation, hooks, staged execution | Qwen3.8-27B official full-vocabulary logits and 32 greedy tokens on Metal/F16; Q8 and Q4 execution smokes | Qwen3.5-0.8B, Metal/F16, base text (v0.1.0 record; passed again at `f4c56c4`, [replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)) |
| Qwen3-MoE | `qwen3_moe` | GQA with Q/K norm, sparse top-k experts, sliding windows, batched cached decoding, router hooks | Qwen3-30B-A3B strict load and Metal smokes (internal consistency only); Q8/Q4 conversions byte-identical and packed smokes and audits repeated at `c2936fe` ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)) | — |
| DeepSeek-V2 | `deepseek_v2` | MLA with absorbed latent decoding, YaRN, softmax or sigmoid (`noaux_tc`) routing, group-limited selection, shared experts | DeepSeek-V2-Lite strict load and internal Metal smoke (reproduced at `f4c56c4`); tiny Transformers goldens for V2 and V3-style routing | — |
| Muse Glimmer | `muse_glimmer_text` | Sliding/full hybrid, RoPE/NoPE schedule, sandwich norms, output gates, final soft cap, score-bounded attention | Muse-Glimmer-30B official prompt, cached, and hidden-trace references on Metal/F16 (minimum cosine `0.999943377`, 32/32 greedy); Q8/Q4 conversions byte-identical and packed Metal smokes at `c2936fe` ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)) | — |
| Nemotron-H | `nemotron_h` | Interleaved Mamba-2, GQA, dense and MoE blocks, one-stage MTP speculation | Nemotron 3.5 Lightning strict load and internal Metal smoke | — |
| T5 / FLAN / UL2 | `t5` (`AutoSeq2SeqLm`) | Encoder–decoder, relative-position bias, cross-attention cache, batched decoding, cache-aware beam search | FLAN-T5-XXL official references on Metal/F32 | FLAN-T5-small, Flex/F32 and Metal/F32 (v0.1.0 record; both passed again at `f4c56c4`, [replay record](acceptance/measurements/real-checkpoint-replay-20261016.json), and at `c2936fe`, [Mac validation record](acceptance/measurements/mac-validation-20261017.json)) |
| Llama family | `llama`, `mistral`, `qwen2`, `qwen3`, `smollm3`, `olmo2` | Dense GQA decoder with every Hugging Face RoPE rule, QKV bias, per-head or full-width Q/K norm, post-sublayer norm, NoPE layers, sliding windows | Tiny goldens for all six types; Flex and Metal F32 logits and 32/32 greedy parity for SmolLM-135M, Qwen3-4B, SmolLM3-3B, Llama-3.1-8B, and Qwen2-7B (against an F64 reference; on Metal under the documented `2.5e-3` tolerance), repeated at `53f167b`; half precision there: all Metal/F16 gates pass, Flex/BF16 passes 10 of 15 ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)) | — |
| Gemma 2 / Gemma 3 text | `gemma2`, `gemma3_text`, and the text tower of `gemma3` | Sandwich `(1 + w)` RMSNorm, `sqrt(d)` embedding scale, soft caps, local/global schedules with ring caches, per-layer-type RoPE | Tiny goldens for both; gemma-3-4b-it on Flex and Metal F32, including 1,237-token prompts across the 1,024-token window (repeated at `c2936fe`); Flex/BF16 passes 4 of 5 gates there, Metal/F16 fails because the model overflows F16 ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)) | — |
| gpt-oss | `gpt_oss` | Attention sinks, alternating sliding/full attention, YaRN, biased top-k-then-softmax router, clamped SwiGLU experts, exact MXFP4 decoding | Tiny golden; gpt-oss-20b truncated to one and four blocks on Flex and Metal F32 (within `1.3e-4`; repeated at `c2936fe`, [Mac validation record](acceptance/measurements/mac-validation-20261017.json)) | — |
| GPT-2, GPT-NeoX / Pythia | not registered | Stateless forward, hooks, staged execution, strict local import | Published-weight interpretability studies ([Greater-Than](examples/mechanistic/greater_than/README.md), [Tuned Lens](examples/mechanistic/tuned_lens/README.md)) | — |
| Custom modular models | serializable specs | Every shared mechanism above, composed from `DecoderSpec` or `Seq2SeqSpec` | Synthetic parity against several family graphs | — (experiments, not checkpoints) |

`AutoCausalLm` dispatches every causal `model_type` in the table and implements
the `HookedModel` and `StagedModel` experiment traits for all of them;
`AutoSeq2SeqLm` dispatches `t5` and implements `HookedModel`.

## v0.1 release qualification

The recorded [v0.1 source-gate and public-example replay](acceptance/measurements/v0_1-source-gates-20260910.json)
passes both line-coverage thresholds at that snapshot: 90.005% crate-wide and
95.011% shared/core. This excludes 79 verified test-only files and was measured
at `b4d6410` with 1,184 passing tests. Manual review binds every retained native
region to authored production function items for that exact
source/configuration; it adds no generated-code exclusions. This is not a
general macro-closure or branch-coverage proof. All 17 source/MSRV/backend/
package checks and both unchanged public examples also pass at that snapshot.
The [refreshed timing and sampled-memory records](acceptance/measurements/small-model-performance-20260910/manifest.json)
are available, including FLAN's first Flex CPU baseline. The
[release decision](acceptance/measurements/v0_1-release-20260910.json) records
the user's explicit approval of the controlled Qwen baseline. Its historical
slowdown from roughly 36 to 27 tokens/s remains unexplained and documented;
it has not been fixed or erased. Future comparisons retain the 5% alert and
confirmed 10% failure thresholds. See [the release gates](RELEASE_GATES.md).

Both public small-model examples, `flan_t5_small` and `qwen35_chat`, passed
their unchanged entry points on Rust 1.92 with local checkpoints, including a
[final post-tokenizer-fix replay](acceptance/measurements/v0_1-source-gates-20260910.json)
with exact previous outputs and before/after source, checkpoint, and binary
hashes. The [earlier execution record](acceptance/measurements/public-examples-20260909.json)
is retained separately. These are functional runs, not throughput
measurements.

These records describe the source snapshots they name. Later work (see
[CHANGELOG.md](CHANGELOG.md)) changed both profiles' execution paths: batched
cached decoding, grouped-query attention on native KV heads, DeltaNet chunked
prefill, and T5's single block graph. FLAN-T5-small's Flex/F32 official-reference
gate was re-run during that work with byte-identical vector digests. At source
`f4c56c4` both profiles' gates passed again, as recorded in the
[real-checkpoint replay](acceptance/measurements/real-checkpoint-replay-20261016.json): FLAN-T5-small on Flex/F32 and
Metal/F32 with zero `1e-4` tolerance failures, and Qwen3.5-0.8B on Metal/F16
with prompt cosine `0.999998523`, cached decisions at least `0.999997671`, and
teacher-forced rows at least `0.999997517`. The Qwen3.5-0.8B public example,
replayed at `8a4404a`, printed the same 32-token text as its September record
([large-checkpoint replay](acceptance/measurements/large-checkpoint-replay-20261016.json)). These replays are evidence, not a new support declaration;
[the release gates](RELEASE_GATES.md#replays-required-after-the-polish-cycle)
list what each record needs and the replays still outstanding.

## Real-checkpoint evidence

The status below is evidence-based and intentionally narrower than the
[configuration matrix](#configuration-compatibility). “Strict loader
implemented” means that the production local loader, family tensor mapping,
and synthetic failure tests exist. “Real weights applied” means the named
pinned artifact was actually streamed into a model and executed. Rows for the
Llama, Gemma, and gpt-oss families cite their ignored test gates rather than
an `acceptance/` record: those runs compare against references generated
locally by [`scripts/generate_causal_lm_reference.py`](scripts/generate_causal_lm_reference.py)
(Transformers 5.2.0, PyTorch 2.10.0, CPU, eager attention). Their Flex and
Metal F32 runs at source `f4c56c4` are recorded in the
[real-checkpoint replay](acceptance/measurements/real-checkpoint-replay-20261016.json) as experimental validation; none
promotes a checkpoint to advertised support.

| Target | Executable transformer/runtime surface | Checkpoint evidence | Current acceptance state |
|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` | Qwen3.5/Qwen3.8 hybrid graph with DeltaNet and full-attention layers, recurrent/KV decode state, gated GQA, partial MRoPE, SwiGLU, and optional batch-one, one-layer MTP speculation | Strict local multimodal mapping and real weight application passed. The checked artifact has 488 physical tensors; the base graph consumes 320 tensors, produces one tied alias, and explicitly reports 153 visual plus 15 omitted MTP tensors. | [Real-weight record](acceptance/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17.json): Metal F16 logit/reference and 32-token greedy parity, cached/stateless parity, and a Q4 packed smoke run passed. The parity gate passed again at `f4c56c4` (prompt cosine `0.999998523`; [replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)). The bounded Metal/F16 base-text profile is supported for v0.1 with the documented historical timing limitation. Populated Q4 32K execution stress also passes, but Q4/Q8 quality, real-weight MTP and long-context numerical quality remain experimental. |
| `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Same executable hybrid and MTP runtime as the bring-up model, with its exact 64-layer schedule | The pinned 18-shard artifact's exact configuration and all 1,199 physical headers match the committed inventory. Strict loading applied all 866 text-plus-MTP tensors and allowlisted exactly 333 visual tensors; the tokenizer and no-thinking chat template also passed. | [Real-weight record](acceptance/qwen3.8-27b-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.json): Metal F16 matched the official full-vocabulary prompt logits at `0.999999881` cosine, the exact top-1, all 32 cached greedy tokens, and stateless top-1 at every generated prefix. MTP and recurrent decode smokes plus strict mixed Q8/F16 and Q4/F16 smokes also passed. At `8a4404a` ([large-checkpoint replay](acceptance/measurements/large-checkpoint-replay-20261016.json)) the official gate passed again (prompt cosine `0.999999873`, exact top-1, 32/32 greedy tokens), as did the text-plus-MTP smoke and fresh create-only Q8 and Q4 conversions that installed exactly and ran on Metal. That smoke's higher peak footprint (68.2 GB against 66.5 GB for the base source in alternating rounds) was bisected to source-ordered installation of the vocabulary matrices; with outsized tensors installed first at `c15f24e` it peaks at 60.6 GB with bitwise-identical logits, and the official gate passes again ([reference-precision follow-ups](acceptance/measurements/reference-precision-followups-20261016.json)). Quantized reference/quality, standardized throughput and output hashing, and 32K stress remain open; not advertised. |
| `Qwen/Qwen3-30B-A3B` at `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | Executable GQA decoder with Q/K normalization, top-8 routing, sparse selected-expert execution, and cached decode | The pinned 16-shard artifact's exact configuration and all 18,867 BF16 tensor headers match the committed inventory. Strict source-ordered loading applied every tensor as F16 with 18,480 explicit renames, opened each shard exactly once, and retained the tokenizer. Provenance-bound Q8 and Q4 conversion packed 18,625 matrices and retained 242 tensors as F16. | [Real-weight record](acceptance/qwen3-30b-a3b-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39.json): six-token F16, mixed Q8/F16, and mixed Q4/F16 Metal stateless, cached-prefill, 50-hidden-trace, and recurrent-decode smokes passed. Both packed decode audits launched all 1,345 expected matmuls without whole-weight dequantization. The F16, Q8 and Q4 smokes and both audits passed again at `8a4404a` from the existing v2 sidecars ([large-checkpoint replay](acceptance/measurements/large-checkpoint-replay-20261016.json); top-1 `1096`, cached and stateless bitwise equal). At `c2936fe` ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)) fresh Q8 and Q4 conversions reproduced both authenticated Burnpacks byte for byte, and the Q8 and Q4 smokes and audits passed again with logit, trace, and decode digests identical to `8a4404a`. Official reference logits, 32-token generation, quantized quality, standardized performance, and 32K stress remain open; not advertised. |
| `deepseek-ai/DeepSeek-V2-Lite` at `604d5664dddd88a0433dbae533b7fe9472482de0` | Executable YaRN/MLA decoder with reference unabsorbed prefill, compressed latent-cache absorbed decode, routed top-k experts, and shared experts | The pinned four-shard artifact's exact configuration and all 5,291 BF16 tensor headers match the committed inventory. Strict source-ordered loading applied every tensor as F16 with 5,099 explicit renames, opened each shard exactly once, and retained the tokenizer. | [Real-weight record](acceptance/deepseek-v2-lite-604d5664dddd88a0433dbae533b7fe9472482de0.json): a six-token Metal F16 stateless, reference-unabsorbed cached-prefill, 29-hidden-trace, compressed-cache, and absorbed-decode smoke passed. Cached/stateless cosine was `1.0`; absorbed decode matched the internal stateless path at `0.999998964`; the metadata gate and this smoke passed again at `f4c56c4` with the same cosines ([replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)). No official-reference numerical comparison has run. Reference logits, 32-token generation, quantization, and 32K stress remain open; not advertised. |
| `meta-models/Muse-Glimmer-30B` at `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | Executable text decoder with sliding/full attention, RoPE/NoPE scheduling, sandwich norms, attention output gates, final-logit softcap, and score-bounded query chunking for exact eager-F16 inference | The exact two-shard artifact's control files, payload hashes, 1,436 BF16 headers, 627-tensor text mapping, 809-key visual allowlist, F16 load plan, and tokenizer round trip passed. Strict Metal loading applied all 627 language tensors as F16, including 626 explicit renames and BF16-to-F16 conversions, with zero missing, unexpected, mismatched, or error diagnostics. Provenance-bound Q8/G64 and Q4/G64 conversions each packed 417 matrices, retained 210 tensors as F16, and fully reopened their exact v2 sidecars. | [Real-weight record](acceptance/muse-glimmer-30b-a4e59da52a7bc87ae7251dd5545c0dd437c44b68.json): the authenticated seven-file schema-3 oracle covers full-vocabulary prompt logits, every cached and teacher-forced decision row, and stateless, cached-prefill, and first recurrent-decode hidden traces. Burn's minimum official-reference cosine was `0.999943377`, and all 32 official greedy tokens matched. Both real sidecars passed stateless, cached-prefill, and recurrent Metal execution; each trace observed all 417 packed decode matmuls and no whole-weight dequantization. At `4f9502a` ([large-checkpoint replay](acceptance/measurements/large-checkpoint-replay-20261016.json)) the official gate reproduced every recorded cosine (minimum `0.999943377`, 32/32 greedy tokens), and the streaming audit and both packed smokes and audits passed again. At `c15f24e`, which installs its two vocabulary matrices first (the streaming audit's peak falls from 68.9 to 66.2 GB in alternating rounds), the official gate reproduced every recorded cosine again and the streaming audit passed ([reference-precision follow-ups](acceptance/measurements/reference-precision-followups-20261016.json)). At `c2936fe` ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)) fresh Q8 and Q4 conversions reproduced both authenticated Burnpacks byte for byte under the conversion test's new three-open shard order, and both packed Metal smokes passed (top-1 `85068` and `90`, cached cosine `1.0`); in alternating rounds against `c15f24e`'s parent the Q8 smoke peaked about 1.6 GB higher in two of three head rounds. A synthetic Metal gate exercised the 32K attention/cache/position envelope with official head counts but reduced head width and no dense blocks; full-model 32K acceptance remains open. Quantized reference/quality, real-checkpoint padded/context-boundary coverage, and standardized performance also remain open; not advertised. |
| `google/flan-t5-small` at `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` | Executable T5 encoder-decoder with bidirectional encoding, causal decoding, cross-attention caches, relative-position bias, shared encoder/decoder embeddings, an independent LM head, and cached beam search | The official one-file SafeTensors artifact was content-authenticated, strictly loaded, and all 190 F32 tensors were applied through the public automatic loader on Flex and Metal. | [Real-weight record](acceptance/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json): complete encoder, decoder-hidden, first-logit, cached-32, teacher-32, padded-batch, and 512-token boundary vectors match an authenticated Transformers 5.2 CPU/F32 oracle on both accepted backends at minimum cosine `0.999999999999`, with zero `1e-4` tolerance failures. All 32 greedy decisions and exact four-beam translation tokens match. Both gates passed again at `f4c56c4` ([replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)) and at `c2936fe` with all seven output digests unchanged ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)). The bounded Flex/F32 and Metal/F32 profiles are supported for v0.1; other revisions, dtypes and quantized quality remain experimental. |
| `google/flan-t5-xxl` at `ae7c9136adc7555eeccc78cdd960dfd60fb346ce` | Executable 24-layer encoder/24-layer decoder T5 graph with gated GELU, relative-position bias, cross-attention caches, greedy decode, and beam search | All five official F32 SafeTensors shards were content-verified. Strict loading applied all 558 runtime parameters, verified the physical embedding equivalence, performed 433 checkpoint-layout transposes, and retained the tokenizer. | [Real-weight record](acceptance/flan-t5-xxl-ae7c9136adc7555eeccc78cdd960dfd60fb346ce.json): Metal matched five official Transformers/MPS F32 tensor fixtures at minimum cosine `0.999999999998`, all 32 cached greedy tokens, all teacher-forced top-1 rows, and the exact four-beam continuation. At `8a4404a` the gate passed again with all seven output digests bitwise identical ([large-checkpoint replay](acceptance/measurements/large-checkpoint-replay-20261016.json)). Padded/context-boundary, quantized quality/32K, and standardized performance gates remain open; not advertised. |
| `google/flan-ul2` at `452d74ce28ac4a7f211d6ba3ef0717027f7a8074` | The executable T5 runtime represents UL2's gated-SiLU topology | Strict T5-family mapping exists. The pinned target still requires the planned verified offline PyTorch-to-SafeTensors conversion at this revision. | No real-weight acceptance record; not advertised. |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` | Executable heterogeneous Mamba2/GQA/dense-MLP/MoE graph with ReLU-squared experts, shared expert, recurrent/KV state, and a one-stage next-token head with transactional batch-one speculation | The exact fourteen-shard artifact was authenticated and all 6,513 tensors were strictly streamed into Metal F16, including 270 MTP tensors. The exact MTP body, independent KV bank, partial target/MTP commits, rollback, greedy verification, and lossless stochastic rejection pass tiny Flex tests. The Nano revision `bf77c3174f68ad409e1c2aa60daeb46e32d1c606` is configuration-compatible only. | [Real-weight record](acceptance/nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json): stateless, cached-prefill, and recurrent base-model smokes passed; recurrent versus teacher-forced logits had cosine `0.999376514` and identical top-1; the smoke passed again at `4f9502a` (`0.999370862`). The official corresponding-path gate first failed two teacher-forced rows against a `0.999` floor on the F16 oracle, which the oracle itself misses by the same amount (its teacher row 18 is `0.994466` from the official F32 rows). Judged against the official class's F32 rows, with each path's floor set by the official F16 evaluations' own distance from F32, it passes at `9a9c730` ([reference-precision follow-ups](acceptance/measurements/reference-precision-followups-20261016.json)): prompt `0.999224`, teacher rows at least `0.997612`, cached rows at least `0.998777`, 32/32 top-1 on both paths ([analysis](NEMOTRON_H_ACCEPTANCE.md#first-official-comparison)). Real-weight MTP parity, 32-token generation, quantization, and 32K remain open; not advertised. |
| `HuggingFaceTB/SmolLM-135M` at `1d461723eec654e65efdc40cf49301c89c0c92f4` (`llama`) | Llama-family dense decoder: GQA, RoPE, SwiGLU, tied head, batched cached decoding | Strict automatic loading; BF16 weights cast exactly to F32 while streaming. | [Flex/F32 gate](tests/llama_family_artifacts.rs) against an F32 reference: every prompt position's full-vocabulary logits within `9.1e-5`, a left-padded batched prefill within `3.3e-5`, all 32 cached greedy tokens identical (source `65645e0`). At `f4c56c4` ([replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)) the gate passed on Flex and Metal F32 within `1.34e-4` at every prompt position, with 32/32 greedy tokens, and again with identical lines at `53f167b`. Half precision at `53f167b` ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)): Metal/F16 passes all three gates; Flex/BF16 passes batched prefill but fails prompt logits and cached greedy (prompt 1, position 0: `1.436` against the official-calibrated `1.322`), which the record traces to the twice-rounded half-precision SiLU and the bound rule's width. Not advertised. |
| `Qwen/Qwen3-4B` at `531c80e289d6cff3a7cd8c0db8110231d23a6f7a` (`qwen3`) | Llama-family graph with per-head Q/K RMSNorm and explicit `head_dim` | Strict automatic loading; BF16 cast to F32. | [Flex/F32 gate](tests/llama_family_artifacts.rs): prompt logits within `1.41e-4`, batched prefill within `1.35e-4`, 32/32 greedy tokens (source `65645e0`). At `f4c56c4` ([replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)) the gate passed on Flex and Metal F32 within `1.41e-4`, with 32/32 greedy tokens. At `53f167b` the F32 gates passed again with identical lines; Metal/F16 passes all three half-precision gates, and Flex/BF16 passes prompt logits and batched prefill but its cached decode step 1 deviates `0.5625` from the stateless pass against the `0.5` tolerance ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)). Not advertised. |
| `HuggingFaceTB/SmolLM3-3B` at `1fa6fa281b5c635c89171a2b53769609cc1199c6` (`smollm3`) | Llama-family graph with periodic NoPE layers | Strict automatic loading; BF16 cast to F32. | [Flex/F32 gate](tests/llama_family_artifacts.rs): prompt logits within `2.2e-4`, `1.5e-4`, and `3.8e-4`, batched prefill within `1.3e-4`, 32/32 greedy tokens (source `dd92f3b`). At `f4c56c4` ([replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)) the gate passed on Flex and Metal F32 within `3.83e-4`, with 32/32 greedy tokens. At `53f167b` the F32 gates passed again with identical lines; Metal/F16 passes all three half-precision gates, and Flex/BF16 passes batched prefill but fails prompt logits (prompt 1 mean `0.0971` against `0.0947`) and cached greedy (decode step 13, `0.516` against the `0.5` tolerance) ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)). Not advertised. |
| `Qwen/Qwen2-7B` at `453ed1575b739b5b03ce3758b23befdb0967f40e` (`qwen2`) | Llama-family graph with bias on the Q/K/V projections only | Strict automatic loading; BF16 cast to F32. | [Flex/F32 gate](tests/llama_family_artifacts.rs) against an F32 reference: batched prefill within `7.9e-5` and 32/32 greedy tokens, but one 8-token prompt deviated by `1.80e-3`, above the `1e-3` default (`8720be1`, `dd92f3b`). That deviation is mostly the reference's own rounding: the first position's hidden states reach magnitudes near 7,000, where one F32 unit in the last place is about `5e-4`, and against an F64 bundle (itself checked against an independent layer-streamed F64 forward within `2.1e-5`) F32 Transformers deviates by `1.6e-3` there. At `f4c56c4` ([replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)), against that F64 bundle, Flex stays within `2.9e-4` (`1.8e-4` at that position) and Metal reaches `1.9e-3` at one first-position logit, comparable to (slightly above) F32 Transformers' own `1.6e-3` there, so the Metal run uses `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE=2.5e-3`. Every other position stays below `3e-4`, and both backends reproduce all 32 greedy tokens. At `53f167b` the F32 gates passed again with identical lines and about 2 GB lower peaks, and all six half-precision gates pass; Metal/F16's prompt-1 mean sits at 96% of its bound, and Transformers' own F16 evaluation of this checkpoint is non-finite ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)). Not advertised. |
| `meta-llama/Llama-3.1-8B` at `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` (`llama`) | Llama-family graph with Llama 3 RoPE scaling (factor 8, frequency factors 1 and 4, original context 8,192) | Strict automatic loading; BF16 cast to F32. | [Flex and Metal F32 gate](tests/llama_family_artifacts.rs) against an F32 reference at `f4c56c4` ([replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)): every prompt position within `7.6e-5`, 32/32 greedy tokens. This is the only real-checkpoint run of Llama 3 RoPE scaling. At `53f167b` the F32 gates passed again with identical lines and about 2 GB lower peaks, and all six half-precision gates pass, Flex/BF16 matching all 27 decisive greedy steps ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)). Not advertised. |
| `google/gemma-3-4b-it` at `093f9f388b31de276ce2de164bdc2081324b9767` (`gemma3` text tower) | Gemma 3 text decoder: sandwich `(1 + w)` RMSNorm, `sqrt(d)` embedding scale, 5:1 local/global attention with a 1,024-token window and ring caches, per-layer-type RoPE | Strict automatic loading of the language model; exactly the 439 vision-tower tensors are reported as intentionally ignored. | [Flex/F32 gate](tests/gemma_artifacts.rs) against `Gemma3ForConditionalGeneration` on text inputs: short-prompt logits within `7.2e-5`, `1.3e-4`, and `1.6e-4`, batched prefill within `1.1e-4`, 32/32 greedy tokens. One 1,237-token prompt crossing the window: within `5.5e-4` before and `1.2e-3` at or beyond the window (tolerance `5e-3`; Transformers' own eager and SDPA graphs differ by `3.2e-3` there), disabling the window moves logits by up to 36, and a cached run whose 29 sliding layers wrap their rings stays within `1.5e-4` with 8/8 greedy tokens (source `c5fd877`; gate code identical at `dd92f3b`). At `f4c56c4` ([replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)) the gate passed on Flex and Metal F32, including the 1,237-token prompts (largest deviation beyond the window `1.18e-3`, 8/8 long greedy tokens), and again with identical lines at `53f167b` and `c2936fe`. Half precision ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)): at `c2936fe`, which evaluates half-precision tanh GELU in F32, Flex/BF16 passes prompt logits (`0.346`, `0.514`, `0.582` against `0.460`, `0.711`, `0.897`), batched prefill, and both long-context gates, and fails cached greedy (decode step 10, `0.75` against the `0.5` tolerance); Metal/F16 fails every gate on non-finite logits because the model overflows F16, as Transformers' own F16 evaluation does. Longer contexts are not gated; not advertised. |
| `openai/gpt-oss-20b` at `6cee5e81ee83917806bbde320786a8fb61efebee` (`gpt_oss`) | gpt-oss sparse decoder: learned sinks, alternating 128-token sliding and full attention, YaRN, biased top-k-then-softmax router, clamped SwiGLU experts | MXFP4 expert tensors are decoded at load, bit for bit equal to Transformers' `convert_moe_packed_tensors` on the checkpoint's own experts; the gate loads truncated stacks strictly, allowlisting the unread blocks. | [Flex/F32 gate](tests/gpt_oss_artifacts.rs) against truncated F32 references: one block within `7.6e-5` (mean `3.4e-6`); four blocks within `1.3e-4` (mean below `5e-6`), a 183-token prompt decoded token by token through a wrapping 128-slot ring within `7.6e-5`, and 32/32 greedy tokens (source `743a578`). Half-precision runs deviate from the F32 reference by 1.4 to 1.9 logits at some position, as Transformers' own BF16 run does (`1.73`). At `f4c56c4` ([replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)) the one- and four-block F32 gates passed on Flex and Metal (within `1.3e-4`, 32/32 greedy tokens, batched and long cached decoding). They passed again at `c2936fe` with every line identical and peak footprints about 2.0 GB lower on Flex and 4.9 GB (one block) or 8.8 GB (four blocks) lower on Metal ([Mac validation record](acceptance/measurements/mac-validation-20261017.json)). At `8a4404a` ([large-checkpoint replay](acceptance/measurements/large-checkpoint-replay-20261016.json)) a full 24-block BF16 reference was generated: Flex/BF16 passes every gate (within `1.62`, mean at most `0.064`), and Metal/F16's `2.66` at one prompt measured that BF16 bundle's own rounding: against a full 24-block F32 reference generated block by block from BF16 ([reference-precision follow-ups](acceptance/measurements/reference-precision-followups-20261016.json)), Metal/F16 stays within `1.15` (mean at most `0.025`) and Flex/BF16 within `2.19` (mean at most `0.112`), while Transformers' own BF16 run is `2.665` away (mean at most `0.108`, recorded from the two bundles). All six gates pass under a half-precision maximum of `2.7`, that official deviation rounded up, and the earlier mean bound of `0.15`. Metal/F16 against the BF16 bundle (`2.66`) compares two half-precision roundings and is recorded as informational, not as a gate pass. Not advertised. |

### Tiny reference goldens

Each family added since v0.1 also has committed tiny random-weight checkpoints
under `tests/fixtures/tiny_goldens/`, generated with Transformers 5.2.0 and
PyTorch 2.10.0 by [`scripts/generate_tiny_goldens.py`](scripts/generate_tiny_goldens.py).
They check full-vocabulary logits at every valid position of an unpadded and a
left-padded row, every hidden state, stateless-versus-cached agreement with
ragged rows, and greedy continuations at an F32 tolerance of `1e-5`:

| Test | Cases | Largest observed logit gap |
| --- | --- | --- |
| [`tiny_goldens_llama`](tests/tiny_goldens_llama.rs) | `llama3`, `llama-bias-tied`, `qwen2`, `qwen3`, `mistral-sliding`, `smollm3`, `olmo2` | `7.2e-7` |
| [`tiny_goldens_gemma`](tests/tiny_goldens_gemma.rs) | `gemma2` (soft caps, alternating windows), `gemma3-text` (Q/K norm, per-type RoPE), plus a saturated attention soft cap | `1.5e-6` |
| [`tiny_goldens_gpt_oss`](tests/tiny_goldens_gpt_oss.rs) | `gpt-oss` (sinks, sliding/full, YaRN, router, clamped SwiGLU), plus the load-balancing loss | about `1e-6` |

Shared components carry their own reference goldens in their unit tests: the
router against Transformers' Mixtral, Qwen3-MoE, DeepSeek-V2/V3, and gpt-oss
routing; every RoPE scaling rule against `modeling_rope_utils.py`; learned
sinks against gpt-oss eager attention; and differential attention against
DiffLlama. These establish mechanism parity, not checkpoint support.

## Quantized checkpoints

Native Q8/Q4 Burnpack sidecars and automatic packed loading are described in
the [README](README.md#quantization). Automatic packed loading of
`qwen3_5_text`, `qwen3_moe`, and `muse_glimmer_text` installs every packed
projection exactly: as packed QFloat on Metal, and as the exact dequantized
F32 values on Burn Flex 0.21, whose quantized transpose would re-block them.
Their real-checkpoint evidence:

- Qwen3.5-0.8B Q4 has passed a finite full-model Metal smoke test and exact
  cached/stateless consistency, but its first token differs from F16 and the
  quality corpus has not passed. Its
  [populated Q4 32K stress](acceptance/measurements/qwen3.5-0.8b-q4-32k-stress-20260909.json)
  passes finite-state, overflow-atomicity, and reset checks; this is execution
  evidence, not long-context quality.
- Qwen3.8-27B Q8 and Q4 have each passed strict streaming conversion and a
  clean single-backend Metal smoke: 497 matrices were packed in groups of 64,
  369 tensors remained outside each sidecar and loaded as F16, and both mixed
  graphs exercised stateless, cached, MTP, and recurrent paths. Quantized
  reference/longer-generation parity, the quality corpus, standardized
  throughput and output hashing, and 32K stress remain open.
- Qwen3-30B-A3B Q8 and Q4 have passed provenance-bound streaming conversion and
  mixed F16/packed Metal stateless, cached-prefill, and recurrent decode
  smokes. Each sidecar contains 18,625 packed matrices; 242 embeddings, norms,
  and routers remain F16, and runtime MoE dispatch evaluates only selected
  experts. The Q4 and Q8 decode audits each observed all 1,345 expected packed
  matmuls and no whole-weight dequantization. Official-reference and 32-token
  generation parity, quantized quality, standardized performance, and 32K
  stress remain open.
- Muse-Glimmer's exact 627-tensor F16 load plan classifies 417 matrices for
  Burnpack (all five attention projections, all three SwiGLU projections in
  each of 52 layers, and the independent LM head) and retains the embedding
  plus 209 normalization vectors. Every eligible input width aligns to G64. A
  nonzero two-layer fixture has passed Q8 and Q4 streaming conversion,
  automatic mixed loading, stateless/prefill/recurrent parity, complete dtype
  accounting, and a negative gate proving row-unaligned artifacts remain
  reference-only. The real Q8/G64 conversion revalidated all 417 payloads and
  published a `28,165,645,337`-byte v2 artifact with SHA-256
  `91cfd35bfe0a389cbc2eed2189c4e33312bdce1a0ff50d8452acf06d00f0e8eb`; the
  strict conversion/provenance gate took `1,570.845 s`. A separate traced
  Metal gate reauthenticated that sidecar and its source provenance, then ran
  stateless, cached-prefill, and recurrent decode with cached/stateless cosine
  `1.0`, launching all 417 packed Q8 matmuls with no whole-weight
  dequantization. The real Q4/G64 conversion published and fully reopened its
  `14,911,372,485`-byte v2 artifact with SHA-256
  `05b69f67799b4dc0228b4298e39d466d753fcc5d6484f61c8920c551017daed3`; its
  traced Metal gate likewise ran all 417 packed Q4 matmuls without
  whole-weight dequantization and reported cached/stateless cosine `1.0`.
  Quantized official-reference and quality evaluation plus full-model 32K
  execution remain pending.
- gpt-oss MXFP4 experts are decoded exactly to the load dtype; there is no
  packed MXFP4 execution and no memory saving yet.

Neither Q8 nor Q4 is advertised for production inference.

## Configuration compatibility

A checked configuration means the crate can deserialize the official text
topology (including wrapper and legacy aliases) and reject contradictory
tensor shapes before model allocation. It does not prove that a checkpoint
loads or computes the right logits.

| Family | Reference checkpoint/config | Serialized identity | Wrapper and aliases | Topology represented and validated |
|---|---|---|---|---|
| Qwen3.5 / Qwen3.8 dense hybrid | [Qwen3.8-27B `config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json), [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B/blob/feea018b31f89dc0950e61da42577a7a4ab09169/config.json) | Inner text config remains `qwen3_5_text` for Qwen3.8 | Direct or nested `text_config`; `rope_scaling` alias; `Qwen38TextConfig` Rust alias | Explicit or interval-derived GDN versus full-attention schedule, gated-Q projection, GQA, linear head/state widths, convolution, multimodal RoPE sectioning, and MTP depth/embedding policy |
| Qwen3 MoE | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json) | `qwen3_moe` | Direct or any supported text wrapper; `num_local_experts` and `rope_scaling` aliases | GQA/head dimensions, dense-versus-MoE cadence, dense overrides, expert width/count/top-k, router normalization, RoPE, and optional sliding window |
| DeepSeek-V2 / V2-Lite | [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json) | `deepseek_v2` | Direct or `text_config`, `language_config`, or `llm_config`; old/new RoPE key aliases; nullable Q LoRA rank | MLA Q/K content and rotary widths, independent V width, Q/KV latent ranks, dense prefix and MoE cadence, grouped top-k capacity, shared/routed experts, and YaRN constraints |
| Muse Glimmer text decoder | [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/config.json), [Transformers attention reference](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/muse_glimmer/modeling_muse_glimmer.py) | Inner `muse_glimmer_text` | Nested `text_config` or direct text config; `hidden_act` and `rope_scaling` aliases | GQA, explicit or backward-derived sliding/full schedule, per-layer RoPE/NoPE, sliding window, QK scaling, FP32 softmax for half inputs, post norms, and final-logit scaling/softcap |
| T5 / T5 v1.1 / FLAN-T5 | [FLAN-T5-XL](https://huggingface.co/google/flan-t5-xl/blob/8772db7a7a11f7b08e6be7d7088f7a7fd4813bc5/config.json), [pinned upstream T5 schema](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/t5/configuration_t5.py) | `t5` | Canonical names plus `hidden_size`, `head_dim`, `intermediate_size`, and standard head/depth aliases | Independent residual and attention-inner widths, asymmetric encoder/decoder depth, relative-position buckets/range, dense versus gated FFN layout, output-scaling flag, and decoder start/token IDs |
| UL2 | [Google UL2](https://huggingface.co/google/ul2/blob/71c34c5c6e3e54c5bf2cee360c3edfaa42acc1b3/config.json) | `t5` | `Ul2Config` Rust alias; accepts historic derived `dense_act_fn` and `is_gated_act` | T5 encoder-decoder topology plus gated-SiLU FFN consistency |
| Nemotron-H / Nemotron Nano | [Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/config.json), [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/config.json) | `nemotron_h` | Compact `hybrid_override_pattern` or modern `layers_block_type`/`layer_types`; historic `mamba_*` aliases; `attention_head_dim`; bare legacy `Infinity` and Transformers 5's `{"__float__": "Infinity"}`; `sliding_window` accepted and ignored, as in the pinned `cfa309` graph (experiments opt into a window with `NemotronHConfig::attention_window`) | Exact heterogeneous Mamba/attention/MLP/MoE sequence, NoPE attention with GQA, Mamba state/head/group/convolution/time-step topology, expert routing, redundant norm consistency, and optional MTP block sequence |
| Llama family | [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M/blob/1d461723eec654e65efdc40cf49301c89c0c92f4/config.json), [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/blob/531c80e289d6cff3a7cd8c0db8110231d23a6f7a/config.json), [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/1fa6fa281b5c635c89171a2b53769609cc1199c6/config.json), [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B/blob/453ed1575b739b5b03ce3758b23befdb0967f40e/config.json); Transformers 5.2.0 `configuration_*.py` | `llama`, `mistral`, `qwen2`, `qwen3`, `smollm3`, `olmo2` | One superset schema whose absent fields take the checkpoint's own `model_type` defaults; `rope_scaling` and `rope_parameters` through the shared RoPE translator | GQA and `head_dim`, attention/MLP bias or Q/K/V-only bias, per-head (Qwen3) or full-width (OLMo 2) Q/K RMSNorm, post-sublayer norms, every Hugging Face RoPE type, Mistral and Qwen sliding-window schedules (`sliding_window`, `use_sliding_window`, `max_window_layers`, `layer_types`), SmolLM3 NoPE layers, tied heads; `quantization_config` is rejected |
| Gemma 2 / Gemma 3 text | [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it/blob/093f9f388b31de276ce2de164bdc2081324b9767/config.json); Transformers 5.2.0 `configuration_gemma2.py` / `configuration_gemma3.py` | `gemma2`, `gemma3_text`; multimodal `gemma3` through `text_config` | Gemma 3 RoPE mappings exactly as Transformers 5.2 `save_pretrained` writes them, including inert `rope_type`/`rope_theta` keys beside per-layer-type objects | Sandwich norms, `query_pre_attn_scalar`, attention and final soft caps (rejected where the reference never applies them), `sliding_window_pattern` or `layer_types` schedules, separate local/global RoPE bases with scaling on global layers, per-head Q/K norm (Gemma 3) |
| gpt-oss | [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json); Transformers 5.2.0 `configuration_gpt_oss.py` | `gpt_oss` | Dense or MXFP4 (`quantization_config.quant_method = "mxfp4"`) expert weights | GQA with learned sinks and biased projections, `layer_types` sliding/full schedule, YaRN with `truncate` (legacy `rope_scaling` accepted), expert count and top-k, biased router; `swiglu_limit` and `hidden_act` must equal the values the reference hard-codes (`7.0`, `silu`) |

## Loader behavior

`ModelConfig::from_json_str` first parses strict JSON after one deliberately narrow
compatibility repair: an unquoted positive `Infinity` token is converted to `null`.
NVIDIA's original Nemotron-H configs used that Python JSON extension for an
unbounded time-step maximum. Configs written by Transformers 5 spell the same
value `{"__float__": "Infinity"}`, which is also read as `null`; a tagged
`-Infinity` or `NaN`, or an unknown tag, is rejected with its JSON path.
Strings containing the word `Infinity` are unchanged.

Layer and expert counts are bounded before anything is allocated:
`models::config::MAX_LAYERS` (2^18 blocks per stack), `MAX_EXPERTS` (2^20
routed experts), and, for per-expert checkpoint layouts, `MAX_CHECKPOINT_TENSORS`
(ten million tensors), so no accepted configuration describes a contract
beyond the artifact limit. Automatic loading also checks, before it builds a
family's tensor contract, that the discovered checkpoint holds at least as
many tensors as the configuration's depth (or exact one-to-one inventory)
needs, failing with `CheckpointError::TooFewTensors` unless an `allow_missing`
matcher could excuse them.

For a multimodal repository, only `text_config` is consumed. The compatibility
aliases `language_config` and `llm_config` support older/custom wrappers. Vision
metadata is ignored because this crate milestone covers transformer text backbones
only. Unknown non-topology metadata is also ignored so harmless Hub metadata does
not break loading; unknown enum values and contradictory shape fields fail closed.

`AutoCausalLm::load_local` dispatches `deepseek_v2`, `gemma2` (alias
`gemma3_text`), `gpt_oss`, `llama` (aliases `mistral`, `qwen2`, `qwen3`,
`smollm3`, `olmo2`), `muse_glimmer_text`, `nemotron_h`, `qwen3_5_text`, and
`qwen3_moe`. `AutoSeq2SeqLm::load_local` dispatches `t5`. Both APIs are
local-only: they discover SafeTensors, validate the architecture-owned mapping
before applying weights, attach an optional local tokenizer, and return the
final strict `LoadReport`. Planning failures are `Error::CheckpointPlan` with
the typed `CheckpointError`, and a failed tensor stream is
`Error::CheckpointExecution` with its partial report. Unknown checkpoint
tensors fail closed unless an architecture explicitly allowlists them, as
Qwen, Muse, and multimodal Gemma 3 do for their vision towers. For an alias family, `AutoCausalConfig::model_type`
returns the primary type (`llama`); the checkpoint's own type is available from
`LoadedModel::source().model_type()`.

Discovery prefers neither artifact silently: when a directory holds both
`model.safetensors` and `model.safetensors.index.json`, it loads through the
index only if the index names exactly that single file, and otherwise reports
ambiguous artifacts. An index in a subdirectory resolves its shards beside
itself. Header-entry errors name the shard and tensor and are reported before
any layout error; element counts follow SafeTensors' left-to-right overflow
rule, and out-of-section byte ranges are reported at the first violation in
byte-offset order.

`generation_config.json` is normalized with Hugging Face semantics: sampling
defaults for `top_k`/`top_p`, and, for causal loads, `min_p` (when sampling),
`no_repeat_ngram_size`, and `min_new_tokens` (clamped to `max_new_tokens`, with
the declared value reported). Settings the runtime does not apply are kept in
`LoadedModel::unsupported_generation_settings()` instead of being dropped
silently. Seq2seq loads keep `repetition_penalty`, `presence_penalty`, and
sampling neutral in their defaults and report the three processor settings,
`repetition_penalty`, and `do_sample` with its non-neutral `temperature`,
`top_k`, and `top_p`, because beam search does not apply them.

Chat templates render through a sandboxed MiniJinja environment with Hugging
Face `apply_chat_template` semantics (Python `str()` conversions in filters,
`tojson`, `strftime_now`, tool calls, `continue_final_message`, documents).
Template discovery (`chat_template.jinja` and `additional_chat_templates/*.jinja`
over `tokenizer_config.json`, skipping paths that are not regular files) and
selection (`tool_use` when tools are passed) follow Transformers 5.2.
On the pinned Qwen3.5, Muse Glimmer, and Nemotron-H templates the rendered
text matches Transformers 5.2.0 byte for byte when mapping keys are supplied in
sorted order; templates see mapping keys sorted, which is the main remaining
difference. The tokenization module documentation lists the others and the
deliberate differences for artifacts it treats as unsafe or malformed. This is
tokenizer/template parity evidence, not checkpoint support.

## Known gaps and non-claims

- A schema match alone still does not prove checkpoint compatibility. The strict
  family contract compares every observed name, shape, and dtype before weight
  application, and only a real-artifact run establishes evidence for one pinned
  revision.
- Qwen3.8 uses the Qwen3.5 architecture discriminator in the official artifact.
  No invented `qwen3_8_text` serialized value is accepted.
- Qwen3.8's complete text-plus-MTP weights now load and execute on Metal F16,
  while its mixed Q8/F16 and Q4/F16 graphs have each passed a strict smoke run.
  The F16 graph also matches the pinned official full-vocabulary logits and all
  32 stable-margin greedy tokens. Quantized reference/quality,
  standardized-throughput and output-hash, and 32K gates are still pending.
- Muse Glimmer vision topology is intentionally not represented. Only its nested
  text decoder is extracted.
- Nemotron-H's historical configs are not RFC-compliant JSON because they contain
  `Infinity`; only the published positive-infinity case is repaired, in both
  the bare and the Transformers 5 tagged spelling. `NaN` and negative infinity
  remain errors in both spellings.
- Qwen3 MoE's pinned 30B-A3B artifact has passed complete F16 weight application,
  provenance-bound Q8/Q4 conversion, and short Metal stateless/cached/decode
  smokes in both packed formats. Q4 and Q8 both passed their 1,345-launch
  packed-kernel audits without whole-weight dequantization. Official reference,
  generation, quantized-quality, standardized-performance, and 32K gates remain
  open.
- DeepSeek V2 Lite has passed complete F16 weight application and a short Metal
  stateless/reference-prefill/compressed-cache/absorbed-decode smoke. Its
  cached and decode comparisons are internal Burn-path parity, not official
  numerical-reference acceptance; generation, quantization, and 32K gates also
  remain open.
- Muse Glimmer has authenticated its exact real artifact and seven-file official
  oracle, strictly applied all 627 text tensors as Metal F16, and passed its
  short-context numerical and 32-token parity gate. Its real Q8 conversion,
  full sidecar reopen, mixed packed-Metal execution, and 417-launch decode audit
  passed. Its Q4 sidecar conversion, full reopen, and 417-launch Metal audit also
  passed, while quantized reference/quality, real-32K,
  real-checkpoint padded/context-boundary, and standardized-performance gates
  remain open. FLAN-UL2 has an executable
  synthetic runtime and strict loader but no real-artifact inference acceptance
  record. Nemotron-H 3.5 has passed strict real-weight loading and a short
  Burn-internal Metal base-model smoke, and now has a separately authenticated
  official base oracle; Burn-versus-official and real-weight MTP parity remain
  open.
- MTP execution, tokenization, cached generation, T5 beam search, and native Q4/Q8
  are runtime facilities rather than configuration claims. Automatic packed
  checkpoint loading is limited to `muse_glimmer_text`, `qwen3_5_text`, and
  `qwen3_moe` (exact dequantized F32 projections on Flex, packed QFloat on
  Metal); v2 sidecars bind exact checkpoint, load-plan, and payload hashes,
  while legacy v1 artifacts retain full source-payload validation. Qwen3.8 and
  Qwen3-MoE Q8/Q4 Metal smokes passed, while Muse has traced Q8 and Q4 Metal
  evidence; their quantized-reference and quality gates remain open.
- Qwen and Nemotron-H MTP speculative generation accepts batch size one only and
  defaults to two draft tokens per cycle. A different positive draft count is
  configurable; ordinary inference of both families runs batched.
- The Llama-family, Gemma, and truncated gpt-oss real-checkpoint gates have
  run on Flex and Metal in F32, most recently under `c15f24e`'s install order
  at `53f167b` and `c2936fe`, with lines identical to `f4c56c4`
  ([Mac validation record](acceptance/measurements/mac-validation-20261017.json));
  the full-model gpt-oss half-precision gates ran at `c15f24e`. The
  Llama-family and Gemma half-precision profiles have run once: Metal/F16
  passes on the Llama family and cannot pass on gemma-3-4b-it, which
  overflows F16, and Flex/BF16 fails five Llama-family gates and Gemma's
  cached greedy gate pending decisions on the calibration rule, the decode
  tolerance, and half-precision SiLU (see that record). Qwen2-7B is
  qualified against an F64 reference, and its Metal run needs the documented
  `2.5e-3` tolerance (above). Gemma contexts beyond 1,237 tokens and Gemma 2
  real weights are not gated. Full 24-block gpt-oss references exist in BF16 and F32; half-precision runs
  are qualified against the F32 one, where Flex/BF16 and Metal/F16 pass
  ([reference-precision follow-ups](acceptance/measurements/reference-precision-followups-20261016.json)).
- MXFP4 experts are decoded to the load dtype, so gpt-oss-20b needs about
  39 GiB in F16/BF16; there is no packed MXFP4 execution. On Flex, batched
  decoding of wide MoE layers is not faster than row-by-row decoding.
- Three model decode states support speculative transactions, through two
  interfaces. DeepSeek-V2 implements `cache::TransactionalCache` (`mark`, one
  uniform `commit_prefix`, `rollback`). Qwen3.5 and Nemotron-H instead expose
  inherent `mark`, `commit_prefixes(mark, target, mtp)`, and `rollback`,
  because their MTP head commits its own prefix; generic code bounded on
  `TransactionalCache` therefore reaches only DeepSeek-V2. The Llama-family,
  Gemma, gpt-oss, Qwen3-MoE, Muse Glimmer, T5, and modular decode states have
  no transactions; they roll back with `truncate`, which a sliding ring
  supports only within its window.
- Multimodal execution, tensor/expert parallelism, external GGUF/AWQ/GPTQ/FP8
  formats, server/distributed execution, and a separate training framework
  remain out of scope; training uses native Burn (see [TRAINING.md](TRAINING.md)).
