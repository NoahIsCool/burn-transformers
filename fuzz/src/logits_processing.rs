//! Logits processing with token history, checked against a scalar `f64` oracle.
//!
//! [`LogitsProcessor`] applies, per row, logit bias, hard constraints (EOS
//! before `min_new_tokens`, no-repeat n-grams), repetition/presence/frequency
//! penalties, and, when sampling, temperature, top-k, top-p, and min-p. The
//! harness decodes a small batch (vocabulary at most 8, at most 3 rows of at most
//! 6 tokens each, ragged prompt boundaries) and checks:
//!
//! 1. **Rejection.** Rows with an out-of-vocabulary token or a prompt length past
//!    the row end are rejected by [`TokenHistory::from_token_rows`] and
//!    [`LogitsProcessor::process_with_prompt_lengths`]. The tensor-prompt path
//!    ([`TokenHistory::from_prompt`]) reads token values only when a
//!    history-dependent control needs them, so it rejects an out-of-vocabulary
//!    prompt token exactly when a penalty or the n-gram constraint is active.
//! 2. **Path equivalence (bitwise).** Processing through a history rebuilt from
//!    host rows, an incrementally recorded history (tensor prompt plus one
//!    [`TokenHistory::record`] per ragged step), and the stateless
//!    `process_with_prompt_lengths` give identical bits. The legacy
//!    `process_with_mask` path, where every supplied token counts as generated,
//!    equals rows with zero-length prompts.
//! 3. **Pre-warper values.** Greedy processing equals [`expected_pre_warper`]
//!    within `1e-4` relative error, with `-inf` exactly where a constraint bans a
//!    token.
//! 4. **Warpers.** Sampling output keeps each surviving value bit-for-bit (the
//!    greedy value divided by the `f32` temperature) and sets the rest to `-inf`.
//!    Survivors form a top set of the scaled values (no dropped value exceeds
//!    a kept one, even one ulp apart: top-p ranks by logit as Hugging Face
//!    does, finding TOPP-TIE-1, fixed) whose size equals the
//!    count of top-k (every value at least the `k`-th greatest, so ties with
//!    the boundary survive), then top-p (sorted mass *before* an item below
//!    `p`), then min-p (`z >= max + ln(min_p)`); counts within `1e-5` of a
//!    boundary may resolve either way.
//! 5. **Selection.** Greedy picks a row maximum; seeded sampling picks a token
//!    with positive probability, is reproducible from a cloned processor, and
//!    reports a finite non-positive log probability.
//!
//! Logits lie on the grid `k / 4` for `k` in `-16..=16`, and every parameter of
//! the default mode comes from a small table of moderate values, so the
//! oracle's `f64` arithmetic and the processor's `f32` arithmetic agree to
//! rounding. A second, *extreme* mode draws parameters from
//! [`crate::INTERESTING_FLOATS`] (including the `f32` boundaries) and checks
//! that every configuration [`GenerationConfig::validate`] accepts keeps the
//! processed logits finite or `-inf` and the selection defined (findings
//! GEN-F32-1..4, fixed: controls whose `f32` rounding was zero or infinite
//! used to validate and then produce NaN).

use std::collections::{BTreeMap, BTreeSet};

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use transformers::{
    GenerationConfig, LogitsProcessor, PenaltyScope, SamplingStrategy, TokenHistory, TokenMask,
};

use crate::Bytes;

/// Host backend of every tensor in this harness.
type Backend = Flex;

/// Largest vocabulary the harness builds.
const MAX_VOCABULARY: usize = 8;
/// Largest batch the harness builds.
const MAX_BATCH: usize = 3;
/// Largest row (prompt plus generated tokens) the harness builds.
const MAX_ROW: usize = 6;
/// Value written into masked tensor slots; never a valid token ID.
const SENTINEL: i32 = -7;
/// Relative tolerance between the `f32` processor and the `f64` oracle.
const TOLERANCE: f64 = 1e-4;
/// Distance from a top-p or min-p boundary inside which either decision is accepted.
const BOUNDARY_SLACK: f64 = 1e-5;

/// One decoded processing problem.
#[derive(Debug, Clone)]
struct Case {
    /// Vocabulary size `V` (1..=8).
    vocabulary: usize,
    /// Configuration; valid by construction.
    config: GenerationConfig,
    /// Each row's prompt followed by its generated tokens; may hold invalid IDs.
    rows: Vec<Vec<u32>>,
    /// Each row's prompt length; may exceed the row length.
    prompts: Vec<usize>,
    /// Row-major `[batch, V]` logits on the quarter grid.
    logits: Vec<f32>,
    /// Whether parameters came from the extreme table.
    extreme: bool,
}

impl Case {
    /// Number of rows.
    fn batch(&self) -> usize {
        self.rows.len()
    }

    /// Whether every row passes the host-row validation of `from_token_rows`.
    fn rows_are_valid(&self) -> bool {
        self.rows
            .iter()
            .zip(&self.prompts)
            .all(|(row, &prompt)| prompt <= row.len() && self.in_vocabulary(row))
    }

    /// Whether every token of `tokens` lies inside the vocabulary.
    fn in_vocabulary(&self, tokens: &[u32]) -> bool {
        tokens
            .iter()
            .all(|&token| (token as usize) < self.vocabulary)
    }

    /// Whether the configuration reads token values of the history at all.
    ///
    /// Mirrors the documented rule of `TokenHistory::from_prompt`: values are
    /// read (and validated) only for penalties that count tokens and for the
    /// n-gram constraint. The same controls make a processor refuse a history
    /// built for a configuration without them.
    fn reads_token_values(&self) -> bool {
        let config = &self.config;
        config.repetition_penalty != 1.0
            || config.presence_penalty != 0.0
            || config.frequency_penalty != 0.0
            || config.no_repeat_ngram_size > 0
    }

    /// The `[batch, V]` logits tensor.
    fn logits_tensor(&self) -> Tensor<Backend, 2> {
        Tensor::from_data(
            TensorData::new(self.logits.clone(), [self.batch(), self.vocabulary]),
            &Default::default(),
        )
    }

    /// Host rows borrowed as slices.
    fn row_slices(&self) -> Vec<&[u32]> {
        self.rows.iter().map(Vec::as_slice).collect()
    }
}

/// Decodes a table value, or a boundary float in extreme mode.
fn parameter(bytes: &mut Bytes<'_>, extreme: bool, table: &[f64]) -> f64 {
    if extreme && bytes.flag() {
        bytes.interesting_float()
    } else {
        bytes.pick(table)
    }
}

/// Decodes a valid configuration for `vocabulary` tokens.
///
/// Byte layout: strategy, temperature, optional top-k, optional top-p,
/// optional min-p, repetition, presence, and frequency penalties, scope, n-gram
/// size (0..=3), `min_new_tokens` (0..=3), up to two logit-bias entries, and up to
/// two EOS IDs. In extreme mode each float may instead be a boundary value; a
/// draw the documented rules reject (divisors must round to normal positive
/// `f32` values, additive controls to finite ones) falls back to the neutral
/// value, so the result always validates.
fn decode_config(bytes: &mut Bytes<'_>, vocabulary: usize, extreme: bool) -> GenerationConfig {
    let strategy = if bytes.flag() {
        SamplingStrategy::Sample
    } else {
        SamplingStrategy::Greedy
    };
    let mut config = GenerationConfig {
        strategy,
        max_new_tokens: 8,
        temperature: parameter(bytes, extreme, &[0.25, 0.5, 1.0, 1.5, 2.0]),
        top_k: bytes.flag().then(|| bytes.below(vocabulary) + 1),
        top_p: bytes
            .flag()
            .then(|| parameter(bytes, extreme, &[0.05, 0.3, 0.5, 0.9, 0.999, 1.0])),
        min_p: bytes
            .flag()
            .then(|| parameter(bytes, extreme, &[0.0, 0.01, 0.2, 0.5, 1.0])),
        repetition_penalty: parameter(bytes, extreme, &[1.0, 0.5, 1.25, 2.0]),
        presence_penalty: parameter(bytes, extreme, &[0.0, 0.5, -1.0, 2.0]),
        frequency_penalty: parameter(bytes, extreme, &[0.0, 0.25, -0.5, 1.0]),
        additive_penalty_scope: if bytes.flag() {
            PenaltyScope::PromptAndGenerated
        } else {
            PenaltyScope::Generated
        },
        no_repeat_ngram_size: bytes.below(4),
        min_new_tokens: bytes.below(4),
        ..GenerationConfig::default()
    };
    for _ in 0..bytes.below(3) {
        let token = u32::try_from(bytes.below(vocabulary)).expect("small vocabulary");
        let bias = parameter(bytes, extreme, &[-3.0, -0.5, 0.0, 0.75, 2.0]);
        config.logit_bias.insert(token, bias);
    }
    config.eos_token_ids = (0..bytes.below(3))
        .map(|_| u32::try_from(bytes.below(vocabulary)).expect("small vocabulary"))
        .collect();
    // Replace invalid extreme draws by neutral values, field by field, using
    // the documented F32 rules (independent of the library's implementation).
    let neutral = GenerationConfig::default();
    let positive = |value: f64| value.is_finite() && value > 0.0;
    let normal_f32 = |value: f64| {
        let rounded = value as f32;
        rounded.is_finite() && rounded >= f32::MIN_POSITIVE
    };
    let finite_f32 = |value: f64| (value as f32).is_finite();
    if !normal_f32(config.temperature) {
        config.temperature = neutral.temperature;
    }
    if config.top_p.is_some_and(|p| !(positive(p) && p <= 1.0)) {
        config.top_p = None;
    }
    if config.min_p.is_some_and(|p| !(0.0..=1.0).contains(&p)) {
        config.min_p = None;
    }
    if !normal_f32(config.repetition_penalty) {
        config.repetition_penalty = neutral.repetition_penalty;
    }
    if !finite_f32(config.presence_penalty) {
        config.presence_penalty = 0.0;
    }
    if !finite_f32(config.frequency_penalty) {
        config.frequency_penalty = 0.0;
    }
    config.logit_bias.retain(|_, bias| finite_f32(*bias));
    config
}

/// Decodes one case; see [`exercise`] for the byte layout.
fn decode_case(bytes: &mut Bytes<'_>) -> Case {
    let mode = bytes.byte();
    let extreme = mode >= 224;
    let vocabulary = bytes.below(MAX_VOCABULARY) + 1;
    let batch = bytes.below(MAX_BATCH) + 1;
    let config = decode_config(bytes, vocabulary, extreme);
    let mut rows = Vec::with_capacity(batch);
    let mut prompts = Vec::with_capacity(batch);
    for _ in 0..batch {
        let length = bytes.below(MAX_ROW + 1);
        let row: Vec<u32> = (0..length)
            .map(|_| match bytes.byte() {
                255 => u32::MAX,
                254 => u32::try_from(vocabulary).expect("small vocabulary"),
                value => u32::from(value) % u32::try_from(vocabulary).expect("small vocabulary"),
            })
            .collect();
        let prompt = match bytes.byte() {
            255 => length + 1,
            value => usize::from(value) % (length + 1),
        };
        rows.push(row);
        prompts.push(prompt);
    }
    let logits = (0..batch * vocabulary)
        .map(|_| (f32::from(bytes.byte() % 33) - 16.0) / 4.0)
        .collect();
    Case {
        vocabulary,
        config,
        rows,
        prompts,
        logits,
        extreme,
    }
}

/// Reads a float tensor to host values.
fn host(tensor: Tensor<Backend, 2>) -> Vec<f32> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .expect("F32 logits read back")
}

/// Pre-warper processed logits of one row, computed in `f64` from the documented rules.
///
/// Order: bias; `-inf` for EOS while the row has fewer than `min_new_tokens`
/// generated tokens and for tokens completing an n-gram already in the row;
/// sign-aware repetition penalty on every token seen in the row; then
/// `z -= f * c + p * [c > 0]` over the scope's counts.
fn expected_pre_warper(
    config: &GenerationConfig,
    row: &[u32],
    prompt: usize,
    logits: &[f32],
) -> Vec<f64> {
    let mut scores: Vec<f64> = logits.iter().map(|&value| f64::from(value)).collect();
    for (&token, &bias) in &config.logit_bias {
        scores[token as usize] += bias;
    }
    let generated = &row[prompt..];
    if generated.len() < config.min_new_tokens {
        for &token in &config.eos_token_ids {
            scores[token as usize] = f64::NEG_INFINITY;
        }
    }
    let size = config.no_repeat_ngram_size;
    if size > 0 && row.len() + 1 >= size {
        let prefix = &row[row.len() + 1 - size..];
        for window in row.windows(size) {
            if &window[..size - 1] == prefix {
                scores[window[size - 1] as usize] = f64::NEG_INFINITY;
            }
        }
    }
    let penalty = config.repetition_penalty;
    if penalty != 1.0 {
        for &token in row.iter().collect::<BTreeSet<_>>() {
            let score = &mut scores[token as usize];
            *score = if *score < 0.0 {
                *score * penalty
            } else {
                *score / penalty
            };
        }
    }
    let counted = match config.additive_penalty_scope {
        PenaltyScope::Generated => generated,
        PenaltyScope::PromptAndGenerated => row,
    };
    let mut counts = BTreeMap::<u32, f64>::new();
    for &token in counted {
        *counts.entry(token).or_default() += 1.0;
    }
    for (token, count) in counts {
        scores[token as usize] -= config.frequency_penalty * count + config.presence_penalty;
    }
    scores
}

/// Asserts that `actual` matches `expected` within [`TOLERANCE`], with exact infinities.
fn assert_close(actual: &[f32], expected: &[f64], context: &str) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(actual);
        if expected.is_infinite() {
            assert_eq!(actual, expected, "{context}: token {index}");
        } else {
            let scale = expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= TOLERANCE * scale,
                "{context}: token {index} is {actual}, expected {expected}"
            );
        }
    }
}

/// Inclusive range of survivor counts a threshold rule may produce.
///
/// `values` are sorted in descending order; `keeps(j)` says whether sorted
/// item `j` survives and `margin(j)` how far it is from the rule's boundary.
/// Items within [`BOUNDARY_SLACK`] of the boundary may resolve either way.
fn survivor_range(
    count: usize,
    keeps: impl Fn(usize) -> bool,
    margin: impl Fn(usize) -> f64,
) -> (usize, usize) {
    let mut low = 0;
    let mut high = 0;
    for index in 0..count {
        let ambiguous = margin(index).abs() < BOUNDARY_SLACK;
        if keeps(index) || ambiguous {
            high = index + 1;
        }
        if keeps(index) && !ambiguous && low == index {
            low = index + 1;
        }
    }
    (low, high)
}

/// Checks the sampling warpers of one row against the greedy (pre-warper) row.
fn check_warpers(config: &GenerationConfig, vocabulary: usize, greedy: &[f32], sampled: &[f32]) {
    let temperature = config.temperature as f32;
    let scaled: Vec<f32> = greedy.iter().map(|&value| value / temperature).collect();
    // Surviving values are the scaled values bit for bit; the rest are -inf.
    let mut survivors = Vec::new();
    for (index, (&kept, &value)) in sampled.iter().zip(&scaled).enumerate() {
        if kept == f32::NEG_INFINITY {
            continue;
        }
        assert_eq!(
            kept.to_bits(),
            value.to_bits(),
            "token {index} was rescaled"
        );
        survivors.push(value);
    }
    let mut finite: Vec<f64> = scaled
        .iter()
        .filter(|value| value.is_finite())
        .map(|&value| f64::from(value))
        .collect();
    finite.sort_by(|left, right| right.total_cmp(left));
    // Survivors form a top set: nothing outside beats anything inside, with no
    // slack. Top-k and min-p are thresholds on the values, and top-p ranks by
    // logit as Hugging Face's `TopPLogitsWarper` does, so even logits one ulp
    // apart never split (finding TOPP-TIE-1, fixed; campaign regression
    // `crash-cfb301be` once dropped 0.8 while 0.79999995 survived).
    if let Some(weakest) = survivors.iter().copied().reduce(f32::min) {
        for (&kept, &value) in sampled.iter().zip(&scaled) {
            if kept == f32::NEG_INFINITY && value.is_finite() {
                assert!(value <= weakest, "{value} dropped while {weakest} survived");
            }
        }
    }
    // Top-k keeps every finite value at least the k-th greatest, so values
    // tied with that boundary all survive (Hugging Face's `TopKLogitsWarper`
    // rule, `scores < topk(scores, k)[..., -1]` is removed); with fewer than
    // `k` finite values all of them survive.
    let mut count = match config.top_k.filter(|&k| k < vocabulary) {
        Some(k) if k <= finite.len() => {
            let boundary = finite[k - 1];
            finite.iter().filter(|&&value| value >= boundary).count()
        }
        _ => finite.len(),
    };
    let (mut low, mut high) = (count, count);
    // Top-p keeps sorted items whose preceding mass is below p.
    if let Some(top_p) = config.top_p.filter(|&p| p < 1.0)
        && count > 0
    {
        let maximum = finite[0];
        let weights: Vec<f64> = finite[..count]
            .iter()
            .map(|value| (value - maximum).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let before: Vec<f64> = weights
            .iter()
            .scan(0.0, |mass, &weight| {
                let preceding = *mass;
                *mass += weight / total;
                Some(preceding)
            })
            .collect();
        (low, high) = survivor_range(count, |j| before[j] < top_p, |j| before[j] - top_p);
        count = high;
    }
    // Min-p keeps values at least max + ln(min_p).
    if let Some(min_p) = config.min_p.filter(|&p| p > 0.0)
        && count > 0
    {
        let threshold = finite[0] + min_p.ln();
        let (min_low, min_high) =
            survivor_range(count, |j| finite[j] >= threshold, |j| finite[j] - threshold);
        low = low.min(min_low);
        high = min_high;
    }
    assert!(
        (low..=high).contains(&survivors.len()),
        "{} survivors, expected {low}..={high}",
        survivors.len()
    );
}

/// Checks greedy or seeded selection on processed rows.
fn check_selection(processor: &LogitsProcessor<Backend>, processed: &[f32], vocabulary: usize) {
    let batch = processed.len() / vocabulary;
    let tensor = Tensor::<Backend, 2>::from_data(
        TensorData::new(processed.to_vec(), [batch, vocabulary]),
        &Default::default(),
    );
    let selection = processor
        .clone()
        .select_with_log_probs(tensor.clone())
        .expect("valid processed logits");
    let again = processor.clone().select(tensor).expect("valid logits");
    let tokens = selection
        .token_ids
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .expect("token IDs");
    let repeated = again
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .expect("token IDs");
    assert_eq!(
        tokens, repeated,
        "a cloned processor must reproduce its draw"
    );
    let log_probabilities = selection
        .log_probabilities
        .into_data()
        .convert::<f64>()
        .to_vec::<f64>()
        .expect("log probabilities");
    for row in 0..batch {
        let values = &processed[row * vocabulary..(row + 1) * vocabulary];
        if !values.iter().any(|value| value.is_finite()) {
            continue;
        }
        let token = usize::try_from(tokens[row]).expect("non-negative token");
        assert!(token < vocabulary);
        assert!(
            values[token].is_finite(),
            "selected a zero-probability token"
        );
        if processor.config().strategy == SamplingStrategy::Greedy {
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(values[token], maximum);
        }
        let log_probability = log_probabilities[row];
        assert!(log_probability.is_finite() && log_probability <= 1e-6);
    }
}

/// Builds the incremental history: a masked tensor prompt, then one record per step.
fn incremental_history(case: &Case) -> transformers::Result<TokenHistory<Backend>> {
    let device = Default::default();
    let batch = case.batch();
    let width = case.prompts.iter().copied().max().unwrap_or(0);
    let mut ids = Vec::with_capacity(batch * width);
    let mut valid = Vec::with_capacity(batch * width);
    for (row, &prompt) in case.rows.iter().zip(&case.prompts) {
        for &token in &row[..prompt] {
            ids.push(i32::try_from(token).unwrap_or(i32::MAX));
            valid.push(1_i32);
        }
        for _ in prompt..width {
            ids.push(SENTINEL);
            valid.push(0);
        }
    }
    let prompt =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(ids, [batch, width]), &device);
    let mask = (width > 0).then(|| {
        TokenMask::new(
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(valid, [batch, width]), &device)
                .bool(),
        )
    });
    let mut history = TokenHistory::from_prompt(&case.config, case.vocabulary, prompt, mask)?;
    let steps = case
        .rows
        .iter()
        .zip(&case.prompts)
        .map(|(row, &prompt)| row.len() - prompt)
        .max()
        .unwrap_or(0);
    for step in 0..steps {
        let mut tokens = Vec::with_capacity(batch);
        let mut active = Vec::with_capacity(batch);
        for (row, &prompt) in case.rows.iter().zip(&case.prompts) {
            let token = row.get(prompt + step).copied();
            tokens.push(token.unwrap_or(0));
            active.push(token.is_some());
        }
        history.record(&tokens, &active)?;
    }
    Ok(history)
}

/// Processes through the legacy stateless path, where every row token counts as generated.
fn legacy_processing(
    processor: &LogitsProcessor<Backend>,
    case: &Case,
) -> transformers::Result<Vec<f32>> {
    let device = Default::default();
    let batch = case.batch();
    let width = case.rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut ids = Vec::with_capacity(batch * width);
    let mut valid = Vec::with_capacity(batch * width);
    for row in &case.rows {
        for column in 0..width {
            match row.get(column) {
                Some(&token) => {
                    ids.push(i32::try_from(token).unwrap_or(i32::MAX));
                    valid.push(1_i32);
                }
                None => {
                    ids.push(SENTINEL);
                    valid.push(0);
                }
            }
        }
    }
    let history =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(ids, [batch, width]), &device);
    let mask = TokenMask::new(
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(valid, [batch, width]), &device)
            .bool(),
    );
    processor
        .process_with_mask(case.logits_tensor(), Some(history), Some(mask))
        .map(host)
}

/// Checks the rejection rules of every construction path for invalid rows.
fn check_rejections(case: &Case, processor: &LogitsProcessor<Backend>) {
    let device = Default::default();
    let rows = case.row_slices();
    assert!(
        TokenHistory::<Backend>::from_token_rows(
            &case.config,
            case.vocabulary,
            &rows,
            &case.prompts,
            &device
        )
        .is_err()
    );
    assert!(
        processor
            .process_with_prompt_lengths(case.logits_tensor(), &rows, &case.prompts)
            .is_err()
    );
    // The tensor path validates prompt values only when it reads them, and
    // every recorded token always.
    if case
        .prompts
        .iter()
        .zip(&case.rows)
        .all(|(&p, row)| p <= row.len())
    {
        let prompt_valid = case
            .rows
            .iter()
            .zip(&case.prompts)
            .all(|(row, &prompt)| case.in_vocabulary(&row[..prompt]));
        let generated_valid = case
            .rows
            .iter()
            .zip(&case.prompts)
            .all(|(row, &prompt)| case.in_vocabulary(&row[prompt..]));
        let expected = (prompt_valid || !case.reads_token_values()) && generated_valid;
        assert_eq!(incremental_history(case).is_ok(), expected);
    }
}

/// Runs every path on one decoded case.
fn check_case(case: &Case) {
    let device = Default::default();
    assert!(case.config.validate(case.vocabulary).is_ok(), "{case:?}");
    let processor = LogitsProcessor::<Backend>::new(case.config.clone(), case.vocabulary, &device)
        .expect("valid configuration");
    // Wrong row counts are rejected before any history is built.
    let rows = case.row_slices();
    assert!(
        processor
            .process_with_prompt_lengths(case.logits_tensor(), &rows[1..], &case.prompts[1..])
            .is_err()
            || case.batch() == 1
    );
    if !case.rows_are_valid() {
        check_rejections(case, &processor);
        return;
    }
    let stateless = host(
        processor
            .process_with_prompt_lengths(case.logits_tensor(), &rows, &case.prompts)
            .expect("valid rows"),
    );
    let from_rows =
        TokenHistory::from_token_rows(&case.config, case.vocabulary, &rows, &case.prompts, &device)
            .expect("valid rows");
    let rebuilt = host(
        processor
            .process_with_history(case.logits_tensor(), &from_rows)
            .expect("compatible history"),
    );
    assert_bitwise(&stateless, &rebuilt, "rebuilt history");
    let incremental = incremental_history(case).expect("valid incremental history");
    let recorded = host(
        processor
            .process_with_history(case.logits_tensor(), &incremental)
            .expect("compatible history"),
    );
    assert_bitwise(&stateless, &recorded, "recorded history");
    // The legacy tensor path counts every token as generated.
    let zeros = vec![0; case.batch()];
    let as_generated =
        TokenHistory::from_token_rows(&case.config, case.vocabulary, &rows, &zeros, &device)
            .expect("valid rows");
    let generated_only = host(
        processor
            .process_with_history(case.logits_tensor(), &as_generated)
            .expect("compatible history"),
    );
    let legacy = legacy_processing(&processor, case).expect("valid legacy history");
    assert_bitwise(&generated_only, &legacy, "legacy masked history");
    // A history built for other controls is refused when the processor reads them.
    let plain = TokenHistory::<Backend>::empty(
        &GenerationConfig::default(),
        case.vocabulary,
        case.batch(),
        &device,
    )
    .expect("empty history");
    assert_eq!(
        processor
            .process_with_history(case.logits_tensor(), &plain)
            .is_ok(),
        !case.reads_token_values()
    );

    let greedy = if case.config.strategy == SamplingStrategy::Greedy {
        stateless.clone()
    } else {
        let config = GenerationConfig {
            strategy: SamplingStrategy::Greedy,
            ..case.config.clone()
        };
        let greedy = LogitsProcessor::<Backend>::new(config, case.vocabulary, &device)
            .expect("valid configuration");
        host(
            greedy
                .process_with_prompt_lengths(case.logits_tensor(), &rows, &case.prompts)
                .expect("valid rows"),
        )
    };
    let vocabulary = case.vocabulary;
    // Every valid configuration keeps processed logits finite or `-inf`.
    assert!(
        stateless
            .iter()
            .chain(&greedy)
            .all(|value| value.is_finite() || *value == f32::NEG_INFINITY),
        "{case:?}: {stateless:?}"
    );
    for (row_index, (row, &prompt)) in case.rows.iter().zip(&case.prompts).enumerate() {
        let span = row_index * vocabulary..(row_index + 1) * vocabulary;
        if case.extreme {
            continue;
        }
        let expected = expected_pre_warper(&case.config, row, prompt, &case.logits[span.clone()]);
        assert_close(&greedy[span.clone()], &expected, "pre-warper logits");
        if case.config.strategy == SamplingStrategy::Sample {
            check_warpers(
                &case.config,
                vocabulary,
                &greedy[span.clone()],
                &stateless[span],
            );
        }
    }
    check_selection(&processor, &stateless, vocabulary);
}

/// Asserts bitwise equality of two processed tensors.
fn assert_bitwise(expected: &[f32], actual: &[f32], context: &str) {
    assert_eq!(expected.len(), actual.len());
    for (index, (left, right)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "{context}: element {index} differs ({left} vs {right})"
        );
    }
}

/// Decodes and checks one logits-processing problem.
///
/// Byte layout: a mode byte (at least 224 selects extreme parameters), the
/// vocabulary (1..=8), the batch (1..=3), the configuration of
/// [`decode_config`], then per row its length (0..=6), its tokens (254 and 255
/// decode to out-of-vocabulary IDs), and its prompt length (255 is one past the
/// row end), and finally one byte per logit, mapped to `(b mod 33 - 16) / 4`.
pub fn exercise(input: &[u8]) {
    let mut bytes = Bytes::new(input);
    let case = decode_case(&mut bytes);
    check_case(&case);
}

#[cfg(test)]
#[path = "logits_processing_tests.rs"]
mod tests;
