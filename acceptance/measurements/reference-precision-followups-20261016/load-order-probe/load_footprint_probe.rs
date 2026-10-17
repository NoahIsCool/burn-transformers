//! EXPERIMENT ONLY (not committed): load one checkpoint on Metal, synchronize, and idle.
//!
//! TRANSFORMERS_PROBE_DIR names the checkpoint; TRANSFORMERS_PROBE_KIND is `causal` or `seq2seq`;
//! TRANSFORMERS_PROBE_DTYPE is `f16` or `preserve`; TRANSFORMERS_PROBE_REPOSITORY/_REVISION are
//! the provenance some family policies require; TRANSFORMERS_PROBE_PHYSICAL=1 selects physical
//! index alias accounting (Nemotron).

#![cfg(all(feature = "metal", feature = "tokenizers"))]

use burn::{backend::Metal, tensor::backend::Backend};
use transformers::{
    AutoCausalLm, AutoLoadOptions, AutoSeq2SeqLm,
    checkpoint::{DTypePolicy, IndexAliasSizeAccounting, LoadOptions, TensorDType},
};

/// Loads, synchronizes, prints, and idles so a sampler sees the steady post-load footprint.
#[test]
#[ignore = "experiment"]
fn probe_post_load_footprint() {
    let root = std::path::PathBuf::from(std::env::var_os("TRANSFORMERS_PROBE_DIR").expect("dir"));
    let kind = std::env::var("TRANSFORMERS_PROBE_KIND").unwrap_or_else(|_| "causal".into());
    let dtype_policy = match std::env::var("TRANSFORMERS_PROBE_DTYPE").as_deref() {
        Ok("preserve") => DTypePolicy::Preserve,
        _ => DTypePolicy::Cast(TensorDType::F16),
    };
    let mut checkpoint = LoadOptions {
        dtype_policy,
        ..LoadOptions::default()
    };
    if std::env::var("TRANSFORMERS_PROBE_PHYSICAL").as_deref() == Ok("1") {
        checkpoint.index_alias_size_accounting = IndexAliasSizeAccounting::Physical;
    }
    let options = AutoLoadOptions {
        checkpoint,
        load_tokenizer: false,
        repository: std::env::var("TRANSFORMERS_PROBE_REPOSITORY").ok(),
        revision: std::env::var("TRANSFORMERS_PROBE_REVISION").ok(),
        ..AutoLoadOptions::default()
    };
    let device = Default::default();
    let started = std::time::Instant::now();
    let opens = if kind == "seq2seq" {
        let loaded = AutoSeq2SeqLm::<Metal>::load_local(&root, &options, &device).expect("load");
        <Metal as Backend>::sync(&device).expect("sync");
        let opens = loaded.execution_stats().map(|stats| stats.source.shard_opens);
        eprintln!("probe loaded seq2seq in {:.1}s shard_opens={opens:?}", started.elapsed().as_secs_f64());
        std::thread::sleep(std::time::Duration::from_secs(3));
        drop(loaded);
        opens
    } else {
        let loaded = AutoCausalLm::<Metal>::load_local(&root, &options, &device).expect("load");
        <Metal as Backend>::sync(&device).expect("sync");
        let opens = loaded.execution_stats().map(|stats| stats.source.shard_opens);
        eprintln!("probe loaded causal in {:.1}s shard_opens={opens:?}", started.elapsed().as_secs_f64());
        std::thread::sleep(std::time::Duration::from_secs(3));
        drop(loaded);
        opens
    };
    assert!(opens.is_some());
}
