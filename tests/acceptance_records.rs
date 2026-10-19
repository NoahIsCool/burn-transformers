//! Structural gates for committed checkpoint and bounded v0.1 release evidence.
//!
//! Support is an exact checkpoint/backend/dtype claim, not a family-wide claim.
//! The closed allowlist makes any expanded support or altered rebaseline decision
//! require an explicit reviewed change to these gates.
//!
//! The Qwen3.5-0.8B performance reference has two layers. The v0.1 release
//! record names the September 10 model-loop series, and it keeps its bytes
//! (pinned here by digest). The October 18 re-baseline record supersedes that
//! series for future comparisons: it carries the maintainer's authorization,
//! the gated metrics (warm, natural-loop and strict-load medians at the 5%
//! alert / 10% confirmed-failure thresholds) and the ungated cold first token,
//! and its reference medians are what `scripts/qwen35_benchmark_protocol.py`
//! judges a candidate against.

use std::{collections::BTreeSet, fs, path::PathBuf};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// Approved shared release record, always resolved under the acceptance directory.
const RELEASE_RECORD: &str = "measurements/v0_1-release-20260910.json";
/// Digest of the release record, which the October 18 re-baseline supersedes without editing it.
const RELEASE_RECORD_SHA256: &str =
    "0fa1fe79ad85528c1a62bc897528ee8bee1e1c5b3617ba07f2cf8b1911239405";
/// Phase reserved for the approved bounded release profiles.
const SUPPORTED_PHASE: &str = "v0_1_supported_bounded_profile";
/// Model-loop series the v0.1 release record approved, superseded for future comparisons on October 18.
const QWEN_BASELINE: &str = "small-model-performance-20260910/qwen-controlled-current-summary.json";
/// Content identity of the approved compact timing series, not a new measurement.
const QWEN_BASELINE_SHA256: &str =
    "1a1c989ad9fe1b4d0ee87b8d35e0c41a4005ca61f6677ffc58b711a844cee899";
/// Re-baseline record whose reference medians future Qwen3.5-0.8B comparisons use.
const QWEN_REBASELINE_RECORD: &str = "measurements/qwen3.5-0.8b-metal-rebaseline-20261018.json";
/// The maintainer's October 18 decision, quoted exactly.
const REBASELINE_REASON: &str = "re-baseline the Qwen3.5-0.8B benchmarks on the current host (macOS 27.0.1, same Mac15,11), and report cold first token as a diagnostic rather than a pass/fail gate.";
/// Gated metrics of the baseline benchmark: strict load, twelve warm medians, and the model-API loop.
const BASELINE_GATED: [&str; 14] = [
    "strict_load",
    "source_32.prefill",
    "source_32.first_token_latency",
    "source_32.decode_model_only",
    "source_32.decode_including_greedy_readback",
    "source_128.prefill",
    "source_128.first_token_latency",
    "source_128.decode_model_only",
    "source_128.decode_including_greedy_readback",
    "source_512.prefill",
    "source_512.first_token_latency",
    "source_512.decode_model_only",
    "source_512.decode_including_greedy_readback",
    "natural_loop",
];
/// Gated metrics of the public-coordinator benchmark: strict load and the coordinator loop.
const PUBLIC_GATED: [&str; 2] = ["strict_load", "natural_loop"];
/// Profile identifiers listed by the release record, without duplicate aliases.
const PROFILE_IDS: [&str; 3] = [
    "qwen3.5-0.8b-metal-f16",
    "flan-t5-small-flex-f32",
    "flan-t5-small-metal-f32",
];
/// Exact checkpoint-record references relative to the measurements directory.
const CHECKPOINT_RECORDS: [&str; 2] = [
    "../qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17.json",
    "../flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json",
];
/// Approved Qwen checkpoint identity.
const QWEN: (&str, &str) = (
    "Qwen/Qwen3.5-0.8B",
    "2fc06364715b967f1860aea9cf38778875588b17",
);
/// Approved encoder-decoder checkpoint identity.
const FLAN: (&str, &str) = (
    "google/flan-t5-small",
    "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
);

/// An exact profile ID, backend spelling and floating-point dtype.
type Profile = (&'static str, &'static str, &'static str);

/// Returns the complete profile set for an exactly approved checkpoint.
fn approved_profiles(repository: &str, revision: &str) -> Option<&'static [Profile]> {
    match (repository, revision) {
        QWEN => Some(&[("qwen3.5-0.8b-metal-f16", "Metal", "F16")]),
        FLAN => Some(&[
            ("flan-t5-small-flex-f32", "Flex", "F32"),
            ("flan-t5-small-metal-f32", "Metal", "F32"),
        ]),
        _ => None,
    }
}

/// Pins the certified Cargo feature combination for each execution profile.
fn approved_features(id: &str) -> Option<&'static [&'static str]> {
    match id {
        "qwen3.5-0.8b-metal-f16" => Some(&["std", "metal", "tokenizers", "chat-templates"]),
        "flan-t5-small-flex-f32" => Some(&["std", "flex", "tokenizers"]),
        "flan-t5-small-metal-f32" => Some(&["std", "metal", "tokenizers"]),
        _ => None,
    }
}

/// Requires a present nonblank string, not null or whitespace-only text.
fn nonempty_text<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("{field} must be nonempty text"))
}

/// Compares an exact string set while rejecting missing, duplicate and malformed entries.
fn exact_string_set(value: &Value, expected: &[&str], field: &str) -> Result<(), String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{field} must be an array"))?;
    let mut actual = BTreeSet::new();
    for value in values {
        let value = nonempty_text(value, field)?;
        if !actual.insert(value) {
            return Err(format!("{field} contains duplicate {value}"));
        }
    }
    if actual != expected.iter().copied().collect() {
        return Err(format!("{field} must contain exactly the approved entries"));
    }
    Ok(())
}

/// Validates the release identity and explicitly authorized, unexplained rebaseline.
fn validate_release(record: &Value) -> Result<(), String> {
    for (field, expected) in [
        ("schema_version", json!(1)),
        ("version", json!("0.1.0")),
        ("date", json!("2026-09-10")),
        ("release_gate_promotion", json!(true)),
        ("published_or_tagged", json!(false)),
    ] {
        if record[field] != expected {
            return Err(format!("release.{field} must equal {expected}"));
        }
    }
    for (field, expected) in [
        ("user_authorized_rebaseline", true),
        ("historical_slowdown_explained", false),
        ("historical_measurements_modified", false),
        ("new_performance_run_claimed", false),
    ] {
        if record["performance_policy"][field].as_bool() != Some(expected) {
            return Err(format!(
                "release.performance_policy.{field} must be {expected}"
            ));
        }
    }
    for (field, expected) in [
        ("future_alert_percent", json!(5)),
        ("future_failure_percent_after_controlled_rerun", json!(10)),
        ("qwen_model_loop_baseline", json!(QWEN_BASELINE)),
        (
            "qwen_model_loop_baseline_sha256",
            json!(QWEN_BASELINE_SHA256),
        ),
    ] {
        if record["performance_policy"][field] != expected {
            return Err(format!(
                "release.performance_policy.{field} must equal {expected}"
            ));
        }
    }
    exact_string_set(
        &record["promoted_profiles"],
        &PROFILE_IDS,
        "promoted_profiles",
    )?;
    exact_string_set(
        &record["checkpoint_records"],
        &CHECKPOINT_RECORDS,
        "checkpoint_records",
    )
}

/// Requires a positive, finite number of seconds.
fn positive_seconds(value: &Value, field: &str) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .ok_or_else(|| format!("{field} must be positive finite seconds"))
}

/// Validates the October 18 re-baseline: its authorization, what it supersedes, and its gates.
///
/// The record changes no support claim and edits no earlier record; it names
/// the release record and the September 10 series it supersedes by their
/// digests. Its reference must hold exactly the gated metrics of each
/// benchmark, each with a positive median, and an explicitly ungated cold
/// first token; nothing else may gate.
fn validate_rebaseline(record: &Value) -> Result<(), String> {
    for (pointer, expected) in [
        ("/schema_version", json!(1)),
        ("/advertised_supported", json!(false)),
        ("/support_claims_changed", json!(false)),
        ("/repository", json!(QWEN.0)),
        ("/revision", json!(QWEN.1)),
        ("/profile", json!(PROFILE_IDS[0])),
        (
            "/performance_policy/user_authorized_rebaseline",
            json!(true),
        ),
        (
            "/performance_policy/authorization_date",
            json!("2026-10-18"),
        ),
        (
            "/performance_policy/quoted_reason",
            json!(REBASELINE_REASON),
        ),
        (
            "/performance_policy/historical_measurements_modified",
            json!(false),
        ),
        ("/performance_policy/future_alert_percent", json!(5)),
        (
            "/performance_policy/future_failure_percent_after_controlled_rerun",
            json!(10),
        ),
        ("/performance_policy/cold_first_token_gated", json!(false)),
        (
            "/performance_policy/diagnostic_metrics",
            json!(["cold_first_token"]),
        ),
        (
            "/performance_policy/cold_first_token_reported_as",
            json!(["median", "range"]),
        ),
        (
            "/performance_policy/supersedes/record",
            json!(RELEASE_RECORD.trim_start_matches("measurements/")),
        ),
        (
            "/performance_policy/supersedes/record_sha256",
            json!(RELEASE_RECORD_SHA256),
        ),
        (
            "/performance_policy/supersedes/qwen_model_loop_baseline",
            json!(QWEN_BASELINE),
        ),
        (
            "/performance_policy/supersedes/qwen_model_loop_baseline_sha256",
            json!(QWEN_BASELINE_SHA256),
        ),
        ("/reference/label", json!("head")),
    ] {
        if record.pointer(pointer) != Some(&expected) {
            return Err(format!("rebaseline{pointer} must equal {expected}"));
        }
    }
    for (benchmark, metrics) in [
        ("baseline", &BASELINE_GATED[..]),
        ("public", &PUBLIC_GATED[..]),
    ] {
        if record["performance_policy"]["gated_metrics"][benchmark] != json!(metrics) {
            return Err(format!(
                "rebaseline gated_metrics.{benchmark} must list exactly the gated metrics"
            ));
        }
        let reference = &record["reference"]["benchmarks"][benchmark];
        let gated = reference["gated"]
            .as_object()
            .ok_or_else(|| format!("rebaseline reference {benchmark} gated must be an object"))?;
        if gated.keys().map(String::as_str).collect::<BTreeSet<_>>()
            != metrics.iter().copied().collect()
        {
            return Err(format!(
                "rebaseline reference {benchmark} must gate exactly its gated metrics"
            ));
        }
        for (metric, statistic) in gated {
            positive_seconds(
                &statistic["median"],
                &format!("rebaseline reference {benchmark}.{metric}.median"),
            )?;
        }
        let cold = &reference["diagnostic"]["cold_first_token"];
        if cold["gated"] != json!(false) {
            return Err(format!(
                "rebaseline reference {benchmark} cold_first_token must be ungated"
            ));
        }
        positive_seconds(
            &cold["median"],
            &format!("rebaseline reference {benchmark}.cold_first_token.median"),
        )?;
    }
    Ok(())
}

/// Pins execution metadata and requires nonblank scope/limit prose for human review.
fn validate_profiles(record: &Value, expected: &[Profile]) -> Result<(), String> {
    if record["phase"] != SUPPORTED_PHASE {
        return Err(format!("promoted phase must be {SUPPORTED_PHASE}"));
    }
    if record["release_record"] != RELEASE_RECORD {
        return Err(format!("release_record must be {RELEASE_RECORD}"));
    }
    let profiles = record["supported_profiles"]
        .as_array()
        .ok_or("supported_profiles must be an array")?;
    let mut seen = BTreeSet::new();
    for profile in profiles {
        let id = nonempty_text(&profile["id"], "profile.id")?;
        if !seen.insert(id) {
            return Err(format!("duplicate profile ID {id}"));
        }
        let (_, backend, dtype) = expected
            .iter()
            .find(|(allowed, _, _)| *allowed == id)
            .ok_or_else(|| format!("unapproved profile ID {id}"))?;
        if profile["backend"].as_str() != Some(*backend) {
            return Err(format!("profile {id} backend must be {backend}"));
        }
        if profile["dtype"].as_str() != Some(*dtype) {
            return Err(format!("profile {id} dtype must be {dtype}"));
        }
        if id == PROFILE_IDS[0] && profile["recurrent_state_dtype"] != "F32" {
            return Err("Qwen recurrent_state_dtype must be F32".into());
        }
        exact_string_set(
            &profile["features"],
            approved_features(id).ok_or_else(|| format!("no approved features for {id}"))?,
            "profile.features",
        )?;
        nonempty_text(&profile["scope"], "profile.scope")?;
        let limits = profile["known_limits"]
            .as_array()
            .filter(|limits| !limits.is_empty())
            .ok_or_else(|| format!("profile {id} known_limits must be a nonempty array"))?;
        for limit in limits {
            nonempty_text(limit, "profile.known_limits[]")?;
        }
    }
    if seen != expected.iter().map(|(id, _, _)| *id).collect() {
        return Err("supported_profiles must contain the exact approved profile set".into());
    }
    Ok(())
}

/// Preserves pinned metadata and rejects every support claim outside the exact release allowlist.
fn validate_records(records: &[(String, Value)], release: &Value) -> Result<(), String> {
    validate_release(release)?;
    let mut advertised = BTreeSet::new();
    for (filename, record) in records {
        if record["schema_version"] != 1 {
            return Err(format!("{filename}: schema_version must be 1"));
        }
        let repository = nonempty_text(&record["repository"], "repository")?;
        if !repository.contains('/') {
            return Err(format!(
                "{filename}: repository must identify a Hub repository"
            ));
        }
        let revision = nonempty_text(&record["revision"], "revision")?;
        if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "{filename}: revision must be a 40-digit hexadecimal SHA"
            ));
        }
        if !filename.contains(revision) {
            return Err(format!("{filename}: filename must contain revision"));
        }
        nonempty_text(&record["phase"], "phase")?;
        if record["checks"]
            .as_object()
            .is_none_or(|checks| checks.is_empty())
        {
            return Err(format!("{filename}: checks must be a nonempty object"));
        }
        let supported = record["advertised_supported"]
            .as_bool()
            .ok_or_else(|| format!("{filename}: advertised_supported must be a boolean"))?;
        if supported {
            let profiles = approved_profiles(repository, revision)
                .ok_or_else(|| format!("{filename}: checkpoint is not approved for v0.1"))?;
            if !advertised.insert((repository, revision)) {
                return Err(format!("{filename}: duplicate advertised checkpoint"));
            }
            if !CHECKPOINT_RECORDS
                .iter()
                .any(|relative| relative.strip_prefix("../") == Some(filename))
            {
                return Err(format!(
                    "{filename}: advertised record is not referenced by release"
                ));
            }
            validate_profiles(record, profiles)?;
        }
    }
    if advertised != BTreeSet::from([QWEN, FLAN]) {
        return Err("exactly the approved Qwen and FLAN checkpoints must be advertised".into());
    }
    Ok(())
}

/// Returns all top-level checkpoint records in deterministic filename order.
fn acceptance_records() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("acceptance");
    let mut records = fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("read {}: {error}", root.display()))
        .map(|entry| entry.expect("read acceptance directory entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect::<Vec<_>>();
    records.sort();
    records
}

/// Exactly two pinned checkpoints and three bounded profiles advertise v0.1 support.
#[test]
fn committed_acceptance_records_are_pinned_and_match_v0_1_release() {
    let paths = acceptance_records();
    assert!(
        paths.len() >= 7,
        "the known v1 evidence set must not silently disappear"
    );
    let records = paths
        .iter()
        .map(|path| {
            let bytes = fs::read(path).unwrap_or_else(|error| {
                panic!("read acceptance record {}: {error}", path.display())
            });
            let record = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
                panic!("parse acceptance record {}: {error}", path.display())
            });
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("acceptance filenames must be UTF-8")
                .to_owned();
            (filename, record)
        })
        .collect::<Vec<_>>();
    let release_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("acceptance")
        .join(RELEASE_RECORD);
    let bytes = fs::read(&release_path).unwrap_or_else(|error| {
        panic!(
            "read referenced release {}: {error}",
            release_path.display()
        )
    });
    let release = serde_json::from_slice(&bytes).expect("referenced release must be valid JSON");
    validate_records(&records, &release).expect("committed v0.1 support claims must be exact");
}

/// Creates compact metadata fixtures without copying committed evidence or numerical results.
fn valid_release_fixture() -> (Vec<(String, Value)>, Value) {
    let records = [QWEN, FLAN]
        .into_iter()
        .zip(CHECKPOINT_RECORDS)
        .map(|((repository, revision), reference)| {
            let profiles = approved_profiles(repository, revision)
                .unwrap()
                .iter()
                .map(|(id, backend, dtype)| {
                    let mut profile = json!({
                        "id": id, "backend": backend, "dtype": dtype,
                        "features": approved_features(id).unwrap(),
                        "scope": "bounded synthetic fixture",
                        "known_limits": ["not a numerical acceptance fixture"]
                    });
                    if *id == PROFILE_IDS[0] {
                        profile["recurrent_state_dtype"] = json!("F32");
                    }
                    profile
                })
                .collect::<Vec<_>>();
            (
                reference.strip_prefix("../").unwrap().into(),
                json!({
                    "schema_version": 1, "repository": repository, "revision": revision,
                    "phase": SUPPORTED_PHASE, "advertised_supported": true,
                    "release_record": RELEASE_RECORD, "supported_profiles": profiles,
                    "checks": {"fixture_only": true}
                }),
            )
        })
        .collect();
    (
        records,
        json!({
            "schema_version": 1, "version": "0.1.0", "date": "2026-09-10",
            "release_gate_promotion": true, "promoted_profiles": PROFILE_IDS,
            "published_or_tagged": false,
            "checkpoint_records": CHECKPOINT_RECORDS,
            "performance_policy": {"user_authorized_rebaseline": true,
                "historical_slowdown_explained": false,
                "historical_measurements_modified": false,
                "new_performance_run_claimed": false,
                "future_alert_percent": 5,
                "future_failure_percent_after_controlled_rerun": 10,
                "qwen_model_loop_baseline": QWEN_BASELINE,
                "qwen_model_loop_baseline_sha256": QWEN_BASELINE_SHA256}
        }),
    )
}

/// Mutations cannot broaden support, remove profiles or hide malformed profile metadata.
#[test]
fn release_allowlist_rejects_invalid_support_claims_and_profile_metadata() {
    let (records, release) = valid_release_fixture();
    validate_records(&records, &release).expect("exact synthetic allowlist is valid");
    for (pointer, value, expected) in [
        (
            "/repository",
            json!("Qwen/Qwen3.8-27B"),
            "checkpoint is not approved",
        ),
        (
            "/advertised_supported",
            json!(false),
            "exactly the approved",
        ),
        ("/advertised_supported", json!("true"), "must be a boolean"),
        ("/revision", json!("main"), "40-digit hexadecimal"),
        ("/phase", json!("pre_release"), "promoted phase"),
        (
            "/release_record",
            json!("../release.json"),
            "release_record must be",
        ),
        ("/checks", json!({}), "checks must be"),
        (
            "/supported_profiles",
            json!([]),
            "exact approved profile set",
        ),
        (
            "/supported_profiles/0/id",
            json!("qwen3.5-0.8b-metal-q4"),
            "unapproved profile ID",
        ),
        (
            "/supported_profiles/0/backend",
            json!("CUDA"),
            "backend must be Metal",
        ),
        (
            "/supported_profiles/0/dtype",
            json!("F32"),
            "dtype must be F16",
        ),
        (
            "/supported_profiles/0/features",
            Value::Null,
            "features must be an array",
        ),
        (
            "/supported_profiles/0/recurrent_state_dtype",
            json!("F16"),
            "recurrent_state_dtype must be F32",
        ),
        (
            "/supported_profiles/0/recurrent_state_dtype",
            Value::Null,
            "recurrent_state_dtype must be F32",
        ),
        (
            "/supported_profiles/0/features",
            json!([false]),
            "features must be nonempty text",
        ),
        (
            "/supported_profiles/0/features",
            json!(["std", "std"]),
            "duplicate",
        ),
        (
            "/supported_profiles/0/features",
            json!([]),
            "exactly the approved entries",
        ),
        (
            "/supported_profiles/0/features",
            json!(["std", "metal", "tokenizers", "chat-templates", "rocm"]),
            "exactly the approved entries",
        ),
        (
            "/supported_profiles/0/scope",
            json!("  "),
            "scope must be nonempty text",
        ),
        (
            "/supported_profiles/0/known_limits",
            json!([]),
            "known_limits must be a nonempty array",
        ),
        (
            "/supported_profiles/0/known_limits",
            json!([null]),
            "known_limits[] must be nonempty text",
        ),
    ] {
        let mut changed = records.clone();
        *changed[0]
            .1
            .pointer_mut(pointer)
            .expect("fixture field exists") = value;
        let error = validate_records(&changed, &release).expect_err(pointer);
        assert!(error.contains(expected), "{pointer}: {error}");
    }
    let mut duplicate = records.clone();
    duplicate.push(records[0].clone());
    assert!(
        validate_records(&duplicate, &release)
            .unwrap_err()
            .contains("duplicate advertised")
    );
    let mut missing_state_dtype = records.clone();
    missing_state_dtype[0].1["supported_profiles"][0]
        .as_object_mut()
        .unwrap()
        .remove("recurrent_state_dtype");
    assert!(
        validate_records(&missing_state_dtype, &release)
            .unwrap_err()
            .contains("recurrent_state_dtype must be F32")
    );
    let mut missing = records.clone();
    missing[1].1["supported_profiles"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(
        validate_records(&missing, &release)
            .unwrap_err()
            .contains("exact approved profile set")
    );
    let mut duplicate = records.clone();
    let first = duplicate[1].1["supported_profiles"][0].clone();
    duplicate[1].1["supported_profiles"]
        .as_array_mut()
        .unwrap()
        .push(first);
    assert!(
        validate_records(&duplicate, &release)
            .unwrap_err()
            .contains("duplicate profile ID")
    );
    let mut unapproved = records.clone();
    let mut other = records[0].clone();
    other.0 = format!("historical-{}.json", QWEN.1);
    other.1["repository"] = json!("Qwen/Qwen3.8-27B");
    other.1["advertised_supported"] = json!(false);
    unapproved.push(other);
    validate_records(&unapproved, &release)
        .expect("unadvertised historical evidence remains valid");
    unapproved[2].1["advertised_supported"] = json!(true);
    assert!(
        validate_records(&unapproved, &release)
            .unwrap_err()
            .contains("checkpoint is not approved")
    );
    let mut wrong_revision = records.clone();
    let revision = "a".repeat(40);
    wrong_revision[0].0 = format!("qwen3.5-0.8b-{revision}.json");
    wrong_revision[0].1["revision"] = json!(revision);
    assert!(
        validate_records(&wrong_revision, &release)
            .unwrap_err()
            .contains("checkpoint is not approved")
    );
}

/// Release identity, references and rebaseline authorization are exact typed commitments.
#[test]
fn release_allowlist_rejects_mutated_release_decisions() {
    let (records, release) = valid_release_fixture();
    for (pointer, value, expected) in [
        ("/schema_version", json!(2), "schema_version"),
        ("/version", json!("0.2.0"), "version"),
        ("/date", json!("2026-09-11"), "date"),
        ("/published_or_tagged", json!(true), "published_or_tagged"),
        (
            "/performance_policy/historical_measurements_modified",
            json!(true),
            "historical_measurements_modified",
        ),
        (
            "/performance_policy/new_performance_run_claimed",
            json!(true),
            "new_performance_run_claimed",
        ),
        (
            "/performance_policy/future_alert_percent",
            json!(6),
            "future_alert_percent",
        ),
        (
            "/performance_policy/future_failure_percent_after_controlled_rerun",
            json!(11),
            "future_failure_percent_after_controlled_rerun",
        ),
        (
            "/performance_policy/qwen_model_loop_baseline",
            json!("unapproved-series.json"),
            "qwen_model_loop_baseline",
        ),
        (
            "/performance_policy/qwen_model_loop_baseline_sha256",
            json!("0".repeat(64)),
            "qwen_model_loop_baseline_sha256",
        ),
        (
            "/release_gate_promotion",
            json!(false),
            "release_gate_promotion",
        ),
        (
            "/promoted_profiles",
            json!([PROFILE_IDS[0]]),
            "promoted_profiles",
        ),
        (
            "/promoted_profiles",
            json!([PROFILE_IDS[0], PROFILE_IDS[0], PROFILE_IDS[2]]),
            "duplicate",
        ),
        (
            "/checkpoint_records",
            json!([CHECKPOINT_RECORDS[0]]),
            "checkpoint_records",
        ),
        (
            "/checkpoint_records",
            json!([CHECKPOINT_RECORDS[0], "../unknown.json"]),
            "checkpoint_records",
        ),
        (
            "/performance_policy/user_authorized_rebaseline",
            json!(false),
            "user_authorized_rebaseline",
        ),
        (
            "/performance_policy/user_authorized_rebaseline",
            json!("true"),
            "user_authorized_rebaseline",
        ),
        (
            "/performance_policy/historical_slowdown_explained",
            json!(true),
            "historical_slowdown_explained",
        ),
        (
            "/performance_policy/historical_slowdown_explained",
            Value::Null,
            "historical_slowdown_explained",
        ),
    ] {
        let mut changed = release.clone();
        *changed.pointer_mut(pointer).expect("fixture field exists") = value;
        let error = validate_records(&records, &changed).expect_err(pointer);
        assert!(error.contains(expected), "{pointer}: {error}");
    }
}

/// Reads one committed record below the acceptance directory.
fn committed_record(relative: &str) -> (Vec<u8>, Value) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("acceptance")
        .join(relative);
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let record = serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
    (bytes, record)
}

/// The re-baseline supersedes the release record's Qwen baseline, which keeps its bytes.
#[test]
fn committed_rebaseline_supersedes_the_unchanged_release_baseline() {
    let (release_bytes, release) = committed_record(RELEASE_RECORD);
    assert_eq!(
        format!("{:x}", Sha256::digest(&release_bytes)),
        RELEASE_RECORD_SHA256,
        "the superseded release record must keep its bytes"
    );
    validate_release(&release).expect("the release record stays valid");
    let (_, rebaseline) = committed_record(QWEN_REBASELINE_RECORD);
    validate_rebaseline(&rebaseline).expect("the committed re-baseline must be exact");
}

/// Creates a compact re-baseline fixture with placeholder medians, not measured results.
fn valid_rebaseline_fixture() -> Value {
    let reference = |metrics: &[&str]| {
        json!({
            "gated": metrics.iter().map(|metric| ((*metric).to_owned(), json!({"median": 1.0})))
                .collect::<serde_json::Map<_, _>>(),
            "diagnostic": {"cold_first_token": {"gated": false, "median": 0.5}}
        })
    };
    json!({
        "schema_version": 1, "advertised_supported": false, "support_claims_changed": false,
        "repository": QWEN.0, "revision": QWEN.1, "profile": PROFILE_IDS[0],
        "performance_policy": {
            "user_authorized_rebaseline": true, "authorization_date": "2026-10-18",
            "quoted_reason": REBASELINE_REASON, "historical_measurements_modified": false,
            "future_alert_percent": 5, "future_failure_percent_after_controlled_rerun": 10,
            "cold_first_token_gated": false, "diagnostic_metrics": ["cold_first_token"],
            "cold_first_token_reported_as": ["median", "range"],
            "gated_metrics": {"baseline": BASELINE_GATED, "public": PUBLIC_GATED},
            "supersedes": {
                "record": "v0_1-release-20260910.json", "record_sha256": RELEASE_RECORD_SHA256,
                "qwen_model_loop_baseline": QWEN_BASELINE,
                "qwen_model_loop_baseline_sha256": QWEN_BASELINE_SHA256
            }
        },
        "reference": {"label": "head", "benchmarks": {
            "baseline": reference(&BASELINE_GATED), "public": reference(&PUBLIC_GATED)
        }}
    })
}

/// Mutations cannot drop the authorization, gate cold first token, or change what is superseded.
#[test]
fn rebaseline_rejects_mutated_decisions_and_gates() {
    let record = valid_rebaseline_fixture();
    validate_rebaseline(&record).expect("the synthetic re-baseline is valid");
    for (pointer, value, expected) in [
        ("/advertised_supported", json!(true), "advertised_supported"),
        (
            "/support_claims_changed",
            json!(true),
            "support_claims_changed",
        ),
        ("/profile", json!("qwen3.5-0.8b-metal-q4"), "profile"),
        (
            "/performance_policy/user_authorized_rebaseline",
            json!(false),
            "user_authorized_rebaseline",
        ),
        (
            "/performance_policy/user_authorized_rebaseline",
            json!("true"),
            "user_authorized_rebaseline",
        ),
        (
            "/performance_policy/authorization_date",
            json!("2026-10-19"),
            "authorization_date",
        ),
        (
            "/performance_policy/quoted_reason",
            json!("re-baseline"),
            "quoted_reason",
        ),
        (
            "/performance_policy/historical_measurements_modified",
            json!(true),
            "historical_measurements_modified",
        ),
        (
            "/performance_policy/future_alert_percent",
            json!(6),
            "future_alert_percent",
        ),
        (
            "/performance_policy/future_failure_percent_after_controlled_rerun",
            json!(15),
            "future_failure_percent_after_controlled_rerun",
        ),
        (
            "/performance_policy/cold_first_token_gated",
            json!(true),
            "cold_first_token_gated",
        ),
        (
            "/performance_policy/diagnostic_metrics",
            json!([]),
            "diagnostic_metrics",
        ),
        (
            "/performance_policy/cold_first_token_reported_as",
            json!(["median"]),
            "cold_first_token_reported_as",
        ),
        (
            "/performance_policy/supersedes/record_sha256",
            json!("0".repeat(64)),
            "record_sha256",
        ),
        (
            "/performance_policy/supersedes/qwen_model_loop_baseline",
            json!("unapproved-series.json"),
            "qwen_model_loop_baseline",
        ),
        (
            "/performance_policy/gated_metrics/public",
            json!(["strict_load", "natural_loop", "cold_first_token"]),
            "gated_metrics.public",
        ),
        (
            "/performance_policy/gated_metrics/baseline",
            json!(PUBLIC_GATED),
            "gated_metrics.baseline",
        ),
        ("/reference/label", json!("september"), "label"),
        (
            "/reference/benchmarks/baseline/gated/natural_loop/median",
            json!(0.0),
            "positive finite",
        ),
        (
            "/reference/benchmarks/baseline/gated/strict_load/median",
            Value::Null,
            "positive finite",
        ),
        (
            "/reference/benchmarks/public/gated",
            json!([]),
            "must be an object",
        ),
        (
            "/reference/benchmarks/public/diagnostic/cold_first_token/gated",
            json!(true),
            "must be ungated",
        ),
        (
            "/reference/benchmarks/baseline/diagnostic/cold_first_token/median",
            json!(-1.0),
            "positive finite",
        ),
    ] {
        let mut changed = record.clone();
        *changed.pointer_mut(pointer).expect("fixture field exists") = value;
        let error = validate_rebaseline(&changed).expect_err(pointer);
        assert!(error.contains(expected), "{pointer}: {error}");
    }
    let mut gated_cold = record.clone();
    gated_cold["reference"]["benchmarks"]["public"]["gated"]["cold_first_token"] =
        json!({"median": 0.5});
    assert!(
        validate_rebaseline(&gated_cold)
            .unwrap_err()
            .contains("gate exactly")
    );
    let mut missing = record.clone();
    missing["reference"]["benchmarks"]["baseline"]["gated"]
        .as_object_mut()
        .unwrap()
        .remove("source_512.prefill");
    assert!(
        validate_rebaseline(&missing)
            .unwrap_err()
            .contains("gate exactly")
    );
}
