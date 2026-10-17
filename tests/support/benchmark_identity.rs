//! Source, toolchain and host identity recorded in every benchmark report.
//!
//! A timing is evidence only when its report names what produced it. Each
//! report built with [`source_identity`] carries three kinds of identity:
//!
//! - **Compiled source**: the SHA-256 of every harness source compiled into the
//!   executable (taken with `include_bytes!`, so it describes the binary, not the
//!   checkout), the executable's own SHA-256, the crate version, enabled Cargo
//!   features, build profile, and the Rust toolchain the binary was built with.
//! - **Build tree**: `git rev-parse HEAD` and `git status --porcelain` run at
//!   report time in `CARGO_MANIFEST_DIR`, the tree the binary was compiled
//!   from. When that tree has moved on since the build (for example a base
//!   revision built once and timed later, interleaved with a head build), these
//!   describe the later tree. The optional `TRANSFORMERS_BENCH_SOURCE_LABEL`
//!   lets the operator name the binary's source revision explicitly, and each
//!   compiled source records whether it still matches the build tree.
//! - **Host**: OS, architecture, available parallelism, and the one-, five- and
//!   fifteen-minute load averages before and after timing, plus the optional
//!   operator-supplied `RunEnvironment` record named by
//!   `TRANSFORMERS_BENCH_METADATA_JSON` (validated when present).
//!
//! Metadata commands never abort a run: a failing probe is preserved as
//! `{"value": null, "error": ...}` instead of being guessed.

use std::{fs, io::Read, path::Path, process::Command};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::benchmark_metrics::RunEnvironment;

/// Burn version pinned by `Cargo.toml` and the committed lockfile.
const BURN_VERSION: &str = "0.21.0";

/// Streams one regular file's SHA-256 without staging it in memory.
///
/// # Panics
///
/// Panics with the path when the file cannot be opened or read, because a
/// report must not silently omit the identity of what it measured.
pub fn file_sha256(path: &Path) -> String {
    let mut file =
        fs::File::open(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    format!("{:x}", digest.finalize())
}

/// Runs a small metadata command in `directory` and preserves failure as evidence.
///
/// Returns `{"value": trimmed stdout, "error": null}` on success and
/// `{"value": null, "error": message}` when the command fails or cannot start.
pub fn command_text(program: &str, arguments: &[&str], directory: &Path) -> Value {
    match Command::new(program)
        .args(arguments)
        .current_dir(directory)
        .output()
    {
        Ok(output) if output.status.success() => {
            json!({"value": String::from_utf8_lossy(&output.stdout).trim(), "error": null})
        }
        Ok(output) => {
            json!({"value": null, "error": String::from_utf8_lossy(&output.stderr).trim()})
        }
        Err(error) => json!({"value": null, "error": error.to_string()}),
    }
}

/// Records every runtime-affecting crate feature so incompatible series cannot be merged.
pub fn cargo_features() -> Vec<&'static str> {
    [
        ("std", cfg!(feature = "std")),
        ("flex", cfg!(feature = "flex")),
        ("metal", cfg!(feature = "metal")),
        ("cpu", cfg!(feature = "cpu")),
        ("cuda", cfg!(feature = "cuda")),
        ("rocm", cfg!(feature = "rocm")),
        ("autodiff", cfg!(feature = "autodiff")),
        ("tokenizers", cfg!(feature = "tokenizers")),
        ("chat-templates", cfg!(feature = "chat-templates")),
        ("kernel-tracing", cfg!(feature = "kernel-tracing")),
        (
            "attention-kernel-tracing",
            cfg!(feature = "attention-kernel-tracing"),
        ),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect()
}

/// Returns the verbose identity of the toolchain this executable was compiled with.
///
/// The second value is the `RUSTUP_TOOLCHAIN` captured at compile time. Without
/// it the compiler cannot be named reliably (the `rustc` on `PATH` at run time
/// may differ), so the first value records that as an explicit error.
pub fn compiler_identity(directory: &Path) -> (Value, Option<&'static str>) {
    let toolchain = option_env!("RUSTUP_TOOLCHAIN");
    let compiler = match toolchain {
        Some(toolchain) => command_text(
            "rustc",
            &[&format!("+{toolchain}"), "--version", "--verbose"],
            directory,
        ),
        None => {
            json!({"value": null, "error": "RUSTUP_TOOLCHAIN was not present when this executable was compiled"})
        }
    };
    (compiler, toolchain)
}

/// Samples the host's one-, five- and fifteen-minute load averages.
///
/// Uses `sysctl -n vm.loadavg` on macOS and `/proc/loadavg` elsewhere; the raw
/// text is kept because the two formats differ.
pub fn host_load() -> Value {
    if cfg!(target_os = "macos") {
        command_text("sysctl", &["-n", "vm.loadavg"], Path::new("/"))
    } else {
        match fs::read_to_string("/proc/loadavg") {
            Ok(text) => json!({"value": text.trim(), "error": null}),
            Err(error) => json!({"value": null, "error": error.to_string()}),
        }
    }
}

/// Reads the optional operator environment record named by `TRANSFORMERS_BENCH_METADATA_JSON`.
///
/// # Panics
///
/// Panics when the variable is set but the file is unreadable, malformed, or
/// incomplete, so a mislabeled run fails before it is timed.
fn operator_environment() -> Value {
    let Some(path) = std::env::var_os("TRANSFORMERS_BENCH_METADATA_JSON") else {
        return Value::Null;
    };
    let bytes = fs::read(&path).expect("read TRANSFORMERS_BENCH_METADATA_JSON");
    let environment: RunEnvironment =
        serde_json::from_slice(&bytes).expect("complete benchmark environment JSON");
    environment.validate().expect("valid benchmark environment");
    serde_json::to_value(environment).expect("serialize benchmark environment")
}

/// Builds the provenance record for a report; see the module documentation.
///
/// `compiled_sources` pairs each harness source's repository-relative path with
/// the bytes `include_bytes!` compiled into this executable. Call it before
/// timing so the recorded load average describes the run's start.
///
/// # Panics
///
/// Panics when `TRANSFORMERS_BENCH_SOURCE_LABEL` is set but blank, when the
/// operator environment record is invalid, or when the executable or lockfile
/// cannot be hashed.
pub fn source_identity(compiled_sources: &[(&str, &[u8])]) -> Value {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source_label = std::env::var("TRANSFORMERS_BENCH_SOURCE_LABEL").ok();
    if let Some(label) = &source_label {
        assert!(
            !label.trim().is_empty(),
            "TRANSFORMERS_BENCH_SOURCE_LABEL must name a source revision when set"
        );
    }
    let sources = compiled_sources
        .iter()
        .map(|(path, bytes)| {
            let compiled = format!("{:x}", Sha256::digest(bytes));
            let on_disk = fs::read(manifest_dir.join(path))
                .ok()
                .map(|disk| format!("{:x}", Sha256::digest(disk)));
            json!({
                "path": path,
                "compiled_sha256": compiled,
                "matches_build_tree": on_disk.as_deref() == Some(compiled.as_str()),
            })
        })
        .collect::<Vec<_>>();
    let (compiler, toolchain) = compiler_identity(manifest_dir);
    json!({
        "source_label": source_label,
        "build_tree": manifest_dir.display().to_string(),
        "crate_commit": command_text("git", &["rev-parse", "HEAD"], manifest_dir),
        "crate_worktree_status": command_text(
            "git",
            &["status", "--porcelain", "--untracked-files=all", "--", "."],
            manifest_dir,
        ),
        "git_queries_describe": "the build tree at report time, which can be newer than the binary; source_label and compiled_sources identify the binary",
        "compiled_sources": sources,
        "benchmark_executable_sha256": file_sha256(
            &std::env::current_exe().expect("benchmark executable path"),
        ),
        "cargo_lock_sha256": file_sha256(&manifest_dir.join("Cargo.lock")),
        "crate_version": env!("CARGO_PKG_VERSION"),
        "burn_version": BURN_VERSION,
        "rustc": compiler,
        "compiled_rustup_toolchain": toolchain,
        "profile": if cfg!(debug_assertions) {
            "debug assertions enabled (not a timing profile)"
        } else {
            "release (debug assertions disabled)"
        },
        "cargo_features": cargo_features(),
        "host": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "available_parallelism": std::thread::available_parallelism()
                .map(usize::from)
                .ok(),
            "load_average_before": host_load(),
        },
        "operator_environment": operator_environment(),
    })
}

#[cfg(test)]
mod tests {
    //! Fast checks that reports name their sources without timing anything.

    use std::path::Path;

    use sha2::{Digest, Sha256};

    use super::{cargo_features, command_text, file_sha256, host_load, source_identity};

    /// Holds the shared child-process lock while a test spawns metadata probes.
    ///
    /// On macOS a spawned child can hold another test's dropped socket open;
    /// see `benchmark_metrics::CHILD_PROCESS_LOCK`. The guard is an `Option` so
    /// callers bind it the same way on every platform.
    #[cfg(target_os = "macos")]
    fn spawning_children() -> Option<std::sync::MutexGuard<'static, ()>> {
        Some(
            crate::benchmark_metrics::CHILD_PROCESS_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// The only socket test that needs the lock runs on macOS alone, so no guard is held.
    #[cfg(not(target_os = "macos"))]
    fn spawning_children() -> Option<std::sync::MutexGuard<'static, ()>> {
        None
    }

    /// A compiled source is identified by the hash of the bytes in the binary,
    /// matched against the build tree, alongside toolchain, features and host.
    #[test]
    fn source_identity_names_compiled_sources_toolchain_and_host() {
        let _children = spawning_children();
        let bytes: &[u8] = include_bytes!("benchmark_identity.rs");
        let identity = source_identity(&[("tests/support/benchmark_identity.rs", bytes)]);
        let source = &identity["compiled_sources"][0];
        assert_eq!(source["path"], "tests/support/benchmark_identity.rs");
        assert_eq!(
            source["compiled_sha256"],
            format!("{:x}", Sha256::digest(bytes))
        );
        assert_eq!(source["matches_build_tree"], true);
        assert_eq!(
            identity["cargo_features"],
            serde_json::to_value(cargo_features()).unwrap()
        );
        assert!(identity["host"]["available_parallelism"].as_u64().unwrap() >= 1);
        assert_eq!(identity["burn_version"], "0.21.0");
        assert_eq!(
            identity["benchmark_executable_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        // Git may be unavailable on a build host; either outcome is explicit.
        let commit = &identity["crate_commit"];
        assert!(commit["value"].is_string() != commit["error"].is_string());
    }

    /// A source that differs from the build tree is reported, not rejected.
    #[test]
    fn edited_sources_are_flagged_against_the_build_tree() {
        let _children = spawning_children();
        let identity = source_identity(&[("Cargo.toml", b"not the manifest".as_slice())]);
        assert_eq!(identity["compiled_sources"][0]["matches_build_tree"], false);
        let missing = source_identity(&[("tests/support/missing.rs", b"".as_slice())]);
        assert_eq!(missing["compiled_sources"][0]["matches_build_tree"], false);
    }

    /// Failing metadata probes are preserved as explicit errors.
    #[test]
    fn failing_probes_are_recorded_as_errors() {
        let _children = spawning_children();
        let missing = command_text(
            "transformers-benchmark-probe-that-does-not-exist",
            &[],
            Path::new("/"),
        );
        assert!(missing["value"].is_null());
        assert!(missing["error"].is_string());
        let failing = command_text("git", &["definitely-not-a-git-command"], Path::new("/"));
        assert!(failing["value"].is_null());
        let load = host_load();
        assert!(load["value"].is_string() != load["error"].is_string());
    }

    /// File hashes stream the exact bytes on disk.
    #[test]
    fn file_hashes_match_in_memory_digests() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(file_sha256(&path), format!("{:x}", Sha256::digest(bytes)));
    }
}
