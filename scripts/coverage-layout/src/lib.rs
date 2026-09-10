//! Conservative, source-only supplementary classification for coverage audits.
//!
//! This tool does not consume LLVM coverage, alter denominators, expand macros,
//! or decide whether a release gate passed. It evaluates configuration twice:
//! with `test = false` and `test = true`, leaving every other atom unknown.
//! Only syntax proven absent without `test` and potentially present with it is
//! a test-only candidate. Every reference to a physical source file contributes
//! a context; production or unresolved ownership in any context prevents a
//! physical line from being classified exclusively as test-only.
//! These are candidates within the parsed graph, not exclusion proofs: macros
//! can create additional references. Reports explicitly remain unqualified for
//! coverage exclusions until independently bound to complete compiler evidence.

#![deny(missing_docs)]

mod cfg;
mod scanner;
mod source;
mod test_audit;

pub use scanner::classify;
pub use source::{
    Classification, Context, Diagnostic, FileReport, LineRun, Position, Region, Report, SourceSpan,
};
pub use test_audit::{TestAudit, TestSite};
