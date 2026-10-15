//! Allocation-backed identities for runtime cache objects.
//!
//! An [`IdentityToken`] is a shared, payload-free allocation, and equality
//! compares its address. While any handle is alive the allocator cannot reuse
//! that address, so a token equals only its own clones: never a token created
//! later, nor one owned by another cache or decode state. This needs no global
//! counter and cannot wrap. Objects that hand out derived values — read plans,
//! step write sets — stamp them with their token so a value presented to the
//! wrong object is rejected instead of silently reinterpreted.

use core::fmt;
use std::sync::Arc;

/// Unique identity of one cache object, shared by every value it stamps.
#[derive(Clone)]
pub(crate) struct IdentityToken(
    /// Address identity shared by cloned handles; the payload carries no state.
    Arc<()>,
);

impl IdentityToken {
    /// Allocates an identity distinct from every other live token.
    pub(crate) fn new() -> Self {
        Self(Arc::new(()))
    }
}

impl fmt::Debug for IdentityToken {
    /// Formats the token without exposing its process-specific address.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("IdentityToken(..)")
    }
}

impl PartialEq for IdentityToken {
    /// Compares allocation identity: only clones of one token are equal.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for IdentityToken {}

#[cfg(test)]
#[path = "identity_tests.rs"]
mod tests;
