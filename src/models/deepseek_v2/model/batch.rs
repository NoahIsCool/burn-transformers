//! Former home of the batched-row host plan, now [`crate::models::shared::block_plan`].
//!
//! The plan is family-independent (DeepSeek-V2 and Muse Glimmer share it), so
//! it lives with the other shared model glue and is verified by
//! `verus/models.rs`. This module only re-exports the two helpers DeepSeek's
//! decode path names through its old path.

pub(crate) use crate::models::shared::block_plan::{first_uncacheable_context, rotary_contexts};
