//! Former home of the ragged-row block layout, now [`crate::models::shared::block`].
//!
//! The layout is family-independent (DeepSeek-V2 and Muse Glimmer share it),
//! so it lives with the other shared model glue. This module only re-exports
//! [`BlockLayout`](crate::models::shared::block::BlockLayout) for DeepSeek's
//! decode path, which names it through its old path.

pub(crate) use crate::models::shared::block::BlockLayout;
