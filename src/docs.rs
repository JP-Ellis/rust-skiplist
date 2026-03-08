//! Long-form explanation documents (enabled by the `docs` feature).
//!
//! This module is only compiled when the `docs` feature is active.  It is
//! automatically included on [docs.rs](https://docs.rs/skiplist).
//!
//! | Page | Contents |
//! |------|----------|
//! [`concepts`] | Skip list theory, collection taxonomy, comparator design, capacity formula |
//! [`internals`] | Node ownership, pointer provenance, `NonNull`-over-`Box` rationale (for contributors) |
//! [`partial_ord`] | The `partial-ord` feature, NaN caveats, `total_cmp` alternative |

#[expect(
    missing_docs,
    reason = "content provided by include_str! when building docs"
)]
#[cfg_attr(doc, doc = include_str!("../docs/concepts.md"))]
pub mod concepts {}

#[expect(
    missing_docs,
    reason = "content provided by include_str! when building docs"
)]
#[cfg_attr(doc, doc = include_str!("../docs/internals.md"))]
pub mod internals {}

#[cfg(feature = "partial-ord")]
#[expect(
    missing_docs,
    reason = "content provided by include_str! when building docs"
)]
#[cfg_attr(doc, doc = include_str!("../docs/partial_ord.md"))]
pub mod partial_ord {}
