//! Long-form explanation documents (enabled by the `docs` feature).
//!
//! This module is only compiled when the `docs` feature is active.  It is
//! automatically included on [docs.rs](https://docs.rs/skiplist).
//!
//! | Page | Contents |
//! |------|----------|
//! [`concepts`] | Skip list theory, collection taxonomy, comparator design, capacity formula |
//! [`cursor`] | Gap cursor model, `lower_bound`/`upper_bound` semantics, how-to guides (requires `cursor` feature) |
//! [`internals`] | Node ownership, pointer provenance, `NonNull`-over-`Box` rationale (for contributors) |
//! [`partial_ord`] | The `partial-ord` feature, NaN caveats, `total_cmp` alternative |
//! [`no_std`] | Features, seeding without std, constructor availability |
//!
#![cfg_attr(not(feature = "cursor"), doc = "[`cursor`]: crate#features")]
#![cfg_attr(not(feature = "partial-ord"), doc = "[`partial_ord`]: crate#features")]
#![expect(
    clippy::inline_modules,
    reason = "each page is an empty module whose docs come from include_str!"
)]

#[expect(
    missing_docs,
    reason = "content provided by include_str! when building docs"
)]
#[cfg_attr(doc, doc = include_str!("../docs/concepts.md"))]
#[cfg_attr(
    all(doc, feature = "partial-ord"),
    doc = "[`PartialOrdComparator`]: crate::PartialOrdComparator"
)]
#[cfg_attr(
    all(doc, not(feature = "partial-ord")),
    doc = "[`PartialOrdComparator`]: crate#features"
)]
#[cfg_attr(
    all(doc, not(feature = "partial-ord")),
    doc = "[`crate::docs::partial_ord`]: crate#features"
)]
pub mod concepts {}

#[expect(
    missing_docs,
    reason = "content provided by include_str! when building docs"
)]
#[cfg_attr(doc, doc = include_str!("../docs/internals.md"))]
pub mod internals {}

#[cfg(feature = "cursor")]
#[expect(
    missing_docs,
    reason = "content provided by include_str! when building docs"
)]
#[cfg_attr(doc, doc = include_str!("../docs/cursor.md"))]
pub mod cursor {}

#[cfg(feature = "partial-ord")]
#[expect(
    missing_docs,
    reason = "content provided by include_str! when building docs"
)]
#[cfg_attr(doc, doc = include_str!("../docs/partial_ord.md"))]
pub mod partial_ord {}

#[expect(
    missing_docs,
    reason = "content provided by include_str! when building docs"
)]
#[cfg_attr(doc, doc = include_str!("../docs/no_std.md"))]
pub mod no_std {}
