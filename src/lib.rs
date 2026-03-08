//! Skip list collections with `$O(\log n)$` average-case performance for access,
//! insertion, and removal.  Four variants cover the common use cases:
//! positional sequences, sorted bags, sorted sets, and sorted maps, all with
//! pluggable ordering via the [`Comparator`] trait.
//!
//! # Collections
//!
//! | Collection | Ordering | Duplicates | Primary use case |
//! |---|---|---|---|
//! | [`SkipList`] | Insertion order | Yes | Positional sequence with `$O(\log n)$` insert/remove/access anywhere |
//! | [`OrderedSkipList`] | Sorted | Yes | Sorted bag (multiple equal values kept in order) |
//! | [`SkipSet`] | Sorted | No | Sorted set (each value at most once) |
//! | [`SkipMap`] | Sorted by key | No (unique keys) | Sorted key-value map |
//!
//! # Comparators
//!
//! Ordered collections are parameterised over a [`Comparator`]:
//!
//! - [`OrdComparator`]: uses `T: Ord` (the default; zero overhead).
//! - [`FnComparator`]: wraps any `fn(&T, &T) -> Ordering` or compatible
//!   closure; use this for custom orderings without a new type.
//! - [`PartialOrdComparator`]: uses `T: PartialOrd`; panics on incomparable
//!   values.  Available only with the `partial-ord` feature.
//!
//! For design rationale see [`docs::concepts`].
//!
//! ## The `partial-ord` feature
//!
//! Enable `partial-ord` to unlock [`PartialOrdComparator`]:
//!
//! ```toml
//! [dependencies]
//! skiplist = { version = "...", features = ["partial-ord"] }
//! ```
//!
//! **Caution:** inserting or looking up a `NaN` value will panic at runtime.
//! For floating-point keys, prefer [`FnComparator`] with [`f64::total_cmp`]
//! which provides a true total order with no panics.
//! See [`docs::partial_ord`] for full guidance.
//!
//! # Ordering Requirements
//!
//! The ordered collections rely on a well-behaved comparison function.
//! Specifically, given some ordering function `$f(a, b)$`, it **must** satisfy
//! the following properties:
//!
//! - Be well defined: `$f(a, b)$` should always return the same value.
//! - Be anti-symmetric: `$f(a, b) = \text{Greater}$` if and only if `$f(b, a) = \text{Less}$`,
//!   and `$f(a, b) = \text{Equal} = f(b, a)$`.
//! - Be transitive: if `$f(a, b) = \text{Greater}$` and `$f(b, c) = \text{Greater}$` then
//!   `$f(a, c) = \text{Greater}$`.
//!
//! **Failure to satisfy these properties can result in unexpected behavior at
//! best, and at worst will cause a segfault, null deref, or some other bad
//! behavior.**
//!
//! # Further Reading
//!
//! The [`docs`] module (enabled by the `docs` feature) contains long-form
//! explanation pages:
//!
//! - [`docs::concepts`]: skip list theory, collection taxonomy, comparator
//!   design, and the capacity / levels formula.
//! - [`docs::internals`]: node ownership, pointer provenance, and the
//!   `NonNull`-over-`Box` rationale for contributors.
//! - [`docs::partial_ord`]: the `partial-ord` feature, NaN caveats, and the
//!   `FnComparator(f64::total_cmp)` alternative.

// Internal terminology used throughout this crate:
//   - "height" of a node: how many forward links it holds (minimum 1).
//   - "level 0": the bottom-most layer, which contains every element.
//   - "level k" (k > 0): a sparser layer whose nodes also appear at level k-1.
// Higher-level links allow the search algorithm to skip over many nodes at
// once, yielding `$O(\log n)$` average traversal cost.

#![expect(
    clippy::pub_use,
    reason = "creating the main public API from private modules"
)]

pub mod comparator;
#[cfg(feature = "docs")]
pub mod docs;
pub mod level_generator;
mod node;
pub mod ordered_skip_list;
pub mod skip_list;
pub mod skip_map;
pub mod skip_set;

#[cfg(feature = "partial-ord")]
pub use comparator::PartialOrdComparator;
pub use comparator::{Comparator, FnComparator, OrdComparator};
pub use ordered_skip_list::OrderedSkipList;
pub use skip_list::SkipList;
pub use skip_map::SkipMap;
pub use skip_set::SkipSet;

/// Convenience re-exports for glob import.
///
/// ```rust
/// use skiplist::prelude::*;
/// ```
pub mod prelude {
    #[cfg(feature = "partial-ord")]
    pub use crate::PartialOrdComparator;
    pub use crate::{
        Comparator, FnComparator, OrdComparator, OrderedSkipList, SkipList, SkipMap, SkipSet,
    };
}
