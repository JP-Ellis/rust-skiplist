//! A skiplist is a probabilistic data structure for storing sorted elements
//! with `O(log n)` average-case performance for access, insertion, and
//! removal.
//!
//! # Understanding Skip Lists
//!
//! A skip list layers multiple linked lists on top of each other. The bottom
//! layer (level 0) contains every element. Each higher layer contains a
//! randomly chosen subset of the elements below it, with each node promoted
//! independently with a fixed probability (typically 0.5). A search starts at
//! the top layer and drops down whenever the next node would overshoot the
//! target, arriving at the answer in O(log n) expected steps.
//!
//! ```text
//! Level 3:  head ----------> [20] ---------------------------------> tail
//! Level 2:  head --> [10] -> [20] ---------> [50] -----------------> tail
//! Level 1:  head --> [10] -> [20] -> [30] -> [50] -> [70] ---------> tail
//! Level 0:  head --> [10] -> [20] -> [30] -> [40] -> [50] -> [70] -> tail
//! ```
//!
//! To look up 40: start at level 3 and skip past 20, then drop to level 2
//! (50 is too far), then level 1 (50 is still too far), then step forward
//! at level 0 to find 40. Only four comparisons, rather than the four
//! linear steps that a plain linked list would require.
//!
//! ## When to use a skip list
//!
//! Skip lists are a good fit when you need:
//!
//! - **Sorted sequences with duplicates** ([`OrderedSkipList`]): unlike
//!   `BTreeSet`, an ordered skip list happily holds multiple copies of the same
//!   value.
//! - **Rank-based access** ([`SkipList`]): random access by position in O(log
//!   n), like a `Vec` but with O(log n) insertion and removal anywhere.
//! - **A sorted map or set with a custom ordering** ([`SkipMap`], [`SkipSet`]):
//!   any comparison function can be plugged in via the [`Comparator`] trait,
//!   without needing the element type to implement [`Ord`].
//!
//! If all you need is a set or map with unique keys and the default `Ord`
//! ordering, the standard library's `BTreeSet` / `BTreeMap` will often be
//! faster in practice due to better cache locality. Skip lists shine when
//! duplicate-tolerant ordering, custom comparators, or rank queries matter.
//!
//! # Available Collections
//!
//! - [`SkipList`]: a sequence container with O(log n) positional access
//!   (elements need not be `Ord`).
//! - [`OrderedSkipList`]: a sorted sequence where duplicates are permitted.
//! - [`SkipSet`]: a sorted set (no duplicates).
//! - [`SkipMap`]: a sorted map from keys to values.
//!
//! # Comparators
//!
//! The ordered collections are parameterized over a [`Comparator`]. The
//! default is [`OrdComparator`], which uses the element's [`Ord`]
//! implementation. A custom ordering can be supplied via [`FnComparator`].
//!
//! ## The `partial-ord` feature
//!
//! By default this crate requires elements of ordered collections to implement
//! [`Ord`], which guarantees a total order. Some types only implement
//! [`PartialOrd`] because certain values are incomparable; the most notable
//! example is floating-point numbers (`f32`, `f64`), where `NaN` is
//! incomparable with everything, including itself.
//!
//! Enabling the `partial-ord` feature unlocks [`PartialOrdComparator`], which
//! delegates to [`PartialOrd`] and **panics** if a comparison returns `None`
//! (i.e. if an incomparable value is encountered).
//!
//! ```toml
//! # Cargo.toml
//! [dependencies]
//! skiplist = { version = "...", features = ["partial-ord"] }
//! ```
//!
//! Once enabled, pass [`PartialOrdComparator`] as the comparator type:
//!
//! ```rust
//! # #[cfg(feature = "partial-ord")] {
//! use skiplist::{OrderedSkipList, PartialOrdComparator};
//!
//! let mut scores: OrderedSkipList<f64, _, PartialOrdComparator> =
//!     OrderedSkipList::with_comparator(PartialOrdComparator);
//!
//! // Insert an odd number of values in arbitrary order.
//! for &v in &[4.5, 1.2, 8.3, 3.7, 6.1] {
//!     scores.insert(v);
//! }
//!
//! // The list is kept sorted, so the middle element is the median.
//! let median = scores.get_by_index(scores.len() / 2);
//! assert_eq!(median, Some(&4.5));
//! # }
//! ```
//!
//! ### Caveats
//!
//! - **`NaN` panics at runtime.** Inserting or looking up a `NaN` value will
//!   panic immediately. There is no compile-time protection against this. Only
//!   use `PartialOrdComparator` when you can guarantee that incomparable values
//!   will never reach the collection.
//! - **No `NaN`-equality guarantee.** Even if a `NaN` somehow entered the
//!   collection without panicking, the skip list's ordering invariants would be
//!   broken, leading to incorrect results. The panic is a safety net, not a
//!   correctness guarantee.
//! - **`f32`/`f64` do not implement `Ord`.** [`FnComparator`] with
//!   [`f64::total_cmp`] gives you the IEEE 754-2019 total order: finite values
//!   sort numerically, `-0.0` sorts before `+0.0`, and all `NaN` bit patterns
//!   sort after every finite value (and after `+∞`). This is a true total order
//!   with no panics.
//!
//!   ```rust
//!   use skiplist::{OrderedSkipList, FnComparator};
//!
//!   let mut list: OrderedSkipList<f64, _, _> =
//!       OrderedSkipList::with_comparator(FnComparator(f64::total_cmp));
//!
//!   list.insert(3.0_f64);
//!   list.insert(f64::NAN);
//!   list.insert(1.0_f64);
//!
//!   // Finite values sort numerically; NaN sorts last.
//!   assert_eq!(list.first(), Some(&1.0));
//!   assert!(list.last().is_some_and(|v| v.is_nan()));
//!   ```
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
//! # Examples
//!
//! ```rust
//! use skiplist::SkipMap;
//!
//! let mut map = SkipMap::new();
//! map.insert("one", 1);
//! map.insert("two", 2);
//! map.insert("three", 3);
//!
//! assert_eq!(map.get("two"), Some(&2));
//! assert_eq!(map.len(), 3);
//! ```

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
