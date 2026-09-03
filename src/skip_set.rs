//! A skip-list-backed ordered set.
//!
//! [`SkipSet`] keeps each element **at most once**, sorted according to a
//! [`Comparator<T>`].  Insert, lookup, and remove are `$O(\log n)$` on average.
//! It wraps [`OrderedSkipList`] and adds uniqueness enforcement plus
//! set-algebra operations (union, intersection, difference, symmetric
//! difference).
//!
//! The ordering is parameterised by `C: Comparator<T>` so that custom
//! orderings can be used without requiring [`Ord`] on the element type.
//!
//! # Key Invariants
//!
//! - Each value is stored at most once.  Inserting a value that already exists
//!   is a no-op (`insert` returns `false`).
//! - Values are always kept in sorted order.
//! - Values cannot be mutated in place because that could silently break the
//!   sorted invariant.  To update a value, [`take`] it and re-`insert` the new
//!   one.
//!
//! # Intentional Omissions
//!
//! - **No `IterMut`.**  Mutable references to stored values could break the
//!   sorted invariant without reinsertion.
//!
//! # Method Summary
//!
//! **Constructors:** [`new`], [`with_level_generator`], [`with_comparator`],
//!   [`with_comparator_and_level_generator`].
//!
//! **Access:** [`contains`], [`get`], [`get_by_index`], [`first`], [`last`],
//!   [`rank`].
//!
//! **Insertion:** [`insert`], [`replace`], [`get_or_insert`],
//!   [`get_or_insert_with`].
//!
//! **Removal:** [`remove`], [`take`], [`pop_first`], [`pop_last`], [`retain`],
//!   [`drain`], [`extract_if`].
//!
//! **Set operations:** [`union`], [`intersection`], [`difference`],
//!   [`symmetric_difference`], [`is_subset`], [`is_superset`],
//!   [`is_disjoint`].
//!
//! **Structural:** [`len`], [`is_empty`], [`clear`], [`split_off`],
//!   [`split_off_index`], [`append`].
//!
//! **Iteration:** [`iter`], [`into_iter`].
//!
//! # Examples
//!
//! ```rust
//! use skiplist::SkipSet;
//!
//! let mut set = SkipSet::new();
//!
//! // insert returns true for new elements, false for duplicates.
//! assert!(set.insert(30));
//! assert!(set.insert(10));
//! assert!(set.insert(20));
//! assert!(!set.insert(10)); // duplicate; no-op
//!
//! assert_eq!(set.len(), 3);
//! assert_eq!(set.first(), Some(&10));
//!
//! // Iteration is in sorted order.
//! let values: Vec<_> = set.iter().copied().collect();
//! assert_eq!(values, [10, 20, 30]);
//!
//! // Set operations.
//! let other: SkipSet<i32> = [20, 30, 40].into_iter().collect();
//! let union: Vec<_> = set.union(&other).copied().collect();
//! assert_eq!(union, [10, 20, 30, 40]);
//! ```
//!
//! [`OrderedSkipList`]: crate::OrderedSkipList
//! [`Comparator<T>`]: crate::comparator::Comparator
//! [`new`]: SkipSet::new
//! [`with_level_generator`]: SkipSet::with_level_generator
//! [`with_comparator`]: SkipSet::with_comparator
//! [`with_comparator_and_level_generator`]: SkipSet::with_comparator_and_level_generator
//! [`contains`]: SkipSet::contains
//! [`get`]: SkipSet::get
//! [`get_by_index`]: SkipSet::get_by_index
//! [`first`]: SkipSet::first
//! [`last`]: SkipSet::last
//! [`rank`]: SkipSet::rank
//! [`insert`]: SkipSet::insert
//! [`replace`]: SkipSet::replace
//! [`get_or_insert`]: SkipSet::get_or_insert
//! [`get_or_insert_with`]: SkipSet::get_or_insert_with
//! [`remove`]: SkipSet::remove
//! [`take`]: SkipSet::take
//! [`pop_first`]: SkipSet::pop_first
//! [`pop_last`]: SkipSet::pop_last
//! [`retain`]: SkipSet::retain
//! [`drain`]: SkipSet::drain
//! [`extract_if`]: SkipSet::extract_if
//! [`union`]: SkipSet::union
//! [`intersection`]: SkipSet::intersection
//! [`difference`]: SkipSet::difference
//! [`symmetric_difference`]: SkipSet::symmetric_difference
//! [`is_subset`]: SkipSet::is_subset
//! [`is_superset`]: SkipSet::is_superset
//! [`is_disjoint`]: SkipSet::is_disjoint
//! [`len`]: SkipSet::len
//! [`is_empty`]: SkipSet::is_empty
//! [`clear`]: SkipSet::clear
//! [`split_off`]: SkipSet::split_off
//! [`split_off_index`]: SkipSet::split_off_index
//! [`append`]: SkipSet::append
//! [`iter`]: SkipSet::iter
//! [`into_iter`]: SkipSet::into_iter

use crate::{
    comparator::{Comparator, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    ordered_skip_list::OrderedSkipList,
};

mod access;
#[cfg(feature = "cursor")]
pub mod cursor;
#[cfg(feature = "cursor")]
pub use cursor::{Cursor, CursorMut, UnorderedValueError};
mod entry;
mod filter;
mod insert_remove;
mod iter;
mod ops;
mod set_ops;
mod structural;
mod traits;

pub use entry::{Entry, OccupiedEntry, VacantEntry};
pub use iter::{Drain, ExtractIf, IntoIter, Iter};
pub use set_ops::{Difference, Intersection, SymmetricDifference, Union};

/// An ordered set that stores each element at most once.
///
/// `SkipSet<T, N, C, G>` keeps its elements sorted according to the total
/// order defined by `C: Comparator<T>` and rejects duplicates (elements
/// comparing `Equal`).  Insert, lookup, and remove are `$O(\log n)$` on average.
///
/// The const generic `N` (default `16`) sets the maximum number of levels
/// used internally; increase it when you expect more than roughly `$2^N$`
/// elements.  `G` controls how levels are chosen for new elements; the
/// default ([`Geometric`]) works well in practice.
///
/// # Constructors
///
/// | Constructor                                   | `T: Ord`? | Comparator        | Generator       |
/// |-----------------------------------------------|:---------:|:-----------------:|:---------------:|
/// | [`new()`]                                     | required  | [`OrdComparator`] | [`Geometric`]   |
/// | [`with_level_generator(g)`]                   | required  | [`OrdComparator`] | `g`             |
/// | [`with_comparator(c)`]                        | not req.  | `c`               | [`Geometric`]   |
/// | [`with_comparator_and_level_generator(c, g)`] | not req.  | `c`               | `g`             |
///
/// [`new()`]: SkipSet::new
/// [`with_level_generator(g)`]: SkipSet::with_level_generator
/// [`with_comparator(c)`]: SkipSet::with_comparator
/// [`with_comparator_and_level_generator(c, g)`]: SkipSet::with_comparator_and_level_generator
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::SkipSet;
///
/// let set = SkipSet::<u32>::new();
/// assert!(set.is_empty());
/// ```
pub struct SkipSet<
    T,
    const N: usize = 16,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// The underlying ordered skip list that stores the elements and maintains
    /// the sorted invariant.
    inner: OrderedSkipList<T, N, C, G>,
}

// MARK: Constructors (OrdComparator, default level generator)

impl<T: Ord, const N: usize> SkipSet<T, N, OrdComparator, Geometric> {
    /// Creates an empty skip set using the natural [`Ord`] ordering and the
    /// default level generator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set = SkipSet::<i32>::new();
    /// assert!(set.is_empty());
    /// assert_eq!(set.len(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::with_comparator_and_level_generator(OrdComparator, Geometric::default())
    }
}

// MARK: Constructors (OrdComparator, custom level generator)

impl<T: Ord, G: LevelGenerator, const N: usize> SkipSet<T, N, OrdComparator, G> {
    /// Creates an empty skip set using the natural [`Ord`] ordering and the
    /// supplied level generator.
    ///
    /// Use this when you need precise control over the level distribution
    /// (e.g., a different probability or level count than the defaults).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use skiplist::level_generator::geometric::Geometric;
    ///
    /// let g = Geometric::new(8, 0.5).expect("valid parameters");
    /// let set = SkipSet::<i32>::with_level_generator(g);
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_level_generator(generator: G) -> Self {
        Self::with_comparator_and_level_generator(OrdComparator, generator)
    }
}

// MARK: Constructors (custom comparator, default level generator)

impl<T, C: Comparator<T>, const N: usize> SkipSet<T, N, C, Geometric> {
    /// Creates an empty skip set with the supplied comparator and the default
    /// level generator.
    ///
    /// Use this when you need a custom ordering without implementing [`Ord`] on
    /// the element type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use skiplist::comparator::FnComparator;
    ///
    /// // Largest-first ordering.
    /// let set: SkipSet<i32, 16, _> =
    ///     SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_comparator(comparator: C) -> Self {
        Self::with_comparator_and_level_generator(comparator, Geometric::default())
    }
}

// MARK: Generic methods available for any C + G

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> SkipSet<T, N, C, G> {
    /// Creates an empty skip set with the supplied comparator and level
    /// generator.
    ///
    /// This is the base constructor; all other constructors delegate to it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use skiplist::comparator::FnComparator;
    /// use skiplist::level_generator::geometric::Geometric;
    ///
    /// let g = Geometric::new(8, 0.25).expect("valid parameters");
    /// let set: SkipSet<i32, 8, _> =
    ///     SkipSet::with_comparator_and_level_generator(
    ///         FnComparator(|a: &i32, b: &i32| b.cmp(a)),
    ///         g,
    ///     );
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_comparator_and_level_generator(comparator: C, generator: G) -> Self {
        Self {
            inner: OrderedSkipList::with_comparator_and_level_generator(comparator, generator),
        }
    }

    /// Returns the number of elements in the set.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set = SkipSet::<i32>::new();
    /// assert_eq!(set.len(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the set contains no elements.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set = SkipSet::<i32>::new();
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// MARK: Default

impl<T: Ord, const N: usize> Default for SkipSet<T, N, OrdComparator, Geometric> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::SkipSet;
    use crate::{comparator::FnComparator, level_generator::geometric::Geometric};

    // MARK: new

    #[test]
    fn new_is_empty() {
        let set = SkipSet::<i32>::new();
        assert!(set.is_empty());
    }

    #[test]
    fn new_len_zero() {
        let set = SkipSet::<i32>::new();
        assert_eq!(set.len(), 0);
    }

    // MARK: with_level_generator

    #[test]
    fn with_level_generator_is_empty() {
        let g = Geometric::new(8, 0.5).expect("valid parameters");
        let set = SkipSet::<i32>::with_level_generator(g);
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn with_level_generator_custom_params() {
        let g = Geometric::new(4, 0.25).expect("valid parameters");
        let set = SkipSet::<String>::with_level_generator(g);
        assert!(set.is_empty());
    }

    // MARK: with_comparator

    #[test]
    fn with_comparator_is_empty() {
        let set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    // MARK: with_comparator_and_level_generator

    #[test]
    fn with_comparator_and_level_generator_is_empty() {
        let g = Geometric::new(8, 0.25).expect("valid parameters");
        let set: SkipSet<i32, 8, _> = SkipSet::with_comparator_and_level_generator(
            FnComparator(|a: &i32, b: &i32| b.cmp(a)),
            g,
        );
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    // MARK: default

    #[test]
    fn default_is_empty() {
        let set = SkipSet::<i32>::default();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }
}
