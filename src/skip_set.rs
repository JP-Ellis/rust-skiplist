//! A skip-list-backed ordered set.
//!
//! This module provides [`SkipSet`], an ordered set that stores each element
//! at most once (no duplicates).  Insert, lookup, and remove are O(log n) on
//! average.
//!
//! The ordering is parameterised by a [`Comparator<T>`] so that custom
//! orderings can be used without requiring [`Ord`] on the element type.
//!
//! # Example
//!
//! ```rust
//! use skiplist::skip_set::SkipSet;
//!
//! let set = SkipSet::<i32>::new();
//! assert!(set.is_empty());
//! assert_eq!(set.len(), 0);
//! ```
//!
//! [`Comparator<T>`]: crate::comparator::Comparator

use crate::{
    comparator::{Comparator, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    ordered_skip_list::OrderedSkipList,
};

/// An ordered set backed by a skip list.
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
