//! Index-based skip list.
//!
//! This module provides [`SkipList<T, G>`], a general-purpose sequence with
//! O(log n) insert, remove, and random access by index.  It is a useful
//! alternative to [`Vec`] when elements are frequently inserted or removed in
//! the middle of the list while indexed access is still needed.
//!
//! # Example
//!
//! ```rust
//! use skiplist::skip_list::SkipList;
//!
//! let list = SkipList::<i32>::new();
//! assert!(list.is_empty());
//! assert_eq!(list.len(), 0);
//! ```

use crate::level_generator::{LevelGenerator, geometric::Geometric};
use crate::node::Node;

/// Default number of levels used by [`SkipList::new`] and
/// [`SkipList::with_capacity`].
const DEFAULT_LEVELS: usize = 16;

/// Default promotion probability used by [`SkipList::new`] and
/// [`SkipList::with_capacity`].
const DEFAULT_P: f64 = 0.5;

/// An index-based skip list.
///
/// `SkipList<T, G>` stores elements in insertion order and provides O(log n)
/// insert, remove, and indexed access.  Unlike [`Vec`], inserting or removing
/// in the middle does not shift elements; unlike a plain linked list,
/// arbitrary-index access is O(log n) rather than O(n).
///
/// The generic parameter `G` controls how node tower heights are chosen.  The
/// default ([`Geometric`]) works well in practice; supply a custom
/// [`LevelGenerator`] via [`SkipList::with_level_generator`] if you need
/// different behaviour.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_list::SkipList;
///
/// let list = SkipList::<u32>::new();
/// assert!(list.is_empty());
/// ```
pub struct SkipList<T, G: LevelGenerator = Geometric> {
    /// Sentinel head node. Never holds a value; its `links` array has length
    /// equal to the maximum number of levels.
    head: Box<Node<T>>,
    /// Cached element count. Updated by every insert / remove operation.
    len: usize,
    /// Level generator used to determine the tower height of each new node.
    generator: G,
}

// ── Constructors that require the default Geometric generator ────────────────

impl<T> SkipList<T> {
    /// Creates an empty skip list with the default level generator
    /// (`Geometric { levels: 16, p: 0.5 }`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let list = SkipList::<i32>::new();
    /// assert!(list.is_empty());
    /// assert_eq!(list.len(), 0);
    /// ```
    ///
    /// # Panics
    ///
    /// Never panics.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        #[expect(
            clippy::expect_used,
            reason = "DEFAULT_LEVELS and DEFAULT_P are compile-time constants \
                      whose validity is guaranteed by their definitions"
        )]
        let generator = Geometric::new(DEFAULT_LEVELS, DEFAULT_P)
            .expect("DEFAULT_LEVELS and DEFAULT_P are valid Geometric parameters");
        Self::with_level_generator(generator)
    }

    /// Creates an empty skip list with `max_levels` as the level-count hint.
    ///
    /// The level count controls how many skip-link levels the internal
    /// structure will maintain.  A larger value improves performance for very
    /// large lists at the cost of slightly higher per-node memory use.
    /// `max_levels` is clamped to a minimum of 1.
    ///
    /// The default `p = 0.5` promotion probability is used.
    ///
    /// # Panics
    ///
    /// Never panics; `max_levels` is clamped to `>= 1` and the default
    /// `p = 0.5` is always a valid [`Geometric`] probability.
    #[inline]
    #[must_use]
    pub fn with_capacity(max_levels: usize) -> Self {
        let levels = max_levels.max(1);
        #[expect(
            clippy::expect_used,
            reason = "`levels` is clamped to >= 1 and DEFAULT_P is a valid probability"
        )]
        let generator = Geometric::new(levels, DEFAULT_P)
            .expect("`levels >= 1` and `DEFAULT_P` are valid Geometric parameters");
        Self::with_level_generator(generator)
    }
}

// ── Generic methods available for any LevelGenerator ────────────────────────

impl<T, G: LevelGenerator> SkipList<T, G> {
    /// Creates an empty skip list driven by the supplied level generator.
    ///
    /// The generator controls the distribution of tower heights assigned to
    /// newly inserted nodes, which affects the time and space trade-offs of
    /// the skip list.  Use [`SkipList::new`] or [`SkipList::with_capacity`]
    /// for the common case.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    /// use skiplist::level_generator::geometric::Geometric;
    ///
    /// let generator = Geometric::new(8, 0.5).expect("valid parameters");
    /// let list = SkipList::<i32, 16, _>::with_level_generator(generator);
    /// assert!(list.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_level_generator(generator: G) -> Self {
        let max_levels = generator.total();
        Self {
            head: Box::new(Node::new(max_levels)),
            len: 0,
            generator,
        }
    }

    /// Returns the number of elements in the list.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let list = SkipList::<i32>::new();
    /// assert_eq!(list.len(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the list contains no elements.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let list = SkipList::<i32>::new();
    /// assert!(list.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ── Default ──────────────────────────────────────────────────────────────────

impl<T> Default for SkipList<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use pretty_assertions::{assert_eq, assert_ne};

    use super::SkipList;
    use crate::level_generator::geometric::Geometric;

    #[test]
    fn new_is_empty() {
        let list = SkipList::<i32>::new();
        assert!(list.is_empty());
    }

    #[test]
    fn new_len_zero() {
        let list = SkipList::<i32>::new();
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn with_capacity_is_empty() {
        let list = SkipList::<i32>::with_capacity(8);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn with_capacity_zero_clamped() {
        // max_levels = 0 is clamped to 1; must not panic
        let list = SkipList::<i32>::with_capacity(0);
        assert!(list.is_empty());
    }

    #[test]
    fn with_level_generator_custom() {
        let g = Geometric::new(4, 0.25).expect("valid parameters");
        let list = SkipList::<String>::with_level_generator(g);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn default_is_empty() {
        let list = SkipList::<i32>::default();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
    }
}
