//! Index-based skip list.
//!
//! This module provides [`SkipList`], a general-purpose sequence with
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

#![expect(
    clippy::pub_use,
    reason = "re-exporting public types from private submodules"
)]

use core::ptr::NonNull;

use crate::{
    level_generator::{LevelGenerator, geometric::Geometric},
    node::{
        Node,
        visitor::{IndexVisitor, Visitor},
    },
};

mod access;
mod filter;
mod insert_remove;
mod iter;
mod push_pop;
mod structural;
mod traits;

pub use iter::{Drain, ExtractIf, IntoIter, Iter, IterMut};

/// Default promotion probability used by [`SkipList::new`] and
/// [`SkipList::with_capacity`].
const DEFAULT_P: f64 = 0.5;

/// An index-based skip list.
///
/// `SkipList<T, N, G>` stores elements in insertion order and provides `$O(\log
/// n)$` insert, remove, and indexed access.  Unlike [`Vec`], inserting or
/// removing in the middle does not shift elements; unlike a plain linked list,
/// arbitrary-index access is `$O(\log n)$` rather than `$O(n)$`.
///
/// The const generic `$N$` (default `16`) sets the maximum number of skip-link
/// levels per node; increase it when you expect more than `$\sim 2^N$`
/// elements. The type parameter `G` controls how node tower heights are chosen.
/// The default ([`Geometric`]) works well in practice; supply a custom
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
pub struct SkipList<T, const N: usize = 16, G: LevelGenerator = Geometric> {
    /// Sentinel head node. Never holds a value; its `links` array has length
    /// equal to the maximum number of levels.
    head: Box<Node<T, N>>,
    /// Non-owning pointer to the last data node, or `None` when the list is
    /// empty.  Maintained by every insert and remove operation to provide
    /// `$O(1)$` [`back`](SkipList::back) / [`back_mut`](SkipList::back_mut)
    /// access.
    tail: Option<NonNull<Node<T, N>>>,
    /// Cached element count. Updated by every insert / remove operation.
    len: usize,
    /// Level generator used to determine the tower height of each new node.
    generator: G,
}

// MARK: Constructors (default level generator)

impl<T, const N: usize> SkipList<T, N, Geometric> {
    /// Creates an empty skip list with default settings.
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
        Self::with_level_generator(Geometric::default())
    }

    /// Creates an empty skip list pre-configured for the expected number of
    /// elements.
    ///
    /// Use this when you know roughly how many elements the list will hold.
    /// The skip list will be tuned so that skip links span the right number of
    /// nodes for that size, giving good average-case performance without
    /// wasting memory on unnecessary levels for small lists or degrading for
    /// large ones.
    ///
    /// By default, there is an upper limit of `N = 16` levels, which is
    /// optimal for up to about ~2^16 = 65,536 elements. If you need
    /// significantly more levels, increase the const generic parameter `N`
    /// when declaring the list type, e.g. `SkipList<T, 32>`.  The
    /// `with_capacity` method will automatically clamp the level count to `N`
    /// to avoid overrunning the node links array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// // Expect around a hundred elements.
    /// let list = SkipList::<i32>::with_capacity(100);
    /// assert!(list.is_empty());
    ///
    /// // Expect a large number of elements.
    /// let big = SkipList::<i32, 32>::with_capacity(240_000_000);
    /// assert!(big.is_empty());
    /// ```
    ///
    /// # Panics
    ///
    /// Never panics.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        // Derive 1 + ceil(log2(capacity)), clamped to a minimum of 1.
        //
        // With p = 0.5 this is the level count at which the expected number of
        // nodes at the topmost level is ~1 when the list holds `capacity`
        // elements.
        //
        // ceil(log2(n)) = bit_width(n - 1) = BITS - leading_zeros(n - 1)  for n >= 2.
        let levels = if capacity <= 1 {
            1
        } else {
            #[expect(
                clippy::as_conversions,
                reason = "usize::BITS is a u32 value, but should always a valid usize value, \
                even if usize is smaller than 32 bits."
            )]
            let ceil_log2 =
                usize::BITS.saturating_sub(capacity.saturating_sub(1).leading_zeros()) as usize;
            ceil_log2.saturating_add(1)
        };
        // Clamp to the compile-time capacity N so that the node links array is
        // never overrun.  For capacities larger than ~2^N, this means fewer
        // levels than ideal, but the list remains correct.
        #[expect(
            clippy::expect_used,
            reason = "`levels` is always >= 1 and DEFAULT_P is a valid probability"
        )]
        let generator = Geometric::new(levels.min(N), DEFAULT_P)
            .expect("`levels >= 1` and `DEFAULT_P` are valid Geometric parameters");
        Self::with_level_generator(generator)
    }
}

// MARK: Generic methods available for any LevelGenerator

impl<T, G: LevelGenerator, const N: usize> SkipList<T, N, G> {
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
        debug_assert!(
            max_levels <= N,
            "generator.total() ({max_levels}) exceeds node capacity ({N})"
        );
        Self {
            head: Box::new(Node::new(max_levels.min(N))),
            tail: None,
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

    /// Returns a [`NonNull`] pointer to the data node at the given 0-based
    /// `index`.  The caller must ensure `index < self.len`.
    #[expect(
        clippy::expect_used,
        reason = "index < self.len is validated by the debug_assert and all callers; \
                  the expect fires only on internal invariant violations"
    )]
    #[inline]
    fn node_ptr_at(&self, index: usize) -> NonNull<Node<T, N>> {
        debug_assert!(
            index < self.len,
            "index {index} out of bounds (len={})",
            self.len
        );
        IndexVisitor::new(&self.head, index.saturating_add(1))
            .traverse()
            .map(NonNull::from)
            .expect("node at index exists because index < self.len")
    }
}

// MARK: Default

impl<T, const N: usize> Default for SkipList<T, N, Geometric> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

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
        // capacity = 0 results in 1 level; must not panic
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
