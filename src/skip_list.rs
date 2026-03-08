//! Index-based skip list.
//!
//! [`SkipList`] is a general-purpose sequence that stores elements in
//! **insertion order** and provides `$O(\log n)$` insert, remove, and indexed
//! access by position.  It is a useful alternative to [`Vec`] when elements are
//! frequently inserted or removed anywhere in the sequence, and to a plain
//! linked list when `$O(\log n)$` random access by index is needed.
//!
//! Elements need not implement [`Ord`]; the list imposes no ordering on them.
//!
//! # Key Invariants
//!
//! - Elements are stored in insertion order.
//! - Rank (index) of an element equals its position in that insertion-order
//!   sequence.
//! - No ordering constraint exists on element values.
//!
//! # Intentional Omissions
//!
//! - There is no `sort` or `sort_by` method.  Use [`OrderedSkipList`] if you
//!   need a sorted collection.
//! - [`IterMut`] is provided because mutation of elements cannot break any
//!   ordering invariant (there is none).
//!
//! # Method Summary
//!
//! **Constructors:** [`new`], [`with_capacity`], [`with_level_generator`],
//!   [`with_comparator_and_level_generator`][SkipList::with_level_generator].
//!
//! **Access:** [`get`], [`get_mut`], [`front`], [`back`], [`front_mut`],
//!   [`back_mut`].
//!
//! **Insertion:** [`insert`], [`push_front`], [`push_back`].
//!
//! **Removal:** [`remove`], [`pop_front`], [`pop_back`], [`retain`],
//!   [`retain_mut`], [`dedup_by`], [`drain`], [`extract_if`].
//!
//! **Iteration:** [`iter`], [`iter_mut`], [`into_iter`].
//!
//! **Structural:** [`len`], [`is_empty`], [`clear`], [`split_off`], [`append`].
//!
//! # Examples
//!
//! ```rust
//! use skiplist::SkipList;
//!
//! let mut list = SkipList::<i32>::new();
//!
//! // Elements are kept in insertion order, not sorted order.
//! list.push_back(30);
//! list.push_back(10);
//! list.push_back(20);
//!
//! // O(log n) indexed access.
//! assert_eq!(list.get(0), Some(&30));
//! assert_eq!(list.get(1), Some(&10));
//!
//! // O(log n) mid-sequence insert (no element shifting).
//! list.insert(1, 99);
//! assert_eq!(list.get(1), Some(&99));
//!
//! // Iterate in insertion order.
//! let values: Vec<_> = list.iter().copied().collect();
//! assert_eq!(values, [30, 99, 10, 20]);
//! ```
//!
//! [`OrderedSkipList`]: crate::OrderedSkipList
//! [`new`]: SkipList::new
//! [`with_capacity`]: SkipList::with_capacity
//! [`with_level_generator`]: SkipList::with_level_generator
//! [`get`]: SkipList::get
//! [`get_mut`]: SkipList::get_mut
//! [`front`]: SkipList::front
//! [`back`]: SkipList::back
//! [`front_mut`]: SkipList::front_mut
//! [`back_mut`]: SkipList::back_mut
//! [`insert`]: SkipList::insert
//! [`push_front`]: SkipList::push_front
//! [`push_back`]: SkipList::push_back
//! [`remove`]: SkipList::remove
//! [`pop_front`]: SkipList::pop_front
//! [`pop_back`]: SkipList::pop_back
//! [`retain`]: SkipList::retain
//! [`retain_mut`]: SkipList::retain_mut
//! [`dedup_by`]: SkipList::dedup_by
//! [`drain`]: SkipList::drain
//! [`extract_if`]: SkipList::extract_if
//! [`iter`]: SkipList::iter
//! [`iter_mut`]: SkipList::iter_mut
//! [`into_iter`]: SkipList::into_iter
//! [`len`]: SkipList::len
//! [`is_empty`]: SkipList::is_empty
//! [`clear`]: SkipList::clear
//! [`split_off`]: SkipList::split_off
//! [`append`]: SkipList::append

#![expect(
    clippy::pub_use,
    reason = "re-exporting public types from private submodules"
)]

use core::ptr::NonNull;

use crate::{
    level_generator::{LevelGenerator, geometric::Geometric},
    node::{
        Node,
        visitor::{IndexMutVisitor, Visitor},
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
    ///
    /// Stored as `NonNull` rather than `Box` to preserve a single root
    /// provenance tag across all accesses.  See [`crate::docs::internals`]
    /// for the full NonNull-over-Box rationale.
    ///
    /// # Invariant
    ///
    /// `head` was allocated via `Box::into_raw(Box::new(...))` in
    /// [`with_level_generator`] and is exclusively owned by this `SkipList`.
    /// It is freed in `Drop`.
    head: NonNull<Node<T, N>>,
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

// MARK: Send / Sync
//
// `NonNull<T>` is neither `Send` nor `Sync`, so `SkipList` would not be
// auto-Send/Sync.  We provide the impls manually: `SkipList<T,N,G>` is the
// sole owner of every heap-allocated node; no raw pointer is shared across
// threads without `&mut`.  The bounds mirror those of `Vec<T>`.
//
// SAFETY: `SkipList<T,N,G>` exclusively owns all nodes.  No raw pointer is
// shared across threads without exclusive access.
unsafe impl<T: Send, G: LevelGenerator + Send, const N: usize> Send for SkipList<T, N, G> {}
// SAFETY: `SkipList<T,N,G>` exclusively owns all nodes.  No raw pointer is
// shared across threads without exclusive access.
unsafe impl<T: Sync, G: LevelGenerator + Sync, const N: usize> Sync for SkipList<T, N, G> {}

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
    /// The level generator is initialised so that skip links are tuned for
    /// `capacity` elements, giving good average-case performance without
    /// unnecessary levels.  The level count is clamped to `N` so the node
    /// links array is never overrun.
    ///
    /// See [`crate::docs::concepts`] for the level-selection formula and
    /// guidance on choosing `N`.
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
        // Allocate the sentinel head node as a raw pointer so that all
        // subsequent accesses share the same provenance tag.  Storing it as
        // `Box<Node>` would cause Miri (Tree Borrows) to assign a new Reserved
        // child tag on every Box-retag, making sibling writes through other
        // child tags appear as foreign writes that disable the Box's tag.
        //
        // SAFETY: `Box::into_raw` transfers ownership of the allocation to this
        // struct.  It is freed in `Drop::drop` via `Box::from_raw`.
        let head = unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(max_levels.min(N)))))
        };
        Self {
            head,
            tail: None,
            len: 0,
            generator,
        }
    }

    /// Returns a shared reference to the sentinel head node.
    ///
    /// # Safety invariant
    ///
    /// `self.head` is always a live, valid allocation for the lifetime of
    /// `&self`.
    #[inline]
    fn head_ref(&self) -> &Node<T, N> {
        // SAFETY: `self.head` was allocated in `with_level_generator` and
        // remains valid for `&self`'s lifetime.
        unsafe { self.head.as_ref() }
    }

    /// Returns an exclusive reference to the sentinel head node.
    ///
    /// # Safety invariant
    ///
    /// `self.head` is always a live, valid allocation for the lifetime of
    /// `&mut self`.  `&mut self` guarantees exclusive access to all nodes.
    #[inline]
    fn head_mut(&mut self) -> &mut Node<T, N> {
        // SAFETY: `self.head` was allocated in `with_level_generator` and
        // remains valid.  `&mut self` guarantees no other live references.
        unsafe { self.head.as_mut() }
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
        // Rank is 1-based: rank 1 is the first data node (immediately after
        // the sentinel head).  Add 1 to convert the 0-based index.
        IndexMutVisitor::new(self.head, index.saturating_add(1))
            .traverse()
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

// MARK: Drop

impl<T, G: LevelGenerator, const N: usize> Drop for SkipList<T, N, G> {
    #[inline]
    fn drop(&mut self) {
        // Reconstruct the `Box` that owns the head allocation and drop it.
        // `Node::drop` walks the entire `next` chain and frees every data node
        // one at a time (O(n), non-recursive).
        //
        // SAFETY: `self.head` was allocated via `Box::into_raw(Box::new(...))`
        // in `with_level_generator` and is exclusively owned by this
        // `SkipList` for its entire lifetime.
        unsafe { drop(Box::from_raw(self.head.as_ptr())) };
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
