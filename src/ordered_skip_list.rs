//! Value-ordered skip list.
//!
//! This module provides [`OrderedSkipList`], a skip list that maintains its
//! elements in sorted order according to a [`Comparator<T>`].  Insert, lookup,
//! and remove are O(log n) on average.  Duplicate elements are allowed.
//!
//! Unlike [`SkipList`], which stores elements in insertion order, an
//! `OrderedSkipList` imposes a total order on its elements and always keeps
//! them sorted.  The ordering is parameterised by a [`Comparator<T>`] so that
//! custom orderings can be used without requiring [`Ord`] on the element type.
//!
//! # Example
//!
//! ```rust
//! use skiplist::ordered_skip_list::OrderedSkipList;
//!
//! let list = OrderedSkipList::<i32>::new();
//! assert!(list.is_empty());
//! assert_eq!(list.len(), 0);
//! ```
//!
//! [`SkipList`]: crate::skip_list::SkipList
//! [`Comparator<T>`]: crate::comparator::Comparator

use core::ptr::NonNull;

use crate::{
    comparator::{Comparator, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    node::Node,
};

mod insert_remove;

/// An ordered skip list parameterised by a comparator.
///
/// `OrderedSkipList<T, N, C, G>` maintains all elements in the total order
/// defined by `C: Comparator<T>`.  Duplicate elements are permitted and appear
/// adjacent to one another in the iteration order.
///
/// The const generic `N` (default `16`) sets the maximum number of skip-link
/// levels per node; increase it when you expect more than `$\sim 2^N$` elements.  `G`
/// controls how tower heights are chosen; the default ([`Geometric`]) works
/// well in practice.
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
/// [`new()`]: OrderedSkipList::new
/// [`with_level_generator(g)`]: OrderedSkipList::with_level_generator
/// [`with_comparator(c)`]: OrderedSkipList::with_comparator
/// [`with_comparator_and_level_generator(c, g)`]: OrderedSkipList::with_comparator_and_level_generator
///
/// # Examples
///
/// ```rust
/// use skiplist::ordered_skip_list::OrderedSkipList;
///
/// let list = OrderedSkipList::<u32>::new();
/// assert!(list.is_empty());
/// ```
pub struct OrderedSkipList<
    T,
    const N: usize = 16,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// Sentinel head node. Never holds a value; its `links` array has length
    /// equal to the maximum number of levels.
    ///
    /// Stored as `NonNull` rather than `Box` so that all accesses (shared via
    /// `head_ref` or exclusive via `head_mut`) share the same provenance tag.
    /// Under Tree Borrows, `Box<T>` receives "Unique" retagging, which creates
    /// a Reserved child tag; writes through any sibling tag then disable the
    /// Box's tag, causing false UB reports in Miri.  `NonNull` with a single
    /// provenance tag avoids this problem.
    ///
    /// # Invariant
    ///
    /// `head` was allocated via `Box::into_raw(Box::new(...))` in
    /// [`with_comparator_and_level_generator`][Self::with_comparator_and_level_generator]
    /// and is exclusively owned by this `OrderedSkipList`.  It is freed in
    /// `Drop`.
    head: NonNull<Node<T, N>>,
    /// Non-owning pointer to the last data node, or `None` when the list is
    /// empty.  Maintained by every insert and remove to provide `$O(1)$`
    /// [`last`](OrderedSkipList::last) access.
    tail: Option<NonNull<Node<T, N>>>,
    /// Cached element count.  Updated by every insert / remove operation.
    len: usize,
    /// Comparator defining the element ordering.
    comparator: C,
    /// Level generator used to determine the tower height of each new node.
    generator: G,
}

// MARK: Send / Sync
//
// `NonNull<T>` is neither `Send` nor `Sync`, so `OrderedSkipList` would not
// be auto-Send/Sync.  We provide the impls manually: `OrderedSkipList` is the
// sole owner of every heap-allocated node; no raw pointer is shared across
// threads without `&mut`.  The bounds mirror those of `BTreeMap<K, V>`.

// SAFETY: `OrderedSkipList<T,N,C,G>` exclusively owns all nodes.  No raw
// pointer is shared across threads without exclusive access.
unsafe impl<T: Send, C: Comparator<T> + Send, G: LevelGenerator + Send, const N: usize> Send
    for OrderedSkipList<T, N, C, G>
{
}
// SAFETY: `OrderedSkipList<T,N,C,G>` exclusively owns all nodes.  No raw
// pointer is shared across threads without exclusive access.
unsafe impl<T: Sync, C: Comparator<T> + Sync, G: LevelGenerator + Sync, const N: usize> Sync
    for OrderedSkipList<T, N, C, G>
{
}

// MARK: Constructors (OrdComparator, default level generator)

impl<T: Ord, const N: usize> OrderedSkipList<T, N, OrdComparator, Geometric> {
    /// Creates an empty ordered skip list using the natural [`Ord`] ordering
    /// and the default level generator (`Geometric { levels: 16, p: 0.5 }`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let list = OrderedSkipList::<i32>::new();
    /// assert!(list.is_empty());
    /// assert_eq!(list.len(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::with_comparator_and_level_generator(OrdComparator, Geometric::default())
    }
}

// MARK: Constructors (OrdComparator, custom level generator)

impl<T: Ord, G: LevelGenerator, const N: usize> OrderedSkipList<T, N, OrdComparator, G> {
    /// Creates an empty ordered skip list using the natural [`Ord`] ordering
    /// and the supplied level generator.
    ///
    /// Use this when you need precise control over the skip-link distribution
    /// (e.g., a different probability or level count than the default 16/0.5).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use skiplist::level_generator::geometric::Geometric;
    ///
    /// let g = Geometric::new(8, 0.5).expect("valid parameters");
    /// let list = OrderedSkipList::<i32>::with_level_generator(g);
    /// assert!(list.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_level_generator(generator: G) -> Self {
        Self::with_comparator_and_level_generator(OrdComparator, generator)
    }
}

// MARK: Constructors (custom comparator, default level generator)

impl<T, C: Comparator<T>, const N: usize> OrderedSkipList<T, N, C, Geometric> {
    /// Creates an empty ordered skip list with the supplied comparator and the
    /// default level generator (`Geometric { levels: 16, p: 0.5 }`).
    ///
    /// Use this when you need a custom ordering without implementing [`Ord`] on
    /// the element type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use skiplist::comparator::FnComparator;
    ///
    /// // Largest-first ordering.
    /// let list: OrderedSkipList<i32, 16, _> =
    ///     OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
    /// assert!(list.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_comparator(comparator: C) -> Self {
        Self::with_comparator_and_level_generator(comparator, Geometric::default())
    }
}

// MARK: Generic methods available for any C + G

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> OrderedSkipList<T, N, C, G> {
    /// Creates an empty ordered skip list with the supplied comparator and
    /// level generator.
    ///
    /// This is the base constructor; all other constructors delegate to it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use skiplist::comparator::FnComparator;
    /// use skiplist::level_generator::geometric::Geometric;
    ///
    /// let g = Geometric::new(8, 0.25).expect("valid parameters");
    /// let list: OrderedSkipList<i32, 8, _> =
    ///     OrderedSkipList::with_comparator_and_level_generator(
    ///         FnComparator(|a: &i32, b: &i32| b.cmp(a)),
    ///         g,
    ///     );
    /// assert!(list.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_comparator_and_level_generator(comparator: C, generator: G) -> Self {
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
            comparator,
            generator,
        }
    }

    /// Returns a shared reference to the sentinel head node.
    ///
    /// The head sentinel is always valid for the lifetime of `&self`.
    #[inline]
    fn head_ref(&self) -> &Node<T, N> {
        // SAFETY: `self.head` was allocated in `with_comparator_and_level_generator`
        // and remains valid for `&self`'s lifetime.
        unsafe { self.head.as_ref() }
    }

    /// Returns an exclusive reference to the sentinel head node.
    ///
    /// The head sentinel is always valid for the lifetime of `&mut self`.
    /// `&mut self` guarantees exclusive access to all nodes.
    #[inline]
    fn head_mut(&mut self) -> &mut Node<T, N> {
        // SAFETY: `self.head` was allocated in `with_comparator_and_level_generator`
        // and remains valid.  `&mut self` guarantees no other live references.
        unsafe { self.head.as_mut() }
    }

    /// Returns the number of elements in the list.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// assert_eq!(list.len(), 0);
    /// list.insert(1);
    /// list.insert(2);
    /// assert_eq!(list.len(), 2);
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
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// assert!(list.is_empty());
    /// list.insert(42);
    /// assert!(!list.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// MARK: Default

impl<T: Ord, const N: usize> Default for OrderedSkipList<T, N, OrdComparator, Geometric> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// MARK: Drop

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> Drop for OrderedSkipList<T, N, C, G> {
    #[inline]
    fn drop(&mut self) {
        // Reconstruct the `Box` that owns the head allocation and drop it.
        // `Node::drop` walks the entire `next` chain and frees every data node
        // one at a time (O(n), non-recursive).
        //
        // SAFETY: `self.head` was allocated via `Box::into_raw(Box::new(...))`
        // in `with_comparator_and_level_generator` and is exclusively owned by
        // this `OrderedSkipList` for its entire lifetime.
        unsafe { drop(Box::from_raw(self.head.as_ptr())) };
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::OrderedSkipList;
    use crate::{comparator::FnComparator, level_generator::geometric::Geometric};

    // MARK: new

    #[test]
    fn new_is_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert!(list.is_empty());
    }

    #[test]
    fn new_len_zero() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(list.len(), 0);
    }

    // MARK: with_level_generator

    #[test]
    fn with_level_generator_is_empty() {
        let g = Geometric::new(8, 0.5).expect("valid parameters");
        let list = OrderedSkipList::<i32>::with_level_generator(g);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn with_level_generator_custom_params() {
        let g = Geometric::new(4, 0.25).expect("valid parameters");
        let list = OrderedSkipList::<String>::with_level_generator(g);
        assert!(list.is_empty());
    }

    // MARK: with_comparator

    #[test]
    fn with_comparator_is_empty() {
        let list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    // MARK: with_comparator_and_level_generator

    #[test]
    fn with_comparator_and_level_generator_is_empty() {
        let g = Geometric::new(8, 0.25).expect("valid parameters");
        let list: OrderedSkipList<i32, 8, _> = OrderedSkipList::with_comparator_and_level_generator(
            FnComparator(|a: &i32, b: &i32| b.cmp(a)),
            g,
        );
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    // MARK: default

    #[test]
    fn default_is_empty() {
        let list = OrderedSkipList::<i32>::default();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
    }
}
