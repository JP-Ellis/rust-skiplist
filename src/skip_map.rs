//! Key-value ordered skip map.
//!
//! This module provides [`SkipMap`], a skip list that maps keys to values,
//! maintaining key-value pairs in sorted key order according to a
//! [`Comparator<K>`].  Insert, lookup, and remove are O(log n) on average.
//! Duplicate keys are allowed (behaves like a multimap).
//!
//! Unlike [`SkipList`], which stores elements in insertion order, a
//! `SkipMap` imposes a total order on its keys and always keeps pairs
//! sorted by key. The ordering is parameterised by a [`Comparator<K>`] so
//! that custom orderings can be used without requiring [`Ord`] on the key
//! type.
//!
//! # Example
//!
//! ```rust
//! use skiplist::skip_map::SkipMap;
//!
//! let map = SkipMap::<&str, i32>::new();
//! assert!(map.is_empty());
//! assert_eq!(map.len(), 0);
//! ```
//!
//! [`SkipList`]: crate::skip_list::SkipList
//! [`Comparator<K>`]: crate::comparator::Comparator

mod access;

use core::ptr::NonNull;

use crate::{
    comparator::{Comparator, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    node::Node,
};

/// A key-value ordered skip map parameterised by a comparator.
///
/// `SkipMap<K, V, N, C, G>` maintains all key-value pairs in the total order
/// defined by `C: Comparator<K>`.  Duplicate keys are permitted and appear
/// adjacent to one another in the iteration order.
///
/// The const generic `N` (default `16`) sets the maximum number of skip-link
/// levels per node; increase it when you expect more than `$\sim 2^N$` elements. `G`
/// controls how tower heights are chosen; the default ([`Geometric`]) works
/// well in practice.
///
/// # Constructors
///
/// | Constructor                                   | `K: Ord`? | Comparator        | Generator       |
/// |-----------------------------------------------|:---------:|:-----------------:|:---------------:|
/// | [`new()`]                                     | required  | [`OrdComparator`] | [`Geometric`]   |
/// | [`with_level_generator(g)`]                   | required  | [`OrdComparator`] | `g`             |
/// | [`with_comparator(c)`]                        | not req.  | `c`               | [`Geometric`]   |
/// | [`with_comparator_and_level_generator(c, g)`] | not req.  | `c`               | `g`             |
///
/// [`new()`]: SkipMap::new
/// [`with_level_generator(g)`]: SkipMap::with_level_generator
/// [`with_comparator(c)`]: SkipMap::with_comparator
/// [`with_comparator_and_level_generator(c, g)`]: SkipMap::with_comparator_and_level_generator
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let map = SkipMap::<&str, u32>::new();
/// assert!(map.is_empty());
/// ```
pub struct SkipMap<
    K,
    V,
    const N: usize = 16,
    C: Comparator<K> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    // Sentinel head node. Never holds a value; its `links` array has length
    // equal to the maximum number of levels.
    //
    // Stored as a raw `NonNull` pointer (not `Box`) so that all accesses,
    // whether shared (`head_ref`) or exclusive (`head_mut`), share the same
    // provenance tag. Under Tree Borrows, `Box<T>` receives special
    // "Unique" retagging that creates a new Reserved child tag; write
    // accesses through any sibling tag then disable the Box's tag, causing
    // false UB reports. Using `NonNull` with a single provenance tag avoids
    // this problem entirely.
    //
    // Invariant: `head` was allocated via `Box::into_raw(Box::new(...))` in
    // `with_comparator_and_level_generator` and is exclusively owned by this
    // `SkipMap`. It is freed in `Drop`.
    head: NonNull<Node<(K, V), N>>,
    /// Non-owning pointer to the last data node, or `None` when the map is
    /// empty. Maintained by every insert and remove operation to provide `$O(1)$`
    /// last-entry access.
    tail: Option<NonNull<Node<(K, V), N>>>,
    /// Cached element count. Updated by every insert / remove operation.
    len: usize,
    /// Comparator defining the key ordering.
    comparator: C,
    /// Level generator used to determine the tower height of each new node.
    generator: G,
}

// MARK: Send / Sync
//
// `NonNull<T>` is neither `Send` nor `Sync`, so `SkipMap` would not be
// auto-Send/Sync. We provide the impls manually: `SkipMap` is the sole
// owner of every heap-allocated node; no raw pointer is shared across
// threads without `&mut`. The bounds mirror those of `BTreeMap<K, V>`.

// SAFETY: `SkipMap<K,V,N,C,G>` exclusively owns all nodes. No raw pointer
// is shared across threads without exclusive access.
unsafe impl<K: Send, V: Send, C: Comparator<K> + Send, G: LevelGenerator + Send, const N: usize>
    Send for SkipMap<K, V, N, C, G>
{
}
// SAFETY: `SkipMap<K,V,N,C,G>` exclusively owns all nodes. No raw pointer
// is shared across threads without exclusive access.
unsafe impl<K: Sync, V: Sync, C: Comparator<K> + Sync, G: LevelGenerator + Sync, const N: usize>
    Sync for SkipMap<K, V, N, C, G>
{
}

// MARK: Constructors (OrdComparator, default level generator)

impl<K: Ord, V, const N: usize> SkipMap<K, V, N, OrdComparator, Geometric> {
    /// Creates an empty skip map using the natural [`Ord`] key ordering
    /// and the default level generator (`Geometric { levels: 16, p: 0.5 }`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let map = SkipMap::<&str, i32>::new();
    /// assert!(map.is_empty());
    /// assert_eq!(map.len(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::with_comparator_and_level_generator(OrdComparator, Geometric::default())
    }
}

// MARK: Constructors (OrdComparator, custom level generator)

impl<K: Ord, V, G: LevelGenerator, const N: usize> SkipMap<K, V, N, OrdComparator, G> {
    /// Creates an empty skip map using the natural [`Ord`] key ordering
    /// and the supplied level generator.
    ///
    /// Use this when you need precise control over the skip-link distribution
    /// (e.g., a different probability or level count than the default 16/0.5).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    /// use skiplist::level_generator::geometric::Geometric;
    ///
    /// let g = Geometric::new(8, 0.5).expect("valid parameters");
    /// let map = SkipMap::<&str, i32>::with_level_generator(g);
    /// assert!(map.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_level_generator(generator: G) -> Self {
        Self::with_comparator_and_level_generator(OrdComparator, generator)
    }
}

// MARK: Constructors (custom comparator, default level generator)

impl<K, V, C: Comparator<K>, const N: usize> SkipMap<K, V, N, C, Geometric> {
    /// Creates an empty skip map with the supplied comparator and the
    /// default level generator (`Geometric { levels: 16, p: 0.5 }`).
    ///
    /// Use this when you need a custom key ordering without implementing
    /// [`Ord`] on the key type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    /// use skiplist::comparator::FnComparator;
    ///
    /// // Largest-key-first ordering.
    /// let map: SkipMap<i32, &str, 16, _> =
    ///     SkipMap::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
    /// assert!(map.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_comparator(comparator: C) -> Self {
        Self::with_comparator_and_level_generator(comparator, Geometric::default())
    }
}

// MARK: Generic methods available for any C + G

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> SkipMap<K, V, N, C, G> {
    /// Creates an empty skip map with the supplied comparator and level
    /// generator.
    ///
    /// This is the base constructor; all other constructors delegate to it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    /// use skiplist::comparator::FnComparator;
    /// use skiplist::level_generator::geometric::Geometric;
    ///
    /// let g = Geometric::new(8, 0.25).expect("valid parameters");
    /// let map: SkipMap<i32, &str, 8, _> =
    ///     SkipMap::with_comparator_and_level_generator(
    ///         FnComparator(|a: &i32, b: &i32| b.cmp(a)),
    ///         g,
    ///     );
    /// assert!(map.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_comparator_and_level_generator(comparator: C, generator: G) -> Self {
        let max_levels = generator.total();
        // `debug_assert` fires in debug builds to catch misconfigured
        // generators early. The `.min(N)` below is a defensive release-build
        // safety net; both are intentional and complement each other.
        debug_assert!(
            max_levels <= N,
            "generator.total() ({max_levels}) exceeds node capacity ({N})"
        );
        // Allocate the sentinel head node as a raw pointer so that all
        // subsequent accesses share the same provenance tag. Storing it as
        // `Box<Node>` would cause Miri (Tree Borrows) to assign a new Reserved
        // child tag on every Box-retag, making sibling writes through other
        // child tags appear as foreign writes that disable the Box's tag.
        //
        // SAFETY: `Box::into_raw` transfers ownership of the allocation to this
        // struct. It is freed in `Drop::drop` via `Box::from_raw`.
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
    /// Safety: `self.head` is always a live, valid allocation for the lifetime
    /// of `&self`.
    #[inline]
    fn head_ref(&self) -> &Node<(K, V), N> {
        // SAFETY: `self.head` was allocated in `with_comparator_and_level_generator`
        // and remains valid for `&self`'s lifetime.
        unsafe { self.head.as_ref() }
    }

    /// Returns an exclusive reference to the sentinel head node.
    ///
    /// Safety: `self.head` is always a live, valid allocation for the lifetime
    /// of `&mut self`. `&mut self` guarantees exclusive access to all nodes.
    #[inline]
    fn head_mut(&mut self) -> &mut Node<(K, V), N> {
        // SAFETY: `self.head` was allocated in `with_comparator_and_level_generator`
        // and remains valid. `&mut self` guarantees no other live references.
        unsafe { self.head.as_mut() }
    }

    /// Returns the number of key-value pairs in the map.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<&str, i32>::new();
    /// assert_eq!(map.len(), 0);
    /// map.insert("a", 1);
    /// assert_eq!(map.len(), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the map contains no key-value pairs.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<&str, i32>::new();
    /// assert!(map.is_empty());
    /// map.insert("a", 1);
    /// assert!(!map.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// MARK: Default

impl<K: Ord, V, const N: usize> Default for SkipMap<K, V, N, OrdComparator, Geometric> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// MARK: Drop

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> Drop for SkipMap<K, V, N, C, G> {
    #[inline]
    fn drop(&mut self) {
        // Reconstruct the `Box` that owns the head allocation and drop it.
        // `Node::drop` walks the entire `next` chain and frees every data node
        // one at a time (O(n), non-recursive).
        //
        // SAFETY: `self.head` was allocated via `Box::into_raw(Box::new(...))`
        // in `with_comparator_and_level_generator` and is exclusively owned by
        // this `SkipMap` for its entire lifetime.
        unsafe { drop(Box::from_raw(self.head.as_ptr())) };
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::SkipMap;
    use crate::{comparator::FnComparator, level_generator::geometric::Geometric};

    // MARK: new

    #[test]
    fn new_is_empty() {
        let map = SkipMap::<&str, i32>::new();
        assert!(map.is_empty());
    }

    #[test]
    fn new_len_zero() {
        let map = SkipMap::<&str, i32>::new();
        assert_eq!(map.len(), 0);
    }

    // MARK: with_level_generator

    #[test]
    fn with_level_generator_is_empty() {
        let g = Geometric::new(8, 0.5).expect("valid parameters");
        let map = SkipMap::<&str, i32>::with_level_generator(g);
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn with_level_generator_custom_params() {
        let g = Geometric::new(4, 0.25).expect("valid parameters");
        let map = SkipMap::<String, String>::with_level_generator(g);
        assert!(map.is_empty());
    }

    // MARK: with_comparator

    #[test]
    fn with_comparator_is_empty() {
        let map: SkipMap<i32, &str, 16, _> =
            SkipMap::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    // MARK: with_comparator_and_level_generator

    #[test]
    fn with_comparator_and_level_generator_is_empty() {
        let g = Geometric::new(8, 0.25).expect("valid parameters");
        let map: SkipMap<i32, &str, 8, _> = SkipMap::with_comparator_and_level_generator(
            FnComparator(|a: &i32, b: &i32| b.cmp(a)),
            g,
        );
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    // MARK: default

    #[test]
    fn default_is_empty() {
        let map = SkipMap::<&str, i32>::default();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }
}
