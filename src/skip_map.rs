//! Key-value ordered skip map.
//!
//! [`SkipMap`] maintains **unique** key-value pairs sorted by key according to
//! a [`Comparator<K>`].  Insert, lookup, and remove are `$O(\log n)$` on average.
//! Inserting a key that already exists replaces the existing value and returns
//! the old one, identical to [`BTreeMap`] semantics.
//!
//! Values can be mutated in place (via [`get_mut`]) because changing a value
//! does not affect key order.
//!
//! The key ordering is parameterised by `C: Comparator<K>` so that custom
//! orderings can be used without requiring [`Ord`] on the key type.
//!
//! # Key Invariants
//!
//! - Each key appears at most once.
//! - Keys are always stored in sorted order.
//! - Values may be mutated in place; keys may not.
//!
//! # Intentional Omissions
//!
//! - **No `keys_mut` or `IterMut` over keys.**  Exposing mutable key
//!   references would allow callers to break the sorted invariant without
//!   reinserting.
//!
//! # Method Summary
//!
//! **Constructors:** [`new`], [`with_level_generator`], [`with_comparator`],
//!   [`with_comparator_and_level_generator`].
//!
//! **Access:** [`get`], [`get_mut`], [`get_key_value`], [`get_by_index`],
//!   [`contains_key`], [`first_key_value`], [`last_key_value`], [`rank`],
//!   [`comparator`], [`entry`].
//!
//! **Insertion:** [`insert`], [`merge`].
//!
//! **Removal:** [`remove`], [`remove_entry`], [`pop_first`], [`pop_last`],
//!   [`retain`], [`drain`], [`extract_if`].
//!
//! **Structural:** [`len`], [`is_empty`], [`clear`], [`split_off`], [`append`].
//!
//! **Iteration:** [`iter`], [`iter_mut`], [`keys`], [`values`], [`values_mut`],
//!   [`into_keys`], [`into_values`], [`into_iter`].
//!
//! # Examples
//!
//! ```rust
//! use skiplist::SkipMap;
//!
//! let mut map = SkipMap::<&str, usize>::new();
//!
//! // Inserting new keys returns None.
//! assert_eq!(map.insert("banana", 2), None);
//! assert_eq!(map.insert("apple",  1), None);
//! assert_eq!(map.insert("cherry", 3), None);
//!
//! // Inserting a duplicate key replaces the value and returns the old one.
//! assert_eq!(map.insert("apple", 10), Some(1));
//!
//! // Lookup by key.
//! assert_eq!(map.get(&"apple"), Some(&10));
//!
//! // Iteration is in sorted key order.
//! let pairs: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
//! assert_eq!(pairs, [("apple", 10), ("banana", 2), ("cherry", 3)]);
//! ```
//!
//! [`SkipList`]: crate::skip_list::SkipList
//! [`Comparator<K>`]: crate::comparator::Comparator
//! [`BTreeMap`]: std::collections::BTreeMap
//! [`new`]: SkipMap::new
//! [`with_level_generator`]: SkipMap::with_level_generator
//! [`with_comparator`]: SkipMap::with_comparator
//! [`with_comparator_and_level_generator`]: SkipMap::with_comparator_and_level_generator
//! [`get`]: SkipMap::get
//! [`get_mut`]: SkipMap::get_mut
//! [`get_key_value`]: SkipMap::get_key_value
//! [`get_by_index`]: SkipMap::get_by_index
//! [`contains_key`]: SkipMap::contains_key
//! [`first_key_value`]: SkipMap::first_key_value
//! [`last_key_value`]: SkipMap::last_key_value
//! [`rank`]: SkipMap::rank
//! [`comparator`]: SkipMap::comparator
//! [`entry`]: SkipMap::entry
//! [`insert`]: SkipMap::insert
//! [`merge`]: SkipMap::merge
//! [`remove`]: SkipMap::remove
//! [`remove_entry`]: SkipMap::remove_entry
//! [`pop_first`]: SkipMap::pop_first
//! [`pop_last`]: SkipMap::pop_last
//! [`retain`]: SkipMap::retain
//! [`drain`]: SkipMap::drain
//! [`extract_if`]: SkipMap::extract_if
//! [`len`]: SkipMap::len
//! [`is_empty`]: SkipMap::is_empty
//! [`clear`]: SkipMap::clear
//! [`split_off`]: SkipMap::split_off
//! [`append`]: SkipMap::append
//! [`iter`]: SkipMap::iter
//! [`iter_mut`]: SkipMap::iter_mut
//! [`keys`]: SkipMap::keys
//! [`values`]: SkipMap::values
//! [`values_mut`]: SkipMap::values_mut
//! [`into_keys`]: SkipMap::into_keys
//! [`into_values`]: SkipMap::into_values
//! [`into_iter`]: SkipMap::into_iter

mod access;
#[cfg(feature = "cursor")]
pub mod cursor;
#[cfg(feature = "cursor")]
pub use cursor::{Cursor, CursorMut, UnorderedKeyError};
mod entry;
mod filter;
mod insert_remove;
mod iter;
mod ops;
mod structural;
mod traits;

use core::ptr::NonNull;

pub use entry::{Entry, OccupiedEntry, OccupiedError, VacantEntry};
pub use iter::{
    Drain, ExtractIf, IntoIter, IntoKeys, IntoValues, Iter, IterMut, Keys, Values, ValuesMut,
};

use crate::{
    comparator::{Comparator, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    node::Node,
};

/// A key-value ordered skip map parameterised by a comparator.
///
/// `SkipMap<K, V, N, C, G>` maintains unique key-value pairs in the total
/// order defined by `C: Comparator<K>`. Each key maps to exactly one value;
/// inserting a key that is already present replaces the existing value and
/// returns the old one.
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
    /// Sentinel head node. Never holds a value; its `links` array has length
    /// equal to the maximum number of levels.
    ///
    /// Stored as `NonNull` rather than `Box` to preserve a single root
    /// provenance tag across all accesses.  See `crate::docs::internals`
    /// for the full NonNull-over-Box rationale.
    ///
    /// Invariant: `head` was allocated via `Box::into_raw(Box::new(...))` in
    /// `with_comparator_and_level_generator` and is exclusively owned by this
    /// `SkipMap`. It is freed in `Drop`.
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
