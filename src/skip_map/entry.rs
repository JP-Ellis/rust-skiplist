//! Entry API for [`SkipMap`](super::SkipMap).
//!
//! This module provides the [`Entry`] enum and the [`OccupiedEntry`] and
//! [`VacantEntry`] types that enable in-place, conditional insertion.

use core::{fmt, iter, mem, ptr::NonNull};

use arrayvec::ArrayVec;

use super::SkipMap;
use crate::{
    comparator::{Comparator, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    node::{
        Node,
        link::Link,
        visitor::{OrdIndexMutVisitor, Visitor},
    },
};

// MARK: OccupiedEntry

/// A view into an occupied entry in a [`SkipMap`].
///
/// It is part of the [`Entry`] enum returned by [`SkipMap::entry`].
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::{Entry, SkipMap};
///
/// let mut map = SkipMap::<i32, &str>::new();
/// map.insert(1, "a");
/// if let Entry::Occupied(mut e) = map.entry(1) {
///     assert_eq!(e.key(), &1);
///     assert_eq!(e.insert("b"), "a");
/// }
/// assert_eq!(map.get(&1), Some(&"b"));
/// ```
pub struct OccupiedEntry<
    'a,
    K,
    V,
    const N: usize = 16,
    C: Comparator<K> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// Pointer to the found node.
    node: NonNull<Node<(K, V), N>>,
    /// Precursor nodes at each skip level, used when splicing out or
    /// re-wiring links during removal.
    precursors: ArrayVec<NonNull<Node<(K, V), N>>, N>,
    /// Exclusive mutable borrow of the owning map.
    map: &'a mut SkipMap<K, V, N, C, G>,
}

impl<K: fmt::Debug, V: fmt::Debug, const N: usize, C: Comparator<K>, G: LevelGenerator> fmt::Debug
    for OccupiedEntry<'_, K, V, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .field("value", self.get())
            .finish()
    }
}

impl<'a, K, V, const N: usize, C: Comparator<K>, G: LevelGenerator>
    OccupiedEntry<'a, K, V, N, C, G>
{
    /// Returns a reference to this entry's key.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::{Entry, SkipMap};
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(1, "a");
    /// if let Entry::Occupied(e) = map.entry(1) {
    ///     assert_eq!(e.key(), &1);
    /// }
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "data nodes always have a value; the head sentinel \
                  is never exposed as an OccupiedEntry"
    )]
    #[inline]
    #[must_use]
    pub fn key(&self) -> &K {
        // SAFETY: self.node is a valid data node for &self's lifetime.
        &unsafe { self.node.as_ref() }.value().expect("data node").0
    }

    /// Returns a shared reference to the entry's value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::{Entry, SkipMap};
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// map.insert(1, 10);
    /// if let Entry::Occupied(e) = map.entry(1) {
    ///     assert_eq!(e.get(), &10);
    /// }
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "data nodes always have a value; the head sentinel \
                  is never exposed as an OccupiedEntry"
    )]
    #[inline]
    #[must_use]
    pub fn get(&self) -> &V {
        // SAFETY: self.node is a valid data node for &self's lifetime.
        &unsafe { self.node.as_ref() }.value().expect("data node").1
    }

    /// Returns a mutable reference to the entry's value.
    ///
    /// If you need a reference with the map's lifetime, use [`into_mut`].
    ///
    /// [`into_mut`]: OccupiedEntry::into_mut
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::{Entry, SkipMap};
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// map.insert(1, 10);
    /// if let Entry::Occupied(mut e) = map.entry(1) {
    ///     *e.get_mut() += 5;
    ///     assert_eq!(e.get(), &15);
    /// }
    /// assert_eq!(map.get(&1), Some(&15));
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "data nodes always have a value; the head sentinel \
                  is never exposed as an OccupiedEntry"
    )]
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        // SAFETY: self.node is a valid data node. self.map is &'a mut, so no
        // other mutable reference to this node exists while this borrow lives.
        unsafe { &mut self.node.as_mut().value_mut().expect("data node").1 }
    }

    /// Converts the entry into a mutable reference to its value in the map
    /// with the map's lifetime.
    ///
    /// If you need a reference that only lives as long as this entry, use
    /// [`get_mut`] instead.
    ///
    /// [`get_mut`]: OccupiedEntry::get_mut
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::{Entry, SkipMap};
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// map.insert(1, 10);
    /// let v = match map.entry(1) {
    ///     Entry::Occupied(e) => e.into_mut(),
    ///     Entry::Vacant(e) => e.insert(0),
    ///     _ => unreachable!(),
    /// };
    /// *v += 5;
    /// assert_eq!(map.get(&1), Some(&15));
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "data nodes always have a value; the head sentinel \
                  is never exposed as an OccupiedEntry"
    )]
    #[inline]
    #[must_use]
    pub fn into_mut(self) -> &'a mut V {
        let mut node = self.node;
        // SAFETY: `node` is a valid, heap-allocated data node owned by `map`.
        // `self.map` held `&'a mut SkipMap`, guaranteeing exclusive access to
        // the node for the lifetime `'a`. The returned `&'a mut V` extends
        // that exclusivity to the value component for the same lifetime.
        unsafe { &mut node.as_mut().value_mut().expect("data node").1 }
    }

    /// Sets the value of the entry, returning the old value.
    ///
    /// Unlike [`remove`](OccupiedEntry::remove), the entry remains usable
    /// after this call; you can call `insert` again or access the new value
    /// via [`get`](OccupiedEntry::get).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::{Entry, SkipMap};
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(1, "a");
    /// if let Entry::Occupied(mut e) = map.entry(1) {
    ///     assert_eq!(e.insert("b"), "a");
    ///     assert_eq!(e.get(), &"b"); // entry still usable after insert
    /// }
    /// assert_eq!(map.get(&1), Some(&"b"));
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "data nodes always have a value; the head sentinel \
                  is never exposed as an OccupiedEntry"
    )]
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        // SAFETY: self.node is a valid data node. &mut self guarantees
        // exclusive access. Replacing the value in place preserves all
        // structural invariants: the key is never touched, so the
        // ordering invariant holds.
        unsafe {
            let pair = self.node.as_mut().value_mut().expect("data node");
            mem::replace(&mut pair.1, value)
        }
    }

    /// Removes the entry from the map, returning the value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::{Entry, SkipMap};
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(1, "a");
    /// if let Entry::Occupied(e) = map.entry(1) {
    ///     assert_eq!(e.remove(), "a");
    /// }
    /// assert!(map.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Removes the entry from the map, returning the key-value pair.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::{Entry, SkipMap};
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(1, "a");
    /// if let Entry::Occupied(e) = map.entry(1) {
    ///     assert_eq!(e.remove_entry(), (1, "a"));
    /// }
    /// assert!(map.is_empty());
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "new_dist >= 1 because skip-link distances are computed from consecutive \
                  node ranks; decrement_distance requires distance >= 2 for spanning links; \
                  take_value on a data node always returns Some; \
                  all expects fire only on internal invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l iterates 0..max_levels; precursors[l] is valid because \
                  OrdIndexMutVisitor fills all max_levels entries; \
                  links_mut()[l] is valid because l < node.level() <= max_levels; \
                  precursors[0] is valid because max_levels >= 1"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "head level read, link splicing, and node pop touch provably disjoint \
                  heap nodes; splitting across blocks would require unsafe-crossing \
                  raw-pointer variables"
    )]
    #[inline]
    #[must_use]
    pub fn remove_entry(self) -> (K, V) {
        let Self {
            node: target_ptr,
            precursors,
            map,
        } = self;

        // SAFETY: map.head is valid for the lifetime of &mut map.
        let max_levels = unsafe { map.head.as_ref() }.level();

        // SAFETY: target_ptr is a live data node owned by `map`.
        // precursors[l] for l < target_height have their level-l link
        // pointing to target_ptr (skip-list invariant + OrdIndexMutVisitor
        // semantics for Equal).  For l >= target_height, precursors[l] is
        // the last node at level l whose link spans past target_ptr.
        // No other &mut references to any node exist.
        let (key, val, new_tail) = unsafe {
            let target_height = target_ptr.as_ref().level();
            let target_raw = target_ptr.as_ptr();

            // Splice out target_ptr with accurate distance maintenance.
            //
            // For l < target_height:
            //   pred.links[l] → target (dist d1), target.links[l] → succ (d2).
            //   New: pred.links[l] → succ (dist d1 + d2 - 1) or None.
            // For l >= target_height:
            //   pred.links[l] spans over target to some successor at a rank
            //   one greater than before; decrement the distance.
            for (l, pred_nn) in precursors.iter().enumerate().take(max_levels) {
                let pred_ptr = pred_nn.as_ptr();
                if l < target_height {
                    let old_link = (*pred_ptr).links_mut()[l].take();
                    let target_link = (*target_raw).links_mut()[l].take();
                    (*pred_ptr).links_mut()[l] = match (old_link, target_link) {
                        (Some(pred_to_target), Some(target_to_succ)) => {
                            let new_dist = pred_to_target
                                .distance()
                                .get()
                                .saturating_add(target_to_succ.distance().get())
                                .saturating_sub(1);
                            Some(Link::new(target_to_succ.node(), new_dist).expect("new_dist >= 1"))
                        }
                        (_, None) => None,
                        (None, tgt) => tgt,
                    };
                } else if let Some(link) = (*pred_ptr).links_mut()[l].as_mut() {
                    link.decrement_distance()
                        .expect("skip link spanning target has distance >= 2");
                }
            }

            // Capture predecessor before removing the node.
            let new_tail = (*target_raw).prev();
            let mut popped = (*target_raw).pop();
            let (k, v) = popped.take_value().expect("data node has a value");
            (k, v, new_tail)
        };

        if map.tail == Some(target_ptr) {
            map.tail = if map.len == 1 { None } else { new_tail };
        }
        map.len = map.len.saturating_sub(1);

        (key, val)
    }
}

// MARK: VacantEntry

/// A view into a vacant entry in a [`SkipMap`].
///
/// It is part of the [`Entry`] enum returned by [`SkipMap::entry`].
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::{Entry, SkipMap};
///
/// let mut map = SkipMap::<i32, &str>::new();
/// if let Entry::Vacant(e) = map.entry(1) {
///     assert_eq!(e.key(), &1);
///     e.insert("hello");
/// }
/// assert_eq!(map.get(&1), Some(&"hello"));
/// ```
pub struct VacantEntry<
    'a,
    K,
    V,
    const N: usize = 16,
    C: Comparator<K> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// The key that was looked up.
    key: K,
    /// The node that the traversal stopped on (last node with key < searched
    /// key).  The new node will be inserted immediately after this node.
    current: NonNull<Node<(K, V), N>>,
    /// Rank of `current` at the time of traversal (head = 0, first data = 1).
    current_rank: usize,
    /// Precursor node at each skip level: the insertion point for
    /// wiring the new node's skip links.
    precursors: ArrayVec<NonNull<Node<(K, V), N>>, N>,
    /// Rank (distance from head) of each precursor node at the time
    /// of traversal; used to compute accurate skip-link distances.
    precursor_distances: ArrayVec<usize, N>,
    /// Exclusive mutable borrow of the owning map.
    map: &'a mut SkipMap<K, V, N, C, G>,
}

impl<K: fmt::Debug, V, const N: usize, C: Comparator<K>, G: LevelGenerator> fmt::Debug
    for VacantEntry<'_, K, V, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VacantEntry").field(&self.key).finish()
    }
}

impl<'a, K, V, const N: usize, C: Comparator<K>, G: LevelGenerator> VacantEntry<'a, K, V, N, C, G> {
    /// Returns a reference to this entry's key.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::{Entry, SkipMap};
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// if let Entry::Vacant(e) = map.entry(1) {
    ///     assert_eq!(e.key(), &1);
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Consumes the entry, returning the key that was looked up.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::{Entry, SkipMap};
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// if let Entry::Vacant(e) = map.entry(1) {
    ///     assert_eq!(e.into_key(), 1);
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn into_key(self) -> K {
        self.key
    }

    /// Inserts a value into the map at this key, returning a mutable
    /// reference to the inserted value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::{Entry, SkipMap};
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// if let Entry::Vacant(e) = map.entry(1) {
    ///     e.insert("hello");
    /// }
    /// assert_eq!(map.get(&1), Some(&"hello"));
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "value_mut on the freshly-created data node returned by \
                  insert_raw always returns Some; this expect fires only on \
                  internal invariant violations"
    )]
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        let (node, _, _map) = self.insert_raw(value);
        // SAFETY: node is a valid, heap-allocated data node owned by `map`.
        // `map` was exclusively borrowed for `'a`, so no other mutable
        // reference to this value exists for `'a`.
        unsafe { &mut (*node.as_ptr()).value_mut().expect("data node").1 }
    }

    /// Inserts a value and returns an [`OccupiedEntry`] for the newly
    /// inserted entry.
    ///
    /// Unlike [`insert`](VacantEntry::insert), this gives you the
    /// [`OccupiedEntry`] so you can inspect or modify the entry further
    /// without a second map lookup.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::{Entry, SkipMap};
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// if let Entry::Vacant(e) = map.entry(1) {
    ///     let occupied = e.insert_entry("hello");
    ///     assert_eq!(occupied.key(), &1);
    ///     assert_eq!(occupied.get(), &"hello");
    /// }
    /// ```
    #[inline]
    pub fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V, N, C, G> {
        let (node, precursors, map) = self.insert_raw(value);
        OccupiedEntry {
            node,
            precursors,
            map,
        }
    }

    /// Performs the actual insertion and returns the new node pointer, the
    /// precursor array (updated so each level now points to the inserted
    /// node), and the map borrow.  Both `insert` and `insert_entry`
    /// delegate here.
    #[expect(
        clippy::expect_used,
        reason = "Link::new distances are >= 1 by construction; increment_distance \
                  overflow requires > usize::MAX nodes; precursors[0] always exists \
                  because max_levels >= 1; value_mut on a freshly-created data node \
                  always returns Some; all expects fire only on internal invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "precursors[0] is valid: max_levels >= 1 so precursors.len() >= 1; \
                  precursors[l] and new_raw.links_mut()[l] are bounded by \
                  l < height <= node.level() = max_levels"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "insertion and link wiring touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer \
                  variables"
    )]
    #[expect(
        clippy::type_complexity,
        reason = "return type is a private implementation detail; \
                  a type alias here would add noise without clarity"
    )]
    fn insert_raw(
        self,
        value: V,
    ) -> (
        NonNull<Node<(K, V), N>>,
        ArrayVec<NonNull<Node<(K, V), N>>, N>,
        &'a mut SkipMap<K, V, N, C, G>,
    ) {
        let Self {
            key,
            current,
            current_rank,
            precursors,
            precursor_distances,
            map,
        } = self;

        // height ∈ [0, total]: number of skip links to allocate.
        let height = map.generator.level();
        // new_rank = current_rank + 1 (current is the last node with key < our key).
        let new_rank = current_rank.saturating_add(1);

        // SAFETY: All raw pointers originate from `NonNull<Node<(K,V), N>>`
        // values captured during traversal. They point into heap allocations
        // exclusively owned by `map`. No safe `&mut` references to any node
        // exist while this block runs.
        let new_node_nonnull: NonNull<Node<(K, V), N>> = unsafe {
            // Insert after `current` (the last node strictly less than `key`).
            let new_raw: *mut Node<(K, V), N> =
                Node::insert_after(current, Node::with_value(height, (key, value))).as_ptr();

            // Wire skip links with accurate distances (identical pattern to
            // SkipMap::insert's structural path).
            for (l, (pred_nn, pred_rank)) in precursors
                .iter()
                .copied()
                .zip(precursor_distances.iter().copied())
                .enumerate()
            {
                let pred_ptr = pred_nn.as_ptr();
                if l < height {
                    let distance = new_rank.saturating_sub(pred_rank);
                    let old_link = (*pred_ptr).links_mut()[l].take();
                    (*pred_ptr).links_mut()[l] = Some(
                        Link::new(NonNull::new_unchecked(new_raw), distance)
                            .expect("distance >= 1"),
                    );
                    (*new_raw).links_mut()[l] = if let Some(old) = old_link {
                        let new_d = old
                            .distance()
                            .get()
                            .saturating_sub(distance)
                            .saturating_add(1);
                        Some(Link::new(old.node(), new_d).expect("new_d >= 1"))
                    } else {
                        None
                    };
                } else if let Some(link) = (*pred_ptr).links_mut()[l].as_mut() {
                    link.increment_distance()
                        .expect("distance overflow requires > usize::MAX nodes");
                }
            }

            NonNull::new_unchecked(new_raw)
        };

        // The new node is the tail if it has no successor.
        // SAFETY: `new_node_nonnull` was just created from `Box::into_raw`
        // above; it is properly aligned, fully initialized, and no other
        // reference to it exists yet.
        let is_new_tail = unsafe { new_node_nonnull.as_ref() }.next().is_none();
        if is_new_tail {
            map.tail = Some(new_node_nonnull);
        }
        map.len = map.len.saturating_add(1);

        (new_node_nonnull, precursors, map)
    }
}

// MARK: Entry

/// A view into a single entry in a [`SkipMap`], which may be vacant or
/// occupied.
///
/// This enum is constructed by the [`entry`] method on [`SkipMap`].
///
/// [`entry`]: SkipMap::entry
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::{Entry, SkipMap};
///
/// let mut map = SkipMap::<i32, i32>::new();
/// // Insert a value if absent, or update it if present.
/// *map.entry(1).or_insert(0) += 10;
/// assert_eq!(map.get(&1), Some(&10));
///
/// match map.entry(1) {
///     Entry::Occupied(e) => assert_eq!(e.get(), &10),
///     Entry::Vacant(_) => panic!("expected occupied"),
/// }
/// ```
#[expect(
    clippy::exhaustive_enums,
    reason = "Entry is intentionally exhaustive with only Occupied and Vacant variants"
)]
pub enum Entry<
    'a,
    K,
    V,
    const N: usize = 16,
    C: Comparator<K> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V, N, C, G>),
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V, N, C, G>),
}

impl<K: fmt::Debug, V: fmt::Debug, const N: usize, C: Comparator<K>, G: LevelGenerator> fmt::Debug
    for Entry<'_, K, V, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Occupied(e) => f.debug_tuple("Entry::Occupied").field(e).finish(),
            Self::Vacant(e) => f.debug_tuple("Entry::Vacant").field(e).finish(),
        }
    }
}

impl<'a, K, V, const N: usize, C: Comparator<K>, G: LevelGenerator> Entry<'a, K, V, N, C, G> {
    /// Returns a reference to this entry's key.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// assert_eq!(map.entry(1).key(), &1);
    /// ```
    #[inline]
    #[must_use]
    pub fn key(&self) -> &K {
        match self {
            Self::Occupied(e) => e.key(),
            Self::Vacant(e) => e.key(),
        }
    }

    /// Ensures a value is in the entry by inserting `default` if vacant,
    /// and returns a mutable reference to the value in the map.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// map.entry(1).or_insert(10);
    /// assert_eq!(map.get(&1), Some(&10));
    /// // Already present: value is unchanged.
    /// map.entry(1).or_insert(20);
    /// assert_eq!(map.get(&1), Some(&10));
    /// ```
    #[inline]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => e.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of `default`
    /// if vacant, and returns a mutable reference to the value in the map.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, String>::new();
    /// map.entry(1).or_insert_with(|| "hello".to_string());
    /// assert_eq!(map.get(&1).map(String::as_str), Some("hello"));
    /// ```
    #[inline]
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => e.insert(default()),
        }
    }

    /// Ensures a value is in the entry by inserting the result of `default`
    /// (called with the entry's key) if vacant, and returns a mutable
    /// reference to the value in the map.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// map.entry(5).or_insert_with_key(|k| *k * 10);
    /// assert_eq!(map.get(&5), Some(&50));
    /// ```
    #[inline]
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => {
                let v = default(e.key());
                e.insert(v)
            }
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential insertion, and returns the entry.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// map.insert(1, 10);
    /// map.entry(1).and_modify(|v| *v += 1).or_insert(0);
    /// assert_eq!(map.get(&1), Some(&11));
    /// map.entry(2).and_modify(|v| *v += 1).or_insert(0);
    /// assert_eq!(map.get(&2), Some(&0));
    /// ```
    #[inline]
    #[must_use]
    pub fn and_modify<F: FnOnce(&mut V)>(self, f: F) -> Self {
        match self {
            Self::Occupied(mut e) => {
                f(e.get_mut());
                Self::Occupied(e)
            }
            Self::Vacant(e) => Self::Vacant(e),
        }
    }

    /// Ensures a value is in the entry by inserting `V::default()` if vacant,
    /// and returns a mutable reference to the value in the map.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// map.entry(1).or_default();
    /// assert_eq!(map.get(&1), Some(&0));
    /// ```
    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => e.insert(V::default()),
        }
    }

    /// Inserts `value` and returns an [`OccupiedEntry`] for the entry.
    ///
    /// If the entry is vacant the value is inserted and the resulting
    /// [`OccupiedEntry`] is returned.  If it is already occupied the
    /// existing value is replaced with `value` and the entry is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    ///
    /// // Vacant: inserts and returns the entry.
    /// let e = map.entry(1).insert_entry("a");
    /// assert_eq!(e.get(), &"a");
    ///
    /// // Occupied: replaces the value and still returns the entry.
    /// let e = map.entry(1).insert_entry("b");
    /// assert_eq!(e.get(), &"b");
    /// assert_eq!(map.get(&1), Some(&"b"));
    /// ```
    #[inline]
    pub fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V, N, C, G> {
        match self {
            Self::Occupied(mut e) => {
                e.insert(value);
                e
            }
            Self::Vacant(e) => e.insert_entry(value),
        }
    }
}

// MARK: OccupiedError

/// The error returned by [`SkipMap::try_insert`] when the key is already
/// present in the map.
///
/// Contains the existing [`OccupiedEntry`] and the value that was not
/// inserted.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let mut map = SkipMap::<i32, &str>::new();
/// map.insert(1, "a");
/// let err = map.try_insert(1, "b").unwrap_err();
/// assert_eq!(err.value, "b");
/// assert_eq!(err.entry.get(), &"a");
/// ```
#[non_exhaustive]
pub struct OccupiedError<
    'a,
    K,
    V,
    const N: usize = 16,
    C: Comparator<K> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// The existing occupied entry.
    pub entry: OccupiedEntry<'a, K, V, N, C, G>,
    /// The value that was not inserted.
    pub value: V,
}

impl<K: fmt::Debug, V: fmt::Debug, const N: usize, C: Comparator<K>, G: LevelGenerator> fmt::Debug
    for OccupiedError<'_, K, V, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedError")
            .field("entry", &self.entry)
            .field("value", &self.value)
            .finish()
    }
}

impl<K: fmt::Debug, V: fmt::Debug, const N: usize, C: Comparator<K>, G: LevelGenerator> fmt::Display
    for OccupiedError<'_, K, V, N, C, G>
{
    #[expect(
        clippy::use_debug,
        reason = "the key is displayed via its Debug impl, mirroring BTreeMap::OccupiedError"
    )]
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "entry with key {:?} already exists", self.entry.key())
    }
}

impl<K: fmt::Debug, V: fmt::Debug, const N: usize, C: Comparator<K>, G: LevelGenerator>
    std::error::Error for OccupiedError<'_, K, V, N, C, G>
{
}

// MARK: SkipMap: entry / try_insert / first_entry / last_entry

impl<K, V, const N: usize, C: Comparator<K>, G: LevelGenerator> SkipMap<K, V, N, C, G> {
    /// Returns a view into a single entry in the map for in-place
    /// manipulation.
    ///
    /// This is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// *map.entry(1).or_insert(0) += 10;
    /// assert_eq!(map.get(&1), Some(&10));
    /// *map.entry(1).or_insert(0) += 10;
    /// assert_eq!(map.get(&1), Some(&20));
    /// ```
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, N, C, G> {
        let (current_rank, current, found, precursors, precursor_distances) = {
            // Copy head (NonNull is Copy) so the closure borrows only
            // self.comparator, not the whole of self.
            let head = self.head;
            let cmp = |entry: &(K, V), k: &K| self.comparator.compare(&entry.0, k);
            let mut visitor = OrdIndexMutVisitor::new(head, &key, cmp);
            visitor.traverse();
            let rank = visitor.current_rank_internal();
            let (current, found, precursors, precursor_distances) = visitor.into_parts();
            (rank, current, found, precursors, precursor_distances)
        };

        if found {
            Entry::Occupied(OccupiedEntry {
                node: current,
                precursors,
                map: self,
            })
        } else {
            Entry::Vacant(VacantEntry {
                key,
                current,
                current_rank,
                precursors,
                precursor_distances,
                map: self,
            })
        }
    }

    /// Tries to insert a key-value pair, returning a mutable reference to
    /// the new value on success.
    ///
    /// If the key is already present, an [`OccupiedError`] containing the
    /// existing [`OccupiedEntry`] and the `value` that was not inserted is
    /// returned.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Errors
    ///
    /// Returns [`Err(OccupiedError)`][OccupiedError] when the key already
    /// exists in the map.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// assert!(map.try_insert(1, "a").is_ok());
    /// let err = map.try_insert(1, "b").unwrap_err();
    /// assert_eq!(err.value, "b");
    /// assert_eq!(err.entry.get(), &"a");
    /// ```
    #[inline]
    pub fn try_insert(
        &mut self,
        key: K,
        value: V,
    ) -> Result<&mut V, OccupiedError<'_, K, V, N, C, G>> {
        match self.entry(key) {
            Entry::Occupied(entry) => Err(OccupiedError { entry, value }),
            Entry::Vacant(entry) => Ok(entry.insert(value)),
        }
    }

    /// Returns an [`OccupiedEntry`] for the minimum-key entry, or `None` if
    /// the map is empty.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// assert!(map.first_entry().is_none());
    /// map.insert(3, "c");
    /// map.insert(1, "a");
    /// let entry = map.first_entry().unwrap();
    /// assert_eq!(entry.key(), &1);
    /// assert_eq!(entry.remove(), "a");
    /// assert_eq!(map.first_key_value(), Some((&3, &"c")));
    /// ```
    #[inline]
    pub fn first_entry(&mut self) -> Option<OccupiedEntry<'_, K, V, N, C, G>> {
        let (node, max_levels) = {
            // SAFETY: self.head is valid for &mut self's lifetime.
            let head_ref = unsafe { self.head.as_ref() };
            (head_ref.next()?, head_ref.level())
        };
        // The first data node's precursor at every skip level is the head
        // sentinel: head.links[l] for l < first_height points directly to
        // the first node with distance 1, and for l >= first_height head's
        // link spans over it (or is absent).  The splice logic in
        // remove_entry handles both cases correctly with head as the precursor.
        let precursors = iter::repeat_n(self.head, max_levels).collect();
        Some(OccupiedEntry {
            node,
            precursors,
            map: self,
        })
    }

    /// Returns an [`OccupiedEntry`] for the maximum-key entry, or `None` if
    /// the map is empty.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// assert!(map.last_entry().is_none());
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// let entry = map.last_entry().unwrap();
    /// assert_eq!(entry.key(), &3);
    /// assert_eq!(entry.remove(), "c");
    /// assert_eq!(map.last_key_value(), Some((&1, &"a")));
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "l is in 0..max_levels; precursors[l] is always in bounds (len == \
                  max_levels); links_mut()[l] is safe because the precursor at level l \
                  is always a node with height > l: every node advanced to via a level-l \
                  link has height >= l+1, and head (the fallback when no advance occurs) \
                  has height max_levels > any valid l"
    )]
    #[inline]
    pub fn last_entry(&mut self) -> Option<OccupiedEntry<'_, K, V, N, C, G>> {
        let tail_ptr = self.tail?;
        // SAFETY: tail_ptr is a valid data node for &mut self's lifetime.
        let tail_height = unsafe { tail_ptr.as_ref() }.level();
        // SAFETY: self.head is valid for &mut self's lifetime.
        let max_levels = unsafe { self.head.as_ref() }.level();

        // Initialise all precursor entries to head; the single traversal
        // below overwrites them all.
        let mut precursors: ArrayVec<NonNull<Node<(K, V), N>>, N> =
            iter::repeat_n(self.head, max_levels).collect();
        let mut current = self.head;

        // Traverse all levels from the highest to the lowest, advancing
        // `current` forward at each level until we can go no further.
        //
        // * For l >= tail_height: the tail has no tower slot at those levels,
        //   so no link can point to it.  We advance to the end of the level-l
        //   chain (link is None) and record the last node there.  Its link[l]
        //   is None, so remove_entry's distance-decrement branch is skipped.
        //
        // * For l < tail_height: we stop when the level-l link points
        //   directly to the tail.  This node is the correct level-l precursor
        //   for the bridging step in remove_entry.
        //
        // Invariant: at the start of each outer-loop iteration, current.level()
        // > l, so links.get(l) is always in bounds and links_mut()[l] in
        // remove_entry is always safe.
        for l in (0..max_levels).rev() {
            loop {
                // SAFETY: current is a valid node in this list, live for
                // the duration of &mut self.  No exclusive reference exists.
                let maybe_link = unsafe { current.as_ref() }
                    .links()
                    .get(l)
                    .and_then(|lk| lk.as_ref());
                match maybe_link {
                    None => break,
                    Some(link) if l < tail_height && link.node() == tail_ptr => break,
                    Some(link) => current = link.node(),
                }
            }
            precursors[l] = current;
        }

        Some(OccupiedEntry {
            node: tail_ptr,
            precursors,
            map: self,
        })
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::SkipMap;
    use crate::skip_map::entry::Entry;

    // MARK: entry: occupied / vacant

    #[test]
    fn entry_occupied_on_existing_key() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        assert!(matches!(map.entry(1), Entry::Occupied(_)));
    }

    #[test]
    fn entry_vacant_on_missing_key() {
        let mut map = SkipMap::<i32, &str>::new();
        assert!(matches!(map.entry(1), Entry::Vacant(_)));
    }

    // MARK: OccupiedEntry::key / get / get_mut / into_mut

    #[test]
    fn occupied_entry_key() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(42, "x");
        if let Entry::Occupied(e) = map.entry(42) {
            assert_eq!(e.key(), &42);
        } else {
            panic!("expected Occupied");
        }
    }

    #[test]
    fn occupied_entry_get() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        if let Entry::Occupied(e) = map.entry(1) {
            assert_eq!(e.get(), &10);
        } else {
            panic!("expected Occupied");
        }
    }

    #[test]
    fn occupied_entry_get_mut() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        if let Entry::Occupied(mut e) = map.entry(1) {
            *e.get_mut() += 5;
            assert_eq!(e.get(), &15);
        } else {
            panic!("expected Occupied");
        }
        assert_eq!(map.get(&1), Some(&15));
    }

    #[test]
    fn occupied_entry_into_mut() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        let v = match map.entry(1) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(_) => panic!("expected Occupied"),
        };
        *v += 5;
        assert_eq!(map.get(&1), Some(&15));
    }

    // MARK: OccupiedEntry::insert (replace value)

    #[test]
    fn occupied_entry_insert_returns_old_value() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        if let Entry::Occupied(mut e) = map.entry(1) {
            assert_eq!(e.insert("b"), "a");
        } else {
            panic!("expected Occupied");
        }
        assert_eq!(map.get(&1), Some(&"b"));
    }

    #[test]
    fn occupied_insert_mut() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        if let Entry::Occupied(mut e) = map.entry(1) {
            assert_eq!(e.insert("b"), "a");
            // Entry is still usable after first insert.
            assert_eq!(e.insert("c"), "b");
            assert_eq!(e.get(), &"c");
        } else {
            panic!("expected Occupied");
        }
        assert_eq!(map.get(&1), Some(&"c"));
    }

    // MARK: OccupiedEntry::remove / remove_entry

    #[test]
    fn occupied_entry_remove() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        if let Entry::Occupied(e) = map.entry(1) {
            assert_eq!(e.remove(), "a");
        } else {
            panic!("expected Occupied");
        }
        assert!(map.is_empty());
    }

    #[test]
    fn occupied_entry_remove_entry() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(7, "seven");
        if let Entry::Occupied(e) = map.entry(7) {
            assert_eq!(e.remove_entry(), (7, "seven"));
        } else {
            panic!("expected Occupied");
        }
        assert!(map.is_empty());
    }

    #[test]
    fn occupied_entry_remove_first_of_many() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 1..=5_i32 {
            map.insert(i, i * 10);
        }
        if let Entry::Occupied(e) = map.entry(1) {
            assert_eq!(e.remove(), 10);
        }
        assert_eq!(map.len(), 4);
        assert_eq!(map.first_key_value(), Some((&2, &20)));
    }

    #[test]
    fn occupied_entry_remove_last_of_many() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 1..=5_i32 {
            map.insert(i, i * 10);
        }
        if let Entry::Occupied(e) = map.entry(5) {
            assert_eq!(e.remove(), 50);
        }
        assert_eq!(map.len(), 4);
        assert_eq!(map.last_key_value(), Some((&4, &40)));
    }

    #[test]
    fn occupied_entry_remove_middle_of_many() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 1..=5_i32 {
            map.insert(i, i * 10);
        }
        if let Entry::Occupied(e) = map.entry(3) {
            assert_eq!(e.remove(), 30);
        }
        assert_eq!(map.len(), 4);
        assert_eq!(map.get(&3), None);
    }

    // MARK: VacantEntry::key / into_key / insert

    #[test]
    fn vacant_entry_key() {
        let mut map = SkipMap::<i32, &str>::new();
        if let Entry::Vacant(e) = map.entry(99) {
            assert_eq!(e.key(), &99);
        } else {
            panic!("expected Vacant");
        }
    }

    #[test]
    fn vacant_entry_into_key() {
        let mut map = SkipMap::<i32, &str>::new();
        if let Entry::Vacant(e) = map.entry(42) {
            assert_eq!(e.into_key(), 42);
        } else {
            panic!("expected Vacant");
        }
        assert!(map.is_empty());
    }

    #[test]
    fn vacant_entry_insert() {
        let mut map = SkipMap::<i32, &str>::new();
        if let Entry::Vacant(e) = map.entry(1) {
            let v = e.insert("hello");
            assert_eq!(*v, "hello");
        } else {
            panic!("expected Vacant");
        }
        assert_eq!(map.get(&1), Some(&"hello"));
    }

    #[test]
    fn vacant_entry_insert_into_empty_map() {
        let mut map = SkipMap::<i32, i32>::new();
        if let Entry::Vacant(e) = map.entry(5) {
            e.insert(50);
        }
        assert_eq!(map.len(), 1);
        assert_eq!(map.first_key_value(), Some((&5, &50)));
        assert_eq!(map.last_key_value(), Some((&5, &50)));
    }

    #[test]
    fn vacant_insert_entry_returns_occupied() {
        let mut map = SkipMap::<i32, &str>::new();
        if let Entry::Vacant(e) = map.entry(7) {
            let occupied = e.insert_entry("hello");
            assert_eq!(occupied.key(), &7);
            assert_eq!(occupied.get(), &"hello");
        } else {
            panic!("expected Vacant");
        }
        assert_eq!(map.get(&7), Some(&"hello"));
    }

    #[test]
    fn vacant_entry_insert_new_maximum() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        map.insert(2, 20);
        if let Entry::Vacant(e) = map.entry(5) {
            e.insert(50);
        }
        assert_eq!(map.last_key_value(), Some((&5, &50)));
    }

    // MARK: Entry convenience methods

    #[test]
    fn entry_or_insert_vacant() {
        let mut map = SkipMap::<i32, i32>::new();
        let v = map.entry(1).or_insert(10);
        assert_eq!(*v, 10);
        assert_eq!(map.get(&1), Some(&10));
    }

    #[test]
    fn entry_or_insert_occupied() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        let v = map.entry(1).or_insert(99);
        assert_eq!(*v, 10);
        assert_eq!(map.get(&1), Some(&10));
    }

    #[test]
    fn entry_or_insert_with_vacant() {
        let mut map = SkipMap::<i32, String>::new();
        map.entry(1).or_insert_with(|| "hello".to_owned());
        assert_eq!(map.get(&1).map(String::as_str), Some("hello"));
    }

    #[test]
    fn entry_or_insert_with_occupied() {
        let mut map = SkipMap::<i32, String>::new();
        map.insert(1, "original".to_owned());
        map.entry(1).or_insert_with(|| "new".to_owned());
        assert_eq!(map.get(&1).map(String::as_str), Some("original"));
    }

    #[test]
    fn entry_or_insert_with_key_vacant() {
        let mut map = SkipMap::<i32, i32>::new();
        map.entry(5).or_insert_with_key(|k| *k * 10);
        assert_eq!(map.get(&5), Some(&50));
    }

    #[test]
    fn entry_or_insert_with_key_occupied() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(5, 99);
        map.entry(5).or_insert_with_key(|k| *k * 10);
        assert_eq!(map.get(&5), Some(&99));
    }

    #[test]
    fn entry_and_modify_occupied() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        map.entry(1).and_modify(|v| *v += 1).or_insert(0);
        assert_eq!(map.get(&1), Some(&11));
    }

    #[test]
    fn entry_and_modify_vacant() {
        let mut map = SkipMap::<i32, i32>::new();
        map.entry(2).and_modify(|v| *v += 1).or_insert(0);
        assert_eq!(map.get(&2), Some(&0));
    }

    #[test]
    fn entry_or_default_vacant() {
        let mut map = SkipMap::<i32, i32>::new();
        map.entry(1).or_default();
        assert_eq!(map.get(&1), Some(&0));
    }

    #[test]
    fn entry_or_default_occupied() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 42);
        map.entry(1).or_default();
        assert_eq!(map.get(&1), Some(&42));
    }

    #[test]
    fn entry_key_occupied() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        assert_eq!(map.entry(1).key(), &1);
    }

    #[test]
    fn entry_key_vacant() {
        let mut map = SkipMap::<i32, &str>::new();
        assert_eq!(map.entry(99).key(), &99);
    }

    // MARK: Entry::insert_entry

    #[test]
    fn entry_insert_entry_vacant() {
        let mut map = SkipMap::<i32, &str>::new();
        let e = map.entry(1).insert_entry("a");
        assert_eq!(e.key(), &1);
        assert_eq!(e.get(), &"a");
        assert_eq!(map.get(&1), Some(&"a"));
    }

    #[test]
    fn entry_insert_entry_occupied() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        let e = map.entry(1).insert_entry("b");
        assert_eq!(e.key(), &1);
        assert_eq!(e.get(), &"b");
        assert_eq!(map.get(&1), Some(&"b"));
    }

    // MARK: try_insert

    #[test]
    fn try_insert_absent_key() {
        let mut map = SkipMap::<i32, &str>::new();
        assert_eq!(
            *map.try_insert(1, "a").expect("absent key should succeed"),
            "a"
        );
        assert_eq!(map.get(&1), Some(&"a"));
    }

    #[test]
    fn try_insert_present_key_returns_error() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        let Err(err) = map.try_insert(1, "b") else {
            panic!("expected Err from try_insert on occupied key");
        };
        assert_eq!(err.value, "b");
        assert_eq!(err.entry.get(), &"a");
    }

    #[test]
    fn try_insert_error_entry_can_update() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        let Err(err) = map.try_insert(1, 99) else {
            panic!("expected Err from try_insert on occupied key");
        };
        *err.entry.into_mut() += 5;
        assert_eq!(map.get(&1), Some(&15));
    }

    // MARK: first_entry

    #[test]
    fn first_entry_empty_is_none() {
        let mut map = SkipMap::<i32, &str>::new();
        assert!(map.first_entry().is_none());
    }

    #[test]
    fn first_entry_single_element() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(42, "x");
        let e = map.first_entry().expect("non-empty");
        assert_eq!(e.key(), &42);
        assert_eq!(e.get(), &"x");
    }

    #[test]
    fn first_entry_remove_updates_map() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(3, 30);
        map.insert(1, 10);
        map.insert(2, 20);
        let e = map.first_entry().expect("non-empty");
        assert_eq!(e.key(), &1);
        assert_eq!(e.remove(), 10);
        assert_eq!(map.len(), 2);
        assert_eq!(map.first_key_value(), Some((&2, &20)));
    }

    #[test]
    fn first_entry_remove_only_element() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        assert_eq!(map.first_entry().expect("non-empty").remove(), "a");
        assert!(map.is_empty());
        assert_eq!(map.first_key_value(), None);
        assert_eq!(map.last_key_value(), None);
    }

    // MARK: last_entry

    #[test]
    fn last_entry_empty_is_none() {
        let mut map = SkipMap::<i32, &str>::new();
        assert!(map.last_entry().is_none());
    }

    #[test]
    fn last_entry_single_element() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(42, "x");
        let e = map.last_entry().expect("non-empty");
        assert_eq!(e.key(), &42);
        assert_eq!(e.get(), &"x");
    }

    #[test]
    fn last_entry_remove_updates_map() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        map.insert(3, 30);
        map.insert(2, 20);
        let e = map.last_entry().expect("non-empty");
        assert_eq!(e.key(), &3);
        assert_eq!(e.remove(), 30);
        assert_eq!(map.len(), 2);
        assert_eq!(map.last_key_value(), Some((&2, &20)));
    }

    #[test]
    fn last_entry_remove_only_element() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        assert_eq!(map.last_entry().expect("non-empty").remove(), "a");
        assert!(map.is_empty());
        assert_eq!(map.first_key_value(), None);
        assert_eq!(map.last_key_value(), None);
    }

    // MARK: first_entry / last_entry: type bounds check (OccupiedEntry not bound to Debug)

    #[test]
    fn occupied_entry_remove_entry_large_map() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 0..20_i32 {
            map.insert(i, i * 10);
        }
        for i in (0..20_i32).step_by(2) {
            if let Entry::Occupied(e) = map.entry(i) {
                assert_eq!(e.remove(), i * 10);
            }
        }
        assert_eq!(map.len(), 10);
        for i in (1..20_i32).step_by(2) {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
        for i in (0..20_i32).step_by(2) {
            assert_eq!(map.get(&i), None);
        }
    }
}
