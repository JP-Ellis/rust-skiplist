//! Standard trait implementations for [`SkipMap`](super::SkipMap):
//! `Debug`, `Clone`, `PartialEq`, `Eq`, `Hash`, `Extend`, `FromIterator`.

use core::{
    fmt,
    hash::{Hash, Hasher},
    ptr::NonNull,
};

use super::SkipMap;
use crate::{comparator::Comparator, level_generator::LevelGenerator, node::Node};

// MARK: Debug

impl<K: fmt::Debug, V: fmt::Debug, C: Comparator<K>, G: LevelGenerator, const N: usize> fmt::Debug
    for SkipMap<K, V, N, C, G>
{
    /// Formats the map as `{k1: v1, k2: v2, ...}` in ascending key order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(2, "b");
    /// map.insert(1, "a");
    /// assert_eq!(format!("{map:?}"), r#"{1: "a", 2: "b"}"#);
    /// ```
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

// MARK: Clone

impl<K: Clone, V: Clone, C: Comparator<K> + Clone, G: LevelGenerator + Clone, const N: usize> Clone
    for SkipMap<K, V, N, C, G>
{
    /// Returns a deep clone of the map.
    ///
    /// The cloned map contains the same key-value pairs in the same sorted
    /// order. The comparator and level generator are both cloned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    ///
    /// let cloned = map.clone();
    /// assert_eq!(map, cloned);
    ///
    /// // Mutations to the clone do not affect the original.
    /// // (Illustrated by checking lengths differ after inserting into clone.)
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "`value()` returns None only for the head sentinel, which is never visited \
                  in the data-node walk; the expect fires only on invariant violations"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "insert_after and rebuild touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    fn clone(&self) -> Self {
        let max_levels = self.head_ref().level();

        // SAFETY: Box::into_raw transfers ownership; freed in Drop.
        let new_head =
            unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(max_levels)))) };
        let mut new_map = Self {
            head: new_head,
            tail: None,
            len: self.len,
            comparator: self.comparator.clone(),
            generator: self.generator.clone(),
        };

        if self.is_empty() {
            return new_map;
        }

        // Walk self's sequential next chain, cloning each node at the same
        // tower height. insert_after wires the prev/next chain; rebuild()
        // will wire all skip links in a single subsequent pass.
        //
        // SAFETY: self.head is a valid, live head sentinel. Every src_nn is a
        // live data node owned by self. new_head / prev_nn are exclusively
        // owned by new_map and have no other live references.
        let tail = unsafe {
            let mut prev_nn = new_head;
            let mut src_opt = self.head.as_ref().next();

            while let Some(src_nn) = src_opt {
                let src_node = src_nn.as_ref();
                let height = src_node.level();
                let kv = src_node.value().expect("data node has a value").clone();
                // insert_after requires the inserted node to be detached;
                // Node::with_value creates a detached node (prev=None, next=None,
                // all links=None).
                prev_nn = Node::insert_after(prev_nn, Node::with_value(height, kv));
                src_opt = src_node.next();
            }

            Node::rebuild(new_head)
        };
        new_map.tail = tail;

        new_map
    }
}

// MARK: PartialEq / Eq

impl<
    K: PartialEq,
    V: PartialEq,
    C1: Comparator<K>,
    C2: Comparator<K>,
    G1: LevelGenerator,
    G2: LevelGenerator,
    const N: usize,
> PartialEq<SkipMap<K, V, N, C2, G2>> for SkipMap<K, V, N, C1, G1>
{
    /// Returns `true` if both maps have the same length and all corresponding
    /// key-value pairs compare equal (in ascending key order).
    ///
    /// The comparators (`C1` and `C2`) and level generators do not need to
    /// match.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut a = SkipMap::<i32, &str>::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// let mut b = SkipMap::<i32, &str>::new();
    /// b.insert(2, "b");
    /// b.insert(1, "a");
    ///
    /// assert_eq!(a, b);
    ///
    /// b.insert(3, "c");
    /// assert_ne!(a, b);
    /// ```
    #[inline]
    fn eq(&self, other: &SkipMap<K, V, N, C2, G2>) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|((ak, av), (bk, bv))| ak == bk && av == bv)
    }
}

impl<K: Eq, V: Eq, C: Comparator<K>, G: LevelGenerator, const N: usize> Eq
    for SkipMap<K, V, N, C, G>
{
}

// MARK: Hash

impl<K: Hash, V: Hash, C: Comparator<K>, G: LevelGenerator, const N: usize> Hash
    for SkipMap<K, V, N, C, G>
{
    /// Hashes the length followed by each key-value pair in sorted order.
    ///
    /// Two maps with the same entries in the same sorted order produce
    /// identical hashes. Maps that compare equal via [`PartialEq`] will
    /// always produce the same hash.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use core::hash::{Hash, Hasher};
    /// use std::collections::hash_map::DefaultHasher;
    /// use skiplist::skip_map::SkipMap;
    ///
    /// fn hash_of<T: Hash>(value: &T) -> u64 {
    ///     let mut h = DefaultHasher::new();
    ///     value.hash(&mut h);
    ///     h.finish()
    /// }
    ///
    /// let mut a = SkipMap::<i32, i32>::new();
    /// a.insert(1, 10);
    /// a.insert(2, 20);
    ///
    /// let mut b = SkipMap::<i32, i32>::new();
    /// b.insert(2, 20);
    /// b.insert(1, 10);
    ///
    /// // Equal maps produce the same hash regardless of insertion order.
    /// assert_eq!(hash_of(&a), hash_of(&b));
    /// ```
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for (k, v) in self {
            k.hash(state);
            v.hash(state);
        }
    }
}

// MARK: Extend

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> Extend<(K, V)>
    for SkipMap<K, V, N, C, G>
{
    /// Inserts all key-value pairs from `iter` into the map.
    ///
    /// Each pair is inserted at its sorted key position. When a key already
    /// exists, the existing value is replaced with the new value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.extend([(3, "c"), (1, "a"), (2, "b")]);
    ///
    /// let keys: Vec<i32> = map.keys().copied().collect();
    /// assert_eq!(keys, [1, 2, 3]);
    ///
    /// // Extending with a duplicate key replaces the existing value.
    /// map.extend([(1, "updated")]);
    /// assert_eq!(map.get(&1), Some(&"updated"));
    /// ```
    #[inline]
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<'a, K: Copy + 'a, V: Copy + 'a, C: Comparator<K>, G: LevelGenerator, const N: usize>
    Extend<(&'a K, &'a V)> for SkipMap<K, V, N, C, G>
{
    /// Copies all key-value reference pairs from `iter` and inserts them into
    /// the map.
    ///
    /// This is a convenience overload for iterators that yield `(&K, &V)` when
    /// both `K` and `V` implement [`Copy`]. When a key already exists, the
    /// existing value is replaced.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// let pairs = [(1, 10), (2, 20), (3, 30)];
    /// map.extend(pairs.iter().map(|(k, v)| (k, v)));
    ///
    /// assert_eq!(map.len(), 3);
    /// assert_eq!(map.get(&2), Some(&20));
    /// ```
    #[inline]
    fn extend<I: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: I) {
        self.extend(iter.into_iter().map(|(&k, &v)| (k, v)));
    }
}

// MARK: FromIterator

impl<K, V, C: Comparator<K> + Default, G: LevelGenerator + Default, const N: usize>
    FromIterator<(K, V)> for SkipMap<K, V, N, C, G>
{
    /// Creates a map from an iterator of key-value pairs.
    ///
    /// Pairs are inserted in their sorted key order regardless of the
    /// iteration order of the source. The map uses the default comparator and
    /// level generator for the type parameters `C` and `G`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let pairs = [(3, "c"), (1, "a"), (2, "b")];
    /// let map: SkipMap<i32, &str> = pairs.into_iter().collect();
    /// let keys: Vec<i32> = map.keys().copied().collect();
    /// assert_eq!(keys, [1, 2, 3]);
    /// ```
    #[inline]
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Self::with_comparator_and_level_generator(C::default(), G::default());
        map.extend(iter);
        map
    }
}

#[cfg(test)]
mod tests {
    use core::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    use pretty_assertions::{assert_eq, assert_ne};

    use super::super::SkipMap;
    use crate::comparator::FnComparator;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut h = DefaultHasher::new();
        value.hash(&mut h);
        h.finish()
    }

    // MARK: Debug

    #[test]
    fn debug_empty() {
        let map = SkipMap::<i32, i32>::new();
        assert_eq!(format!("{map:?}"), "{}");
    }

    #[test]
    fn debug_single() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        assert_eq!(format!("{map:?}"), r#"{1: "a"}"#);
    }

    #[test]
    fn debug_multiple_sorted_order() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(3, "c");
        map.insert(1, "a");
        map.insert(2, "b");
        assert_eq!(format!("{map:?}"), r#"{1: "a", 2: "b", 3: "c"}"#);
    }

    // MARK: Clone

    #[test]
    fn clone_empty() {
        let map = SkipMap::<i32, i32>::new();
        let cloned = map.clone();
        assert!(cloned.is_empty());
    }

    #[test]
    fn clone_elements() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(3, "c");
        map.insert(1, "a");
        map.insert(2, "b");
        let cloned = map.clone();
        let kvs: Vec<(i32, &str)> = cloned.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(kvs, [(1, "a"), (2, "b"), (3, "c")]);
    }

    #[test]
    fn clone_is_independent() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(10, 100);
        map.insert(20, 200);
        let mut cloned = map.clone();
        cloned.insert(30, 300);
        assert_eq!(map.len(), 2);
        assert_eq!(cloned.len(), 3);
    }

    #[test]
    fn clone_preserves_comparator() {
        let mut map: SkipMap<i32, &str, 16, _> =
            SkipMap::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        map.insert(1, "a");
        map.insert(3, "c");
        map.insert(2, "b");
        let cloned = map.clone();
        let keys: Vec<i32> = cloned.keys().copied().collect();
        assert_eq!(keys, [3, 2, 1]);
    }

    // MARK: PartialEq / Eq

    #[test]
    fn eq_empty_maps() {
        let a = SkipMap::<i32, i32>::new();
        let b = SkipMap::<i32, i32>::new();
        assert_eq!(a, b);
    }

    #[test]
    fn eq_same_entries() {
        let mut a = SkipMap::<i32, &str>::new();
        a.insert(1, "a");
        a.insert(2, "b");
        let mut b = SkipMap::<i32, &str>::new();
        b.insert(2, "b");
        b.insert(1, "a");
        assert_eq!(a, b);
    }

    #[test]
    fn ne_different_lengths() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        let b = SkipMap::<i32, i32>::new();
        assert_ne!(a, b);
    }

    #[test]
    fn ne_different_values() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        let mut b = SkipMap::<i32, i32>::new();
        b.insert(1, 99);
        assert_ne!(a, b);
    }

    #[test]
    fn ne_different_keys() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        let mut b = SkipMap::<i32, i32>::new();
        b.insert(2, 10);
        assert_ne!(a, b);
    }

    // MARK: Hash

    #[test]
    fn hash_empty_maps_equal() {
        let a = SkipMap::<i32, i32>::new();
        let b = SkipMap::<i32, i32>::new();
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn hash_equal_maps_equal() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        a.insert(2, 20);
        let mut b = SkipMap::<i32, i32>::new();
        b.insert(2, 20);
        b.insert(1, 10);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn hash_different_maps_likely_unequal() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        let mut b = SkipMap::<i32, i32>::new();
        b.insert(1, 20);
        assert_ne!(hash_of(&a), hash_of(&b));
    }

    // MARK: Extend

    #[test]
    fn extend_owned_pairs() {
        let mut map = SkipMap::<i32, &str>::new();
        map.extend([(3, "c"), (1, "a"), (2, "b")]);
        let keys: Vec<i32> = map.keys().copied().collect();
        assert_eq!(keys, [1, 2, 3]);
    }

    #[test]
    fn extend_ref_pairs() {
        let mut map = SkipMap::<i32, i32>::new();
        let pairs = [(1, 10), (2, 20), (3, 30)];
        map.extend(pairs.iter().map(|(k, v)| (k, v)));
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&20));
    }

    #[test]
    fn extend_replaces_existing_value() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "old");
        map.extend([(1, "new")]);
        assert_eq!(map.get(&1), Some(&"new"));
        assert_eq!(map.len(), 1);
    }

    // MARK: FromIterator

    #[test]
    fn from_iter_empty() {
        let map: SkipMap<i32, i32> = core::iter::empty().collect();
        assert!(map.is_empty());
    }

    #[test]
    fn from_iter_sorted() {
        let pairs = [(3, "c"), (1, "a"), (2, "b")];
        let map: SkipMap<i32, &str> = pairs.into_iter().collect();
        let keys: Vec<i32> = map.keys().copied().collect();
        assert_eq!(keys, [1, 2, 3]);
    }

    #[test]
    fn from_iter_duplicates_replaced() {
        let pairs = [(1, "first"), (1, "second")];
        let map: SkipMap<i32, &str> = pairs.into_iter().collect();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"second"));
    }
}
