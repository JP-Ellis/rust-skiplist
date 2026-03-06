//! Iteration support for [`SkipMap`](super::SkipMap): `iter`, `iter_mut`,
//! `keys`, `values`, `values_mut`, `range`, `range_mut`, `drain`,
//! `extract_if`, all iterator types, and [`IntoIterator`] implementations.

use core::{
    cmp::Ordering,
    fmt,
    iter::FusedIterator,
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    ptr::NonNull,
};

use crate::{
    comparator::{Comparator, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    node::{
        Node,
        visitor::{OrdVisitor, Visitor},
    },
};

use super::SkipMap;

// MARK: Private helpers

/// Counts the elements between `front` and `back` (both inclusive) by
/// following `next` links.  Returns `Some(count)` if `back` is reachable from
/// `front`, or `None` if the chain ends without encountering `back`.
///
/// # Safety
///
/// `front` and `back` must be valid, live `Node` pointers in the same skip
/// map, both of which remain valid for the duration of this call.
unsafe fn count_inclusive<T, const N: usize>(
    front: NonNull<Node<T, N>>,
    back: NonNull<Node<T, N>>,
) -> Option<usize> {
    let mut count: usize = 0;
    let mut ptr = front;
    loop {
        count = count.saturating_add(1);
        if ptr == back {
            return Some(count);
        }
        // SAFETY: ptr is a live node pointer; caller guarantees validity.
        ptr = unsafe { ptr.as_ref() }.next()?;
    }
}

impl<K, V, const N: usize, C: Comparator<K>, G: LevelGenerator> SkipMap<K, V, N, C, G> {
    /// Returns a raw pointer to the first node satisfying the lower bound, or
    /// `None` when no such node exists.
    ///
    /// - Unbounded   : first data node (head's successor)
    /// - Included(k) : first node with key >= k
    /// - Excluded(k) : first node with key > k
    fn lower_bound_ptr(&self, bound: Bound<&K>) -> Option<NonNull<Node<(K, V), N>>> {
        match bound {
            Bound::Unbounded => {
                // SAFETY: self.head is always valid for &self.
                unsafe { self.head.as_ref() }.next()
            }
            Bound::Included(k) => {
                // Advance only through nodes strictly < k; stop as soon as we
                // hit a node == k or > k.  After traversal, current() is the
                // last node < k (or the head sentinel when all nodes >= k), so
                // current().next() is the first node >= k.
                let head = self.head_ref();
                let cmp = |entry: &(K, V), target: &K| {
                    match self.comparator.compare(&entry.0, target) {
                        Ordering::Less => Ordering::Less,
                        // Treat Equal and Greater the same: do not advance past them.
                        Ordering::Equal | Ordering::Greater => Ordering::Greater,
                    }
                };
                let mut visitor = OrdVisitor::new(head, k, cmp);
                visitor.traverse();
                visitor.current().next()
            }
            Bound::Excluded(k) => {
                // Advance through nodes <= k; stop just before > k.
                // After traversal, current() is the last node <= k (or head),
                // so current().next() is the first node > k.
                let head = self.head_ref();
                let cmp =
                    |entry: &(K, V), target: &K| match self.comparator.compare(&entry.0, target) {
                        Ordering::Less | Ordering::Equal => Ordering::Less,
                        Ordering::Greater => Ordering::Greater,
                    };
                let mut visitor = OrdVisitor::new(head, k, cmp);
                visitor.traverse();
                visitor.current().next()
            }
        }
    }

    /// Returns a raw pointer to the last node satisfying the upper bound, or
    /// `None` when no such node exists.
    ///
    /// - Unbounded   : tail (last data node)
    /// - Included(k) : last node with key <= k
    /// - Excluded(k) : last node with key < k
    fn upper_bound_ptr(&self, bound: Bound<&K>) -> Option<NonNull<Node<(K, V), N>>> {
        match bound {
            Bound::Unbounded => self.tail,
            Bound::Included(k) => {
                // Advance through all nodes <= k; stop before > k.
                // After traversal, current() is the last node <= k (or head).
                let head = self.head_ref();
                let cmp =
                    |entry: &(K, V), target: &K| match self.comparator.compare(&entry.0, target) {
                        Ordering::Less | Ordering::Equal => Ordering::Less,
                        Ordering::Greater => Ordering::Greater,
                    };
                let mut visitor = OrdVisitor::new(head, k, cmp);
                visitor.traverse();
                let current = visitor.current();
                // Return None when the sentinel head has no value (all nodes > k).
                // NonNull::from a shared reference is safe: no unsafe needed.
                current.value().is_some().then(|| NonNull::from(current))
            }
            Bound::Excluded(k) => {
                // Advance only through nodes strictly < k.
                // After traversal, current() is the last node < k (or head).
                let head = self.head_ref();
                let cmp =
                    |entry: &(K, V), target: &K| match self.comparator.compare(&entry.0, target) {
                        Ordering::Less => Ordering::Less,
                        Ordering::Equal | Ordering::Greater => Ordering::Greater,
                    };
                let mut visitor = OrdVisitor::new(head, k, cmp);
                visitor.traverse();
                let current = visitor.current();
                current.value().is_some().then(|| NonNull::from(current))
            }
        }
    }

    // MARK: Iteration methods

    /// Returns an iterator over shared key-value references, yielded in key
    /// order from smallest to largest.
    ///
    /// The iterator also supports [`DoubleEndedIterator`], allowing traversal
    /// in reverse order.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(3, "c");
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    ///
    /// let pairs: Vec<_> = map.iter().collect();
    /// assert_eq!(pairs, [(&1, &"a"), (&2, &"b"), (&3, &"c")]);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V, N> {
        Iter {
            // SAFETY: self.head is a valid, exclusively-owned head sentinel.
            front: unsafe { self.head.as_ref() }.next(),
            back: self.tail,
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over shared key and mutable value references,
    /// yielded in key order.
    ///
    /// Only the value may be mutated through this iterator; the key is always
    /// shared (`&K`) to preserve the ordering invariant.
    ///
    /// The iterator also supports [`DoubleEndedIterator`].
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// map.insert(3, 30);
    /// map.insert(1, 10);
    /// map.insert(2, 20);
    ///
    /// for (_k, v) in map.iter_mut() {
    ///     *v *= 2;
    /// }
    ///
    /// let pairs: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
    /// assert_eq!(pairs, [(1, 20), (2, 40), (3, 60)]);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, N> {
        IterMut {
            // SAFETY: self.head is a valid, exclusively-owned head sentinel.
            front: unsafe { self.head.as_ref().next() },
            back: self.tail,
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over shared references to the keys of the map, in
    /// key order.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(3, "c");
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    ///
    /// let keys: Vec<i32> = map.keys().copied().collect();
    /// assert_eq!(keys, [1, 2, 3]);
    /// ```
    #[inline]
    pub fn keys(&self) -> Keys<'_, K, V, N> {
        Keys { inner: self.iter() }
    }

    /// Returns an iterator over shared references to the values of the map,
    /// in key order.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(3, "c");
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    ///
    /// let values: Vec<&&str> = map.values().collect();
    /// assert_eq!(values, [&"a", &"b", &"c"]);
    /// ```
    #[inline]
    pub fn values(&self) -> Values<'_, K, V, N> {
        Values { inner: self.iter() }
    }

    /// Returns an iterator over mutable references to the values of the map,
    /// in key order.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// map.insert(1, 10);
    /// map.insert(2, 20);
    ///
    /// for v in map.values_mut() {
    ///     *v += 1;
    /// }
    ///
    /// let values: Vec<i32> = map.values().copied().collect();
    /// assert_eq!(values, [11, 21]);
    /// ```
    #[inline]
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V, N> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }

    /// Consumes the map and returns an iterator over the keys in key order.
    ///
    /// This operation is `$O(1)$` to construct.  Iteration is `$O(n)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(3, "c");
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    ///
    /// let keys: Vec<i32> = map.into_keys().collect();
    /// assert_eq!(keys, [1, 2, 3]);
    /// ```
    #[inline]
    pub fn into_keys(self) -> IntoKeys<K, V, N, C, G> {
        IntoKeys {
            inner: IntoIter { list: self },
        }
    }

    /// Consumes the map and returns an iterator over the values in key order.
    ///
    /// This operation is `$O(1)$` to construct.  Iteration is `$O(n)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// map.insert(3, 30);
    /// map.insert(1, 10);
    /// map.insert(2, 20);
    ///
    /// let values: Vec<i32> = map.into_values().collect();
    /// assert_eq!(values, [10, 20, 30]);
    /// ```
    #[inline]
    pub fn into_values(self) -> IntoValues<K, V, N, C, G> {
        IntoValues {
            inner: IntoIter { list: self },
        }
    }

    /// Returns an iterator over shared key-value references whose keys fall
    /// within the given bound range, in key order.
    ///
    /// Finding the start and end nodes is `$O(\log n)$`; counting elements in the
    /// range is `$O(k)$` where k is the number of elements in the range.
    ///
    /// # Panics
    ///
    /// Panics if the lower bound is greater than the upper bound according to
    /// the map's comparator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// for (k, v) in [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")] {
    ///     map.insert(k, v);
    /// }
    ///
    /// let slice: Vec<_> = map.range(2..=4).collect();
    /// assert_eq!(slice, [(&2, &"b"), (&3, &"c"), (&4, &"d")]);
    /// ```
    #[expect(
        clippy::panic,
        reason = "mirrors BTreeMap::range: panics on invalid range bounds, \
                  not on internal invariant violations"
    )]
    #[expect(
        clippy::shadow_reuse,
        reason = "rebinding front/back after bound-resolution helpers is the \
                  clearest way to express the narrowing from Option to the final triple"
    )]
    #[inline]
    pub fn range<R: RangeBounds<K>>(&self, range: R) -> Iter<'_, K, V, N> {
        let lo = range.start_bound();
        let hi = range.end_bound();

        // Mirrors the BTreeMap contract.
        if let (Bound::Included(a) | Bound::Excluded(a), Bound::Included(b) | Bound::Excluded(b)) =
            (lo, hi)
        {
            match self.comparator.compare(a, b) {
                Ordering::Greater => {
                    panic!("range start is after range end in SkipMap");
                }
                Ordering::Equal => {
                    assert!(
                        matches!((lo, hi), (Bound::Included(_), Bound::Included(_))),
                        "range start is after range end in SkipMap"
                    );
                }
                Ordering::Less => {}
            }
        }

        let front = self.lower_bound_ptr(lo);
        let back = self.upper_bound_ptr(hi);

        let (front, back, len) = match (front, back) {
            (None, _) | (_, None) => (None, None, 0),
            (Some(f), Some(b)) => {
                // SAFETY: f and b are valid node pointers in this map, live for
                // &self.  count_inclusive walks the next chain.
                match unsafe { count_inclusive(f, b) } {
                    Some(len) => (Some(f), Some(b), len),
                    None => (None, None, 0),
                }
            }
        };

        Iter {
            front,
            back,
            len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over shared-key / mutable-value references for
    /// pairs whose keys fall within the given bound range, in key order.
    ///
    /// Finding the start and end nodes is `$O(\log n)$`; counting elements in the
    /// range is `$O(k)$`.
    ///
    /// # Panics
    ///
    /// Panics if the lower bound is greater than the upper bound according to
    /// the map's comparator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// for (k, v) in [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)] {
    ///     map.insert(k, v);
    /// }
    ///
    /// for (_k, v) in map.range_mut(2..=4) {
    ///     *v *= 10;
    /// }
    ///
    /// let pairs: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
    /// assert_eq!(pairs, [(1, 10), (2, 200), (3, 300), (4, 400), (5, 50)]);
    /// ```
    #[expect(
        clippy::panic,
        reason = "mirrors BTreeMap::range_mut: panics on invalid range bounds"
    )]
    #[expect(
        clippy::shadow_reuse,
        reason = "rebinding front/back after bound-resolution is the clearest expression"
    )]
    #[inline]
    pub fn range_mut<R: RangeBounds<K>>(&mut self, range: R) -> IterMut<'_, K, V, N> {
        let lo = range.start_bound();
        let hi = range.end_bound();

        if let (Bound::Included(a) | Bound::Excluded(a), Bound::Included(b) | Bound::Excluded(b)) =
            (lo, hi)
        {
            match self.comparator.compare(a, b) {
                Ordering::Greater => {
                    panic!("range start is after range end in SkipMap");
                }
                Ordering::Equal => {
                    assert!(
                        matches!((lo, hi), (Bound::Included(_), Bound::Included(_))),
                        "range start is after range end in SkipMap"
                    );
                }
                Ordering::Less => {}
            }
        }

        // lower_bound_ptr and upper_bound_ptr are &self methods that return raw
        // pointers (no lingering borrow), so they compose fine with &mut self.
        let front = self.lower_bound_ptr(lo);
        let back = self.upper_bound_ptr(hi);

        let (front, back, len) = match (front, back) {
            (None, _) | (_, None) => (None, None, 0),
            (Some(f), Some(b)) => {
                // SAFETY: f and b are valid node pointers in this map, live for
                // &mut self.  count_inclusive walks the next chain.
                match unsafe { count_inclusive(f, b) } {
                    Some(len) => (Some(f), Some(b), len),
                    None => (None, None, 0),
                }
            }
        };

        IterMut {
            front,
            back,
            len,
            _marker: PhantomData,
        }
    }

    /// Removes all key-value pairs from the map and returns them as an
    /// iterator in key order.
    ///
    /// After the call the map is empty.  Elements are collected eagerly before
    /// the `Drain` is returned: the map is cleared regardless of whether the
    /// caller consumes the iterator.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(3, "c");
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    ///
    /// let drained: Vec<_> = map.drain().collect();
    /// assert_eq!(drained, [(1, "a"), (2, "b"), (3, "c")]);
    /// assert!(map.is_empty());
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "`take_value()` returns None only for the head sentinel, which is never \
                  in the drain range; the expect fires only on invariant violations, not \
                  on any user-observable code path"
    )]
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, K, V> {
        let mut drained: Vec<(K, V)> = Vec::with_capacity(self.len);

        // SAFETY: All raw pointers come from heap allocations owned by this
        // SkipMap.  We hold &mut self.  The keep closure returns false for every
        // node, so all nodes are dropped via the on_drop closure.
        let (new_len, new_tail) = unsafe {
            Node::filter_rebuild(
                self.head,
                |_cur| false,
                |mut boxed| {
                    drained.push(boxed.take_value().expect("data node has a value"));
                },
            )
        };
        self.tail = new_tail;
        self.len = new_len;

        Drain {
            iter: drained.into_iter(),
            _marker: PhantomData,
        }
    }

    /// Creates a lazy iterator that removes and yields every key-value pair
    /// for which `pred(&key, &mut value)` returns `true`.
    ///
    /// Pairs for which `pred` returns `false` are kept in the map.  The
    /// predicate receives a shared reference to the key and a mutable reference
    /// to the value.
    ///
    /// If the `ExtractIf` iterator is dropped before being fully consumed,
    /// the predicate is **not** called for the remaining entries; they all
    /// stay in the map.  The map remains valid and fully usable after the
    /// iterator is dropped.
    ///
    /// This operation is `$O(n)$` for a full traversal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, i32>::new();
    /// for (k, v) in [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)] {
    ///     map.insert(k, v);
    /// }
    ///
    /// let extracted: Vec<_> = map.extract_if(|_k, v| *v % 20 == 0).collect();
    /// assert_eq!(extracted, [(2, 20), (4, 40)]);
    ///
    /// let remaining: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
    /// assert_eq!(remaining, [(1, 10), (3, 30), (5, 50)]);
    /// ```
    #[inline]
    pub fn extract_if<F>(&mut self, pred: F) -> ExtractIf<'_, K, V, C, G, F, N>
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        // SAFETY: self.head is a valid, exclusively-owned head sentinel.
        let current = unsafe { self.head.as_ref().next() };
        ExtractIf {
            current,
            any_removed: false,
            list: self,
            pred,
        }
    }
}

// MARK: IntoIterator

impl<'a, K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> IntoIterator
    for &'a SkipMap<K, V, N, C, G>
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> IntoIterator
    for &'a mut SkipMap<K, V, N, C, G>
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> IntoIterator
    for SkipMap<K, V, N, C, G>
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, N, C, G>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { list: self }
    }
}

// MARK: Iter

/// An iterator over shared key-value references of a [`SkipMap`], yielded in
/// key order.
///
/// This struct is created by the [`SkipMap::iter`] and [`SkipMap::range`]
/// methods.  See their documentation for more.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let mut map = SkipMap::<i32, &str>::new();
/// map.insert(1, "a");
/// map.insert(2, "b");
/// map.insert(3, "c");
///
/// let mut iter = map.iter();
/// assert_eq!(iter.next(), Some((&1, &"a")));
/// assert_eq!(iter.next_back(), Some((&3, &"c")));
/// assert_eq!(iter.next(), Some((&2, &"b")));
/// assert_eq!(iter.next(), None);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Iter<'a, K, V, const N: usize = 16> {
    /// Pointer to the next element to yield from the front, or `None` when
    /// the iterator is exhausted or the map was empty.
    front: Option<NonNull<Node<(K, V), N>>>,
    /// Pointer to the next element to yield from the back, or `None` when
    /// the iterator is exhausted or the map was empty.
    back: Option<NonNull<Node<(K, V), N>>>,
    /// Number of elements remaining.  Guards against yielding more than
    /// `len` items even when `front` and `back` pointers cross mid-map
    /// during interleaved `next`/`next_back` calls.
    len: usize,
    /// Ties the iterator's lifetime to `&'a SkipMap` and expresses
    /// covariance in `K` and `V`.
    _marker: PhantomData<&'a (K, V)>,
}

// SAFETY: Iter<'a, K, V> yields `(&'a K, &'a V)` (shared references).
// Sending it to another thread requires K: Sync and V: Sync because the
// receiving thread will read K and V values through shared references.
unsafe impl<K: Sync, V: Sync, const N: usize> Send for Iter<'_, K, V, N> {}

// SAFETY: Sharing &Iter<'a, K, V> across threads is safe when K: Sync and
// V: Sync.  Advancing the iterator requires &mut Iter, preventing data races.
unsafe impl<K: Sync, V: Sync, const N: usize> Sync for Iter<'_, K, V, N> {}

impl<K, V, const N: usize> Clone for Iter<'_, K, V, N> {
    #[inline]
    fn clone(&self) -> Self {
        Iter {
            front: self.front,
            back: self.back,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug, const N: usize> fmt::Debug for Iter<'_, K, V, N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.clone()).finish()
    }
}

impl<'a, K, V, const N: usize> Iterator for Iter<'a, K, V, N> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let front_ptr = self.front?;
        // SAFETY: front_ptr was derived from a heap-allocated Node<(K,V), N>
        // owned by the SkipMap that created this Iter.  The iterator holds a
        // shared borrow of that map for lifetime 'a, ensuring every node
        // remains allocated and reachable for the iterator's entire lifetime.
        // No &mut references to any node exist while this shared Iter is alive.
        let node: &'a Node<(K, V), N> = unsafe { front_ptr.as_ref() };
        self.front = node.next();
        self.len = self.len.saturating_sub(1);
        let kv = node.value()?;
        Some((&kv.0, &kv.1))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for Iter<'a, K, V, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let back_ptr = self.back?;
        // SAFETY: Same provenance argument as front_ptr in next().
        // back_ptr points to a live data node for the 'a lifetime.
        let node: &'a Node<(K, V), N> = unsafe { back_ptr.as_ref() };
        // Walk backward.  The head sentinel has no value; the filter ensures
        // `back` becomes None when we step past the first data node.
        // `len` independently prevents accessing a stale `back` pointer.
        // SAFETY: prev() returns a valid pointer into the same map allocation.
        self.back = node
            .prev()
            .filter(|p| unsafe { p.as_ref() }.value().is_some());
        self.len = self.len.saturating_sub(1);
        let kv = node.value()?;
        Some((&kv.0, &kv.1))
    }
}

impl<K, V, const N: usize> ExactSizeIterator for Iter<'_, K, V, N> {}

impl<K, V, const N: usize> FusedIterator for Iter<'_, K, V, N> {}

// MARK: IterMut

/// An iterator over shared key and mutable value references of a [`SkipMap`],
/// yielded in key order.
///
/// This struct is created by the [`SkipMap::iter_mut`] and
/// [`SkipMap::range_mut`] methods.  See their documentation for more.
///
/// Unlike [`Iter`], `IterMut` does not implement [`Clone`]: cloning would
/// allow two independent iterators each holding `&mut V` references to the
/// same entries, violating Rust's aliasing rules.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let mut map = SkipMap::<i32, i32>::new();
/// map.insert(1, 10);
/// map.insert(2, 20);
/// map.insert(3, 30);
///
/// for (_k, v) in map.iter_mut() {
///     *v *= 2;
/// }
///
/// let pairs: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
/// assert_eq!(pairs, [(1, 20), (2, 40), (3, 60)]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IterMut<'a, K, V, const N: usize = 16> {
    /// Pointer to the next element to yield from the front.
    front: Option<NonNull<Node<(K, V), N>>>,
    /// Pointer to the next element to yield from the back.
    back: Option<NonNull<Node<(K, V), N>>>,
    /// Number of elements remaining.
    len: usize,
    /// Ties the iterator's lifetime to `&'a mut SkipMap` and expresses
    /// invariance in `K` and `V` (required for mutable references).
    _marker: PhantomData<&'a mut (K, V)>,
}

// SAFETY: IterMut<'a, K, V> yields `(&'a K, &'a mut V)`.  Sending it to
// another thread requires K: Send and V: Send.
unsafe impl<K: Send, V: Send, const N: usize> Send for IterMut<'_, K, V, N> {}

// SAFETY: Sharing &IterMut<'a, K, V> across threads is safe when K: Sync and
// V: Sync.  Advancing the iterator requires &mut IterMut.
unsafe impl<K: Sync, V: Sync, const N: usize> Sync for IterMut<'_, K, V, N> {}

impl<K: fmt::Debug, V: fmt::Debug, const N: usize> fmt::Debug for IterMut<'_, K, V, N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Traverse via shared references for display.  We hold &self, so no
        // mutable access is ongoing.
        let mut builder = f.debug_map();
        let mut node_ptr = self.front;
        let mut remaining = self.len;
        while remaining > 0 {
            let Some(ptr) = node_ptr else { break };
            // SAFETY: ptr is a valid, aligned pointer to a live Node<(K,V), N>
            // for lifetime 'a.  We only read through it here, and &self
            // prevents concurrent mutable access.
            let node = unsafe { ptr.as_ref() };
            if let Some(kv) = node.value() {
                builder.entry(&kv.0, &kv.1);
            }
            node_ptr = node.next();
            remaining = remaining.saturating_sub(1);
        }
        builder.finish()
    }
}

impl<'a, K, V, const N: usize> Iterator for IterMut<'a, K, V, N> {
    type Item = (&'a K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let mut front_ptr = self.front?;
        // SAFETY: front_ptr was derived from a heap-allocated Node<(K,V), N>
        // owned by the SkipMap that created this IterMut.  The iterator holds
        // an exclusive borrow of that map for lifetime 'a, ensuring every
        // node remains allocated and non-aliased.  We advance self.front
        // before returning, so no two calls can yield a reference to the same
        // node.
        let node: &'a mut Node<(K, V), N> = unsafe { front_ptr.as_mut() };
        self.front = node.next();
        self.len = self.len.saturating_sub(1);
        let kv: &'a mut (K, V) = node.value_mut()?;
        Some((&kv.0, &mut kv.1))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for IterMut<'a, K, V, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let mut back_ptr = self.back?;
        // SAFETY: Same provenance argument as front_ptr in next().
        // back_ptr points to a live data node for the 'a lifetime, and no
        // other mutable reference to it exists while this IterMut is alive.
        let node: &'a mut Node<(K, V), N> = unsafe { back_ptr.as_mut() };
        // Walk backward.  The head sentinel has no value; the filter ensures
        // `back` becomes None when we step past the first data node.
        // SAFETY: prev() returns a valid pointer into the same map allocation.
        self.back = node
            .prev()
            .filter(|p| unsafe { p.as_ref() }.value().is_some());
        self.len = self.len.saturating_sub(1);
        let kv: &'a mut (K, V) = node.value_mut()?;
        Some((&kv.0, &mut kv.1))
    }
}

impl<K, V, const N: usize> ExactSizeIterator for IterMut<'_, K, V, N> {}

impl<K, V, const N: usize> FusedIterator for IterMut<'_, K, V, N> {}

// MARK: Keys

/// An iterator over shared references to the keys of a [`SkipMap`], in key
/// order.
///
/// This struct is created by the [`SkipMap::keys`] method.  See its
/// documentation for more.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let mut map = SkipMap::<i32, &str>::new();
/// map.insert(3, "c");
/// map.insert(1, "a");
/// map.insert(2, "b");
///
/// let keys: Vec<i32> = map.keys().copied().collect();
/// assert_eq!(keys, [1, 2, 3]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Keys<'a, K, V, const N: usize = 16> {
    /// Underlying key-value iterator.
    inner: Iter<'a, K, V, N>,
}

// SAFETY: Keys delegates all access to Iter; same Send/Sync bounds apply.
unsafe impl<K: Sync, V: Sync, const N: usize> Send for Keys<'_, K, V, N> {}
// SAFETY: Keys delegates all access to Iter; same Send/Sync bounds apply.
unsafe impl<K: Sync, V: Sync, const N: usize> Sync for Keys<'_, K, V, N> {}

impl<K, V, const N: usize> Clone for Keys<'_, K, V, N> {
    #[inline]
    fn clone(&self) -> Self {
        Keys {
            inner: self.inner.clone(),
        }
    }
}

impl<K: fmt::Debug, V, const N: usize> fmt::Debug for Keys<'_, K, V, N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, K, V, const N: usize> Iterator for Keys<'a, K, V, N> {
    type Item = &'a K;

    #[inline]
    fn next(&mut self) -> Option<&'a K> {
        self.inner.next().map(|(k, _)| k)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for Keys<'a, K, V, N> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a K> {
        self.inner.next_back().map(|(k, _)| k)
    }
}

impl<K, V, const N: usize> ExactSizeIterator for Keys<'_, K, V, N> {}

impl<K, V, const N: usize> FusedIterator for Keys<'_, K, V, N> {}

// MARK: Values

/// An iterator over shared references to the values of a [`SkipMap`], in key
/// order.
///
/// This struct is created by the [`SkipMap::values`] method.  See its
/// documentation for more.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let mut map = SkipMap::<i32, &str>::new();
/// map.insert(3, "c");
/// map.insert(1, "a");
/// map.insert(2, "b");
///
/// let values: Vec<&&str> = map.values().collect();
/// assert_eq!(values, [&"a", &"b", &"c"]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Values<'a, K, V, const N: usize = 16> {
    /// Underlying key-value iterator.
    inner: Iter<'a, K, V, N>,
}

// SAFETY: Values delegates all access to Iter; same Send/Sync bounds apply.
unsafe impl<K: Sync, V: Sync, const N: usize> Send for Values<'_, K, V, N> {}
// SAFETY: Values delegates all access to Iter; same Send/Sync bounds apply.
unsafe impl<K: Sync, V: Sync, const N: usize> Sync for Values<'_, K, V, N> {}

impl<K, V, const N: usize> Clone for Values<'_, K, V, N> {
    #[inline]
    fn clone(&self) -> Self {
        Values {
            inner: self.inner.clone(),
        }
    }
}

impl<K, V: fmt::Debug, const N: usize> fmt::Debug for Values<'_, K, V, N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, K, V, const N: usize> Iterator for Values<'a, K, V, N> {
    type Item = &'a V;

    #[inline]
    fn next(&mut self) -> Option<&'a V> {
        self.inner.next().map(|(_, v)| v)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for Values<'a, K, V, N> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a V> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

impl<K, V, const N: usize> ExactSizeIterator for Values<'_, K, V, N> {}

impl<K, V, const N: usize> FusedIterator for Values<'_, K, V, N> {}

// MARK: ValuesMut

/// An iterator over mutable references to the values of a [`SkipMap`], in key
/// order.
///
/// This struct is created by the [`SkipMap::values_mut`] method.  See its
/// documentation for more.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let mut map = SkipMap::<i32, i32>::new();
/// map.insert(1, 10);
/// map.insert(2, 20);
/// map.insert(3, 30);
///
/// for v in map.values_mut() {
///     *v += 1;
/// }
///
/// let values: Vec<i32> = map.values().copied().collect();
/// assert_eq!(values, [11, 21, 31]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ValuesMut<'a, K, V, const N: usize = 16> {
    /// Underlying mutable key-value iterator.
    inner: IterMut<'a, K, V, N>,
}

// SAFETY: ValuesMut delegates all access to IterMut; same Send/Sync bounds.
unsafe impl<K: Send, V: Send, const N: usize> Send for ValuesMut<'_, K, V, N> {}
// SAFETY: ValuesMut delegates all access to IterMut; same Send/Sync bounds.
unsafe impl<K: Sync, V: Sync, const N: usize> Sync for ValuesMut<'_, K, V, N> {}

impl<K, V: fmt::Debug, const N: usize> fmt::Debug for ValuesMut<'_, K, V, N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Traverse via shared references for display.
        let mut builder = f.debug_list();
        let mut node_ptr = self.inner.front;
        let mut remaining = self.inner.len;
        while remaining > 0 {
            let Some(ptr) = node_ptr else { break };
            // SAFETY: ptr is a valid pointer to a live Node for lifetime 'a.
            // We only read through it here, and &self prevents concurrent mutation.
            let node = unsafe { ptr.as_ref() };
            if let Some(kv) = node.value() {
                builder.entry(&kv.1);
            }
            node_ptr = node.next();
            remaining = remaining.saturating_sub(1);
        }
        builder.finish()
    }
}

impl<'a, K, V, const N: usize> Iterator for ValuesMut<'a, K, V, N> {
    type Item = &'a mut V;

    #[inline]
    fn next(&mut self) -> Option<&'a mut V> {
        self.inner.next().map(|(_, v)| v)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for ValuesMut<'a, K, V, N> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut V> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

impl<K, V, const N: usize> ExactSizeIterator for ValuesMut<'_, K, V, N> {}

impl<K, V, const N: usize> FusedIterator for ValuesMut<'_, K, V, N> {}

// MARK: IntoIter

/// An owning iterator over the key-value pairs of a [`SkipMap`], in key order.
///
/// This struct is created by the [`IntoIterator`] implementation for
/// [`SkipMap`].  Iteration consumes the map.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let mut map = SkipMap::<i32, &str>::new();
/// map.insert(1, "a");
/// map.insert(2, "b");
/// map.insert(3, "c");
///
/// let mut iter = map.into_iter();
/// assert_eq!(iter.next(), Some((1, "a")));
/// assert_eq!(iter.next_back(), Some((3, "c")));
/// assert_eq!(iter.next(), Some((2, "b")));
/// assert_eq!(iter.next(), None);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IntoIter<
    K,
    V,
    const N: usize = 16,
    C: Comparator<K> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// The remaining entries.  `pop_first` / `pop_last` drive iteration;
    /// dropping [`IntoIter`] drops the remaining entries via [`SkipMap::Drop`].
    list: SkipMap<K, V, N, C, G>,
}

// SAFETY: IntoIter<K, V, N, C, G> owns its entries.  Sending it to another
// thread is safe when K, V, C, and G are Send.
unsafe impl<K: Send, V: Send, C: Comparator<K> + Send, G: LevelGenerator + Send, const N: usize>
    Send for IntoIter<K, V, N, C, G>
{
}

// SAFETY: Sharing &IntoIter is safe when K, V, C, and G are Sync.  Advancing
// the iterator requires &mut IntoIter, preventing concurrent mutation.
unsafe impl<K: Sync, V: Sync, C: Comparator<K> + Sync, G: LevelGenerator + Sync, const N: usize>
    Sync for IntoIter<K, V, N, C, G>
{
}

impl<K: fmt::Debug, V: fmt::Debug, C: Comparator<K>, G: LevelGenerator, const N: usize> fmt::Debug
    for IntoIter<K, V, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.list.iter()).finish()
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> Iterator
    for IntoIter<K, V, N, C, G>
{
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.list.pop_first()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.list.len(), Some(self.list.len()))
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> DoubleEndedIterator
    for IntoIter<K, V, N, C, G>
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.list.pop_last()
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> ExactSizeIterator
    for IntoIter<K, V, N, C, G>
{
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> FusedIterator
    for IntoIter<K, V, N, C, G>
{
}

// MARK: IntoKeys

/// An owning iterator over the keys of a [`SkipMap`], in key order.
///
/// This struct is created by the [`SkipMap::into_keys`] method.  See its
/// documentation for more.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let mut map = SkipMap::<i32, &str>::new();
/// map.insert(3, "c");
/// map.insert(1, "a");
/// map.insert(2, "b");
///
/// let keys: Vec<i32> = map.into_keys().collect();
/// assert_eq!(keys, [1, 2, 3]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IntoKeys<
    K,
    V,
    const N: usize = 16,
    C: Comparator<K> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// Underlying owning key-value iterator.
    inner: IntoIter<K, V, N, C, G>,
}

// SAFETY: IntoKeys delegates all access to IntoIter; same Send/Sync bounds.
unsafe impl<K: Send, V: Send, C: Comparator<K> + Send, G: LevelGenerator + Send, const N: usize>
    Send for IntoKeys<K, V, N, C, G>
{
}

// SAFETY: IntoKeys delegates all access to IntoIter; same Send/Sync bounds.
unsafe impl<K: Sync, V: Sync, C: Comparator<K> + Sync, G: LevelGenerator + Sync, const N: usize>
    Sync for IntoKeys<K, V, N, C, G>
{
}

impl<K: fmt::Debug, V, C: Comparator<K>, G: LevelGenerator, const N: usize> fmt::Debug
    for IntoKeys<K, V, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.inner.list.keys()).finish()
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> Iterator
    for IntoKeys<K, V, N, C, G>
{
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<K> {
        self.inner.next().map(|(k, _)| k)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> DoubleEndedIterator
    for IntoKeys<K, V, N, C, G>
{
    #[inline]
    fn next_back(&mut self) -> Option<K> {
        self.inner.next_back().map(|(k, _)| k)
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> ExactSizeIterator
    for IntoKeys<K, V, N, C, G>
{
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> FusedIterator
    for IntoKeys<K, V, N, C, G>
{
}

// MARK: IntoValues

/// An owning iterator over the values of a [`SkipMap`], in key order.
///
/// This struct is created by the [`SkipMap::into_values`] method.  See its
/// documentation for more.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let mut map = SkipMap::<i32, i32>::new();
/// map.insert(3, 30);
/// map.insert(1, 10);
/// map.insert(2, 20);
///
/// let values: Vec<i32> = map.into_values().collect();
/// assert_eq!(values, [10, 20, 30]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IntoValues<
    K,
    V,
    const N: usize = 16,
    C: Comparator<K> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// Underlying owning key-value iterator.
    inner: IntoIter<K, V, N, C, G>,
}

// SAFETY: IntoValues delegates all access to IntoIter; same Send/Sync bounds.
unsafe impl<K: Send, V: Send, C: Comparator<K> + Send, G: LevelGenerator + Send, const N: usize>
    Send for IntoValues<K, V, N, C, G>
{
}

// SAFETY: IntoValues delegates all access to IntoIter; same Send/Sync bounds.
unsafe impl<K: Sync, V: Sync, C: Comparator<K> + Sync, G: LevelGenerator + Sync, const N: usize>
    Sync for IntoValues<K, V, N, C, G>
{
}

impl<K, V: fmt::Debug, C: Comparator<K>, G: LevelGenerator, const N: usize> fmt::Debug
    for IntoValues<K, V, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.inner.list.values()).finish()
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> Iterator
    for IntoValues<K, V, N, C, G>
{
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<V> {
        self.inner.next().map(|(_, v)| v)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> DoubleEndedIterator
    for IntoValues<K, V, N, C, G>
{
    #[inline]
    fn next_back(&mut self) -> Option<V> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> ExactSizeIterator
    for IntoValues<K, V, N, C, G>
{
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> FusedIterator
    for IntoValues<K, V, N, C, G>
{
}

// MARK: Drain

/// An owning iterator over all key-value pairs drained from a [`SkipMap`].
///
/// This struct is created by the [`SkipMap::drain`] method.  All entries are
/// removed from the map eagerly when `drain` is called.  The removed entries
/// are yielded by this iterator in key order.  The map is left empty
/// regardless of whether the `Drain` is fully consumed.
///
/// Supports both forward and backward iteration ([`DoubleEndedIterator`]).
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let mut map = SkipMap::<i32, &str>::new();
/// map.insert(3, "c");
/// map.insert(1, "a");
/// map.insert(2, "b");
///
/// let drained: Vec<_> = map.drain().collect();
/// assert_eq!(drained, [(1, "a"), (2, "b"), (3, "c")]);
/// assert!(map.is_empty());
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Drain<'a, K, V> {
    /// The already-removed entries in key order.
    iter: std::vec::IntoIter<(K, V)>,
    /// Ties the `Drain`'s lifetime to the `&'a mut SkipMap` that created it,
    /// preventing the map from being used while this `Drain` is alive.
    _marker: PhantomData<&'a mut (K, V)>,
}

// SAFETY: Drain<'a, K, V> owns its yielded entries.  Sending it to another
// thread requires K: Send and V: Send.
unsafe impl<K: Send, V: Send> Send for Drain<'_, K, V> {}

// SAFETY: Sharing &Drain<'a, K, V> across threads is safe when K: Sync and
// V: Sync.  Advancing the iterator requires &mut Drain.
unsafe impl<K: Sync, V: Sync> Sync for Drain<'_, K, V> {}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Drain<'_, K, V> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter.as_slice()).finish()
    }
}

impl<K, V> Iterator for Drain<'_, K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for Drain<'_, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<K, V> FusedIterator for Drain<'_, K, V> {}

// MARK: ExtractIf

/// A lazy iterator that removes and yields key-value pairs satisfying a
/// predicate from a [`SkipMap`].
///
/// This struct is created by the [`SkipMap::extract_if`] method.  The
/// predicate is called once per entry in key order.  Entries for which it
/// returns `true` are removed and yielded; all others remain in place.
///
/// If the iterator is dropped before being fully consumed the predicate is
/// **not** called for the remaining entries; they all stay in the map.
///
/// Does **not** implement [`DoubleEndedIterator`] or [`ExactSizeIterator`].
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_map::SkipMap;
///
/// let mut map = SkipMap::<i32, i32>::new();
/// for (k, v) in [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)] {
///     map.insert(k, v);
/// }
///
/// let extracted: Vec<_> = map.extract_if(|_k, v| *v % 20 == 0).collect();
/// assert_eq!(extracted, [(2, 20), (4, 40)]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<
    'a,
    K,
    V,
    C: Comparator<K> = OrdComparator,
    G: LevelGenerator = Geometric,
    F = fn(&K, &mut V) -> bool,
    const N: usize = 16,
> where
    F: FnMut(&K, &mut V) -> bool,
{
    /// Mutable borrow of the owning map, needed to update `len` and `tail` on
    /// each removal and to rebuild skip links on drop.
    list: &'a mut SkipMap<K, V, N, C, G>,
    /// Raw pointer to the next node to visit, or `None` when exhausted.
    current: Option<NonNull<Node<(K, V), N>>>,
    /// Set to `true` the first time an entry is removed.  Used to skip the
    /// `$O(n)$` skip-link rebuild in `Drop::drop` when nothing was removed.
    any_removed: bool,
    /// User-supplied filter predicate.
    pred: F,
}

// SAFETY: ExtractIf yields owned (K, V) pairs and holds &'a mut SkipMap.
// Sending it to another thread requires K: Send, V: Send, C: Send, G: Send,
// and F: Send.
unsafe impl<
    K: Send,
    V: Send,
    C: Comparator<K> + Send,
    G: LevelGenerator + Send,
    F: Send,
    const N: usize,
> Send for ExtractIf<'_, K, V, C, G, F, N>
where
    F: FnMut(&K, &mut V) -> bool,
{
}

// SAFETY: Sharing &ExtractIf requires K: Sync, V: Sync, C: Sync, G: Sync,
// and F: Sync.  Advancing the iterator requires &mut ExtractIf.
unsafe impl<
    K: Sync,
    V: Sync,
    C: Comparator<K> + Sync,
    G: LevelGenerator + Sync,
    F: Sync,
    const N: usize,
> Sync for ExtractIf<'_, K, V, C, G, F, N>
where
    F: FnMut(&K, &mut V) -> bool,
{
}

impl<K: fmt::Debug, V: fmt::Debug, C: Comparator<K>, G: LevelGenerator, F, const N: usize>
    fmt::Debug for ExtractIf<'_, K, V, C, G, F, N>
where
    F: FnMut(&K, &mut V) -> bool,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display the unvisited entries reachable from `current`.
        // We hold &self so no mutable access is in progress.
        let mut builder = f.debug_map();
        let mut ptr = self.current;
        while let Some(nn) = ptr {
            // SAFETY: nn points to a live Node<(K,V)> owned by the SkipMap
            // that created this ExtractIf.  We only read through it here.
            let node = unsafe { nn.as_ref() };
            if let Some(kv) = node.value() {
                builder.entry(&kv.0, &kv.1);
            }
            ptr = node.next();
        }
        builder.finish()
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, F, const N: usize> Iterator
    for ExtractIf<'_, K, V, C, G, F, N>
where
    F: FnMut(&K, &mut V) -> bool,
{
    type Item = (K, V);

    #[expect(
        clippy::unwrap_in_result,
        clippy::expect_used,
        reason = "`value_mut()` and `take_value()` return None only for the head \
              sentinel, which is never reachable via the data-node walk; the \
              expect fires only on invariant violations"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-pointer dereference, value_mut(), tail-update read, and pop() \
              all touch provably disjoint heap nodes; splitting across blocks \
              would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        loop {
            let current_nn = self.current?;
            // SAFETY: current_nn was derived from a heap-allocated Node<(K,V)>
            // owned by the SkipMap that created this ExtractIf.  We hold
            // &'a mut SkipMap exclusively for the iterator's lifetime,
            // ensuring every node remains allocated and non-aliased.
            // We capture next_opt before any mutation of the current node.
            unsafe {
                let current: *mut Node<(K, V), N> = current_nn.as_ptr();
                let next_opt = (*current).next();

                let kv_ref = (*current).value_mut().expect("data node has value");
                if (self.pred)(&kv_ref.0, &mut kv_ref.1) {
                    self.current = next_opt;
                    self.any_removed = true;
                    self.list.len = self.list.len.saturating_sub(1);
                    // If this node was the tail, update the tail pointer to the
                    // predecessor data node (or None if the map is now empty).
                    if self.list.tail == Some(current_nn) {
                        // SAFETY: prev() returns a valid pointer into the same
                        // map allocation.
                        self.list.tail = (*current).prev().filter(|p| p.as_ref().value().is_some());
                    }
                    let mut boxed = (*current).pop();
                    return Some(boxed.take_value().expect("data node has value"));
                }
                self.current = next_opt;
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // The predicate outcome is unknown, so the lower bound is 0.
        // At most all remaining map entries could be extracted.
        (0, Some(self.list.len))
    }
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, F, const N: usize> FusedIterator
    for ExtractIf<'_, K, V, C, G, F, N>
where
    F: FnMut(&K, &mut V) -> bool,
{
}

impl<K, V, C: Comparator<K>, G: LevelGenerator, F, const N: usize> Drop
    for ExtractIf<'_, K, V, C, G, F, N>
where
    F: FnMut(&K, &mut V) -> bool,
{
    #[inline]
    fn drop(&mut self) {
        if !self.any_removed {
            // Nothing was removed; skip links are still valid.
            return;
        }
        // Rebuild all skip links in one O(n) forward pass over the prev/next
        // chain.  The prev/next chain is already correct (each `pop()` in
        // `Iterator::next` spliced out the removed node), so we only need to
        // re-derive the level-indexed skip links.
        //
        // SAFETY: &'a mut SkipMap is held exclusively.  All raw pointers
        // originate from its heap allocations.
        let (_, new_tail) = unsafe { Node::filter_rebuild(self.list.head, |_| true, |_| {}) };
        self.list.tail = new_tail;
        // self.list.len is already correct: decremented in Iterator::next
        // once per removed entry.
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use core::ops::Bound;

    use pretty_assertions::assert_eq;

    use super::super::SkipMap;
    use crate::comparator::FnComparator;

    fn map_from<const N: usize>(pairs: [(&'static str, i32); N]) -> SkipMap<&'static str, i32> {
        let mut m = SkipMap::new();
        for (k, v) in pairs {
            m.insert(k, v);
        }
        m
    }

    fn imap_from<const N: usize>(pairs: [(i32, i32); N]) -> SkipMap<i32, i32> {
        let mut m = SkipMap::new();
        for (k, v) in pairs {
            m.insert(k, v);
        }
        m
    }

    // MARK: iter

    #[test]
    fn iter_empty() {
        let map = SkipMap::<i32, i32>::new();
        let mut iter = map.iter();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_single_element() {
        let map = imap_from([(1, 10)]);
        let mut iter = map.iter();
        assert_eq!(iter.next(), Some((&1, &10)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_single_from_back() {
        let map = imap_from([(1, 10)]);
        let mut iter = map.iter();
        assert_eq!(iter.next_back(), Some((&1, &10)));
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_forward_sorted_order() {
        let map = map_from([("b", 2), ("a", 1), ("c", 3)]);
        let pairs: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [("a", 1), ("b", 2), ("c", 3)]);
    }

    #[test]
    fn iter_backward_sorted_order() {
        let map = map_from([("b", 2), ("a", 1), ("c", 3)]);
        let pairs: Vec<_> = map.iter().rev().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [("c", 3), ("b", 2), ("a", 1)]);
    }

    #[test]
    fn iter_double_ended_alternating() {
        let map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        let mut iter = map.iter();
        assert_eq!(iter.next(), Some((&1, &10)));
        assert_eq!(iter.next_back(), Some((&5, &50)));
        assert_eq!(iter.next(), Some((&2, &20)));
        assert_eq!(iter.next_back(), Some((&4, &40)));
        assert_eq!(iter.next(), Some((&3, &30)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_size_hint() {
        let map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let mut iter = map.iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        iter.next();
        iter.next();
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn iter_exact_size() {
        let map = imap_from([(1, 10), (2, 20), (3, 30)]);
        assert_eq!(map.iter().len(), 3);
    }

    #[test]
    fn iter_clone() {
        let map = imap_from([(1, 10), (2, 20)]);
        let iter = map.iter();
        let clone = iter.clone();
        let a: Vec<_> = iter.collect();
        let b: Vec<_> = clone.collect();
        assert_eq!(a, b);
    }

    // MARK: iter_mut

    #[test]
    fn iter_mut_empty() {
        let mut map = SkipMap::<i32, i32>::new();
        let mut iter = map.iter_mut();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_mut_mutate_values() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30)]);
        for (_k, v) in map.iter_mut() {
            *v *= 2;
        }
        let pairs: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [(1, 20), (2, 40), (3, 60)]);
    }

    #[test]
    fn iter_mut_keys_are_immutable() {
        let mut map = imap_from([(1, 10), (2, 20)]);
        let keys: Vec<i32> = map.iter_mut().map(|(&k, _)| k).collect();
        assert_eq!(keys, [1, 2]);
    }

    #[test]
    fn iter_mut_double_ended() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let mut iter = map.iter_mut();
        assert_eq!(iter.next().map(|(&k, &mut v)| (k, v)), Some((1, 10)));
        assert_eq!(iter.next_back().map(|(&k, &mut v)| (k, v)), Some((3, 30)));
        assert_eq!(iter.next().map(|(&k, &mut v)| (k, v)), Some((2, 20)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_mut_size_hint() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let mut iter = map.iter_mut();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
    }

    // MARK: keys

    #[test]
    fn keys_empty() {
        let map = SkipMap::<i32, &str>::new();
        assert_eq!(map.keys().next(), None);
    }

    #[test]
    fn keys_forward() {
        let map = imap_from([(3, 30), (1, 10), (2, 20)]);
        let keys: Vec<i32> = map.keys().copied().collect();
        assert_eq!(keys, [1, 2, 3]);
    }

    #[test]
    fn keys_backward() {
        let map = imap_from([(3, 30), (1, 10), (2, 20)]);
        let keys: Vec<i32> = map.keys().copied().rev().collect();
        assert_eq!(keys, [3, 2, 1]);
    }

    #[test]
    fn keys_size_hint() {
        let map = imap_from([(1, 10), (2, 20)]);
        assert_eq!(map.keys().len(), 2);
    }

    // MARK: values

    #[test]
    fn values_empty() {
        let map = SkipMap::<i32, i32>::new();
        assert_eq!(map.values().next(), None);
    }

    #[test]
    fn values_forward() {
        let map = imap_from([(3, 30), (1, 10), (2, 20)]);
        let values: Vec<i32> = map.values().copied().collect();
        assert_eq!(values, [10, 20, 30]);
    }

    #[test]
    fn values_backward() {
        let map = imap_from([(3, 30), (1, 10), (2, 20)]);
        let values: Vec<i32> = map.values().copied().rev().collect();
        assert_eq!(values, [30, 20, 10]);
    }

    // MARK: values_mut

    #[test]
    fn values_mut_empty() {
        let mut map = SkipMap::<i32, i32>::new();
        assert_eq!(map.values_mut().next(), None);
    }

    #[test]
    fn values_mut_modify() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30)]);
        for v in map.values_mut() {
            *v += 5;
        }
        let values: Vec<i32> = map.values().copied().collect();
        assert_eq!(values, [15, 25, 35]);
    }

    #[test]
    fn values_mut_backward() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let values: Vec<i32> = map.values_mut().map(|&mut v| v).rev().collect();
        assert_eq!(values, [30, 20, 10]);
    }

    // MARK: range

    #[test]
    fn range_empty_map() {
        let map = SkipMap::<i32, i32>::new();
        let mut iter = map.range(1..=5);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn range_unbounded() {
        let map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let pairs: Vec<_> = map.range(..).map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn range_lo_included() {
        let map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        let pairs: Vec<_> = map.range(3..).map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [(3, 30), (4, 40), (5, 50)]);
    }

    #[test]
    fn range_hi_included() {
        let map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        let pairs: Vec<_> = map.range(..=3).map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn range_hi_excluded() {
        let map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        let pairs: Vec<_> = map.range(..3).map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [(1, 10), (2, 20)]);
    }

    #[test]
    fn range_both_included() {
        let map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        let pairs: Vec<_> = map.range(2..=4).map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [(2, 20), (3, 30), (4, 40)]);
    }

    #[test]
    fn range_lo_excluded() {
        let map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        let pairs: Vec<_> = map
            .range((Bound::Excluded(2), Bound::Included(4)))
            .map(|(&k, &v)| (k, v))
            .collect();
        assert_eq!(pairs, [(3, 30), (4, 40)]);
    }

    #[test]
    fn range_no_elements_in_gap() {
        let map = imap_from([(1, 10), (5, 50)]);
        let pairs: Vec<_> = map.range(2..=4).map(|(&k, &v)| (k, v)).collect();
        assert!(pairs.is_empty());
    }

    #[test]
    fn range_single_element() {
        let map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let pairs: Vec<_> = map.range(2..=2).map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [(2, 20)]);
    }

    #[test]
    fn range_backward() {
        let map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        let pairs: Vec<_> = map.range(2..=4).rev().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [(4, 40), (3, 30), (2, 20)]);
    }

    #[test]
    fn range_size_hint() {
        let map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        assert_eq!(map.range(2..=4).len(), 3);
    }

    #[test]
    #[should_panic(expected = "range start is after range end")]
    fn range_panics_lo_gt_hi() {
        let map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let _ = map.range(3..1);
    }

    #[test]
    #[should_panic(expected = "range start is after range end")]
    fn range_panics_excluded_equal() {
        let map = imap_from([(1, 10)]);
        let _ = map.range((Bound::Excluded(1), Bound::Included(1)));
    }

    // MARK: range_mut

    #[test]
    fn range_mut_empty_map() {
        let mut map = SkipMap::<i32, i32>::new();
        let mut iter = map.range_mut(1..=5);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn range_mut_modify_values() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        for (_k, v) in map.range_mut(2..=4) {
            *v *= 10;
        }
        let pairs: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [(1, 10), (2, 200), (3, 300), (4, 400), (5, 50)]);
    }

    #[test]
    fn range_mut_backward() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let pairs: Vec<_> = map.range_mut(..).rev().map(|(&k, &mut v)| (k, v)).collect();
        assert_eq!(pairs, [(3, 30), (2, 20), (1, 10)]);
    }

    #[test]
    #[should_panic(expected = "range start is after range end")]
    fn range_mut_panics_lo_gt_hi() {
        let mut map = imap_from([(1, 10), (3, 30)]);
        let _ = map.range_mut(3..1);
    }

    // MARK: into_iter

    #[test]
    fn into_iter_empty() {
        let map = SkipMap::<i32, i32>::new();
        let mut iter = map.into_iter();
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn into_iter_forward() {
        let map = imap_from([(3, 30), (1, 10), (2, 20)]);
        let pairs: Vec<_> = map.into_iter().collect();
        assert_eq!(pairs, [(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn into_iter_backward() {
        let map = imap_from([(3, 30), (1, 10), (2, 20)]);
        let pairs: Vec<_> = map.into_iter().rev().collect();
        assert_eq!(pairs, [(3, 30), (2, 20), (1, 10)]);
    }

    #[test]
    fn into_iter_alternating() {
        let map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let mut iter = map.into_iter();
        assert_eq!(iter.next(), Some((1, 10)));
        assert_eq!(iter.next_back(), Some((3, 30)));
        assert_eq!(iter.next(), Some((2, 20)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn into_iter_size_hint() {
        let map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let iter = map.into_iter();
        assert_eq!(iter.len(), 3);
    }

    // MARK: into_keys

    #[test]
    fn into_keys_empty() {
        let map = SkipMap::<i32, i32>::new();
        assert_eq!(map.into_keys().next(), None);
    }

    #[test]
    fn into_keys_forward() {
        let map = imap_from([(3, 30), (1, 10), (2, 20)]);
        let keys: Vec<i32> = map.into_keys().collect();
        assert_eq!(keys, [1, 2, 3]);
    }

    #[test]
    fn into_keys_backward() {
        let map = imap_from([(3, 30), (1, 10), (2, 20)]);
        let keys: Vec<i32> = map.into_keys().rev().collect();
        assert_eq!(keys, [3, 2, 1]);
    }

    // MARK: into_values

    #[test]
    fn into_values_empty() {
        let map = SkipMap::<i32, i32>::new();
        assert_eq!(map.into_values().next(), None);
    }

    #[test]
    fn into_values_forward() {
        let map = imap_from([(3, 30), (1, 10), (2, 20)]);
        let values: Vec<i32> = map.into_values().collect();
        assert_eq!(values, [10, 20, 30]);
    }

    #[test]
    fn into_values_backward() {
        let map = imap_from([(3, 30), (1, 10), (2, 20)]);
        let values: Vec<i32> = map.into_values().rev().collect();
        assert_eq!(values, [30, 20, 10]);
    }

    // MARK: drain

    #[test]
    fn drain_empty() {
        let mut map = SkipMap::<i32, i32>::new();
        let drained: Vec<_> = map.drain().collect();
        assert!(drained.is_empty());
        assert!(map.is_empty());
    }

    #[test]
    fn drain_all() {
        let mut map = imap_from([(3, 30), (1, 10), (2, 20)]);
        let drained: Vec<_> = map.drain().collect();
        assert_eq!(drained, [(1, 10), (2, 20), (3, 30)]);
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn drain_backward() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let drained: Vec<_> = map.drain().rev().collect();
        assert_eq!(drained, [(3, 30), (2, 20), (1, 10)]);
    }

    #[test]
    fn drain_map_usable_after() {
        let mut map = imap_from([(1, 10), (2, 20)]);
        drop(map.drain());
        assert!(map.is_empty());
        map.insert(5, 50);
        assert_eq!(map.len(), 1);
    }

    // MARK: extract_if

    #[test]
    fn extract_if_none_match() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let extracted: Vec<_> = map.extract_if(|_k, _v| false).collect();
        assert!(extracted.is_empty());
        assert_eq!(map.len(), 3);
        let pairs: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, [(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn extract_if_all_match() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30)]);
        let extracted: Vec<_> = map.extract_if(|_k, _v| true).collect();
        assert_eq!(extracted, [(1, 10), (2, 20), (3, 30)]);
        assert!(map.is_empty());
    }

    #[test]
    fn extract_if_by_value() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        let extracted: Vec<_> = map.extract_if(|_k, v| *v % 20 == 0).collect();
        assert_eq!(extracted, [(2, 20), (4, 40)]);
        let remaining: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(remaining, [(1, 10), (3, 30), (5, 50)]);
    }

    #[test]
    fn extract_if_by_key() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40)]);
        let extracted: Vec<_> = map.extract_if(|k, _v| k % 2 == 0).collect();
        assert_eq!(extracted, [(2, 20), (4, 40)]);
        let remaining: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(remaining, [(1, 10), (3, 30)]);
    }

    #[test]
    fn extract_if_mutates_value() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30)]);
        // Increment all values; only extract those > 15 after increment.
        let extracted: Vec<_> = map
            .extract_if(|_k, v| {
                *v += 5;
                *v > 25
            })
            .collect();
        // After increment: 15, 25, 35.  Only 35 > 25.
        assert_eq!(extracted, [(3, 35)]);
        let remaining: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(remaining, [(1, 15), (2, 25)]);
    }

    #[test]
    fn extract_if_partial_drop() {
        let mut map = imap_from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        {
            let mut iter = map.extract_if(|_k, _v| true);
            let first = iter.next();
            assert_eq!(first, Some((1, 10)));
            // Drop iter here; remaining entries stay in the map.
        }
        // Entries 2..5 remain because the predicate was not called for them.
        assert_eq!(map.len(), 4);
        let remaining: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(remaining, [(2, 20), (3, 30), (4, 40), (5, 50)]);
    }

    #[test]
    fn extract_if_custom_comparator() {
        let mut map: SkipMap<i32, i32, 16, _> =
            SkipMap::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);
        // Stored as [3, 2, 1] (reverse order)
        let extracted: Vec<_> = map.extract_if(|k, _v| k % 2 != 0).collect();
        // Odd keys in traversal order (3 then 1)
        assert_eq!(extracted, [(3, 30), (1, 10)]);
        let remaining: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(remaining, [(2, 20)]);
    }
}
