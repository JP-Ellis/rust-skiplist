//! List-restructuring methods for [`SkipMap`](super::SkipMap):
//! `clear`, `append`, `split_off`, and `merge`.

use alloc::boxed::Box;
use core::{cmp::Ordering, ptr::NonNull};

use super::SkipMap;
use crate::{
    comparator::{Comparator, ComparatorKey},
    level_generator::LevelGenerator,
    node::{
        Node,
        visitor::{OrdIndexVisitor, Visitor},
    },
};

impl<K, V, const N: usize, C: Comparator<K>, G: LevelGenerator> SkipMap<K, V, N, C, G> {
    /// Removes all entries from the map.
    ///
    /// The comparator and level generator are preserved; entries can be
    /// inserted again immediately after calling `clear`.
    ///
    /// This operation is `$O(n)$`: all n entries must be dropped.
    ///
    /// If an entry's destructor panics, the remaining entries are kept in the
    /// map rather than leaked, and the map stays usable; call `clear` again to
    /// drop them.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// map.clear();
    /// assert!(map.is_empty());
    /// assert_eq!(map.len(), 0);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        // Detach the chain and reset the bookkeeping before running a single
        // destructor.  An entry `Drop` that unwinds then finds the map already
        // empty and consistent, rather than leaving `tail` pointing at a node
        // the unwind has freed.
        //
        // SAFETY: `self.head` is a live, exclusively-owned sentinel, and the
        // chain hanging off it is owned by this map alone.
        let chain = unsafe {
            let head = self.head_mut();
            for link in head.links_mut() {
                *link = None;
            }
            head.take_next_chain()
        };
        self.tail = None;
        self.len = 0;

        let head = self.head;
        // SAFETY: `head` is the exclusively-owned sentinel, its `next` was
        // emptied above, and `chain` holds the nodes that hung off it, each
        // allocated by `Box::into_raw` and referenced from nowhere else.
        unsafe { Node::drop_chain(head, head, &mut self.len, &mut self.tail, chain) };
    }

    /// Moves all entries from `other` into `self`, leaving `other` empty.
    ///
    /// Entries from `other` are merged into `self` at their sorted key positions
    /// according to `self`'s comparator.  After the call `self` contains all
    /// entries from both maps in sorted order and `other` is empty.  If a key
    /// from `other` already exists in `self`, the value from `other` overwrites
    /// the value in `self`.
    ///
    /// When the key ranges are strictly disjoint (every key of `other` is
    /// strictly greater than every key of `self`, or every key of `other` is
    /// strictly less than every key of `self`), the node chains are spliced in
    /// `$O(\log n)$` time.
    /// Otherwise, when the key ranges overlap (including when
    /// `self.last_key == other.first_key`), each entry of `other` is inserted
    /// individually in `$O(m \log(n+m))$` time.
    ///
    /// The splice compares only the boundary keys and trusts the
    /// comparator for the rest.  A [`Comparator`] that is not a total order
    /// can therefore leave the result unordered.
    ///
    /// [`Comparator`]: crate::comparator::Comparator
    ///
    /// # Panics
    ///
    /// If the comparator violates its contract, this method may panic after
    /// earlier steps have taken effect; the collection stays valid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut a = SkipMap::<i32, &str>::new();
    /// a.insert(1, "one");
    /// a.insert(3, "three");
    ///
    /// let mut b = SkipMap::<i32, &str>::new();
    /// b.insert(4, "four");
    /// b.insert(5, "five");
    ///
    /// a.append(&mut b);
    /// assert!(b.is_empty());
    /// assert_eq!(a.len(), 4);
    /// let keys: Vec<i32> = a.keys().copied().collect();
    /// assert_eq!(keys, [1, 3, 4, 5]);
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "tail/head nodes always have a value; expects fire only on invariant violations"
    )]
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        if other.is_empty() {
            return;
        }

        // The splice conditions use strict `Less` (not `!= Greater`) so that
        // an equal boundary key falls through to `insert`, which replaces the
        // existing value, matching the `BTreeMap::append` contract.
        //
        // Comparator trust: the comparisons below run before any write and
        // only pick a path.  The splices never consult the comparator, so a
        // wrong answer can unsort the map but not mislink it.
        //
        // Forward splice: self is empty, or self.last_key < other.first_key.
        let can_concat = self.is_empty()
            || self.tail.is_some_and(|tail_nn| {
                // SAFETY: tail_nn is a live data node owned by self; the
                // borrow ends before any mutation below.
                let self_last_key: &K = &unsafe { tail_nn.as_ref() }
                    .value()
                    .expect("tail node has a value")
                    .0;
                // SAFETY: other.head is a valid sentinel; next_as_ref returns
                // a short-lived shared reference that is not retained.
                let other_first_key: &K = &unsafe { other.head.as_ref() }
                    .next_as_ref()
                    .and_then(|n| n.value())
                    .expect("other is non-empty so its first data node has a value")
                    .0;
                self.comparator.compare(self_last_key, other_first_key) == Ordering::Less
            });
        if can_concat {
            // SAFETY: we hold `&mut self` and `&mut other`, two distinct
            // maps, and `other` is non-empty.
            unsafe {
                Node::concat(
                    self.head,
                    &mut self.len,
                    &mut self.tail,
                    other.head,
                    &mut other.len,
                    &mut other.tail,
                );
            }
            return;
        }

        // Reverse splice: other.last_key < self.first_key.  self is non-empty
        // here because can_concat covers self.is_empty().
        let can_prepend = other.tail.is_some_and(|other_tail_nn| {
            // SAFETY: other_tail_nn is a live data node owned by other.
            let other_last_key: &K = &unsafe { other_tail_nn.as_ref() }
                .value()
                .expect("other tail node has a value")
                .0;
            // SAFETY: self.head is a valid sentinel; next_as_ref returns a
            // short-lived shared reference that is not retained.
            let self_first_key: &K = &unsafe { self.head.as_ref() }
                .next_as_ref()
                .and_then(|n| n.value())
                .expect("self is non-empty so its first data node has a value")
                .0;
            self.comparator.compare(other_last_key, self_first_key) == Ordering::Less
        });
        if can_prepend {
            // `other` goes first: swap the two chains, then append.
            core::mem::swap(&mut self.head, &mut other.head);
            core::mem::swap(&mut self.tail, &mut other.tail);
            core::mem::swap(&mut self.len, &mut other.len);
            // SAFETY: we hold `&mut self` and `&mut other`, two distinct
            // maps, and `other` now holds the non-empty original `self`.
            unsafe {
                Node::concat(
                    self.head,
                    &mut self.len,
                    &mut self.tail,
                    other.head,
                    &mut other.len,
                    &mut other.tail,
                );
            }
        } else {
            while let Some((k, v)) = other.pop_first() {
                self.insert(k, v);
            }
        }
    }

    /// Splits the map at the given key, returning a new map containing all
    /// entries with key `>= key`.
    ///
    /// After the call, `self` retains all entries with key `< key`, and the
    /// returned map contains all entries with key `>= key`.
    ///
    /// This operation runs in `$O(\log n)$` time.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut a = SkipMap::<i32, &str>::new();
    /// a.insert(1, "one");
    /// a.insert(2, "two");
    /// a.insert(3, "three");
    /// a.insert(4, "four");
    /// a.insert(5, "five");
    ///
    /// let b = a.split_off(&3);
    /// let a_keys: Vec<i32> = a.keys().copied().collect();
    /// let b_keys: Vec<i32> = b.keys().copied().collect();
    /// assert_eq!(a_keys, [1, 2]);
    /// assert_eq!(b_keys, [3, 4, 5]);
    /// ```
    #[inline]
    #[must_use]
    pub fn split_off<Q>(&mut self, key: &Q) -> Self
    where
        Q: ?Sized,
        C: Clone + ComparatorKey<K, Q>,
        G: Clone,
    {
        // The comparator only picks the split rank; the split itself goes by
        // rank, so a wrong answer moves the wrong entries but keeps both maps
        // well formed.
        let at = {
            let cmp = |entry: &(K, V), q: &Q| match self.comparator.compare_key(&entry.0, q) {
                Ordering::Less => Ordering::Less,
                Ordering::Equal | Ordering::Greater => Ordering::Greater,
            };
            let mut visitor = OrdIndexVisitor::new(self.head_ref(), key, cmp);
            visitor.traverse();
            visitor.current_rank()
        };

        let max_levels = self.head_ref().level();
        // SAFETY: `Box::into_raw` transfers ownership; freed in `Drop`.
        let head =
            unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(max_levels)))) };
        let mut result = Self {
            head,
            tail: None,
            len: 0,
            comparator: self.comparator.clone(),
            generator: self.generator.clone(),
        };
        if at == self.len {
            return result;
        }

        // SAFETY: we hold `&mut self`, `at < self.len`, and `result.head` is a
        // fresh empty head of the same height.
        let last_kept = unsafe { Node::split_after(self.head, at, result.head) };
        result.len = self.len.wrapping_sub(at);
        result.tail = self.tail;
        self.len = at;
        self.tail = (at > 0).then_some(last_kept);
        result
    }

    /// Merges all entries from `other` into `self`, consuming `other`.
    ///
    /// For each entry `(key, value)` taken from `other` (in ascending key
    /// order):
    ///
    /// - If `self` does not contain `key`, the entry is inserted directly.
    /// - If `self` already contains `key`, `conflict(&key, existing, incoming)`
    ///   is called.  The returned value replaces the existing entry.
    ///
    /// The conflict closure receives the existing value by ownership (the
    /// existing entry is removed before the closure runs) and must return the
    /// value to store.
    ///
    /// This operation is `$O(m \log(n+m))$` where `m = other.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut a = SkipMap::<i32, i32>::new();
    /// a.insert(1, 10);
    /// a.insert(2, 20);
    ///
    /// let mut b = SkipMap::<i32, i32>::new();
    /// b.insert(2, 200);
    /// b.insert(3, 300);
    ///
    /// // On conflict: sum the two values.
    /// a.merge(b, |_k, old, new| old + new);
    ///
    /// assert_eq!(a.get(&1), Some(&10));
    /// assert_eq!(a.get(&2), Some(&220)); // 20 + 200
    /// assert_eq!(a.get(&3), Some(&300));
    /// assert_eq!(a.len(), 3);
    /// ```
    #[inline]
    pub fn merge<F>(&mut self, other: Self, mut conflict: F)
    where
        F: FnMut(&K, V, V) -> V,
    {
        for (k, v) in other {
            if let Some(old_v) = self.remove_impl(&k).map(|(_, val)| val) {
                let merged = conflict(&k, old_v, v);
                self.insert(k, merged);
            } else {
                self.insert(k, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::panic::AssertUnwindSafe;
    use std::panic::catch_unwind;

    use pretty_assertions::assert_eq;

    use super::super::SkipMap;
    use crate::test_util::{PanicOnDrop, bombs};

    /// A map of `k -> k` for `k` in `keys`, with tower heights from `seed`.
    fn seeded(keys: core::ops::Range<usize>, seed: u32) -> SkipMap<usize, usize> {
        let mut map = SkipMap::with_level_generator(crate::test_util::seeded_geometric(seed));
        for k in keys {
            map.insert(k, k);
        }
        map
    }

    #[test]
    fn split_off_every_key_is_well_formed() {
        let max_len: usize = if cfg!(miri) { 12 } else { 40 };
        for (len, seed) in (0..max_len).zip(0_u32..) {
            for at in 0..=len {
                let mut kept = seeded(0..len, seed);
                let moved = kept.split_off(&at);
                kept.assert_well_formed();
                moved.assert_well_formed();
                assert!(kept.keys().copied().eq(0..at));
                assert!(moved.keys().copied().eq(at..len));
            }
        }
    }

    #[test]
    fn append_spliced_in_both_directions_is_well_formed() {
        let max_len: usize = if cfg!(miri) { 8 } else { 20 };
        for left in 0..max_len {
            for right in 0..max_len {
                let mut low = seeded(0..left, 1);
                let mut high = seeded(left..left + right, 2);
                low.append(&mut high);
                low.assert_well_formed();
                high.assert_well_formed();
                assert!(low.keys().copied().eq(0..left + right));

                let mut first = seeded(0..left, 3);
                let mut second = seeded(left..left + right, 4);
                second.append(&mut first);
                second.assert_well_formed();
                first.assert_well_formed();
                assert!(second.keys().copied().eq(0..left + right));
                second.insert(left + right, 0);
                second.assert_well_formed();
            }
        }
    }

    // MARK: clear

    #[test]
    fn clear_empty() {
        let mut map = SkipMap::<i32, i32>::new();
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn clear_non_empty() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        map.insert(2, "b");
        map.insert(3, "c");
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.first_key_value(), None);
        assert_eq!(map.last_key_value(), None);
    }

    #[test]
    fn clear_reusable() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 0..10 {
            map.insert(i, i * 10);
        }
        map.clear();
        assert!(map.is_empty());
        map.insert(42, 420);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&42), Some(&420));
    }

    #[test]
    fn clear_idempotent() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 0..5 {
            map.insert(i, i);
        }
        map.clear();
        map.clear();
        assert!(map.is_empty());
    }

    // MARK: append

    #[test]
    fn append_empty_other() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        let mut b = SkipMap::<i32, i32>::new();
        a.append(&mut b);
        assert_eq!(a.len(), 1);
        assert!(b.is_empty());
    }

    #[test]
    fn append_to_empty_self() {
        let mut a = SkipMap::<i32, i32>::new();
        let mut b = SkipMap::<i32, i32>::new();
        b.insert(1, 10);
        b.insert(2, 20);
        a.append(&mut b);
        assert_eq!(a.len(), 2);
        assert!(b.is_empty());
        let keys: Vec<i32> = a.keys().copied().collect();
        assert_eq!(keys, [1, 2]);
    }

    #[test]
    fn append_disjoint_fast_path() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        a.insert(3, 30);

        let mut b = SkipMap::<i32, i32>::new();
        b.insert(4, 40);
        b.insert(5, 50);

        a.append(&mut b);
        assert!(b.is_empty());
        assert_eq!(a.len(), 4);
        let kvs: Vec<(i32, i32)> = a.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(kvs, [(1, 10), (3, 30), (4, 40), (5, 50)]);
    }

    #[test]
    fn append_overlapping_slow_path() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        a.insert(3, 30);

        let mut b = SkipMap::<i32, i32>::new();
        b.insert(2, 20);
        b.insert(4, 40);

        a.append(&mut b);
        assert!(b.is_empty());
        assert_eq!(a.len(), 4);
        let keys: Vec<i32> = a.keys().copied().collect();
        assert_eq!(keys, [1, 2, 3, 4]);
    }

    #[test]
    fn append_equal_boundary_replaces_value() {
        // self.last_key == other.first_key: comparator returns Equal, so the
        // slow path is taken and other's value replaces self's value for that
        // key (matching BTreeMap::append semantics: no duplicates are created).
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        a.insert(3, 30);

        let mut b = SkipMap::<i32, i32>::new();
        b.insert(3, 300); // equal to a's last key

        a.append(&mut b);
        assert!(b.is_empty());
        // Key 3's value is overwritten by b's value; no duplicate is created.
        assert_eq!(a.len(), 2);
        let pairs: Vec<(i32, i32)> = a.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(pairs, [(1, 10), (3, 300)]);
    }

    #[test]
    fn append_both_empty() {
        let mut a = SkipMap::<i32, i32>::new();
        let mut b = SkipMap::<i32, i32>::new();
        a.append(&mut b);
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn append_large_disjoint() {
        let mut a = SkipMap::<i32, i32>::new();
        for i in 0..50 {
            a.insert(i, i);
        }
        let mut b = SkipMap::<i32, i32>::new();
        for i in 50..100 {
            b.insert(i, i);
        }
        a.append(&mut b);
        assert_eq!(a.len(), 100);
        assert!(b.is_empty());
        let keys: Vec<i32> = a.keys().copied().collect();
        let expected: Vec<i32> = (0..100).collect();
        assert_eq!(keys, expected);
    }

    #[test]
    fn append_reverse_disjoint_fast_path() {
        // Reverse fast path: every key of other is strictly less than every
        // key of self.
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(3, 30);
        a.insert(4, 40);

        let mut b = SkipMap::<i32, i32>::new();
        b.insert(1, 10);
        b.insert(2, 20);

        a.append(&mut b);
        assert!(b.is_empty());
        assert_eq!(a.len(), 4);
        let kvs: Vec<(i32, i32)> = a.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(kvs, [(1, 10), (2, 20), (3, 30), (4, 40)]);
    }

    #[test]
    fn append_reverse_equal_boundary_slow_path() {
        // Equal boundary: other.last_key == self.first_key.  The reverse fast
        // path requires strictly Less, so this falls through to the slow path,
        // which replaces the existing value (BTreeMap contract).
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(2, 20);
        a.insert(3, 30);

        let mut b = SkipMap::<i32, i32>::new();
        b.insert(1, 10);
        b.insert(2, 200); // equal to a's first key

        a.append(&mut b);
        assert!(b.is_empty());
        assert_eq!(a.len(), 3);
        let kvs: Vec<(i32, i32)> = a.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(kvs, [(1, 10), (2, 200), (3, 30)]);
    }

    #[test]
    fn append_reverse_large_disjoint() {
        let mut a = SkipMap::<i32, i32>::new();
        for i in 50..100 {
            a.insert(i, i);
        }
        let mut b = SkipMap::<i32, i32>::new();
        for i in 0..50 {
            b.insert(i, i);
        }
        a.append(&mut b);
        assert_eq!(a.len(), 100);
        assert!(b.is_empty());
        let keys: Vec<i32> = a.keys().copied().collect();
        let expected: Vec<i32> = (0..100).collect();
        assert_eq!(keys, expected);
    }

    // MARK: link_structure

    #[test]
    fn link_structure() -> anyhow::Result<()> {
        use insta::assert_snapshot;

        use crate::level_generator::geometric::Geometric;

        let g = Geometric::new_with_seed(4, 0.5, 42)?;
        let mut map = SkipMap::<i32, i32, 4>::with_level_generator(g);
        for i in 1..=10 {
            map.insert(i, i * 10);
        }

        if !cfg!(miri) {
            let display = map.head_ref().display()?;
            assert_snapshot!(display, @"
            [03]: 00 -------------------> 04
            [02]: 00 -------------> 03 -> 04 -------------------------------> 10
            [01]: 00 -------------> 03 -> 04 -------------> 07 -> 08 -------> 10
            [->]: 00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10
            [<-]: 00 <- 01 <- 02 <- 03 <- 04 <- 05 <- 06 <- 07 <- 08 <- 09 <- 10

            [00|03] None
            [01|00] Some((1, 10))
            [02|00] Some((2, 20))
            [03|02] Some((3, 30))
            [04|03] Some((4, 40))
            [05|00] Some((5, 50))
            [06|00] Some((6, 60))
            [07|01] Some((7, 70))
            [08|01] Some((8, 80))
            [09|00] Some((9, 90))
            [10|02] Some((10, 100))
            ");
        }
        Ok(())
    }

    // MARK: split_off

    #[test]
    fn split_off_empty_map() {
        let mut a = SkipMap::<i32, i32>::new();
        let b = a.split_off(&3);
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn split_off_key_in_middle() {
        let mut a = SkipMap::<i32, &str>::new();
        a.insert(1, "one");
        a.insert(2, "two");
        a.insert(3, "three");
        a.insert(4, "four");
        a.insert(5, "five");

        let b = a.split_off(&3);
        let a_keys: Vec<i32> = a.keys().copied().collect();
        let b_keys: Vec<i32> = b.keys().copied().collect();
        assert_eq!(a_keys, [1, 2]);
        assert_eq!(b_keys, [3, 4, 5]);
    }

    #[test]
    fn split_off_key_before_all() {
        let mut a = SkipMap::<i32, i32>::new();
        for i in 1..=5 {
            a.insert(i, i);
        }
        let b = a.split_off(&0); // key less than all elements
        assert!(a.is_empty());
        assert_eq!(b.len(), 5);
    }

    #[test]
    fn split_off_key_after_all() {
        let mut a = SkipMap::<i32, i32>::new();
        for i in 1..=5 {
            a.insert(i, i);
        }
        let b = a.split_off(&10); // key greater than all elements
        assert_eq!(a.len(), 5);
        assert!(b.is_empty());
    }

    #[test]
    fn split_off_first_key() {
        let mut a = SkipMap::<i32, i32>::new();
        for i in 1..=5 {
            a.insert(i, i);
        }
        let b = a.split_off(&1);
        assert!(a.is_empty());
        assert_eq!(b.len(), 5);
    }

    #[test]
    fn split_off_last_key() {
        let mut a = SkipMap::<i32, i32>::new();
        for i in 1..=5 {
            a.insert(i, i);
        }
        let b = a.split_off(&5);
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 1);
        assert_eq!(b.first_key_value(), Some((&5, &5)));
    }

    #[test]
    fn split_off_missing_key() {
        let mut a = SkipMap::<i32, i32>::new();
        for i in [1, 2, 4, 5] {
            a.insert(i, i);
        }
        // Split at 3 (not present): elements >= 3 are 4 and 5.
        let b = a.split_off(&3);
        let a_keys: Vec<i32> = a.keys().copied().collect();
        let b_keys: Vec<i32> = b.keys().copied().collect();
        assert_eq!(a_keys, [1, 2]);
        assert_eq!(b_keys, [4, 5]);
    }

    #[test]
    fn split_off_len_sum_correct() {
        let mut a = SkipMap::<i32, i32>::new();
        for i in 0..20 {
            a.insert(i, i);
        }
        let orig_len = a.len();
        let b = a.split_off(&10);
        assert_eq!(a.len() + b.len(), orig_len);
    }

    #[test]
    fn split_off_links_consistent() {
        let mut a = SkipMap::<i32, i32>::new();
        for i in 0..20 {
            a.insert(i, i);
        }
        let b = a.split_off(&10);
        let a_keys: Vec<i32> = a.keys().copied().collect();
        let b_keys: Vec<i32> = b.keys().copied().collect();
        assert_eq!(a_keys, (0..10).collect::<Vec<_>>());
        assert_eq!(b_keys, (10..20).collect::<Vec<_>>());
    }

    // MARK: merge

    #[test]
    fn merge_empty_other() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        let b = SkipMap::<i32, i32>::new();
        a.merge(b, |_, old, _| old);
        assert_eq!(a.len(), 1);
        assert_eq!(a.get(&1), Some(&10));
    }

    #[test]
    fn merge_into_empty_self() {
        let mut a = SkipMap::<i32, i32>::new();
        let mut b = SkipMap::<i32, i32>::new();
        b.insert(1, 10);
        b.insert(2, 20);
        a.merge(b, |_, old, _| old);
        assert_eq!(a.len(), 2);
        let keys: Vec<i32> = a.keys().copied().collect();
        assert_eq!(keys, [1, 2]);
    }

    #[test]
    fn merge_no_conflict() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        a.insert(3, 30);
        let mut b = SkipMap::<i32, i32>::new();
        b.insert(2, 20);
        b.insert(4, 40);
        a.merge(b, |_, old, _| old);
        assert_eq!(a.len(), 4);
        let keys: Vec<i32> = a.keys().copied().collect();
        assert_eq!(keys, [1, 2, 3, 4]);
    }

    #[test]
    fn merge_sum_on_conflict() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(1, 10);
        a.insert(2, 20);
        let mut b = SkipMap::<i32, i32>::new();
        b.insert(2, 200);
        b.insert(3, 300);
        a.merge(b, |_, old, new| old + new);
        assert_eq!(a.get(&1), Some(&10));
        assert_eq!(a.get(&2), Some(&220));
        assert_eq!(a.get(&3), Some(&300));
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn merge_keep_existing_on_conflict() {
        let mut a = SkipMap::<i32, &str>::new();
        a.insert(1, "original");
        let mut b = SkipMap::<i32, &str>::new();
        b.insert(1, "incoming");
        a.merge(b, |_, old, _| old);
        assert_eq!(a.get(&1), Some(&"original"));
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn merge_keep_incoming_on_conflict() {
        let mut a = SkipMap::<i32, &str>::new();
        a.insert(1, "original");
        let mut b = SkipMap::<i32, &str>::new();
        b.insert(1, "incoming");
        a.merge(b, |_, _, new| new);
        assert_eq!(a.get(&1), Some(&"incoming"));
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn merge_conflict_receives_key() {
        let mut a = SkipMap::<i32, i32>::new();
        a.insert(5, 50);
        let mut b = SkipMap::<i32, i32>::new();
        b.insert(5, 5);
        let mut seen_key = None;
        a.merge(b, |&k, old, new| {
            seen_key = Some(k);
            old + new
        });
        assert_eq!(seen_key, Some(5));
        assert_eq!(a.get(&5), Some(&55));
    }

    #[test]
    fn merge_all_conflict() {
        let mut a = SkipMap::<i32, i32>::new();
        let mut b = SkipMap::<i32, i32>::new();
        for i in 0..10 {
            a.insert(i, i);
            b.insert(i, i * 100);
        }
        a.merge(b, |_, old, new| old + new);
        assert_eq!(a.len(), 10);
        for i in 0..10_i32 {
            assert_eq!(a.get(&i), Some(&(i + i * 100)));
        }
    }

    // MARK: Borrow<Q> split_off (String / &str)

    #[test]
    fn split_off_str_on_string_key() {
        let mut map: SkipMap<String, i32> = SkipMap::new();
        map.insert("apple".to_owned(), 1);
        map.insert("banana".to_owned(), 2);
        map.insert("cherry".to_owned(), 3);
        map.insert("date".to_owned(), 4);
        // split at "cherry": left keeps apple, banana; right gets cherry, date
        let right = map.split_off("cherry");
        assert_eq!(
            map.iter()
                .map(|(k, &v)| (k.as_str(), v))
                .collect::<Vec<_>>(),
            [("apple", 1), ("banana", 2)]
        );
        assert_eq!(
            right
                .iter()
                .map(|(k, &v)| (k.as_str(), v))
                .collect::<Vec<_>>(),
            [("cherry", 3), ("date", 4)]
        );
    }

    // MARK: panic safety

    #[test]
    fn clear_panicking_drop_leaves_map_empty() {
        let mut map = SkipMap::<u32, PanicOnDrop>::new();
        for bomb in bombs(16, 15) {
            map.insert(bomb.id(), bomb);
        }

        let result = catch_unwind(AssertUnwindSafe(|| map.clear()));
        assert!(result.is_err());

        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert!(map.first_key_value().is_none());
        assert!(map.last_key_value().is_none());
        assert_eq!(map.iter().count(), 0);

        map.insert(100, PanicOnDrop::new(100));
        assert_eq!(map.last_key_value().map(|(k, _)| *k), Some(100));
    }

    /// The entries a panicking `clear` never reached stay in the map; they
    /// must not be leaked, and a second `clear` must drop them.
    #[test]
    fn clear_panicking_drop_keeps_unreached_entries() {
        let mut map = SkipMap::<u32, PanicOnDrop>::new();
        for bomb in bombs(16, 7) {
            map.insert(bomb.id(), bomb);
        }

        let result = catch_unwind(AssertUnwindSafe(|| map.clear()));
        assert!(result.is_err());

        let expected = [8, 9, 10, 11, 12, 13, 14, 15];
        let keys: Vec<u32> = map.keys().copied().collect();
        assert_eq!(keys, expected);
        assert_eq!(map.len(), expected.len());
        assert_eq!(map.last_key_value().map(|(k, _)| *k), Some(15));
        for key in expected {
            assert_eq!(map.get(&key).map(PanicOnDrop::id), Some(key));
        }

        map.clear();
        assert!(map.is_empty());
    }

    // MARK: split_off under a lying comparator

    #[test]
    fn split_off_chaos_is_well_formed() {
        use core::panic::AssertUnwindSafe;
        use std::panic::catch_unwind;

        use crate::test_util::{Chaos, assert_contract_panic, seeded_geometric, seeds};

        for seed in seeds() {
            let mut list: SkipMap<u8, u8, 16, Chaos, _> =
                SkipMap::with_comparator_and_level_generator(
                    Chaos::new(seed),
                    seeded_geometric(seed),
                );
            for k in 0_u8..24 {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    list.insert(k, 0_u8);
                }));
                assert_contract_panic(result, &[]);
            }
            list.assert_well_formed();

            for probe in [0_u8, 7, 12, 23, 40] {
                let mut other = None;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    other = Some(list.split_off(&probe));
                }));
                assert_contract_panic(result, &[]);
                list.assert_well_formed();
                if let Some(second) = other {
                    second.assert_well_formed();
                }
            }
        }
    }

    // MARK: head trimming

    #[test]
    fn insert_trims_the_head_after_removals() {
        let n: usize = if cfg!(miri) { 100 } else { 1_000 };
        let mut map = seeded(0..n, 3);
        while map.len() > 1 {
            map.pop_last();
        }
        map.insert(1, 1);
        // SAFETY: `head` is the live head of `map`, which owns every node it
        // reaches.
        let head = unsafe { map.head.as_ref() };
        let mut tallest = 0;
        let mut current = head.next_as_ref();
        while let Some(node) = current {
            tallest = tallest.max(node.level());
            current = node.next_as_ref();
        }
        assert_eq!(head.level(), tallest.max(1));
        map.assert_well_formed();
    }
}
