//! List-restructuring methods for [`SkipMap`](super::SkipMap):
//! `clear`, `append`, `split_off`, and `merge`.

use core::{cmp::Ordering, ptr::NonNull};

use super::SkipMap;
use crate::{
    comparator::Comparator,
    level_generator::LevelGenerator,
    node::{
        Node,
        visitor::{OrdMutVisitor, Visitor},
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
        let max_levels = self.head_ref().level();
        // Drop the old sentinel in-place.  `Node::drop` iterates the entire
        // `next` chain and frees each node one at a time, so this is O(n) and
        // non-recursive regardless of map size.  Then write a fresh sentinel
        // into the same allocation.
        //
        // SAFETY: `self.head` is a live, exclusively-owned allocation;
        // `drop_in_place` drops the old `Node<(K,V), N>` (and its linked chain),
        // leaving the memory valid but uninitialized.
        unsafe { core::ptr::drop_in_place(self.head.as_ptr()) };
        // SAFETY: The allocation is still live after `drop_in_place`; `write`
        // initializes it with a fresh sentinel, and no destructor runs on the
        // `write` side.
        unsafe { self.head.as_ptr().write(Node::new(max_levels)) };
        self.tail = None;
        self.len = 0;
    }

    /// Moves all entries from `other` into `self`, leaving `other` empty.
    ///
    /// Entries from `other` are merged into `self` at their sorted key positions
    /// according to `self`'s comparator.  After the call `self` contains all
    /// entries from both maps in sorted order and `other` is empty.  If a key
    /// from `other` already exists in `self`, the value from `other` overwrites
    /// the value in `self`.
    ///
    /// When every key of `other` is strictly greater than every key of `self`
    /// (the two sorted ranges are disjoint), this runs in `$O(n+m)$` time.
    /// Otherwise, when the key ranges overlap (including when
    /// `self.last_key == other.first_key`), each entry of `other` is inserted
    /// individually in `$O(m \log(n+m))$` time.
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
        clippy::missing_panics_doc,
        reason = "tail/head nodes always have a value; expects fire only on invariant violations"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "take_next_chain, set_head_next, and filter_rebuild touch provably disjoint \
                  heap nodes; splitting across blocks would require unsafe-crossing raw-pointer \
                  variables"
    )]
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        if other.is_empty() {
            return;
        }

        // Fast path: when self is empty, or every key of other is strictly
        // greater than every key of self, splice the raw node chains and
        // rebuild skip links in one O(n+m) pass.
        //
        // The condition is strictly `Less` (not `!= Greater`) so that when the
        // last key of self equals the first key of other, we fall through to
        // the slow path.  The slow path calls `insert`, which replaces the
        // existing value, matching the BTreeMap::append contract.
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
            // SAFETY: other.head is a valid head sentinel; we hold &mut other.
            let first_of_other = unsafe { (*other.head.as_ptr()).take_next_chain() };

            // Clear other.head's skip links: after take_next_chain they may
            // still point to nodes now belonging to self's chain.
            for link in other.head_mut().links_mut() {
                *link = None;
            }
            other.tail = None;
            other.len = 0;

            if let Some(first_nn) = first_of_other {
                let attach = self.tail.unwrap_or(self.head);
                // SAFETY: The attachment point (self.tail or self.head) has
                // next == None.  first_nn.prev was set to None by take_next_chain.
                unsafe { (*attach.as_ptr()).set_head_next(first_nn) };

                // Rebuild all skip links in one O(n+m) pass.
                //
                // SAFETY: self.head is exclusively owned; all reachable nodes
                // are live heap allocations with no other live references.
                let (new_len, new_tail) =
                    unsafe { Node::filter_rebuild(self.head, |_| true, |_| {}) };
                self.tail = new_tail;
                self.len = new_len;
            }
        } else {
            // Slow path: overlapping key ranges, insert each entry individually.
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
    #[expect(
        clippy::indexing_slicing,
        reason = "precursors[0] is valid because max_levels >= 1 (guaranteed by the \
                  LevelGenerator invariant), so precursors.len() >= 1"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "set_head_next is an unsafe fn called on a raw-pointer dereference; \
                  the two operations are on the same freshly allocated, exclusively-owned \
                  node and are provably disjoint from the take_next_chain and filter_rebuild \
                  calls above and below"
    )]
    #[inline]
    #[must_use]
    pub fn split_off(&mut self, key: &K) -> Self
    where
        C: Clone,
        G: Clone,
    {
        let max_levels = self.head_ref().level();

        // Use OrdMutVisitor to find the last node strictly < key.
        // The comparator projects entry.0 (the key) before comparing.
        //
        // `self.head` is `NonNull` (a `Copy` type) so copying it does not
        // borrow `self`.  The closure only borrows `self.comparator` (shared),
        // which is a distinct field.  Both borrows are released when `visitor`
        // is consumed by `into_parts()`.
        let pivot_nn = {
            let head = self.head;
            let cmp = |entry: &(K, V), k: &K| self.comparator.compare(&entry.0, k);
            let mut visitor = OrdMutVisitor::new(head, key, cmp);
            visitor.traverse();
            let (_current, _found, precursors) = visitor.into_parts();
            precursors[0]
        };

        // SAFETY: `Box::into_raw` transfers ownership; freed in `Drop`.
        let result_head =
            unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(max_levels)))) };
        let mut result = Self {
            head: result_head,
            tail: None,
            len: 0,
            comparator: self.comparator.clone(),
            generator: self.generator.clone(),
        };

        // Detach the ">= key" suffix from pivot_nn (the last node < key).
        //
        // SAFETY: `pivot_nn` is a valid, live node in this map (either the
        // head sentinel or a data node) for the duration of `&mut self`.  No
        // other live `&mut` references to any node exist.
        let first_nn = unsafe { (*pivot_nn.as_ptr()).take_next_chain() };

        if let Some(nn) = first_nn {
            // SAFETY: `result.head` is a freshly allocated, exclusively-owned
            // sentinel node.  `nn` is the first node of the detached chain
            // with no other live references.
            unsafe { (*result.head.as_ptr()).set_head_next(nn) };
        }

        // Rebuild skip links for both halves and compute their element counts.
        //
        // SAFETY: `self.head` is exclusively owned; every node reachable from
        // it is a live heap allocation with no other live references.
        let (self_len, self_tail) = unsafe { Node::filter_rebuild(self.head, |_| true, |_| {}) };
        // SAFETY: `result.head` is exclusively owned; every node reachable
        // from it is a live heap allocation with no other live references.
        let (result_len, result_tail) =
            unsafe { Node::filter_rebuild(result.head, |_| true, |_| {}) };

        self.tail = self_tail;
        self.len = self_len;
        result.tail = result_tail;
        result.len = result_len;

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
            if let Some(old_v) = self.remove(&k) {
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
    use pretty_assertions::assert_eq;

    use super::super::SkipMap;

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
}
