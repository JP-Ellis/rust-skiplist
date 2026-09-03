//! Predicate-based filtering for [`SkipMap`](super::SkipMap): `retain`.

use super::SkipMap;
use crate::{comparator::Comparator, level_generator::LevelGenerator, node::Node};

impl<K, V, const N: usize, C: Comparator<K>, G: LevelGenerator> SkipMap<K, V, N, C, G> {
    /// Retains only the key-value pairs for which the predicate returns `true`.
    ///
    /// All pairs are visited in ascending key order.  The predicate receives a
    /// shared reference to the key and an exclusive reference to the value,
    /// allowing values to be mutated in-place.  Pairs for which the predicate
    /// returns `false` are removed and dropped.
    ///
    /// This operation runs in `$O(n)$` time: every pair is visited exactly once.
    ///
    /// If `f` or an entry's destructor panics, the entries the walk had not
    /// yet reached are kept in the map, as is the entry `f` panicked on; the
    /// map stays usable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(1, "one");
    /// map.insert(2, "two");
    /// map.insert(3, "three");
    ///
    /// // Keep only pairs with even keys.
    /// map.retain(|k, _v| k % 2 == 0);
    ///
    /// assert_eq!(map.len(), 1);
    /// assert_eq!(map.get(&2), Some(&"two"));
    /// assert_eq!(map.get(&1), None);
    /// assert_eq!(map.get(&3), None);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "`value_mut()` returns None only for the head sentinel, which is never \
                  visited in the data-node walk; the expect fires only on invariant violations"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "calling filter_rebuild (unsafe fn) and dereferencing cur inside the keep \
                  closure are provably disjoint"
    )]
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        if self.is_empty() {
            return;
        }

        // SAFETY: All raw pointers originate from heap allocations owned by
        // this SkipMap.  We hold &mut self so no other reference to any node
        // exists.  The closure reads/mutates the value and returns before any
        // structural mutation occurs.
        // `filter_rebuild` publishes the new length and tail itself, so
        // an unwind out of the predicate or out of an element's destructor
        // cannot skip the update.
        unsafe {
            Node::filter_rebuild(
                self.head,
                &mut self.len,
                &mut self.tail,
                |cur| {
                    // SAFETY: cur is a live, heap-allocated data node; no
                    // other reference to this node exists within the closure.
                    let (k, v) = (*cur).value_mut().expect("data node has a value");
                    f(k, v)
                },
                |_| {},
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use pretty_assertions::assert_eq;

    use super::super::SkipMap;
    use crate::test_util::{PanicOnDrop, bombs};

    // MARK: retain

    #[test]
    fn retain_empty() {
        let mut map = SkipMap::<i32, i32>::new();
        map.retain(|_, _| true);
        assert!(map.is_empty());
        map.retain(|_, _| false);
        assert!(map.is_empty());
    }

    #[test]
    fn retain_all_kept() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 1..=5 {
            map.insert(i, i * 10);
        }
        map.retain(|_, _| true);
        assert_eq!(map.len(), 5);
        let got: Vec<(i32, i32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(got, [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
    }

    #[test]
    fn retain_all_dropped() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 1..=5 {
            map.insert(i, i * 10);
        }
        map.retain(|_, _| false);
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.first_key_value(), None);
        assert_eq!(map.last_key_value(), None);
    }

    #[test]
    fn retain_by_key() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearly expresses the intent of keeping even keys"
        )]
        map.retain(|k, _| k % 2 == 0);

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&2), Some(&"two"));
        assert_eq!(map.get(&1), None);
        assert_eq!(map.get(&3), None);
    }

    #[test]
    fn retain_by_value() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 1..=5 {
            map.insert(i, i * 10);
        }
        map.retain(|_, v| *v > 20);
        assert_eq!(map.len(), 3);
        let got: Vec<i32> = map.keys().copied().collect();
        assert_eq!(got, [3, 4, 5]);
    }

    #[test]
    fn retain_mutates_value() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 1..=4 {
            map.insert(i, i);
        }
        // Double odd values, drop even values.
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearly expresses the intent of keeping odd values"
        )]
        map.retain(|_, v| {
            if *v % 2 != 0 {
                *v *= 2;
                true
            } else {
                false
            }
        });
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&1), Some(&2));
        assert_eq!(map.get(&3), Some(&6));
        assert_eq!(map.get(&2), None);
        assert_eq!(map.get(&4), None);
    }

    #[test]
    fn retain_tail_pointer_correct() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 1..=5 {
            map.insert(i, i * 10);
        }
        map.retain(|k, _| *k <= 3);
        assert_eq!(map.last_key_value(), Some((&3, &30)));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn retain_links_consistent() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 0..20 {
            map.insert(i, i * 10);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearly expresses the intent of keeping even keys"
        )]
        map.retain(|k, _| k % 2 == 0);
        let got: Vec<i32> = map.keys().copied().collect();
        let expected: Vec<i32> = (0..20).step_by(2).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn retain_single_kept() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(42, "answer");
        map.retain(|_, _| true);
        assert_eq!(map.len(), 1);
        assert_eq!(map.first_key_value(), Some((&42, &"answer")));
        assert_eq!(map.last_key_value(), Some((&42, &"answer")));
    }

    #[test]
    fn retain_single_dropped() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(42, "answer");
        map.retain(|_, _| false);
        assert!(map.is_empty());
        assert_eq!(map.first_key_value(), None);
        assert_eq!(map.last_key_value(), None);
    }

    #[test]
    fn retain_after_retain_is_correct() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 0..10 {
            map.insert(i, i * 10);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearly expresses the intent of keeping even keys"
        )]
        map.retain(|k, _| k % 2 == 0);
        map.retain(|k, _| *k > 2);
        let got: Vec<i32> = map.keys().copied().collect();
        assert_eq!(got, [4, 6, 8]);
    }

    // MARK: panic safety

    #[test]
    fn retain_panicking_drop_keeps_unvisited_entries() {
        let mut map = SkipMap::<u32, PanicOnDrop>::new();
        for bomb in bombs(16, 7) {
            map.insert(bomb.id(), bomb);
        }

        let result = catch_unwind(AssertUnwindSafe(|| map.retain(|_, _| false)));
        assert!(result.is_err());

        let expected = [8, 9, 10, 11, 12, 13, 14, 15];
        let keys: Vec<u32> = map.keys().copied().collect();
        assert_eq!(keys, expected);
        assert_eq!(map.len(), expected.len());
        assert_eq!(map.first_key_value().map(|(k, _)| *k), Some(8));
        assert_eq!(map.last_key_value().map(|(k, _)| *k), Some(15));
        for key in expected {
            assert_eq!(map.get(&key).map(PanicOnDrop::id), Some(key));
        }
    }

    #[test]
    fn retain_panicking_predicate_keeps_unvisited_entries() {
        let mut map = SkipMap::<u32, PanicOnDrop>::new();
        for bomb in bombs(16, 16) {
            map.insert(bomb.id(), bomb);
        }

        let result = catch_unwind(AssertUnwindSafe(|| {
            map.retain(|key, _value| {
                assert!(*key != 7, "predicate panic");
                false
            });
        }));
        assert!(result.is_err());

        let expected = [7, 8, 9, 10, 11, 12, 13, 14, 15];
        let keys: Vec<u32> = map.keys().copied().collect();
        assert_eq!(keys, expected);
        assert_eq!(map.len(), expected.len());
        assert_eq!(map.last_key_value().map(|(k, _)| *k), Some(15));
        for key in expected {
            assert_eq!(map.get(&key).map(PanicOnDrop::id), Some(key));
        }
    }
}
