//! Key-based read and limited mutable access for [`SkipMap`](super::SkipMap).

use crate::{
    comparator::Comparator,
    level_generator::LevelGenerator,
    node::visitor::{IndexVisitor, OrdIndexVisitor, OrdMutVisitor, OrdVisitor, Visitor},
};

use super::SkipMap;

impl<K, V, const N: usize, C: Comparator<K>, G: LevelGenerator> SkipMap<K, V, N, C, G> {
    /// Returns references to the first (smallest-key) key-value pair, or
    /// `None` if the map is empty.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// assert_eq!(map.first_key_value(), None);
    /// map.insert(3, "c");
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// assert_eq!(map.first_key_value(), Some((&1, &"a")));
    /// ```
    #[inline]
    #[must_use]
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        let kv = self.head_ref().next_as_ref()?.value()?;
        Some((&kv.0, &kv.1))
    }

    /// Returns references to the last (largest-key) key-value pair, or `None`
    /// if the map is empty.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// assert_eq!(map.last_key_value(), None);
    /// map.insert(3, "c");
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// assert_eq!(map.last_key_value(), Some((&3, &"c")));
    /// ```
    #[inline]
    #[must_use]
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        // SAFETY: self.tail is Some iff len > 0, an invariant maintained by all
        // mutating operations.  The pointer remains valid for the lifetime of
        // &self.
        let kv = unsafe { self.tail?.as_ref() }.value()?;
        Some((&kv.0, &kv.1))
    }

    /// Returns `true` if the map contains any entry with the given key.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    ///
    /// assert!(map.contains_key(&1));
    /// assert!(!map.contains_key(&2));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        let cmp = |entry: &(K, V), k: &K| self.comparator.compare(&entry.0, k);
        OrdVisitor::new(self.head_ref(), key, cmp)
            .traverse()
            .is_some()
    }

    /// Returns a shared reference to the value for the given key, or `None`
    /// if the key is absent.
    ///
    /// When duplicate keys exist this may return any one of the matching
    /// values.
    ///
    /// This operation is `$O(\log n)$` on average.
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
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&3), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        let cmp = |entry: &(K, V), k: &K| self.comparator.compare(&entry.0, k);
        let node = OrdVisitor::new(self.head_ref(), key, cmp).traverse()?;
        Some(&node.value()?.1)
    }

    /// Returns a shared reference to the key-value pair whose key equals
    /// `key`, or `None` if the key is absent.
    ///
    /// When duplicate keys exist this may return any one of the matching
    /// entries.
    ///
    /// This operation is `$O(\log n)$` on average.
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
    /// assert_eq!(map.get_key_value(&1), Some((&1, &"a")));
    /// assert_eq!(map.get_key_value(&3), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
        let cmp = |entry: &(K, V), k: &K| self.comparator.compare(&entry.0, k);
        let node = OrdVisitor::new(self.head_ref(), key, cmp).traverse()?;
        let kv = node.value()?;
        Some((&kv.0, &kv.1))
    }

    /// Returns a mutable reference to the value for the given key, or `None`
    /// if the key is absent.
    ///
    /// Only the *value* may be mutated through this reference; modifying the
    /// key is not possible because doing so could violate the ordering
    /// invariant.
    ///
    /// When duplicate keys exist this may return any one of the matching
    /// entries.
    ///
    /// This operation is `$O(\log n)$` on average.
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
    /// if let Some(v) = map.get_mut(&1) {
    ///     *v += 5;
    /// }
    /// assert_eq!(map.get(&1), Some(&15));
    /// ```
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let cmp = |entry: &(K, V), k: &K| self.comparator.compare(&entry.0, k);
        let mut visitor = OrdMutVisitor::new(self.head, key, cmp);
        visitor.traverse();
        let (mut current, found, _precursors) = visitor.into_parts();
        if found {
            // SAFETY: `current` is a valid, exclusively-owned node pointer.
            // We only mutate the value component; the key is never touched, so
            // the ordering invariant is preserved.
            unsafe { Some(&mut current.as_mut().value_mut()?.1) }
        } else {
            None
        }
    }

    /// Returns the key-value pair at the given 0-based `index`, or `None` if
    /// `index` is out of bounds.
    ///
    /// This operation is `$O(\log n)$` on average.
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
    /// assert_eq!(map.get_by_index(0), Some((&1, &"a")));
    /// assert_eq!(map.get_by_index(1), Some((&2, &"b")));
    /// assert_eq!(map.get_by_index(2), Some((&3, &"c")));
    /// assert_eq!(map.get_by_index(3), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get_by_index(&self, index: usize) -> Option<(&K, &V)> {
        if index >= self.len {
            return None;
        }
        let node = IndexVisitor::new(self.head_ref(), index.saturating_add(1)).traverse()?;
        let kv = node.value()?;
        Some((&kv.0, &kv.1))
    }

    /// Returns the 0-based index of the first entry whose key compares equal
    /// to `key`, or `None` if no such entry is present.
    ///
    /// When duplicate keys exist the index of the first occurrence is returned.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(10, "ten");
    /// map.insert(20, "twenty");
    /// map.insert(30, "thirty");
    ///
    /// assert_eq!(map.rank(&10), Some(0));
    /// assert_eq!(map.rank(&20), Some(1));
    /// assert_eq!(map.rank(&30), Some(2));
    /// assert_eq!(map.rank(&99), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn rank(&self, key: &K) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        let cmp = |entry: &(K, V), k: &K| self.comparator.compare(&entry.0, k);
        let mut visitor = OrdIndexVisitor::new(self.head_ref(), key, cmp);
        visitor.traverse();
        visitor.found().then(|| visitor.rank())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::SkipMap;
    use crate::comparator::FnComparator;

    // MARK: first_key_value

    #[test]
    fn first_key_value_empty() {
        let map = SkipMap::<i32, &str>::new();
        assert_eq!(map.first_key_value(), None);
    }

    // Non-empty tests for first_key_value are exercised in
    // skip_map/insert_remove.rs once insert is implemented.

    // MARK: last_key_value

    #[test]
    fn last_key_value_empty() {
        let map = SkipMap::<i32, &str>::new();
        assert_eq!(map.last_key_value(), None);
    }

    // Non-empty tests for last_key_value are exercised in
    // skip_map/insert_remove.rs once insert is implemented.

    // MARK: contains_key

    #[test]
    fn contains_key_empty() {
        let map = SkipMap::<i32, &str>::new();
        assert!(!map.contains_key(&1));
    }

    #[test]
    fn contains_key_empty_custom_comparator() {
        let map: SkipMap<i32, &str, 16, _> =
            SkipMap::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        assert!(!map.contains_key(&99));
    }

    // MARK: get

    #[test]
    fn get_empty() {
        let map = SkipMap::<i32, &str>::new();
        assert_eq!(map.get(&1), None);
    }

    #[test]
    fn get_empty_custom_comparator() {
        let map: SkipMap<i32, &str, 16, _> =
            SkipMap::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        assert_eq!(map.get(&99), None);
    }

    // MARK: get_key_value

    #[test]
    fn get_key_value_empty() {
        let map = SkipMap::<i32, &str>::new();
        assert_eq!(map.get_key_value(&1), None);
    }

    // MARK: get_mut

    #[test]
    fn get_mut_empty() {
        let mut map = SkipMap::<i32, i32>::new();
        assert_eq!(map.get_mut(&1), None);
    }

    #[test]
    fn get_mut_empty_custom_comparator() {
        let mut map: SkipMap<i32, i32, 16, _> =
            SkipMap::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        assert_eq!(map.get_mut(&42), None);
    }

    // MARK: get_by_index

    #[test]
    fn get_by_index_empty() {
        let map = SkipMap::<i32, &str>::new();
        assert_eq!(map.get_by_index(0), None);
    }

    #[test]
    fn get_by_index_in_order() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(3, "c");
        map.insert(1, "a");
        map.insert(2, "b");
        assert_eq!(map.get_by_index(0), Some((&1, &"a")));
        assert_eq!(map.get_by_index(1), Some((&2, &"b")));
        assert_eq!(map.get_by_index(2), Some((&3, &"c")));
        assert_eq!(map.get_by_index(3), None);
    }

    #[test]
    fn get_by_index_out_of_bounds() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        assert_eq!(map.get_by_index(1), None);
        assert_eq!(map.get_by_index(usize::MAX), None);
    }

    #[test]
    fn get_by_index_large_map() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 0..20 {
            map.insert(i, i * 10);
        }
        for (i, expected_key) in (0..20_usize).zip(0..20_i32) {
            assert_eq!(
                map.get_by_index(i),
                Some((&expected_key, &(expected_key * 10)))
            );
        }
        assert_eq!(map.get_by_index(20), None);
    }

    // MARK: rank

    #[test]
    fn rank_empty() {
        let map = SkipMap::<i32, i32>::new();
        assert_eq!(map.rank(&1), None);
    }

    #[test]
    fn rank_present_keys() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(10, "ten");
        map.insert(20, "twenty");
        map.insert(30, "thirty");
        assert_eq!(map.rank(&10), Some(0));
        assert_eq!(map.rank(&20), Some(1));
        assert_eq!(map.rank(&30), Some(2));
    }

    #[test]
    fn rank_absent_key() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        map.insert(3, "c");
        assert_eq!(map.rank(&2), None);
        assert_eq!(map.rank(&99), None);
    }

    #[test]
    fn rank_first_of_duplicates() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(2, 21); // duplicate key
        map.insert(3, 30);
        // rank returns the index of the first matching key
        assert_eq!(map.rank(&2), Some(1));
    }

    #[test]
    fn get_by_index_and_rank_roundtrip() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in [5, 2, 8, 1, 9] {
            map.insert(i, i);
        }
        // Sorted order: 1, 2, 5, 8, 9
        for (idx, &key) in [1, 2, 5, 8, 9].iter().enumerate() {
            assert_eq!(map.rank(&key), Some(idx));
            assert_eq!(map.get_by_index(idx).map(|(&k, _)| k), Some(key));
        }
    }
}
