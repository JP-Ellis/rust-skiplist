//! Value-based read access for [`SkipSet`](super::SkipSet).

use core::ops::Index;

use crate::{comparator::Comparator, level_generator::LevelGenerator, skip_set::SkipSet};

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> SkipSet<T, N, C, G> {
    /// Returns a shared reference to the first (smallest) element, or `None`
    /// if the set is empty.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set = SkipSet::<i32>::new();
    /// assert_eq!(set.first(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn first(&self) -> Option<&T> {
        self.inner.first()
    }

    /// Returns a shared reference to the last (largest) element, or `None`
    /// if the set is empty.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set = SkipSet::<i32>::new();
    /// assert_eq!(set.last(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        self.inner.last()
    }

    /// Returns `true` if the set contains an element that compares equal to
    /// `value`.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set = SkipSet::<i32>::new();
    /// assert!(!set.contains(&1));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains(&self, value: &T) -> bool {
        self.inner.contains(value)
    }

    /// Returns a shared reference to the element in the set that compares
    /// equal to `value`, or `None` if no such element is present.
    ///
    /// Because `SkipSet` forbids duplicates, at most one element can compare
    /// equal to `value`.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set = SkipSet::<i32>::new();
    /// assert_eq!(set.get(&1), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self, value: &T) -> Option<&T> {
        self.inner.get_fast(value)
    }

    /// Returns a shared reference to the element at the given 0-based `index`
    /// in sorted order, or `None` if `index` is out of bounds.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// for v in [3, 1, 2] { set.insert(v); }
    ///
    /// assert_eq!(set.get_by_index(0), Some(&1));
    /// assert_eq!(set.get_by_index(1), Some(&2));
    /// assert_eq!(set.get_by_index(2), Some(&3));
    /// assert_eq!(set.get_by_index(3), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get_by_index(&self, index: usize) -> Option<&T> {
        self.inner.get_by_index(index)
    }

    /// Returns the 0-based rank (sorted position) of `value` in the set, or
    /// `None` if `value` is not present.
    ///
    /// Because `SkipSet` forbids duplicates, the rank of a present element is
    /// unambiguous.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// for v in [10, 20, 30] { set.insert(v); }
    ///
    /// assert_eq!(set.rank(&10), Some(0));
    /// assert_eq!(set.rank(&20), Some(1));
    /// assert_eq!(set.rank(&30), Some(2));
    /// assert_eq!(set.rank(&99), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn rank(&self, value: &T) -> Option<usize> {
        self.inner.rank(value)
    }
}

// MARK: Index

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> Index<usize> for SkipSet<T, N, C, G> {
    type Output = T;

    /// Returns a shared reference to the element at `index` in sorted order.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// for v in [3, 1, 2] { set.insert(v); }
    ///
    /// assert_eq!(set[0], 1);
    /// assert_eq!(set[1], 2);
    /// assert_eq!(set[2], 3);
    /// ```
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        reason = "delegates to OrderedSkipList::index which panics with a clear message on OOB"
    )]
    fn index(&self, index: usize) -> &T {
        &self.inner[index]
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::SkipSet;
    use crate::{comparator::FnComparator, ordered_skip_list::OrderedSkipList};

    fn make_set(values: &[i32]) -> SkipSet<i32> {
        let mut list = OrderedSkipList::new();
        for &v in values {
            list.insert(v);
        }
        SkipSet { inner: list }
    }

    // MARK: first

    #[test]
    fn first_empty() {
        let set = SkipSet::<i32>::new();
        assert_eq!(set.first(), None);
    }

    #[test]
    fn first_single() {
        let set = make_set(&[42]);
        assert_eq!(set.first(), Some(&42));
    }

    #[test]
    fn first_multiple() {
        let set = make_set(&[3, 1, 2]);
        assert_eq!(set.first(), Some(&1));
    }

    #[test]
    fn first_custom_comparator() {
        // Largest-first ordering: "first" is the largest value.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        for v in [3, 1, 2] {
            list.insert(v);
        }
        let set: SkipSet<i32, 16, _> = SkipSet { inner: list };
        assert_eq!(set.first(), Some(&3));
    }

    // MARK: last

    #[test]
    fn last_empty() {
        let set = SkipSet::<i32>::new();
        assert_eq!(set.last(), None);
    }

    #[test]
    fn last_single() {
        let set = make_set(&[42]);
        assert_eq!(set.last(), Some(&42));
    }

    #[test]
    fn last_multiple() {
        let set = make_set(&[3, 1, 2]);
        assert_eq!(set.last(), Some(&3));
    }

    #[test]
    fn last_custom_comparator() {
        // Largest-first ordering: "last" is the smallest value.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        for v in [3, 1, 2] {
            list.insert(v);
        }
        let set: SkipSet<i32, 16, _> = SkipSet { inner: list };
        assert_eq!(set.last(), Some(&1));
    }

    // MARK: contains

    #[test]
    fn contains_empty() {
        let set = SkipSet::<i32>::new();
        assert!(!set.contains(&1));
    }

    #[test]
    fn contains_present_first() {
        let set = make_set(&[1, 2, 3]);
        assert!(set.contains(&1));
    }

    #[test]
    fn contains_present_last() {
        let set = make_set(&[1, 2, 3]);
        assert!(set.contains(&3));
    }

    #[test]
    fn contains_present_middle() {
        let set = make_set(&[1, 2, 3]);
        assert!(set.contains(&2));
    }

    #[test]
    fn contains_absent_less() {
        let set = make_set(&[5, 10]);
        assert!(!set.contains(&1));
    }

    #[test]
    fn contains_absent_greater() {
        let set = make_set(&[1, 5]);
        assert!(!set.contains(&99));
    }

    #[test]
    fn contains_absent_between() {
        let set = make_set(&[1, 3, 5]);
        assert!(!set.contains(&2));
        assert!(!set.contains(&4));
    }

    #[test]
    fn contains_custom_comparator() {
        // Largest-first ordering.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        for v in [3, 1, 2] {
            list.insert(v);
        }
        let set: SkipSet<i32, 16, _> = SkipSet { inner: list };
        assert!(set.contains(&2));
        assert!(!set.contains(&4));
    }

    // MARK: get

    #[test]
    fn get_empty() {
        let set = SkipSet::<i32>::new();
        assert_eq!(set.get(&1), None);
    }

    #[test]
    fn get_present() {
        let set = make_set(&[1, 2, 3]);
        assert_eq!(set.get(&2), Some(&2));
    }

    #[test]
    fn get_absent() {
        let set = make_set(&[1, 3]);
        assert_eq!(set.get(&2), None);
    }

    #[test]
    fn get_first_element() {
        let set = make_set(&[1, 2, 3]);
        assert_eq!(set.get(&1), Some(&1));
    }

    #[test]
    fn get_last_element() {
        let set = make_set(&[1, 2, 3]);
        assert_eq!(set.get(&3), Some(&3));
    }

    #[test]
    fn get_custom_comparator() {
        // Largest-first ordering.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        for v in [3, 1, 2] {
            list.insert(v);
        }
        let set: SkipSet<i32, 16, _> = SkipSet { inner: list };
        assert_eq!(set.get(&2), Some(&2));
        assert_eq!(set.get(&4), None);
    }

    // MARK: get_by_index

    #[test]
    fn get_by_index_empty() {
        let set = SkipSet::<i32>::new();
        assert_eq!(set.get_by_index(0), None);
    }

    #[test]
    fn get_by_index_in_bounds() {
        let set = make_set(&[3, 1, 2]);
        assert_eq!(set.get_by_index(0), Some(&1));
        assert_eq!(set.get_by_index(1), Some(&2));
        assert_eq!(set.get_by_index(2), Some(&3));
    }

    #[test]
    fn get_by_index_out_of_bounds() {
        let set = make_set(&[1, 2, 3]);
        assert_eq!(set.get_by_index(3), None);
        assert_eq!(set.get_by_index(99), None);
    }

    #[test]
    fn get_by_index_single_element() {
        let set = make_set(&[42]);
        assert_eq!(set.get_by_index(0), Some(&42));
        assert_eq!(set.get_by_index(1), None);
    }

    #[test]
    fn get_by_index_large() {
        let values: Vec<i32> = (0..100).collect();
        let set = make_set(&values);
        for i in 0..100_usize {
            let expected = i32::try_from(i).expect("i < 100");
            assert_eq!(set.get_by_index(i), Some(&expected));
        }
        assert_eq!(set.get_by_index(100), None);
    }

    #[test]
    fn get_by_index_custom_comparator() {
        // Reverse ordering: stored [3, 2, 1].
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        for v in [1, 2, 3] {
            list.insert(v);
        }
        let set: SkipSet<i32, 16, _> = SkipSet { inner: list };
        assert_eq!(set.get_by_index(0), Some(&3));
        assert_eq!(set.get_by_index(1), Some(&2));
        assert_eq!(set.get_by_index(2), Some(&1));
        assert_eq!(set.get_by_index(3), None);
    }

    // MARK: rank

    #[test]
    fn rank_empty() {
        let set = SkipSet::<i32>::new();
        assert_eq!(set.rank(&1), None);
    }

    #[test]
    fn rank_not_found() {
        let set = make_set(&[1, 3, 5]);
        assert_eq!(set.rank(&2), None);
        assert_eq!(set.rank(&99), None);
    }

    #[test]
    fn rank_single() {
        let set = make_set(&[42]);
        assert_eq!(set.rank(&42), Some(0));
    }

    #[test]
    fn rank_first() {
        let set = make_set(&[10, 20, 30]);
        assert_eq!(set.rank(&10), Some(0));
    }

    #[test]
    fn rank_last() {
        let set = make_set(&[10, 20, 30]);
        assert_eq!(set.rank(&30), Some(2));
    }

    #[test]
    fn rank_middle() {
        let set = make_set(&[3, 1, 4, 2]);
        assert_eq!(set.rank(&1), Some(0));
        assert_eq!(set.rank(&2), Some(1));
        assert_eq!(set.rank(&3), Some(2));
        assert_eq!(set.rank(&4), Some(3));
    }

    #[test]
    fn rank_large() {
        let values: Vec<i32> = (0..50).collect();
        let set = make_set(&values);
        for i in 0..50_i32 {
            let expected = usize::try_from(i).expect("non-negative");
            assert_eq!(set.rank(&i), Some(expected));
        }
    }

    #[test]
    fn rank_custom_comparator() {
        // Reverse ordering: stored [3, 2, 1].
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        for v in [1, 2, 3] {
            list.insert(v);
        }
        let set: SkipSet<i32, 16, _> = SkipSet { inner: list };
        assert_eq!(set.rank(&3), Some(0));
        assert_eq!(set.rank(&2), Some(1));
        assert_eq!(set.rank(&1), Some(2));
        assert_eq!(set.rank(&99), None);
    }

    // MARK: Index

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "testing valid indexing with known in-bounds indices"
    )]
    fn index_in_bounds() {
        let set = make_set(&[3, 1, 2]);
        assert_eq!(set[0], 1);
        assert_eq!(set[1], 2);
        assert_eq!(set[2], 3);
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "testing out-of-bounds indexing panics"
    )]
    #[should_panic(expected = "index out of bounds")]
    fn index_out_of_bounds_panics() {
        let set = make_set(&[1, 2, 3]);
        let _: i32 = set[3];
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "testing out-of-bounds indexing panics on empty set"
    )]
    #[should_panic(expected = "index out of bounds")]
    fn index_empty_panics() {
        let set = SkipSet::<i32>::new();
        let _: i32 = set[0];
    }
}
