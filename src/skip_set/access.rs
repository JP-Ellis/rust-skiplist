//! Value-based read access for [`SkipSet`](super::SkipSet).

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
        self.inner.get_by_value(value)
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
}
