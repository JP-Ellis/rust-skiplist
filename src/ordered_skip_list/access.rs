//! Value-based read access for [`OrderedSkipList`](super::OrderedSkipList).

use core::cmp::Ordering;

use crate::{
    comparator::Comparator,
    level_generator::LevelGenerator,
    node::visitor::{IndexVisitor, OrdIndexVisitor, OrdVisitor, Visitor},
    ordered_skip_list::OrderedSkipList,
};

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> OrderedSkipList<T, N, C, G> {
    /// Returns a shared reference to the first (smallest) element, or `None`
    /// if the list is empty.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// assert_eq!(list.first(), None);
    /// list.insert(3);
    /// list.insert(1);
    /// list.insert(2);
    /// assert_eq!(list.first(), Some(&1));
    /// ```
    #[inline]
    #[must_use]
    pub fn first(&self) -> Option<&T> {
        self.head_ref().next_as_ref()?.value()
    }

    /// Returns a shared reference to the last (largest) element, or `None`
    /// if the list is empty.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// assert_eq!(list.last(), None);
    /// list.insert(3);
    /// list.insert(1);
    /// list.insert(2);
    /// assert_eq!(list.last(), Some(&3));
    /// ```
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        // SAFETY: self.tail is Some iff len > 0, an invariant maintained by all
        // mutating operations. The pointer remains valid for the lifetime of &self.
        unsafe { self.tail?.as_ref().value() }
    }

    /// Returns `true` if the list contains an element that compares equal to
    /// `value`.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(1);
    /// list.insert(3);
    ///
    /// assert!(list.contains(&1));
    /// assert!(!list.contains(&2));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains(&self, value: &T) -> bool {
        let head = self.head_ref();
        let cmp = |v: &T, t: &T| self.comparator.compare(v, t);
        OrdVisitor::new(head, value, cmp).traverse().is_some()
    }

    /// Returns a shared reference to the first element that compares equal to
    /// `value`, or `None` if no such element is present.
    ///
    /// When duplicates exist the reference points to the first (earliest-
    /// inserted) occurrence.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(1);
    /// list.insert(2);
    /// list.insert(3);
    ///
    /// assert_eq!(list.get_by_value(&2), Some(&2));
    /// assert_eq!(list.get_by_value(&4), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get_by_value(&self, value: &T) -> Option<&T> {
        let head = self.head_ref();
        let cmp = |v: &T, t: &T| self.comparator.compare(v, t);
        OrdVisitor::new(head, value, cmp).traverse()?.value()
    }

    /// Returns a shared reference to the element at the given 0-based `index`,
    /// or `None` if `index` is out of bounds.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(3);
    /// list.insert(1);
    /// list.insert(2);
    ///
    /// assert_eq!(list.get(0), Some(&1));
    /// assert_eq!(list.get(1), Some(&2));
    /// assert_eq!(list.get(2), Some(&3));
    /// assert_eq!(list.get(3), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        IndexVisitor::new(self.head_ref(), index.saturating_add(1))
            .traverse()
            .and_then(|node| node.value())
    }

    /// Returns the 0-based index of the first element that compares equal to
    /// `value`, or `None` if no such element is present.
    ///
    /// When duplicates exist the index of the first occurrence is returned.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(10);
    /// list.insert(20);
    /// list.insert(30);
    ///
    /// assert_eq!(list.rank(&10), Some(0));
    /// assert_eq!(list.rank(&20), Some(1));
    /// assert_eq!(list.rank(&30), Some(2));
    /// assert_eq!(list.rank(&99), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn rank(&self, value: &T) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        let head = self.head_ref();
        let cmp = |v: &T, t: &T| self.comparator.compare(v, t);
        let mut visitor = OrdIndexVisitor::new(head, value, cmp);
        visitor.traverse();
        visitor.found().then(|| visitor.rank())
    }

    /// Returns the number of elements that compare equal to `value`.
    ///
    /// Returns `0` if no such element is present.
    ///
    /// This operation is `$O(\log n + k)$` where k is the count of equal elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(1);
    /// list.insert(2);
    /// list.insert(2);
    /// list.insert(3);
    ///
    /// assert_eq!(list.count(&2), 2);
    /// assert_eq!(list.count(&1), 1);
    /// assert_eq!(list.count(&99), 0);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "found() being true guarantees traverse() returned Some; \
                  the expect fires only on an internal invariant violation"
    )]
    #[inline]
    #[must_use]
    pub fn count(&self, value: &T) -> usize {
        if self.is_empty() {
            return 0;
        }
        // Use OrdIndexVisitor (Less-only advancement) to always land on the
        // *first* occurrence of `value`. OrdVisitor follows Equal skip links
        // which can skip earlier duplicates when multiple equal nodes exist.
        let head = self.head_ref();
        let cmp = |v: &T, t: &T| self.comparator.compare(v, t);
        let mut visitor = OrdIndexVisitor::new(head, value, cmp);
        let first = visitor.traverse();
        if !visitor.found() {
            return 0;
        }
        let first_node = first.expect("found implies Some");
        let mut count = 1_usize;
        let mut cur = first_node.next_as_ref();
        while let Some(node) = cur {
            match node.value() {
                Some(v) if self.comparator.compare(v, value) == Ordering::Equal => {
                    count = count.saturating_add(1);
                    cur = node.next_as_ref();
                }
                _ => break,
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::OrderedSkipList;
    use crate::{comparator::FnComparator, level_generator::geometric::Geometric};

    // MARK: contains

    #[test]
    fn contains_empty_list() {
        let list = OrderedSkipList::<i32>::new();
        assert!(!list.contains(&1));
    }

    #[test]
    fn contains_single_element_present() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(5);
        assert!(list.contains(&5));
    }

    #[test]
    fn contains_single_element_absent_less() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(5);
        assert!(!list.contains(&1));
    }

    #[test]
    fn contains_single_element_absent_greater() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(5);
        assert!(!list.contains(&10));
    }

    #[test]
    fn contains_first_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert!(list.contains(&1));
    }

    #[test]
    fn contains_last_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert!(list.contains(&3));
    }

    #[test]
    fn contains_middle_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert!(list.contains(&2));
    }

    #[test]
    fn contains_absent_between_elements() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(3);
        list.insert(5);
        assert!(!list.contains(&2));
        assert!(!list.contains(&4));
    }

    #[test]
    fn contains_absent_before_first() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(5);
        list.insert(10);
        assert!(!list.contains(&1));
    }

    #[test]
    fn contains_absent_after_last() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(5);
        assert!(!list.contains(&99));
    }

    #[test]
    fn contains_duplicate() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(2);
        list.insert(2);
        list.insert(2);
        assert!(list.contains(&2));
    }

    #[test]
    fn contains_custom_comparator() {
        // Largest-first ordering.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(3);
        list.insert(1);
        list.insert(2);
        assert!(list.contains(&2));
        assert!(!list.contains(&4));
    }

    // MARK: get_by_value

    #[test]
    fn get_by_value_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(list.get_by_value(&1), None);
    }

    #[test]
    fn get_by_value_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get_by_value(&2), Some(&2));
    }

    #[test]
    fn get_by_value_not_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(3);
        assert_eq!(list.get_by_value(&2), None);
    }

    #[test]
    fn get_by_value_first() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get_by_value(&1), Some(&1));
    }

    #[test]
    fn get_by_value_last() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get_by_value(&3), Some(&3));
    }

    #[test]
    fn get_by_value_duplicate_returns_a_match() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get_by_value(&2), Some(&2));
    }

    #[test]
    fn get_by_value_custom_comparator() {
        // Largest-first ordering.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(3);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.get_by_value(&2), Some(&2));
        assert_eq!(list.get_by_value(&4), None);
    }

    // MARK: get

    #[test]
    fn get_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(list.get(0), None);
    }

    #[test]
    fn get_in_bounds() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [3, 1, 2] {
            list.insert(i);
        }
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&3));
    }

    #[test]
    fn get_out_of_bounds() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        assert_eq!(list.get(1), None);
        assert_eq!(list.get(99), None);
    }

    #[test]
    fn get_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.get(0), Some(&42));
        assert_eq!(list.get(1), None);
    }

    #[test]
    fn get_large_list() {
        let mut list = OrderedSkipList::<usize>::new();
        for i in 0..100_usize {
            list.insert(i);
        }
        for i in 0..100_usize {
            assert_eq!(list.get(i), Some(&i));
        }
        assert_eq!(list.get(100), None);
    }

    #[test]
    fn get_with_duplicates() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&2));
        assert_eq!(list.get(3), Some(&3));
    }

    #[test]
    fn get_after_removals() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        list.remove_first(&3);
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&4));
        assert_eq!(list.get(3), Some(&5));
        assert_eq!(list.get(4), None);
    }

    #[test]
    fn get_custom_comparator() {
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(1);
        list.insert(2);
        list.insert(3);
        // Stored as [3, 2, 1] with reverse ordering.
        assert_eq!(list.get(0), Some(&3));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&1));
    }

    // MARK: rank

    #[test]
    fn rank_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(list.rank(&1), None);
    }

    #[test]
    fn rank_not_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(3);
        assert_eq!(list.rank(&2), None);
        assert_eq!(list.rank(&99), None);
    }

    #[test]
    fn rank_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.rank(&42), Some(0));
    }

    #[test]
    fn rank_first_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(10);
        list.insert(20);
        list.insert(30);
        assert_eq!(list.rank(&10), Some(0));
    }

    #[test]
    fn rank_last_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(10);
        list.insert(20);
        list.insert(30);
        assert_eq!(list.rank(&30), Some(2));
    }

    #[test]
    fn rank_middle_element() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [3, 1, 4, 2] {
            list.insert(i);
        }
        assert_eq!(list.rank(&1), Some(0));
        assert_eq!(list.rank(&2), Some(1));
        assert_eq!(list.rank(&3), Some(2));
        assert_eq!(list.rank(&4), Some(3));
    }

    #[test]
    fn rank_large_list() {
        let mut list = OrderedSkipList::<usize>::new();
        for i in 0..50_usize {
            list.insert(i);
        }
        for i in 0..50_usize {
            assert_eq!(list.rank(&i), Some(i));
        }
    }

    #[test]
    fn rank_with_duplicate_returns_first() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        // First occurrence of 2 is at index 1.
        assert_eq!(list.rank(&2), Some(1));
    }

    #[test]
    fn rank_custom_comparator() {
        // Reverse ordering: stored [3, 2, 1].
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.rank(&3), Some(0));
        assert_eq!(list.rank(&2), Some(1));
        assert_eq!(list.rank(&1), Some(2));
    }

    // MARK: count

    #[test]
    fn count_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(list.count(&1), 0);
    }

    #[test]
    fn count_not_present() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(3);
        assert_eq!(list.count(&2), 0);
    }

    #[test]
    fn count_single_occurrence() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.count(&2), 1);
    }

    #[test]
    fn count_multiple_occurrences() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.count(&2), 3);
    }

    #[test]
    fn count_all_same() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(5);
        list.insert(5);
        list.insert(5);
        assert_eq!(list.count(&5), 3);
    }

    #[test]
    fn count_first_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.count(&1), 2);
        assert_eq!(list.count(&2), 1);
    }

    #[test]
    fn count_custom_comparator() {
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(2);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.count(&2), 2);
        assert_eq!(list.count(&3), 1);
        assert_eq!(list.count(&99), 0);
    }

    // MARK: first

    #[test]
    fn first_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(list.first(), None);
    }

    #[test]
    fn first_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.first(), Some(&42));
    }

    #[test]
    fn first_multiple_elements() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(3);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.first(), Some(&1));
    }

    #[test]
    fn first_custom_comparator() {
        // Largest-first ordering: "first" is the element that sorts first,
        // i.e. the largest value.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(3);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.first(), Some(&3));
    }

    // MARK: last

    #[test]
    fn last_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(list.last(), None);
    }

    #[test]
    fn last_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.last(), Some(&42));
    }

    #[test]
    fn last_multiple_elements() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(3);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.last(), Some(&3));
    }

    #[test]
    fn last_custom_comparator() {
        // Largest-first ordering: "last" is the element that sorts last,
        // i.e. the smallest value.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(3);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.last(), Some(&1));
    }
}
