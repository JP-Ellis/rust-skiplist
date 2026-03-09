//! Value-based read access for [`OrderedSkipList`](super::OrderedSkipList).

use core::{cmp::Ordering, ops::Index};

use crate::{
    comparator::{Comparator, ComparatorKey},
    level_generator::LevelGenerator,
    node::visitor::{IndexVisitor, OrdIndexVisitor, OrdVisitor, Visitor},
    ordered_skip_list::OrderedSkipList,
};

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> OrderedSkipList<T, N, C, G> {
    /// Returns a shared reference to the comparator used by this list.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use skiplist::comparator::OrdComparator;
    ///
    /// let list = OrderedSkipList::<i32>::new();
    /// let _cmp: &OrdComparator = list.comparator();
    /// ```
    #[inline]
    #[must_use]
    pub fn comparator(&self) -> &C {
        &self.comparator
    }

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
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        let head = self.head_ref();
        let cmp = |v: &T, q: &Q| self.comparator.compare_key(v, q);
        OrdVisitor::new(head, value, cmp).traverse().is_some()
    }

    /// Returns a shared reference to an element that compares equal to
    /// `value`, or `None` if no such element is present.
    ///
    /// This is an alias for [`get_fast`](OrderedSkipList::get_fast).
    /// When duplicates exist this may return any one of the equal occurrences;
    /// use [`get_first`] or [`get_last`] when the specific occurrence matters.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// [`get_first`]: OrderedSkipList::get_first
    /// [`get_last`]: OrderedSkipList::get_last
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
    /// assert_eq!(list.get(&2), Some(&2));
    /// assert_eq!(list.get(&4), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        self.get_fast(value)
    }

    /// Returns a shared reference to an element that compares equal to
    /// `value`, or `None` if no such element is present.
    ///
    /// When duplicates exist this may return any one of the equal occurrences;
    /// use [`get_first`] or [`get_last`] when the specific occurrence matters.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// [`get_first`]: OrderedSkipList::get_first
    /// [`get_last`]: OrderedSkipList::get_last
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
    /// assert_eq!(list.get_fast(&2), Some(&2));
    /// assert_eq!(list.get_fast(&4), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get_fast<Q>(&self, value: &Q) -> Option<&T>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        let head = self.head_ref();
        let cmp = |v: &T, q: &Q| self.comparator.compare_key(v, q);
        OrdVisitor::new(head, value, cmp).traverse()?.value()
    }

    /// Returns a shared reference to the **first** (lowest-rank) element that
    /// compares equal to `value`, or `None` if no such element is present.
    ///
    /// When duplicates exist this always returns the occurrence at the lowest
    /// rank, consistent with [`rank`] and [`count`].
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// [`rank`]: OrderedSkipList::rank
    /// [`count`]: OrderedSkipList::count
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
    /// // get_first returns rank 1 (the first 2).
    /// assert_eq!(list.get_first(&2), Some(&2));
    /// assert_eq!(list.rank(&2), Some(1));
    /// ```
    #[inline]
    #[must_use]
    pub fn get_first<Q>(&self, value: &Q) -> Option<&T>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        let head = self.head_ref();
        let cmp = |v: &T, q: &Q| self.comparator.compare_key(v, q);
        // OrdIndexVisitor advances only on strict Less, guaranteeing that
        // traversal lands on the *first* occurrence when duplicates exist.
        let mut visitor = OrdIndexVisitor::new(head, value, cmp);
        visitor.traverse()?.value()
    }

    /// Returns a shared reference to the **last** (highest-rank) element that
    /// compares equal to `value`, or `None` if no such element is present.
    ///
    /// When duplicates exist this always returns the occurrence at the highest
    /// rank.
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
    /// list.insert(2);
    /// list.insert(3);
    ///
    /// // get_last returns rank 2 (the second 2).
    /// assert_eq!(list.get_last(&2), Some(&2));
    /// assert_eq!(list.rank(&2), Some(1)); // first is still at rank 1
    /// ```
    #[inline]
    #[must_use]
    pub fn get_last<Q>(&self, value: &Q) -> Option<&T>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        let head = self.head_ref();
        // Treat Equal as Less so that traversal advances through *all* equal
        // nodes. After exhaustion, current() is the last equal node (or a
        // node that compares Less if none exist).
        let cmp_past = |v: &T, q: &Q| match self.comparator.compare_key(v, q) {
            core::cmp::Ordering::Equal | core::cmp::Ordering::Less => core::cmp::Ordering::Less,
            core::cmp::Ordering::Greater => core::cmp::Ordering::Greater,
        };
        let mut visitor = OrdVisitor::new(head, value, cmp_past);
        visitor.traverse();
        let current = visitor.current();
        // Verify the last node visited actually equals the target (it would be
        // a Less node when no matching element exists at all).
        current
            .value()
            .filter(|v| self.comparator.compare_key(v, value) == core::cmp::Ordering::Equal)
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
    /// assert_eq!(list.get_by_index(0), Some(&1));
    /// assert_eq!(list.get_by_index(1), Some(&2));
    /// assert_eq!(list.get_by_index(2), Some(&3));
    /// assert_eq!(list.get_by_index(3), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get_by_index(&self, index: usize) -> Option<&T> {
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
    pub fn rank<Q>(&self, value: &Q) -> Option<usize>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        if self.is_empty() {
            return None;
        }
        let head = self.head_ref();
        let cmp = |v: &T, q: &Q| self.comparator.compare_key(v, q);
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
    pub fn count<Q>(&self, value: &Q) -> usize
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        if self.is_empty() {
            return 0;
        }
        // Use OrdIndexVisitor (Less-only advancement) to always land on the
        // *first* occurrence of `value`. OrdVisitor follows Equal skip links
        // which can skip earlier duplicates when multiple equal nodes exist.
        let head = self.head_ref();
        let cmp = |v: &T, q: &Q| self.comparator.compare_key(v, q);
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
                Some(v) if self.comparator.compare_key(v, value) == Ordering::Equal => {
                    count = count.saturating_add(1);
                    cur = node.next_as_ref();
                }
                _ => break,
            }
        }
        count
    }
}

// MARK: Index

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> Index<usize>
    for OrderedSkipList<T, N, C, G>
{
    type Output = T;

    /// Returns a reference to the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
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
    /// assert_eq!(list[0], 1);
    /// assert_eq!(list[1], 2);
    /// assert_eq!(list[2], 3);
    /// ```
    #[inline]
    #[expect(
        clippy::unwrap_used,
        reason = "index < self.len was just asserted, so get() always returns Some"
    )]
    fn index(&self, index: usize) -> &T {
        assert!(
            index < self.len,
            "index out of bounds: the len is {} but the index is {index}",
            self.len
        );
        self.get_by_index(index).unwrap()
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

    // MARK: get / get_fast

    #[test]
    fn get_fast_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(list.get_fast(&1), None);
    }

    #[test]
    fn get_fast_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get_fast(&2), Some(&2));
    }

    #[test]
    fn get_fast_not_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(3);
        assert_eq!(list.get_fast(&2), None);
    }

    #[test]
    fn get_fast_first() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get_fast(&1), Some(&1));
    }

    #[test]
    fn get_fast_last() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get_fast(&3), Some(&3));
    }

    #[test]
    fn get_fast_duplicate_returns_some_match() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get_fast(&2), Some(&2));
    }

    #[test]
    fn get_fast_custom_comparator() {
        // Largest-first ordering.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(3);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.get_fast(&2), Some(&2));
        assert_eq!(list.get_fast(&4), None);
    }

    #[test]
    fn get_delegates_to_get_fast() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        for v in [1, 2, 3, 4] {
            assert_eq!(list.get(&v), list.get_fast(&v));
        }
    }

    // MARK: get_first

    #[test]
    fn get_first_value_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(list.get_first(&1), None);
    }

    #[test]
    fn get_first_value_not_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(3);
        assert_eq!(list.get_first(&2), None);
    }

    #[test]
    fn get_first_value_no_duplicates() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get_first(&2), Some(&2));
    }

    #[test]
    fn get_first_value_returns_first_occurrence() {
        // Use a multi-level list so skip links actually span duplicates.
        let mut list = OrderedSkipList::<i32>::new();
        for _ in 0..50 {
            list.insert(2);
        }
        list.insert(1);
        list.insert(3);
        // rank(&2) is 1 (the first 2 is at index 1 after the leading 1).
        let found = list.get_first(&2);
        assert_eq!(found, Some(&2));
        // Verify it is genuinely the first: rank of that pointer == rank(&2).
        let rank_first = list.rank(&2).expect("present");
        assert_eq!(list.get_by_index(rank_first), found);
    }

    #[test]
    fn get_first_value_custom_comparator() {
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(3);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.get_first(&2), Some(&2));
        assert_eq!(list.get_first(&4), None);
    }

    // MARK: get_last

    #[test]
    fn get_last_value_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(list.get_last(&1), None);
    }

    #[test]
    fn get_last_value_not_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(3);
        assert_eq!(list.get_last(&2), None);
    }

    #[test]
    fn get_last_value_no_duplicates() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get_last(&2), Some(&2));
    }

    #[test]
    fn get_last_value_returns_last_occurrence() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        // count(&2) == 3; last occurrence is at rank count + rank_first - 1 == 3.
        let last_rank = list.rank(&2).expect("present") + list.count(&2) - 1;
        let found = list.get_last(&2);
        assert_eq!(found, Some(&2));
        assert_eq!(list.get_by_index(last_rank), found);
    }

    #[test]
    fn get_last_value_all_equal() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(5);
        list.insert(5);
        list.insert(5);
        let found = list.get_last(&5);
        assert_eq!(found, Some(&5));
        assert_eq!(list.get_by_index(list.len() - 1), found);
    }

    #[test]
    fn get_last_value_custom_comparator() {
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(3);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.get_last(&2), Some(&2));
        assert_eq!(list.get_last(&4), None);
    }

    // MARK: get_by_index

    #[test]
    fn get_by_index_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(list.get_by_index(0), None);
    }

    #[test]
    fn get_by_index_in_bounds() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [3, 1, 2] {
            list.insert(i);
        }
        assert_eq!(list.get_by_index(0), Some(&1));
        assert_eq!(list.get_by_index(1), Some(&2));
        assert_eq!(list.get_by_index(2), Some(&3));
    }

    #[test]
    fn get_by_index_out_of_bounds() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        assert_eq!(list.get_by_index(1), None);
        assert_eq!(list.get_by_index(99), None);
    }

    #[test]
    fn get_by_index_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.get_by_index(0), Some(&42));
        assert_eq!(list.get_by_index(1), None);
    }

    #[test]
    fn get_by_index_large_list() {
        let mut list = OrderedSkipList::<usize>::new();
        for i in 0..100_usize {
            list.insert(i);
        }
        for i in 0..100_usize {
            assert_eq!(list.get_by_index(i), Some(&i));
        }
        assert_eq!(list.get_by_index(100), None);
    }

    #[test]
    fn get_by_index_with_duplicates() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.get_by_index(0), Some(&1));
        assert_eq!(list.get_by_index(1), Some(&2));
        assert_eq!(list.get_by_index(2), Some(&2));
        assert_eq!(list.get_by_index(3), Some(&3));
    }

    #[test]
    fn get_by_index_after_removals() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        list.take_first(&3);
        assert_eq!(list.get_by_index(0), Some(&1));
        assert_eq!(list.get_by_index(1), Some(&2));
        assert_eq!(list.get_by_index(2), Some(&4));
        assert_eq!(list.get_by_index(3), Some(&5));
        assert_eq!(list.get_by_index(4), None);
    }

    #[test]
    fn get_by_index_custom_comparator() {
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(1);
        list.insert(2);
        list.insert(3);
        // Stored as [3, 2, 1] with reverse ordering.
        assert_eq!(list.get_by_index(0), Some(&3));
        assert_eq!(list.get_by_index(1), Some(&2));
        assert_eq!(list.get_by_index(2), Some(&1));
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

    // MARK: Index

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing valid indexing behavior with known in-bounds indices"
    )]
    fn index_in_bounds() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [3, 1, 2] {
            list.insert(i);
        }
        assert_eq!(list[0], 1);
        assert_eq!(list[1], 2);
        assert_eq!(list[2], 3);
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing out-of-bounds indexing panics with a known out-of-bounds index"
    )]
    #[should_panic(expected = "index out of bounds")]
    fn index_out_of_bounds() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        _ = list[1];
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing out-of-bounds indexing panics on an empty list"
    )]
    #[should_panic(expected = "index out of bounds")]
    fn index_empty_list_panics() {
        let list = OrderedSkipList::<i32>::new();
        _ = list[0];
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing valid indexing behavior with a known in-bounds index"
    )]
    fn index_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list[0], 42);
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing valid indexing behavior with known in-bounds indices under a custom comparator"
    )]
    fn index_custom_comparator() {
        // Reverse ordering: stored [3, 2, 1].
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list[0], 3);
        assert_eq!(list[1], 2);
        assert_eq!(list[2], 1);
    }

    // MARK: Borrow<Q> lookups (String / &str)

    #[test]
    fn contains_str_on_string_element() {
        let mut list = OrderedSkipList::<String>::new();
        list.insert("apple".to_owned());
        list.insert("banana".to_owned());
        assert!(list.contains("apple"));
        assert!(list.contains("banana"));
        assert!(!list.contains("cherry"));
    }

    #[test]
    fn get_fast_str_on_string_element() {
        let mut list = OrderedSkipList::<String>::new();
        list.insert("apple".to_owned());
        list.insert("banana".to_owned());
        assert_eq!(list.get_fast("apple"), Some(&"apple".to_owned()));
        assert_eq!(list.get_fast("cherry"), None);
    }

    #[test]
    fn rank_str_on_string_element() {
        let mut list = OrderedSkipList::<String>::new();
        list.insert("apple".to_owned());
        list.insert("banana".to_owned());
        list.insert("cherry".to_owned());
        // Sorted: apple(0), banana(1), cherry(2)
        assert_eq!(list.rank("apple"), Some(0));
        assert_eq!(list.rank("banana"), Some(1));
        assert_eq!(list.rank("cherry"), Some(2));
        assert_eq!(list.rank("date"), None);
    }

    #[test]
    fn count_str_on_string_element() {
        let mut list = OrderedSkipList::<String>::new();
        list.insert("apple".to_owned());
        list.insert("apple".to_owned());
        list.insert("banana".to_owned());
        assert_eq!(list.count("apple"), 2);
        assert_eq!(list.count("banana"), 1);
        assert_eq!(list.count("cherry"), 0);
    }
}
