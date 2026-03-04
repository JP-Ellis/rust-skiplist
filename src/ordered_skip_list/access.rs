//! Value-based read access for [`OrderedSkipList`](super::OrderedSkipList).

use crate::{
    comparator::Comparator,
    level_generator::LevelGenerator,
    node::visitor::{OrdVisitor, Visitor},
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
