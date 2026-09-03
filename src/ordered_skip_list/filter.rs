//! Predicate-based filtering methods for [`OrderedSkipList`](super::OrderedSkipList):
//! `retain`, `dedup`, `dedup_by`, `dedup_by_key`.

use crate::{
    comparator::Comparator, level_generator::LevelGenerator, node::Node,
    ordered_skip_list::OrderedSkipList,
};

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> OrderedSkipList<T, N, C, G> {
    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, removes all elements `e` for which `f(&e)` returns
    /// `false`. The elements are visited in ascending order and, in the
    /// kept subset, their relative (sorted) order is preserved.
    ///
    /// Note: `retain_mut` is not provided because mutating elements in-place
    /// could break the ordering invariant.
    ///
    /// This operation is `$O(n)$`: every element is visited once.
    ///
    /// If `f` or an element's destructor panics, the elements the walk had
    /// not yet reached are kept in the list, as is the element `f` panicked
    /// on; the list stays usable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.insert(i);
    /// }
    /// list.retain(|&x| x % 2 == 0);
    /// let collected: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(collected, [2, 4]);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "`value()` returns None only for the head sentinel, which is never \
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
        F: FnMut(&T) -> bool,
    {
        if self.is_empty() {
            return;
        }

        // SAFETY: All raw pointers originate from heap allocations owned by
        // this OrderedSkipList.  We hold &mut self so no other reference to
        // any node exists.  The closure reads the value and returns before any
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
                    // SAFETY: cur is a live, heap-allocated data node.
                    f((*cur).value().expect("data node has a value"))
                },
                |_| {},
            );
        }
    }

    /// Removes all but the first of consecutive equal elements as determined
    /// by a predicate.
    ///
    /// The `same_bucket` predicate receives mutable references to two
    /// consecutive elements, `(later, earlier)`, and returns `true` if
    /// `later` should be removed. The `later` element may be mutated before
    /// the predicate returns; it is dropped when the predicate returns `true`.
    ///
    /// This matches the semantics of [`Vec::dedup_by`].
    ///
    /// Because `OrderedSkipList` keeps elements in sorted order, all
    /// occurrences of equal values are always adjacent, so this method
    /// effectively removes **all** duplicates when used with an equality
    /// predicate (see [`dedup`](OrderedSkipList::dedup)).
    ///
    /// # Note on `b` mutation
    ///
    /// Mutating the *retained* element `b` in a way that changes its sorted
    /// position leaves the list in an inconsistent state. Only mutate `a`
    /// (the element being dropped) or mutate `b` in a way that preserves its
    /// relative order.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// If `same_bucket` or an element's destructor panics, the elements the
    /// walk had not yet reached are kept in the list, as is the element
    /// `same_bucket` panicked on; the list stays usable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<String>::new();
    /// for s in ["bar", "Bar", "foo"] {
    ///     list.insert(s.to_string());
    /// }
    /// list.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    /// let collected: Vec<String> = list.iter().cloned().collect();
    /// assert_eq!(collected, ["Bar", "foo"]);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "`value_mut()` returns None only for the head sentinel, which is never \
                  visited in the data-node walk; the expect fires only on invariant violations"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "calling filter_rebuild (unsafe fn) and dereferencing cur and prev_ptr inside \
                  the keep closure are provably disjoint heap operations"
    )]
    #[inline]
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        if self.len() <= 1 {
            return;
        }

        // Pointer to the last kept node, used for comparison inside the closure.
        let mut prev_kept: Option<*mut Node<T, N>> = None;

        // SAFETY: All raw pointers originate from heap allocations owned by
        // this OrderedSkipList.  We hold &mut self so no other reference to
        // any node exists.  The keep closure calls value_mut() on two distinct
        // raw node pointers, producing non-aliasing exclusive references
        // because each node is a separately heap-allocated Box.  The closure
        // returns before any structural mutation occurs.
        // `filter_rebuild` publishes the new length and tail itself, so
        // an unwind out of the predicate or out of an element's destructor
        // cannot skip the update.
        unsafe {
            Node::filter_rebuild(
                self.head,
                &mut self.len,
                &mut self.tail,
                |cur| {
                    let keep = match prev_kept {
                        None => true,
                        Some(prev_ptr) => {
                            // SAFETY: cur and prev_ptr point to distinct, live,
                            // heap-allocated Nodes; their value_mut references do not alias.
                            let a: &mut T = (*cur).value_mut().expect("data node has a value");
                            let b: &mut T = (*prev_ptr).value_mut().expect("data node has a value");
                            !same_bucket(a, b)
                        }
                    };
                    if keep {
                        prev_kept = Some(cur);
                    }
                    keep
                },
                |_| {},
            );
        }
    }

    /// Removes all but the first of consecutive equal elements.
    ///
    /// Equivalent to [`dedup_by`](OrderedSkipList::dedup_by) with a predicate
    /// that compares elements using [`PartialEq`].
    ///
    /// Because `OrderedSkipList` keeps elements in sorted order, all equal
    /// elements are always adjacent, so this effectively removes **all**
    /// duplicate values from the list.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// Panic behaviour matches [`dedup_by`](OrderedSkipList::dedup_by).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for i in [1, 1, 2, 3, 3] {
    ///     list.insert(i);
    /// }
    /// list.dedup();
    /// let collected: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(collected, [1, 2, 3]);
    /// ```
    #[inline]
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.dedup_by(|a, b| a == b);
    }

    /// Removes all but the first of consecutive elements for which a key
    /// function returns equal values.
    ///
    /// Equivalent to [`dedup_by`](OrderedSkipList::dedup_by) with a predicate
    /// that compares the keys derived from each element.
    ///
    /// Because `OrderedSkipList` keeps elements in sorted order, elements with
    /// the same key are always adjacent, so this effectively removes **all**
    /// elements with a duplicate key.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// Panic behaviour matches [`dedup_by`](OrderedSkipList::dedup_by).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for i in [10, 20, 21, 30, 31, 32] {
    ///     list.insert(i);
    /// }
    /// list.dedup_by_key(|i| *i / 10);
    /// let collected: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(collected, [10, 20, 30]);
    /// ```
    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        // Cache the key of the last retained element so `key` is called on it
        // exactly once, even if `key` mutates its argument.  This matches the
        // semantics of [`Vec::dedup_by_key`].
        let mut last_key: Option<K> = None;
        self.dedup_by(|a, b| {
            // `b` is the retained element; compute its key only on first use.
            let kb = last_key.get_or_insert_with(|| key(b));
            let ka = key(a);
            if ka == *kb {
                true // drop `a`
            } else {
                last_key = Some(ka); // `a` will be retained; cache its key
                false // keep `a`
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use pretty_assertions::assert_eq;

    use super::super::OrderedSkipList;
    use crate::test_util::{PanicOnDrop, bombs};

    // MARK: retain

    #[test]
    fn retain_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        list.retain(|_| true);
        assert!(list.is_empty());
        list.retain(|_| false);
        assert!(list.is_empty());
    }

    #[test]
    fn retain_all_kept() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        list.retain(|_| true);
        assert_eq!(list.len(), 5);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn retain_all_dropped() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        list.retain(|_| false);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.first(), None);
        assert_eq!(list.last(), None);
    }

    #[test]
    fn retain_even_elements() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=6 {
            list.insert(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping even numbers"
        )]
        list.retain(|&x| x % 2 == 0);
        assert_eq!(list.len(), 3);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [2, 4, 6]);
    }

    #[test]
    fn retain_preserves_sorted_order() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 0..10 {
            list.insert(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping multiples of 3"
        )]
        list.retain(|&x| x % 3 == 0);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [0, 3, 6, 9]);
    }

    #[test]
    fn retain_links_consistent() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 0..20 {
            list.insert(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping even numbers"
        )]
        list.retain(|&x| x % 2 == 0);
        // Verify via iter() to exercise skip links.
        let got: Vec<i32> = list.iter().copied().collect();
        let expected: Vec<i32> = (0..20).step_by(2).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn retain_single_element_kept() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        list.retain(|_| true);
        assert_eq!(list.len(), 1);
        assert_eq!(list.first(), Some(&42));
        assert_eq!(list.last(), Some(&42));
    }

    #[test]
    fn retain_single_element_dropped() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        list.retain(|_| false);
        assert!(list.is_empty());
        assert_eq!(list.first(), None);
        assert_eq!(list.last(), None);
    }

    #[test]
    fn retain_tail_pointer_correct() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        list.retain(|&x| x <= 3);
        assert_eq!(list.last(), Some(&3));
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn retain_after_retain_is_correct() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 0..10 {
            list.insert(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping even numbers"
        )]
        list.retain(|&x| x % 2 == 0);
        list.retain(|&x| x > 2);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [4, 6, 8]);
    }

    #[test]
    fn retain_with_duplicates_in_ordered_list() {
        let mut list = OrderedSkipList::<i32>::new();
        for _ in 0..3 {
            list.insert(1);
            list.insert(2);
            list.insert(3);
        }
        list.retain(|&x| x > 1);
        assert_eq!(list.len(), 6);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [2, 2, 2, 3, 3, 3]);
    }

    // MARK: dedup

    #[test]
    fn dedup_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        list.dedup();
        assert!(list.is_empty());
    }

    #[test]
    fn dedup_single() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        list.dedup();
        assert_eq!(list.len(), 1);
        assert_eq!(list.first(), Some(&42));
        assert_eq!(list.last(), Some(&42));
    }

    #[test]
    fn dedup_all_same() {
        let mut list = OrderedSkipList::<i32>::new();
        for _ in 0..4 {
            list.insert(1);
        }
        list.dedup();
        assert_eq!(list.len(), 1);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1]);
    }

    #[test]
    fn dedup_no_duplicates() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3, 4] {
            list.insert(i);
        }
        list.dedup();
        assert_eq!(list.len(), 4);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 3, 4]);
    }

    #[test]
    fn dedup_removes_all_duplicates() {
        // In a sorted list, dedup removes all duplicates, not just adjacent pairs.
        let mut list = OrderedSkipList::<i32>::new();
        for i in [1, 1, 2, 2, 3, 3] {
            list.insert(i);
        }
        list.dedup();
        assert_eq!(list.len(), 3);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 3]);
    }

    #[test]
    fn dedup_multiple_groups() {
        let mut list = OrderedSkipList::<i32>::new();
        for _ in 0..3 {
            list.insert(1);
            list.insert(2);
            list.insert(3);
        }
        list.dedup();
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 3]);
    }

    #[test]
    fn dedup_links_consistent() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [1, 1, 2, 2, 3, 3, 4, 4] {
            list.insert(i);
        }
        list.dedup();
        let expected = [1, 2, 3, 4];
        assert_eq!(list.len(), expected.len());
        let via_iter: Vec<i32> = list.iter().copied().collect();
        assert_eq!(via_iter, expected);
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(list.get_by_index(i), Some(&v));
        }
        assert_eq!(list.get_by_index(expected.len()), None);
    }

    #[test]
    fn dedup_tail_pointer_correct() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3, 3, 3] {
            list.insert(i);
        }
        list.dedup();
        assert_eq!(list.last(), Some(&3));
        assert_eq!(list.len(), 3);
    }

    // MARK: dedup_by

    #[test]
    fn dedup_by_case_insensitive() {
        let mut list = OrderedSkipList::<String>::new();
        for s in ["bar", "Bar", "foo"] {
            list.insert(s.to_owned());
        }
        // "Bar" < "bar" in ASCII order, so the list is ["Bar", "bar", "foo"].
        list.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
        let got: Vec<String> = list.into_iter().collect();
        assert_eq!(got, ["Bar", "foo"]);
    }

    #[test]
    fn dedup_by_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        list.dedup_by(|a, b| a == b);
        assert!(list.is_empty());
    }

    // MARK: dedup_by_key

    #[test]
    fn dedup_by_key_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        #[expect(
            clippy::integer_division,
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of deduplicating by the tens digit"
        )]
        list.dedup_by_key(|i| *i / 10);
        assert!(list.is_empty());
    }

    #[test]
    fn dedup_by_key_mutating_key_called_once_per_retained_element() {
        // Key fn mutates: replaces the element with 0 and returns the original
        // value.  If `key` were called more than once on the retained element,
        // the second call would return 0 and later duplicates would be
        // incorrectly kept.
        let mut list = OrderedSkipList::<i32>::new();
        for _ in 0..3 {
            list.insert(5);
        }
        list.dedup_by_key(|x| {
            let v = *x;
            *x = 0;
            v
        });
        assert_eq!(list.len(), 1);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [0]);
    }

    #[test]
    fn dedup_by_key_floored_decade() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [10, 20, 21, 30, 31, 32] {
            list.insert(i);
        }
        #[expect(
            clippy::integer_division,
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of deduplicating by the tens digit"
        )]
        list.dedup_by_key(|i| *i / 10);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [10, 20, 30]);
    }

    // MARK: panic safety

    fn assert_survivors(list: &OrderedSkipList<PanicOnDrop>, expected: &[u32]) {
        let ids: Vec<u32> = list.iter().map(PanicOnDrop::id).collect();
        assert_eq!(ids, expected);
        assert_eq!(list.len(), expected.len());
        assert_eq!(list.first().map(PanicOnDrop::id), expected.first().copied());
        assert_eq!(list.last().map(PanicOnDrop::id), expected.last().copied());
        // Indexed lookup goes through the rebuilt skip links, not the `next`
        // chain the assertions above walk.
        for (index, id) in expected.iter().enumerate() {
            assert_eq!(list.get_by_index(index).map(PanicOnDrop::id), Some(*id));
        }
        assert!(list.get_by_index(expected.len()).is_none());
    }

    fn armed_list() -> OrderedSkipList<PanicOnDrop> {
        let mut list = OrderedSkipList::<PanicOnDrop>::new();
        for bomb in bombs(16, 7) {
            list.insert(bomb);
        }
        list
    }

    #[test]
    fn retain_panicking_drop_keeps_unvisited_elements() {
        let mut list = armed_list();

        let result = catch_unwind(AssertUnwindSafe(|| list.retain(|_| false)));
        assert!(result.is_err());

        assert_survivors(&list, &[8, 9, 10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn dedup_by_panicking_drop_keeps_unvisited_elements() {
        let mut list = armed_list();

        let result = catch_unwind(AssertUnwindSafe(|| list.dedup_by(|_, _| true)));
        assert!(result.is_err());

        assert_survivors(&list, &[0, 8, 9, 10, 11, 12, 13, 14, 15]);
    }

    /// A list whose elements all destruct cleanly, so the only unwind can come
    /// from the predicate.
    fn unarmed_list() -> OrderedSkipList<PanicOnDrop> {
        let mut list = OrderedSkipList::<PanicOnDrop>::new();
        for bomb in bombs(16, 16) {
            list.insert(bomb);
        }
        list
    }

    #[test]
    fn retain_panicking_predicate_keeps_unvisited_elements() {
        let mut list = unarmed_list();

        let result = catch_unwind(AssertUnwindSafe(|| {
            list.retain(|element| {
                assert!(element.id() != 7, "predicate panic");
                false
            });
        }));
        assert!(result.is_err());

        assert_survivors(&list, &[7, 8, 9, 10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn dedup_by_panicking_predicate_keeps_unvisited_elements() {
        let mut list = unarmed_list();

        let result = catch_unwind(AssertUnwindSafe(|| {
            list.dedup_by(|later, _earlier| {
                assert!(later.id() != 7, "predicate panic");
                true
            });
        }));
        assert!(result.is_err());

        assert_survivors(&list, &[0, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    }
}
