//! Predicate-based filtering methods for [`OrderedSkipList`](super::OrderedSkipList):
//! `retain`.

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
        let (new_rank, new_tail) = unsafe {
            Node::filter_rebuild(
                self.head,
                |cur| {
                    // SAFETY: cur is a live, heap-allocated data node.
                    f((*cur).value().expect("data node has a value"))
                },
                |_| {},
            )
        };
        self.tail = new_tail;
        self.len = new_rank;
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::OrderedSkipList;

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
    fn retain_with_duplicates() {
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
}
