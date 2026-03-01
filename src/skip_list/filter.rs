//! Predicate-based and deduplication methods for [`SkipList`](super::SkipList):
//! `retain`, `retain_mut`, `dedup`, `dedup_by`, and `dedup_by_key`.

use crate::{level_generator::LevelGenerator, node::Node, skip_list::SkipList};

impl<T, G: LevelGenerator> SkipList<T, G> {
    /// Removes all but the first of consecutive equal elements as determined
    /// by a comparator.
    ///
    /// The `same_bucket` predicate receives mutable references to two
    /// consecutive elements as `(later, earlier)` and returns `true` if
    /// `later` should be removed.  The `later` element may be mutated before
    /// the predicate returns; it is dropped when the predicate returns `true`.
    ///
    /// This matches the semantics of [`Vec::dedup_by`].
    ///
    /// This operation is `$O(n)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<String>::new();
    /// for s in ["foo", "bar", "Bar", "baz", "bar"] {
    ///     list.push_back(s.to_string());
    /// }
    /// list.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    /// let collected: Vec<String> = list.into_iter().collect();
    /// assert_eq!(collected, ["foo", "bar", "baz", "bar"]);
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
        let mut prev_kept: Option<*mut Node<T>> = None;

        // SAFETY: All raw pointers originate from heap allocations owned by
        // this SkipList.  We hold &mut self so no other reference to any node
        // exists.  The keep closure calls value_mut() on two distinct raw node
        // pointers, producing non-aliasing exclusive references because each
        // node is a separately heap-allocated Box.  The closure returns before
        // any structural mutation occurs.
        let (new_rank, new_tail) = unsafe {
            self.head.filter_rebuild(
                |cur| {
                    let keep = match prev_kept {
                        None => true,
                        Some(prev_ptr) => {
                            // SAFETY: cur and prev_ptr point to distinct, live,
                            // heap-allocated Nodes; their value_mut references do not alias.
                            let a: &mut T =
                                (*cur).value_mut().expect("data node has a value");
                            let b: &mut T =
                                (*prev_ptr).value_mut().expect("data node has a value");
                            !same_bucket(a, b)
                        }
                    };
                    if keep {
                        prev_kept = Some(cur);
                    }
                    keep
                },
                |_| {},
            )
        };
        self.tail = new_tail;
        self.len = new_rank;
    }

    /// Removes all but the first of consecutive equal elements.
    ///
    /// Equivalent to [`dedup_by`](SkipList::dedup_by) with a predicate that
    /// compares elements using [`PartialEq`].
    ///
    /// This operation is `$O(n)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// for i in [1, 1, 2, 3, 3, 2] {
    ///     list.push_back(i);
    /// }
    /// list.dedup();
    /// let collected: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(collected, [1, 2, 3, 2]);
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
    /// Equivalent to [`dedup_by`](SkipList::dedup_by) with a predicate that
    /// compares the keys derived from each element.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// for i in [10, 20, 21, 30, 31, 32] {
    ///     list.push_back(i);
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
        self.dedup_by(|a, b| key(a) == key(b));
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, removes all elements `e` for which `f(&e)` returns
    /// `false`.  The elements are visited in order and, in the kept subset,
    /// their relative order is preserved.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.push_back(i);
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
        // this SkipList.  We hold &mut self so no other reference to any node
        // exists.  The closure reads the value and returns before any
        // structural mutation occurs.
        let (new_rank, new_tail) = unsafe {
            self.head.filter_rebuild(
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

    /// Retains only the elements specified by the predicate, passing a mutable
    /// reference to each element.
    ///
    /// In other words, removes all elements `e` for which `f(&mut e)` returns
    /// `false`.  The elements are visited in order; the predicate may mutate
    /// retained elements before it returns `true`.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.push_back(i);
    /// }
    /// list.retain_mut(|x| {
    ///     if *x % 2 == 0 {
    ///         *x *= 10;
    ///         true
    ///     } else {
    ///         false
    ///     }
    /// });
    /// let collected: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(collected, [20, 40]);
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
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        if self.is_empty() {
            return;
        }

        // SAFETY: All raw pointers originate from heap allocations owned by
        // this SkipList.  We hold &mut self so no other reference to any node
        // exists.  The &mut T borrow created by `value_mut()` expires before
        // `filter_rebuild` performs any structural mutation on the node.
        let (new_rank, new_tail) = unsafe {
            self.head.filter_rebuild(
                |cur| {
                    // SAFETY: cur is a live, heap-allocated data node.
                    f((*cur).value_mut().expect("data node has a value"))
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

    use super::super::SkipList;

    // MARK: retain

    #[test]
    fn retain_empty() {
        let mut list = SkipList::<i32>::new();
        list.retain(|_| true);
        assert!(list.is_empty());
        list.retain(|_| false);
        assert!(list.is_empty());
    }

    #[test]
    fn retain_all_kept() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        list.retain(|_| true);
        assert_eq!(list.len(), 5);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn retain_all_dropped() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        list.retain(|_| false);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn retain_even_elements() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=6 {
            list.push_back(i);
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
    fn retain_preserves_order() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10 {
            list.push_back(i);
        }
        // Keep 0, 3, 6, 9
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
        let mut list = SkipList::<i32>::new();
        for i in 0..20 {
            list.push_back(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping even numbers"
        )]
        list.retain(|&x| x % 2 == 0);
        for (idx, expected) in (0..20).step_by(2).enumerate() {
            assert_eq!(list.get(idx), Some(&expected));
        }
        assert_eq!(list.get(list.len()), None);
    }

    #[test]
    fn retain_single_element_kept() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        list.retain(|_| true);
        assert_eq!(list.len(), 1);
        assert_eq!(list.front(), Some(&42));
        assert_eq!(list.back(), Some(&42));
        assert_eq!(list.get(0), Some(&42));
    }

    #[test]
    fn retain_single_element_dropped() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        list.retain(|_| false);
        assert!(list.is_empty());
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn retain_tail_pointer_correct() {
        // Verify that back() returns the last retained element.
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        // Keep 1, 2, 3; drop 4 and 5
        list.retain(|&x| x <= 3);
        assert_eq!(list.back(), Some(&3));
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn retain_after_retain_is_correct() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10 {
            list.push_back(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping even numbers"
        )]
        list.retain(|&x| x % 2 == 0); // 0,2,4,6,8
        list.retain(|&x| x > 2); // 4,6,8
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [4, 6, 8]);
    }

    // MARK: retain_mut

    #[test]
    fn retain_mut_can_modify_before_keeping() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of modifying even numbers in place and keeping them"
        )]
        list.retain_mut(|x| {
            if *x % 2 == 0 {
                *x *= 10;
                true
            } else {
                false
            }
        });
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [20, 40]);
    }

    #[test]
    fn retain_mut_drops_correctly() {
        let mut list = SkipList::<i32>::new();
        for i in 0..5 {
            list.push_back(i);
        }
        list.retain_mut(|_| false);
        assert!(list.is_empty());
    }

    #[test]
    fn retain_mut_links_consistent() {
        let mut list = SkipList::<i32>::new();
        for i in 0..20 {
            list.push_back(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping multiples of 3 \
                and modifying them in place"
        )]
        list.retain_mut(|x| {
            if *x % 3 == 0 {
                *x += 100;
                true
            } else {
                false
            }
        });
        // Retained: 0+100=100, 3+100=103, 6+100=106, 9+100=109, 12+100=112, 15+100=115, 18+100=118
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [100, 103, 106, 109, 112, 115, 118]);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(list.get(i), Some(v));
        }
    }

    // MARK: dedup

    #[test]
    fn dedup_empty() {
        let mut list = SkipList::<i32>::new();
        list.dedup();
        assert!(list.is_empty());
    }

    #[test]
    fn dedup_single() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        list.dedup();
        assert_eq!(list.len(), 1);
        assert_eq!(list.front(), Some(&42));
        assert_eq!(list.back(), Some(&42));
    }

    #[test]
    fn dedup_all_same() {
        let mut list = SkipList::<i32>::new();
        for _ in 0..4 {
            list.push_back(1);
        }
        list.dedup();
        assert_eq!(list.len(), 1);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1]);
    }

    #[test]
    fn dedup_no_duplicates() {
        let mut list = SkipList::<i32>::new();
        for i in [1, 2, 3, 4] {
            list.push_back(i);
        }
        list.dedup();
        assert_eq!(list.len(), 4);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 3, 4]);
    }

    #[test]
    fn dedup_adjacent_pairs() {
        let mut list = SkipList::<i32>::new();
        for i in [1, 1, 2, 2, 3, 3] {
            list.push_back(i);
        }
        list.dedup();
        assert_eq!(list.len(), 3);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 3]);
    }

    #[test]
    fn dedup_non_adjacent() {
        let mut list = SkipList::<i32>::new();
        for i in [1, 2, 1, 2] {
            list.push_back(i);
        }
        list.dedup();
        assert_eq!(list.len(), 4);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 1, 2]);
    }

    #[test]
    fn dedup_leading_duplicates() {
        let mut list = SkipList::<i32>::new();
        for i in [1, 1, 2, 3] {
            list.push_back(i);
        }
        list.dedup();
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 3]);
    }

    #[test]
    fn dedup_trailing_duplicates() {
        let mut list = SkipList::<i32>::new();
        for i in [1, 2, 3, 3] {
            list.push_back(i);
        }
        list.dedup();
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 3]);
    }

    #[test]
    fn dedup_links_consistent() {
        let mut list = SkipList::<i32>::new();
        for i in [1, 1, 2, 2, 3, 3, 4, 4] {
            list.push_back(i);
        }
        list.dedup();
        let expected = [1, 2, 3, 4];
        assert_eq!(list.len(), expected.len());
        // Verify via iter() and get() to exercise both prev/next and skip links.
        let via_iter: Vec<i32> = list.iter().copied().collect();
        assert_eq!(via_iter, expected);
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(list.get(i), Some(&v));
        }
        assert_eq!(list.get(expected.len()), None);
    }

    #[test]
    fn dedup_by_case_insensitive() {
        let mut list = SkipList::<String>::new();
        for s in ["foo", "bar", "Bar", "baz", "bar"] {
            list.push_back(s.to_owned());
        }
        list.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
        let got: Vec<String> = list.into_iter().collect();
        assert_eq!(got, ["foo", "bar", "baz", "bar"]);
    }

    #[test]
    fn dedup_by_key_empty() {
        let mut list = SkipList::<i32>::new();
        #[expect(
            clippy::integer_division,
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of deduplicating by the tens digit"
        )]
        list.dedup_by_key(|i| *i / 10);
        assert!(list.is_empty());
    }

    #[test]
    fn dedup_by_key_floored_half() {
        let mut list = SkipList::<i32>::new();
        for i in [10, 20, 21, 30, 31, 32] {
            list.push_back(i);
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
}
