//! Positional write methods for [`SkipList`](super::SkipList):
//! `insert`, `remove`, and `swap`.

use core::ptr::NonNull;

use crate::{
    level_generator::LevelGenerator,
    node::{
        Node,
        link::Link,
        visitor::{IndexMutVisitor, Visitor},
    },
    skip_list::SkipList,
};

impl<T, G: LevelGenerator, const N: usize> SkipList<T, N, G> {
    /// Inserts `value` at position `index`, shifting all elements at `index..`
    /// one position to the right.
    ///
    /// This operation is `$O(\log n)$` expected.
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(1);
    /// list.push_back(3);
    /// list.insert(1, 2);
    /// assert_eq!(list.len(), 3);
    /// // list is now [1, 2, 3]
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "Link::new distances are computed to be ≥ 1; \
                  increment_distance overflow requires > usize::MAX nodes; \
                  all expects fire only on internal invariant violations, not user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < height ≤ max_levels = new_raw.links.len(); \
                  pred_ptr was reached at level l during traversal so pred_ptr.links.len() > l; \
                  all accesses are bounded by max_levels = head.links.len()"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "insertion and link wiring touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        assert!(
            index <= self.len,
            "insertion index (is {index}) should be <= len (is {})",
            self.len
        );

        // height ∈ [0, total]: number of skip links to allocate.
        let height = self.generator.level();
        // new_rank: the rank of the new node after insertion
        // (head = rank 0; element at index i has rank i + 1).
        let new_rank = index.saturating_add(1);

        // IndexMutVisitor records, for each level l, the last node whose
        // level-l skip-link would overshoot or is absent at new_rank.
        // into_parts() releases the &mut borrow so the subsequent unsafe block
        // can take raw pointers to nodes.
        let (current, precursors, precursor_distances) = {
            let mut visitor = IndexMutVisitor::new(self.head, new_rank);
            visitor.traverse();
            visitor.into_parts()
        };

        // SAFETY: All raw pointers come from NonNull<Node<T, N>> captured during
        // traversal.  They originate from heap allocations owned by this SkipList.
        // No safe &mut references to any node exist while this block runs.
        // Different precursors may alias (same predecessor, different levels) but
        // each iteration accesses a distinct links[l] index.
        let new_node_nonnull: NonNull<Node<T, N>> = unsafe {
            // When inserting in the middle (index < len), `current` is the node
            // currently at rank `new_rank`; the new node must go before it, so
            // insert after its immediate base-chain predecessor.
            // When inserting at the end (index == len), `current` is the tail;
            // insert after it directly.
            let insert_after_ptr = if index < self.len {
                (*current.as_ptr())
                    .prev()
                    .expect("node at rank >= 1 always has a predecessor")
            } else {
                current
            };
            let new_raw: *mut Node<T, N> =
                Node::insert_after(insert_after_ptr, Node::with_value(height, value)).as_ptr();

            // Wire skip links.
            //
            // For l < height (new node's tower reaches this level):
            //   Before: pred (rank D) ---[d]---> X (rank D + d)
            //   After:  pred (rank D) ---[new_rank − D]---> new_node (rank new_rank)
            //           new_node      ---[D + d + 1 − new_rank]---> X (rank D + d + 1)
            //
            // For l ≥ height (new node has no tower here):
            //   pred.links[l] still points to the same X (now at rank D + d + 1);
            //   increment its distance by 1.
            for (l, (pred_nn, pred_rank)) in precursors
                .iter()
                .copied()
                .zip(precursor_distances.iter().copied())
                .enumerate()
            {
                let pred_ptr = pred_nn.as_ptr();
                if l < height {
                    let distance = new_rank.saturating_sub(pred_rank);
                    let old_link = (*pred_ptr).links_mut()[l].take();
                    (*pred_ptr).links_mut()[l] = Some(
                        Link::new(NonNull::new_unchecked(new_raw), distance)
                            .expect("distance >= 1"),
                    );
                    (*new_raw).links_mut()[l] = if let Some(old) = old_link {
                        let new_d = old
                            .distance()
                            .get()
                            .saturating_sub(distance)
                            .saturating_add(1);
                        Some(Link::new(old.node(), new_d).expect("new_d >= 1"))
                    } else {
                        None
                    };
                } else if let Some(link) = (*pred_ptr).links_mut()[l].as_mut() {
                    link.increment_distance()
                        .expect("distance overflow requires > usize::MAX nodes");
                }
            }

            // SAFETY: new_raw comes from insert_after (Box::into_raw), so it
            // is non-null.  Return it so self.tail can be updated outside.
            NonNull::new_unchecked(new_raw)
        };

        if index == self.len {
            self.tail = Some(new_node_nonnull);
        }
        self.len = self.len.saturating_add(1);
    }

    /// Removes and returns the element at position `index`.
    ///
    /// All elements after `index` shift one position to the left.
    ///
    /// This operation is `$O(\log n)$` expected.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(1);
    /// list.push_back(2);
    /// list.push_back(3);
    /// assert_eq!(list.remove(1), 2);
    /// assert_eq!(list.len(), 2);
    /// // list is now [1, 3]
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "precursors[0].links[0] exists because index < len guarantees a node at target_rank; \
                  take_value is Some for any body/tail node; \
                  Link::new distance is computed to be >= 1; \
                  all expects fire only on internal invariant violations, not user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < target_height <= max_levels = target.links.len(); \
                  precursors[l] was reached at level l so precursors[l].links.len() > l; \
                  all accesses are bounded by max_levels = head.links.len()"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link rewiring and node pop touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        assert!(
            index < self.len,
            "removal index (is {index}) should be < len (is {})",
            self.len
        );

        // head = rank 0; the element at logical index i is at rank i + 1.
        let target_rank = index.saturating_add(1);

        // IndexMutVisitor records the predecessor at each level.
        // `current` from into_parts() is the target node itself.
        // into_parts() releases the &mut borrow so the unsafe block can use raw ptrs.
        let (target_node, precursors, precursor_distances) = {
            let mut visitor = IndexMutVisitor::new(self.head, target_rank);
            visitor.traverse();
            visitor.into_parts()
        };

        // SAFETY: All raw pointers come from NonNull<Node<T, N>> captured during
        // traversal.  They originate from heap allocations owned by this SkipList.
        // No safe &mut references to any node exist while this block runs.
        let (value, new_tail) = unsafe {
            let target_ptr: *mut Node<T, N> = target_node.as_ptr();
            let target_height = (*target_ptr).level();

            // Rewire skip links around the removed node.
            //
            // For l < target_height (target participates in the level-l list):
            //   Before: pred (rank P) --[T−P]--> target (rank T) --[d]--> X (rank T+d)
            //   After:  pred (rank P) --[(T−P)+d−1]--> X (now at rank T+d−1)
            //   If target.links[l] is None (target was the last at level l):
            //     pred.links[l] ← None
            //
            // For l >= target_height (target has no skip-link slot at this level):
            //   pred.links[l] already skips over target; decrement its distance by 1.
            for (l, (pred_nn, pred_rank)) in precursors
                .iter()
                .copied()
                .zip(precursor_distances.iter().copied())
                .enumerate()
            {
                let pred_ptr = pred_nn.as_ptr();
                if l < target_height {
                    let pred_to_target = target_rank.saturating_sub(pred_rank);
                    let old_link = (*target_ptr).links_mut()[l].take();
                    (*pred_ptr).links_mut()[l] = if let Some(target_link) = old_link {
                        let new_dist = pred_to_target
                            .saturating_add(target_link.distance().get())
                            .saturating_sub(1);
                        Some(Link::new(target_link.node(), new_dist).expect("new_dist >= 1"))
                    } else {
                        None
                    };
                } else if let Some(link) = (*pred_ptr).links_mut()[l].as_mut() {
                    link.decrement_distance()
                        .expect("skip list invariant: distance >= 2 before decrement");
                }
            }

            // Capture predecessor before removing the node.
            let new_tail = (*target_ptr).prev();

            // Detach the target node from the prev/next chain and take its value.
            let mut popped = (*target_ptr).pop();
            (
                popped.take_value().expect("target node always has a value"),
                new_tail,
            )
        };

        if index.saturating_add(1) == self.len {
            self.tail = if self.len == 1 { None } else { new_tail };
        }
        self.len = self.len.saturating_sub(1);
        value
    }

    /// Swaps two elements in the list.
    ///
    /// If `a == b`, this is a no-op.
    ///
    /// This operation is `$O(\log n)$`.
    ///
    /// # Panics
    ///
    /// Panics if `a >= self.len()` or `b >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(1);
    /// list.push_back(2);
    /// list.push_back(3);
    /// list.swap(0, 2);
    /// assert_eq!(list.get(0), Some(&3));
    /// assert_eq!(list.get(1), Some(&2));
    /// assert_eq!(list.get(2), Some(&1));
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "the expect calls fire only on internal invariant violations; \
                  a < len and b < len are asserted at the top of this function, \
                  guaranteeing that the target nodes and their values exist"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "two value_mut accesses and a ptr::swap on provably distinct heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        assert!(
            a < self.len,
            "swap index a (is {a}) should be < len (is {})",
            self.len
        );
        assert!(
            b < self.len,
            "swap index b (is {b}) should be < len (is {})",
            self.len
        );
        if a == b {
            return;
        }

        let ptr_a = self.node_ptr_at(a);
        let ptr_b = self.node_ptr_at(b);

        // SAFETY: a != b guarantees the two target nodes are distinct.  We hold
        // &mut self, so no other live references to any node exist.
        // core::ptr::swap on two distinct value pointers does not violate
        // Rust's aliasing rules.
        unsafe {
            let val_a: *mut T = (*ptr_a.as_ptr()).value_mut().expect("node a has a value");
            let val_b: *mut T = (*ptr_b.as_ptr()).value_mut().expect("node b has a value");
            core::ptr::swap(val_a, val_b);
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::SkipList;

    // MARK: insert

    #[test]
    fn insert_into_empty() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.insert(0, 42);
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
        assert_eq!(
            list.head_ref().next_as_ref().and_then(|n| n.value()),
            Some(&42)
        );
        assert!(
            list.head_ref()
                .next_as_ref()
                .and_then(|n| n.next_as_ref())
                .is_none()
        );
    }

    #[test]
    fn insert_at_front() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(2);
        list.push_back(3);
        list.insert(0, 1);
        assert_eq!(list.len(), 3);
        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next_as_ref().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next_as_ref().is_none());
    }

    #[test]
    fn insert_at_back() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.insert(2, 3);
        assert_eq!(list.len(), 3);
        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next_as_ref().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next_as_ref().is_none());
    }

    #[test]
    fn insert_in_middle() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(3);
        list.insert(1, 2);
        assert_eq!(list.len(), 3);
        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next_as_ref().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next_as_ref().is_none());
    }

    #[test]
    fn insert_len_increments() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.insert(0, i);
            assert_eq!(list.len(), i.saturating_add(1));
        }
    }

    #[test]
    #[should_panic(expected = "insertion index (is 5) should be <= len (is 3)")]
    fn insert_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.insert(5, 99);
    }

    #[test]
    fn insert_interleaved_with_pop() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(3); // [1, 3]
        list.insert(1, 2); // [1, 2, 3]
        assert_eq!(list.pop_front(), Some(1)); // [2, 3]
        list.insert(0, 0); // [0, 2, 3]
        assert_eq!(list.pop_back(), Some(3)); // [0, 2]
        list.insert(2, 4); // [0, 2, 4]
        assert_eq!(list.len(), 3);
        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&0));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next_as_ref().expect("n3");
        assert_eq!(n3.value(), Some(&4));
        assert!(n3.next_as_ref().is_none());
    }

    #[test]
    fn insert_multiple_positions() {
        let mut list = SkipList::<i32>::with_capacity(1);
        // Build [0, 1, 2, 3, 4] by inserting at various positions.
        list.insert(0, 2); // [2]
        list.insert(0, 0); // [0, 2]
        list.insert(1, 1); // [0, 1, 2]
        list.insert(3, 4); // [0, 1, 2, 4]
        list.insert(3, 3); // [0, 1, 2, 3, 4]
        assert_eq!(list.len(), 5);
        let mut node = list.head_ref().next_as_ref().expect("first");
        for expected in 0..5_i32 {
            assert_eq!(node.value(), Some(&expected));
            if expected < 4 {
                node = node.next_as_ref().expect("next");
            }
        }
    }

    // MARK: remove

    #[test]
    fn remove_only_element() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(42);
        assert_eq!(list.remove(0), 42);
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
        assert!(list.head_ref().next_as_ref().is_none());
    }

    #[test]
    fn remove_at_front() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.remove(0), 1);
        assert_eq!(list.len(), 2);
        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&2));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&3));
        assert!(n2.next_as_ref().is_none());
    }

    #[test]
    fn remove_at_back() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.remove(2), 3);
        assert_eq!(list.len(), 2);
        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        assert!(n2.next_as_ref().is_none());
    }

    #[test]
    fn remove_in_middle() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.remove(1), 2);
        assert_eq!(list.len(), 2);
        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&3));
        assert!(n2.next_as_ref().is_none());
    }

    #[test]
    fn remove_len_decrements() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.push_back(i);
        }
        for i in (0..50_usize).rev() {
            assert_eq!(list.len(), i.saturating_add(1));
            list.remove(0);
        }
        assert_eq!(list.len(), 0);
    }

    #[test]
    #[should_panic(expected = "removal index (is 3) should be < len (is 3)")]
    fn remove_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.remove(3);
    }

    #[test]
    #[should_panic(expected = "removal index (is 0) should be < len (is 0)")]
    fn remove_from_empty() {
        let mut list = SkipList::<i32>::new();
        list.remove(0);
    }

    #[test]
    fn remove_interleaved_with_insert() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(2); // [1, 2]
        list.push_back(3); // [1, 2, 3]
        assert_eq!(list.remove(1), 2); // [1, 3]
        list.insert(1, 4); // [1, 4, 3]
        assert_eq!(list.remove(0), 1); // [4, 3]
        list.push_back(5); // [4, 3, 5]
        assert_eq!(list.remove(2), 5); // [4, 3]
        assert_eq!(list.len(), 2);
        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&4));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&3));
        assert!(n2.next_as_ref().is_none());
    }

    #[test]
    fn remove_all_elements() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=5_i32 {
            list.push_back(i);
        }
        assert_eq!(list.remove(2), 3); // [1, 2, 4, 5]
        assert_eq!(list.remove(0), 1); // [2, 4, 5]
        assert_eq!(list.remove(2), 5); // [2, 4]
        assert_eq!(list.remove(1), 4); // [2]
        assert_eq!(list.remove(0), 2); // []
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
    }

    #[test]
    #[expect(
        clippy::integer_division,
        clippy::integer_division_remainder_used,
        reason = "removing from middle"
    )]
    fn remove_all_in_order_from_middle() {
        let n = 50_usize;
        let mut list = SkipList::<usize>::new();
        for i in 0..n {
            list.push_back(i);
        }
        while !list.is_empty() {
            list.remove(list.len() / 2);
        }
        assert_eq!(list.len(), 0);
        assert!(list.head_ref().next_as_ref().is_none());
    }

    // MARK: swap

    #[test]
    fn swap_basic() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.swap(0, 2);
        assert_eq!(list.get(0), Some(&3));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&1));
    }

    #[test]
    fn swap_same_index() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.swap(1, 1);
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&3));
    }

    #[test]
    fn swap_adjacent() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.swap(1, 2);
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&3));
        assert_eq!(list.get(2), Some(&2));
    }

    #[test]
    fn swap_two_elements() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        list.push_back(20);
        list.swap(0, 1);
        assert_eq!(list.get(0), Some(&20));
        assert_eq!(list.get(1), Some(&10));
    }

    #[test]
    fn swap_front_back() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.push_back(i);
        }
        list.swap(0, 4);
        assert_eq!(list.get(0), Some(&5));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&3));
        assert_eq!(list.get(3), Some(&4));
        assert_eq!(list.get(4), Some(&1));
    }

    #[test]
    fn swap_preserves_len() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.swap(0, 2);
        assert_eq!(list.len(), 3);
    }

    #[test]
    #[should_panic(expected = "swap index a (is 3) should be < len (is 3)")]
    fn swap_out_of_bounds_a() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.swap(3, 0);
    }

    #[test]
    #[should_panic(expected = "swap index b (is 3) should be < len (is 3)")]
    fn swap_out_of_bounds_b() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.swap(0, 3);
    }

    #[test]
    #[should_panic(expected = "swap index a (is 0) should be < len (is 0)")]
    fn swap_empty() {
        let mut list = SkipList::<i32>::new();
        list.swap(0, 0);
    }

    #[test]
    fn swap_large() {
        let n: usize = 100;
        let mut list = SkipList::<usize>::new();
        for i in 0..n {
            list.push_back(i);
        }
        // Swap each element with its mirror across the midpoint.
        #[expect(
            clippy::integer_division,
            clippy::integer_division_remainder_used,
            reason = "swapping across midpoint"
        )]
        for i in 0..(n / 2) {
            list.swap(i, n - 1 - i);
        }
        // After all swaps, element at index i should be n - 1 - i.
        for i in 0..n {
            assert_eq!(list.get(i), Some(&(n - 1 - i)));
        }
        assert_eq!(list.len(), n);
    }
}
