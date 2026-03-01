//! Positional write methods for [`SkipList`](super::SkipList):
//! `insert`, `remove`, and `swap`.

use core::ptr::NonNull;

use crate::{
    level_generator::LevelGenerator,
    node::{Node, link::Link},
    skip_list::SkipList,
};

impl<T, G: LevelGenerator> SkipList<T, G> {
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
        reason = "traversal, insertion, and link wiring all touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        assert!(
            index <= self.len,
            "insertion index (is {index}) should be <= len (is {})",
            self.len
        );

        let height = self.generator.level().saturating_add(1);
        let max_levels = self.head.level();
        // new_rank: the rank of the new node after insertion
        // (head = rank 0; element at index i has rank i + 1).
        let new_rank = index.saturating_add(1);

        // SAFETY: All raw pointers originate from heap allocations owned by this
        // SkipList.  No safe &mut references to any node exist while this block
        // runs.  Different update[] entries may alias (same predecessor node,
        // different levels) but each iteration accesses a distinct links[l]
        // index, so no simultaneous aliasing occurs.
        let new_node_nonnull: NonNull<Node<T>> = unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;

            // update[l] = (predecessor at level l, its rank from head).
            // Initialised to (head, 0): head is always a valid predecessor.
            let mut update: Vec<(*mut Node<T>, usize)> = vec![(head_ptr, 0_usize); max_levels];
            let mut current: *mut Node<T> = head_ptr;
            let mut current_rank: usize = 0;

            for l in (0..max_levels).rev() {
                while let Some(link) = (*current).links()[l].as_ref() {
                    let next_rank = current_rank.saturating_add(link.distance().get());
                    if next_rank >= new_rank {
                        break;
                    }
                    current_rank = next_rank;
                    current = NonNull::from(link.node()).as_ptr();
                }
                update[l] = (current, current_rank);
            }

            // Insert the new node right after the level-0 predecessor.
            let (pred0_ptr, _) = update[0];
            (*pred0_ptr).insert_after(Node::with_value(height, value));

            // Obtain a raw pointer to the new node without holding a live &mut.
            let new_raw: *mut Node<T> = NonNull::from(
                (*pred0_ptr)
                    .next_mut()
                    .expect("node was just inserted after predecessor"),
            )
            .as_ptr();

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
            for (l, &(pred_ptr, pred_rank)) in update.iter().enumerate() {
                if l < height {
                    let distance = new_rank.saturating_sub(pred_rank);
                    let old_link = (*pred_ptr).links_mut()[l].take();
                    (*pred_ptr).links_mut()[l] =
                        Some(Link::new(&*new_raw, distance).expect("distance >= 1"));
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

            // SAFETY: new_raw was derived from NonNull::from(next_mut()), so it
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
        reason = "update[0].links[0] exists because index < len guarantees a node at target_rank; \
                  take_value is Some for any body/tail node; \
                  Link::new distance is computed to be >= 1; \
                  all expects fire only on internal invariant violations, not user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < target_height <= max_levels = target.links.len(); \
                  update[l].0 was reached at level l so update[l].0.links.len() > l; \
                  all accesses are bounded by max_levels = head.links.len()"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "traversal, link rewiring, and node pop all touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        assert!(
            index < self.len,
            "removal index (is {index}) should be < len (is {})",
            self.len
        );

        let max_levels = self.head.level();
        // head = rank 0; the element at logical index i is at rank i + 1.
        let target_rank = index.saturating_add(1);

        // SAFETY: All raw pointers originate from heap allocations owned by this
        // SkipList.  No safe &mut references to any node exist while this block
        // runs.  Different update[] entries may alias (same predecessor node,
        // different levels) but each iteration accesses a distinct links[l]
        // index, so no simultaneous aliasing occurs.
        let (value, pred0) = unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;

            // update[l] = (predecessor at level l, its rank from head).
            // Initialised to (head, 0): head is always a valid predecessor.
            let mut update: Vec<(*mut Node<T>, usize)> = vec![(head_ptr, 0_usize); max_levels];
            let mut current: *mut Node<T> = head_ptr;
            let mut current_rank: usize = 0;

            for l in (0..max_levels).rev() {
                while let Some(link) = (*current).links()[l].as_ref() {
                    let next_rank = current_rank.saturating_add(link.distance().get());
                    if next_rank >= target_rank {
                        break;
                    }
                    current_rank = next_rank;
                    current = NonNull::from(link.node()).as_ptr();
                }
                update[l] = (current, current_rank);
            }

            // The target is update[0]'s level-0 successor.
            // index < len guarantees this link exists.
            let target_ptr: *mut Node<T> = NonNull::from(
                (*update[0].0).links()[0]
                    .as_ref()
                    .expect("update[0].links[0] points to the target node")
                    .node(),
            )
            .as_ptr();

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
            //   The distance is >= 2 because pred_rank < target_rank and
            //   next_rank > target_rank (no level-l link can point to target at these
            //   levels, since target was never wired into this level during insertion).
            for (l, &(pred_ptr, pred_rank)) in update.iter().enumerate() {
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

            // Detach the target node from the prev/next chain and take its value.
            // Also capture the level-0 predecessor for the tail update below.
            let pred0 = update[0].0;
            let mut popped = (*target_ptr).pop();
            (
                popped.take_value().expect("target node always has a value"),
                pred0,
            )
        };

        if index.saturating_add(1) == self.len {
            self.tail = if self.len == 1 {
                None
            } else {
                // SAFETY: pred0 is a live data node pointer owned by this SkipList.
                Some(unsafe { NonNull::new_unchecked(pred0) })
            };
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
        reason = "two independent skip-list traversals and a ptr::swap on provably distinct \
                  heap nodes; splitting across blocks would require unsafe-crossing variables"
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

        let max_levels = self.head.level();

        // SAFETY: Both indices are in bounds (asserted above) and a != b guarantees
        // the two target nodes are distinct. We hold &mut self, so no other live
        // references to any node exist. core::ptr::swap on two distinct value
        // pointers does not violate Rust's aliasing rules.
        unsafe {
            // Traversal for index `a`: same algorithm as `get_mut`.
            let target_rank_a = a.saturating_add(1);
            let mut current: *const Node<T> = &raw const *self.head;
            let mut current_rank: usize = 0;
            for l in (0..max_levels).rev() {
                while let Some(Some(link)) = (*current).links().get(l) {
                    let next_rank = current_rank.saturating_add(link.distance().get());
                    if next_rank >= target_rank_a {
                        break;
                    }
                    current_rank = next_rank;
                    current = link.node();
                }
            }
            let node_a = NonNull::from(
                (*current)
                    .links()
                    .first()
                    .expect("level-0 link exists because a < len")
                    .as_ref()
                    .expect("level-0 link is Some because a < len guarantees a node")
                    .node(),
            );
            let ptr_a: *mut T = (*node_a.as_ptr()).value_mut().expect("node a has a value");

            // Traversal for index `b`.
            let target_rank_b = b.saturating_add(1);
            current = &raw const *self.head;
            current_rank = 0;
            for l in (0..max_levels).rev() {
                while let Some(Some(link)) = (*current).links().get(l) {
                    let next_rank = current_rank.saturating_add(link.distance().get());
                    if next_rank >= target_rank_b {
                        break;
                    }
                    current_rank = next_rank;
                    current = link.node();
                }
            }
            let node_b = NonNull::from(
                (*current)
                    .links()
                    .first()
                    .expect("level-0 link exists because b < len")
                    .as_ref()
                    .expect("level-0 link is Some because b < len guarantees a node")
                    .node(),
            );
            let ptr_b: *mut T = (*node_b.as_ptr()).value_mut().expect("node b has a value");

            core::ptr::swap(ptr_a, ptr_b);
        }
    }
}

#[expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test code, safety guarantees can be relaxed"
)]
#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::SkipList;
    use crate::node::link::Link;

    // MARK: insert

    #[test]
    fn insert_into_empty() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.insert(0, 42);
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
        assert_eq!(list.head.next().and_then(|n| n.value()), Some(&42));
        assert!(list.head.next().and_then(|n| n.next()).is_none());
    }

    #[test]
    fn insert_at_front() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(2);
        list.push_back(3);
        list.insert(0, 1);
        assert_eq!(list.len(), 3);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next().is_none());
    }

    #[test]
    fn insert_at_back() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.insert(2, 3);
        assert_eq!(list.len(), 3);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next().is_none());
    }

    #[test]
    fn insert_in_middle() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(3);
        list.insert(1, 2);
        assert_eq!(list.len(), 3);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next().is_none());
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

    /// With `with_capacity(1)` the generator assigns height = 1 to every node.
    /// Verify that links are correctly maintained after `insert(1, 2)` into [1, 3]:
    ///
    /// ```text
    /// Before: head ---[1]---> n1(1) ---[1]---> n3(3, links=None)
    /// After:  head ---[1]---> n1(1) ---[1]---> new(2) ---[1]---> n3(3, links=None)
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn insert_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(3); // [1, 3]
        list.insert(1, 2); // [1, 2, 3]

        assert_eq!(list.len(), 3);
        // head.links[0] → n1(1) at distance 1 (unchanged)
        {
            let link: &Link<_> = list.head.links()[0].as_ref().expect("head link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&1));
        }
        // n1(1).links[0] → new(2) at distance 1
        {
            let n1 = list.head.next().expect("n1");
            assert_eq!(n1.value(), Some(&1));
            let link: &Link<_> = n1.links()[0].as_ref().expect("n1 link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&2));
        }
        // new(2).links[0] → n3(3) at distance 1
        {
            let new_node = list.head.next().expect("n1").next().expect("new_node");
            assert_eq!(new_node.value(), Some(&2));
            let link: &Link<_> = new_node.links()[0].as_ref().expect("new_node link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&3));
        }
        // n3(3).links[0] = None
        {
            let n3 = list
                .head
                .next()
                .expect("n1")
                .next()
                .expect("new_node")
                .next()
                .expect("n3");
            assert_eq!(n3.value(), Some(&3));
            assert!(n3.links()[0].is_none());
        }
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
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&0));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next().expect("n3");
        assert_eq!(n3.value(), Some(&4));
        assert!(n3.next().is_none());
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
        let mut node = list.head.next().expect("first");
        for expected in 0..5_i32 {
            assert_eq!(node.value(), Some(&expected));
            if expected < 4 {
                node = node.next().expect("next");
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
        assert!(list.head.next().is_none());
    }

    #[test]
    fn remove_at_front() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.remove(0), 1);
        assert_eq!(list.len(), 2);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&2));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&3));
        assert!(n2.next().is_none());
    }

    #[test]
    fn remove_at_back() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.remove(2), 3);
        assert_eq!(list.len(), 2);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        assert!(n2.next().is_none());
    }

    #[test]
    fn remove_in_middle() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.remove(1), 2);
        assert_eq!(list.len(), 2);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&3));
        assert!(n2.next().is_none());
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

    /// With `with_capacity(1)` the generator assigns height = 1 to every node.
    /// Verify that links are correctly maintained after removing the middle element
    /// from [1, 2, 3]:
    ///
    /// ```text
    /// Before: head ---[1]---> n1(1) ---[1]---> n2(2) ---[1]---> n3(3, None)
    /// After:  head ---[1]---> n1(1) ---[1]---> n3(3, None)
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn remove_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(2); // [1, 2]
        list.push_back(3); // [1, 2, 3]
        assert_eq!(list.remove(1), 2); // [1, 3]

        assert_eq!(list.len(), 2);
        // head.links[0] → n1(1) at distance 1 (unchanged)
        {
            let link: &Link<_> = list.head.links()[0].as_ref().expect("head link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&1));
        }
        // n1(1).links[0] → n3(3) at distance 1 (previously pointed to n2)
        {
            let n1 = list.head.next().expect("n1");
            assert_eq!(n1.value(), Some(&1));
            let link: &Link<_> = n1.links()[0].as_ref().expect("n1 link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&3));
        }
        // n3(3).links[0] = None
        {
            let n3 = list.head.next().expect("n1").next().expect("n3");
            assert_eq!(n3.value(), Some(&3));
            assert!(n3.links()[0].is_none());
        }
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
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&4));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&3));
        assert!(n2.next().is_none());
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
        assert!(list.head.next().is_none());
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
