//! Ordered insertion and removal for [`OrderedSkipList`](super::OrderedSkipList).

use core::ptr::NonNull;

use crate::{
    comparator::Comparator,
    level_generator::LevelGenerator,
    node::{
        Node,
        link::Link,
        visitor::{OrdMutVisitor, Visitor},
    },
    ordered_skip_list::OrderedSkipList,
};

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> OrderedSkipList<T, N, C, G> {
    /// Removes and returns the first (smallest) element, or `None` if the
    /// list is empty.
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
    /// assert_eq!(list.pop_first(), Some(1));
    /// assert_eq!(list.pop_first(), Some(2));
    /// assert_eq!(list.pop_first(), Some(3));
    /// assert_eq!(list.pop_first(), None);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        clippy::unwrap_in_result,
        reason = "head.next is Some because is_empty() was checked first; \
                  all expects fire only on internal invariant violations, not user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l is bounded by front_height ≤ max_levels, which equals the length \
                  of the links slice on every node, so all accesses are in bounds"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link unwiring, pointer extraction, and node pop all touch provably \
                  disjoint heap nodes; splitting across blocks would require \
                  unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn pop_first(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        // SAFETY: All raw pointers originate from heap allocations owned by this
        // OrderedSkipList.  No safe &mut references to any node exist while this
        // block runs.  head_ptr and front_ptr are distinct heap allocations; all
        // slice accesses are bounded by front_height ≤ max_levels = links.len().
        let value = unsafe {
            let head_ptr: *mut Node<T, N> = self.head.as_ptr();

            // front_ptr is the node at rank 1.  The list is non-empty, so
            // head.next is Some.  Converting the &mut to NonNull releases the
            // borrow immediately, leaving no live &mut when we later use head_ptr.
            let front_ptr: *mut Node<T, N> =
                NonNull::from((*head_ptr).next_as_mut().expect("list is non-empty")).as_ptr();

            let front_height = (*front_ptr).level();

            // Splice out front_node: move its skip links back to head.
            //
            // Unlike SkipList::pop_front, we do NOT adjust distances for levels
            // l >= front_height.  Distances in OrderedSkipList are all stored as
            // 1 (intentionally wrong until OrdIndexMutVisitor in Step 11).
            // Decrementing them would underflow NonZeroUsize.
            //
            // For l < front_height: head.links[l] pointed to front_node.
            //   Replace with front_node.links[l] (the next node at that level).
            // For l >= front_height: head.links[l] already skips over front_node
            //   and needs no change (pointer is still correct; distance remains
            //   its placeholder value of 1).
            for l in 0..front_height {
                (*head_ptr).links_mut()[l] = (*front_ptr).links_mut()[l].take();
            }

            let mut popped = (*front_ptr).pop();
            popped.take_value()
        };

        self.len = self.len.saturating_sub(1);
        if self.len == 0 {
            self.tail = None;
        }
        value
    }

    /// Removes and returns the last (largest) element, or `None` if the list
    /// is empty.
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
    /// assert_eq!(list.pop_last(), Some(3));
    /// assert_eq!(list.pop_last(), Some(2));
    /// assert_eq!(list.pop_last(), Some(1));
    /// assert_eq!(list.pop_last(), None);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        clippy::unwrap_in_result,
        reason = "self.tail is Some because is_empty() was checked first; \
                  all expects fire only on internal invariant violations, not user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l is bounded by tail_height ≤ max_levels, which equals the length \
                  of the links slice on every node, so all accesses are in bounds"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link clearing and node pop touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn pop_last(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let tail_ptr: NonNull<Node<T, N>> = self.tail.expect("non-empty list has a tail");

        // SAFETY: tail_ptr is a live, valid data node for the lifetime of
        // &mut self.  No other &mut reference to it exists.
        let tail_height = unsafe { tail_ptr.as_ref() }.level();

        // Find the predecessor of the tail at each skip level using a
        // pointer-equality forward traversal.
        //
        // At each level l < tail_height we advance from `current` while the
        // level-l link does NOT point to the tail.  When we stop, `current`
        // is the unique predecessor at level l (the node whose link[l] = tail).
        // We do NOT reset `current` between levels: the skip-list structure
        // guarantees we can only advance forward.
        //
        // For levels l >= tail_height, no node can link directly to the tail
        // (tail has no tower slot at those levels), so no links need clearing.
        let precursors: [NonNull<Node<T, N>>; N] = {
            let mut arr = [self.head; N];
            let mut current = self.head;

            for l in (0..tail_height).rev() {
                loop {
                    // SAFETY: `current` is a valid node in this list, live for
                    // the duration of &mut self.  No exclusive reference exists.
                    let maybe_link = unsafe { current.as_ref() }
                        .links()
                        .get(l)
                        .and_then(|lk| lk.as_ref());
                    match maybe_link {
                        None => break, // no link at this level; current is precursor
                        Some(link) if link.node() == tail_ptr => break, // current is precursor
                        Some(link) => current = link.node(),
                    }
                }
                arr[l] = current;
            }
            arr
        };

        // SAFETY: All raw pointers come from NonNull<Node<T, N>> values captured
        // during traversal or from self.tail.  No safe &mut references to any
        // node exist while this block runs.
        let value = unsafe {
            let tail_raw = tail_ptr.as_ptr();

            // Clear all skip links pointing to the tail.
            // For levels >= tail_height no link points to the tail: no-op.
            for (l, pred_nn) in precursors.iter().enumerate().take(tail_height) {
                (*pred_nn.as_ptr()).links_mut()[l] = None;
            }

            let mut popped = (*tail_raw).pop();
            popped.take_value()
        };

        // Update the cached tail pointer.  When the list becomes empty,
        // precursors[0] equals head; we set tail to None rather than pointing
        // at the sentinel.
        self.tail = if self.len == 1 {
            None
        } else {
            Some(precursors[0])
        };
        self.len = self.len.saturating_sub(1);
        value
    }

    /// Inserts `value` into the list at its sorted position.
    ///
    /// All elements are maintained in the order defined by the list's
    /// comparator.  Duplicate values are permitted; the new element is
    /// inserted before any existing elements that compare equal to it.
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
    /// assert_eq!(list.len(), 3);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "Link::new with distance 1 always succeeds (distance >= 1 is satisfied); \
                  precursors[0] always exists because max_levels >= 1; \
                  all expects fire only on internal invariant violations, not on user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "precursors[0] is valid: max_levels >= 1 so precursors.len() >= 1; \
                  precursors[l].links_mut()[l] is valid: the visitor invariant guarantees \
                  each precursor node has a link slot at the level it was recorded for"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "insertion and link wiring touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn insert(&mut self, value: T) {
        let height = self.generator.level().saturating_add(1);

        // Use OrdMutVisitor to locate the insertion point and collect precursors.
        //
        // `self.head` is a `NonNull` (a `Copy` type), so copying it does not
        // borrow `self`.  The closure borrows only `self.comparator` (shared),
        // which is a distinct field from `self.head`.  Both borrows coexist
        // safely and are released when `visitor` is dropped via `into_parts()`.
        let precursors = {
            let head = self.head;
            let cmp = |v: &T, t: &T| self.comparator.compare(v, t);
            let mut visitor = OrdMutVisitor::new(head, &value, cmp);
            visitor.traverse();
            let (_current, _found, precursors) = visitor.into_parts();
            precursors
        };
        // `visitor` and `cmp` are dropped here, releasing the borrow on
        // `self.comparator`.  `value` is no longer borrowed by the visitor.

        // SAFETY: All raw pointers originate from `NonNull<Node<T, N>>` values
        // captured during traversal.  They point into heap allocations exclusively
        // owned by this `OrderedSkipList`.  No safe `&mut` references to any node
        // exist while this block runs.  The pointer `new_raw` is distinct from
        // every precursor: it is freshly allocated by `Node::insert_after`.
        let new_node_nonnull: NonNull<Node<T, N>> = unsafe {
            let new_raw: *mut Node<T, N> =
                Node::insert_after(precursors[0], Node::with_value(height, value)).as_ptr();

            // Wire skip links for levels 0..height.
            //
            // For each level l < height (the new node's tower reaches this level):
            //   pred.links[l]     ← Link(new_node, 1)
            //   new_node.links[l] ← Link(old_target, 1)  (or None if pred had no link)
            //
            // Levels l >= height are left unchanged: the new node has no tower
            // slot at those levels, so the existing links already skip over it.
            //
            // Note: link distances are stored as 1 for all links because
            // `OrdMutVisitor` does not track rank distances.  Accurate distance
            // maintenance for rank queries (`get(index)`, `rank(value)`) is
            // deferred to Phase 2 step 11, when `OrdIndexMutVisitor` is added.
            for (l, pred_nn) in precursors.iter().enumerate().take(height) {
                let pred_ptr = pred_nn.as_ptr();
                let old_link = (*pred_ptr).links_mut()[l].take();
                (*pred_ptr).links_mut()[l] = Some(
                    Link::new(NonNull::new_unchecked(new_raw), 1)
                        .expect("distance 1 is always valid"),
                );
                (*new_raw).links_mut()[l] = old_link
                    .map(|old| Link::new(old.node(), 1).expect("distance 1 is always valid"));
            }

            NonNull::new_unchecked(new_raw)
        };

        // Update the cached tail pointer if the new node is the last element.
        //
        // The new node is the tail iff `precursors[0]` (the level-0 predecessor)
        // was previously the tail, or the list was empty (tail = None) and the
        // new node was inserted right after the head.
        let is_new_tail = self.tail.is_none_or(|tail| precursors[0] == tail);
        if is_new_tail {
            self.tail = Some(new_node_nonnull);
        }

        self.len = self.len.saturating_add(1);
    }

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::OrderedSkipList;
    use crate::{comparator::FnComparator, level_generator::geometric::Geometric};

    // MARK: insert

    #[test]
    fn insert_into_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
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
    fn insert_maintains_sorted_order() {
        // Use a single-level generator so tower heights are deterministic.
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(3);
        list.insert(1);
        list.insert(4);
        list.insert(2);
        assert_eq!(list.len(), 4);

        let mut node = list.head_ref().next_as_ref().expect("first");
        let mut prev_val = i32::MIN;
        for _ in 0..4 {
            let v = *node.value().expect("value");
            assert!(v >= prev_val, "{v} < {prev_val}: list is not sorted");
            prev_val = v;
            node = match node.next_as_ref() {
                Some(n) => n,
                None => break,
            };
        }
    }

    #[test]
    fn insert_duplicates_adjacent() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.len(), 4);

        let mut values = Vec::new();
        let mut cur = list.head_ref().next_as_ref();
        while let Some(n) = cur {
            values.push(*n.value().expect("value"));
            cur = n.next_as_ref();
        }
        assert_eq!(values, [1, 2, 2, 3]);
    }

    #[test]
    fn insert_at_front_updates_order() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(5);
        list.insert(3);
        list.insert(1);
        assert_eq!(list.len(), 3);

        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&3));
        let n3 = n2.next_as_ref().expect("n3");
        assert_eq!(n3.value(), Some(&5));
        assert!(n3.next_as_ref().is_none());
    }

    #[test]
    fn insert_at_back_updates_tail() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.len(), 3);
        // The last value in the chain must be 3.
        let mut cur = list.head_ref().next_as_ref().expect("first");
        while let Some(next) = cur.next_as_ref() {
            cur = next;
        }
        assert_eq!(cur.value(), Some(&3));
    }

    #[test]
    fn insert_len_increments() {
        let mut list = OrderedSkipList::<usize>::new();
        for i in 0..50_usize {
            list.insert(i);
            assert_eq!(list.len(), i + 1);
        }
    }

    #[test]
    fn insert_reverse_order_still_sorted() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        for i in (0..10_i32).rev() {
            list.insert(i);
        }
        assert_eq!(list.len(), 10);

        let mut cur = list.head_ref().next_as_ref().expect("first");
        for expected in 0..10_i32 {
            assert_eq!(cur.value(), Some(&expected));
            cur = match cur.next_as_ref() {
                Some(n) => n,
                None => break,
            };
        }
    }

    #[test]
    fn insert_mixed_order_large() {
        let mut list = OrderedSkipList::<i32>::new();
        // Insert a shuffled sequence and verify sorted output.
        let values = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0];
        for &v in &values {
            list.insert(v);
        }
        assert_eq!(list.len(), 10);

        let mut cur = list.head_ref().next_as_ref().expect("first");
        for expected in 0..10_i32 {
            assert_eq!(cur.value(), Some(&expected));
            cur = match cur.next_as_ref() {
                Some(n) => n,
                None => break,
            };
        }
    }

    // MARK: pop_first

    #[test]
    fn pop_first_from_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        assert_eq!(list.pop_first(), None);
    }

    #[test]
    fn pop_first_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.pop_first(), Some(42));
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        // Second pop on now-empty list
        assert_eq!(list.pop_first(), None);
    }

    #[test]
    fn pop_first_returns_ascending_order() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(3);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.pop_first(), Some(1));
        assert_eq!(list.pop_first(), Some(2));
        assert_eq!(list.pop_first(), Some(3));
        assert_eq!(list.pop_first(), None);
    }

    #[test]
    fn pop_first_len_decrements() {
        let mut list = OrderedSkipList::<usize>::new();
        for i in 0..20_usize {
            list.insert(i);
        }
        for remaining in (0..20_usize).rev() {
            list.pop_first();
            assert_eq!(list.len(), remaining);
        }
        assert_eq!(list.pop_first(), None);
    }

    #[test]
    fn pop_first_duplicate_at_front() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.pop_first(), Some(1));
        assert_eq!(list.pop_first(), Some(1));
        assert_eq!(list.pop_first(), Some(2));
        assert_eq!(list.pop_first(), None);
    }

    #[test]
    fn pop_first_custom_comparator() {
        // Largest-first ordering: pop_first removes the largest element.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(1);
        list.insert(3);
        list.insert(2);
        assert_eq!(list.pop_first(), Some(3));
        assert_eq!(list.pop_first(), Some(2));
        assert_eq!(list.pop_first(), Some(1));
        assert_eq!(list.pop_first(), None);
    }

    // MARK: pop_last

    #[test]
    fn pop_last_from_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        assert_eq!(list.pop_last(), None);
    }

    #[test]
    fn pop_last_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.pop_last(), Some(42));
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        // Second pop on now-empty list
        assert_eq!(list.pop_last(), None);
    }

    #[test]
    fn pop_last_returns_descending_order() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(3);
        list.insert(1);
        list.insert(2);
        assert_eq!(list.pop_last(), Some(3));
        assert_eq!(list.pop_last(), Some(2));
        assert_eq!(list.pop_last(), Some(1));
        assert_eq!(list.pop_last(), None);
    }

    #[test]
    fn pop_last_len_decrements() {
        let mut list = OrderedSkipList::<usize>::new();
        for i in 0..20_usize {
            list.insert(i);
        }
        for remaining in (0..20_usize).rev() {
            list.pop_last();
            assert_eq!(list.len(), remaining);
        }
        assert_eq!(list.pop_last(), None);
    }

    #[test]
    fn pop_last_duplicate_at_back() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        assert_eq!(list.pop_last(), Some(2));
        assert_eq!(list.pop_last(), Some(2));
        assert_eq!(list.pop_last(), Some(1));
        assert_eq!(list.pop_last(), None);
    }

    #[test]
    fn pop_last_custom_comparator() {
        // Largest-first ordering: pop_last removes the smallest element.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(1);
        list.insert(3);
        list.insert(2);
        assert_eq!(list.pop_last(), Some(1));
        assert_eq!(list.pop_last(), Some(2));
        assert_eq!(list.pop_last(), Some(3));
        assert_eq!(list.pop_last(), None);
    }

    #[test]
    fn pop_first_and_pop_last_interleaved() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=6 {
            list.insert(i); // [1, 2, 3, 4, 5, 6]
        }
        assert_eq!(list.pop_last(), Some(6)); // [1, 2, 3, 4, 5]
        assert_eq!(list.pop_first(), Some(1)); // [2, 3, 4, 5]
        assert_eq!(list.pop_last(), Some(5)); // [2, 3, 4]
        assert_eq!(list.pop_first(), Some(2)); // [3, 4]
        assert_eq!(list.pop_last(), Some(4)); // [3]
        assert_eq!(list.pop_first(), Some(3)); // []
        assert_eq!(list.pop_last(), None);
        assert_eq!(list.pop_first(), None);
        assert_eq!(list.len(), 0);
    }
}
