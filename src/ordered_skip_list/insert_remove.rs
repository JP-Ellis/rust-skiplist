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
    use crate::level_generator::geometric::Geometric;

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
}
