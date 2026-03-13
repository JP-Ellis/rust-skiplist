//! End-of-list write methods for [`SkipList`](super::SkipList):
//! `push_front`, `push_back`, `pop_front`, and `pop_back`.

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
    /// Inserts `value` at the front of the list.
    ///
    /// The new element becomes the element at index 0, shifting all existing
    /// elements one position to the right.  This operation is `$O(\log n)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_front(1);
    /// list.push_front(2);
    /// assert_eq!(list.len(), 2);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "insert_after guarantees head.next is Some; Link::new(_, 1) and \
                  increment_distance cannot fail under any reachable condition"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l is bounded by height <= max_levels, which equals the length \
                  of the links slice on every node, so all accesses are in bounds"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "head_ptr and new_raw are provably distinct heap allocations; \
                  batching the operations avoids repeating the same SAFETY preamble"
    )]
    #[inline]
    pub fn push_front(&mut self, value: T) {
        // height ∈ [0, total]: number of skip links to allocate.
        let height = self.generator.level();
        let max_levels = self.head_ref().level();

        // SAFETY: Node::with_value produces a detached node (prev = next = None),
        // which is the only kind insert_after accepts.
        // SAFETY: insert_after returns node_ptr with the allocation's root
        // provenance (Box::into_raw), so storing it in Links is safe.
        let new_node_ptr: NonNull<Node<T, N>> =
            unsafe { Node::insert_after(self.head, Node::with_value(height, value)) };

        // Update skip links so the structure remains consistent.
        //
        // Before insertion (new_node not yet wired):
        //   head.links[l] → X at distance d   (X is at rank d)
        //
        // After inserting new_node at rank 1:
        //   head is rank 0, new_node is rank 1, X is now at rank d+1.
        //   distance head → new_node = 1
        //   distance new_node → X   = (d+1) - 1 = d  (unchanged)
        //
        // For levels 0..height:
        //   head.links[l]     ← Link { new_node, 1 }
        //   new_node.links[l] ← old head link (distance unchanged)
        //
        // For levels height..max_levels:
        //   head.links[l].distance += 1  (target node shifted one rank right)
        //
        // SAFETY: head_ptr and new_raw point to distinct, heap-allocated Node<T>
        // values.  No safe references to either node exist during this block.
        unsafe {
            let head_ptr: *mut Node<T, N> = self.head.as_ptr();
            let new_raw: *mut Node<T, N> = new_node_ptr.as_ptr();

            for l in 0..height {
                let old = (*head_ptr).links_mut()[l].take();
                (*new_raw).links_mut()[l] = old;
                (*head_ptr).links_mut()[l] = Some(
                    Link::new(NonNull::new_unchecked(new_raw), 1)
                        .expect("distance 1 is always valid"),
                );
            }

            for l in height..max_levels {
                if let Some(link) = (*head_ptr).links_mut()[l].as_mut() {
                    link.increment_distance()
                        .expect("distance overflow requires > usize::MAX nodes");
                }
            }
        }

        // If the list was empty the new node is also the tail.
        if self.len == 0 {
            self.tail = Some(new_node_ptr);
        }
        self.len = self.len.saturating_add(1);
    }

    /// Appends `value` to the back of the list.
    ///
    /// The new element becomes the element at index `self.len()`, placed after
    /// all existing elements.  This operation is `$O(\log n)$` expected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.len(), 2);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "insert_after guarantees the tail's next is Some immediately after; \
                  distance equals new_rank − pred_rank where pred_rank ≤ self.len < new_rank \
                  so distance ≥ 1 always; overflow requires > usize::MAX nodes"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l is bounded by height ≤ max_levels, which equals the length of precursors[] \
                  and the links slice on every node, so all accesses are in bounds"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "insertion and link wiring touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn push_back(&mut self, value: T) {
        // height ∈ [0, total]: number of skip links to allocate.
        let height = self.generator.level();

        // IndexMutVisitor with target = len + 1 advances to the end of the list
        // (all ranks are ≤ len < len+1), recording the rightmost predecessor at
        // each level.  into_parts() releases the &mut borrow.
        let (tail_ptr, precursors, precursor_distances) = {
            let mut visitor = IndexMutVisitor::new(self.head, self.len.saturating_add(1));
            visitor.traverse();
            visitor.into_parts()
        };

        // SAFETY: All raw pointers come from NonNull<Node<T, N>> captured during
        // traversal.  They originate from heap allocations owned by this SkipList.
        // No safe &mut references to any node exist while this block runs.
        let new_node_nonnull: NonNull<Node<T, N>> = unsafe {
            // `tail_ptr` is the rightmost node (or head if the list is empty).
            // insert_after returns node_ptr with the allocation's root provenance
            // (Box::into_raw), so storing it in Links avoids sibling-tag issues.
            let new_raw: *mut Node<T, N> =
                Node::insert_after(tail_ptr, Node::with_value(height, value)).as_ptr();

            // new_rank is the 0-based rank of new_node (head = 0, elements 1..=n).
            // pred_rank ≤ self.len < new_rank, so distance ≥ 1 for all levels.
            let new_rank = self.len.saturating_add(1);

            for (l, (pred_nn, pred_rank)) in precursors
                .iter()
                .copied()
                .zip(precursor_distances.iter().copied())
                .enumerate()
                .take(height)
            {
                // pred_rank <= self.len < new_rank, so saturating_sub == plain sub here.
                let distance = new_rank.saturating_sub(pred_rank);
                (*pred_nn.as_ptr()).links_mut()[l] = Some(
                    Link::new(NonNull::new_unchecked(new_raw), distance).expect("distance >= 1"),
                );
                // new_node.links[l] remains None (Node::with_value initialises all to None).
            }
            // Levels height..max_levels need no update: new_node is at the end
            // and no existing skip link spans past the old tail.

            // SAFETY: new_raw comes from insert_after (Box::into_raw), so it
            // is non-null.  Return it so self.tail can be updated outside.
            NonNull::new_unchecked(new_raw)
        };

        self.tail = Some(new_node_nonnull);
        self.len = self.len.saturating_add(1);
    }

    /// Removes and returns the first element, or `None` if the list is empty.
    ///
    /// This operation is `$O(\log n)$` expected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.pop_front(), Some(1));
    /// assert_eq!(list.pop_front(), Some(2));
    /// assert_eq!(list.pop_front(), None);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        clippy::unwrap_in_result,
        reason = "head.next is Some because is_empty() was checked first; \
                  decrement_distance cannot underflow because head.links[l] for \
                  l ≥ front_height cannot point to front_node (front_node.level() ≤ l), \
                  so its distance is ≥ 2 before the decrement; \
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
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let max_levels = self.head_ref().level();

        // SAFETY: All raw pointers originate from heap allocations owned by this
        // SkipList.  No safe &mut references to any node exist while this block
        // runs.  head_ptr and front_ptr are distinct heap allocations; all slice
        // accesses are bounded by front_height ≤ max_levels = links.len().
        let value = unsafe {
            let head_ptr: *mut Node<T, N> = self.head.as_ptr();

            // front_ptr is the node at rank 1.  The list is non-empty, so
            // head.next is Some.  Converting the &mut to NonNull releases the
            // borrow immediately, leaving no live &mut when we later use head_ptr.
            let front_ptr: *mut Node<T, N> =
                NonNull::from((*head_ptr).next_as_mut().expect("list is non-empty")).as_ptr();

            let front_height = (*front_ptr).level();

            // Restore skip links as if front_node never existed.
            //
            // Invariant (maintained by push_front / push_back):
            //   For l < front_height, head.links[l] = Link(front_node, 1).
            //   front_node.links[l] points to the next node at level l.
            // Moving that link back to head is the exact inverse of push_front's wiring.
            for l in 0..front_height {
                (*head_ptr).links_mut()[l] = (*front_ptr).links_mut()[l].take();
            }

            // For l ≥ front_height, head.links[l] cannot point to front_node
            // (front_node has no link tower at those levels), so the target node
            // is at rank ≥ 2.  Removing front_node shifts every node left by 1,
            // so each such distance decrements from d ≥ 2 to d−1 ≥ 1.
            for l in front_height..max_levels {
                if let Some(link) = (*head_ptr).links_mut()[l].as_mut() {
                    link.decrement_distance().expect(
                        "skip list invariant: target at rank ≥ 2 so distance ≥ 2 before decrement",
                    );
                }
            }

            // Detach front_node from the prev/next chain.
            // pop() sets: head.next = front_node.next
            //             front_node.next.prev = &head  (if next exists)
            let mut popped = (*front_ptr).pop();
            popped.take_value()
        };

        self.len = self.len.saturating_sub(1);
        // If that was the only element the list is now empty.
        if self.len == 0 {
            self.tail = None;
        }
        value
    }

    /// Removes and returns the last element, or `None` if the list is empty.
    ///
    /// This operation is `$O(\log n)$` expected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.pop_back(), Some(2));
    /// assert_eq!(list.pop_back(), Some(1));
    /// assert_eq!(list.pop_back(), None);
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "l is bounded by max_levels = head.links.len(); any node reachable at level l \
                  has links.len() > l by the skip-list invariant, so all accesses are in bounds"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link clearing, prev capture, and node pop touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        // IndexMutVisitor with target = self.len advances to the tail node.
        // `current` from into_parts() is the tail itself.
        let (tail_node, precursors, _) = {
            let mut visitor = IndexMutVisitor::new(self.head, self.len);
            visitor.traverse();
            visitor.into_parts()
        };

        // SAFETY: All raw pointers come from NonNull<Node<T, N>> captured during
        // traversal.  They originate from heap allocations owned by this SkipList.
        // No safe &mut references to any node exist while this block runs.
        let (value, new_tail) = unsafe {
            let back_ptr: *mut Node<T, N> = tail_node.as_ptr();
            let back_height = (*back_ptr).level();

            // Remove skip links that pointed to the tail.
            // For l < back_height, precursors[l].links[l] = Link(tail, d): clear it.
            // For l >= back_height, no level-l link reaches the tail, so leave as-is.
            for (l, pred_nn) in precursors.iter().enumerate().take(back_height) {
                (*pred_nn.as_ptr()).links_mut()[l] = None;
            }

            // Capture the predecessor before removing the node.
            let new_tail = (*back_ptr).prev();

            // Detach the tail from the prev/next chain.
            let mut popped = (*back_ptr).pop();
            (popped.take_value(), new_tail)
        };

        // Update the cached tail pointer.  When the list had exactly one element
        // new_tail points to the head sentinel; we set tail to None instead.
        self.tail = if self.len == 1 { None } else { new_tail };
        self.len = self.len.saturating_sub(1);
        value
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::SkipList;

    // MARK: push_front

    #[test]
    fn push_front_into_empty() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_front(42);
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
    fn push_front_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_front(1);
        list.push_front(2);
        list.push_front(3);
        assert_eq!(list.len(), 3);

        // Last pushed element is at the front: 3 → 2 → 1
        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&3));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next_as_ref().expect("n3");
        assert_eq!(n3.value(), Some(&1));
        assert!(n3.next_as_ref().is_none());
    }

    #[test]
    fn push_front_len_increments() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.push_front(i);
            assert_eq!(list.len(), i + 1);
        }
    }

    /// With `with_capacity(1)` the generator always assigns height = 0.
    /// After two `push_front` calls all data nodes have no skip links, so all
    /// skip-link slots remain `None` and navigation is via `prev`/`next` only:
    ///
    /// ```text
    /// head → node(20) → node(10)
    /// head.links[0] = None  (no skip-link chains)
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn push_front_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);

        list.push_front(10);
        // No skip links: all data nodes have height 0.
        assert!(list.head_ref().links()[0].is_none());

        list.push_front(20);
        // Still no skip links.
        assert!(list.head_ref().links()[0].is_none());

        // Verify prev/next chain: head → 20 → 10.
        // links() is empty for height-0 nodes; use first() to avoid indexing.
        let front = list.head_ref().next_as_ref().expect("front node");
        assert_eq!(front.value(), Some(&20));
        assert!(front.links().first().is_none_or(|l| l.is_none()));

        let second = front.next_as_ref().expect("second node");
        assert_eq!(second.value(), Some(&10));
        assert!(second.links().first().is_none_or(|l| l.is_none()));
    }

    // MARK: push_back

    #[test]
    fn push_back_into_empty() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(42);
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
    fn push_back_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.len(), 3);

        // Elements are in insertion order: 1 → 2 → 3
        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next_as_ref().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next_as_ref().is_none());
    }

    #[test]
    fn push_back_len_increments() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.push_back(i);
            assert_eq!(list.len(), i + 1);
        }
    }

    /// With `with_capacity(1)` the generator always assigns height = 0.
    /// After two `push_back` calls all data nodes have no skip links, so all
    /// skip-link slots remain `None` and navigation is via `prev`/`next` only:
    ///
    /// ```text
    /// head → node(10) → node(20)
    /// head.links[0] = None  (no skip-link chains)
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn push_back_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);

        list.push_back(10);
        // No skip links: all data nodes have height 0.
        assert!(list.head_ref().links()[0].is_none());

        list.push_back(20);
        // Still no skip links.
        assert!(list.head_ref().links()[0].is_none());

        // Verify prev/next chain: head → 10 → 20.
        // links() is empty for height-0 nodes; use first() to avoid indexing.
        let front = list.head_ref().next_as_ref().expect("first node");
        assert_eq!(front.value(), Some(&10));
        assert!(front.links().first().is_none_or(|l| l.is_none()));

        let second = front.next_as_ref().expect("second node");
        assert_eq!(second.value(), Some(&20));
        assert!(second.links().first().is_none_or(|l| l.is_none()));
    }

    #[test]
    fn push_back_after_push_front() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_front(2); // [2]
        list.push_back(3); // [2, 3]
        list.push_front(1); // [1, 2, 3]
        assert_eq!(list.len(), 3);

        let n1 = list.head_ref().next_as_ref().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next_as_ref().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next_as_ref().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next_as_ref().is_none());
    }

    // MARK: pop_front

    #[test]
    fn pop_front_from_empty() {
        let mut list = SkipList::<i32>::new();
        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn pop_front_single_element() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(42);
        assert_eq!(list.pop_front(), Some(42));
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(list.head_ref().next_as_ref().is_none());
        // Second pop on now-empty list
        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn pop_front_returns_in_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.pop_front(), Some(2));
        assert_eq!(list.pop_front(), Some(3));
        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn pop_front_len_decrements() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.push_back(i);
        }
        for remaining in (0..50_usize).rev() {
            list.pop_front();
            assert_eq!(list.len(), remaining);
        }
        assert_eq!(list.pop_front(), None);
    }

    /// With `with_capacity(1)` the generator always assigns height = 0.
    /// All data nodes have no skip links; head.links[0] is always None.
    /// Popping from the front only updates the prev/next chain.
    ///
    /// ```text
    /// Initial:  head → n1(10) → n2(20) → n3(30)   [no skip links]
    /// After 1st pop_front (removes 10):
    ///   head → n2(20) → n3(30)
    /// After 2nd pop_front (removes 20):
    ///   head → n3(30)
    /// After 3rd pop_front (removes 30):
    ///   head  (empty)
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn pop_front_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);

        // No skip links at any point; head.links[0] is always None.
        assert!(list.head_ref().links()[0].is_none());

        // First pop_front: removes 10
        assert_eq!(list.pop_front(), Some(10));
        assert_eq!(list.len(), 2);
        assert!(list.head_ref().links()[0].is_none());
        assert_eq!(
            list.head_ref().next_as_ref().and_then(|n| n.value()),
            Some(&20)
        );

        // Second pop_front: removes 20
        assert_eq!(list.pop_front(), Some(20));
        assert_eq!(list.len(), 1);
        assert!(list.head_ref().links()[0].is_none());
        assert_eq!(
            list.head_ref().next_as_ref().and_then(|n| n.value()),
            Some(&30)
        );

        // Third pop_front: removes 30
        assert_eq!(list.pop_front(), Some(30));
        assert_eq!(list.len(), 0);
        assert!(list.head_ref().links()[0].is_none());
        assert!(list.head_ref().next_as_ref().is_none());
    }

    #[test]
    fn pop_front_interleaved_with_push() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(2); // [1, 2]
        assert_eq!(list.pop_front(), Some(1)); // [2]
        list.push_front(0); // [0, 2]
        list.push_back(3); // [0, 2, 3]
        assert_eq!(list.pop_front(), Some(0)); // [2, 3]
        assert_eq!(list.pop_front(), Some(2)); // [3]
        assert_eq!(list.pop_front(), Some(3)); // []
        assert_eq!(list.pop_front(), None);
        assert_eq!(list.len(), 0);
    }

    // MARK: pop_back

    #[test]
    fn pop_back_from_empty() {
        let mut list = SkipList::<i32>::new();
        assert_eq!(list.pop_back(), None);
    }

    #[test]
    fn pop_back_single_element() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(42);
        assert_eq!(list.pop_back(), Some(42));
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(list.head_ref().next_as_ref().is_none());
        // Second pop on now-empty list
        assert_eq!(list.pop_back(), None);
    }

    #[test]
    fn pop_back_returns_in_reverse_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.pop_back(), Some(3));
        assert_eq!(list.pop_back(), Some(2));
        assert_eq!(list.pop_back(), Some(1));
        assert_eq!(list.pop_back(), None);
    }

    #[test]
    fn pop_back_len_decrements() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.push_back(i);
        }
        for remaining in (0..50_usize).rev() {
            list.pop_back();
            assert_eq!(list.len(), remaining);
        }
        assert_eq!(list.pop_back(), None);
    }

    /// With `with_capacity(1)` the generator always assigns height = 0.
    /// All data nodes have no skip links; head.links[0] is always None.
    /// Popping from the back only updates the prev/next chain and tail pointer.
    ///
    /// ```text
    /// Initial:  head → n1(10) → n2(20) → n3(30)   [no skip links]
    /// After 1st pop_back (removes 30):
    ///   head → n1(10) → n2(20)
    /// After 2nd pop_back (removes 20):
    ///   head → n1(10)
    /// After 3rd pop_back (removes 10):
    ///   head  (empty)
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn pop_back_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);

        // No skip links at any point; head.links[0] is always None.
        assert!(list.head_ref().links()[0].is_none());

        // First pop_back: removes 30; tail becomes node(20).
        assert_eq!(list.pop_back(), Some(30));
        assert_eq!(list.len(), 2);
        assert!(list.head_ref().links()[0].is_none());
        assert_eq!(list.back(), Some(&20));

        // Second pop_back: removes 20; tail becomes node(10).
        assert_eq!(list.pop_back(), Some(20));
        assert_eq!(list.len(), 1);
        assert!(list.head_ref().links()[0].is_none());
        assert_eq!(list.back(), Some(&10));

        // Third pop_back: removes 10; list is now empty.
        assert_eq!(list.pop_back(), Some(10));
        assert_eq!(list.len(), 0);
        assert!(list.head_ref().links()[0].is_none());
        assert!(list.head_ref().next_as_ref().is_none());
    }

    #[test]
    fn pop_back_interleaved_with_push() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(2); // [1, 2]
        assert_eq!(list.pop_back(), Some(2)); // [1]
        list.push_front(0); // [0, 1]
        list.push_back(3); // [0, 1, 3]
        assert_eq!(list.pop_back(), Some(3)); // [0, 1]
        assert_eq!(list.pop_back(), Some(1)); // [0]
        assert_eq!(list.pop_back(), Some(0)); // []
        assert_eq!(list.pop_back(), None);
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn pop_back_and_pop_front_together() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=6 {
            list.push_back(i); // [1, 2, 3, 4, 5, 6]
        }
        assert_eq!(list.pop_back(), Some(6)); // [1, 2, 3, 4, 5]
        assert_eq!(list.pop_front(), Some(1)); // [2, 3, 4, 5]
        assert_eq!(list.pop_back(), Some(5)); // [2, 3, 4]
        assert_eq!(list.pop_front(), Some(2)); // [3, 4]
        assert_eq!(list.pop_back(), Some(4)); // [3]
        assert_eq!(list.pop_front(), Some(3)); // []
        assert_eq!(list.pop_back(), None);
        assert_eq!(list.pop_front(), None);
        assert_eq!(list.len(), 0);
    }
}
