//! List-restructuring methods for [`SkipList`](super::SkipList):
//! `clear`, `truncate`, `split_off`, and `append`.

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

impl<T, G: LevelGenerator> SkipList<T, G> {
    /// Removes all elements from the list.
    ///
    /// The level generator is preserved; elements can be inserted again
    /// immediately after calling `clear`.
    ///
    /// This operation is `$O(n)$`: all n elements must be dropped.
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
    /// list.clear();
    /// assert!(list.is_empty());
    /// assert_eq!(list.len(), 0);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        let max_levels = self.head.level();
        // Replacing `*self.head` with a fresh sentinel node drops the old
        // sentinel in-place.  `Node::drop` iterates the entire `next` chain
        // and frees each node one at a time, so this is O(n) and
        // non-recursive regardless of list length.
        *self.head = Node::new(max_levels);
        self.tail = None;
        self.len = 0;
    }

    /// Shortens the list, keeping only the first `len` elements and dropping
    /// the rest.
    ///
    /// If `len >= self.len()`, this is a no-op.
    ///
    /// This operation is `$O(\log n + k)$` where k = `self.len() - len` is the
    /// number of elements removed: `$O(\log n)$` to locate the new tail and update
    /// the skip links, then `$O(k)$` to drop k values.
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
    /// list.truncate(3);
    /// assert_eq!(list.len(), 3);
    /// assert_eq!(list.get(0), Some(&1));
    /// assert_eq!(list.get(1), Some(&2));
    /// assert_eq!(list.get(2), Some(&3));
    /// assert_eq!(list.get(3), None);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "the node at rank `len` exists because 0 < len < self.len was checked before \
                  the traversal; the expect fires only on invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        clippy::needless_range_loop,
        reason = "l < max_levels = head.links.len(); every node in precursors[] was reached \
                  via a level-l link so its links.len() > l; all accesses are in bounds; \
                  l is used for both precursors[l] and links_mut()[l] so a plain index loop is \
                  the clearest expression"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link clearing and truncate_next touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        if len == 0 {
            self.clear();
            return;
        }

        // 0 < len < self.len: keep elements at ranks 1..=len; drop the rest.
        let max_levels = self.head.level();

        // IndexMutVisitor with target = len records, for each level l, the last
        // node at level l with rank < len.  precursors[0].links[0] points to the
        // new tail at rank len.  into_parts() releases the &mut borrow.
        let (_, precursors, _) = {
            let mut visitor = IndexMutVisitor::new(&mut self.head, len);
            visitor.traverse();
            visitor.into_parts()
        };

        // SAFETY: All raw pointers come from NonNull<Node<T>> captured during
        // traversal.  They originate from heap allocations owned by this SkipList.
        // No safe references to any node exist while this block runs.
        let new_tail_ptr: *mut Node<T> = unsafe {
            // The new tail is the level-0 successor of precursors[0].
            // It exists because 0 < len < self.len.
            let new_tail_ptr: *mut Node<T> = NonNull::from(
                (*precursors[0].as_ptr()).links()[0]
                    .as_ref()
                    .expect("the node at rank `len` exists because len < self.len")
                    .node(),
            )
            .as_ptr();

            let new_tail_height = (*new_tail_ptr).level();

            // Clear the new tail's own forward skip links: they point to
            // nodes that are about to be freed.
            for link in (*new_tail_ptr).links_mut() {
                *link = None;
            }

            // For levels at or above the new tail's height, the predecessor at
            // each such level may have a skip link that spans past the cut;
            // clear it.
            for l in new_tail_height..max_levels {
                (*precursors[l].as_ptr()).links_mut()[l] = None;
            }

            (*new_tail_ptr).truncate_next();

            new_tail_ptr
        };

        // SAFETY: new_tail_ptr is a live, heap-allocated node owned by this
        // SkipList; it will not be freed until the list itself is dropped.
        self.tail = Some(unsafe { NonNull::new_unchecked(new_tail_ptr) });
        self.len = len;
    }

    /// Splits the list at the given index, returning a new list containing
    /// all elements from index `at` onward.
    ///
    /// After the call, `self` contains elements at indices `[0, at)` and the
    /// returned list contains elements previously at indices `[at, len)`,
    /// renumbered from 0 in the returned list.
    ///
    /// Navigation to the split point is `$O(\log n)$`. Both halves are left in a
    /// consistent state after the call.
    ///
    /// # Panics
    ///
    /// Panics if `at > self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut a = SkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     a.push_back(i);
    /// }
    ///
    /// let b = a.split_off(3);
    /// let a_vals: Vec<i32> = a.iter().copied().collect();
    /// let b_vals: Vec<i32> = b.iter().copied().collect();
    /// assert_eq!(a_vals, [1, 2, 3]);
    /// assert_eq!(b_vals, [4, 5]);
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "`take_next_chain` returns None only if there is no successor, \
                  which cannot happen when at < self.len (validated above); \
                  Link::new(dist) succeeds because dist >= 1 by construction"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "take_next_chain, set_head_next, NonNull::new_unchecked, and \
                  rebuild_skip_links all touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer \
                  variables"
    )]
    #[inline]
    #[must_use]
    pub fn split_off(&mut self, at: usize) -> Self
    where
        G: Clone,
    {
        assert!(
            at <= self.len,
            "split_off index {at} is out of bounds (len = {})",
            self.len
        );

        let tail_len = self.len.saturating_sub(at);
        let max_levels = self.head.level();

        // Edge case: nothing to split off, return an empty list.
        if tail_len == 0 {
            return Self {
                head: Box::new(Node::new(max_levels)),
                tail: None,
                len: 0,
                generator: self.generator.clone(),
            };
        }

        // Edge case: split at position 0, transfer everything to the result.
        if at == 0 {
            let old_len = self.len;
            let mut result = Self {
                head: Box::new(Node::new(max_levels)),
                tail: None, // set by rebuild below
                len: old_len,
                generator: self.generator.clone(),
            };

            // Transfer the entire node chain from self.head to result.head.
            // SAFETY: We hold &mut self; no other references to any node
            // exist.  take_next_chain detaches cleanly.  set_head_next
            // wires the first node to result.head.
            unsafe {
                let head_ptr: *mut Node<T> = &raw mut *self.head;
                if let Some(first_nn) = (*head_ptr).take_next_chain() {
                    result.head.set_head_next(first_nn);
                }
            }

            // Clear self.head's skip links (now all-None).
            for link in self.head.links_mut() {
                *link = None;
            }
            self.tail = None;
            self.len = 0;

            // Rebuild result's skip links (result.head is new; data nodes'
            // inter-node links are stale for the new head).
            // SAFETY: result.head is exclusively owned; all nodes reachable
            // via result.head.next are live heap allocations.
            unsafe {
                result.tail = result.head.rebuild();
            }

            return result;
        }

        // General case: 0 < at < self.len.
        //
        // Navigate to the pivot (node[at - 1], the last node to keep in
        // self), detach the tail chain, wire it to a fresh head, then
        // rebuild skip links for both halves.
        //
        // SAFETY: at > 0 and at < self.len, so node_ptr_at(at - 1) returns
        // a valid data node.  We hold &mut self, so exclusive access is
        // guaranteed throughout.
        unsafe {
            let pivot: *mut Node<T> = self.node_ptr_at(at.saturating_sub(1)).as_ptr();

            // Detach nodes [at ..] from the pivot.  Guaranteed to succeed
            // because at < self.len means the pivot has at least one
            // successor.
            let first_of_tail = (*pivot)
                .take_next_chain()
                .expect("pivot has a successor because at < self.len");

            // Build the returned list.
            let mut result = Self {
                head: Box::new(Node::new(max_levels)),
                tail: None, // set by rebuild below
                len: tail_len,
                generator: self.generator.clone(),
            };
            result.head.set_head_next(first_of_tail);

            self.tail = Some(NonNull::new_unchecked(pivot));
            self.len = at;

            // Rebuild skip links for self (nodes 0 .. at).
            self.tail = self.head.rebuild();

            // Rebuild skip links for result (nodes at .. original_len).
            result.tail = result.head.rebuild();

            result
        }
    }

    /// Moves all elements from `other` to the end of `self`, leaving `other` empty.
    ///
    /// Elements are appended in the order they appear in `other`.
    ///
    /// When both lists share the same maximum level count (which is always the
    /// case for lists created with [`new`](SkipList::new) or
    /// [`with_capacity`](SkipList::with_capacity)), the operation runs in
    /// `$O(\log n)$` time. When the maximum level counts differ, the operation
    /// falls back to `$O(k \log(n+k))$` via repeated
    /// [`push_back`](SkipList::push_back) calls.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut a = SkipList::<i32>::new();
    /// a.push_back(1);
    /// a.push_back(2);
    ///
    /// let mut b = SkipList::<i32>::new();
    /// b.push_back(3);
    /// b.push_back(4);
    ///
    /// a.append(&mut b);
    /// assert!(b.is_empty());
    /// assert_eq!(a.iter().copied().collect::<Vec<_>>(), [1, 2, 3, 4]);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "take_next_chain returns Some because other.is_empty() was checked above; \
                  Link::new distance = self_len + other_link.distance - pred_rank >= 1 always \
                  because other_link.distance >= 1 (NonZeroUsize) and pred_rank <= self_len"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < max_levels = precursors.len() = head.links.len() = other_head.links.len(); \
                  l indexes both precursors[l] and links[][l] simultaneously"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "chain detach, prev/next splice, and cross-seam link wiring all touch \
                  provably disjoint heap nodes; splitting across blocks would require \
                  unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        if other.is_empty() {
            return;
        }

        let self_len = self.len;
        let other_len = other.len;
        let max_levels = self.head.level();

        // Slow path: different level-generator sizes, fall back to re-insertion
        // to avoid out-of-bounds indexing in the fast-path link wiring below.
        if max_levels != other.head.level() {
            while let Some(val) = other.pop_front() {
                self.push_back(val);
            }
            return;
        }

        // Fast path: splice chains and rewrite only the cross-seam skip links.
        //
        // IndexMutVisitor with target = self_len + 1 advances to the end of self,
        // recording the rightmost predecessor at each level.
        // into_parts() releases the &mut borrow on self.head.
        let (_, precursors, precursor_distances) = {
            let mut visitor = IndexMutVisitor::new(&mut self.head, self_len.saturating_add(1));
            visitor.traverse();
            visitor.into_parts()
        };

        // SAFETY: We hold `&mut self` and `&mut other`, giving exclusive access to
        // every node in both lists.  `other_head_ptr` is derived from `other`'s
        // disjoint `Box<Node<T>>` allocation.  Within the block each node field is
        // accessed at most once, so no simultaneous aliasing occurs.
        unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;
            let other_head_ptr: *mut Node<T> = &raw mut *other.head;

            // Step 1: detach other's node chain from other's sentinel head.
            let first_nn = (*other_head_ptr)
                .take_next_chain()
                .expect("other is non-empty");

            // Step 2: splice other's chain onto the end of self's chain.
            // If self is non-empty, attach after self.tail; otherwise attach
            // directly after self's sentinel head.
            if let Some(tail_nn) = self.tail {
                (*tail_nn.as_ptr()).set_head_next(first_nn);
            } else {
                (*head_ptr).set_head_next(first_nn);
            }

            // Step 3: wire the cross-seam skip links.
            //
            // For each level l, if other's head had a link at level l pointing
            // into other's chain, compute a new distance for the corresponding
            // predecessor in self:
            //
            //   distance = (self_len + other_link.distance) - pred_rank
            //
            // distance >= 1 because other_link.distance >= 1 (NonZeroUsize) and
            // pred_rank <= self_len.
            for l in 0..max_levels {
                if let Some(other_link) = (*other_head_ptr).links()[l].as_ref() {
                    let pred_ptr = precursors[l].as_ptr();
                    let pred_rank = precursor_distances[l];
                    let distance = self_len
                        .saturating_add(other_link.distance().get())
                        .saturating_sub(pred_rank);
                    (*pred_ptr).links_mut()[l] = Some(
                        Link::new(other_link.node(), distance)
                            .expect("distance >= 1 by construction"),
                    );
                }
            }

            let new_tail = other.tail;
            for link in (*other_head_ptr).links_mut() {
                *link = None;
            }
            other.tail = None;
            other.len = 0;

            self.len = self_len.saturating_add(other_len);
            self.tail = new_tail;
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::SkipList;

    // MARK: clear

    #[test]
    fn clear_empty_list() {
        let mut list = SkipList::<i32>::new();
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn clear_single_element() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn clear_multiple_elements() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=10 {
            list.push_back(i);
        }
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
        assert!(list.head.next().is_none());
    }

    #[test]
    fn clear_usable_after_clear() {
        // After clear, the list can accept new insertions.
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        list.clear();
        list.push_back(99);
        list.push_front(0);
        assert_eq!(list.len(), 2);
        assert_eq!(list.front(), Some(&0));
        assert_eq!(list.back(), Some(&99));
    }

    #[test]
    fn clear_then_clear_again() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.clear();
        list.clear(); // second clear on already-empty list
        assert!(list.is_empty());
    }

    #[test]
    fn clear_large_list() {
        // Large list to exercise the iterative Drop path.
        let n = 1_000_usize;
        let mut list = SkipList::<usize>::new();
        for i in 0..n {
            list.push_back(i);
        }
        list.clear();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
    }

    // MARK: truncate

    #[test]
    fn truncate_noop_when_len_equals_current() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.truncate(3);
        assert_eq!(list.len(), 3);
        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn truncate_noop_when_len_greater() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.truncate(5);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn truncate_to_zero_clears_list() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.truncate(0);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn truncate_to_one() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.truncate(1);
        assert_eq!(list.len(), 1);
        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.back(), Some(&1));
        assert!(list.head.next().and_then(|n| n.next()).is_none());
    }

    #[test]
    fn truncate_keeps_correct_elements() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=5 {
            list.push_back(i); // [1, 2, 3, 4, 5]
        }
        list.truncate(3); // [1, 2, 3]
        assert_eq!(list.len(), 3);
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&3));
        assert_eq!(list.get(3), None);
    }

    #[test]
    fn truncate_back_pointer_updated() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        list.truncate(3);
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn truncate_front_unchanged() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        list.truncate(3);
        assert_eq!(list.front(), Some(&1));
    }

    #[test]
    fn truncate_empty_list() {
        let mut list = SkipList::<i32>::new();
        list.truncate(0);
        assert!(list.is_empty());
        list.truncate(5);
        assert!(list.is_empty());
    }

    #[test]
    fn truncate_usable_after_truncate() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=5 {
            list.push_back(i);
        }
        list.truncate(2); // [1, 2]
        list.push_back(99); // [1, 2, 99]
        assert_eq!(list.len(), 3);
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&99));
        assert_eq!(list.back(), Some(&99));
    }

    #[test]
    fn truncate_then_truncate_more() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=10 {
            list.push_back(i);
        }
        list.truncate(7); // [1..=7]
        list.truncate(4); // [1..=4]
        assert_eq!(list.len(), 4);
        assert_eq!(list.back(), Some(&4));
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&3));
        assert_eq!(list.get(3), Some(&4));
    }

    #[test]
    fn truncate_large_list() {
        const N: usize = 1_000;
        const HALF: usize = 500;
        let mut list = SkipList::<usize>::new();
        for i in 0..N {
            list.push_back(i);
        }
        list.truncate(HALF); // keep first 500
        assert_eq!(list.len(), HALF);
        for i in 0..HALF {
            assert_eq!(list.get(i), Some(&i));
        }
        assert_eq!(list.back(), Some(&(HALF - 1)));
    }

    // MARK: split_off

    #[test]
    fn split_off_empty_list() {
        let mut a = SkipList::<i32>::new();
        let b = a.split_off(0);
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn split_off_at_end_returns_empty() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=5 {
            a.push_back(i);
        }
        let b = a.split_off(5);
        assert_eq!(a.len(), 5);
        assert!(b.is_empty());
        let a_vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(a_vals, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn split_off_at_zero_transfers_all() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=5 {
            a.push_back(i);
        }
        let b = a.split_off(0);
        assert!(a.is_empty());
        assert_eq!(b.len(), 5);
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(b_vals, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn split_off_middle() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=5 {
            a.push_back(i);
        }
        let b = a.split_off(3);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [1, 2, 3]);
        assert_eq!(b_vals, [4, 5]);
    }

    #[test]
    fn split_off_len_correct() {
        let mut a = SkipList::<i32>::new();
        for i in 0..10 {
            a.push_back(i);
        }
        let b = a.split_off(4);
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 6);
    }

    #[test]
    fn split_off_front_back_correct() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=6 {
            a.push_back(i);
        }
        let b = a.split_off(3);
        assert_eq!(a.front(), Some(&1));
        assert_eq!(a.back(), Some(&3));
        assert_eq!(b.front(), Some(&4));
        assert_eq!(b.back(), Some(&6));
    }

    #[test]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "Numbers are small in test, and therefore not affected"
    )]
    fn split_off_get_works_after() {
        let mut a = SkipList::<i32>::new();
        for i in 0..10 {
            a.push_back(i);
        }
        let b = a.split_off(5);
        for i in 0..5 {
            assert_eq!(a.get(i), Some(&(i as i32)));
        }
        for i in 0..5 {
            assert_eq!(b.get(i), Some(&((i + 5) as i32)));
        }
    }

    #[test]
    fn split_off_single_element_at_zero() {
        let mut a = SkipList::<i32>::new();
        a.push_back(42);
        let b = a.split_off(0);
        assert!(a.is_empty());
        assert_eq!(b.len(), 1);
        assert_eq!(b.get(0), Some(&42));
    }

    #[test]
    fn split_off_single_element_at_one() {
        let mut a = SkipList::<i32>::new();
        a.push_back(42);
        let b = a.split_off(1);
        assert_eq!(a.len(), 1);
        assert_eq!(a.get(0), Some(&42));
        assert!(b.is_empty());
    }

    #[test]
    fn split_off_then_push() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=4 {
            a.push_back(i);
        }
        let mut b = a.split_off(2);
        a.push_back(99);
        b.push_back(100);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [1, 2, 99]);
        assert_eq!(b_vals, [3, 4, 100]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn split_off_out_of_bounds_panics() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=3 {
            a.push_back(i);
        }
        drop(a.split_off(4));
    }

    #[test]
    fn split_off_large_list() {
        let n: usize = 200;
        let mut a = SkipList::<usize>::new();
        for i in 0..n {
            a.push_back(i);
        }
        #[expect(
            clippy::integer_division,
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of splitting at one-third of the list length"
        )]
        let at = n / 3;
        let b = a.split_off(at);
        assert_eq!(a.len(), at);
        assert_eq!(b.len(), n - at);
        // Verify every element via get() to exercise skip links.
        for i in 0..at {
            assert_eq!(a.get(i), Some(&i));
        }
        for i in 0..(n - at) {
            assert_eq!(b.get(i), Some(&(i + at)));
        }
    }

    // MARK: append

    #[test]
    fn append_both_empty() {
        let mut a = SkipList::<i32>::new();
        let mut b = SkipList::<i32>::new();
        a.append(&mut b);
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn append_other_empty() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=3 {
            a.push_back(i);
        }
        let mut b = SkipList::<i32>::new();
        a.append(&mut b);
        assert_eq!(a.len(), 3);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [1, 2, 3]);
    }

    #[test]
    fn append_self_empty() {
        let mut a = SkipList::<i32>::new();
        let mut b = SkipList::<i32>::new();
        for i in 1..=3 {
            b.push_back(i);
        }
        a.append(&mut b);
        assert_eq!(a.len(), 3);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [1, 2, 3]);
    }

    #[test]
    fn append_basic() {
        let mut a = SkipList::<i32>::new();
        a.push_back(1);
        a.push_back(2);
        let mut b = SkipList::<i32>::new();
        b.push_back(3);
        b.push_back(4);
        a.append(&mut b);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [1, 2, 3, 4]);
    }

    #[test]
    fn append_len_correct() {
        let mut a = SkipList::<i32>::new();
        for i in 0..5 {
            a.push_back(i);
        }
        let mut b = SkipList::<i32>::new();
        for i in 5..8 {
            b.push_back(i);
        }
        a.append(&mut b);
        assert_eq!(a.len(), 8);
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn append_front_back_correct() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=3 {
            a.push_back(i);
        }
        let mut b = SkipList::<i32>::new();
        for i in 4..=6 {
            b.push_back(i);
        }
        a.append(&mut b);
        assert_eq!(a.front(), Some(&1));
        assert_eq!(a.back(), Some(&6));
    }

    #[test]
    fn append_then_push() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=2 {
            a.push_back(i);
        }
        let mut b = SkipList::<i32>::new();
        for i in 3..=4 {
            b.push_back(i);
        }
        a.append(&mut b);
        a.push_back(99);
        b.push_back(100);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(a_vals, [1, 2, 3, 4, 99]);
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(b_vals, [100]);
    }

    #[test]
    fn append_get_works() {
        let n = 5_usize;
        let m = 5_usize;
        let mut a = SkipList::<usize>::new();
        for i in 0..n {
            a.push_back(i);
        }
        let mut b = SkipList::<usize>::new();
        for i in n..(n + m) {
            b.push_back(i);
        }
        a.append(&mut b);
        assert_eq!(a.len(), n + m);
        for i in 0..(n + m) {
            assert_eq!(a.get(i), Some(&i));
        }
    }

    #[test]
    fn append_large_list() {
        let n: usize = 200;
        let m: usize = 150;
        let mut a = SkipList::<usize>::new();
        for i in 0..n {
            a.push_back(i);
        }
        let mut b = SkipList::<usize>::new();
        for i in n..(n + m) {
            b.push_back(i);
        }
        a.append(&mut b);
        assert_eq!(a.len(), n + m);
        assert!(b.is_empty());
        // Verify every element via get() to exercise skip links.
        for i in 0..(n + m) {
            assert_eq!(a.get(i), Some(&i));
        }
        // Also verify via iter().
        let vals: Vec<usize> = a.iter().copied().collect();
        let expected: Vec<usize> = (0..(n + m)).collect();
        assert_eq!(vals, expected);
    }
}
