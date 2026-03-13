//! List-restructuring methods for [`OrderedSkipList`](super::OrderedSkipList):
//! `clear`, `append`, `split_off`, and `truncate`.

use core::{cmp::Ordering, ptr::NonNull};

use crate::{
    comparator::{Comparator, ComparatorKey},
    level_generator::LevelGenerator,
    node::{
        Node,
        visitor::{IndexMutVisitor, OrdMutVisitor, Visitor},
    },
    ordered_skip_list::OrderedSkipList,
};

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> OrderedSkipList<T, N, C, G> {
    /// Removes all elements from the list.
    ///
    /// The comparator and level generator are preserved; elements can be
    /// inserted again immediately after calling `clear`.
    ///
    /// This operation is `$O(n)$`: all n elements must be dropped.
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
    /// list.clear();
    /// assert!(list.is_empty());
    /// assert_eq!(list.len(), 0);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        let max_levels = self.head_ref().level();
        // Drop the old sentinel in-place.  `Node::drop` iterates the entire
        // `next` chain and frees each node one at a time, so this is O(n) and
        // non-recursive regardless of list length.  Then write a fresh
        // sentinel into the same allocation.
        //
        // SAFETY: `self.head` is a live, exclusively-owned allocation;
        // `drop_in_place` drops the old `Node<T, N>` (and its linked chain),
        // leaving the memory valid but uninitialized.
        unsafe { core::ptr::drop_in_place(self.head.as_ptr()) };
        // SAFETY: The allocation is still live after `drop_in_place`; `write`
        // initializes it with a fresh sentinel (no destructor runs on the
        // `write` side).
        unsafe { self.head.as_ptr().write(Node::new(max_levels)) };
        self.tail = None;
        self.len = 0;
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// Elements from `other` are merged into `self` at their sorted positions
    /// according to `self`'s comparator.  After the call `self` contains all
    /// elements from both lists in sorted order and `other` is empty.
    ///
    /// When the two sorted ranges are non-overlapping (every element of
    /// `other` is >= every element of `self`, or every element of `other`
    /// is <= every element of `self`), the raw node chains are spliced and
    /// skip links are rebuilt in one `$O(n+m)$` pass.  When the ranges
    /// overlap, each element of `other` is inserted individually in
    /// `$O(m \log(n+m))$` time.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut a = OrderedSkipList::<i32>::new();
    /// a.insert(1);
    /// a.insert(3);
    ///
    /// let mut b = OrderedSkipList::<i32>::new();
    /// b.insert(2);
    /// b.insert(4);
    ///
    /// a.append(&mut b);
    /// assert!(b.is_empty());
    /// let collected: Vec<i32> = a.iter().copied().collect();
    /// assert_eq!(collected, [1, 2, 3, 4]);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "tail/head nodes always have a value; expects fire only on invariant violations"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "take_next_chain, set_head_next, and filter_rebuild touch provably disjoint \
                  heap nodes; splitting across blocks would require unsafe-crossing raw-pointer \
                  variables"
    )]
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        if other.is_empty() {
            return;
        }

        // Forward fast path: self is empty, or every element of other is >=
        // every element of self (self.last <= other.first).  Splice other's
        // chain after self's tail and rebuild skip links in one O(n+m) pass.
        let can_concat = self.is_empty()
            || self.tail.is_some_and(|tail_nn| {
                // SAFETY: tail_nn is a live data node; other.head is a valid sentinel.
                let self_last: &T = unsafe { tail_nn.as_ref() }
                    .value()
                    .expect("tail node has a value");
                // SAFETY: `other.head` is the sentinel node allocated for
                // `other` and is valid for the entire lifetime of `other`;
                // `other` is a shared reference so no mutation can alias.
                let other_first: &T = unsafe { other.head.as_ref() }
                    .next_as_ref()
                    .and_then(|n| n.value())
                    .expect("other is non-empty so its first data node has a value");
                self.comparator.compare(self_last, other_first) != Ordering::Greater
            });

        if can_concat {
            // Detach other's entire node chain from its head sentinel.
            //
            // SAFETY: other.head is a valid head sentinel; we hold &mut other.
            let first_of_other = unsafe { (*other.head.as_ptr()).take_next_chain() };

            // Clear other.head's skip links: after take_next_chain they may
            // still point to nodes now belonging to self's chain.
            for link in other.head_mut().links_mut() {
                *link = None;
            }
            other.tail = None;
            other.len = 0;

            if let Some(first_nn) = first_of_other {
                // Attach other's chain after self's tail (or self's head when empty).
                let attach = self.tail.unwrap_or(self.head);
                // SAFETY: The attachment point (self.tail or self.head) has
                // next == None.  first_nn.prev was set to None by take_next_chain.
                unsafe { (*attach.as_ptr()).set_head_next(first_nn) };

                // Rebuild all skip links in one O(n+m) pass.
                //
                // SAFETY: self.head is exclusively owned; all reachable nodes
                // are live heap allocations with no other live references.
                let (new_len, new_tail) =
                    unsafe { Node::filter_rebuild(self.head, |_| true, |_| {}) };
                self.tail = new_tail;
                self.len = new_len;
            }
        } else {
            // Reverse fast path: every element of other is <= every element of
            // self (other.last <= self.first).  Prepend other's chain before
            // self's chain and rebuild skip links in one O(n+m) pass.
            //
            // self is non-empty here because can_concat handles self.is_empty().
            let other_tail_saved = other.tail;
            let can_prepend = other_tail_saved.is_some_and(|other_tail_nn| {
                // SAFETY: other_tail_nn is a live data node owned by other.
                let other_last: &T = unsafe { other_tail_nn.as_ref() }
                    .value()
                    .expect("other tail node has a value");
                // SAFETY: self.head is a valid sentinel; next_as_ref returns a
                // short-lived shared reference that is not retained.
                let self_first: &T = unsafe { self.head.as_ref() }
                    .next_as_ref()
                    .and_then(|n| n.value())
                    .expect("self is non-empty so its first data node has a value");
                self.comparator.compare(other_last, self_first) != Ordering::Greater
            });

            if can_prepend {
                let other_tail_nn = other_tail_saved.expect("other is non-empty in this branch");

                // Detach other's chain and clear its sentinel's skip links.
                //
                // SAFETY: other.head is a valid head sentinel; we hold &mut other.
                let first_of_other = unsafe { (*other.head.as_ptr()).take_next_chain() };
                for link in other.head_mut().links_mut() {
                    *link = None;
                }
                other.tail = None;
                other.len = 0;

                // Detach self's existing chain so we can reattach it after
                // other's chain.
                //
                // SAFETY: self.head is a valid head sentinel; we hold &mut self.
                let first_of_self = unsafe { (*self.head.as_ptr()).take_next_chain() };

                if let Some(first_other_nn) = first_of_other {
                    // Wire: self.head -> other's chain -> self's old chain.
                    //
                    // SAFETY: self.head.next is None after take_next_chain.
                    // first_other_nn.prev was cleared by take_next_chain.
                    unsafe { (*self.head.as_ptr()).set_head_next(first_other_nn) };

                    if let Some(first_self_nn) = first_of_self {
                        // SAFETY: other_tail_nn.next is None (it was the tail
                        // of other's chain).  first_self_nn.prev was cleared
                        // by take_next_chain above.
                        unsafe { (*other_tail_nn.as_ptr()).set_head_next(first_self_nn) };
                    }

                    // Rebuild all skip links in one O(n+m) pass.
                    //
                    // SAFETY: self.head is exclusively owned; all reachable
                    // nodes are live heap allocations with no other live
                    // references.
                    let (new_len, new_tail) =
                        unsafe { Node::filter_rebuild(self.head, |_| true, |_| {}) };
                    self.tail = new_tail;
                    self.len = new_len;
                }
            } else {
                while let Some(v) = other.pop_first() {
                    self.insert(v);
                }
            }
        }
    }

    /// Splits the list at the given value, returning a new list containing
    /// all elements with value `>= value`.
    ///
    /// After the call, `self` retains all elements with value `< value`, and
    /// the returned list contains all elements with value `>= value`.  When
    /// duplicates of `value` are present they all move to the returned list.
    ///
    /// This operation runs in `$O(n)$` time.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut a = OrderedSkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     a.insert(i);
    /// }
    ///
    /// let b = a.split_off(&3);
    /// let a_vals: Vec<i32> = a.iter().copied().collect();
    /// let b_vals: Vec<i32> = b.iter().copied().collect();
    /// assert_eq!(a_vals, [1, 2]);
    /// assert_eq!(b_vals, [3, 4, 5]);
    /// ```
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "set_head_next is an unsafe fn called on a raw-pointer dereference; \
                  the two operations are on the same freshly allocated, exclusively-owned \
                  node and are provably disjoint from the take_next_chain and filter_rebuild \
                  calls above and below"
    )]
    #[inline]
    #[must_use]
    pub fn split_off<Q>(&mut self, value: &Q) -> Self
    where
        Q: ?Sized,
        C: Clone + ComparatorKey<T, Q>,
        G: Clone,
    {
        let max_levels = self.head_ref().level();

        // Use OrdMutVisitor to find the last node strictly < value.
        // If all nodes are >= value, the visitor leaves precursors[0] = head.
        //
        // `self.head` is `NonNull` (a `Copy` type) so copying it does not
        // borrow `self`.  The closure only borrows `self.comparator` (shared),
        // which is a distinct field.  Both borrows are released when `visitor`
        // is consumed by `into_parts()`.
        let pivot_nn = {
            let head = self.head;
            let cmp = |v: &T, q: &Q| self.comparator.compare_key(v, q);
            let mut visitor = OrdMutVisitor::new(head, value, cmp);
            visitor.traverse();
            let (current, found, _precursors) = visitor.into_parts();
            // `found=true`:  current = first node with value == split_value;
            //                pivot = the node just before it (prev()).
            // `found=false`: current = last node with value < split_value
            //                (or head if all values >= split_value);
            //                pivot = current.
            if found {
                // SAFETY: current is a live data node; prev() gives the
                // predecessor node (always Some for a non-head data node).
                unsafe { current.as_ref() }.prev().unwrap_or(head)
            } else {
                current
            }
        };

        // SAFETY: `Box::into_raw` transfers ownership; freed in `Drop`.
        let result_head =
            unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(max_levels)))) };
        let mut result = Self {
            head: result_head,
            tail: None,
            len: 0,
            comparator: self.comparator.clone(),
            generator: self.generator.clone(),
        };

        // Detach the "≥ value" suffix from pivot_nn.
        //
        // SAFETY: `pivot_nn` is a valid, live node in this list (either the
        // head sentinel or a data node) for the duration of `&mut self`.  No
        // other live `&mut` references to any node exist.
        let first_nn = unsafe { (*pivot_nn.as_ptr()).take_next_chain() };

        if let Some(nn) = first_nn {
            // SAFETY: `result.head` is a freshly allocated, exclusively-owned
            // sentinel node.  `nn` is the first node of the detached chain
            // with no other live references.
            unsafe { (*result.head.as_ptr()).set_head_next(nn) };
        }

        // Rebuild skip links for both halves and compute their element counts.
        //
        // `filter_rebuild` clears all forward skip links from the head, walks
        // the sequential chain in O(n), re-wires skip links, and returns
        // `(element_count, tail)`.
        //
        // SAFETY: `self.head` is exclusively owned; every node reachable from
        // it is a live heap allocation with no other live references.
        let (self_len, self_tail) = unsafe { Node::filter_rebuild(self.head, |_| true, |_| {}) };
        // SAFETY: `result.head` is exclusively owned; every node reachable
        // from it is a live heap allocation with no other live references.
        let (result_len, result_tail) =
            unsafe { Node::filter_rebuild(result.head, |_| true, |_| {}) };

        self.tail = self_tail;
        self.len = self_len;
        result.tail = result_tail;
        result.len = result_len;

        result
    }

    /// Splits the list at the given 0-based index, returning a new list
    /// containing all elements from position `at` onwards.
    ///
    /// After the call, `self` retains elements at positions `0..at`, and the
    /// returned list contains elements at positions `at..`.  This operation
    /// runs in `$O(n)$` time.
    ///
    /// # Panics
    ///
    /// Panics if `at > self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut a = OrderedSkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     a.insert(i);
    /// }
    ///
    /// let b = a.split_off_index(2);
    /// let a_vals: Vec<i32> = a.iter().copied().collect();
    /// let b_vals: Vec<i32> = b.iter().copied().collect();
    /// assert_eq!(a_vals, [1, 2]);
    /// assert_eq!(b_vals, [3, 4, 5]);
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "the pivot node exists because 0 < at < self.len was checked before the \
                  traversal; the expect fires only on invariant violations"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "set_head_next is an unsafe fn called on a raw-pointer dereference; \
                  the two operations are on the same freshly allocated, exclusively-owned \
                  node and are provably disjoint from the take_next_chain and rebuild \
                  calls above and below"
    )]
    #[inline]
    #[must_use]
    pub fn split_off_index(&mut self, at: usize) -> Self
    where
        C: Clone,
        G: Clone,
    {
        assert!(
            at <= self.len,
            "split_off_index index {at} is out of bounds (len = {})",
            self.len
        );

        let tail_len = self.len.saturating_sub(at);
        let max_levels = self.head_ref().level();

        // Edge case: nothing to split off.
        if tail_len == 0 {
            // SAFETY: `Box::into_raw` transfers ownership; freed in `Drop`.
            let head =
                unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(max_levels)))) };
            return Self {
                head,
                tail: None,
                len: 0,
                comparator: self.comparator.clone(),
                generator: self.generator.clone(),
            };
        }

        // Edge case: split at position 0, transfer everything.
        if at == 0 {
            let old_len = self.len;
            // SAFETY: `Box::into_raw` transfers ownership; freed in `Drop`.
            let result_head =
                unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(max_levels)))) };
            let mut result = Self {
                head: result_head,
                tail: None, // set by rebuild below
                len: old_len,
                comparator: self.comparator.clone(),
                generator: self.generator.clone(),
            };

            // SAFETY: We hold &mut self; no other references to any node exist.
            unsafe {
                let head_ptr: *mut Node<T, N> = self.head.as_ptr();
                if let Some(first_nn) = (*head_ptr).take_next_chain() {
                    (*result.head.as_ptr()).set_head_next(first_nn);
                }
            }

            for link in self.head_mut().links_mut() {
                *link = None;
            }
            self.tail = None;
            self.len = 0;

            // Rebuild result's skip links.
            // SAFETY: result.head is exclusively owned; all reachable nodes are live.
            unsafe {
                result.tail = Node::rebuild(result.head);
            }

            return result;
        }

        // General case: 0 < at < self.len.
        //
        // Navigate to the pivot (the last node to keep in self, at 0-based
        // index `at - 1`, internal rank `at`), detach the tail chain, wire it
        // to a fresh head, then rebuild skip links for both halves.
        //
        // head = internal rank 0, first data node = rank 1, so 0-based index
        // i corresponds to internal rank i + 1.  The last kept node is at
        // 0-based index `at - 1` = internal rank `at`.
        // `IndexMutVisitor::new(head, at).traverse()` returns the node at
        // internal rank `at` as its `current`.
        let (pivot_nn, _, _) = {
            let mut visitor = IndexMutVisitor::new(self.head, at);
            visitor.traverse();
            visitor.into_parts()
        };

        // SAFETY: at > 0 and at < self.len guarantee the pivot exists.
        // We hold &mut self so exclusive access is guaranteed.
        unsafe {
            let pivot: *mut Node<T, N> = pivot_nn.as_ptr();

            let first_of_tail = (*pivot)
                .take_next_chain()
                .expect("pivot has a successor because at < self.len");

            // SAFETY: `Box::into_raw` transfers ownership; freed in `Drop`.
            let result_head =
                NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(max_levels))));
            let mut result = Self {
                head: result_head,
                tail: None, // set by rebuild below
                len: tail_len,
                comparator: self.comparator.clone(),
                generator: self.generator.clone(),
            };
            (*result.head.as_ptr()).set_head_next(first_of_tail);

            self.len = at;

            self.tail = Node::rebuild(self.head);
            result.tail = Node::rebuild(result.head);

            result
        }
    }

    /// Shortens the list, keeping only the first `len` elements and dropping
    /// the rest.
    ///
    /// If `len >= self.len()`, this is a no-op.
    ///
    /// This operation is `$O(\log n + k)$` where k is the number of elements
    /// removed.
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
    /// list.truncate(3);
    /// assert_eq!(list.len(), 3);
    /// assert_eq!(list.get_by_index(0), Some(&1));
    /// assert_eq!(list.get_by_index(1), Some(&2));
    /// assert_eq!(list.get_by_index(2), Some(&3));
    /// assert_eq!(list.get_by_index(3), None);
    /// ```
    #[expect(
        clippy::indexing_slicing,
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

        let max_levels = self.head_ref().level();

        // IndexMutVisitor with target = len stops when current = the node at
        // rank len (the new tail).  into_parts() returns that node as `current`.
        let (new_tail_node, precursors, _) = {
            let mut visitor = IndexMutVisitor::new(self.head, len);
            visitor.traverse();
            visitor.into_parts()
        };

        // SAFETY: All raw pointers come from NonNull<Node<T, N>> captured during
        // traversal.  They originate from heap allocations owned by this
        // OrderedSkipList.  No safe references to any node exist while this block
        // runs.
        let new_tail_ptr: *mut Node<T, N> = unsafe {
            let new_tail_ptr: *mut Node<T, N> = new_tail_node.as_ptr();

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
        // OrderedSkipList; it will not be freed until the list itself is dropped.
        self.tail = Some(unsafe { NonNull::new_unchecked(new_tail_ptr) });
        self.len = len;
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::OrderedSkipList;
    use crate::comparator::FnComparator;

    // MARK: clear

    #[test]
    fn clear_empty_list() {
        let mut list = OrderedSkipList::<i32>::new();
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.first(), None);
        assert_eq!(list.last(), None);
    }

    #[test]
    fn clear_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.first(), None);
        assert_eq!(list.last(), None);
    }

    #[test]
    fn clear_multiple_elements() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=10 {
            list.insert(i);
        }
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.first(), None);
        assert_eq!(list.last(), None);
        assert!(list.head_ref().next_as_ref().is_none());
    }

    #[test]
    fn clear_usable_after_clear() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        list.clear();
        list.insert(99);
        list.insert(0);
        assert_eq!(list.len(), 2);
        assert_eq!(list.first(), Some(&0));
        assert_eq!(list.last(), Some(&99));
    }

    #[test]
    fn clear_then_clear_again() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.clear();
        list.clear();
        assert!(list.is_empty());
    }

    #[test]
    fn clear_large_list() {
        let n = 1_000_usize;
        let mut list = OrderedSkipList::<usize>::new();
        for i in 0..n {
            list.insert(i);
        }
        list.clear();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
    }

    // MARK: append

    #[test]
    fn append_both_empty() {
        let mut a = OrderedSkipList::<i32>::new();
        let mut b = OrderedSkipList::<i32>::new();
        a.append(&mut b);
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn append_other_empty() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 1..=3 {
            a.insert(i);
        }
        let mut b = OrderedSkipList::<i32>::new();
        a.append(&mut b);
        assert_eq!(a.len(), 3);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [1, 2, 3]);
    }

    #[test]
    fn append_self_empty() {
        let mut a = OrderedSkipList::<i32>::new();
        let mut b = OrderedSkipList::<i32>::new();
        for i in 1..=3 {
            b.insert(i);
        }
        a.append(&mut b);
        assert_eq!(a.len(), 3);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [1, 2, 3]);
    }

    #[test]
    fn append_non_overlapping() {
        let mut a = OrderedSkipList::<i32>::new();
        a.insert(1);
        a.insert(2);
        let mut b = OrderedSkipList::<i32>::new();
        b.insert(3);
        b.insert(4);
        a.append(&mut b);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [1, 2, 3, 4]);
    }

    #[test]
    fn append_interleaved() {
        let mut a = OrderedSkipList::<i32>::new();
        a.insert(1);
        a.insert(3);
        a.insert(5);
        let mut b = OrderedSkipList::<i32>::new();
        b.insert(2);
        b.insert(4);
        b.insert(6);
        a.append(&mut b);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn append_len_correct() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 0..5 {
            a.insert(i);
        }
        let mut b = OrderedSkipList::<i32>::new();
        for i in 5..8 {
            b.insert(i);
        }
        a.append(&mut b);
        assert_eq!(a.len(), 8);
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn append_sorted_order_preserved() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in [5, 3, 1] {
            a.insert(i);
        }
        let mut b = OrderedSkipList::<i32>::new();
        for i in [6, 4, 2] {
            b.insert(i);
        }
        a.append(&mut b);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn append_with_duplicates() {
        let mut a = OrderedSkipList::<i32>::new();
        a.insert(1);
        a.insert(2);
        let mut b = OrderedSkipList::<i32>::new();
        b.insert(2);
        b.insert(3);
        a.append(&mut b);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [1, 2, 2, 3]);
    }

    #[test]
    fn append_first_last_correct() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in [2, 4] {
            a.insert(i);
        }
        let mut b = OrderedSkipList::<i32>::new();
        for i in [1, 5] {
            b.insert(i);
        }
        a.append(&mut b);
        assert_eq!(a.first(), Some(&1));
        assert_eq!(a.last(), Some(&5));
    }

    #[test]
    fn append_custom_comparator() {
        // Largest-first ordering.  Use a fn pointer so both lists share the
        // same concrete comparator type.
        let f: fn(&i32, &i32) -> core::cmp::Ordering = |a, b| b.cmp(a);
        let mut a: OrderedSkipList<i32, 16, _> = OrderedSkipList::with_comparator(FnComparator(f));
        a.insert(3);
        a.insert(5);
        let mut b: OrderedSkipList<i32, 16, _> = OrderedSkipList::with_comparator(FnComparator(f));
        b.insert(1);
        b.insert(4);
        a.append(&mut b);
        assert!(b.is_empty());
        // Stored as [5, 4, 3, 1] (largest first).
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [5, 4, 3, 1]);
    }

    #[test]
    fn append_reverse_non_overlapping() {
        // Reverse fast path: every element of other is <= every element of self.
        let mut a = OrderedSkipList::<i32>::new();
        a.insert(3);
        a.insert(4);
        let mut b = OrderedSkipList::<i32>::new();
        b.insert(1);
        b.insert(2);
        a.append(&mut b);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [1, 2, 3, 4]);
    }

    #[test]
    fn append_reverse_equal_boundary() {
        // Reverse fast path also covers the equal-boundary case (other.last ==
        // self.first), since OrderedSkipList allows duplicates.
        let mut a = OrderedSkipList::<i32>::new();
        a.insert(2);
        a.insert(3);
        let mut b = OrderedSkipList::<i32>::new();
        b.insert(1);
        b.insert(2);
        a.append(&mut b);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(vals, [1, 2, 2, 3]);
    }

    #[test]
    fn append_reverse_first_last_correct() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in [3, 5] {
            a.insert(i);
        }
        let mut b = OrderedSkipList::<i32>::new();
        for i in [1, 2] {
            b.insert(i);
        }
        a.append(&mut b);
        assert_eq!(a.first(), Some(&1));
        assert_eq!(a.last(), Some(&5));
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn append_reverse_large() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 50..100 {
            a.insert(i);
        }
        let mut b = OrderedSkipList::<i32>::new();
        for i in 0..50 {
            b.insert(i);
        }
        a.append(&mut b);
        assert_eq!(a.len(), 100);
        assert!(b.is_empty());
        let vals: Vec<i32> = a.iter().copied().collect();
        let expected: Vec<i32> = (0..100).collect();
        assert_eq!(vals, expected);
    }

    // MARK: split_off

    #[test]
    fn split_off_empty_list() {
        let mut a = OrderedSkipList::<i32>::new();
        let b = a.split_off(&3);
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn split_off_value_after_all() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            a.insert(i);
        }
        let b = a.split_off(&10);
        assert_eq!(a.len(), 5);
        assert!(b.is_empty());
        let a_vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(a_vals, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn split_off_value_before_all() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 3..=5 {
            a.insert(i);
        }
        let b = a.split_off(&1);
        assert!(a.is_empty());
        assert_eq!(b.len(), 3);
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(b_vals, [3, 4, 5]);
    }

    #[test]
    fn split_off_middle() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            a.insert(i);
        }
        let b = a.split_off(&3);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [1, 2]);
        assert_eq!(b_vals, [3, 4, 5]);
    }

    #[test]
    fn split_off_len_correct() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 0..10 {
            a.insert(i);
        }
        let b = a.split_off(&4);
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 6);
    }

    #[test]
    fn split_off_first_last_correct() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 1..=6 {
            a.insert(i);
        }
        let b = a.split_off(&4);
        assert_eq!(a.first(), Some(&1));
        assert_eq!(a.last(), Some(&3));
        assert_eq!(b.first(), Some(&4));
        assert_eq!(b.last(), Some(&6));
    }

    #[test]
    fn split_off_at_value_not_in_list() {
        // Split at 2.5 → a keeps [1, 2], b gets [3, 4, 5].
        let mut a = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3, 4, 5] {
            a.insert(i);
        }
        let b = a.split_off(&3); // split at 3 (first element >= 3)
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [1, 2]);
        assert_eq!(b_vals, [3, 4, 5]);
    }

    #[test]
    fn split_off_duplicates_all_go_to_result() {
        let mut a = OrderedSkipList::<i32>::new();
        for _ in 0..3 {
            a.insert(2);
        }
        a.insert(1);
        a.insert(3);
        // list: [1, 2, 2, 2, 3]; split_off(&2) -> a=[1], b=[2,2,2,3]
        let b = a.split_off(&2);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [1]);
        assert_eq!(b_vals, [2, 2, 2, 3]);
    }

    #[test]
    fn split_off_single_element_goes_to_result() {
        let mut a = OrderedSkipList::<i32>::new();
        a.insert(42);
        let b = a.split_off(&42);
        assert!(a.is_empty());
        assert_eq!(b.len(), 1);
        assert_eq!(b.first(), Some(&42));
    }

    #[test]
    fn split_off_single_element_stays_in_self() {
        let mut a = OrderedSkipList::<i32>::new();
        a.insert(42);
        let b = a.split_off(&100);
        assert_eq!(a.len(), 1);
        assert_eq!(a.first(), Some(&42));
        assert!(b.is_empty());
    }

    #[test]
    fn split_off_usable_after_split() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 1..=4 {
            a.insert(i);
        }
        let mut b = a.split_off(&3);
        a.insert(99);
        b.insert(100);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [1, 2, 99]);
        assert_eq!(b_vals, [3, 4, 100]);
    }

    #[test]
    fn split_off_iter_consistent() {
        let n = 100_usize;
        let mut a = OrderedSkipList::<usize>::new();
        for i in 0..n {
            a.insert(i);
        }
        let b = a.split_off(&50);
        assert_eq!(a.len(), 50);
        assert_eq!(b.len(), 50);
        let a_vals: Vec<usize> = a.iter().copied().collect();
        let b_vals: Vec<usize> = b.iter().copied().collect();
        let expected_a: Vec<usize> = (0..50).collect();
        let expected_b: Vec<usize> = (50..100).collect();
        assert_eq!(a_vals, expected_a);
        assert_eq!(b_vals, expected_b);
    }

    #[test]
    fn split_off_custom_comparator() {
        // Largest-first ordering.  Use a fn pointer so that `FnComparator<F>`
        // satisfies `Clone` (fn pointers are `Copy`).
        //
        // With reverse ordering the list stores [5, 4, 3, 2, 1].
        // split_off(&3) keeps elements "< 3" in that ordering (i.e. 5, 4) in
        // self and moves elements ">= 3" (i.e. 3, 2, 1) to the result.
        let f: fn(&i32, &i32) -> core::cmp::Ordering = |a, b| b.cmp(a);
        let mut a: OrderedSkipList<i32, 16, _> = OrderedSkipList::with_comparator(FnComparator(f));
        for i in 1..=5 {
            a.insert(i); // stored as [5, 4, 3, 2, 1]
        }
        let b = a.split_off(&3);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [5, 4]);
        assert_eq!(b_vals, [3, 2, 1]);
    }

    // MARK: split_off_index

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn split_off_index_panics_on_out_of_bounds() {
        let mut a = OrderedSkipList::<i32>::new();
        a.insert(1);
        drop(a.split_off_index(2));
    }

    #[test]
    fn split_off_index_empty_list_at_zero() {
        let mut a = OrderedSkipList::<i32>::new();
        let b = a.split_off_index(0);
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn split_off_index_at_len_returns_empty() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            a.insert(i);
        }
        let b = a.split_off_index(5);
        assert_eq!(a.len(), 5);
        assert!(b.is_empty());
        let a_vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(a_vals, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn split_off_index_at_zero_transfers_all() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            a.insert(i);
        }
        let b = a.split_off_index(0);
        assert!(a.is_empty());
        assert_eq!(b.len(), 5);
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(b_vals, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn split_off_index_middle() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            a.insert(i);
        }
        let b = a.split_off_index(2);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [1, 2]);
        assert_eq!(b_vals, [3, 4, 5]);
    }

    #[test]
    fn split_off_index_at_one() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 1..=4 {
            a.insert(i);
        }
        let b = a.split_off_index(1);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [1]);
        assert_eq!(b_vals, [2, 3, 4]);
    }

    #[test]
    fn split_off_index_len_correct() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 0..10 {
            a.insert(i);
        }
        let b = a.split_off_index(4);
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 6);
    }

    #[test]
    fn split_off_index_first_last_correct() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 1..=6 {
            a.insert(i);
        }
        let b = a.split_off_index(3);
        assert_eq!(a.first(), Some(&1));
        assert_eq!(a.last(), Some(&3));
        assert_eq!(b.first(), Some(&4));
        assert_eq!(b.last(), Some(&6));
    }

    #[test]
    fn split_off_index_usable_after_split() {
        let mut a = OrderedSkipList::<i32>::new();
        for i in 1..=4 {
            a.insert(i);
        }
        let mut b = a.split_off_index(2);
        a.insert(99);
        b.insert(100);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [1, 2, 99]);
        assert_eq!(b_vals, [3, 4, 100]);
    }

    #[test]
    fn split_off_index_iter_consistent() {
        let n = 100_usize;
        let mut a = OrderedSkipList::<usize>::new();
        for i in 0..n {
            a.insert(i);
        }
        let b = a.split_off_index(50);
        assert_eq!(a.len(), 50);
        assert_eq!(b.len(), 50);
        let a_vals: Vec<usize> = a.iter().copied().collect();
        let b_vals: Vec<usize> = b.iter().copied().collect();
        let expected_a: Vec<usize> = (0..50).collect();
        let expected_b: Vec<usize> = (50..100).collect();
        assert_eq!(a_vals, expected_a);
        assert_eq!(b_vals, expected_b);
    }

    #[test]
    fn split_off_index_single_element_stays_in_self() {
        let mut a = OrderedSkipList::<i32>::new();
        a.insert(42);
        let b = a.split_off_index(1);
        assert_eq!(a.len(), 1);
        assert_eq!(a.first(), Some(&42));
        assert!(b.is_empty());
    }

    #[test]
    fn split_off_index_single_element_moves_to_result() {
        let mut a = OrderedSkipList::<i32>::new();
        a.insert(42);
        let b = a.split_off_index(0);
        assert!(a.is_empty());
        assert_eq!(b.len(), 1);
        assert_eq!(b.first(), Some(&42));
    }

    #[test]
    fn split_off_index_custom_comparator() {
        // Reverse ordering: stored [5, 4, 3, 2, 1].
        let f: fn(&i32, &i32) -> core::cmp::Ordering = |a, b| b.cmp(a);
        let mut a: OrderedSkipList<i32, 16, _> = OrderedSkipList::with_comparator(FnComparator(f));
        for i in 1..=5 {
            a.insert(i);
        }
        let b = a.split_off_index(2);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [5, 4]);
        assert_eq!(b_vals, [3, 2, 1]);
    }

    // MARK: truncate

    #[test]
    fn truncate_noop_when_len_equals_current() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=3 {
            list.insert(i);
        }
        list.truncate(3);
        assert_eq!(list.len(), 3);
        assert_eq!(list.first(), Some(&1));
        assert_eq!(list.last(), Some(&3));
    }

    #[test]
    fn truncate_noop_when_len_greater() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.truncate(5);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn truncate_to_zero_clears_list() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=3 {
            list.insert(i);
        }
        list.truncate(0);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.first(), None);
        assert_eq!(list.last(), None);
    }

    #[test]
    fn truncate_to_one() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=3 {
            list.insert(i);
        }
        list.truncate(1);
        assert_eq!(list.len(), 1);
        assert_eq!(list.first(), Some(&1));
        assert_eq!(list.last(), Some(&1));
        assert!(
            list.head_ref()
                .next_as_ref()
                .and_then(|n| n.next_as_ref())
                .is_none()
        );
    }

    #[test]
    fn truncate_keeps_correct_elements() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        list.truncate(3);
        assert_eq!(list.len(), 3);
        assert_eq!(list.get_by_index(0), Some(&1));
        assert_eq!(list.get_by_index(1), Some(&2));
        assert_eq!(list.get_by_index(2), Some(&3));
        assert_eq!(list.get_by_index(3), None);
    }

    #[test]
    fn truncate_tail_pointer_updated() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        list.truncate(3);
        assert_eq!(list.last(), Some(&3));
    }

    #[test]
    fn truncate_first_unchanged() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        list.truncate(3);
        assert_eq!(list.first(), Some(&1));
    }

    #[test]
    fn truncate_empty_list() {
        let mut list = OrderedSkipList::<i32>::new();
        list.truncate(0);
        assert!(list.is_empty());
        list.truncate(5);
        assert!(list.is_empty());
    }

    #[test]
    fn truncate_usable_after_truncate() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        list.truncate(2); // [1, 2]
        list.insert(99); // [1, 2, 99]
        assert_eq!(list.len(), 3);
        assert_eq!(list.get_by_index(0), Some(&1));
        assert_eq!(list.get_by_index(1), Some(&2));
        assert_eq!(list.get_by_index(2), Some(&99));
        assert_eq!(list.last(), Some(&99));
    }

    #[test]
    fn truncate_then_truncate_more() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=10 {
            list.insert(i);
        }
        list.truncate(7); // [1..=7]
        list.truncate(4); // [1..=4]
        assert_eq!(list.len(), 4);
        assert_eq!(list.last(), Some(&4));
        assert_eq!(list.get_by_index(0), Some(&1));
        assert_eq!(list.get_by_index(1), Some(&2));
        assert_eq!(list.get_by_index(2), Some(&3));
        assert_eq!(list.get_by_index(3), Some(&4));
    }

    #[test]
    fn truncate_large_list() {
        const N: usize = 1_000;
        const HALF: usize = 500;
        let mut list = OrderedSkipList::<usize>::new();
        for i in 0..N {
            list.insert(i);
        }
        list.truncate(HALF);
        assert_eq!(list.len(), HALF);
        for i in 0..HALF {
            assert_eq!(list.get_by_index(i), Some(&i));
        }
        assert_eq!(list.last(), Some(&(HALF - 1)));
    }

    #[test]
    fn truncate_with_duplicates() {
        let mut list = OrderedSkipList::<i32>::new();
        for _ in 0..3 {
            list.insert(1);
            list.insert(2);
        }
        // list: [1, 1, 1, 2, 2, 2]
        list.truncate(4); // keep [1, 1, 1, 2]
        assert_eq!(list.len(), 4);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 1, 1, 2]);
        assert_eq!(list.last(), Some(&2));
    }

    #[test]
    fn truncate_single_element_kept() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        list.truncate(1);
        assert_eq!(list.len(), 1);
        assert_eq!(list.first(), Some(&42));
        assert_eq!(list.last(), Some(&42));
    }

    #[test]
    fn truncate_single_element_dropped() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        list.truncate(0);
        assert!(list.is_empty());
        assert_eq!(list.first(), None);
        assert_eq!(list.last(), None);
    }

    // MARK: Borrow<Q> split_off (String / &str)

    #[test]
    fn split_off_str_on_string_element() {
        let mut list = OrderedSkipList::<String>::new();
        for s in ["apple", "banana", "cherry", "date"] {
            list.insert(s.to_owned());
        }
        // split at "cherry": self keeps apple, banana; right gets cherry, date
        let right = list.split_off("cherry");
        assert_eq!(
            list.iter().map(String::as_str).collect::<Vec<_>>(),
            ["apple", "banana"]
        );
        assert_eq!(
            right.iter().map(String::as_str).collect::<Vec<_>>(),
            ["cherry", "date"]
        );
    }
}
