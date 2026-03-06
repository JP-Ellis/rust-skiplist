//! Ordered insertion and removal for [`OrderedSkipList`](super::OrderedSkipList).

use core::{cmp::Ordering, ptr::NonNull};

use crate::{
    comparator::Comparator,
    level_generator::LevelGenerator,
    node::{
        Node,
        link::Link,
        visitor::{IndexMutVisitor, OrdIndexMutVisitor, OrdMutVisitor, Visitor},
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
                  decrement_distance panics only if a distance would underflow to 0, which \
                  cannot happen because every skip link spanning front has distance >= 2; \
                  all expects fire only on internal invariant violations, not user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l is bounded by front_height ≤ max_levels and max_levels ≤ max_levels, \
                  which equals the length of the links slice on every node, so all \
                  accesses are in bounds"
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

        let max_levels = self.head_ref().level();

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
            // For l < front_height: head.links[l] pointed to front_node with
            //   distance 1 (front is always adjacent to head).  The new distance
            //   is 1 + front.links[l].distance - 1 = front.links[l].distance.
            //   So copying front_node.links[l] directly is correct.
            // For l >= front_height: head.links[l] skips over front_node to a
            //   node at rank r.  After removing front, that node is now at rank
            //   r - 1, so the distance decreases by 1.
            for l in 0..front_height {
                (*head_ptr).links_mut()[l] = (*front_ptr).links_mut()[l].take();
            }
            for l in front_height..max_levels {
                if let Some(link) = (*head_ptr).links_mut()[l].as_mut() {
                    link.decrement_distance()
                        .expect("skip link spanning front node has distance >= 2");
                }
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
        reason = "Link::new distances are computed to be >= 1; \
                  increment_distance overflow requires > usize::MAX nodes; \
                  precursors[0] always exists because max_levels >= 1; \
                  all expects fire only on internal invariant violations, not on user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "precursors[0] is valid: max_levels >= 1 so precursors.len() >= 1; \
                  precursors[l].links_mut()[l] is valid: OrdIndexMutVisitor guarantees \
                  each precursor node has a link slot at the level it was recorded for; \
                  new_raw.links_mut()[l] is valid: l < height <= new_raw.level()"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "insertion and link wiring touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn insert(&mut self, value: T) {
        let height = self.generator.level().saturating_add(1);

        // Use OrdIndexMutVisitor to locate the insertion point, collect precursors,
        // and track rank distances for accurate skip-link maintenance.
        //
        // `self.head` is a `NonNull` (a `Copy` type), so copying it does not
        // borrow `self`.  The closure borrows only `self.comparator` (shared),
        // which is a distinct field from `self.head`.  Both borrows coexist
        // safely and are released when `visitor` is dropped via `into_parts()`.
        let (precursors, precursor_distances) = {
            let head = self.head;
            let cmp = |v: &T, t: &T| self.comparator.compare(v, t);
            let mut visitor = OrdIndexMutVisitor::new(head, &value, cmp);
            visitor.traverse();
            let (_current, _found, precursors, precursor_distances) = visitor.into_parts();
            (precursors, precursor_distances)
        };
        // `visitor` and `cmp` are dropped here, releasing the borrow on
        // `self.comparator`.  `value` is no longer borrowed by the visitor.

        // The new node's internal rank (head = rank 0, first data node = rank 1).
        let new_rank = precursor_distances[0].saturating_add(1);

        // SAFETY: All raw pointers originate from `NonNull<Node<T, N>>` values
        // captured during traversal.  They point into heap allocations exclusively
        // owned by this `OrderedSkipList`.  No safe `&mut` references to any node
        // exist while this block runs.  The pointer `new_raw` is distinct from
        // every precursor: it is freshly allocated by `Node::insert_after`.
        let new_node_nonnull: NonNull<Node<T, N>> = unsafe {
            let new_raw: *mut Node<T, N> =
                Node::insert_after(precursors[0], Node::with_value(height, value)).as_ptr();

            // Wire skip links with accurate distances.
            //
            // For l < height (new node's tower reaches this level):
            //   Before: pred (rank D) --[d]--> X (rank D + d)
            //   After:  pred (rank D) --[new_rank − D]--> new_node (rank new_rank)
            //           new_node      --[D + d + 1 − new_rank]--> X (rank D + d + 1)
            //
            // For l >= height (new node has no tower slot here):
            //   pred.links[l] still points to X (now at rank D + d + 1);
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

    /// Removes and returns the first (earliest) element that compares equal to
    /// `value`, or `None` if no such element exists.
    ///
    /// When duplicates are present only the first occurrence is removed; the
    /// rest remain in the list.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(1);
    /// list.insert(2);
    /// list.insert(2);
    /// list.insert(3);
    /// assert_eq!(list.take_first(&2), Some(2));
    /// assert_eq!(list.len(), 3);
    /// assert!(list.contains(&2)); // second 2 is still there
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "Link::new distance is pred_to_target + target_to_succ - 1 >= 1; \
                  decrement_distance panics only on underflow to 0 which cannot happen \
                  for valid skip-link distances; all expects are invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l iterates 0..max_levels; precursors[l] is valid because \
                  OrdMutVisitor fills all max_levels entries; \
                  links_mut()[l] is valid because l < node.level() = max_levels"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link splicing and node pop touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn take_first(&mut self, value: &T) -> Option<T> {
        let (target_ptr, found, precursors) = {
            let head = self.head;
            let cmp = |v: &T, t: &T| self.comparator.compare(v, t);
            let mut visitor = OrdMutVisitor::new(head, value, cmp);
            visitor.traverse();
            visitor.into_parts()
        };

        if !found {
            return None;
        }

        let max_levels = self.head_ref().level();

        // SAFETY: `found` is true, so `target_ptr` is a live data node owned by
        // this list.  `precursors[l]` for l < target_height have their level-l
        // link pointing to `target_ptr` (skip-list invariant + OrdMutVisitor
        // semantics for Equal).  For l >= target_height, `precursors[l]` is the
        // last node at level l whose link spans past `target_ptr`.
        // No other &mut references to any node exist.
        let val = unsafe {
            let target_height = target_ptr.as_ref().level();
            let target_raw = target_ptr.as_ptr();

            // Splice out target_ptr with accurate distance maintenance.
            //
            // For l < target_height: pred.links[l] → target (dist d1),
            //   target.links[l] → succ (dist d2) or None.
            //   New: pred.links[l] → succ (dist d1 + d2 - 1) or None.
            // For l >= target_height: pred.links[l] spans over target to some
            //   node at a rank 1 higher than before; decrement the distance.
            for (l, pred_nn) in precursors.iter().enumerate().take(max_levels) {
                let pred_ptr = pred_nn.as_ptr();
                if l < target_height {
                    let old_link = (*pred_ptr).links_mut()[l].take();
                    let target_link = (*target_raw).links_mut()[l].take();
                    (*pred_ptr).links_mut()[l] = match (old_link, target_link) {
                        (Some(pred_to_target), Some(target_to_succ)) => {
                            let new_dist = pred_to_target
                                .distance()
                                .get()
                                .saturating_add(target_to_succ.distance().get())
                                .saturating_sub(1);
                            Some(Link::new(target_to_succ.node(), new_dist).expect("new_dist >= 1"))
                        }
                        (_, None) => None,
                        (None, tgt) => tgt,
                    };
                } else if let Some(link) = (*pred_ptr).links_mut()[l].as_mut() {
                    link.decrement_distance()
                        .expect("skip link spanning target has distance >= 2");
                }
            }

            let mut popped = (*target_raw).pop();
            popped.take_value()
        };

        if self.tail == Some(target_ptr) {
            self.tail = if self.len == 1 {
                None
            } else {
                Some(precursors[0])
            };
        }
        self.len = self.len.saturating_sub(1);
        val
    }

    /// Removes and returns the last (latest) element that compares equal to
    /// `value`, or `None` if no such element exists.
    ///
    /// When duplicates are present only the last occurrence is removed; the
    /// rest remain in the list.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(1);
    /// list.insert(2);
    /// list.insert(2);
    /// list.insert(3);
    /// assert_eq!(list.take_last(&2), Some(2));
    /// assert_eq!(list.len(), 3);
    /// assert!(list.contains(&2)); // first 2 is still there
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "Link::new distance is pred_to_target + target_to_succ - 1 >= 1; \
                  decrement_distance panics only on underflow to 0 which cannot happen \
                  for valid skip-link distances; all expects are invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l iterates 0..target_height or 0..max_levels; arr[l] and \
                  cmp_past_precursors[l] are in-bounds for the same reason; \
                  cmp_past_precursors[0] is always initialised because max_levels >= 1"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "pointer-equality scan, link splicing, and node pop all touch \
                  provably disjoint heap nodes; splitting across blocks would require \
                  unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn take_last(&mut self, value: &T) -> Option<T> {
        // Use a modified comparator that treats Equal as Less so that the
        // visitor advances past ALL equal nodes.  After traversal,
        // `cmp_past_precursors[0]` is the last node whose value compares equal
        // to `value`, or a node with a smaller value if none exist.
        // For l >= target_height, `cmp_past_precursors[l]` is the node whose
        // skip-link at level l spans over target and must be decremented.
        let (target_ptr, cmp_past_precursors) = {
            let head = self.head;
            let cmp_past = |v: &T, t: &T| match self.comparator.compare(v, t) {
                Ordering::Equal | Ordering::Less => Ordering::Less,
                Ordering::Greater => Ordering::Greater,
            };
            let mut visitor = OrdMutVisitor::new(head, value, cmp_past);
            visitor.traverse();
            let (_current, _found, precursors) = visitor.into_parts();
            let target = precursors[0];
            (target, precursors)
        };

        // Verify that target_ptr is actually equal to `value`; the head
        // sentinel has no value, and a node with a smaller value means the
        // target is absent from the list.
        // SAFETY: target_ptr comes from OrdMutVisitor's precursors array, which
        // is pre-filled with `self.head` (always valid) and updated only to
        // node pointers reachable from head: all live for &mut self's lifetime.
        let is_equal = unsafe { target_ptr.as_ref() }
            .value()
            .is_some_and(|v| self.comparator.compare(v, value) == Ordering::Equal);
        if !is_equal {
            return None;
        }

        // SAFETY: target_ptr is a live, valid data node for the lifetime of
        // &mut self.  No other &mut reference to it exists.
        let target_height = unsafe { target_ptr.as_ref() }.level();
        let max_levels = self.head_ref().level();

        // Find the direct level-l predecessor of target_ptr for each level
        // l < target_height via a pointer-equality forward scan.
        // These are used for the distance-preserving splice and tail update.
        let scan_precursors: [NonNull<Node<T, N>>; N] = {
            let mut arr = [self.head; N];
            let mut current = self.head;

            for l in (0..target_height).rev() {
                loop {
                    // SAFETY: `current` is a valid node in this list, live for
                    // the duration of &mut self.  No exclusive reference exists.
                    let maybe_link = unsafe { current.as_ref() }
                        .links()
                        .get(l)
                        .and_then(|lk| lk.as_ref());
                    match maybe_link {
                        None => break,
                        Some(link) if link.node() == target_ptr => break,
                        Some(link) => current = link.node(),
                    }
                }
                arr[l] = current;
            }
            arr
        };

        // SAFETY: All raw pointers come from NonNull<Node<T, N>> values
        // captured above.  No safe &mut references to any node exist.
        let val = unsafe {
            let target_raw = target_ptr.as_ptr();

            // For l < target_height: distance-preserving splice via scan_precursors.
            // For l >= target_height: decrement distance of cmp_past_precursors[l].
            for l in 0..max_levels {
                if l < target_height {
                    let pred_ptr = scan_precursors[l].as_ptr();
                    let old_link = (*pred_ptr).links_mut()[l].take();
                    let target_link = (*target_raw).links_mut()[l].take();
                    (*pred_ptr).links_mut()[l] = match (old_link, target_link) {
                        (Some(pred_to_target), Some(target_to_succ)) => {
                            let new_dist = pred_to_target
                                .distance()
                                .get()
                                .saturating_add(target_to_succ.distance().get())
                                .saturating_sub(1);
                            Some(Link::new(target_to_succ.node(), new_dist).expect("new_dist >= 1"))
                        }
                        (_, None) => None,
                        (None, tgt) => tgt,
                    };
                } else if let Some(link) =
                    (*cmp_past_precursors[l].as_ptr()).links_mut()[l].as_mut()
                {
                    link.decrement_distance()
                        .expect("skip link spanning target has distance >= 2");
                }
            }

            let mut popped = (*target_raw).pop();
            popped.take_value()
        };

        if self.tail == Some(target_ptr) {
            self.tail = if self.len == 1 {
                None
            } else {
                Some(scan_precursors[0])
            };
        }
        self.len = self.len.saturating_sub(1);
        val
    }

    /// Removes and returns any element that compares equal to `value`, or
    /// `None` if no such element exists.
    ///
    /// This is an alias for [`take_first`](OrderedSkipList::take_first), which
    /// always removes the first (earliest) equal occurrence.  Use [`take_fast`]
    /// if the specific occurrence removed does not matter and you want a
    /// potentially faster path.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// [`take_fast`]: OrderedSkipList::take_fast
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
    /// assert_eq!(list.take(&2), Some(2));
    /// assert_eq!(list.len(), 2);
    /// assert!(!list.contains(&2));
    /// ```
    #[inline]
    pub fn take(&mut self, value: &T) -> Option<T> {
        self.take_first(value)
    }

    /// Removes all elements that compare equal to `value` and returns the
    /// number of elements removed.
    ///
    /// This operation is `$O(k \log n)$` where k is the number of occurrences of
    /// `value`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(1);
    /// list.insert(2);
    /// list.insert(2);
    /// list.insert(2);
    /// list.insert(3);
    /// assert_eq!(list.remove_all(&2), 3);
    /// assert_eq!(list.len(), 2);
    /// assert!(!list.contains(&2));
    /// ```
    #[inline]
    pub fn remove_all(&mut self, value: &T) -> usize {
        let mut count = 0_usize;
        while self.take_first(value).is_some() {
            count = count.saturating_add(1);
        }
        count
    }

    /// Removes and returns an element that compares equal to `value`, or
    /// `None` if no such element exists.
    ///
    /// Unlike [`take`] / [`take_first`], which always remove the first (earliest)
    /// equal occurrence, this method may remove any occurrence when duplicates
    /// are present.  Prefer [`take`] for routine removals where consistency
    /// about which duplicate is removed matters; use `take_fast` only when
    /// any occurrence is acceptable.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// [`take`]: OrderedSkipList::take
    /// [`take_first`]: OrderedSkipList::take_first
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
    /// assert_eq!(list.take_fast(&2), Some(2));
    /// assert_eq!(list.len(), 2);
    /// assert!(!list.contains(&2));
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "Link::new distance is pred_to_target + target_to_succ - 1 >= 1; \
                  decrement_distance panics only on underflow to 0 which cannot happen \
                  for valid skip-link distances; all expects are invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l iterates 0..max_levels; scan_precursors[l] and first_precursors[l] are \
                  valid because OrdMutVisitor fills all max_levels entries; \
                  links_mut()[l] is valid because l < node.level() = max_levels"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "pointer-equality scan, link splicing, and node pop all touch \
                  provably disjoint heap nodes; splitting across blocks would require \
                  unsafe-crossing raw-pointer variables"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "This method's complexity arises from the fast-path optimization, \
                  which requires replicating much of the logic of take_first while \
                  using raw pointers and pointer-equality scans to locate the target \
                  and its predecessors more quickly. Extracting helper methods would \
                  require passing many raw-pointer variables across unsafe boundaries, \
                  which would be more error-prone than keeping all the logic in one place."
    )]
    #[inline]
    pub fn take_fast(&mut self, value: &T) -> Option<T> {
        // Pass 1: find the "fast" target, the first equal node encountered
        // during skip traversal (tends to be a higher-level node).
        //
        // We replicate OrdVisitor's descent inline using NonNull pointers
        // (Reserved provenance) rather than &Node (Frozen provenance).
        // NonNull::from(&ref) would capture a Frozen tag that forbids
        // subsequent writes, so we obtain fast_ptr exclusively via link.node()
        // and node.next(), both of which preserve Reserved provenance.
        let fast_ptr: NonNull<Node<T, N>> = {
            // SAFETY: self.head is a valid node for &mut self's lifetime.
            let max_levels = unsafe { self.head.as_ref() }.level();
            let cmp = |v: &T| self.comparator.compare(v, value);
            let mut current = self.head;
            let mut level = max_levels;
            let mut result: Option<NonNull<Node<T, N>>> = None;

            'search: loop {
                // Examine skip links from level-1 down to 0, following the
                // first link that does not overshoot (Less or Equal).
                // This mirrors OrdVisitor::step()'s top-down descent.
                // SAFETY: current is a valid heap-allocated node.
                let links = unsafe { current.as_ref() }.links();
                let check_up_to = links.len().min(level);
                let mut advanced_via_skip = false;

                for l in (0..check_up_to).rev() {
                    if let Some(link) = links.get(l).and_then(|lk| lk.as_ref()) {
                        // link.node() carries Reserved (writable) provenance.
                        let next = link.node();
                        // SAFETY: next is a valid heap-allocated node.
                        let ord = unsafe { next.as_ref() }
                            .value()
                            .map_or(Ordering::Less, &cmp);
                        if ord != Ordering::Greater {
                            current = next;
                            level = l.saturating_add(1);
                            if ord == Ordering::Equal {
                                result = Some(current);
                                break 'search;
                            }
                            advanced_via_skip = true;
                            break; // Re-examine links on new current at new level.
                        }
                    }
                }

                if !advanced_via_skip {
                    // No skip link can advance us; fall back to sequential next.
                    level = 0;
                    // SAFETY: current is a valid node; next() → Reserved NonNull.
                    match unsafe { current.as_ref() }.next() {
                        None => break 'search,
                        Some(next) => {
                            // SAFETY: next is a valid heap-allocated node.
                            let ord = unsafe { next.as_ref() }
                                .value()
                                .map_or(Ordering::Less, &cmp);
                            match ord {
                                Ordering::Greater => break 'search,
                                Ordering::Less => current = next,
                                Ordering::Equal => {
                                    result = Some(next);
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }

            result?
        };

        // Pass 2: locate the first-equal precursors using OrdMutVisitor (Less-only
        // advancement) as the starting point for the pointer-equality scan.
        let (first_ptr, found, first_precursors) = {
            let head = self.head;
            let cmp = |v: &T, t: &T| self.comparator.compare(v, t);
            let mut visitor = OrdMutVisitor::new(head, value, cmp);
            visitor.traverse();
            visitor.into_parts()
        };

        if !found {
            // Pass 1 succeeded, so this is an internal invariant violation.
            return None;
        }

        let max_levels = self.head_ref().level();

        // SAFETY: `found` is true, so `first_ptr` and `fast_ptr` are live data
        // nodes owned by this list.  No other &mut references to any node exist
        // while this block runs.
        //
        // The scan reads links via shared (immutable) access only; structural
        // mutations happen only in the splice loop afterwards.  Both phases touch
        // provably disjoint heap nodes.
        let (val, level0_pred) = unsafe {
            let fast_height = fast_ptr.as_ref().level();
            let fast_raw = fast_ptr.as_ptr();

            // Build `scan_precursors`: for l < fast_height this holds the exact
            // level-l predecessor of `fast_ptr` found by a pointer-equality scan;
            // for l >= fast_height it holds `first_precursors[l]` (the node whose
            // level-l link spans over fast_ptr, valid because whenever fast_ptr !=
            // first_ptr the skip link that bypasses first_ptr requires
            // first_ptr.height <= link_level < fast_ptr.height, so fast_height >
            // first_height and first_precursors[l] for l >= fast_height spans over
            // fast_ptr).
            let mut scan_precursors = [self.head; N];
            for l in 0..max_levels {
                scan_precursors[l] = first_precursors[l];
            }

            if fast_ptr != first_ptr {
                let mut current = self.head;
                for l in (0..fast_height).rev() {
                    loop {
                        let maybe_link = current.as_ref().links().get(l).and_then(|lk| lk.as_ref());
                        match maybe_link {
                            None => break,
                            Some(link) if link.node() == fast_ptr => break,
                            Some(link) => current = link.node(),
                        }
                    }
                    scan_precursors[l] = current;
                }
            }

            // Splice out fast_ptr with accurate distance maintenance:
            //   l < fast_height  → distance-preserving link replacement (scan_precursors)
            //   l >= fast_height → distance-only decrement (first_precursors == scan_precursors)
            for (l, &scan_pred) in scan_precursors.iter().enumerate().take(max_levels) {
                let pred_ptr = scan_pred.as_ptr();
                if l < fast_height {
                    let old_link: Option<Link<T, N>> = (*pred_ptr).links_mut()[l].take();
                    let target_link: Option<Link<T, N>> = (*fast_raw).links_mut()[l].take();
                    (*pred_ptr).links_mut()[l] = match (old_link, target_link) {
                        (Some(pred_to_target), Some(target_to_succ)) => {
                            let new_dist = pred_to_target
                                .distance()
                                .get()
                                .saturating_add(target_to_succ.distance().get())
                                .saturating_sub(1);
                            Some(Link::new(target_to_succ.node(), new_dist).expect("new_dist >= 1"))
                        }
                        (_, None) => None,
                        (None, tgt) => tgt,
                    };
                } else if let Some(link) = (*pred_ptr).links_mut()[l].as_mut() {
                    link.decrement_distance()
                        .expect("skip link spanning target has distance >= 2");
                }
            }

            let mut popped = (*fast_raw).pop();
            (popped.take_value(), scan_precursors[0])
        };

        if self.tail == Some(fast_ptr) {
            self.tail = if self.len == 1 {
                None
            } else {
                Some(level0_pred)
            };
        }
        self.len = self.len.saturating_sub(1);
        val
    }

    /// Removes and returns the element at the given index.
    ///
    /// The index is 0-based: `0` refers to the first (smallest) element and
    /// `self.len() - 1` refers to the last (largest) element.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
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
    /// assert_eq!(list.remove(1), 2);
    /// assert_eq!(list.len(), 2);
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "Link::new distance is pred_to_target + target_to_succ - 1 >= 1; \
                  decrement_distance panics only on underflow to 0 which cannot happen \
                  for valid skip-link distances; all expects are invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l iterates 0..max_levels; precursors[l] is valid because \
                  IndexMutVisitor fills all max_levels entries; \
                  links_mut()[l] is valid because l < node.level() = max_levels"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link splicing and node pop touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        assert!(
            index < self.len,
            "index out of bounds: the len is {} but the index is {index}",
            self.len
        );

        // IndexMutVisitor uses 1-based rank: head = rank 0, first data = rank 1.
        // To target 0-based `index` we pass `index + 1`.
        let (target_ptr, precursors) = {
            let head = self.head;
            let mut visitor = IndexMutVisitor::new(head, index.saturating_add(1));
            visitor.traverse();
            debug_assert!(
                visitor.found(),
                "index {index} was asserted < len but visitor did not find it"
            );
            let (current, precursors, _precursor_distances) = visitor.into_parts();
            (current, precursors)
        };

        let max_levels = self.head_ref().level();

        // SAFETY: `target_ptr` is a live data node owned by this list.
        // `index < self.len` was asserted above so the visitor found it.
        // `precursors[l]` for l < target_height have their level-l link
        // pointing to `target_ptr`.  For l >= target_height, `precursors[l]`
        // is the last node whose level-l link spans past `target_ptr`.
        // No other &mut references to any node exist while this block runs.
        let val = unsafe {
            let target_height = target_ptr.as_ref().level();
            let target_raw = target_ptr.as_ptr();

            for (l, pred_nn) in precursors.iter().enumerate().take(max_levels) {
                let pred_ptr = pred_nn.as_ptr();
                if l < target_height {
                    let old_link = (*pred_ptr).links_mut()[l].take();
                    let target_link = (*target_raw).links_mut()[l].take();
                    (*pred_ptr).links_mut()[l] = match (old_link, target_link) {
                        (Some(pred_to_target), Some(target_to_succ)) => {
                            let new_dist = pred_to_target
                                .distance()
                                .get()
                                .saturating_add(target_to_succ.distance().get())
                                .saturating_sub(1);
                            Some(Link::new(target_to_succ.node(), new_dist).expect("new_dist >= 1"))
                        }
                        (_, None) => None,
                        (None, tgt) => tgt,
                    };
                } else if let Some(link) = (*pred_ptr).links_mut()[l].as_mut() {
                    link.decrement_distance()
                        .expect("skip link spanning target has distance >= 2");
                }
            }

            let mut popped = (*target_raw).pop();
            popped.take_value()
        };

        if self.tail == Some(target_ptr) {
            self.tail = if self.len == 1 {
                None
            } else {
                Some(precursors[0])
            };
        }
        self.len = self.len.saturating_sub(1);
        val.expect("removed node has a value")
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

    // MARK: take_first

    #[test]
    fn take_first_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        assert_eq!(list.take_first(&1), None);
    }

    #[test]
    fn take_first_single_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.take_first(&42), Some(42));
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn take_first_single_not_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.take_first(&99), None);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn take_first_from_front() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.take_first(&1), Some(1));
        assert_eq!(list.len(), 2);
        assert_eq!(list.first(), Some(&2));
    }

    #[test]
    fn take_first_from_back_updates_tail() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.take_first(&3), Some(3));
        assert_eq!(list.len(), 2);
        assert_eq!(list.last(), Some(&2));
    }

    #[test]
    fn take_first_from_middle() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.take_first(&2), Some(2));
        assert_eq!(list.len(), 2);
        assert_eq!(list.first(), Some(&1));
        assert_eq!(list.last(), Some(&3));
    }

    #[test]
    fn take_first_not_found_between_elements() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(3);
        assert_eq!(list.take_first(&2), None);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn take_first_duplicate_removes_first_occurrence() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.take_first(&2), Some(2));
        assert_eq!(list.len(), 3);
        // One copy of 2 remains.
        assert!(list.contains(&2));
    }

    #[test]
    fn take_first_len_decrements() {
        let mut list = OrderedSkipList::<usize>::new();
        for i in 1..=10_usize {
            list.insert(i);
        }
        for i in 1..=10_usize {
            assert_eq!(list.take_first(&i), Some(i));
            assert_eq!(list.len(), 10 - i);
        }
        assert_eq!(list.take_first(&1), None);
    }

    #[test]
    fn take_first_custom_comparator() {
        // Largest-first ordering: take_first removes by comparator order.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(3); // stored as [3, 2, 1]
        list.insert(1);
        list.insert(2);
        assert_eq!(list.take_first(&2), Some(2));
        assert_eq!(list.len(), 2);
        assert!(!list.contains(&2));
    }

    // MARK: take_last

    #[test]
    fn take_last_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        assert_eq!(list.take_last(&1), None);
    }

    #[test]
    fn take_last_single_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.take_last(&42), Some(42));
        assert!(list.is_empty());
    }

    #[test]
    fn take_last_single_not_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.take_last(&99), None);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn take_last_no_duplicates_same_as_take_first() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.take_last(&2), Some(2));
        assert_eq!(list.len(), 2);
        assert!(!list.contains(&2));
    }

    #[test]
    fn take_last_duplicate_removes_last_occurrence() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.take_last(&2), Some(2));
        assert_eq!(list.len(), 3);
        // One copy of 2 still remains.
        assert!(list.contains(&2));
    }

    #[test]
    fn take_last_all_equal_removes_last() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(2);
        list.insert(2);
        list.insert(2);
        assert_eq!(list.take_last(&2), Some(2));
        assert_eq!(list.len(), 2);
        assert_eq!(list.take_last(&2), Some(2));
        assert_eq!(list.len(), 1);
        assert_eq!(list.take_last(&2), Some(2));
        assert!(list.is_empty());
        assert_eq!(list.take_last(&2), None);
    }

    #[test]
    fn take_last_from_back_updates_tail() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.take_last(&3), Some(3));
        assert_eq!(list.len(), 2);
        assert_eq!(list.last(), Some(&2));
    }

    #[test]
    fn take_last_not_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(3);
        assert_eq!(list.take_last(&2), None);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn take_last_custom_comparator() {
        // Largest-first ordering.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(3); // stored as [3, 2, 1]
        list.insert(1);
        list.insert(2);
        assert_eq!(list.take_last(&2), Some(2));
        assert_eq!(list.len(), 2);
        assert!(!list.contains(&2));
    }

    // MARK: take

    #[test]
    fn take_delegates_to_take_first() {
        // take() must agree with take_first() on every input.
        let mut list1 = OrderedSkipList::<i32>::new();
        let mut list2 = OrderedSkipList::<i32>::new();
        for v in [1, 2, 2, 3] {
            list1.insert(v);
            list2.insert(v);
        }
        // Remove 2: both lists should produce the same result and leave the
        // same remaining element.
        assert_eq!(list1.take(&2), list2.take_first(&2));
        assert_eq!(list1.len(), list2.len());
        let v1: Vec<i32> = list1.iter().copied().collect();
        let v2: Vec<i32> = list2.iter().copied().collect();
        assert_eq!(v1, v2);
    }

    #[test]
    fn take_returns_none_when_absent() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(3);
        assert_eq!(list.take(&2), None);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn take_removes_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.take(&2), Some(2));
        assert_eq!(list.len(), 2);
        assert!(!list.contains(&2));
    }

    // MARK: take_fast

    #[test]
    fn take_fast_returns_none_when_absent() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(3);
        assert_eq!(list.take_fast(&2), None);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn take_fast_no_duplicates_same_as_take() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(3);
        // With no duplicates take_fast and take must agree.
        assert_eq!(list.take_fast(&2), Some(2));
        assert_eq!(list.len(), 2);
        assert!(!list.contains(&2));
    }

    #[test]
    fn take_fast_with_duplicates_removes_one() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        let result = list.take_fast(&2);
        assert_eq!(result, Some(2));
        assert_eq!(list.len(), 4);
        // At least two copies of 2 remain.
        assert!(list.contains(&2));
    }

    #[test]
    fn take_fast_empty_list() {
        let mut list = OrderedSkipList::<i32>::new();
        assert_eq!(list.take_fast(&1), None);
    }

    #[test]
    fn take_fast_len_decrements() {
        let mut list = OrderedSkipList::<i32>::new();
        for _ in 0..5 {
            list.insert(42);
        }
        for remaining in (0..5_usize).rev() {
            assert_eq!(list.take_fast(&42), Some(42));
            assert_eq!(list.len(), remaining);
        }
        assert_eq!(list.take_fast(&42), None);
    }

    #[test]
    fn take_fast_links_consistent() {
        // Insert many duplicates; remove all via take_fast and verify the list
        // remains structurally valid after each removal.
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(0);
        for _ in 0..20 {
            list.insert(1);
        }
        list.insert(2);
        while let Some(v) = list.take_fast(&1) {
            assert_eq!(v, 1);
            // Structural check: iter must still produce a sorted sequence.
            let got: Vec<i32> = list.iter().copied().collect();
            let mut sorted = got.clone();
            sorted.sort_unstable();
            assert_eq!(got, sorted, "list is not sorted after take_fast");
        }
        assert_eq!(list.len(), 2);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [0, 2]);
    }

    // MARK: remove_all

    #[test]
    fn remove_all_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        assert_eq!(list.remove_all(&1), 0);
    }

    #[test]
    fn remove_all_not_found() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(3);
        assert_eq!(list.remove_all(&2), 0);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn remove_all_single_occurrence() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.remove_all(&2), 1);
        assert_eq!(list.len(), 2);
        assert!(!list.contains(&2));
    }

    #[test]
    fn remove_all_multiple_occurrences() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.remove_all(&2), 3);
        assert_eq!(list.len(), 2);
        assert!(!list.contains(&2));
        assert!(list.contains(&1));
        assert!(list.contains(&3));
    }

    #[test]
    fn remove_all_entire_list() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(5);
        list.insert(5);
        list.insert(5);
        assert_eq!(list.remove_all(&5), 3);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn remove_all_custom_comparator() {
        // Largest-first ordering.
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        assert_eq!(list.remove_all(&2), 2);
        assert_eq!(list.len(), 2);
        assert!(!list.contains(&2));
    }

    // MARK: remove (by index)

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn remove_empty_panics() {
        let mut list = OrderedSkipList::<i32>::new();
        list.remove(0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn remove_out_of_bounds_panics() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.remove(2); // len is 2; index 2 is out of bounds
    }

    #[test]
    fn remove_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(list.remove(0), 42);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.first(), None);
        assert_eq!(list.last(), None);
    }

    #[test]
    fn remove_first_element() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3] {
            list.insert(i);
        }
        assert_eq!(list.remove(0), 1);
        assert_eq!(list.len(), 2);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [2, 3]);
    }

    #[test]
    fn remove_last_element() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3] {
            list.insert(i);
        }
        assert_eq!(list.remove(2), 3);
        assert_eq!(list.len(), 2);
        assert_eq!(list.last(), Some(&2));
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2]);
    }

    #[test]
    fn remove_middle_element() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3, 4, 5] {
            list.insert(i);
        }
        assert_eq!(list.remove(2), 3);
        assert_eq!(list.len(), 4);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 4, 5]);
    }

    #[test]
    fn remove_decrements_len() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 0..10 {
            list.insert(i);
        }
        for expected_len in (0..10).rev() {
            list.remove(0);
            assert_eq!(list.len(), expected_len);
        }
    }

    #[test]
    fn remove_tail_pointer_correct() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3, 4, 5] {
            list.insert(i);
        }
        // Remove the tail: index 4 (value 5)
        list.remove(4);
        assert_eq!(list.last(), Some(&4));
        assert_eq!(list.len(), 4);
    }

    #[test]
    fn remove_links_consistent() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 0..20 {
            list.insert(i);
        }
        // Remove all even-indexed elements (values 0, 2, 4, …, 18).
        for offset in 0..10 {
            list.remove(offset); // after each removal the indices shift
        }
        // Should have values 1, 3, 5, …, 19 remaining.
        let got: Vec<i32> = list.iter().copied().collect();
        let expected: Vec<i32> = (0..20).step_by(2).map(|x| x + 1).collect();
        assert_eq!(got, expected);
        // Verify rank-based access is also consistent.
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(list.get_by_index(i), Some(&v));
        }
    }

    #[test]
    fn remove_custom_comparator() {
        // Largest-first ordering: list is [3, 2, 1].
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(1);
        list.insert(2);
        list.insert(3);
        // Index 0 is 3 (largest first), index 1 is 2, index 2 is 1.
        assert_eq!(list.remove(1), 2);
        assert_eq!(list.len(), 2);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [3, 1]);
    }

    #[test]
    fn remove_with_duplicates() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        for _ in 0..3 {
            list.insert(1);
            list.insert(2);
        }
        // List: [1, 1, 1, 2, 2, 2].  Remove index 2 (third 1).
        assert_eq!(list.remove(2), 1);
        assert_eq!(list.len(), 5);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 1, 2, 2, 2]);
    }
}
