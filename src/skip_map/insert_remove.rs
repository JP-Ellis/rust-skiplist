//! Insertion and removal operations for [`SkipMap`](super::SkipMap).

use core::ptr::NonNull;

use super::SkipMap;
use crate::{
    comparator::{Comparator, ComparatorKey},
    level_generator::LevelGenerator,
    node::{
        Node,
        link::Link,
        visitor::{OrdIndexMutVisitor, OrdMutVisitor, Visitor},
    },
};

impl<K, V, const N: usize, C: Comparator<K>, G: LevelGenerator> SkipMap<K, V, N, C, G> {
    /// Inserts a key-value pair into the map.
    ///
    /// If the map already contained an entry with this key, the value is
    /// updated in place and the old value is returned as `Some(old_value)`.
    /// Otherwise, the entry is inserted at its sorted position and `None`
    /// is returned.
    ///
    /// When duplicate keys are present the first matching entry (in sorted
    /// order) is the one that is updated.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// assert_eq!(map.insert(1, "a"), None);
    /// assert_eq!(map.insert(2, "b"), None);
    /// assert_eq!(map.insert(1, "c"), Some("a"));
    /// assert_eq!(map.len(), 2);
    /// assert_eq!(map.get(&1), Some(&"c"));
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        clippy::unwrap_in_result,
        reason = "Link::new distances are computed to be >= 1; \
                  increment_distance overflow requires > usize::MAX nodes; \
                  precursors[0] always exists because max_levels >= 1; \
                  value_mut on a data node always returns Some; \
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
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // height ∈ [0, total]: number of skip links to allocate.
        let height = self.generator.level();

        // `self.head` is a `NonNull` (a `Copy` type), so copying it does not
        // borrow `self`.  The closure borrows only `self.comparator` (shared),
        // which is a distinct field from `self.head`.  Both borrows coexist
        // safely and are released when `visitor` is dropped via `into_parts()`.
        let (current_rank, current, found, precursors, precursor_distances) = {
            let head = self.head;
            let cmp = |entry: &(K, V), k: &K| self.comparator.compare(&entry.0, k);
            let mut visitor = OrdIndexMutVisitor::new(head, &key, cmp);
            visitor.traverse();
            let rank = visitor.current_rank_internal();
            let (current, found, precursors, precursor_distances) = visitor.into_parts();
            (rank, current, found, precursors, precursor_distances)
        };
        // `visitor` and `cmp` are dropped here, releasing the borrow on
        // `self.comparator`.  `key` is no longer borrowed by the visitor.

        if found {
            // The key already exists at `current`.  Swap the value in place;
            // no structural change means skip links remain valid.
            //
            // SAFETY: `found` is true, so `current` is a live data node owned
            // by this SkipMap.  No other &mut reference to it exists while we
            // hold `&mut self`.
            let old = unsafe {
                let pair = current
                    .as_ptr()
                    .as_mut()
                    .expect("non-null")
                    .value_mut()
                    .expect("data node has value");
                core::mem::replace(&mut pair.1, value)
            };
            return Some(old);
        }

        // `found == false`: current is the last node with key < `key`.
        // new_rank = current_rank + 1.
        let new_rank = current_rank.saturating_add(1);

        // SAFETY: All raw pointers originate from `NonNull<Node<(K,V), N>>`
        // values captured during traversal.  They point into heap allocations
        // exclusively owned by this `SkipMap`.  No safe `&mut` references to
        // any node exist while this block runs.  The pointer `new_raw` is
        // distinct from every precursor: it is freshly allocated by
        // `Node::insert_after`.
        let new_node_nonnull: NonNull<Node<(K, V), N>> = unsafe {
            // `found == false`: current is the last node strictly less than key;
            // insert the new node immediately after it.
            let new_raw: *mut Node<(K, V), N> =
                Node::insert_after(current, Node::with_value(height, (key, value))).as_ptr();

            // Wire skip links with accurate distances.
            //
            // For l < height (new node's tower reaches this level):
            //   Before: pred (rank D) --[d]--> X (rank D + d)
            //   After:  pred (rank D) --[new_rank - D]--> new_node (rank new_rank)
            //           new_node      --[D + d + 1 - new_rank]--> X (rank D + d + 1)
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

        // The new node is the tail if it has no successor.
        // SAFETY: `new_node_nonnull` was just created from `Box::into_raw`
        // above; it is properly aligned, fully initialized, and no other
        // reference to it exists yet.
        let is_new_tail = unsafe { new_node_nonnull.as_ref() }.next().is_none();
        if is_new_tail {
            self.tail = Some(new_node_nonnull);
        }

        self.len = self.len.saturating_add(1);
        None
    }

    /// Removes the entry with the given key and returns its value, or `None`
    /// if the key is absent.
    ///
    /// When duplicate keys are present the first matching entry (in sorted
    /// order) is removed.
    ///
    /// This operation is O(log n) on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// assert_eq!(map.len(), 1);
    /// ```
    #[inline]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: ?Sized,
        C: ComparatorKey<K, Q>,
    {
        self.remove_impl_q(key).map(|(_, v)| v)
    }

    /// Removes the entry with the given key and returns the key-value pair,
    /// or `None` if the key is absent.
    ///
    /// When duplicate keys are present the first matching entry (in sorted
    /// order) is removed.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(7, "seven");
    /// assert_eq!(map.remove_entry(&7), Some((7, "seven")));
    /// assert_eq!(map.remove_entry(&7), None);
    /// ```
    #[inline]
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        Q: ?Sized,
        C: ComparatorKey<K, Q>,
    {
        self.remove_impl_q(key)
    }

    /// Non-generic removal used internally (e.g. by `merge`) where the key
    /// type is always `K`.  Avoids requiring `C: ComparatorKey<K, K>` on
    /// callers that only ever pass `&K`.
    #[expect(
        clippy::expect_used,
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
    pub(super) fn remove_impl(&mut self, key: &K) -> Option<(K, V)> {
        let (target_ptr, found, precursors) = {
            let head = self.head;
            let cmp = |entry: &(K, V), k: &K| self.comparator.compare(&entry.0, k);
            let mut visitor = OrdMutVisitor::new(head, key, cmp);
            visitor.traverse();
            visitor.into_parts()
        };

        if !found {
            return None;
        }

        let max_levels = self.head_ref().level();

        // SAFETY: `found` is true, so `target_ptr` is a live data node owned
        // by this SkipMap.  `precursors[l]` for l < target_height have their
        // level-l link pointing to `target_ptr` (skip-list invariant +
        // OrdMutVisitor semantics for Equal).  For l >= target_height,
        // `precursors[l]` is the last node at level l whose link spans past
        // `target_ptr`.  No other &mut references to any node exist.
        let (kv, new_tail) = unsafe {
            let target_height = target_ptr.as_ref().level();
            let target_raw = target_ptr.as_ptr();

            // Splice out target_ptr with accurate distance maintenance.
            //
            // For l < target_height: pred.links[l] points to target (dist d1),
            //   target.links[l] points to succ (dist d2) or None.
            //   New: pred.links[l] points to succ (dist d1 + d2 - 1) or None.
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

            // Capture predecessor before removing the node.
            let new_tail = (*target_raw).prev();
            let mut popped = (*target_raw).pop();
            (popped.take_value(), new_tail)
        };

        if self.tail == Some(target_ptr) {
            self.tail = if self.len == 1 { None } else { new_tail };
        }
        self.len = self.len.saturating_sub(1);
        kv
    }

    /// Generic removal for the public `remove` / `remove_entry` API.
    ///
    /// Accepts any borrowed form `Q` of `K` via [`ComparatorKey`].
    #[expect(
        clippy::expect_used,
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
    fn remove_impl_q<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        Q: ?Sized,
        C: ComparatorKey<K, Q>,
    {
        let (target_ptr, found, precursors) = {
            let head = self.head;
            let cmp = |entry: &(K, V), q: &Q| self.comparator.compare_key(&entry.0, q);
            let mut visitor = OrdMutVisitor::new(head, key, cmp);
            visitor.traverse();
            visitor.into_parts()
        };

        if !found {
            return None;
        }

        let max_levels = self.head_ref().level();

        // SAFETY: same as `remove_impl`.
        let (kv, new_tail) = unsafe {
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

            // Capture predecessor before removing the node.
            let new_tail = (*target_raw).prev();
            let mut popped = (*target_raw).pop();
            (popped.take_value(), new_tail)
        };

        if self.tail == Some(target_ptr) {
            self.tail = if self.len == 1 { None } else { new_tail };
        }
        self.len = self.len.saturating_sub(1);
        kv
    }

    /// Removes and returns the minimum-key entry, or `None` if the map is
    /// empty.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(2, "b");
    /// map.insert(1, "a");
    /// assert_eq!(map.pop_first(), Some((1, "a")));
    /// assert_eq!(map.pop_first(), Some((2, "b")));
    /// assert_eq!(map.pop_first(), None);
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
        reason = "l is bounded by front_height <= max_levels, which equals the length \
                  of the links slice on every node, so all accesses are in bounds"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link unwiring, pointer extraction, and node pop all touch provably \
                  disjoint heap nodes; splitting across blocks would require \
                  unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        if self.is_empty() {
            return None;
        }

        let max_levels = self.head_ref().level();

        // SAFETY: All raw pointers originate from heap allocations owned by
        // this SkipMap.  No safe &mut references to any node exist while this
        // block runs.  head_ptr and front_ptr are distinct heap allocations;
        // all slice accesses are bounded by front_height <= max_levels =
        // links.len().
        let kv = unsafe {
            let head_ptr: *mut Node<(K, V), N> = self.head.as_ptr();

            // front_ptr is the node at rank 1.  The list is non-empty, so
            // head.next is Some.  Converting the &mut to NonNull releases the
            // borrow immediately, leaving no live &mut when we later use
            // head_ptr.
            let front_ptr: *mut Node<(K, V), N> =
                NonNull::from((*head_ptr).next_as_mut().expect("list is non-empty")).as_ptr();

            let front_height = (*front_ptr).level();

            // Splice out front_node: move its skip links back to head.
            //
            // For l < front_height: head.links[l] pointed to front_node with
            //   distance 1 (front is always adjacent to head).  The new
            //   distance is 1 + front.links[l].distance - 1 =
            //   front.links[l].distance.  So copying front_node.links[l]
            //   directly is correct.
            // For l >= front_height: head.links[l] skips over front_node to a
            //   node at rank r.  After removing front, that node is now at
            //   rank r - 1, so the distance decreases by 1.
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
        kv
    }

    /// Removes and returns the maximum-key entry, or `None` if the map is
    /// empty.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// assert_eq!(map.pop_last(), Some((2, "b")));
    /// assert_eq!(map.pop_last(), Some((1, "a")));
    /// assert_eq!(map.pop_last(), None);
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
        reason = "l is bounded by tail_height <= max_levels, which equals the length \
                  of the links slice on every node, so all accesses are in bounds"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link clearing and node pop touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        if self.is_empty() {
            return None;
        }

        let tail_ptr: NonNull<Node<(K, V), N>> = self.tail.expect("non-empty map has a tail");

        // SAFETY: tail_ptr is a live, valid data node for the lifetime of
        // &mut self.  No other &mut reference to it exists.
        let tail_height = unsafe { tail_ptr.as_ref() }.level();

        // Find the predecessor of the tail at each skip level using a
        // pointer-equality forward traversal.
        //
        // At each level l < tail_height, advance from `current` while the
        // level-l link does NOT point to the tail.  When the loop stops,
        // `current` is the unique predecessor at level l (the node whose
        // link[l] == tail).  `current` is not reset between levels: the
        // skip-list structure guarantees we can only advance forward.
        //
        // For levels l >= tail_height, no node can link directly to the tail
        // (tail has no tower slot at those levels), so no links need clearing.
        let precursors: [NonNull<Node<(K, V), N>>; N] = {
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
                        None => break,
                        Some(link) if link.node() == tail_ptr => break,
                        Some(link) => current = link.node(),
                    }
                }
                arr[l] = current;
            }
            arr
        };

        // SAFETY: All raw pointers come from NonNull<Node<(K,V), N>> values
        // captured during traversal or from self.tail.  No safe &mut
        // references to any node exist while this block runs.
        let (kv, new_tail) = unsafe {
            let tail_raw = tail_ptr.as_ptr();

            // Clear all skip links pointing to the tail.
            // For levels >= tail_height no link points to the tail: no-op.
            for (l, pred_nn) in precursors.iter().enumerate().take(tail_height) {
                (*pred_nn.as_ptr()).links_mut()[l] = None;
            }

            // Capture predecessor before removing the node.
            let new_tail = (*tail_raw).prev();
            let mut popped = (*tail_raw).pop();
            (popped.take_value(), new_tail)
        };

        self.tail = if self.len == 1 { None } else { new_tail };
        self.len = self.len.saturating_sub(1);
        kv
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::SkipMap;
    use crate::comparator::FnComparator;

    // MARK: insert

    #[test]
    fn insert_into_empty_returns_none() {
        let mut map = SkipMap::<i32, &str>::new();
        assert_eq!(map.insert(1, "a"), None);
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
    }

    #[test]
    fn insert_duplicate_key_returns_old_value() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        let old = map.insert(1, "b");
        assert_eq!(old, Some("a"));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"b"));
    }

    #[test]
    fn insert_multiple_unique_keys_maintains_order() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(3, 30);
        map.insert(1, 10);
        map.insert(2, 20);
        assert_eq!(map.len(), 3);
        assert_eq!(map.first_key_value(), Some((&1, &10)));
        assert_eq!(map.last_key_value(), Some((&3, &30)));
    }

    #[test]
    fn insert_updates_tail() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 1);
        assert_eq!(map.last_key_value(), Some((&1, &1)));
        map.insert(5, 5);
        assert_eq!(map.last_key_value(), Some((&5, &5)));
        map.insert(3, 3);
        // 3 is inserted before 5; tail remains 5.
        assert_eq!(map.last_key_value(), Some((&5, &5)));
    }

    #[test]
    fn insert_single_element_first_and_last_agree() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(42, "x");
        assert_eq!(map.first_key_value(), Some((&42, &"x")));
        assert_eq!(map.last_key_value(), Some((&42, &"x")));
    }

    #[test]
    fn insert_large_number_of_entries() {
        let mut map = SkipMap::<usize, usize>::new();
        for i in 0..100_usize {
            assert_eq!(map.insert(i, i * 10), None);
        }
        assert_eq!(map.len(), 100);
        for i in 0..100_usize {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn insert_custom_comparator_reverse_order() {
        // Reverse comparator: larger keys sort first.
        let mut map: SkipMap<i32, &str, 16, _> =
            SkipMap::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        map.insert(1, "a");
        map.insert(3, "c");
        map.insert(2, "b");
        // In reverse order, 3 is "first" (smallest), 1 is "last" (largest).
        assert_eq!(map.first_key_value(), Some((&3, &"c")));
        assert_eq!(map.last_key_value(), Some((&1, &"a")));
    }

    // MARK: remove / remove_entry

    #[test]
    fn remove_absent_key_returns_none() {
        let mut map = SkipMap::<i32, &str>::new();
        assert_eq!(map.remove(&99), None);
    }

    #[test]
    fn remove_only_element() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        assert_eq!(map.remove(&1), Some("a"));
        assert!(map.is_empty());
        assert_eq!(map.first_key_value(), None);
        assert_eq!(map.last_key_value(), None);
    }

    #[test]
    fn remove_first_of_many() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);
        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.len(), 2);
        assert_eq!(map.first_key_value(), Some((&2, &20)));
    }

    #[test]
    fn remove_last_updates_tail() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);
        assert_eq!(map.remove(&3), Some(30));
        assert_eq!(map.len(), 2);
        assert_eq!(map.last_key_value(), Some((&2, &20)));
    }

    #[test]
    fn remove_middle_element() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);
        assert_eq!(map.remove(&2), Some(20));
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&2), None);
        assert_eq!(map.first_key_value(), Some((&1, &10)));
        assert_eq!(map.last_key_value(), Some((&3, &30)));
    }

    #[test]
    fn remove_entry_returns_key_and_value() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(7, "seven");
        assert_eq!(map.remove_entry(&7), Some((7, "seven")));
        assert!(map.is_empty());
    }

    #[test]
    fn remove_entry_absent_returns_none() {
        let mut map = SkipMap::<i32, &str>::new();
        assert_eq!(map.remove_entry(&1), None);
    }

    #[test]
    fn remove_then_contains_key_is_false() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(5, 50);
        map.remove(&5);
        assert!(!map.contains_key(&5));
    }

    // MARK: pop_first

    #[test]
    fn pop_first_empty_returns_none() {
        let mut map = SkipMap::<i32, i32>::new();
        assert_eq!(map.pop_first(), None);
    }

    #[test]
    fn pop_first_single_element_empties_map() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(5, 50);
        assert_eq!(map.pop_first(), Some((5, 50)));
        assert!(map.is_empty());
    }

    #[test]
    fn pop_first_returns_minimum_key_sequence() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(3, 30);
        map.insert(1, 10);
        map.insert(2, 20);
        assert_eq!(map.pop_first(), Some((1, 10)));
        assert_eq!(map.pop_first(), Some((2, 20)));
        assert_eq!(map.pop_first(), Some((3, 30)));
        assert_eq!(map.pop_first(), None);
    }

    #[test]
    fn pop_first_updates_first_key_value() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);
        map.pop_first();
        assert_eq!(map.first_key_value(), Some((&2, &20)));
    }

    // MARK: pop_last

    #[test]
    fn pop_last_empty_returns_none() {
        let mut map = SkipMap::<i32, i32>::new();
        assert_eq!(map.pop_last(), None);
    }

    #[test]
    fn pop_last_single_element_empties_map() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(5, 50);
        assert_eq!(map.pop_last(), Some((5, 50)));
        assert!(map.is_empty());
    }

    #[test]
    fn pop_last_returns_maximum_key_sequence() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);
        assert_eq!(map.pop_last(), Some((3, 30)));
        assert_eq!(map.pop_last(), Some((2, 20)));
        assert_eq!(map.pop_last(), Some((1, 10)));
        assert_eq!(map.pop_last(), None);
    }

    #[test]
    fn pop_last_updates_last_key_value() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);
        map.pop_last();
        assert_eq!(map.last_key_value(), Some((&2, &20)));
    }

    // MARK: cross-method invariant checks

    #[test]
    fn len_consistent_across_insert_and_remove() {
        let mut map = SkipMap::<i32, i32>::new();
        for i in 0..10_i32 {
            map.insert(i, i * 10);
        }
        assert_eq!(map.len(), 10);
        for i in 0..5_i32 {
            map.remove(&i);
        }
        assert_eq!(map.len(), 5);
    }

    #[test]
    fn first_last_updated_after_all_pops() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 1);
        map.insert(2, 2);
        map.insert(3, 3);
        map.pop_first();
        assert_eq!(map.first_key_value(), Some((&2, &2)));
        map.pop_last();
        assert_eq!(map.last_key_value(), Some((&2, &2)));
    }

    #[test]
    fn get_after_insert_remove_roundtrip() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(10, "ten");
        map.insert(20, "twenty");
        map.remove(&10);
        assert_eq!(map.get(&10), None);
        assert_eq!(map.get(&20), Some(&"twenty"));
    }

    #[test]
    fn get_mut_after_insert() {
        let mut map = SkipMap::<i32, i32>::new();
        map.insert(1, 10);
        map.insert(2, 20);
        *map.get_mut(&1).expect("key present") += 5;
        assert_eq!(map.get(&1), Some(&15));
        assert_eq!(map.get(&2), Some(&20));
    }

    // MARK: Borrow<Q> removals

    #[test]
    fn remove_str_on_string_key() {
        let mut map: SkipMap<String, i32> = SkipMap::new();
        map.insert("hello".to_owned(), 1);
        map.insert("world".to_owned(), 2);
        assert_eq!(map.remove("hello"), Some(1));
        assert!(!map.contains_key("hello"));
        assert_eq!(map.remove("missing"), None);
    }

    #[test]
    fn remove_entry_str_on_string_key() {
        let mut map: SkipMap<String, i32> = SkipMap::new();
        map.insert("hello".to_owned(), 1);
        map.insert("world".to_owned(), 2);
        assert_eq!(map.remove_entry("hello"), Some(("hello".to_owned(), 1)));
        assert!(!map.contains_key("hello"));
        assert_eq!(map.remove_entry("missing"), None);
    }
}
