//! List-restructuring methods for [`SkipList`](super::SkipList):
//! `clear`, `truncate`, `split_off`, and `append`.

use alloc::boxed::Box;
use core::ptr::NonNull;

use crate::{
    level_generator::LevelGenerator,
    node::{
        Node,
        visitor::{IndexMutVisitor, Visitor},
    },
    skip_list::SkipList,
};

impl<T, G: LevelGenerator, const N: usize> SkipList<T, N, G> {
    /// Removes all elements from the list.
    ///
    /// The level generator is preserved; elements can be inserted again
    /// immediately after calling `clear`.
    ///
    /// This operation is `$O(n)$`: all n elements must be dropped.
    ///
    /// If an element's destructor panics, the remaining elements are kept in
    /// the list rather than leaked, and the list stays usable; call `clear`
    /// again to drop them.
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
        // Detach the chain and reset the bookkeeping before running a single
        // destructor.  An element `Drop` that unwinds then finds the list
        // already empty and consistent, rather than leaving `tail` pointing at
        // a node the unwind has freed.
        //
        // SAFETY: `self.head` is a live, exclusively-owned sentinel, and the
        // chain hanging off it is owned by this list alone.
        let chain = unsafe {
            let head = self.head_mut();
            for link in head.links_mut() {
                *link = None;
            }
            head.take_next_chain()
        };
        self.tail = None;
        self.len = 0;

        let head = self.head;
        // SAFETY: `head` is the exclusively-owned sentinel, its `next` was
        // emptied above, and `chain` holds the nodes that hung off it, each
        // allocated by `Box::into_raw` and referenced from nowhere else.
        unsafe { Node::drop_chain(head, head, &mut self.len, &mut self.tail, chain) };
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
    /// If an element's destructor panics, the elements it had not yet reached
    /// are kept in the list rather than leaked, so the list may still be
    /// longer than `len`.
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
        clippy::indexing_slicing,
        reason = "l < max_levels = head.links.len(); every node in precursors[] was reached \
                  via a level-l link so its links.len() > l; all accesses are in bounds; \
                  l is used for both precursors[l] and links_mut()[l] so a plain index loop is \
                  the clearest expression"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link clearing, detaching the suffix and freeing it touch provably \
                  disjoint heap nodes; splitting across blocks would require \
                  unsafe-crossing raw-pointer variables"
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
        let max_levels = self.head_ref().level();

        // IndexMutVisitor with target = len advances directly to the node at rank len.
        // `current` from into_parts() is the new tail node itself.
        // into_parts() releases the &mut borrow.
        let head = self.head;
        let (new_tail_node, precursors, _) = {
            let mut visitor = IndexMutVisitor::new(self.head, len);
            visitor.traverse();
            visitor.into_parts()
        };

        // SAFETY: All raw pointers come from NonNull<Node<T, N>> captured during
        // traversal.  They originate from heap allocations owned by this SkipList.
        // No safe references to any node exist while this block runs.
        unsafe {
            // `current` is the node at rank len (the new tail).
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

            // Commit the new tail and length before any element is freed:
            // freeing runs element destructors, and one that unwinds must not
            // leave `tail` pointing into the region it has freed.
            self.tail = Some(new_tail_node);
            self.len = len;

            // Detach the removed suffix, then free it.  An element destructor
            // that unwinds leaves the rest of the suffix re-attached after the
            // new tail rather than leaked.
            let suffix = (*new_tail_ptr).take_next_chain();
            Node::drop_chain(head, new_tail_node, &mut self.len, &mut self.tail, suffix);
        }
    }

    /// Splits the list at the given index, returning a new list containing
    /// all elements from index `at` onward.
    ///
    /// After the call, `self` contains elements at indices `[0, at)` and the
    /// returned list contains elements previously at indices `[at, len)`,
    /// renumbered from 0 in the returned list.
    ///
    /// This operation runs in `$O(\log n)$` time: only the links that cross
    /// the split point move, so every element keeps its tower.
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

        let max_levels = self.head_ref().level();
        // SAFETY: `Box::into_raw` transfers ownership; freed in `Drop`.
        let head =
            unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(max_levels)))) };
        let mut result = Self {
            head,
            tail: None,
            len: 0,
            generator: self.generator.clone(),
        };
        if at == self.len {
            return result;
        }

        // SAFETY: we hold `&mut self`, `at < self.len`, and `result.head` is a
        // fresh empty head of the same height.
        let last_kept = unsafe { Node::split_after(self.head, at, result.head) };
        result.len = self.len.wrapping_sub(at);
        result.tail = self.tail;
        self.len = at;
        self.tail = (at > 0).then_some(last_kept);
        result
    }

    /// Moves all elements from `other` to the end of `self`, leaving `other` empty.
    ///
    /// Elements are appended in the order they appear in `other`.
    ///
    /// This operation runs in `$O(\log n)$` time.
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
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        if other.is_empty() {
            return;
        }

        // SAFETY: we hold `&mut self` and `&mut other`, two distinct lists,
        // and `other` is non-empty.
        unsafe {
            Node::concat(
                self.head,
                &mut self.len,
                &mut self.tail,
                other.head,
                &mut other.len,
                &mut other.tail,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use core::panic::AssertUnwindSafe;
    use std::panic::catch_unwind;

    use pretty_assertions::assert_eq;

    use super::super::SkipList;
    use crate::{
        node::Node,
        test_util::{PanicOnDrop, bombs},
    };

    /// Panic unless `list`'s nodes are well formed.
    fn assert_well_formed<T>(list: &SkipList<T>) {
        // SAFETY: `head` is the live head of `list`, which owns every node it
        // reaches.
        unsafe { Node::assert_well_formed(list.head, list.len, list.tail) };
    }

    /// A list of `0..len` with seeded tower heights.
    fn seeded(len: usize, seed: u64) -> SkipList<usize> {
        let mut list = SkipList::with_capacity_and_seed(64, seed);
        list.extend(0..len);
        list
    }

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
        assert!(list.head_ref().next_as_ref().is_none());
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
        let n: usize = if cfg!(miri) { 100 } else { 1_000 };
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
        assert!(
            list.head_ref()
                .next_as_ref()
                .and_then(|n| n.next_as_ref())
                .is_none()
        );
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
        const N: usize = if cfg!(miri) { 100 } else { 1_000 };
        const HALF: usize = if cfg!(miri) { 50 } else { 500 };
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
        let n: usize = if cfg!(miri) { 50 } else { 200 };
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

    #[test]
    fn split_off_every_index_is_well_formed() {
        for (len, seed) in (0..40).zip(0_u64..) {
            for at in 0..=len {
                let mut list = seeded(len, seed);
                let tail = list.split_off(at);
                assert_well_formed(&list);
                assert_well_formed(&tail);
                assert!(list.iter().copied().eq(0..at));
                assert!(tail.iter().copied().eq(at..len));
            }
        }
    }

    // MARK: head height

    #[test]
    fn head_is_as_tall_as_the_tallest_tower() {
        for len in [0, 1, 2, 10, 100, 1000] {
            let list = seeded(len, 3);
            // SAFETY: `head` is the live head of `list`, which owns every node
            // it reaches.
            let head = unsafe { list.head.as_ref() };
            let mut tallest = 0;
            let mut current = head.next_as_ref();
            while let Some(node) = current {
                tallest = tallest.max(node.level());
                current = node.next_as_ref();
            }
            assert_eq!(head.level(), tallest.max(1), "len {len}");
        }
    }

    // MARK: append

    #[test]
    fn append_every_pair_of_lengths_is_well_formed() {
        for left in 0..24 {
            for right in 0..24 {
                let mut a = seeded(left, 1);
                let mut b = SkipList::with_capacity_and_seed(64, 2);
                b.extend(left..left + right);
                a.append(&mut b);
                assert_well_formed(&a);
                assert_well_formed(&b);
                assert!(a.iter().copied().eq(0..left + right));
                assert!(b.is_empty());
                a.push_back(left + right);
                assert_well_formed(&a);
            }
        }
    }

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
        let n: usize = if cfg!(miri) { 50 } else { 200 };
        let m: usize = if cfg!(miri) { 40 } else { 150 };
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

    // MARK: link_structure

    #[test]
    fn link_structure() -> anyhow::Result<()> {
        use insta::assert_snapshot;

        use crate::level_generator::geometric::Geometric;

        let g = Geometric::new_with_seed(4, 0.5, 42)?;
        let mut list = SkipList::<i32, 4>::with_level_generator(g);
        for i in 1..=10 {
            list.push_back(i);
        }

        if !cfg!(miri) {
            let display = list.head_ref().display()?;
            assert_snapshot!(display, @"
            [03]: 00 -------------------> 04
            [02]: 00 -------------> 03 -> 04 -------------------------------> 10
            [01]: 00 -------------> 03 -> 04 -------------> 07 -> 08 -------> 10
            [->]: 00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10
            [<-]: 00 <- 01 <- 02 <- 03 <- 04 <- 05 <- 06 <- 07 <- 08 <- 09 <- 10

            [00|03] None
            [01|00] Some(1)
            [02|00] Some(2)
            [03|02] Some(3)
            [04|03] Some(4)
            [05|00] Some(5)
            [06|00] Some(6)
            [07|01] Some(7)
            [08|01] Some(8)
            [09|00] Some(9)
            [10|02] Some(10)
            ");
        }
        Ok(())
    }

    // MARK: panic safety

    #[test]
    fn clear_panicking_drop_leaves_list_empty() {
        let mut list = SkipList::<PanicOnDrop>::new();
        for bomb in bombs(16, 15) {
            list.push_back(bomb);
        }

        let result = catch_unwind(AssertUnwindSafe(|| list.clear()));
        assert!(result.is_err());

        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
        assert!(list.front().is_none());
        assert!(list.back().is_none());
        assert_eq!(list.iter().count(), 0);

        list.push_back(PanicOnDrop::new(100));
        assert_eq!(list.back().map(PanicOnDrop::id), Some(100));
    }

    #[test]
    fn truncate_panicking_drop_leaves_list_truncated() {
        let mut list = SkipList::<PanicOnDrop>::new();
        for bomb in bombs(16, 15) {
            list.push_back(bomb);
        }

        let result = catch_unwind(AssertUnwindSafe(|| list.truncate(4)));
        assert!(result.is_err());

        let ids: Vec<u32> = list.iter().map(PanicOnDrop::id).collect();
        assert_eq!(ids, [0, 1, 2, 3]);
        assert_eq!(list.len(), 4);
        assert_eq!(list.front().map(PanicOnDrop::id), Some(0));
        assert_eq!(list.back().map(PanicOnDrop::id), Some(3));
        assert!(list.get(4).is_none());

        list.push_back(PanicOnDrop::new(100));
        assert_eq!(list.back().map(PanicOnDrop::id), Some(100));
    }

    /// The elements a panicking `clear` never reached stay in the list; they
    /// must not be leaked, and a second `clear` must drop them.
    #[test]
    fn clear_panicking_drop_keeps_unreached_elements() {
        let mut list = SkipList::<PanicOnDrop>::new();
        for bomb in bombs(16, 7) {
            list.push_back(bomb);
        }

        let result = catch_unwind(AssertUnwindSafe(|| list.clear()));
        assert!(result.is_err());

        let ids: Vec<u32> = list.iter().map(PanicOnDrop::id).collect();
        assert_eq!(ids, [8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(list.len(), 8);
        assert_eq!(list.front().map(PanicOnDrop::id), Some(8));
        assert_eq!(list.back().map(PanicOnDrop::id), Some(15));
        for (index, id) in ids.iter().enumerate() {
            assert_eq!(list.get(index).map(PanicOnDrop::id), Some(*id));
        }

        list.clear();
        assert!(list.is_empty());
    }

    #[test]
    fn truncate_panicking_drop_keeps_unreached_elements() {
        let mut list = SkipList::<PanicOnDrop>::new();
        for bomb in bombs(16, 7) {
            list.push_back(bomb);
        }

        let result = catch_unwind(AssertUnwindSafe(|| list.truncate(4)));
        assert!(result.is_err());

        let ids: Vec<u32> = list.iter().map(PanicOnDrop::id).collect();
        assert_eq!(ids, [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(list.len(), 12);
        assert_eq!(list.back().map(PanicOnDrop::id), Some(15));
        for (index, id) in ids.iter().enumerate() {
            assert_eq!(list.get(index).map(PanicOnDrop::id), Some(*id));
        }
    }
}
