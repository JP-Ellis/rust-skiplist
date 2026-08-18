//! Shared mutable cursor state for skip list variant cursors.
//!
//! [`RawCursorMut`] holds the three fields that are structurally identical
//! between `OrderedSkipList::CursorMut` and `SkipMap::CursorMut`:
//! the current node pointer, its 1-based rank, and an optional all-level
//! precursor cache.
//!
//! All methods operate on the skip list structure only (pointer wiring, rank
//! tracking).  Value-type-specific logic (ordering checks, tail/len updates,
//! value extraction) stays in the outer `CursorMut` wrappers.

use core::{cmp::Ordering, ptr::NonNull};

use arrayvec::ArrayVec;

use crate::node::{
    Node,
    visitor::{IndexMutVisitor, Visitor},
};

/// Shared mutable cursor state for skip list variant cursors.
///
/// Contains:
/// - `current` — the node on the left side of the gap (head sentinel = leftmost gap).
/// - `current_rank` — 1-based position of `current`; 0 means head sentinel.
/// - `precursors` — optional cache of all-level predecessors of the gap.
///
/// The cache is `None` after a backward step (impossible to update incrementally)
/// and rebuilt lazily before the next structural modification.
#[expect(
    clippy::field_scoped_visibility_modifiers,
    reason = "pub(crate) fields allow both OSL and SkipMap CursorMut to read/write \
              current, current_rank, and precursors directly without accessor boilerplate"
)]
pub(crate) struct RawCursorMut<V, const N: usize> {
    /// The node on the left side of the gap.  Points to the head sentinel
    /// when the cursor is at the leftmost gap.
    pub(crate) current: NonNull<Node<V, N>>,
    /// 1-based rank of `current` within the list: 0 = head sentinel.
    pub(crate) current_rank: usize,
    /// Cached all-level precursors for the current gap, if valid.
    ///
    /// `Some((precursors, precursor_distances))` when the cache is valid:
    ///   `precursors[l]` = last node at level `l` before the current gap,
    ///   `precursor_distances[l]` = its 1-based rank.
    ///
    /// `None` after a backward step (`retreat()`).  Rebuilt lazily before
    /// the next structural modification.
    #[expect(
        clippy::type_complexity,
        reason = "two parallel ArrayVecs form the precursor cache; a named type would \
                  not reduce complexity at the call sites"
    )]
    pub(crate) precursors: Option<(ArrayVec<NonNull<Node<V, N>>, N>, ArrayVec<usize, N>)>,
}

// MARK: Internal gap-finding helper

/// Compute the gap position for a cursor lower-bound construction.
///
/// Returns `(current_nonnull, current_1based_rank)` where `current` is the
/// node on the **left** side of the gap (or the head sentinel for the
/// leftmost gap).
///
/// - `advance_on_equal = false` advances only on `Less`
///   (used by `Included` lower-bound and `Excluded` upper-bound).
/// - `advance_on_equal = true` advances on `Less` **or** `Equal`
///   (used by `Excluded` lower-bound and `Included` upper-bound).
///
/// # Safety
///
/// `head` must be a valid, live pointer to the head sentinel of the list for
/// the duration of this call.
pub(crate) unsafe fn gap_find<T, Q, F, const N: usize>(
    head: NonNull<Node<T, N>>,
    q: &Q,
    compare: F,
    advance_on_equal: bool,
) -> (NonNull<Node<T, N>>, usize)
where
    Q: ?Sized,
    F: Fn(&T, &Q) -> Ordering,
{
    let mut current = head;
    let mut rank: usize = 0;
    // SAFETY: caller guarantees head is valid.
    let max_levels = unsafe { head.as_ref() }.level();

    // Skip-link phase: process levels from high to low.
    // At each level, advance as far as possible before dropping to the next.
    for level in (0..max_levels).rev() {
        loop {
            // SAFETY: `current` is valid throughout — it starts as `head` and
            // is only updated to nodes reachable via valid skip links.
            let current_ref = unsafe { current.as_ref() };
            let Some(link) = current_ref.links().get(level).and_then(|l| l.as_ref()) else {
                break;
            };
            let dest = link.node();
            // SAFETY: `dest` is a valid heap-allocated node.
            let ord = unsafe { dest.as_ref() }
                .value()
                .map_or(Ordering::Less, |v| compare(v, q));
            let should_advance = match ord {
                Ordering::Less => true,
                Ordering::Equal => advance_on_equal,
                Ordering::Greater => false,
            };
            if should_advance {
                rank = rank.saturating_add(link.distance().get());
                current = dest;
                // Stay at the same level: try again from the new position.
            } else {
                break;
            }
        }
    }

    // Sequential phase: walk the base-layer doubly-linked list one step at a time.
    //
    // Note: `links[l]` are skip-links, not DLL pointers — even `links[0]` can span
    // multiple nodes.  After the skip-link phase above, `current` may be several DLL
    // hops before the target.  The `next()` method returns the immediate DLL successor,
    // so we use it here to land on the precise gap.
    // SAFETY: `current` is valid; each `next` obtained from it is also valid.
    while let Some(next) = unsafe { current.as_ref() }.next() {
        // SAFETY: `next` is a valid node.
        let ord = unsafe { next.as_ref() }
            .value()
            .map_or(Ordering::Less, |v| compare(v, q));
        let should_advance = match ord {
            Ordering::Less => true,
            Ordering::Equal => advance_on_equal,
            Ordering::Greater => false,
        };
        if should_advance {
            rank = rank.saturating_add(1);
            current = next;
        } else {
            break;
        }
    }

    (current, rank)
}

impl<V, const N: usize> RawCursorMut<V, N> {
    /// Constructs a new `RawCursorMut` at `current` with the given rank.
    ///
    /// The precursor cache starts empty and is populated lazily by
    /// [`ensure_precursors`](Self::ensure_precursors).
    pub(crate) fn new(current: NonNull<Node<V, N>>, current_rank: usize) -> Self {
        Self {
            current,
            current_rank,
            precursors: None,
        }
    }

    /// Ensures the precursor cache is populated for the current gap position.
    ///
    /// If the cache is already valid this is a no-op.  Otherwise it runs an
    /// [`IndexMutVisitor`] traversal targeting `current_rank + 1` and stores
    /// the result.
    pub(crate) fn ensure_precursors(&mut self, head: NonNull<Node<V, N>>) {
        if self.precursors.is_some() {
            return;
        }
        let target = self.current_rank.saturating_add(1);
        let (_, precursors, precursor_distances) = {
            let mut visitor = IndexMutVisitor::new(head, target);
            visitor.traverse();
            visitor.into_parts()
        };
        self.precursors = Some((precursors, precursor_distances));
    }

    /// Advance the cursor one position to the right.
    ///
    /// Updates the precursor cache incrementally (if active) and returns the
    /// new `current` node.  Returns `None` if already at the rightmost gap.
    #[expect(
        clippy::indexing_slicing,
        reason = "l < next_level ≤ N = precursors.len(); head always has N levels"
    )]
    pub(crate) fn advance(&mut self) -> Option<NonNull<Node<V, N>>> {
        // SAFETY: `current` is a valid node.
        let next = unsafe { self.current.as_ref() }.next()?;
        let new_rank = self.current_rank.saturating_add(1);
        if let Some((ref mut precursors, ref mut precursor_distances)) = self.precursors {
            // SAFETY: `next` is a valid node.
            let next_level = unsafe { next.as_ref() }.level().min(N);
            for l in 0..next_level {
                precursors[l] = next;
                precursor_distances[l] = new_rank;
            }
        }
        self.current = next;
        self.current_rank = new_rank;
        Some(next)
    }

    /// Retreat the cursor one position to the left.
    ///
    /// Invalidates the precursor cache and returns the old `current` node
    /// (the element that was just straddled).  Returns `None` if already at
    /// the leftmost gap (head sentinel).
    #[expect(
        clippy::expect_used,
        clippy::unwrap_in_result,
        reason = "every data node always has a predecessor; fires only on internal corruption"
    )]
    pub(crate) fn retreat(&mut self) -> Option<NonNull<Node<V, N>>> {
        // Returns `None` if `current` is the head sentinel (no value).
        // SAFETY: `current` is valid.
        unsafe { self.current.as_ref() }.value()?;
        let old = self.current;
        // SAFETY: every data node has a predecessor (at minimum the head sentinel).
        let prev = unsafe { self.current.as_ref() }
            .prev()
            .expect("data node always has a predecessor");
        self.current = prev;
        self.current_rank = self.current_rank.saturating_sub(1);
        // Backwards traversal cannot update precursors incrementally.
        self.precursors = None;
        Some(old)
    }

    /// Remove the node immediately to the right of the cursor.
    ///
    /// Returns `(boxed_node, target_ptr)` on success:
    /// - `boxed_node` — owned removed node; caller calls `.take_value()` and drops it.
    /// - `target_ptr` — raw pointer to the removed node, for tail-pointer checks before
    ///   the box is dropped.
    ///
    /// Restores the precursor cache after removal (the removed node's absence
    /// does not invalidate the cached predecessors for the same gap).
    ///
    /// Returns `None` if there is no right neighbour.
    #[expect(
        clippy::expect_used,
        clippy::unwrap_in_result,
        reason = "ensure_precursors unconditionally sets Some; the expect encodes an \
                  internal invariant"
    )]
    #[expect(
        clippy::type_complexity,
        reason = "two-element return avoids an extra heap allocation; a named type \
                  would not simplify the single call site in each outer cursor"
    )]
    pub(crate) fn splice_out_next(
        &mut self,
        head: NonNull<Node<V, N>>,
    ) -> Option<(Box<Node<V, N>>, NonNull<Node<V, N>>)> {
        // SAFETY: `current` is valid.
        let target_ptr = unsafe { self.current.as_ref() }.next()?;
        // Target must be a data node (no tail sentinel in this design).
        // SAFETY: `target_ptr` is a valid node.
        unsafe { target_ptr.as_ref() }.value()?;

        self.ensure_precursors(head);
        let (precursors, precursor_distances) = self
            .precursors
            .take()
            .expect("ensure_precursors populated the cache");

        // SAFETY: target_ptr and all precursor pointers are valid and disjoint.
        let boxed = unsafe { Node::splice_out(target_ptr, &precursors) };

        // Precursors remain valid after removal; restore the cache.
        self.precursors = Some((precursors, precursor_distances));

        Some((boxed, target_ptr))
    }

    /// Remove the node at 1-based `rank` from the list.
    ///
    /// Returns `(boxed_node, predecessor)` on success:
    /// - `boxed_node` — owned removed node; caller calls `.take_value()` and drops it.
    /// - `predecessor` — base-layer DLL predecessor of the removed node (used by
    ///   the caller to update its `current` pointer and the list's `tail`).
    ///
    /// Returns `None` if `rank` is out of bounds.
    ///
    /// # Note
    ///
    /// This is a free function (no `&mut self`) because `remove_prev` needs to
    /// remove `self.current` while that pointer is still live in `&mut self`.
    /// Taking `self` as a mutable reference during the visitor traversal would
    /// create aliasing issues at the use site.
    #[expect(
        clippy::expect_used,
        clippy::unwrap_in_result,
        reason = "every data node always has a predecessor; fires only on internal corruption"
    )]
    #[expect(
        clippy::type_complexity,
        reason = "two-element return mirrors the OSL/SkipMap pattern; a named type \
                  would not simplify the single call site in each outer cursor"
    )]
    pub(crate) fn splice_out_at_rank(
        rank: usize,
        head: NonNull<Node<V, N>>,
    ) -> Option<(Box<Node<V, N>>, NonNull<Node<V, N>>)> {
        let (target_ptr, precursors, _) = {
            let mut visitor = IndexMutVisitor::new(head, rank);
            visitor.traverse();
            if !visitor.found() {
                return None;
            }
            visitor.into_parts()
        };

        // Capture the base-layer predecessor before `splice_out` zeroes the
        // `prev` pointer.  `precursors[0]` can lag behind the DLL predecessor
        // when a level-0 node was not tracked for every intermediate hop.
        // SAFETY: target_ptr is a valid, live data node.
        let predecessor = unsafe { target_ptr.as_ref() }
            .prev()
            .expect("data node always has a predecessor");

        // SAFETY: target_ptr and all precursors are valid and disjoint.
        let boxed = unsafe { Node::splice_out(target_ptr, &precursors) };

        Some((boxed, predecessor))
    }

    /// Wire skip links for a freshly base-layer-inserted node, then optionally
    /// advance the cursor to that node.
    ///
    /// `new_node` must have already been inserted in the DLL via
    /// `Node::insert_after(self.current, …)` before calling this method.
    ///
    /// # Precondition
    ///
    /// `self.precursors` must be `Some`.  Call
    /// [`ensure_precursors`](Self::ensure_precursors) first.
    #[expect(
        clippy::expect_used,
        reason = "precursors must be Some before insert_at_gap; the expect encodes an \
                  internal invariant, not an error path"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < new_level ≤ N = precursors.len(); head always has N levels"
    )]
    pub(crate) fn insert_at_gap(
        &mut self,
        new_node: NonNull<Node<V, N>>,
        new_rank: usize,
        height: usize,
        move_cursor: bool,
    ) {
        let (mut precursors, mut precursor_distances) = self
            .precursors
            .take()
            .expect("precursors must be Some before insert_at_gap");

        // SAFETY: new_node is freshly inserted in the base layer and not yet
        // wired into any skip links; precursors are the correct all-level
        // predecessors for new_rank.
        unsafe {
            Node::wire_links(
                new_node,
                new_rank,
                height,
                &precursors,
                &precursor_distances,
            );
        };

        if move_cursor {
            // SAFETY: new_node is valid — we just created it.
            let new_level = unsafe { new_node.as_ref() }.level().min(N);
            for l in 0..new_level {
                precursors[l] = new_node;
                precursor_distances[l] = new_rank;
            }
            self.current = new_node;
            self.current_rank = new_rank;
        }

        self.precursors = Some((precursors, precursor_distances));
    }
}

#[expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test code, safety guarantees can be relaxed"
)]
#[cfg(test)]
mod tests {
    use core::ptr::NonNull;

    use anyhow::Result;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::{RawCursorMut, gap_find};
    use crate::node::{
        Node,
        tests::{MAX_LEVELS, skiplist},
    };

    // The fixture builds: head(0) -> v1=10(1) -> v2=20(2) -> v3=30(3) -> v4=40(4)
    // Skip links:
    //   head.links[1] -> v3 (distance 3)
    //   head.links[0] -> v2 (distance 2)
    //   v2.links[0]   -> v3 (distance 1)
    //   v3.links[0]   -> v4 (distance 1)

    // MARK: new

    #[rstest]
    fn new_initialises_fields(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;

        let cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        assert_eq!(cursor.current, head);
        assert_eq!(cursor.current_rank, 0);
        assert!(cursor.precursors.is_none());

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: ensure_precursors

    #[rstest]
    fn ensure_precursors_populates_cache(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        assert!(cursor.precursors.is_none());
        cursor.ensure_precursors(head);
        assert!(cursor.precursors.is_some());

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn ensure_precursors_is_idempotent(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        cursor.ensure_precursors(head);
        let snapshot = cursor.precursors.clone();
        // Second call must not overwrite the already-valid cache.
        cursor.ensure_precursors(head);
        assert_eq!(cursor.precursors, snapshot);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: advance

    #[rstest]
    fn advance_from_head_reaches_first_node(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        let next_nn = cursor
            .advance()
            .expect("first advance from head should succeed");
        assert_eq!(unsafe { next_nn.as_ref() }.value(), Some(&10_u8));
        assert_eq!(cursor.current, next_nn);
        assert_eq!(cursor.current_rank, 1);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn advance_traverses_all_nodes(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        for (expected_rank, expected_value) in [(1, 10_u8), (2, 20), (3, 30), (4, 40)] {
            let next_nn = cursor.advance().expect("advance should succeed");
            assert_eq!(unsafe { next_nn.as_ref() }.value(), Some(&expected_value));
            assert_eq!(cursor.current_rank, expected_rank);
        }

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn advance_at_last_node_returns_none(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        for _ in 0..4 {
            cursor.advance();
        }
        // Cursor is now at v4 (rank 4); there is no further right neighbour.
        let result = cursor.advance();
        assert!(result.is_none());
        assert_eq!(unsafe { cursor.current.as_ref() }.value(), Some(&40_u8));
        assert_eq!(cursor.current_rank, 4);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn advance_maintains_precursor_cache(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        cursor.ensure_precursors(head);
        cursor.advance();
        // A forward step must keep the cache valid, not wipe it.
        assert!(cursor.precursors.is_some());

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: retreat

    #[rstest]
    fn retreat_at_head_returns_none(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        let result = cursor.retreat();
        assert!(result.is_none());
        assert_eq!(cursor.current, head);
        assert_eq!(cursor.current_rank, 0);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn retreat_from_first_node_returns_to_head(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let v1_ptr = unsafe { head.as_ref() }.next().expect("v1");
        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(v1_ptr, 1);
        // retreat returns the node we just stepped off (v1).
        let old = cursor.retreat();
        assert_eq!(old, Some(v1_ptr));
        assert_eq!(cursor.current, head);
        assert_eq!(cursor.current_rank, 0);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn retreat_invalidates_precursor_cache(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let v1_ptr = unsafe { head.as_ref() }.next().expect("v1");
        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(v1_ptr, 1);
        cursor.ensure_precursors(head);
        assert!(cursor.precursors.is_some());
        cursor.retreat();
        assert!(cursor.precursors.is_none());

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: splice_out_next

    #[rstest]
    fn splice_out_next_removes_right_neighbor(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        let (mut boxed, _target_ptr) = cursor
            .splice_out_next(head)
            .expect("head has a right neighbour");
        assert_eq!(boxed.take_value(), Some(10_u8));
        drop(boxed);
        // Cursor must remain at head.
        assert_eq!(cursor.current, head);
        assert_eq!(cursor.current_rank, 0);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn splice_out_next_at_last_node_returns_none(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        for _ in 0..4 {
            cursor.advance();
        }
        // Cursor is at v4; there is no right neighbour to remove.
        let result = cursor.splice_out_next(head);
        assert!(result.is_none());

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn splice_out_next_preserves_precursor_cache(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        cursor.ensure_precursors(head);
        let (mut boxed, _) = cursor
            .splice_out_next(head)
            .expect("head has a right neighbour");
        boxed.take_value();
        drop(boxed);
        // The cache for the current gap remains valid after the removal.
        assert!(cursor.precursors.is_some());

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: splice_out_at_rank

    #[rstest]
    fn splice_out_at_rank_removes_correct_node(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // Rank 2 is v2 (value 20).
        let (mut boxed, _predecessor) =
            RawCursorMut::<u8, MAX_LEVELS>::splice_out_at_rank(2, head).expect("rank 2 exists");
        assert_eq!(boxed.take_value(), Some(20_u8));
        drop(boxed);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn splice_out_at_rank_out_of_bounds_returns_none(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let result = RawCursorMut::<u8, MAX_LEVELS>::splice_out_at_rank(99, head);
        assert!(result.is_none());

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: insert_at_gap

    #[rstest]
    fn insert_at_gap_without_move_keeps_cursor_at_current(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        cursor.ensure_precursors(head);

        let mut new_node = Node::<u8, MAX_LEVELS>::new(1);
        new_node.value = Some(5_u8);
        let new_ptr = unsafe { Node::insert_after(head, new_node) };
        cursor.insert_at_gap(new_ptr, 1, 1, false);

        // Cursor stays at head (rank 0).
        assert_eq!(cursor.current, head);
        assert_eq!(cursor.current_rank, 0);
        assert!(cursor.precursors.is_some());

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn insert_at_gap_with_move_advances_cursor_to_new_node(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        let mut cursor = RawCursorMut::<u8, MAX_LEVELS>::new(head, 0);
        cursor.ensure_precursors(head);

        let mut new_node = Node::<u8, MAX_LEVELS>::new(1);
        new_node.value = Some(5_u8);
        let new_ptr = unsafe { Node::insert_after(head, new_node) };
        cursor.insert_at_gap(new_ptr, 1, 1, true);

        // Cursor advances to the newly inserted node.
        assert_eq!(cursor.current, new_ptr);
        assert_eq!(cursor.current_rank, 1);
        assert!(cursor.precursors.is_some());

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: gap_find
    //
    // Fixture: head(0) -> v1=10(1) -> v2=20(2) -> v3=30(3) -> v4=40(4)
    // Skip links:
    //   head.links[1] -> v3=30 (distance 3)   ← level-1 shortcut
    //   head.links[0] -> v2=20 (distance 2)
    //   v2.links[0]   -> v3=30 (distance 1)
    //   v3.links[0]   -> v4=40 (distance 1)
    //
    // advance_on_equal = false  →  lower-bound: gap lands BEFORE the first
    //                              node whose value is >= q.
    // advance_on_equal = true   →  upper-bound: gap lands AFTER the last
    //                              node whose value is <= q.

    // ---- advance_on_equal = false (lower-bound) --------------------------------

    #[rstest]
    fn gap_find_lower_bound_below_range(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // q=5 < 10: all skip links point to values >= q, so we never advance.
        // Result: (head, 0).
        let (current, rank) = unsafe { gap_find(head, &5_u8, std::cmp::Ord::cmp, false) };
        assert_eq!(current, head);
        assert_eq!(rank, 0);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn gap_find_lower_bound_on_first_element(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // q=10 == v1: advance_on_equal=false, so we stop before v1.
        // Result: (head, 0).
        let (current, rank) = unsafe { gap_find(head, &10_u8, std::cmp::Ord::cmp, false) };
        assert_eq!(current, head);
        assert_eq!(rank, 0);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn gap_find_lower_bound_between_elements(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // q=15 is between v1=10 and v2=20: skip links all overshoot, so the
        // sequential phase advances past v1 and stops before v2.
        // Result: (v1=10, 1).
        let (current, rank) = unsafe { gap_find(head, &15_u8, std::cmp::Ord::cmp, false) };
        assert_eq!(unsafe { current.as_ref() }.value(), Some(&10_u8));
        assert_eq!(rank, 1);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn gap_find_lower_bound_on_middle_element_via_skip_link(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // q=30 == v3: the level-1 skip link head->v3 is Equal, so it is NOT
        // taken (advance_on_equal=false).  The level-0 link head->v2 is Less,
        // so we land on v2=20, then the sequential phase sees v3=Equal and stops.
        // Result: (v2=20, 2).
        let (current, rank) = unsafe { gap_find(head, &30_u8, std::cmp::Ord::cmp, false) };
        assert_eq!(unsafe { current.as_ref() }.value(), Some(&20_u8));
        assert_eq!(rank, 2);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn gap_find_lower_bound_above_range(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // q=50 > 40: skip links and sequential phase advance all the way to
        // the last node.
        // Result: (v4=40, 4).
        let (current, rank) = unsafe { gap_find(head, &50_u8, std::cmp::Ord::cmp, false) };
        assert_eq!(unsafe { current.as_ref() }.value(), Some(&40_u8));
        assert_eq!(rank, 4);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // ---- advance_on_equal = true (upper-bound) ---------------------------------

    #[rstest]
    fn gap_find_upper_bound_below_range(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // q=5 < 10: nothing to advance past.
        // Result: (head, 0).
        let (current, rank) = unsafe { gap_find(head, &5_u8, std::cmp::Ord::cmp, true) };
        assert_eq!(current, head);
        assert_eq!(rank, 0);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn gap_find_upper_bound_on_first_element(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // q=10 == v1: advance_on_equal=true, so we advance past v1 and stop
        // before v2=20 (Greater).
        // Result: (v1=10, 1).
        let (current, rank) = unsafe { gap_find(head, &10_u8, std::cmp::Ord::cmp, true) };
        assert_eq!(unsafe { current.as_ref() }.value(), Some(&10_u8));
        assert_eq!(rank, 1);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn gap_find_upper_bound_between_elements(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // q=15 is between v1=10 and v2=20: identical result to lower-bound
        // because no element equals q.
        // Result: (v1=10, 1).
        let (current, rank) = unsafe { gap_find(head, &15_u8, std::cmp::Ord::cmp, true) };
        assert_eq!(unsafe { current.as_ref() }.value(), Some(&10_u8));
        assert_eq!(rank, 1);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn gap_find_upper_bound_on_middle_element_via_skip_link(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // q=30 == v3: the level-1 skip link head->v3 is Equal, and
        // advance_on_equal=true, so we jump straight to v3 (rank 3).
        // The level-0 link from v3 points to v4=40 (Greater), so we stop.
        // Result: (v3=30, 3).
        let (current, rank) = unsafe { gap_find(head, &30_u8, std::cmp::Ord::cmp, true) };
        assert_eq!(unsafe { current.as_ref() }.value(), Some(&30_u8));
        assert_eq!(rank, 3);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn gap_find_upper_bound_above_range(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // q=50 > 40: advance all the way to the last node.
        // Result: (v4=40, 4).
        let (current, rank) = unsafe { gap_find(head, &50_u8, std::cmp::Ord::cmp, true) };
        assert_eq!(unsafe { current.as_ref() }.value(), Some(&40_u8));
        assert_eq!(rank, 4);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }
}
