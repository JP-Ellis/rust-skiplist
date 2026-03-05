//! Mutable ordered visitor with rank/distance tracking.

use core::{cmp::Ordering, iter, marker::PhantomData, ptr::NonNull};

use arrayvec::ArrayVec;

use crate::node::{
    Node,
    visitor::{Step, Visitor, VisitorMut},
};

/// Mutable ordered visitor that also tracks skip-link distances.
///
/// Traverses the list in value order while recording, for each skip-link
/// level, the last node whose forward link at that level points to a node
/// that is greater than or equal to the target. These "precursor" nodes are
/// the ones whose links must be rewritten during insert or remove.
///
/// In addition to precursors, the visitor tracks the cumulative rank (number
/// of data nodes skipped) so that corrected link distances can be computed
/// after a structural change.
///
/// # Rank convention
///
/// `rank` is 1-based internally: the head sentinel has rank 0 and the first
/// data node has rank 1. The public [`rank`] method subtracts 1 to return
/// the 0-based position. Call it only when [`found`] is `true`.
///
/// # Precursor distances
///
/// `precursor_distances[l]` holds the internal rank at the time the precursor
/// for level `l` was recorded. Combined with the skip-link distance stored in
/// the precursor's link at level `l`, this allows insert and remove to compute
/// corrected link distances after structural changes.
///
/// # Safety
///
/// The caller must guarantee that `head` and all nodes reachable from it
/// remain valid and exclusively accessible for the lifetime of this visitor.
/// All `NonNull` pointers inside it point into the same list.
///
/// [`rank`]: OrdIndexMutVisitor::rank
/// [`found`]: crate::node::visitor::Visitor::found
pub(crate) struct OrdIndexMutVisitor<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> {
    /// Raw pointer to the current node.
    ///
    /// Stored as `NonNull` rather than `&'a mut` to avoid holding an exclusive
    /// reference while `precursors` may alias the same allocation at a
    /// different level.
    current: NonNull<Node<T, N>>,
    /// Highest level still under consideration.
    level: usize,
    /// Whether the target has been found.
    found: bool,
    /// The value being searched for.
    target: &'a Q,
    /// Comparator: `cmp(node_value, target)`.
    cmp: F,
    /// For each level `l`: the last node at level `l` whose skip-link at
    /// level `l` points to `>= target` (or has no link at that level).
    /// Pre-filled with `head` so that every level has a valid precursor even
    /// when the target is before the first real node.
    precursors: ArrayVec<NonNull<Node<T, N>>, N>,
    /// Current rank during traversal (1-based: 0 = head sentinel, 1 = first
    /// data node). Incremented by `link.distance()` when following a skip
    /// link and by 1 when stepping via the sequential `next` pointer.
    rank: usize,
    /// Internal rank at the time each level's precursor was recorded.
    /// Pre-filled with 0 (rank of the head sentinel).
    precursor_distances: ArrayVec<usize, N>,
    /// Suppresses `Send`/`Sync` auto-impls and marks variance as invariant
    /// in `T` (the visitor holds raw mutable pointers into the list).
    _marker: PhantomData<*mut Node<T, N>>,
}

impl<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize>
    OrdIndexMutVisitor<'a, T, Q, F, N>
{
    /// Create a new mutable ordered visitor with rank tracking, starting at
    /// `head`.
    ///
    /// # Arguments
    ///
    /// - `head`: Raw pointer to the head node of the skip list. The pointer
    ///   must carry mutable (Reserved/Active) provenance so that it can be
    ///   accessed for writes without violating Tree Borrows.
    /// - `target`: The value to search for.
    /// - `cmp`: Comparator returning `Ordering` for a node value vs. the target.
    ///
    /// # Safety
    ///
    /// `head` must be valid and exclusively accessible for the lifetime `'a`
    /// of the returned visitor.
    pub(crate) fn new(head: NonNull<Node<T, N>>, target: &'a Q, cmp: F) -> Self {
        // SAFETY: caller guarantees head is valid for 'a.
        let max_levels = unsafe { head.as_ref() }.level();
        let current = head;
        Self {
            current,
            level: max_levels,
            found: false,
            target,
            cmp,
            precursors: iter::repeat_n(current, max_levels).collect(),
            rank: 0,
            precursor_distances: iter::repeat_n(0_usize, max_levels).collect(),
            _marker: PhantomData,
        }
    }

    /// The internal rank at the time each level's precursor was recorded.
    ///
    /// `precursor_distances()[l]` is the 1-based rank at the moment the
    /// precursor for level `l` was recorded. Used by insert and remove to
    /// compute the new skip-link distances after structural changes.
    pub(crate) fn precursor_distances(&self) -> &[usize] {
        &self.precursor_distances
    }

    /// Returns the 0-based rank of the found node.
    ///
    /// This is the number of data nodes that precede the found node in sorted
    /// order (i.e., the 0-based index into the sorted sequence).
    ///
    /// # Precondition
    ///
    /// Only meaningful after [`traverse`] returns `Some(...)`, i.e., when
    /// [`found`] is `true`. The value is `self.rank - 1`, converting
    /// from the internal 1-based tracking to the 0-based external convention.
    ///
    /// [`traverse`]: crate::node::visitor::Visitor::traverse
    /// [`found`]: crate::node::visitor::Visitor::found
    pub(crate) fn rank(&self) -> usize {
        self.rank.saturating_sub(1)
    }

    /// Consume the visitor, releasing the borrow on the list and returning
    /// the internal state as owned values.
    ///
    /// Returns `(current, found, precursors, precursor_distances)` where:
    /// - `current` is the last node advanced to during traversal (the found
    ///   node if `found`, otherwise the last node before the target position).
    /// - `found` is `true` if the target value was present in the list.
    /// - `precursors[l]` is the last node at level `l` before the target.
    /// - `precursor_distances[l]` is the internal rank at that precursor.
    #[expect(clippy::type_complexity, reason = "internal code")]
    pub(crate) fn into_parts(
        self,
    ) -> (
        NonNull<Node<T, N>>,
        bool,
        ArrayVec<NonNull<Node<T, N>>, N>,
        ArrayVec<usize, N>,
    ) {
        (
            self.current,
            self.found,
            self.precursors,
            self.precursor_distances,
        )
    }
}

impl<T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> Visitor
    for OrdIndexMutVisitor<'_, T, Q, F, N>
{
    /// Node references are raw const pointers to avoid lifetime conflicts with
    /// the `&mut` borrow. Callers that need a shared reference can convert
    /// via [`NonNull::as_ref`] inside an `unsafe` block.
    type NodeRef = NonNull<Node<T, N>>;

    fn current(&self) -> Self::NodeRef {
        self.current
    }

    fn level(&self) -> usize {
        self.level
    }

    fn found(&self) -> bool {
        self.found
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "`level` comes from `(0..self.level).rev()` where `self.level` is \
                  initialised to `max_levels = precursors.len()` and only decreases, \
                  so `level < max_levels == precursors.len()` is always true"
    )]
    fn step(&mut self) -> Step<Self::NodeRef> {
        if self.found {
            return Step::FoundTarget;
        }

        // Borrow only the links slice for the duration of the for loop, then
        // drop the reference so `self.current` can be re-borrowed for `next`.
        {
            // SAFETY: `self.current` was obtained from a valid `&mut Node<T, N>`
            // during construction or from a link/next pointer during traversal.
            // No other `&mut` to the same node exists while we hold `self`.
            let current_ref: &Node<T, N> = unsafe { self.current.as_ref() };
            let links = current_ref.links();

            for level in (0..self.level).rev() {
                // Use `.get()` rather than `.zip()` so that levels beyond
                // this node's actual height are treated as `None` (overshoot),
                // not silently skipped: every level must get a precursor
                // recorded.
                let maybe_link = links.get(level).and_then(|l| l.as_ref());

                if let Some(link) = maybe_link {
                    let node_ptr = link.node();
                    // Treat a value-less node (head sentinel) as strictly less
                    // than any target. In a well-formed skip list every
                    // skip-link target has a value, so this is defensive.
                    // SAFETY: node_ptr is a valid heap-allocated node.
                    let ord = unsafe { node_ptr.as_ref() }
                        .value()
                        .map_or(Ordering::Less, |v| (self.cmp)(v, self.target));

                    if ord == Ordering::Less {
                        // This link advances us strictly closer to the target.
                        // Do NOT record a precursor here: the current node's
                        // link[level] points to < target, so it is not the
                        // precursor for this level. We will record it later
                        // if the same level overshoots at the new node.
                        self.current = node_ptr;
                        self.rank = self.rank.saturating_add(link.distance().get());
                        // Use level + 1 so the same level is reconsidered at
                        // the new node (mirrors the fix in `IndexVisitor`).
                        self.level = level.saturating_add(1);
                        return Step::Advanced(self.current);
                    }
                }

                // Link overshoots the target (Greater), points exactly to the
                // target (Equal), or is absent (node has no link at this
                // level): current is the precursor for this level.
                self.precursors[level] = self.current;
                self.precursor_distances[level] = self.rank;
            }
        }

        // No skip-link can advance us; fall back to the sequential next pointer.
        self.level = 0;
        // SAFETY: `self.current` is valid for `'a` and no other `&mut` exists.
        if let Some(next_nn) = unsafe { self.current.as_ref() }.next() {
            // SAFETY: next_nn is a valid heap-allocated node.
            let ord = unsafe { next_nn.as_ref() }
                .value()
                .map_or(Ordering::Less, |v| (self.cmp)(v, self.target));
            if ord == Ordering::Greater {
                // The next sequential node is already past the target; the
                // target is absent from the list.
                return Step::Exhausted;
            }
            self.current = next_nn;
            self.rank = self.rank.saturating_add(1);
            self.found = ord == Ordering::Equal;
            return Step::Advanced(self.current);
        }

        Step::Exhausted
    }
}

impl<T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> VisitorMut
    for OrdIndexMutVisitor<'_, T, Q, F, N>
{
    type NodeMut = NonNull<Node<T, N>>;
    type Precursor = NonNull<Node<T, N>>;

    fn current_mut(&mut self) -> Self::NodeMut {
        self.current
    }

    fn precursors(&self) -> &[Self::Precursor] {
        &self.precursors
    }
}

#[expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test code, covered by miri, so safety guarantees can be relaxed"
)]
#[cfg(test)]
mod tests {
    use core::ptr::NonNull;

    use anyhow::Result;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::OrdIndexMutVisitor;
    use crate::node::{
        Node,
        tests::{MAX_LEVELS, skiplist},
        visitor::{Step, Visitor, VisitorMut},
    };

    // The fixture builds: head -> v1(10) -> v2(20) -> v3(30) -> v4(40)
    // with skip links: head->v3 (dist 3, level 1), head->v2 (dist 2, level 0),
    //                  v2->v3 (dist 1, level 0), v3->v4 (dist 1, level 0).

    #[rstest]
    fn find_existing_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexMutVisitor::new(head, &30_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        // SAFETY: pointer is valid for the duration of `head`'s lifetime.
        let value = found.map(|ptr| unsafe { ptr.as_ref() }.value().copied());
        assert_eq!(value, Some(Some(30)));
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn find_first_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexMutVisitor::new(head, &10_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        let value = found.map(|ptr| unsafe { ptr.as_ref() }.value().copied());
        assert_eq!(value, Some(Some(10)));
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn find_last_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexMutVisitor::new(head, &40_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        let value = found.map(|ptr| unsafe { ptr.as_ref() }.value().copied());
        assert_eq!(value, Some(Some(40)));
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn value_not_found(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexMutVisitor::new(head, &25_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn value_beyond_list(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexMutVisitor::new(head, &99_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// `rank()` returns the correct 0-based position for each element.
    #[rstest]
    fn rank_first_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexMutVisitor::new(head, &10_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.rank(), 0);
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn rank_second_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexMutVisitor::new(head, &20_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.rank(), 1);
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn rank_third_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexMutVisitor::new(head, &30_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.rank(), 2);
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn rank_last_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexMutVisitor::new(head, &40_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.rank(), 3);
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// After traversal, every precursor must point to a node whose value is
    /// strictly less than the target (or the head sentinel with no value).
    #[rstest]
    fn precursors_are_before_target(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // Target = 30 (node v3).
        let mut visitor = OrdIndexMutVisitor::new(head, &30_u8, Ord::cmp);
        while let Step::Advanced(_) = visitor.step() {}

        for &ptr in visitor.precursors() {
            let value = unsafe { ptr.as_ref() }.value().copied();
            // Value must be None (head) or < 30.
            assert!(
                value.is_none_or(|v| v < 30),
                "precursor value {value:?} should be < 30"
            );
        }
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// `precursor_distances` are each <= the internal rank of the found node
    /// (they record the rank at a node strictly before the target).
    #[rstest]
    fn precursor_distances_at_most_found_rank(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;
        // Fixture ranks: v1=0, v2=1, v3=2, v4=3 (0-based).
        // Internal ranks: head=0, v1=1, v2=2, v3=3, v4=4.
        // Searching for v3 (30): found at internal rank 3.
        // precursor_distances should all be <= 3.
        let mut visitor = OrdIndexMutVisitor::new(head, &30_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        // Internal rank of the found node = rank() + 1 (converting 0-based back to 1-based).
        let internal_rank = visitor.rank().saturating_add(1);
        for &dist in visitor.precursor_distances() {
            assert!(
                dist <= internal_rank,
                "precursor_distance {dist} should be <= found internal rank {internal_rank}"
            );
        }
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// Stepping past the end of the list produces `Exhausted`.
    #[rstest]
    fn exhausted_when_target_out_of_range(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexMutVisitor::new(head, &99_u8, Ord::cmp);

        loop {
            let s = visitor.step();
            match s {
                Step::Advanced(_) => (),
                Step::Exhausted => {
                    assert!(!visitor.found());
                    break;
                }
                Step::FoundTarget => panic!("should not find target 99"),
            }
        }
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// `current_mut()` returns the same pointer as `current()`.
    #[rstest]
    fn current_mut_matches_current(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexMutVisitor::new(head, &20_u8, Ord::cmp);
        visitor.traverse();

        assert_eq!(visitor.current(), visitor.current_mut());
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// `precursors()` and `precursor_distances()` each have one entry per level.
    #[rstest]
    fn precursors_length(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is valid for the duration of this test.
        let max_levels = unsafe { head.as_ref() }.level();
        let mut visitor = OrdIndexMutVisitor::new(head, &20_u8, Ord::cmp);
        visitor.traverse();

        assert_eq!(visitor.precursors().len(), max_levels);
        assert_eq!(visitor.precursor_distances().len(), max_levels);
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }
}
