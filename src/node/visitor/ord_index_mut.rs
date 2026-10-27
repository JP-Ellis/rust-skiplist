//! Mutable ordered visitor with rank/distance tracking.

use core::{cmp::Ordering, marker::PhantomData, ptr::NonNull};

use crate::{
    comparator::total_order_violation,
    node::{
        Node,
        level_array::LevelArray,
        visitor::{Visitor, VisitorMut},
    },
};

/// The precursors an [`OrdIndexMutVisitor`] records, with their ranks.
///
/// The caller owns it, so the walk's result outlives the visitor (and the
/// borrow of the target) without a copy out of the visitor.
pub(crate) struct Path<T, const N: usize> {
    /// For each level `l`: the last node at level `l` whose skip-link at
    /// level `l` points to `>= target` (or has no link at that level).
    /// Pre-filled with `head` so that every level has a valid precursor even
    /// when the target is before the first real node.
    precursors: LevelArray<NonNull<Node<T, N>>, N>,
    /// Internal rank at the time each level's precursor was recorded.
    /// Pre-filled with 0 (rank of the head sentinel).
    distances: LevelArray<usize, N>,
}

impl<T, const N: usize> Path<T, N> {
    /// Create a path with `head` as the precursor at every level of `head`.
    ///
    /// # Safety
    ///
    /// `head` must point to a valid node.
    pub(crate) unsafe fn new(head: NonNull<Node<T, N>>) -> Self {
        // SAFETY: the caller guarantees `head` is valid.
        let max_levels = unsafe { head.as_ref() }.level();
        Self {
            precursors: LevelArray::from_fn(max_levels, |_| head),
            distances: LevelArray::from_fn(max_levels, |_| 0_usize),
        }
    }

    /// Extends the path to `height` levels after the head has grown to
    /// match. The head is the precursor, at rank 0, on every new level.
    pub(crate) fn grow(&mut self, head: NonNull<Node<T, N>>, height: usize) {
        self.precursors.grow(height, |_| head);
        self.distances.grow(height, |_| 0);
    }

    /// The precursor at each level.
    pub(crate) fn precursors(&self) -> &[NonNull<Node<T, N>>] {
        &self.precursors
    }

    /// The internal rank of each level's precursor.
    pub(crate) fn distances(&self) -> &[usize] {
        &self.distances
    }

    /// Split into `(precursors, distances)`.
    pub(crate) fn into_parts(self) -> (LevelArray<NonNull<Node<T, N>>, N>, LevelArray<usize, N>) {
        (self.precursors, self.distances)
    }
}

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
/// `rank` is 1-based: the head sentinel has rank 0 and the first data node
/// has rank 1.
///
/// # Precursor distances
///
/// `path.distances()[l]` holds the internal rank at the time the precursor
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
    /// The precursors and their ranks, owned by the caller.
    path: &'a mut Path<T, N>,
    /// The link target of the lowest-level precursor recorded so far, or
    /// `None` while every recorded precursor's link is absent. The traversal
    /// panics rather than advance onto it after a `Less` answer.
    bound: Option<NonNull<Node<T, N>>>,
    /// Current rank during traversal (1-based: 0 = head sentinel, 1 = first
    /// data node). Incremented by `link.distance()` when following a skip
    /// link and by 1 when stepping via the sequential `next` pointer.
    rank: usize,
    /// Suppresses `Send`/`Sync` auto-impls and marks variance as invariant
    /// in `T` (the visitor holds raw mutable pointers into the list).
    _marker: PhantomData<*mut Node<T, N>>,
}

impl<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize>
    OrdIndexMutVisitor<'a, T, Q, F, N>
{
    /// Create a new mutable ordered visitor with rank tracking, starting at
    /// `head` and recording into `path`.
    ///
    /// # Arguments
    ///
    /// - `head`: Raw pointer to the head node of the skip list. The pointer
    ///   must carry mutable (Reserved/Active) provenance so that it can be
    ///   accessed for writes without violating Tree Borrows.
    /// - `path`: A fresh [`Path::new`] for the same `head`.
    /// - `target`: The value to search for.
    /// - `cmp`: Comparator returning `Ordering` for a node value vs. the target.
    ///
    /// # Safety
    ///
    /// `head` must be valid and exclusively accessible for the lifetime `'a`
    /// of the returned visitor.
    pub(crate) fn new(
        head: NonNull<Node<T, N>>,
        path: &'a mut Path<T, N>,
        target: &'a Q,
        cmp: F,
    ) -> Self {
        // SAFETY: caller guarantees head is valid for 'a.
        let max_levels = unsafe { head.as_ref() }.level();
        debug_assert!(
            path.precursors.len() == max_levels,
            "path is not for this head"
        );
        Self {
            current: head,
            level: max_levels,
            found: false,
            target,
            cmp,
            path,
            bound: None,
            rank: 0,
            _marker: PhantomData,
        }
    }

    /// Returns the internal 1-based rank of the current node after traversal.
    ///
    /// This is `self.rank` directly: head sentinel = 0, first data node = 1.
    pub(crate) fn current_rank_internal(&self) -> usize {
        self.rank
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

    // The walk state lives in locals so the hot loop does not reload and
    // store it through `self` on every link.
    //
    // In a well-formed list, `current` lies at or before `bound`, and every
    // recorded precursor's link targets `bound` or later. Refusing a `Less`
    // advance onto `bound` keeps the final position at or before each target.
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        reason = "`level` starts at `max_levels == precursors.len()` and never \
                  grows, so every `l < level` indexes `precursors` and `distances`"
    )]
    fn traverse(&mut self) -> Option<Self::NodeRef> {
        if self.found {
            return Some(self.current);
        }
        let mut current = self.current;
        let mut level = self.level;
        let mut bound = self.bound;
        let mut rank = self.rank;
        let path = &mut *self.path;
        let distances: &mut [usize] = &mut path.distances;
        let cmp = &self.cmp;
        let target = self.target;
        let precursors: &mut [NonNull<Node<T, N>>] = &mut path.precursors;
        let result = 'walk: loop {
            // SAFETY: `current` is the head or a node reached through a link
            // or `next` pointer, so it is live, and the visitor holds the only
            // access to the list.
            let links = unsafe { current.as_ref() }.links();
            for l in (0..level).rev() {
                // A level above this node's tower has no link, so it records
                // a precursor like a link that overshoots.
                let maybe_link = links.get(l).and_then(Option::as_ref);
                if let Some(link) = maybe_link {
                    let node = link.node();
                    // SAFETY: a link targets a live node of the list and never
                    // the head, so the node holds a value.
                    let value = unsafe { Node::value_at(node) };
                    let ord = cmp(value, target);
                    if ord == Ordering::Less {
                        // A link from before `bound` at a level below every
                        // recorded one cannot pass `bound`; reaching it with
                        // `Less` contradicts the answer that recorded it.
                        if Some(node) == bound {
                            total_order_violation();
                        }
                        current = node;
                        // Ranks are bounded by the list length, so these sums never wrap.
                        rank = rank.wrapping_add(link.distance().get());
                        // Reconsider the same level at the new node.
                        level = l.wrapping_add(1);
                        continue 'walk;
                    }
                }
                // The link reaches a node at or past the target, or is
                // absent: `current` is the precursor at level `l`.
                precursors[l] = current;
                distances[l] = rank;
                if let Some(link) = maybe_link {
                    bound = Some(link.node());
                } else {
                    // A node before `bound` has a link at every level below
                    // `bound`'s tower.
                    debug_assert!(bound.is_none(), "absent link below the bound");
                }
            }
            level = 0;
            // SAFETY: `current` is live and the visitor holds the only access
            // to the list.
            let Some(next) = unsafe { current.as_ref() }.next() else {
                break None;
            };
            // SAFETY: a `next` pointer targets a live node of the list and
            // never the head, so the node holds a value.
            let value = unsafe { Node::value_at(next) };
            let ord = cmp(value, target);
            if ord == Ordering::Greater {
                break None;
            }
            if ord == Ordering::Less && Some(next) == bound {
                total_order_violation();
            }
            current = next;
            rank = rank.wrapping_add(1);
            if ord == Ordering::Equal {
                self.found = true;
                break Some(next);
            }
        };
        self.current = current;
        self.level = level;
        self.bound = bound;
        self.rank = rank;
        result
    }
}

impl<T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> VisitorMut
    for OrdIndexMutVisitor<'_, T, Q, F, N>
{
    type Precursor = NonNull<Node<T, N>>;

    fn precursors(&self) -> &[Self::Precursor] {
        &self.path.precursors
    }
}

#[expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test code, covered by miri, so safety guarantees can be relaxed"
)]
#[cfg(test)]
mod tests {
    use core::{cell::Cell, cmp::Ordering, panic::AssertUnwindSafe, ptr::NonNull};
    use std::panic::catch_unwind;

    use anyhow::Result;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::{OrdIndexMutVisitor, Path};
    use crate::{
        comparator::Comparator,
        node::{
            Node,
            tests::{MAX_LEVELS, skiplist, tall_skiplist, well_formed_skiplist},
            visitor::{
                Visitor, VisitorMut,
                tests::{
                    LEVELS, SHAPES, assert_consistent_stop, contains, count_below, list, nodes,
                    outcome, rank_of, scan_precursors, targets,
                },
            },
        },
        test_util::{Chaos, TOTAL_ORDER_PANIC, seeds},
    };

    type Ptr = NonNull<Node<u8, MAX_LEVELS>>;

    /// The values along the base chain, in order.
    fn base_chain(head: Ptr) -> Vec<u8> {
        let mut values = Vec::new();
        let mut current = unsafe { head.as_ref() }.next();
        while let Some(ptr) = current {
            let node = unsafe { ptr.as_ref() };
            values.extend(node.value().copied());
            current = node.next();
        }
        values
    }

    // The fixture builds: head -> v1(10) -> v2(20) -> v3(30) -> v4(40)
    // with skip links: head->v3 (dist 3, level 1), head->v2 (dist 2, level 0),
    //                  v2->v3 (dist 1, level 0), v3->v4 (dist 1, level 0).

    #[rstest]
    fn find_existing_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &30_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        // SAFETY: pointer is valid for the duration of `head`'s lifetime.
        let value = found.map(|ptr| unsafe { ptr.as_ref() }.value().copied());
        assert_eq!(value, Some(Some(30)));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn find_first_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &10_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        let value = found.map(|ptr| unsafe { ptr.as_ref() }.value().copied());
        assert_eq!(value, Some(Some(10)));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn find_last_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &40_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        let value = found.map(|ptr| unsafe { ptr.as_ref() }.value().copied());
        assert_eq!(value, Some(Some(40)));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn value_not_found(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &25_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn value_beyond_list(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &99_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// `rank()` returns the correct 0-based position for each element.
    #[rstest]
    fn rank_first_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &10_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.current_rank_internal().wrapping_sub(1), 0);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn rank_second_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &20_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.current_rank_internal().wrapping_sub(1), 1);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn rank_third_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &30_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.current_rank_internal().wrapping_sub(1), 2);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn rank_last_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &40_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.current_rank_internal().wrapping_sub(1), 3);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// After traversal, every precursor must point to a node whose value is
    /// strictly less than the target (or the head sentinel with no value).
    #[rstest]
    fn precursors_are_before_target(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // Target = 30 (node v3).
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &30_u8, Ord::cmp);
        visitor.traverse();

        for &ptr in visitor.precursors() {
            let value = unsafe { ptr.as_ref() }.value().copied();
            // Value must be None (head) or < 30.
            assert!(
                value.is_none_or(|v| v < 30),
                "precursor value {value:?} should be < 30"
            );
        }
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
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &30_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        let internal_rank = visitor.current_rank_internal();
        for &dist in visitor.path.distances.iter() {
            assert!(
                dist <= internal_rank,
                "precursor_distance {dist} should be <= found internal rank {internal_rank}"
            );
        }
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// A walk past the end stops at the last node.
    #[rstest]
    fn stops_at_tail_when_target_out_of_range(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &99_u8, Ord::cmp);

        assert!(visitor.traverse().is_none());
        assert!(!visitor.found());
        let value = unsafe { visitor.current().as_ref() }.value().copied();
        assert_eq!(value, Some(40));
        assert_eq!(visitor.current_rank_internal(), 4);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// `precursors()` and the precursor distances each have one entry per level.
    #[rstest]
    fn precursors_length(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is valid for the duration of this test.
        let max_levels = unsafe { head.as_ref() }.level();
        let mut path = unsafe { Path::new(head) };
        let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &20_u8, Ord::cmp);
        visitor.traverse();

        assert_eq!(visitor.precursors().len(), max_levels);
        assert_eq!(visitor.path.distances.len(), max_levels);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// A comparator that answers `Greater` the first time it sees `30` and
    /// `Less` on every later call.
    ///
    /// On `well_formed_skiplist` the second call comes from the base chain;
    /// on `tall_skiplist` it comes from a level-0 skip link.
    #[rstest]
    #[case::base_chain(well_formed_skiplist())]
    #[case::skip_link(tall_skiplist())]
    fn contradiction_on_bound_panics(#[case] fixture: Result<(Ptr, Ptr)>) -> Result<()> {
        let (head, tail) = fixture?;
        let calls = Cell::new(0_u32);
        let cmp = |value: &u8, target: &u8| {
            if *value == 30 {
                let seen = calls.get();
                calls.set(seen.saturating_add(1));
                if seen == 0 {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            } else {
                value.cmp(target)
            }
        };

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut path = unsafe { Path::new(head) };
            let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &35_u8, cmp);
            visitor.traverse();
        }));

        let payload = result.expect_err("traversal must panic on the contradiction");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&TOTAL_ORDER_PANIC));
        assert_eq!(calls.get(), 2);
        unsafe { Node::assert_well_formed(head, 4, Some(tail)) };
        assert_eq!(base_chain(head), [10, 20, 30, 40]);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// A consistent comparator never trips the bound, and the result matches
    /// a linear scan for every target, rank included.
    #[rstest]
    #[case::well_formed(well_formed_skiplist())]
    #[case::tall(tall_skiplist())]
    fn consistent_comparator_never_panics(#[case] fixture: Result<(Ptr, Ptr)>) -> Result<()> {
        let (head, _tail) = fixture?;
        let values = base_chain(head);

        let mut outcomes = Vec::new();
        for target in 0_u8..=50 {
            let mut path = unsafe { Path::new(head) };
            let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &target, Ord::cmp);
            visitor.traverse();
            let current = unsafe { visitor.current().as_ref() }.value().copied();
            let outcome = (visitor.found(), current);
            let expected = (
                values.contains(&target),
                values.iter().copied().filter(|&v| v <= target).max(),
            );
            assert_eq!(outcome, expected, "target {target}");
            let rank = values.iter().filter(|&&v| v <= target).count();
            assert_eq!(visitor.current_rank_internal(), rank, "target {target}");
            outcomes.push((target, outcome));
        }

        let spot_checks: Vec<_> = outcomes
            .into_iter()
            .filter(|(target, _)| [5, 10, 25, 30, 45].contains(target))
            .collect();
        assert_eq!(
            spot_checks,
            [
                (5, (false, None)),
                (10, (true, Some(10))),
                (25, (false, Some(20))),
                (30, (true, Some(30))),
                (45, (false, Some(40))),
            ]
        );
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: traverse matches a scan

    type EqPtr = NonNull<Node<u8, LEVELS>>;

    /// Where a walk towards `target` stops: its result, `found`, `current`,
    /// its rank, the precursors and their ranks, or the panic message.
    type Stop = Result<(Option<EqPtr>, bool, EqPtr, usize, Vec<EqPtr>, Vec<usize>), String>;

    fn walk(head: EqPtr, target: u8, cmp: impl Fn(&u8, &u8) -> Ordering) -> Stop {
        outcome(|| {
            let mut path = unsafe { Path::new(head) };
            let mut visitor = OrdIndexMutVisitor::new(head, &mut path, &target, cmp);
            let result = visitor.traverse();
            (
                result,
                visitor.found(),
                visitor.current(),
                visitor.current_rank_internal(),
                visitor.precursors().to_vec(),
                visitor.path.distances.to_vec(),
            )
        })
    }

    /// With a consistent comparator, `traverse` stops on the first node equal
    /// to each target, or after the last node below it, and records as each
    /// level's precursor, with its rank, the last node below the target
    /// whose tower has a link at the level.
    #[test]
    fn traverse_matches_scan() {
        for entries in SHAPES {
            let head = list(entries);
            let nodes = nodes(head);
            for target in targets() {
                let context = format!("shape {entries:?}, target {target}");
                let (result, found, current, rank, precursors, distances) =
                    walk(head, target, Ord::cmp).expect(&context);
                let below = count_below(entries, target);
                let expected_found = contains(entries, target);
                let stop = below.wrapping_add(usize::from(expected_found));
                assert_eq!(found, expected_found, "{context}");
                assert_eq!(
                    result,
                    nodes.get(stop).copied().filter(|_| found),
                    "{context}"
                );
                assert_eq!(rank_of(&nodes, current), stop, "{context}");
                assert_eq!(rank, stop, "{context}");
                let expected = scan_precursors(entries, below);
                let ranks: Vec<usize> = precursors.iter().map(|&p| rank_of(&nodes, p)).collect();
                assert_eq!(ranks, expected, "{context}");
                assert_eq!(distances, expected, "{context}");
            }
            unsafe { drop(Box::from_raw(head.as_ptr())) };
        }
    }

    /// With a chaotic comparator, `traverse` either refuses with the total
    /// order panic or stops at a position consistent with the list's links,
    /// with every rank matching its node.
    #[test]
    fn traverse_under_chaos_stops_consistently() {
        let (mut refused, mut completed) = (0_usize, 0_usize);
        for entries in SHAPES {
            let head = list(entries);
            let nodes = nodes(head);
            for seed in seeds() {
                let chaos = Chaos::new(seed);
                let context = format!("shape {entries:?}, seed {seed}");
                match walk(head, 25, |v, t| chaos.compare(v, t)) {
                    Ok((_, _, current, rank, precursors, distances)) => {
                        assert_consistent_stop(&nodes, current, &precursors, &context);
                        assert_eq!(rank, rank_of(&nodes, current), "{context}");
                        let ranks: Vec<usize> =
                            precursors.iter().map(|&p| rank_of(&nodes, p)).collect();
                        assert_eq!(distances, ranks, "{context}");
                        completed = completed.saturating_add(1);
                    }
                    Err(message) => {
                        assert_eq!(message, TOTAL_ORDER_PANIC, "{context}");
                        refused = refused.saturating_add(1);
                    }
                }
            }
            unsafe { drop(Box::from_raw(head.as_ptr())) };
        }
        assert!(refused > 0, "no seed tripped the bound");
        assert!(completed > 0, "every seed tripped the bound");
    }
}
