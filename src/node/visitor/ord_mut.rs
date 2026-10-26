//! Mutable ordered visitor.

use core::{cmp::Ordering, marker::PhantomData, ptr::NonNull};

use crate::{
    comparator::total_order_violation,
    node::{
        Node,
        level_array::LevelArray,
        visitor::{Visitor, VisitorMut},
    },
};

/// Mutable ordered visitor.
///
/// Traverses a skip list whose nodes are sorted by some ordering, stopping at
/// the first node that compares equal to the target, while recording the
/// *precursor* nodes needed to update skip-links during insert or remove.
///
/// A caller-supplied comparator `F` is used instead of `T: Ord` so that
/// `SkipMap` can search by key and `insert_by` / `remove_by` style APIs can
/// supply their own orderings.
///
/// # Precursors
///
/// During traversal the visitor maintains one precursor per level: for each
/// level `l`, `precursors[l]` is the last node whose skip-link at level `l`
/// either points to the target position or would overshoot it. After
/// traversal these are the nodes whose links must be rewritten on insert or
/// remove.
///
/// Unlike `OrdIndexMutVisitor`, no distance information is stored alongside
/// the precursors. Operations that use this visitor (e.g. `split_off`) only
/// need to relink nodes; they do not need to update skip-link distances or
/// answer rank queries, so the extra tracking is omitted.
///
/// # Safety
///
/// The caller must guarantee that `head` and all nodes reachable from it
/// remain valid and exclusively accessible for the lifetime of this visitor.
/// All `NonNull` pointers inside it point into the same list.
pub(crate) struct OrdMutVisitor<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> {
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
    precursors: LevelArray<NonNull<Node<T, N>>, N>,
    /// The link target of the lowest-level precursor recorded so far, or
    /// `None` while every recorded precursor's link is absent. The traversal
    /// panics rather than advance onto it after a `Less` answer.
    bound: Option<NonNull<Node<T, N>>>,
    /// Suppresses `Send`/`Sync` auto-impls and marks variance as invariant in
    /// `T` (the visitor holds raw mutable pointers into the list).
    _marker: PhantomData<*mut Node<T, N>>,
}

impl<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> OrdMutVisitor<'a, T, Q, F, N> {
    /// Create a new mutable ordered visitor starting at `head`.
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
            // Every level starts with `head` as its precursor: the head is
            // always before every real node, so it is a valid precursor for
            // any target value.
            precursors: LevelArray::from_fn(max_levels, |_| current),
            bound: None,
            _marker: PhantomData,
        }
    }

    /// Consume the visitor, releasing the borrow on the list and returning
    /// the internal state as owned values.
    ///
    /// Returns `(current, found, precursors)` where:
    /// - `current` is the last node advanced to during traversal (the found
    ///   node if `found`, otherwise the last node before the target position).
    /// - `found` is `true` if the target value was present in the list.
    /// - `precursors[l]` is the last node at level `l` before the target.
    #[expect(clippy::type_complexity, reason = "internal code")]
    pub(crate) fn into_parts(
        self,
    ) -> (
        NonNull<Node<T, N>>,
        bool,
        LevelArray<NonNull<Node<T, N>>, N>,
    ) {
        (self.current, self.found, self.precursors)
    }
}

impl<T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> Visitor
    for OrdMutVisitor<'_, T, Q, F, N>
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
                  grows, so every `l < level` indexes `precursors`"
    )]
    fn traverse(&mut self) -> Option<Self::NodeRef> {
        if self.found {
            return Some(self.current);
        }
        let mut current = self.current;
        let mut level = self.level;
        let mut bound = self.bound;
        let cmp = &self.cmp;
        let target = self.target;
        let precursors: &mut [NonNull<Node<T, N>>] = &mut self.precursors;
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
                    // A value-less node compares `Less`. Links never target
                    // the head, so this is defensive.
                    // SAFETY: a link targets a live node of the list.
                    let ord = unsafe { node.as_ref() }
                        .value()
                        .map_or(Ordering::Less, |v| cmp(v, target));
                    if ord == Ordering::Less {
                        // A link from before `bound` at a level below every
                        // recorded one cannot pass `bound`; reaching it with
                        // `Less` contradicts the answer that recorded it.
                        if Some(node) == bound {
                            total_order_violation();
                        }
                        current = node;
                        // Reconsider the same level at the new node.
                        level = l.wrapping_add(1);
                        continue 'walk;
                    }
                }
                // The link reaches a node at or past the target, or is
                // absent: `current` is the precursor at level `l`.
                precursors[l] = current;
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
            // SAFETY: a `next` pointer targets a live node of the list.
            let ord = unsafe { next.as_ref() }
                .value()
                .map_or(Ordering::Less, |v| cmp(v, target));
            if ord == Ordering::Greater {
                break None;
            }
            if ord == Ordering::Less && Some(next) == bound {
                total_order_violation();
            }
            current = next;
            if ord == Ordering::Equal {
                self.found = true;
                break Some(next);
            }
        };
        self.current = current;
        self.level = level;
        self.bound = bound;
        result
    }
}

impl<T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> VisitorMut
    for OrdMutVisitor<'_, T, Q, F, N>
{
    type Precursor = NonNull<Node<T, N>>;

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
    use core::{cell::Cell, cmp::Ordering, panic::AssertUnwindSafe, ptr::NonNull};
    use std::panic::catch_unwind;

    use anyhow::Result;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::OrdMutVisitor;
    use crate::{
        comparator::Comparator,
        node::{
            Node,
            tests::{MAX_LEVELS, skiplist, tall_skiplist, well_formed_skiplist},
            visitor::{
                Visitor, VisitorMut,
                tests::{
                    LEVELS, SHAPES, assert_consistent_stop, contains, count_below, list, nodes,
                    outcome, rank_of, scan_precursors,
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

    #[rstest]
    fn find_existing_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdMutVisitor::new(head, &30_u8, Ord::cmp);

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
        let mut visitor = OrdMutVisitor::new(head, &10_u8, Ord::cmp);

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
        let mut visitor = OrdMutVisitor::new(head, &40_u8, Ord::cmp);

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
        let mut visitor = OrdMutVisitor::new(head, &25_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn value_beyond_list(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdMutVisitor::new(head, &99_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// After traversal, every precursor must point to a node whose value is
    /// strictly less than the target (or the head sentinel with no value).
    #[rstest]
    fn precursors_are_before_target(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // Target = 30 (node v3).
        let mut visitor = OrdMutVisitor::new(head, &30_u8, Ord::cmp);
        visitor.traverse();

        for &ptr in visitor.precursors() {
            let value = unsafe { ptr.as_ref() }.value().copied();
            assert!(
                value.is_none_or(|v| v < 30),
                "precursor value {value:?} should be < 30"
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
        let mut visitor = OrdMutVisitor::new(head, &99_u8, Ord::cmp);

        assert!(visitor.traverse().is_none());
        assert!(!visitor.found());
        let value = unsafe { visitor.current().as_ref() }.value().copied();
        assert_eq!(value, Some(40));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// `precursors()` has one entry per level.
    #[rstest]
    fn precursors_length(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is valid for the duration of this test.
        let max_levels = unsafe { head.as_ref() }.level();
        let mut visitor = OrdMutVisitor::new(head, &20_u8, Ord::cmp);
        visitor.traverse();

        assert_eq!(visitor.precursors().len(), max_levels);
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
            let mut visitor = OrdMutVisitor::new(head, &35_u8, cmp);
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
    /// a linear scan for every target.
    #[rstest]
    #[case::well_formed(well_formed_skiplist())]
    #[case::tall(tall_skiplist())]
    fn consistent_comparator_never_panics(#[case] fixture: Result<(Ptr, Ptr)>) -> Result<()> {
        let (head, _tail) = fixture?;
        let values = base_chain(head);

        let mut outcomes = Vec::new();
        for target in 0_u8..=50 {
            let mut visitor = OrdMutVisitor::new(head, &target, Ord::cmp);
            visitor.traverse();
            let current = unsafe { visitor.current().as_ref() }.value().copied();
            let outcome = (visitor.found(), current);
            let expected = (
                values.contains(&target),
                values.iter().copied().filter(|&v| v <= target).max(),
            );
            assert_eq!(outcome, expected, "target {target}");
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

    /// Where a walk towards `target` stops: its result, `found`, `current`
    /// and precursors, or the panic message.
    type Stop = Result<(Option<EqPtr>, bool, EqPtr, Vec<EqPtr>), String>;

    fn walk(head: EqPtr, target: u8, cmp: impl Fn(&u8, &u8) -> Ordering) -> Stop {
        outcome(|| {
            let mut visitor = OrdMutVisitor::new(head, &target, cmp);
            let result = visitor.traverse();
            (
                result,
                visitor.found(),
                visitor.current(),
                visitor.precursors().to_vec(),
            )
        })
    }

    /// With a consistent comparator, `traverse` stops on the first node equal
    /// to each target, or after the last node below it, and records as each
    /// level's precursor the last node below the target whose tower has a
    /// link at the level.
    #[test]
    fn traverse_matches_scan() {
        for entries in SHAPES {
            let head = list(entries);
            let nodes = nodes(head);
            for target in 0_u8..=95 {
                let context = format!("shape {entries:?}, target {target}");
                let (result, found, current, precursors) =
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
                let ranks: Vec<usize> = precursors.iter().map(|&p| rank_of(&nodes, p)).collect();
                assert_eq!(ranks, scan_precursors(entries, below), "{context}");
            }
            unsafe { drop(Box::from_raw(head.as_ptr())) };
        }
    }

    /// With a chaotic comparator, `traverse` either refuses with the total
    /// order panic or stops at a position consistent with the list's links.
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
                    Ok((_, _, current, precursors)) => {
                        assert_consistent_stop(&nodes, current, &precursors, &context);
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
