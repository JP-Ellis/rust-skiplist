//! Mutable ordered visitor.

use core::{cmp::Ordering, marker::PhantomData, ptr::NonNull};

use crate::{
    comparator::total_order_violation,
    node::{
        Node,
        level_array::LevelArray,
        visitor::{Step, Visitor, VisitorMut},
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

    // Runs `step` to completion with the walk state held in locals, so the
    // hot loop does not reload and store it through `self` on every link.
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        reason = "`l < level <= max_levels == precursors.len()`, as in `step`"
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
            // SAFETY: as in `step`.
            let links = unsafe { current.as_ref() }.links();
            for l in (0..level).rev() {
                let maybe_link = links.get(l).and_then(Option::as_ref);
                if let Some(link) = maybe_link {
                    let node = link.node();
                    // SAFETY: as in `step`.
                    let ord = unsafe { node.as_ref() }
                        .value()
                        .map_or(Ordering::Less, |v| cmp(v, target));
                    if ord == Ordering::Less {
                        if Some(node) == bound {
                            total_order_violation();
                        }
                        current = node;
                        level = l.wrapping_add(1);
                        continue 'walk;
                    }
                }
                precursors[l] = current;
                if let Some(link) = maybe_link {
                    bound = Some(link.node());
                } else {
                    debug_assert!(bound.is_none(), "absent link below the bound");
                }
            }
            level = 0;
            // SAFETY: as in `step`.
            let Some(next) = unsafe { current.as_ref() }.next() else {
                break None;
            };
            // SAFETY: as in `step`.
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

    // In a well-formed list, `current` lies at or before `bound`, and every
    // recorded precursor's link targets `bound` or later. Refusing a `Less`
    // advance onto `bound` keeps the final position at or before each target.
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
                        // A link from before `bound` at a level below every
                        // recorded one cannot pass `bound`; reaching it with
                        // `Less` contradicts the answer that recorded it.
                        if Some(node_ptr) == self.bound {
                            total_order_violation();
                        }
                        // This link advances us strictly closer to the target.
                        // Do NOT record a precursor here: the current node's
                        // link[level] points to < target, so it is not the
                        // precursor for this level. We will record it later
                        // if the same level overshoots at the new node.
                        self.current = node_ptr;
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
                if let Some(link) = maybe_link {
                    self.bound = Some(link.node());
                } else {
                    // A node before `bound` always has a link at every level
                    // below `bound`'s tower.
                    debug_assert!(self.bound.is_none(), "absent link below the bound");
                }
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
            if ord == Ordering::Less && Some(next_nn) == self.bound {
                total_order_violation();
            }
            self.current = next_nn;
            self.found = ord == Ordering::Equal;
            return Step::Advanced(self.current);
        }

        Step::Exhausted
    }
}

impl<T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> VisitorMut
    for OrdMutVisitor<'_, T, Q, F, N>
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
    use core::{
        cell::{Cell, RefCell},
        cmp::Ordering,
        panic::AssertUnwindSafe,
        ptr::NonNull,
    };
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
                Step, Visitor, VisitorMut,
                tests::{LEVELS, SHAPES, list, outcome, step_to_end},
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
        while let Step::Advanced(_) = visitor.step() {}

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

    /// Stepping past the end of the list produces `Exhausted`.
    #[rstest]
    fn exhausted_when_target_out_of_range(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdMutVisitor::new(head, &99_u8, Ord::cmp);

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
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// `current_mut()` returns the same pointer as `current()`.
    #[rstest]
    fn current_mut_matches_current(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdMutVisitor::new(head, &20_u8, Ord::cmp);
        visitor.traverse();

        assert_eq!(visitor.current(), visitor.current_mut());
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

    // MARK: traverse matches step

    type EqPtr = NonNull<Node<u8, LEVELS>>;

    /// The state of an `OrdMutVisitor` once it stops.
    #[derive(Debug, PartialEq, Eq)]
    struct Snapshot {
        result: Option<EqPtr>,
        current: EqPtr,
        level: usize,
        found: bool,
        bound: Option<EqPtr>,
        precursors: Vec<EqPtr>,
    }

    impl Snapshot {
        fn new<F>(visitor: &OrdMutVisitor<'_, u8, u8, F, LEVELS>, result: Option<EqPtr>) -> Self
        where
            F: Fn(&u8, &u8) -> Ordering,
        {
            Self {
                result,
                current: visitor.current,
                level: visitor.level,
                found: visitor.found(),
                bound: visitor.bound,
                precursors: visitor.precursors().to_vec(),
            }
        }
    }

    /// Walks `head` towards `target` with `step` or `traverse`.
    ///
    /// Returns the final state, or the panic message, together with every
    /// value handed to the comparator in call order.
    fn walk(
        head: EqPtr,
        target: u8,
        use_traverse: bool,
        cmp: impl Fn(&u8, &u8) -> Ordering,
    ) -> (Result<Snapshot, String>, Vec<u8>) {
        let calls = RefCell::new(Vec::new());
        let logged = |value: &u8, probe: &u8| {
            calls.borrow_mut().push(*value);
            cmp(value, probe)
        };
        let state = outcome(|| {
            let mut visitor = OrdMutVisitor::new(head, &target, logged);
            let result = if use_traverse {
                visitor.traverse()
            } else {
                step_to_end(&mut visitor)
            };
            Snapshot::new(&visitor, result)
        });
        (state, calls.into_inner())
    }

    /// With a consistent comparator, `traverse` makes the same comparisons
    /// and ends in the same state as the `step` loop for every target
    /// before, between, on and after the values of each shape.
    #[test]
    fn traverse_matches_step() {
        for entries in SHAPES {
            let head = list(entries);
            for target in 0_u8..=95 {
                let stepped = walk(head, target, false, Ord::cmp);
                let traversed = walk(head, target, true, Ord::cmp);
                assert!(stepped.0.is_ok(), "shape {entries:?}, target {target}");
                assert_eq!(traversed, stepped, "shape {entries:?}, target {target}");
            }
            unsafe { drop(Box::from_raw(head.as_ptr())) };
        }
    }

    /// With a chaotic comparator, `traverse` panics exactly when the `step`
    /// loop does, after the same comparisons, and otherwise ends in the same
    /// state.
    #[test]
    fn traverse_matches_step_under_chaos() {
        let (mut refused, mut completed) = (0_usize, 0_usize);
        for entries in SHAPES {
            let head = list(entries);
            for seed in seeds() {
                let (step_chaos, traverse_chaos) = (Chaos::new(seed), Chaos::new(seed));
                let stepped = walk(head, 25, false, |v, t| step_chaos.compare(v, t));
                let traversed = walk(head, 25, true, |v, t| traverse_chaos.compare(v, t));
                assert_eq!(traversed, stepped, "shape {entries:?}, seed {seed}");
                match stepped.0 {
                    Ok(_) => completed = completed.saturating_add(1),
                    Err(message) => {
                        assert_eq!(message, TOTAL_ORDER_PANIC);
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
