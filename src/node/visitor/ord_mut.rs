//! Mutable ordered visitor.

use core::{cmp::Ordering, iter, marker::PhantomData, ptr::NonNull};

use arrayvec::ArrayVec;

use crate::node::{
    Node,
    visitor::{Step, Visitor, VisitorMut},
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
/// Unlike `IndexMutVisitor`, no distance information is stored alongside the
/// precursors because ordered collections do not need to maintain link
/// distances (they do not support O(1) rank queries).
///
/// # Safety
///
/// The visitor borrows `head` mutably for its lifetime `'a`.  All `NonNull`
/// pointers inside it point into the same list and are valid for `'a`.
struct OrdMutVisitor<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> {
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
    /// Ties the raw pointer lifetime to `'a`.
    _marker: PhantomData<&'a mut Node<T, N>>,
}

impl<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> OrdMutVisitor<'a, T, Q, F, N> {
    /// Create a new mutable ordered visitor starting at `head`.
    ///
    /// # Arguments
    ///
    /// - `head`: The head node of the skip list.
    /// - `target`: The value to search for.
    /// - `cmp`: Comparator returning `Ordering` for a node value vs. the target.
    fn new(head: &'a mut Node<T, N>, target: &'a Q, cmp: F) -> Self {
        let max_levels = head.level();
        let current = NonNull::from_mut(head);
        Self {
            current,
            level: max_levels,
            found: false,
            target,
            cmp,
            // Every level starts with `head` as its precursor: the head is
            // always before every real node, so it is a valid precursor for
            // any target value.
            precursors: iter::repeat_n(current, max_levels).collect(),
            _marker: PhantomData,
        }
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
    use core::ptr::NonNull;

    use anyhow::Result;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::OrdMutVisitor;
    use crate::node::{
        Node,
        tests::{MAX_LEVELS, skiplist},
        visitor::{Step, Visitor, VisitorMut},
    };

    #[rstest]
    fn find_existing_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let mut head = skiplist?;
        let mut visitor = OrdMutVisitor::new(unsafe { head.as_mut() }, &30_u8, Ord::cmp);

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
        let mut head = skiplist?;
        let mut visitor = OrdMutVisitor::new(unsafe { head.as_mut() }, &10_u8, Ord::cmp);

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
        let mut head = skiplist?;
        let mut visitor = OrdMutVisitor::new(unsafe { head.as_mut() }, &40_u8, Ord::cmp);

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
        let mut head = skiplist?;
        let mut visitor = OrdMutVisitor::new(unsafe { head.as_mut() }, &25_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn value_beyond_list(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let mut head = skiplist?;
        let mut visitor = OrdMutVisitor::new(unsafe { head.as_mut() }, &99_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// After traversal, every precursor must point to a node whose value is
    /// strictly less than the target (or the head sentinel with no value).
    #[rstest]
    fn precursors_are_before_target(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let mut head = skiplist?;
        // Target = 30 (node v3).
        let mut visitor = OrdMutVisitor::new(unsafe { head.as_mut() }, &30_u8, Ord::cmp);
        while let Step::Advanced(_) = visitor.step() {}

        for &ptr in visitor.precursors() {
            let value = unsafe { ptr.as_ref() }.value().copied();
            assert!(
                value.is_none_or(|v| v < 30),
                "precursor value {value:?} should be < 30"
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
        let mut head = skiplist?;
        let mut visitor = OrdMutVisitor::new(unsafe { head.as_mut() }, &99_u8, Ord::cmp);

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
        let mut head = skiplist?;
        let mut visitor = OrdMutVisitor::new(unsafe { head.as_mut() }, &20_u8, Ord::cmp);
        visitor.traverse();

        assert_eq!(visitor.current(), visitor.current_mut());
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// `precursors()` has one entry per level.
    #[rstest]
    fn precursors_length(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let mut head = skiplist?;
        // SAFETY: head is valid for the duration of this test.
        let max_levels = unsafe { head.as_ref() }.level();
        let mut visitor = OrdMutVisitor::new(unsafe { head.as_mut() }, &20_u8, Ord::cmp);
        visitor.traverse();

        assert_eq!(visitor.precursors().len(), max_levels);
        drop(visitor);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }
}
