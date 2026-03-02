//! Mutable index-based visitor.

use core::{iter, marker::PhantomData, ptr::NonNull};

use arrayvec::ArrayVec;

use crate::node::{
    Node,
    visitor::{Step, Visitor, VisitorMut},
};

/// Mutable index-based visitor.
///
/// This visitor is used to find a node by its index and simultaneously record
/// the *precursor* nodes needed to update skip-links when inserting or removing
/// a node at that position.
///
/// # Precursors
///
/// During traversal, the visitor maintains one precursor per level: for each
/// level `l`, `precursors[l]` is the last node whose skip-link at level `l`
/// either points exactly to the target position or would overshoot it.  After
/// traversal these are the nodes whose links must be rewritten.
///
/// # Safety
///
/// The visitor borrows `head` mutably for its lifetime `'a`.  All `NonNull`
/// pointers inside it point into the same list and are valid for `'a`.
pub(crate) struct IndexMutVisitor<'a, T, const N: usize> {
    /// Raw pointer to the current node.
    ///
    /// Stored as `NonNull` rather than `&'a mut` to avoid holding an exclusive
    /// reference while `precursors` may alias the same allocation at a
    /// different level.
    current: NonNull<Node<T, N>>,
    /// 0-based index of the current node within the list.
    index: usize,
    /// Highest level still under consideration.
    level: usize,
    /// Target index to find.
    target: usize,
    /// For each level `l`: the last node at level `l` whose skip-link at
    /// level `l` points to `>= target` (or has no link at that level).
    /// Pre-filled with `head` so that every level has a valid precursor even
    /// when the target is before the first real node.
    precursors: ArrayVec<NonNull<Node<T, N>>, N>,
    /// For each level `l`: the cumulative distance already traversed at level
    /// `l` when the precursor at that level was recorded.  Together with the
    /// precursor's link distance this is sufficient to compute the new link
    /// distances after insertion or removal.
    precursor_distances: ArrayVec<usize, N>,
    /// Ties the raw pointer lifetime to `'a`.
    _marker: PhantomData<&'a mut Node<T, N>>,
}

impl<'a, T, const N: usize> IndexMutVisitor<'a, T, N> {
    /// Create a new mutable index visitor starting at `head`.
    ///
    /// # Arguments
    ///
    /// - `head`: The head node of the skip list.
    /// - `target`: The 0-based index to locate.
    pub(crate) fn new(head: &'a mut Node<T, N>, target: usize) -> Self {
        let max_levels = head.level();
        let current = NonNull::from(&*head);
        Self {
            current,
            index: 0,
            level: max_levels,
            target,
            // Every level starts with `head` as its precursor: the head is
            // always before every real node, so it is a valid precursor for
            // any target index.
            precursors: iter::repeat_n(current, max_levels).collect(),
            precursor_distances: iter::repeat_n(0, max_levels).collect(),
            _marker: PhantomData,
        }
    }

    /// The distances from each precursor to the target position.
    ///
    /// `precursor_distances()[l]` is the total distance traversed at level
    /// `l` before the precursor at that level was recorded.  Used by insert
    /// and remove operations to compute new link distances.
    pub(crate) fn precursor_distances(&self) -> &[usize] {
        &self.precursor_distances
    }

    /// Consume the visitor, releasing the `&mut` borrow on the head node and
    /// returning the internal state as owned values.
    ///
    /// Returns `(current, precursors, precursor_distances)` where:
    /// - `current` is the last node advanced to during traversal (the tail node
    ///   for an exhausted traversal, or the target node if found).
    /// - `precursors[l]` is the last node at level `l` before the target.
    /// - `precursor_distances[l]` is the cumulative rank of that precursor.
    #[expect(clippy::type_complexity, reason = "internal code")]
    pub(crate) fn into_parts(
        self,
    ) -> (
        NonNull<Node<T, N>>,
        ArrayVec<NonNull<Node<T, N>>, N>,
        ArrayVec<usize, N>,
    ) {
        (self.current, self.precursors, self.precursor_distances)
    }
}

impl<T, const N: usize> Visitor for IndexMutVisitor<'_, T, N> {
    /// Node references are raw const pointers to avoid lifetime conflicts with
    /// the `&mut` borrow.  Callers that need a shared reference can convert
    /// via [`NonNull::as_ref`] inside an `unsafe` block.
    type NodeRef = NonNull<Node<T, N>>;

    fn current(&self) -> Self::NodeRef {
        self.current
    }

    fn level(&self) -> usize {
        self.level
    }

    fn found(&self) -> bool {
        self.index == self.target
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "`level` comes from `(0..self.level).rev()` where `self.level` is \
                  initialised to `max_levels = precursors.len()` and only decreases, \
                  so `level < max_levels == precursors.len()` is always true"
    )]
    fn step(&mut self) -> Step<Self::NodeRef> {
        if self.found() {
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
                    let next_index = self.index.saturating_add(link.distance().get());
                    if next_index < self.target {
                        // This link advances us strictly closer to the target.
                        // Do NOT record a precursor here: the current node's
                        // link[level] points to < target, so it is not the
                        // precursor for this level.  We will record it later
                        // if the same level overshoots at the new node.
                        self.current = NonNull::from(link.node());
                        // Use level + 1 so the same level is reconsidered at
                        // the new node (mirrors the fix in `IndexVisitor`).
                        self.level = level.saturating_add(1);
                        self.index = next_index;
                        return Step::Advanced(self.current);
                    }
                }

                // Link overshoots the target (next_index >= target), points
                // exactly to the target (next_index == target), or is absent
                // (node has no link at this level): current is the precursor.
                self.precursors[level] = self.current;
                self.precursor_distances[level] = self.index;
            }
        }

        // No skip-link can advance us further.  Step via `next` to close the
        // final gap.  Since found() is false we know self.index < self.target,
        // so self.index + 1 <= self.target and the advance is always valid.
        self.level = 0;
        // SAFETY: `self.current` is valid for `'a` and no other `&mut` exists.
        let next_opt = unsafe { self.current.as_ref() }.next();
        if let Some(next) = next_opt {
            self.current = NonNull::from(next);
            self.index = self.index.saturating_add(1);
            return Step::Advanced(self.current);
        }

        Step::Exhausted
    }
}

impl<T, const N: usize> VisitorMut for IndexMutVisitor<'_, T, N> {
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
    reason = "test code, safety guarantees can be relaxed"
)]
#[cfg(test)]
mod tests {
    use anyhow::Result;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::IndexMutVisitor;
    use crate::node::{
        Node,
        tests::{MAX_LEVELS, skiplist},
        visitor::{Step, Visitor, VisitorMut},
    };

    #[rstest]
    fn find_index_2(skiplist: Result<Box<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let mut head = skiplist?;
        let mut visitor = IndexMutVisitor::new(&mut head, 2);

        let found = visitor.traverse();

        assert!(visitor.found());
        // SAFETY: pointer is valid for the duration of `head`'s lifetime.
        let value = found.map(|ptr| unsafe { ptr.as_ref() }.value().copied());
        assert_eq!(value, Some(Some(20)));
        Ok(())
    }

    #[rstest]
    fn find_index_not_found(skiplist: Result<Box<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let mut head = skiplist?;
        let mut visitor = IndexMutVisitor::new(&mut head, 5);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        Ok(())
    }

    /// After traversal, every precursor must point to a node whose index is
    /// strictly less than the target.
    #[rstest]
    fn precursors_are_before_target(skiplist: Result<Box<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let mut head = skiplist?;

        // Target = 3 (node v3, value 3).
        let mut visitor = IndexMutVisitor::new(&mut head, 3);
        while let Step::Advanced(_) = visitor.step() {}

        for &dist in visitor.precursor_distances() {
            assert!(dist < 3, "precursor distance {dist} should be < 3");
        }
        Ok(())
    }

    /// Stepping past the end of the list produces `Exhausted`.
    #[rstest]
    fn exhausted_when_target_out_of_range(
        skiplist: Result<Box<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let mut head = skiplist?;
        let mut visitor = IndexMutVisitor::new(&mut head, 99);

        loop {
            let s = visitor.step();
            match s {
                Step::Advanced(_) => {}
                Step::Exhausted => {
                    assert!(!visitor.found());
                    break;
                }
                Step::FoundTarget => panic!("should not find target 99"),
            }
        }
        Ok(())
    }

    /// `current_mut()` returns the same pointer as `current()`.
    #[rstest]
    fn current_mut_matches_current(skiplist: Result<Box<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let mut head = skiplist?;
        let mut visitor = IndexMutVisitor::new(&mut head, 2);
        visitor.traverse();

        assert_eq!(visitor.current(), visitor.current_mut());
        Ok(())
    }

    /// `precursors()` has one entry per level.
    #[rstest]
    fn precursors_length(skiplist: Result<Box<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let mut head = skiplist?;
        let max_levels = head.level();
        let mut visitor = IndexMutVisitor::new(&mut head, 2);
        visitor.traverse();

        assert_eq!(visitor.precursors().len(), max_levels);
        assert_eq!(visitor.precursor_distances().len(), max_levels);
        Ok(())
    }
}
