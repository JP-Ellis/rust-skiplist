//! Mutable index-based visitor.

use core::{marker::PhantomData, ptr::NonNull};

use crate::node::{
    Node,
    level_array::LevelArray,
    visitor::{Visitor, VisitorMut},
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
/// The caller must guarantee that the `NonNull<Node>` passed to [`new`]
/// remains valid and exclusively accessible for the lifetime of this visitor.
/// All `NonNull` pointers stored inside the visitor point into the same list.
///
/// [`new`]: IndexMutVisitor::new
pub(crate) struct IndexMutVisitor<T, const N: usize> {
    /// Raw pointer to the current node.
    ///
    /// Stored as `NonNull` rather than `&mut` to avoid holding an exclusive
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
    precursors: LevelArray<NonNull<Node<T, N>>, N>,
    /// For each level `l`: the cumulative distance already traversed at level
    /// `l` when the precursor at that level was recorded.  Together with the
    /// precursor's link distance this is sufficient to compute the new link
    /// distances after insertion or removal.
    precursor_distances: LevelArray<usize, N>,
    /// Variance marker: acts like `*mut Node<T, N>`, invariant in `T`,
    /// not automatically `Send`/`Sync`.
    _marker: PhantomData<*mut Node<T, N>>,
}

impl<T, const N: usize> IndexMutVisitor<T, N> {
    /// Create a new mutable index visitor starting at `head`.
    ///
    /// # Arguments
    ///
    /// - `head`: A raw `NonNull` pointer to the head node of the skip list.
    ///   Must carry mutable provenance so that precursor pointers stored during
    ///   traversal can be used for subsequent writes.
    /// - `target`: The 0-based index to locate.
    ///
    /// # Safety
    ///
    /// `head` must be valid and exclusively accessible for the lifetime of the
    /// returned visitor.
    pub(crate) fn new(head: NonNull<Node<T, N>>, target: usize) -> Self {
        // `head` must come from a `&mut` borrow (Reserved/Active provenance in
        // Tree Borrows terms): the raw pointer is stored in `precursors` and
        // later used as the self pointer for `insert_after`, which requires
        // write access. Deriving it from a shared reference would violate
        // Tree Borrows when the write occurs.
        // SAFETY: caller guarantees head is valid.
        let max_levels = unsafe { head.as_ref() }.level();
        let current = head;
        Self {
            current,
            index: 0,
            level: max_levels,
            target,
            // Every level starts with `head` as its precursor: the head is
            // always before every real node, so it is a valid precursor for
            // any target index.
            precursors: LevelArray::from_fn(max_levels, |_| current),
            precursor_distances: LevelArray::from_fn(max_levels, |_| 0_usize),
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
        LevelArray<NonNull<Node<T, N>>, N>,
        LevelArray<usize, N>,
    ) {
        (self.current, self.precursors, self.precursor_distances)
    }
}

impl<T, const N: usize> Visitor for IndexMutVisitor<T, N> {
    // `NodeRef` is `NonNull` (not `&Node`) to avoid holding a shared reference
    // while `precursors` may alias the same allocation. Callers that need a
    // shared reference can call `NonNull::as_ref` inside an `unsafe` block.
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

    // The walk state lives in locals so the hot loop does not reload and
    // store it through `self` on every link.
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        reason = "`level` starts at `max_levels == precursors.len()` and never \
                  grows, so every `l < level` indexes `precursors` and `distances`"
    )]
    fn traverse(&mut self) -> Option<Self::NodeRef> {
        let target = self.target;
        let mut current = self.current;
        let mut index = self.index;
        let mut level = self.level;
        let precursors: &mut [NonNull<Node<T, N>>] = &mut self.precursors;
        let distances: &mut [usize] = &mut self.precursor_distances;
        let result = 'walk: loop {
            if index == target {
                break Some(current);
            }
            // SAFETY: `current` is the head or a node reached through a link
            // or `next` pointer, so it is live, and the visitor holds the only
            // access to the list.
            let links = unsafe { current.as_ref() }.links();
            for l in (0..level).rev() {
                // A level above this node's tower has no link, so it records
                // a precursor like a link that overshoots.
                if let Some(link) = links.get(l).and_then(Option::as_ref) {
                    // Ranks are bounded by the list length, so these sums never wrap.
                    let next_index = index.wrapping_add(link.distance().get());
                    if next_index < target {
                        current = link.node();
                        // Reconsider the same level at the new node.
                        level = l.wrapping_add(1);
                        index = next_index;
                        continue 'walk;
                    }
                }
                // The link reaches the target or beyond, or is absent:
                // `current` is the last node before the target at level `l`.
                precursors[l] = current;
                distances[l] = index;
            }
            // `next` keeps the pointer's write provenance, so a precursor
            // reached through it can still anchor `insert_after`.
            level = 0;
            // SAFETY: `current` is live and the visitor holds the only access
            // to the list.
            let Some(next) = unsafe { current.as_ref() }.next() else {
                break None;
            };
            current = next;
            index = index.wrapping_add(1);
        };
        self.current = current;
        self.index = index;
        self.level = level;
        result
    }
}

impl<T, const N: usize> VisitorMut for IndexMutVisitor<T, N> {
    type Precursor = NonNull<Node<T, N>>;

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
    use core::ptr::NonNull;

    use anyhow::Result;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::IndexMutVisitor;
    use crate::node::{
        Node,
        tests::{MAX_LEVELS, skiplist},
        visitor::{
            Visitor, VisitorMut,
            tests::{SHAPES, list, nodes, rank_of, scan_precursors},
        },
    };

    #[rstest]
    fn find_index_2(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = IndexMutVisitor::new(head, 2);

        let found = visitor.traverse();

        assert!(visitor.found());
        // SAFETY: pointer is valid for the duration of `head`'s lifetime.
        let value = found.map(|ptr| unsafe { ptr.as_ref() }.value().copied());
        assert_eq!(value, Some(Some(20)));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn find_index_not_found(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = IndexMutVisitor::new(head, 5);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// After traversal, every precursor must point to a node whose index is
    /// strictly less than the target.
    #[rstest]
    fn precursors_are_before_target(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;

        // Target = 3 (node v3, value 3).
        let mut visitor = IndexMutVisitor::new(head, 3);
        visitor.traverse();

        for &dist in visitor.precursor_distances() {
            assert!(dist < 3, "precursor distance {dist} should be < 3");
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
        let mut visitor = IndexMutVisitor::new(head, 99);

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
        let mut visitor = IndexMutVisitor::new(head, 2);
        visitor.traverse();

        assert_eq!(visitor.precursors().len(), max_levels);
        assert_eq!(visitor.precursor_distances().len(), max_levels);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: traverse matches a scan

    /// `traverse` reaches the node at each rank before, inside and after
    /// each shape, and records as each level's precursor the last node
    /// before that rank whose tower has a link at the level.
    #[test]
    fn traverse_matches_scan() {
        for entries in SHAPES {
            let head = list(entries);
            let nodes = nodes(head);
            for target in 0..entries.len().saturating_add(3) {
                let mut visitor = IndexMutVisitor::new(head, target);
                let result = visitor.traverse();
                let context = format!("shape {entries:?}, target {target}");
                assert_eq!(result, nodes.get(target).copied(), "{context}");
                assert_eq!(visitor.found(), result.is_some(), "{context}");
                assert_eq!(
                    rank_of(&nodes, visitor.current()),
                    target.min(entries.len()),
                    "{context}"
                );
                let expected = scan_precursors(entries, target.saturating_sub(1));
                let precursors: Vec<usize> = visitor
                    .precursors()
                    .iter()
                    .map(|&ptr| rank_of(&nodes, ptr))
                    .collect();
                assert_eq!(precursors, expected, "{context}");
                assert_eq!(visitor.precursor_distances(), expected, "{context}");
            }
            unsafe { drop(Box::from_raw(head.as_ptr())) };
        }
    }
}
