//! Index-based visitor for rank-based traversal of a skip list.

use crate::node::{Node, visitor::Visitor};

/// Traverses a skip list to locate a node at a given 0-based rank.
///
/// Starts at the head sentinel (which does not hold a value) and advances
/// level-by-level using skip links, then falls back to the level-0 next
/// pointer. After traversal, `found()` returns `true` if the target index
/// was reached exactly.
pub(crate) struct IndexVisitor<'a, T, const N: usize> {
    /// The current node being visited.
    current: &'a Node<T, N>,
    /// The cumulative rank of `current` relative to the head (head = 0).
    index: usize,
    /// The highest skip-link level still under consideration.
    // Starts at node.level() and decreases as we descend the tower.
    level: usize,
    /// The rank we are searching for.
    target: usize,
}

impl<'a, T, const N: usize> IndexVisitor<'a, T, N> {
    /// Create a new index visitor starting at `node` (the head sentinel).
    ///
    /// # Arguments
    ///
    /// * `node` - The head sentinel of the skip list (rank 0, no value).
    /// * `target` - The 0-based rank of the node to locate.
    pub(crate) fn new(node: &'a Node<T, N>, target: usize) -> Self {
        Self {
            current: node,
            index: 0,
            level: node.level(),
            target,
        }
    }
}

impl<'a, T, const N: usize> Visitor for IndexVisitor<'a, T, N> {
    type NodeRef = &'a Node<T, N>;

    fn current(&self) -> Self::NodeRef {
        self.current
    }

    fn level(&self) -> usize {
        self.level
    }

    fn found(&self) -> bool {
        self.index == self.target
    }

    // The walk state lives in locals so the loop does not reload and store
    // it through `self` on every link.
    #[inline]
    fn traverse(&mut self) -> Option<Self::NodeRef> {
        let target = self.target;
        let mut current = self.current;
        let mut index = self.index;
        let mut level = self.level;
        let result = 'walk: loop {
            if index == target {
                break Some(current);
            }
            for (l, maybe_link) in (0..level).zip(current.links()).rev() {
                // Ranks are bounded by the list length, so these sums never wrap.
                if let Some(link) = maybe_link
                    && index.wrapping_add(link.distance().get()) <= target
                {
                    // SAFETY: a link targets a live node of the list, and the
                    // list outlives `'a`.
                    current = unsafe { link.node().as_ref() };
                    // Reconsider the same level at the new node; resuming one
                    // level lower degrades the walk to O(n) when consecutive
                    // nodes share a high level.
                    level = l.wrapping_add(1);
                    index = index.wrapping_add(link.distance().get());
                    continue 'walk;
                }
            }
            level = 0;
            let Some(next) = current.next_as_ref() else {
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

    use super::IndexVisitor;
    use crate::node::{
        Node,
        tests::{MAX_LEVELS, skiplist},
        visitor::{
            Visitor,
            tests::{SHAPES, list, nodes, rank_of},
        },
    };

    /// A walk past the end stops at the last node.
    #[rstest]
    fn index_traverser_stops_at_tail(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is a valid heap-allocated node for the duration of this test.
        let mut traverser = IndexVisitor::new(unsafe { head.as_ref() }, 5);

        assert!(traverser.traverse().is_none());
        assert!(!traverser.found());
        assert_eq!(traverser.current().value(), Some(&40));

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn index_traverser_traverse(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is a valid heap-allocated node for the duration of this test.
        let mut traverser = IndexVisitor::new(unsafe { head.as_ref() }, 2);

        let node = traverser.traverse();
        assert!(traverser.found());
        assert_eq!(node.and_then(|n| n.value()), Some(&20));

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn index_traverser_traverse_not_found(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is a valid heap-allocated node for the duration of this test.
        let mut traverser = IndexVisitor::new(unsafe { head.as_ref() }, 5);

        let node = traverser.traverse();
        assert!(!traverser.found());
        assert!(node.is_none());

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: traverse matches a scan

    /// `traverse` reaches the node at each rank before, inside and after
    /// each shape, and otherwise stops at the tail.
    #[test]
    fn traverse_matches_scan() {
        for entries in SHAPES {
            let head = list(entries);
            let nodes = nodes(head);
            let head_ref = unsafe { head.as_ref() };
            for target in 0..entries.len().saturating_add(3) {
                let mut visitor = IndexVisitor::new(head_ref, target);
                let result = visitor.traverse().map(core::ptr::from_ref);
                let expected = nodes.get(target).map(|node| node.as_ptr().cast_const());
                let context = format!("shape {entries:?}, target {target}");
                assert_eq!(result, expected, "{context}");
                assert_eq!(visitor.found(), expected.is_some(), "{context}");
                let current = NonNull::from(visitor.current());
                assert_eq!(
                    rank_of(&nodes, current),
                    target.min(entries.len()),
                    "{context}"
                );
            }
            unsafe { drop(Box::from_raw(head.as_ptr())) };
        }
    }
}
