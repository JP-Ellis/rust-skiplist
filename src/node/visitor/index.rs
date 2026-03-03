//! Index-based visitor for rank-based traversal of a skip list.

use crate::node::{
    Node,
    visitor::{Step, Visitor},
};

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

    fn step(&mut self) -> Step<Self::NodeRef> {
        if self.found() {
            return Step::FoundTarget;
        }

        for (level, maybe_link) in (0..self.level).zip(self.current.links()).rev() {
            if let Some(link) = maybe_link
                && self.index.saturating_add(link.distance().get()) <= self.target
            {
                // SAFETY: link.node() is a valid heap-allocated node
                // that lives at least as long as the visitor's `'a`.
                self.current = unsafe { link.node().as_ref() };
                // Use level + 1 so the same level is reconsidered at the new
                // node; setting to `level` would skip it, degrading O(log n)
                // traversal to O(n) when consecutive nodes share a high level.
                self.level = level.saturating_add(1);
                self.index = self.index.saturating_add(link.distance().get());
                return Step::Advanced(self.current);
            }
        }

        // No skip link advanced far enough: fall through to the level-0 next
        // pointer. By construction, `self.index < self.target` here, so if a
        // next node exists we can always advance by 1.
        self.level = 0;
        if let Some(next) = self.current.next_as_ref() {
            self.current = next;
            self.index = self.index.saturating_add(1);
            return Step::Advanced(self.current);
        }

        Step::Exhausted
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

    use super::IndexVisitor;
    use crate::node::{
        Node,
        tests::{MAX_LEVELS, skiplist},
        visitor::{Step, Visitor},
    };

    /// Test the `IndexVisitor` for finding a node by index.
    #[rstest]
    fn index_traverser_step(skiplist: Result<Box<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut traverser = IndexVisitor::new(&head, 2);

        let mut last_node = None;
        while let Step::Advanced(node) = traverser.step() {
            last_node = Some(node);
        }

        assert!(traverser.found());
        assert_eq!(last_node.and_then(|n| n.value()), Some(&20));

        Ok(())
    }

    #[rstest]
    fn index_traverser_step_none(skiplist: Result<Box<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut traverser = IndexVisitor::new(&head, 5);

        let mut last_node = None;
        while let Step::Advanced(node) = traverser.step() {
            last_node = Some(node);
        }

        assert!(!traverser.found());
        assert_eq!(last_node.and_then(|n| n.value()), Some(&40));

        Ok(())
    }

    #[rstest]
    fn index_traverser_traverse(skiplist: Result<Box<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut traverser = IndexVisitor::new(&head, 2);

        let node = traverser.traverse();
        assert!(traverser.found());
        assert_eq!(node.and_then(|n| n.value()), Some(&20));

        Ok(())
    }

    #[rstest]
    fn index_traverser_traverse_not_found(
        skiplist: Result<Box<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;
        let mut traverser = IndexVisitor::new(&head, 5);

        let node = traverser.traverse();
        assert!(!traverser.found());
        assert!(node.is_none());

        Ok(())
    }
}
