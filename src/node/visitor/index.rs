//! Index-based visitor for rank-based traversal of a skip list.

use crate::node::{
    Node,
    visitor::{Step, Visitor},
};

/// Traverses a skip list to locate a node at a given 0-based rank.
///
/// This visitor is used to find a node by its index. It starts at the head
/// of the skiplist and traverses down the list until it reaches the target
/// index.
struct IndexVisitor<'a, T> {
    /// The current node being visited.
    current: &'a Node<T>,
    /// The current index of the visitor.
    index: usize,
    /// The highest skip-link level still under consideration.
    // Starts at node.level() and decreases as we descend the tower.
    level: usize,
    /// The rank we are searching for.
    target: usize,
}

impl<'a, T> IndexVisitor<'a, T> {
    /// Create a new index visitor.
    ///
    /// This creates a new index traverser starting at the head of the skiplist.
    ///
    /// # Arguments
    ///
    /// - `node`: The head of the skiplist.
    /// - `target`: The target index to find.
    fn new(node: &'a Node<T>, target: usize) -> Self {
        Self {
            current: node,
            index: 0,
            level: node.level(),
            target,
        }
    }
}

impl<'a, T> Visitor for IndexVisitor<'a, T> {
    type NodeRef = &'a Node<T>;

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
            if let Some(link) = maybe_link {
                if self.index.saturating_add(link.distance().get()) <= self.target {
                    self.current = link.node();
                    // Use level + 1 so the same level is reconsidered at the new
                    // node; setting to `level` would skip it, degrading O(log n)
                    // traversal to O(n) when consecutive nodes share a high level.
                    self.level = level.saturating_add(1);
                    self.index = self.index.saturating_add(link.distance().get());
                    return Step::Advanced(self.current);
                }
            }
        }

        // No skip link advanced far enough: fall through to the level-0 next
        // pointer. By construction, `self.index < self.target` here, so if a
        // next node exists we can always advance by 1.
        self.level = 0;
        if let Some(next) = self.current.next() {
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
    use rstest::{fixture, rstest};

    use super::IndexVisitor;
    use crate::node::{
        Node,
        link::Link,
        visitor::{Step, Visitor},
    };

    const MAX_LEVELS: usize = 3;

    /// Build a simple skiplist.
    ///
    /// The values are 1, 2, 3, 4; and the links are as follows:
    ///
    /// head
    /// head -------------> 03
    /// head -------> 02 -> 03 -> 04
    /// head -> 01 -> 02 -> 03 -> 04
    #[fixture]
    fn minimal_skiplist() -> Result<Box<Node<u8>>> {
        let mut head = Box::new(Node::new(MAX_LEVELS));
        let mut v1 = Node::new(0);
        let mut v2 = Node::new(1);
        let mut v3 = Node::new(1);
        let mut v4 = Node::new(0);

        // Internal values
        v1.value = Some(1);
        v2.value = Some(2);
        v3.value = Some(3);
        v4.value = Some(4);

        unsafe {
            head.insert_after(v1);
            head.next_mut().expect("v1 not found").insert_after(v2);
            head.next_mut()
                .expect("v1 not found")
                .next_mut()
                .expect("v2 not found")
                .insert_after(v3);
            head.next_mut()
                .expect("v1 not found")
                .next_mut()
                .expect("v2 not found")
                .next_mut()
                .expect("v3 not found")
                .insert_after(v4);
        }

        // Build higher level links:

        let head_v3: Link<_>;
        let head_v2: Link<_>;
        let v2_v3: Link<_>;
        let v3_v4: Link<_>;
        {
            let v1_ref = head.next().expect("v1 not found");
            let v2_ref = v1_ref.next().expect("v2 not found");
            let v3_ref = v2_ref.next().expect("v3 not found");
            let v4_ref = v3_ref.next().expect("tail not found");
            head_v3 = Link::new(v3_ref, 3)?;
            head_v2 = Link::new(v2_ref, 2)?;
            v2_v3 = Link::new(v3_ref, 1)?;
            v3_v4 = Link::new(v4_ref, 1)?;
        }

        unsafe {
            head.links[1] = Some(head_v3);
            head.links[0] = Some(head_v2);

            head.next_mut()
                .expect("v1 not found")
                .next_mut()
                .expect("v2 not found")
                .links[0] = Some(v2_v3);

            head.next_mut()
                .expect("v1 not found")
                .next_mut()
                .expect("v2 not found")
                .next_mut()
                .expect("v3 not found")
                .links[0] = Some(v3_v4);
        }

        Ok(head)
    }

    /// Test the `IndexVisitor` for finding a node by index.
    #[rstest]
    fn index_traverser_step(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = minimal_skiplist?;
        let mut traverser = IndexVisitor::new(&head, 2);

        let mut last_node = None;
        while let Step::Advanced(node) = traverser.step() {
            last_node = Some(node);
        }

        assert!(traverser.found());
        assert_eq!(last_node.and_then(|n| n.value()), Some(&2));

        Ok(())
    }

    #[rstest]
    fn index_traverser_step_none(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = minimal_skiplist?;
        let mut traverser = IndexVisitor::new(&head, 5);

        let mut last_node = None;
        while let Step::Advanced(node) = traverser.step() {
            last_node = Some(node);
        }

        assert!(!traverser.found());
        assert_eq!(last_node.and_then(|n| n.value()), Some(&4));

        Ok(())
    }

    #[rstest]
    fn index_traverser_traverse(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = minimal_skiplist?;
        let mut traverser = IndexVisitor::new(&head, 2);

        let node = traverser.traverse();
        assert!(traverser.found());
        assert_eq!(node.and_then(|n| n.value()), Some(&2));

        Ok(())
    }

    #[rstest]
    fn index_traverser_traverse_not_found(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = minimal_skiplist?;
        let mut traverser = IndexVisitor::new(&head, 5);

        let node = traverser.traverse();
        assert!(!traverser.found());
        assert!(node.is_none());

        Ok(())
    }
}
