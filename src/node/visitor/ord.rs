//! Read-only ordered visitor.

use core::cmp::Ordering;

use crate::node::{
    Node,
    visitor::{Step, Visitor},
};

/// Read-only ordered visitor for sorted skip lists.
///
/// Traverses a skip list whose nodes are sorted by some ordering, stopping at
/// the first node that compares equal to the target. A caller-supplied
/// comparator `F` is used instead of a blanket `T: Ord` bound so that:
///
/// - `SkipMap` can search by key without unwrapping the key-value pair.
/// - `get_by` / `contains_by` style APIs can provide their own ordering.
///
/// # Ordering contract
///
/// The comparator must be consistent with the node ordering in the list: for
/// every adjacent pair `(a, b)`, `cmp(a.value, target)` must not be
/// `Ordering::Greater` when `cmp(b.value, target)` is `Ordering::Less`.
/// Violating this contract may cause the visitor to return a false negative.
struct OrdVisitor<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering> {
    /// Current node under consideration.
    current: &'a Node<T>,
    /// Highest level still under consideration.
    level: usize,
    /// Whether the target has been found.
    found: bool,
    /// The value being searched for.
    target: &'a Q,
    /// Comparator: `cmp(node_value, target)`.
    cmp: F,
}

impl<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering> OrdVisitor<'a, T, Q, F> {
    /// Create a new ordered visitor starting at `head`.
    ///
    /// # Arguments
    ///
    /// - `head`: The head node of the skip list.
    /// - `target`: The value to search for.
    /// - `cmp`: Comparator returning `Ordering` for a node value vs. the target.
    fn new(head: &'a Node<T>, target: &'a Q, cmp: F) -> Self {
        Self {
            current: head,
            level: head.level(),
            found: false,
            target,
            cmp,
        }
    }
}

impl<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering> Visitor for OrdVisitor<'a, T, Q, F> {
    type NodeRef = &'a Node<T>;

    fn current(&self) -> Self::NodeRef {
        self.current
    }

    fn level(&self) -> usize {
        self.level
    }

    fn found(&self) -> bool {
        self.found
    }

    fn step(&mut self) -> Step<Self::NodeRef> {
        if self.found {
            return Step::FoundTarget;
        }

        for (level, maybe_link) in (0..self.level).zip(self.current.links()).rev() {
            if let Some(link) = maybe_link {
                let node = link.node();
                // A node with no value (head sentinel) is treated as strictly
                // less than any target so that links to sentinel-like nodes are
                // always followed. In a well-formed skip list every skip-link
                // target has a value, so this branch is defensive.
                let ord = node
                    .value()
                    .map_or(Ordering::Less, |v| (self.cmp)(v, self.target));
                if ord != Ordering::Greater {
                    self.current = node;
                    // Retain the same level on the next step (mirrors the
                    // level + 1 fix used in `IndexVisitor`).
                    self.level = level.saturating_add(1);
                    if ord == Ordering::Equal {
                        self.found = true;
                    }
                    return Step::Advanced(self.current);
                }
            }
        }

        // No skip-link can advance us; fall back to the sequential next pointer.
        self.level = 0;
        if let Some(next) = self.current.next() {
            let ord = next
                .value()
                .map_or(Ordering::Less, |v| (self.cmp)(v, self.target));
            if ord == Ordering::Greater {
                // The next sequential node is already past the target, so the
                // target is absent from the list.
                return Step::Exhausted;
            }
            self.current = next;
            self.found = ord == Ordering::Equal;
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

    use super::OrdVisitor;
    use crate::node::{
        Node,
        link::Link,
        visitor::{Step, Visitor},
    };

    const MAX_LEVELS: usize = 3;

    /// Sorted skip list with values [10, 20, 30, 40].
    ///
    /// ```text
    /// head ----------------------------> 30          (level 2, distance 3)
    /// head -----------> 20 -----------> 30 -> 40    (level 1, distance 2 / 1 / 1)
    /// head -> 10 -> 20 -> 30 -> 40                  (sequential)
    /// ```
    #[fixture]
    fn sorted_skiplist() -> Result<Box<Node<u8>>> {
        let mut head = Box::new(Node::new(MAX_LEVELS));
        let mut v1 = Node::new(0);
        let mut v2 = Node::new(1);
        let mut v3 = Node::new(1);
        let mut v4 = Node::new(0);

        v1.value = Some(10);
        v2.value = Some(20);
        v3.value = Some(30);
        v4.value = Some(40);

        #[expect(
            clippy::multiple_unsafe_ops_per_block,
            reason = "Building the skip list"
        )]
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

        let head_v3: Link<_>;
        let head_v2: Link<_>;
        let v2_v3: Link<_>;
        let v3_v4: Link<_>;
        {
            let v1_ref = head.next().expect("v1 not found");
            let v2_ref = v1_ref.next().expect("v2 not found");
            let v3_ref = v2_ref.next().expect("v3 not found");
            let v4_ref = v3_ref.next().expect("v4 not found");
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

    #[rstest]
    fn find_existing_value(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = sorted_skiplist?;
        let mut visitor = OrdVisitor::new(&head, &30_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        assert_eq!(found.and_then(|n| n.value().copied()), Some(30));
        Ok(())
    }

    #[rstest]
    fn find_first_value(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = sorted_skiplist?;
        let mut visitor = OrdVisitor::new(&head, &10_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        assert_eq!(found.and_then(|n| n.value().copied()), Some(10));
        Ok(())
    }

    #[rstest]
    fn find_last_value(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = sorted_skiplist?;
        let mut visitor = OrdVisitor::new(&head, &40_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        assert_eq!(found.and_then(|n| n.value().copied()), Some(40));
        Ok(())
    }

    #[rstest]
    fn value_not_in_list(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = sorted_skiplist?;
        let mut visitor = OrdVisitor::new(&head, &25_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        Ok(())
    }

    #[rstest]
    fn value_before_first(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = sorted_skiplist?;
        let mut visitor = OrdVisitor::new(&head, &5_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        Ok(())
    }

    #[rstest]
    fn value_beyond_list(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = sorted_skiplist?;
        let mut visitor = OrdVisitor::new(&head, &99_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        Ok(())
    }

    /// Calling `step()` again after `found()` returns `true` must immediately
    /// return `FoundTarget` without advancing.
    #[rstest]
    fn step_after_found(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = sorted_skiplist?;
        let mut visitor = OrdVisitor::new(&head, &20_u8, Ord::cmp);

        visitor.traverse();
        assert!(visitor.found());

        assert!(matches!(visitor.step(), Step::FoundTarget));
        Ok(())
    }
}
