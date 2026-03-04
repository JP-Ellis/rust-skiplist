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
pub(crate) struct OrdVisitor<'a, 'b, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> {
    /// Current node under consideration.
    current: &'a Node<T, N>,
    /// Highest level still under consideration.
    level: usize,
    /// Whether the target has been found.
    found: bool,
    /// The value being searched for. `'b` is independent of `'a` so that
    /// callers can pass a shorter-lived target while the returned node ref
    /// still carries the longer list lifetime `'a`.
    target: &'b Q,
    /// Comparator: `cmp(node_value, target)`.
    cmp: F,
}

impl<'a, 'b, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize>
    OrdVisitor<'a, 'b, T, Q, F, N>
{
    /// Create a new ordered visitor starting at `head`.
    ///
    /// # Arguments
    ///
    /// - `head`: The head node of the skip list. The returned node ref
    ///   carries the same lifetime `'a`.
    /// - `target`: The value to search for. Its lifetime `'b` is independent
    ///   of `'a` and need not outlive the traversal result.
    /// - `cmp`: Comparator returning `Ordering` for a node value vs. the target.
    pub(crate) fn new(head: &'a Node<T, N>, target: &'b Q, cmp: F) -> Self {
        Self {
            current: head,
            level: head.level(),
            found: false,
            target,
            cmp,
        }
    }
}

impl<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> Visitor
    for OrdVisitor<'a, '_, T, Q, F, N>
{
    type NodeRef = &'a Node<T, N>;

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
                // SAFETY: link.node() is a valid heap-allocated node
                // that lives at least as long as the visitor's `'a`.
                let node = unsafe { link.node().as_ref() };
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
        if let Some(next) = self.current.next_as_ref() {
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
    use core::ptr::NonNull;

    use anyhow::Result;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::OrdVisitor;
    use crate::node::{
        Node,
        tests::{MAX_LEVELS, skiplist},
        visitor::{Step, Visitor},
    };

    #[rstest]
    fn find_existing_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is a valid heap-allocated node for the duration of this test.
        let mut visitor = OrdVisitor::new(unsafe { head.as_ref() }, &30_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        assert_eq!(found.and_then(|n| n.value().copied()), Some(30));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn find_first_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is a valid heap-allocated node for the duration of this test.
        let mut visitor = OrdVisitor::new(unsafe { head.as_ref() }, &10_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        assert_eq!(found.and_then(|n| n.value().copied()), Some(10));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn find_last_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is a valid heap-allocated node for the duration of this test.
        let mut visitor = OrdVisitor::new(unsafe { head.as_ref() }, &40_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        assert_eq!(found.and_then(|n| n.value().copied()), Some(40));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn value_not_in_list(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is a valid heap-allocated node for the duration of this test.
        let mut visitor = OrdVisitor::new(unsafe { head.as_ref() }, &25_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn value_before_first(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is a valid heap-allocated node for the duration of this test.
        let mut visitor = OrdVisitor::new(unsafe { head.as_ref() }, &5_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn value_beyond_list(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is a valid heap-allocated node for the duration of this test.
        let mut visitor = OrdVisitor::new(unsafe { head.as_ref() }, &99_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// Calling `step()` again after `found()` returns `true` must immediately
    /// return `FoundTarget` without advancing.
    #[rstest]
    fn step_after_found(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is a valid heap-allocated node for the duration of this test.
        let mut visitor = OrdVisitor::new(unsafe { head.as_ref() }, &20_u8, Ord::cmp);

        visitor.traverse();
        assert!(visitor.found());

        assert!(matches!(visitor.step(), Step::FoundTarget));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }
}
