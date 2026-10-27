//! Read-only ordered visitor.

use core::cmp::Ordering;

use crate::node::{Node, visitor::Visitor};

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

    fn traverse(&mut self) -> Option<Self::NodeRef> {
        'walk: while !self.found {
            for (level, maybe_link) in (0..self.level).zip(self.current.links()).rev() {
                if let Some(link) = maybe_link {
                    // SAFETY: a link targets a live node of the list, and the
                    // list outlives `'a`.
                    let node = unsafe { link.node().as_ref() };
                    // SAFETY: links never target the head, so the node holds a
                    // value.
                    let value = unsafe { node.value_unchecked() };
                    let ord = (self.cmp)(value, self.target);
                    if ord != Ordering::Greater {
                        self.current = node;
                        // Reconsider the same level at the new node.
                        self.level = level.saturating_add(1);
                        self.found = ord == Ordering::Equal;
                        continue 'walk;
                    }
                }
            }

            // No skip-link can advance us; fall back to the sequential next
            // pointer.
            self.level = 0;
            let next = self.current.next_as_ref()?;
            // SAFETY: `next` pointers never target the head, so the node
            // holds a value.
            let value = unsafe { next.value_unchecked() };
            let ord = (self.cmp)(value, self.target);
            if ord == Ordering::Greater {
                // The next sequential node is already past the target, so the
                // target is absent from the list.
                return None;
            }
            self.current = next;
            self.found = ord == Ordering::Equal;
        }
        Some(self.current)
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
        visitor::{
            Visitor,
            tests::{SHAPES, contains, count_below, list, nodes, rank_of, targets},
        },
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

    /// Calling `traverse()` again after it found the target returns the same
    /// node without advancing.
    #[rstest]
    fn traverse_after_found(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        // SAFETY: head is a valid heap-allocated node for the duration of this test.
        let mut visitor = OrdVisitor::new(unsafe { head.as_ref() }, &20_u8, Ord::cmp);

        let first = visitor.traverse().map(core::ptr::from_ref);
        assert!(visitor.found());

        assert_eq!(visitor.traverse().map(core::ptr::from_ref), first);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: traverse matches a scan

    /// `traverse` stops on a node equal to each target, or after the last
    /// node below it.
    #[test]
    fn traverse_matches_scan() {
        for entries in SHAPES {
            let head = list(entries);
            let nodes = nodes(head);
            let head_ref = unsafe { head.as_ref() };
            for target in targets() {
                let context = format!("shape {entries:?}, target {target}");
                let mut visitor = OrdVisitor::new(head_ref, &target, Ord::cmp);
                let result = visitor.traverse();
                let found = contains(entries, target);
                assert_eq!(visitor.found(), found, "{context}");
                assert_eq!(result.is_some(), found, "{context}");
                let current = visitor.current();
                if found {
                    assert_eq!(current.value(), Some(&target), "{context}");
                } else {
                    assert_eq!(
                        rank_of(&nodes, NonNull::from(current)),
                        count_below(entries, target),
                        "{context}"
                    );
                }
            }
            unsafe { drop(Box::from_raw(head.as_ptr())) };
        }
    }
}
