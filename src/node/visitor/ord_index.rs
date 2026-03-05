//! Read-only ordered visitor with rank/distance tracking.

use core::cmp::Ordering;

use crate::node::{
    Node,
    visitor::{Step, Visitor},
};

/// Read-only ordered visitor that also tracks skip-link distances.
///
/// Combines value-ordered traversal with rank computation, enabling `$O(\log n)$`
/// rank queries without recording precursor nodes.
///
/// Unlike [`OrdVisitor`], this visitor advances only through nodes strictly
/// *less than* the target at the skip-link stage (never on `Equal`). This
/// guarantees that, when duplicates are present, traversal always lands on
/// the *first* occurrence in the sorted list, which is the correct starting
/// point for `rank` and `count` queries.
///
/// # Rank convention
///
/// Internally, `rank` is 1-based: the head sentinel has rank 0 and the first
/// data node has rank 1. The public [`rank`] method subtracts 1 to return
/// the 0-based position. Call it only when [`found`] is `true`.
///
/// # Ordering contract
///
/// The comparator must be consistent with the node ordering in the list: for
/// every adjacent pair `(a, b)`, `cmp(a.value, target)` must not be
/// `Ordering::Greater` when `cmp(b.value, target)` is `Ordering::Less`.
/// Violating this contract may cause the visitor to return a false negative.
///
/// [`OrdVisitor`]: crate::node::visitor::OrdVisitor
/// [`rank`]: OrdIndexVisitor::rank
/// [`found`]: crate::node::visitor::Visitor::found
pub(crate) struct OrdIndexVisitor<'a, 'b, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> {
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
    /// Current rank during traversal (1-based: 0 = head sentinel, 1 = first
    /// data node). Incremented by `link.distance()` when following a skip
    /// link and by 1 when stepping via the sequential `next` pointer.
    rank: usize,
}

impl<'a, 'b, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize>
    OrdIndexVisitor<'a, 'b, T, Q, F, N>
{
    /// Create a new read-only ordered visitor with rank tracking, starting at
    /// `head`.
    ///
    /// # Arguments
    ///
    /// - `head`: The head node of the skip list.
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
            rank: 0,
        }
    }

    /// Returns the 0-based rank of the found node.
    ///
    /// This is the number of data nodes that precede the found node in sorted
    /// order (i.e., the 0-based index into the sorted sequence).
    ///
    /// # Precondition
    ///
    /// Only meaningful after [`traverse`] returns `Some(...)`, i.e., when
    /// [`found`] is `true`. The value is `self.rank - 1`, converting from
    /// the internal 1-based tracking to the 0-based external convention.
    ///
    /// [`traverse`]: crate::node::visitor::Visitor::traverse
    /// [`found`]: crate::node::visitor::Visitor::found
    pub(crate) fn rank(&self) -> usize {
        self.rank.saturating_sub(1)
    }
}

impl<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering, const N: usize> Visitor
    for OrdIndexVisitor<'a, '_, T, Q, F, N>
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
                // SAFETY: link.node() is a valid heap-allocated node that
                // lives at least as long as the visitor's `'a`.
                let node: &'a Node<T, N> = unsafe { link.node().as_ref() };
                // A node with no value (head sentinel) is treated as strictly
                // less than any target. In a well-formed skip list every
                // skip-link target has a value, so this branch is defensive.
                let ord = node
                    .value()
                    .map_or(Ordering::Less, |v| (self.cmp)(v, self.target));

                if ord == Ordering::Less {
                    // Advance only on strict Less to ensure we land on the
                    // *first* occurrence of any duplicate value. This differs
                    // from `OrdVisitor`, which also follows Equal links and
                    // may skip earlier duplicates.
                    self.current = node;
                    self.rank = self.rank.saturating_add(link.distance().get());
                    // Retain the same level on the next step (mirrors the
                    // level + 1 fix used in `IndexVisitor`).
                    self.level = level.saturating_add(1);
                    return Step::Advanced(self.current);
                }
            }
            // Link absent, Equal, or Greater: drop to next lower level.
        }

        // No skip-link can advance us; fall back to the sequential next pointer.
        self.level = 0;
        if let Some(next) = self.current.next_as_ref() {
            let ord = next
                .value()
                .map_or(Ordering::Less, |v| (self.cmp)(v, self.target));
            if ord == Ordering::Greater {
                // The next sequential node is already past the target; the
                // target is absent from the list.
                return Step::Exhausted;
            }
            self.current = next;
            self.rank = self.rank.saturating_add(1);
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

    use super::OrdIndexVisitor;
    use crate::node::{
        Node,
        tests::{MAX_LEVELS, skiplist},
        visitor::{Step, Visitor},
    };

    // The fixture builds: head -> v1(10) -> v2(20) -> v3(30) -> v4(40)
    // with skip links: head->v3 (dist 3, level 1), head->v2 (dist 2, level 0),
    //                  v2->v3 (dist 1, level 0), v3->v4 (dist 1, level 0).

    #[rstest]
    fn find_existing_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexVisitor::new(unsafe { head.as_ref() }, &30_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        assert_eq!(found.and_then(|n| n.value().copied()), Some(30));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn find_first_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexVisitor::new(unsafe { head.as_ref() }, &10_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        assert_eq!(found.and_then(|n| n.value().copied()), Some(10));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn find_last_value(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexVisitor::new(unsafe { head.as_ref() }, &40_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        assert_eq!(found.and_then(|n| n.value().copied()), Some(40));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn value_not_found(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexVisitor::new(unsafe { head.as_ref() }, &25_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn value_beyond_list(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexVisitor::new(unsafe { head.as_ref() }, &99_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// `rank()` returns the correct 0-based position for each element.
    #[rstest]
    fn rank_first_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexVisitor::new(unsafe { head.as_ref() }, &10_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.rank(), 0);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn rank_second_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexVisitor::new(unsafe { head.as_ref() }, &20_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.rank(), 1);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn rank_third_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexVisitor::new(unsafe { head.as_ref() }, &30_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.rank(), 2);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn rank_last_element(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexVisitor::new(unsafe { head.as_ref() }, &40_u8, Ord::cmp);
        visitor.traverse();

        assert!(visitor.found());
        assert_eq!(visitor.rank(), 3);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// `rank()` is not meaningful when `found()` is `false`; stepping to
    /// exhaustion must not set `found`.
    #[rstest]
    fn exhausted_when_target_out_of_range(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexVisitor::new(unsafe { head.as_ref() }, &99_u8, Ord::cmp);

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

    /// Calling `step()` again after `found()` is `true` returns `FoundTarget`
    /// without advancing.
    #[rstest]
    fn step_after_found(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut visitor = OrdIndexVisitor::new(unsafe { head.as_ref() }, &20_u8, Ord::cmp);
        visitor.traverse();
        assert!(visitor.found());

        assert!(matches!(visitor.step(), Step::FoundTarget));
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }
}
