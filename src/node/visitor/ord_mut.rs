//! Mutable ordered visitor.

use core::{cmp::Ordering, marker::PhantomData, ptr::NonNull};

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
struct OrdMutVisitor<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering> {
    /// Raw pointer to the current node.
    ///
    /// Stored as `NonNull` rather than `&'a mut` to avoid holding an exclusive
    /// reference while `precursors` may alias the same allocation at a
    /// different level.
    current: NonNull<Node<T>>,
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
    precursors: Vec<NonNull<Node<T>>>,
    /// Ties the raw pointer lifetime to `'a`.
    _marker: PhantomData<&'a mut Node<T>>,
}

impl<'a, T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering> OrdMutVisitor<'a, T, Q, F> {
    /// Create a new mutable ordered visitor starting at `head`.
    ///
    /// # Arguments
    ///
    /// - `head`: The head node of the skip list.
    /// - `target`: The value to search for.
    /// - `cmp`: Comparator returning `Ordering` for a node value vs. the target.
    fn new(head: &'a mut Node<T>, target: &'a Q, cmp: F) -> Self {
        let max_levels = head.level();
        let current = NonNull::from(&*head);
        Self {
            current,
            level: max_levels,
            found: false,
            target,
            cmp,
            // Every level starts with `head` as its precursor: the head is
            // always before every real node, so it is a valid precursor for
            // any target value.
            precursors: vec![current; max_levels],
            _marker: PhantomData,
        }
    }
}

impl<T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering> Visitor for OrdMutVisitor<'_, T, Q, F> {
    /// Node references are raw const pointers to avoid lifetime conflicts with
    /// the `&mut` borrow. Callers that need a shared reference can convert
    /// via [`NonNull::as_ref`] inside an `unsafe` block.
    type NodeRef = NonNull<Node<T>>;

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
            // SAFETY: `self.current` was obtained from a valid `&mut Node<T>`
            // during construction or from a link/next pointer during traversal.
            // No other `&mut` to the same node exists while we hold `self`.
            let current_ref: &Node<T> = unsafe { self.current.as_ref() };
            let links = current_ref.links();

            for level in (0..self.level).rev() {
                // Use `.get()` rather than `.zip()` so that levels beyond
                // this node's actual height are treated as `None` (overshoot),
                // not silently skipped: every level must get a precursor
                // recorded.
                let maybe_link = links.get(level).and_then(|l| l.as_ref());

                if let Some(link) = maybe_link {
                    let node = link.node();
                    // Treat a value-less node (head sentinel) as strictly less
                    // than any target. In a well-formed skip list every
                    // skip-link target has a value, so this is defensive.
                    let ord = node
                        .value()
                        .map_or(Ordering::Less, |v| (self.cmp)(v, self.target));

                    if ord == Ordering::Less {
                        // This link advances us strictly closer to the target.
                        // Do NOT record a precursor here: the current node's
                        // link[level] points to < target, so it is not the
                        // precursor for this level. We will record it later
                        // if the same level overshoots at the new node.
                        self.current = NonNull::from(node);
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
        let next_opt = unsafe { self.current.as_ref() }.next();
        if let Some(next) = next_opt {
            let ord = next
                .value()
                .map_or(Ordering::Less, |v| (self.cmp)(v, self.target));
            if ord == Ordering::Greater {
                // The next sequential node is already past the target; the
                // target is absent from the list.
                return Step::Exhausted;
            }
            self.current = NonNull::from(next);
            self.found = ord == Ordering::Equal;
            return Step::Advanced(self.current);
        }

        Step::Exhausted
    }
}

impl<T, Q: ?Sized, F: Fn(&T, &Q) -> Ordering> VisitorMut for OrdMutVisitor<'_, T, Q, F> {
    type NodeMut = NonNull<Node<T>>;
    type Precursor = NonNull<Node<T>>;

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
    use anyhow::Result;
    use pretty_assertions::assert_eq;
    use rstest::{fixture, rstest};

    use super::OrdMutVisitor;
    use crate::node::{
        Node,
        link::Link,
        visitor::{Step, Visitor, VisitorMut},
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
        let mut head = sorted_skiplist?;
        let mut visitor = OrdMutVisitor::new(&mut head, &30_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        // SAFETY: pointer is valid for the duration of `head`'s lifetime.
        let value = found.map(|ptr| unsafe { ptr.as_ref() }.value().copied());
        assert_eq!(value, Some(Some(30)));
        Ok(())
    }

    #[rstest]
    fn find_first_value(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = sorted_skiplist?;
        let mut visitor = OrdMutVisitor::new(&mut head, &10_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        let value = found.map(|ptr| unsafe { ptr.as_ref() }.value().copied());
        assert_eq!(value, Some(Some(10)));
        Ok(())
    }

    #[rstest]
    fn find_last_value(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = sorted_skiplist?;
        let mut visitor = OrdMutVisitor::new(&mut head, &40_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(visitor.found());
        let value = found.map(|ptr| unsafe { ptr.as_ref() }.value().copied());
        assert_eq!(value, Some(Some(40)));
        Ok(())
    }

    #[rstest]
    fn value_not_found(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = sorted_skiplist?;
        let mut visitor = OrdMutVisitor::new(&mut head, &25_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        Ok(())
    }

    #[rstest]
    fn value_beyond_list(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = sorted_skiplist?;
        let mut visitor = OrdMutVisitor::new(&mut head, &99_u8, Ord::cmp);

        let found = visitor.traverse();

        assert!(!visitor.found());
        assert!(found.is_none());
        Ok(())
    }

    /// After traversal, every precursor must point to a node whose value is
    /// strictly less than the target (or the head sentinel with no value).
    #[rstest]
    fn precursors_are_before_target(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = sorted_skiplist?;
        // Target = 30 (node v3).
        let mut visitor = OrdMutVisitor::new(&mut head, &30_u8, Ord::cmp);
        while let Step::Advanced(_) = visitor.step() {}

        for &ptr in visitor.precursors() {
            let value = unsafe { ptr.as_ref() }.value().copied();
            assert!(
                value.is_none_or(|v| v < 30),
                "precursor value {value:?} should be < 30"
            );
        }
        Ok(())
    }

    /// Stepping past the end of the list produces `Exhausted`.
    #[rstest]
    fn exhausted_when_target_out_of_range(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = sorted_skiplist?;
        let mut visitor = OrdMutVisitor::new(&mut head, &99_u8, Ord::cmp);

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
        Ok(())
    }

    /// `current_mut()` returns the same pointer as `current()`.
    #[rstest]
    fn current_mut_matches_current(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = sorted_skiplist?;
        let mut visitor = OrdMutVisitor::new(&mut head, &20_u8, Ord::cmp);
        visitor.traverse();

        assert_eq!(visitor.current(), visitor.current_mut());
        Ok(())
    }

    /// `precursors()` has one entry per level.
    #[rstest]
    fn precursors_length(sorted_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = sorted_skiplist?;
        let max_levels = head.level();
        let mut visitor = OrdMutVisitor::new(&mut head, &20_u8, Ord::cmp);
        visitor.traverse();

        assert_eq!(visitor.precursors().len(), max_levels);
        Ok(())
    }
}
