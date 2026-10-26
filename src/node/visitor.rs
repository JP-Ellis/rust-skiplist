//! Visitor traits over skip list nodes.
//!
//! This module defines the traits for visitor structs, used to locate nodes
//! efficiently during traversal. Given a list of the form:
//!
//! ```text
//! [3] head --------------------------> 6
//! [2] head ---------------------> 5 -> 6
//! [1] head ------> 2 -----------> 5 -> 6 ------> 8
//! [*] head -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10
//! ```
//!
//! Starting from the head node, the paths to targets will be:
//!
//! - If trying to reach node 9, the links traversed would be `head -> 6 -> 8 ->
//!   9`.
//! - If trying to reach node 3, the algorithm would try the `head -> 6` and
//!   `head -> 5` links first, realise they overshoot the target, and then take
//!   the path `head -> 2 -> 3`.
//!
//! If the only reason for visiting the list is to find a node, the visitor
//! implementation does not need to record the links traversed. When the intent
//! is to insert or remove a node, the visitor must track the links that will
//! need to be rewritten.
//!
//! In the `head -> 9` example above, the links that need to be modified are at
//! nodes `6[3]`, `6[2]`, and `8[1]`, as well as the immediately previous node
//! `8`; and in the `head -> 3` example, the links that need to be modified are
//! `head[3]`, `head[2]`, and `2[1]`.

mod index;
mod index_mut;
mod ord;
mod ord_index;
mod ord_index_mut;
mod ord_mut;

pub(crate) use index::IndexVisitor;
pub(crate) use index_mut::IndexMutVisitor;
pub(crate) use ord::OrdVisitor;
pub(crate) use ord_index::OrdIndexVisitor;
pub(crate) use ord_index_mut::OrdIndexMutVisitor;
pub(crate) use ord_mut::OrdMutVisitor;

/// Basic interface for a visitor.
///
/// This trait defines the interface for visiting nodes.
pub(crate) trait Visitor {
    /// Node reference associated type.
    type NodeRef;

    /// The current node being traversed.
    fn current(&self) -> Self::NodeRef;

    /// The current level of the visitor.
    ///
    /// The visitor will always start at the highest level, and work its way
    /// down to the lowest level.
    fn level(&self) -> usize;

    /// Whether the traverser has reached its destination.
    fn found(&self) -> bool;

    /// Walk towards the target until it is reached or cannot be reached.
    ///
    /// Each move follows the highest skip link that does not pass the target,
    /// and falls back to the node's `next` pointer when none does.
    ///
    /// Returns the target node if it was reached, otherwise `None`. Either
    /// way, [`current`][Visitor::current] is left at the last node the walk
    /// reached, and calling `traverse` again returns the same result.
    fn traverse(&mut self) -> Option<Self::NodeRef>;
}

/// Extension to the [`Visitor`] trait for mutation.
///
/// This trait extends the [`Visitor`] trait to expose the links around the
/// target, which insert and remove operations rewrite.
///
/// During traversal the visitor records, for each level `l`, the last node
/// whose skip-link at level `l` points to the target position or beyond.
/// These precursor nodes are the ones whose links must be rewritten when
/// a node is inserted or removed.
pub(crate) trait VisitorMut: Visitor {
    /// Opaque pointer to a precursor node (typically `NonNull<Node<V>>`).
    ///
    /// This is `Copy` so that precursor slices can be indexed cheaply and
    /// individual entries can be read without consuming the array.
    type Precursor: Copy;

    /// Precursor nodes, one per level.
    ///
    /// After traversal, `precursors()[l]` is the last node at level `l` whose
    /// skip-link at level `l` points to the target position or beyond. These
    /// are the nodes whose links must be updated when inserting or removing a
    /// node at the visited position.
    ///
    /// The slice has one entry per level (length equals the maximum number of
    /// levels for this skip list).
    fn precursors(&self) -> &[Self::Precursor];
}

#[cfg(test)]
pub(crate) mod tests {
    //! Helpers that check each visitor's `traverse` against a linear scan of
    //! the list.

    use core::{panic::AssertUnwindSafe, ptr::NonNull};
    use std::panic::catch_unwind;

    use crate::node::Node;

    /// The head height of every list built by [`list`].
    pub(crate) const LEVELS: usize = 4;

    /// List shapes as `(value, height)` pairs in base-chain order.
    ///
    /// Values are ascending, so the ordered visitors can walk every shape.
    pub(crate) const SHAPES: &[&[(u8, usize)]] = &[
        &[],
        &[(10, 0)],
        &[(10, LEVELS)],
        &[(10, 0), (20, 0), (30, 0), (40, 0), (50, 0)],
        &[
            (10, 0),
            (20, 2),
            (30, 1),
            (40, LEVELS),
            (50, 0),
            (60, 3),
            (70, 1),
            (80, 0),
        ],
        &[(10, LEVELS), (20, LEVELS), (30, LEVELS), (40, LEVELS)],
        &[
            (10, 1),
            (20, 0),
            (20, 3),
            (20, 2),
            (30, 0),
            (30, 1),
            (40, 0),
        ],
    ];

    /// Builds a well-formed list from `(value, height)` pairs and returns
    /// its head.
    ///
    /// The caller frees the list with `drop(Box::from_raw(head.as_ptr()))`.
    pub(crate) fn list(entries: &[(u8, usize)]) -> NonNull<Node<u8, LEVELS>> {
        // SAFETY: `Box::into_raw` never returns null.
        let head = unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(LEVELS)))) };
        let mut last = head;
        for &(value, height) in entries {
            // SAFETY: `last` is the live tail of the list under construction.
            last = unsafe { Node::insert_after(last, Node::with_value(height, value)) };
        }
        let (mut len, mut tail) = (0, None);
        // SAFETY: `head` is exclusively owned and every data node is live.
        unsafe { Node::filter_rebuild(head, &mut len, &mut tail, |_| true, |_| {}) };
        head
    }

    /// Every node of the list at `head` in base-chain order, the head first,
    /// so each node's position is its rank.
    pub(crate) fn nodes(head: NonNull<Node<u8, LEVELS>>) -> Vec<NonNull<Node<u8, LEVELS>>> {
        let mut nodes = vec![head];
        let mut current = head;
        // SAFETY: every node of the list is live.
        while let Some(next) = unsafe { current.as_ref() }.next() {
            nodes.push(next);
            current = next;
        }
        nodes
    }

    /// The rank of the node at `ptr` within `nodes`.
    pub(crate) fn rank_of(
        nodes: &[NonNull<Node<u8, LEVELS>>],
        ptr: NonNull<Node<u8, LEVELS>>,
    ) -> usize {
        nodes
            .iter()
            .position(|&node| node == ptr)
            .expect("pointer is a node of the list")
    }

    /// The rank of each level's precursor for a target placed after the
    /// first `before` data nodes of `entries`.
    ///
    /// At level `l` that is the last of those nodes whose tower has a link
    /// at `l`, or the head (rank 0) when none does.
    pub(crate) fn scan_precursors(entries: &[(u8, usize)], before: usize) -> Vec<usize> {
        (0..LEVELS)
            .map(|l| {
                entries
                    .iter()
                    .take(before)
                    .rposition(|&(_, height)| height > l)
                    .map_or(0, |i| i.wrapping_add(1))
            })
            .collect()
    }

    /// Asserts that `current` and `precursors` describe a stop a forward walk
    /// can make, whatever the comparator answered.
    ///
    /// Each level's precursor lies at or after the one above it and at or
    /// before `current`, and its link at that level, if any, reaches
    /// `current` or beyond.
    pub(crate) fn assert_consistent_stop(
        nodes: &[NonNull<Node<u8, LEVELS>>],
        current: NonNull<Node<u8, LEVELS>>,
        precursors: &[NonNull<Node<u8, LEVELS>>],
        context: &str,
    ) {
        let current_rank = rank_of(nodes, current);
        let mut above = 0;
        for (l, &precursor) in precursors.iter().enumerate().rev() {
            let rank = rank_of(nodes, precursor);
            assert!(
                rank >= above,
                "{context}: level {l} precursor before the one above"
            );
            assert!(
                rank <= current_rank,
                "{context}: level {l} precursor past current"
            );
            // SAFETY: every precursor is a live node of the list.
            let maybe_link = unsafe { precursor.as_ref() }
                .links()
                .get(l)
                .and_then(Option::as_ref);
            if let Some(link) = maybe_link {
                assert!(
                    rank_of(nodes, link.node()) >= current_rank,
                    "{context}: level {l} link stops short of current"
                );
            }
            above = rank;
        }
    }

    /// The number of `entries` whose value is below `target`.
    pub(crate) fn count_below(entries: &[(u8, usize)], target: u8) -> usize {
        entries.iter().filter(|&&(value, _)| value < target).count()
    }

    /// Whether some entry holds `target`.
    pub(crate) fn contains(entries: &[(u8, usize)], target: u8) -> bool {
        entries.iter().any(|&(value, _)| value == target)
    }

    /// Runs `walk`, catching a panic as its message.
    pub(crate) fn outcome<S>(walk: impl FnOnce() -> S) -> Result<S, String> {
        catch_unwind(AssertUnwindSafe(walk)).map_err(|payload| {
            payload
                .downcast_ref::<&str>()
                .map(|message| (*message).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_owned())
        })
    }
}
