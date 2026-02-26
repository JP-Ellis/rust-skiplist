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

/// Outcome of a single [`Visitor::step`] call.
#[derive(Debug, PartialEq, Eq)]
enum Step<NodeRef> {
    /// The visitor advanced to a new node.
    Advanced(NodeRef),
    /// The visitor has reached its target and can step no further.
    FoundTarget,
    /// The visitor has exhausted the list without finding the target.
    Exhausted,
}

/// Basic interface for a visitor.
///
/// This trait defines the interface for visiting nodes.
trait Visitor {
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

    /// Step towards the target.
    ///
    /// This attempts to take a single step towards the target. This may be
    /// either through a link, or by moving through the node's `next` pointer.
    ///
    /// Returns [`Step::Advanced`] with the new node when a step was taken,
    /// [`Step::FoundTarget`] when the target has been reached and no further
    /// steps are possible, or [`Step::Exhausted`] when the list has been
    /// exhausted without finding the target.
    fn step(&mut self) -> Step<Self::NodeRef>;

    /// Traverse until we reach the target, or we can no longer step.
    ///
    /// This will continue to step until we reach the target, or we can no
    /// longer step. If we reach the target, this will return the target node
    /// reference, otherwise it will return `None`.
    ///
    /// # Invariant
    ///
    /// When [`Step::FoundTarget`] is returned by [`step`][Visitor::step],
    /// `current` holds the last node returned by [`Step::Advanced`], which is
    /// the target node itself (the visitor advanced *to* the target, then the
    /// following call signals it cannot advance further). This is why
    /// `Some(current)` (not a new call to [`current`][Visitor::current]) is
    /// the correct value to return.
    fn traverse(&mut self) -> Option<Self::NodeRef> {
        let mut current = self.current();
        loop {
            match self.step() {
                Step::Advanced(node) => current = node,
                Step::FoundTarget => return Some(current),
                Step::Exhausted => return None,
            }
        }
    }
}

/// Extension to the [`Visitor`] trait for mutation.
///
/// This trait extends the [`Visitor`] trait to allow for mutation of the
/// current node and the links around it during insert and remove operations.
///
/// During traversal the visitor records, for each level `l`, the last node
/// whose skip-link at level `l` points to the target position or beyond.
/// These precursor nodes are the ones whose links must be rewritten when
/// a node is inserted or removed.
trait VisitorMut: Visitor {
    /// Mutable node reference returned by [`current_mut`][VisitorMut::current_mut].
    type NodeMut;

    /// Opaque pointer to a precursor node (typically `NonNull<Node<V>>`).
    ///
    /// This is `Copy` so that precursor slices can be indexed cheaply and
    /// individual entries can be read without consuming the array.
    type Precursor: Copy;

    /// Get a mutable reference to the current (target) node.
    fn current_mut(&mut self) -> Self::NodeMut;

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
