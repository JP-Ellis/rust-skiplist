//! Index-based skip list.
//!
//! This module provides [`SkipList<T, G>`], a general-purpose sequence with
//! O(log n) insert, remove, and random access by index.  It is a useful
//! alternative to [`Vec`] when elements are frequently inserted or removed in
//! the middle of the list while indexed access is still needed.
//!
//! # Example
//!
//! ```rust
//! use skiplist::skip_list::SkipList;
//!
//! let list = SkipList::<i32>::new();
//! assert!(list.is_empty());
//! assert_eq!(list.len(), 0);
//! ```

use core::ptr::NonNull;

use crate::{
    level_generator::{LevelGenerator, geometric::Geometric},
    node::{Node, link::Link},
};

/// Default number of levels used by [`SkipList::new`] and
/// [`SkipList::with_capacity`].
const DEFAULT_LEVELS: usize = 16;

/// Default promotion probability used by [`SkipList::new`] and
/// [`SkipList::with_capacity`].
const DEFAULT_P: f64 = 0.5;

/// An index-based skip list.
///
/// `SkipList<T, G>` stores elements in insertion order and provides O(log n)
/// insert, remove, and indexed access.  Unlike [`Vec`], inserting or removing
/// in the middle does not shift elements; unlike a plain linked list,
/// arbitrary-index access is O(log n) rather than O(n).
///
/// The generic parameter `G` controls how node tower heights are chosen.  The
/// default ([`Geometric`]) works well in practice; supply a custom
/// [`LevelGenerator`] via [`SkipList::with_level_generator`] if you need
/// different behaviour.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_list::SkipList;
///
/// let list = SkipList::<u32>::new();
/// assert!(list.is_empty());
/// ```
pub struct SkipList<T, G: LevelGenerator = Geometric> {
    /// Sentinel head node. Never holds a value; its `links` array has length
    /// equal to the maximum number of levels.
    head: Box<Node<T>>,
    /// Cached element count. Updated by every insert / remove operation.
    len: usize,
    /// Level generator used to determine the tower height of each new node.
    generator: G,
}

// MARK: Constructors (default level generator)

impl<T> SkipList<T> {
    /// Creates an empty skip list with the default level generator
    /// (`Geometric { levels: 16, p: 0.5 }`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let list = SkipList::<i32>::new();
    /// assert!(list.is_empty());
    /// assert_eq!(list.len(), 0);
    /// ```
    ///
    /// # Panics
    ///
    /// Never panics.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        #[expect(
            clippy::expect_used,
            reason = "DEFAULT_LEVELS and DEFAULT_P are compile-time constants \
                      whose validity is guaranteed by their definitions"
        )]
        let generator = Geometric::new(DEFAULT_LEVELS, DEFAULT_P)
            .expect("DEFAULT_LEVELS and DEFAULT_P are valid Geometric parameters");
        Self::with_level_generator(generator)
    }

    /// Creates an empty skip list with `max_levels` as the level-count hint.
    ///
    /// The level count controls how many skip-link levels the internal
    /// structure will maintain.  A larger value improves performance for very
    /// large lists at the cost of slightly higher per-node memory use.
    /// `max_levels` is clamped to a minimum of 1.
    ///
    /// The default `p = 0.5` promotion probability is used.
    ///
    /// # Panics
    ///
    /// Never panics; `max_levels` is clamped to `>= 1` and the default
    /// `p = 0.5` is always a valid [`Geometric`] probability.
    #[inline]
    #[must_use]
    pub fn with_capacity(max_levels: usize) -> Self {
        let levels = max_levels.max(1);
        #[expect(
            clippy::expect_used,
            reason = "`levels` is clamped to >= 1 and DEFAULT_P is a valid probability"
        )]
        let generator = Geometric::new(levels, DEFAULT_P)
            .expect("`levels >= 1` and `DEFAULT_P` are valid Geometric parameters");
        Self::with_level_generator(generator)
    }
}

// MARK: Generic methods available for any LevelGenerator

impl<T, G: LevelGenerator> SkipList<T, G> {
    /// Creates an empty skip list driven by the supplied level generator.
    ///
    /// The generator controls the distribution of tower heights assigned to
    /// newly inserted nodes, which affects the time and space trade-offs of
    /// the skip list.  Use [`SkipList::new`] or [`SkipList::with_capacity`]
    /// for the common case.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    /// use skiplist::level_generator::geometric::Geometric;
    ///
    /// let generator = Geometric::new(8, 0.5).expect("valid parameters");
    /// let list = SkipList::<i32, 16, _>::with_level_generator(generator);
    /// assert!(list.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_level_generator(generator: G) -> Self {
        let max_levels = generator.total();
        Self {
            head: Box::new(Node::new(max_levels)),
            len: 0,
            generator,
        }
    }

    /// Returns the number of elements in the list.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let list = SkipList::<i32>::new();
    /// assert_eq!(list.len(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the list contains no elements.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let list = SkipList::<i32>::new();
    /// assert!(list.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inserts `value` at the front of the list.
    ///
    /// The new element becomes the element at index 0, shifting all existing
    /// elements one position to the right.  This operation is O(log n).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_front(1);
    /// list.push_front(2);
    /// assert_eq!(list.len(), 2);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "insert_after guarantees head.next is Some; Link::new(_, 1) and \
                  increment_distance cannot fail under any reachable condition"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l is bounded by height <= max_levels, which equals the length \
                  of the links slice on every node, so all accesses are in bounds"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "head_ptr and new_raw are provably distinct heap allocations; \
                  batching the operations avoids repeating the same SAFETY preamble"
    )]
    #[inline]
    pub fn push_front(&mut self, value: T) {
        // height ∈ [1, max_levels]: generator.level() ∈ [0, total).
        let height = self.generator.level().saturating_add(1);
        let max_levels = self.head.level();

        // SAFETY: Node::with_value produces a detached node (prev = next = None),
        // which is the only kind insert_after accepts.
        unsafe { self.head.insert_after(Node::with_value(height, value)) };

        // SAFETY: insert_after placed new_node immediately after head on the heap.
        // Converting the &mut to NonNull releases the borrow on this line, so
        // no live &mut exists when we later obtain raw pointers to both nodes.
        let new_node_ptr: NonNull<Node<T>> = unsafe {
            NonNull::from(
                self.head
                    .next_mut()
                    .expect("node was just inserted after head"),
            )
        };

        // Update skip links so the structure remains consistent.
        //
        // Before insertion (new_node not yet wired):
        //   head.links[l] → X at distance d   (X is at rank d)
        //
        // After inserting new_node at rank 1:
        //   head is rank 0, new_node is rank 1, X is now at rank d+1.
        //   distance head → new_node = 1
        //   distance new_node → X   = (d+1) - 1 = d  (unchanged)
        //
        // For levels 0..height:
        //   head.links[l]     ← Link { new_node, 1 }
        //   new_node.links[l] ← old head link (distance unchanged)
        //
        // For levels height..max_levels:
        //   head.links[l].distance += 1  (target node shifted one rank right)
        //
        // SAFETY: head_ptr and new_raw point to distinct, heap-allocated Node<T>
        // values.  No safe references to either node exist during this block.
        unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;
            let new_raw: *mut Node<T> = new_node_ptr.as_ptr();

            for l in 0..height {
                let old = (*head_ptr).links_mut()[l].take();
                (*new_raw).links_mut()[l] = old;
                (*head_ptr).links_mut()[l] =
                    Some(Link::new(&*new_raw, 1).expect("distance 1 is always valid"));
            }

            for l in height..max_levels {
                if let Some(link) = (*head_ptr).links_mut()[l].as_mut() {
                    link.increment_distance()
                        .expect("distance overflow requires > usize::MAX nodes");
                }
            }
        }

        self.len = self.len.saturating_add(1);
    }

    /// Appends `value` to the back of the list.
    ///
    /// The new element becomes the element at index `self.len()`, placed after
    /// all existing elements.  This operation is O(log n) expected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.len(), 2);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "insert_after guarantees the tail's next is Some immediately after; \
                  distance equals new_rank − pred_rank where pred_rank ≤ self.len < new_rank \
                  so distance ≥ 1 always; overflow requires > usize::MAX nodes"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l is bounded by height ≤ max_levels, which equals the length of update[] \
                  and the links slice on every node, so all accesses are in bounds"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "traversal, insertion, and link wiring all touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn push_back(&mut self, value: T) {
        // height ∈ [1, max_levels]: generator.level() ∈ [0, total).
        let height = self.generator.level().saturating_add(1);
        let max_levels = self.head.level();

        // SAFETY: All raw pointers originate from heap allocations owned by this
        // SkipList.  No safe &mut references to any node exist while this block
        // runs.  Each node's fields are accessed at most once per iteration, so
        // there is no simultaneous aliasing.
        unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;

            // update[l] = (last node at level l, its 0-based rank from head).
            // Initialised to (head, 0) so levels with no reachable nodes default
            // to head as the predecessor.
            let mut update: Vec<(*mut Node<T>, usize)> = vec![(head_ptr, 0_usize); max_levels];

            // Standard "find-update-array" traversal: advance as far right as
            // possible at each level (we always insert at the very end).
            // current and current_rank carry over between levels — this is
            // intentional and correct for the rightmost-insertion case.
            let mut current: *mut Node<T> = head_ptr;
            let mut current_rank: usize = 0;

            for l in (0..max_levels).rev() {
                while let Some(link) = (*current).links()[l].as_ref() {
                    current_rank = current_rank.saturating_add(link.distance().get());
                    current = NonNull::from(link.node()).as_ptr();
                }
                update[l] = (current, current_rank);
            }

            // `current` is now the tail node (or head if the list is empty).
            (*current).insert_after(Node::with_value(height, value));

            // Obtain a stable raw pointer to new_node.  Converting the &mut
            // returned by next_mut() to NonNull releases the borrow on this line,
            // so no live &mut exists when we later use new_raw.
            let new_raw: *mut Node<T> = NonNull::from(
                (*current)
                    .next_mut()
                    .expect("node was just inserted after tail"),
            )
            .as_ptr();

            // new_rank is the 0-based rank of new_node (head = 0, elements 1..=n).
            // pred_rank ≤ self.len < new_rank, so distance ≥ 1 for all levels.
            let new_rank = self.len.saturating_add(1);

            for (l, &(pred_ptr, pred_rank)) in update.iter().enumerate().take(height) {
                // pred_rank <= self.len < new_rank, so saturating_sub == plain sub here.
                let distance = new_rank.saturating_sub(pred_rank);
                (*pred_ptr).links_mut()[l] =
                    Some(Link::new(&*new_raw, distance).expect("distance >= 1"));
                // new_node.links[l] remains None (Node::with_value initialises all to None).
            }
            // Levels height..max_levels need no update: new_node is at the end
            // and no existing skip link spans past the old tail.
        }

        self.len = self.len.saturating_add(1);
    }

    /// Removes and returns the first element, or `None` if the list is empty.
    ///
    /// This operation is O(log n) expected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.pop_front(), Some(1));
    /// assert_eq!(list.pop_front(), Some(2));
    /// assert_eq!(list.pop_front(), None);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        clippy::unwrap_in_result,
        reason = "head.next is Some because is_empty() was checked first; \
                  decrement_distance cannot underflow because head.links[l] for \
                  l ≥ front_height cannot point to front_node (front_node.level() ≤ l), \
                  so its distance is ≥ 2 before the decrement; \
                  all expects fire only on internal invariant violations, not user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l is bounded by front_height ≤ max_levels, which equals the length \
                  of the links slice on every node, so all accesses are in bounds"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link unwiring, pointer extraction, and node pop all touch provably \
                  disjoint heap nodes; splitting across blocks would require \
                  unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let max_levels = self.head.level();

        // SAFETY: All raw pointers originate from heap allocations owned by this
        // SkipList.  No safe &mut references to any node exist while this block
        // runs.  head_ptr and front_ptr are distinct heap allocations; all slice
        // accesses are bounded by front_height ≤ max_levels = links.len().
        let value = unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;

            // front_ptr is the node at rank 1.  The list is non-empty, so
            // head.next is Some.  Converting the &mut to NonNull releases the
            // borrow immediately, leaving no live &mut when we later use head_ptr.
            let front_ptr: *mut Node<T> =
                NonNull::from((*head_ptr).next_mut().expect("list is non-empty")).as_ptr();

            let front_height = (*front_ptr).level();

            // Restore skip links as if front_node never existed.
            //
            // Invariant (maintained by push_front / push_back):
            //   For l < front_height, head.links[l] = Link(front_node, 1).
            //   front_node.links[l] points to the next node at level l.
            // Moving that link back to head is the exact inverse of push_front's wiring.
            for l in 0..front_height {
                (*head_ptr).links_mut()[l] = (*front_ptr).links_mut()[l].take();
            }

            // For l ≥ front_height, head.links[l] cannot point to front_node
            // (front_node has no link tower at those levels), so the target node
            // is at rank ≥ 2.  Removing front_node shifts every node left by 1,
            // so each such distance decrements from d ≥ 2 to d−1 ≥ 1.
            for l in front_height..max_levels {
                if let Some(link) = (*head_ptr).links_mut()[l].as_mut() {
                    link.decrement_distance().expect(
                        "skip list invariant: target at rank ≥ 2 so distance ≥ 2 before decrement",
                    );
                }
            }

            // Detach front_node from the prev/next chain.
            // pop() sets: head.next = front_node.next
            //             front_node.next.prev = &head  (if next exists)
            let mut popped = (*front_ptr).pop();
            popped.take_value()
        };

        self.len = self.len.saturating_sub(1);
        value
    }
}

// MARK: Default

impl<T> Default for SkipList<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use pretty_assertions::{assert_eq, assert_ne};

    use super::SkipList;
    use crate::{level_generator::geometric::Geometric, node::link::Link};

    #[test]
    fn new_is_empty() {
        let list = SkipList::<i32>::new();
        assert!(list.is_empty());
    }

    #[test]
    fn new_len_zero() {
        let list = SkipList::<i32>::new();
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn with_capacity_is_empty() {
        let list = SkipList::<i32>::with_capacity(8);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn with_capacity_zero_clamped() {
        // max_levels = 0 is clamped to 1; must not panic
        let list = SkipList::<i32>::with_capacity(0);
        assert!(list.is_empty());
    }

    #[test]
    fn with_level_generator_custom() {
        let g = Geometric::new(4, 0.25).expect("valid parameters");
        let list = SkipList::<String>::with_level_generator(g);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn default_is_empty() {
        let list = SkipList::<i32>::default();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
    }

    // MARK: push_front

    #[test]
    fn push_front_into_empty() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_front(42);
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
        assert_eq!(list.head.next().and_then(|n| n.value()), Some(&42));
        assert!(list.head.next().and_then(|n| n.next()).is_none());
    }

    #[test]
    fn push_front_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_front(1);
        list.push_front(2);
        list.push_front(3);
        assert_eq!(list.len(), 3);

        // Last pushed element is at the front: 3 → 2 → 1
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&3));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next().expect("n3");
        assert_eq!(n3.value(), Some(&1));
        assert!(n3.next().is_none());
    }

    #[test]
    fn push_front_len_increments() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.push_front(i);
            assert_eq!(list.len(), i + 1);
        }
    }

    /// With `with_capacity(1)` the generator always assigns height = 1.
    /// After two `push_front` calls the skip-link structure must be:
    ///
    /// ```text
    /// head.links[0] → second_node (value 20) at distance 1
    /// second_node.links[0] → first_node (value 10) at distance 1
    /// first_node.links is empty (height 1 = index-0 only)
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn push_front_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);

        list.push_front(10);
        // After first push: head.links[0] → node(10) at distance 1
        {
            let link: &Link<_> = list.head.links()[0].as_ref().expect("head link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&10));
        }

        list.push_front(20);
        // After second push: head.links[0] → node(20) at distance 1
        {
            let link: &Link<_> = list.head.links()[0].as_ref().expect("head link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&20));
        }
        // node(20).links[0] → node(10) at distance 1
        {
            let front = list.head.next().expect("front node");
            assert_eq!(front.value(), Some(&20));
            let link: &Link<_> = front.links()[0].as_ref().expect("front link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&10));
        }
    }

    // MARK: push_back

    #[test]
    fn push_back_into_empty() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(42);
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
        assert_eq!(list.head.next().and_then(|n| n.value()), Some(&42));
        assert!(list.head.next().and_then(|n| n.next()).is_none());
    }

    #[test]
    fn push_back_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.len(), 3);

        // Elements are in insertion order: 1 → 2 → 3
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next().is_none());
    }

    #[test]
    fn push_back_len_increments() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.push_back(i);
            assert_eq!(list.len(), i + 1);
        }
    }

    /// With `with_capacity(1)` the generator always assigns height = 1.
    /// After two `push_back` calls the skip-link structure must be:
    ///
    /// ```text
    /// head.links[0]         → first_node  (value 10) at distance 1
    /// first_node.links[0]   → second_node (value 20) at distance 1
    /// second_node.links[0]  = None
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn push_back_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);

        list.push_back(10);
        // After first push: head.links[0] → node(10) at distance 1
        {
            let link: &Link<_> = list.head.links()[0].as_ref().expect("head link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&10));
        }

        list.push_back(20);
        // head.links[0] still → node(10) at distance 1 (unchanged)
        {
            let link: &Link<_> = list.head.links()[0].as_ref().expect("head link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&10));
        }
        // node(10).links[0] → node(20) at distance 1
        {
            let front = list.head.next().expect("front node");
            assert_eq!(front.value(), Some(&10));
            let link: &Link<_> = front.links()[0].as_ref().expect("front link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&20));
        }
        // node(20).links[0] = None
        {
            let second = list
                .head
                .next()
                .expect("first node")
                .next()
                .expect("second node");
            assert_eq!(second.value(), Some(&20));
            assert!(second.links()[0].is_none());
        }
    }

    #[test]
    fn push_back_after_push_front() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_front(2); // [2]
        list.push_back(3); // [2, 3]
        list.push_front(1); // [1, 2, 3]
        assert_eq!(list.len(), 3);

        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next().is_none());
    }

    // MARK: pop_front

    #[test]
    fn pop_front_from_empty() {
        let mut list = SkipList::<i32>::new();
        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn pop_front_single_element() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(42);
        assert_eq!(list.pop_front(), Some(42));
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(list.head.next().is_none());
        // Second pop on now-empty list
        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn pop_front_returns_in_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.pop_front(), Some(2));
        assert_eq!(list.pop_front(), Some(3));
        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn pop_front_len_decrements() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.push_back(i);
        }
        for remaining in (0..50_usize).rev() {
            list.pop_front();
            assert_eq!(list.len(), remaining);
        }
        assert_eq!(list.pop_front(), None);
    }

    /// With `with_capacity(1)` the generator always assigns height = 1.
    /// After three `push_back` calls followed by two `pop_front` calls
    /// the skip-link structure must be kept consistent.
    ///
    /// ```text
    /// Initial:  head → n1(10,d=1) → n2(20,d=1) → n3(30,None)
    /// After 1st pop_front (removes 10):
    ///   head → n2(20,d=1) → n3(30,None)
    /// After 2nd pop_front (removes 20):
    ///   head → n3(30,None)
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn pop_front_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);

        // First pop_front: removes 10
        assert_eq!(list.pop_front(), Some(10));
        assert_eq!(list.len(), 2);
        {
            let link: &Link<_> = list.head.links()[0]
                .as_ref()
                .expect("head link after 1st pop");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&20));
        }

        // Second pop_front: removes 20
        assert_eq!(list.pop_front(), Some(20));
        assert_eq!(list.len(), 1);
        {
            let link: &Link<_> = list.head.links()[0]
                .as_ref()
                .expect("head link after 2nd pop");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&30));
        }

        // Third pop_front: removes 30, head link becomes None
        assert_eq!(list.pop_front(), Some(30));
        assert_eq!(list.len(), 0);
        assert!(list.head.links()[0].is_none());
    }

    #[test]
    fn pop_front_interleaved_with_push() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(2); // [1, 2]
        assert_eq!(list.pop_front(), Some(1)); // [2]
        list.push_front(0); // [0, 2]
        list.push_back(3); // [0, 2, 3]
        assert_eq!(list.pop_front(), Some(0)); // [2, 3]
        assert_eq!(list.pop_front(), Some(2)); // [3]
        assert_eq!(list.pop_front(), Some(3)); // []
        assert_eq!(list.pop_front(), None);
        assert_eq!(list.len(), 0);
    }
}
