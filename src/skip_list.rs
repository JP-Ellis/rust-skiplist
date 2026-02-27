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

use core::{
    ops::{Index, IndexMut},
    ptr::NonNull,
};

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

    /// Returns a reference to the element at position `index`, or `None` if
    /// `index` is out of bounds.
    ///
    /// This operation is O(log n) expected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(10);
    /// list.push_back(20);
    /// list.push_back(30);
    /// assert_eq!(list.get(0), Some(&10));
    /// assert_eq!(list.get(1), Some(&20));
    /// assert_eq!(list.get(2), Some(&30));
    /// assert_eq!(list.get(3), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }

        let target_rank = index.saturating_add(1);
        let mut current: &Node<T> = &self.head;
        let mut current_rank: usize = 0;

        for l in (0..self.head.level()).rev() {
            while let Some(Some(link)) = current.links().get(l) {
                let next_rank = current_rank.saturating_add(link.distance().get());
                if next_rank >= target_rank {
                    break;
                }
                current_rank = next_rank;
                current = link.node();
            }
        }

        current
            .links()
            .first()
            .and_then(Option::as_ref)
            .and_then(|link| link.node().value())
    }

    /// Returns a mutable reference to the element at position `index`, or
    /// `None` if `index` is out of bounds.
    ///
    /// This operation is O(log n) expected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(10);
    /// list.push_back(20);
    /// if let Some(v) = list.get_mut(0) {
    ///     *v = 42;
    /// }
    /// assert_eq!(list.get(0), Some(&42));
    /// assert_eq!(list.get(1), Some(&20));
    /// ```
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-pointer traversal and the final value_mut call are all on distinct, \
                  provably non-aliasing nodes owned by self; splitting into separate unsafe \
                  blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }

        let target_rank = index.saturating_add(1);
        let max_levels = self.head.level();

        // SAFETY: All raw pointers originate from heap allocations owned by this
        // SkipList. We have &mut self, guaranteeing exclusive access to all nodes.
        // We traverse using *const pointers (read-only) to avoid holding a live
        // &mut while reading links. The target pointer is converted to *mut only
        // when calling value_mut(); the returned &mut T has lifetime tied to
        // &mut self, and no other reference to the same node can exist.
        unsafe {
            let mut current: *const Node<T> = &raw const *self.head;
            let mut current_rank: usize = 0;

            for l in (0..max_levels).rev() {
                while let Some(Some(link)) = (*current).links().get(l) {
                    let next_rank = current_rank.saturating_add(link.distance().get());
                    if next_rank >= target_rank {
                        break;
                    }
                    current_rank = next_rank;
                    current = link.node();
                }
            }

            let target: NonNull<Node<T>> =
                NonNull::from((*current).links().first()?.as_ref()?.node());
            (*target.as_ptr()).value_mut()
        }
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

    /// Removes and returns the last element, or `None` if the list is empty.
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
    /// assert_eq!(list.pop_back(), Some(2));
    /// assert_eq!(list.pop_back(), Some(1));
    /// assert_eq!(list.pop_back(), None);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        clippy::unwrap_in_result,
        reason = "update[0].links[0] is Some because the list is non-empty and the traversal \
                  stops at the predecessor of the tail; \
                  all expects fire only on internal invariant violations, not user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l is bounded by max_levels = head.links.len(); any node reachable at level l \
                  has links.len() > l by the skip-list invariant, so all accesses are in bounds"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "traversal, link clearing, and node pop all touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let max_levels = self.head.level();

        // SAFETY: All raw pointers originate from heap allocations owned by this
        // SkipList.  No safe &mut references to any node exist while this block
        // runs.  update[] entries and back_ptr are distinct heap allocations; all
        // slice accesses are bounded by max_levels = head.links.len().
        let value = unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;

            // Build the update array: for each level l, update[l] is the
            // rightmost node from which the level-l link points to the tail
            // (i.e., following it would reach target_rank = self.len).
            //
            // Invariant maintained by push_back:
            //   For l < back_height, update[l].links[l] = Link(tail, d).
            //   For l >= back_height, update[l].links[l] is None or points
            //   to a node before the tail (never set by push_back at those
            //   levels, so no level-l link reaches the tail).
            let target_rank = self.len;
            let mut update: Vec<*mut Node<T>> = vec![head_ptr; max_levels];
            let mut current: *mut Node<T> = head_ptr;
            let mut current_rank: usize = 0;

            for l in (0..max_levels).rev() {
                // Advance as far right as possible WITHOUT reaching the tail.
                while let Some(link) = (*current).links()[l].as_ref() {
                    let next_rank = current_rank.saturating_add(link.distance().get());
                    if next_rank >= target_rank {
                        break;
                    }
                    current_rank = next_rank;
                    current = NonNull::from(link.node()).as_ptr();
                }
                update[l] = current;
            }

            // The tail is the node at level 0 immediately after update[0].
            // The list is non-empty, so this link must exist.
            let back_ptr: *mut Node<T> = NonNull::from(
                (*update[0]).links()[0]
                    .as_ref()
                    .expect("update[0].links[0] points to tail in a non-empty list")
                    .node(),
            )
            .as_ptr();

            let back_height = (*back_ptr).level();

            // Remove skip links that pointed to the tail.
            // For l < back_height, update[l].links[l] = Link(tail, d) — clear it.
            // For l >= back_height, no level-l link reaches the tail — leave as-is.
            for (l, &pred_ptr) in update.iter().enumerate().take(back_height) {
                (*pred_ptr).links_mut()[l] = None;
            }

            // Detach the tail from the prev/next chain.
            // pop() sets: tail.prev.next = None
            //             (tail.next is already None for the tail node)
            let mut popped = (*back_ptr).pop();
            popped.take_value()
        };

        self.len = self.len.saturating_sub(1);
        value
    }

    /// Inserts `value` at position `index`, shifting all elements at `index..`
    /// one position to the right.
    ///
    /// This operation is O(log n) expected.
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(1);
    /// list.push_back(3);
    /// list.insert(1, 2);
    /// assert_eq!(list.len(), 3);
    /// // list is now [1, 2, 3]
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "Link::new distances are computed to be ≥ 1; \
                  increment_distance overflow requires > usize::MAX nodes; \
                  all expects fire only on internal invariant violations, not user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < height ≤ max_levels = new_raw.links.len(); \
                  pred_ptr was reached at level l during traversal so pred_ptr.links.len() > l; \
                  all accesses are bounded by max_levels = head.links.len()"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "traversal, insertion, and link wiring all touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        assert!(
            index <= self.len,
            "insertion index (is {index}) should be <= len (is {})",
            self.len
        );

        let height = self.generator.level().saturating_add(1);
        let max_levels = self.head.level();
        // new_rank: the rank of the new node after insertion
        // (head = rank 0; element at index i has rank i + 1).
        let new_rank = index.saturating_add(1);

        // SAFETY: All raw pointers originate from heap allocations owned by this
        // SkipList.  No safe &mut references to any node exist while this block
        // runs.  Different update[] entries may alias (same predecessor node,
        // different levels) but each iteration accesses a distinct links[l]
        // index, so no simultaneous aliasing occurs.
        unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;

            // update[l] = (predecessor at level l, its rank from head).
            // Initialised to (head, 0): head is always a valid predecessor.
            let mut update: Vec<(*mut Node<T>, usize)> = vec![(head_ptr, 0_usize); max_levels];
            let mut current: *mut Node<T> = head_ptr;
            let mut current_rank: usize = 0;

            for l in (0..max_levels).rev() {
                while let Some(link) = (*current).links()[l].as_ref() {
                    let next_rank = current_rank.saturating_add(link.distance().get());
                    if next_rank >= new_rank {
                        break;
                    }
                    current_rank = next_rank;
                    current = NonNull::from(link.node()).as_ptr();
                }
                update[l] = (current, current_rank);
            }

            // Insert the new node right after the level-0 predecessor.
            let (pred0_ptr, _) = update[0];
            (*pred0_ptr).insert_after(Node::with_value(height, value));

            // Obtain a raw pointer to the new node without holding a live &mut.
            let new_raw: *mut Node<T> = NonNull::from(
                (*pred0_ptr)
                    .next_mut()
                    .expect("node was just inserted after predecessor"),
            )
            .as_ptr();

            // Wire skip links.
            //
            // For l < height (new node's tower reaches this level):
            //   Before: pred (rank D) ---[d]---> X (rank D + d)
            //   After:  pred (rank D) ---[new_rank − D]---> new_node (rank new_rank)
            //           new_node      ---[D + d + 1 − new_rank]---> X (rank D + d + 1)
            //
            // For l ≥ height (new node has no tower here):
            //   pred.links[l] still points to the same X (now at rank D + d + 1);
            //   increment its distance by 1.
            for (l, &(pred_ptr, pred_rank)) in update.iter().enumerate() {
                if l < height {
                    let distance = new_rank.saturating_sub(pred_rank);
                    let old_link = (*pred_ptr).links_mut()[l].take();
                    (*pred_ptr).links_mut()[l] =
                        Some(Link::new(&*new_raw, distance).expect("distance >= 1"));
                    (*new_raw).links_mut()[l] = if let Some(old) = old_link {
                        let new_d = old
                            .distance()
                            .get()
                            .saturating_sub(distance)
                            .saturating_add(1);
                        Some(Link::new(old.node(), new_d).expect("new_d >= 1"))
                    } else {
                        None
                    };
                } else if let Some(link) = (*pred_ptr).links_mut()[l].as_mut() {
                    link.increment_distance()
                        .expect("distance overflow requires > usize::MAX nodes");
                }
            }
        }

        self.len = self.len.saturating_add(1);
    }

    /// Removes and returns the element at position `index`.
    ///
    /// All elements after `index` shift one position to the left.
    ///
    /// This operation is O(log n) expected.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(1);
    /// list.push_back(2);
    /// list.push_back(3);
    /// assert_eq!(list.remove(1), 2);
    /// assert_eq!(list.len(), 2);
    /// // list is now [1, 3]
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "update[0].links[0] exists because index < len guarantees a node at target_rank; \
                  take_value is Some for any body/tail node; \
                  Link::new distance is computed to be >= 1; \
                  all expects fire only on internal invariant violations, not user input"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < target_height <= max_levels = target.links.len(); \
                  update[l].0 was reached at level l so update[l].0.links.len() > l; \
                  all accesses are bounded by max_levels = head.links.len()"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "traversal, link rewiring, and node pop all touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        assert!(
            index < self.len,
            "removal index (is {index}) should be < len (is {})",
            self.len
        );

        let max_levels = self.head.level();
        // head = rank 0; the element at logical index i is at rank i + 1.
        let target_rank = index.saturating_add(1);

        // SAFETY: All raw pointers originate from heap allocations owned by this
        // SkipList.  No safe &mut references to any node exist while this block
        // runs.  Different update[] entries may alias (same predecessor node,
        // different levels) but each iteration accesses a distinct links[l]
        // index, so no simultaneous aliasing occurs.
        let value = unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;

            // update[l] = (predecessor at level l, its rank from head).
            // Initialised to (head, 0): head is always a valid predecessor.
            let mut update: Vec<(*mut Node<T>, usize)> = vec![(head_ptr, 0_usize); max_levels];
            let mut current: *mut Node<T> = head_ptr;
            let mut current_rank: usize = 0;

            for l in (0..max_levels).rev() {
                while let Some(link) = (*current).links()[l].as_ref() {
                    let next_rank = current_rank.saturating_add(link.distance().get());
                    if next_rank >= target_rank {
                        break;
                    }
                    current_rank = next_rank;
                    current = NonNull::from(link.node()).as_ptr();
                }
                update[l] = (current, current_rank);
            }

            // The target is update[0]'s level-0 successor.
            // index < len guarantees this link exists.
            let target_ptr: *mut Node<T> = NonNull::from(
                (*update[0].0).links()[0]
                    .as_ref()
                    .expect("update[0].links[0] points to the target node")
                    .node(),
            )
            .as_ptr();

            let target_height = (*target_ptr).level();

            // Rewire skip links around the removed node.
            //
            // For l < target_height (target participates in the level-l list):
            //   Before: pred (rank P) --[T−P]--> target (rank T) --[d]--> X (rank T+d)
            //   After:  pred (rank P) --[(T−P)+d−1]--> X (now at rank T+d−1)
            //   If target.links[l] is None (target was the last at level l):
            //     pred.links[l] ← None
            //
            // For l >= target_height (target has no skip-link slot at this level):
            //   pred.links[l] already skips over target; decrement its distance by 1.
            //   The distance is >= 2 because pred_rank < target_rank and
            //   next_rank > target_rank (no level-l link can point to target at these
            //   levels, since target was never wired into this level during insertion).
            for (l, &(pred_ptr, pred_rank)) in update.iter().enumerate() {
                if l < target_height {
                    let pred_to_target = target_rank.saturating_sub(pred_rank);
                    let old_link = (*target_ptr).links_mut()[l].take();
                    (*pred_ptr).links_mut()[l] = if let Some(target_link) = old_link {
                        let new_dist = pred_to_target
                            .saturating_add(target_link.distance().get())
                            .saturating_sub(1);
                        Some(Link::new(target_link.node(), new_dist).expect("new_dist >= 1"))
                    } else {
                        None
                    };
                } else if let Some(link) = (*pred_ptr).links_mut()[l].as_mut() {
                    link.decrement_distance()
                        .expect("skip list invariant: distance >= 2 before decrement");
                }
            }

            // Detach the target node from the prev/next chain and take its value.
            let mut popped = (*target_ptr).pop();
            popped.take_value().expect("target node always has a value")
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

// MARK: Index

impl<T, G: LevelGenerator> Index<usize> for SkipList<T, G> {
    type Output = T;

    /// Returns a reference to the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[inline]
    #[expect(
        clippy::unwrap_used,
        reason = "index < self.len was just asserted, so get() always returns Some"
    )]
    fn index(&self, index: usize) -> &T {
        assert!(
            index < self.len,
            "index out of bounds: the len is {} but the index is {index}",
            self.len
        );
        self.get(index).unwrap()
    }
}

impl<T, G: LevelGenerator> IndexMut<usize> for SkipList<T, G> {
    /// Returns a mutable reference to the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[inline]
    #[expect(
        clippy::unwrap_used,
        reason = "index < self.len was just asserted, so get_mut() always returns Some"
    )]
    fn index_mut(&mut self, index: usize) -> &mut T {
        assert!(
            index < self.len,
            "index out of bounds: the len is {} but the index is {index}",
            self.len
        );
        self.get_mut(index).unwrap()
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

    // MARK: pop_back

    #[test]
    fn pop_back_from_empty() {
        let mut list = SkipList::<i32>::new();
        assert_eq!(list.pop_back(), None);
    }

    #[test]
    fn pop_back_single_element() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(42);
        assert_eq!(list.pop_back(), Some(42));
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(list.head.next().is_none());
        // Second pop on now-empty list
        assert_eq!(list.pop_back(), None);
    }

    #[test]
    fn pop_back_returns_in_reverse_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.pop_back(), Some(3));
        assert_eq!(list.pop_back(), Some(2));
        assert_eq!(list.pop_back(), Some(1));
        assert_eq!(list.pop_back(), None);
    }

    #[test]
    fn pop_back_len_decrements() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.push_back(i);
        }
        for remaining in (0..50_usize).rev() {
            list.pop_back();
            assert_eq!(list.len(), remaining);
        }
        assert_eq!(list.pop_back(), None);
    }

    /// With `with_capacity(1)` the generator always assigns height = 1.
    /// After three `push_back` calls followed by two `pop_back` calls
    /// the skip-link structure must be kept consistent.
    ///
    /// ```text
    /// Initial:  head → n1(10,d=1) → n2(20,d=1) → n3(30,d=1) → None
    /// After 1st pop_back (removes 30):
    ///   head → n1(10,d=1) → n2(20,d=1) → None
    /// After 2nd pop_back (removes 20):
    ///   head → n1(10,d=1) → None
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn pop_back_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);

        // First pop_back: removes 30
        assert_eq!(list.pop_back(), Some(30));
        assert_eq!(list.len(), 2);
        {
            // head.links[0] still → node(10) at distance 1
            let link: &Link<_> = list.head.links()[0]
                .as_ref()
                .expect("head link after 1st pop_back");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&10));
        }
        {
            // node(10).links[0] → node(20) at distance 1
            let n1 = list.head.next().expect("n1");
            let link: &Link<_> = n1.links()[0].as_ref().expect("n1 link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&20));
        }
        {
            // node(20).links[0] = None (was cleared by pop_back)
            let n2 = list.head.next().expect("n1").next().expect("n2");
            assert!(n2.links()[0].is_none());
        }

        // Second pop_back: removes 20
        assert_eq!(list.pop_back(), Some(20));
        assert_eq!(list.len(), 1);
        {
            // head.links[0] → node(10) at distance 1
            let link: &Link<_> = list.head.links()[0]
                .as_ref()
                .expect("head link after 2nd pop_back");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&10));
        }

        // Third pop_back: removes 10, head link becomes None
        assert_eq!(list.pop_back(), Some(10));
        assert_eq!(list.len(), 0);
        assert!(list.head.links()[0].is_none());
    }

    #[test]
    fn pop_back_interleaved_with_push() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(2); // [1, 2]
        assert_eq!(list.pop_back(), Some(2)); // [1]
        list.push_front(0); // [0, 1]
        list.push_back(3); // [0, 1, 3]
        assert_eq!(list.pop_back(), Some(3)); // [0, 1]
        assert_eq!(list.pop_back(), Some(1)); // [0]
        assert_eq!(list.pop_back(), Some(0)); // []
        assert_eq!(list.pop_back(), None);
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn pop_back_and_pop_front_together() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=6 {
            list.push_back(i); // [1, 2, 3, 4, 5, 6]
        }
        assert_eq!(list.pop_back(), Some(6)); // [1, 2, 3, 4, 5]
        assert_eq!(list.pop_front(), Some(1)); // [2, 3, 4, 5]
        assert_eq!(list.pop_back(), Some(5)); // [2, 3, 4]
        assert_eq!(list.pop_front(), Some(2)); // [3, 4]
        assert_eq!(list.pop_back(), Some(4)); // [3]
        assert_eq!(list.pop_front(), Some(3)); // []
        assert_eq!(list.pop_back(), None);
        assert_eq!(list.pop_front(), None);
        assert_eq!(list.len(), 0);
    }

    // MARK: insert

    #[test]
    fn insert_into_empty() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.insert(0, 42);
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
        assert_eq!(list.head.next().and_then(|n| n.value()), Some(&42));
        assert!(list.head.next().and_then(|n| n.next()).is_none());
    }

    #[test]
    fn insert_at_front() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(2);
        list.push_back(3);
        list.insert(0, 1);
        assert_eq!(list.len(), 3);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next().is_none());
    }

    #[test]
    fn insert_at_back() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.insert(2, 3);
        assert_eq!(list.len(), 3);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next().is_none());
    }

    #[test]
    fn insert_in_middle() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(3);
        list.insert(1, 2);
        assert_eq!(list.len(), 3);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next().expect("n3");
        assert_eq!(n3.value(), Some(&3));
        assert!(n3.next().is_none());
    }

    #[test]
    fn insert_len_increments() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.insert(0, i);
            assert_eq!(list.len(), i.saturating_add(1));
        }
    }

    #[test]
    #[should_panic(expected = "insertion index (is 5) should be <= len (is 3)")]
    fn insert_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.insert(5, 99);
    }

    /// With `with_capacity(1)` the generator assigns height = 1 to every node.
    /// Verify that links are correctly maintained after `insert(1, 2)` into [1, 3]:
    ///
    /// ```text
    /// Before: head ---[1]---> n1(1) ---[1]---> n3(3, links=None)
    /// After:  head ---[1]---> n1(1) ---[1]---> new(2) ---[1]---> n3(3, links=None)
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn insert_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(3); // [1, 3]
        list.insert(1, 2); // [1, 2, 3]

        assert_eq!(list.len(), 3);
        // head.links[0] → n1(1) at distance 1 (unchanged)
        {
            let link: &Link<_> = list.head.links()[0].as_ref().expect("head link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&1));
        }
        // n1(1).links[0] → new(2) at distance 1
        {
            let n1 = list.head.next().expect("n1");
            assert_eq!(n1.value(), Some(&1));
            let link: &Link<_> = n1.links()[0].as_ref().expect("n1 link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&2));
        }
        // new(2).links[0] → n3(3) at distance 1
        {
            let new_node = list.head.next().expect("n1").next().expect("new_node");
            assert_eq!(new_node.value(), Some(&2));
            let link: &Link<_> = new_node.links()[0].as_ref().expect("new_node link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&3));
        }
        // n3(3).links[0] = None
        {
            let n3 = list
                .head
                .next()
                .expect("n1")
                .next()
                .expect("new_node")
                .next()
                .expect("n3");
            assert_eq!(n3.value(), Some(&3));
            assert!(n3.links()[0].is_none());
        }
    }

    #[test]
    fn insert_interleaved_with_pop() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(3); // [1, 3]
        list.insert(1, 2); // [1, 2, 3]
        assert_eq!(list.pop_front(), Some(1)); // [2, 3]
        list.insert(0, 0); // [0, 2, 3]
        assert_eq!(list.pop_back(), Some(3)); // [0, 2]
        list.insert(2, 4); // [0, 2, 4]
        assert_eq!(list.len(), 3);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&0));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        let n3 = n2.next().expect("n3");
        assert_eq!(n3.value(), Some(&4));
        assert!(n3.next().is_none());
    }

    #[test]
    fn insert_multiple_positions() {
        let mut list = SkipList::<i32>::with_capacity(1);
        // Build [0, 1, 2, 3, 4] by inserting at various positions.
        list.insert(0, 2); // [2]
        list.insert(0, 0); // [0, 2]
        list.insert(1, 1); // [0, 1, 2]
        list.insert(3, 4); // [0, 1, 2, 4]
        list.insert(3, 3); // [0, 1, 2, 3, 4]
        assert_eq!(list.len(), 5);
        let mut node = list.head.next().expect("first");
        for expected in 0..5_i32 {
            assert_eq!(node.value(), Some(&expected));
            if expected < 4 {
                node = node.next().expect("next");
            }
        }
    }

    // MARK: remove

    #[test]
    fn remove_only_element() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(42);
        assert_eq!(list.remove(0), 42);
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
        assert!(list.head.next().is_none());
    }

    #[test]
    fn remove_at_front() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.remove(0), 1);
        assert_eq!(list.len(), 2);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&2));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&3));
        assert!(n2.next().is_none());
    }

    #[test]
    fn remove_at_back() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.remove(2), 3);
        assert_eq!(list.len(), 2);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&2));
        assert!(n2.next().is_none());
    }

    #[test]
    fn remove_in_middle() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.remove(1), 2);
        assert_eq!(list.len(), 2);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&1));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&3));
        assert!(n2.next().is_none());
    }

    #[test]
    fn remove_len_decrements() {
        let mut list = SkipList::<usize>::new();
        for i in 0..50_usize {
            list.push_back(i);
        }
        for i in (0..50_usize).rev() {
            assert_eq!(list.len(), i.saturating_add(1));
            list.remove(0);
        }
        assert_eq!(list.len(), 0);
    }

    #[test]
    #[should_panic(expected = "removal index (is 3) should be < len (is 3)")]
    fn remove_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.remove(3);
    }

    #[test]
    #[should_panic(expected = "removal index (is 0) should be < len (is 0)")]
    fn remove_from_empty() {
        let mut list = SkipList::<i32>::new();
        list.remove(0);
    }

    /// With `with_capacity(1)` the generator assigns height = 1 to every node.
    /// Verify that links are correctly maintained after removing the middle element
    /// from [1, 2, 3]:
    ///
    /// ```text
    /// Before: head ---[1]---> n1(1) ---[1]---> n2(2) ---[1]---> n3(3, None)
    /// After:  head ---[1]---> n1(1) ---[1]---> n3(3, None)
    /// ```
    #[expect(
        clippy::indexing_slicing,
        reason = "links slice length is known to be 1 for with_capacity(1)"
    )]
    #[test]
    fn remove_links_with_single_level() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(2); // [1, 2]
        list.push_back(3); // [1, 2, 3]
        assert_eq!(list.remove(1), 2); // [1, 3]

        assert_eq!(list.len(), 2);
        // head.links[0] → n1(1) at distance 1 (unchanged)
        {
            let link: &Link<_> = list.head.links()[0].as_ref().expect("head link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&1));
        }
        // n1(1).links[0] → n3(3) at distance 1 (previously pointed to n2)
        {
            let n1 = list.head.next().expect("n1");
            assert_eq!(n1.value(), Some(&1));
            let link: &Link<_> = n1.links()[0].as_ref().expect("n1 link");
            assert_eq!(link.distance().get(), 1);
            assert_eq!(link.node().value(), Some(&3));
        }
        // n3(3).links[0] = None
        {
            let n3 = list.head.next().expect("n1").next().expect("n3");
            assert_eq!(n3.value(), Some(&3));
            assert!(n3.links()[0].is_none());
        }
    }

    #[test]
    fn remove_interleaved_with_insert() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1); // [1]
        list.push_back(2); // [1, 2]
        list.push_back(3); // [1, 2, 3]
        assert_eq!(list.remove(1), 2); // [1, 3]
        list.insert(1, 4); // [1, 4, 3]
        assert_eq!(list.remove(0), 1); // [4, 3]
        list.push_back(5); // [4, 3, 5]
        assert_eq!(list.remove(2), 5); // [4, 3]
        assert_eq!(list.len(), 2);
        let n1 = list.head.next().expect("n1");
        assert_eq!(n1.value(), Some(&4));
        let n2 = n1.next().expect("n2");
        assert_eq!(n2.value(), Some(&3));
        assert!(n2.next().is_none());
    }

    #[test]
    fn remove_all_elements() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=5_i32 {
            list.push_back(i);
        }
        assert_eq!(list.remove(2), 3); // [1, 2, 4, 5]
        assert_eq!(list.remove(0), 1); // [2, 4, 5]
        assert_eq!(list.remove(2), 5); // [2, 4]
        assert_eq!(list.remove(1), 4); // [2]
        assert_eq!(list.remove(0), 2); // []
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
    }

    #[test]
    #[expect(
        clippy::integer_division,
        clippy::integer_division_remainder_used,
        reason = "removing from middle"
    )]
    fn remove_all_in_order_from_middle() {
        let n = 50_usize;
        let mut list = SkipList::<usize>::new();
        for i in 0..n {
            list.push_back(i);
        }
        while !list.is_empty() {
            list.remove(list.len() / 2);
        }
        assert_eq!(list.len(), 0);
        assert!(list.head.next().is_none());
    }

    // MARK: get / get_mut

    #[test]
    fn get_from_empty() {
        let list = SkipList::<i32>::new();
        assert_eq!(list.get(0), None);
    }

    #[test]
    fn get_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.get(3), None);
        assert_eq!(list.get(100), None);
    }

    #[test]
    fn get_first() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list.get(0), Some(&10));
    }

    #[test]
    fn get_last() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list.get(2), Some(&30));
    }

    #[test]
    fn get_middle() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list.get(1), Some(&20));
    }

    #[test]
    fn get_all_elements() {
        let n = 50_usize;
        let mut list = SkipList::<usize>::new();
        for i in 0..n {
            list.push_back(i);
        }
        for i in 0..n {
            assert_eq!(list.get(i), Some(&i));
        }
    }

    #[test]
    fn get_single_element() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(42);
        assert_eq!(list.get(0), Some(&42));
        assert_eq!(list.get(1), None);
    }

    #[test]
    fn get_mut_from_empty() {
        let mut list = SkipList::<i32>::new();
        assert_eq!(list.get_mut(0), None);
    }

    #[test]
    fn get_mut_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        assert_eq!(list.get_mut(1), None);
        assert_eq!(list.get_mut(99), None);
    }

    #[test]
    fn get_mut_modify() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);

        // Modify first element
        if let Some(v) = list.get_mut(0) {
            *v = 100;
        }
        assert_eq!(list.get(0), Some(&100));
        assert_eq!(list.get(1), Some(&20));
        assert_eq!(list.get(2), Some(&30));

        // Modify last element
        if let Some(v) = list.get_mut(2) {
            *v = 300;
        }
        assert_eq!(list.get(0), Some(&100));
        assert_eq!(list.get(1), Some(&20));
        assert_eq!(list.get(2), Some(&300));

        // Modify middle element
        if let Some(v) = list.get_mut(1) {
            *v = 200;
        }
        assert_eq!(list.get(0), Some(&100));
        assert_eq!(list.get(1), Some(&200));
        assert_eq!(list.get(2), Some(&300));
    }

    #[test]
    fn get_mut_all_elements() {
        let n = 30_usize;
        let mut list = SkipList::<usize>::new();
        for i in 0..n {
            list.push_back(i);
        }
        for i in 0..n {
            if let Some(v) = list.get_mut(i) {
                *v = i * 10;
            }
        }
        for i in 0..n {
            assert_eq!(list.get(i), Some(&(i * 10)));
        }
    }

    // MARK: Index / IndexMut

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing valid indexing behavior with known indices"
    )]
    fn index_basic() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list[0], 10);
        assert_eq!(list[1], 20);
        assert_eq!(list[2], 30);
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing valid mut indexing behavior with known indices"
    )]
    fn index_mut_basic() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        list[0] = 100;
        list[2] = 300;
        assert_eq!(list[0], 100);
        assert_eq!(list[1], 20);
        assert_eq!(list[2], 300);
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing out-of-bounds indexing behavior with known indices"
    )]
    #[should_panic(expected = "index out of bounds: the len is 3 but the index is 3")]
    fn index_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        _ = list[3];
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing out-of-bounds mut indexing behavior with known indices"
    )]
    #[should_panic(expected = "index out of bounds: the len is 0 but the index is 0")]
    fn index_empty() {
        let list = SkipList::<i32>::new();
        _ = list[0];
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing out-of-bounds mut indexing behavior with known indices"
    )]
    #[should_panic(expected = "index out of bounds: the len is 3 but the index is 5")]
    fn index_mut_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list[5] = 99;
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing valid indexing behavior after mutations with known indices"
    )]
    fn index_after_mutations() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10_i32 {
            list.push_back(i);
        }
        list.remove(3); // [0,1,2,4,5,6,7,8,9]
        list.insert(3, 42); // [0,1,2,42,4,5,6,7,8,9]
        assert_eq!(list[3], 42);
        assert_eq!(list[4], 4);
    }
}
