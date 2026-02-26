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
}
