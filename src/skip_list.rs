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
    fmt,
    iter::FusedIterator,
    marker::PhantomData,
    ops::{Bound, Index, IndexMut, RangeBounds},
    ptr::NonNull,
};

use crate::{
    level_generator::{LevelGenerator, geometric::Geometric},
    node::{Node, link::Link},
};

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
    /// Non-owning pointer to the last data node, or `None` when the list is
    /// empty.  Maintained by every insert and remove operation to provide O(1)
    /// [`back`](SkipList::back) / [`back_mut`](SkipList::back_mut) access.
    tail: Option<NonNull<Node<T>>>,
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
        Self::with_level_generator(Geometric::default())
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
            tail: None,
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

    /// Returns a reference to the first element, or `None` if the list is
    /// empty.
    ///
    /// This operation is O(1).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// assert_eq!(list.front(), None);
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.front(), Some(&1));
    /// ```
    #[inline]
    #[must_use]
    pub fn front(&self) -> Option<&T> {
        self.head.next()?.value()
    }

    /// Returns a mutable reference to the first element, or `None` if the
    /// list is empty.
    ///
    /// This operation is O(1).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(10);
    /// list.push_back(20);
    /// if let Some(v) = list.front_mut() {
    ///     *v = 99;
    /// }
    /// assert_eq!(list.front(), Some(&99));
    /// assert_eq!(list.get(1), Some(&20));
    /// ```
    #[inline]
    #[must_use]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        // SAFETY: &mut self guarantees exclusive access to all nodes.  We
        // convert the shared reference from next() to a raw pointer and
        // re-borrow it as &mut T, which is sound because no other reference to
        // the front node can exist concurrently.  The `?` propagates None when
        // the list is empty.  The returned &mut T is bounded by &mut self.
        unsafe {
            let front_ptr: *mut Node<T> = NonNull::from(self.head.next()?).as_ptr();
            (*front_ptr).value_mut()
        }
    }

    /// Returns a reference to the last element, or `None` if the list is
    /// empty.
    ///
    /// This operation is O(1).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// assert_eq!(list.back(), None);
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.back(), Some(&2));
    /// ```
    #[inline]
    #[must_use]
    pub fn back(&self) -> Option<&T> {
        // SAFETY: self.tail is Some iff len > 0, an invariant maintained by all
        // mutating operations.  The pointer remains valid for the lifetime of &self.
        unsafe { self.tail?.as_ref().value() }
    }

    /// Returns a mutable reference to the last element, or `None` if the list
    /// is empty.
    ///
    /// This operation is O(1).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(10);
    /// list.push_back(20);
    /// if let Some(v) = list.back_mut() {
    ///     *v = 99;
    /// }
    /// assert_eq!(list.back(), Some(&99));
    /// assert_eq!(list.get(0), Some(&10));
    /// ```
    #[inline]
    #[must_use]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        // SAFETY: self.tail is Some iff len > 0, an invariant maintained by all
        // mutating operations.  &mut self guarantees exclusive access, so no
        // other reference to the tail node exists.  The returned &mut T is
        // bounded by &mut self's lifetime.
        unsafe { self.tail?.as_mut().value_mut() }
    }

    /// Removes all elements from the list.
    ///
    /// The level generator is preserved; elements can be inserted again
    /// immediately after calling `clear`.
    ///
    /// This operation is O(n): all n elements must be dropped.
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
    /// list.clear();
    /// assert!(list.is_empty());
    /// assert_eq!(list.len(), 0);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        let max_levels = self.head.level();
        // Replacing `*self.head` with a fresh sentinel node drops the old
        // sentinel in-place.  `Node::drop` iterates the entire `next` chain
        // and frees each node one at a time, so this is O(n) and
        // non-recursive regardless of list length.
        *self.head = Node::new(max_levels);
        self.tail = None;
        self.len = 0;
    }

    /// Shortens the list, keeping only the first `len` elements and dropping
    /// the rest.
    ///
    /// If `len >= self.len()`, this is a no-op.
    ///
    /// This operation is O(log n + k) where k = `self.len() − len` is the
    /// number of elements removed: O(log n) to locate the new tail and update
    /// the skip links, then O(k) to drop k values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.push_back(i);
    /// }
    /// list.truncate(3);
    /// assert_eq!(list.len(), 3);
    /// assert_eq!(list.get(0), Some(&1));
    /// assert_eq!(list.get(1), Some(&2));
    /// assert_eq!(list.get(2), Some(&3));
    /// assert_eq!(list.get(3), None);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "the node at rank `len` exists because 0 < len < self.len was checked before \
                  entering the unsafe block; the expect fires only on invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        clippy::needless_range_loop,
        reason = "l < max_levels = head.links.len(); every node in update[] was reached \
                  via a level-l link so its links.len() > l; all accesses are in bounds; \
                  l is used for both update[l] and links_mut()[l] so a plain index loop is \
                  the clearest expression"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "traversal, link clearing, and truncate_next all touch provably disjoint \
                  heap nodes; splitting across blocks would require unsafe-crossing raw-pointer \
                  variables"
    )]
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        if len == 0 {
            self.clear();
            return;
        }

        // 0 < len < self.len: keep elements at ranks 1..=len; drop the rest.
        let max_levels = self.head.level();

        // SAFETY: All raw pointers come from heap allocations owned by this
        // SkipList.  No safe references to any node exist while this block
        // runs.  Each node in update[] is distinct from the others (they are
        // at different levels in a skip-list traversal), and each links[] slot
        // is accessed at most once per level.
        let new_tail_ptr: *mut Node<T> = unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;

            // update[l] = (predecessor at level l, its rank).
            // Using target_rank = len and break when next_rank >= len means
            // update[l] holds the last node at level l with rank < len.
            // Therefore update[0].links[0] points to the new tail at rank len.
            let mut update: Vec<(*mut Node<T>, usize)> = vec![(head_ptr, 0_usize); max_levels];
            let mut current: *mut Node<T> = head_ptr;
            let mut current_rank: usize = 0;

            for l in (0..max_levels).rev() {
                while let Some(link) = (*current).links()[l].as_ref() {
                    let next_rank = current_rank.saturating_add(link.distance().get());
                    if next_rank >= len {
                        break;
                    }
                    current_rank = next_rank;
                    current = NonNull::from(link.node()).as_ptr();
                }
                update[l] = (current, current_rank);
            }

            // The new tail is the level-0 successor of update[0].
            // It exists because 0 < len < self.len.
            let new_tail_ptr: *mut Node<T> = NonNull::from(
                (*update[0].0).links()[0]
                    .as_ref()
                    .expect("the node at rank `len` exists because len < self.len")
                    .node(),
            )
            .as_ptr();

            let new_tail_height = (*new_tail_ptr).level();

            // Clear the new tail's own forward skip links: they point to
            // nodes that are about to be freed.
            for link in (*new_tail_ptr).links_mut() {
                *link = None;
            }

            // For levels at or above the new tail's height, the new tail does
            // not participate in those skip-link lists.  The predecessor at
            // each such level may have a skip link that spans past the cut;
            // clear it.
            for l in new_tail_height..max_levels {
                (*update[l].0).links_mut()[l] = None;
            }

            // Iteratively free all nodes after the new tail.
            // truncate_next() sets new_tail.next = None and drops the rest
            // of the chain in O(k) without recursion.
            (*new_tail_ptr).truncate_next();

            new_tail_ptr
        };

        // SAFETY: new_tail_ptr is a live, heap-allocated node owned by this
        // SkipList; it will not be freed until the list itself is dropped.
        self.tail = Some(unsafe { NonNull::new_unchecked(new_tail_ptr) });
        self.len = len;
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

        // If the list was empty the new node is also the tail.
        if self.len == 0 {
            self.tail = Some(new_node_ptr);
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
        let new_node_nonnull: NonNull<Node<T>> = unsafe {
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

            // SAFETY: new_raw was derived from NonNull::from(next_mut()), so it
            // is non-null.  Return it so self.tail can be updated outside.
            NonNull::new_unchecked(new_raw)
        };

        self.tail = Some(new_node_nonnull);
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
        // If that was the only element the list is now empty.
        if self.len == 0 {
            self.tail = None;
        }
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
        let (value, pred0) = unsafe {
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

            // Capture the level-0 predecessor before the node is popped.  When
            // len == 1, this equals head_ptr; the tail update below handles that.
            let pred0: *mut Node<T> = update[0];

            // Detach the tail from the prev/next chain.
            // pop() sets: tail.prev.next = None
            //             (tail.next is already None for the tail node)
            let mut popped = (*back_ptr).pop();
            (popped.take_value(), pred0)
        };

        // Update the cached tail pointer.  When the list becomes empty pred0
        // equals head_ptr; we set tail to None rather than pointing at head.
        self.tail = if self.len == 1 {
            None
        } else {
            // SAFETY: pred0 is a live data node pointer owned by this SkipList.
            Some(unsafe { NonNull::new_unchecked(pred0) })
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
        let new_node_nonnull: NonNull<Node<T>> = unsafe {
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

            // SAFETY: new_raw was derived from NonNull::from(next_mut()), so it
            // is non-null.  Return it so self.tail can be updated outside.
            NonNull::new_unchecked(new_raw)
        };

        // When inserting at the end the new node becomes the tail.
        if index == self.len {
            self.tail = Some(new_node_nonnull);
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
        let (value, pred0) = unsafe {
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
            // Also capture the level-0 predecessor for the tail update below.
            let pred0 = update[0].0;
            let mut popped = (*target_ptr).pop();
            (
                popped.take_value().expect("target node always has a value"),
                pred0,
            )
        };

        // When the last element is removed, update the cached tail pointer.
        if index.saturating_add(1) == self.len {
            self.tail = if self.len == 1 {
                None
            } else {
                // SAFETY: pred0 is a live data node pointer owned by this SkipList.
                Some(unsafe { NonNull::new_unchecked(pred0) })
            };
        }
        self.len = self.len.saturating_sub(1);
        value
    }

    // MARK: Iteration

    /// Returns an iterator over shared references to the elements of the list,
    /// from front to back.
    ///
    /// The iterator also supports [`DoubleEndedIterator`], allowing traversal
    /// in reverse order.  Advancing from both ends toward the middle is also
    /// supported: calls to [`Iterator::next`] and
    /// [`DoubleEndedIterator::next_back`] can be interleaved freely.
    ///
    /// This operation is O(1).
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
    ///
    /// let collected: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(collected, [1, 2, 3]);
    ///
    /// let reversed: Vec<i32> = list.iter().copied().rev().collect();
    /// assert_eq!(reversed, [3, 2, 1]);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            front: self.head.next().map(NonNull::from),
            back: self.tail,
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over mutable references to the elements of the
    /// list, from front to back.
    ///
    /// The iterator also supports [`DoubleEndedIterator`], allowing traversal
    /// in reverse order.  Advancing from both ends toward the middle is also
    /// supported: calls to [`Iterator::next`] and
    /// [`DoubleEndedIterator::next_back`] can be interleaved freely.
    ///
    /// This operation is O(1).
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
    ///
    /// for v in list.iter_mut() {
    ///     *v *= 2;
    /// }
    ///
    /// let collected: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(collected, [2, 4, 6]);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            front: self.head.next().map(NonNull::from),
            back: self.tail,
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over shared references to elements within the
    /// given index range.
    ///
    /// The iterator supports [`DoubleEndedIterator`], [`ExactSizeIterator`],
    /// and [`FusedIterator`].  Setting up the iterator navigates to the
    /// start and end nodes in O(log n) via the skip links.
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or if the
    /// end is greater than [`self.len()`][SkipList::len].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.push_back(i);
    /// }
    ///
    /// let slice: Vec<i32> = list.range(1..4).copied().collect();
    /// assert_eq!(slice, [2, 3, 4]);
    ///
    /// let reversed: Vec<i32> = list.range(1..4).copied().rev().collect();
    /// assert_eq!(reversed, [4, 3, 2]);
    /// ```
    #[inline]
    pub fn range<R: RangeBounds<usize>>(&self, range: R) -> Iter<'_, T> {
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s.saturating_add(1),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&e) => e.saturating_add(1),
            Bound::Excluded(&e) => e,
            Bound::Unbounded => self.len,
        };
        assert!(
            start <= end,
            "range start (is {start}) must be ≤ end (is {end})"
        );
        assert!(
            end <= self.len,
            "range end (is {end}) must be ≤ len (is {})",
            self.len
        );
        let count = end.saturating_sub(start);
        if count == 0 {
            return Iter {
                front: None,
                back: None,
                len: 0,
                _marker: PhantomData,
            };
        }
        Iter {
            front: Some(self.node_ptr_at(start)),
            back: Some(self.node_ptr_at(end.saturating_sub(1))),
            len: count,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over mutable references to elements within the
    /// given index range.
    ///
    /// The iterator supports [`DoubleEndedIterator`], [`ExactSizeIterator`],
    /// and [`FusedIterator`].  Setting up the iterator navigates to the
    /// start and end nodes in O(log n) via the skip links.
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or if the
    /// end is greater than [`self.len()`][SkipList::len].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.push_back(i);
    /// }
    ///
    /// for v in list.range_mut(1..4) {
    ///     *v *= 10;
    /// }
    /// let collected: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(collected, [1, 20, 30, 40, 5]);
    /// ```
    #[inline]
    pub fn range_mut<R: RangeBounds<usize>>(&mut self, range: R) -> IterMut<'_, T> {
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s.saturating_add(1),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&e) => e.saturating_add(1),
            Bound::Excluded(&e) => e,
            Bound::Unbounded => self.len,
        };
        assert!(
            start <= end,
            "range start (is {start}) must be ≤ end (is {end})"
        );
        assert!(
            end <= self.len,
            "range end (is {end}) must be ≤ len (is {})",
            self.len
        );
        let count = end.saturating_sub(start);
        if count == 0 {
            return IterMut {
                front: None,
                back: None,
                len: 0,
                _marker: PhantomData,
            };
        }
        IterMut {
            front: Some(self.node_ptr_at(start)),
            back: Some(self.node_ptr_at(end.saturating_sub(1))),
            len: count,
            _marker: PhantomData,
        }
    }

    /// Returns a [`NonNull`] pointer to the data node at the given 0-based
    /// `index`.  The caller must ensure `index < self.len`.
    #[expect(
        clippy::expect_used,
        reason = "the level-0 successor of the found predecessor is guaranteed to \
                  exist because callers (range / range_mut / split_off) validate \
                  index < self.len before calling this helper; the expect fires \
                  only on internal invariant violations"
    )]
    #[inline]
    fn node_ptr_at(&self, index: usize) -> NonNull<Node<T>> {
        debug_assert!(
            index < self.len,
            "index {index} out of bounds (len={})",
            self.len
        );
        let target_rank = index.saturating_add(1);
        let mut current: &Node<T> = &self.head;
        let mut current_rank: usize = 0;
        let levels = self.head.level();
        for l in (0..levels).rev() {
            while let Some(Some(link)) = current.links().get(l) {
                let next_rank = current_rank.saturating_add(link.distance().get());
                if next_rank >= target_rank {
                    break;
                }
                current_rank = next_rank;
                current = link.node();
            }
        }
        NonNull::from(
            current
                .links()
                .first()
                .and_then(Option::as_ref)
                .expect("node at index exists because index < self.len")
                .node(),
        )
    }

    /// Splits the list at the given index, returning a new list containing
    /// all elements from index `at` onward.
    ///
    /// After the call, `self` contains elements at indices `[0, at)` and the
    /// returned list contains elements previously at indices `[at, len)`,
    /// renumbered from 0 in the returned list.
    ///
    /// Skip links are rebuilt for both halves in a single O(n) pass each.
    /// Navigation to the split point is O(log n).
    ///
    /// # Panics
    ///
    /// Panics if `at > self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut a = SkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     a.push_back(i);
    /// }
    ///
    /// let b = a.split_off(3);
    /// let a_vals: Vec<i32> = a.iter().copied().collect();
    /// let b_vals: Vec<i32> = b.iter().copied().collect();
    /// assert_eq!(a_vals, [1, 2, 3]);
    /// assert_eq!(b_vals, [4, 5]);
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "`take_next_chain` returns None only if there is no successor, \
                  which cannot happen when at < self.len (validated above); \
                  Link::new(dist) succeeds because dist >= 1 by construction"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "take_next_chain, set_head_next, NonNull::new_unchecked, and \
                  rebuild_skip_links all touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer \
                  variables"
    )]
    #[inline]
    #[must_use]
    pub fn split_off(&mut self, at: usize) -> Self
    where
        G: Clone,
    {
        assert!(
            at <= self.len,
            "split_off index {at} is out of bounds (len = {})",
            self.len
        );

        let tail_len = self.len.saturating_sub(at);
        let max_levels = self.head.level();

        // ── Edge case: nothing to split off ──────────────────────────────
        if tail_len == 0 {
            return Self {
                head: Box::new(Node::new(max_levels)),
                tail: None,
                len: 0,
                generator: self.generator.clone(),
            };
        }

        // ── Edge case: split at position 0 — transfer everything ─────────
        if at == 0 {
            let old_len = self.len;
            let mut result = Self {
                head: Box::new(Node::new(max_levels)),
                tail: None, // set by rebuild below
                len: old_len,
                generator: self.generator.clone(),
            };

            // Transfer the entire node chain from self.head to result.head.
            // SAFETY: We hold &mut self; no other references to any node
            // exist.  take_next_chain detaches cleanly.  set_head_next
            // wires the first node to result.head.
            unsafe {
                let head_ptr: *mut Node<T> = &raw mut *self.head;
                if let Some(first_nn) = (*head_ptr).take_next_chain() {
                    result.head.set_head_next(first_nn);
                }
            }

            // Clear self.head's skip links (now all-None).
            for link in self.head.links_mut() {
                *link = None;
            }
            self.tail = None;
            self.len = 0;

            // Rebuild result's skip links (result.head is new; data nodes'
            // inter-node links are stale for the new head).
            // SAFETY: result.head is exclusively owned; all nodes reachable
            // via result.head.next are live heap allocations.
            unsafe {
                result.tail = Self::rebuild_skip_links(&mut result.head, max_levels);
            }

            return result;
        }

        // ── General case: 0 < at < self.len ──────────────────────────────
        //
        // Navigate to the pivot (node[at − 1], the last node to keep in
        // self), detach the tail chain, wire it to a fresh head, then
        // rebuild skip links for both halves.
        //
        // SAFETY: at > 0 and at < self.len, so node_ptr_at(at − 1) returns
        // a valid data node.  We hold &mut self, so exclusive access is
        // guaranteed throughout.
        unsafe {
            let pivot: *mut Node<T> = self.node_ptr_at(at.saturating_sub(1)).as_ptr();

            // Detach nodes [at ..] from the pivot.  Guaranteed to succeed
            // because at < self.len means the pivot has at least one
            // successor.
            let first_of_tail = (*pivot)
                .take_next_chain()
                .expect("pivot has a successor because at < self.len");

            // Build the returned list.
            let mut result = Self {
                head: Box::new(Node::new(max_levels)),
                tail: None, // set by rebuild below
                len: tail_len,
                generator: self.generator.clone(),
            };
            result.head.set_head_next(first_of_tail);

            // Update self's tail and length.
            self.tail = Some(NonNull::new_unchecked(pivot));
            self.len = at;

            // Rebuild skip links for self (nodes 0 .. at).
            self.tail = Self::rebuild_skip_links(&mut self.head, max_levels);

            // Rebuild skip links for result (nodes at .. original_len).
            result.tail = Self::rebuild_skip_links(&mut result.head, max_levels);

            result
        }
    }

    /// Rebuilds all skip links for the list rooted at `head` in a single
    /// O(n) forward pass.  Returns the last data node as a `NonNull`, or
    /// `None` if the list is empty.
    ///
    /// After the call every skip link in every node (including `head`) is
    /// consistent with the current prev/next chain.
    ///
    /// # Safety
    ///
    /// `head` must be the exclusively-owned head sentinel of a valid
    /// prev/next chain of heap-allocated [`Node<T>`] instances.  No other
    /// live reference to any node in the chain may exist.
    #[expect(
        clippy::expect_used,
        reason = "Link::new(dist) succeeds because dist >= 1 by construction \
                  (new_rank is incremented before use and pred_rank < new_rank)"
    )]
    #[expect(
        clippy::indexing_slicing,
        clippy::needless_range_loop,
        reason = "l < node_height <= max_levels = predecessors.len() = head.links.len(); \
                  l indexes both predecessors[l] and links_mut()[l] simultaneously"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-pointer traversal, link clearing, and link wiring all touch \
                  provably disjoint heap nodes; grouping them avoids unsafe-crossing \
                  raw-pointer variables"
    )]
    unsafe fn rebuild_skip_links(
        head: &mut Node<T>,
        max_levels: usize,
    ) -> Option<NonNull<Node<T>>> {
        // Clear head's skip links; they will be fully rebuilt below.
        for link in head.links_mut() {
            *link = None;
        }

        let head_ptr: *mut Node<T> = head as *mut Node<T>;
        let mut predecessors: Vec<(*mut Node<T>, usize)> = vec![(head_ptr, 0_usize); max_levels];
        let mut new_rank: usize = 0;
        let mut new_tail: Option<NonNull<Node<T>>> = None;

        // SAFETY: head_ptr and all nodes reachable via next are live,
        // exclusively-owned, heap-allocated Node<T> instances.  We do not
        // create any overlapping references.
        unsafe {
            let mut cur_opt = (*head_ptr).next().map(NonNull::from);
            while let Some(cur_nn) = cur_opt {
                let cur: *mut Node<T> = cur_nn.as_ptr();
                cur_opt = (*cur).next().map(NonNull::from);

                new_rank = new_rank.saturating_add(1);
                new_tail = Some(cur_nn);

                // Clear this node's forward skip links; they will be re-wired
                // when its successors are processed.
                for link in (*cur).links_mut() {
                    *link = None;
                }

                let height = (*cur).level();
                for l in 0..height {
                    let (pred_ptr, pred_rank) = predecessors[l];
                    let dist = new_rank.saturating_sub(pred_rank);
                    (*pred_ptr).links_mut()[l] =
                        Some(Link::new(&*cur, dist).expect("dist >= 1 by construction"));
                    predecessors[l] = (cur, new_rank);
                }
            }
        }

        new_tail
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, removes all elements `e` for which `f(&e)` returns
    /// `false`.  The elements are visited in order and, in the kept subset,
    /// their relative order is preserved.
    ///
    /// This operation is O(n): every element is visited once and skip links
    /// are rebuilt in a single linear pass.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.push_back(i);
    /// }
    /// list.retain(|&x| x % 2 == 0);
    /// let collected: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(collected, [2, 4]);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "`value()` returns None only for the head sentinel, which is never \
                  visited in the data-node walk; the expect fires only on invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        clippy::needless_range_loop,
        reason = "l < node_height ≤ max_levels = predecessors.len() = head.links.len(); \
                  l is used for both predecessors[l] and links_mut()[l] so an index loop \
                  is the clearest expression"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-pointer traversal, link clearing, link wiring, and node pop all \
                  touch provably disjoint heap nodes; splitting across blocks would \
                  require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        if self.is_empty() {
            return;
        }

        let max_levels = self.head.level();

        // SAFETY: All raw pointers originate from heap allocations owned by
        // this SkipList.  We hold &mut self so no other reference to any node
        // exists.  We save `next_opt` before calling `pop()`, so the walk
        // remains correct even after removing a node.  Each level's predecessor
        // pointer is set once (when its successor is processed) and then only
        // read, so there is no simultaneous aliasing.
        unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;

            // Clear head's skip links; they will be fully rebuilt below.
            for link in (*head_ptr).links_mut() {
                *link = None;
            }

            // predecessors[l] = (node_ptr, rank_of_that_node_in_new_list)
            let mut predecessors: Vec<(*mut Node<T>, usize)> =
                vec![(head_ptr, 0_usize); max_levels];
            let mut new_rank: usize = 0;
            let mut new_tail: Option<NonNull<Node<T>>> = None;

            let mut current_opt: Option<NonNull<Node<T>>> = (*head_ptr).next().map(NonNull::from);

            while let Some(current_nn) = current_opt {
                let current: *mut Node<T> = current_nn.as_ptr();
                // Save the successor before any mutation.
                let next_opt: Option<NonNull<Node<T>>> = (*current).next().map(NonNull::from);

                let keep = f((*current).value().expect("data node has a value"));

                if keep {
                    new_rank = new_rank.saturating_add(1);
                    new_tail = Some(current_nn);

                    // Clear this node's own forward skip links; they will be
                    // re-wired when its successors are processed below.
                    for link in (*current).links_mut() {
                        *link = None;
                    }

                    // Wire each predecessor level to this node.
                    let node_height = (*current).level();
                    for l in 0..node_height {
                        let (pred_ptr, pred_rank) = predecessors[l];
                        let dist = new_rank.saturating_sub(pred_rank);
                        (*pred_ptr).links_mut()[l] = Some(
                            Link::new(&*current, dist).expect("dist = new_rank - pred_rank ≥ 1"),
                        );
                        predecessors[l] = (current, new_rank);
                    }
                } else {
                    // Detach from the prev/next chain and free.
                    let boxed = (*current).pop();
                    drop(boxed);
                }

                current_opt = next_opt;
            }

            self.tail = new_tail;
            self.len = new_rank;
        }
    }

    /// Retains only the elements specified by the predicate, passing a mutable
    /// reference to each element.
    ///
    /// In other words, removes all elements `e` for which `f(&mut e)` returns
    /// `false`.  The elements are visited in order; the predicate may mutate
    /// retained elements before it returns `true`.
    ///
    /// This operation is O(n).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.push_back(i);
    /// }
    /// list.retain_mut(|x| {
    ///     if *x % 2 == 0 {
    ///         *x *= 10;
    ///         true
    ///     } else {
    ///         false
    ///     }
    /// });
    /// let collected: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(collected, [20, 40]);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "`value_mut()` returns None only for the head sentinel, which is never \
                  visited in the data-node walk; the expect fires only on invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        clippy::needless_range_loop,
        reason = "l < node_height ≤ max_levels = predecessors.len() = head.links.len(); \
                  l is used for both predecessors[l] and links_mut()[l] so an index loop \
                  is the clearest expression"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-pointer traversal, link clearing, link wiring, and node pop all \
                  touch provably disjoint heap nodes; splitting across blocks would \
                  require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        if self.is_empty() {
            return;
        }

        let max_levels = self.head.level();

        // SAFETY: All raw pointers originate from heap allocations owned by
        // this SkipList.  We hold &mut self so no other reference to any node
        // exists.  `value_mut()` yields an exclusive reference that expires
        // before we call `links_mut()` or `pop()` on the same node —
        // the predicate returns its `bool` result before any structural
        // mutation occurs.  We save `next_opt` before any mutation.
        unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;

            // Clear head's skip links; they will be fully rebuilt below.
            for link in (*head_ptr).links_mut() {
                *link = None;
            }

            let mut predecessors: Vec<(*mut Node<T>, usize)> =
                vec![(head_ptr, 0_usize); max_levels];
            let mut new_rank: usize = 0;
            let mut new_tail: Option<NonNull<Node<T>>> = None;

            let mut current_opt: Option<NonNull<Node<T>>> = (*head_ptr).next().map(NonNull::from);

            while let Some(current_nn) = current_opt {
                let current: *mut Node<T> = current_nn.as_ptr();
                let next_opt: Option<NonNull<Node<T>>> = (*current).next().map(NonNull::from);

                // The &mut T borrow ends when `f` returns; no aliasing with
                // the subsequent `links_mut()` / `pop()` calls.
                let keep = f((*current).value_mut().expect("data node has a value"));

                if keep {
                    new_rank = new_rank.saturating_add(1);
                    new_tail = Some(current_nn);

                    for link in (*current).links_mut() {
                        *link = None;
                    }

                    let node_height = (*current).level();
                    for l in 0..node_height {
                        let (pred_ptr, pred_rank) = predecessors[l];
                        let dist = new_rank.saturating_sub(pred_rank);
                        (*pred_ptr).links_mut()[l] = Some(
                            Link::new(&*current, dist).expect("dist = new_rank - pred_rank ≥ 1"),
                        );
                        predecessors[l] = (current, new_rank);
                    }
                } else {
                    let boxed = (*current).pop();
                    drop(boxed);
                }

                current_opt = next_opt;
            }

            self.tail = new_tail;
            self.len = new_rank;
        }
    }

    /// Removes the elements in the given index range from the list and returns
    /// them as an iterator.
    ///
    /// The range is specified by index (0-based, same as [`SkipList::get`]).
    /// All valid Rust range expressions are supported: `..`, `a..`, `..b`,
    /// `..=b`, `a..b`, `a..=b`.
    ///
    /// Elements outside the range are retained and remain accessible via the
    /// list after the `Drain` is consumed or dropped.
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or if the end
    /// is greater than `self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.push_back(i);
    /// }
    ///
    /// // Drain the middle three elements.
    /// let drained: Vec<i32> = list.drain(1..4).collect();
    /// assert_eq!(drained, [2, 3, 4]);
    /// assert_eq!(list.len(), 2);
    /// let remaining: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(remaining, [1, 5]);
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "`take_value()` returns None only for the head sentinel, which is never \
                  in the drain range; the expect fires only on invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        clippy::needless_range_loop,
        reason = "l < node_height ≤ max_levels = predecessors.len() = head.links.len(); \
                  l is used for both predecessors[l] and links_mut()[l] so an index loop \
                  is the clearest expression"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-pointer traversal, link clearing, link wiring, and node pop all \
                  touch provably disjoint heap nodes; splitting across blocks would \
                  require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T>
    where
        R: RangeBounds<usize>,
    {
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s.saturating_add(1),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&e) => e.saturating_add(1),
            Bound::Excluded(&e) => e,
            Bound::Unbounded => self.len,
        };

        assert!(
            start <= end,
            "drain range start (is {start}) must be ≤ end (is {end})"
        );
        assert!(
            end <= self.len,
            "drain range end (is {end}) must be ≤ len (is {})",
            self.len
        );

        let drain_count = end.saturating_sub(start);
        let mut drained: Vec<T> = Vec::with_capacity(drain_count);

        if drain_count == 0 {
            return Drain {
                iter: drained.into_iter(),
                _marker: PhantomData,
            };
        }

        let max_levels = self.head.level();

        // SAFETY: Same as retain — all raw pointers come from heap allocations
        // owned by this SkipList.  We hold &mut self.  next_opt is saved before
        // any mutation.  Drained nodes are popped (prev/next updated) and then
        // freed after their value is extracted.
        unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.head;

            // Clear head's skip links; rebuilt below.
            for link in (*head_ptr).links_mut() {
                *link = None;
            }

            let mut predecessors: Vec<(*mut Node<T>, usize)> =
                vec![(head_ptr, 0_usize); max_levels];
            let mut new_rank: usize = 0;
            let mut new_tail: Option<NonNull<Node<T>>> = None;
            let mut current_index: usize = 0;

            let mut current_opt: Option<NonNull<Node<T>>> = (*head_ptr).next().map(NonNull::from);

            while let Some(current_nn) = current_opt {
                let current: *mut Node<T> = current_nn.as_ptr();
                let next_opt: Option<NonNull<Node<T>>> = (*current).next().map(NonNull::from);

                let in_drain_range = current_index >= start && current_index < end;

                if in_drain_range {
                    // Extract value then free the node.
                    let mut boxed = (*current).pop();
                    drained.push(boxed.take_value().expect("data node has a value"));
                    drop(boxed);
                } else {
                    // Keep this node and wire skip links.
                    new_rank = new_rank.saturating_add(1);
                    new_tail = Some(current_nn);

                    for link in (*current).links_mut() {
                        *link = None;
                    }

                    let node_height = (*current).level();
                    for l in 0..node_height {
                        let (pred_ptr, pred_rank) = predecessors[l];
                        let dist = new_rank.saturating_sub(pred_rank);
                        (*pred_ptr).links_mut()[l] = Some(
                            Link::new(&*current, dist).expect("dist = new_rank - pred_rank ≥ 1"),
                        );
                        predecessors[l] = (current, new_rank);
                    }
                }

                current_index = current_index.saturating_add(1);
                current_opt = next_opt;
            }

            self.tail = new_tail;
            self.len = new_rank;
        }

        Drain {
            iter: drained.into_iter(),
            _marker: PhantomData,
        }
    }

    /// Creates a lazy iterator that removes and yields every element for
    /// which `pred` returns `true`.
    ///
    /// Elements for which `pred` returns `false` are kept in the list.
    /// The predicate receives a `&mut T` so it may inspect or mutate the
    /// element before deciding whether to extract it.
    ///
    /// If the `ExtractIf` iterator is dropped before being fully consumed,
    /// the predicate is **not** called for the remaining elements — they all
    /// stay in the list.  The list remains valid and fully usable after the
    /// iterator is dropped.
    ///
    /// This operation is O(n) for a full traversal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.push_back(i);
    /// }
    ///
    /// let evens: Vec<i32> = list.extract_if(|x| *x % 2 == 0).collect();
    /// assert_eq!(evens, [2, 4]);
    /// let remaining: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(remaining, [1, 3, 5]);
    /// ```
    #[inline]
    pub fn extract_if<F>(&mut self, pred: F) -> ExtractIf<'_, T, G, F>
    where
        F: FnMut(&mut T) -> bool,
    {
        let current = self.head.next().map(NonNull::from);
        ExtractIf {
            current,
            any_removed: false,
            list: self,
            pred,
        }
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

// MARK: IntoIterator

impl<'a, T, G: LevelGenerator> IntoIterator for &'a SkipList<T, G> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, G: LevelGenerator> IntoIterator for &'a mut SkipList<T, G> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, G: LevelGenerator> IntoIterator for SkipList<T, G> {
    type Item = T;
    type IntoIter = IntoIter<T, G>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { list: self }
    }
}

// MARK: Iter

/// An iterator over shared references to the elements of a [`SkipList`].
///
/// This struct is created by the [`SkipList::iter`] method.  See its
/// documentation for more.
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
///
/// let mut iter = list.iter();
/// assert_eq!(iter.next(), Some(&1));
/// assert_eq!(iter.next_back(), Some(&3));
/// assert_eq!(iter.next(), Some(&2));
/// assert_eq!(iter.next(), None);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Iter<'a, T> {
    /// Pointer to the next element to yield from the front, or `None` when
    /// the iterator is exhausted or the list was empty.
    front: Option<NonNull<Node<T>>>,
    /// Pointer to the next element to yield from the back, or `None` when
    /// the iterator is exhausted or the list was empty.
    back: Option<NonNull<Node<T>>>,
    /// Number of elements remaining.  Guards against yielding more than
    /// `len` items even when `front` and `back` pointers become stale after
    /// crossing mid-list during interleaved `next`/`next_back` calls.
    len: usize,
    /// Ties the iterator's lifetime to `&'a SkipList` and expresses
    /// covariance in `T`.
    _marker: PhantomData<&'a T>,
}

// SAFETY: Iter<'a, T> yields `&'a T` (shared, non-owning references).
// Sending it to another thread requires T: Sync because the receiving
// thread will read T values through a shared reference derived from the
// raw pointer carried by this type.
unsafe impl<T: Sync> Send for Iter<'_, T> {}

// SAFETY: Sharing &Iter<'a, T> across threads is safe when T: Sync.
// Concurrent callers need &mut Iter to advance it, so data races on the
// iterator's own fields are prevented by the requirement for exclusive
// access through &mut.
unsafe impl<T: Sync> Sync for Iter<'_, T> {}

impl<T> Clone for Iter<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Iter {
            front: self.front,
            back: self.back,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let front_ptr = self.front?;
        // SAFETY: front_ptr was derived from a heap-allocated Node<T> owned
        // by the SkipList that created this Iter.  The iterator holds a
        // shared borrow of that list for lifetime 'a, ensuring every node
        // remains allocated and reachable for the iterator's entire lifetime.
        // No &mut references to any node exist while this shared Iter is
        // alive.
        let node: &'a Node<T> = unsafe { front_ptr.as_ref() };
        self.front = node.next().map(NonNull::from);
        self.len = self.len.saturating_sub(1);
        node.value()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let back_ptr = self.back?;
        // SAFETY: Same provenance argument as front_ptr in next().
        // back_ptr points to a live data node for the 'a lifetime.
        let node: &'a Node<T> = unsafe { back_ptr.as_ref() };
        // Walk backward.  The head sentinel has no value; the filter ensures
        // `back` becomes None when we step past the first data node.
        // `len` independently prevents accessing a stale `back` pointer.
        self.back = node
            .prev()
            .filter(|p| p.value().is_some())
            .map(NonNull::from);
        self.len = self.len.saturating_sub(1);
        node.value()
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<T> FusedIterator for Iter<'_, T> {}

// MARK: IterMut

/// An iterator over mutable references to the elements of a [`SkipList`].
///
/// This struct is created by the [`SkipList::iter_mut`] method.  See its
/// documentation for more.
///
/// Unlike [`Iter`], `IterMut` does not implement [`Clone`]: cloning would
/// allow two independent iterators each holding `&mut T` references to the
/// same elements, violating Rust's aliasing rules.
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
///
/// let mut iter = list.iter_mut();
/// assert_eq!(iter.next(), Some(&mut 1));
/// assert_eq!(iter.next_back(), Some(&mut 3));
/// assert_eq!(iter.next(), Some(&mut 2));
/// assert_eq!(iter.next(), None);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IterMut<'a, T> {
    /// Pointer to the next element to yield from the front, or `None` when
    /// the iterator is exhausted or the list was empty.
    front: Option<NonNull<Node<T>>>,
    /// Pointer to the next element to yield from the back, or `None` when
    /// the iterator is exhausted or the list was empty.
    back: Option<NonNull<Node<T>>>,
    /// Number of elements remaining.  Guards against yielding more than
    /// `len` items even when `front` and `back` pointers become stale after
    /// crossing mid-list during interleaved `next`/`next_back` calls.
    len: usize,
    /// Ties the iterator's lifetime to `&'a mut SkipList` and expresses
    /// invariance in `T` (required for mutable references).
    _marker: PhantomData<&'a mut T>,
}

// SAFETY: IterMut<'a, T> yields `&'a mut T` (exclusive references).
// Sending it to another thread requires T: Send because the receiving
// thread will get exclusive access to T values through those references.
unsafe impl<T: Send> Send for IterMut<'_, T> {}

// SAFETY: Sharing &IterMut<'a, T> across threads is safe when T: Sync.
// Advancing the iterator requires &mut IterMut, so concurrent advancement
// is prevented by the requirement for exclusive access.
unsafe impl<T: Sync> Sync for IterMut<'_, T> {}

impl<T: fmt::Debug> fmt::Debug for IterMut<'_, T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Traverse via shared references for the purposes of display.
        // We hold &self, so no mutable access is ongoing.
        let mut builder = f.debug_list();
        let mut node_ptr = self.front;
        let mut remaining = self.len;
        while remaining > 0 {
            let Some(ptr) = node_ptr else { break };
            // SAFETY: ptr is a valid, aligned pointer to a live Node<T> for
            // lifetime 'a.  We only read through it here (no mutation), and
            // we hold &self which prevents concurrent mutable access.
            let node = unsafe { ptr.as_ref() };
            if let Some(v) = node.value() {
                builder.entry(v);
            }
            node_ptr = node.next().map(NonNull::from);
            remaining = remaining.saturating_sub(1);
        }
        builder.finish()
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let mut front_ptr = self.front?;
        // SAFETY: front_ptr was derived from a heap-allocated Node<T> owned
        // by the SkipList that created this IterMut.  The iterator holds an
        // exclusive borrow of that list for lifetime 'a, ensuring every node
        // remains allocated and non-aliased for the iterator's entire
        // lifetime.  We advance self.front before returning, so no two calls
        // to next() can yield a reference to the same node.
        let node: &'a mut Node<T> = unsafe { front_ptr.as_mut() };
        // node.next() is an immutable reborrow; it ends before value_mut().
        self.front = node.next().map(NonNull::from);
        self.len = self.len.saturating_sub(1);
        node.value_mut()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let mut back_ptr = self.back?;
        // SAFETY: Same provenance argument as front_ptr in next().
        // back_ptr points to a live data node for the 'a lifetime, and no
        // other mutable reference to it exists while this IterMut is alive.
        let node: &'a mut Node<T> = unsafe { back_ptr.as_mut() };
        // Walk backward.  The head sentinel has no value; the filter ensures
        // `back` becomes None when we step past the first data node.
        // `len` independently prevents accessing a stale `back` pointer.
        self.back = node
            .prev()
            .filter(|p| p.value().is_some())
            .map(NonNull::from);
        self.len = self.len.saturating_sub(1);
        node.value_mut()
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {}

impl<T> FusedIterator for IterMut<'_, T> {}

// MARK: IntoIter

/// An owning iterator over the elements of a [`SkipList`].
///
/// This struct is created by the [`IntoIterator`] implementation for
/// [`SkipList`].  Iteration consumes the list.
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
///
/// let mut iter = list.into_iter();
/// assert_eq!(iter.next(), Some(1));
/// assert_eq!(iter.next_back(), Some(3));
/// assert_eq!(iter.next(), Some(2));
/// assert_eq!(iter.next(), None);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IntoIter<T, G: LevelGenerator = Geometric> {
    /// The remaining elements.  `pop_front` / `pop_back` drive iteration;
    /// dropping `IntoIter` drops the remaining elements via the `SkipList`
    /// `Drop` impl.
    list: SkipList<T, G>,
}

// SAFETY: IntoIter<T, G> owns its elements.  Sending it to another thread
// is safe when T is Send (the elements will be accessed on the new thread).
// G: LevelGenerator is Send-safe by the same argument.
unsafe impl<T: Send, G: LevelGenerator + Send> Send for IntoIter<T, G> {}

// SAFETY: Sharing &IntoIter<T, G> is safe when T: Sync and G: Sync.
// Advancing the iterator requires &mut IntoIter, which prevents concurrent
// mutation through shared references.
unsafe impl<T: Sync, G: LevelGenerator + Sync> Sync for IntoIter<T, G> {}

impl<T: fmt::Debug, G: LevelGenerator> fmt::Debug for IntoIter<T, G> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.list.iter()).finish()
    }
}

impl<T, G: LevelGenerator> Iterator for IntoIter<T, G> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.list.pop_front()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.list.len(), Some(self.list.len()))
    }
}

impl<T, G: LevelGenerator> DoubleEndedIterator for IntoIter<T, G> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.list.pop_back()
    }
}

impl<T, G: LevelGenerator> ExactSizeIterator for IntoIter<T, G> {}

impl<T, G: LevelGenerator> FusedIterator for IntoIter<T, G> {}

// MARK: Drain

/// An owning iterator over a sub-range of elements drained from a
/// [`SkipList`].
///
/// This struct is created by the [`SkipList::drain`] method.  Elements in the
/// specified range are removed from the list eagerly when `drain` is called.
/// The removed elements are yielded by this iterator.  Elements outside the
/// range remain in the list regardless of whether the `Drain` is fully
/// consumed.
///
/// Supports both forward and backward iteration
/// ([`DoubleEndedIterator`]).  Does **not** implement
/// [`ExactSizeIterator`].
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_list::SkipList;
///
/// let mut list = SkipList::<i32>::new();
/// for i in 1..=5 {
///     list.push_back(i);
/// }
///
/// let drained: Vec<i32> = list.drain(1..4).collect();
/// assert_eq!(drained, [2, 3, 4]);
/// assert_eq!(list.len(), 2);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Drain<'a, T> {
    /// The already-removed values in front-to-back order.
    iter: std::vec::IntoIter<T>,
    /// Ties the `Drain`'s lifetime to the `&'a mut SkipList` that created it,
    /// preventing the list from being used while this `Drain` is alive.
    _marker: PhantomData<&'a mut T>,
}

// SAFETY: Drain<'a, T> owns its yielded elements.  Sending it to another
// thread requires T: Send because the receiving thread will own T values.
unsafe impl<T: Send> Send for Drain<'_, T> {}

// SAFETY: Sharing &Drain<'a, T> across threads is safe when T: Sync.
// Advancing the iterator requires &mut Drain, preventing concurrent mutation.
unsafe impl<T: Sync> Sync for Drain<'_, T> {}

impl<T: fmt::Debug> fmt::Debug for Drain<'_, T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter.as_slice()).finish()
    }
}

impl<T> Iterator for Drain<'_, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T> DoubleEndedIterator for Drain<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<T> FusedIterator for Drain<'_, T> {}

// MARK: ExtractIf

/// A lazy iterator that removes and yields elements satisfying a predicate.
///
/// This struct is created by the [`SkipList::extract_if`] method.  The
/// predicate is called once per element, in forward order.  Elements for
/// which it returns `true` are removed from the list and yielded; all others
/// remain in place.
///
/// If the iterator is dropped before being fully consumed the predicate is
/// **not** called for the remaining elements — they all stay in the list and
/// the list remains fully usable.
///
/// Does **not** implement [`DoubleEndedIterator`] or [`ExactSizeIterator`].
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_list::SkipList;
///
/// let mut list = SkipList::<i32>::new();
/// for i in 1..=5 {
///     list.push_back(i);
/// }
///
/// let evens: Vec<i32> = list.extract_if(|x| *x % 2 == 0).collect();
/// assert_eq!(evens, [2, 4]);
/// let remaining: Vec<i32> = list.iter().copied().collect();
/// assert_eq!(remaining, [1, 3, 5]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<'a, T, G: LevelGenerator = Geometric, F = fn(&mut T) -> bool>
where
    F: FnMut(&mut T) -> bool,
{
    /// Mutable borrow of the owning list (needed to rebuild skip links on
    /// drop and to update `len` and `tail` on each removal).
    list: &'a mut SkipList<T, G>,
    /// Raw pointer to the next node to visit, or `None` when the iterator
    /// has been exhausted.
    current: Option<NonNull<Node<T>>>,
    /// Set to `true` the first time an element is removed.  Used to skip
    /// the O(n) skip-link rebuild in `Drop::drop` when nothing was removed.
    any_removed: bool,
    /// User-supplied filter predicate.
    pred: F,
}

// SAFETY: ExtractIf<'a, T, G, F> yields owned T values and holds
// &'a mut SkipList<T, G>.  Sending it to another thread requires
// T: Send, G: Send, and F: Send.
unsafe impl<T: Send, G: LevelGenerator + Send, F: Send> Send for ExtractIf<'_, T, G, F> where
    F: FnMut(&mut T) -> bool
{
}

// SAFETY: Sharing &ExtractIf<'a, T, G, F> requires T: Sync, G: Sync, F: Sync.
// Advancing the iterator requires &mut ExtractIf, preventing concurrent mutation.
unsafe impl<T: Sync, G: LevelGenerator + Sync, F: Sync> Sync for ExtractIf<'_, T, G, F> where
    F: FnMut(&mut T) -> bool
{
}

impl<T: fmt::Debug, G: LevelGenerator, F> fmt::Debug for ExtractIf<'_, T, G, F>
where
    F: FnMut(&mut T) -> bool,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display the unvisited elements reachable from `current`.
        // We hold &self so no mutable access is in progress.
        let mut builder = f.debug_list();
        let mut ptr = self.current;
        while let Some(nn) = ptr {
            // SAFETY: nn points to a live Node<T> owned by the SkipList that
            // created this ExtractIf.  We only read through it here, and
            // &self prevents concurrent mutable access.
            let node = unsafe { nn.as_ref() };
            if let Some(v) = node.value() {
                builder.entry(v);
            }
            ptr = node.next().map(NonNull::from);
        }
        builder.finish()
    }
}

impl<T, G: LevelGenerator, F> Iterator for ExtractIf<'_, T, G, F>
where
    F: FnMut(&mut T) -> bool,
{
    type Item = T;

    #[expect(
        clippy::unwrap_in_result,
        clippy::expect_used,
        reason = "`value_mut()` and `take_value()` return None only for the head \
              sentinel, which is never reachable via the data-node walk; the \
              expect fires only on invariant violations"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-pointer dereference, value_mut(), tail-update read, and pop() \
              all touch provably disjoint heap nodes; splitting across blocks \
              would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    fn next(&mut self) -> Option<T> {
        // Walk forward until the predicate matches or the list is exhausted.
        loop {
            let current_nn = self.current?;
            // SAFETY: current_nn was derived from a heap-allocated Node<T>
            // owned by the SkipList that created this ExtractIf.  We hold
            // &'a mut SkipList exclusively for the iterator's lifetime,
            // ensuring every node remains allocated and non-aliased.
            // We capture next_opt before any mutation of the current node.
            unsafe {
                let current: *mut Node<T> = current_nn.as_ptr();
                let next_opt = (*current).next().map(NonNull::from);

                let value_ref = (*current).value_mut().expect("data node has value");
                if (self.pred)(value_ref) {
                    self.current = next_opt;
                    self.any_removed = true;
                    self.list.len = self.list.len.saturating_sub(1);
                    // If this node was the tail, update the tail pointer to
                    // the predecessor data node (or None if the list is now
                    // empty).
                    if self.list.tail == Some(current_nn) {
                        self.list.tail = (*current)
                            .prev()
                            .filter(|p| p.value().is_some())
                            .map(NonNull::from);
                    }
                    let mut boxed = (*current).pop();
                    return Some(boxed.take_value().expect("data node has value"));
                }
                self.current = next_opt;
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // The predicate outcome is unknown, so the lower bound is 0.
        // At most all remaining list elements could be extracted.
        (0, Some(self.list.len))
    }
}

impl<T, G: LevelGenerator, F> FusedIterator for ExtractIf<'_, T, G, F> where F: FnMut(&mut T) -> bool
{}

impl<T, G: LevelGenerator, F> Drop for ExtractIf<'_, T, G, F>
where
    F: FnMut(&mut T) -> bool,
{
    #[expect(
        clippy::expect_used,
        reason = "`Link::new` returns None only when dist == 0, which cannot happen \
              because dist = new_rank - pred_rank and new_rank > pred_rank \
              whenever a predecessor exists"
    )]
    #[expect(
        clippy::indexing_slicing,
        clippy::needless_range_loop,
        reason = "l < node_height ≤ max_levels = predecessors.len() = head.links.len(); \
              l is used for both predecessors[l] and links_mut()[l] so an index \
              loop is the clearest expression"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-pointer traversal, link clearing, and link wiring all touch \
              provably disjoint heap nodes; splitting across blocks would require \
              unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    fn drop(&mut self) {
        if !self.any_removed {
            // Nothing was removed; skip links are still valid.
            return;
        }
        // Rebuild all skip links in one O(n) forward pass over the prev/next
        // chain.  The same algorithm is used by `retain` and `retain_mut`.
        // The prev/next chain is already correct (each `pop()` in `next()`
        // spliced out the removed node), so we only need to re-derive the
        // level-indexed skip links.
        //
        // SAFETY: &'a mut SkipList is held exclusively.  All raw pointers
        // originate from its heap allocations.  We save the successor before
        // clearing links, so traversal is correct throughout.
        unsafe {
            let head_ptr: *mut Node<T> = &raw mut *self.list.head;
            let max_levels = (*head_ptr).level();

            for link in (*head_ptr).links_mut() {
                *link = None;
            }

            let mut predecessors: Vec<(*mut Node<T>, usize)> =
                vec![(head_ptr, 0_usize); max_levels];
            let mut new_rank: usize = 0;
            let mut new_tail: Option<NonNull<Node<T>>> = None;

            let mut cur_opt = (*head_ptr).next().map(NonNull::from);
            while let Some(cur_nn) = cur_opt {
                let cur: *mut Node<T> = cur_nn.as_ptr();
                cur_opt = (*cur).next().map(NonNull::from);

                new_rank = new_rank.saturating_add(1);
                new_tail = Some(cur_nn);

                for link in (*cur).links_mut() {
                    *link = None;
                }
                let height = (*cur).level();
                for l in 0..height {
                    let (pred_ptr, pred_rank) = predecessors[l];
                    let dist = new_rank.saturating_sub(pred_rank);
                    (*pred_ptr).links_mut()[l] =
                        Some(Link::new(&*cur, dist).expect("dist = new_rank - pred_rank ≥ 1"));
                    predecessors[l] = (cur, new_rank);
                }
            }
            self.list.tail = new_tail;
            // self.list.len is already correct: decremented in Iterator::next
            // once per removed element.
        }
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

    // MARK: front / back

    #[test]
    fn front_empty() {
        let list = SkipList::<i32>::new();
        assert_eq!(list.front(), None);
    }

    #[test]
    fn back_empty() {
        let list = SkipList::<i32>::new();
        assert_eq!(list.back(), None);
    }

    #[test]
    fn front_single() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        assert_eq!(list.front(), Some(&42));
    }

    #[test]
    fn back_single() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        assert_eq!(list.back(), Some(&42));
    }

    #[test]
    fn front_and_back_are_same_for_single_element() {
        let mut list = SkipList::<i32>::new();
        list.push_back(7);
        assert_eq!(list.front(), list.back());
    }

    #[test]
    fn front_returns_first_after_push_back() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list.front(), Some(&10));
    }

    #[test]
    fn back_returns_last_after_push_back() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list.back(), Some(&30));
    }

    #[test]
    fn front_returns_first_after_push_front() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        list.push_front(99);
        assert_eq!(list.front(), Some(&99));
        assert_eq!(list.back(), Some(&10));
    }

    #[test]
    fn back_unchanged_after_push_front() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_front(0);
        // front is new element, back is still 2
        assert_eq!(list.front(), Some(&0));
        assert_eq!(list.back(), Some(&2));
    }

    #[test]
    fn front_unchanged_after_push_back() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        // front stays 1 no matter how many push_backs
        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn front_mut_modifies_first_element() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        list.push_back(20);
        *list.front_mut().expect("non-empty") = 99;
        assert_eq!(list.front(), Some(&99));
        assert_eq!(list.back(), Some(&20));
    }

    #[test]
    fn back_mut_modifies_last_element() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        list.push_back(20);
        *list.back_mut().expect("non-empty") = 99;
        assert_eq!(list.front(), Some(&10));
        assert_eq!(list.back(), Some(&99));
    }

    #[test]
    fn front_mut_empty_returns_none() {
        let mut list = SkipList::<i32>::new();
        assert_eq!(list.front_mut(), None);
    }

    #[test]
    fn back_mut_empty_returns_none() {
        let mut list = SkipList::<i32>::new();
        assert_eq!(list.back_mut(), None);
    }

    #[test]
    fn front_none_after_pop_front_empties_list() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.pop_front();
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn back_none_after_pop_back_empties_list() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.pop_back();
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn back_updates_after_pop_back() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.pop_back();
        assert_eq!(list.back(), Some(&2));
        list.pop_back();
        assert_eq!(list.back(), Some(&1));
    }

    #[test]
    fn front_updates_after_pop_front() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.pop_front();
        assert_eq!(list.front(), Some(&2));
        list.pop_front();
        assert_eq!(list.front(), Some(&3));
    }

    #[test]
    fn back_updates_after_insert_at_end() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.insert(2, 99); // append
        assert_eq!(list.back(), Some(&99));
    }

    #[test]
    fn back_unchanged_after_insert_in_middle() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.insert(1, 99); // middle
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn back_updates_after_remove_last() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.remove(2); // remove last
        assert_eq!(list.back(), Some(&2));
    }

    #[test]
    fn back_unchanged_after_remove_middle() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.remove(1); // remove middle
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn back_consistent_with_get_last() {
        let mut list = SkipList::<i32>::with_capacity(4);
        for i in 0..20_i32 {
            list.push_back(i);
        }
        assert_eq!(list.back(), list.get(list.len() - 1));
    }

    // MARK: clear

    #[test]
    fn clear_empty_list() {
        let mut list = SkipList::<i32>::new();
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn clear_single_element() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn clear_multiple_elements() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=10 {
            list.push_back(i);
        }
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
        assert!(list.head.next().is_none());
    }

    #[test]
    fn clear_usable_after_clear() {
        // After clear, the list can accept new insertions.
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        list.clear();
        list.push_back(99);
        list.push_front(0);
        assert_eq!(list.len(), 2);
        assert_eq!(list.front(), Some(&0));
        assert_eq!(list.back(), Some(&99));
    }

    #[test]
    fn clear_then_clear_again() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.clear();
        list.clear(); // second clear on already-empty list
        assert!(list.is_empty());
    }

    #[test]
    fn clear_large_list() {
        // Large list to exercise the iterative Drop path.
        let n = 1_000_usize;
        let mut list = SkipList::<usize>::new();
        for i in 0..n {
            list.push_back(i);
        }
        list.clear();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
    }

    // MARK: truncate

    #[test]
    fn truncate_noop_when_len_equals_current() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.truncate(3);
        assert_eq!(list.len(), 3);
        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn truncate_noop_when_len_greater() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.truncate(5);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn truncate_to_zero_clears_list() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.truncate(0);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn truncate_to_one() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.truncate(1);
        assert_eq!(list.len(), 1);
        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.back(), Some(&1));
        assert!(list.head.next().and_then(|n| n.next()).is_none());
    }

    #[test]
    fn truncate_keeps_correct_elements() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=5 {
            list.push_back(i); // [1, 2, 3, 4, 5]
        }
        list.truncate(3); // [1, 2, 3]
        assert_eq!(list.len(), 3);
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&3));
        assert_eq!(list.get(3), None);
    }

    #[test]
    fn truncate_back_pointer_updated() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        list.truncate(3);
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn truncate_front_unchanged() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        list.truncate(3);
        assert_eq!(list.front(), Some(&1));
    }

    #[test]
    fn truncate_empty_list() {
        let mut list = SkipList::<i32>::new();
        list.truncate(0);
        assert!(list.is_empty());
        list.truncate(5);
        assert!(list.is_empty());
    }

    #[test]
    fn truncate_usable_after_truncate() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=5 {
            list.push_back(i);
        }
        list.truncate(2); // [1, 2]
        list.push_back(99); // [1, 2, 99]
        assert_eq!(list.len(), 3);
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&99));
        assert_eq!(list.back(), Some(&99));
    }

    #[test]
    fn truncate_then_truncate_more() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=10 {
            list.push_back(i);
        }
        list.truncate(7); // [1..=7]
        list.truncate(4); // [1..=4]
        assert_eq!(list.len(), 4);
        assert_eq!(list.back(), Some(&4));
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(2), Some(&3));
        assert_eq!(list.get(3), Some(&4));
    }

    #[test]
    fn truncate_large_list() {
        const N: usize = 1_000;
        const HALF: usize = 500;
        let mut list = SkipList::<usize>::new();
        for i in 0..N {
            list.push_back(i);
        }
        list.truncate(HALF); // keep first 500
        assert_eq!(list.len(), HALF);
        for i in 0..HALF {
            assert_eq!(list.get(i), Some(&i));
        }
        assert_eq!(list.back(), Some(&(HALF - 1)));
    }

    // MARK: iter

    #[test]
    fn iter_empty() {
        let list = SkipList::<i32>::new();
        let mut iter = list.iter();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_single_element() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(&42));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_single_element_from_back() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        let mut iter = list.iter();
        assert_eq!(iter.next_back(), Some(&42));
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_forward_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, [1, 2, 3]);
    }

    #[test]
    fn iter_backward_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let collected: Vec<i32> = list.iter().copied().rev().collect();
        assert_eq!(collected, [3, 2, 1]);
    }

    #[test]
    fn iter_double_ended_alternating() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=5_i32 {
            list.push_back(i);
        }
        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), Some(&5));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next_back(), Some(&4));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_double_ended_meets_in_middle_odd() {
        // 3 elements: consume 1 from front, 1 from back → 1 left in middle
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(&10));
        assert_eq!(iter.next_back(), Some(&30));
        // Only the middle element remains
        assert_eq!(iter.next(), Some(&20));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_double_ended_meets_in_middle_even() {
        // 4 elements: alternate until both are exhausted
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        list.push_back(40);
        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(&10));
        assert_eq!(iter.next_back(), Some(&40));
        assert_eq!(iter.next(), Some(&20));
        assert_eq!(iter.next_back(), Some(&30));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_size_hint_decrements() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let mut iter = list.iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        iter.next_back();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        iter.next();
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn iter_exact_size() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10_i32 {
            list.push_back(i);
        }
        let mut iter = list.iter();
        assert_eq!(iter.len(), 10);
        iter.next();
        assert_eq!(iter.len(), 9);
        iter.next_back();
        assert_eq!(iter.len(), 8);
    }

    #[test]
    fn iter_fused_returns_none_repeatedly() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(&1));
        // Exhausted — subsequent calls must all return None (FusedIterator)
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_clone_yields_same_elements() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let iter = list.iter();
        let clone = iter.clone();
        let v1: Vec<i32> = iter.copied().collect();
        let v2: Vec<i32> = clone.copied().collect();
        assert_eq!(v1, v2);
        assert_eq!(v1, [1, 2, 3]);
    }

    #[test]
    fn iter_does_not_consume_list() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        // Iterating multiple times yields the same elements each time.
        let v1: Vec<i32> = list.iter().copied().collect();
        let v2: Vec<i32> = list.iter().copied().collect();
        assert_eq!(v1, v2);
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn into_iter_for_ref() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let collected: Vec<i32> = (&list).into_iter().copied().collect();
        assert_eq!(collected, [1, 2, 3]);
    }

    #[test]
    fn iter_after_push_front() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(3);
        list.push_front(2);
        list.push_front(1);
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, [1, 2, 3]);
        let reversed: Vec<i32> = list.iter().copied().rev().collect();
        assert_eq!(reversed, [3, 2, 1]);
    }

    #[test]
    fn iter_after_mutations() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=5_i32 {
            list.push_back(i); // [1, 2, 3, 4, 5]
        }
        list.remove(2); // [1, 2, 4, 5]
        list.insert(2, 9); // [1, 2, 9, 4, 5]
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, [1, 2, 9, 4, 5]);
    }

    #[test]
    fn iter_large_list_forward() {
        const N: usize = 200;
        let mut list = SkipList::<usize>::new();
        for i in 0..N {
            list.push_back(i);
        }
        for (i, v) in list.iter().enumerate() {
            assert_eq!(*v, i);
        }
    }

    #[test]
    fn iter_large_list_backward() {
        const N: usize = 200;
        let mut list = SkipList::<usize>::new();
        for i in 0..N {
            list.push_back(i);
        }
        for (i, v) in list.iter().rev().enumerate() {
            assert_eq!(*v, N - 1 - i);
        }
    }

    // MARK: iter_mut

    #[test]
    fn iter_mut_empty() {
        let mut list = SkipList::<i32>::new();
        let mut iter = list.iter_mut();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_mut_single_element() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        let mut iter = list.iter_mut();
        assert_eq!(iter.next(), Some(&mut 42));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_mut_single_element_from_back() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        let mut iter = list.iter_mut();
        assert_eq!(iter.next_back(), Some(&mut 42));
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    #[expect(
        clippy::explicit_iter_loop,
        reason = "explicitly exercising iter_mut(); the &mut shorthand tests a different code path"
    )]
    fn iter_mut_modifies_elements() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        for v in list.iter_mut() {
            *v *= 10;
        }
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, [10, 20, 30]);
    }

    #[test]
    fn iter_mut_forward_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let collected: Vec<i32> = list.iter_mut().map(|v| *v).collect();
        assert_eq!(collected, [1, 2, 3]);
    }

    #[test]
    fn iter_mut_backward_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let collected: Vec<i32> = list.iter_mut().map(|v| *v).rev().collect();
        assert_eq!(collected, [3, 2, 1]);
    }

    #[test]
    fn iter_mut_double_ended_alternating() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=5_i32 {
            list.push_back(i);
        }
        let mut iter = list.iter_mut();
        assert_eq!(iter.next(), Some(&mut 1));
        assert_eq!(iter.next_back(), Some(&mut 5));
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next_back(), Some(&mut 4));
        assert_eq!(iter.next(), Some(&mut 3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_mut_double_ended_meets_in_middle_odd() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        let mut iter = list.iter_mut();
        assert_eq!(iter.next(), Some(&mut 10));
        assert_eq!(iter.next_back(), Some(&mut 30));
        assert_eq!(iter.next(), Some(&mut 20));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_mut_double_ended_meets_in_middle_even() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        list.push_back(40);
        let mut iter = list.iter_mut();
        assert_eq!(iter.next(), Some(&mut 10));
        assert_eq!(iter.next_back(), Some(&mut 40));
        assert_eq!(iter.next(), Some(&mut 20));
        assert_eq!(iter.next_back(), Some(&mut 30));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_mut_size_hint_decrements() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let mut iter = list.iter_mut();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        iter.next_back();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        iter.next();
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn iter_mut_exact_size() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10_i32 {
            list.push_back(i);
        }
        let mut iter = list.iter_mut();
        assert_eq!(iter.len(), 10);
        iter.next();
        assert_eq!(iter.len(), 9);
        iter.next_back();
        assert_eq!(iter.len(), 8);
    }

    #[test]
    fn iter_mut_fused_returns_none_repeatedly() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        let mut iter = list.iter_mut();
        assert_eq!(iter.next(), Some(&mut 1));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_mut_does_not_consume_list_after_drop() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        {
            let _iter = list.iter_mut(); // create and immediately drop
        }
        assert_eq!(list.len(), 3);
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, [1, 2, 3]);
    }

    #[test]
    fn into_iter_for_mut_ref() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let collected: Vec<i32> = (&mut list).into_iter().map(|v| *v).collect();
        assert_eq!(collected, [1, 2, 3]);
    }

    // MARK: into_iter (consuming)

    #[test]
    fn into_iter_empty() {
        let list = SkipList::<i32>::new();
        let mut iter = list.into_iter();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn into_iter_single_element() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        let mut iter = list.into_iter();
        assert_eq!(iter.next(), Some(42));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn into_iter_single_element_from_back() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        let mut iter = list.into_iter();
        assert_eq!(iter.next_back(), Some(42));
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn into_iter_forward_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let collected: Vec<i32> = list.into_iter().collect();
        assert_eq!(collected, [1, 2, 3]);
    }

    #[test]
    fn into_iter_backward_order() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let collected: Vec<i32> = list.into_iter().rev().collect();
        assert_eq!(collected, [3, 2, 1]);
    }

    #[test]
    fn into_iter_double_ended_alternating() {
        let mut list = SkipList::<i32>::with_capacity(1);
        for i in 1..=5_i32 {
            list.push_back(i);
        }
        let mut iter = list.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next_back(), Some(5));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next_back(), Some(4));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn into_iter_double_ended_meets_in_middle_even() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        list.push_back(40);
        let mut iter = list.into_iter();
        assert_eq!(iter.next(), Some(10));
        assert_eq!(iter.next_back(), Some(40));
        assert_eq!(iter.next(), Some(20));
        assert_eq!(iter.next_back(), Some(30));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn into_iter_size_hint_decrements() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let mut iter = list.into_iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        iter.next_back();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        iter.next();
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn into_iter_exact_size() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10_i32 {
            list.push_back(i);
        }
        let mut iter = list.into_iter();
        assert_eq!(iter.len(), 10);
        iter.next();
        assert_eq!(iter.len(), 9);
        iter.next_back();
        assert_eq!(iter.len(), 8);
    }

    #[test]
    fn into_iter_fused_returns_none_repeatedly() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        let mut iter = list.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn into_iter_drops_remaining_on_drop() {
        // Use a large list to exercise normal drop behaviour.
        let mut list = SkipList::<i32>::new();
        for i in 0..100_i32 {
            list.push_back(i);
        }
        let mut iter = list.into_iter();
        // Partially consume
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next_back(), Some(99));
        // Drop the iterator; remaining elements must not leak.
        drop(iter);
    }

    #[test]
    fn into_iter_consuming_via_for_loop() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        let mut sum = 0;
        for v in list {
            sum += v;
        }
        assert_eq!(sum, 60);
    }

    // MARK: retain

    #[test]
    fn retain_empty() {
        let mut list = SkipList::<i32>::new();
        list.retain(|_| true);
        assert!(list.is_empty());
        list.retain(|_| false);
        assert!(list.is_empty());
    }

    #[test]
    fn retain_all_kept() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        list.retain(|_| true);
        assert_eq!(list.len(), 5);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn retain_all_dropped() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        list.retain(|_| false);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn retain_even_elements() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=6 {
            list.push_back(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping even numbers"
        )]
        list.retain(|&x| x % 2 == 0);
        assert_eq!(list.len(), 3);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [2, 4, 6]);
    }

    #[test]
    fn retain_preserves_order() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10 {
            list.push_back(i);
        }
        // Keep 0, 3, 6, 9
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping multiples of 3"
        )]
        list.retain(|&x| x % 3 == 0);
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [0, 3, 6, 9]);
    }

    #[test]
    fn retain_links_consistent() {
        let mut list = SkipList::<i32>::new();
        for i in 0..20 {
            list.push_back(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping even numbers"
        )]
        list.retain(|&x| x % 2 == 0);
        for (idx, expected) in (0..20).step_by(2).enumerate() {
            assert_eq!(list.get(idx), Some(&expected));
        }
        assert_eq!(list.get(list.len()), None);
    }

    #[test]
    fn retain_single_element_kept() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        list.retain(|_| true);
        assert_eq!(list.len(), 1);
        assert_eq!(list.front(), Some(&42));
        assert_eq!(list.back(), Some(&42));
        assert_eq!(list.get(0), Some(&42));
    }

    #[test]
    fn retain_single_element_dropped() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        list.retain(|_| false);
        assert!(list.is_empty());
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn retain_tail_pointer_correct() {
        // Verify that back() returns the last retained element.
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        // Keep 1, 2, 3; drop 4 and 5
        list.retain(|&x| x <= 3);
        assert_eq!(list.back(), Some(&3));
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn retain_after_retain_is_correct() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10 {
            list.push_back(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping even numbers"
        )]
        list.retain(|&x| x % 2 == 0); // 0,2,4,6,8
        list.retain(|&x| x > 2); // 4,6,8
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [4, 6, 8]);
    }

    // MARK: retain_mut

    #[test]
    fn retain_mut_can_modify_before_keeping() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of modifying even numbers in place and keeping them"
        )]
        list.retain_mut(|x| {
            if *x % 2 == 0 {
                *x *= 10;
                true
            } else {
                false
            }
        });
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [20, 40]);
    }

    #[test]
    fn retain_mut_drops_correctly() {
        let mut list = SkipList::<i32>::new();
        for i in 0..5 {
            list.push_back(i);
        }
        list.retain_mut(|_| false);
        assert!(list.is_empty());
    }

    #[test]
    fn retain_mut_links_consistent() {
        let mut list = SkipList::<i32>::new();
        for i in 0..20 {
            list.push_back(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of keeping multiples of 3 \
                and modifying them in place"
        )]
        list.retain_mut(|x| {
            if *x % 3 == 0 {
                *x += 100;
                true
            } else {
                false
            }
        });
        // Retained: 0+100=100, 3+100=103, 6+100=106, 9+100=109, 12+100=112, 15+100=115, 18+100=118
        let got: Vec<i32> = list.iter().copied().collect();
        assert_eq!(got, [100, 103, 106, 109, 112, 115, 118]);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(list.get(i), Some(v));
        }
    }

    // MARK: drain

    #[test]
    fn drain_full_range() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let got: Vec<i32> = list.drain(..).collect();
        assert_eq!(got, [1, 2, 3, 4, 5]);
        assert!(list.is_empty());
    }

    #[test]
    fn drain_empty_range() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let got: Vec<i32> = list.drain(2..2).collect();
        assert!(got.is_empty());
        assert_eq!(list.len(), 5);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn drain_front() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let got: Vec<i32> = list.drain(..2).collect();
        assert_eq!(got, [1, 2]);
        assert_eq!(list.len(), 3);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [3, 4, 5]);
    }

    #[test]
    fn drain_back() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let got: Vec<i32> = list.drain(3..).collect();
        assert_eq!(got, [4, 5]);
        assert_eq!(list.len(), 3);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [1, 2, 3]);
    }

    #[test]
    fn drain_middle() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let got: Vec<i32> = list.drain(1..4).collect();
        assert_eq!(got, [2, 3, 4]);
        assert_eq!(list.len(), 2);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [1, 5]);
    }

    #[test]
    fn drain_inclusive_range() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let got: Vec<i32> = list.drain(1..=3).collect();
        assert_eq!(got, [2, 3, 4]);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn drain_double_ended() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let mut drain = list.drain(..);
        assert_eq!(drain.next(), Some(1));
        assert_eq!(drain.next_back(), Some(5));
        assert_eq!(drain.next(), Some(2));
        assert_eq!(drain.next_back(), Some(4));
        assert_eq!(drain.next(), Some(3));
        assert_eq!(drain.next(), None);
        assert_eq!(drain.next_back(), None);
    }

    #[test]
    fn drain_drop_remaining() {
        // Drop the Drain without consuming all elements; list must still be valid.
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        {
            let mut drain = list.drain(1..4);
            // Consume just one element and drop the rest.
            assert_eq!(drain.next(), Some(2));
            // `drain` is dropped here; 3 and 4 are freed.
        }
        assert_eq!(list.len(), 2);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [1, 5]);
    }

    #[test]
    fn drain_len_correct_after() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10 {
            list.push_back(i);
        }
        _ = list.drain(3..7);
        assert_eq!(list.len(), 6);
    }

    #[test]
    fn drain_links_consistent_after() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10 {
            list.push_back(i);
        }
        _ = list.drain(3..7);
        // Remaining: 0,1,2,7,8,9
        let expected = [0, 1, 2, 7, 8, 9];
        for (idx, &v) in expected.iter().enumerate() {
            assert_eq!(list.get(idx), Some(&v));
        }
        assert_eq!(list.get(list.len()), None);
    }

    #[test]
    fn drain_size_hint() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let mut drain = list.drain(1..4);
        assert_eq!(drain.size_hint(), (3, Some(3)));
        drain.next();
        assert_eq!(drain.size_hint(), (2, Some(2)));
    }

    #[test]
    fn drain_fused() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        let mut drain = list.drain(..);
        assert_eq!(drain.next(), Some(1));
        assert_eq!(drain.next(), None);
        assert_eq!(drain.next(), None);
        assert_eq!(drain.next_back(), None);
    }

    #[test]
    #[expect(
        clippy::reversed_empty_ranges,
        reason = "Intentional test of invalid range handling in drain()"
    )]
    #[should_panic(expected = "drain range start")]
    fn drain_panics_start_gt_end() {
        let mut list = SkipList::<i32>::new();
        for i in 0..5 {
            list.push_back(i);
        }
        _ = list.drain(3..1);
    }

    #[test]
    #[should_panic(expected = "drain range end")]
    fn drain_panics_end_gt_len() {
        let mut list = SkipList::<i32>::new();
        for i in 0..5 {
            list.push_back(i);
        }
        _ = list.drain(0..10);
    }

    // MARK: extract_if

    #[test]
    fn extract_if_empty() {
        let mut list = SkipList::<i32>::new();
        let extracted: Vec<i32> = list.extract_if(|_| true).collect();
        assert!(extracted.is_empty());
        assert!(list.is_empty());
    }

    #[test]
    fn extract_if_none_match() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let extracted: Vec<i32> = list.extract_if(|_| false).collect();
        assert!(extracted.is_empty());
        assert_eq!(list.len(), 5);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn extract_if_all_match() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let extracted: Vec<i32> = list.extract_if(|_| true).collect();
        assert_eq!(extracted, [1, 2, 3, 4, 5]);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn extract_if_evens() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of extracting even numbers"
        )]
        let extracted: Vec<i32> = list.extract_if(|x| *x % 2 == 0).collect();
        assert_eq!(extracted, [2, 4]);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [1, 3, 5]);
    }

    #[test]
    fn extract_if_preserves_order() {
        let mut list = SkipList::<i32>::new();
        for i in [5, 1, 4, 2, 3] {
            list.push_back(i);
        }
        // Extract values > 3; they appear in insertion order.
        let extracted: Vec<i32> = list.extract_if(|x| *x > 3).collect();
        assert_eq!(extracted, [5, 4]);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [1, 2, 3]);
    }

    #[test]
    fn extract_if_remaining_in_list() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=6 {
            list.push_back(i);
        }
        // Extract odd numbers.
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of extracting odd numbers"
        )]
        let extracted: Vec<i32> = list.extract_if(|x| *x % 2 != 0).collect();
        assert_eq!(extracted, [1, 3, 5]);
        assert_eq!(list.len(), 3);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [2, 4, 6]);
    }

    #[test]
    #[expect(
        clippy::integer_division_remainder_used,
        reason = "clearer to express the intent of extracting multiples of 3"
    )]
    fn extract_if_links_consistent() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10 {
            list.push_back(i);
        }
        // Extract elements divisible by 3: 0, 3, 6, 9.
        _ = list.extract_if(|x| *x % 3 == 0).count();
        // Remaining: 1, 2, 4, 5, 7, 8
        let expected = [1, 2, 4, 5, 7, 8];
        assert_eq!(list.len(), expected.len());
        for (idx, &v) in expected.iter().enumerate() {
            assert_eq!(list.get(idx), Some(&v));
        }
        assert_eq!(list.get(list.len()), None);
    }

    #[test]
    fn extract_if_drop_early() {
        // Drop the ExtractIf mid-iteration; unvisited elements stay in the list.
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        {
            #[expect(
                clippy::integer_division_remainder_used,
                reason = "clearer to express the intent of extracting even numbers"
            )]
            let mut it = list.extract_if(|x| *x % 2 == 0);
            // Advance once: visits 1 (kept), then 2 (extracted).
            assert_eq!(it.next(), Some(2));
            // Drop here; 3, 4, 5 are not visited and stay in the list.
        }
        // 1 was kept, 2 was extracted, 3/4/5 were never visited.
        assert_eq!(list.len(), 4);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [1, 3, 4, 5]);
    }

    #[test]
    fn extract_if_tail_updated() {
        // Verify that back() returns the correct node after the tail is removed.
        let mut list = SkipList::<i32>::new();
        for i in 1..=4 {
            list.push_back(i);
        }
        // Extract elements >= 3 (i.e. 3 and 4, including the tail).
        let extracted: Vec<i32> = list.extract_if(|x| *x >= 3).collect();
        assert_eq!(extracted, [3, 4]);
        assert_eq!(list.back(), Some(&2));
    }

    #[test]
    fn extract_if_len_correct() {
        // Verify that `list.len` is decremented on each extraction, observable
        // via `size_hint` (whose upper bound reflects the current list length).
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of extracting even numbers"
        )]
        let mut it = list.extract_if(|x| *x % 2 == 0);
        // Before any extraction: upper bound = 5.
        assert_eq!(it.size_hint(), (0, Some(5)));
        // Extract 2 (visits 1 kept, 2 extracted).
        assert_eq!(it.next(), Some(2));
        // list.len is now 4.
        assert_eq!(it.size_hint(), (0, Some(4)));
        // Extract 4 (visits 3 kept, 4 extracted).
        assert_eq!(it.next(), Some(4));
        // list.len is now 3.
        assert_eq!(it.size_hint(), (0, Some(3)));
        drop(it);
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn extract_if_fused() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        let mut it = list.extract_if(|_| true);
        assert_eq!(it.next(), Some(1));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
    }

    #[test]
    fn extract_if_mut_predicate() {
        // Predicate receives &mut T — verify it can mutate values before keeping.
        let mut list = SkipList::<i32>::new();
        for i in 1..=4 {
            list.push_back(i);
        }
        // Double even elements, extract only the odd ones.
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of modifying even numbers in place and extracting odd numbers"
        )]
        let extracted: Vec<i32> = list
            .extract_if(|x| {
                if *x % 2 == 0 {
                    *x *= 10;
                    false
                } else {
                    true
                }
            })
            .collect();
        assert_eq!(extracted, [1, 3]);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [20, 40]);
    }

    // MARK: range

    #[test]
    fn range_empty_list() {
        let list = SkipList::<i32>::new();
        let v: Vec<i32> = list.range(0..0).copied().collect();
        assert!(v.is_empty());
    }

    #[test]
    fn range_empty_range() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let v: Vec<i32> = list.range(2..2).copied().collect();
        assert!(v.is_empty());
    }

    #[test]
    fn range_full() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let from_range: Vec<i32> = list.range(..).copied().collect();
        let from_iter: Vec<i32> = list.iter().copied().collect();
        assert_eq!(from_range, from_iter);
    }

    #[test]
    fn range_half_open() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let v: Vec<i32> = list.range(1..4).copied().collect();
        assert_eq!(v, [2, 3, 4]);
    }

    #[test]
    fn range_inclusive() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let v: Vec<i32> = list.range(1..=3).copied().collect();
        assert_eq!(v, [2, 3, 4]);
    }

    #[test]
    fn range_single() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let v: Vec<i32> = list.range(2..=2).copied().collect();
        assert_eq!(v, [3]);
    }

    #[test]
    fn range_double_ended() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let mut it = list.range(1..4);
        assert_eq!(it.next(), Some(&2));
        assert_eq!(it.next_back(), Some(&4));
        assert_eq!(it.next(), Some(&3));
        assert_eq!(it.next(), None);
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn range_rev() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let v: Vec<i32> = list.range(1..4).copied().rev().collect();
        assert_eq!(v, [4, 3, 2]);
    }

    #[test]
    fn range_exact_size() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let it = list.range(1..4);
        assert_eq!(it.len(), 3);
    }

    #[test]
    fn range_mut_modify() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        for v in list.range_mut(1..4) {
            *v *= 10;
        }
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, [1, 20, 30, 40, 5]);
    }

    #[test]
    fn range_mut_double_ended() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        let v: Vec<i32> = list.range_mut(1..4).map(|x| *x).rev().collect();
        assert_eq!(v, [4, 3, 2]);
    }

    #[test]
    #[expect(
        clippy::reversed_empty_ranges,
        reason = "Intentional test of invalid range handling in range()"
    )]
    #[should_panic(expected = "range start")]
    fn range_panic_start_gt_end() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        _ = list.range(3..1);
    }

    #[test]
    #[should_panic(expected = "range end")]
    fn range_panic_end_gt_len() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        _ = list.range(0..10);
    }

    // MARK: split_off

    #[test]
    fn split_off_empty_list() {
        let mut a = SkipList::<i32>::new();
        let b = a.split_off(0);
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn split_off_at_end_returns_empty() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=5 {
            a.push_back(i);
        }
        let b = a.split_off(5);
        assert_eq!(a.len(), 5);
        assert!(b.is_empty());
        let a_vals: Vec<i32> = a.iter().copied().collect();
        assert_eq!(a_vals, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn split_off_at_zero_transfers_all() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=5 {
            a.push_back(i);
        }
        let b = a.split_off(0);
        assert!(a.is_empty());
        assert_eq!(b.len(), 5);
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(b_vals, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn split_off_middle() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=5 {
            a.push_back(i);
        }
        let b = a.split_off(3);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [1, 2, 3]);
        assert_eq!(b_vals, [4, 5]);
    }

    #[test]
    fn split_off_len_correct() {
        let mut a = SkipList::<i32>::new();
        for i in 0..10 {
            a.push_back(i);
        }
        let b = a.split_off(4);
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 6);
    }

    #[test]
    fn split_off_front_back_correct() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=6 {
            a.push_back(i);
        }
        let b = a.split_off(3);
        assert_eq!(a.front(), Some(&1));
        assert_eq!(a.back(), Some(&3));
        assert_eq!(b.front(), Some(&4));
        assert_eq!(b.back(), Some(&6));
    }

    #[test]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "Numbers are small in test, and therefore not affected"
    )]
    fn split_off_get_works_after() {
        let mut a = SkipList::<i32>::new();
        for i in 0..10 {
            a.push_back(i);
        }
        let b = a.split_off(5);
        for i in 0..5 {
            assert_eq!(a.get(i), Some(&(i as i32)));
        }
        for i in 0..5 {
            assert_eq!(b.get(i), Some(&((i + 5) as i32)));
        }
    }

    #[test]
    fn split_off_single_element_at_zero() {
        let mut a = SkipList::<i32>::new();
        a.push_back(42);
        let b = a.split_off(0);
        assert!(a.is_empty());
        assert_eq!(b.len(), 1);
        assert_eq!(b.get(0), Some(&42));
    }

    #[test]
    fn split_off_single_element_at_one() {
        let mut a = SkipList::<i32>::new();
        a.push_back(42);
        let b = a.split_off(1);
        assert_eq!(a.len(), 1);
        assert_eq!(a.get(0), Some(&42));
        assert!(b.is_empty());
    }

    #[test]
    fn split_off_then_push() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=4 {
            a.push_back(i);
        }
        let mut b = a.split_off(2);
        a.push_back(99);
        b.push_back(100);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [1, 2, 99]);
        assert_eq!(b_vals, [3, 4, 100]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn split_off_out_of_bounds_panics() {
        let mut a = SkipList::<i32>::new();
        for i in 1..=3 {
            a.push_back(i);
        }
        _ = a.split_off(4);
    }

    #[test]
    fn split_off_large_list() {
        let n: usize = 200;
        let mut a = SkipList::<usize>::new();
        for i in 0..n {
            a.push_back(i);
        }
        #[expect(
            clippy::integer_division,
            clippy::integer_division_remainder_used,
            reason = "clearer to express the intent of splitting at one-third of the list length"
        )]
        let at = n / 3;
        let b = a.split_off(at);
        assert_eq!(a.len(), at);
        assert_eq!(b.len(), n - at);
        // Verify every element via get() to exercise skip links.
        for i in 0..at {
            assert_eq!(a.get(i), Some(&i));
        }
        for i in 0..(n - at) {
            assert_eq!(b.get(i), Some(&(i + at)));
        }
    }
}
