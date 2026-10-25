//! Node implementation for the skip list.
//!
//! This module documents the internal design of skip list nodes.  A
//! user-facing version of this content, covering ownership, pointer
//! provenance, and the `NonNull`-over-`Box` rationale, is available at
//! [`crate::docs::internals`].
//!
//! Underlying all the operations of the skip list, skip map and ordered skip
//! list is the node. Each node owns the next node, and has a link to the
//! previous node. The node also has a level, which corresponds to how 'high'
//! the node reaches (or equivalently, how many links it has above the immediate
//! neighbour links).
//!
//! As a concrete example, consider the following list:
//!
//! ```text
//! [4] head
//! [3] head --------------------------> 6
//! [2] head ---------------------> 5 -> 6
//! [1] head ------> 2 -----------> 5 -> 6 ------> 8
//! [*] head -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10
//! ```
//!
//! Traversal of the list at the 0th level is straightforward using the node's
//! previous and next pointers (the latter being the one that owns the next
//! node).
//!
//! The skip list's benefit comes from the ability to skip over nodes at higher
//! levels. For example, to move from node 1 to node 10, the above list would be
//! traversed in the order head -> 6 -> 8 -> 9 -> 10.
//!
//! # Considerations
//!
//! There are a number of considerations to take into account when working with
//! the [`Node`] directly. These concerns are managed by the higher-level skip
//! list implementations.
//!
//! ## Ownership
//!
//! The nodes are placed on the heap using [`Box::new`]. The [`Box`] is then
//! leaked into a [`NonNull`] pointer, which is used to create the links between
//! nodes.
//!
//! The [`NonNull`] pointer allows access to the node, but Rust does not keep
//! track of the ownership of the node. As a result, we must do so manually.
//! Specifically:
//!
//! 1. A skip list or skip map owns the head node.
//! 2. Each node owns the immediately following node through the `next` pointer.
//!
//! As a result of (2), getting a mutable reference to the next node through
//! [`next_as_mut`][Node::next_as_mut] requires that no other mutable reference
//! to that node exists, which is why the method is marked `unsafe`.
//!
//! ## Pointers
//!
//! When managing pointers, it is also important to ensure that they are not
//! invalidated. There are a few ways this can happen:
//!
//! 1. If the node is dropped without updating the appropriate links, the
//!    pointers will be left dangling.
//! 2. If the node is moved. This can happen when the node is moved to the heap
//!    through a [`Box`], or when the node is moved from one function's stack to
//!    another.
//!    - This last point is particularly critical. It is _not_ safe to have a
//!      function return a [`Node`] unless that node is detached from the list.
//!
//! ## Linking
//!
//! In addition to the immediate `prev` and `next` pointers, each node has a
//! list of links. These links are used to skip over nodes at higher levels.
//!
//! The links never own the nodes they point to.
//!
//! Some important considerations/observations (especially when considering
//! safety):
//!
//! - A node's 'level' is the number of links it has (excluding the
//!   `prev`/`next` links). In the above example, node (1) has a level of 0,
//!   node (2) a level of 1 and node (5) a level of 2.
//!   **A node at level 0 has no skip links and participates only in the
//!   sequential base layer** (reachable via `prev`/`next` only).  This is
//!   the most common case for a well-tuned generator.
//! - The head node always has the maximum number of levels. In the example
//!   above, the head node has a level of 4, even if it doesn't contain a fourth
//!   link. That fourth link will be created when a new node of level 4 is
//!   inserted.
//! - If a node is linked to at level `n` (as numbered above), then the
//!   following are true:
//!   - The node has at least `n` links.
//!   - The node is reachable for all levels `0..=n`.
//!
//! There are methods to pop the current node, or to insert a new node before or
//! after the current node. These methods _only_ work on regular nodes and do
//! _not_ alter the links of the node, or surrounding nodes. As a result  while
//! traversing the list to find a specific node, it is important to keep track
//! of the links to the node and links over the node.
//!
//! The implementation of the node has some similarities to the way the standard
//! library's [`LinkedList`][std::collections::LinkedList] is implemented, with
//! the node being similar to
//! [`linked_list::Node`](https://doc.rust-lang.org/stable/src/alloc/collections/linked_list.rs.html).
//!
//! ## Pointer and Provenance Safety
//!
//! `Node` exposes three accessor families for `next` and `prev`:
//!
//! | Method | Returns | Provenance | When to use |
//! |--------|---------|------------|-------------|
//! | [`next`][Node::next] / [`prev`][Node::prev] | `Option<NonNull<Self>>` | Original (Reserved/Active) | **Default choice.** Any time the pointer is stored, passed to a function, used across loop iterations, or may later be written through. |
//! | [`next_as_ref`][Node::next_as_ref] / [`prev_as_ref`][Node::prev_as_ref] | `Option<&Self>` | Frozen (shared reborrow) | **Inline, read-only access only.** Safe when the result is consumed within the same expression and never converted back to a raw pointer. Example: `node.next_as_ref()?.value()`. |
//! | [`next_as_mut`][Node::next_as_mut] / [`prev_as_mut`][Node::prev_as_mut] | `Option<&mut Self>` | Exclusive reborrow | Short-lived inline mutation; see individual method safety docs. |
//!
//! ### The Frozen anti-pattern
//!
//! **Never** write `node.next_as_ref().map(NonNull::from)` (or the `prev`
//! equivalent).  Calling `next_as_ref()` performs a *shared* reborrow of the
//! neighbouring node, which Tree Borrows records with a **Frozen** provenance
//! tag.  Converting the resulting `&Node` back to `NonNull` via `NonNull::from`
//! silently carries that Frozen tag into the raw pointer.  Any subsequent write
//! through that pointer, or through any child pointer derived from it, is
//! undefined behaviour under Tree Borrows.
//!
//! If you need a raw pointer to the next/previous node, use `next()` / `prev()`
//! directly: they return the stored `NonNull` without creating any reborrow.

#![expect(dead_code, reason = "library is still being implemented")]

use alloc::boxed::Box;
#[cfg(any(debug_assertions, test))]
use alloc::collections::BTreeMap;
#[cfg(any(debug_assertions, test))]
use alloc::{format, string::String};
#[cfg(any(debug_assertions, test))]
use core::fmt::Write;
use core::{
    cmp::Ordering,
    fmt::{self, Debug},
    ptr::NonNull,
};

use crate::{
    comparator::total_order_violation,
    node::{
        level_array::LevelArray,
        link::Link,
        visitor::{IndexMutVisitor, Visitor, VisitorMut},
    },
};

pub(crate) mod cursor_raw;
pub(crate) mod level_array;
pub(crate) mod link;
pub(crate) mod visitor;

/// The type of node in the skip list.
#[derive(Debug)]
pub(crate) enum NodeType {
    /// A head node.
    ///
    /// This is identified by having no `prev` pointer and no value. In most
    /// circumstances, this node will have a `next` pointer, but this is not
    /// strictly necessary (e.g., for an empty skip list).
    Head,
    /// A tail node.
    ///
    /// This is identified by having a `prev` pointer, but no `next` pointer. It
    /// also _must_ have a value.
    Tail,
    /// A body node.
    ///
    /// This is identified by having both a `prev` and `next` pointer, and
    /// _must_ have a value.
    Body,
    /// A detached node.
    ///
    /// This is identified by having neither a `prev` nor `next` pointer, and
    /// having a value. This is typically encountered when a node has been
    /// popped from a list, or when a new node has been created and is yet to
    /// be inserted into a list.
    ///
    /// Note that if it has no value, it is categorized as a head node.
    Detached,
}

/// A node in the skip list.
pub(crate) struct Node<V, const N: usize> {
    /// Owning reference to the next node.
    next: Option<NonNull<Self>>,
    /// Non-owning reference to the previous node.
    prev: Option<NonNull<Self>>,
    /// Links to subsequent nodes.
    ///
    /// Uses a fixed-capacity inline array (no separate heap allocation) for
    /// better cache locality and fewer allocations per node.
    links: LevelArray<Option<Link<V, N>>, N>,
    /// The value of the node.
    value: Option<V>,
}

// Node interface methods.
impl<V, const N: usize> Node<V, N> {
    /// Create a new head-sentinel node with the given number of skip-link slots.
    ///
    /// # Parameters
    ///
    /// * `max_levels` - The number of skip-link slots to initialise. Must be
    ///   `<= N`; a larger value panics.
    #[inline]
    #[must_use]
    pub(crate) fn new(max_levels: usize) -> Self {
        debug_assert!(
            max_levels <= N,
            "max_levels ({max_levels}) exceeds node capacity ({N})"
        );
        Self {
            next: None,
            prev: None,
            links: LevelArray::from_fn(max_levels, |_| None),
            value: None,
        }
    }

    /// Create a new data node that already holds a value.
    ///
    /// # Arguments
    ///
    /// * `height` - The number of skip-link slots to initialise. Must be
    ///   `<= N`. Same semantics as the `max_levels` parameter of
    ///   [`Node::new`].
    /// * `value` - The value the node will hold.
    #[inline]
    #[must_use]
    pub(crate) fn with_value(height: usize, value: V) -> Self {
        debug_assert!(height <= N, "height ({height}) exceeds node capacity ({N})");
        Self {
            next: None,
            prev: None,
            links: LevelArray::from_fn(height, |_| None),
            value: Some(value),
        }
    }

    /// Get the node's level.
    ///
    /// The level of a node corresponds to how many links it has (excluding the
    /// `prev`/`next` links).
    #[inline]
    pub(crate) fn level(&self) -> usize {
        self.links.len()
    }

    /// Raises the node's level to `height`, filling the new top levels with
    /// absent links. Does nothing if the node is already at least that tall.
    ///
    /// A head is at least as tall as the tallest tower in its list: every
    /// insert calls this on the head before linking a new tower.
    ///
    /// # Panics
    ///
    /// Panics if `height > N`.
    #[inline]
    pub(crate) fn grow_to(&mut self, height: usize) {
        self.links.grow(height, |_| None);
    }

    /// Drops a head's empty top levels, keeping at least one level.
    ///
    /// An empty level `l` means no tower reaches `l`, so every level above it
    /// is empty too. Removals leave a head too tall. `SkipList::insert_mut`,
    /// `push_front_mut`, `push_back_mut`, `OrderedSkipList::insert` and
    /// `SkipMap::insert` call this before their walk, as does the
    /// ordered-list insert behind `get_or_insert_with` and `replace`. Cursor,
    /// entry and bulk (`extend`, `from_iter`) inserts do not. A steady insert
    /// and remove workload therefore does not ratchet the head up to the
    /// tallest tower it ever held.
    #[inline]
    pub(crate) fn trim(&mut self) {
        let floor = self.links.len().min(1);
        let height = self
            .links
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |top| top.wrapping_add(1));
        self.links.truncate(height.max(floor));
    }

    /// Get a shared reference to the next node.
    ///
    /// The returned reference has **Frozen** provenance under Tree Borrows.
    /// Use it for **inline, read-only access only**: consume the result in the
    /// same expression and never convert it back to a raw pointer.
    ///
    /// > ⚠ **Do not call `.map(NonNull::from)` on the result.**  Doing so
    /// > silently captures the Frozen provenance tag; any subsequent write
    /// > through that pointer is UB.  Use [`next`][Self::next] instead when a
    /// > raw pointer is needed.
    #[inline]
    pub(crate) fn next_as_ref(&self) -> Option<&Self> {
        // SAFETY: The pointer can never be null, and the value is
        // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
        self.next.map(|ptr| unsafe { ptr.as_ref() })
    }

    /// Get a mutable reference to the next node.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other mutable references to the next
    /// node exist while this reference is held. Borrowing `self` mutably is
    /// necessary but not sufficient: the same next node can be reached through
    /// a different parent's `next` pointer, creating aliasing `&mut`
    /// references.
    ///
    /// > ⚠ **Do not convert the result to `NonNull` and store it.**  Use
    /// > [`next`][Self::next] when a storable raw pointer is needed.
    #[inline]
    pub(crate) unsafe fn next_as_mut(&mut self) -> Option<&mut Self> {
        // SAFETY: The pointer can never be null, and the value is
        // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
        self.next.map(|mut ptr| unsafe { ptr.as_mut() })
    }

    /// Returns the raw `NonNull` pointer to the next node, preserving its
    /// original provenance.
    ///
    /// Unlike [`next_as_ref`][Self::next_as_ref], no shared reborrow is created
    /// and the provenance tag is not downgraded to Frozen under Tree Borrows.
    /// Use this method any time the pointer is stored, passed to a function,
    /// used across loop iterations, or may later be written through.
    #[inline]
    pub(crate) fn next(&self) -> Option<NonNull<Self>> {
        self.next
    }

    /// Get a shared reference to the previous node.
    ///
    /// The returned reference has **Frozen** provenance under Tree Borrows.
    /// Use it for **inline, read-only access only**: consume the result in the
    /// same expression and never convert it back to a raw pointer.
    ///
    /// > ⚠ **Do not call `.map(NonNull::from)` on the result.**  Doing so
    /// > silently captures the Frozen provenance tag; any subsequent write
    /// > through that pointer is UB.  Use [`prev`][Self::prev] instead when a
    /// > raw pointer is needed.
    #[inline]
    pub(crate) fn prev_as_ref(&self) -> Option<&Self> {
        // SAFETY: The pointer can never be null, and the value is
        // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
        self.prev.map(|ptr| unsafe { ptr.as_ref() })
    }

    /// Get a mutable reference to the previous node.
    ///
    /// # Arguments
    ///
    /// Unlike [`next_as_mut`][Node::next_as_mut], the current node does _not_
    /// own the previous node. The fact that `self` is borrowed mutably does not
    /// imply that the previous node is not being used elsewhere. As a result,
    /// the caller must ensure that the previous node is not being used
    /// elsewhere.
    ///
    /// > ⚠ **Do not convert the result to `NonNull` and store it.**  Use
    /// > [`prev`][Self::prev] when a storable raw pointer is needed.
    #[inline]
    pub(crate) unsafe fn prev_as_mut(&mut self) -> Option<&mut Self> {
        // SAFETY: The pointer can never be null, and the value is
        // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
        self.prev.map(|mut ptr| unsafe { ptr.as_mut() })
    }

    /// Returns the raw `NonNull` pointer to the previous node, preserving its
    /// original provenance.
    ///
    /// Unlike [`prev_as_ref`][Self::prev_as_ref], no shared reborrow is created
    /// and the provenance tag is not downgraded to Frozen under Tree Borrows.
    /// Use this method any time the pointer is stored, passed to a function,
    /// used across loop iterations, or may later be written through.
    #[inline]
    pub(crate) fn prev(&self) -> Option<NonNull<Self>> {
        self.prev
    }

    /// Get a reference to the value.
    #[inline]
    pub(crate) fn value(&self) -> Option<&V> {
        self.value.as_ref()
    }

    /// Get the value of a node that is not the head sentinel.
    ///
    /// # Safety
    ///
    /// The node must not be the head. The head is the only linked node without
    /// a value. Links and `next` pointers never target the head, so every node
    /// reached through either qualifies. Removals take a node's value only
    /// after unlinking it, so a node still reachable from a list holds one.
    #[inline]
    pub(crate) unsafe fn value_unchecked(&self) -> &V {
        debug_assert!(
            self.value.is_some(),
            "value_unchecked on a node without a value"
        );
        // SAFETY: the caller guarantees the node is not the head, so it holds
        // a value.
        unsafe { self.value.as_ref().unwrap_unchecked() }
    }

    /// Get the value of the node at `node`, which is not the head sentinel.
    ///
    /// # Safety
    ///
    /// `node` must be live for `'a` and must not be the head (see
    /// [`value_unchecked`][Self::value_unchecked]).
    #[inline]
    pub(crate) unsafe fn value_at<'a>(node: NonNull<Self>) -> &'a V {
        // SAFETY: the caller guarantees `node` is live for `'a`.
        let node_ref: &'a Self = unsafe { node.as_ref() };
        // SAFETY: the caller guarantees `node` is not the head.
        unsafe { node_ref.value_unchecked() }
    }

    /// Get a mutable reference to the value.
    #[inline]
    pub(crate) fn value_mut(&mut self) -> Option<&mut V> {
        self.value.as_mut()
    }

    /// Take the value out of the node, leaving `None` in its place.
    ///
    /// Used by removal operations (e.g. [`SkipList::pop_front`]) to transfer
    /// ownership of the stored value after the node has been unlinked.
    #[inline]
    pub(crate) fn take_value(&mut self) -> Option<V> {
        self.value.take()
    }

    /// Returns the skip-link slots for this node.
    ///
    /// Each slot at index `l` holds the link to the next node reachable at
    /// skip level `l`, or `None` if no such forward node exists at that level.
    #[inline]
    pub(crate) fn links(&self) -> &[Option<Link<V, N>>] {
        &self.links
    }

    /// Returns a mutable view of the skip-link slots for this node.
    ///
    /// Used when wiring or clearing skip links during insertion, removal,
    /// or a full link rebuild.
    #[inline]
    pub(crate) fn links_mut(&mut self) -> &mut [Option<Link<V, N>>] {
        &mut self.links
    }

    /// Classifies this node based on its `prev`, `next`, and `value` fields.
    ///
    /// Classification rules:
    /// - No `prev`, no value: [`NodeType::Head`] (sentinel; may or may not have `next`).
    /// - No `prev`, no `next`, has value: [`NodeType::Detached`] (unlinked data node).
    /// - Has `prev` and `next`, has value: [`NodeType::Body`].
    /// - Has `prev`, no `next`, has value: [`NodeType::Tail`].
    /// - Any other combination is unreachable given correct node construction.
    #[inline]
    fn node_type(&self) -> NodeType {
        match (
            self.prev.is_some(),
            self.next.is_some(),
            self.value.is_some(),
        ) {
            (false, _, false) => NodeType::Head,
            (false, false, true) => NodeType::Detached,
            (true, true, true) => NodeType::Body,
            (true, false, true) => NodeType::Tail,
            _ => unreachable!("Invalid node state"),
        }
    }

    /// Remove the node from the list.
    ///
    /// This method removes the node from the list, returning the node on its
    /// own. It modifies the immediately preceding and following nodes so that
    /// their `next` and `prev` pointers are updated to point to each other. It
    /// also transfers the ownership of the node to the caller, and ensures that
    /// the ownership of the following node is transferred to the preceding
    /// node.
    ///
    /// This method must not be called on the head node, as it will result in
    /// the rest of the list becoming unreachable.
    ///
    /// # Safety
    ///
    /// The caller must uphold **all** of the following:
    ///
    /// 1. **Heap allocation**: `self` must have been originally allocated via
    ///    [`Box::new`] and then stored (e.g. via [`Box::into_raw`] into a
    ///    [`NonNull`] pointer).
    /// 2. **Node type**: `self` must not be a [`NodeType::Head`] or
    ///    [`NodeType::Detached`] node.  Popping the head would make the rest
    ///    of the list unreachable; popping a detached node has no well-defined
    ///    meaning.
    /// 3. **Links**: This method does not update the skip links of the node or
    ///    the surrounding nodes. The caller must update those links afterwards
    ///    so that no dangling link pointers remain.
    #[inline]
    pub(crate) unsafe fn pop(&mut self) -> Box<Self> {
        let _node_type = self.node_type();
        debug_assert!(
            !matches!(self.node_type(), NodeType::Head | NodeType::Detached),
            "Cannot pop head or detached node"
        );

        let mut prev = self.prev.take();
        let mut next = self.next.take();

        if let Some(prev_ptr) = prev.as_mut() {
            // SAFETY: The pointer can never be null, and the value is
            // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
            unsafe { prev_ptr.as_mut() }.next = next;
        }

        if let Some(next_ptr) = next.as_mut() {
            // SAFETY: The pointer can never be null, and the value is
            // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
            unsafe { next_ptr.as_mut() }.prev = prev;
        }

        // SAFETY: The previous node's `next` pointer has been updated to point
        // to the next node, and it is now safe to transfer ownership of the
        // node to the caller.
        unsafe { Box::from_raw(self) }
    }

    /// Joins a head node to a tail node, creating a single sequence of nodes.
    ///
    /// This method takes ownership of `head` (consuming it) and splices the
    /// nodes that follow it onto `this` (the tail).  After the call, `this` is
    /// no longer a tail node, it now points to what was the first real node
    /// after `head`.
    ///
    /// # Safety
    ///
    /// - `this` must be a valid, exclusively-accessible pointer to the tail
    ///   node.  It is stored directly in the successor's `prev` without a
    ///   reborrow, so the caller must pass its long-lived pointer.
    /// - This method does not alter the skip links of the nodes. The caller must
    ///   update those links afterwards so that no dangling link pointers remain.
    #[inline]
    pub(crate) unsafe fn join(mut this: NonNull<Self>, mut head: Self) {
        // SAFETY: the caller guarantees `this` is valid and exclusive.
        let tail = unsafe { this.as_mut() };
        debug_assert!(
            matches!(tail.node_type(), NodeType::Tail),
            "Can only join to tail node"
        );
        debug_assert!(
            matches!(head.node_type(), NodeType::Head),
            "Can only join with head node"
        );

        // Copy the head's `next` pointer to the tail node, which transfers the
        // ownership of the `head.next` reference to `tail.next`.
        tail.next = head.next.take();

        // Update the `prev` pointer of the next node in the sequence to point
        // to the tail node.
        if let Some(mut head_next) = tail.next {
            // SAFETY: The pointer can never be null, and the value is
            // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
            unsafe { head_next.as_mut() }.prev = Some(this);
        }
        // `head` is dropped here; it has already had its `next` taken so only
        // the now-empty sentinel is freed and no nodes are lost.
    }

    /// Insert a new node after the current node.
    ///
    /// The new node to be inserted must be a standalone node and not part of a
    /// list (i.e., it must not have a `prev` or `next` pointer).
    ///
    /// Inserts `node` into the list immediately after `self_ptr`.
    ///
    /// Takes ownership of the node to insert and updates the `next`/`prev`
    /// pointers of the predecessor, the new node, and the former successor so
    /// that they form a consistent doubly-linked sequence.
    ///
    /// # Safety
    ///
    /// - `self_ptr` must be a valid, exclusively-accessible pointer to a live
    ///   `Node`.  No other `&mut` borrow of that node may exist when this
    ///   function is called.
    /// - This function does not update the skip-links of any node; the caller
    ///   is responsible for keeping skip-link distances consistent.
    ///
    /// # Tree Borrows note
    ///
    /// `self_ptr` is stored directly in `node.prev` (no reborrow), and the new
    /// node is allocated via `Box::into_raw` (preserves root provenance).
    /// See [`crate::docs::internals`] for the full explanation.
    #[inline]
    pub(crate) unsafe fn insert_after(
        mut self_ptr: NonNull<Self>,
        mut node: Self,
    ) -> NonNull<Self> {
        debug_assert!(
            matches!(node.node_type(), NodeType::Detached),
            "Can only insert detached nodes."
        );

        // SAFETY: caller guarantees self_ptr is valid and exclusively accessible.
        let self_ref = unsafe { self_ptr.as_mut() };

        // Store self_ptr directly: no new child reborrow tag is created.
        node.prev = Some(self_ptr);
        node.next = self_ref.next;

        // Box::into_raw preserves root provenance; see crate::docs::internals.
        // SAFETY: Box::into_raw returns a non-null pointer.
        let node_ptr = unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(node))) };

        // If self has a 'next' node, update its 'prev' pointer to point to the
        // new node.
        if let Some(next_ptr) = self_ref.next.as_mut() {
            // SAFETY: The pointer can never be null, and the value is
            // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
            unsafe { next_ptr.as_mut() }.prev = Some(node_ptr);
        }

        // Update the predecessor's `next` pointer.  Writing through self_ref
        // (a child of self_ptr) is NOT foreign to self_ptr, so the tag stored
        // in node.prev above is never Disabled.
        self_ref.next = Some(node_ptr);
        node_ptr
    }

    /// Detaches the chain of nodes that follow `self` and returns a raw
    /// pointer to the first detached node, or `None` if there is no successor.
    ///
    /// After this call:
    /// - `self.next` is `None`.
    /// - The returned node's `prev` is `None`.
    ///
    /// The caller takes ownership of the detached chain and must eventually
    /// free it (typically by passing it to [`Node::set_head_next`] to attach
    /// it to a fresh head sentinel).
    ///
    /// # Safety
    ///
    /// The caller must hold exclusive access to `self` and to all nodes in
    /// the detached chain.  No other live reference to those nodes may exist
    /// after this call.
    #[inline]
    pub(crate) unsafe fn take_next_chain(&mut self) -> Option<NonNull<Self>> {
        let first = self.next.take()?;
        // SAFETY: `first` is a live, heap-allocated node.  We took ownership
        // via `self.next.take()`, establishing exclusive access before
        // clearing the back-pointer.
        unsafe { &mut *first.as_ptr() }.prev = None;
        Some(first)
    }

    /// Wires `first` as the immediate successor of `this`.
    ///
    /// After this call, `this.next = Some(first)` and
    /// `first.prev = Some(this)`.
    ///
    /// `this` itself is stored as the back-pointer, so `first.prev` keeps the
    /// provenance the caller holds.  Both writes go through raw pointers, so
    /// no reborrow of `this` exists for a later write to invalidate.
    ///
    /// # Safety
    ///
    /// - `this` must point to a live, heap-allocated node with `next == None`,
    ///   and must carry the long-lived provenance the container uses for that
    ///   node (the `Box::into_raw` pointer, or a pointer stored in a link).
    /// - `first` must point to a live, heap-allocated node with `prev == None`.
    /// - The caller must hold exclusive access to both nodes, and no reference
    ///   to either may be live across this call.
    #[inline]
    pub(crate) unsafe fn set_head_next(this: NonNull<Self>, first: NonNull<Self>) {
        debug_assert!(
            // SAFETY: `this` is live; no mutable reference to it is live.
            unsafe { this.as_ref() }.next.is_none(),
            "set_head_next: this.next must be None"
        );
        // SAFETY: `first` is a live, heap-allocated node exclusively owned by
        // the caller.  The place expression writes through the raw pointer
        // without creating a reference.
        unsafe { (*first.as_ptr()).prev = Some(this) };
        // SAFETY: `this` is a live, heap-allocated node exclusively owned by
        // the caller.  The write goes through `this` itself, so the copy just
        // stored in `first.prev` keeps its permissions.
        unsafe { (*this.as_ptr()).next = Some(first) };
    }

    /// Frees `first` and every node that follows it.
    ///
    /// Each node's `next` is taken before the node is freed, so this is
    /// `$O(k)$` and non-recursive in the number of nodes freed.
    ///
    /// Freeing a node runs its element's destructor, which may unwind.  The
    /// nodes not yet freed are then re-attached after `attach` and every skip
    /// link is rebuilt, so the container keeps the elements it has not
    /// destroyed instead of leaking them.  `out_len` and `out_tail`
    /// describe that survivor chain and are written on the unwind path
    /// alone; the normal path frees everything and leaves both untouched.
    ///
    /// # Safety
    ///
    /// - The caller must commit the container's post-call length and tail
    ///   before calling.  Nothing is published unless an element destructor
    ///   unwinds, so a caller that defers the update leaves `tail` pointing
    ///   into the region this call frees.
    /// - `head` must be the exclusively-owned head sentinel of the container.
    /// - `attach` must be a live node in that container with `next == None`,
    ///   and must be the node the chain hung off before it was detached.
    /// - `first`, and every node reachable from it, must have been allocated
    ///   via `Box::into_raw` and detached from the container, with no other
    ///   live reference to any of them.
    #[inline]
    pub(crate) unsafe fn drop_chain(
        head: NonNull<Self>,
        attach: NonNull<Self>,
        out_len: &mut usize,
        out_tail: &mut Option<NonNull<Self>>,
        first: Option<NonNull<Self>>,
    ) {
        let mut walk = DropChain {
            head,
            attach,
            current: first,
            out_len,
            out_tail,
        };

        while let Some(ptr) = walk.current {
            // SAFETY: every node in the chain was allocated by `Box::into_raw`
            // and detached from the container, so this `Box` is its only owner.
            let mut boxed: Box<Self> = unsafe { Box::from_raw(ptr.as_ptr()) };
            // Commit the advance before the element's destructor runs, and
            // leave the node's own `next` empty so its `Drop` stops here.
            walk.current = boxed.next.take();
            drop(boxed);
        }
    }

    /// Detaches the first data node after `head` along the base chain, or
    /// returns `None` when the list is empty.
    ///
    /// Skip links are not touched, so links to the node are left dangling.
    ///
    /// # Safety
    ///
    /// `head` must be the head of a live list, and nothing may follow that
    /// list's skip links afterwards: only `next`, `prev` and `Drop`.
    pub(crate) unsafe fn pop_chain_front(head: NonNull<Self>) -> Option<Box<Self>> {
        // SAFETY: the caller guarantees `head` is live.
        let mut first = unsafe { head.as_ref() }.next()?;
        // SAFETY: `first` is a live data node of this list.
        let node = unsafe { first.as_mut() };
        // SAFETY: `node` is a data node; the caller accepts the dangling skip
        // links `pop` leaves behind.
        Some(unsafe { node.pop() })
    }

    /// Detaches `tail` along the base chain and returns it with its
    /// predecessor (the head when `tail` was the only data node).
    ///
    /// Skip links are not touched, so links to the node are left dangling.
    ///
    /// # Safety
    ///
    /// `tail` must be the last data node of a live list, and nothing may
    /// follow that list's skip links afterwards: only `next`, `prev` and
    /// `Drop`.
    pub(crate) unsafe fn pop_chain_back(
        mut tail: NonNull<Self>,
    ) -> (Box<Self>, Option<NonNull<Self>>) {
        // SAFETY: the caller guarantees `tail` is live.
        let node = unsafe { tail.as_mut() };
        let prev = node.prev();
        // SAFETY: `node` is a data node; the caller accepts the dangling skip
        // links `pop` leaves behind.
        (unsafe { node.pop() }, prev)
    }

    /// Finds a node whose value `cmp` reports as `Equal`, or `None`.
    ///
    /// Descends from the head's top level and stops at the first `Equal`
    /// met, so with duplicates it returns any one of them. The pointer comes
    /// from a link or `next`, so it keeps the provenance needed to write
    /// through it. A comparator that is not a total order can make the walk
    /// miss a match or return a wrong node, but the walk only moves forward
    /// and always ends.
    ///
    /// # Safety
    ///
    /// `head` must be the head of a live list.
    pub(crate) unsafe fn find_ptr<F>(head: NonNull<Self>, cmp: F) -> Option<NonNull<Self>>
    where
        F: Fn(&V) -> Ordering,
    {
        let mut current = head;
        // SAFETY: the caller guarantees `head` is live.
        let mut level = unsafe { head.as_ref() }.level();
        'walk: loop {
            // SAFETY: `current` is `head` or was reached through a link or
            // `next`, so it is live.
            let links = unsafe { current.as_ref() }.links();
            for l in (0..level).rev() {
                let Some(link) = links.get(l).and_then(Option::as_ref) else {
                    continue;
                };
                let node = link.node();
                // SAFETY: a live link target, never the head; see
                // `Node::value_unchecked`.
                let value = unsafe { Node::value_at(node) };
                match cmp(value) {
                    Ordering::Less => {
                        current = node;
                        level = l.wrapping_add(1);
                        continue 'walk;
                    }
                    Ordering::Equal => return Some(node),
                    Ordering::Greater => {}
                }
            }
            // SAFETY: as above.
            let next = unsafe { current.as_ref() }.next()?;
            // SAFETY: `next` is a live list node, never the head; see
            // `Node::value_unchecked`.
            let value = unsafe { Node::value_at(next) };
            match cmp(value) {
                Ordering::Less => {
                    current = next;
                    level = 0;
                }
                Ordering::Equal => return Some(next),
                Ordering::Greater => return None,
            }
        }
    }

    /// For each level of the head's tower, the last node at that level
    /// before `tail`.
    ///
    /// Descends from the head's top level, each level resuming where the one
    /// above stopped, so the walk is `$O(\log n)$` on average. Below the
    /// tail's height the returned node links to `tail`; at and above it the
    /// node is the last one on its level. Only pointers are compared, so the
    /// result does not depend on any comparator.
    ///
    /// # Safety
    ///
    /// `head` must be the head of a live list whose last node is `tail`.
    pub(crate) unsafe fn tail_precursors(
        head: NonNull<Self>,
        tail: NonNull<Self>,
    ) -> LevelArray<NonNull<Self>, N> {
        // SAFETY: the caller guarantees `head` is live.
        let max_levels = unsafe { head.as_ref() }.level();
        let mut precursors = LevelArray::from_fn(max_levels, |_| head);
        let mut current = head;
        for (l, precursor) in precursors.iter_mut().enumerate().rev() {
            // SAFETY: `current` is `head` or was reached through a link, so it
            // is live.
            while let Some(link) = unsafe { current.as_ref() }
                .links()
                .get(l)
                .and_then(Option::as_ref)
                && link.node() != tail
            {
                current = link.node();
            }
            *precursor = current;
        }
        precursors
    }

    /// Moves every node after internal rank `at` to the empty list headed by
    /// `new_head`, and returns the node at rank `at` (the head when `at` is
    /// 0), now the last node left in the list.
    ///
    /// Each level's link that crosses the cut moves to `new_head`, so moved
    /// nodes keep their towers and the split is `$O(\log n)$`. The walk is by
    /// rank, so the result does not depend on any comparator.
    ///
    /// # Safety
    ///
    /// - `head` must be the exclusively-owned head of a well-formed list with
    ///   more than `at` data nodes.
    /// - `new_head` must be a live, exclusively-owned head of an empty list,
    ///   with the same height as `head`, and not part of `head`'s list.
    #[expect(
        clippy::expect_used,
        reason = "`at` is below the length, so the walk reaches rank `at + 1`, whose node \
                  has a predecessor; a crossing link ends past rank `at`, so its new \
                  distance is at least 1"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < precursors.len() == head.level() == new_head.level(), and each \
                  precursor at level l has a link slot at l"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "the chain cut and the link moves touch disjoint fields of exclusively \
                  owned nodes"
    )]
    pub(crate) unsafe fn split_after(
        head: NonNull<Self>,
        at: usize,
        new_head: NonNull<Self>,
    ) -> NonNull<Self> {
        let mut visitor = IndexMutVisitor::new(head, at.wrapping_add(1));
        let first_moved = visitor.traverse().expect("the list is longer than `at`");
        let precursors = visitor.precursors();
        let ranks = visitor.precursor_distances();
        // SAFETY: the caller guarantees exclusive access to both lists. Every
        // precursor is the last node at its level with rank at most `at`, so
        // its link at that level, if any, ends in the moved part.
        unsafe {
            let last_kept = first_moved
                .as_ref()
                .prev()
                .expect("a data node has a predecessor");
            let first = (*last_kept.as_ptr())
                .take_next_chain()
                .expect("`first_moved` follows `last_kept`");
            Node::set_head_next(new_head, first);
            for (l, (&pred, &rank)) in precursors.iter().zip(ranks).enumerate() {
                if let Some(link) = (*pred.as_ptr()).links_mut()[l].take() {
                    let distance = rank.wrapping_add(link.distance().get()).wrapping_sub(at);
                    (*new_head.as_ptr()).links_mut()[l] =
                        Some(Link::new(link.node(), distance).expect("the link crosses the cut"));
                }
            }
            last_kept
        }
    }

    /// Moves every node of the list headed by `other_head` to the end of the
    /// list headed by `head`, leaving the other list empty.
    ///
    /// The shorter head is first raised to the taller one's height. Only the
    /// links that end the first list are then rewritten, so the splice is
    /// `$O(\log n)$`. No comparator is consulted. Both lists' lengths and
    /// tails are updated to match.
    ///
    /// # Safety
    ///
    /// - `head` must be the exclusively-owned head of a well-formed list of
    ///   `*len` data nodes whose last node is `*tail` (`None` when empty).
    /// - `other_head` must be the exclusively-owned head of a different,
    ///   non-empty, well-formed list of `*other_len` data nodes whose last
    ///   node is `*other_tail`.
    #[expect(
        clippy::expect_used,
        reason = "the other list is non-empty, and a link from its head ends at rank 1 or \
                  later, past every node of the first list"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < head.level() == other_head.level() once both are raised; the last node \
                  at level l has a link slot at l"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "the chain splice and the link moves touch disjoint fields of exclusively \
                  owned nodes"
    )]
    pub(crate) unsafe fn concat(
        head: NonNull<Self>,
        len: &mut usize,
        tail: &mut Option<NonNull<Self>>,
        other_head: NonNull<Self>,
        other_len: &mut usize,
        other_tail: &mut Option<NonNull<Self>>,
    ) {
        // SAFETY: the caller guarantees both heads are live and exclusively
        // owned, and no reference to either is held.
        let (head_mut, other_head_mut) =
            unsafe { (&mut *head.as_ptr(), &mut *other_head.as_ptr()) };
        let height = head_mut.level().max(other_head_mut.level());
        head_mut.grow_to(height);
        other_head_mut.grow_to(height);

        // Every rank is below `len + 1`, so the walk ends at the tail having
        // recorded the last node at each level.
        let mut visitor = IndexMutVisitor::new(head, len.wrapping_add(1));
        visitor.traverse();
        let precursors = visitor.precursors();
        let ranks = visitor.precursor_distances();
        let len_before = *len;
        // SAFETY: the caller guarantees exclusive access to both lists, which
        // share no nodes.
        unsafe {
            let first = (*other_head.as_ptr())
                .take_next_chain()
                .expect("the other list is non-empty");
            Node::set_head_next(tail.unwrap_or(head), first);
            for (l, (&pred, &rank)) in precursors.iter().zip(ranks).enumerate() {
                if let Some(link) = (*other_head.as_ptr()).links_mut()[l].take() {
                    let distance = len_before
                        .wrapping_add(link.distance().get())
                        .wrapping_sub(rank);
                    (*pred.as_ptr()).links_mut()[l] =
                        Some(Link::new(link.node(), distance).expect("the link crosses the seam"));
                }
            }
        }
        *len = len_before.saturating_add(*other_len);
        *tail = other_tail.take();
        *other_len = 0;
    }

    /// Filter and rebuild skip links in a single `$O(n)$` forward pass.
    ///
    /// Walks the `next` chain starting from `head` (the head sentinel).
    /// For each data node:
    ///
    /// - If `keep(raw_ptr)` returns `true`, the node is retained and its skip
    ///   links are re-wired into the rebuilt list.
    /// - If `keep(raw_ptr)` returns `false`, `on_drop(boxed_node)` is called
    ///   and the node is removed from the chain.
    ///
    /// `out_len` receives the count of retained nodes and `out_tail` the last
    /// retained node, or `None` if all nodes were removed.
    ///
    /// Both `keep` and `on_drop` run caller code that may unwind: `keep` is a
    /// user predicate, and `on_drop` runs the element's destructor.  The
    /// results are written through `out_len` and `out_tail` rather than
    /// returned so that the walk can publish them on the unwind path too,
    /// where a returned value would be discarded.  A panic therefore cannot
    /// leave the container claiming nodes the walk has already freed.  Every
    /// node the walk had not yet reached is retained, as is the node `keep`
    /// panicked on.
    ///
    /// # Safety
    ///
    /// - `head` must be the exclusively-owned head sentinel of a valid
    ///   prev/next chain of heap-allocated [`Node<V>`] instances.
    /// - No other live reference to any node in the chain may exist during the
    ///   call.
    /// - The `keep` closure must not structurally modify the chain (no
    ///   insertion, removal, or pointer update).  It may read or mutate node
    ///   values before returning.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-pointer traversal, link clearing and the optional pop all touch \
                  provably disjoint heap nodes; grouping them avoids unsafe-crossing \
                  raw-pointer variables"
    )]
    pub(crate) unsafe fn filter_rebuild<F, D>(
        head: NonNull<Self>,
        out_len: &mut usize,
        out_tail: &mut Option<NonNull<Self>>,
        mut keep: F,
        mut on_drop: D,
    ) where
        F: FnMut(*mut Self) -> bool,
        D: FnMut(Box<Self>),
    {
        let head_ptr: *mut Self = head.as_ptr();
        // SAFETY: head is a valid, exclusively-owned head sentinel per the
        // function's safety contract.
        let (max_levels, first) = unsafe {
            for link in (*head_ptr).links_mut() {
                *link = None;
            }
            ((*head_ptr).level(), (*head_ptr).next())
        };

        let mut walk = Rebuild {
            predecessors: LevelArray::from_fn(max_levels, |_| (head_ptr, 0_usize)),
            rank: 0,
            tail: None,
            current: first,
            out_len,
            out_tail,
        };

        while let Some(cur_nn) = walk.current {
            let cur: *mut Self = cur_nn.as_ptr();
            // SAFETY: cur is a live, exclusively-owned data node in the chain.
            // Read the successor before any structural mutation.
            let next = unsafe { (*cur).next() };

            // `keep` may unwind.  Leaving `walk.current` on `cur` until it
            // returns makes the guard retain `cur` along with the rest.
            if keep(cur) {
                walk.current = next;
                // SAFETY: cur is live and still linked into the chain.
                unsafe { walk.retain(cur_nn) };
            } else {
                // The element's destructor may unwind, so commit the advance
                // before `cur` is unlinked and destroyed.
                walk.current = next;
                // SAFETY: cur is a live data node; `pop` detaches it from the
                // chain and transfers ownership to the returned Box.
                on_drop(unsafe { (*cur).pop() });
            }
        }

        // Dropping `walk` publishes the length and tail.
    }

    // MARK: Precursor check

    /// Panics unless every level below the height of `target` has its
    /// precursor linking straight to `target`.
    ///
    /// For each `l < target.level()`, `precursors[l].links()[l]` must be
    /// `Some` and point at `target`.  A removal that splices through a
    /// precursor failing this check would leave a link to a freed node, so
    /// the check calls [`total_order_violation`] instead.  It reads links
    /// only and compares pointers, so it makes no comparator calls.
    ///
    /// After a completed [`OrdMutVisitor`] walk that found `target`, the
    /// check always passes, so those callers run it in debug builds only.
    /// Let `X` be the node `precursors[l]` links to at level
    /// `l < target.level()`. The walk refuses to advance onto `X` (its
    /// bound), and no link at level `l` or below can skip `X`, whose height
    /// exceeds `l`. So `target` lies at or before `X`. A `target` strictly
    /// before `X` would be reached by that level-`l` link first. So
    /// `target` is `X`. The argument holds for any comparator. Precursors
    /// built another way must still run the check.
    ///
    /// [`OrdMutVisitor`]: crate::node::visitor::OrdMutVisitor
    ///
    /// # Panics
    ///
    /// Panics with the total-order message when a precursor does not link
    /// to `target`.
    ///
    /// # Safety
    ///
    /// `target` and every node in `precursors` must be live nodes of one
    /// list, with no live `&mut` reference to any of them.
    pub(crate) unsafe fn check_precursors(precursors: &[NonNull<Self>], target: NonNull<Self>) {
        // SAFETY: the caller guarantees `target` is live; the reference is
        // short-lived.
        let height = unsafe { target.as_ref() }.level();
        debug_assert!(
            precursors.len() >= height,
            "one precursor per level of the target"
        );
        for (l, pred) in precursors.iter().take(height).enumerate() {
            // SAFETY: the caller guarantees every precursor is live; the
            // reference is short-lived.
            let link = unsafe { pred.as_ref() }
                .links()
                .get(l)
                .and_then(Option::as_ref);
            if link.is_none_or(|found| found.node() != target) {
                total_order_violation();
            }
        }
    }

    // MARK: Cursor surgery
    /// Unlinks `target` from the skip links and the base-layer doubly-linked
    /// list, returning it as an owned [`Box`].
    ///
    /// For each level `l` in `0..precursors.len()`, `precursors[l]` is the
    /// level-`l` predecessor of `target`.  The function patches those
    /// predecessors so that they skip over `target` (either by merging the
    /// two link distances or by decrementing the spanning link distance) and
    /// then calls [`pop`](Self::pop) to detach `target` from the base layer.
    ///
    /// The caller is responsible for extracting the value (via
    /// [`take_value`](Self::take_value)) and for updating any tail and length
    /// bookkeeping on the containing collection.
    ///
    /// # Safety
    ///
    /// - `target` must be a live, linked data node - not the head sentinel.
    /// - `precursors[l]` must be the correct level-`l` predecessor of
    ///   `target` for every `l` in `0..precursors.len()`.
    /// - No other live references to `target` or any node in `precursors`
    ///   may exist for the duration of this call.
    #[expect(
        clippy::expect_used,
        reason = "distances are always >= 1 in a valid list; expects fire only on \
                  internal invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l iterates 0..precursors.len(); links_mut()[l] is valid because \
                  precursors.len() == max_levels == links.len()"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link surgery touches provably disjoint pred and target nodes"
    )]
    pub(crate) unsafe fn splice_out(
        target: NonNull<Self>,
        precursors: &[NonNull<Self>],
    ) -> Box<Self> {
        // SAFETY: caller guarantees target is a valid, live node.
        let target_height = unsafe { target.as_ref() }.level();
        let target_raw = target.as_ptr();

        for (l, &pred_nn) in precursors.iter().enumerate() {
            let pred_ptr = pred_nn.as_ptr();
            if l < target_height {
                // SAFETY: pred_ptr and target_raw are valid, live, and disjoint.
                let old_link = unsafe { (*pred_ptr).links_mut()[l].take() };
                // SAFETY: same — target_raw is valid and disjoint from pred_ptr.
                let target_link = unsafe { (*target_raw).links_mut()[l].take() };
                // SAFETY: pred_ptr is valid; the assignment cannot alias target_raw
                // because l < target_height implies target has a link at level l,
                // and a node cannot be its own level-l predecessor.
                unsafe {
                    (*pred_ptr).links_mut()[l] = match (old_link, target_link) {
                        (Some(p2t), Some(t2s)) => {
                            let new_dist = p2t
                                .distance()
                                .get()
                                .saturating_add(t2s.distance().get())
                                .saturating_sub(1);
                            Some(Link::new(t2s.node(), new_dist).expect("new_dist >= 1"))
                        }
                        (_, None) => None,
                        (None, tgt) => {
                            debug_assert!(
                                false,
                                "level-{l} predecessor had no link pointing to target"
                            );
                            tgt
                        }
                    };
                }
            } else {
                // SAFETY: pred_ptr is valid; `l >= target_height` means this is a
                // spanning link whose distance must be decremented.
                if let Some(link) = unsafe { (*pred_ptr).links_mut()[l].as_mut() } {
                    link.decrement_distance()
                        .expect("skip link spanning target has distance >= 2");
                }
            }
        }

        // SAFETY: caller guarantees target is a valid, live data node.
        // All skip links pointing to it have been removed above, so the only
        // remaining pointer to target is the prev-node's `next` ownership;
        // pop() transfers that ownership to the returned Box.
        unsafe { (*target_raw).pop() }
    }

    /// Wires the skip links for a newly inserted node.
    ///
    /// `new_node` must already be inserted in the base-layer doubly-linked
    /// list (via [`insert_after`](Self::insert_after)) before this is called.
    /// For each level `l` in `0..precursors.len()`:
    ///
    /// - If `l < height`: patches `precursors[l]` to point to `new_node`
    ///   and sets `new_node.links[l]` to point to the old successor.
    /// - If `l >= height`: increments the spanning predecessor link by 1.
    ///
    /// # Safety
    ///
    /// - `new_node` must be freshly inserted in the base layer and not yet
    ///   wired into any skip links.
    /// - `precursors[l]` must be the correct level-`l` predecessor of
    ///   `new_node` for every `l` in `0..precursors.len()`.
    /// - `precursor_ranks[l]` must be the rank of `precursors[l]`.
    /// - `precursors.len() == precursor_ranks.len()`.
    #[expect(
        clippy::expect_used,
        reason = "distances are always >= 1 during a valid insert; expects fire only on \
                  internal invariant violations"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l iterates 0..precursors.len(); links_mut()[l] is valid because \
                  precursors.len() == max_levels == links.len()"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "wiring pred and new_node links touches provably disjoint heap nodes"
    )]
    pub(crate) unsafe fn wire_links(
        new_node: NonNull<Self>,
        new_rank: usize,
        height: usize,
        precursors: &[NonNull<Self>],
        precursor_ranks: &[usize],
    ) {
        let new_raw = new_node.as_ptr();

        for (l, (&pred_nn, &pred_rank)) in precursors.iter().zip(precursor_ranks.iter()).enumerate()
        {
            let pred_ptr = pred_nn.as_ptr();
            if l < height {
                let distance = new_rank.saturating_sub(pred_rank);
                // SAFETY: pred_ptr and new_raw are valid, live, and disjoint.
                let old_link = unsafe { (*pred_ptr).links_mut()[l].take() };
                // SAFETY: pred_ptr and new_raw are valid and disjoint; new_raw is
                // freshly inserted so no other node's link points to it yet.
                unsafe {
                    (*pred_ptr).links_mut()[l] = Some(
                        Link::new(NonNull::new_unchecked(new_raw), distance)
                            .expect("distance >= 1"),
                    );
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
                }
            } else {
                // SAFETY: pred_ptr is valid; `l >= height` means this is a
                // spanning link whose distance must be incremented.
                if let Some(link) = unsafe { (*pred_ptr).links_mut()[l].as_mut() } {
                    link.increment_distance()
                        .expect("distance overflow requires > usize::MAX nodes");
                }
            }
        }
    }
}

/// Incremental state of a [`Node::drop_chain`] walk.
///
/// Dropping the guard re-attaches the nodes the walk has not yet freed and
/// rebuilds every skip link, so an element destructor that unwinds leaves its
/// surviving neighbours owned by the container rather than stranded.
struct DropChain<'a, V, const N: usize> {
    /// Head sentinel of the container, used to rebuild the skip links.
    head: NonNull<Node<V, N>>,
    /// Node the survivors are re-attached after.  Its `next` is `None` for as
    /// long as the walk runs.
    attach: NonNull<Node<V, N>>,
    /// Next node to free, or `None` once the chain is exhausted.
    current: Option<NonNull<Node<V, N>>>,
    /// Container length to publish into.
    out_len: &'a mut usize,
    /// Container tail to publish into.
    out_tail: &'a mut Option<NonNull<Node<V, N>>>,
}

impl<V, const N: usize> Drop for DropChain<'_, V, N> {
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "re-attaching the survivors and rebuilding the links are one recovery \
                  step on the same chain; splitting them would obscure that"
    )]
    fn drop(&mut self) {
        let Some(first) = self.current.take() else {
            return;
        };

        // SAFETY: `first` heads a chain of live, exclusively-owned nodes whose
        // predecessor has already been freed, and `attach.next` is None.
        // `set_head_next` overwrites `first.prev`, so the freed predecessor is
        // never read.  Re-attaching only rewires pointers and rebuilding only
        // retains, so neither can unwind a second time.
        unsafe {
            Node::set_head_next(self.attach, first);
            Node::filter_rebuild(self.head, self.out_len, self.out_tail, |_| true, |_| {});
        }
    }
}

/// The last node at every level of a list, with its rank, for appending
/// after the tail without a walk per element.
///
/// Valid only while nothing else changes the list: any other insertion or
/// removal leaves it stale.
pub(crate) struct Frontier<V, const N: usize> {
    /// Head of the list, raised when a pushed tower is taller.
    head: NonNull<Node<V, N>>,
    /// Last node at each level and its rank (the head has rank 0).
    last: LevelArray<(NonNull<Node<V, N>>, usize), N>,
}

impl<V, const N: usize> Frontier<V, N> {
    /// Records the last node at every level of the list headed by `head`.
    ///
    /// # Safety
    ///
    /// `head` must be the head of a live list of `len` data nodes.
    pub(crate) unsafe fn new(head: NonNull<Node<V, N>>, len: usize) -> Self {
        // Every rank is below `len + 1`, so the walk ends at the tail having
        // recorded the last node at each level.
        let mut visitor = IndexMutVisitor::new(head, len.wrapping_add(1));
        visitor.traverse();
        let (_, precursors, distances) = visitor.into_parts();
        Self {
            head,
            // `from_fn` also fills the unused slots above the head's height.
            last: LevelArray::from_fn(precursors.len(), |l| {
                (
                    precursors.get(l).copied().unwrap_or(head),
                    distances.get(l).copied().unwrap_or(0),
                )
            }),
        }
    }

    /// Links a new node holding `value`, with a tower of `height`, after
    /// `tail`, and returns it.
    ///
    /// # Safety
    ///
    /// `tail` must be the last node (or the head of an empty list) at rank
    /// `rank` of the list this frontier was built for, unchanged since except
    /// through this method.
    ///
    /// # Panics
    ///
    /// Panics if `height > N`.
    #[expect(
        clippy::expect_used,
        reason = "each recorded rank is at most `rank`, so the distance is at least 1"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < height, the height of both the new node and `pred`"
    )]
    pub(crate) unsafe fn push(
        &mut self,
        tail: NonNull<Node<V, N>>,
        rank: usize,
        height: usize,
        value: V,
    ) -> NonNull<Node<V, N>> {
        // `last` is always as tall as the head.
        if height > self.last.len() {
            let head = self.head;
            // SAFETY: the head is live and no reference to it is held.
            unsafe { (*head.as_ptr()).grow_to(height) };
            self.last.grow(height, |_| (head, 0));
        }
        // SAFETY: the caller guarantees `tail` is the live last node.
        let new = unsafe { Node::insert_after(tail, Node::with_value(height, value)) };
        let new_rank = rank.wrapping_add(1);
        // Levels at or above `height` keep their links: no link spans past
        // the old tail.
        for (l, (pred, pred_rank)) in self.last.iter_mut().enumerate().take(height) {
            let link = Link::new(new, new_rank.wrapping_sub(*pred_rank)).expect("distance >= 1");
            // SAFETY: `pred` is a live node of the list, the last at level
            // `l`, so it has a link slot at that level.
            unsafe { pred.as_mut() }.links_mut()[l] = Some(link);
            *pred = new;
            *pred_rank = new_rank;
        }
        new
    }
}

/// A sorted list that [`extend_appending`] can append to.
///
/// # Safety
///
/// [`chain`][Self::chain] must return the head, length and tail of one live
/// list.
pub(crate) unsafe trait Appendable<V, const N: usize> {
    /// Whether `item` sorts strictly after the last element; `true` for an
    /// empty list.
    fn after_tail(&self, item: &V) -> bool;

    /// Draws the tower height for a new node.
    fn level(&mut self) -> usize;

    /// The list's head, with its length and tail for the caller to update.
    #[expect(clippy::type_complexity, reason = "internal code")]
    fn chain(
        &mut self,
    ) -> (
        NonNull<Node<V, N>>,
        &mut usize,
        &mut Option<NonNull<Node<V, N>>>,
    );
}

/// Adds every item from `iter` to `list`, appending in O(1) each item that
/// sorts after the last element and passing the rest to `insert`.
///
/// Sorted input therefore costs O(n) after one O(log n) walk. An item that
/// does not sort after the tail, such as an equal one, takes the container's
/// own insertion path, so duplicates are handled exactly as `insert` would.
pub(crate) fn extend_appending<L, V, const N: usize>(
    list: &mut L,
    iter: impl IntoIterator<Item = V>,
    mut insert: impl FnMut(&mut L, V),
) where
    L: Appendable<V, N>,
{
    let mut frontier: Option<Frontier<V, N>> = None;
    for item in iter {
        if !list.after_tail(&item) {
            frontier = None;
            insert(list, item);
            continue;
        }
        let height = list.level();
        let (head, len, tail) = list.chain();
        // SAFETY: `Appendable` guarantees `head` heads a live list of `*len`
        // nodes.
        let appender = frontier.get_or_insert_with(|| unsafe { Frontier::new(head, *len) });
        let last = tail.unwrap_or(head);
        // SAFETY: `last` is the last node at rank `*len`, and the list has
        // changed only through `appender` since it was built.
        let node = unsafe { appender.push(last, *len, height, item) };
        *tail = Some(node);
        *len = len.saturating_add(1);
    }
}

/// Incremental state of a [`Node::filter_rebuild`] walk.
///
/// Dropping the guard retains every node the walk has not yet reached and
/// publishes the resulting length and tail, so the `next` chain, the skip
/// links and the container's bookkeeping agree on every exit path — including
/// an unwind out of the predicate or out of an element's destructor.
struct Rebuild<'a, V, const N: usize> {
    /// Last node wired at each level, paired with its rank in the rebuilt
    /// list.  Level `l` starts at the head sentinel with rank 0.
    predecessors: LevelArray<(*mut Node<V, N>, usize), N>,
    /// Number of nodes retained so far.
    rank: usize,
    /// Last node retained so far.
    tail: Option<NonNull<Node<V, N>>>,
    /// Next node to visit, or `None` once the chain is exhausted.
    current: Option<NonNull<Node<V, N>>>,
    /// Container length to publish into.
    out_len: &'a mut usize,
    /// Container tail to publish into.
    out_tail: &'a mut Option<NonNull<Node<V, N>>>,
}

impl<V, const N: usize> Rebuild<'_, V, N> {
    /// Wires `cur` into the rebuilt list as the next retained node.
    ///
    /// # Safety
    ///
    /// `cur` must be a live, exclusively-owned data node still linked into the
    /// chain, and every pointer in `predecessors` must still be live.
    #[expect(
        clippy::expect_used,
        reason = "Link::new(dist) returns Err only when dist == 0; \
                  dist = rank - pred_rank and rank > pred_rank whenever a \
                  predecessor is recorded, so dist >= 1 always"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < node_height <= max_levels = predecessors.len() = head.level(); \
                  l indexes both predecessors[l] and links_mut()[l] so a plain index \
                  loop is the clearest expression"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "link clearing and link wiring touch provably disjoint heap nodes; \
                  grouping them avoids unsafe-crossing raw-pointer variables"
    )]
    unsafe fn retain(&mut self, cur: NonNull<Node<V, N>>) {
        let ptr: *mut Node<V, N> = cur.as_ptr();
        self.rank = self.rank.saturating_add(1);
        self.tail = Some(cur);

        // SAFETY: ptr and every recorded predecessor are live, exclusively
        // owned, heap-allocated nodes.
        unsafe {
            // Clear this node's forward links; they are re-wired below.
            for link in (*ptr).links_mut() {
                *link = None;
            }

            let height = (*ptr).level();
            for l in 0..height {
                let (pred_ptr, pred_rank) = self.predecessors[l];
                let dist = self.rank.saturating_sub(pred_rank);
                (*pred_ptr).links_mut()[l] =
                    Some(Link::new(cur, dist).expect("dist >= 1 by construction"));
                self.predecessors[l] = (ptr, self.rank);
            }
        }
    }

    /// Retains every node the walk has not yet reached.
    ///
    /// # Safety
    ///
    /// Every node reachable from `current` must be live and exclusively owned.
    unsafe fn retain_rest(&mut self) {
        while let Some(cur) = self.current {
            // SAFETY: cur is a live, exclusively-owned data node in the chain.
            self.current = unsafe { (*cur.as_ptr()).next() };
            // SAFETY: cur is live and still linked into the chain.
            unsafe { self.retain(cur) };
        }
    }
}

impl<V, const N: usize> Drop for Rebuild<'_, V, N> {
    fn drop(&mut self) {
        // SAFETY: every node still reachable from `current` is live and
        // exclusively owned by the container being rebuilt.  Retaining a node
        // only rewires links, so this cannot unwind a second time.
        unsafe { self.retain_rest() };
        *self.out_len = self.rank;
        *self.out_tail = self.tail;
    }
}

// MARK: Structural checker

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    reason = "test-only checker; ranks are bounded by len and levels by N"
)]
impl<V, const N: usize> Node<V, N> {
    /// Panic unless the list rooted at `head` is structurally sound.
    ///
    /// One forward walk of the base chain checks that:
    ///
    /// - `next` and `prev` mirror each other, and only the head has no value;
    /// - no node is taller than the head;
    /// - `len` equals the number of data nodes and `tail` is the last node
    ///   (`None` when the list is empty);
    /// - each skip link at level `l` targets the next node with a level-`l`
    ///   tower, at a distance equal to their rank difference;
    /// - the last tower at each level has no link at that level.
    ///
    /// # Safety
    ///
    /// `head` must point at a live list whose nodes are all live.
    pub(crate) unsafe fn assert_well_formed(
        head: NonNull<Self>,
        len: usize,
        tail: Option<NonNull<Self>>,
    ) {
        // Last node seen with a tower at each level, with its rank.
        let mut last_tower: Vec<Option<(NonNull<Self>, usize)>> = vec![None; N];
        let mut current = head;
        let mut rank = 0_usize;

        // SAFETY: the caller guarantees `head` is live; the reference is
        // short-lived.
        let head_ref = unsafe { head.as_ref() };
        assert!(head_ref.prev().is_none(), "head has a prev pointer");
        assert!(head_ref.value().is_none(), "head holds a value");
        for slot in last_tower.iter_mut().take(head_ref.level()) {
            *slot = Some((head, 0));
        }

        // SAFETY: `current` is always a live node of the list.
        while let Some(next) = unsafe { current.as_ref() }.next() {
            rank += 1;
            assert!(
                rank <= len,
                "base chain has more than len ({len}) data nodes"
            );
            // SAFETY: `next` is a live node of the list.
            let node = unsafe { next.as_ref() };
            assert!(
                node.prev() == Some(current),
                "prev of node at rank {rank} does not point back to rank {}",
                rank - 1
            );
            assert!(node.value().is_some(), "node at rank {rank} has no value");
            assert!(
                node.level() <= head_ref.level(),
                "node at rank {rank} has height {}, above the head height {}",
                node.level(),
                head_ref.level()
            );

            for (level, slot) in last_tower.iter_mut().enumerate().take(node.level()) {
                if let Some((owner, owner_rank)) = *slot {
                    // SAFETY: `owner` is a live node of the list.
                    let owner_ref = unsafe { owner.as_ref() };
                    let link = owner_ref.links()[level].as_ref().unwrap_or_else(|| {
                        panic!(
                            "node at rank {owner_rank} has no level-{level} link, \
                             but rank {rank} is the next level-{level} tower"
                        )
                    });
                    assert!(
                        link.node() == next,
                        "level-{level} link of rank {owner_rank} does not target the next \
                         tower (rank {rank})"
                    );
                    assert!(
                        link.distance().get() == rank - owner_rank,
                        "level-{level} link of rank {owner_rank} has distance {}, expected {}",
                        link.distance().get(),
                        rank - owner_rank
                    );
                }
                *slot = Some((next, rank));
            }
            current = next;
        }

        assert!(
            rank == len,
            "data node count {rank} does not match len {len}"
        );
        assert!(
            tail == if rank == 0 { None } else { Some(current) },
            "tail does not point at the last node"
        );
        for (level, slot) in last_tower.iter().enumerate() {
            if let Some((owner, owner_rank)) = *slot {
                // SAFETY: `owner` is a live node of the list.
                let owner_ref = unsafe { owner.as_ref() };
                assert!(
                    owner_ref.links().get(level).is_none_or(Option::is_none),
                    "last level-{level} tower (rank {owner_rank}) still has a link"
                );
            }
        }
    }
}

#[cfg(any(debug_assertions, test))]
#[allow(
    clippy::allow_attributes,
    clippy::use_debug,
    dead_code,
    reason = "Used for debugging"
)]
impl<V: Debug, const N: usize> Node<V, N> {
    /// Generate a map of pointers to node indices.
    #[inline]
    fn ptr_index_map(&self) -> BTreeMap<NonNull<Self>, usize> {
        let mut hm = BTreeMap::new();
        let mut current = self;
        let mut index = 0_usize;
        loop {
            hm.insert(NonNull::from(current), index);
            if let Some(next) = current.next_as_ref() {
                current = next;
                index = index.saturating_add(1);
            } else {
                break;
            }
        }
        hm
    }

    /// Display the node and all subsequent nodes.
    ///
    /// This is only used for debugging purposes. If the node or its links are
    /// not properly initialized or contain invalid links, this method may
    /// result in undefined behavior.
    ///
    /// The output will be of the form:
    ///
    /// ```text
    /// [03] head -------------------------------------------> 08
    /// [02] head -------------------------> 05 -------------> 08
    /// [01] head -------> 02 -------------> 05 -------> 07 -> 08
    /// [->] head -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08
    /// [<-] head <- 01 <- 02 <- 03 <- 04 <- 05 <- 06 <- 07 <- 08
    ///
    /// values:
    /// head: None
    /// 1: Some(...)
    /// ...
    /// tail: None
    /// ```
    #[inline]
    pub(crate) fn display(&self) -> Result<String, fmt::Error> {
        let mut output = String::new();
        write!(
            output,
            "{}\n\n{}",
            self.display_links()?,
            self.display_values()?
        )?;
        Ok(output)
    }

    /// Display the links of the node.
    ///
    /// This is only used for debugging purposes. If the node or its links are
    /// not properly initialized or contain invalid links, this method may
    /// result in undefined behavior.
    ///
    /// The output will be of the form:
    ///
    /// ```text
    /// [03] 00
    /// [02] 00 -------------------------> 05
    /// [01] 00 -------> 02 -------------> 05
    /// [->] 00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06
    /// [<-] 00 <- 01 <- 02 <- 03 <- 04 <- 05 <- 06
    ///
    /// [00|02] None
    /// [01|00] Some(..)
    /// [02|01] Some(..)
    /// [03|00] Some(..)
    /// [04|00] Some(..)
    /// [05|02] Some(..)
    /// [06|00] Some(..)
    /// ```
    ///
    /// The first section displays the links between nodes at each level. The
    /// `[->]` level displays the sequence of `next` pointers, while the `[<-]`
    /// level displays the sequence of `prev` pointers. The numbers indicate the
    /// positions of the index within the list (with `00` being the head).
    ///
    /// The second section displays the value of each node, in the form
    /// `[index|level] value`.
    #[inline]
    fn display_links(&self) -> Result<String, fmt::Error> {
        let hm = self.ptr_index_map();
        let mut output = String::new();

        for level in (0..self.level()).rev() {
            write!(output, "[{:02}]: ", level.saturating_add(1))?;

            let mut current = self;
            loop {
                if let Some(index) = hm.get(&NonNull::from(current)) {
                    write!(output, "{index:02}")?;
                } else {
                    writeln!(output, "??")?;
                    break;
                }

                if let Some(Some(link)) = current.links().get(level) {
                    write!(
                        output,
                        " {}-> ",
                        "------".repeat(link.distance().get().saturating_sub(1))
                    )?;
                    // SAFETY: link.node() is a valid heap-allocated node.
                    current = unsafe { link.node().as_ref() };
                } else {
                    writeln!(output)?;
                    break;
                }
            }
        }

        write!(output, "[->]: ")?;
        let mut current = self;
        loop {
            if let Some(index) = hm.get(&NonNull::from(current)) {
                write!(output, "{index:02}")?;
            } else {
                write!(output, "??")?;
            }

            if let Some(next) = current.next_as_ref() {
                write!(output, " -> ")?;
                current = next;
            } else {
                break;
            }
        }

        // Display the lowest level of reverse links. Since we're going
        // backwards, we build the string in reverse and prepend to it.
        let mut rev_string = String::new();
        loop {
            if let Some(index) = hm.get(&NonNull::from(current)) {
                rev_string.insert_str(0, &format!("{index:02}"));
            } else {
                rev_string.insert_str(0, "??");
            }

            if let Some(prev) = current.prev_as_ref() {
                rev_string.insert_str(0, " <- ");
                current = prev;
            } else {
                break;
            }
        }
        write!(output, "\n[<-]: {rev_string}")?;

        Ok(output)
    }

    /// Display the value of each node.
    ///
    /// This is only used for debugging purposes. If the node or its links are
    /// not properly initialized or contain invalid links, this method may
    /// result in undefined behavior.
    ///
    /// The output will be of the form:
    ///
    /// ```text
    /// [00|04] None
    /// [01|02] Some(1)
    /// [02|00] Some(2)
    /// [03|03] Some(3)
    /// ```
    ///
    /// which is of the form `[index|level] value`.
    #[inline]
    fn display_values(&self) -> Result<String, fmt::Error> {
        let mut output = String::new();
        let mut current = self;
        let mut index = 0_usize;
        loop {
            writeln!(
                output,
                "[{:02}|{:02}] {:?}",
                index,
                current.level(),
                current.value()
            )?;
            if let Some(node) = current.next_as_ref() {
                current = node;
                index = index.saturating_add(1);
            } else {
                break;
            }
        }
        Ok(output)
    }

    /// Display the pointer addresses of the nodes.
    ///
    /// This is only used for debugging purposes. If the node or its links are
    /// not properly initialized or contain invalid links, this method may
    /// result in undefined behavior.
    #[inline]
    fn display_ptrs(&self) -> Result<String, fmt::Error> {
        let mut output = String::new();
        let mut current = self;
        let mut index = 0_usize;
        loop {
            writeln!(
                output,
                "[{:02}] {:?} <- {:?} -> {:?}",
                index,
                current.prev,
                NonNull::from(current),
                current.next
            )?;
            if let Some(node) = current.next_as_ref() {
                current = node;
                index = index.saturating_add(1);
            } else {
                break;
            }
        }
        Ok(output)
    }
}

impl<V: Debug, const N: usize> Debug for Node<V, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("next", &self.next)
            .field("prev", &self.prev)
            .field("links", &self.links)
            .field("value", &self.value)
            .finish()
    }
}

impl<V, const N: usize> Drop for Node<V, N> {
    /// Drop a node and all subsequent nodes in the chain.
    ///
    /// `NonNull<T>` does not own its pointee, so without this implementation
    /// every node after the head would be leaked when the head is dropped.
    ///
    /// # Implementation note
    ///
    /// We iterate using raw pointers (`NonNull::as_ptr`) rather than the safe
    /// `next_as_mut()` method.  Using `next_as_mut()` inside `drop` would
    /// create a `&mut Node<V>` that aliases the `Box<Node<V>>` we are about to
    /// construct, which is undefined behaviour.  Working at the raw-pointer
    /// level avoids that aliasing entirely.
    fn drop(&mut self) {
        // Walk the chain, reconstructing each heap-allocated node as a `Box`
        // and immediately dropping it.  Each `Box::from_raw` is safe because:
        //   1. Every node reachable via `next` was allocated with `Box::new`
        //      and stored via `Box::into_raw` (see `insert_after`).
        //   2. We take `next` out of the node before reconstructing it as a
        //      `Box`, so this path is only visited once per node.
        // Skip links become dangling as nodes are freed, but they are owned by
        // the nodes themselves and are freed together with each node.
        while let Some(next_ptr) = self.next.take() {
            // SAFETY: `next_ptr` points to a heap-allocated `Node<V>` that
            // was created via `Box::new` + `Box::into_raw` in `insert_after`.
            // We have taken ownership by removing it from `self.next`, so no
            // other `Box` wraps this pointer.
            let mut next_box: Box<Node<V, N>> = unsafe { Box::from_raw(next_ptr.as_ptr()) };
            // Take `next_box.next` so that *its* drop does not recurse further
            // (the loop will handle it in the next iteration instead).
            self.next = next_box.next.take();
            drop(next_box);
        }
    }
}

#[expect(
    clippy::undocumented_unsafe_blocks,
    clippy::multiple_unsafe_ops_per_block,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "test code, covered by miri, so safety guarantees can be relaxed"
)]
#[cfg(test)]
pub(crate) mod tests {
    use core::ptr::NonNull;

    use anyhow::Result;
    use insta::assert_snapshot;
    use pretty_assertions::{assert_eq, assert_matches};
    use rstest::{fixture, rstest};

    use crate::node::{Node, NodeType, link::Link};

    pub(crate) const MAX_LEVELS: usize = 3;

    // MARK: layout

    #[test]
    fn node_size_is_unchanged_by_level_array() {
        // prev + next + value + links + height, with Option<Link> niche-packed.
        // 16 slots of Option<Link<u64, 16>> (16 bytes each) = 256 bytes.
        let links = 16 * core::mem::size_of::<Option<Link<u64, 16>>>();
        let base = 2 * core::mem::size_of::<Option<NonNull<Node<u64, 16>>>>()
            + core::mem::size_of::<Option<u64>>();
        // The u8 height pads to the pointer alignment.
        assert_eq!(core::mem::size_of::<Node<u64, 16>>(), base + links + 8);
        assert_eq!(core::mem::size_of::<Option<Link<u64, 16>>>(), 16);
    }

    // MARK: new

    #[test]
    fn node_new() {
        let node = Node::<(), MAX_LEVELS>::new(MAX_LEVELS);
        assert!(node.next.is_none());
        assert!(node.prev.is_none());
        assert_eq!(node.links.len(), MAX_LEVELS);
        assert!(node.value.is_none());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "value_unchecked on a node without a value")]
    fn value_unchecked_on_head_panics() {
        let head = Node::<u64, MAX_LEVELS>::new(MAX_LEVELS);
        // SAFETY: deliberately violated; the debug assertion must fire before
        // the value is read.
        let _value: &u64 = unsafe { head.value_unchecked() };
    }

    #[test]
    fn new_node_properties() {
        let mut node = Node::<(), MAX_LEVELS>::new(MAX_LEVELS);

        assert!(node.next_as_ref().is_none());
        assert!(unsafe { node.next_as_mut() }.is_none());

        assert!(node.prev_as_ref().is_none());
        assert!(unsafe { node.prev_as_mut() }.is_none());

        assert!(node.value().is_none());
        assert!(node.value_mut().is_none());
    }

    /// Build a simple skiplist with values [10, 20, 30, 40].
    ///
    /// The links are as follows:
    ///
    /// head
    /// head -------------> 30
    /// head -------> 20 -> 30 -> 40
    /// head -> 10 -> 20 -> 30 -> 40
    #[fixture]
    pub(crate) fn skiplist() -> Result<NonNull<Node<u8, MAX_LEVELS>>> {
        // Allocate the head sentinel via Box::into_raw so that `head_nn`
        // carries **root allocation provenance** (T_alloc, no parent tag).
        // Returning NonNull (not Box::from_raw) avoids Miri creating a
        // protected function-call sibling tag at the test boundary.  All
        // accesses to head's allocation (including the Box::deref_mut calls
        // inside filter_rebuild) go through children of T_alloc, which are
        // never foreign to T_alloc (= v1.prev).  See MIRI.md Issue 3 /
        // Anti-pattern B.
        let mut head_nn = unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(Node::<u8, MAX_LEVELS>::new(
                MAX_LEVELS,
            ))))
        };

        let mut v1 = Node::<u8, MAX_LEVELS>::new(0);
        let mut v2 = Node::<u8, MAX_LEVELS>::new(1);
        let mut v3 = Node::<u8, MAX_LEVELS>::new(1);
        let mut v4 = Node::<u8, MAX_LEVELS>::new(0);

        v1.value = Some(10);
        v2.value = Some(20);
        v3.value = Some(30);
        v4.value = Some(40);

        unsafe {
            // Chain insert_after return values: each returned NonNull carries
            // root allocation provenance (Box::into_raw), so storing it as the
            // successor's `prev` pointer never creates sibling tags.  See
            // MIRI.md Issue 4 / Anti-pattern B.
            let v1_ptr = Node::insert_after(head_nn, v1);
            let mut v2_ptr = Node::insert_after(v1_ptr, v2);
            let mut v3_ptr = Node::insert_after(v2_ptr, v3);
            let v4_ptr = Node::insert_after(v3_ptr, v4);

            // Build higher level links using root-provenance pointers directly:
            // eliminates NonNull::from(&node_ref) Frozen-tag construction and
            // next_as_mut() chain navigation.  See MIRI.md Anti-patterns A & B.
            //
            // head -------------> v3
            // head -------> v2 -> v3 -> v4
            // head -> v1 -> v2 -> v3 -> v4
            head_nn.as_mut().links[1] = Some(Link::new(v3_ptr, 3)?);
            head_nn.as_mut().links[0] = Some(Link::new(v2_ptr, 2)?);
            v2_ptr.as_mut().links[0] = Some(Link::new(v3_ptr, 1)?);
            v3_ptr.as_mut().links[0] = Some(Link::new(v4_ptr, 1)?);
        }

        Ok(head_nn)
    }

    // MARK: assert_well_formed

    type Ptr = NonNull<Node<u8, MAX_LEVELS>>;

    /// The [`skiplist`] fixture with its skip links rebuilt from node heights.
    ///
    /// The fixture wires `head.links[1]` at a node of height 1, so its links
    /// are not a function of its heights.  Rebuilding yields the consistent
    /// layout the checker expects:
    ///
    /// ```text
    /// head -> v1 -> v2 -> v3 -> v4
    /// head ------> v2 -> v3
    /// ```
    pub(crate) fn well_formed_skiplist() -> Result<(Ptr, Ptr)> {
        let head = skiplist()?;
        let mut len = 0;
        let mut tail = None;
        // SAFETY: head is exclusively owned and every data node is live.
        unsafe { Node::filter_rebuild(head, &mut len, &mut tail, |_| true, |_| {}) };
        Ok((head, tail.expect("fixture is non-empty")))
    }

    /// A well-formed list of `[10, 20, 30, 40]` where `30` has a level-1
    /// tower, returned as `(head, tail)`.
    ///
    /// ```text
    /// head -------------> 30
    /// head ------> 20 -> 30
    /// head -> 10 -> 20 -> 30 -> 40
    /// ```
    #[expect(
        clippy::unnecessary_wraps,
        reason = "shares a signature with `well_formed_skiplist` in `#[case]` lists"
    )]
    pub(crate) fn tall_skiplist() -> Result<(Ptr, Ptr)> {
        // SAFETY: `Box::into_raw` never returns null.
        let head = unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(Node::<u8, MAX_LEVELS>::new(
                MAX_LEVELS,
            ))))
        };
        // SAFETY: each insert targets the live node returned by the last one.
        unsafe {
            let v1 = Node::insert_after(head, Node::with_value(0, 10));
            let v2 = Node::insert_after(v1, Node::with_value(1, 20));
            let v3 = Node::insert_after(v2, Node::with_value(2, 30));
            Node::insert_after(v3, Node::with_value(0, 40));
        }
        let mut len = 0;
        let mut tail = None;
        // SAFETY: head is exclusively owned and every data node is live.
        unsafe { Node::filter_rebuild(head, &mut len, &mut tail, |_| true, |_| {}) };
        Ok((head, tail.expect("fixture is non-empty")))
    }

    /// Run the checker on the rebuilt fixture after `corrupt` has edited it,
    /// with the given `len`, and report whether it panicked.
    fn well_formed_panics(
        len: usize,
        corrupt: impl FnOnce(NonNull<Node<u8, MAX_LEVELS>>) -> Result<()>,
    ) -> Result<bool> {
        let (head, tail) = well_formed_skiplist()?;
        corrupt(head)?;
        let panicked = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| unsafe {
            Node::assert_well_formed(head, len, Some(tail));
        }))
        .is_err();
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(panicked)
    }

    #[test]
    fn well_formed_accepts_fixture() -> Result<()> {
        assert!(!well_formed_panics(4, |_| Ok(()))?);
        Ok(())
    }

    #[test]
    fn well_formed_rejects_wrong_distance() -> Result<()> {
        // head.links[0] -> v2 really spans 2 positions.
        let panicked = well_formed_panics(4, |mut head| {
            let target = unsafe { head.as_ref() }.links()[0].as_ref().unwrap().node();
            unsafe { head.as_mut() }.links_mut()[0] = Some(Link::new(target, 3)?);
            Ok(())
        })?;
        assert!(panicked);
        Ok(())
    }

    #[test]
    fn well_formed_rejects_link_over_tower() -> Result<()> {
        // v2 and v3 both have a level-0 tower.  Retarget head.links[0] from v2
        // to v3 at v3's true distance (3), so only the check that a link lands
        // on the next tower can catch it.
        let panicked = well_formed_panics(4, |mut head| {
            let v2 = unsafe { head.as_ref() }.links()[0].as_ref().unwrap().node();
            let v3 = unsafe { v2.as_ref() }.links()[0].as_ref().unwrap().node();
            unsafe { head.as_mut() }.links_mut()[0] = Some(Link::new(v3, 3)?);
            Ok(())
        })?;
        assert!(panicked);
        Ok(())
    }

    #[test]
    fn well_formed_rejects_wrong_len() -> Result<()> {
        assert!(well_formed_panics(5, |_| Ok(()))?);
        Ok(())
    }

    #[test]
    fn well_formed_rejects_node_taller_than_head() -> Result<()> {
        // head (height 1) -> v1 (height 2).  Every link is consistent, so only
        // the height check can catch it.
        let mut head = unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(Node::<u8, MAX_LEVELS>::new(1))))
        };
        let v1 = unsafe { Node::insert_after(head, Node::with_value(2, 10)) };
        unsafe { head.as_mut() }.links_mut()[0] = Some(Link::new(v1, 1)?);

        let result = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| unsafe {
            Node::assert_well_formed(head, 1, Some(v1));
        }));

        let payload = result.expect_err("a node taller than the head must panic");
        assert_eq!(
            payload.downcast_ref::<String>().map(String::as_str),
            Some("node at rank 1 has height 2, above the head height 1")
        );
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: check_precursors

    /// The data node at 1-based `rank` along the base chain.
    fn nth(head: Ptr, rank: usize) -> Ptr {
        let mut node = head;
        for _ in 0..rank {
            // SAFETY: every node reached from a fixture head is live.
            node = unsafe { node.as_ref() }
                .next()
                .expect("rank within the list");
        }
        node
    }

    #[test]
    fn check_precursors_accepts_true_precursors() -> Result<()> {
        // head -> v1 -> v2 -> v3 -> v4
        // head ------> v2 -> v3
        let (head, tail) = well_formed_skiplist()?;
        let (v1, v2, v3) = (nth(head, 1), nth(head, 2), nth(head, 3));
        // SAFETY: every pointer is a live node of the fixture, which this
        // test owns and frees last.
        unsafe {
            Node::check_precursors(&[head, head, head], v1);
            Node::check_precursors(&[head, head, head], v2);
            Node::check_precursors(&[v2, head, head], v3);
            Node::check_precursors(&[v3, head, head], tail);
            Node::assert_well_formed(head, 4, Some(tail));
            drop(Box::from_raw(head.as_ptr()));
        }
        Ok(())
    }

    #[rstest]
    // `head.links[0]` targets v2, the level-0 predecessor of v3.
    #[case::level_0_wrong_target(well_formed_skiplist(), 3, [0, 0, 0])]
    // 30 has a level-1 tower; `20` has no level-1 link.
    #[case::level_1_missing_link(tall_skiplist(), 3, [2, 2, 0])]
    fn check_precursors_rejects_wrong_link(
        #[case] fixture: Result<(Ptr, Ptr)>,
        #[case] target_rank: usize,
        #[case] precursor_ranks: [usize; MAX_LEVELS],
    ) -> Result<()> {
        use crate::test_util::TOTAL_ORDER_PANIC;

        let (head, tail) = fixture?;
        let target = nth(head, target_rank);
        let precursors = precursor_ranks.map(|rank| nth(head, rank));

        // SAFETY: every pointer is a live node of the fixture.
        let result = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| unsafe {
            Node::check_precursors(&precursors, target);
        }));

        let payload = result.expect_err("a precursor that misses the target must panic");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&TOTAL_ORDER_PANIC));
        // SAFETY: the check only reads, so the fixture is intact; the test
        // owns it and frees it last.
        unsafe {
            Node::assert_well_formed(head, 4, Some(tail));
            drop(Box::from_raw(head.as_ptr()));
        }
        Ok(())
    }

    // MARK: display

    #[rstest]
    fn node_display(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            assert_snapshot!(
                unsafe { head.as_ref() }.display()?,
                @"
                [03]: 00
                [02]: 00 -------------> 03
                [01]: 00 -------> 02 -> 03 -> 04
                [->]: 00 -> 01 -> 02 -> 03 -> 04
                [<-]: 00 <- 01 <- 02 <- 03 <- 04

                [00|03] None
                [01|00] Some(10)
                [02|01] Some(20)
                [03|01] Some(30)
                [04|00] Some(40)
                "
            );
        }

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn node_properties(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let head_ref = unsafe { head.as_ref() };

        assert_matches!(head_ref.node_type(), NodeType::Head);
        assert_eq!(head_ref.level(), 3);

        let mut node = head_ref.next_as_ref().expect("v1 not found");
        assert_matches!(node.node_type(), NodeType::Body);
        assert_eq!(node.level(), 0);

        node = node.next_as_ref().expect("v2 not found");
        assert_matches!(node.node_type(), NodeType::Body);
        assert_eq!(node.level(), 1);

        node = node.next_as_ref().expect("v3 not found");
        assert_matches!(node.node_type(), NodeType::Body);
        assert_eq!(node.level(), 1);

        node = node.next_as_ref().expect("v4 not found");
        assert_matches!(node.node_type(), NodeType::Tail);
        assert_eq!(node.level(), 0);

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: pop

    #[rstest]
    fn pop_node(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;

        // SAFETY: `head` is a valid node with `v1` as its next.
        let v1 = unsafe { (*head.as_ptr()).next_as_mut() }.expect("v1 not found");
        // SAFETY: `v1` is a valid node with `v2` as its next.
        let v2 = unsafe { v1.next_as_mut() }.expect("v2 not found");
        let detached_node = unsafe { v2.pop() };

        assert_eq!(detached_node.value, Some(20));
        assert_matches!(detached_node.node_type(), NodeType::Detached);

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // Note: The sequence of values should be valid. It is fine for the
            // links to be invalid as the node has been popped without updating the
            // links.
            assert_snapshot!(
                unsafe { head.as_ref() }.display()?,
                @"
                [03]: 00
                [02]: 00 -------------> 02
                [01]: 00 -------> ??
                [->]: 00 -> 01 -> 02 -> 03
                [<-]: 00 <- 01 <- 02 <- 03

                [00|03] None
                [01|00] Some(10)
                [02|01] Some(30)
                [03|00] Some(40)
                "
            );
        }

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: insert_after

    #[rstest]
    fn insert_after_head_node(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut new_node = Node::<u8, MAX_LEVELS>::new(MAX_LEVELS);
        new_node.value = Some(100);

        unsafe { Node::insert_after(head, new_node) };

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // Note: The sequence of values should be valid. The links are not
            // updated and therefore may result in UB when displayed.
            assert_snapshot!(
                unsafe { head.as_ref() }.display()?,
                @"
                [03]: 00
                [02]: 00 -------------> 04
                [01]: 00 -------> 03 -> 04 -> 05
                [->]: 00 -> 01 -> 02 -> 03 -> 04 -> 05
                [<-]: 00 <- 01 <- 02 <- 03 <- 04 <- 05

                [00|03] None
                [01|03] Some(100)
                [02|00] Some(10)
                [03|01] Some(20)
                [04|01] Some(30)
                [05|00] Some(40)
                "
            );
        }

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn insert_after_body_node(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut new_node = Node::<u8, MAX_LEVELS>::new(MAX_LEVELS);
        new_node.value = Some(100);

        // SAFETY: `head` is a valid node with `v1` as its next.
        let v1 = unsafe { (*head.as_ptr()).next_as_mut() }.expect("v1 not found");
        // SAFETY: `v1` has exclusive access and no other references to its successor exist.
        unsafe { Node::insert_after(NonNull::from_mut(v1), new_node) };

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // Note: The sequence of values should be valid. The links are not
            // updated and therefore may result in UB when displayed.
            assert_snapshot!(
                unsafe { head.as_ref() }.display()?,
                @"
                [03]: 00
                [02]: 00 -------------> 04
                [01]: 00 -------> 03 -> 04 -> 05
                [->]: 00 -> 01 -> 02 -> 03 -> 04 -> 05
                [<-]: 00 <- 01 <- 02 <- 03 <- 04 <- 05

                [00|03] None
                [01|00] Some(10)
                [02|03] Some(100)
                [03|01] Some(20)
                [04|01] Some(30)
                [05|00] Some(40)
                "
            );
        }

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn insert_after_tail_node(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let mut new_node = Node::<u8, MAX_LEVELS>::new(MAX_LEVELS);
        new_node.value = Some(100);

        // SAFETY: Each node is valid with the next node as its successor.
        let v1 = unsafe { (*head.as_ptr()).next_as_mut() }.expect("v1 not found");
        let v2 = unsafe { v1.next_as_mut() }.expect("v2 not found");
        let v3 = unsafe { v2.next_as_mut() }.expect("v3 not found");
        let v4 = unsafe { v3.next_as_mut() }.expect("v4 not found");
        // SAFETY: `v4` has exclusive access and no other references to its successor exist.
        unsafe { Node::insert_after(NonNull::from_mut(v4), new_node) };

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // Note: The sequence of values should be valid. The links are not
            // updated and therefore may result in UB when displayed.
            assert_snapshot!(
                unsafe { head.as_ref() }.display()?,
                @"
                [03]: 00
                [02]: 00 -------------> 03
                [01]: 00 -------> 02 -> 03 -> 04
                [->]: 00 -> 01 -> 02 -> 03 -> 04 -> 05
                [<-]: 00 <- 01 <- 02 <- 03 <- 04 <- 05

                [00|03] None
                [01|00] Some(10)
                [02|01] Some(20)
                [03|01] Some(30)
                [04|00] Some(40)
                [05|03] Some(100)
                "
            );
        }

        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: filter_rebuild

    #[rstest]
    fn filter_rebuild_empty_list() {
        let mut head = Box::new(Node::<u8, MAX_LEVELS>::new(MAX_LEVELS));
        let (mut new_len, mut new_tail) = (0, None);
        unsafe {
            Node::filter_rebuild(
                NonNull::from_mut(&mut *head),
                &mut new_len,
                &mut new_tail,
                |_| true,
                |_| {},
            );
        }
        assert_eq!(new_len, 0);
        assert!(new_tail.is_none());
        assert!(head.next_as_ref().is_none());
    }

    #[rstest]
    fn filter_rebuild_keep_all(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let (mut new_len, mut new_tail) = (0, None);
        unsafe { Node::filter_rebuild(head, &mut new_len, &mut new_tail, |_| true, |_| {}) };

        assert_eq!(new_len, 4);
        let vals: Vec<u8> = {
            let mut v = Vec::new();
            let mut cur = unsafe { head.as_ref() }.next_as_ref();
            while let Some(n) = cur {
                v.push(*n.value().expect("data node"));
                cur = n.next_as_ref();
            }
            v
        };
        assert_eq!(vals, [10, 20, 30, 40]);
        let tail_val = unsafe { new_tail.expect("tail exists").as_ref() }
            .value()
            .copied();
        assert_eq!(tail_val, Some(40));

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // Links are rebuilt from node heights: v1=h0, v2=h1, v3=h1, v4=h0.
            // head.links[0] → v2 (distance 2), head.links[1..] → None.
            // v2.links[0] → v3 (distance 1), v3.links[0] → None.
            assert_snapshot!(
                unsafe { head.as_ref() }.display()?,
                @"
                [03]: 00
                [02]: 00
                [01]: 00 -------> 02 -> 03
                [->]: 00 -> 01 -> 02 -> 03 -> 04
                [<-]: 00 <- 01 <- 02 <- 03 <- 04

                [00|03] None
                [01|00] Some(10)
                [02|01] Some(20)
                [03|01] Some(30)
                [04|00] Some(40)
                "
            );
        }
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn filter_rebuild_keep_none(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let mut dropped_vals: Vec<u8> = Vec::new();
        let head = skiplist?;
        let (mut new_len, mut new_tail) = (0, None);
        unsafe {
            Node::filter_rebuild(
                head,
                &mut new_len,
                &mut new_tail,
                |_| false,
                |mut b| dropped_vals.push(b.take_value().expect("data node")),
            );
        }

        assert_eq!(new_len, 0);
        assert!(new_tail.is_none());
        assert!(unsafe { head.as_ref() }.next_as_ref().is_none());
        // on_drop called in traversal order.
        assert_eq!(dropped_vals, [10, 20, 30, 40]);

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // All nodes dropped; head has no links and no successors.
            assert_snapshot!(
                unsafe { head.as_ref() }.display()?,
                @"
                [03]: 00
                [02]: 00
                [01]: 00
                [->]: 00
                [<-]: 00

                [00|03] None
                "
            );
        }
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn filter_rebuild_keep_first_and_third(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        // Keep v1 (value 10) and v3 (value 30); drop v2 and v4.
        let head = skiplist?;
        let (mut new_len, mut new_tail) = (0, None);
        unsafe {
            Node::filter_rebuild(
                head,
                &mut new_len,
                &mut new_tail,
                |cur| {
                    let v = (*cur).value().copied();
                    v == Some(10) || v == Some(30)
                },
                |_| {},
            );
        }

        assert_eq!(new_len, 2);
        let vals: Vec<u8> = {
            let mut v = Vec::new();
            let mut cur = unsafe { head.as_ref() }.next_as_ref();
            while let Some(n) = cur {
                v.push(*n.value().expect("data node"));
                cur = n.next_as_ref();
            }
            v
        };
        assert_eq!(vals, [10, 30]);
        let tail_val = unsafe { new_tail.expect("tail exists").as_ref() }
            .value()
            .copied();
        assert_eq!(tail_val, Some(30));

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // v1 (h0) and v3 (h1) retained.  head.links[0] → v3 (distance 2).
            assert_snapshot!(
                unsafe { head.as_ref() }.display()?,
                @"
                [03]: 00
                [02]: 00
                [01]: 00 -------> 02
                [->]: 00 -> 01 -> 02
                [<-]: 00 <- 01 <- 02

                [00|03] None
                [01|00] Some(10)
                [02|01] Some(30)
                "
            );
        }
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    #[rstest]
    fn filter_rebuild_on_drop_receives_correct_values(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let mut dropped: Vec<u8> = Vec::new();
        let head = skiplist?;
        // Keep v2 and v4; drop v1 and v3.
        unsafe {
            Node::filter_rebuild(
                head,
                &mut 0,
                &mut None,
                |cur| {
                    let v = (*cur).value().copied();
                    v == Some(20) || v == Some(40)
                },
                |mut b| dropped.push(b.take_value().expect("data node")),
            );
        }

        // Dropped values must arrive in traversal order: v1 then v3.
        assert_eq!(dropped, [10, 30]);
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// After dropping nodes with no skip-link slots, all head links must be
    /// `None` because no retained node can anchor a skip link.
    #[rstest]
    fn filter_rebuild_links_consistent_after_partial_keep(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;
        // Drop v2 and v3 (both height 1); keep v1 (height 0) and v4 (height 0).
        unsafe {
            Node::filter_rebuild(
                head,
                &mut 0,
                &mut None,
                |cur| {
                    let v = (*cur).value().copied();
                    v != Some(20) && v != Some(30)
                },
                |_| {},
            );
        }

        // Chain must be head -> v1 -> v4.
        let vals: Vec<u8> = {
            let mut v = Vec::new();
            let mut cur = unsafe { head.as_ref() }.next_as_ref();
            while let Some(n) = cur {
                v.push(*n.value().expect("data node"));
                cur = n.next_as_ref();
            }
            v
        };
        assert_eq!(vals, [10, 40]);
        // v1 and v4 both have height 0, so all head skip links must be None.
        for link in unsafe { head.as_ref() }.links() {
            assert!(link.is_none(), "head skip link should be None");
        }

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // Neither kept node has skip-link slots, so all levels show only the head.
            assert_snapshot!(
                unsafe { head.as_ref() }.display()?,
                @"
                [03]: 00
                [02]: 00
                [01]: 00
                [->]: 00 -> 01 -> 02
                [<-]: 00 <- 01 <- 02

                [00|03] None
                [01|00] Some(10)
                [02|00] Some(40)
                "
            );
        }
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// Rebuilding skip links over a keep-all pass must produce correct distances.
    #[rstest]
    fn filter_rebuild_keep_all_links_rebuilt(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        // Fixture node heights: v1=0, v2=1, v3=1, v4=0.
        //
        // After a keep-all rebuild:
        //   head.links[0] → v2 (distance 2)  (head → v1 (height 0, no link) → v2 (height 1))
        //   head.links[1] → None             (no height-2 node exists)
        //   head.links[2] → None             (no height-3 node exists)
        //   v2.links[0]   → v3 (distance 1)
        //   v3.links[0]   → None             (v4 has height 0 so nothing wires into v3.links[0])
        let head = skiplist?;
        unsafe {
            Node::filter_rebuild(head, &mut 0, &mut None, |_| true, |_| {});
        }

        let link0 = unsafe { head.as_ref() }.links()[0]
            .as_ref()
            .expect("head.links[0] must be Some");
        assert_eq!(unsafe { link0.node().as_ref() }.value().copied(), Some(20));
        assert_eq!(link0.distance().get(), 2);

        assert!(
            unsafe { head.as_ref() }.links()[1].is_none(),
            "head.links[1] must be None"
        );
        assert!(
            unsafe { head.as_ref() }.links()[2].is_none(),
            "head.links[2] must be None"
        );

        let v2 = unsafe { head.as_ref() }
            .next_as_ref()
            .expect("v1")
            .next_as_ref()
            .expect("v2");
        let v2_link0 = v2.links()[0].as_ref().expect("v2.links[0] must be Some");
        assert_eq!(
            unsafe { v2_link0.node().as_ref() }.value().copied(),
            Some(30)
        );
        assert_eq!(v2_link0.distance().get(), 1);

        let v3 = v2.next_as_ref().expect("v3");
        assert!(v3.links()[0].is_none(), "v3.links[0] must be None");

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // Same structural outcome as filter_rebuild_keep_all.
            assert_snapshot!(
                unsafe { head.as_ref() }.display()?,
                @"
                [03]: 00
                [02]: 00
                [01]: 00 -------> 02 -> 03
                [->]: 00 -> 01 -> 02 -> 03 -> 04
                [<-]: 00 <- 01 <- 02 <- 03 <- 04

                [00|03] None
                [01|00] Some(10)
                [02|01] Some(20)
                [03|01] Some(30)
                [04|00] Some(40)
                "
            );
        }
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    // MARK: join

    #[test]
    fn join_two_nonempty_lists() -> Result<()> {
        // Build list1: head1 → v1(10) → v2(20), all height 0 (no skip links).
        let mut head1 = Box::new(Node::<u8, MAX_LEVELS>::new(MAX_LEVELS));
        unsafe {
            Node::insert_after(NonNull::from_mut(&mut *head1), Node::with_value(0, 10_u8));
            Node::insert_after(
                NonNull::from_mut(head1.next_as_mut().expect("v1")),
                Node::with_value(0, 20_u8),
            );
        }

        // Build list2: head2 → v3(30) → v4(40), all height 0 (no skip links).
        let mut head2 = Box::new(Node::<u8, MAX_LEVELS>::new(MAX_LEVELS));
        unsafe {
            Node::insert_after(NonNull::from_mut(&mut *head2), Node::with_value(0, 30_u8));
            Node::insert_after(
                NonNull::from_mut(head2.next_as_mut().expect("v3")),
                Node::with_value(0, 40_u8),
            );
        }

        unsafe {
            // Find v2 (tail of list1) and join. Moving *head2 out of the Box
            // frees the allocation; join immediately updates v3.prev to point
            // to v2, so the stale back-pointer is never dereferenced.
            let v2 = head1.next_as_mut().expect("v1").next_as_mut().expect("v2");
            debug_assert!(matches!(v2.node_type(), NodeType::Tail));
            Node::join(NonNull::from_mut(v2), *head2);
        }

        let vals: Vec<u8> = {
            let mut v = Vec::new();
            let mut cur = head1.next_as_ref();
            while let Some(n) = cur {
                v.push(*n.value().expect("data node"));
                cur = n.next_as_ref();
            }
            v
        };
        assert_eq!(vals, [10, 20, 30, 40]);

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // All nodes height 0; head1 has no skip links.
            assert_snapshot!(
                head1.display()?,
                @"
                [03]: 00
                [02]: 00
                [01]: 00
                [->]: 00 -> 01 -> 02 -> 03 -> 04
                [<-]: 00 <- 01 <- 02 <- 03 <- 04

                [00|03] None
                [01|00] Some(10)
                [02|00] Some(20)
                [03|00] Some(30)
                [04|00] Some(40)
                "
            );
        }
        Ok(())
    }

    #[test]
    fn join_empty_second_list() -> Result<()> {
        // Build list1: head1 → v1(10) → v2(20), all height 0.
        let mut head1 = Box::new(Node::<u8, MAX_LEVELS>::new(MAX_LEVELS));
        unsafe {
            Node::insert_after(NonNull::from_mut(&mut *head1), Node::with_value(0, 10_u8));
            Node::insert_after(
                NonNull::from_mut(head1.next_as_mut().expect("v1")),
                Node::with_value(0, 20_u8),
            );
        }

        // Joining an empty head2 (no nodes) must leave list1 unchanged.
        let head2 = Box::new(Node::<u8, MAX_LEVELS>::new(MAX_LEVELS));
        // SAFETY: Moving *head2 frees the allocation; join sees head.next = None
        // and leaves self.next unchanged.
        unsafe {
            let v2 = head1.next_as_mut().expect("v1").next_as_mut().expect("v2");
            debug_assert!(matches!(v2.node_type(), NodeType::Tail));
            Node::join(NonNull::from_mut(v2), *head2);
        }

        let vals: Vec<u8> = {
            let mut v = Vec::new();
            let mut cur = head1.next_as_ref();
            while let Some(n) = cur {
                v.push(*n.value().expect("data node"));
                cur = n.next_as_ref();
            }
            v
        };
        assert_eq!(vals, [10, 20]);

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            assert_snapshot!(
                head1.display()?,
                @"
                [03]: 00
                [02]: 00
                [01]: 00
                [->]: 00 -> 01 -> 02
                [<-]: 00 <- 01 <- 02

                [00|03] None
                [01|00] Some(10)
                [02|00] Some(20)
                "
            );
        }
        Ok(())
    }

    // MARK: take_next_chain / set_head_next

    #[test]
    fn take_next_chain_returns_none_on_empty_head() {
        let mut head = Box::new(Node::<u8, MAX_LEVELS>::new(MAX_LEVELS));
        // SAFETY: head is exclusively owned; there is no chain to detach.
        let result = unsafe { head.take_next_chain() };
        assert!(result.is_none());
        assert!(head.next_as_ref().is_none());
    }

    #[rstest]
    fn take_next_chain_and_set_head_next_round_trip(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let head = skiplist?;

        // Detach the chain from head.
        // SAFETY: `head` is exclusively owned; no other references exist.
        let first_nn = unsafe { (*head.as_ptr()).take_next_chain() }.expect("fixture is non-empty");

        assert!(
            unsafe { head.as_ref() }.next_as_ref().is_none(),
            "head must be empty after take"
        );
        // SAFETY: `first_nn` is a live heap-allocated node whose ownership was
        // transferred to us by take_next_chain.
        assert!(
            unsafe { first_nn.as_ref() }.prev_as_ref().is_none(),
            "first detached node must have no prev"
        );

        // Wire the chain onto a fresh head.
        // SAFETY: `Box::into_raw` never returns null.
        let head2_nn = unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(Node::<u8, MAX_LEVELS>::new(
                MAX_LEVELS,
            ))))
        };
        // SAFETY: head2 has no current successor; first_nn is the exclusive
        // owner of the detached chain.
        unsafe { Node::set_head_next(head2_nn, first_nn) };
        // SAFETY: `head2_nn` came from `Box::into_raw` and no other reference
        // to it exists.
        let head2 = unsafe { Box::from_raw(head2_nn.as_ptr()) };

        let vals: Vec<u8> = {
            let mut v = Vec::new();
            let mut cur = head2.next_as_ref();
            while let Some(n) = cur {
                v.push(*n.value().expect("data node"));
                cur = n.next_as_ref();
            }
            v
        };
        assert_eq!(vals, [10, 20, 30, 40]);

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // head2 is a fresh sentinel with no skip links.  The data-node
            // inter-node links are stale relative to the old head but they do
            // not affect the prev/next chain display.
            assert_snapshot!(
                head2.display()?,
                @"
                [03]: 00
                [02]: 00
                [01]: 00
                [->]: 00 -> 01 -> 02 -> 03 -> 04
                [<-]: 00 <- 01 <- 02 <- 03 <- 04

                [00|03] None
                [01|00] Some(10)
                [02|01] Some(20)
                [03|01] Some(30)
                [04|00] Some(40)
                "
            );
        }
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }

    /// The back-pointer that `set_head_next` stores in `first.prev` stays
    /// usable for writes after the call returns.
    ///
    /// `Node::pop` writes `head.next` through `first.prev`.  Under Tree
    /// Borrows that write is only allowed when `first.prev` carries the
    /// head's allocation provenance.
    #[test]
    fn miri_set_head_next_provenance() {
        // SAFETY: `Box::into_raw` never returns null.
        let head = unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(Node::<u8, MAX_LEVELS>::new(
                MAX_LEVELS,
            ))))
        };
        // SAFETY: `Box::into_raw` never returns null.
        let first = unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(Node::<u8, MAX_LEVELS>::with_value(
                1, 10,
            ))))
        };

        // SAFETY: `head` is a fresh sentinel with `next == None`; `first` is
        // a fresh detached node with `prev == None`.
        unsafe { Node::set_head_next(head, first) };

        // SAFETY: `first` is the only data node after `head`; nothing else
        // references it.
        let popped = unsafe { (*first.as_ptr()).pop() };
        assert_eq!(popped.value(), Some(&10));
        // SAFETY: `head` is live; no mutable reference to it exists.
        assert!(unsafe { head.as_ref() }.next_as_ref().is_none());

        // SAFETY: `head` came from `Box::into_raw` and has no successors.
        drop(unsafe { Box::from_raw(head.as_ptr()) });
    }

    // MARK: rebuild

    #[rstest]
    fn rebuild_produces_correct_links(
        skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>,
    ) -> Result<()> {
        let mut head = skiplist?;

        // Clear all skip links so that rebuild starts from a blank slate.
        for link in unsafe { head.as_mut() }.links_mut() {
            *link = None;
        }
        let mut cur = unsafe { (*head.as_ptr()).next_as_mut() };
        while let Some(n) = cur {
            for link in n.links_mut() {
                *link = None;
            }
            cur = unsafe { n.next_as_mut() };
        }

        let mut tail = None;
        // SAFETY: head is exclusively owned; all data nodes are live heap
        // allocations reachable through the next chain.
        unsafe { Node::filter_rebuild(head, &mut 0, &mut tail, |_| true, |_| {}) };

        assert_eq!(
            unsafe { tail.expect("non-empty").as_ref() }
                .value()
                .copied(),
            Some(40)
        );

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // Links rebuilt from node heights: same outcome as filter_rebuild_keep_all.
            assert_snapshot!(
                unsafe { head.as_ref() }.display()?,
                @"
                [03]: 00
                [02]: 00
                [01]: 00 -------> 02 -> 03
                [->]: 00 -> 01 -> 02 -> 03 -> 04
                [<-]: 00 <- 01 <- 02 <- 03 <- 04

                [00|03] None
                [01|00] Some(10)
                [02|01] Some(20)
                [03|01] Some(30)
                [04|00] Some(40)
                "
            );
        }
        unsafe { drop(Box::from_raw(head.as_ptr())) };
        Ok(())
    }
}
