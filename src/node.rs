//! Node implementation for the skip list.
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

use core::{
    fmt::{self, Debug, Write},
    iter,
    ptr::NonNull,
};
#[cfg(any(debug_assertions, test))]
use std::collections::HashMap;

use arrayvec::ArrayVec;

use crate::node::link::Link;

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
    links: ArrayVec<Option<Link<V, N>>, N>,
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
    ///   `<= N`. Passing a value greater than `N` panics in debug builds
    ///   and truncates silently in release builds.
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
            links: iter::repeat_with(|| None).take(max_levels).collect(),
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
            links: iter::repeat_with(|| None).take(height).collect(),
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
    #[expect(
        clippy::unnecessary_box_returns,
        reason = "pop() recovers the existing Box allocation created by insert_after(); \
                  returning Box<Self> signals heap ownership to callers"
    )]
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

    /// Drops all nodes following `self` and sets `self.next` to `None`.
    ///
    /// This is an `$O(k)$` iterative operation, where k is the number of nodes
    /// freed.  The caller is responsible for clearing any skip links that
    /// pointed to the freed nodes before or after this call.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no live references (including non-owning
    /// skip-link pointers that will be dereferenced) to the nodes being freed
    /// remain after this call.
    #[inline]
    pub(crate) unsafe fn truncate_next(&mut self) {
        // Same iterative pattern as `Drop for Node<V>`, applied to the tail
        // of the chain rather than the node itself.
        let mut current = self.next.take();
        while let Some(ptr) = current {
            // SAFETY: Every node reachable via `next` was heap-allocated via
            // `Box::new` and then stored via `Box::into_raw` in
            // `insert_after`.  We take ownership by removing the pointer from
            // the previous node's `next` before reconstructing the `Box`, so
            // no other owner exists.
            let mut boxed: Box<Self> = unsafe { Box::from_raw(ptr.as_ptr()) };
            current = boxed.next.take();
            // Drop `boxed` here.  Because `next` was already taken, the
            // node's own `Drop` impl will do nothing further.
            drop(boxed);
        }
    }

    /// Joins a head node to a tail node, creating a single sequence of nodes.
    ///
    /// This method takes ownership of `head` (consuming it) and splices the
    /// nodes that follow it onto `self` (the tail).  After the call, `self` is
    /// no longer a tail node, it now points to what was the first real node
    /// after `head`.
    ///
    /// # Safety
    ///
    /// This method does not alter the skip links of the nodes. The caller must
    /// update those links afterwards so that no dangling link pointers remain.
    #[inline]
    pub(crate) unsafe fn join(&mut self, mut head: Self) {
        debug_assert!(
            matches!(self.node_type(), NodeType::Tail),
            "Can only join to tail node"
        );
        debug_assert!(
            matches!(head.node_type(), NodeType::Head),
            "Can only join with head node"
        );

        // Copy the head's `next` pointer to the tail node, which transfers the
        // ownership of the `head.next` reference to `self.next`.
        self.next = head.next.take();

        // Update the `prev` pointer of the next node in the sequence to point
        // to the tail node.
        if let Some(mut head_next) = self.next {
            // SAFETY: The pointer can never be null, and the value is
            // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
            unsafe { head_next.as_mut() }.prev = Some(NonNull::from(&mut *self));
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
    /// `self_ptr` is stored directly in `node.prev`.  Storing `self_ptr`
    /// (rather than a new child reborrow of it) means that subsequent writes
    /// through any child of `self_ptr` are NOT foreign to `self_ptr`, so
    /// `self_ptr` is never transitioned to Disabled.
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

        // Box::into_raw preserves the allocation's root provenance without
        // creating an intermediate &mut reborrow layer.  Storing this
        // node_ptr in Link objects means that subsequent accesses to the
        // new node's memory go through children of node_ptr (not siblings),
        // so node_ptr is never Disabled by foreign writes under Tree Borrows.
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

    /// Wires `first` as the immediate successor of this head sentinel.
    ///
    /// After this call, `self.next = Some(first)` and
    /// `first.prev = Some(NonNull(self))`.
    ///
    /// # Safety
    ///
    /// - `self.next` must be `None` before this call.
    /// - `first` must point to a live, heap-allocated node with `prev == None`.
    #[inline]
    pub(crate) unsafe fn set_head_next(&mut self, first: NonNull<Self>) {
        debug_assert!(self.next.is_none(), "set_head_next: self.next must be None");
        let self_nn = NonNull::from(&mut *self);
        // SAFETY: `first` is a live, heap-allocated node with `prev == None`.
        // We are the exclusive owner of both `self` and `first` at this point.
        unsafe { &mut *first.as_ptr() }.prev = Some(self_nn);
        self.next = Some(first);
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
    /// Returns `(new_len, new_tail)` where `new_len` is the count of retained
    /// nodes and `new_tail` is the last retained node, or `None` if all nodes
    /// were removed.
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
        clippy::expect_used,
        reason = "Link::new(dist) returns Err only when dist == 0; \
                  dist = new_rank - pred_rank and new_rank > pred_rank \
                  whenever a predecessor is recorded, so dist >= 1 always"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "l < node_height <= max_levels = predecessors.len() = self.level(); \
                  l indexes both predecessors[l] and links_mut()[l] so a plain index \
                  loop is the clearest expression"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-pointer traversal, link clearing, optional pop, and link wiring \
                  all touch provably disjoint heap nodes; grouping them avoids \
                  unsafe-crossing raw-pointer variables"
    )]
    pub(crate) unsafe fn filter_rebuild<F, D>(
        head: NonNull<Self>,
        mut keep: F,
        mut on_drop: D,
    ) -> (usize, Option<NonNull<Self>>)
    where
        F: FnMut(*mut Self) -> bool,
        D: FnMut(Box<Self>),
    {
        let head_ptr: *mut Self = head.as_ptr();
        // SAFETY: head is a valid, exclusively-owned head sentinel per the
        // function's safety contract.
        let max_levels = unsafe { (*head_ptr).level() };
        let mut predecessors: ArrayVec<(*mut Self, usize), N> =
            iter::repeat_n((head_ptr, 0_usize), max_levels).collect();
        let mut new_rank: usize = 0;
        let mut new_tail: Option<NonNull<Self>> = None;

        // SAFETY: head_ptr and all nodes reachable via next are live,
        // exclusively-owned, heap-allocated Node<V> instances.  `keep` does
        // not structurally modify the chain before returning.
        unsafe {
            for link in (*head_ptr).links_mut() {
                *link = None;
            }

            let mut current_opt = (*head_ptr).next();
            while let Some(cur_nn) = current_opt {
                let cur: *mut Self = cur_nn.as_ptr();
                // Save successor before any structural mutation.
                let next_opt = (*cur).next();

                if keep(cur) {
                    new_rank = new_rank.saturating_add(1);
                    new_tail = Some(cur_nn);

                    // Clear this node's forward links; they will be re-wired.
                    for link in (*cur).links_mut() {
                        *link = None;
                    }

                    let height = (*cur).level();
                    for l in 0..height {
                        let (pred_ptr, pred_rank) = predecessors[l];
                        let dist = new_rank.saturating_sub(pred_rank);
                        (*pred_ptr).links_mut()[l] = Some(
                            Link::new(NonNull::new_unchecked(cur), dist)
                                .expect("dist >= 1 by construction"),
                        );
                        predecessors[l] = (cur, new_rank);
                    }
                } else {
                    on_drop((*cur).pop());
                }

                current_opt = next_opt;
            }
        }

        (new_rank, new_tail)
    }

    /// Rebuilds all skip links in a single `$O(n)$` forward pass, retaining every
    /// node.
    ///
    /// This is the keep-all specialisation of [`filter_rebuild`](Self::filter_rebuild).
    /// Returns the last data node as a [`NonNull`], or `None` if the chain is empty.
    ///
    /// # Safety
    ///
    /// Same as [`filter_rebuild`](Self::filter_rebuild): `head` must be the
    /// exclusively-owned head sentinel of a valid prev/next chain, with no
    /// other live references to any node in the chain.
    #[inline]
    pub(crate) unsafe fn rebuild(head: NonNull<Self>) -> Option<NonNull<Self>> {
        // SAFETY: forwarded from caller.
        let (_, tail) = unsafe { Self::filter_rebuild(head, |_| true, |_| {}) };
        tail
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
    fn ptr_index_map(&self) -> HashMap<NonNull<Self>, usize> {
        let mut hm = HashMap::new();
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
    fn display(&self) -> Result<String, fmt::Error> {
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

    // MARK: new

    #[test]
    fn node_new() {
        let node = Node::<(), MAX_LEVELS>::new(MAX_LEVELS);
        assert!(node.next.is_none());
        assert!(node.prev.is_none());
        assert_eq!(node.links.len(), MAX_LEVELS);
        assert!(node.value.is_none());
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
        assert!(matches!(detached_node.node_type(), NodeType::Detached));

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
        let (new_len, new_tail) =
            unsafe { Node::filter_rebuild(NonNull::from_mut(&mut *head), |_| true, |_| {}) };
        assert_eq!(new_len, 0);
        assert!(new_tail.is_none());
        assert!(head.next_as_ref().is_none());
    }

    #[rstest]
    fn filter_rebuild_keep_all(skiplist: Result<NonNull<Node<u8, MAX_LEVELS>>>) -> Result<()> {
        let head = skiplist?;
        let (new_len, new_tail) = unsafe { Node::filter_rebuild(head, |_| true, |_| {}) };

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
        let (new_len, new_tail) = unsafe {
            Node::filter_rebuild(
                head,
                |_| false,
                |mut b| dropped_vals.push(b.take_value().expect("data node")),
            )
        };

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
        let (new_len, new_tail) = unsafe {
            Node::filter_rebuild(
                head,
                |cur| {
                    let v = (*cur).value().copied();
                    v == Some(10) || v == Some(30)
                },
                |_| {},
            )
        };

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
            Node::filter_rebuild(head, |_| true, |_| {});
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
            v2.join(*head2);
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
            v2.join(*head2);
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

    // MARK: truncate_next

    #[test]
    fn truncate_next_on_middle_node() -> Result<()> {
        // Build head → v1(10) → v2(20) → v3(30), all height 0 (no skip links).
        let mut head = Box::new(Node::<u8, MAX_LEVELS>::new(MAX_LEVELS));
        unsafe {
            Node::insert_after(NonNull::from_mut(&mut *head), Node::with_value(0, 10_u8));
            Node::insert_after(
                NonNull::from_mut(head.next_as_mut().expect("v1")),
                Node::with_value(0, 20_u8),
            );
            Node::insert_after(
                NonNull::from_mut(head.next_as_mut().expect("v1").next_as_mut().expect("v2")),
                Node::with_value(0, 30_u8),
            );
        }

        // Truncate after v1; v2 and v3 are freed.
        // SAFETY: v1 is exclusively owned and no other references to v2/v3 exist.
        unsafe { head.next_as_mut().expect("v1").truncate_next() };

        let vals: Vec<u8> = {
            let mut v = Vec::new();
            let mut cur = head.next_as_ref();
            while let Some(n) = cur {
                v.push(*n.value().expect("data node"));
                cur = n.next_as_ref();
            }
            v
        };
        assert_eq!(vals, [10]);

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            assert_snapshot!(
                head.display()?,
                @"
                [03]: 00
                [02]: 00
                [01]: 00
                [->]: 00 -> 01
                [<-]: 00 <- 01

                [00|03] None
                [01|00] Some(10)
                "
            );
        }
        Ok(())
    }

    #[test]
    fn truncate_next_empties_list() -> Result<()> {
        // Build head → v1(10), height 0.
        let mut head = Box::new(Node::<u8, MAX_LEVELS>::new(MAX_LEVELS));
        unsafe { Node::insert_after(NonNull::from_mut(&mut *head), Node::with_value(0, 10_u8)) };

        // Truncate directly on head; v1 is freed.
        // SAFETY: head is exclusively owned and no other references to v1 exist.
        unsafe {
            head.truncate_next();
        }

        assert!(head.next_as_ref().is_none());

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            assert_snapshot!(
                head.display()?,
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
        let mut head2 = Box::new(Node::<u8, MAX_LEVELS>::new(MAX_LEVELS));
        // SAFETY: head2 has no current successor; first_nn is the exclusive
        // owner of the detached chain.
        unsafe { head2.set_head_next(first_nn) };

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

        // SAFETY: head is exclusively owned; all data nodes are live heap
        // allocations reachable through the next chain.
        let tail = unsafe { Node::rebuild(head) };

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
