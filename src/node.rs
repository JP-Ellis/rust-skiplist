//! Node implementation for the skip list.
//!
//! Underlying all the operations of the skip list, skip map and ordered skip
//! list is the node. Each node owns the next node, and has a link to the
//! previous node. The node also has a level, which corresponds to how 'high'
//! the node reaches.
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
//! The skip-list's benefit comes from the ability to skip over nodes at higher
//! levels. For example, to move from node 1 to node 10, the above list would be
//! traversed in the order head -> 6 -> 8 -> 9 -> 10.
//!
//! # Consideration
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
//! As a result of (2), it is generally safer to get a mutable reference to the
//! next node through [`next_mut`][Node::next_mut], as that requires a mutable
//! reference to the current node.
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

#![expect(dead_code, reason = "Temporary until put to use")]

use core::{
    fmt::{self, Debug, Write},
    iter,
    ptr::NonNull,
};
#[cfg(debug_assertions)]
use std::collections::HashMap;

use crate::node::link::Link;

mod link;

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
pub(crate) struct Node<V> {
    /// Owning reference to the next node.
    next: Option<NonNull<Self>>,
    /// Non-owning reference to the previous node.
    prev: Option<NonNull<Self>>,
    /// Links to subsequent nodes.
    links: Vec<Option<Link<V>>>,
    /// The value of the node.
    value: Option<V>,
}

impl<V> Node<V> {
    /// Create a new node.
    ///
    /// # Parameters
    ///
    /// - `max_levels`: The maximum number of levels in the skip list.
    #[inline]
    #[must_use]
    pub(crate) fn new(max_levels: usize) -> Self {
        Self {
            next: None,
            prev: None,
            links: iter::repeat_with(|| None).take(max_levels).collect(),
            value: None,
        }
    }

    /// Get the node's level.
    ///
    /// The level of a node corresponds to how many links it has (excluding the
    /// `prev`/`next` links).
    #[inline]
    #[must_use]
    pub(crate) fn level(&self) -> usize {
        self.links.len()
    }

    /// Get a reference to the next node.
    #[inline]
    #[must_use]
    pub(crate) fn next(&self) -> Option<&Self> {
        // SAFETY: The pointer can never be null, and the value is
        // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
        self.next.map(|ptr| unsafe { ptr.as_ref() })
    }

    /// Get a mutable reference to the next node.
    ///
    /// # Safety
    ///
    /// As `self`, which is borrowed mutably, is the owner of the next node, it
    /// should generally be safe to get a mutable reference to the next node
    /// provided that no other references are using the next node. This
    /// requirement _should_ be enforced by the fact that `self` is borrowed
    /// mutably.
    #[inline]
    #[must_use]
    pub(crate) fn next_mut(&mut self) -> Option<&mut Self> {
        // SAFETY: The pointer can never be null, and the value is
        // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
        self.next.map(|mut ptr| unsafe { ptr.as_mut() })
    }

    /// Get a reference to the previous node.
    #[inline]
    #[must_use]
    pub(crate) fn prev(&self) -> Option<&Self> {
        // SAFETY: The pointer can never be null, and the value is
        // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
        self.prev.map(|ptr| unsafe { ptr.as_ref() })
    }

    /// Get a mutable reference to the previous node.
    ///
    /// # Safety
    ///
    /// Unlike [`next_mut`][Self::next_mut], the current node does _not_ own the
    /// previous node. The fact that `self` is borrowed mutably does not imply
    /// that the previous node is not being used elsewhere. As a result, the
    /// caller must ensure that the previous node is not being used elsewhere.
    #[inline]
    #[must_use]
    pub(crate) unsafe fn prev_mut(&mut self) -> Option<&mut Self> {
        // SAFETY: The pointer can never be null, and the value is
        // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
        self.prev.map(|mut ptr| unsafe { ptr.as_mut() })
    }

    /// Get a reference to the value.
    #[inline]
    #[must_use]
    pub(crate) fn value(&self) -> Option<&V> {
        self.value.as_ref()
    }

    /// Get a mutable reference to the value.
    #[inline]
    #[must_use]
    pub(crate) fn value_mut(&mut self) -> Option<&mut V> {
        self.value.as_mut()
    }

    /// Get a reference to the list of links.
    ///
    /// This is used to get a list of links to the next nodes at each level.
    #[inline]
    #[must_use]
    pub(crate) fn links(&self) -> &Vec<Option<Link<V>>> {
        &self.links
    }

    /// Get a mutable reference to the list of links.
    #[inline]
    #[must_use]
    pub(crate) fn links_mut(&mut self) -> &mut Vec<Option<Link<V>>> {
        &mut self.links
    }

    /// Identify the type of node.
    ///
    /// This is used to determine the type of node in the skip list. The head
    /// and tail nodes must not have a value, while regular nodes must have a
    /// value. Additionally, the head node must have a `next` pointer and no
    /// `prev` pointer, and the tail node must have a `prev` pointer and no
    /// `next` pointer.
    #[inline]
    #[must_use]
    pub(crate) fn node_type(&self) -> NodeType {
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
    /// This method does not alter the links of the node, or surrounding nodes.
    /// As a result, the caller must ensure that
    #[inline]
    #[expect(clippy::unnecessary_box_returns, reason = "Waiting for stabilization")]
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

    /// Join two sequences of nodes.
    ///
    /// Joins a head node to a tail node, creating a single sequence of nodes,
    /// returning the head node.
    ///
    /// This method takes ownership of the new sequence of nodes, and joins it
    /// to the tail node.
    ///
    /// # Safety
    ///
    /// This method re-allocates the node on the heap. As a result, any links
    /// pointing to the node being joined will be invalidated.
    ///
    /// This method does not alter the links of the node being joined, or
    /// surrounding nodes. As a result, while traversing the list to find a
    /// specific node, it is important to keep track of the links to the node
    /// and links over the node.
    #[inline]
    pub(crate) unsafe fn join(&mut self, mut head: Self) -> Self {
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

        head
    }

    /// Insert a new node after the current node.
    ///
    /// The new node to be inserted must be a standalone node and not part of a
    /// list (i.e., it must not have a `prev` or `next` pointer).
    ///
    /// This method takes ownership of the node to insert, and inserts it after
    /// the current node. It modifies the current node, the new node, and the
    /// node following the current node so that their `next` and `prev` pointers
    /// are updated to point to each other.
    ///
    /// # Safety
    ///
    /// This method does not alter the links of the node being inserted, or
    /// surrounding nodes. As a result, while traversing the list to find a
    /// specific node, it is important to keep track of the links to the node and
    /// links over the node.
    #[inline]
    pub(crate) unsafe fn insert_after(&mut self, mut node: Self) {
        debug_assert!(
            matches!(node.node_type(), NodeType::Detached),
            "Can only insert detached nodes."
        );

        node.prev = Some(NonNull::from(&mut *self));
        node.next = self.next;

        // Move ownership of the new node through box-and-leak to ensure that the
        // node is allocated on the heap and not deallocated when the function
        // returns.
        let node_ptr = NonNull::from(Box::leak(Box::new(node)));

        // If self has a 'next' node, update its 'prev' pointer to point to the
        // new node.
        if let Some(next_ptr) = self.next.as_mut() {
            // SAFETY: The pointer can never be null, and the value is
            // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
            unsafe { next_ptr.as_mut() }.prev = Some(node_ptr);
        }

        // Finally, update self's 'next' pointer to point to the new node
        self.next = Some(node_ptr);
    }

    /// Generate a map of pointers to node indices.
    #[inline]
    #[allow(clippy::allow_attributes, dead_code, reason = "Used for debugging")]
    fn ptr_index_map(&self) -> HashMap<NonNull<Self>, usize> {
        let mut map = HashMap::new();
        let mut current = self;
        let mut index = 0_usize;
        map.insert(NonNull::from(self), index);
        while let Some(node) = current.next() {
            index = index.saturating_add(1);
            map.insert(NonNull::from(node), index);
            current = node;
        }
        map
    }
}

impl<V: Debug> Node<V> {
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
    #[allow(clippy::allow_attributes, dead_code, reason = "Used for debugging")]
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
    #[allow(clippy::allow_attributes, dead_code, reason = "Used for debugging")]
    pub(crate) fn display_links(&self) -> Result<String, fmt::Error> {
        let hm = self.ptr_index_map();
        let mut output = String::new();

        // Display the links
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
                    current = link.node();
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

            if let Some(next) = current.next() {
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

            if let Some(prev) = current.prev() {
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
    #[allow(clippy::allow_attributes, dead_code, reason = "Used for debugging")]
    #[expect(clippy::use_debug, reason = "This is an internal method for debugging")]
    pub(crate) fn display_values(&self) -> Result<String, fmt::Error> {
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
            if let Some(node) = current.next() {
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
    #[expect(clippy::use_debug, reason = "This is an internal method for debugging")]
    pub(crate) fn display_ptrs(&self) -> Result<String, fmt::Error> {
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
            if let Some(node) = current.next() {
                current = node;
                index = index.saturating_add(1);
            } else {
                break;
            }
        }
        Ok(output)
    }
}

impl<V: Debug> Debug for Node<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("next", &self.next)
            .field("prev", &self.prev)
            .field("links", &self.links)
            .field("value", &self.value)
            .finish()
    }
}

// impl<V> Drop for Node<V> {
//     /// Drop a node
//     ///
//     /// We cannot simply drop the node, as that would result in the next node
//     /// leaking. We need to drop the next node (if it exists) before dropping
//     /// the current node.
//     fn drop(&mut self) {
//         while self
//             .next_mut()
//             .map(|node|
//                 // SAFETY: Some links will be left dangling during the
//                 // drop, but once finished, all links are also dropped.
//                 unsafe { node.pop() })
//             .is_some()
//         {}
//     }
// }

#[expect(clippy::undocumented_unsafe_blocks, reason = "Test code")]
#[cfg(test)]
mod tests {
    use anyhow::Result;
    use insta::assert_snapshot;
    use pretty_assertions::assert_eq;
    use rstest::{fixture, rstest};

    use crate::node::{Node, NodeType, link::Link};

    const MAX_LEVELS: usize = 2;

    #[test]
    fn node_new() {
        let node = Node::<()>::new(MAX_LEVELS);
        assert!(node.next.is_none());
        assert!(node.prev.is_none());
        assert_eq!(node.links.len(), MAX_LEVELS);
        assert!(node.value.is_none());
    }

    #[test]
    fn new_node_properties() {
        let mut node = Node::<()>::new(MAX_LEVELS);

        assert!(node.next().is_none());
        assert!(node.next_mut().is_none());

        assert!(node.prev().is_none());
        assert!(unsafe { node.prev_mut() }.is_none());

        assert!(node.value().is_none());
        assert!(node.value_mut().is_none());
    }

    /// Build a simple skiplist.
    ///
    /// The values are 1, 2, 3, 4; and the links are as follows:
    ///
    /// head -------------> 03
    /// head -------> 02 -> 03 -> 04
    /// head -> 01 -> 02 -> 03 -> 04
    #[fixture]
    fn minimal_skiplist() -> Result<Box<Node<u8>>> {
        let mut head = Box::new(Node::new(MAX_LEVELS));
        let mut v1 = Node::new(0);
        let mut v2 = Node::new(1);
        let mut v3 = Node::new(1);
        let mut v4 = Node::new(0);

        // Internal values
        v1.value = Some(1);
        v2.value = Some(2);
        v3.value = Some(3);
        v4.value = Some(4);

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

        // Build higher level links:
        //
        // head -------------> v3
        // head -------> v2 -> v3 -> v4
        // head -> v1 -> v2 -> v3 -> v4

        let head_v3: Link<_>;
        let head_v2: Link<_>;
        let v2_v3: Link<_>;
        let v3_v4: Link<_>;
        {
            let v1_ref = head.next().expect("v1 not found");
            let v2_ref = v1_ref.next().expect("v2 not found");
            let v3_ref = v2_ref.next().expect("v3 not found");
            let v4_ref = v3_ref.next().expect("tail not found");
            head_v3 = Link::new(v3_ref, 3)?;
            head_v2 = Link::new(v2_ref, 2)?;
            v2_v3 = Link::new(v3_ref, 1)?;
            v3_v4 = Link::new(v4_ref, 1)?;
        }

        #[expect(clippy::indexing_slicing, reason = "Serves as an additional assertion")]
        {
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

    #[cfg_attr(miri, ignore)] // Insta does not work with miri
    #[rstest]
    fn node_display(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = minimal_skiplist?;

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            assert_snapshot!(
                head.display()?,
                @"
                [02]: 00 -------------> 03
                [01]: 00 -------> 02 -> 03 -> 04
                [->]: 00 -> 01 -> 02 -> 03 -> 04
                [<-]: 00 <- 01 <- 02 <- 03 <- 04

                [00|02] None
                [01|00] Some(1)
                [02|01] Some(2)
                [03|01] Some(3)
                [04|00] Some(4)
                "
            );
        }

        Ok(())
    }

    #[rstest]
    fn node_properties(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let head = minimal_skiplist?;

        matches!(head.node_type(), NodeType::Head);
        assert_eq!(head.level(), 2);

        let mut node = head.next().expect("v1 not found");
        matches!(node.node_type(), NodeType::Body);
        assert_eq!(node.level(), 0);

        node = node.next().expect("v2 not found");
        matches!(node.node_type(), NodeType::Body);
        assert_eq!(node.level(), 1);

        node = node.next().expect("v3 not found");
        matches!(node.node_type(), NodeType::Body);
        assert_eq!(node.level(), 1);

        node = node.next().expect("v4 not found");
        matches!(node.node_type(), NodeType::Tail);
        assert_eq!(node.level(), 0);

        Ok(())
    }

    #[cfg_attr(miri, ignore)] // Insta does not work with miri
    #[rstest]
    fn pop_node(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = minimal_skiplist?;

        let v1 = head.next_mut().expect("v1 not found");
        let v2 = v1.next_mut().expect("v2 not found");
        let detached_node = unsafe { v2.pop() };

        assert_eq!(detached_node.value, Some(2));
        assert!(matches!(detached_node.node_type(), NodeType::Detached));

        // Insta is incompatible with Miri
        if cfg!(miri) {
            head.display()?;
        } else {
            // Note: The sequence of values should be valid. It is fine for the
            // links to be invalid as the node has been popped without updating the
            // links.
            assert_snapshot!(
                head.display()?,
                @"
                [02]: 00 -------------> 02
                [01]: 00 -------> ??
                [->]: 00 -> 01 -> 02 -> 03
                [<-]: 00 <- 01 <- 02 <- 03

                [00|02] None
                [01|00] Some(1)
                [02|01] Some(3)
                [03|00] Some(4)
                "
            );
        }

        Ok(())
    }

    #[cfg_attr(miri, ignore)] // Insta does not work with miri
    #[rstest]
    fn insert_after_head_node(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = minimal_skiplist?;
        let mut new_node = Node::new(99);
        new_node.value = Some(100);

        unsafe {
            head.insert_after(new_node);
        }

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // Note: The sequence of values should be valid. The links are not
            // updated and therefore may result in UB when displayed.
            assert_snapshot!(
                head.display_values()?,
                @"
                [00|02] None
                [01|99] Some(100)
                [02|00] Some(1)
                [03|01] Some(2)
                [04|01] Some(3)
                [05|00] Some(4)
                "
            );
        }

        Ok(())
    }

    #[cfg_attr(miri, ignore)] // Insta does not work with miri
    #[rstest]
    fn insert_after_body_node(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = minimal_skiplist?;
        let mut new_node = Node::new(99);
        new_node.value = Some(100);

        // SAFETY: `head` is a valid node with `v1` as its next.
        let v1 = unsafe { head.next_mut() }.expect("v1 not found");
        // SAFETY: `v1` has exclusive access and no other references to its successor exist.
        unsafe { v1.insert_after(new_node) };

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // Note: The sequence of values should be valid. The links are not
            // updated and therefore may result in UB when displayed.
            assert_snapshot!(
                head.display_values()?,
                @"
                [00|02] None
                [01|00] Some(1)
                [02|99] Some(100)
                [03|01] Some(2)
                [04|01] Some(3)
                [05|00] Some(4)
                "
            );
        }

        Ok(())
    }

    // MARK: filter_rebuild

    #[rstest]
    fn filter_rebuild_empty_list() {
        let mut head = Box::new(Node::<u8>::new(MAX_LEVELS));
        let (new_len, new_tail) = unsafe { head.filter_rebuild(|_| true, |_| {}) };
        assert_eq!(new_len, 0);
        assert!(new_tail.is_none());
        assert!(head.next().is_none());
    }

    #[rstest]
    fn filter_rebuild_keep_all(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = minimal_skiplist?;
        let (new_len, new_tail) = unsafe { head.filter_rebuild(|_| true, |_| {}) };

        assert_eq!(new_len, 4);
        let vals: Vec<u8> = {
            let mut v = Vec::new();
            let mut cur = head.next();
            while let Some(n) = cur {
                v.push(*n.value().expect("data node"));
                cur = n.next();
            }
            v
        };
        assert_eq!(vals, [1, 2, 3, 4]);
        let tail_val = unsafe { new_tail.expect("tail exists").as_ref() }
            .value()
            .copied();
        assert_eq!(tail_val, Some(4));
        Ok(())
    }

    #[rstest]
    fn filter_rebuild_keep_none(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut dropped_vals: Vec<u8> = Vec::new();
        let mut head = minimal_skiplist?;
        let (new_len, new_tail) = unsafe {
            head.filter_rebuild(
                |_| false,
                |mut b| dropped_vals.push(b.take_value().expect("data node")),
            )
        };

        assert_eq!(new_len, 0);
        assert!(new_tail.is_none());
        assert!(head.next().is_none());
        // on_drop called in traversal order.
        assert_eq!(dropped_vals, [1, 2, 3, 4]);
        Ok(())
    }

    #[rstest]
    fn filter_rebuild_keep_first_and_third(
        minimal_skiplist: Result<Box<Node<u8>>>,
    ) -> Result<()> {
        // Keep v1 (value 1) and v3 (value 3); drop v2 and v4.
        let mut head = minimal_skiplist?;
        let (new_len, new_tail) = unsafe {
            head.filter_rebuild(
                |cur| {
                    let v = (*cur).value().copied();
                    v == Some(1) || v == Some(3)
                },
                |_| {},
            )
        };

        assert_eq!(new_len, 2);
        let vals: Vec<u8> = {
            let mut v = Vec::new();
            let mut cur = head.next();
            while let Some(n) = cur {
                v.push(*n.value().expect("data node"));
                cur = n.next();
            }
            v
        };
        assert_eq!(vals, [1, 3]);
        let tail_val = unsafe { new_tail.expect("tail exists").as_ref() }
            .value()
            .copied();
        assert_eq!(tail_val, Some(3));
        Ok(())
    }

    #[rstest]
    fn filter_rebuild_on_drop_receives_correct_values(
        minimal_skiplist: Result<Box<Node<u8>>>,
    ) -> Result<()> {
        let mut dropped: Vec<u8> = Vec::new();
        let mut head = minimal_skiplist?;
        // Keep v2 and v4; drop v1 and v3.
        unsafe {
            head.filter_rebuild(
                |cur| {
                    let v = (*cur).value().copied();
                    v == Some(2) || v == Some(4)
                },
                |mut b| dropped.push(b.take_value().expect("data node")),
            );
        }

        // Dropped values must arrive in traversal order: v1 then v3.
        assert_eq!(dropped, [1, 3]);
        Ok(())
    }

    /// After dropping nodes with no skip-link slots, all head links must be
    /// `None` because no retained node can anchor a skip link.
    #[rstest]
    fn filter_rebuild_links_consistent_after_partial_keep(
        minimal_skiplist: Result<Box<Node<u8>>>,
    ) -> Result<()> {
        let mut head = minimal_skiplist?;
        // Drop v2 and v3 (both height 1); keep v1 (height 0) and v4 (height 0).
        unsafe {
            head.filter_rebuild(
                |cur| {
                    let v = (*cur).value().copied();
                    v != Some(2) && v != Some(3)
                },
                |_| {},
            );
        }

        // Chain must be head -> v1 -> v4.
        let vals: Vec<u8> = {
            let mut v = Vec::new();
            let mut cur = head.next();
            while let Some(n) = cur {
                v.push(*n.value().expect("data node"));
                cur = n.next();
            }
            v
        };
        assert_eq!(vals, [1, 4]);
        // v1 and v4 both have height 0, so all head skip links must be None.
        for link in head.links() {
            assert!(link.is_none(), "head skip link should be None");
        }
        Ok(())
    }

    /// Rebuilding skip links over a keep-all pass must produce correct distances.
    #[rstest]
    fn filter_rebuild_keep_all_links_rebuilt(
        minimal_skiplist: Result<Box<Node<u8>>>,
    ) -> Result<()> {
        // Fixture node heights: v1=0, v2=1, v3=1, v4=0.
        //
        // After a keep-all rebuild:
        //   head.links[0] → v2 (distance 2)  — head → v1 (height 0, no link) → v2 (height 1)
        //   head.links[1] → None             — no height-2 node exists
        //   v2.links[0]   → v3 (distance 1)
        //   v3.links[0]   → None             — v4 has height 0 so nothing wires into v3.links[0]
        let mut head = minimal_skiplist?;
        unsafe {
            head.filter_rebuild(|_| true, |_| {});
        }

        let link0 = head.links()[0].as_ref().expect("head.links[0] must be Some");
        assert_eq!(link0.node().value().copied(), Some(2));
        assert_eq!(link0.distance().get(), 2);

        assert!(head.links()[1].is_none(), "head.links[1] must be None");

        let v2 = head.next().expect("v1").next().expect("v2");
        let v2_link0 = v2.links()[0].as_ref().expect("v2.links[0] must be Some");
        assert_eq!(v2_link0.node().value().copied(), Some(3));
        assert_eq!(v2_link0.distance().get(), 1);

        let v3 = v2.next().expect("v3");
        assert!(v3.links()[0].is_none(), "v3.links[0] must be None");
        Ok(())
    }

    #[cfg_attr(miri, ignore)] // Insta does not work with miri
    #[rstest]
    fn insert_after_tail_node(minimal_skiplist: Result<Box<Node<u8>>>) -> Result<()> {
        let mut head = minimal_skiplist?;
        let mut new_node = Node::new(99);
        new_node.value = Some(100);

        // SAFETY: Each node is valid with the next node as its successor.
        let v1 = unsafe { head.next_mut() }.expect("v1 not found");
        let v2 = unsafe { v1.next_mut() }.expect("v2 not found");
        let v3 = unsafe { v2.next_mut() }.expect("v3 not found");
        let v4 = unsafe { v3.next_mut() }.expect("v4 not found");
        // SAFETY: `v4` has exclusive access and no other references to its successor exist.
        unsafe { v4.insert_after(new_node) };

        // Insta is incompatible with Miri
        if !cfg!(miri) {
            // Note: The sequence of values should be valid. The links are not
            // updated and therefore may result in UB when displayed.
            assert_snapshot!(
                head.display_values()?,
                @"
                [00|02] None
                [01|00] Some(1)
                [02|01] Some(2)
                [03|01] Some(3)
                [04|00] Some(4)
                [05|99] Some(100)
                "
            );
        }

        Ok(())
    }
}
