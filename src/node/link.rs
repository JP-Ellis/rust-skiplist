//! Skip-link connecting two nodes in the list.
//!
//! Each node owns its immediate successor, but also carries zero or more
//! skip links to nodes further ahead. These links allow traversal to jump
//! over runs of nodes, giving the skip list its `$O(\log n)$` characteristics.

use std::{num::NonZeroUsize, ptr::NonNull};

use thiserror::Error;

use crate::node::Node;

/// A skip link between two nodes in the list.
///
/// Stores a raw pointer to the target node and the number of list positions
/// separating the link's owner from that target. The distance is always
/// at least 1 (enforced by [`NonZeroUsize`]).
///
/// This is a non-owning reference: dropping a `Link` does not drop or
/// deallocate the target node.
#[derive(Debug)]
pub(crate) struct Link<V> {
    /// The next node in the list.
    ///
    /// This is _not_ an owning reference and therefore when the link is
    /// dropped, the node should not be dropped.
    node: NonNull<Node<V>>,
    /// Distance to the next node.
    distance: NonZeroUsize,
}

impl<V> Link<V> {
    /// Create a new link.
    ///
    /// * `node` - Raw pointer to the target node. The caller must ensure the
    ///   pointer is valid for the lifetime of this link. The pointer is stored
    ///   as-is; no reborrow is performed, so the provenance of the supplied
    ///   pointer is preserved.
    /// * `distance` - Number of list positions to the target node. Must be
    ///   at least 1; passing 0 returns [`LinkError::InvalidDistance`].
    ///
    /// - `next`: The node that this link points to.
    /// - `distance`: The distance to the next node.
    pub(crate) fn new(node: &Node<V>, distance: usize) -> Result<Self, LinkError> {
        Ok(Link {
            node: NonNull::from(node),
            distance: NonZeroUsize::new(distance).ok_or(LinkError::InvalidDistance)?,
        })
    }

    /// Get a reference to the next node.
    pub(crate) fn next(&self) -> &Node<V> {
        // SAFETY: The pointer can never be null, and the value is
        // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
        unsafe { self.node.as_ref() }
    }

    /// Get a mutable reference to the next node.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the reference is not used elsewhere while
    /// this mutable reference is held. In most cases, this is not a problem as
    /// the caller will have a mutable reference to the source node.
    pub(crate) unsafe fn next_mut(&mut self) -> &mut Node<V> {
        // SAFETY: The pointer can never be null, and the value is
        // [convertible](https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion).
        unsafe { self.node.as_mut() }
    }

    /// Return the distance to the target node.
    pub(crate) fn distance(&self) -> NonZeroUsize {
        self.distance
    }

    /// Increment the distance by one.
    ///
    /// Call this when a new node is inserted within the span of this link,
    /// so that the recorded distance remains accurate.
    ///
    /// # Errors
    ///
    /// Returns [`LinkError::DistanceOverflow`] if the distance would exceed
    /// [`usize::MAX`].
    pub(crate) fn increment_distance(&mut self) -> Result<NonZeroUsize, LinkError> {
        self.distance = self
            .distance
            .checked_add(1)
            .ok_or(LinkError::DistanceOverflow)?;
        Ok(self.distance)
    }

    /// Decrement the distance by one.
    ///
    /// Call this when a node within the span of this link is removed, so
    /// that the recorded distance remains accurate.
    ///
    /// # Errors
    ///
    /// Returns [`LinkError::DistanceUnderflow`] if the result would be zero,
    /// which would violate the non-zero distance invariant.
    pub(crate) fn decrement_distance(&mut self) -> Result<NonZeroUsize, LinkError> {
        self.distance = self
            .distance
            .get()
            .checked_sub(1)
            .and_then(NonZeroUsize::new)
            .ok_or(LinkError::DistanceUnderflow)?;
        Ok(self.distance)
    }
}

/// Errors that can occur when working with links.
#[derive(Debug, Error)]
pub(crate) enum LinkError {
    /// The distance between two nodes has overflowed.
    #[error("distance overflow")]
    DistanceOverflow,
    /// The distance between two nodes has underflowed, becoming zero.
    #[error("distance underflow")]
    DistanceUnderflow,
    /// The distance between is invalid
    #[error("invalid distance")]
    InvalidDistance,
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroUsize, ptr};

    use anyhow::{Result, anyhow};
    use pretty_assertions::assert_eq;

    use super::{Link, LinkError};
    use crate::node::Node;

    #[test]
    fn link_new() -> Result<()> {
        let node: Node<i32> = Node::new(3);
        let link = Link::new(&node, 1)?;
        assert_eq!(ptr::from_ref(link.next()), ptr::from_ref(&node));
        assert_eq!(
            link.distance(),
            NonZeroUsize::new(1).ok_or(anyhow!("Invalid distance"))?
        );
        Ok(())
    }

    #[test]
    fn link_increment_distance() -> Result<()> {
        let node: Node<i32> = Node::new(3);
        let mut link = Link::new(&node, 1)?;
        assert_eq!(
            link.increment_distance()
                .expect("Distance increment failed"),
            NonZeroUsize::new(2).expect("NonZeroUsize::new failed")
        );
        Ok(())
    }

    #[test]
    fn link_decrement_distance() -> Result<()> {
        let node: Node<i32> = Node::new(3);
        let mut link = Link::new(&node, 2)?;
        assert_eq!(
            link.decrement_distance()
                .expect("Distance decrement failed"),
            NonZeroUsize::new(1).expect("NonZeroUsize::new failed")
        );
        Ok(())
    }

    #[test]
    fn link_decrement_distance_underflow() -> Result<()> {
        let node: Node<i32> = Node::new(3);
        let mut link = Link::new(&node, 1)?;
        assert!(matches!(
            link.decrement_distance(),
            Err(LinkError::DistanceUnderflow)
        ));
        Ok(())
    }

    #[test]
    fn link_increment_distance_overflow() -> Result<()> {
        let node: Node<i32> = Node::new(3);
        let mut link = Link::new(&node, usize::MAX)?;
        assert!(matches!(
            link.increment_distance(),
            Err(LinkError::DistanceOverflow)
        ));
        Ok(())
    }
}
