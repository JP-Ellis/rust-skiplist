//! Skip-link connecting two nodes in the list.
//!
//! Each node owns its immediate successor, but also carries zero or more
//! skip links to nodes further ahead. These links allow traversal to jump
//! over runs of nodes, giving the skip list its `$O(\log n)$` characteristics.

use core::{error::Error, fmt, num::NonZeroUsize, ptr::NonNull};

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
pub(crate) struct Link<V, const N: usize> {
    /// Raw, non-owning pointer to the target node.
    // Not an owning reference; the link owner must not free this node.
    node: NonNull<Node<V, N>>,
    /// Number of list positions between the link owner and the target.
    // Invariant: distance >= 1, enforced by NonZeroUsize.
    distance: NonZeroUsize,
}

impl<V, const N: usize> Link<V, N> {
    /// Create a new link pointing to `node` at the given `distance`.
    ///
    /// # Arguments
    ///
    /// * `node` - Raw pointer to the target node. The caller must ensure the
    ///   pointer is valid for the lifetime of this link. The pointer is stored
    ///   as-is; no reborrow is performed, so the provenance of the supplied
    ///   pointer is preserved.
    /// * `distance` - Number of list positions to the target node. Must be
    ///   at least 1; passing 0 returns [`LinkError::InvalidDistance`].
    ///
    /// # Errors
    ///
    /// Returns [`LinkError::InvalidDistance`] if `distance` is zero.
    pub(crate) fn new(node: NonNull<Node<V, N>>, distance: usize) -> Result<Self, LinkError> {
        Ok(Link {
            node,
            distance: NonZeroUsize::new(distance).ok_or(LinkError::InvalidDistance)?,
        })
    }

    /// Return the raw pointer to the target node.
    ///
    /// The caller is responsible for choosing the appropriate access mode:
    ///
    /// - Read-only: `unsafe { link.node().as_ref() }`
    /// - Mutable: `unsafe { link.node().as_mut() }`
    /// - Raw pointer arithmetic: `link.node().as_ptr()`
    // Returning NonNull rather than &Node avoids creating a shared reborrow
    // here, which under Tree Borrows would downgrade the pointer's Reserved
    // tag to Frozen and break callers that need mutable access (e.g., the
    // mutable visitors). Callers must choose the reborrow that matches their
    // actual access pattern.
    pub(crate) fn node(&self) -> NonNull<Node<V, N>> {
        self.node
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
#[derive(Debug)]
pub(crate) enum LinkError {
    /// The distance between two nodes has overflowed.
    DistanceOverflow,
    /// The distance between two nodes has underflowed, becoming zero.
    DistanceUnderflow,
    /// The supplied distance is invalid (zero is not permitted).
    InvalidDistance,
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DistanceOverflow => write!(f, "distance overflow"),
            Self::DistanceUnderflow => write!(f, "distance underflow"),
            Self::InvalidDistance => write!(f, "invalid distance"),
        }
    }
}

impl Error for LinkError {}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroUsize, ptr::NonNull};

    use anyhow::{Result, anyhow};
    use pretty_assertions::assert_eq;

    use super::{Link, LinkError};
    use crate::node::Node;

    #[test]
    fn link_new() -> Result<()> {
        let node: Node<i32, 3> = Node::new(3);
        let link = Link::new(NonNull::from(&node), 1)?;
        assert_eq!(link.node(), NonNull::from(&node));
        assert_eq!(
            link.distance(),
            NonZeroUsize::new(1).ok_or(anyhow!("Invalid distance"))?
        );
        Ok(())
    }

    #[test]
    fn link_increment_distance() -> Result<()> {
        let node: Node<i32, 3> = Node::new(3);
        let mut link = Link::new(NonNull::from(&node), 1)?;
        assert_eq!(
            link.increment_distance()?,
            NonZeroUsize::new(2).expect("NonZeroUsize::new failed")
        );
        Ok(())
    }

    #[test]
    fn link_decrement_distance() -> Result<()> {
        let node: Node<i32, 3> = Node::new(3);
        let mut link = Link::new(NonNull::from(&node), 2)?;
        assert_eq!(
            link.decrement_distance()?,
            NonZeroUsize::new(1).expect("NonZeroUsize::new failed")
        );
        Ok(())
    }

    #[test]
    fn link_decrement_distance_underflow() -> Result<()> {
        let node: Node<i32, 3> = Node::new(3);
        let mut link = Link::new(NonNull::from(&node), 1)?;
        assert!(matches!(
            link.decrement_distance(),
            Err(LinkError::DistanceUnderflow)
        ));
        Ok(())
    }

    #[test]
    fn link_increment_distance_overflow() -> Result<()> {
        let node: Node<i32, 3> = Node::new(3);
        let mut link = Link::new(NonNull::from(&node), usize::MAX)?;
        assert!(matches!(
            link.increment_distance(),
            Err(LinkError::DistanceOverflow)
        ));
        Ok(())
    }
}
