use std::{fmt, iter, ptr};

pub enum NextNotFound {
    NoMoreElements,
    NoSuchElement,
}

// ////////////////////////////////////////////////////////////////////////////
// SkipNode
// ////////////////////////////////////////////////////////////////////////////

/// SkipNodes are make up the SkipList.  The SkipList owns the first head-node
/// (which has no value) and each node has ownership of the next node through
/// `next`.
///
/// The node has a `level` which corresponds to how 'high' the node reaches, and
/// this should be equal to the length of the vector of links.  There is a
/// corresponding vector of link lengths which is used to reach a particular
/// index.
///
/// Lastly, each node contains a link to the immediately previous node in case
/// one needs to parse the list backwards.
///
/// In cases where the value is not applicable, `None` should be used.  In
/// particular, as there is no tail node, the value of `next` in the last node
/// should be `None`.
#[derive(Clone, Debug)]
pub struct SkipNode<V> {
    // key and value should never be None, with the sole exception being the
    // head node.
    pub value: Option<V>,
    // how high the node reaches.  This should be equal to the vector length.
    pub level: usize,
    // The immediately next element (and owns that next node).
    pub next: Option<Box<SkipNode<V>>>,
    // The immediately previous element.
    pub prev: Option<*mut SkipNode<V>>,
    // Vector of links to the next node at the respective level.  This vector
    // *must* be of length `self.level + 1`.  links[0] stores a pointer to the
    // same node as next.
    pub links: Vec<Option<*mut SkipNode<V>>>,
    // The corresponding length of each link
    pub links_len: Vec<usize>,
}

// ///////////////////////////////////////////////
// Inherent methods
// ///////////////////////////////////////////////

impl<V> SkipNode<V> {
    /// Create a new head node.
    pub fn head(total_levels: usize) -> Self {
        SkipNode {
            value: None,
            level: total_levels - 1,
            next: None,
            prev: None,
            links: iter::repeat(None).take(total_levels).collect(),
            links_len: iter::repeat(0).take(total_levels).collect(),
        }
    }

    /// Create a new SkipNode with the given value.  The values of `prev` and
    /// `next` will all be `None` and have to be adjusted.
    pub fn new(value: V, level: usize) -> Self {
        SkipNode {
            value: Some(value),
            level,
            next: None,
            prev: None,
            links: iter::repeat(None).take(level + 1).collect(),
            links_len: iter::repeat(0).take(level + 1).collect(),
        }
    }

    /// Consumes the node returning the value it contains.
    pub fn into_inner(mut self) -> Option<V> {
        self.value.take()
    }

    /// Returns `true` is the node is a head-node.
    pub fn is_head(&self) -> bool {
        self.prev.is_none()
    }

    /// Try to move to next nth node at specified level.
    /// If it's impossible, then move as far as possible.
    /// Returns a reference to the new node and the distance travelled.
    pub fn advance_len(&self, level: usize, mut max_distance: usize) -> (&Self, usize) {
        self.advance_while(level, move |current_node, _| {
            let travelled = current_node.links_len[level];
            if travelled <= max_distance {
                max_distance -= travelled;
                return true;
            } else {
                return false;
            }
        })
    }

    /// Try to move to next nth node at specified level.
    /// If it's impossible, then move as far as possible.
    /// Returns a mutable reference to the new node and the distance travelled.
    pub fn advance_len_mut(&mut self, level: usize, mut max_distance: usize) -> (&mut Self, usize) {
        self.advance_while_mut(level, move |current_node, _| {
            let travelled = current_node.links_len[level];
            if travelled <= max_distance {
                max_distance -= travelled;
                return true;
            } else {
                return false;
            }
        })
    }

    pub fn advance_while(
        &self,
        level: usize,
        mut pred: impl FnMut(&Self, &Self) -> bool,
    ) -> (&Self, usize) {
        let mut current = self;
        let mut travelled = 0;
        loop {
            match current.advance_if(level, &mut pred) {
                Ok((node, steps)) => {
                    current = node;
                    travelled += steps;
                }
                Err(node) => return (node, travelled),
            }
        }
    }

    pub fn advance_while_mut(
        &mut self,
        level: usize,
        mut pred: impl FnMut(&Self, &Self) -> bool,
    ) -> (&mut Self, usize) {
        let mut current = self;
        let mut travelled = 0;
        loop {
            match current.advance_if_mut(level, &mut pred) {
                Ok((node, steps)) => {
                    current = node;
                    travelled += steps;
                }
                Err(node) => return (node, travelled),
            }
        }
    }

    // Due to Rust lifetime semantics, the lifetime of result is the same as self.
    // Sometimes Rust cannot determine the result is unused, e.g. in a loop.
    // As a result, self might be  borrowed forever and caller cannot return that value
    // if they call this function in a loop.
    // Therefore this function always return self when it fails to advance.
    pub fn advance_if_mut<'a>(
        &'a mut self,
        level: usize,
        predicate: impl FnOnce(&Self, &Self) -> bool,
    ) -> Result<(&'a mut Self, usize), &'a mut Self> {
        let next = unsafe { self.links[level].and_then(|ptr| ptr.as_mut()) };
        match next {
            Some(next) if predicate(self, next) => Ok((next, self.links_len[level])),
            _ => Err(self),
        }
    }

    pub fn advance_if<'a>(
        &'a self,
        level: usize,
        predicate: impl FnOnce(&Self, &Self) -> bool,
    ) -> Result<(&'a Self, usize), &'a Self> {
        let next = unsafe { self.links[level].and_then(|ptr| ptr.as_ref()) };
        match next {
            Some(next) if predicate(self, next) => Ok((next, self.links_len[level])),
            _ => Err(self),
        }
    }
}

impl<V> Drop for SkipNode<V> {
    fn drop(&mut self) {
        while let Some(mut node) = self.next.take() {
            self.next = node.next.take();
        }
    }
}

// ///////////////////////////////////////////////
// Trait implementation
// ///////////////////////////////////////////////

impl<V> fmt::Display for SkipNode<V>
where
    V: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(ref v) = self.value {
            write!(f, "{}", v)
        } else {
            Ok(())
        }
    }
}

// /////////////////////////////////
// Iterators
// /////////////////////////////////
// Since Iterators (currently) only pop from front and back,
// they can be shared by some data structures.
// There's no need for a dummy head (that contains no value) in the iterator.
// so the members are named first and last instaed of head/end to avoid confusion.

/// Consuming iterator.  
pub struct IntoIter<T> {
    pub(crate) first: Option<Box<SkipNode<T>>>,
    pub(crate) last: *mut SkipNode<T>,
    pub(crate) size: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let mut popped_node = self.first.take()?;
        self.size -= 1;
        self.first = popped_node.next.take().map(|mut node| {
            node.prev = None;
            node
        });
        if self.first.is_none() {
            self.last = ptr::null_mut();
        }
        popped_node.into_inner()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.first.is_none() {
            return None;
        }
        assert!(!self.last.is_null());
        unsafe {
            let new_last = (*self.last).prev;
            let popped_node = match new_last {
                Some(new_last) => {
                    self.last = new_last;
                    (*new_last).next.take().unwrap()
                }
                None => self.first.take().unwrap(),
            };
            self.size -= 1;
            popped_node.into_inner()
        }
    }
}
