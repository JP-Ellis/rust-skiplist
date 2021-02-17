use std::{fmt, iter, ptr};

// ////////////////////////////////////////////////////////////////////////////
// SkipNode
// ////////////////////////////////////////////////////////////////////////////

/// SkipNodes are make up the SkipList.  The SkipList owns the first head-node
/// (which has no value) and each node has ownership of the next node through
/// `next`.
///
/// The node has a `level` which corresponds to how 'high' the node reaches.
///
/// A node of `level` n has (n + 1) links to next nodes, which are stored in
/// a vector. level 0 points to the same node as `self.next`.
///
/// There is a corresponding vector of link lengths which contains the distance
/// between current node and the next node. If there's no next node, the distance
/// is distance between current node and last reachable node.
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
    // how high the node reaches.
    pub level: usize,
    // The immediately next element (and owns that next node).
    pub next: Option<Box<SkipNode<V>>>,
    // The immediately previous element.
    pub prev: *mut SkipNode<V>,
    // Vector of links to the next node at the respective level.  This vector
    // *must* be of length `self.level + 1`.  links[0] stores a pointer to the
    // same node as next.
    pub links: Vec<*mut SkipNode<V>>,
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
            prev: ptr::null_mut(),
            links: iter::repeat(ptr::null_mut()).take(total_levels).collect(),
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
            prev: ptr::null_mut(),
            links: iter::repeat(ptr::null_mut()).take(level + 1).collect(),
            links_len: iter::repeat(0).take(level + 1).collect(),
        }
    }

    /// Consumes the node returning the value it contains.
    pub fn into_inner(mut self) -> Option<V> {
        self.value.take()
    }

    /// Returns `true` is the node is a head-node.
    pub fn is_head(&self) -> bool {
        self.prev.is_null()
    }

    /// Distance between current node and the given node at specified level.
    /// If no node is given, then return distance between current node and the
    /// last possible node.
    /// If the node is not reachable on given level, return Err(()).
    pub fn distance(&self, level: usize, target: Option<&Self>) -> Result<usize, ()> {
        let distance = match target {
            Some(target) => {
                let (dest, distance) = self.advance_while(level, |current, _| {
                    current as *const _ != target as *const _
                });
                if dest as *const _ != target as *const _ {
                    return Err(());
                }
                distance
            }
            None => {
                let (dest, distance) = self.advance_while(level, |_, _| true);
                dest.links_len[level] + distance
            }
        };
        Ok(distance)
    }

    /// Try to move to next nth node at specified level.
    /// If it's impossible, then move as far as possible.
    /// Returns a reference to the new node and the distance travelled.
    pub fn advance_atmost(&self, level: usize, mut max_distance: usize) -> (&Self, usize) {
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
    pub fn advance_atmost_mut(
        &mut self,
        level: usize,
        mut max_distance: usize,
    ) -> (&mut Self, usize) {
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
        let next = unsafe { self.links[level].as_mut() };
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
        let next = unsafe { self.links[level].as_mut() };
        match next {
            Some(next) if predicate(self, next) => Ok((next, self.links_len[level])),
            _ => Err(self),
        }
    }

    /// Find the node after distance units, then insert a new node after that node.
    pub fn insert<'a>(&'a mut self, new_node: Box<Self>, distance: usize) {
        let locater = {
            let mut distance_left = distance;
            move |node: &'a mut Self, level| {
                let (dest, distance) = node.advance_atmost_mut(level, distance_left);
                distance_left -= distance;
                (dest, distance)
            }
        };
        self._insert(self.level, new_node, locater);
    }

    /// Locater finds the element that's before the target in a level.
    /// If the target does not exist in that level,
    /// it finds the last element that's not before target.
    /// In either case it also returns the distance travelled.
    ///
    /// The return value is distance traveled, used for fixup.
    pub fn _insert<'a, F>(
        &'a mut self,
        level: usize,
        mut new_node: Box<Self>,
        mut locater: F,
    ) -> (&'a mut Self, usize)
    where
        F: FnMut(&'a mut Self, usize) -> (&'a mut Self, usize),
    {
        let (prev_node, prev_distance) = locater(self, level);
        let prev_node_p = prev_node as *mut Self;
        if level == 0 {
            if let Some(mut tail) = prev_node.next.take() {
                tail.prev = new_node.as_mut() as *mut _;
                new_node.links[level] = tail.as_mut();
                new_node.next = Some(tail);
                new_node.links_len[level] = 1;
            }
            new_node.prev = prev_node as *mut _;
            prev_node.links[0] = new_node.as_mut() as *mut _;
            prev_node.links_len[level] = 1;
            prev_node.next = Some(new_node);
            return (prev_node.next.as_mut().unwrap(), prev_distance + 1);
        } else {
            let (inserted_node, insert_distance) = prev_node._insert(level - 1, new_node, locater);
            unsafe {
                if level <= inserted_node.level {
                    inserted_node.links[level] = (*prev_node_p).links[level];
                    inserted_node.links_len[level] =
                        (*prev_node_p).links_len[level] + 1 - insert_distance;
                    (*prev_node_p).links[level] = inserted_node as *mut _;
                    (*prev_node_p).links_len[level] = insert_distance;
                } else {
                    (*prev_node_p).links_len[level] += 1;
                }
            }
            return (inserted_node, insert_distance + prev_distance);
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
            node.prev = ptr::null_mut();
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
            let popped_node = match new_last.as_mut() {
                Some(new_last) => {
                    self.last = new_last as *mut _;
                    (*new_last).next.take().unwrap()
                }
                None => self.first.take().unwrap(),
            };
            self.size -= 1;
            popped_node.into_inner()
        }
    }
}
