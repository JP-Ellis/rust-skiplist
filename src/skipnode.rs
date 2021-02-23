use std::marker::PhantomData;
use std::{fmt, iter, ptr};

/// Minimum levels required for a list of size n.
pub fn levels_required(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        let num_bits = std::mem::size_of::<usize>() * 8;
        num_bits - n.leading_zeros() as usize
    }
}

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
/// a vector.
///
/// The node linked by level 0 should be considered owned by this node.
///
/// There is a corresponding vector of link lengths which contains the distance
/// between current node and the next node. If there's no next node, the distance
/// is distance between current node and last reachable node.
///
/// Lastly, each node contains a link to the immediately previous node in case
/// one needs to parse the list backwards.
#[derive(Clone, Debug)]
pub struct SkipNode<V> {
    // key and value should never be None, with the sole exception being the
    // head node.
    pub value: Option<V>,
    // how high the node reaches.
    pub level: usize,
    // The immediately previous element.
    pub prev: *mut SkipNode<V>,
    // Vector of links to the next node at the respective level.  This vector
    // *must* be of length `self.level + 1`.  links[0] stores a pointer to the
    // next node, which will have to be dropped.
    pub links: Vec<*mut SkipNode<V>>,
    // The corresponding length of each link
    pub links_len: Vec<usize>,
    // Owns self.link[0]
    _phantom_link: PhantomData<SkipNode<V>>,
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
            prev: ptr::null_mut(),
            links: iter::repeat(ptr::null_mut()).take(total_levels).collect(),
            links_len: iter::repeat(0).take(total_levels).collect(),
            _phantom_link: PhantomData,
        }
    }

    /// Create a new SkipNode with the given value.
    /// All pointers default to null.
    pub fn new(value: V, level: usize) -> Self {
        SkipNode {
            value: Some(value),
            level,
            prev: ptr::null_mut(),
            links: iter::repeat(ptr::null_mut()).take(level + 1).collect(),
            links_len: iter::repeat(0).take(level + 1).collect(),
            _phantom_link: PhantomData,
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

    pub fn next_ref(&self) -> Option<&Self> {
        // SAFETY: all links either points to something or is null.
        unsafe { self.links[0].as_ref() }
    }

    pub fn next_mut(&mut self) -> Option<&mut Self> {
        // SAFETY: all links either points to something or is null.
        unsafe { self.links[0].as_mut() }
    }

    /// Takes the next node and set next_node.prev as null.
    ///
    /// SAFETY: please make sure no link at level 1 or greater becomes dangling.
    pub unsafe fn take_tail(&mut self) -> Option<Box<Self>> {
        let next = self.links[0];
        if next.is_null() {
            None
        } else {
            let mut next = Box::from_raw(next);
            next.prev = ptr::null_mut();
            self.links[0] = ptr::null_mut();
            self.links_len[0] = 0;
            Some(next)
        }
    }

    /// Replace the next node.
    /// Return the old node.
    ///
    /// SAFETY: please makes sure all links are fixed.
    pub unsafe fn replace_tail(&mut self, mut new_next: Box<Self>) -> Option<Box<Self>> {
        let mut old_next = self.take_tail();
        if let Some(old_next) = old_next.as_mut() {
            old_next.prev = ptr::null_mut();
        }
        new_next.prev = self as *mut _;
        self.links[0] = Box::into_raw(new_next);
        self.links_len[0] = 1;
        old_next
    }
    // /////////////////////////////
    // Value Manipulation
    // /////////////////////////////
    //
    // Methods that care about values carried by the nodes.

    #[must_use]
    pub fn retain<F>(&mut self, mut pred: F) -> usize
    where
        F: FnMut(Option<&V>, &V) -> bool,
    {
        assert!(self.is_head());
        let mut removed = 0;
        let mut level_head: Vec<_> = iter::repeat(self as *mut Self)
            .take(self.level + 1)
            .collect();
        let mut node = self;
        unsafe {
            while let Some(mut next_node) = node.take_tail() {
                if pred(node.value.as_ref(), next_node.value.as_ref().unwrap()) {
                    for x in &mut level_head[0..=next_node.level] {
                        *x = next_node.as_mut() as *mut _;
                    }
                    node.replace_tail(next_node);
                    node = node.next_mut().unwrap();
                } else {
                    removed += 1;
                    for (level, head) in level_head
                        .iter_mut()
                        .map(|&mut x| x.as_mut().unwrap())
                        .enumerate()
                        .skip(1)
                    // should use take_next()/replace_next() to manage 0th level.
                    {
                        if level <= next_node.level {
                            assert_eq!(head.links[level], next_node.as_mut() as *mut _);
                            head.links_len[level] += next_node.links_len[level];
                            head.links_len[level] -= 1;
                            head.links[level] = next_node.links[level];
                        } else {
                            head.links_len[level] -= 1;
                        }
                    }
                    if let Some(new_next) = next_node.take_tail() {
                        node.replace_tail(new_next);
                    }
                }
            }
        }
        removed
    }

    // /////////////////////////////
    // Pointer Manipulations
    // /////////////////////////////
    //
    // Methods that care about the whole node.
    //

    /// Distance between current node and the given node at specified level.
    /// If no node is given, then return distance between current node and the
    /// last possible node.
    /// If the node is not reachable on given level, return Err(()).
    pub fn distance_at_level(&self, level: usize, target: Option<&Self>) -> Result<usize, ()> {
        let distance = match target {
            Some(target) => {
                let (dest, distance) = self.advance_while_at_level(level, |current, _| {
                    current as *const _ != target as *const _
                });
                if dest as *const _ != target as *const _ {
                    return Err(());
                }
                distance
            }
            None => {
                let (dest, distance) = self.advance_while_at_level(level, |_, _| true);
                dest.links_len[level] + distance
            }
        };
        Ok(distance)
    }

    /// Move for max_distance units.
    /// Returns None if it's not possible.
    pub fn advance(&self, max_distance: usize) -> Option<&Self> {
        let level = self.level;
        let mut node = self;
        let mut distance_left = max_distance;
        for level in (0..=level).rev() {
            let (new_node, steps) = node.advance_at_level(level, distance_left);
            distance_left -= steps;
            node = new_node;
        }
        if distance_left == 0 {
            Some(node)
        } else {
            None
        }
    }

    /// Move for max_distance units.
    /// Returns None if it's not possible.
    pub fn advance_mut(&mut self, max_distance: usize) -> Option<&mut Self> {
        let level = self.level;
        let mut node = self;
        let mut distance_left = max_distance;
        for level in (0..=level).rev() {
            let (new_node, steps) = node.advance_at_level_mut(level, distance_left);
            distance_left -= steps;
            node = new_node;
        }
        if distance_left == 0 {
            Some(node)
        } else {
            None
        }
    }

    /// Move to the last node reachable from this node.
    pub fn last(&self) -> &Self {
        (0..=self.level).rev().fold(self, |node, level| {
            node.advance_while_at_level(level, |_, _| true).0
        })
    }

    /// Move to the last node reachable from this node.
    pub fn last_mut(&mut self) -> &mut Self {
        (0..=self.level).rev().fold(self, |node, level| {
            node.advance_while_at_level_mut(level, |_, _| true).0
        })
    }

    /// Try to move for the given distance, only using links at the specified level.
    /// If it's impossible, then move as far as possible.
    ///
    /// Returns a reference to the new node and the distance travelled.
    pub fn advance_at_level(&self, level: usize, mut max_distance: usize) -> (&Self, usize) {
        self.advance_while_at_level(level, move |current_node, _| {
            let travelled = current_node.links_len[level];
            if travelled <= max_distance {
                max_distance -= travelled;
                true
            } else {
                false
            }
        })
    }

    /// Try to move for the given distance, only using links at the specified level.
    /// If it's impossible, then move as far as possible.
    ///
    /// Returns a mutable reference to the new node and the distance travelled.
    pub fn advance_at_level_mut(
        &mut self,
        level: usize,
        mut max_distance: usize,
    ) -> (&mut Self, usize) {
        self.advance_while_at_level_mut(level, move |current_node, _| {
            let travelled = current_node.links_len[level];
            if travelled <= max_distance {
                max_distance -= travelled;
                true
            } else {
                false
            }
        })
    }

    /// Keep moving at the specified level as long as pred is true.
    /// pred takes reference to current node and next node.
    pub fn advance_while_at_level(
        &self,
        level: usize,
        mut pred: impl FnMut(&Self, &Self) -> bool,
    ) -> (&Self, usize) {
        let mut current = self;
        let mut travelled = 0;
        loop {
            match current.next_if_at_level(level, &mut pred) {
                Ok((node, steps)) => {
                    current = node;
                    travelled += steps;
                }
                Err(node) => return (node, travelled),
            }
        }
    }

    /// Keep moving at the specified level as long as pred is true.
    /// pred takes reference to current node and next node.
    pub fn advance_while_at_level_mut(
        &mut self,
        level: usize,
        mut pred: impl FnMut(&Self, &Self) -> bool,
    ) -> (&mut Self, usize) {
        let mut current = self;
        let mut travelled = 0;
        loop {
            match current.next_if_at_level_mut(level, &mut pred) {
                Ok((node, steps)) => {
                    current = node;
                    travelled += steps;
                }
                Err(node) => return (node, travelled),
            }
        }
    }

    // The following methods return `Err(self)` if they fail.
    //
    // In Rust, the lifetime of returned value is the same as `self`.
    // Therefore if you return something that's borrowed from `self` in a branch,
    // `self` is considered borrowed in other branches.
    //
    // e.g.
    // ```
    // fn some_method(&mut self) -> Option<&mut Self>;
    //
    // fn caller(&mut self) {
    //     match self.some_method(){
    //         Some(x) => return x, // oops now `self` is borrowed until the function returns...
    //         None => return self, // Now you cannot use `self` in other branches..
    //     }                        // including returning it!
    // }
    // ```
    // While in this example you can restructure the code to fix that,
    // it's much more difficult when loops are involved.
    // The following methods are usually used in loops, so they return `Err(self)`
    // when they fail, to ease the pain.

    /// Move to the next node at given level if the given predicate is true.
    /// The predicate takes reference to the current node and the next node.
    pub fn next_if_at_level_mut(
        &mut self,
        level: usize,
        predicate: impl FnOnce(&Self, &Self) -> bool,
    ) -> Result<(&mut Self, usize), &mut Self> {
        // SAFETY: all links either points to something or is null.
        let next = unsafe { self.links[level].as_mut() };
        match next {
            Some(next) if predicate(self, next) => Ok((next, self.links_len[level])),
            _ => Err(self),
        }
    }

    /// Move to the next node at given level if the given predicate is true.
    /// The predicate takes reference to the current node and the next node.
    pub fn next_if_at_level(
        &self,
        level: usize,
        predicate: impl FnOnce(&Self, &Self) -> bool,
    ) -> Result<(&Self, usize), &Self> {
        // SAFETY: all links either points to something or is null.
        let next = unsafe { self.links[level].as_mut() };
        match next {
            Some(next) if predicate(self, next) => Ok((next, self.links_len[level])),
            _ => Err(self),
        }
    }

    /// Insert a node after given distance after the list head.
    ///
    /// Requries that there's nothing before the node and the new node can't be at a higher level.
    ///
    /// Return the reference to the new node if successful.
    /// Give back the input node if not succssful.
    pub fn insert(
        &mut self,
        new_node: Box<Self>,
        distance_to_parent: usize,
    ) -> Result<&mut Self, Box<Self>> {
        assert!(self.prev.is_null(), "Only the head may insert nodes!");
        assert!(
            self.level >= new_node.level,
            "You may not insert nodes with level higher than the head!"
        );
        // SAFETY: This operation is safe because there's no node before self and it's inserting at
        // the highest level.
        let (node, distance_to_new_node) = unsafe {
            self._insert(self.level, new_node, {
                let mut distance_left = distance_to_parent;
                move |node: &mut Self, level| {
                    let (dest, distance) = node.advance_at_level_mut(level, distance_left);
                    distance_left -= distance;
                    if level == 0 && distance_left != 0 {
                        None
                    } else {
                        Some((dest, distance))
                    }
                }
            })
        }?;
        assert_eq!(distance_to_parent + 1, distance_to_new_node);
        Ok(node)
    }

    /// Move for distance units, and remove the node after it.
    ///
    /// Requries that there's nothing before the node and the new node can't be at a higher level.
    ///
    /// If that node exists, remove that node and retrun it.
    pub fn remove(&mut self, distance_to_parent: usize) -> Option<Box<Self>> {
        assert!(self.prev.is_null(), "Only the head may remove nodes!");
        // SAFETY: This operation is safe because there's no node before self and the head can be
        // assumed as at highest level.
        let (node, distance_to_removed) = unsafe {
            self._remove(self.level, {
                let mut distance_left = distance_to_parent;
                move |node: &mut Self, level| {
                    let (dest, distance) = node.advance_at_level_mut(level, distance_left);
                    distance_left -= distance;
                    if dest.links[0].is_null() {
                        None
                    } else {
                        Some((dest, distance))
                    }
                }
            })?
        };
        assert_eq!(
            distance_to_removed,
            distance_to_parent + 1,
            "Expected to remove node at {} but somehow the node at {} is removed instead",
            distance_to_parent + 1,
            distance_to_removed
        );
        Some(node)
    }

    /// A helper method for insertion.
    ///
    /// Locater finds the node before the target position in a level,
    /// as well as the distance from input node to that node.
    /// If it fails to find such position, return None.
    ///
    /// At level 0 it either returns the target's parent or None.
    ///
    /// It may be stateful and use the information from previous
    /// calls to determine the result.
    /// In short, it may remember how far it has already travelled.
    ///
    /// If the insertion succeeds, return a mutable reference to the new node
    /// and the distance between self and the new node.
    /// If the insertion fails, fails return the given node.
    ///
    /// SAFETY: This function only fixes links at or after `self`,
    /// and only fixes links at or below current level.
    pub unsafe fn _insert<'a, F>(
        &'a mut self,
        level: usize,
        mut new_node: Box<Self>,
        mut locater: F,
    ) -> Result<(&'a mut Self, usize), Box<Self>>
    where
        F: FnMut(&'a mut Self, usize) -> Option<(&'a mut Self, usize)>,
    {
        // This function first finds the node before the insert position using locater.
        let (prev_node, prev_distance) = match locater(self, level) {
            Some(res) => res,
            None => return Err(new_node),
        };
        let prev_node_p = prev_node as *mut Self;
        if level != 0 {
            // If it's not the last level, recursively call itself to insert at lower level.
            // This call fixes all links below current level.
            // After this call we proceed to fix links at the current level.
            let (inserted_node, insert_distance) =
                prev_node._insert(level - 1, new_node, locater)?;
            // prev_node._insert() borrows `prev_node`, so we need to create a new reference to it.
            // SAFETY: It's safe because it can never alias with `inserted_node`.
            let prev_node = &mut *prev_node_p;
            // Fix links of prev_node and inserted_node at this level.
            if level <= inserted_node.level {
                inserted_node.links[level] = prev_node.links[level];
                inserted_node.links_len[level] = prev_node.links_len[level] + 1 - insert_distance;
                prev_node.links[level] = inserted_node as *mut _;
                prev_node.links_len[level] = insert_distance;
            } else {
                // Already pointing to the correct node; fix length.
                prev_node.links_len[level] += 1;
            }
            Ok((inserted_node, insert_distance + prev_distance))
        } else {
            // {take|replace}_tail takes care of links at level 0.
            // SAFETY: The caller takes care of links at other levels.
            if let Some(tail) = prev_node.take_tail() {
                new_node.replace_tail(tail);
            }
            prev_node.replace_tail(new_node);
            Ok((prev_node.next_mut().unwrap(), prev_distance + 1))
        }
    }

    /// A helper method for removal.
    ///
    /// Locater takes a mutable reference and a level,
    /// then returns the node after which the node target node may exist
    /// and the distance it moved from the given reference.
    ///
    /// If the locater is certain that such node does not exist at any level,
    /// the locater should return None.
    /// At level 0 it either returns the target's parent or None.
    ///
    /// The locater may be stateful and use the information from previous
    /// calls to determine the result.
    /// In short, it may remember how far it has already travelled.
    ///
    /// Returns None if failed to remove a node.
    /// Otherwise it returns the removed node and
    /// the distance between the current node and the removed node.
    ///
    /// SAFETY: This function only fixes links at or after `self`,
    /// and only fixes links at or below current level.
    pub unsafe fn _remove<'a, F>(
        &'a mut self,
        level: usize,
        mut locater: F,
    ) -> Option<(Box<Self>, usize)>
    where
        F: FnMut(&'a mut Self, usize) -> Option<(&'a mut Self, usize)>,
    {
        // This function first finds the node to remove using locater.
        let (prev_node, prev_distance) = locater(self, level)?;
        let prev_node_p = prev_node as *mut Self;
        if level != 0 {
            // If it's not the last level, recursively call itself to remove at lower level.
            // This call fixes all links below current level.
            // After this call we proceed to fix links at the current level.
            let (removed_node, distance) = prev_node._remove(level - 1, locater)?;
            // Rust consider prev_node as borrowed until we return for some reason.
            // We create a new mutable reference to that.
            // It's safe because nothing aliases it.
            let prev_node = &mut *prev_node_p;
            // Fix links of prev_node at this level.
            if level <= removed_node.level {
                prev_node.links[level] = removed_node.links[level];
                assert_eq!(prev_node.links_len[level], distance);
                prev_node.links_len[level] = distance + removed_node.links_len[level] - 1;
            } else {
                // Already pointing to the correct node; fix length.
                prev_node.links_len[level] -= 1;
            }
            Some((removed_node, prev_distance + distance))
        } else {
            // {take|replace}_tail takes care of links at level 0.
            // SAFETY: The caller takes care of links at other levels.
            let mut removed_node = prev_node.take_tail()?;
            if let Some(new_tail) = removed_node.take_tail() {
                prev_node.replace_tail(new_tail);
            }
            Some((removed_node, prev_distance + 1))
        }
    }
}

impl<V> Drop for SkipNode<V> {
    fn drop(&mut self) {
        // SAFETY: all nodes are going to be dropped; its okay that its links (except those at
        // level 0) become dangling.
        unsafe {
            let mut node = self.take_tail();
            while let Some(mut node_inner) = node {
                node = node_inner.take_tail();
            }
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

// ///////////////////////////////////////////////
// Helper Traits
// ///////////////////////////////////////////////

// Converting Option<&T> to *_ T becomes more and more annoying...
trait AsPtr<T> {
    fn as_ptr(&self) -> *const T;
}

trait AsPtrMut<T> {
    fn as_ptr_mut(&mut self) -> *mut T;
}

impl<T> AsPtr<T> for Option<&T> {
    fn as_ptr(&self) -> *const T {
        self.map_or(ptr::null(), |inner_ref| inner_ref)
    }
}

impl<T> AsPtr<T> for Option<&mut T> {
    fn as_ptr(&self) -> *const T {
        self.as_ref().map_or(ptr::null(), |inner: &&mut T| &**inner)
    }
}

impl<T> AsPtrMut<T> for Option<&mut T> {
    fn as_ptr_mut(&mut self) -> *mut T {
        self.as_mut()
            .map_or(ptr::null_mut(), |inner: &mut &mut T| *inner)
    }
}

// /////////////////////////////////
// Iterators
// /////////////////////////////////
// Since Iterators (currently) only pop from front and back,
// they can be shared by some data structures.
// There's no need for a dummy head (that contains no value) in the iterator.
// so the members are named first and last instaed of head/end to avoid confusion.

/// Iterator by reference
pub struct Iter<'a, T> {
    pub(crate) first: Option<&'a SkipNode<T>>,
    pub(crate) last: Option<&'a SkipNode<T>>,
    pub(crate) size: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let current_node = self.first?;
        if ptr::eq(current_node, self.last.as_ptr()) {
            self.first = None;
            self.last = None;
        } else {
            self.first = current_node.next_ref();
        }
        self.size -= 1;
        current_node.value.as_ref()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let last_node = self.last?;

        if ptr::eq(self.first.as_ptr(), last_node) {
            self.first = None;
            self.last = None;
        } else {
            // SAFETY: The iterator is not empty yet.
            unsafe {
                self.last = last_node.prev.as_ref();
            }
        }
        self.size -= 1;
        last_node.value.as_ref()
    }
}

/// Iterator by mutable reference
pub struct IterMut<'a, T> {
    pub(crate) first: Option<&'a mut SkipNode<T>>,
    pub(crate) last: *mut SkipNode<T>,
    pub(crate) size: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let current_node = self.first.take()?;
        if ptr::eq(current_node, self.last) {
            self.first = None;
            self.last = ptr::null_mut();
        } else {
            // calling current_node.next_mut() borrows it, so we need to use a pointer.
            let p = current_node.next_mut().unwrap() as *mut SkipNode<T>;
            // SAFETY: p.as_mut() is safe because it points to a valid object.
            // There's no aliasing issue since nobody else holds a reference to current_node
            // until this function returns, and the returned reference does not points to a node.
            unsafe {
                self.first = p.as_mut();
            }
        }
        self.size -= 1;
        current_node.value.as_mut()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.last.is_null() {
            return None;
        }
        // There can be at most one mutable reference to the first node.
        // We need to take it from self.first before doing anything,
        // including simple comparison.
        let first = self.first.take().unwrap();
        // SAFETY: we already checked self.last points to something.
        let before_last = unsafe { (*self.last).prev };
        let popped_node = if first as *mut _ == self.last {
            self.first = None;
            self.last = ptr::null_mut();
            first
        } else if first as *mut _ == before_last {
            // self.first aliasing before_last
            // I'm not sure if it's necessarity,
            // But lets try not to access (*before_last) to avoid UB.
            self.last = first as *mut _;
            let popped = first.next_mut().unwrap();
            // SAFETY: we already checked self.last points to something.
            self.first = unsafe { self.last.as_mut() };
            popped
        } else {
            self.first.replace(first); // we took self.first, put it back.
            self.last = before_last;
            // SAFETY: before_last.next_mut() points to the old self.last.
            unsafe { (*before_last).next_mut().unwrap() }
        };
        self.size -= 1;
        popped_node.value.as_mut()
    }
}

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
        // SAFETY: no need to fix links at upper levels inside iterators.
        self.first = unsafe { popped_node.take_tail() };
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
        #[allow(clippy::question_mark)]
        if self.first.is_none() {
            return None;
        }
        assert!(
            !self.last.is_null(),
            "The IntoIter should be empty but IntoIter.last somehow still contains something"
        );

        // SAFETY: we already checked self.last is not null.
        let new_last = unsafe { (*self.last).prev };
        let popped_node = if new_last.is_null() {
            self.first.take().unwrap()
        } else {
            // SAFETY: new_last is not null there's no need to fix links at upper levels inside
            // iterators.
            unsafe { (*new_last).take_tail().unwrap() }
        };
        self.last = new_last;
        self.size -= 1;
        popped_node.into_inner()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_level_required() {
        assert_eq!(levels_required(0), 1);
        assert_eq!(levels_required(1), 1);
        assert_eq!(levels_required(2), 2);
        assert_eq!(levels_required(3), 2);
        assert_eq!(levels_required(1023), 10);
        assert_eq!(levels_required(1024), 11);
    }

    fn level_for_index(mut n: usize) -> usize {
        let mut cnt = 0;
        while n & 0x1 == 1 {
            cnt += 1;
            n /= 2;
        }
        cnt
    }

    #[test]
    fn test_level_index() {
        assert_eq!(level_for_index(0), 0);
        assert_eq!(level_for_index(1), 1);
        assert_eq!(level_for_index(2), 0);
        assert_eq!(level_for_index(3), 2);
        assert_eq!(level_for_index(4), 0);
        assert_eq!(level_for_index(5), 1);
        assert_eq!(level_for_index(6), 0);
        assert_eq!(level_for_index(7), 3);
        assert_eq!(level_for_index(8), 0);
        assert_eq!(level_for_index(9), 1);
        assert_eq!(level_for_index(10), 0);
        assert_eq!(level_for_index(11), 2);
    }

    /// Make a list of size n
    /// levels are evenly spread out
    fn new_list_for_test(n: usize) -> SkipNode<usize> {
        let max_level = levels_required(n);
        let mut head = SkipNode::<usize>::head(max_level);
        assert_eq!(head.links.len(), max_level);
        let mut nodes: Vec<_> = (0..n)
            .map(|n| {
                let new_node = Box::new(SkipNode::new(n, level_for_index(n)));
                Box::into_raw(new_node)
            })
            .collect();
        unsafe {
            let node_max_level = nodes.iter().map(|&node| (*node).level).max();
            if let Some(node_max_level) = node_max_level {
                assert_eq!(node_max_level + 1, max_level);
            }
            for level in 0..max_level {
                let mut last_node = &mut head as *mut SkipNode<usize>;
                let mut len_left = n;
                for &mut node_ptr in nodes
                    .iter_mut()
                    .filter(|&&mut node_ptr| level <= (*node_ptr).level)
                {
                    if level == 0 {
                        (*node_ptr).prev = last_node;
                    }
                    (*last_node).links[level] = node_ptr;
                    (*last_node).links_len[level] = 1 << level;
                    last_node = node_ptr;
                    len_left -= 1 << level;
                }
                (*last_node).links_len[level] = len_left;
            }
        }
        return head;
    }

    /////////////////////////////////////////////////////////
    // Those tests are supposed to be run using Miri to detect UB.
    // The size of those test are limited since Miri doesn't run very fast.
    /////////////////////////////////////////////////////////

    #[test]
    fn miri_test_insert() {
        let mut list = new_list_for_test(50);
        list.insert(Box::new(SkipNode::new(100, 0)), 25).unwrap();
        list.insert(Box::new(SkipNode::new(101, 1)), 25).unwrap();
        list.insert(Box::new(SkipNode::new(102, 2)), 25).unwrap();
        list.insert(Box::new(SkipNode::new(103, 3)), 25).unwrap();
        list.insert(Box::new(SkipNode::new(104, 4)), 25).unwrap();
    }

    #[test]
    fn miri_test_remove() {
        let mut list = new_list_for_test(50);
        for i in (0..50).rev() {
            list.remove(i).unwrap();
        }
    }

    #[test]
    fn miri_test_distance() {
        let list = new_list_for_test(50);
        for i in 0..=list.level {
            let _ = list.distance_at_level(i, None);
        }
    }

    #[test]
    fn miri_test_iter() {
        let list = new_list_for_test(50);
        let first = list.next_ref();
        let last = Some(list.last());
        let mut iter = Iter {
            first,
            last,
            size: 50,
        };
        for _ in 0..25 {
            let _ = iter.next();
            let _ = iter.next_back();
        }
    }

    #[test]
    fn miri_test_iter_mut() {
        let mut list = new_list_for_test(50);
        let mut first = list.next_mut();
        let last = first.as_mut().unwrap().last_mut() as *mut SkipNode<usize>;
        let mut iter = IterMut {
            first,
            last,
            size: 50,
        };
        for _ in 0..25 {
            let _ = iter.next();
            let _ = iter.next_back();
        }
    }

    #[test]
    fn miri_test_into_iter() {
        let mut list = new_list_for_test(50);
        let mut first = unsafe {Some(list.take_tail().unwrap())};
        let last = first.as_mut().unwrap().last_mut() as *mut SkipNode<usize>;
        let mut iter = IntoIter{
            first,
            last,
            size: 50,
        };
        for _ in 0..25 {
            let _ = iter.next();
            let _ = iter.next_back();
        }
    }

    #[test]
    fn miri_test_retain() {
        let mut list = new_list_for_test(50);
        let _ = list.retain(|_, val| val % 2 == 0);
    }
}
