//! Iteration support for [`OrderedSkipList`](super::OrderedSkipList): `iter`,
//! `range`, `drain`, `extract_if`, all iterator types, and
//! [`IntoIterator`] implementations.
//!
//! Note: `IterMut` is intentionally absent.  Mutating an element in place
//! could violate the sort-order invariant, so only shared (`&T`) iteration is
//! exposed.

use core::{
    cmp::Ordering,
    fmt,
    iter::FusedIterator,
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    ptr::NonNull,
};

use crate::{
    comparator::{Comparator, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    node::{
        Node,
        visitor::{OrdVisitor, Visitor},
    },
    ordered_skip_list::OrderedSkipList,
};

// MARK: Private helpers

/// Counts elements between `front` and `back` (both inclusive) by following
/// next links. Returns Some(count) if `back` is reachable from `front`, or
/// None if the chain ends without encountering `back`.
///
/// # Safety
///
/// Both `front` and `back` must be valid, live Node pointers in the same
/// skip list, both remaining valid for the duration of this call.
unsafe fn count_inclusive<T, const N: usize>(
    front: NonNull<Node<T, N>>,
    back: NonNull<Node<T, N>>,
) -> Option<usize> {
    let mut count: usize = 0;
    let mut ptr = front;
    loop {
        count = count.saturating_add(1);
        if ptr == back {
            return Some(count);
        }
        // SAFETY: ptr is a live node pointer; caller guarantees validity.
        ptr = unsafe { ptr.as_ref() }.next()?;
    }
}

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> OrderedSkipList<T, N, C, G> {
    /// Returns a raw pointer to the first node that satisfies the given lower
    /// bound, or `None` when no such node exists.
    ///
    /// - `Unbounded`   -> first data node (head's successor)
    /// - `Included(v)` -> first node with value >= v
    /// - `Excluded(v)` -> first node with value > v
    fn lower_bound_ptr(&self, bound: Bound<&T>) -> Option<NonNull<Node<T, N>>> {
        match bound {
            Bound::Unbounded => {
                // SAFETY: self.head is always valid for &self.
                unsafe { self.head.as_ref() }.next()
            }
            Bound::Included(v) => {
                // Advance only through nodes strictly < v; stop as soon as we
                // hit a node == v or > v.  After traversal, current() is the
                // last node < v (or the head sentinel when all nodes ≥ v), so
                // current().next() is the first node ≥ v.
                let head = self.head_ref();
                let cmp = |node_val: &T, target: &T| {
                    match self.comparator.compare(node_val, target) {
                        Ordering::Less => Ordering::Less,
                        // Treat Equal and Greater the same: do not advance past them.
                        Ordering::Equal | Ordering::Greater => Ordering::Greater,
                    }
                };
                let mut visitor = OrdVisitor::new(head, v, cmp);
                visitor.traverse();
                visitor.current().next()
            }
            Bound::Excluded(v) => {
                // Advance through nodes ≤ v; stop just before > v.
                // After traversal, current() is the last node ≤ v (or head),
                // so current().next() is the first node > v.
                let head = self.head_ref();
                let cmp = |node_val: &T, target: &T| match self.comparator.compare(node_val, target)
                {
                    Ordering::Less | Ordering::Equal => Ordering::Less,
                    Ordering::Greater => Ordering::Greater,
                };
                let mut visitor = OrdVisitor::new(head, v, cmp);
                visitor.traverse();
                visitor.current().next()
            }
        }
    }

    /// Returns a raw pointer to the last node that satisfies the given upper
    /// bound, or `None` when no such node exists.
    ///
    /// - `Unbounded`   -> tail (last data node)
    /// - `Included(v)` -> last node with value <= v
    /// - `Excluded(v)` -> last node with value < v
    fn upper_bound_ptr(&self, bound: Bound<&T>) -> Option<NonNull<Node<T, N>>> {
        match bound {
            Bound::Unbounded => self.tail,
            Bound::Included(v) => {
                // Advance through all nodes ≤ v; stop before > v.
                // After traversal, current() is the last node ≤ v (or head).
                let head = self.head_ref();
                let cmp = |node_val: &T, target: &T| match self.comparator.compare(node_val, target)
                {
                    Ordering::Less | Ordering::Equal => Ordering::Less,
                    Ordering::Greater => Ordering::Greater,
                };
                let mut visitor = OrdVisitor::new(head, v, cmp);
                visitor.traverse();
                let current = visitor.current();
                // NonNull::from a shared reference is safe; no unsafe needed here.
                // Return None when the sentinel head has no value (all nodes > v).
                current.value().is_some().then(|| NonNull::from(current))
            }
            Bound::Excluded(v) => {
                // Advance only through nodes strictly < v.
                // After traversal, current() is the last node < v (or head).
                let head = self.head_ref();
                let cmp = |node_val: &T, target: &T| match self.comparator.compare(node_val, target)
                {
                    Ordering::Less => Ordering::Less,
                    Ordering::Equal | Ordering::Greater => Ordering::Greater,
                };
                let mut visitor = OrdVisitor::new(head, v, cmp);
                visitor.traverse();
                let current = visitor.current();
                current.value().is_some().then(|| NonNull::from(current))
            }
        }
    }

    // MARK: Iteration methods

    /// Returns an iterator over shared references to the elements of the list,
    /// yielded in sorted order from smallest to largest.
    ///
    /// The iterator also supports [`DoubleEndedIterator`], allowing traversal
    /// in reverse order.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(3);
    /// list.insert(1);
    /// list.insert(2);
    ///
    /// let collected: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(collected, [1, 2, 3]);
    ///
    /// let reversed: Vec<i32> = list.iter().copied().rev().collect();
    /// assert_eq!(reversed, [3, 2, 1]);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T, N> {
        Iter {
            // SAFETY: self.head is a valid, exclusively-owned head sentinel.
            front: unsafe { self.head.as_ref() }.next(),
            back: self.tail,
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over shared references to elements whose values
    /// fall within the given bound range.
    ///
    /// Bounds are inclusive or exclusive on either end; all standard Rust range
    /// expressions (`..`, `a..`, `..b`, `..=b`, `a..b`, `a..=b`) are accepted.
    ///
    /// Finding the start and end nodes is `$O(\log n)$`; counting the elements in
    /// the range is `$O(k)$` where k is the number of elements in the range.
    ///
    /// # Panics
    ///
    /// Panics if the lower bound is greater than the upper bound according to
    /// the list's comparator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.insert(i);
    /// }
    ///
    /// let slice: Vec<i32> = list.range(2..=4).copied().collect();
    /// assert_eq!(slice, [2, 3, 4]);
    ///
    /// let reversed: Vec<i32> = list.range(2..=4).copied().rev().collect();
    /// assert_eq!(reversed, [4, 3, 2]);
    /// ```
    #[expect(
        clippy::panic,
        reason = "mirrors BTreeMap::range / BTreeSet::range: panics on invalid range bounds, \
                  not on internal invariant violations"
    )]
    #[expect(
        clippy::shadow_reuse,
        reason = "rebinding front/back after the bound-resolution helpers is the \
                  clearest way to express the narrowing from Option to the final triple"
    )]
    #[inline]
    pub fn range<R: RangeBounds<T>>(&self, range: R) -> Iter<'_, T, N> {
        let lo = range.start_bound();
        let hi = range.end_bound();

        // Validate that the range bounds are logically consistent.
        // Mirrors the BTreeMap / BTreeSet contract:
        //   - lo > hi → panic
        //   - lo == hi and both Excluded → panic (no element can satisfy lo < x < lo)
        //   - lo == hi and at least one Included → valid (possibly empty)
        if let (Bound::Included(a) | Bound::Excluded(a), Bound::Included(b) | Bound::Excluded(b)) =
            (lo, hi)
        {
            match self.comparator.compare(a, b) {
                Ordering::Greater => {
                    panic!("range start is after range end in OrderedSkipList");
                }
                Ordering::Equal => {
                    assert!(
                        !matches!((lo, hi), (Bound::Excluded(_), Bound::Excluded(_))),
                        "range start and end are equal and excluded in OrderedSkipList"
                    );
                }
                Ordering::Less => {}
            }
        }
        // at least one Unbounded → always valid

        let front = self.lower_bound_ptr(lo);
        let back = self.upper_bound_ptr(hi);

        let (front, back, len) = match (front, back) {
            (None, _) | (_, None) => (None, None, 0),
            (Some(f), Some(b)) => {
                // SAFETY: f and b are valid node pointers in this list, live
                // for &self.  count_inclusive walks the next chain.  If back
                // is not reachable from front the range is valid but empty
                // (bounds are in range but no elements happen to fall there).
                match unsafe { count_inclusive(f, b) } {
                    Some(len) => (Some(f), Some(b), len),
                    None => (None, None, 0),
                }
            }
        };

        Iter {
            front,
            back,
            len,
            _marker: PhantomData,
        }
    }

    /// Removes all elements from the list and returns them as an iterator in
    /// sorted order.
    ///
    /// After the call the list is empty.  Elements are collected eagerly before
    /// the `Drain` is returned: the list is cleared regardless of whether the
    /// caller consumes the iterator.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// # Panics
    ///
    /// Does not panic under normal use.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(3);
    /// list.insert(1);
    /// list.insert(2);
    ///
    /// let drained: Vec<i32> = list.drain().collect();
    /// assert_eq!(drained, [1, 2, 3]);
    /// assert!(list.is_empty());
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "`take_value()` returns None only for the head sentinel, which is never \
                  in the drain range; the expect fires only on invariant violations"
    )]
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, T> {
        let mut drained: Vec<T> = Vec::with_capacity(self.len);

        // SAFETY: All raw pointers come from heap allocations owned by this
        // OrderedSkipList.  We hold &mut self.  The keep closure returns false
        // for every node, so all nodes are dropped via the on_drop closure that
        // extracts their values.
        let (new_len, new_tail) = unsafe {
            Node::filter_rebuild(
                self.head,
                |_cur| false,
                |mut boxed| {
                    drained.push(boxed.take_value().expect("data node has a value"));
                },
            )
        };
        self.tail = new_tail;
        self.len = new_len;

        Drain {
            iter: drained.into_iter(),
            _marker: PhantomData,
        }
    }

    /// Removes all elements whose values fall within the given range and
    /// returns them as an iterator in sorted order.
    ///
    /// The bound syntax is identical to [`range`](OrderedSkipList::range):
    /// all standard Rust range expressions (`..`, `a..`, `..b`, `..=b`,
    /// `a..b`, `a..=b`) are accepted.
    ///
    /// Elements outside the range remain in the list.  All elements inside
    /// the range are collected eagerly before the `Drain` is returned, so
    /// the list is updated regardless of whether the caller consumes the
    /// iterator.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// # Panics
    ///
    /// Panics if the lower bound is greater than the upper bound according to
    /// the list's comparator (mirrors [`BTreeMap::drain_filter`] and
    /// [`range`](OrderedSkipList::range)).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.insert(i);
    /// }
    ///
    /// let mid: Vec<i32> = list.drain_range(2..=4).collect();
    /// assert_eq!(mid, [2, 3, 4]);
    /// let remaining: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(remaining, [1, 5]);
    /// ```
    #[expect(
        clippy::panic,
        reason = "mirrors OrderedSkipList::range: panics on invalid range bounds, \
                  not on internal invariant violations"
    )]
    #[expect(
        clippy::expect_used,
        reason = "`value()` / `take_value()` return None only for the head sentinel, \
                  which is never visited in the data-node walk; \
                  the expect fires only on invariant violations"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "calling filter_rebuild (unsafe fn) and dereferencing cur inside the keep \
                  closure are provably disjoint"
    )]
    #[inline]
    pub fn drain_range<R: RangeBounds<T>>(&mut self, range: R) -> Drain<'_, T> {
        let lo = range.start_bound();
        let hi = range.end_bound();

        // Validate bounds using the same contract as range().
        if let (Bound::Included(a) | Bound::Excluded(a), Bound::Included(b) | Bound::Excluded(b)) =
            (lo, hi)
        {
            match self.comparator.compare(a, b) {
                Ordering::Greater => {
                    panic!("range start is after range end in OrderedSkipList");
                }
                Ordering::Equal => {
                    assert!(
                        matches!((lo, hi), (Bound::Included(_), Bound::Included(_))),
                        "range start is after range end in OrderedSkipList"
                    );
                }
                Ordering::Less => {}
            }
        }

        if self.is_empty() {
            return Drain {
                iter: Vec::new().into_iter(),
                _marker: PhantomData,
            };
        }

        let mut drained: Vec<T> = Vec::new();

        // `self.head` is NonNull (Copy), so copying it does not borrow `self`.
        // The keep closure borrows only `self.comparator` (shared) and `lo`/`hi`
        // (borrowed from `range`, which outlives this call), both distinct from
        // the mutable `self.tail`/`self.len` updates that follow.
        let head = self.head;
        let comparator = &self.comparator;
        let mut past_hi = false;

        // SAFETY: All raw pointers originate from heap allocations owned by
        // this OrderedSkipList.  We hold &mut self so no other reference to
        // any node exists.  The keep closure reads the value before any
        // structural mutation occurs.
        let (new_len, new_tail) = unsafe {
            Node::filter_rebuild(
                head,
                |cur| {
                    if past_hi {
                        return true; // all remaining nodes are past upper bound: keep
                    }
                    // SAFETY: cur is a live, heap-allocated data node.
                    let val: &T = (*cur).value().expect("data node has a value");

                    let above_lo = match lo {
                        Bound::Unbounded => true,
                        Bound::Included(l) => comparator.compare(val, l) != Ordering::Less,
                        Bound::Excluded(l) => comparator.compare(val, l) == Ordering::Greater,
                    };
                    if !above_lo {
                        return true; // before range: keep
                    }

                    let below_hi = match hi {
                        Bound::Unbounded => true,
                        Bound::Included(h) => comparator.compare(val, h) != Ordering::Greater,
                        Bound::Excluded(h) => comparator.compare(val, h) == Ordering::Less,
                    };
                    if !below_hi {
                        past_hi = true;
                        return true; // past upper bound: keep
                    }

                    false // inside range: drain
                },
                |mut boxed| {
                    drained.push(boxed.take_value().expect("data node has a value"));
                },
            )
        };
        self.tail = new_tail;
        self.len = new_len;

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
    /// element before deciding whether to extract it.  Note, however, that
    /// mutating an element in a way that changes its sort position may violate
    /// the ordering invariant.
    ///
    /// If the `ExtractIf` iterator is dropped before being fully consumed,
    /// the predicate is **not** called for the remaining elements; they all
    /// stay in the list.  The list remains valid and fully usable after the
    /// iterator is dropped.
    ///
    /// This operation is `$O(n)$` for a full traversal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for i in 1..=5 {
    ///     list.insert(i);
    /// }
    ///
    /// let evens: Vec<i32> = list.extract_if(|x| *x % 2 == 0).collect();
    /// assert_eq!(evens, [2, 4]);
    /// let remaining: Vec<i32> = list.iter().copied().collect();
    /// assert_eq!(remaining, [1, 3, 5]);
    /// ```
    #[inline]
    pub fn extract_if<F>(&mut self, pred: F) -> ExtractIf<'_, T, C, G, F, N>
    where
        F: FnMut(&mut T) -> bool,
    {
        // SAFETY: self.head is a valid, exclusively-owned head sentinel.
        let current = unsafe { self.head.as_ref().next() };
        ExtractIf {
            current,
            any_removed: false,
            list: self,
            pred,
        }
    }
}

// MARK: IntoIterator

impl<'a, T, C: Comparator<T>, G: LevelGenerator, const N: usize> IntoIterator
    for &'a OrderedSkipList<T, N, C, G>
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> IntoIterator
    for OrderedSkipList<T, N, C, G>
{
    type Item = T;
    type IntoIter = IntoIter<T, N, C, G>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { list: self }
    }
}

// MARK: Iter

/// An iterator over shared references to the elements of an
/// [`OrderedSkipList`], yielded in sorted order.
///
/// This struct is created by the [`OrderedSkipList::iter`] method and by the
/// [`OrderedSkipList::range`] method.  See their documentation for more.
///
/// # Examples
///
/// ```rust
/// use skiplist::ordered_skip_list::OrderedSkipList;
///
/// let mut list = OrderedSkipList::<i32>::new();
/// list.insert(1);
/// list.insert(2);
/// list.insert(3);
///
/// let mut iter = list.iter();
/// assert_eq!(iter.next(), Some(&1));
/// assert_eq!(iter.next_back(), Some(&3));
/// assert_eq!(iter.next(), Some(&2));
/// assert_eq!(iter.next(), None);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Iter<'a, T, const N: usize = 16> {
    /// Pointer to the next element to yield from the front, or None when the
    /// iterator is exhausted or the list was empty.
    front: Option<NonNull<Node<T, N>>>,
    /// Pointer to the next element to yield from the back, or None when the
    /// iterator is exhausted or the list was empty.
    back: Option<NonNull<Node<T, N>>>,
    /// Number of elements remaining. Guards against yielding more than `len`
    /// items even when `front` and `back` pointers cross mid-list during
    /// interleaved `next`/`next_back` calls.
    len: usize,
    /// Ties the iterator's lifetime to `&'a OrderedSkipList` and expresses
    /// covariance in T.
    _marker: PhantomData<&'a T>,
}

// SAFETY: Iter<'a, T> yields `&'a T` (shared, non-owning references).
// Sending it to another thread requires T: Sync because the receiving thread
// will read T values through a shared reference derived from the raw pointer
// carried by this type.
unsafe impl<T: Sync, const N: usize> Send for Iter<'_, T, N> {}

// SAFETY: Sharing &Iter<'a, T> across threads is safe when T: Sync.
// Concurrent callers need &mut Iter to advance it, so data races on the
// iterator's own fields are prevented by the requirement for exclusive access
// through &mut.
unsafe impl<T: Sync, const N: usize> Sync for Iter<'_, T, N> {}

impl<T, const N: usize> Clone for Iter<'_, T, N> {
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

impl<T: fmt::Debug, const N: usize> fmt::Debug for Iter<'_, T, N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, T, const N: usize> Iterator for Iter<'a, T, N> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let front_ptr = self.front?;
        // SAFETY: front_ptr was derived from a heap-allocated Node<T, N> owned
        // by the OrderedSkipList that created this Iter.  The iterator holds a
        // shared borrow of that list for lifetime 'a, ensuring every node
        // remains allocated and reachable for the iterator's entire lifetime.
        // No &mut references to any node exist while this shared Iter is alive.
        let node: &'a Node<T, N> = unsafe { front_ptr.as_ref() };
        self.front = node.next();
        self.len = self.len.saturating_sub(1);
        node.value()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for Iter<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let back_ptr = self.back?;
        // SAFETY: Same provenance argument as front_ptr in next().
        // back_ptr points to a live data node for the 'a lifetime.
        let node: &'a Node<T, N> = unsafe { back_ptr.as_ref() };
        // Walk backward.  The head sentinel has no value; the filter ensures
        // `back` becomes None when we step past the first data node.
        // `len` independently prevents accessing a stale `back` pointer.
        // SAFETY: prev() returns a valid pointer into the same list allocation.
        self.back = node
            .prev()
            .filter(|p| unsafe { p.as_ref() }.value().is_some());
        self.len = self.len.saturating_sub(1);
        node.value()
    }
}

impl<T, const N: usize> ExactSizeIterator for Iter<'_, T, N> {}

impl<T, const N: usize> FusedIterator for Iter<'_, T, N> {}

// MARK: IntoIter

/// An owning iterator over the elements of an [`OrderedSkipList`], yielded in
/// sorted order.
///
/// This struct is created by the [`IntoIterator`] implementation for
/// [`OrderedSkipList`].  Iteration consumes the list.
///
/// # Examples
///
/// ```rust
/// use skiplist::ordered_skip_list::OrderedSkipList;
///
/// let mut list = OrderedSkipList::<i32>::new();
/// list.insert(1);
/// list.insert(2);
/// list.insert(3);
///
/// let mut iter = list.into_iter();
/// assert_eq!(iter.next(), Some(1));
/// assert_eq!(iter.next_back(), Some(3));
/// assert_eq!(iter.next(), Some(2));
/// assert_eq!(iter.next(), None);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IntoIter<
    T,
    const N: usize = 16,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// The remaining elements. `pop_first` / `pop_last` drive iteration;
    /// dropping [`IntoIter`] drops the remaining elements via the
    /// [`OrderedSkipList`] Drop impl.
    list: OrderedSkipList<T, N, C, G>,
}

// SAFETY: IntoIter<T, N, C, G> owns its elements.  Sending it to another
// thread is safe when T, C, and G are Send.
unsafe impl<T: Send, C: Comparator<T> + Send, G: LevelGenerator + Send, const N: usize> Send
    for IntoIter<T, N, C, G>
{
}

// SAFETY: Sharing &IntoIter is safe when T, C, and G are Sync.  Advancing the
// iterator requires &mut IntoIter, preventing concurrent mutation.
unsafe impl<T: Sync, C: Comparator<T> + Sync, G: LevelGenerator + Sync, const N: usize> Sync
    for IntoIter<T, N, C, G>
{
}

impl<T: fmt::Debug, C: Comparator<T>, G: LevelGenerator, const N: usize> fmt::Debug
    for IntoIter<T, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.list.iter()).finish()
    }
}

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> Iterator for IntoIter<T, N, C, G> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.list.pop_first()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.list.len(), Some(self.list.len()))
    }
}

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> DoubleEndedIterator
    for IntoIter<T, N, C, G>
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.list.pop_last()
    }
}

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> ExactSizeIterator
    for IntoIter<T, N, C, G>
{
}

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> FusedIterator
    for IntoIter<T, N, C, G>
{
}

// MARK: Drain

/// An owning iterator over all elements drained from an [`OrderedSkipList`].
///
/// This struct is created by the [`OrderedSkipList::drain`] method.  All
/// elements are removed from the list eagerly when `drain` is called.  The
/// removed elements are yielded by this iterator in sorted order.  The list
/// is left empty regardless of whether the `Drain` is fully consumed.
///
/// Supports both forward and backward iteration ([`DoubleEndedIterator`]).
///
/// # Examples
///
/// ```rust
/// use skiplist::ordered_skip_list::OrderedSkipList;
///
/// let mut list = OrderedSkipList::<i32>::new();
/// list.insert(3);
/// list.insert(1);
/// list.insert(2);
///
/// let drained: Vec<i32> = list.drain().collect();
/// assert_eq!(drained, [1, 2, 3]);
/// assert!(list.is_empty());
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Drain<'a, T> {
    /// The already-removed values in sorted order, collected eagerly.
    iter: std::vec::IntoIter<T>,
    /// Ties the Drain's lifetime to the &'a mut [`OrderedSkipList`] that
    /// created it, preventing the list from being used while this Drain is
    /// alive.
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

/// A lazy iterator that removes and yields elements satisfying a predicate
/// from an [`OrderedSkipList`].
///
/// This struct is created by the [`OrderedSkipList::extract_if`] method.  The
/// predicate is called once per element, in sorted order.  Elements for which
/// it returns `true` are removed and yielded; all others remain in place.
///
/// If the iterator is dropped before being fully consumed the predicate is
/// **not** called for the remaining elements; they all stay in the list and
/// the list remains fully usable.
///
/// Does **not** implement [`DoubleEndedIterator`] or [`ExactSizeIterator`].
///
/// # Examples
///
/// ```rust
/// use skiplist::ordered_skip_list::OrderedSkipList;
///
/// let mut list = OrderedSkipList::<i32>::new();
/// for i in 1..=5 {
///     list.insert(i);
/// }
///
/// let evens: Vec<i32> = list.extract_if(|x| *x % 2 == 0).collect();
/// assert_eq!(evens, [2, 4]);
/// let remaining: Vec<i32> = list.iter().copied().collect();
/// assert_eq!(remaining, [1, 3, 5]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<
    'a,
    T,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
    F = fn(&mut T) -> bool,
    const N: usize = 16,
> where
    F: FnMut(&mut T) -> bool,
{
    /// Mutable borrow of the owning list; needed to rebuild skip links on drop
    /// and to update len and tail on each removal.
    list: &'a mut OrderedSkipList<T, N, C, G>,
    /// Raw pointer to the next node to visit, or None when the iterator has
    /// been exhausted.
    current: Option<NonNull<Node<T, N>>>,
    /// Set to true the first time an element is removed. Used to skip the `$O(n)$`
    /// skip-link rebuild in [`Drop::drop`] when nothing was removed.
    any_removed: bool,
    /// User-supplied filter predicate.
    pred: F,
}

// SAFETY: ExtractIf<'a, T, C, G, F, N> yields owned T values and holds
// &'a mut OrderedSkipList<T, N, C, G>.  Sending it to another thread requires
// T: Send, C: Send, G: Send, and F: Send.
unsafe impl<T: Send, C: Comparator<T> + Send, G: LevelGenerator + Send, F: Send, const N: usize>
    Send for ExtractIf<'_, T, C, G, F, N>
where
    F: FnMut(&mut T) -> bool,
{
}

// SAFETY: Sharing &ExtractIf requires T: Sync, C: Sync, G: Sync, F: Sync.
// Advancing the iterator requires &mut ExtractIf, preventing concurrent
// mutation.
unsafe impl<T: Sync, C: Comparator<T> + Sync, G: LevelGenerator + Sync, F: Sync, const N: usize>
    Sync for ExtractIf<'_, T, C, G, F, N>
where
    F: FnMut(&mut T) -> bool,
{
}

impl<T: fmt::Debug, C: Comparator<T>, G: LevelGenerator, F, const N: usize> fmt::Debug
    for ExtractIf<'_, T, C, G, F, N>
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
            // SAFETY: nn points to a live Node<T> owned by the OrderedSkipList
            // that created this ExtractIf.  We only read through it here, and
            // &self prevents concurrent mutable access.
            let node = unsafe { nn.as_ref() };
            if let Some(v) = node.value() {
                builder.entry(v);
            }
            ptr = node.next();
        }
        builder.finish()
    }
}

impl<T, C: Comparator<T>, G: LevelGenerator, F, const N: usize> Iterator
    for ExtractIf<'_, T, C, G, F, N>
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
        loop {
            let current_nn = self.current?;
            // SAFETY: current_nn was derived from a heap-allocated Node<T>
            // owned by the OrderedSkipList that created this ExtractIf.  We
            // hold &'a mut OrderedSkipList exclusively for the iterator's
            // lifetime, ensuring every node remains allocated and non-aliased.
            // We capture next_opt before any mutation of the current node.
            unsafe {
                let current: *mut Node<T, N> = current_nn.as_ptr();
                let next_opt = (*current).next();

                let value_ref = (*current).value_mut().expect("data node has value");
                if (self.pred)(value_ref) {
                    self.current = next_opt;
                    self.any_removed = true;
                    self.list.len = self.list.len.saturating_sub(1);
                    // If this node was the tail, update the tail pointer to the
                    // predecessor data node (or None if the list is now empty).
                    if self.list.tail == Some(current_nn) {
                        // SAFETY: prev() returns a valid pointer into the same
                        // list allocation.
                        self.list.tail = (*current).prev().filter(|p| p.as_ref().value().is_some());
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

impl<T, C: Comparator<T>, G: LevelGenerator, F, const N: usize> FusedIterator
    for ExtractIf<'_, T, C, G, F, N>
where
    F: FnMut(&mut T) -> bool,
{
}

impl<T, C: Comparator<T>, G: LevelGenerator, F, const N: usize> Drop
    for ExtractIf<'_, T, C, G, F, N>
where
    F: FnMut(&mut T) -> bool,
{
    #[inline]
    fn drop(&mut self) {
        if !self.any_removed {
            // Nothing was removed; skip links are still valid.
            return;
        }
        // Rebuild all skip links in one O(n) forward pass over the prev/next
        // chain.  The prev/next chain is already correct (each `pop()` in
        // `Iterator::next` spliced out the removed node), so we only need to
        // re-derive the level-indexed skip links.
        //
        // SAFETY: &'a mut OrderedSkipList is held exclusively.  All raw
        // pointers originate from its heap allocations.
        let (_, new_tail) = unsafe { Node::filter_rebuild(self.list.head, |_| true, |_| {}) };
        self.list.tail = new_tail;
        // self.list.len is already correct: decremented in Iterator::next
        // once per removed element.
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use core::ops::Bound;

    use pretty_assertions::assert_eq;

    use super::super::OrderedSkipList;
    use crate::{comparator::FnComparator, level_generator::geometric::Geometric};

    // MARK: iter

    #[test]
    fn iter_empty() {
        let list = OrderedSkipList::<i32>::new();
        let mut iter = list.iter();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(&42));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_single_element_from_back() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        let mut iter = list.iter();
        assert_eq!(iter.next_back(), Some(&42));
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_forward_sorted_order() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(3);
        list.insert(1);
        list.insert(2);
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, [1, 2, 3]);
    }

    #[test]
    fn iter_backward_sorted_order() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(3);
        list.insert(1);
        list.insert(2);
        let collected: Vec<i32> = list.iter().copied().rev().collect();
        assert_eq!(collected, [3, 2, 1]);
    }

    #[test]
    fn iter_double_ended_alternating() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
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
    fn iter_size_hint_decrements() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
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
        let mut list = OrderedSkipList::<i32>::new();
        for i in 0..10_i32 {
            list.insert(i);
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
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_clone_yields_same_elements() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        let iter = list.iter();
        let clone = iter.clone();
        let v1: Vec<i32> = iter.copied().collect();
        let v2: Vec<i32> = clone.copied().collect();
        assert_eq!(v1, v2);
        assert_eq!(v1, [1, 2, 3]);
    }

    #[test]
    fn iter_does_not_consume_list() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        let v1: Vec<i32> = list.iter().copied().collect();
        let v2: Vec<i32> = list.iter().copied().collect();
        assert_eq!(v1, v2);
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn iter_with_duplicates() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(2);
        list.insert(1);
        list.insert(2);
        list.insert(3);
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, [1, 2, 2, 3]);
    }

    #[test]
    fn iter_custom_comparator() {
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        list.insert(3);
        list.insert(1);
        list.insert(2);
        // Iteration order is the list's internal order: 3, 2, 1 (largest first).
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, [3, 2, 1]);
    }

    #[test]
    fn into_iter_for_ref() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(3);
        list.insert(1);
        list.insert(2);
        let collected: Vec<i32> = (&list).into_iter().copied().collect();
        assert_eq!(collected, [1, 2, 3]);
    }

    // MARK: into_iter (owned)

    #[test]
    fn into_iter_forward() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(3);
        list.insert(1);
        list.insert(2);
        let collected: Vec<i32> = list.into_iter().collect();
        assert_eq!(collected, [1, 2, 3]);
    }

    #[test]
    fn into_iter_backward() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(3);
        list.insert(1);
        list.insert(2);
        let collected: Vec<i32> = list.into_iter().rev().collect();
        assert_eq!(collected, [3, 2, 1]);
    }

    #[test]
    fn into_iter_double_ended() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=4_i32 {
            list.insert(i);
        }
        let mut iter = list.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next_back(), Some(4));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next_back(), Some(3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn into_iter_empty() {
        let list = OrderedSkipList::<i32>::new();
        let mut iter = list.into_iter();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn into_iter_size_hint() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        let mut iter = list.into_iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
    }

    // MARK: range

    #[test]
    fn range_full() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let collected: Vec<i32> = list.range(..).copied().collect();
        assert_eq!(collected, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn range_included_both_ends() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let collected: Vec<i32> = list.range(2..=4).copied().collect();
        assert_eq!(collected, [2, 3, 4]);
    }

    #[test]
    fn range_excluded_end() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let collected: Vec<i32> = list.range(2..4).copied().collect();
        assert_eq!(collected, [2, 3]);
    }

    #[test]
    fn range_unbounded_start() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let collected: Vec<i32> = list.range(..=3).copied().collect();
        assert_eq!(collected, [1, 2, 3]);
    }

    #[test]
    fn range_unbounded_end() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let collected: Vec<i32> = list.range(3..).copied().collect();
        assert_eq!(collected, [3, 4, 5]);
    }

    #[test]
    fn range_single_element() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let collected: Vec<i32> = list.range(3..=3).copied().collect();
        assert_eq!(collected, [3]);
    }

    #[test]
    fn range_empty_no_match() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(5);
        let collected: Vec<i32> = list.range(2..=4).copied().collect();
        assert_eq!(collected, []);
    }

    #[test]
    fn range_beyond_list() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        let collected: Vec<i32> = list.range(10..).copied().collect();
        assert_eq!(collected, []);
    }

    #[test]
    fn range_before_list() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(5);
        list.insert(10);
        let collected: Vec<i32> = list.range(..=3).copied().collect();
        assert_eq!(collected, []);
    }

    #[test]
    fn range_with_duplicates() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        list.insert(1);
        list.insert(2);
        list.insert(2);
        list.insert(3);
        let collected: Vec<i32> = list.range(2..=2).copied().collect();
        assert_eq!(collected, [2, 2]);
    }

    #[test]
    fn range_excluded_start_duplicates() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        for i in [1, 2, 2, 2, 3] {
            list.insert(i);
        }
        // Excluded start at 2 should skip all 2s.
        let collected: Vec<i32> = list
            .range((Bound::Excluded(&2), Bound::Unbounded))
            .copied()
            .collect();
        assert_eq!(collected, [3]);
    }

    #[test]
    fn range_excluded_end_duplicates() {
        let g = Geometric::new(1, 0.5).expect("valid parameters");
        let mut list = OrderedSkipList::<i32, 1>::with_level_generator(g);
        for i in [1, 2, 2, 2, 3] {
            list.insert(i);
        }
        // Excluded end at 2 should not include any 2s.
        let collected: Vec<i32> = list.range(..2).copied().collect();
        assert_eq!(collected, [1]);
    }

    #[test]
    fn range_double_ended() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let mut iter = list.range(1..=5);
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), Some(&5));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next_back(), Some(&4));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn range_size_hint() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let mut iter = list.range(2..=4);
        assert_eq!(iter.size_hint(), (3, Some(3)));
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
    }

    #[test]
    #[expect(
        clippy::reversed_empty_ranges,
        reason = "intentionally testing invalid range detection"
    )]
    #[should_panic(expected = "range start is after range end")]
    fn range_invalid_panics() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        // 4..=2 is invalid.
        drop(list.range(4..=2));
    }

    #[test]
    fn range_empty_list() {
        let list = OrderedSkipList::<i32>::new();
        let collected: Vec<i32> = list.range(..).copied().collect();
        assert_eq!(collected, []);
    }

    // MARK: drain

    #[test]
    fn drain_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        let drained: Vec<i32> = list.drain().collect();
        assert_eq!(drained, []);
        assert!(list.is_empty());
    }

    #[test]
    fn drain_all_sorted() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(3);
        list.insert(1);
        list.insert(2);
        let drained: Vec<i32> = list.drain().collect();
        assert_eq!(drained, [1, 2, 3]);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn drain_leaves_list_empty() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 0..10_i32 {
            list.insert(i);
        }
        let _drain = list.drain();
        assert!(list.is_empty());
    }

    #[test]
    fn drain_backwards() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(3);
        list.insert(1);
        list.insert(2);
        let drained: Vec<i32> = list.drain().rev().collect();
        assert_eq!(drained, [3, 2, 1]);
    }

    #[test]
    fn drain_double_ended() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=4_i32 {
            list.insert(i);
        }
        let mut drain = list.drain();
        assert_eq!(drain.next(), Some(1));
        assert_eq!(drain.next_back(), Some(4));
        assert_eq!(drain.next(), Some(2));
        assert_eq!(drain.next_back(), Some(3));
        assert_eq!(drain.next(), None);
    }

    #[test]
    fn drain_reinsert_after() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        drop(list.drain());
        list.insert(10);
        assert_eq!(list.len(), 1);
        assert_eq!(list.first(), Some(&10));
    }

    // MARK: drain_range

    #[test]
    fn drain_range_empty_list() {
        let mut list = OrderedSkipList::<i32>::new();
        let drained: Vec<i32> = list.drain_range(1..=3).collect();
        assert!(drained.is_empty());
        assert!(list.is_empty());
    }

    #[test]
    fn drain_range_no_match() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3] {
            list.insert(i);
        }
        let drained: Vec<i32> = list.drain_range(10..=20).collect();
        assert!(drained.is_empty());
        assert_eq!(list.len(), 3);
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    fn drain_range_all() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        let drained: Vec<i32> = list.drain_range(..).collect();
        assert_eq!(drained, [1, 2, 3, 4, 5]);
        assert!(list.is_empty());
    }

    #[test]
    fn drain_range_inclusive_range() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        let drained: Vec<i32> = list.drain_range(2..=4).collect();
        assert_eq!(drained, [2, 3, 4]);
        assert_eq!(list.len(), 2);
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), [1, 5]);
    }

    #[test]
    fn drain_range_exclusive_range() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        let drained: Vec<i32> = list.drain_range(2..4).collect();
        assert_eq!(drained, [2, 3]);
        assert_eq!(list.len(), 3);
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), [1, 4, 5]);
    }

    #[test]
    fn drain_range_from_bound() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        let drained: Vec<i32> = list.drain_range(3..).collect();
        assert_eq!(drained, [3, 4, 5]);
        assert_eq!(list.len(), 2);
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), [1, 2]);
    }

    #[test]
    fn drain_range_to_bound() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        let drained: Vec<i32> = list.drain_range(..=3).collect();
        assert_eq!(drained, [1, 2, 3]);
        assert_eq!(list.len(), 2);
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), [4, 5]);
    }

    #[test]
    fn drain_range_tail_pointer_correct() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        // Drain the last two elements; tail should move to 3.
        let drained: Vec<i32> = list.drain_range(4..).collect();
        assert_eq!(drained, [4, 5]);
        assert_eq!(list.last(), Some(&3));
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn drain_range_links_consistent() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 0..20 {
            list.insert(i);
        }
        let drained: Vec<i32> = list.drain_range(5..15).collect();
        assert_eq!(drained, (5..15).collect::<Vec<_>>());
        assert_eq!(list.len(), 10);
        // Verify rank-based access and iteration are consistent.
        let got: Vec<i32> = list.iter().copied().collect();
        let expected: Vec<i32> = (0..5).chain(15..20).collect();
        assert_eq!(got, expected);
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(list.get(i), Some(&v));
        }
    }

    #[test]
    fn drain_range_with_duplicates() {
        let mut list = OrderedSkipList::<i32>::new();
        for _ in 0..3 {
            list.insert(2);
        }
        list.insert(1);
        list.insert(3);
        let drained: Vec<i32> = list.drain_range(2..=2).collect();
        assert_eq!(drained, [2, 2, 2]);
        assert_eq!(list.len(), 2);
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), [1, 3]);
    }

    #[test]
    fn drain_range_list_usable_after() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        drop(list.drain_range(2..=4));
        list.insert(10);
        assert_eq!(list.len(), 3);
        assert_eq!(list.last(), Some(&10));
    }

    #[test]
    #[should_panic(expected = "range start is after range end")]
    #[expect(
        clippy::reversed_empty_ranges,
        reason = "intentionally testing invalid range detection in drain_range"
    )]
    fn drain_range_inverted_panics() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5 {
            list.insert(i);
        }
        _ = list.drain_range(4..=2);
    }

    #[test]
    #[expect(
        clippy::reversed_empty_ranges,
        reason = "false positive: 4..=2 is well-ordered in the list's internal \
            order (largest to smallest)"
    )]
    fn drain_range_custom_comparator() {
        // Largest-first ordering: list is [5, 4, 3, 2, 1].
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        for i in 1..=5 {
            list.insert(i);
        }
        let drained: Vec<i32> = list.drain_range(4..=2).collect();
        assert_eq!(drained, [4, 3, 2]);
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), [5, 1]);
    }

    // MARK: extract_if

    #[test]
    fn extract_if_remove_none() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let removed: Vec<i32> = list.extract_if(|_| false).collect();
        assert_eq!(removed, []);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn extract_if_remove_all() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let removed: Vec<i32> = list.extract_if(|_| true).collect();
        assert_eq!(removed, [1, 2, 3, 4, 5]);
        assert!(list.is_empty());
    }

    #[test]
    #[expect(
        clippy::integer_division_remainder_used,
        reason = "clearer to express the intent of extracting even numbers"
    )]
    fn extract_if_remove_evens() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let evens: Vec<i32> = list.extract_if(|x| *x % 2 == 0).collect();
        assert_eq!(evens, [2, 4]);
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [1, 3, 5]);
    }

    #[test]
    #[expect(
        clippy::integer_division_remainder_used,
        reason = "clearer to express the intent of extracting even numbers"
    )]
    fn extract_if_drop_before_exhaustion() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        {
            // Only consume the first yielded element, then drop.
            let mut iter = list.extract_if(|x| *x % 2 == 0);
            assert_eq!(iter.next(), Some(2)); // remove 2
            // iter drops here; predicate not called for 3, 4, 5
        }
        // 4 should still be in the list (predicate was not called for it).
        let remaining: Vec<i32> = list.iter().copied().collect();
        assert_eq!(remaining, [1, 3, 4, 5]);
    }

    #[test]
    fn extract_if_list_still_valid_after_drop() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=4_i32 {
            list.insert(i);
        }
        {
            let mut iter = list.extract_if(|x| *x == 2);
            assert_eq!(iter.next(), Some(2));
        }
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, [1, 3, 4]);
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn extract_if_empty_list() {
        let mut list = OrderedSkipList::<i32>::new();
        let removed: Vec<i32> = list.extract_if(|_| true).collect();
        assert_eq!(removed, []);
    }

    #[test]
    fn extract_if_size_hint() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in 1..=5_i32 {
            list.insert(i);
        }
        let iter = list.extract_if(|_| false);
        // Lower bound is always 0 (predicate outcome unknown).
        assert_eq!(iter.size_hint().0, 0);
    }
}
