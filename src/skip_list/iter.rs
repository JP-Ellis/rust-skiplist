//! Iteration support for [`SkipList`](super::SkipList): `iter`, `iter_mut`,
//! `range`, `range_mut`, `drain`, `extract_if`, all iterator types, and
//! [`IntoIterator`] implementations.

use core::{
    fmt,
    iter::FusedIterator,
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    ptr::NonNull,
};

use crate::{
    level_generator::{LevelGenerator, geometric::Geometric},
    node::Node,
    skip_list::SkipList,
};

impl<T, G: LevelGenerator, const N: usize> SkipList<T, N, G> {
    // MARK: Iteration

    /// Returns an iterator over shared references to the elements of the list,
    /// from front to back.
    ///
    /// The iterator also supports [`DoubleEndedIterator`], allowing traversal
    /// in reverse order.  Advancing from both ends toward the middle is also
    /// supported: calls to [`Iterator::next`] and
    /// [`DoubleEndedIterator::next_back`] can be interleaved freely.
    ///
    /// This operation is `$O(1)$`.
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
    pub fn iter(&self) -> Iter<'_, T, N> {
        Iter {
            // SAFETY: self.head is a valid, exclusively-owned head sentinel.
            front: unsafe { self.head.as_ref() }.next(),
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
    /// This operation is `$O(1)$`.
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
    pub fn iter_mut(&mut self) -> IterMut<'_, T, N> {
        IterMut {
            // SAFETY: self.head is a valid, exclusively-owned head sentinel.
            front: unsafe { self.head.as_ref().next() },
            back: self.tail,
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over shared references to elements within the
    /// given index range.
    ///
    /// The iterator supports [`DoubleEndedIterator`], [`ExactSizeIterator`],
    /// and [`FusedIterator`].  Setting up the iterator runs in `$O(\log n)$`.
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
    pub fn range<R: RangeBounds<usize>>(&self, range: R) -> Iter<'_, T, N> {
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
    /// and [`FusedIterator`].  Setting up the iterator runs in `$O(\log n)$`.
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
    pub fn range_mut<R: RangeBounds<usize>>(&mut self, range: R) -> IterMut<'_, T, N> {
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

        // Track the position of each node as the closure visits them in order.
        let mut current_index: usize = 0;

        // SAFETY: All raw pointers come from heap allocations owned by this
        // SkipList.  We hold &mut self.  The keep closure does not structurally
        // modify the chain; the on_drop closure pops the node and extracts the
        // value before the box is dropped.
        let (new_rank, new_tail) = unsafe {
            Node::filter_rebuild(
                self.head,
                |_cur| {
                    let in_range = current_index >= start && current_index < end;
                    current_index = current_index.saturating_add(1);
                    !in_range
                },
                |mut boxed| {
                    drained.push(boxed.take_value().expect("data node has a value"));
                },
            )
        };
        self.tail = new_tail;
        self.len = new_rank;

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
    /// the predicate is **not** called for the remaining elements: they all
    /// stay in the list.  The list remains valid and fully usable after the
    /// iterator is dropped.
    ///
    /// This operation is `$O(n)$` for a full traversal.
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
    pub fn extract_if<F>(&mut self, pred: F) -> ExtractIf<'_, T, G, F, N>
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

impl<'a, T, G: LevelGenerator, const N: usize> IntoIterator for &'a SkipList<T, N, G> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, G: LevelGenerator, const N: usize> IntoIterator for &'a mut SkipList<T, N, G> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, G: LevelGenerator, const N: usize> IntoIterator for SkipList<T, N, G> {
    type Item = T;
    type IntoIter = IntoIter<T, N, G>;

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
pub struct Iter<'a, T, const N: usize = 16> {
    /// Pointer to the next element to yield from the front, or `None` when
    /// the iterator is exhausted or the list was empty.
    front: Option<NonNull<Node<T, N>>>,
    /// Pointer to the next element to yield from the back, or `None` when
    /// the iterator is exhausted or the list was empty.
    back: Option<NonNull<Node<T, N>>>,
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
unsafe impl<T: Sync, const N: usize> Send for Iter<'_, T, N> {}

// SAFETY: Sharing &Iter<'a, T> across threads is safe when T: Sync.
// Concurrent callers need &mut Iter to advance it, so data races on the
// iterator's own fields are prevented by the requirement for exclusive
// access through &mut.
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
        // by the SkipList that created this Iter.  The iterator holds a
        // shared borrow of that list for lifetime 'a, ensuring every node
        // remains allocated and reachable for the iterator's entire lifetime.
        // No &mut references to any node exist while this shared Iter is
        // alive.
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
pub struct IterMut<'a, T, const N: usize = 16> {
    /// Pointer to the next element to yield from the front, or `None` when
    /// the iterator is exhausted or the list was empty.
    front: Option<NonNull<Node<T, N>>>,
    /// Pointer to the next element to yield from the back, or `None` when
    /// the iterator is exhausted or the list was empty.
    back: Option<NonNull<Node<T, N>>>,
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
unsafe impl<T: Send, const N: usize> Send for IterMut<'_, T, N> {}

// SAFETY: Sharing &IterMut<'a, T> across threads is safe when T: Sync.
// Advancing the iterator requires &mut IterMut, so concurrent advancement
// is prevented by the requirement for exclusive access.
unsafe impl<T: Sync, const N: usize> Sync for IterMut<'_, T, N> {}

impl<T: fmt::Debug, const N: usize> fmt::Debug for IterMut<'_, T, N> {
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
            node_ptr = node.next();
            remaining = remaining.saturating_sub(1);
        }
        builder.finish()
    }
}

impl<'a, T, const N: usize> Iterator for IterMut<'a, T, N> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let mut front_ptr = self.front?;
        // SAFETY: front_ptr was derived from a heap-allocated Node<T, N> owned
        // by the SkipList that created this IterMut.  The iterator holds an
        // exclusive borrow of that list for lifetime 'a, ensuring every node
        // remains allocated and non-aliased for the iterator's entire
        // lifetime.  We advance self.front before returning, so no two calls
        // to next() can yield a reference to the same node.
        let node: &'a mut Node<T, N> = unsafe { front_ptr.as_mut() };
        self.front = node.next();
        self.len = self.len.saturating_sub(1);
        node.value_mut()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for IterMut<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        let mut back_ptr = self.back?;
        // SAFETY: Same provenance argument as front_ptr in next().
        // back_ptr points to a live data node for the 'a lifetime, and no
        // other mutable reference to it exists while this IterMut is alive.
        let node: &'a mut Node<T, N> = unsafe { back_ptr.as_mut() };
        // Walk backward.  The head sentinel has no value; the filter ensures
        // `back` becomes None when we step past the first data node.
        // `len` independently prevents accessing a stale `back` pointer.
        // SAFETY: prev() returns a valid pointer into the same list allocation.
        self.back = node
            .prev()
            .filter(|p| unsafe { p.as_ref() }.value().is_some());
        self.len = self.len.saturating_sub(1);
        node.value_mut()
    }
}

impl<T, const N: usize> ExactSizeIterator for IterMut<'_, T, N> {}

impl<T, const N: usize> FusedIterator for IterMut<'_, T, N> {}

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
pub struct IntoIter<T, const N: usize = 16, G: LevelGenerator = Geometric> {
    /// The remaining elements.  `pop_front` / `pop_back` drive iteration;
    /// dropping `IntoIter` drops the remaining elements via the
    /// [`SkipList::Drop`] impl.
    list: SkipList<T, N, G>,
}

// SAFETY: IntoIter<T, N, G> owns its elements.  Sending it to another thread
// is safe when T is Send (the elements will be accessed on the new thread).
// G: LevelGenerator is Send-safe by the same argument.
unsafe impl<T: Send, G: LevelGenerator + Send, const N: usize> Send for IntoIter<T, N, G> {}

// SAFETY: Sharing &IntoIter<T, N, G> is safe when T: Sync and G: Sync.
// Advancing the iterator requires &mut IntoIter, which prevents concurrent
// mutation through shared references.
unsafe impl<T: Sync, G: LevelGenerator + Sync, const N: usize> Sync for IntoIter<T, N, G> {}

impl<T: fmt::Debug, G: LevelGenerator, const N: usize> fmt::Debug for IntoIter<T, N, G> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.list.iter()).finish()
    }
}

impl<T, G: LevelGenerator, const N: usize> Iterator for IntoIter<T, N, G> {
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

impl<T, G: LevelGenerator, const N: usize> DoubleEndedIterator for IntoIter<T, N, G> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.list.pop_back()
    }
}

impl<T, G: LevelGenerator, const N: usize> ExactSizeIterator for IntoIter<T, N, G> {}

impl<T, G: LevelGenerator, const N: usize> FusedIterator for IntoIter<T, N, G> {}

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
/// **not** called for the remaining elements: they all stay in the list and
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
pub struct ExtractIf<
    'a,
    T,
    G: LevelGenerator = Geometric,
    F = fn(&mut T) -> bool,
    const N: usize = 16,
> where
    F: FnMut(&mut T) -> bool,
{
    /// Mutable borrow of the owning list (needed to rebuild skip links on
    /// drop and to update `len` and `tail` on each removal).
    list: &'a mut SkipList<T, N, G>,
    /// Raw pointer to the next node to visit, or `None` when the iterator
    /// has been exhausted.
    current: Option<NonNull<Node<T, N>>>,
    /// Set to `true` the first time an element is removed.  Used to skip
    /// the `$O(n)$` skip-link rebuild in `Drop::drop` when nothing was removed.
    any_removed: bool,
    /// User-supplied filter predicate.
    pred: F,
}

// SAFETY: ExtractIf<'a, T, G, F> yields owned T values and holds
// &'a mut SkipList<T, G>.  Sending it to another thread requires
// T: Send, G: Send, and F: Send.
unsafe impl<T: Send, G: LevelGenerator + Send, F: Send, const N: usize> Send
    for ExtractIf<'_, T, G, F, N>
where
    F: FnMut(&mut T) -> bool,
{
}

// SAFETY: Sharing &ExtractIf<'a, T, G, F, N> requires T: Sync, G: Sync, F: Sync.
// Advancing the iterator requires &mut ExtractIf, preventing concurrent mutation.
unsafe impl<T: Sync, G: LevelGenerator + Sync, F: Sync, const N: usize> Sync
    for ExtractIf<'_, T, G, F, N>
where
    F: FnMut(&mut T) -> bool,
{
}

impl<T: fmt::Debug, G: LevelGenerator, F, const N: usize> fmt::Debug for ExtractIf<'_, T, G, F, N>
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
            ptr = node.next();
        }
        builder.finish()
    }
}

impl<T, G: LevelGenerator, F, const N: usize> Iterator for ExtractIf<'_, T, G, F, N>
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
                let current: *mut Node<T, N> = current_nn.as_ptr();
                let next_opt = (*current).next();

                let value_ref = (*current).value_mut().expect("data node has value");
                if (self.pred)(value_ref) {
                    self.current = next_opt;
                    self.any_removed = true;
                    self.list.len = self.list.len.saturating_sub(1);
                    // If this node was the tail, update the tail pointer to
                    // the predecessor data node (or None if the list is now
                    // empty).
                    if self.list.tail == Some(current_nn) {
                        // SAFETY: prev() returns a valid pointer into the
                        // same list allocation.
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

impl<T, G: LevelGenerator, F, const N: usize> FusedIterator for ExtractIf<'_, T, G, F, N> where
    F: FnMut(&mut T) -> bool
{
}

impl<T, G: LevelGenerator, F, const N: usize> Drop for ExtractIf<'_, T, G, F, N>
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
        // SAFETY: &'a mut SkipList is held exclusively.  All raw pointers
        // originate from its heap allocations.
        let (_, new_tail) = unsafe { Node::filter_rebuild(self.list.head, |_| true, |_| {}) };
        self.list.tail = new_tail;
        // self.list.len is already correct: decremented in Iterator::next
        // once per removed element.
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::SkipList;

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
        // Exhausted: subsequent calls must all return None (FusedIterator)
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
            let _iter = list.iter_mut();
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
        drop(list.drain(3..7));
        assert_eq!(list.len(), 6);
    }

    #[test]
    fn drain_links_consistent_after() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10 {
            list.push_back(i);
        }
        drop(list.drain(3..7));
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
        drop(list.drain(3..1));
    }

    #[test]
    #[should_panic(expected = "drain range end")]
    fn drain_panics_end_gt_len() {
        let mut list = SkipList::<i32>::new();
        for i in 0..5 {
            list.push_back(i);
        }
        drop(list.drain(0..10));
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
        // Predicate receives &mut T, so verify it can mutate values before keeping.
        let mut list = SkipList::<i32>::new();
        for i in 1..=4 {
            list.push_back(i);
        }
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
        drop(list.range(3..1));
    }

    #[test]
    #[should_panic(expected = "range end")]
    fn range_panic_end_gt_len() {
        let mut list = SkipList::<i32>::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        drop(list.range(0..10));
    }
}
