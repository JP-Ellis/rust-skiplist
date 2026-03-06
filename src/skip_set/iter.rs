//! Iteration support for [`SkipSet`](super::SkipSet): `iter`, `into_iter`,
//! `range`, `drain`, `extract_if`, all iterator types, and [`IntoIterator`]
//! implementations.
//!
//! Note: `IterMut` is intentionally absent.  Mutating an element in place
//! could violate the sort-order / uniqueness invariants, so only shared
//! (`&T`) iteration is exposed.

use core::{
    cmp::Ordering,
    fmt,
    iter::FusedIterator,
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    ptr::NonNull,
};

// Re-export the read-only iterator types from `ordered_skip_list::iter`.
// They carry no `C` or `G` type parameter and are identical for any set.
pub use crate::ordered_skip_list::{Drain, Iter};
use crate::{
    comparator::{Comparator, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    node::Node,
    ordered_skip_list::IntoIter as OslIntoIter,
    skip_set::SkipSet,
};

// MARK: IntoIter

/// An owning iterator over the elements of a [`SkipSet`], yielded in sorted
/// order.
///
/// This struct is created by the `into_iter` method on [`SkipSet`] (provided
/// by the [`IntoIterator`] trait).
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::SkipSet;
///
/// let mut set = SkipSet::<i32>::new();
/// for v in [3, 1, 2] { set.insert(v); }
///
/// let collected: Vec<i32> = set.into_iter().collect();
/// assert_eq!(collected, [1, 2, 3]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IntoIter<
    T,
    const N: usize = 16,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// Delegates to the ordered-skip-list owning iterator.
    inner: OslIntoIter<T, N, C, G>,
}

// SAFETY: `IntoIter` owns all elements.  Sending it to another thread
// requires `T: Send`, `C: Send`, and `G: Send`.
unsafe impl<T: Send, C: Comparator<T> + Send, G: LevelGenerator + Send, const N: usize> Send
    for IntoIter<T, N, C, G>
{
}

// SAFETY: Sharing `&IntoIter` requires `T: Sync`, `C: Sync`, and `G: Sync`.
// Advancing the iterator requires `&mut IntoIter`, preventing concurrent
// mutation.
unsafe impl<T: Sync, C: Comparator<T> + Sync, G: LevelGenerator + Sync, const N: usize> Sync
    for IntoIter<T, N, C, G>
{
}

impl<T: fmt::Debug, C: Comparator<T>, G: LevelGenerator, const N: usize> fmt::Debug
    for IntoIter<T, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoIter").finish_non_exhaustive()
    }
}

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> Iterator for IntoIter<T, N, C, G> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> DoubleEndedIterator
    for IntoIter<T, N, C, G>
{
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.inner.next_back()
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

// MARK: ExtractIf

/// A lazy iterator that removes and yields elements from a [`SkipSet`]
/// satisfying a predicate, optionally restricted to a value range.
///
/// This struct is created by the [`SkipSet::extract_if`] method.  The
/// predicate is called once per in-range element, in sorted order.  Elements
/// for which the predicate returns `true` are removed and yielded; all others
/// remain in the set.
///
/// If the iterator is dropped before being fully consumed, the predicate is
/// **not** called for the remaining elements: they all stay in the set and
/// the set remains fully usable.
///
/// Does **not** implement [`DoubleEndedIterator`] or [`ExactSizeIterator`].
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::SkipSet;
///
/// let mut set = SkipSet::<i32>::new();
/// for v in 1..=5 { set.insert(v); }
///
/// let evens: Vec<i32> = set.extract_if(.., |x| x % 2 == 0).collect();
/// assert_eq!(evens, [2, 4]);
/// let remaining: Vec<i32> = set.iter().copied().collect();
/// assert_eq!(remaining, [1, 3, 5]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<
    'a,
    T,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
    R: RangeBounds<T> = core::ops::RangeFull,
    F = fn(&T) -> bool,
    const N: usize = 16,
> where
    F: FnMut(&T) -> bool,
{
    /// Mutable borrow of the owning set.
    set: &'a mut SkipSet<T, N, C, G>,
    /// Raw pointer to the next node to visit, or `None` when exhausted.
    current: Option<NonNull<Node<T, N>>>,
    /// Range restriction: elements below lo are skipped, elements above hi
    /// cause the iterator to stop.
    range: R,
    /// Set to `true` once the iterator has passed the upper bound.
    past_hi: bool,
    /// Set to `true` the first time an element is removed.  Used to skip the
    /// `$O(n)$` skip-link rebuild in `Drop::drop` when nothing was removed.
    any_removed: bool,
    /// User-supplied filter predicate.
    pred: F,
    /// Marks the iterator as invariant over `T` and tied to the `'a` borrow.
    _marker: PhantomData<&'a mut T>,
}

// SAFETY: `ExtractIf` yields owned `T` values and holds `&'a mut SkipSet`.
// Sending it to another thread requires `T`, `C`, `G`, `R`, and `F` to be
// `Send`.
unsafe impl<
    T: Send,
    C: Comparator<T> + Send,
    G: LevelGenerator + Send,
    R: RangeBounds<T> + Send,
    F: Send,
    const N: usize,
> Send for ExtractIf<'_, T, C, G, R, F, N>
where
    F: FnMut(&T) -> bool,
{
}

// SAFETY: Sharing `&ExtractIf` requires `T`, `C`, `G`, `R`, and `F` to be
// `Sync`.  Advancing the iterator requires `&mut ExtractIf`.
unsafe impl<
    T: Sync,
    C: Comparator<T> + Sync,
    G: LevelGenerator + Sync,
    R: RangeBounds<T> + Sync,
    F: Sync,
    const N: usize,
> Sync for ExtractIf<'_, T, C, G, R, F, N>
where
    F: FnMut(&T) -> bool,
{
}

impl<T: fmt::Debug, C: Comparator<T>, G: LevelGenerator, R, F, const N: usize> fmt::Debug
    for ExtractIf<'_, T, C, G, R, F, N>
where
    R: RangeBounds<T>,
    F: FnMut(&T) -> bool,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display the unvisited elements still reachable from `current`.
        let mut builder = f.debug_list();
        let mut ptr = self.current;
        while let Some(nn) = ptr {
            // SAFETY: nn points to a live Node owned by the SkipSet that
            // created this ExtractIf.  We only read through it here, and &self
            // prevents concurrent mutable access.
            let node = unsafe { nn.as_ref() };
            if let Some(v) = node.value() {
                builder.entry(v);
            }
            ptr = node.next();
        }
        builder.finish()
    }
}

impl<T, C: Comparator<T>, G: LevelGenerator, R, F, const N: usize> Iterator
    for ExtractIf<'_, T, C, G, R, F, N>
where
    R: RangeBounds<T>,
    F: FnMut(&T) -> bool,
{
    type Item = T;

    #[expect(
        clippy::unwrap_in_result,
        clippy::expect_used,
        reason = "`value()` and `take_value()` return `None` only for the head \
                  sentinel, which is never reachable via the data-node walk; the \
                  `expect` fires only on invariant violations"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-pointer dereference, value(), tail-update, and pop() all \
                  touch provably disjoint heap nodes; splitting across blocks \
                  would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    fn next(&mut self) -> Option<T> {
        loop {
            if self.past_hi {
                return None;
            }
            let current_nn = self.current?;
            let current_ptr: *mut Node<T, N> = current_nn.as_ptr();

            // Phase 1: determine what to do.
            // Use an inner block so that `cmp` and `value_ref` (both
            // borrows) are provably dropped before the mutable operations
            // in Phase 2.
            let (next_opt, do_skip, do_stop, do_remove) = {
                // SAFETY: current_nn was derived from a heap-allocated
                // Node<T> owned by the SkipSet that created this
                // ExtractIf.  We hold &'a mut SkipSet exclusively, so no
                // other live reference to these nodes exists.
                let value_ref: &T = unsafe { (*current_ptr).value() }.expect("data node has value");
                // SAFETY: Same provenance as value_ref above; current_ptr still valid.
                let next_opt = unsafe { (*current_ptr).next() };
                let cmp = self.set.inner.comparator();

                let after_lo = match self.range.start_bound() {
                    Bound::Unbounded => true,
                    Bound::Included(lo) => cmp.compare(value_ref, lo) != Ordering::Less,
                    Bound::Excluded(lo) => cmp.compare(value_ref, lo) == Ordering::Greater,
                };
                let in_hi = match self.range.end_bound() {
                    Bound::Unbounded => true,
                    Bound::Included(hi) => cmp.compare(value_ref, hi) != Ordering::Greater,
                    Bound::Excluded(hi) => cmp.compare(value_ref, hi) == Ordering::Less,
                };
                let do_remove = after_lo && in_hi && (self.pred)(value_ref);

                (next_opt, !after_lo, after_lo && !in_hi, do_remove)
                // `cmp` and `value_ref` dropped here.
            };

            // Phase 2: act on the decision made above.
            if do_skip {
                self.current = next_opt;
                continue;
            }
            if do_stop {
                self.past_hi = true;
                return None;
            }
            self.current = next_opt;
            if do_remove {
                self.any_removed = true;
                // SAFETY: current_ptr is still valid (the node has not
                // been freed); we hold exclusive access via &'a mut SkipSet.
                unsafe {
                    self.set.inner.decrement_len();
                    self.set.inner.update_tail_before_pop(current_nn);
                    let mut boxed = (*current_ptr).pop();
                    return Some(boxed.take_value().expect("data node has value"));
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // The predicate outcome is unknown; at most all remaining elements
        // could be extracted.
        (0, Some(self.set.len()))
    }
}

impl<T, C: Comparator<T>, G: LevelGenerator, R, F, const N: usize> FusedIterator
    for ExtractIf<'_, T, C, G, R, F, N>
where
    R: RangeBounds<T>,
    F: FnMut(&T) -> bool,
{
}

impl<T, C: Comparator<T>, G: LevelGenerator, R, F, const N: usize> Drop
    for ExtractIf<'_, T, C, G, R, F, N>
where
    R: RangeBounds<T>,
    F: FnMut(&T) -> bool,
{
    #[inline]
    fn drop(&mut self) {
        if !self.any_removed {
            // Nothing was removed; skip links are still valid.
            return;
        }
        // Rebuild all skip links in one O(n) forward pass over the
        // prev/next chain.  Each `pop()` in `Iterator::next` maintained the
        // prev/next chain; we only need to re-derive the level-indexed links.
        //
        // SAFETY: &'a mut SkipSet is held exclusively.  All raw pointers
        // originate from its heap allocations.
        unsafe { self.set.inner.rebuild_skip_links() };
    }
}

// MARK: Iteration methods on SkipSet

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> SkipSet<T, N, C, G> {
    /// Returns an iterator over shared references to the elements of the set,
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
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// for v in [3, 1, 2] { set.insert(v); }
    ///
    /// let forward: Vec<i32> = set.iter().copied().collect();
    /// assert_eq!(forward, [1, 2, 3]);
    ///
    /// let reverse: Vec<i32> = set.iter().copied().rev().collect();
    /// assert_eq!(reverse, [3, 2, 1]);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T, N> {
        self.inner.iter()
    }

    /// Returns an iterator over shared references to elements whose values
    /// fall within the given range, yielded in sorted order.
    ///
    /// All standard Rust range expressions (`..`, `a..`, `..b`, `..=b`,
    /// `a..b`, `a..=b`) are accepted.  Finding the start and end nodes is
    /// `$O(\log n)$`; yielding each element is `$O(1)$`.
    ///
    /// # Panics
    ///
    /// Panics if the lower bound is greater than the upper bound according to
    /// the set's comparator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// for v in 1..=5 { set.insert(v); }
    ///
    /// let slice: Vec<i32> = set.range(2..=4).copied().collect();
    /// assert_eq!(slice, [2, 3, 4]);
    /// ```
    #[inline]
    pub fn range<R: RangeBounds<T>>(&self, range: R) -> Iter<'_, T, N> {
        self.inner.range(range)
    }

    /// Clears the set, returning all elements in sorted order as a
    /// [`Drain`] iterator.
    ///
    /// All elements are removed regardless of whether the iterator is
    /// consumed.  The set is empty once [`drain`](Self::drain) returns.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// for v in [3, 1, 2] { set.insert(v); }
    ///
    /// let drained: Vec<i32> = set.drain().collect();
    /// assert_eq!(drained, [1, 2, 3]);
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, T> {
        self.inner.drain()
    }

    /// Creates a lazy iterator that removes and yields elements in the given
    /// `range` for which `pred` returns `true`.
    ///
    /// Elements for which `pred` returns `false` stay in the set.  Elements
    /// outside `range` are never passed to `pred` and always remain.  If the
    /// iterator is dropped before being fully consumed, `pred` is not called
    /// for the remaining in-range elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// for v in 1..=6 { set.insert(v); }
    ///
    /// // Remove even numbers in [2, 5].
    /// let removed: Vec<i32> = set.extract_if(2..=5, |x| x % 2 == 0).collect();
    /// assert_eq!(removed, [2, 4]);
    /// let remaining: Vec<i32> = set.iter().copied().collect();
    /// assert_eq!(remaining, [1, 3, 5, 6]);
    /// ```
    #[inline]
    pub fn extract_if<R: RangeBounds<T>, F: FnMut(&T) -> bool>(
        &mut self,
        range: R,
        pred: F,
    ) -> ExtractIf<'_, T, C, G, R, F, N> {
        // Start just past the head sentinel (first data node).
        // SAFETY: self.inner.head_ptr() is always a valid head sentinel.
        let current = unsafe { self.inner.head_ptr().as_ref() }.next();
        ExtractIf {
            set: self,
            current,
            range,
            past_hi: false,
            any_removed: false,
            pred,
            _marker: PhantomData,
        }
    }
}

// MARK: IntoIterator

impl<'a, T, C: Comparator<T>, G: LevelGenerator, const N: usize> IntoIterator
    for &'a SkipSet<T, N, C, G>
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> IntoIterator for SkipSet<T, N, C, G> {
    type Item = T;
    type IntoIter = IntoIter<T, N, C, G>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.inner.into_iter(),
        }
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::{ExtractIf, IntoIter, SkipSet};
    use crate::comparator::FnComparator;

    // --- helpers ---

    fn make_set(values: &[i32]) -> SkipSet<i32> {
        let mut s = SkipSet::new();
        for &v in values {
            s.insert(v);
        }
        s
    }

    fn to_vec(set: &SkipSet<i32>) -> Vec<i32> {
        set.iter().copied().collect()
    }

    // MARK: iter

    #[test]
    fn iter_empty() {
        let set = make_set(&[]);
        let v: Vec<i32> = set.iter().copied().collect();
        assert!(v.is_empty());
    }

    #[test]
    fn iter_single() {
        let set = make_set(&[42]);
        assert_eq!(set.iter().copied().collect::<Vec<_>>(), [42]);
    }

    #[test]
    fn iter_sorted() {
        let set = make_set(&[3, 1, 4, 1, 5]);
        assert_eq!(set.iter().copied().collect::<Vec<_>>(), [1, 3, 4, 5]);
    }

    #[test]
    fn iter_reverse() {
        let set = make_set(&[3, 1, 2]);
        assert_eq!(set.iter().copied().rev().collect::<Vec<_>>(), [3, 2, 1]);
    }

    #[test]
    fn iter_double_ended_meet_in_middle() {
        let set = make_set(&[1, 2, 3, 4, 5]);
        let mut it = set.iter().copied();
        assert_eq!(it.next(), Some(1));
        assert_eq!(it.next_back(), Some(5));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next_back(), Some(4));
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next_back(), None);
        assert_eq!(it.next(), None);
    }

    // MARK: range

    #[test]
    fn range_full() {
        let set = make_set(&[1, 2, 3, 4, 5]);
        assert_eq!(set.range(..).copied().collect::<Vec<_>>(), [1, 2, 3, 4, 5]);
    }

    #[test]
    fn range_inclusive() {
        let set = make_set(&[1, 2, 3, 4, 5]);
        assert_eq!(set.range(2..=4).copied().collect::<Vec<_>>(), [2, 3, 4]);
    }

    #[test]
    fn range_exclusive() {
        let set = make_set(&[1, 2, 3, 4, 5]);
        assert_eq!(set.range(2..4).copied().collect::<Vec<_>>(), [2, 3]);
    }

    #[test]
    fn range_empty() {
        let set = make_set(&[1, 2, 3]);
        assert!(set.range(5..10).copied().collect::<Vec<i32>>().is_empty());
    }

    // MARK: drain

    #[test]
    fn drain_empty() {
        let mut set = make_set(&[]);
        let v: Vec<i32> = set.drain().collect();
        assert!(v.is_empty());
        assert!(set.is_empty());
    }

    #[test]
    fn drain_all() {
        let mut set = make_set(&[3, 1, 2]);
        let v: Vec<i32> = set.drain().collect();
        assert_eq!(v, [1, 2, 3]);
        assert!(set.is_empty());
    }

    #[test]
    fn drain_set_empty_after_partial_consume() {
        let mut set = make_set(&[1, 2, 3]);
        let d = set.drain();
        // drain eagerly removes all elements; dropping the iterator without
        // consuming does not put them back.
        drop(d);
        assert!(set.is_empty());
    }

    // MARK: into_iter (SkipSet)

    #[test]
    fn into_iter_forward() {
        let set = make_set(&[3, 1, 2]);
        let v: Vec<i32> = set.into_iter().collect();
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn into_iter_reverse() {
        let set = make_set(&[3, 1, 2]);
        let v: Vec<i32> = set.into_iter().rev().collect();
        assert_eq!(v, [3, 2, 1]);
    }

    #[test]
    fn into_iter_empty() {
        let set = make_set(&[]);
        let v: Vec<i32> = set.into_iter().collect();
        assert!(v.is_empty());
    }

    #[test]
    fn into_iter_size_hint() {
        let set = make_set(&[1, 2, 3]);
        let mut it = set.into_iter();
        assert_eq!(it.size_hint(), (3, Some(3)));
        it.next();
        assert_eq!(it.size_hint(), (2, Some(2)));
    }

    // MARK: IntoIter: Debug

    #[test]
    fn into_iter_debug_non_exhaustive() {
        let set = make_set(&[1]);
        let it = set.into_iter();
        let s = format!("{it:?}");
        assert!(s.contains("IntoIter"));
    }

    // MARK: into_iter (&SkipSet)

    #[test]
    fn ref_into_iter_forward() {
        let set = make_set(&[3, 1, 2]);
        let v: Vec<i32> = (&set).into_iter().copied().collect();
        assert_eq!(v, [1, 2, 3]);
    }

    // MARK: IntoIter type check

    #[test]
    fn into_iter_is_correct_type() {
        let set = make_set(&[1, 2, 3]);
        let _it: IntoIter<i32> = set.into_iter();
    }

    // MARK: extract_if: basic

    #[test]
    fn extract_if_empty_set() {
        let mut set = make_set(&[]);
        let removed: Vec<i32> = set.extract_if(.., |_| true).collect();
        assert!(removed.is_empty());
        assert!(set.is_empty());
    }

    #[test]
    fn extract_if_none_match() {
        let mut set = make_set(&[1, 2, 3]);
        let removed: Vec<i32> = set.extract_if(.., |_| false).collect();
        assert!(removed.is_empty());
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    #[test]
    fn extract_if_all_match() {
        let mut set = make_set(&[1, 2, 3]);
        let removed: Vec<i32> = set.extract_if(.., |_| true).collect();
        assert_eq!(removed, [1, 2, 3]);
        assert!(set.is_empty());
    }

    #[test]
    #[expect(
        clippy::integer_division_remainder_used,
        reason = "test intent is clearest with %"
    )]
    fn extract_if_even() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        let removed: Vec<i32> = set.extract_if(.., |x| x % 2 == 0).collect();
        assert_eq!(removed, [2, 4]);
        assert_eq!(to_vec(&set), [1, 3, 5]);
    }

    #[test]
    #[expect(
        clippy::integer_division_remainder_used,
        reason = "test intent is clearest with %"
    )]
    fn extract_if_odd() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        let removed: Vec<i32> = set.extract_if(.., |x| x % 2 != 0).collect();
        assert_eq!(removed, [1, 3, 5]);
        assert_eq!(to_vec(&set), [2, 4]);
    }

    // MARK: extract_if: range restrictions

    #[test]
    #[expect(
        clippy::integer_division_remainder_used,
        reason = "test intent is clearest with %"
    )]
    fn extract_if_range_inclusive() {
        let mut set = make_set(&[1, 2, 3, 4, 5, 6]);
        let removed: Vec<i32> = set.extract_if(2..=5, |x| x % 2 == 0).collect();
        assert_eq!(removed, [2, 4]);
        assert_eq!(to_vec(&set), [1, 3, 5, 6]);
    }

    #[test]
    fn extract_if_range_exclusive() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        let removed: Vec<i32> = set.extract_if(2..5, |_| true).collect();
        assert_eq!(removed, [2, 3, 4]);
        assert_eq!(to_vec(&set), [1, 5]);
    }

    #[test]
    fn extract_if_range_from() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        let removed: Vec<i32> = set.extract_if(3.., |_| true).collect();
        assert_eq!(removed, [3, 4, 5]);
        assert_eq!(to_vec(&set), [1, 2]);
    }

    #[test]
    fn extract_if_range_to_inclusive() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        let removed: Vec<i32> = set.extract_if(..=3, |_| true).collect();
        assert_eq!(removed, [1, 2, 3]);
        assert_eq!(to_vec(&set), [4, 5]);
    }

    #[test]
    fn extract_if_range_empty_yields_nothing() {
        let mut set = make_set(&[1, 2, 3]);
        let removed: Vec<i32> = set.extract_if(10..20, |_| true).collect();
        assert!(removed.is_empty());
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    // MARK: extract_if: drop before exhaustion

    #[test]
    fn extract_if_drop_early_preserves_remaining() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        {
            let mut it = set.extract_if(.., |_| true);
            assert_eq!(it.next(), Some(1));
            assert_eq!(it.next(), Some(2));
        }
        // Elements 3, 4, 5 must still be in the set.
        assert_eq!(to_vec(&set), [3, 4, 5]);
    }

    #[test]
    fn extract_if_drop_with_nothing_removed() {
        let mut set = make_set(&[1, 2, 3]);
        {
            let mut it = set.extract_if(.., |_| false);
            it.next(); // called but predicate returned false
            // Drop; nothing was removed, so no rebuild needed.
        }
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    // MARK: extract_if: size_hint

    #[test]
    fn extract_if_size_hint() {
        let mut set = make_set(&[1, 2, 3]);
        let it = set.extract_if(.., |_| true);
        let (lo, hi) = it.size_hint();
        assert_eq!(lo, 0);
        assert_eq!(hi, Some(3));
    }

    // MARK: extract_if: Debug

    #[test]
    fn extract_if_debug() {
        let mut set = make_set(&[1, 2, 3]);
        let it = set.extract_if(.., |_| false);
        let s = format!("{it:?}");
        assert!(s.contains('1') || s.contains('['));
    }

    // MARK: extract_if: ExtractIf type check

    #[test]
    fn extract_if_is_correct_type() {
        let mut set = make_set(&[1, 2, 3]);
        let _it: ExtractIf<'_, i32, _, _, _, _, 16> = set.extract_if(.., |_| true);
    }

    // MARK: extract_if: custom comparator

    #[test]
    fn extract_if_custom_comparator() {
        use core::cmp::Ordering;
        #[expect(
            clippy::trivially_copy_pass_by_ref,
            reason = "must match Comparator<T> signature"
        )]
        fn rev(x: &i32, y: &i32) -> Ordering {
            y.cmp(x)
        }
        let fnptr: fn(&i32, &i32) -> Ordering = rev;
        let mut set: SkipSet<i32, 16, _> = SkipSet::with_comparator(FnComparator(fnptr));
        for v in [1, 2, 3, 4, 5] {
            set.insert(v);
        }
        // Descending order: [5, 4, 3, 2, 1].
        // Range `..=3` in descending order means values ≥ 3 (5, 4, 3).
        let removed: Vec<i32> = set.extract_if(..=3, |_| true).collect();
        assert_eq!(removed, [5, 4, 3]);
        let remaining: Vec<i32> = set.iter().copied().collect();
        assert_eq!(remaining, [2, 1]);
    }
}
