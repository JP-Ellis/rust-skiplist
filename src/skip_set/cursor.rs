//! Gap cursor types for [`SkipSet`].
//!
//! > **Note:** This module requires the `cursor` feature and is **unstable**.
//! > The API may change in a future release without prior notice.
//!
//! These are thin wrappers around the [`OrderedSkipList`] cursors.  The only
//! behavioural difference is that `CursorMut` rejects insertions of values
//! that are **equal** to a neighbour, enforcing the set uniqueness invariant.
//!
//! Cursors are produced by the `lower_bound` / `upper_bound` family of methods
//! on [`SkipSet`].
//!
//! [`OrderedSkipList`]: crate::ordered_skip_list::OrderedSkipList

use core::{fmt, marker::PhantomData, ops::Bound};

pub use crate::ordered_skip_list::cursor::UnorderedValueError;
use crate::{
    comparator::{Comparator, ComparatorKey, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    ordered_skip_list::cursor::{Cursor as OslCursor, CursorMut as OslCursorMut},
    skip_set::SkipSet,
};

// MARK: Cursor

/// A read-only gap cursor into a [`SkipSet`].
///
/// Points at a **gap between two adjacent elements** (or at the leftmost /
/// rightmost gap).  Implements [`Copy`] and [`Clone`].
///
/// Obtain a cursor via [`SkipSet::lower_bound`] or [`SkipSet::upper_bound`].
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::SkipSet;
/// use core::ops::Bound;
///
/// let set: SkipSet<i32> = [1, 2, 3].into_iter().collect();
/// let mut cur = set.lower_bound(Bound::Included(&2));
/// assert_eq!(cur.peek_prev(), Some(&1));
/// assert_eq!(cur.peek_next(), Some(&2));
/// ```
pub struct Cursor<
    'a,
    T,
    const N: usize = 16,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// The underlying [`OrderedSkipList`] cursor.
    ///
    /// [`OrderedSkipList`]: crate::ordered_skip_list::OrderedSkipList
    inner: OslCursor<'a, T, N, C, G>,
}

impl<T, const N: usize, C: Comparator<T>, G: LevelGenerator> Clone for Cursor<'_, T, N, C, G> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const N: usize, C: Comparator<T>, G: LevelGenerator> Copy for Cursor<'_, T, N, C, G> {}

impl<'a, T, const N: usize, C: Comparator<T>, G: LevelGenerator> Cursor<'a, T, N, C, G> {
    /// Constructs a new cursor wrapping an [`OrderedSkipList`] cursor.
    ///
    /// [`OrderedSkipList`]: crate::ordered_skip_list::OrderedSkipList
    pub(super) fn new(inner: OslCursor<'a, T, N, C, G>) -> Self {
        Self { inner }
    }

    /// Returns a shared reference to the element immediately to the **right**
    /// of the cursor without moving it.
    ///
    /// Returns `None` when at the rightmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let set: SkipSet<i32> = [1, 2].into_iter().collect();
    /// let cur = set.lower_bound(Bound::Unbounded);
    /// assert_eq!(cur.peek_next(), Some(&1));
    /// ```
    #[must_use]
    #[inline]
    pub fn peek_next(&self) -> Option<&'a T> {
        self.inner.peek_next()
    }

    /// Returns a shared reference to the element immediately to the **left**
    /// of the cursor without moving it.
    ///
    /// Returns `None` when at the leftmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let set: SkipSet<i32> = [1, 2].into_iter().collect();
    /// let cur = set.upper_bound(Bound::Unbounded);
    /// assert_eq!(cur.peek_prev(), Some(&2));
    /// ```
    #[must_use]
    #[inline]
    pub fn peek_prev(&self) -> Option<&'a T> {
        self.inner.peek_prev()
    }

    /// Advances the cursor one position to the right and returns the element
    /// that was just straddled.
    ///
    /// Returns `None` (without moving) when already at the rightmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let set: SkipSet<i32> = [10, 20, 30].into_iter().collect();
    /// let mut cur = set.lower_bound(Bound::Unbounded);
    /// assert_eq!(cur.next(), Some(&10));
    /// assert_eq!(cur.next(), Some(&20));
    /// ```
    #[expect(
        clippy::should_implement_trait,
        reason = "cursor navigation method, not an iterator"
    )]
    #[inline]
    pub fn next(&mut self) -> Option<&'a T> {
        self.inner.next()
    }

    /// Retreats the cursor one position to the left and returns the element
    /// that was just straddled.
    ///
    /// Returns `None` (without moving) when already at the leftmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let set: SkipSet<i32> = [10, 20, 30].into_iter().collect();
    /// let mut cur = set.upper_bound(Bound::Unbounded);
    /// assert_eq!(cur.prev(), Some(&30));
    /// assert_eq!(cur.prev(), Some(&20));
    /// ```
    #[inline]
    pub fn prev(&mut self) -> Option<&'a T> {
        self.inner.prev()
    }
}

impl<T: fmt::Debug, const N: usize, C: Comparator<T>, G: LevelGenerator> fmt::Debug
    for Cursor<'_, T, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cursor")
            .field("peek_prev", &self.peek_prev())
            .field("peek_next", &self.peek_next())
            .finish()
    }
}

// MARK: CursorMut

/// A mutable gap cursor into a [`SkipSet`].
///
/// Points at the **gap between two adjacent elements**.  Supports inserting
/// and removing elements while enforcing the set uniqueness invariant:
/// `insert_after` and `insert_before` return [`UnorderedValueError`] if the
/// value is equal to a neighbour, in addition to the usual out-of-order check.
///
/// Obtain a mutable cursor via [`SkipSet::lower_bound_mut`] or
/// [`SkipSet::upper_bound_mut`].
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::SkipSet;
/// use skiplist::skip_set::cursor::UnorderedValueError;
/// use core::ops::Bound;
///
/// let mut set: SkipSet<i32> = [1, 3].into_iter().collect();
/// {
///     let mut cur = set.lower_bound_mut(Bound::Included(&2));
///     cur.insert_after(2).expect("2 is in order");
/// }
/// assert!(set.contains(&2));
/// ```
#[expect(
    clippy::module_name_repetitions,
    reason = "CursorMut lives in the cursor module; the repetition is intentional for clarity"
)]
pub struct CursorMut<
    'a,
    T,
    const N: usize = 16,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// The underlying [`OrderedSkipList`] mutable cursor.
    ///
    /// [`OrderedSkipList`]: crate::ordered_skip_list::OrderedSkipList
    inner: OslCursorMut<'a, T, N, C, G>,
    /// Phantom marker for the exclusive borrow of the set.
    _marker: PhantomData<&'a mut SkipSet<T, N, C, G>>,
}

impl<'a, T, const N: usize, C: Comparator<T>, G: LevelGenerator> CursorMut<'a, T, N, C, G> {
    /// Constructs a new mutable cursor wrapping an [`OrderedSkipList`] mutable cursor.
    ///
    /// [`OrderedSkipList`]: crate::ordered_skip_list::OrderedSkipList
    pub(super) fn new(inner: OslCursorMut<'a, T, N, C, G>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// Returns a read-only cursor at the same position.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let mut set: SkipSet<i32> = [1].into_iter().collect();
    /// let cur = set.lower_bound_mut(Bound::Unbounded);
    /// let ro = cur.as_cursor();
    /// assert_eq!(ro.peek_next(), Some(&1));
    /// ```
    #[must_use]
    #[inline]
    pub fn as_cursor(&self) -> Cursor<'_, T, N, C, G> {
        Cursor::new(self.inner.as_cursor())
    }

    /// Returns a shared reference to the element immediately to the **right**
    /// of the cursor without moving it.
    ///
    /// Returns `None` when at the rightmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let mut set: SkipSet<i32> = [42].into_iter().collect();
    /// let mut cur = set.lower_bound_mut(Bound::Unbounded);
    /// assert_eq!(cur.peek_next(), Some(&42));
    /// ```
    #[must_use]
    #[inline]
    pub fn peek_next(&self) -> Option<&T> {
        self.inner.peek_next()
    }

    /// Returns a shared reference to the element immediately to the **left**
    /// of the cursor without moving it.
    ///
    /// Returns `None` when at the leftmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let mut set: SkipSet<i32> = [42].into_iter().collect();
    /// let mut cur = set.upper_bound_mut(Bound::Unbounded);
    /// assert_eq!(cur.peek_prev(), Some(&42));
    /// ```
    #[must_use]
    #[inline]
    pub fn peek_prev(&self) -> Option<&T> {
        self.inner.peek_prev()
    }

    /// Advances the cursor one position to the right and returns the element
    /// that was just straddled.
    ///
    /// Returns `None` (without moving) when already at the rightmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let mut set: SkipSet<i32> = [1, 2].into_iter().collect();
    /// let mut cur = set.lower_bound_mut(Bound::Unbounded);
    /// assert_eq!(cur.next(), Some(&1));
    /// ```
    #[expect(
        clippy::should_implement_trait,
        reason = "cursor navigation method, not an iterator"
    )]
    #[inline]
    pub fn next(&mut self) -> Option<&T> {
        self.inner.next()
    }

    /// Retreats the cursor one position to the left and returns the element
    /// that was just straddled.
    ///
    /// Returns `None` (without moving) when already at the leftmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let mut set: SkipSet<i32> = [1, 2].into_iter().collect();
    /// let mut cur = set.upper_bound_mut(Bound::Unbounded);
    /// assert_eq!(cur.prev(), Some(&2));
    /// ```
    #[inline]
    pub fn prev(&mut self) -> Option<&T> {
        self.inner.prev()
    }

    /// Inserts `value` immediately to the **right** of the current gap.
    ///
    /// The cursor position is unchanged after a successful insertion (the new
    /// element becomes the right neighbour).
    ///
    /// Returns [`UnorderedValueError`] if `value` is out of order **or equal**
    /// to a neighbour (sets do not allow duplicates).  The value is returned
    /// inside the error for recovery without cloning.
    ///
    /// # Errors
    ///
    /// Returns [`UnorderedValueError`] when the value would be out of order or
    /// equal to an adjacent element.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let mut set: SkipSet<i32> = [1, 3].into_iter().collect();
    /// {
    ///     let mut cur = set.lower_bound_mut(Bound::Included(&2));
    ///     cur.insert_after(2).expect("2 is in order");
    /// }
    /// assert_eq!(set.len(), 3);
    ///
    /// // Inserting a duplicate is rejected.
    /// let mut cur2 = set.lower_bound_mut(Bound::Included(&2));
    /// assert_eq!(cur2.insert_after(1).unwrap_err().0, 1);
    /// ```
    #[inline]
    pub fn insert_after(&mut self, value: T) -> Result<(), UnorderedValueError<T>> {
        self.inner.insert_after_strict(value)
    }

    /// Inserts `value` immediately to the **left** of the right neighbour
    /// (equivalently, into the current gap), then advances the cursor so the
    /// new element becomes the left neighbour.
    ///
    /// Returns [`UnorderedValueError`] if `value` is out of order **or equal**
    /// to a neighbour.
    ///
    /// # Errors
    ///
    /// Returns [`UnorderedValueError`] when the value would be out of order or
    /// equal to an adjacent element.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use skiplist::skip_set::cursor::UnorderedValueError;
    /// use core::ops::Bound;
    ///
    /// let mut set: SkipSet<i32> = [1, 3].into_iter().collect();
    /// {
    ///     let mut cur = set.lower_bound_mut(Bound::Included(&2));
    ///     cur.insert_before(2).expect("2 is in order");
    ///     assert_eq!(cur.peek_prev(), Some(&2));
    /// }
    /// assert_eq!(set.len(), 3);
    /// ```
    #[inline]
    pub fn insert_before(&mut self, value: T) -> Result<(), UnorderedValueError<T>> {
        self.inner.insert_before_strict(value)
    }

    /// Removes the element immediately to the **right** of the cursor and
    /// returns it.
    ///
    /// Returns `None` if there is no right neighbour.  The cursor position is
    /// unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let mut set: SkipSet<i32> = [1, 2, 3].into_iter().collect();
    /// {
    ///     let mut cur = set.lower_bound_mut(Bound::Unbounded);
    ///     assert_eq!(cur.remove_next(), Some(1));
    /// }
    /// assert_eq!(set.len(), 2);
    /// ```
    #[inline]
    pub fn remove_next(&mut self) -> Option<T> {
        self.inner.remove_next()
    }

    /// Removes the element immediately to the **left** of the cursor and
    /// returns it.
    ///
    /// Returns `None` if there is no left neighbour.  The cursor retreats to
    /// the previous gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let mut set: SkipSet<i32> = [1, 2, 3].into_iter().collect();
    /// {
    ///     let mut cur = set.upper_bound_mut(Bound::Unbounded);
    ///     assert_eq!(cur.remove_prev(), Some(3));
    /// }
    /// assert_eq!(set.len(), 2);
    /// ```
    #[inline]
    pub fn remove_prev(&mut self) -> Option<T> {
        self.inner.remove_prev()
    }
}

impl<T: fmt::Debug, const N: usize, C: Comparator<T>, G: LevelGenerator> fmt::Debug
    for CursorMut<'_, T, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, f)
    }
}

// MARK: Factory methods on SkipSet

impl<T, const N: usize, C: Comparator<T>, G: LevelGenerator> SkipSet<T, N, C, G> {
    /// Returns a read-only cursor positioned at the **lower bound** of `bound`.
    ///
    /// The cursor is placed at the gap immediately **before** the first element
    /// that satisfies the bound:
    ///
    /// | Bound                     | Cursor gap                             |
    /// |---------------------------|----------------------------------------|
    /// | `Unbounded`               | before the first element (leftmost)    |
    /// | `Included(&q)`            | before the first element `>= q`        |
    /// | `Excluded(&q)`            | before the first element `> q`         |
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let set: SkipSet<i32> = [1, 2, 3].into_iter().collect();
    /// let cur = set.lower_bound(Bound::Included(&2));
    /// assert_eq!(cur.peek_prev(), Some(&1));
    /// assert_eq!(cur.peek_next(), Some(&2));
    /// ```
    #[inline]
    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, T, N, C, G>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        Cursor::new(self.inner.lower_bound(bound))
    }

    /// Returns a read-only cursor positioned at the **upper bound** of `bound`.
    ///
    /// The cursor is placed at the gap immediately **after** the last element
    /// that satisfies the bound:
    ///
    /// | Bound                     | Cursor gap                             |
    /// |---------------------------|----------------------------------------|
    /// | `Unbounded`               | after the last element (rightmost)     |
    /// | `Included(&q)`            | after the last element `<= q`          |
    /// | `Excluded(&q)`            | after the last element `< q`           |
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let set: SkipSet<i32> = [1, 2, 3].into_iter().collect();
    /// let cur = set.upper_bound(Bound::Included(&2));
    /// assert_eq!(cur.peek_prev(), Some(&2));
    /// assert_eq!(cur.peek_next(), Some(&3));
    /// ```
    #[inline]
    pub fn upper_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, T, N, C, G>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        Cursor::new(self.inner.upper_bound(bound))
    }

    /// Returns a mutable cursor positioned at the **lower bound** of `bound`.
    ///
    /// See [`lower_bound`] for bound semantics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let mut set: SkipSet<i32> = [1, 3].into_iter().collect();
    /// {
    ///     let mut cur = set.lower_bound_mut(Bound::Included(&2));
    ///     cur.insert_after(2).expect("2 is in order");
    /// }
    /// assert!(set.contains(&2));
    /// ```
    ///
    /// [`lower_bound`]: SkipSet::lower_bound
    #[inline]
    pub fn lower_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, T, N, C, G>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        CursorMut::new(self.inner.lower_bound_mut(bound))
    }

    /// Returns a mutable cursor positioned at the **upper bound** of `bound`.
    ///
    /// See [`upper_bound`] for bound semantics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use core::ops::Bound;
    ///
    /// let mut set: SkipSet<i32> = [1, 3].into_iter().collect();
    /// {
    ///     let mut cur = set.upper_bound_mut(Bound::Included(&2));
    ///     // Gap is between 1 and 3; inserting 2 is valid.
    ///     cur.insert_after(2).expect("2 is in order");
    /// }
    /// assert!(set.contains(&2));
    /// ```
    ///
    /// [`upper_bound`]: SkipSet::upper_bound
    #[inline]
    pub fn upper_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, T, N, C, G>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        CursorMut::new(self.inner.upper_bound_mut(bound))
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use core::ops::Bound;

    use pretty_assertions::assert_eq;

    use super::*;

    fn set_123() -> SkipSet<i32> {
        [1, 2, 3].into_iter().collect()
    }

    // --- Cursor factory ---

    #[test]
    fn lower_bound_unbounded_is_leftmost() {
        let s = set_123();
        let cur = s.lower_bound(Bound::Unbounded);
        assert_eq!(cur.peek_prev(), None);
        assert_eq!(cur.peek_next(), Some(&1));
    }

    #[test]
    fn upper_bound_unbounded_is_rightmost() {
        let s = set_123();
        let cur = s.upper_bound(Bound::Unbounded);
        assert_eq!(cur.peek_prev(), Some(&3));
        assert_eq!(cur.peek_next(), None);
    }

    #[test]
    fn lower_bound_included_exact_match() {
        let s = set_123();
        let cur = s.lower_bound(Bound::Included(&2));
        assert_eq!(cur.peek_prev(), Some(&1));
        assert_eq!(cur.peek_next(), Some(&2));
    }

    #[test]
    fn upper_bound_included_exact_match() {
        let s = set_123();
        let cur = s.upper_bound(Bound::Included(&2));
        assert_eq!(cur.peek_prev(), Some(&2));
        assert_eq!(cur.peek_next(), Some(&3));
    }

    #[test]
    fn cursor_is_copy() {
        let s = set_123();
        let cur1 = s.lower_bound(Bound::Unbounded);
        let cur2 = cur1; // Copy
        assert_eq!(cur1.peek_next(), cur2.peek_next());
    }

    // --- Navigation ---

    #[test]
    fn next_then_prev_round_trip() {
        let s = set_123();
        let mut cur = s.lower_bound(Bound::Unbounded);
        assert_eq!(cur.next(), Some(&1));
        assert_eq!(cur.prev(), Some(&1));
        assert_eq!(cur.peek_prev(), None);
    }

    // --- CursorMut insert_after ---

    #[test]
    fn insert_after_valid() {
        let mut s = set_123();
        // Create a cursor but don't insert here — no room between integers.
        // Use a block so the borrow ends before the next cursor.
        {
            let _cur = s.lower_bound_mut(Bound::Included(&2));
        }
        {
            let mut cur = s.upper_bound_mut(Bound::Unbounded);
            cur.insert_after(4).expect("insert_after 4 should succeed");
        }
        assert_eq!(s.len(), 4);
        assert!(s.contains(&4));
    }

    #[test]
    fn insert_after_rejects_duplicate_right_neighbour() {
        let mut s = set_123();
        // lower_bound(Included(&1)) → gap before 1
        let mut cur = s.lower_bound_mut(Bound::Excluded(&0));
        // right neighbour is 1; inserting 1 is a duplicate
        assert_eq!(cur.insert_after(1), Err(UnorderedValueError(1)));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn insert_after_rejects_duplicate_left_neighbour() {
        let mut s = set_123();
        // upper_bound(Included(&1)) → gap after 1; left neighbour = 1
        let mut cur = s.upper_bound_mut(Bound::Included(&1));
        assert_eq!(cur.insert_after(1), Err(UnorderedValueError(1)));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn insert_after_rejects_out_of_order() {
        let mut s = set_123();
        let mut cur = s.lower_bound_mut(Bound::Included(&2));
        // gap: between 1 and 2; inserting 0 is out of order
        assert_eq!(cur.insert_after(0), Err(UnorderedValueError(0)));
    }

    #[test]
    fn insert_before_valid() {
        let mut s = set_123();
        {
            let mut cur = s.upper_bound_mut(Bound::Unbounded);
            cur.insert_before(4)
                .expect("insert_before 4 should succeed");
            assert_eq!(cur.peek_prev(), Some(&4));
        }
        assert!(s.contains(&4));
    }

    #[test]
    fn insert_before_rejects_duplicate() {
        let mut s = set_123();
        let mut cur = s.upper_bound_mut(Bound::Included(&2));
        // gap after 2, before 3; left neighbour = 2
        assert_eq!(cur.insert_before(2), Err(UnorderedValueError(2)));
    }

    // --- CursorMut remove_next / remove_prev ---

    #[test]
    fn remove_next_removes_right_neighbour() {
        let mut s = set_123();
        {
            let mut cur = s.lower_bound_mut(Bound::Unbounded);
            assert_eq!(cur.remove_next(), Some(1));
        }
        assert_eq!(s.len(), 2);
        assert!(!s.contains(&1));
    }

    #[test]
    fn remove_prev_removes_left_neighbour() {
        let mut s = set_123();
        {
            let mut cur = s.upper_bound_mut(Bound::Unbounded);
            assert_eq!(cur.remove_prev(), Some(3));
        }
        assert_eq!(s.len(), 2);
        assert!(!s.contains(&3));
    }

    #[test]
    fn remove_next_at_rightmost_gap_returns_none() {
        let mut s = set_123();
        let mut cur = s.upper_bound_mut(Bound::Unbounded);
        assert_eq!(cur.remove_next(), None);
        assert_eq!(s.len(), 3);
    }
}
