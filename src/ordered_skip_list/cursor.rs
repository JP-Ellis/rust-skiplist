//! Gap cursor types for [`OrderedSkipList`].
//!
//! > **Note:** This module requires the `cursor` feature and is **unstable**.
//! > The API may change in a future release without prior notice.
//!
//! A cursor points at a **gap between two adjacent elements** (not at an
//! element itself).  The node stored in `current` is the one on the **left**
//! side of the gap; the head sentinel represents the leftmost gap (before the
//! first element) and has no left neighbour.
//!
//! Cursors are produced by the `lower_bound` / `upper_bound` family of methods
//! on [`OrderedSkipList`].

use core::{cmp::Ordering, fmt, marker::PhantomData, ops::Bound, ptr::NonNull};

use crate::{
    comparator::{Comparator, ComparatorKey, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    node::{
        Node,
        cursor_raw::{RawCursorMut, gap_find},
    },
    ordered_skip_list::OrderedSkipList,
};

// MARK: UnorderedValueError

/// Error returned when a value cannot be inserted at the cursor's current
/// position because doing so would violate the sort order.
///
/// The value is returned inside the error so the caller can recover it without
/// cloning.
///
/// # Examples
///
/// ```rust
/// use skiplist::ordered_skip_list::OrderedSkipList;
/// use core::ops::Bound;
///
/// let mut list = OrderedSkipList::<i32>::new();
/// list.insert(1);
/// list.insert(3);
///
/// // Position cursor between 1 and 3.
/// let mut cur = list.lower_bound_mut(Bound::Included(&2));
/// // Inserting 0 here would violate order (0 < 1 = left neighbour).
/// let err = cur.insert_after(0).unwrap_err();
/// assert_eq!(err.0, 0);
/// ```
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnorderedValueError<T>(
    /// The value that could not be inserted.
    pub T,
);

#[expect(
    clippy::use_debug,
    reason = "T may not implement Display; Debug is the best available formatter"
)]
impl<T: fmt::Debug> fmt::Display for UnorderedValueError<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "value {:?} is out of order at the cursor's current gap",
            self.0
        )
    }
}

impl<T: fmt::Debug> std::error::Error for UnorderedValueError<T> {}

// MARK: Cursor (read-only)

/// A read-only gap cursor into an [`OrderedSkipList`].
///
/// A cursor points at the **gap** between two adjacent elements.  The element
/// to the left of the gap (if any) is the "previous" element; the element to
/// the right (if any) is the "next" element.
///
/// Create a cursor via [`OrderedSkipList::lower_bound`] or
/// [`OrderedSkipList::upper_bound`].
///
/// # Lifetime
///
/// The cursor borrows the list for `'a`.  All references returned by the
/// cursor methods carry that same `'a` lifetime, so they remain valid as
/// long as the list is not modified.
///
/// # Examples
///
/// ```rust
/// use skiplist::ordered_skip_list::OrderedSkipList;
/// use core::ops::Bound;
///
/// let mut list = OrderedSkipList::<i32>::new();
/// for v in [1, 3, 5] { list.insert(v); }
///
/// let mut cur = list.lower_bound(Bound::Included(&3));
/// assert_eq!(cur.peek_prev(), Some(&1));
/// assert_eq!(cur.peek_next(), Some(&3));
/// assert_eq!(cur.next(), Some(&3));
/// assert_eq!(cur.peek_prev(), Some(&3));
/// ```
pub struct Cursor<
    'a,
    T,
    const N: usize = 16,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// The node on the left of the gap.  Points to the head sentinel when the
    /// cursor is at the leftmost gap.
    current: NonNull<Node<T, N>>,
    /// 1-based rank of `current` within the list: 0 = head sentinel (leftmost
    /// gap), 1 = after the first element, etc.
    current_rank: usize,
    /// Phantom: binds the `'a` lifetime to the list borrow without requiring
    /// `C: Copy` / `G: Copy` (which would fail because `Geometric` contains
    /// `SmallRng`).
    _marker: PhantomData<&'a OrderedSkipList<T, N, C, G>>,
}

impl<T, const N: usize, C: Comparator<T>, G: LevelGenerator> Clone for Cursor<'_, T, N, C, G> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const N: usize, C: Comparator<T>, G: LevelGenerator> Copy for Cursor<'_, T, N, C, G> {}

impl<'a, T, const N: usize, C: Comparator<T>, G: LevelGenerator> Cursor<'a, T, N, C, G> {
    /// Constructs a new cursor at `current` with the given `current_rank`.
    pub(super) fn new(
        current: NonNull<Node<T, N>>,
        current_rank: usize,
        _list: &'a OrderedSkipList<T, N, C, G>,
    ) -> Self {
        Self {
            current,
            current_rank,
            _marker: PhantomData,
        }
    }

    /// Returns a shared reference to the element immediately to the **right**
    /// of the cursor, without moving the cursor.
    ///
    /// Returns `None` when the cursor is at the rightmost gap (after the last
    /// element or the list is empty).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(1);
    /// list.insert(2);
    ///
    /// let cur = list.lower_bound(Bound::Unbounded);
    /// assert_eq!(cur.peek_next(), Some(&1));
    /// ```
    #[must_use]
    #[inline]
    pub fn peek_next(&self) -> Option<&'a T> {
        // SAFETY: `current` is valid for `'a` (it is inside the list borrowed
        // for `'a`).
        let n = unsafe { self.current.as_ref() }.next()?;
        // SAFETY: `n` is a valid node in the same list.
        unsafe { n.as_ref() }.value()
    }

    /// Returns a shared reference to the element immediately to the **left**
    /// of the cursor, without moving the cursor.
    ///
    /// Returns `None` when the cursor is at the leftmost gap (before the first
    /// element or the list is empty).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(1);
    /// list.insert(2);
    ///
    /// let cur = list.upper_bound(Bound::Unbounded);
    /// assert_eq!(cur.peek_prev(), Some(&2));
    /// ```
    #[must_use]
    #[inline]
    pub fn peek_prev(&self) -> Option<&'a T> {
        // SAFETY: `current` is valid for `'a`.
        unsafe { self.current.as_ref() }.value()
    }

    /// Advances the cursor one position to the right and returns a shared
    /// reference to the element that was just straddled.
    ///
    /// Returns `None` (without moving) when the cursor is already at the
    /// rightmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [10, 20, 30] { list.insert(v); }
    ///
    /// let mut cur = list.lower_bound(Bound::Unbounded);
    /// assert_eq!(cur.next(), Some(&10));
    /// assert_eq!(cur.next(), Some(&20));
    /// assert_eq!(cur.next(), Some(&30));
    /// assert_eq!(cur.next(), None);
    /// ```
    #[expect(
        clippy::should_implement_trait,
        reason = "cursor navigation method, not an iterator"
    )]
    #[inline]
    pub fn next(&mut self) -> Option<&'a T> {
        // SAFETY: `current` is valid for `'a`.
        let next = unsafe { self.current.as_ref() }.next()?;
        self.current = next;
        self.current_rank = self.current_rank.saturating_add(1);
        // SAFETY: `next` is valid for `'a`.
        unsafe { next.as_ref() }.value()
    }

    /// Retreats the cursor one position to the left and returns a shared
    /// reference to the element that was just straddled.
    ///
    /// Returns `None` (without moving) when the cursor is already at the
    /// leftmost gap (before the first element).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [10, 20, 30] { list.insert(v); }
    ///
    /// let mut cur = list.upper_bound(Bound::Unbounded);
    /// assert_eq!(cur.prev(), Some(&30));
    /// assert_eq!(cur.prev(), Some(&20));
    /// assert_eq!(cur.prev(), Some(&10));
    /// assert_eq!(cur.prev(), None);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        clippy::unwrap_in_result,
        reason = "every data node always has a predecessor; fires only on internal corruption"
    )]
    #[inline]
    pub fn prev(&mut self) -> Option<&'a T> {
        // SAFETY: `current` is valid for `'a`.  If `current` has no value it
        // is the head sentinel and we are already at the leftmost gap.
        let val = unsafe { self.current.as_ref() }.value()?;
        // SAFETY: every data node has a predecessor (at minimum the head).
        let prev = unsafe { self.current.as_ref() }
            .prev()
            .expect("data node always has a predecessor");
        self.current = prev;
        self.current_rank = self.current_rank.saturating_sub(1);
        Some(val)
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

/// A mutable gap cursor into an [`OrderedSkipList`].
///
/// A cursor points at the **gap** between two adjacent elements.  In addition
/// to navigation, `CursorMut` supports inserting and removing elements at the
/// cursor position.
///
/// Create a mutable cursor via [`OrderedSkipList::lower_bound_mut`] or
/// [`OrderedSkipList::upper_bound_mut`].
///
/// # Sort-order invariant
///
/// All inserts are validated: a value may only be inserted if it is at most as
/// large as the right neighbour and at least as large as the left neighbour
/// (according to the list's comparator).  Out-of-order inserts return
/// [`Err(UnorderedValueError(value))`][UnorderedValueError].
///
/// # Examples
///
/// ```rust
/// use skiplist::ordered_skip_list::OrderedSkipList;
/// use core::ops::Bound;
///
/// let mut list = OrderedSkipList::<i32>::new();
/// for v in [1, 5] { list.insert(v); }
///
/// // Gap between 1 and 5.
/// {
///     let mut cur = list.lower_bound_mut(Bound::Included(&3));
///     cur.insert_after(3).expect("3 is in order");
/// }
///
/// let vals: Vec<_> = list.iter().copied().collect();
/// assert_eq!(vals, [1, 3, 5]);
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
    /// Shared structural cursor state (current node, rank, precursor cache).
    raw: RawCursorMut<T, N>,
    /// Raw pointer to the list.  The `PhantomData` below records the real
    /// exclusive borrow lifetime `'a`.
    list: *mut OrderedSkipList<T, N, C, G>,
    /// Phantom marker for the exclusive borrow of the list.
    _marker: PhantomData<&'a mut OrderedSkipList<T, N, C, G>>,
}

// SAFETY: `CursorMut` holds an exclusive borrow of `OrderedSkipList`, so the
// same `Send`/`Sync` bounds as the list apply.
unsafe impl<T: Send, const N: usize, C: Comparator<T> + Send, G: LevelGenerator + Send> Send
    for CursorMut<'_, T, N, C, G>
{
}
// SAFETY: same reasoning as Send — exclusive borrow propagates Sync bounds.
unsafe impl<T: Sync, const N: usize, C: Comparator<T> + Sync, G: LevelGenerator + Sync> Sync
    for CursorMut<'_, T, N, C, G>
{
}

impl<T, const N: usize, C: Comparator<T>, G: LevelGenerator> CursorMut<'_, T, N, C, G> {
    /// Constructs a new mutable cursor at `current` with the given rank.
    pub(super) fn new(
        current: NonNull<Node<T, N>>,
        current_rank: usize,
        list: *mut OrderedSkipList<T, N, C, G>,
    ) -> Self {
        Self {
            raw: RawCursorMut::new(current, current_rank),
            list,
            _marker: PhantomData,
        }
    }

    /// Returns a read-only cursor at the same position.
    ///
    /// The read-only cursor borrows the `CursorMut`, so neither can be used
    /// independently while the other is live.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(1);
    ///
    /// let mut cur = list.lower_bound_mut(Bound::Unbounded);
    /// let ro = cur.as_cursor();
    /// assert_eq!(ro.peek_next(), Some(&1));
    /// ```
    #[must_use]
    #[inline]
    pub fn as_cursor(&self) -> Cursor<'_, T, N, C, G> {
        Cursor {
            current: self.raw.current,
            current_rank: self.raw.current_rank,
            _marker: PhantomData,
        }
    }

    /// Returns a shared reference to the element to the **right** of the gap
    /// without moving the cursor.
    ///
    /// Returns `None` when the cursor is at the rightmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(42);
    /// let mut cur = list.lower_bound_mut(Bound::Unbounded);
    /// assert_eq!(cur.peek_next(), Some(&42));
    /// ```
    #[must_use]
    #[inline]
    pub fn peek_next(&self) -> Option<&T> {
        // SAFETY: `current` is valid for the list's lifetime.
        let n = unsafe { self.raw.current.as_ref() }.next()?;
        // SAFETY: `n` is a valid node in the list.
        unsafe { n.as_ref() }.value()
    }

    /// Returns a shared reference to the element to the **left** of the gap
    /// without moving the cursor.
    ///
    /// Returns `None` when the cursor is at the leftmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(42);
    /// let mut cur = list.upper_bound_mut(Bound::Unbounded);
    /// assert_eq!(cur.peek_prev(), Some(&42));
    /// ```
    #[must_use]
    #[inline]
    pub fn peek_prev(&self) -> Option<&T> {
        // SAFETY: `current` is valid for the list's lifetime.
        unsafe { self.raw.current.as_ref() }.value()
    }

    /// Advances the cursor one position to the right and returns a shared
    /// reference to the element that was just straddled.
    ///
    /// Returns `None` (without moving) when already at the rightmost gap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [1, 2, 3] { list.insert(v); }
    ///
    /// let mut cur = list.lower_bound_mut(Bound::Unbounded);
    /// assert_eq!(cur.next(), Some(&1));
    /// assert_eq!(cur.next(), Some(&2));
    /// ```
    #[expect(
        clippy::should_implement_trait,
        reason = "cursor navigation method, not an iterator"
    )]
    #[inline]
    pub fn next(&mut self) -> Option<&T> {
        let next = self.raw.advance()?;
        // SAFETY: `next` is a valid data node — the DLL has no tail sentinel,
        // so every node returned by `advance()` carries a value.
        unsafe { next.as_ref() }.value()
    }

    /// Retreats the cursor one position to the left and returns a shared
    /// reference to the element that was just straddled.
    ///
    /// Returns `None` (without moving) when already at the leftmost gap.
    ///
    /// # Performance
    ///
    /// Invalidates the precursor cache.  The next call to `insert_after`,
    /// `insert_before`, `remove_next`, or `remove_prev` will perform an
    /// O(log n) traversal from the head to rebuild it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [1, 2, 3] { list.insert(v); }
    ///
    /// let mut cur = list.upper_bound_mut(Bound::Unbounded);
    /// assert_eq!(cur.prev(), Some(&3));
    /// assert_eq!(cur.prev(), Some(&2));
    /// ```
    #[inline]
    pub fn prev(&mut self) -> Option<&T> {
        let old = self.raw.retreat()?;
        // SAFETY: `old` is the former `current` node; `retreat()` returns
        // `None` for the head sentinel, so `old` always holds a value.
        unsafe { old.as_ref() }.value()
    }

    /// Inserts `value` immediately to the **right** of the current gap.
    ///
    /// The cursor position does not change after a successful insert: the new
    /// element becomes the new right neighbour (`peek_next`).
    ///
    /// # Errors
    ///
    /// Returns [`Err(UnorderedValueError(value))`][UnorderedValueError] when
    /// `value` would violate the sort order (i.e. it is less than the left
    /// neighbour or greater than the right neighbour).  The value is returned
    /// inside the error so no clone is required.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [1, 5] { list.insert(v); }
    ///
    /// {
    ///     let mut cur = list.lower_bound_mut(Bound::Included(&3));
    ///     cur.insert_after(3).expect("3 is in order");
    /// }
    ///
    /// assert_eq!(list.len(), 3);
    /// assert_eq!(list.get(&3), Some(&3));
    /// ```
    #[inline]
    pub fn insert_after(&mut self, value: T) -> Result<(), UnorderedValueError<T>> {
        self.insert_impl(value, false, true)
    }

    /// Like [`insert_after`] but rejects equal neighbours (used by
    /// [`SkipSet`] which forbids duplicates).
    ///
    /// [`insert_after`]: CursorMut::insert_after
    /// [`SkipSet`]: crate::skip_set::SkipSet
    #[inline]
    pub(crate) fn insert_after_strict(&mut self, value: T) -> Result<(), UnorderedValueError<T>> {
        self.insert_impl(value, false, false)
    }

    /// Inserts `value` immediately to the **left** of the right neighbour
    /// (equivalently, into the current gap), then advances the cursor to sit
    /// after the new element.
    ///
    /// After a successful insert the new element becomes the left neighbour
    /// (`peek_prev`).
    ///
    /// # Errors
    ///
    /// Returns [`Err(UnorderedValueError(value))`][UnorderedValueError] on an
    /// ordering violation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [1, 5] { list.insert(v); }
    ///
    /// {
    ///     let mut cur = list.lower_bound_mut(Bound::Included(&3));
    ///     cur.insert_before(3).expect("3 is in order");
    ///     // Cursor is now after 3; the next element is 5.
    ///     assert_eq!(cur.peek_prev(), Some(&3));
    ///     assert_eq!(cur.peek_next(), Some(&5));
    /// }
    ///
    /// assert_eq!(list.len(), 3);
    /// ```
    #[inline]
    pub fn insert_before(&mut self, value: T) -> Result<(), UnorderedValueError<T>> {
        self.insert_impl(value, true, true)
    }

    /// Like [`insert_before`] but rejects equal neighbours (used by
    /// [`SkipSet`] which forbids duplicates).
    ///
    /// [`insert_before`]: CursorMut::insert_before
    /// [`SkipSet`]: crate::skip_set::SkipSet
    #[inline]
    pub(crate) fn insert_before_strict(&mut self, value: T) -> Result<(), UnorderedValueError<T>> {
        self.insert_impl(value, true, false)
    }

    /// Removes the element immediately to the **right** of the cursor and
    /// returns it.
    ///
    /// The cursor position does not change.  Returns `None` when there is no
    /// right neighbour (rightmost gap or empty list).
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [1, 2, 3] { list.insert(v); }
    ///
    /// {
    ///     let mut cur = list.lower_bound_mut(Bound::Unbounded);
    ///     assert_eq!(cur.remove_next(), Some(1));
    ///     assert_eq!(cur.remove_next(), Some(2));
    /// }
    ///
    /// assert_eq!(list.len(), 1);
    /// ```
    #[inline]
    pub fn remove_next(&mut self) -> Option<T> {
        // SAFETY: `list` is exclusively borrowed for `'a`.
        let list_mut = unsafe { &mut *self.list };
        let (mut boxed, target_ptr) = self.raw.splice_out_next(list_mut.head)?;

        if list_mut.tail == Some(target_ptr) {
            list_mut.tail = if list_mut.len == 1 {
                None
            } else {
                // `self.raw.current` is the immediate base-layer left-neighbour
                // of the removed node.
                Some(self.raw.current)
            };
        }
        list_mut.len = list_mut.len.saturating_sub(1);

        boxed.take_value()
    }

    /// Removes the element immediately to the **left** of the cursor (the
    /// current left neighbour) and returns it.
    ///
    /// After removal the cursor retreats by one position.  Returns `None` when
    /// the cursor is already at the leftmost gap.
    ///
    /// # Performance
    ///
    /// Always performs an O(log n) traversal from the head regardless of cache
    /// state, because the precursors needed to unlink `current` differ from
    /// those cached for the gap after it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [1, 2, 3] { list.insert(v); }
    ///
    /// {
    ///     let mut cur = list.upper_bound_mut(Bound::Unbounded);
    ///     assert_eq!(cur.remove_prev(), Some(3));
    ///     assert_eq!(cur.remove_prev(), Some(2));
    /// }
    ///
    /// assert_eq!(list.len(), 1);
    /// ```
    #[inline]
    pub fn remove_prev(&mut self) -> Option<T> {
        // Must not be at the leftmost gap (head sentinel has no value).
        // SAFETY: `current` is valid.
        unsafe { self.raw.current.as_ref() }.value()?;

        // Cached precursors were for target = current_rank + 1; we now need
        // precursors for current_rank itself.  Invalidate so the next
        // insert/remove recomputes for the new gap position.
        self.raw.precursors = None;

        let target_rank = self.raw.current_rank;
        // Capture target pointer before splice_out invalidates it.
        let removed_ptr = self.raw.current;
        // SAFETY: `list` is exclusively borrowed for `'a`.
        let list_mut = unsafe { &mut *self.list };
        let (mut boxed, predecessor) =
            RawCursorMut::splice_out_at_rank(target_rank, list_mut.head)?;

        if list_mut.tail == Some(removed_ptr) {
            list_mut.tail = if list_mut.len == 1 {
                None
            } else {
                Some(predecessor)
            };
        }
        list_mut.len = list_mut.len.saturating_sub(1);

        // Cursor retreats to the predecessor of the removed node.
        self.raw.current = predecessor;
        self.raw.current_rank = self.raw.current_rank.saturating_sub(1);

        boxed.take_value()
    }

    // --- Private helpers ---

    /// Insert `value` at the current gap.
    ///
    /// `move_cursor = false` → `insert_after` (cursor stays left of new node).
    /// `move_cursor = true`  → `insert_before` (cursor advances to new node).
    /// `allow_equal = true`  — the value may compare `Equal` to a neighbour
    ///   (used by `OrderedSkipList` which permits duplicates).
    /// `allow_equal = false` — the value must compare `Less` on both sides
    ///   (used by `SkipSet` which rejects duplicates).
    fn insert_impl(
        &mut self,
        value: T,
        move_cursor: bool,
        allow_equal: bool,
    ) -> Result<(), UnorderedValueError<T>> {
        // --- Ordering check ---
        // SAFETY: `list` is exclusively borrowed for `'a`.
        let list_ref = unsafe { &*self.list };

        // Left neighbour: must be < value (strict) or <= value (allow_equal).
        // SAFETY: `current` is valid.
        if let Some(prev_val) = unsafe { self.raw.current.as_ref() }.value() {
            let ord = list_ref.comparator.compare(prev_val, &value);
            if ord == Ordering::Greater || (!allow_equal && ord == Ordering::Equal) {
                return Err(UnorderedValueError(value));
            }
        }
        // Right neighbour: must be > value (strict) or >= value (allow_equal).
        // SAFETY: `current` is valid.
        let next_node = unsafe { self.raw.current.as_ref() }.next();
        // SAFETY: any successor node and its value are valid.
        if let Some(next_val) = next_node.and_then(|n| unsafe { n.as_ref() }.value()) {
            let ord = list_ref.comparator.compare(&value, next_val);
            if ord == Ordering::Greater || (!allow_equal && ord == Ordering::Equal) {
                return Err(UnorderedValueError(value));
            }
        }

        // --- Structural insert via cached precursors (rank-based) ---
        // Using the rank guarantees we insert at EXACTLY the cursor's gap,
        // even when duplicate values are present.

        // SAFETY: `list` is exclusively borrowed for `'a`.
        let list_mut = unsafe { &mut *self.list };
        self.raw.ensure_precursors(list_mut.head);
        let height = list_mut.generator.level();
        let new_rank = self.raw.current_rank.saturating_add(1);
        let is_new_tail = list_mut.tail.is_none_or(|tail| self.raw.current == tail);

        // SAFETY: `self.raw.current` is the immediate base-layer left-neighbour
        // of the gap (guaranteed by cursor positioning).
        let new_node_nonnull: NonNull<Node<T, N>> =
            unsafe { Node::insert_after(self.raw.current, Node::with_value(height, value)) };

        // Wire skip links, update precursor cache, and optionally advance cursor.
        self.raw
            .insert_at_gap(new_node_nonnull, new_rank, height, move_cursor);

        // The new node is the tail when `self.raw.current` was the tail (or the
        // list was empty and tail is None).
        if is_new_tail {
            list_mut.tail = Some(new_node_nonnull);
        }
        list_mut.len = list_mut.len.saturating_add(1);

        Ok(())
    }
}

impl<T: fmt::Debug, const N: usize, C: Comparator<T>, G: LevelGenerator> fmt::Debug
    for CursorMut<'_, T, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: `raw.current` is valid for the list's lifetime.
        let peek_prev = unsafe { self.raw.current.as_ref() }.value();
        // SAFETY: `raw.current` is valid; if next exists it is also valid.
        let peek_next = unsafe { self.raw.current.as_ref() }
            .next()
            .and_then(|n| unsafe { n.as_ref() }.value());
        f.debug_struct("CursorMut")
            .field("peek_prev", &peek_prev)
            .field("peek_next", &peek_next)
            .finish()
    }
}

// MARK: Factory methods on OrderedSkipList

impl<T, const N: usize, C: Comparator<T>, G: LevelGenerator> OrderedSkipList<T, N, C, G> {
    /// Returns a read-only cursor positioned at the gap defined by the given
    /// lower bound.
    ///
    /// | Bound                   | Gap position                                   |
    /// |-------------------------|------------------------------------------------|
    /// | `Unbounded`             | Before the first element (leftmost gap)        |
    /// | `Included(&q)`          | After the last element `< q`                   |
    /// | `Excluded(&q)`          | After the last element `<= q`                  |
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [1, 2, 3] { list.insert(v); }
    ///
    /// let cur = list.lower_bound(Bound::Included(&2));
    /// assert_eq!(cur.peek_prev(), Some(&1));
    /// assert_eq!(cur.peek_next(), Some(&2));
    /// ```
    #[inline]
    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, T, N, C, G>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        let (current, rank) = match bound {
            Bound::Unbounded => (self.head, 0),
            Bound::Included(q) => {
                // advance on Less only: land before first >= q
                // SAFETY: self.head is valid for &self.
                unsafe {
                    gap_find(
                        self.head,
                        q,
                        |v, key| self.comparator.compare_key(v, key),
                        false,
                    )
                }
            }
            Bound::Excluded(q) => {
                // advance on Less or Equal: land before first > q
                // SAFETY: self.head is valid for &self.
                unsafe {
                    gap_find(
                        self.head,
                        q,
                        |v, key| self.comparator.compare_key(v, key),
                        true,
                    )
                }
            }
        };
        Cursor::new(current, rank, self)
    }

    /// Returns a read-only cursor positioned at the gap defined by the given
    /// upper bound.
    ///
    /// | Bound                   | Gap position                                   |
    /// |-------------------------|------------------------------------------------|
    /// | `Unbounded`             | After the last element (rightmost gap)         |
    /// | `Included(&q)`          | After the last element `<= q`                  |
    /// | `Excluded(&q)`          | After the last element `< q`                   |
    ///
    /// This operation is `$O(\log n)$` on average (or `$O(1)$` for `Unbounded`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [1, 2, 3] { list.insert(v); }
    ///
    /// let cur = list.upper_bound(Bound::Included(&2));
    /// assert_eq!(cur.peek_prev(), Some(&2));
    /// assert_eq!(cur.peek_next(), Some(&3));
    /// ```
    #[inline]
    pub fn upper_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, T, N, C, G>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        let (current, rank) = match bound {
            Bound::Unbounded => match self.tail {
                Some(tail) => (tail, self.len),
                None => (self.head, 0),
            },
            Bound::Included(q) => {
                // advance on Less or Equal: land after last <= q
                // Same traversal as lower_bound(Excluded).
                // SAFETY: self.head is valid for &self.
                unsafe {
                    gap_find(
                        self.head,
                        q,
                        |v, key| self.comparator.compare_key(v, key),
                        true,
                    )
                }
            }
            Bound::Excluded(q) => {
                // advance on Less only: land after last < q
                // Same traversal as lower_bound(Included).
                // SAFETY: self.head is valid for &self.
                unsafe {
                    gap_find(
                        self.head,
                        q,
                        |v, key| self.comparator.compare_key(v, key),
                        false,
                    )
                }
            }
        };
        Cursor::new(current, rank, self)
    }

    /// Returns a mutable cursor positioned at the gap defined by the given
    /// lower bound.
    ///
    /// See [`lower_bound`][OrderedSkipList::lower_bound] for the bound
    /// semantics table.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [1, 5] { list.insert(v); }
    ///
    /// {
    ///     let mut cur = list.lower_bound_mut(Bound::Included(&3));
    ///     cur.insert_after(3).expect("3 is in order");
    /// }
    /// assert_eq!(list.len(), 3);
    /// ```
    #[inline]
    pub fn lower_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, T, N, C, G>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        let list_ptr = core::ptr::from_mut(self);
        let (current, rank) = match bound {
            Bound::Unbounded => (self.head, 0),
            Bound::Included(q) => {
                // advance on Less only: land before first >= q
                // SAFETY: self.head is valid for &*self.
                unsafe {
                    gap_find(
                        self.head,
                        q,
                        |v, key| self.comparator.compare_key(v, key),
                        false,
                    )
                }
            }
            Bound::Excluded(q) => {
                // advance on Less or Equal: land before first > q
                // SAFETY: self.head is valid for &*self.
                unsafe {
                    gap_find(
                        self.head,
                        q,
                        |v, key| self.comparator.compare_key(v, key),
                        true,
                    )
                }
            }
        };
        CursorMut::new(current, rank, list_ptr)
    }

    /// Returns a mutable cursor positioned at the gap defined by the given
    /// upper bound.
    ///
    /// See [`upper_bound`][OrderedSkipList::upper_bound] for the bound
    /// semantics table.
    ///
    /// This operation is `$O(\log n)$` on average (or `$O(1)$` for `Unbounded`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use core::ops::Bound;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// for v in [1, 5] { list.insert(v); }
    ///
    /// {
    ///     let mut cur = list.upper_bound_mut(Bound::Included(&3));
    ///     cur.insert_after(3).expect("3 is in order");
    /// }
    /// assert_eq!(list.len(), 3);
    /// ```
    #[inline]
    pub fn upper_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, T, N, C, G>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        let list_ptr = core::ptr::from_mut(self);
        let (current, rank) = match bound {
            Bound::Unbounded => match self.tail {
                Some(tail) => (tail, self.len),
                None => (self.head, 0),
            },
            Bound::Included(q) => {
                // advance on Less or Equal: land after last <= q
                // SAFETY: self.head is valid for &*self.
                unsafe {
                    gap_find(
                        self.head,
                        q,
                        |v, key| self.comparator.compare_key(v, key),
                        true,
                    )
                }
            }
            Bound::Excluded(q) => {
                // advance on Less only: land after last < q
                // SAFETY: self.head is valid for &*self.
                unsafe {
                    gap_find(
                        self.head,
                        q,
                        |v, key| self.comparator.compare_key(v, key),
                        false,
                    )
                }
            }
        };
        CursorMut::new(current, rank, list_ptr)
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::ordered_skip_list::OrderedSkipList;

    // Helper: build a list [1, 2, 3].
    fn list_123() -> OrderedSkipList<i32> {
        let mut l = OrderedSkipList::new();
        for v in [1, 2, 3] {
            l.insert(v);
        }
        l
    }

    // MARK: Empty list

    #[test]
    fn empty_lower_bound_unbounded() {
        let l = OrderedSkipList::<i32>::new();
        let cur = l.lower_bound(Bound::Unbounded);
        assert_eq!(cur.peek_prev(), None);
        assert_eq!(cur.peek_next(), None);
    }

    #[test]
    fn empty_upper_bound_unbounded() {
        let l = OrderedSkipList::<i32>::new();
        let cur = l.upper_bound(Bound::Unbounded);
        assert_eq!(cur.peek_prev(), None);
        assert_eq!(cur.peek_next(), None);
    }

    // MARK: lower_bound factory

    #[test]
    fn lower_bound_unbounded_is_leftmost() {
        let l = list_123();
        let cur = l.lower_bound(Bound::Unbounded);
        assert_eq!(cur.peek_prev(), None);
        assert_eq!(cur.peek_next(), Some(&1));
    }

    #[test]
    fn lower_bound_included_exact_match() {
        let l = list_123();
        let cur = l.lower_bound(Bound::Included(&2));
        assert_eq!(cur.peek_prev(), Some(&1));
        assert_eq!(cur.peek_next(), Some(&2));
    }

    #[test]
    fn lower_bound_included_before_all() {
        let l = list_123();
        let cur = l.lower_bound(Bound::Included(&0));
        assert_eq!(cur.peek_prev(), None);
        assert_eq!(cur.peek_next(), Some(&1));
    }

    #[test]
    fn lower_bound_included_after_all() {
        let l = list_123();
        let cur = l.lower_bound(Bound::Included(&99));
        assert_eq!(cur.peek_prev(), Some(&3));
        assert_eq!(cur.peek_next(), None);
    }

    #[test]
    fn lower_bound_excluded_exact_match() {
        let l = list_123();
        let cur = l.lower_bound(Bound::Excluded(&2));
        assert_eq!(cur.peek_prev(), Some(&2));
        assert_eq!(cur.peek_next(), Some(&3));
    }

    // MARK: upper_bound factory

    #[test]
    fn upper_bound_unbounded_is_rightmost() {
        let l = list_123();
        let cur = l.upper_bound(Bound::Unbounded);
        assert_eq!(cur.peek_prev(), Some(&3));
        assert_eq!(cur.peek_next(), None);
    }

    #[test]
    fn upper_bound_included_exact_match() {
        let l = list_123();
        let cur = l.upper_bound(Bound::Included(&2));
        assert_eq!(cur.peek_prev(), Some(&2));
        assert_eq!(cur.peek_next(), Some(&3));
    }

    #[test]
    fn upper_bound_excluded_exact_match() {
        let l = list_123();
        let cur = l.upper_bound(Bound::Excluded(&2));
        assert_eq!(cur.peek_prev(), Some(&1));
        assert_eq!(cur.peek_next(), Some(&2));
    }

    // MARK: Duplicates

    #[test]
    fn lower_bound_included_first_duplicate() {
        let mut l = OrderedSkipList::<i32>::new();
        for v in [1, 2, 2, 3] {
            l.insert(v);
        }
        // lower_bound(Included(&2)) lands before the FIRST 2.
        let cur = l.lower_bound(Bound::Included(&2));
        assert_eq!(cur.peek_prev(), Some(&1));
        assert_eq!(cur.peek_next(), Some(&2));
    }

    #[test]
    fn upper_bound_included_after_last_duplicate() {
        let mut l = OrderedSkipList::<i32>::new();
        for v in [1, 2, 2, 3] {
            l.insert(v);
        }
        // upper_bound(Included(&2)) lands after the LAST 2.
        let cur = l.upper_bound(Bound::Included(&2));
        assert_eq!(cur.peek_prev(), Some(&2));
        assert_eq!(cur.peek_next(), Some(&3));
    }

    // MARK: Navigation

    #[test]
    fn next_traverses_in_order() {
        let l = list_123();
        let mut cur = l.lower_bound(Bound::Unbounded);
        assert_eq!(cur.next(), Some(&1));
        assert_eq!(cur.next(), Some(&2));
        assert_eq!(cur.next(), Some(&3));
        assert_eq!(cur.next(), None);
    }

    #[test]
    fn prev_traverses_in_reverse() {
        let l = list_123();
        let mut cur = l.upper_bound(Bound::Unbounded);
        assert_eq!(cur.prev(), Some(&3));
        assert_eq!(cur.prev(), Some(&2));
        assert_eq!(cur.prev(), Some(&1));
        assert_eq!(cur.prev(), None);
    }

    #[test]
    fn next_then_prev_round_trip() {
        let l = list_123();
        let mut cur = l.lower_bound(Bound::Unbounded);
        assert_eq!(cur.next(), Some(&1)); // straddles 1
        assert_eq!(cur.prev(), Some(&1)); // straddles 1 back
        assert_eq!(cur.peek_prev(), None);
    }

    #[test]
    fn cursor_is_copy() {
        let l = list_123();
        let cur1 = l.lower_bound(Bound::Unbounded);
        let cur2 = cur1; // Copy
        assert_eq!(cur1.peek_next(), cur2.peek_next());
    }

    #[test]
    fn as_cursor_mirrors_cursor_mut_position() {
        let mut l = list_123();
        let cur = l.lower_bound_mut(Bound::Included(&2));
        let ro = cur.as_cursor();
        assert_eq!(ro.peek_prev(), Some(&1));
        assert_eq!(ro.peek_next(), Some(&2));
    }

    // MARK: CursorMut insert_after

    #[test]
    fn insert_after_valid() {
        let mut l = list_123();
        {
            let mut cur = l.lower_bound_mut(Bound::Included(&2));
            // gap: between 1 and 2; duplicates are allowed in OrderedSkipList
            cur.insert_after(2)
                .expect("inserting 2 in order should succeed");
        }
        let vals: Vec<_> = l.iter().copied().collect();
        assert_eq!(vals, [1, 2, 2, 3]);
    }

    #[test]
    fn insert_after_returns_error_on_less_than_left() {
        let mut l = list_123();
        let mut cur = l.lower_bound_mut(Bound::Included(&2));
        // gap: between 1 and 2; inserting 0 is < left neighbour 1
        assert_eq!(cur.insert_after(0), Err(UnorderedValueError(0)));
        assert_eq!(l.len(), 3); // unchanged
    }

    #[test]
    fn insert_after_returns_error_on_greater_than_right() {
        let mut l = list_123();
        let mut cur = l.lower_bound_mut(Bound::Included(&2));
        // gap: between 1 and 2; inserting 3 > right neighbour 2
        assert_eq!(cur.insert_after(3), Err(UnorderedValueError(3)));
    }

    #[test]
    fn insert_after_at_rightmost_gap() {
        let mut l = list_123();
        {
            let mut cur = l.upper_bound_mut(Bound::Unbounded);
            cur.insert_after(4)
                .expect("inserting 4 at rightmost gap should succeed");
        }
        assert_eq!(l.last(), Some(&4));
        assert_eq!(l.len(), 4);
    }

    // MARK: CursorMut insert_before

    #[test]
    fn insert_before_advances_cursor() {
        let mut l = list_123();
        let mut cur = l.lower_bound_mut(Bound::Included(&2));
        // gap: between 1 and 2
        cur.insert_before(2)
            .expect("inserting 2 in order should succeed"); // cursor advances to new 2
        assert_eq!(cur.peek_prev(), Some(&2));
        assert_eq!(cur.peek_next(), Some(&2));
    }

    // MARK: CursorMut remove_next

    #[test]
    fn remove_next_basic() {
        let mut l = list_123();
        {
            let mut cur = l.lower_bound_mut(Bound::Unbounded);
            assert_eq!(cur.remove_next(), Some(1));
            assert_eq!(cur.remove_next(), Some(2));
        }
        assert_eq!(l.len(), 1);
        assert_eq!(l.first(), Some(&3));
    }

    #[test]
    fn remove_next_at_rightmost_gap_returns_none() {
        let mut l = list_123();
        let mut cur = l.upper_bound_mut(Bound::Unbounded);
        assert_eq!(cur.remove_next(), None);
    }

    #[test]
    fn remove_next_updates_tail() {
        let mut l = list_123();
        {
            let mut cur = l.upper_bound_mut(Bound::Excluded(&3));
            // cur is after 2, next = 3 (the tail)
            cur.remove_next();
        }
        assert_eq!(l.last(), Some(&2));
    }

    // MARK: CursorMut remove_prev

    #[test]
    fn remove_prev_basic() {
        let mut l = list_123();
        {
            let mut cur = l.upper_bound_mut(Bound::Unbounded);
            assert_eq!(cur.remove_prev(), Some(3));
            assert_eq!(cur.remove_prev(), Some(2));
        }
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn remove_prev_at_leftmost_gap_returns_none() {
        let mut l = list_123();
        let mut cur = l.lower_bound_mut(Bound::Unbounded);
        assert_eq!(cur.remove_prev(), None);
    }

    #[test]
    fn remove_prev_updates_tail() {
        let mut l = list_123();
        {
            let mut cur = l.upper_bound_mut(Bound::Unbounded);
            cur.remove_prev(); // removes 3
        }
        assert_eq!(l.last(), Some(&2));
    }

    // MARK: CursorMut duplicates

    #[test]
    fn remove_next_with_duplicates() {
        let mut l = OrderedSkipList::<i32>::new();
        for v in [1, 2, 2, 2, 3] {
            l.insert(v);
        }
        {
            let mut cur = l.lower_bound_mut(Bound::Included(&2));
            // gap: between 1 and first 2
            assert_eq!(cur.remove_next(), Some(2));
            assert_eq!(cur.remove_next(), Some(2));
        }
        assert_eq!(l.len(), 3);
        let vals: Vec<_> = l.iter().copied().collect();
        assert_eq!(vals, [1, 2, 3]);
    }

    // MARK: Incremental precursor cache tests

    /// Insert, advance cursor, then insert again — exercises the incremental
    /// precursor update path in `next()`.
    #[test]
    fn insert_then_next_then_insert_again() {
        let mut l = list_123();
        {
            let mut cur = l.lower_bound_mut(Bound::Included(&2));
            // gap: between 1 and 2; insert a duplicate (OSL allows it)
            cur.insert_after(2)
                .expect("inserting 2 in order should succeed");
            // advance past the new 2; precursors update incrementally
            assert_eq!(cur.next(), Some(&2));
            // gap is now between first 2 and original 2; insert another duplicate
            cur.insert_after(2)
                .expect("inserting another 2 should succeed");
        }
        let vals: Vec<_> = l.iter().copied().collect();
        assert_eq!(vals, [1, 2, 2, 2, 3]);
    }

    /// Remove next, then immediately insert after — verifies cache is correctly
    /// restored after `remove_next`.
    #[test]
    fn remove_next_then_insert_after() {
        let mut l = list_123();
        {
            let mut cur = l.lower_bound_mut(Bound::Unbounded);
            // remove 1 (the right neighbour at the leftmost gap)
            assert_eq!(cur.remove_next(), Some(1));
            // gap is now before 2; insert 0 before 2
            cur.insert_after(0)
                .expect("inserting 0 at leftmost gap should succeed");
        }
        let vals: Vec<_> = l.iter().copied().collect();
        assert_eq!(vals, [0, 2, 3]);
    }

    /// `insert_before` + `next` + `insert_after` — exercises `insert_before`
    /// advancing the cursor and the subsequent forward navigation.
    #[test]
    fn insert_before_then_next_then_insert_after() {
        let mut l = list_123();
        {
            let mut cur = l.lower_bound_mut(Bound::Included(&2));
            // gap: between 1 and 2; insert_before advances cursor to new node
            cur.insert_before(2)
                .expect("inserting 2 in order should succeed");
            // cursor is now after the new 2 (between new 2 and original 2)
            assert_eq!(cur.peek_prev(), Some(&2));
            // advance past original 2
            assert_eq!(cur.next(), Some(&2));
            // gap: between original 2 and 3; insert 2.5 (not possible with i32,
            // so insert another 2 — duplicates allowed in OSL)
            cur.insert_after(2)
                .expect("inserting 2 in order should succeed");
        }
        let vals: Vec<_> = l.iter().copied().collect();
        assert_eq!(vals, [1, 2, 2, 2, 3]);
    }
}
