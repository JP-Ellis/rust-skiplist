//! Entry API for [`SkipSet`].
//!
//! [`Entry`] is the return type of [`SkipSet::entry`] and represents either an
//! [`OccupiedEntry`] (an element that compares equal to the probe is already
//! present) or a [`VacantEntry`] (no such element exists).

use core::fmt;

use crate::{
    comparator::{Comparator, ComparatorKey, OrdComparator},
    level_generator::{LevelGenerator, geometric::Geometric},
    skip_set::SkipSet,
};

// MARK: Entry

/// A view into a single entry in a [`SkipSet`], obtained from [`SkipSet::entry`].
///
/// The entry is either occupied (an equal element is already present) or vacant
/// (no equal element exists).
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::{Entry, SkipSet};
///
/// let mut set = SkipSet::<i32>::new();
/// set.insert(1);
///
/// match set.entry(1) {
///     Entry::Occupied(e) => assert_eq!(e.get(), &1),
///     Entry::Vacant(_) => panic!("expected occupied"),
/// }
/// match set.entry(2) {
///     Entry::Occupied(_) => panic!("expected vacant"),
///     Entry::Vacant(e) => { e.insert(); }
/// }
/// assert_eq!(set.len(), 2);
/// ```
#[expect(
    clippy::exhaustive_enums,
    reason = "Entry is intentionally exhaustive with only Occupied and Vacant variants"
)]
pub enum Entry<
    'a,
    T,
    const N: usize = 16,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// An occupied entry: an element comparing equal to the probe is present.
    Occupied(OccupiedEntry<'a, T, N, C, G>),
    /// A vacant entry: no element comparing equal to the probe is present.
    Vacant(VacantEntry<'a, T, N, C, G>),
}

// MARK: VacantEntry

/// A view into a vacant entry in a [`SkipSet`].
///
/// Obtained from [`Entry::Vacant`].
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::{Entry, SkipSet};
///
/// let mut set = SkipSet::<i32>::new();
/// if let Entry::Vacant(e) = set.entry(5) {
///     e.insert();
/// }
/// assert!(set.contains(&5));
/// ```
#[non_exhaustive]
pub struct VacantEntry<
    'a,
    T,
    const N: usize = 16,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// The set for which this entry is vacant.
    set: &'a mut SkipSet<T, N, C, G>,
    /// The probe value used for comparator-based lookups on this entry.
    value: T,
}

// MARK: OccupiedEntry

/// A view into an occupied entry in a [`SkipSet`].
///
/// Obtained from [`Entry::Occupied`].
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::{Entry, SkipSet};
///
/// let mut set = SkipSet::<i32>::new();
/// set.insert(5);
/// if let Entry::Occupied(e) = set.entry(5) {
///     assert_eq!(e.get(), &5);
/// }
/// ```
#[non_exhaustive]
pub struct OccupiedEntry<
    'a,
    T,
    const N: usize = 16,
    C: Comparator<T> = OrdComparator,
    G: LevelGenerator = Geometric,
> {
    /// The set for which this entry is occupied.
    set: &'a mut SkipSet<T, N, C, G>,
    /// The probe value used for comparator-based lookups on this entry.
    value: T,
}

// MARK: Entry methods

impl<'a, T, C: Comparator<T>, G: LevelGenerator, const N: usize> Entry<'a, T, N, C, G> {
    /// Returns a shared reference to the probe key, whether occupied or vacant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// assert_eq!(set.entry(7).key(), &7);
    /// ```
    #[inline]
    #[must_use]
    pub fn key(&self) -> &T {
        match self {
            Self::Occupied(e) => e.key(),
            Self::Vacant(e) => e.key(),
        }
    }

    /// If vacant, inserts the probe key and returns a shared reference to the
    /// element (either newly inserted or already present).
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// set.entry(1).or_insert();
    /// set.entry(1).or_insert(); // no-op: already present
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn or_insert(self) -> &'a T
    where
        C: ComparatorKey<T, T>,
    {
        match self {
            Self::Occupied(e) => e.into_ref(),
            Self::Vacant(VacantEntry { set, value }) => set.inner.get_or_insert(value),
        }
    }

    /// If vacant, calls `f()` to produce the element to insert, then returns a
    /// shared reference to it (or to the already-present element).
    ///
    /// `f` is only called when the entry is vacant.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// set.entry(1).or_insert_with(|| 1);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn or_insert_with<F>(self, f: F) -> &'a T
    where
        F: FnOnce() -> T,
        C: ComparatorKey<T, T>,
    {
        match self {
            Self::Occupied(e) => e.into_ref(),
            Self::Vacant(VacantEntry { set, .. }) => set.inner.get_or_insert(f()),
        }
    }

    /// If vacant, inserts `T::default()` and returns a shared reference to the
    /// element.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// set.entry(0).or_default();
    /// assert!(set.contains(&0));
    /// ```
    #[inline]
    pub fn or_default(self) -> &'a T
    where
        T: Default,
        C: ComparatorKey<T, T>,
    {
        self.or_insert_with(T::default)
    }
}

// MARK: VacantEntry methods

impl<'a, T, C: Comparator<T>, G: LevelGenerator, const N: usize> VacantEntry<'a, T, N, C, G> {
    /// Returns a shared reference to the probe key.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::{Entry, SkipSet};
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// if let Entry::Vacant(e) = set.entry(5) {
    ///     assert_eq!(e.key(), &5);
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn key(&self) -> &T {
        &self.value
    }

    /// Consumes the entry without inserting, returning the probe key.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::{Entry, SkipSet};
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// if let Entry::Vacant(e) = set.entry(5) {
    ///     assert_eq!(e.into_value(), 5);
    /// }
    /// assert!(set.is_empty()); // nothing was inserted
    /// ```
    #[inline]
    #[must_use]
    pub fn into_value(self) -> T {
        self.value
    }

    /// Inserts the probe key into the set and returns a shared reference to it.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::{Entry, SkipSet};
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// if let Entry::Vacant(e) = set.entry(5) {
    ///     assert_eq!(e.insert(), &5);
    /// }
    /// assert!(set.contains(&5));
    /// ```
    #[inline]
    pub fn insert(self) -> &'a T {
        self.set.inner.get_or_insert(self.value)
    }
}

// MARK: OccupiedEntry methods

#[expect(
    clippy::expect_used,
    clippy::missing_panics_doc,
    reason = "OccupiedEntry invariant: the element is present in the set; \
              the expect fires only on internal invariant violation"
)]
impl<'a, T, C: Comparator<T>, G: LevelGenerator, const N: usize> OccupiedEntry<'a, T, N, C, G> {
    /// Returns a shared reference to the probe key.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::{Entry, SkipSet};
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// set.insert(5);
    /// if let Entry::Occupied(e) = set.entry(5) {
    ///     assert_eq!(e.key(), &5);
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn key(&self) -> &T {
        &self.value
    }

    /// Returns a shared reference to the element in the set that compares
    /// equal to the probe key.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::{Entry, SkipSet};
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// set.insert(5);
    /// if let Entry::Occupied(e) = set.entry(5) {
    ///     assert_eq!(e.get(), &5);
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self) -> &T
    where
        C: ComparatorKey<T, T>,
    {
        self.set
            .inner
            .get_fast(&self.value)
            .expect("OccupiedEntry invariant: element is present in set")
    }

    /// Consumes the entry and returns a shared reference to the element valid
    /// for the lifetime `'a` of the original set borrow.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::{Entry, SkipSet};
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// set.insert(5);
    /// let r: &i32 = match set.entry(5) {
    ///     Entry::Occupied(e) => e.into_ref(),
    ///     Entry::Vacant(_) => unreachable!(),
    /// };
    /// assert_eq!(r, &5);
    /// ```
    #[inline]
    #[must_use]
    pub fn into_ref(self) -> &'a T
    where
        C: ComparatorKey<T, T>,
    {
        let Self { set, value } = self;
        set.inner
            .get_fast(&value)
            .expect("OccupiedEntry invariant: element is present in set")
    }

    /// Removes the element from the set and returns it.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::{Entry, SkipSet};
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// set.insert(5);
    /// let v = match set.entry(5) {
    ///     Entry::Occupied(e) => e.remove(),
    ///     Entry::Vacant(_) => unreachable!(),
    /// };
    /// assert_eq!(v, 5);
    /// assert!(!set.contains(&5));
    /// ```
    #[inline]
    pub fn remove(self) -> T
    where
        C: ComparatorKey<T, T>,
    {
        self.set
            .inner
            .take_first(&self.value)
            .expect("OccupiedEntry invariant: element is present in set")
    }
}

// MARK: SkipSet::entry

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> SkipSet<T, N, C, G> {
    /// Returns an [`Entry`] for the given `value`.
    ///
    /// The entry is [`Entry::Occupied`] if an element comparing equal to
    /// `value` is already in the set, or [`Entry::Vacant`] otherwise.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::{Entry, SkipSet};
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// set.entry(1).or_insert();
    /// assert!(set.contains(&1));
    /// ```
    #[inline]
    pub fn entry(&mut self, value: T) -> Entry<'_, T, N, C, G>
    where
        C: ComparatorKey<T, T>,
    {
        if self.inner.contains(&value) {
            Entry::Occupied(OccupiedEntry { set: self, value })
        } else {
            Entry::Vacant(VacantEntry { set: self, value })
        }
    }
}

// MARK: Debug

impl<T: fmt::Debug, C: Comparator<T> + ComparatorKey<T, T>, G: LevelGenerator, const N: usize>
    fmt::Debug for Entry<'_, T, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Occupied(e) => fmt::Debug::fmt(e, f),
            Self::Vacant(e) => fmt::Debug::fmt(e, f),
        }
    }
}

impl<T: fmt::Debug, C: Comparator<T>, G: LevelGenerator, const N: usize> fmt::Debug
    for VacantEntry<'_, T, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VacantEntry")
            .field("key", &self.value)
            .finish()
    }
}

impl<T: fmt::Debug, C: Comparator<T> + ComparatorKey<T, T>, G: LevelGenerator, const N: usize>
    fmt::Debug for OccupiedEntry<'_, T, N, C, G>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("element", self.get())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::Entry;
    use crate::{comparator::FnComparator, skip_set::SkipSet};

    fn make_set(values: &[i32]) -> SkipSet<i32> {
        let mut set = SkipSet::new();
        for &v in values {
            set.insert(v);
        }
        set
    }

    // MARK: entry

    #[test]
    fn entry_vacant_on_empty_set() {
        let mut set = SkipSet::<i32>::new();
        assert!(matches!(set.entry(1), Entry::Vacant(_)));
    }

    #[test]
    fn entry_vacant_when_absent() {
        let mut set = make_set(&[1, 3]);
        assert!(matches!(set.entry(2), Entry::Vacant(_)));
    }

    #[test]
    fn entry_occupied_when_present() {
        let mut set = make_set(&[1, 2, 3]);
        assert!(matches!(set.entry(2), Entry::Occupied(_)));
    }

    #[test]
    fn entry_occupied_first() {
        let mut set = make_set(&[1, 2, 3]);
        assert!(matches!(set.entry(1), Entry::Occupied(_)));
    }

    #[test]
    fn entry_occupied_last() {
        let mut set = make_set(&[1, 2, 3]);
        assert!(matches!(set.entry(3), Entry::Occupied(_)));
    }

    #[test]
    fn entry_custom_comparator_vacant() {
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        set.insert(3);
        set.insert(1);
        assert!(matches!(set.entry(2), Entry::Vacant(_)));
    }

    #[test]
    fn entry_custom_comparator_occupied() {
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        set.insert(3);
        set.insert(1);
        assert!(matches!(set.entry(3), Entry::Occupied(_)));
    }

    // MARK: Entry::key

    #[test]
    fn entry_key_vacant() {
        let mut set = SkipSet::<i32>::new();
        assert_eq!(set.entry(7).key(), &7);
    }

    #[test]
    fn entry_key_occupied() {
        let mut set = make_set(&[7]);
        assert_eq!(set.entry(7).key(), &7);
    }

    // MARK: Entry::or_insert

    #[test]
    fn or_insert_vacant_inserts() {
        let mut set = SkipSet::<i32>::new();
        assert_eq!(set.entry(5).or_insert(), &5);
        assert_eq!(set.len(), 1);
        assert!(set.contains(&5));
    }

    #[test]
    fn or_insert_occupied_does_not_insert() {
        let mut set = make_set(&[5]);
        assert_eq!(set.entry(5).or_insert(), &5);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn or_insert_custom_comparator() {
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        set.insert(3);
        assert_eq!(set.entry(2).or_insert(), &2);
        assert_eq!(set.entry(3).or_insert(), &3);
        assert_eq!(set.len(), 2);
    }

    // MARK: Entry::or_insert_with

    #[test]
    fn or_insert_with_vacant_calls_f() {
        let mut set = SkipSet::<i32>::new();
        let mut called = false;
        let r = set.entry(5).or_insert_with(|| {
            called = true;
            5
        });
        assert_eq!(r, &5);
        assert!(called);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn or_insert_with_occupied_does_not_call_f() {
        let mut set = make_set(&[5]);
        let r = set
            .entry(5)
            .or_insert_with(|| panic!("f should not be called"));
        assert_eq!(r, &5);
        assert_eq!(set.len(), 1);
    }

    // MARK: Entry::or_default

    #[test]
    fn or_default_vacant_inserts_default() {
        let mut set = SkipSet::<i32>::new();
        set.entry(0).or_default();
        assert!(set.contains(&0));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn or_default_occupied_does_not_insert() {
        let mut set = make_set(&[0]);
        set.entry(0).or_default();
        assert_eq!(set.len(), 1);
    }

    // MARK: VacantEntry::key

    #[test]
    fn vacant_entry_key() {
        let mut set = SkipSet::<i32>::new();
        if let Entry::Vacant(e) = set.entry(42) {
            assert_eq!(e.key(), &42);
        } else {
            panic!("expected vacant");
        }
    }

    // MARK: VacantEntry::into_value

    #[test]
    fn vacant_entry_into_value_no_insert() {
        let mut set = SkipSet::<i32>::new();
        if let Entry::Vacant(e) = set.entry(42) {
            assert_eq!(e.into_value(), 42);
        } else {
            panic!("expected vacant");
        }
        assert!(set.is_empty());
    }

    // MARK: VacantEntry::insert

    #[test]
    fn vacant_entry_insert() {
        let mut set = SkipSet::<i32>::new();
        if let Entry::Vacant(e) = set.entry(5) {
            assert_eq!(e.insert(), &5);
        } else {
            panic!("expected vacant");
        }
        assert!(set.contains(&5));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn vacant_entry_insert_at_front() {
        let mut set = make_set(&[5, 10]);
        if let Entry::Vacant(e) = set.entry(1) {
            e.insert();
        } else {
            panic!("expected vacant");
        }
        assert_eq!(set.first(), Some(&1));
    }

    #[test]
    fn vacant_entry_insert_at_back() {
        let mut set = make_set(&[1, 5]);
        if let Entry::Vacant(e) = set.entry(10) {
            e.insert();
        } else {
            panic!("expected vacant");
        }
        assert_eq!(set.last(), Some(&10));
    }

    #[test]
    fn vacant_entry_insert_custom_comparator() {
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        set.insert(3);
        set.insert(1);
        if let Entry::Vacant(e) = set.entry(2) {
            assert_eq!(e.insert(), &2);
        } else {
            panic!("expected vacant");
        }
        assert_eq!(set.len(), 3);
        assert_eq!(set.first(), Some(&3)); // largest-first
    }

    // MARK: OccupiedEntry::key

    #[test]
    fn occupied_entry_key() {
        let mut set = make_set(&[5]);
        if let Entry::Occupied(e) = set.entry(5) {
            assert_eq!(e.key(), &5);
        } else {
            panic!("expected occupied");
        }
    }

    // MARK: OccupiedEntry::get

    #[test]
    fn occupied_entry_get() {
        let mut set = make_set(&[1, 2, 3]);
        if let Entry::Occupied(e) = set.entry(2) {
            assert_eq!(e.get(), &2);
        } else {
            panic!("expected occupied");
        }
    }

    #[test]
    fn occupied_entry_get_first() {
        let mut set = make_set(&[1, 2, 3]);
        if let Entry::Occupied(e) = set.entry(1) {
            assert_eq!(e.get(), &1);
        } else {
            panic!("expected occupied");
        }
    }

    #[test]
    fn occupied_entry_get_custom_comparator() {
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        set.insert(3);
        set.insert(2);
        set.insert(1);
        if let Entry::Occupied(e) = set.entry(2) {
            assert_eq!(e.get(), &2);
        } else {
            panic!("expected occupied");
        }
    }

    // MARK: OccupiedEntry::into_ref

    #[test]
    fn occupied_entry_into_ref() {
        let mut set = make_set(&[1, 2, 3]);
        let r: &i32 = match set.entry(2) {
            Entry::Occupied(e) => e.into_ref(),
            Entry::Vacant(_) => panic!("expected occupied"),
        };
        assert_eq!(r, &2);
    }

    #[test]
    fn occupied_entry_into_ref_lifetime() {
        // Ensure the returned reference lives as long as the set borrow.
        let mut set = make_set(&[42]);
        let r: &i32 = match set.entry(42) {
            Entry::Occupied(e) => e.into_ref(),
            Entry::Vacant(_) => panic!("expected occupied"),
        };
        // `r` is still usable after the entry is consumed.
        assert_eq!(*r, 42);
    }

    // MARK: OccupiedEntry::remove

    #[test]
    fn occupied_entry_remove() {
        let mut set = make_set(&[1, 2, 3]);
        let v = match set.entry(2) {
            Entry::Occupied(e) => e.remove(),
            Entry::Vacant(_) => panic!("expected occupied"),
        };
        assert_eq!(v, 2);
        assert_eq!(set.len(), 2);
        assert!(!set.contains(&2));
    }

    #[test]
    fn occupied_entry_remove_first_element() {
        let mut set = make_set(&[1, 2, 3]);
        let v = match set.entry(1) {
            Entry::Occupied(e) => e.remove(),
            Entry::Vacant(_) => panic!("expected occupied"),
        };
        assert_eq!(v, 1);
        assert_eq!(set.first(), Some(&2));
    }

    #[test]
    fn occupied_entry_remove_last_element() {
        let mut set = make_set(&[1, 2, 3]);
        let v = match set.entry(3) {
            Entry::Occupied(e) => e.remove(),
            Entry::Vacant(_) => panic!("expected occupied"),
        };
        assert_eq!(v, 3);
        assert_eq!(set.last(), Some(&2));
    }

    #[test]
    fn occupied_entry_remove_custom_comparator() {
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        set.insert(3);
        set.insert(2);
        set.insert(1);
        let v = match set.entry(2) {
            Entry::Occupied(e) => e.remove(),
            Entry::Vacant(_) => panic!("expected occupied"),
        };
        assert_eq!(v, 2);
        assert_eq!(set.len(), 2);
        assert!(!set.contains(&2));
    }

    // MARK: Debug

    #[test]
    fn debug_vacant_entry() {
        let mut set = SkipSet::<i32>::new();
        if let Entry::Vacant(e) = set.entry(5) {
            let s = format!("{e:?}");
            assert!(s.contains("VacantEntry"));
            assert!(s.contains('5'));
        } else {
            panic!("expected vacant");
        }
    }

    #[test]
    fn debug_occupied_entry() {
        let mut set = make_set(&[5]);
        if let Entry::Occupied(e) = set.entry(5) {
            let s = format!("{e:?}");
            assert!(s.contains("OccupiedEntry"));
            assert!(s.contains('5'));
        } else {
            panic!("expected occupied");
        }
    }

    #[test]
    fn debug_entry_vacant() {
        let mut set = SkipSet::<i32>::new();
        let s = format!("{:?}", set.entry(5));
        assert!(s.contains("VacantEntry"));
    }

    #[test]
    fn debug_entry_occupied() {
        let mut set = make_set(&[5]);
        let s = format!("{:?}", set.entry(5));
        assert!(s.contains("OccupiedEntry"));
    }
}
