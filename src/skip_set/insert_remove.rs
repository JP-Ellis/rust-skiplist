//! Insertion and removal for [`SkipSet`](super::SkipSet).

use crate::{
    comparator::{Comparator, ComparatorKey},
    level_generator::LevelGenerator,
    skip_set::SkipSet,
};

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> SkipSet<T, N, C, G> {
    /// Inserts `value` into the set if no element comparing equal to it is
    /// already present.
    ///
    /// Returns `true` if `value` was newly inserted, or `false` if an equal
    /// element was already in the set.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// assert!(set.insert(1));
    /// assert!(!set.insert(1)); // duplicate
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        let old_len = self.inner.len();
        self.inner.get_or_insert(value);
        self.inner.len() > old_len
    }

    /// Removes the element equal to `value`, inserts `value` in its place,
    /// and returns the removed element.
    ///
    /// Returns `None` if no equal element was present (the value is still
    /// inserted).
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// assert_eq!(set.replace(1), None); // not present → inserts, returns None
    /// assert_eq!(set.replace(1), Some(1)); // present → replaces, returns old
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn replace(&mut self, value: T) -> Option<T>
    where
        C: ComparatorKey<T, T>,
    {
        let old = self.inner.take_first(&value);
        self.inner.insert(value);
        old
    }

    /// Removes the element equal to `value` from the set and returns it, or
    /// returns `None` if no equal element is present.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// assert_eq!(set.take(&1), None);
    /// ```
    #[inline]
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        self.inner.take_first(value)
    }

    /// Removes the element equal to `value` from the set.
    ///
    /// Returns `true` if the element was present and removed, or `false` if
    /// no equal element was in the set.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// assert!(!set.remove(&1));
    /// ```
    #[inline]
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        Q: ?Sized,
        C: ComparatorKey<T, Q>,
    {
        self.inner.take_first(value).is_some()
    }

    /// Returns a shared reference to the element equal to `value`, inserting
    /// `value` if no equal element is present.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// assert_eq!(set.get_or_insert(1), &1); // absent → inserts
    /// assert_eq!(set.get_or_insert(1), &1); // present → returns existing
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn get_or_insert(&mut self, value: T) -> &T {
        self.inner.get_or_insert(value)
    }

    /// Returns a shared reference to the element equal to `value`, or calls
    /// `f(value)` to produce the inserted element when `value` is absent.
    ///
    /// `f` is only called when no matching element is present.  The value
    /// returned by `f` **must** compare equal to `value` under the set's
    /// comparator; violating this contract breaks the set's ordering invariant.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// set.insert(10);
    /// // 5 is absent; f is called.
    /// assert_eq!(set.get_or_insert_with(5, |v| v), &5);
    /// // 10 is present; f is NOT called.
    /// assert_eq!(set.get_or_insert_with(10, |_| panic!("should not call f")), &10);
    /// assert_eq!(set.len(), 2);
    /// ```
    #[inline]
    pub fn get_or_insert_with<F>(&mut self, value: T, f: F) -> &T
    where
        F: FnOnce(T) -> T,
    {
        self.inner.get_or_insert_with(value, f)
    }

    /// Removes and returns the first (smallest) element, or `None` if the
    /// set is empty.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// assert_eq!(set.pop_first(), None);
    /// ```
    #[inline]
    pub fn pop_first(&mut self) -> Option<T> {
        self.inner.pop_first()
    }

    /// Removes and returns the last (largest) element, or `None` if the set
    /// is empty.
    ///
    /// This operation is `$O(\log n)$` on average.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// assert_eq!(set.pop_last(), None);
    /// ```
    #[inline]
    pub fn pop_last(&mut self) -> Option<T> {
        self.inner.pop_last()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::SkipSet;
    use crate::comparator::FnComparator;

    fn make_set(values: &[i32]) -> SkipSet<i32> {
        let mut set = SkipSet::new();
        for &v in values {
            set.insert(v);
        }
        set
    }

    // MARK: insert

    #[test]
    fn insert_into_empty() {
        let mut set = SkipSet::<i32>::new();
        assert!(set.insert(42));
        assert_eq!(set.len(), 1);
        assert!(!set.is_empty());
    }

    #[test]
    fn insert_duplicate_returns_false() {
        let mut set = SkipSet::<i32>::new();
        assert!(set.insert(1));
        assert!(!set.insert(1));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn insert_distinct_values() {
        let mut set = SkipSet::<i32>::new();
        assert!(set.insert(3));
        assert!(set.insert(1));
        assert!(set.insert(2));
        assert_eq!(set.len(), 3);
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
    }

    #[test]
    fn insert_maintains_sorted_order() {
        let mut set = SkipSet::<i32>::new();
        for v in [5, 3, 1, 4, 2] {
            set.insert(v);
        }
        assert_eq!(set.first(), Some(&1));
        assert_eq!(set.last(), Some(&5));
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn insert_custom_comparator() {
        // Largest-first ordering.
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        assert!(set.insert(2));
        assert!(set.insert(3));
        assert!(set.insert(1));
        assert!(!set.insert(2)); // duplicate under custom comparator
        assert_eq!(set.len(), 3);
        assert_eq!(set.first(), Some(&3)); // largest first
        assert_eq!(set.last(), Some(&1));
    }

    // MARK: replace

    #[test]
    fn replace_absent_inserts_and_returns_none() {
        let mut set = SkipSet::<i32>::new();
        assert_eq!(set.replace(5), None);
        assert_eq!(set.len(), 1);
        assert!(set.contains(&5));
    }

    #[test]
    fn replace_present_returns_old() {
        let mut set = make_set(&[1, 2, 3]);
        assert_eq!(set.replace(2), Some(2));
        assert_eq!(set.len(), 3);
        assert!(set.contains(&2));
    }

    #[test]
    fn replace_first_element() {
        let mut set = make_set(&[1, 2, 3]);
        assert_eq!(set.replace(1), Some(1));
        assert_eq!(set.len(), 3);
        assert_eq!(set.first(), Some(&1));
    }

    #[test]
    fn replace_last_element() {
        let mut set = make_set(&[1, 2, 3]);
        assert_eq!(set.replace(3), Some(3));
        assert_eq!(set.len(), 3);
        assert_eq!(set.last(), Some(&3));
    }

    #[test]
    fn replace_custom_comparator() {
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        set.insert(3);
        set.insert(1);
        assert_eq!(set.replace(3), Some(3));
        assert_eq!(set.len(), 2);
    }

    // MARK: take

    #[test]
    fn take_absent_returns_none() {
        let mut set = SkipSet::<i32>::new();
        assert_eq!(set.take(&1), None);
    }

    #[test]
    fn take_present_removes_and_returns() {
        let mut set = make_set(&[1, 2, 3]);
        assert_eq!(set.take(&2), Some(2));
        assert_eq!(set.len(), 2);
        assert!(!set.contains(&2));
    }

    #[test]
    fn take_first_element() {
        let mut set = make_set(&[1, 2, 3]);
        assert_eq!(set.take(&1), Some(1));
        assert_eq!(set.len(), 2);
        assert_eq!(set.first(), Some(&2));
    }

    #[test]
    fn take_last_element() {
        let mut set = make_set(&[1, 2, 3]);
        assert_eq!(set.take(&3), Some(3));
        assert_eq!(set.len(), 2);
        assert_eq!(set.last(), Some(&2));
    }

    #[test]
    fn take_custom_comparator() {
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        set.insert(3);
        set.insert(1);
        set.insert(2);
        assert_eq!(set.take(&2), Some(2));
        assert_eq!(set.len(), 2);
    }

    // MARK: remove

    #[test]
    fn remove_absent_returns_false() {
        let mut set = SkipSet::<i32>::new();
        assert!(!set.remove(&1));
    }

    #[test]
    fn remove_present_returns_true() {
        let mut set = make_set(&[1, 2, 3]);
        assert!(set.remove(&2));
        assert_eq!(set.len(), 2);
        assert!(!set.contains(&2));
    }

    #[test]
    fn remove_first_element() {
        let mut set = make_set(&[1, 2, 3]);
        assert!(set.remove(&1));
        assert_eq!(set.first(), Some(&2));
    }

    #[test]
    fn remove_last_element() {
        let mut set = make_set(&[1, 2, 3]);
        assert!(set.remove(&3));
        assert_eq!(set.last(), Some(&2));
    }

    #[test]
    fn remove_custom_comparator() {
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        set.insert(3);
        set.insert(1);
        set.insert(2);
        assert!(set.remove(&2));
        assert!(!set.remove(&2)); // already removed
        assert_eq!(set.len(), 2);
    }

    // MARK: get_or_insert

    #[test]
    fn get_or_insert_empty() {
        let mut set = SkipSet::<i32>::new();
        assert_eq!(set.get_or_insert(42), &42);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn get_or_insert_absent_inserts() {
        let mut set = make_set(&[1, 3]);
        assert_eq!(set.get_or_insert(2), &2);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&2));
    }

    #[test]
    fn get_or_insert_present_does_not_insert() {
        let mut set = make_set(&[1, 2, 3]);
        assert_eq!(set.get_or_insert(2), &2);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn get_or_insert_custom_comparator() {
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        set.insert(3);
        set.insert(1);
        assert_eq!(set.get_or_insert(2), &2);
        assert_eq!(set.get_or_insert(3), &3); // present
        assert_eq!(set.len(), 3);
        assert_eq!(set.first(), Some(&3)); // largest first
    }

    // MARK: get_or_insert_with

    #[test]
    fn get_or_insert_with_calls_f_when_absent() {
        let mut set = SkipSet::<i32>::new();
        let mut called = false;
        let v = set.get_or_insert_with(5, |x| {
            called = true;
            x
        });
        assert_eq!(v, &5);
        assert!(called);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn get_or_insert_with_skips_f_when_present() {
        let mut set = make_set(&[1, 2, 3]);
        let v = set.get_or_insert_with(2, |_| panic!("f should not be called"));
        assert_eq!(v, &2);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn get_or_insert_with_custom_comparator() {
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        set.insert(3);
        set.insert(1);
        assert_eq!(set.get_or_insert_with(2, |v| v), &2);
        assert_eq!(set.get_or_insert_with(3, |_| panic!("should not call")), &3);
        assert_eq!(set.len(), 3);
    }

    // MARK: pop_first

    #[test]
    fn pop_first_empty() {
        let mut set = SkipSet::<i32>::new();
        assert_eq!(set.pop_first(), None);
    }

    #[test]
    fn pop_first_single() {
        let mut set = make_set(&[42]);
        assert_eq!(set.pop_first(), Some(42));
        assert!(set.is_empty());
    }

    #[test]
    fn pop_first_multiple() {
        let mut set = make_set(&[3, 1, 2]);
        assert_eq!(set.pop_first(), Some(1));
        assert_eq!(set.pop_first(), Some(2));
        assert_eq!(set.pop_first(), Some(3));
        assert_eq!(set.pop_first(), None);
    }

    #[test]
    fn pop_first_custom_comparator() {
        // Largest-first: first is the largest.
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        for v in [3, 1, 2] {
            set.insert(v);
        }
        assert_eq!(set.pop_first(), Some(3));
        assert_eq!(set.len(), 2);
    }

    // MARK: pop_last

    #[test]
    fn pop_last_empty() {
        let mut set = SkipSet::<i32>::new();
        assert_eq!(set.pop_last(), None);
    }

    #[test]
    fn pop_last_single() {
        let mut set = make_set(&[42]);
        assert_eq!(set.pop_last(), Some(42));
        assert!(set.is_empty());
    }

    #[test]
    fn pop_last_multiple() {
        let mut set = make_set(&[3, 1, 2]);
        assert_eq!(set.pop_last(), Some(3));
        assert_eq!(set.pop_last(), Some(2));
        assert_eq!(set.pop_last(), Some(1));
        assert_eq!(set.pop_last(), None);
    }

    #[test]
    fn pop_last_custom_comparator() {
        // Largest-first: last is the smallest.
        let mut set: SkipSet<i32, 16, _> =
            SkipSet::with_comparator(FnComparator(|a: &i32, b: &i32| b.cmp(a)));
        for v in [3, 1, 2] {
            set.insert(v);
        }
        assert_eq!(set.pop_last(), Some(1));
        assert_eq!(set.len(), 2);
    }

    // MARK: Borrow<Q> removals

    #[test]
    fn take_str_on_string_element() {
        let mut set: SkipSet<String> = SkipSet::new();
        set.insert("hello".to_owned());
        set.insert("world".to_owned());
        assert_eq!(set.take("hello"), Some("hello".to_owned()));
        assert!(!set.contains("hello"));
        assert_eq!(set.take("missing"), None);
    }

    #[test]
    fn remove_str_on_string_element() {
        let mut set: SkipSet<String> = SkipSet::new();
        set.insert("hello".to_owned());
        assert!(set.remove("hello"));
        assert!(!set.remove("missing"));
    }
}
