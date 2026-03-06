//! Standard trait implementations for [`SkipSet`](super::SkipSet):
//! `Debug`, `Clone`, `PartialEq`, `Eq`, `Hash`, `Extend`, `FromIterator`.
//!
//! `IntoIterator` is in `iter.rs`; `Default` is in the root `skip_set.rs`.

use core::{
    fmt,
    hash::{Hash, Hasher},
};

use crate::{comparator::Comparator, level_generator::LevelGenerator, skip_set::SkipSet};

// MARK: Debug

impl<T: fmt::Debug, C: Comparator<T>, G: LevelGenerator, const N: usize> fmt::Debug
    for SkipSet<T, N, C, G>
{
    /// Formats the set using the debug set notation, e.g. `{1, 2, 3}`.
    ///
    /// Elements are printed in sorted order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set: SkipSet<i32> = [3, 1, 2].into();
    /// assert_eq!(format!("{set:?}"), "{1, 2, 3}");
    /// ```
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

// MARK: Clone

impl<T: Clone, C: Comparator<T> + Clone, G: LevelGenerator + Clone, const N: usize> Clone
    for SkipSet<T, N, C, G>
{
    /// Returns a deep clone of the set.
    ///
    /// The cloned set has the same elements in the same sorted order.
    /// The comparator and level generator are both cloned.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set: SkipSet<i32> = [1, 2, 3].into();
    /// let cloned = set.clone();
    /// assert_eq!(set, cloned);
    /// ```
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// MARK: PartialEq / Eq

impl<
    T: PartialEq,
    C1: Comparator<T>,
    C2: Comparator<T>,
    G1: LevelGenerator,
    G2: LevelGenerator,
    const N: usize,
> PartialEq<SkipSet<T, N, C2, G2>> for SkipSet<T, N, C1, G1>
{
    /// Returns `true` if `self` and `other` contain the same elements in the
    /// same sorted order.
    ///
    /// The comparators and level generators do not need to match.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let a: SkipSet<i32> = [1, 2, 3].into();
    /// let b: SkipSet<i32> = [3, 1, 2].into();
    /// assert_eq!(a, b);
    ///
    /// let c: SkipSet<i32> = [1, 2, 4].into();
    /// assert_ne!(a, c);
    /// ```
    #[inline]
    fn eq(&self, other: &SkipSet<T, N, C2, G2>) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Eq, C: Comparator<T>, G: LevelGenerator, const N: usize> Eq for SkipSet<T, N, C, G> {}

// MARK: Hash

impl<T: Hash, C: Comparator<T>, G: LevelGenerator, const N: usize> Hash for SkipSet<T, N, C, G> {
    /// Hashes the length followed by each element in sorted order.
    ///
    /// Two sets that compare equal with `==` will produce the same hash.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    /// use std::collections::hash_map::DefaultHasher;
    /// use std::hash::{Hash, Hasher};
    ///
    /// fn hash_of<T: Hash>(val: &T) -> u64 {
    ///     let mut h = DefaultHasher::new();
    ///     val.hash(&mut h);
    ///     h.finish()
    /// }
    ///
    /// let a: SkipSet<i32> = [1, 2, 3].into();
    /// let b: SkipSet<i32> = [3, 1, 2].into();
    /// assert_eq!(hash_of(&a), hash_of(&b));
    /// ```
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for item in self {
            item.hash(state);
        }
    }
}

// MARK: Extend

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> Extend<T> for SkipSet<T, N, C, G> {
    /// Inserts all items from `iter` into the set, skipping duplicates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set: SkipSet<i32> = [1, 2].into();
    /// set.extend([2, 3, 4]);
    /// let collected: Vec<i32> = set.iter().copied().collect();
    /// assert_eq!(collected, [1, 2, 3, 4]);
    /// ```
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<'a, T: Copy + 'a, C: Comparator<T>, G: LevelGenerator, const N: usize> Extend<&'a T>
    for SkipSet<T, N, C, G>
{
    /// Copies all items from `iter` and inserts them into the set, skipping
    /// duplicates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set: SkipSet<i32> = [1, 2].into();
    /// let values = [2_i32, 3, 4];
    /// set.extend(values.iter());
    /// let collected: Vec<i32> = set.iter().copied().collect();
    /// assert_eq!(collected, [1, 2, 3, 4]);
    /// ```
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

// MARK: FromIterator

impl<T, C: Comparator<T> + Default, G: LevelGenerator + Default, const N: usize> FromIterator<T>
    for SkipSet<T, N, C, G>
{
    /// Creates a set from an iterator.
    ///
    /// Elements are inserted in sorted order regardless of the iteration order
    /// of the source.  Duplicate elements are silently discarded.  The set uses
    /// the default comparator and level generator for the type parameters `C`
    /// and `G`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set: SkipSet<i32> = [3, 1, 2, 1].into_iter().collect();
    /// let collected: Vec<i32> = set.iter().copied().collect();
    /// assert_eq!(collected, [1, 2, 3]);
    /// ```
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::with_comparator_and_level_generator(C::default(), G::default());
        set.extend(iter);
        set
    }
}

// MARK: From

impl<T, C: Comparator<T> + Default, G: LevelGenerator + Default, const N: usize, const M: usize>
    From<[T; M]> for SkipSet<T, N, C, G>
{
    /// Creates a set from a fixed-size array.
    ///
    /// Duplicate elements are silently discarded.  Uses the default comparator
    /// and level generator for `C` and `G`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set: SkipSet<i32> = [3, 1, 2, 1].into();
    /// let collected: Vec<i32> = set.iter().copied().collect();
    /// assert_eq!(collected, [1, 2, 3]);
    /// ```
    #[inline]
    fn from(arr: [T; M]) -> Self {
        arr.into_iter().collect()
    }
}

impl<T, C: Comparator<T> + Default, G: LevelGenerator + Default, const N: usize> From<Vec<T>>
    for SkipSet<T, N, C, G>
{
    /// Creates a set from a `Vec`, consuming it.
    ///
    /// Duplicate elements are silently discarded.  Uses the default comparator
    /// and level generator for `C` and `G`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let set: SkipSet<i32> = vec![3, 1, 2, 1].into();
    /// let collected: Vec<i32> = set.iter().copied().collect();
    /// assert_eq!(collected, [1, 2, 3]);
    /// ```
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        vec.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use pretty_assertions::assert_eq;

    use super::SkipSet;
    use crate::comparator::FnComparator;

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

    fn hash_of<T: Hash>(val: &T) -> u64 {
        let mut h = DefaultHasher::new();
        val.hash(&mut h);
        h.finish()
    }

    // MARK: Debug

    #[test]
    fn debug_empty() {
        let set = SkipSet::<i32>::new();
        assert_eq!(format!("{set:?}"), "{}");
    }

    #[test]
    fn debug_non_empty() {
        let set = make_set(&[1, 2, 3]);
        let s = format!("{set:?}");
        assert!(s.contains('1') && s.contains('2') && s.contains('3'));
    }

    #[test]
    #[expect(
        clippy::unwrap_used,
        reason = "test constructs the set with {1,2,3} so all chars are present"
    )]
    fn debug_sorted() {
        let set = make_set(&[3, 1, 2]);
        // Elements should appear in ascending order.
        let s = format!("{set:?}");
        let pos1 = s.find('1').unwrap();
        let pos2 = s.find('2').unwrap();
        let pos3 = s.find('3').unwrap();
        assert!(pos1 < pos2 && pos2 < pos3);
    }

    // MARK: Clone

    #[test]
    fn clone_empty() {
        let set = SkipSet::<i32>::new();
        let cloned = set.clone();
        assert!(cloned.is_empty());
    }

    #[test]
    fn clone_non_empty() {
        let set = make_set(&[1, 2, 3]);
        let cloned = set.clone();
        assert_eq!(to_vec(&cloned), [1, 2, 3]);
    }

    #[test]
    fn clone_is_independent() {
        let set = make_set(&[1, 2, 3]);
        let mut cloned = set.clone();
        cloned.insert(4);
        assert_eq!(set.len(), 3);
        assert_eq!(cloned.len(), 4);
    }

    #[test]
    fn clone_equals_original() {
        let set = make_set(&[1, 2, 3]);
        assert_eq!(set, set.clone());
    }

    // MARK: PartialEq / Eq

    #[test]
    fn eq_empty_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        assert_eq!(a, b);
    }

    #[test]
    fn eq_same_elements() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[3, 1, 2]);
        assert_eq!(a, b);
    }

    #[test]
    fn eq_different_lengths() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2]);
        assert!(a != b);
    }

    #[test]
    fn eq_different_elements() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 4]);
        assert!(a != b);
    }

    #[test]
    fn eq_empty_nonempty() {
        let a = make_set(&[]);
        let b = make_set(&[1]);
        assert!(a != b);
    }

    #[test]
    fn eq_reflexive() {
        let a = make_set(&[1, 2, 3]);
        assert_eq!(a, a);
    }

    // MARK: Hash

    #[test]
    fn hash_equal_sets_same_hash() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[3, 1, 2]);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn hash_empty_sets_equal() {
        let a = SkipSet::<i32>::new();
        let b = SkipSet::<i32>::new();
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn hash_different_sets_likely_differ() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 4]);
        // Different content → highly likely to differ (not guaranteed, but reliable for small sets).
        assert!(hash_of(&a) != hash_of(&b));
    }

    // MARK: Extend<T>

    #[test]
    fn extend_into_empty() {
        let mut set = SkipSet::<i32>::new();
        set.extend([3, 1, 2]);
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    #[test]
    fn extend_skips_duplicates() {
        let mut set = make_set(&[1, 2]);
        set.extend([2, 3, 3, 4]);
        assert_eq!(to_vec(&set), [1, 2, 3, 4]);
    }

    #[test]
    fn extend_with_empty_iter() {
        let mut set = make_set(&[1, 2, 3]);
        set.extend(core::iter::empty::<i32>());
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    // MARK: Extend<&T>

    #[test]
    fn extend_refs_into_empty() {
        let mut set = SkipSet::<i32>::new();
        let values = [3, 1, 2];
        set.extend(values.iter());
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    #[test]
    fn extend_refs_skips_duplicates() {
        let mut set = make_set(&[1, 2]);
        let values = [2_i32, 3, 3, 4];
        set.extend(values.iter());
        assert_eq!(to_vec(&set), [1, 2, 3, 4]);
    }

    // MARK: FromIterator

    #[test]
    fn from_iter_empty() {
        let set: SkipSet<i32> = core::iter::empty().collect();
        assert!(set.is_empty());
    }

    #[test]
    fn from_iter_sorted() {
        let set: SkipSet<i32> = [3, 1, 2].into_iter().collect();
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    #[test]
    fn from_iter_deduplicates() {
        let set: SkipSet<i32> = [1, 1, 2, 2, 3].into_iter().collect();
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    #[test]
    fn from_iter_custom_comparator() {
        use core::cmp::Ordering;
        #[expect(
            clippy::trivially_copy_pass_by_ref,
            reason = "must match Comparator<T> signature"
        )]
        fn rev(x: &i32, y: &i32) -> Ordering {
            y.cmp(x)
        }
        let fnptr: fn(&i32, &i32) -> Ordering = rev;
        // With a custom comparator, FromIterator uses C::default() which isn't
        // available for FnComparator.  Test via collect() on a standard SkipSet.
        let set: SkipSet<i32> = [3, 1, 2].into_iter().collect();
        assert_eq!(to_vec(&set), [1, 2, 3]);
        // Verify the fn pointer is usable (compiler check).
        drop(SkipSet::<i32, 16, _>::with_comparator(FnComparator(fnptr)));
    }

    // MARK: From<[T; N]>

    #[test]
    fn from_array_sorted() {
        let set: SkipSet<i32> = [3, 1, 2].into();
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    #[test]
    fn from_array_deduplicates() {
        let set: SkipSet<i32> = [1, 1, 2, 2, 3].into();
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    #[test]
    fn from_empty_array() {
        let set: SkipSet<i32> = [].into();
        assert!(set.is_empty());
    }

    // MARK: From<Vec<T>>

    #[test]
    fn from_vec_sorted() {
        let set: SkipSet<i32> = vec![3, 1, 2].into();
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    #[test]
    fn from_vec_deduplicates() {
        let set: SkipSet<i32> = vec![1, 1, 2, 2, 3].into();
        assert_eq!(to_vec(&set), [1, 2, 3]);
    }

    #[test]
    fn from_empty_vec() {
        let set: SkipSet<i32> = vec![].into();
        assert!(set.is_empty());
    }
}
