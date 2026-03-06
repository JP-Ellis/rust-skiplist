//! Set-algebra predicates and lazy iterator adapters for [`SkipSet`](super::SkipSet).
//!
//! This module provides:
//!
//! - Predicate methods: [`is_disjoint`], [`is_subset`], [`is_superset`]
//! - Lazy iterator adapters: [`Difference`], [`Intersection`],
//!   [`SymmetricDifference`], [`Union`]
//!
//! All iterator adapters run in `$O(n + m)$` time and require no extra allocation.
//!
//! [`is_disjoint`]: super::SkipSet::is_disjoint
//! [`is_subset`]: super::SkipSet::is_subset
//! [`is_superset`]: super::SkipSet::is_superset

use core::{cmp::Ordering, fmt, iter::FusedIterator};

use crate::{
    comparator::{Comparator, OrdComparator},
    level_generator::LevelGenerator,
    ordered_skip_list::iter::Iter,
    skip_set::SkipSet,
};

/// Convenience alias for the peekable, sorted-element iterator used inside
/// all four set-operation adapters.
type PeekIter<'a, T, const N: usize> = core::iter::Peekable<Iter<'a, T, N>>;

// MARK: Difference

/// A lazy iterator over the elements of `self` that are not in `other`.
///
/// Constructed by [`SkipSet::difference`].
///
/// Elements are yielded in ascending order (as defined by the set's
/// comparator).  The iterator terminates when `self` is exhausted.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::SkipSet;
///
/// let mut a = SkipSet::<i32>::new();
/// for v in [1, 2, 3, 4] { a.insert(v); }
/// let mut b = SkipSet::<i32>::new();
/// for v in [2, 4] { b.insert(v); }
///
/// let diff: Vec<i32> = a.difference(&b).copied().collect();
/// assert_eq!(diff, [1, 3]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Difference<'a, T, const N: usize = 16, C: Comparator<T> = OrdComparator> {
    /// Peekable iterator over the elements of `self`.
    self_iter: PeekIter<'a, T, N>,
    /// Peekable iterator over the elements of `other`.
    other_iter: PeekIter<'a, T, N>,
    /// Comparator used to determine element order.
    comparator: &'a C,
}

impl<T: fmt::Debug, const N: usize, C: Comparator<T>> fmt::Debug for Difference<'_, T, N, C> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(Difference {
                self_iter: self.self_iter.clone(),
                other_iter: self.other_iter.clone(),
                comparator: self.comparator,
            })
            .finish()
    }
}

impl<'a, T, const N: usize, C: Comparator<T>> Iterator for Difference<'a, T, N, C> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Peek without consuming; extract Ordering so both borrows end
            // before we call next() on either iterator.
            let ord = match (self.self_iter.peek(), self.other_iter.peek()) {
                (None, _) => return None,
                (Some(_), None) => Ordering::Less,
                (Some(a), Some(b)) => self.comparator.compare(*a, *b),
            };
            match ord {
                // Self element is strictly smaller → not in other → yield it.
                Ordering::Less => return self.self_iter.next(),
                // Equal → skip both (self element is in other).
                Ordering::Equal => {
                    self.self_iter.next();
                    self.other_iter.next();
                }
                // Other element is smaller → advance other, keep looking.
                Ordering::Greater => {
                    self.other_iter.next();
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // Upper bound: at most as many elements as remain in self.
        (0, self.self_iter.size_hint().1)
    }
}

impl<T, const N: usize, C: Comparator<T>> FusedIterator for Difference<'_, T, N, C> {}

// MARK: Intersection

/// A lazy iterator over the elements present in both sets.
///
/// Constructed by [`SkipSet::intersection`].
///
/// Elements are yielded in ascending order (as defined by the set's
/// comparator).  The iterator terminates when either set is exhausted.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::SkipSet;
///
/// let mut a = SkipSet::<i32>::new();
/// for v in [1, 2, 3] { a.insert(v); }
/// let mut b = SkipSet::<i32>::new();
/// for v in [2, 3, 4] { b.insert(v); }
///
/// let inter: Vec<i32> = a.intersection(&b).copied().collect();
/// assert_eq!(inter, [2, 3]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Intersection<'a, T, const N: usize = 16, C: Comparator<T> = OrdComparator> {
    /// Peekable iterator over the elements of `self`.
    self_iter: PeekIter<'a, T, N>,
    /// Peekable iterator over the elements of `other`.
    other_iter: PeekIter<'a, T, N>,
    /// Comparator used to determine element order.
    comparator: &'a C,
}

impl<T: fmt::Debug, const N: usize, C: Comparator<T>> fmt::Debug for Intersection<'_, T, N, C> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(Intersection {
                self_iter: self.self_iter.clone(),
                other_iter: self.other_iter.clone(),
                comparator: self.comparator,
            })
            .finish()
    }
}

impl<'a, T, const N: usize, C: Comparator<T>> Iterator for Intersection<'a, T, N, C> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let ord = match (self.self_iter.peek(), self.other_iter.peek()) {
                // Either exhausted → no more common elements.
                (None, _) | (_, None) => return None,
                (Some(a), Some(b)) => self.comparator.compare(*a, *b),
            };
            match ord {
                // Equal → present in both → yield from self.
                Ordering::Equal => {
                    self.other_iter.next();
                    return self.self_iter.next();
                }
                // Self is smaller → not in other → advance self.
                Ordering::Less => {
                    self.self_iter.next();
                }
                // Other is smaller → not in self → advance other.
                Ordering::Greater => {
                    self.other_iter.next();
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // Upper bound: at most min(remaining_self, remaining_other).
        let a = self.self_iter.size_hint().1;
        let b = self.other_iter.size_hint().1;
        let upper = match (a, b) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        };
        (0, upper)
    }
}

impl<T, const N: usize, C: Comparator<T>> FusedIterator for Intersection<'_, T, N, C> {}

// MARK: SymmetricDifference

/// A lazy iterator over the elements in exactly one of the two sets.
///
/// Constructed by [`SkipSet::symmetric_difference`].
///
/// Elements are yielded in ascending order (as defined by the set's
/// comparator).
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::SkipSet;
///
/// let mut a = SkipSet::<i32>::new();
/// for v in [1, 2, 3] { a.insert(v); }
/// let mut b = SkipSet::<i32>::new();
/// for v in [2, 3, 4] { b.insert(v); }
///
/// let sym: Vec<i32> = a.symmetric_difference(&b).copied().collect();
/// assert_eq!(sym, [1, 4]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct SymmetricDifference<'a, T, const N: usize = 16, C: Comparator<T> = OrdComparator> {
    /// Peekable iterator over the elements of `self`.
    self_iter: PeekIter<'a, T, N>,
    /// Peekable iterator over the elements of `other`.
    other_iter: PeekIter<'a, T, N>,
    /// Comparator used to determine element order.
    comparator: &'a C,
}

impl<T: fmt::Debug, const N: usize, C: Comparator<T>> fmt::Debug
    for SymmetricDifference<'_, T, N, C>
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(SymmetricDifference {
                self_iter: self.self_iter.clone(),
                other_iter: self.other_iter.clone(),
                comparator: self.comparator,
            })
            .finish()
    }
}

impl<'a, T, const N: usize, C: Comparator<T>> Iterator for SymmetricDifference<'a, T, N, C> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let ord = match (self.self_iter.peek(), self.other_iter.peek()) {
                // Self exhausted → drain other.
                (None, _) => return self.other_iter.next(),
                // Other exhausted → drain self.
                (Some(_), None) => return self.self_iter.next(),
                (Some(a), Some(b)) => self.comparator.compare(*a, *b),
            };
            match ord {
                // Self element is smaller → only in self → yield it.
                Ordering::Less => return self.self_iter.next(),
                // Other element is smaller → only in other → yield it.
                Ordering::Greater => return self.other_iter.next(),
                // Equal → in both → skip both and continue.
                Ordering::Equal => {
                    self.self_iter.next();
                    self.other_iter.next();
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // Upper bound: at most remaining_self + remaining_other elements.
        let a = self.self_iter.size_hint().1;
        let b = self.other_iter.size_hint().1;
        let upper = match (a, b) {
            (Some(x), Some(y)) => Some(x.saturating_add(y)),
            _ => None,
        };
        (0, upper)
    }
}

impl<T, const N: usize, C: Comparator<T>> FusedIterator for SymmetricDifference<'_, T, N, C> {}

// MARK: Union

/// A lazy iterator over the elements in either or both sets.
///
/// Constructed by [`SkipSet::union`].
///
/// Elements are yielded in ascending order (as defined by the set's
/// comparator).  When an element appears in both sets, it is yielded once.
///
/// # Examples
///
/// ```rust
/// use skiplist::skip_set::SkipSet;
///
/// let mut a = SkipSet::<i32>::new();
/// for v in [1, 2, 3] { a.insert(v); }
/// let mut b = SkipSet::<i32>::new();
/// for v in [2, 3, 4] { b.insert(v); }
///
/// let u: Vec<i32> = a.union(&b).copied().collect();
/// assert_eq!(u, [1, 2, 3, 4]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Union<'a, T, const N: usize = 16, C: Comparator<T> = OrdComparator> {
    /// Peekable iterator over the elements of `self`.
    self_iter: PeekIter<'a, T, N>,
    /// Peekable iterator over the elements of `other`.
    other_iter: PeekIter<'a, T, N>,
    /// Comparator used to determine element order.
    comparator: &'a C,
}

impl<T: fmt::Debug, const N: usize, C: Comparator<T>> fmt::Debug for Union<'_, T, N, C> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(Union {
                self_iter: self.self_iter.clone(),
                other_iter: self.other_iter.clone(),
                comparator: self.comparator,
            })
            .finish()
    }
}

impl<'a, T, const N: usize, C: Comparator<T>> Iterator for Union<'a, T, N, C> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ord = match (self.self_iter.peek(), self.other_iter.peek()) {
            // Self exhausted → drain other.
            (None, _) => return self.other_iter.next(),
            // Other exhausted → drain self.
            (Some(_), None) => return self.self_iter.next(),
            (Some(a), Some(b)) => self.comparator.compare(*a, *b),
        };
        match ord {
            // Self element is smaller → yield it.
            Ordering::Less => self.self_iter.next(),
            // Other element is smaller → yield it.
            Ordering::Greater => self.other_iter.next(),
            // Equal → advance both, yield from self (one occurrence only).
            Ordering::Equal => {
                self.other_iter.next();
                self.self_iter.next()
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // Upper bound: at most remaining_self + remaining_other elements.
        let a = self.self_iter.size_hint().1;
        let b = self.other_iter.size_hint().1;
        let upper = match (a, b) {
            (Some(x), Some(y)) => Some(x.saturating_add(y)),
            _ => None,
        };
        (0, upper)
    }
}

impl<T, const N: usize, C: Comparator<T>> FusedIterator for Union<'_, T, N, C> {}

// MARK: SkipSet methods

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> SkipSet<T, N, C, G> {
    /// Returns `true` if the set has no elements in common with `other`.
    ///
    /// Equivalent to checking that the intersection is empty. Runs in `$O(n + m)$`
    /// time and terminates early on the first common element.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut a = SkipSet::<i32>::new();
    /// for v in [1, 2, 3] { a.insert(v); }
    /// let mut b = SkipSet::<i32>::new();
    /// for v in [4, 5, 6] { b.insert(v); }
    ///
    /// assert!(a.is_disjoint(&b));
    /// b.insert(3);
    /// assert!(!a.is_disjoint(&b));
    /// ```
    #[inline]
    #[must_use]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        let mut ai = self.inner.iter().peekable();
        let mut bi = other.inner.iter().peekable();
        loop {
            let ord = match (ai.peek(), bi.peek()) {
                (None, _) | (_, None) => return true,
                (Some(a), Some(b)) => self.inner.comparator().compare(*a, *b),
            };
            match ord {
                Ordering::Equal => return false,
                Ordering::Less => {
                    ai.next();
                }
                Ordering::Greater => {
                    bi.next();
                }
            }
        }
    }

    /// Returns `true` if every element of `self` is also in `other`.
    ///
    /// Runs in `$O(n + m)$` time and terminates early when an element of `self`
    /// is found that is not present in `other`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut a = SkipSet::<i32>::new();
    /// for v in [1, 2] { a.insert(v); }
    /// let mut b = SkipSet::<i32>::new();
    /// for v in [1, 2, 3] { b.insert(v); }
    ///
    /// assert!(a.is_subset(&b));
    /// assert!(!b.is_subset(&a));
    /// ```
    #[inline]
    #[must_use]
    pub fn is_subset(&self, other: &Self) -> bool {
        if self.len() > other.len() {
            return false;
        }
        let mut ai = self.inner.iter().peekable();
        let mut bi = other.inner.iter().peekable();
        loop {
            let ord = match (ai.peek(), bi.peek()) {
                // All elements of self were found in other.
                (None, _) => return true,
                // Elements remain in self but other is exhausted.
                (Some(_), None) => return false,
                (Some(a), Some(b)) => self.inner.comparator().compare(*a, *b),
            };
            match ord {
                // Found in other; advance both.
                Ordering::Equal => {
                    ai.next();
                    bi.next();
                }
                // Self element not found in other.
                Ordering::Less => return false,
                // Other element smaller; advance other.
                Ordering::Greater => {
                    bi.next();
                }
            }
        }
    }

    /// Returns `true` if every element of `other` is also in `self`.
    ///
    /// Equivalent to `other.is_subset(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut a = SkipSet::<i32>::new();
    /// for v in [1, 2, 3] { a.insert(v); }
    /// let mut b = SkipSet::<i32>::new();
    /// for v in [1, 2] { b.insert(v); }
    ///
    /// assert!(a.is_superset(&b));
    /// assert!(!b.is_superset(&a));
    /// ```
    #[inline]
    #[must_use]
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Returns a lazy iterator over the elements in `self` that are not in
    /// `other`, in ascending order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut a = SkipSet::<i32>::new();
    /// for v in [1, 2, 3, 4] { a.insert(v); }
    /// let mut b = SkipSet::<i32>::new();
    /// for v in [2, 4] { b.insert(v); }
    ///
    /// let diff: Vec<i32> = a.difference(&b).copied().collect();
    /// assert_eq!(diff, [1, 3]);
    /// ```
    #[inline]
    pub fn difference<'a>(&'a self, other: &'a Self) -> Difference<'a, T, N, C> {
        Difference {
            self_iter: self.inner.iter().peekable(),
            other_iter: other.inner.iter().peekable(),
            comparator: self.inner.comparator(),
        }
    }

    /// Returns a lazy iterator over the elements present in both sets, in
    /// ascending order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut a = SkipSet::<i32>::new();
    /// for v in [1, 2, 3] { a.insert(v); }
    /// let mut b = SkipSet::<i32>::new();
    /// for v in [2, 3, 4] { b.insert(v); }
    ///
    /// let inter: Vec<i32> = a.intersection(&b).copied().collect();
    /// assert_eq!(inter, [2, 3]);
    /// ```
    #[inline]
    pub fn intersection<'a>(&'a self, other: &'a Self) -> Intersection<'a, T, N, C> {
        Intersection {
            self_iter: self.inner.iter().peekable(),
            other_iter: other.inner.iter().peekable(),
            comparator: self.inner.comparator(),
        }
    }

    /// Returns a lazy iterator over the elements in exactly one of the two
    /// sets, in ascending order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut a = SkipSet::<i32>::new();
    /// for v in [1, 2, 3] { a.insert(v); }
    /// let mut b = SkipSet::<i32>::new();
    /// for v in [2, 3, 4] { b.insert(v); }
    ///
    /// let sym: Vec<i32> = a.symmetric_difference(&b).copied().collect();
    /// assert_eq!(sym, [1, 4]);
    /// ```
    #[inline]
    pub fn symmetric_difference<'a>(&'a self, other: &'a Self) -> SymmetricDifference<'a, T, N, C> {
        SymmetricDifference {
            self_iter: self.inner.iter().peekable(),
            other_iter: other.inner.iter().peekable(),
            comparator: self.inner.comparator(),
        }
    }

    /// Returns a lazy iterator over the elements in either or both sets, in
    /// ascending order.  Duplicate elements (present in both) are yielded
    /// only once.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut a = SkipSet::<i32>::new();
    /// for v in [1, 2, 3] { a.insert(v); }
    /// let mut b = SkipSet::<i32>::new();
    /// for v in [2, 3, 4] { b.insert(v); }
    ///
    /// let u: Vec<i32> = a.union(&b).copied().collect();
    /// assert_eq!(u, [1, 2, 3, 4]);
    /// ```
    #[inline]
    pub fn union<'a>(&'a self, other: &'a Self) -> Union<'a, T, N, C> {
        Union {
            self_iter: self.inner.iter().peekable(),
            other_iter: other.inner.iter().peekable(),
            comparator: self.inner.comparator(),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use pretty_assertions::assert_eq;

    use super::SkipSet;
    use crate::comparator::FnComparator;

    // Named function pointer so both sets share the same comparator type.
    // Clippy: `&i32` is required here because `Comparator<T>` takes `&T`.
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "matches Comparator<T> signature"
    )]
    fn rev_cmp(x: &i32, y: &i32) -> Ordering {
        y.cmp(x)
    }

    type RevSet = SkipSet<i32, 16, FnComparator<fn(&i32, &i32) -> Ordering>>;

    fn make_rev_set(values: &[i32]) -> RevSet {
        // Coerce the function item to a function pointer via type annotation (no `as` cast).
        let fnptr: fn(&i32, &i32) -> Ordering = rev_cmp;
        let mut set = SkipSet::with_comparator(FnComparator(fnptr));
        for &v in values {
            set.insert(v);
        }
        set
    }

    fn make_set(values: &[i32]) -> SkipSet<i32> {
        let mut set = SkipSet::new();
        for &v in values {
            set.insert(v);
        }
        set
    }

    fn collect_diff(a: &SkipSet<i32>, b: &SkipSet<i32>) -> Vec<i32> {
        a.difference(b).copied().collect()
    }

    fn collect_inter(a: &SkipSet<i32>, b: &SkipSet<i32>) -> Vec<i32> {
        a.intersection(b).copied().collect()
    }

    fn collect_sym(a: &SkipSet<i32>, b: &SkipSet<i32>) -> Vec<i32> {
        a.symmetric_difference(b).copied().collect()
    }

    fn collect_union(a: &SkipSet<i32>, b: &SkipSet<i32>) -> Vec<i32> {
        a.union(b).copied().collect()
    }

    // MARK: is_disjoint

    #[test]
    fn is_disjoint_both_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        assert!(a.is_disjoint(&b));
    }

    #[test]
    fn is_disjoint_self_empty() {
        let a = make_set(&[]);
        let b = make_set(&[1, 2, 3]);
        assert!(a.is_disjoint(&b));
    }

    #[test]
    fn is_disjoint_other_empty() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[]);
        assert!(a.is_disjoint(&b));
    }

    #[test]
    fn is_disjoint_no_overlap() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[4, 5, 6]);
        assert!(a.is_disjoint(&b));
    }

    #[test]
    fn is_disjoint_partial_overlap() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[3, 4, 5]);
        assert!(!a.is_disjoint(&b));
    }

    #[test]
    fn is_disjoint_identical_sets() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 3]);
        assert!(!a.is_disjoint(&b));
    }

    #[test]
    fn is_disjoint_single_shared_element() {
        let a = make_set(&[1, 2, 5]);
        let b = make_set(&[3, 4, 5]);
        assert!(!a.is_disjoint(&b));
    }

    #[test]
    fn is_disjoint_custom_comparator() {
        // Largest-first ordering.
        let a = make_rev_set(&[3, 1]);
        let mut b = make_rev_set(&[4, 2]);
        assert!(a.is_disjoint(&b));
        b.insert(1);
        assert!(!a.is_disjoint(&b));
    }

    // MARK: is_subset

    #[test]
    fn is_subset_both_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        assert!(a.is_subset(&b));
    }

    #[test]
    fn is_subset_empty_is_subset_of_nonempty() {
        let a = make_set(&[]);
        let b = make_set(&[1, 2]);
        assert!(a.is_subset(&b));
    }

    #[test]
    fn is_subset_nonempty_is_not_subset_of_empty() {
        let a = make_set(&[1]);
        let b = make_set(&[]);
        assert!(!a.is_subset(&b));
    }

    #[test]
    fn is_subset_proper_subset() {
        let a = make_set(&[1, 2]);
        let b = make_set(&[1, 2, 3]);
        assert!(a.is_subset(&b));
    }

    #[test]
    fn is_subset_equal_sets() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 3]);
        assert!(a.is_subset(&b));
    }

    #[test]
    fn is_subset_not_subset_due_to_missing_element() {
        let a = make_set(&[1, 2, 4]);
        let b = make_set(&[1, 2, 3]);
        assert!(!a.is_subset(&b));
    }

    #[test]
    fn is_subset_not_subset_larger() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2]);
        assert!(!a.is_subset(&b));
    }

    #[test]
    fn is_subset_disjoint_not_subset() {
        let a = make_set(&[1, 2]);
        let b = make_set(&[3, 4]);
        assert!(!a.is_subset(&b));
    }

    // MARK: is_superset

    #[test]
    fn is_superset_basic() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2]);
        assert!(a.is_superset(&b));
        assert!(!b.is_superset(&a));
    }

    #[test]
    fn is_superset_equal_sets() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 3]);
        assert!(a.is_superset(&b));
        assert!(b.is_superset(&a));
    }

    #[test]
    fn is_superset_empty_is_superset_of_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        assert!(a.is_superset(&b));
    }

    // MARK: difference

    #[test]
    fn difference_both_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        assert_eq!(collect_diff(&a, &b), Vec::<i32>::new());
    }

    #[test]
    fn difference_self_empty() {
        let a = make_set(&[]);
        let b = make_set(&[1, 2, 3]);
        assert_eq!(collect_diff(&a, &b), Vec::<i32>::new());
    }

    #[test]
    fn difference_other_empty() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[]);
        assert_eq!(collect_diff(&a, &b), vec![1, 2, 3]);
    }

    #[test]
    fn difference_disjoint() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[4, 5, 6]);
        assert_eq!(collect_diff(&a, &b), vec![1, 2, 3]);
    }

    #[test]
    fn difference_identical() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 3]);
        assert_eq!(collect_diff(&a, &b), Vec::<i32>::new());
    }

    #[test]
    fn difference_partial_overlap() {
        let a = make_set(&[1, 2, 3, 4]);
        let b = make_set(&[2, 4]);
        assert_eq!(collect_diff(&a, &b), vec![1, 3]);
    }

    #[test]
    fn difference_all_in_other() {
        let a = make_set(&[1, 2]);
        let b = make_set(&[1, 2, 3, 4]);
        assert_eq!(collect_diff(&a, &b), Vec::<i32>::new());
    }

    #[test]
    fn difference_other_is_subset() {
        let a = make_set(&[1, 2, 3, 4, 5]);
        let b = make_set(&[2, 4]);
        assert_eq!(collect_diff(&a, &b), vec![1, 3, 5]);
    }

    #[test]
    fn difference_custom_comparator() {
        // Largest-first: [3, 2, 1]
        let a = make_rev_set(&[1, 2, 3]);
        let b = make_rev_set(&[2]);
        // a = {3,2,1} stored as [3,2,1]; b = {2} stored as [2]
        // difference(a, b) = {3, 1} yielded as [3, 1] (descending)
        let diff: Vec<i32> = a.difference(&b).copied().collect();
        assert_eq!(diff, vec![3, 1]);
    }

    #[test]
    fn difference_size_hint() {
        let a = make_set(&[1, 2, 3, 4]);
        let b = make_set(&[2, 4]);
        let iter = a.difference(&b);
        let (lo, hi) = iter.size_hint();
        assert_eq!(lo, 0);
        assert!(hi.is_some_and(|h| h >= 2)); // at least as big as the result
    }

    // MARK: intersection

    #[test]
    fn intersection_both_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        assert_eq!(collect_inter(&a, &b), Vec::<i32>::new());
    }

    #[test]
    fn intersection_self_empty() {
        let a = make_set(&[]);
        let b = make_set(&[1, 2, 3]);
        assert_eq!(collect_inter(&a, &b), Vec::<i32>::new());
    }

    #[test]
    fn intersection_other_empty() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[]);
        assert_eq!(collect_inter(&a, &b), Vec::<i32>::new());
    }

    #[test]
    fn intersection_disjoint() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[4, 5, 6]);
        assert_eq!(collect_inter(&a, &b), Vec::<i32>::new());
    }

    #[test]
    fn intersection_identical() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 3]);
        assert_eq!(collect_inter(&a, &b), vec![1, 2, 3]);
    }

    #[test]
    fn intersection_partial_overlap() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[2, 3, 4]);
        assert_eq!(collect_inter(&a, &b), vec![2, 3]);
    }

    #[test]
    fn intersection_one_is_subset() {
        let a = make_set(&[1, 2]);
        let b = make_set(&[1, 2, 3, 4]);
        assert_eq!(collect_inter(&a, &b), vec![1, 2]);
    }

    #[test]
    fn intersection_custom_comparator() {
        // Largest-first: stored [3, 2, 1] and [3, 1]
        let a = make_rev_set(&[1, 2, 3]);
        let b = make_rev_set(&[1, 3]);
        let inter: Vec<i32> = a.intersection(&b).copied().collect();
        assert_eq!(inter, vec![3, 1]); // yielded largest-first
    }

    // MARK: symmetric_difference

    #[test]
    fn symmetric_difference_both_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        assert_eq!(collect_sym(&a, &b), Vec::<i32>::new());
    }

    #[test]
    fn symmetric_difference_self_empty() {
        let a = make_set(&[]);
        let b = make_set(&[1, 2, 3]);
        assert_eq!(collect_sym(&a, &b), vec![1, 2, 3]);
    }

    #[test]
    fn symmetric_difference_other_empty() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[]);
        assert_eq!(collect_sym(&a, &b), vec![1, 2, 3]);
    }

    #[test]
    fn symmetric_difference_disjoint() {
        let a = make_set(&[1, 3, 5]);
        let b = make_set(&[2, 4, 6]);
        assert_eq!(collect_sym(&a, &b), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn symmetric_difference_identical() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 3]);
        assert_eq!(collect_sym(&a, &b), Vec::<i32>::new());
    }

    #[test]
    fn symmetric_difference_partial_overlap() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[2, 3, 4]);
        assert_eq!(collect_sym(&a, &b), vec![1, 4]);
    }

    #[test]
    fn symmetric_difference_custom_comparator() {
        // Largest-first.
        let a = make_rev_set(&[1, 2, 3]);
        let b = make_rev_set(&[2, 3, 4]);
        // a = {3,2,1}, b = {4,3,2}; sym_diff = {4,1} yielded as [4, 1]
        let sym: Vec<i32> = a.symmetric_difference(&b).copied().collect();
        assert_eq!(sym, vec![4, 1]);
    }

    // MARK: union

    #[test]
    fn union_both_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        assert_eq!(collect_union(&a, &b), Vec::<i32>::new());
    }

    #[test]
    fn union_self_empty() {
        let a = make_set(&[]);
        let b = make_set(&[1, 2, 3]);
        assert_eq!(collect_union(&a, &b), vec![1, 2, 3]);
    }

    #[test]
    fn union_other_empty() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[]);
        assert_eq!(collect_union(&a, &b), vec![1, 2, 3]);
    }

    #[test]
    fn union_disjoint() {
        let a = make_set(&[1, 3, 5]);
        let b = make_set(&[2, 4, 6]);
        assert_eq!(collect_union(&a, &b), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn union_identical() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 3]);
        assert_eq!(collect_union(&a, &b), vec![1, 2, 3]);
    }

    #[test]
    fn union_partial_overlap() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[2, 3, 4]);
        assert_eq!(collect_union(&a, &b), vec![1, 2, 3, 4]);
    }

    #[test]
    fn union_one_is_subset() {
        let a = make_set(&[1, 2]);
        let b = make_set(&[1, 2, 3, 4]);
        assert_eq!(collect_union(&a, &b), vec![1, 2, 3, 4]);
    }

    #[test]
    fn union_custom_comparator() {
        // Largest-first.
        let a = make_rev_set(&[1, 2, 3]);
        let b = make_rev_set(&[2, 3, 4]);
        // a = {3,2,1}, b = {4,3,2}; union = {4,3,2,1} yielded as [4,3,2,1]
        let u: Vec<i32> = a.union(&b).copied().collect();
        assert_eq!(u, vec![4, 3, 2, 1]);
    }

    #[test]
    fn union_size_hint() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[3, 4, 5]);
        let iter = a.union(&b);
        let (lo, hi) = iter.size_hint();
        assert_eq!(lo, 0);
        assert!(hi.is_some_and(|h| h >= 5)); // at least as big as the result
    }

    // MARK: Debug

    #[test]
    fn debug_difference() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[2]);
        let d = a.difference(&b);
        let s = format!("{d:?}");
        assert_eq!(s, "[1, 3]");
    }

    #[test]
    fn debug_intersection() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[2, 3, 4]);
        let i = a.intersection(&b);
        let s = format!("{i:?}");
        assert_eq!(s, "[2, 3]");
    }

    #[test]
    fn debug_symmetric_difference() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[2, 3, 4]);
        let sd = a.symmetric_difference(&b);
        let s = format!("{sd:?}");
        assert_eq!(s, "[1, 4]");
    }

    #[test]
    fn debug_union() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[3, 4, 5]);
        let u = a.union(&b);
        let s = format!("{u:?}");
        assert_eq!(s, "[1, 2, 3, 4, 5]");
    }
}
