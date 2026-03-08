//! List-restructuring methods for [`SkipSet`](super::SkipSet):
//! `clear`, `append`, `split_off`.

use core::cmp::Ordering;

use crate::{comparator::Comparator, level_generator::LevelGenerator, skip_set::SkipSet};

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> SkipSet<T, N, C, G> {
    /// Removes all elements from the set.
    ///
    /// The comparator and level generator are preserved; elements can be
    /// inserted again immediately after calling `clear`.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// for v in [1, 2, 3] { set.insert(v); }
    /// set.clear();
    /// assert!(set.is_empty());
    /// assert_eq!(set.len(), 0);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// Elements of `other` that are already present in `self` are discarded
    /// (the existing element in `self` is kept).  After the call, `self`
    /// contains the union of both sets.
    ///
    /// This operation is `$O(n+m)$` when the element ranges are strictly
    /// disjoint (`self.last() < other.first()` or `other.last() < self.first()`
    /// according to the comparator), and `$O(m \log(n+m))$` when the ranges
    /// overlap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut a = SkipSet::<i32>::new();
    /// for v in [1, 3, 5] { a.insert(v); }
    ///
    /// let mut b = SkipSet::<i32>::new();
    /// for v in [2, 4, 6] { b.insert(v); }
    ///
    /// a.append(&mut b);
    /// assert!(b.is_empty());
    /// let collected: Vec<i32> = a.iter().copied().collect();
    /// assert_eq!(collected, [1, 2, 3, 4, 5, 6]);
    /// ```
    #[expect(
        clippy::expect_used,
        clippy::missing_panics_doc,
        reason = "self.last/first() / other.last/first() return None only when empty; \
                  both sets are checked to be non-empty before the expect calls"
    )]
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        if other.is_empty() {
            return;
        }

        // Fast path: if self is empty, or the two ranges are strictly disjoint
        // in either direction (self.last < other.first, or other.last <
        // self.first), no duplicates can arise.  We delegate directly to the
        // underlying OrderedSkipList, which picks the appropriate splice
        // direction and rebuilds skip links in O(n+m).
        let disjoint = if self.is_empty() {
            true
        } else {
            let cmp = self.inner.comparator();
            let self_last = self.last().expect("self is non-empty in this branch");
            let other_first = other.first().expect("other is non-empty in this branch");
            let other_last = other.last().expect("other is non-empty in this branch");
            let self_first = self.first().expect("self is non-empty in this branch");
            // Forward: self.last < other.first
            cmp.compare(self_last, other_first) == Ordering::Less
                // Reverse: other.last < self.first
                || cmp.compare(other_last, self_first) == Ordering::Less
        };

        if disjoint {
            self.inner.append(&mut other.inner);
        } else {
            while let Some(v) = other.pop_first() {
                self.insert(v);
            }
        }
    }

    /// Splits the set at `index`, returning a new set containing all elements
    /// at positions `>= index` in sorted order.
    ///
    /// After the call, `self` retains the first `index` elements and the
    /// returned set contains the rest.  If `index >= self.len()`, `self` is
    /// unchanged and the returned set is empty.
    ///
    /// This operation is `$O(n)$` overall.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// for v in 1..=5 { set.insert(v); }
    ///
    /// let right = set.split_off_index(2);
    /// let left_vals: Vec<i32> = set.iter().copied().collect();
    /// let right_vals: Vec<i32> = right.iter().copied().collect();
    /// assert_eq!(left_vals, [1, 2]);
    /// assert_eq!(right_vals, [3, 4, 5]);
    /// ```
    #[inline]
    #[must_use]
    pub fn split_off_index(&mut self, index: usize) -> Self
    where
        C: Clone,
        G: Clone,
    {
        Self {
            inner: self.inner.split_off_index(index.min(self.len())),
        }
    }

    /// Splits the set at `value`, returning a new set containing all elements
    /// with value `>= value`.
    ///
    /// After the call, `self` retains all elements with value `< value`, and
    /// the returned set contains all elements with value `>= value`.
    ///
    /// This operation is `$O(n)$` overall.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut a = SkipSet::<i32>::new();
    /// for v in 1..=5 { a.insert(v); }
    ///
    /// let b = a.split_off(&3);
    /// let a_vals: Vec<i32> = a.iter().copied().collect();
    /// let b_vals: Vec<i32> = b.iter().copied().collect();
    /// assert_eq!(a_vals, [1, 2]);
    /// assert_eq!(b_vals, [3, 4, 5]);
    /// ```
    #[inline]
    #[must_use]
    pub fn split_off(&mut self, value: &T) -> Self
    where
        C: Clone,
        G: Clone,
    {
        Self {
            inner: self.inner.split_off(value),
        }
    }
}

#[cfg(test)]
mod tests {
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

    // MARK: clear

    #[test]
    fn clear_empty() {
        let mut set = make_set(&[]);
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn clear_single() {
        let mut set = make_set(&[42]);
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.first(), None);
        assert_eq!(set.last(), None);
    }

    #[test]
    fn clear_multiple() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn clear_usable_after() {
        let mut set = make_set(&[1, 2, 3]);
        set.clear();
        set.insert(10);
        set.insert(20);
        assert_eq!(to_vec(&set), [10, 20]);
    }

    #[test]
    fn clear_twice() {
        let mut set = make_set(&[1, 2, 3]);
        set.clear();
        set.clear();
        assert!(set.is_empty());
    }

    // MARK: append

    #[test]
    fn append_empty_other() {
        let mut a = make_set(&[1, 2, 3]);
        let mut b = make_set(&[]);
        a.append(&mut b);
        assert_eq!(to_vec(&a), [1, 2, 3]);
        assert!(b.is_empty());
    }

    #[test]
    fn append_to_empty_self() {
        let mut a = make_set(&[]);
        let mut b = make_set(&[1, 2, 3]);
        a.append(&mut b);
        assert_eq!(to_vec(&a), [1, 2, 3]);
        assert!(b.is_empty());
    }

    #[test]
    fn append_disjoint_non_overlapping() {
        // Fast path: [1,3,5] and [7,9,11]
        let mut a = make_set(&[1, 3, 5]);
        let mut b = make_set(&[7, 9, 11]);
        a.append(&mut b);
        assert_eq!(to_vec(&a), [1, 3, 5, 7, 9, 11]);
        assert!(b.is_empty());
    }

    #[test]
    fn append_overlapping() {
        // Slow path: [1,2,3] and [2,3,4]
        let mut a = make_set(&[1, 2, 3]);
        let mut b = make_set(&[2, 3, 4]);
        a.append(&mut b);
        assert_eq!(to_vec(&a), [1, 2, 3, 4]);
        assert!(b.is_empty());
    }

    #[test]
    fn append_duplicates_discarded() {
        let mut a = make_set(&[1, 2, 3]);
        let mut b = make_set(&[1, 2, 3]);
        a.append(&mut b);
        assert_eq!(to_vec(&a), [1, 2, 3]);
        assert!(b.is_empty());
    }

    #[test]
    fn append_boundary_equal() {
        // Boundary: a.last() == b.first(), so the slow path is required to discard the dup.
        let mut a = make_set(&[1, 2, 3]);
        let mut b = make_set(&[3, 4, 5]);
        a.append(&mut b);
        assert_eq!(to_vec(&a), [1, 2, 3, 4, 5]);
        assert!(b.is_empty());
    }

    #[test]
    fn append_both_empty() {
        let mut a = make_set(&[]);
        let mut b = make_set(&[]);
        a.append(&mut b);
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn append_other_subset() {
        let mut a = make_set(&[1, 2, 3, 4, 5]);
        let mut b = make_set(&[2, 4]);
        a.append(&mut b);
        assert_eq!(to_vec(&a), [1, 2, 3, 4, 5]);
        assert!(b.is_empty());
    }

    #[test]
    fn append_self_subset_of_other() {
        let mut a = make_set(&[2, 4]);
        let mut b = make_set(&[1, 2, 3, 4, 5]);
        a.append(&mut b);
        assert_eq!(to_vec(&a), [1, 2, 3, 4, 5]);
        assert!(b.is_empty());
    }

    #[test]
    fn append_large_disjoint() {
        let mut a = make_set(&(1..=50).collect::<Vec<_>>());
        let mut b = make_set(&(51..=100).collect::<Vec<_>>());
        a.append(&mut b);
        assert_eq!(a.len(), 100);
        assert_eq!(a.first(), Some(&1));
        assert_eq!(a.last(), Some(&100));
        assert!(b.is_empty());
    }

    #[test]
    fn append_custom_comparator() {
        use core::cmp::Ordering;
        #[expect(
            clippy::trivially_copy_pass_by_ref,
            reason = "must match Comparator<T> signature"
        )]
        fn rev(x: &i32, y: &i32) -> Ordering {
            y.cmp(x)
        }
        let fnptr: fn(&i32, &i32) -> Ordering = rev;
        let mut a: SkipSet<i32, 16, _> = SkipSet::with_comparator(FnComparator(fnptr));
        for v in [5, 3, 1] {
            a.insert(v);
        }
        let mut b: SkipSet<i32, 16, _> = SkipSet::with_comparator(FnComparator(fnptr));
        for v in [4, 2] {
            b.insert(v);
        }
        // a is [5,3,1], b is [4,2] in descending order. Overlapping, so slow path.
        a.append(&mut b);
        let v: Vec<i32> = a.iter().copied().collect();
        assert_eq!(v, [5, 4, 3, 2, 1]);
        assert!(b.is_empty());
    }

    #[test]
    fn append_reverse_disjoint_fast_path() {
        // Reverse fast path: every element of other is strictly less than
        // every element of self.
        let mut a = make_set(&[4, 5, 6]);
        let mut b = make_set(&[1, 2, 3]);
        a.append(&mut b);
        assert_eq!(to_vec(&a), [1, 2, 3, 4, 5, 6]);
        assert!(b.is_empty());
    }

    #[test]
    fn append_reverse_equal_boundary_slow_path() {
        // Equal boundary: other.last == self.first.  Strictly less is
        // required for the reverse fast path, so this falls to the slow path
        // which discards the duplicate.
        let mut a = make_set(&[2, 3, 4]);
        let mut b = make_set(&[1, 2]);
        a.append(&mut b);
        assert_eq!(to_vec(&a), [1, 2, 3, 4]);
        assert!(b.is_empty());
    }

    #[test]
    fn append_reverse_large_disjoint() {
        let mut a = make_set(&(51..=100).collect::<Vec<_>>());
        let mut b = make_set(&(1..=50).collect::<Vec<_>>());
        a.append(&mut b);
        assert_eq!(a.len(), 100);
        assert_eq!(a.first(), Some(&1));
        assert_eq!(a.last(), Some(&100));
        assert!(b.is_empty());
    }

    // MARK: split_off

    #[test]
    fn split_off_empty() {
        let mut set: SkipSet<i32> = SkipSet::new();
        let other = set.split_off(&5);
        assert!(set.is_empty());
        assert!(other.is_empty());
    }

    #[test]
    fn split_off_all_below() {
        let mut set = make_set(&[1, 2, 3, 4]);
        let other = set.split_off(&10);
        assert_eq!(to_vec(&set), [1, 2, 3, 4]);
        assert!(other.is_empty());
    }

    #[test]
    fn split_off_all_above_or_equal() {
        let mut set = make_set(&[1, 2, 3, 4]);
        let other = set.split_off(&1);
        assert!(set.is_empty());
        assert_eq!(to_vec(&other), [1, 2, 3, 4]);
    }

    #[test]
    fn split_off_middle() {
        let mut a = make_set(&[1, 2, 3, 4, 5]);
        let b = a.split_off(&3);
        assert_eq!(to_vec(&a), [1, 2]);
        assert_eq!(to_vec(&b), [3, 4, 5]);
    }

    #[test]
    fn split_off_value_absent() {
        // Split at a value not in the set.
        let mut a = make_set(&[1, 2, 4, 5]);
        let b = a.split_off(&3);
        assert_eq!(to_vec(&a), [1, 2]);
        assert_eq!(to_vec(&b), [4, 5]);
    }

    #[test]
    fn split_off_single_element_kept() {
        let mut a = make_set(&[5]);
        let b = a.split_off(&10);
        assert_eq!(to_vec(&a), [5]);
        assert!(b.is_empty());
    }

    #[test]
    fn split_off_single_element_moved() {
        let mut a = make_set(&[5]);
        let b = a.split_off(&5);
        assert!(a.is_empty());
        assert_eq!(to_vec(&b), [5]);
    }

    #[test]
    fn split_off_tail_pointer_correct() {
        let mut a = make_set(&[1, 2, 3, 4, 5]);
        let _b = a.split_off(&4);
        assert_eq!(a.last(), Some(&3));
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn split_off_result_tail_correct() {
        let mut a = make_set(&[1, 2, 3, 4, 5]);
        let b = a.split_off(&3);
        assert_eq!(b.first(), Some(&3));
        assert_eq!(b.last(), Some(&5));
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn split_off_custom_comparator() {
        use core::cmp::Ordering;
        #[expect(
            clippy::trivially_copy_pass_by_ref,
            reason = "must match Comparator<T> signature"
        )]
        fn rev(x: &i32, y: &i32) -> Ordering {
            y.cmp(x)
        }
        let fnptr: fn(&i32, &i32) -> Ordering = rev;
        let mut a: SkipSet<i32, 16, _> = SkipSet::with_comparator(FnComparator(fnptr));
        for v in [5, 4, 3, 2, 1] {
            a.insert(v);
        }
        // Descending order: [5, 4, 3, 2, 1].
        // split_off(&3) splits at "≥ 3" in descending ordering, meaning values
        // that compare as >= 3 in descending order, i.e. values ≤ 3 ascending.
        // So a keeps [5, 4] and b gets [3, 2, 1].
        let b = a.split_off(&3);
        let a_vals: Vec<i32> = a.iter().copied().collect();
        let b_vals: Vec<i32> = b.iter().copied().collect();
        assert_eq!(a_vals, [5, 4]);
        assert_eq!(b_vals, [3, 2, 1]);
    }

    // MARK: split_off_index

    #[test]
    fn split_off_index_zero_returns_full_set() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        let right = set.split_off_index(0);
        assert!(set.is_empty());
        assert_eq!(to_vec(&right), [1, 2, 3, 4, 5]);
    }

    #[test]
    fn split_off_index_at_end_returns_empty() {
        let mut set = make_set(&[1, 2, 3]);
        let right = set.split_off_index(3);
        assert_eq!(to_vec(&set), [1, 2, 3]);
        assert!(right.is_empty());
    }

    #[test]
    fn split_off_index_middle() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        let right = set.split_off_index(2);
        assert_eq!(to_vec(&set), [1, 2]);
        assert_eq!(to_vec(&right), [3, 4, 5]);
    }

    #[test]
    fn split_off_index_out_of_bounds_returns_empty() {
        let mut set = make_set(&[1, 2, 3]);
        let right = set.split_off_index(10);
        assert_eq!(to_vec(&set), [1, 2, 3]);
        assert!(right.is_empty());
    }

    #[test]
    fn split_off_index_single_element() {
        let mut set = make_set(&[42]);
        let right = set.split_off_index(0);
        assert!(set.is_empty());
        assert_eq!(to_vec(&right), [42]);
    }

    #[test]
    fn split_off_index_custom_comparator() {
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
        for v in [5, 4, 3, 2, 1] {
            set.insert(v);
        }
        // Descending order: [5, 4, 3, 2, 1]. Split at index 2 → left=[5,4], right=[3,2,1].
        let right = set.split_off_index(2);
        assert_eq!(set.iter().copied().collect::<Vec<_>>(), [5, 4]);
        assert_eq!(right.iter().copied().collect::<Vec<_>>(), [3, 2, 1]);
    }
}
