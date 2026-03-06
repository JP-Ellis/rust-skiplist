//! Predicate-based filtering for [`SkipSet`](super::SkipSet).

use crate::{comparator::Comparator, level_generator::LevelGenerator, skip_set::SkipSet};

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> SkipSet<T, N, C, G> {
    /// Retains only the elements specified by the predicate, removing all
    /// others.
    ///
    /// Elements are visited in ascending (sorted) order. For each element
    /// `e`, `f(&e)` is called; if it returns `false` the element is removed.
    ///
    /// Note: `retain_mut` is intentionally absent, as mutating an element in
    /// place could violate the sort-order or uniqueness invariants.
    ///
    /// This operation runs in `$O(n)$` time.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_set::SkipSet;
    ///
    /// let mut set = SkipSet::<i32>::new();
    /// for v in 1..=5 { set.insert(v); }
    ///
    /// set.retain(|&x| x % 2 == 0);
    /// let collected: Vec<i32> = set.iter().copied().collect();
    /// assert_eq!(collected, [2, 4]);
    /// ```
    #[inline]
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.inner.retain(f);
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

    // MARK: retain

    #[test]
    fn retain_empty_keep_all() {
        let mut set = make_set(&[]);
        set.retain(|_| true);
        assert!(set.is_empty());
    }

    #[test]
    fn retain_empty_drop_all() {
        let mut set = make_set(&[]);
        set.retain(|_| false);
        assert!(set.is_empty());
    }

    #[test]
    fn retain_all_kept() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        set.retain(|_| true);
        assert_eq!(to_vec(&set), [1, 2, 3, 4, 5]);
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn retain_all_dropped() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        set.retain(|_| false);
        assert!(set.is_empty());
        assert_eq!(set.first(), None);
        assert_eq!(set.last(), None);
    }

    #[test]
    #[expect(
        clippy::integer_division_remainder_used,
        reason = "test intent is clearest with %"
    )]
    fn retain_even() {
        let mut set = make_set(&[1, 2, 3, 4, 5, 6]);
        set.retain(|&x| x % 2 == 0);
        assert_eq!(to_vec(&set), [2, 4, 6]);
        assert_eq!(set.len(), 3);
    }

    #[test]
    #[expect(
        clippy::integer_division_remainder_used,
        reason = "test intent is clearest with %"
    )]
    fn retain_odd() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        set.retain(|&x| x % 2 != 0);
        assert_eq!(to_vec(&set), [1, 3, 5]);
    }

    #[test]
    fn retain_single_kept() {
        let mut set = make_set(&[42]);
        set.retain(|_| true);
        assert_eq!(set.len(), 1);
        assert_eq!(set.first(), Some(&42));
        assert_eq!(set.last(), Some(&42));
    }

    #[test]
    fn retain_single_dropped() {
        let mut set = make_set(&[42]);
        set.retain(|_| false);
        assert!(set.is_empty());
        assert_eq!(set.first(), None);
        assert_eq!(set.last(), None);
    }

    #[test]
    fn retain_prefix() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        set.retain(|&x| x <= 3);
        assert_eq!(to_vec(&set), [1, 2, 3]);
        assert_eq!(set.last(), Some(&3));
    }

    #[test]
    fn retain_suffix() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        set.retain(|&x| x >= 3);
        assert_eq!(to_vec(&set), [3, 4, 5]);
        assert_eq!(set.first(), Some(&3));
    }

    #[test]
    fn retain_middle() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        set.retain(|&x| x == 3);
        assert_eq!(to_vec(&set), [3]);
        assert_eq!(set.first(), Some(&3));
        assert_eq!(set.last(), Some(&3));
    }

    #[test]
    fn retain_tail_pointer_updated() {
        let mut set = make_set(&[1, 2, 3, 4, 5]);
        set.retain(|&x| x <= 3);
        assert_eq!(set.last(), Some(&3));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn retain_chained() {
        let mut set = make_set(&[1, 2, 3, 4, 5, 6, 7, 8]);
        #[expect(
            clippy::integer_division_remainder_used,
            reason = "test intent is clearest with %"
        )]
        set.retain(|&x| x % 2 == 0); // 2, 4, 6, 8
        set.retain(|&x| x > 4);
        assert_eq!(to_vec(&set), [6, 8]);
        assert_eq!(set.first(), Some(&6));
        assert_eq!(set.last(), Some(&8));
    }

    #[test]
    fn retain_large() {
        let mut set = make_set(&(1..=100).collect::<Vec<_>>());
        set.retain(|&x| x > 50);
        assert_eq!(set.len(), 50);
        assert_eq!(set.first(), Some(&51));
        assert_eq!(set.last(), Some(&100));
        let v = to_vec(&set);
        let expected: Vec<i32> = (51..=100).collect();
        assert_eq!(v, expected);
    }

    #[test]
    fn retain_custom_comparator() {
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
        // Descending order: [5, 4, 3, 2, 1]. Keep values ≥ 3.
        set.retain(|&x| x >= 3);
        let v: Vec<i32> = set.iter().copied().collect();
        assert_eq!(v, [5, 4, 3]);
    }
}
