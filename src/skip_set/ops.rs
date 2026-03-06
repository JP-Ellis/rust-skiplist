//! Operator overloads (`&`, `|`, `^`, `-`) for [`SkipSet`](super::SkipSet).
//!
//! Each operator mirrors the `std` [`BTreeSet`] convention: `&Set OP &Set -> Set`,
//! and delegates to the corresponding set-operation iterator from
//! [`set_ops`](super::set_ops).
//!
//! | Expression | Set operation        | Method                           |
//! |------------|----------------------|----------------------------------|
//! | `&a & &b`  | Intersection         | [`intersection`]                 |
//! | `&a \| &b` | Union                | [`union`]                        |
//! | `&a ^ &b`  | Symmetric difference | [`symmetric_difference`]         |
//! | `&a - &b`  | Difference           | [`difference`]                   |
//!
//! # Bounds
//!
//! All four operators require:
//!
//! - `T: Clone`: elements are cloned into the output set.
//! - `C: Clone`: the comparator is cloned from `self` so the result uses the
//!   same ordering.
//! - `G: Default`: a fresh level-generator is constructed for the output set.
//!
//! [`BTreeSet`]: std::collections::BTreeSet
//! [`intersection`]: super::SkipSet::intersection
//! [`union`]: super::SkipSet::union
//! [`symmetric_difference`]: super::SkipSet::symmetric_difference
//! [`difference`]: super::SkipSet::difference

use core::ops::{BitAnd, BitOr, BitXor, Sub};

use crate::{comparator::Comparator, level_generator::LevelGenerator, skip_set::SkipSet};

// MARK: BitAnd (&)

impl<T: Clone, C: Comparator<T> + Clone, G: LevelGenerator + Default, const N: usize>
    BitAnd<&SkipSet<T, N, C, G>> for &SkipSet<T, N, C, G>
{
    type Output = SkipSet<T, N, C, G>;

    /// Returns the intersection of `self` and `rhs` as a new [`SkipSet`].
    ///
    /// The output contains every element that appears in both sets. The output
    /// uses `self`'s comparator.
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
    /// let c = &a & &b;
    /// assert_eq!(c.len(), 2);
    /// assert!(c.contains(&2));
    /// assert!(c.contains(&3));
    /// assert!(!c.contains(&1));
    /// assert!(!c.contains(&4));
    /// ```
    #[inline]
    fn bitand(self, rhs: &SkipSet<T, N, C, G>) -> SkipSet<T, N, C, G> {
        let mut out = SkipSet::with_comparator_and_level_generator(
            self.inner.comparator().clone(),
            G::default(),
        );
        for item in self.intersection(rhs) {
            out.insert(item.clone());
        }
        out
    }
}

// MARK: BitOr (|)

impl<T: Clone, C: Comparator<T> + Clone, G: LevelGenerator + Default, const N: usize>
    BitOr<&SkipSet<T, N, C, G>> for &SkipSet<T, N, C, G>
{
    type Output = SkipSet<T, N, C, G>;

    /// Returns the union of `self` and `rhs` as a new [`SkipSet`].
    ///
    /// The output contains every element that appears in either set. When both
    /// sets contain equal elements, the element from `self` is kept. The output
    /// uses `self`'s comparator.
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
    /// let c = &a | &b;
    /// assert_eq!(c.len(), 4);
    /// assert!(c.contains(&1));
    /// assert!(c.contains(&2));
    /// assert!(c.contains(&3));
    /// assert!(c.contains(&4));
    /// ```
    #[inline]
    fn bitor(self, rhs: &SkipSet<T, N, C, G>) -> SkipSet<T, N, C, G> {
        let mut out = SkipSet::with_comparator_and_level_generator(
            self.inner.comparator().clone(),
            G::default(),
        );
        for item in self.union(rhs) {
            out.insert(item.clone());
        }
        out
    }
}

// MARK: BitXor (^)

impl<T: Clone, C: Comparator<T> + Clone, G: LevelGenerator + Default, const N: usize>
    BitXor<&SkipSet<T, N, C, G>> for &SkipSet<T, N, C, G>
{
    type Output = SkipSet<T, N, C, G>;

    /// Returns the symmetric difference of `self` and `rhs` as a new
    /// [`SkipSet`].
    ///
    /// The output contains every element that is in `self` or `rhs` but not in
    /// both. The output uses `self`'s comparator.
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
    /// let c = &a ^ &b;
    /// assert_eq!(c.len(), 2);
    /// assert!(c.contains(&1));
    /// assert!(c.contains(&4));
    /// assert!(!c.contains(&2));
    /// assert!(!c.contains(&3));
    /// ```
    #[inline]
    fn bitxor(self, rhs: &SkipSet<T, N, C, G>) -> SkipSet<T, N, C, G> {
        let mut out = SkipSet::with_comparator_and_level_generator(
            self.inner.comparator().clone(),
            G::default(),
        );
        for item in self.symmetric_difference(rhs) {
            out.insert(item.clone());
        }
        out
    }
}

// MARK: Sub (-)

impl<T: Clone, C: Comparator<T> + Clone, G: LevelGenerator + Default, const N: usize>
    Sub<&SkipSet<T, N, C, G>> for &SkipSet<T, N, C, G>
{
    type Output = SkipSet<T, N, C, G>;

    /// Returns the difference of `self` and `rhs` as a new [`SkipSet`].
    ///
    /// The output contains every element that is in `self` but not in `rhs`.
    /// The output uses `self`'s comparator.
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
    /// let c = &a - &b;
    /// assert_eq!(c.len(), 1);
    /// assert!(c.contains(&1));
    /// assert!(!c.contains(&2));
    /// assert!(!c.contains(&3));
    /// ```
    #[inline]
    fn sub(self, rhs: &SkipSet<T, N, C, G>) -> SkipSet<T, N, C, G> {
        let mut out = SkipSet::with_comparator_and_level_generator(
            self.inner.comparator().clone(),
            G::default(),
        );
        for item in self.difference(rhs) {
            out.insert(item.clone());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use pretty_assertions::assert_eq;

    use super::SkipSet;
    use crate::comparator::FnComparator;

    // --- helpers ---

    fn make_set(values: &[i32]) -> SkipSet<i32> {
        let mut s = SkipSet::new();
        for &v in values {
            s.insert(v);
        }
        s
    }

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "must match Comparator<T> signature"
    )]
    fn rev_cmp(x: &i32, y: &i32) -> Ordering {
        y.cmp(x)
    }

    type RevSet = SkipSet<i32, 16, FnComparator<fn(&i32, &i32) -> Ordering>>;

    fn make_rev_set(values: &[i32]) -> RevSet {
        let fnptr: fn(&i32, &i32) -> Ordering = rev_cmp;
        let mut s = SkipSet::with_comparator(FnComparator(fnptr));
        for &v in values {
            s.insert(v);
        }
        s
    }

    // We only test small integer ranges so iterate [-20, 20].
    fn contents(set: &SkipSet<i32>) -> Vec<i32> {
        (-20_i32..=20).filter(|v| set.contains(v)).collect()
    }

    fn rev_contents(set: &RevSet) -> Vec<i32> {
        (-20_i32..=20).filter(|v| set.contains(v)).collect()
    }

    // MARK: BitAnd (&)

    #[test]
    fn bitand_empty_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        let c = &a & &b;
        assert!(c.is_empty());
    }

    #[test]
    fn bitand_non_empty_and_empty() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[]);
        let c = &a & &b;
        assert!(c.is_empty());
    }

    #[test]
    fn bitand_empty_and_non_empty() {
        let a = make_set(&[]);
        let b = make_set(&[1, 2, 3]);
        let c = &a & &b;
        assert!(c.is_empty());
    }

    #[test]
    fn bitand_disjoint() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[4, 5, 6]);
        let c = &a & &b;
        assert!(c.is_empty());
    }

    #[test]
    fn bitand_overlapping() {
        let a = make_set(&[1, 2, 3, 4]);
        let b = make_set(&[3, 4, 5, 6]);
        let c = &a & &b;
        assert_eq!(contents(&c), [3, 4]);
    }

    #[test]
    fn bitand_subset() {
        let a = make_set(&[2, 3]);
        let b = make_set(&[1, 2, 3, 4]);
        let c = &a & &b;
        assert_eq!(contents(&c), [2, 3]);
    }

    #[test]
    fn bitand_idempotent() {
        let a = make_set(&[1, 2, 3]);
        let c = &a & &a;
        assert_eq!(contents(&c), [1, 2, 3]);
    }

    #[test]
    fn bitand_commutative() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[2, 3, 4]);
        let ab = contents(&(&a & &b));
        let rev = contents(&(&b & &a));
        assert_eq!(ab, rev);
    }

    #[test]
    fn bitand_custom_comparator() {
        // Reverse ordering: set contents are the same values regardless.
        let a = make_rev_set(&[1, 2, 3]);
        let b = make_rev_set(&[2, 3, 4]);
        let c = &a & &b;
        assert_eq!(rev_contents(&c), [2, 3]);
    }

    // MARK: BitOr (|)

    #[test]
    fn bitor_empty_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        let c = &a | &b;
        assert!(c.is_empty());
    }

    #[test]
    fn bitor_non_empty_or_empty() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[]);
        let c = &a | &b;
        assert_eq!(contents(&c), [1, 2, 3]);
    }

    #[test]
    fn bitor_empty_or_non_empty() {
        let a = make_set(&[]);
        let b = make_set(&[1, 2, 3]);
        let c = &a | &b;
        assert_eq!(contents(&c), [1, 2, 3]);
    }

    #[test]
    fn bitor_disjoint() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[4, 5, 6]);
        let c = &a | &b;
        assert_eq!(contents(&c), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn bitor_overlapping() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[2, 3, 4]);
        let c = &a | &b;
        assert_eq!(contents(&c), [1, 2, 3, 4]);
    }

    #[test]
    fn bitor_idempotent() {
        let a = make_set(&[1, 2, 3]);
        let c = &a | &a;
        assert_eq!(contents(&c), [1, 2, 3]);
    }

    #[test]
    fn bitor_commutative() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[2, 3, 4]);
        let ab = contents(&(&a | &b));
        let rev = contents(&(&b | &a));
        assert_eq!(ab, rev);
    }

    #[test]
    fn bitor_custom_comparator() {
        let a = make_rev_set(&[1, 2, 3]);
        let b = make_rev_set(&[2, 3, 4]);
        let c = &a | &b;
        assert_eq!(rev_contents(&c), [1, 2, 3, 4]);
    }

    // MARK: BitXor (^)

    #[test]
    fn bitxor_empty_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        let c = &a ^ &b;
        assert!(c.is_empty());
    }

    #[test]
    fn bitxor_non_empty_xor_empty() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[]);
        let c = &a ^ &b;
        assert_eq!(contents(&c), [1, 2, 3]);
    }

    #[test]
    fn bitxor_same_set() {
        let a = make_set(&[1, 2, 3]);
        let c = &a ^ &a;
        assert!(c.is_empty());
    }

    #[test]
    fn bitxor_disjoint() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[4, 5, 6]);
        let c = &a ^ &b;
        assert_eq!(contents(&c), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn bitxor_overlapping() {
        let a = make_set(&[1, 2, 3, 4]);
        let b = make_set(&[3, 4, 5, 6]);
        let c = &a ^ &b;
        assert_eq!(contents(&c), [1, 2, 5, 6]);
    }

    #[test]
    fn bitxor_commutative() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[2, 3, 4]);
        let ab = contents(&(&a ^ &b));
        let rev = contents(&(&b ^ &a));
        assert_eq!(ab, rev);
    }

    #[test]
    fn bitxor_custom_comparator() {
        let a = make_rev_set(&[1, 2, 3]);
        let b = make_rev_set(&[2, 3, 4]);
        let c = &a ^ &b;
        assert_eq!(rev_contents(&c), [1, 4]);
    }

    // MARK: Sub (-)

    #[test]
    fn sub_empty_empty() {
        let a = make_set(&[]);
        let b = make_set(&[]);
        let c = &a - &b;
        assert!(c.is_empty());
    }

    #[test]
    fn sub_non_empty_minus_empty() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[]);
        let c = &a - &b;
        assert_eq!(contents(&c), [1, 2, 3]);
    }

    #[test]
    fn sub_empty_minus_non_empty() {
        let a = make_set(&[]);
        let b = make_set(&[1, 2, 3]);
        let c = &a - &b;
        assert!(c.is_empty());
    }

    #[test]
    fn sub_disjoint() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[4, 5, 6]);
        let c = &a - &b;
        assert_eq!(contents(&c), [1, 2, 3]);
    }

    #[test]
    fn sub_overlapping() {
        let a = make_set(&[1, 2, 3, 4]);
        let b = make_set(&[3, 4, 5, 6]);
        let c = &a - &b;
        assert_eq!(contents(&c), [1, 2]);
    }

    #[test]
    fn sub_self() {
        let a = make_set(&[1, 2, 3]);
        let c = &a - &a;
        assert!(c.is_empty());
    }

    #[test]
    fn sub_superset() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 3, 4, 5]);
        let c = &a - &b;
        assert!(c.is_empty());
    }

    #[test]
    fn sub_custom_comparator() {
        let a = make_rev_set(&[1, 2, 3, 4]);
        let b = make_rev_set(&[3, 4]);
        let c = &a - &b;
        assert_eq!(rev_contents(&c), [1, 2]);
    }
}
