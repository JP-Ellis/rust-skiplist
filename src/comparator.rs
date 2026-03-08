//! Comparators for ordered skip list variants.
//!
//! Ordered skip list types ([`OrderedSkipList`], [`SkipSet`], and
//! [`SkipMap`]) need a total ordering on their elements. Rather than
//! imposing a blanket `T: Ord` bound on the struct, they accept a
//! *comparator* type parameter `C: Comparator<T>`. This allows:
//!
//! - Using the natural `Ord` ordering via the zero-sized [`OrdComparator`].
//! - Supplying a custom comparison function via [`FnComparator`].
//! - (With the `partial-ord` feature) using a `PartialOrd` ordering that
//!   panics when elements are incomparable via [`PartialOrdComparator`].
//!
//! # Ordering contract
//!
//! Any `Comparator<T>` implementation **must** define a strict total order:
//!
//! - **Totality**: `compare(a, b)` always returns one of `Less`, `Equal`,
//!   `Greater`.
//! - **Antisymmetry**: `compare(a, b) == Greater` iff `compare(b, a) ==
//!   Less`; `compare(a, b) == Equal` implies `compare(b, a) == Equal`.
//! - **Transitivity**: `compare(a, b) != Greater && compare(b, c) !=
//!   Greater` implies `compare(a, c) != Greater`.
//!
//! Violating these properties can cause incorrect behaviour up to and
//! including memory unsafety in the ordered skip list variants.
//!
//! [`OrderedSkipList`]: crate::ordered_skip_list::OrderedSkipList
//! [`SkipSet`]: crate::skip_set::SkipSet
//! [`SkipMap`]: crate::skip_map::SkipMap

#![expect(
    clippy::module_name_repetitions,
    reason = "type names intentionally include `Comparator` to avoid clashing \
        with the core types."
)]

use core::{cmp::Ordering, fmt};

/// A comparator defines a total order over values of type `T`.
///
/// Implementations must satisfy the ordering contract described in the
/// [module documentation][crate::comparator].
///
/// # Examples
///
/// Implementing a custom comparator that orders strings by length:
///
/// ```rust
/// use skiplist::comparator::{Comparator};
/// use core::cmp::Ordering;
///
/// struct ByLength;
///
/// impl Comparator<str> for ByLength {
///     fn compare(&self, a: &str, b: &str) -> Ordering {
///         a.len().cmp(&b.len())
///     }
/// }
///
/// assert_eq!(ByLength.compare("hi", "hello"), Ordering::Less);
/// assert_eq!(ByLength.compare("abc", "xyz"), Ordering::Equal);
/// assert_eq!(ByLength.compare("world", "hi"), Ordering::Greater);
/// ```
pub trait Comparator<T: ?Sized> {
    /// Compare `a` to `b`, returning the ordering of `a` relative to `b`.
    #[must_use]
    fn compare(&self, a: &T, b: &T) -> Ordering;
}

// MARK: OrdComparator

/// A comparator that delegates to the type's [`Ord`] implementation.
///
/// This is the default comparator used by [`OrderedSkipList::new`],
/// [`SkipSet::new`], and [`SkipMap::new`]. It requires `T: Ord`.
///
/// # Examples
///
/// ```rust
/// use skiplist::comparator::{Comparator, OrdComparator};
/// use core::cmp::Ordering;
///
/// let cmp = OrdComparator::default();
/// assert_eq!(cmp.compare(&1, &2), Ordering::Less);
/// assert_eq!(cmp.compare(&3, &3), Ordering::Equal);
/// assert_eq!(cmp.compare(&5, &4), Ordering::Greater);
/// ```
///
/// [`OrderedSkipList::new`]: crate::ordered_skip_list::OrderedSkipList::new
/// [`SkipSet::new`]: crate::skip_set::SkipSet::new
/// [`SkipMap::new`]: crate::skip_map::SkipMap::new
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct OrdComparator;

impl<T: Ord> Comparator<T> for OrdComparator {
    #[inline]
    fn compare(&self, a: &T, b: &T) -> Ordering {
        a.cmp(b)
    }
}

// MARK: FnComparator

/// A comparator backed by a caller-supplied comparison function.
///
/// Use this when you want a custom ordering without implementing `Ord` on the
/// element type. The wrapped function `F` must satisfy the ordering contract
/// described in the [module documentation][crate::comparator].
///
/// # Examples
///
/// ```rust
/// use skiplist::comparator::{Comparator, FnComparator};
/// use core::cmp::Ordering;
///
/// // Reverse ordering (largest-first).
/// let cmp = FnComparator(|a: &i32, b: &i32| b.cmp(a));
/// assert_eq!(cmp.compare(&3, &1), Ordering::Less);
/// assert_eq!(cmp.compare(&1, &3), Ordering::Greater);
/// ```
#[derive(Clone, Copy)]
#[expect(
    clippy::exhaustive_structs,
    reason = "`FnComparator<F>(pub F)` must remain a tuple struct so callers can \
              write `FnComparator(|a, b| …)`; `#[non_exhaustive]` would prevent \
              construction outside this crate"
)]
pub struct FnComparator<F>(pub F);

impl<F> fmt::Debug for FnComparator<F> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnComparator").finish_non_exhaustive()
    }
}

impl<T, F: Fn(&T, &T) -> Ordering> Comparator<T> for FnComparator<F> {
    #[inline]
    fn compare(&self, a: &T, b: &T) -> Ordering {
        (self.0)(a, b)
    }
}

// MARK: PartialOrdComparator

/// A comparator that delegates to the type's [`PartialOrd`] implementation,
/// panicking when elements are incomparable (i.e. `partial_cmp` returns
/// `None`).
///
/// Enabled by the `partial-ord` crate feature. Useful for floating-point
/// types when `NaN` values will never be present.
///
/// # Panics
///
/// Panics in [`compare`][Comparator::compare] when `a.partial_cmp(b)`
/// returns `None`.
///
/// # Examples
///
/// ```rust
/// use skiplist::comparator::{Comparator, PartialOrdComparator};
/// use core::cmp::Ordering;
///
/// let cmp = PartialOrdComparator::default();
/// assert_eq!(cmp.compare(&1.0_f64, &2.0), Ordering::Less);
/// assert_eq!(cmp.compare(&1.5_f64, &1.5), Ordering::Equal);
/// assert_eq!(cmp.compare(&3.0_f64, &2.0), Ordering::Greater);
/// ```
#[cfg(feature = "partial-ord")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PartialOrdComparator;

#[cfg(feature = "partial-ord")]
impl<T: PartialOrd> Comparator<T> for PartialOrdComparator {
    #[inline]
    #[expect(
        clippy::expect_used,
        reason = "panicking on incomparable values is the documented and intentional \
                  behaviour of `PartialOrdComparator`"
    )]
    fn compare(&self, a: &T, b: &T) -> Ordering {
        a.partial_cmp(b)
            .expect("comparison returned None: values are not comparable")
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use pretty_assertions::assert_eq;

    use super::{Comparator, FnComparator, OrdComparator};

    // OrdComparator

    #[test]
    fn ord_comparator_less() {
        assert_eq!(OrdComparator.compare(&1_i32, &2), Ordering::Less);
    }

    #[test]
    fn ord_comparator_equal() {
        assert_eq!(OrdComparator.compare(&42_i32, &42), Ordering::Equal);
    }

    #[test]
    fn ord_comparator_greater() {
        assert_eq!(OrdComparator.compare(&3_i32, &2), Ordering::Greater);
    }

    #[test]
    fn ord_comparator_strings() {
        assert_eq!(OrdComparator.compare(&"apple", &"banana"), Ordering::Less);
    }

    // FnComparator

    #[test]
    fn fn_comparator_natural_order() {
        let cmp = FnComparator(|a: &i32, b: &i32| a.cmp(b));
        assert_eq!(cmp.compare(&1, &2), Ordering::Less);
        assert_eq!(cmp.compare(&2, &2), Ordering::Equal);
        assert_eq!(cmp.compare(&3, &2), Ordering::Greater);
    }

    #[test]
    fn fn_comparator_reverse_order() {
        let cmp = FnComparator(|a: &i32, b: &i32| b.cmp(a));
        assert_eq!(cmp.compare(&3, &1), Ordering::Less);
        assert_eq!(cmp.compare(&1, &1), Ordering::Equal);
        assert_eq!(cmp.compare(&1, &3), Ordering::Greater);
    }

    #[test]
    fn fn_comparator_by_string_length() {
        let cmp = FnComparator(|a: &&str, b: &&str| a.len().cmp(&b.len()));
        assert_eq!(cmp.compare(&"hi", &"hello"), Ordering::Less);
        assert_eq!(cmp.compare(&"abc", &"xyz"), Ordering::Equal);
    }

    // PartialOrdComparator

    #[cfg(feature = "partial-ord")]
    mod partial_ord {
        use core::cmp::Ordering;

        use pretty_assertions::assert_eq;

        use super::super::{Comparator, PartialOrdComparator};

        #[test]
        fn partial_ord_less() {
            assert_eq!(PartialOrdComparator.compare(&1.0_f64, &2.0), Ordering::Less);
        }

        #[test]
        fn partial_ord_equal() {
            assert_eq!(
                PartialOrdComparator.compare(&1.5_f64, &1.5),
                Ordering::Equal
            );
        }

        #[test]
        fn partial_ord_greater() {
            assert_eq!(
                PartialOrdComparator.compare(&3.0_f64, &2.0),
                Ordering::Greater
            );
        }

        #[test]
        #[should_panic(expected = "comparison returned None")]
        fn partial_ord_nan_panics() {
            _ = PartialOrdComparator.compare(&f64::NAN, &1.0);
        }
    }
}
