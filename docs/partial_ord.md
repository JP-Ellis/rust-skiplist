# The `partial-ord` Feature

This page explains when `PartialOrdComparator` is appropriate, what can go wrong with floating-point keys, and the recommended alternatives.

For the API reference, see [`PartialOrdComparator`].

## Background

All ordered collections in this crate require a well-behaved _total_ order: every pair of elements must be comparable and the comparison must be consistent.  The default [`OrdComparator`] enforces this at compile time via the `T: Ord` bound.

Some types, most notably `f32` and `f64`, implement only [`PartialOrd`] because not all values are comparable.  In IEEE 754 arithmetic, `NaN` (Not-a-Number) is incomparable with everything, including itself: `f64::NAN < 1.0` is false, `f64::NAN > 1.0` is false, and `f64::NAN == f64::NAN` is false.

Enabling the `partial-ord` feature unlocks [`PartialOrdComparator`], which delegates to [`PartialOrd`] and **panics immediately** when a comparison returns `None` (i.e. when an incomparable value is encountered).

```toml
# Cargo.toml
[dependencies]
skiplist = { version = "...", features = ["partial-ord"] }
```

## When `PartialOrdComparator` Is Appropriate

Use [`PartialOrdComparator`] when:

-   Your element type implements `PartialOrd` but not `Ord`.
-   You can **guarantee at the call site** that no incomparable values (e.g.  `NaN`) will ever be inserted or looked up.

A common example is a domain that works with measurements that are always finite and non-NaN:

```rust
# #[cfg(feature = "partial-ord")] {
use skiplist::{OrderedSkipList, PartialOrdComparator};

let mut scores: OrderedSkipList<f64, 16, PartialOrdComparator> =
    OrderedSkipList::with_comparator(PartialOrdComparator::default());

for &v in &[4.5, 1.2, 8.3, 3.7, 6.1] {
    scores.insert(v);
}

// The list is kept sorted, so the middle element is the median.
let median = scores.get_by_index(scores.len() / 2);
assert_eq!(median, Some(&4.5));
# }
```

## Caveats

### NaN panics at runtime

Inserting or looking up a `NaN` value will panic immediately.  There is no compile-time protection.  If there is any chance that a `NaN` could reach the collection, prefer `FnComparator(f64::total_cmp)` instead (see below).

### No NaN-equality guarantee

Even if a `NaN` somehow entered the collection without panicking, the skip list's ordering invariants would be broken, leading to incorrect search results and potential memory unsafety.  The panic is a safety net, not a correctness guarantee.

## The Recommended Alternative: `FnComparator(f64::total_cmp)`

For floating-point keys, [`FnComparator`] with [`f64::total_cmp`] gives you the IEEE 754-2019 total order with no panics:

-   Finite values sort numerically.
-   `-0.0` sorts before `+0.0`.
-   All `NaN` bit patterns sort after every finite value and after `+∞`.

This is a true total order: every pair of values is comparable and the ordering is fully consistent.

```rust
use skiplist::{OrderedSkipList, FnComparator};

let mut list: OrderedSkipList<f64, 16, _> =
    OrderedSkipList::with_comparator(FnComparator(f64::total_cmp));

list.insert(3.0_f64);
list.insert(f64::NAN);
list.insert(1.0_f64);

// Finite values sort numerically; NaN sorts last.
assert_eq!(list.first(), Some(&1.0));
assert!(list.last().is_some_and(|v| v.is_nan()));
```

Prefer `FnComparator(f64::total_cmp)` over `PartialOrdComparator` in any context where NaN is possible or where the `partial-ord` feature is undesirable as a dependency.

[`PartialOrdComparator`]: crate::PartialOrdComparator
[`OrdComparator`]: crate::OrdComparator
[`FnComparator`]: crate::FnComparator
