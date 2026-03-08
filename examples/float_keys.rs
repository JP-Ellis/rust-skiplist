//! How-to: Floating-point keys.
//!
//! Goal: I want to use `f64` as a key in a `SkipMap`.
//!
//! Problem: `f64` does not implement `Ord` because `NaN` is incomparable with
//! everything (including itself).  The ordered collections in this crate
//! require a total order by default.
//!
//! Solution: Use `FnComparator(f64::total_cmp)` which provides the IEEE
//! 754-2019 total order: finite values sort numerically, -0.0 sorts before
//! +0.0, and NaN bit patterns sort after +∞.  No panics, no special cases.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example float_keys
//! ```

#![expect(
    clippy::print_stdout,
    clippy::approx_constant,
    reason = "This example is for demonstration, not a test."
)]

use pretty_assertions::assert_eq;
use skiplist::{FnComparator, SkipMap};

fn main() {
    // `f64` does not implement `Ord`, so `SkipMap::<f64, _>::new()` would
    // not compile.  Instead, supply a custom comparator:
    let mut scores: SkipMap<f64, &str, 16, _> =
        SkipMap::with_comparator(FnComparator(f64::total_cmp));

    // Insert measurements with associated labels.
    // (These are truncated approximations for illustration, not using std consts.)
    #[expect(
        clippy::approx_constant,
        reason = "intentional approximations for illustration"
    )]
    let constants: &[(f64, &str)] = &[
        (3.141_592, "pi approximation"),
        (2.718_281, "e approximation"),
        (1.414_213, "sqrt(2) approximation"),
        (0.577_215, "Euler-Mascheroni constant"),
        (1.618_033, "golden ratio"),
    ];
    for &(value, label) in constants {
        scores.insert(value, label);
    }

    // Iteration is in ascending numeric order.
    println!("Constants in ascending order:");
    for (value, label) in &scores {
        println!("  {value:.3} - {label}");
    }

    // Lookup works through the comparator, not `==`.
    assert_eq!(scores.get(&3.141_592), Some(&"pi approximation"));

    // Demonstrate the NaN edge case.
    //
    // With `f64::total_cmp`, NaN sorts after +∞.  This is safe and
    // deterministic, unlike `PartialOrdComparator`, which would panic.
    let mut with_nan: SkipMap<f64, &str, 16, _> =
        SkipMap::with_comparator(FnComparator(f64::total_cmp));

    with_nan.insert(1.0, "one");
    with_nan.insert(f64::INFINITY, "infinity");
    with_nan.insert(f64::NAN, "nan");
    with_nan.insert(-1.0, "minus one");

    println!("\nKeys including NaN and infinity (total order):");
    for (key, label) in &with_nan {
        if key.is_nan() {
            println!("  NaN - {label}");
        } else {
            println!("  {key} - {label}");
        }
    }

    // NaN sorts last.
    let last = with_nan.last_key_value().map(|(_, v)| *v);
    assert_eq!(last, Some("nan"));

    // -1.0 sorts first.
    let first = with_nan.first_key_value().map(|(_, v)| *v);
    assert_eq!(first, Some("minus one"));

    println!("\nDone!");
}
