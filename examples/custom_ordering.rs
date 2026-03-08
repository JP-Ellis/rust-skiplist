//! How-to: Custom ordering without `Ord`.
//!
//! Goal: I want to sort strings by length, then alphabetically, using
//! `OrderedSkipList`, without implementing a new comparator type.
//!
//! `FnComparator` wraps any `fn(&T, &T) -> Ordering` closure and satisfies the
//! `Comparator<T>` trait, so no boilerplate is needed.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example custom_ordering
//! ```

#![expect(
    clippy::print_stdout,
    reason = "This example is for demonstration, not a test."
)]

use std::cmp::Ordering;

use pretty_assertions::assert_eq;
use skiplist::{FnComparator, OrderedSkipList};

fn main() {
    // `Comparator<T>` receives `&T` references, so for an
    // `OrderedSkipList<&str>` the comparator receives `&&str`.
    // We dereference once in the closure to get a plain `&str` for the
    // length and lexicographic comparison.
    let comparator =
        FnComparator(|a: &&str, b: &&str| -> Ordering { a.len().cmp(&b.len()).then(a.cmp(b)) });

    // Construct the list with our custom comparator.
    //
    // `with_comparator` accepts any value that implements `Comparator<T>`.
    // `FnComparator` is the zero-cost wrapper for closures and function
    // pointers.
    let mut list: OrderedSkipList<&str, 16, _> = OrderedSkipList::with_comparator(comparator);

    // Insert strings in arbitrary order.  The list always maintains the
    // custom order, shorter strings come first, ties broken alphabetically.
    list.insert("banana");
    list.insert("fig");
    list.insert("apple");
    list.insert("kiwi");
    list.insert("date");
    list.insert("cherry");
    list.insert("plum");

    // Iterate to observe the custom ordering.
    println!("Fruits sorted by length, then alphabetically:");
    for fruit in &list {
        println!("  {} ({} chars)", fruit, fruit.len());
    }

    // Verify the ordering: "fig" (3) before "date" (4) before "kiwi" (4)
    // before "plum" (4), then "apple" (5), "banana" (6), "cherry" (6).
    let sorted: Vec<&str> = list.iter().copied().collect();
    assert_eq!(
        sorted,
        ["fig", "date", "kiwi", "plum", "apple", "banana", "cherry"]
    );

    // `get_first` uses the comparator, so it works correctly even though
    // equality here means "same length and same text", not pointer identity.
    assert_eq!(list.get_first(&"kiwi"), Some(&"kiwi"));

    // Closures work to use `FnComparator` with a closure when the
    // comparator needs to capture state.
    let min_length: usize = 5;
    let mut long_words: OrderedSkipList<&str, 16, _> =
        OrderedSkipList::with_comparator(FnComparator(move |a: &&str, b: &&str| {
            a.len().cmp(&b.len()).then(a.cmp(b))
        }));

    for fruit in &list {
        if fruit.len() >= min_length {
            long_words.insert(fruit);
        }
    }

    println!("\nFruits with 5+ characters (custom order):");
    for fruit in &long_words {
        println!("  {fruit}");
    }

    let long: Vec<&str> = long_words.iter().copied().collect();
    assert_eq!(long, ["apple", "banana", "cherry"]);

    println!("\nDone!");
}
