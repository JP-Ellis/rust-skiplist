//! Tutorial: Getting Started with skiplist.
//!
//! This example walks you through building a word-frequency counter with
//! `SkipMap`, demonstrating the core operations of the skip list collections.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example getting_started
//! ```

#![expect(
    clippy::print_stdout,
    clippy::use_debug,
    reason = "This example is for demonstration, not a test."
)]

use pretty_assertions::assert_eq;
use skiplist::{FnComparator, SkipMap};

fn main() {
    // Step 1: Create a SkipMap
    //
    // `SkipMap<K, V>` is a sorted key-value map, think `BTreeMap` but with
    // pluggable ordering and O(log n) rank queries.
    //
    // The default constructor uses the natural `Ord` ordering on keys.
    let mut freq: SkipMap<String, usize> = SkipMap::new();

    // Step 2: Count word frequencies
    //
    // `insert` returns `Some(old_value)` when a key already exists, so we can
    // use it together with `get` to update counts.
    let words = ["the", "cat", "sat", "on", "the", "mat", "the", "cat"];
    for word in words {
        let count = freq.get(word).copied().unwrap_or(0);
        freq.insert(word.to_owned(), count.saturating_add(1));
    }

    // Step 3: Inspect individual entries
    //
    // `get` returns `Option<&V>`.  Because `SkipMap` implements `Borrow<Q>`,
    // we can pass a `&str` directly when the key type is `String`.
    println!("Frequency of 'the': {:?}", freq.get("the"));
    println!("Frequency of 'cat': {:?}", freq.get("cat"));
    println!("Frequency of 'dog': {:?}", freq.get("dog"));

    assert_eq!(freq.get("the"), Some(&3));
    assert_eq!(freq.get("cat"), Some(&2));
    assert_eq!(freq.get("dog"), None);

    // Step 4: Iterate in sorted key order
    //
    // This is the payoff for choosing a skip map over a plain hash map: you
    // get O(log n) operations *and* the entries arrive in alphabetical order.
    println!("\nAll word counts (sorted alphabetically):");
    for (word, count) in &freq {
        println!("  {word}: {count}");
    }

    // Step 5: Use a custom ordering
    //
    // `FnComparator` wraps any comparison function.  Here we sort keys by
    // frequency (most common first) instead of alphabetically.  Because
    // `SkipMap` requires a total order on keys we use the word itself as a
    // tiebreaker.
    //
    // Note: this new map uses `(&str, usize)` tuples as keys so that both the
    // count and the word are part of the ordering key.
    let mut by_freq: SkipMap<(&str, usize), (), 16, _> =
        SkipMap::with_comparator(FnComparator(|a: &(&str, usize), b: &(&str, usize)| {
            // Sort by count descending, then alphabetically ascending.
            b.1.cmp(&a.1).then(a.0.cmp(b.0))
        }));

    for (word, count) in &freq {
        by_freq.insert((word.as_str(), *count), ());
    }

    println!("\nAll word counts (most frequent first):");
    for ((word, count), ()) in &by_freq {
        println!("  {word}: {count}");
    }

    // The most frequent word is "the" with 3 occurrences.
    let first_key = by_freq.first_key_value().map(|(k, ())| k);
    assert_eq!(first_key, Some(&("the", 3)));

    println!("\nDone!");
}
