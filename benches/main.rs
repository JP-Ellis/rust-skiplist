//! # Benchmarks
//!
//! Setup and configure benchmarks to compare the performance of our skiplist
//! implementation against the standard library's data structures.

#![expect(
    missing_docs,
    reason = "benchmarking and criterion creates boilerplate"
)]

use criterion::{Criterion, criterion_group, criterion_main};

mod btreemap;
mod hashmap;
mod linkedlist;
mod ordered_skiplist;
mod skiplist;
mod skipmap;
mod vec;
mod vecdeque;

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
    crate::btreemap::insert,
    crate::btreemap::iter,
    crate::btreemap::rand_access,
    crate::hashmap::insert,
    crate::hashmap::iter,
    crate::hashmap::rand_access,
    crate::linkedlist::iter,
    crate::linkedlist::push_back,
    crate::linkedlist::push_front,
    crate::linkedlist::rand_access,
    crate::ordered_skiplist::insert,
    crate::ordered_skiplist::iter,
    crate::ordered_skiplist::rand_access,
    crate::skiplist::iter,
    crate::skiplist::push_back,
    crate::skiplist::push_front,
    crate::skiplist::rand_access,
    crate::skipmap::insert,
    crate::skipmap::iter,
    crate::skipmap::rand_access,
    crate::vec::insert,
    crate::vec::iter,
    crate::vec::push_back,
    crate::vec::push_front,
    crate::vec::rand_access,
    crate::vecdeque::iter,
    crate::vecdeque::push_back,
    crate::vecdeque::push_front,
    crate::vecdeque::rand_access,
);

criterion_main!(benches);
