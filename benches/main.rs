#[macro_use]
extern crate criterion;
extern crate rand;
extern crate skiplist;

mod ordered_skiplist_bench;
mod skiplist_bench;
mod skipmap_bench;
use ordered_skiplist_bench::ordered_skiplist_benchmark;
use skiplist_bench::skiplist_benchmark;
use skipmap_bench::skipmap_benchmark;

criterion_group!(benches, ordered_skiplist_benchmark, skiplist_benchmark, skipmap_benchmark);
criterion_main!(benches);