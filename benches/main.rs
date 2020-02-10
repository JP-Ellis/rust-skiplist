#[macro_use]
extern crate criterion;

mod ordered_skiplist;
mod skiplist;
mod skipmap;

criterion_group!(
    benches,
    crate::ordered_skiplist::benchmark,
    crate::skiplist::benchmark,
    crate::skipmap::benchmark
);
criterion_main!(benches);
