//! Benchmarks for [`Geometric::level`] across promotion probabilities.
//!
//! `q = 0.5` uses the bit sampler; `q = 0.25` and `q = 0.75` invert the
//! truncated geometric CDF.  Each timed iteration draws one level from a
//! 16-level generator.

#![expect(
    clippy::expect_used,
    reason = "`.expect()` is appropriate in benchmarks"
)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use skiplist::level_generator::{LevelGenerator, geometric::Geometric};

/// Times one `level()` draw for each `q`.
fn bench_level(c: &mut Criterion) {
    let mut group = c.benchmark_group("level_generator/geometric");
    for &q in &[0.5_f64, 0.25, 0.75] {
        let mut generator = Geometric::new_with_seed(16, q, 42).expect("valid generator");
        group.bench_with_input(BenchmarkId::new("level", q), &q, |b, _| {
            b.iter(|| black_box(generator.level()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_level);
criterion_main!(benches);
