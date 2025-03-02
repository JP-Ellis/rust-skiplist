//! Benchmarks for the Standard Library's [`HashMap`].

use std::collections::HashMap;

use criterion::{black_box, AxisScale, BenchmarkId, Criterion, PlotConfiguration};
use rand::prelude::*;

/// Benchmarking sizes
const SIZES: [usize; 6] = [1, 10, 100, 1000, 10_000, 100_000];

/// Benchmarking insertion
pub fn insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("HashMap Insert");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let mut sl: HashMap<usize, usize> =
                std::iter::repeat_with(|| rng.gen()).take(size).collect();

            b.iter(|| {
                sl.insert(rng.gen(), rng.gen());
            });
        });
    }
}

/// Benchmarking random access
pub fn rand_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("HashMap Random Access");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let sl: HashMap<usize, usize> = std::iter::repeat_with(|| rng.gen())
                .enumerate()
                .take(size)
                .collect();
            let indices: Vec<_> = std::iter::repeat_with(|| rng.gen_range(0..sl.len()))
                .take(10)
                .collect();

            b.iter(|| {
                for i in &indices {
                    black_box(sl.get(i));
                }
            });
        });
    }
}

/// Benchmarking iteration
pub fn iter(c: &mut Criterion) {
    c.bench_function("HashMap Iter", |b| {
        let mut rng = StdRng::seed_from_u64(0x1234_abcd);
        let sl: HashMap<usize, usize> =
            std::iter::repeat_with(|| rng.gen()).take(100_000).collect();

        b.iter(|| {
            #[expect(clippy::iter_over_hash_type, reason = "for benchmarking")]
            for el in &sl {
                black_box(el);
            }
        });
    });
}
