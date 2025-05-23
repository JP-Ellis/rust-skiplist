//! Benchmarks for the Standard Library's [`BTreeMap`].

use std::collections::BTreeMap;

use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration, black_box};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Benchmarking sizes
const SIZES: [usize; 6] = [1, 10, 100, 1000, 10_000, 100_000];

/// Benchmarking insertion
pub fn insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("BTreeMap Insert");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let mut sl: BTreeMap<u64, u64> =
                std::iter::repeat_with(|| rng.random()).take(size).collect();

            b.iter(|| {
                sl.insert(rng.random(), rng.random());
            });
        });
    }
}

/// Benchmarking random access
pub fn rand_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("BTreeMap Random Access");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let sl: BTreeMap<usize, u64> = std::iter::repeat_with(|| rng.random())
                .enumerate()
                .take(size)
                .collect();
            let indices: Vec<usize> = std::iter::repeat_with(|| rng.random_range(0..sl.len()))
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
    c.bench_function("BTreeMap Iter", |b| {
        let mut rng = StdRng::seed_from_u64(0x1234_abcd);
        let sl: BTreeMap<u64, u64> = std::iter::repeat_with(|| rng.random())
            .take(100_000)
            .collect();

        b.iter(|| {
            for el in &sl {
                black_box(el);
            }
        });
    });
}
