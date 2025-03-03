//! Benchmarks for the Standard Library's [`LinkedList`].

use std::collections::LinkedList;

use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration, black_box};
use rand::prelude::*;

/// Benchmarking sizes
const SIZES: [usize; 6] = [1, 10, 100, 1000, 10_000, 100_000];

/// Benchmarking push front
pub fn push_front(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinkedList Push Front");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let mut sl: LinkedList<usize> =
                std::iter::repeat_with(|| rng.r#gen()).take(size).collect();

            b.iter(|| {
                sl.push_front(rng.r#gen());
            });
        });
    }
}

/// Benchmarking push back
pub fn push_back(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinkedList Push Back");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let mut sl: LinkedList<usize> =
                std::iter::repeat_with(|| rng.r#gen()).take(size).collect();

            b.iter(|| {
                sl.push_back(rng.r#gen());
            });
        });
    }
}

/// Benchmarking random access
pub fn rand_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinkedList Random Access");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let sl: LinkedList<usize> = std::iter::repeat_with(|| rng.r#gen()).take(size).collect();
            let indices: Vec<_> = std::iter::repeat_with(|| rng.gen_range(0..sl.len()))
                .take(10)
                .collect();

            b.iter(|| {
                for &i in &indices {
                    if let Some(el) = sl.iter().nth(i) {
                        black_box(el);
                    }
                }
            });
        });
    }
}

/// Benchmarking iteration
pub fn iter(c: &mut Criterion) {
    c.bench_function("LinkedList Iter", |b| {
        let mut rng = StdRng::seed_from_u64(0x1234_abcd);
        let sl: LinkedList<usize> = std::iter::repeat_with(|| rng.r#gen())
            .take(100_000)
            .collect();

        b.iter(|| {
            for el in &sl {
                black_box(el);
            }
        });
    });
}
