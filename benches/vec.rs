//! Benchmarks for the Standard Library's [`Vec`].

use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration, black_box};
use rand::prelude::*;

/// Benchmarking sizes
const SIZES: [usize; 6] = [1, 10, 100, 1000, 10_000, 100_000];

/// Benchmarking push front
pub fn push_front(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Push Front");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let mut sl: Vec<usize> = std::iter::repeat_with(|| rng.r#gen()).take(size).collect();

            b.iter(|| {
                sl.insert(0, rng.r#gen());
            });
        });
    }
}

/// Benchmarking push back
pub fn push_back(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Push Back");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let mut sl: Vec<usize> = std::iter::repeat_with(|| rng.r#gen()).take(size).collect();

            b.iter(|| {
                sl.push(rng.r#gen());
            });
        });
    }
}

/// Benchmarking insertion
pub fn insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Insert");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let mut sl: Vec<usize> = std::iter::repeat_with(|| rng.r#gen()).take(size).collect();

            b.iter(|| {
                let v = rng.r#gen();
                match sl.binary_search(&v) {
                    Ok(i) | Err(i) => sl.insert(i, v),
                }
            });
        });
    }
}

/// Benchmarking random access
pub fn rand_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Random Access");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let sl: Vec<usize> = std::iter::repeat_with(|| rng.r#gen()).take(size).collect();
            let indices: Vec<_> = std::iter::repeat_with(|| rng.gen_range(0..sl.len()))
                .take(10)
                .collect();

            b.iter(|| {
                for &i in &indices {
                    black_box(sl.get(i));
                }
            });
        });
    }
}

/// Benchmarking iteration
pub fn iter(c: &mut Criterion) {
    c.bench_function("Vec Iter", |b| {
        let mut rng = StdRng::seed_from_u64(0x1234_abcd);
        let sl: Vec<usize> = std::iter::repeat_with(|| rng.r#gen())
            .take(100_000)
            .collect();

        b.iter(|| {
            for el in &sl {
                black_box(el);
            }
        });
    });
}
