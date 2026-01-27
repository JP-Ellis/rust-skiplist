//! Benchmarks for the Standard Library's [`Vec`].

use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration, black_box};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Benchmarking sizes.
const SIZES: [usize; 6] = [1, 10, 100, 1000, 10_000, 100_000];

/// Benchmarking push front.
#[inline]
pub fn push_front(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Push Front");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let mut sl: Vec<u64> = std::iter::repeat_with(|| rng.random()).take(size).collect();

            b.iter(|| {
                sl.insert(0, rng.random());
            });
        });
    }
}

/// Benchmarking push back.
#[inline]
pub fn push_back(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Push Back");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let mut sl: Vec<u64> = std::iter::repeat_with(|| rng.random()).take(size).collect();

            b.iter(|| {
                sl.push(rng.random());
            });
        });
    }
}

/// Benchmarking insertion.
#[inline]
pub fn insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Insert");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let mut sl: Vec<u64> = std::iter::repeat_with(|| rng.random()).take(size).collect();

            b.iter(|| {
                let v = rng.random();
                match sl.binary_search(&v) {
                    Ok(i) | Err(i) => sl.insert(i, v),
                }
            });
        });
    }
}

/// Benchmarking random access.
#[inline]
pub fn rand_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Random Access");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for size in SIZES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234_abcd);
            let sl: Vec<u64> = std::iter::repeat_with(|| rng.random()).take(size).collect();
            let indices: Vec<_> = std::iter::repeat_with(|| rng.random_range(0..sl.len()))
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

/// Benchmarking iteration.
#[inline]
pub fn iter(c: &mut Criterion) {
    c.bench_function("Vec Iter", |b| {
        let mut rng = StdRng::seed_from_u64(0x1234_abcd);
        let sl: Vec<u64> = std::iter::repeat_with(|| rng.random())
            .take(100_000)
            .collect();

        b.iter(|| {
            for el in &sl {
                black_box(el);
            }
        });
    });
}
