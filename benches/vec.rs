use criterion::{black_box, AxisScale, BenchmarkId, Criterion, PlotConfiguration};
use rand::prelude::*;

const STEPS: [usize; 6] = [1, 10, 100, 1000, 10_000, 100_000];

pub fn push_front(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Push Front");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for i in STEPS {
        group.bench_function(BenchmarkId::from_parameter(i), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234abcd);
            let mut sl: Vec<usize> = std::iter::repeat_with(|| rng.gen()).take(i).collect();

            b.iter(|| {
                sl.insert(0, rng.gen());
            })
        });
    }
}

pub fn push_back(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Push Back");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for i in STEPS {
        group.bench_function(BenchmarkId::from_parameter(i), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234abcd);
            let mut sl: Vec<usize> = std::iter::repeat_with(|| rng.gen()).take(i).collect();

            b.iter(|| {
                sl.push(rng.gen());
            })
        });
    }
}

pub fn insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Insert");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for i in STEPS {
        group.bench_function(BenchmarkId::from_parameter(i), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234abcd);
            let mut sl: Vec<usize> = std::iter::repeat_with(|| rng.gen()).take(i).collect();

            b.iter(|| {
                let v = rng.gen();
                match sl.binary_search(&v) {
                    Ok(i) => sl.insert(i, v),
                    Err(i) => sl.insert(i, v),
                }
            })
        });
    }
}
pub fn rand_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec Random Access");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for i in STEPS {
        group.bench_function(BenchmarkId::from_parameter(i), |b| {
            let mut rng = StdRng::seed_from_u64(0x1234abcd);
            let sl: Vec<usize> = std::iter::repeat_with(|| rng.gen()).take(i).collect();
            let indices: Vec<_> = std::iter::repeat_with(|| rng.gen_range(0..sl.len()))
                .take(10)
                .collect();

            b.iter(|| {
                for &i in &indices {
                    black_box(sl[i]);
                }
            })
        });
    }
}

pub fn iter(c: &mut Criterion) {
    c.bench_function("Vec Iter", |b| {
        let mut rng = StdRng::seed_from_u64(0x1234abcd);
        let sl: Vec<usize> = std::iter::repeat_with(|| rng.gen()).take(100_000).collect();

        b.iter(|| {
            for el in &sl {
                black_box(el);
            }
        })
    });
}
