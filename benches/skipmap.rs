use criterion::{black_box, Bencher, Criterion};
use rand::prelude::*;
use skiplist::SkipMap;

fn bench_insert(b: &mut Bencher, base: usize, inserts: usize) {
    let mut sm: SkipMap<u32, u32> = SkipMap::with_capacity(base + inserts);
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

    for _ in 0..base {
        sm.insert(rng.gen(), rng.gen());
    }

    b.iter(|| {
        for _ in 0..inserts {
            sm.insert(rng.gen(), rng.gen());
        }
    });
}

fn bench_iter(b: &mut Bencher, size: usize) {
    let mut sm: SkipMap<usize, usize> = SkipMap::with_capacity(size);
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

    for _ in 0..size {
        sm.insert(rng.gen(), rng.gen());
    }

    b.iter(|| {
        for entry in &sm {
            black_box(entry);
        }
    });
}

pub fn benchmark(c: &mut Criterion) {
    c.bench_function("SkipMap index", |b| {
        let size = 100_000;
        let sm: SkipMap<_, _> = (0..size).map(|x| (x, x)).collect();
        b.iter(|| {
            for i in 0..size {
                assert_eq!(sm[i], i)
            }
        })
    });

    c.bench_function("SkipMap insert 1 (empty)", |b| {
        bench_insert(b, 0, 1);
    });
    c.bench_function("SkipMap insert 10 (empty)", |b| {
        bench_insert(b, 0, 10);
    });
    c.bench_function("SkipMap insert 100 (empty)", |b| {
        bench_insert(b, 0, 100);
    });
    c.bench_function("SkipMap insert 1000 (empty)", |b| {
        bench_insert(b, 0, 1_000);
    });
    c.bench_function("SkipMap insert 10000 (empty)", |b| {
        bench_insert(b, 0, 10_000);
    });

    c.bench_function("SkipMap insert 1 (filled)", |b| {
        bench_insert(b, 100_000, 1);
    });
    c.bench_function("SkipMap insert 10 (filled)", |b| {
        bench_insert(b, 100_000, 10);
    });
    c.bench_function("SkipMap insert 100 (filled)", |b| {
        bench_insert(b, 100_000, 100);
    });
    c.bench_function("SkipMap insert 1000 (filled)", |b| {
        bench_insert(b, 100_000, 1_000);
    });
    c.bench_function("SkipMap insert 10000 (filled)", |b| {
        bench_insert(b, 100_000, 10_000);
    });

    c.bench_function("SkipMap iter 1", |b| {
        bench_iter(b, 1);
    });
    c.bench_function("SkipMap iter 10", |b| {
        bench_iter(b, 10);
    });
    c.bench_function("SkipMap iter 100", |b| {
        bench_iter(b, 100);
    });
    c.bench_function("SkipMap iter 1000", |b| {
        bench_iter(b, 1000);
    });
    c.bench_function("SkipMap iter 10000", |b| {
        bench_iter(b, 10_000);
    });
}
