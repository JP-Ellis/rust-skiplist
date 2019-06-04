use criterion::Criterion;
use criterion::black_box;
use criterion::Bencher;

use rand::{weak_rng, Rng};

use skiplist::SkipMap;


fn bench_insert(b: &mut Bencher, base: usize, inserts: usize) {
    let mut sm: SkipMap<u32, u32> = SkipMap::with_capacity(base + inserts);
    let mut rng = weak_rng();

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
    let mut rng = weak_rng();

    for _ in 0..size {
        sm.insert(rng.gen(), rng.gen());
    }

    b.iter(|| {
        for entry in &sm {
            black_box(entry);
        }
    });
}

pub fn skipmap_benchmark(c: &mut Criterion) {
    c.bench_function("index", |b| {
        let size = 100_000;
        let sm: SkipMap<_, _> = (0..size).map(|x| (x, x)).collect();
        b.iter(|| {
            for i in 0..size {
                assert_eq!(sm[i], i)
            }

        })
    });

    c.bench_function("insert_0_20", |b| {
        bench_insert(b, 0, 20);
    });

    c.bench_function("insert_0_1000", |b| {
        bench_insert(b, 0, 1_000);
    });

    c.bench_function("insert_0_100000", |b| {
        bench_insert(b, 0, 1_00_000);
    });
    c.bench_function("insert_100000_20", |b| {
        bench_insert(b,  1_00_000, 20);
    });
    c.bench_function("iter_20", |b| {
        bench_iter(b, 20);
    });
    c.bench_function("iter_1000", |b| {
        bench_iter(b, 1000);
    });
    c.bench_function("iter_100000", |b| {
        bench_iter(b, 100_000);
    });
}