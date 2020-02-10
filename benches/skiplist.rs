use criterion::{black_box, Bencher, Criterion};
use rand::prelude::*;
use skiplist::SkipList;

fn bench_push_front(b: &mut Bencher, base: usize, inserts: usize) {
    let mut sl: SkipList<u32> = SkipList::with_capacity(base + inserts);
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

    for _ in 0..base {
        sl.push_front(rng.gen());
    }

    b.iter(|| {
        for _ in 0..inserts {
            sl.push_front(rng.gen());
        }
    });
}

fn bench_push_back(b: &mut Bencher, base: usize, inserts: usize) {
    let mut sl: SkipList<u32> = SkipList::with_capacity(base + inserts);
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

    for _ in 0..base {
        sl.push_back(rng.gen());
    }

    b.iter(|| {
        for _ in 0..inserts {
            sl.push_back(rng.gen());
        }
    });
}

fn bench_iter(b: &mut Bencher, size: usize) {
    let mut sl: SkipList<usize> = SkipList::with_capacity(size);
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

    for _ in 0..size {
        sl.push_back(rng.gen());
    }

    b.iter(|| {
        for entry in &sl {
            black_box(entry);
        }
    });
}

pub fn benchmark(c: &mut Criterion) {
    c.bench_function("index", |b| {
        let size = 100_000;
        let sl: SkipList<_> = (0..size).collect();
        b.iter(|| {
            for i in 0..size {
                assert_eq!(sl[i], i)
            }
        })
    });

    c.bench_function("push_front_0_1", |b| {
        bench_push_front(b, 0, 1);
    });
    c.bench_function("push_front_0_10", |b| {
        bench_push_front(b, 0, 10);
    });
    c.bench_function("push_front_0_100", |b| {
        bench_push_front(b, 0, 100);
    });
    c.bench_function("push_front_0_1000", |b| {
        bench_push_front(b, 0, 1000);
    });
    c.bench_function("push_front_0_10000", |b| {
        bench_push_front(b, 0, 10_000);
    });

    c.bench_function("push_front_100000_1", |b| {
        bench_push_front(b, 100_000, 1);
    });
    c.bench_function("push_front_100000_10", |b| {
        bench_push_front(b, 100_000, 10);
    });
    c.bench_function("push_front_100000_100", |b| {
        bench_push_front(b, 100_000, 100);
    });
    c.bench_function("push_front_100000_1000", |b| {
        bench_push_front(b, 100_000, 1000);
    });
    c.bench_function("push_front_100000_10000", |b| {
        bench_push_front(b, 100_000, 10_000);
    });

    c.bench_function("push_back_0_1", |b| {
        bench_push_back(b, 0, 1);
    });
    c.bench_function("push_back_0_10", |b| {
        bench_push_back(b, 0, 10);
    });
    c.bench_function("push_back_0_100", |b| {
        bench_push_back(b, 0, 100);
    });
    c.bench_function("push_back_0_1000", |b| {
        bench_push_back(b, 0, 1000);
    });
    c.bench_function("push_back_0_10000", |b| {
        bench_push_back(b, 0, 10_000);
    });

    c.bench_function("push_back_100000_1", |b| {
        bench_push_back(b, 100_000, 1);
    });
    c.bench_function("push_back_100000_10", |b| {
        bench_push_back(b, 100_000, 10);
    });
    c.bench_function("push_back_100000_100", |b| {
        bench_push_back(b, 100_000, 100);
    });
    c.bench_function("push_back_100000_1000", |b| {
        bench_push_back(b, 100_000, 1000);
    });
    c.bench_function("push_back_100000_10000", |b| {
        bench_push_back(b, 100_000, 10_000);
    });

    c.bench_function("iter_1", |b| {
        bench_iter(b, 1);
    });
    c.bench_function("iter_10", |b| {
        bench_iter(b, 10);
    });
    c.bench_function("iter_100", |b| {
        bench_iter(b, 100);
    });
    c.bench_function("iter_1000", |b| {
        bench_iter(b, 1000);
    });
    c.bench_function("iter_10000", |b| {
        bench_iter(b, 10_000);
    });
}