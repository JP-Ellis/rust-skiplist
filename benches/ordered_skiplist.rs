use criterion::{black_box, Bencher, Criterion};
use rand::prelude::*;
use skiplist::OrderedSkipList;

fn bench_insert(b: &mut Bencher, base: usize, inserts: usize) {
    let mut sl: OrderedSkipList<u32> = OrderedSkipList::with_capacity(base + inserts);
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

    for _ in 0..base {
        sl.insert(rng.gen());
    }

    b.iter(|| {
        for _ in 0..inserts {
            sl.insert(rng.gen());
        }
    });
}

fn bench_iter(b: &mut Bencher, size: usize) {
    let mut sl: OrderedSkipList<usize> = OrderedSkipList::with_capacity(size);
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

    for _ in 0..size {
        sl.insert(rng.gen());
    }

    b.iter(|| {
        for entry in &sl {
            black_box(entry);
        }
    });
}

pub fn benchmark(c: &mut Criterion) {
    c.bench_function("OrderedSkipList index", |b| {
        let size = 100_000;
        let sl: OrderedSkipList<_> = (0..size).collect();
        b.iter(|| {
            for i in 0..size {
                assert_eq!(sl[i], i)
            }
        })
    });

    c.bench_function("OrderedSkipList insert 1 (empty)", |b| {
        bench_insert(b, 0, 1);
    });
    c.bench_function("OrderedSkipList insert 10 (empty)", |b| {
        bench_insert(b, 0, 10);
    });
    c.bench_function("OrderedSkipList insert 100 (empty)", |b| {
        bench_insert(b, 0, 100);
    });
    c.bench_function("OrderedSkipList insert 1000 (empty)", |b| {
        bench_insert(b, 0, 1_000);
    });
    c.bench_function("OrderedSkipList insert 10000 (empty)", |b| {
        bench_insert(b, 0, 10_000);
    });

    c.bench_function("OrderedSkipList insert 1 (filled)", |b| {
        bench_insert(b, 100_000, 1);
    });
    c.bench_function("OrderedSkipList insert 10 (filled)", |b| {
        bench_insert(b, 100_000, 10);
    });
    c.bench_function("OrderedSkipList insert 100 (filled)", |b| {
        bench_insert(b, 100_000, 100);
    });
    c.bench_function("OrderedSkipList insert 1000 (filled)", |b| {
        bench_insert(b, 100_000, 1_000);
    });
    c.bench_function("OrderedSkipList insert 10000 (filled)", |b| {
        bench_insert(b, 100_000, 10_000);
    });

    c.bench_function("OrderedSkipList iter 1", |b| {
        bench_iter(b, 1);
    });
    c.bench_function("OrderedSkipList iter 10", |b| {
        bench_iter(b, 10);
    });
    c.bench_function("OrderedSkipList iter 100", |b| {
        bench_iter(b, 100);
    });
    c.bench_function("OrderedSkipList iter 1000", |b| {
        bench_iter(b, 1000);
    });
    c.bench_function("OrderedSkipList iter 10000", |b| {
        bench_iter(b, 10_000);
    });
}
