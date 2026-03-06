//! Benchmarks for [`SkipSet<T>`] measuring per-operation latency against
//! [`BTreeSet<T>`] and [`HashSet<T>`].
//!
//! **Single-operation benchmarks** (`insert`, `contains`, `remove`,
//! `pop_first`): each timed iteration performs exactly **one** operation on a
//! pre-populated container of `n` elements.  The container is rebuilt during
//! the (excluded) setup phase.
//!
//! **Insert pool**: containers hold *even* values `0, 2, 4, …, 2*(n-1)`.
//! Insert pool values are *odd* `1, 3, 5, …`, cycling modulo `min(N, n)`.
//! Lookup/remove pools draw from even values already in the container.
//!
//! **Set-algebra benchmarks** (`union`, `intersection`): two sets of `n`
//! elements each are built with an overlap of `n/2` elements.
//!
//! **Bulk benchmarks** (`iter`, `retain`, `build_from_iter`): the operation
//! scales with `n`; throughput in elements is the natural metric.

#![expect(
    clippy::expect_used,
    reason = "`.expect()` is appropriate in benchmarks"
)]
#![expect(
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
    reason = "Criterion idiom shadows outer loop var"
)]
#![expect(
    clippy::integer_division,
    clippy::arithmetic_side_effects,
    clippy::integer_division_remainder_used,
    reason = "Performing simple index arithmetic"
)]
#![expect(
    clippy::iter_over_hash_type,
    reason = "Iterating over a HashSet/HashMap for benchmarking is fine"
)]
#![expect(clippy::indexing_slicing, reason = "Simplifying bench code")]

use std::{
    cell::Cell,
    collections::{BTreeSet, HashSet},
    hint::black_box,
    time::Duration,
};

use criterion::{
    AxisScale, BatchSize, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom as _};
use skiplist::SkipSet;

/// Input sizes: 10⁰ … ~10⁵·⁷, covering five orders of magnitude.
const SIZES: &[usize] = &[
    1, 2, 5, //
    10, 20, 50, //
    100, 200, 500, //
    1_000, 2_000, 5_000, //
    10_000, 20_000, 50_000, //
    100_000, 200_000, 500_000, //
];

/// Size of the random-value pool cycled through by single-op benchmarks.
const N_RANDOM_INDICES: usize = 1_000;

// MARK: criterion

criterion_group!(
    name = skip_set;
    config = Criterion::default()
        .with_plots()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets =
        bench_insert,
        bench_contains,
        bench_remove,
        bench_pop_first,
        bench_iter,
        bench_retain,
        bench_build_from_iter,
        bench_union,
        bench_intersection,
);
criterion_main!(skip_set);

// MARK: helpers

/// Build a `SkipSet` of `n` elements with even values `0, 2, …, 2*(n-1)`.
fn build_skip_set(n: usize) -> SkipSet<usize> {
    (0..n).map(|i| i * 2).collect()
}

/// Build a `BTreeSet` of `n` elements with even values `0, 2, …, 2*(n-1)`.
fn build_btree_set(n: usize) -> BTreeSet<usize> {
    (0..n).map(|i| i * 2).collect()
}

/// Build a `HashSet` of `n` elements with even values `0, 2, …, 2*(n-1)`.
fn build_hash_set(n: usize) -> HashSet<usize> {
    (0..n).map(|i| i * 2).collect()
}

/// Odd-valued pool for insert benchmarks.
///
/// Each value `(i % pool_size) * 2 + 1` is odd and not present in the
/// even-valued container, placing the insertion at a uniform random position.
fn insert_pool(n: usize) -> Vec<usize> {
    let pool_size = N_RANDOM_INDICES.min(n);
    (0..N_RANDOM_INDICES)
        .map(|i| (i % pool_size) * 2 + 1)
        .collect()
}

/// Even-valued pool for lookup/remove benchmarks.
///
/// All values are guaranteed to be present in the container (subset of
/// `0, 2, 4, …, 2*(n-1)`).
fn lookup_pool(n: usize) -> Vec<usize> {
    (0..N_RANDOM_INDICES).map(|i| (i % n) * 2).collect()
}

// MARK: insert

/// Single value insert into a pre-built set of `n` elements.
///
/// Odd values from the insert pool are new (not in the even-valued set), so
/// each call performs a real insertion at a uniform random position.
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipSet`  | `$O(\log n)$`            |
/// | `BTreeSet` | `$O(\log n)$`            |
/// | `HashSet`  | `$O(1)$` amortised      |
fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let pool = insert_pool(n);
        let counter = Cell::new(0_usize);

        group.bench_with_input(BenchmarkId::new("SkipSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_skip_set(n), pool[i])
                },
                |(mut set, val)| {
                    black_box(set.insert(black_box(val)));
                    set
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_btree_set(n), pool[i])
                },
                |(mut set, val)| {
                    black_box(set.insert(black_box(val)));
                    set
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_hash_set(n), pool[i])
                },
                |(mut set, val)| {
                    black_box(set.insert(black_box(val)));
                    set
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: contains

/// Single value lookup on a pre-built set of `n` elements.
///
/// The container is built once per size and shared across read-only iterations.
///
/// | Container  | Per-call complexity      |
/// |:-----------|:-------------------------|
/// | `SkipSet`  | `$O(\log n)$`                 |
/// | `BTreeSet` | `$O(\log n)$`                 |
/// | `HashSet`  | `$O(1)$` average             |
fn bench_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let pool = lookup_pool(n);
        let counter = Cell::new(0_usize);

        let skip = build_skip_set(n);
        let btree = build_btree_set(n);
        let hash = build_hash_set(n);

        group.bench_with_input(BenchmarkId::new("SkipSet", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    pool[i]
                },
                |val| {
                    black_box(skip.contains(black_box(&val)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeSet", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    pool[i]
                },
                |val| {
                    black_box(btree.contains(black_box(&val)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashSet", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    pool[i]
                },
                |val| {
                    black_box(hash.contains(black_box(&val)));
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// MARK: remove

/// Single value removal from a pre-built set of `n` elements.
///
/// The value to remove is drawn from the even-value pool (all present).
/// The container is rebuilt in setup; the drop is excluded from timing.
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipSet`  | `$O(\log n)$`            |
/// | `BTreeSet` | `$O(\log n)$`            |
/// | `HashSet`  | `$O(1)$` average        |
fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let pool = lookup_pool(n);
        let counter = Cell::new(0_usize);

        group.bench_with_input(BenchmarkId::new("SkipSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_skip_set(n), pool[i])
                },
                |(mut set, val)| {
                    black_box(set.remove(black_box(&val)));
                    set
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_btree_set(n), pool[i])
                },
                |(mut set, val)| {
                    black_box(set.remove(black_box(&val)));
                    set
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_hash_set(n), pool[i])
                },
                |(mut set, val)| {
                    black_box(set.remove(black_box(&val)));
                    set
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: pop_first

/// Single `pop_first` on a pre-built set of `n` elements.
///
/// `HashSet` has no `pop_first`; the closest equivalent is `iter().next()`
/// (arbitrary first element) followed by `remove`.  This measures the same
/// conceptual operation but without ordering guarantees.
///
/// | Container  | Per-call complexity                        |
/// |:-----------|:-------------------------------------------|
/// | `SkipSet`  | `$O(\log n)$`                                   |
/// | `BTreeSet` | `$O(\log n)$`                                   |
/// | `HashSet`  | `$O(1)$` avg `iter().next()` + `remove`      |
fn bench_pop_first(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop_first");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        group.bench_with_input(BenchmarkId::new("SkipSet", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_set(n),
                |mut set| {
                    black_box(set.pop_first());
                    set
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeSet", n), &n, |b, &n| {
            b.iter_batched(
                || build_btree_set(n),
                |mut set| {
                    black_box(set.pop_first());
                    set
                },
                BatchSize::LargeInput,
            );
        });

        // HashSet has no pop_first; simulate with iter().next() + remove.
        group.bench_with_input(BenchmarkId::new("HashSet", n), &n, |b, &n| {
            b.iter_batched(
                || build_hash_set(n),
                |mut set| {
                    let first = set.iter().next().copied();
                    if let Some(val) = first {
                        set.remove(&val);
                    }
                    black_box(first);
                    set
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: iter

/// Full forward iteration over a pre-built set of `n` elements.
///
/// Each element is passed through `black_box`.  The container is returned so
/// its drop is excluded from timing.
///
/// Throughput is `n` elements (one full pass).
fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("SkipSet", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_set(n),
                |set| {
                    for &v in &set {
                        black_box(v);
                    }
                    set
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeSet", n), &n, |b, &n| {
            b.iter_batched(
                || build_btree_set(n),
                |set| {
                    for &v in &set {
                        black_box(v);
                    }
                    set
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashSet", n), &n, |b, &n| {
            b.iter_batched(
                || build_hash_set(n),
                |set| {
                    for &v in &set {
                        black_box(v);
                    }
                    set
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: retain

/// `retain(|&x| x % 2 == 0)` on a pre-built set of `n` elements, keeping half.
///
/// Since all stored values are even, this retains all of them, measuring the
/// full scan cost without structural mutations.  Use `x % 4 == 0` to actually
/// drop elements.  Here we use `x % 4 == 0` so ~25% are retained.
///
/// Throughput is `n` elements examined (not just those retained).
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipSet`  | `$O(n)$`                |
/// | `BTreeSet` | `$O(n)$`                |
/// | `HashSet`  | `$O(n)$`                |
fn bench_retain(c: &mut Criterion) {
    let mut group = c.benchmark_group("retain");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("SkipSet", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_set(n),
                |mut set| {
                    set.retain(|&x| x % 4 == 0);
                    black_box(set)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeSet", n), &n, |b, &n| {
            b.iter_batched(
                || build_btree_set(n),
                |mut set| {
                    set.retain(|&x| x % 4 == 0);
                    black_box(set)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashSet", n), &n, |b, &n| {
            b.iter_batched(
                || build_hash_set(n),
                |mut set| {
                    set.retain(|&x| x % 4 == 0);
                    black_box(set)
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: build_from_iter

/// Build a set from a randomly shuffled iterator of `n` elements.
///
/// The shuffled `Vec` is pre-generated once per size (outside timing).  Each
/// iteration clones it in setup (untimed), then measures only the `collect`
/// call.
///
/// Throughput is `n` elements (one full build per iteration).
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipSet`  | `$O(n \log n)$`          |
/// | `BTreeSet` | `$O(n \log n)$`          |
/// | `HashSet`  | `$O(n)$` average        |
fn bench_build_from_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_from_iter");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        let shuffled: Vec<usize> = {
            let mut v: Vec<usize> = (0..n).collect();
            let mut rng = SmallRng::seed_from_u64(42);
            v.shuffle(&mut rng);
            v
        };

        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("SkipSet", n), &n, |b, _| {
            b.iter_batched(
                || shuffled.clone(),
                |data| data.into_iter().collect::<SkipSet<usize>>(),
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeSet", n), &n, |b, _| {
            b.iter_batched(
                || shuffled.clone(),
                |data| data.into_iter().collect::<BTreeSet<usize>>(),
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashSet", n), &n, |b, _| {
            b.iter_batched(
                || shuffled.clone(),
                |data| data.into_iter().collect::<HashSet<usize>>(),
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: union

/// Count all elements in the union of two sets of `n` elements each.
///
/// `set_a` = even values `0..2*n`; `set_b` = even values `n..3*n`; overlap
/// = `n/2` elements in `n..2*n`.  Both sets are built in setup; the lazy
/// union iterator is driven to completion by `.count()`.
///
/// Throughput is `n` elements (one set size, representing work proportional
/// to the input size).
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipSet`  | `$O(n)$`                |
/// | `BTreeSet` | `$O(n)$`                |
/// | `HashSet`  | `$O(n)$`                |
fn bench_union(c: &mut Criterion) {
    let mut group = c.benchmark_group("union");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("SkipSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let a: SkipSet<usize> = (0..n).map(|i| i * 2).collect();
                    let b: SkipSet<usize> = (0..n).map(|i| n + i * 2).collect();
                    (a, b)
                },
                |(a, b)| {
                    black_box(a.union(&b).count());
                    (a, b)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let a: BTreeSet<usize> = (0..n).map(|i| i * 2).collect();
                    let b: BTreeSet<usize> = (0..n).map(|i| n + i * 2).collect();
                    (a, b)
                },
                |(a, b)| {
                    black_box(a.union(&b).count());
                    (a, b)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let a: HashSet<usize> = (0..n).map(|i| i * 2).collect();
                    let b: HashSet<usize> = (0..n).map(|i| n + i * 2).collect();
                    (a, b)
                },
                |(a, b)| {
                    black_box(a.union(&b).count());
                    (a, b)
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: intersection

/// Count all elements in the intersection of two sets of `n` elements each.
///
/// `set_a` = even values `0..2*n`; `set_b` = even values `n..3*n`; overlap
/// = `n/2` elements.  Throughput is `max(1, n/2)` elements (the intersection
/// size).
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipSet`  | `$O(n)$`                |
/// | `BTreeSet` | `$O(n)$`                |
/// | `HashSet`  | `$O(n)$` average        |
fn bench_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersection");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        let count = (n / 2).max(1);
        group.throughput(Throughput::Elements(
            u64::try_from(count).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("SkipSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let a: SkipSet<usize> = (0..n).map(|i| i * 2).collect();
                    let b: SkipSet<usize> = (0..n).map(|i| n + i * 2).collect();
                    (a, b)
                },
                |(a, b)| {
                    black_box(a.intersection(&b).count());
                    (a, b)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let a: BTreeSet<usize> = (0..n).map(|i| i * 2).collect();
                    let b: BTreeSet<usize> = (0..n).map(|i| n + i * 2).collect();
                    (a, b)
                },
                |(a, b)| {
                    black_box(a.intersection(&b).count());
                    (a, b)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashSet", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let a: HashSet<usize> = (0..n).map(|i| i * 2).collect();
                    let b: HashSet<usize> = (0..n).map(|i| n + i * 2).collect();
                    (a, b)
                },
                |(a, b)| {
                    black_box(a.intersection(&b).count());
                    (a, b)
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}
