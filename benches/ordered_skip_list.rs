//! Benchmarks for [`OrderedSkipList<T>`] measuring per-operation latency against
//! [`BTreeSet<T>`] and a sorted [`Vec<T>`].
//!
//! **Single-operation benchmarks** (`insert`, `contains`, `take`, `pop_first`):
//! each timed iteration performs exactly **one** operation on a pre-populated
//! container of `n` elements.  The container is rebuilt during the (excluded)
//! setup phase.
//!
//! **Insert / take pool**: containers hold *even* values `0, 2, 4, …, 2*(n-1)`.
//! Insert pool values are *odd* `1, 3, 5, …`, cycling modulo `min(N, n)`, so
//! each insertion lands at a uniformly random position rather than always at
//! the tail.  Lookup/remove pools draw from the even values already in the
//! container.
//!
//! **Bulk benchmarks** (`iter`, `range_iter`, `retain`, `build_from_iter`):
//! the operation scales with `n`; throughput in elements is the natural metric.

#![expect(
    clippy::expect_used,
    reason = "`.expect()` is appropriate in benchmarks"
)]
#![expect(
    clippy::shadow_reuse,
    reason = "Criterion `|b, &n|` idiom shadows outer loop var"
)]
#![expect(
    clippy::integer_division,
    clippy::arithmetic_side_effects,
    clippy::integer_division_remainder_used,
    reason = "Performing simple index arithmetic"
)]
#![expect(clippy::indexing_slicing, reason = "Simplifying bench code")]

use std::{cell::Cell, collections::BTreeSet, hint::black_box, time::Duration};

use criterion::{
    AxisScale, BatchSize, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom as _};
use skiplist::OrderedSkipList;

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
    name = ordered_skip_list;
    config = Criterion::default()
        .with_plots()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets =
        bench_insert,
        bench_contains,
        bench_take,
        bench_pop_first,
        bench_iter,
        bench_range_iter,
        bench_retain,
        bench_build_from_iter,
);
criterion_main!(ordered_skip_list);

// MARK: helpers

/// Build an `OrderedSkipList` of `n` elements with even values `0, 2, …, 2*(n-1)`.
fn build_ordered_skip_list(n: usize) -> OrderedSkipList<usize> {
    (0..n).map(|i| i * 2).collect()
}

/// Build a `BTreeSet` of `n` elements with even values `0, 2, …, 2*(n-1)`.
fn build_btree_set(n: usize) -> BTreeSet<usize> {
    (0..n).map(|i| i * 2).collect()
}

/// Build a sorted `Vec` of `n` elements with even values `0, 2, …, 2*(n-1)`.
fn build_sorted_vec(n: usize) -> Vec<usize> {
    (0..n).map(|i| i * 2).collect()
}

/// Odd-valued pool for insert benchmarks.
///
/// Each value `(i % pool_size) * 2 + 1` is an odd number guaranteed to fall
/// strictly between two adjacent even values in the container, placing the
/// insertion at a uniform random position.
fn insert_pool(n: usize) -> Vec<usize> {
    let pool_size = N_RANDOM_INDICES.min(n);
    (0..N_RANDOM_INDICES)
        .map(|i| (i % pool_size) * 2 + 1)
        .collect()
}

/// Even-valued pool for lookup/remove benchmarks.
///
/// All values are guaranteed to be present in the container (they are a subset
/// of `$0, 2, 4, …, 2*(n-1)$`).
fn lookup_pool(n: usize) -> Vec<usize> {
    (0..N_RANDOM_INDICES).map(|i| (i % n) * 2).collect()
}

// MARK: insert

/// Single value insert into a pre-built container of `n` elements.
///
/// Odd insert values land between existing even values, distributing insertions
/// uniformly across the container (avoids always inserting at the tail, which
/// would give `SortedVec` unrealistically low cost).
///
/// | Container         | Per-call complexity |
/// |:------------------|:--------------------|
/// | `OrderedSkipList` | `$O(\log n)$`        |
/// | `BTreeSet`        | `$O(\log n)$`        |
/// | `SortedVec`       | `$O(n)$` (shift)    |
fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let pool = insert_pool(n);
        let counter = Cell::new(0_usize);

        group.bench_with_input(BenchmarkId::new("OrderedSkipList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_ordered_skip_list(n), pool[i])
                },
                |(mut list, val)| {
                    list.insert(black_box(val));
                    list
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
                    set.insert(black_box(val));
                    set
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SortedVec", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_sorted_vec(n), pool[i])
                },
                |(mut vec, val)| {
                    let idx = vec.binary_search(&val).unwrap_or_else(|i| i);
                    vec.insert(black_box(idx), black_box(val));
                    vec
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: contains

/// Single value lookup on a pre-built container of `n` elements.
///
/// The container is built once per size and shared across read-only iterations.
/// Lookup values are drawn from the even-value pool guaranteed to be present.
///
/// | Container         | Per-call complexity      |
/// |:------------------|:-------------------------|
/// | `OrderedSkipList` | `$O(\log n)$`                 |
/// | `BTreeSet`        | `$O(\log n)$`                 |
/// | `SortedVec`       | `$O(\log n)$` (binary search) |
fn bench_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let pool = lookup_pool(n);
        let counter = Cell::new(0_usize);

        let list = build_ordered_skip_list(n);
        let set = build_btree_set(n);
        let vec = build_sorted_vec(n);

        group.bench_with_input(BenchmarkId::new("OrderedSkipList", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    pool[i]
                },
                |val| {
                    black_box(list.contains(black_box(&val)));
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
                    black_box(set.contains(black_box(&val)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SortedVec", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    pool[i]
                },
                |val| {
                    black_box(vec.binary_search(black_box(&val)).is_ok());
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// MARK: take

/// Single value removal from a pre-built container of `n` elements.
///
/// The value to remove is drawn from the even-value pool (all present in the
/// container).  The container is rebuilt in setup; the drop of the modified
/// container is excluded from timing.
///
/// | Container         | Per-call complexity |
/// |:------------------|:--------------------|
/// | `OrderedSkipList` | `$O(\log n)$`       |
/// | `BTreeSet`        | `$O(\log n)$`       |
/// | `SortedVec`       | `$O(n)$` (shift)    |
fn bench_take(c: &mut Criterion) {
    let mut group = c.benchmark_group("take");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let pool = lookup_pool(n);
        let counter = Cell::new(0_usize);

        group.bench_with_input(BenchmarkId::new("OrderedSkipList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_ordered_skip_list(n), pool[i])
                },
                |(mut list, val)| {
                    black_box(list.take_first(black_box(&val)));
                    list
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
                    black_box(set.take(black_box(&val)));
                    set
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SortedVec", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_sorted_vec(n), pool[i])
                },
                |(mut vec, val)| {
                    if let Ok(idx) = vec.binary_search(black_box(&val)) {
                        black_box(vec.remove(idx));
                    }
                    vec
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: pop_first

/// Single `pop_first` on a pre-built container of `n` elements.
///
/// | Container         | Per-call complexity   |
/// |:------------------|:----------------------|
/// | `OrderedSkipList` | `$O(\log n)$`         |
/// | `BTreeSet`        | `$O(\log n)$`         |
/// | `SortedVec`       | `$O(n)$` (full shift) |
fn bench_pop_first(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop_first");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        group.bench_with_input(BenchmarkId::new("OrderedSkipList", n), &n, |b, &n| {
            b.iter_batched(
                || build_ordered_skip_list(n),
                |mut list| {
                    black_box(list.pop_first());
                    list
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

        group.bench_with_input(BenchmarkId::new("SortedVec", n), &n, |b, &n| {
            b.iter_batched(
                || build_sorted_vec(n),
                |mut vec| {
                    if !vec.is_empty() {
                        black_box(vec.remove(0));
                    }
                    vec
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: iter

/// Full forward iteration over a pre-built container of `n` elements.
///
/// Each element is passed through `black_box` to prevent the loop from being
/// optimised away.  The container is returned so its drop is excluded.
///
/// Throughput is `n` elements (one full pass).
fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("OrderedSkipList", n), &n, |b, &n| {
            b.iter_batched(
                || build_ordered_skip_list(n),
                |list| {
                    for &v in &list {
                        black_box(v);
                    }
                    list
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

        group.bench_with_input(BenchmarkId::new("SortedVec", n), &n, |b, &n| {
            b.iter_batched(
                || build_sorted_vec(n),
                |vec| {
                    for &v in &vec {
                        black_box(v);
                    }
                    vec
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: range_iter

/// Iterate over the middle ~50% of a pre-built container of `n` elements.
///
/// Range `lo..hi` where `lo = (n/4)*2` and `hi = (3*n/4)*2` covers roughly
/// half the even-valued elements.  Throughput is `max(1, n/2)` elements.
///
/// | Container         | Per-call complexity |
/// |:------------------|:--------------------|
/// | `OrderedSkipList` | `$O(\log n + k)$`    |
/// | `BTreeSet`        | `$O(\log n + k)$`    |
/// | `SortedVec`       | `$O(\log n + k)$` binary search + slice iter |
///
/// where `$k \simeq \frac{n}{2}$`.
fn bench_range_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_iter");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        let lo = (n / 4) * 2;
        let hi = (3 * n / 4) * 2;
        let count = (n / 2).max(1);

        group.throughput(Throughput::Elements(
            u64::try_from(count).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("OrderedSkipList", n), &n, |b, &n| {
            b.iter_batched(
                || build_ordered_skip_list(n),
                |list| {
                    for &v in list.range(lo..hi) {
                        black_box(v);
                    }
                    list
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeSet", n), &n, |b, &n| {
            b.iter_batched(
                || build_btree_set(n),
                |set| {
                    for &v in set.range(lo..hi) {
                        black_box(v);
                    }
                    set
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SortedVec", n), &n, |b, &n| {
            b.iter_batched(
                || build_sorted_vec(n),
                |vec| {
                    let lo_idx = vec.partition_point(|&x| x < lo);
                    let hi_idx = vec.partition_point(|&x| x < hi);
                    for &v in &vec[lo_idx..hi_idx] {
                        black_box(v);
                    }
                    vec
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: retain

/// `retain(|&x| x % 4 == 0)` on a pre-built container of `n` elements.
///
/// Keeps roughly 25% of elements, examining all `n`.  Throughput is `n`
/// elements examined (not just those retained).
///
/// | Container         | Per-call complexity |
/// |:------------------|:--------------------|
/// | `OrderedSkipList` | `$O(n)$`            |
/// | `BTreeSet`        | `$O(n)$`            |
/// | `SortedVec`       | `$O(n)$`            |
fn bench_retain(c: &mut Criterion) {
    let mut group = c.benchmark_group("retain");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("OrderedSkipList", n), &n, |b, &n| {
            b.iter_batched(
                || build_ordered_skip_list(n),
                |mut list| {
                    list.retain(|&x| x % 4 == 0);
                    black_box(list)
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

        group.bench_with_input(BenchmarkId::new("SortedVec", n), &n, |b, &n| {
            b.iter_batched(
                || build_sorted_vec(n),
                |mut vec| {
                    vec.retain(|&x| x % 4 == 0);
                    black_box(vec)
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: build_from_iter

/// Build a container from a randomly shuffled iterator of `n` elements.
///
/// The shuffled `Vec` is pre-generated once per size (outside timing).  Each
/// iteration clones it in setup (untimed), then measures only the `collect`
/// call.  For `SortedVec` the equivalent is an in-place sort.
///
/// Throughput is `n` elements (one full build per iteration).
///
/// | Container         | Per-call complexity |
/// |:------------------|:--------------------|
/// | `OrderedSkipList` | `$O(n \log n)$`      |
/// | `BTreeSet`        | `$O(n \log n)$`      |
/// | `SortedVec`       | `$O(n \log n)$`      |
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

        group.bench_with_input(BenchmarkId::new("OrderedSkipList", n), &n, |b, _| {
            b.iter_batched(
                || shuffled.clone(),
                |data| data.into_iter().collect::<OrderedSkipList<usize>>(),
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

        group.bench_with_input(BenchmarkId::new("SortedVec", n), &n, |b, _| {
            b.iter_batched(
                || shuffled.clone(),
                |mut data| {
                    data.sort_unstable();
                    data
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}
