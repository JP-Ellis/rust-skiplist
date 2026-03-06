//! Benchmarks for [`SkipMap<K, V>`] measuring per-operation latency against
//! [`BTreeMap<K, V>`] and [`HashMap<K, V>`].
//!
//! **Single-operation benchmarks** (`insert`, `get`, `get_mut`, `remove`):
//! each timed iteration performs exactly **one** operation on a pre-populated
//! map of `n` entries.  The map is rebuilt during the (excluded) setup phase.
//!
//! **Insert pool**: the map is pre-built with keys `0..n`.  Each insert uses a
//! new key outside that range, guaranteed absent.  Lookup / mutation pools
//! draw from `0..n` keys that are present in the map.
//!
//! **Bulk benchmarks** (`iter`, `retain`, `build_from_iter`): the operation
//! scales with `n`; throughput in elements is the natural metric.

#![expect(
    clippy::expect_used,
    reason = "`.expect()` is appropriate in benchmarks"
)]
#![expect(
    clippy::shadow_reuse,
    reason = "Criterion `|b, &n|` idiom shadows outer loop var"
)]
#![expect(
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
    collections::{BTreeMap, HashMap},
    hint::black_box,
    time::Duration,
};

use criterion::{
    AxisScale, BatchSize, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom as _};
use skiplist::SkipMap;

/// Input sizes: 10⁰ … ~10⁵·⁷, covering five orders of magnitude.
const SIZES: &[usize] = &[
    1, 2, 5, //
    10, 20, 50, //
    100, 200, 500, //
    1_000, 2_000, 5_000, //
    10_000, 20_000, 50_000, //
    100_000, 200_000, 500_000, //
];

/// Size of the random-key pool cycled through by single-op benchmarks.
const N_RANDOM_INDICES: usize = 1_000;

// MARK: criterion

criterion_group!(
    name = skip_map;
    config = Criterion::default()
        .with_plots()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets =
        bench_insert,
        bench_get,
        bench_get_mut,
        bench_remove,
        bench_iter,
        bench_retain,
        bench_build_from_iter,
);
criterion_main!(skip_map);

// MARK: helpers

/// Build a `SkipMap` of `n` entries with keys `0..n`, values `key * 10`.
fn build_skip_map(n: usize) -> SkipMap<usize, usize> {
    (0..n).map(|k| (k, k * 10)).collect()
}

/// Build a `BTreeMap` of `n` entries with keys `0..n`, values `key * 10`.
fn build_btree_map(n: usize) -> BTreeMap<usize, usize> {
    (0..n).map(|k| (k, k * 10)).collect()
}

/// Build a `HashMap` of `n` entries with keys `0..n`, values `key * 10`.
fn build_hash_map(n: usize) -> HashMap<usize, usize> {
    (0..n).map(|k| (k, k * 10)).collect()
}

/// Pool of new keys for insert benchmarks: `n + (i % N)`, all absent from map.
fn insert_pool(n: usize) -> Vec<usize> {
    (0..N_RANDOM_INDICES)
        .map(|i| n + (i % N_RANDOM_INDICES))
        .collect()
}

/// Pool of existing keys for `get` / `get_mut` / `remove` benchmarks.
fn lookup_pool(n: usize) -> Vec<usize> {
    (0..N_RANDOM_INDICES).map(|i| i % n).collect()
}

// MARK: insert

/// Single key-value insert into a pre-built map of `n` entries.
///
/// Insert keys are `>= n` and thus guaranteed to be new (no replacement).
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipMap`  | `$O(\log n)$`            |
/// | `BTreeMap` | `$O(\log n)$`            |
/// | `HashMap`  | `$O(1)$` amortised      |
fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let pool = insert_pool(n);
        let counter = Cell::new(0_usize);

        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_skip_map(n), pool[i])
                },
                |(mut map, key)| {
                    black_box(map.insert(black_box(key), black_box(key * 10)));
                    map
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_btree_map(n), pool[i])
                },
                |(mut map, key)| {
                    black_box(map.insert(black_box(key), black_box(key * 10)));
                    map
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashMap", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_hash_map(n), pool[i])
                },
                |(mut map, key)| {
                    black_box(map.insert(black_box(key), black_box(key * 10)));
                    map
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: get

/// Single immutable key lookup on a pre-built map of `n` entries.
///
/// The map is built once per size and shared across read-only iterations.
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipMap`  | `$O(\log n)$`            |
/// | `BTreeMap` | `$O(\log n)$`            |
/// | `HashMap`  | `$O(1)$` average        |
fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let pool = lookup_pool(n);
        let counter = Cell::new(0_usize);

        let skip = build_skip_map(n);
        let btree = build_btree_map(n);
        let hash = build_hash_map(n);

        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    pool[i]
                },
                |key| {
                    black_box(skip.get(black_box(&key)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    pool[i]
                },
                |key| {
                    black_box(btree.get(black_box(&key)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashMap", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    pool[i]
                },
                |key| {
                    black_box(hash.get(black_box(&key)));
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// MARK: get_mut

/// Single mutable value access on a pre-built map of `n` entries.
///
/// The returned `&mut V` is immediately `black_box`'d to force the lookup.
/// The map is rebuilt per iteration; drop is excluded from timing.
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipMap`  | `$O(\log n)$`            |
/// | `BTreeMap` | `$O(\log n)$`            |
/// | `HashMap`  | `$O(1)$` average        |
fn bench_get_mut(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_mut");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let pool = lookup_pool(n);
        let counter = Cell::new(0_usize);

        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_skip_map(n), pool[i])
                },
                |(mut map, key)| {
                    black_box(map.get_mut(black_box(&key)));
                    map
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_btree_map(n), pool[i])
                },
                |(mut map, key)| {
                    black_box(map.get_mut(black_box(&key)));
                    map
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashMap", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_hash_map(n), pool[i])
                },
                |(mut map, key)| {
                    black_box(map.get_mut(black_box(&key)));
                    map
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: remove

/// Single key removal from a pre-built map of `n` entries.
///
/// Remove keys are drawn from `0..n` (all present).  The map is rebuilt in
/// setup; the drop of the modified map is excluded from timing.
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipMap`  | `$O(\log n)$`            |
/// | `BTreeMap` | `$O(\log n)$`            |
/// | `HashMap`  | `$O(1)$` average        |
fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let pool = lookup_pool(n);
        let counter = Cell::new(0_usize);

        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_skip_map(n), pool[i])
                },
                |(mut map, key)| {
                    black_box(map.remove(black_box(&key)));
                    map
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_btree_map(n), pool[i])
                },
                |(mut map, key)| {
                    black_box(map.remove(black_box(&key)));
                    map
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashMap", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_hash_map(n), pool[i])
                },
                |(mut map, key)| {
                    black_box(map.remove(black_box(&key)));
                    map
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: iter

/// Full forward iteration over all entries in a pre-built map of `n` elements.
///
/// Each `(key, value)` pair is passed through `black_box`.  The map is
/// returned so its drop is excluded from timing.
///
/// Throughput is `n` elements (one full pass).
fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_map(n),
                |map| {
                    for (k, v) in &map {
                        black_box((k, v));
                    }
                    map
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &n, |b, &n| {
            b.iter_batched(
                || build_btree_map(n),
                |map| {
                    for (k, v) in &map {
                        black_box((k, v));
                    }
                    map
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashMap", n), &n, |b, &n| {
            b.iter_batched(
                || build_hash_map(n),
                |map| {
                    for (k, v) in &map {
                        black_box((k, v));
                    }
                    map
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: retain

/// `retain(|&k, _| k % 2 == 0)` on a pre-built map of `n` entries, keeping half.
///
/// Throughput is `n` entries examined (not just those retained).
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipMap`  | `$O(n)$`                |
/// | `BTreeMap` | `$O(n)$`                |
/// | `HashMap`  | `$O(n)$`                |
fn bench_retain(c: &mut Criterion) {
    let mut group = c.benchmark_group("retain");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_map(n),
                |mut map| {
                    map.retain(|&k, _| k % 2 == 0);
                    black_box(map)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &n, |b, &n| {
            b.iter_batched(
                || build_btree_map(n),
                |mut map| {
                    map.retain(|&k, _| k % 2 == 0);
                    black_box(map)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashMap", n), &n, |b, &n| {
            b.iter_batched(
                || build_hash_map(n),
                |mut map| {
                    map.retain(|&k, _| k % 2 == 0);
                    black_box(map)
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: build_from_iter

/// Build a map from a randomly shuffled iterator of `n` `(key, value)` pairs.
///
/// The shuffled `Vec` is pre-generated once per size (outside timing).  Each
/// iteration clones it in setup (untimed), then measures only the `collect`
/// call.
///
/// Throughput is `n` entries (one full build per iteration).
///
/// | Container  | Per-call complexity |
/// |:-----------|:--------------------|
/// | `SkipMap`  | `$O(n \log n)$`          |
/// | `BTreeMap` | `$O(n \log n)$`          |
/// | `HashMap`  | `$O(n)$` average        |
fn bench_build_from_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_from_iter");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        let shuffled: Vec<(usize, usize)> = {
            let mut v: Vec<(usize, usize)> = (0..n).map(|k| (k, k * 10)).collect();
            let mut rng = SmallRng::seed_from_u64(42);
            v.shuffle(&mut rng);
            v
        };

        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, _| {
            b.iter_batched(
                || shuffled.clone(),
                |data| data.into_iter().collect::<SkipMap<usize, usize>>(),
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &n, |b, _| {
            b.iter_batched(
                || shuffled.clone(),
                |data| data.into_iter().collect::<BTreeMap<usize, usize>>(),
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashMap", n), &n, |b, _| {
            b.iter_batched(
                || shuffled.clone(),
                |data| data.into_iter().collect::<HashMap<usize, usize>>(),
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}
