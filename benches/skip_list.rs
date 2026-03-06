//! Benchmarks for [`SkipList<T>`] measuring per-operation latency against
//! [`Vec`], [`VecDeque`], and [`LinkedList`].
//!
//! **Single-operation benchmarks** (`push_*`, `pop_*`, `insert_random`,
//! `remove_random`, `get_random`): each timed iteration performs exactly
//! **one** operation on a pre-populated container of `n` elements.  The
//! container is rebuilt during the (excluded) setup phase; only the single
//! target call is measured.
//!
//! For random-index operations a pool of [`N_RANDOM_INDICES`] pre-generated
//! indices is cycled across successive iterations (wrapping) so that each
//! iteration hits a different position, reducing artificial cache-warm-up
//! bias.
//!
//! **Bulk-operation benchmarks** (`iter`, `retain`, `split_off`,
//! `build_from_iter`): the operation is inherently proportional to `n` and
//! throughput in elements processed is the natural metric.

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
    reason = "Performing simple index arithmetics"
)]
#![expect(clippy::indexing_slicing, reason = "Simplifying test code")]
#![expect(
    clippy::linkedlist,
    reason = "Using LinkedList as a benchmark baseline"
)]

use std::{
    cell::Cell,
    collections::{LinkedList, VecDeque},
    hint::black_box,
    iter,
    time::Duration,
};

use criterion::{
    AxisScale, BatchSize, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use rand::{RngExt as _, SeedableRng, rngs::SmallRng};
use skiplist::skip_list::SkipList;

/// Input sizes: 10⁰ … ~10⁵·⁷, covering five orders of magnitude.
const SIZES: &[usize] = &[
    1, 2, 5, //
    10, 20, 50, //
    100, 200, 500, //
    1_000, 2_000, 5_000, //
    10_000, 20_000, 50_000, //
    100_000, 200_000, 500_000, //
];

/// Size of the random-index pool cycled through by random-access benchmarks.
///
/// Large enough to reduce repetition effects across iterations; small enough
/// to fit in L2 cache (8 000 bytes at 8 bytes per `usize`).
const N_RANDOM_INDICES: usize = 1_000;

// MARK: criterion

criterion_group!(
    name = skiplist;
    config = Criterion::default()
        .with_plots()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets =
        bench_push_front,
        bench_push_back,
        bench_pop_front,
        bench_pop_back,
        bench_insert_random,
        bench_remove_random,
        bench_get_random,
        bench_iter,
        bench_retain,
        bench_split_off,
        bench_build_from_iter,
);
criterion_main!(skiplist);

// MARK: helpers

/// Build a skip list of `n` elements containing `0..n-1` in order.
fn build_skip_list(n: usize) -> SkipList<usize> {
    (0..n).collect()
}

/// Build a `Vec` of `n` elements containing `0..n-1` in order.
fn build_vec(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// Build a `VecDeque` of `n` elements containing `0..n-1` in order.
fn build_vecdeque(n: usize) -> VecDeque<usize> {
    (0..n).collect()
}

/// Build a `LinkedList` of `n` elements containing `0..n-1` in order.
fn build_linked_list(n: usize) -> LinkedList<usize> {
    (0..n).collect()
}

// MARK: push_front

/// Single `push_front` onto a pre-built container of `n` elements.
///
/// The container is rebuilt in the (un-timed) setup phase for each iteration;
/// only the single call is measured.  The mutated container is returned so
/// its drop time is excluded.
///
/// | Container    | Per-call complexity |
/// |:-------------|:--------------------|
/// | `SkipList`   | `$O(\log n)$`       |
/// | `Vec`        | `$O(n)$`            |
/// | `VecDeque`   | `$O(1)$`            |
/// | `LinkedList` | `$O(1)$`            |
fn bench_push_front(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_front");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_list(n),
                |mut list| {
                    list.push_front(black_box(n));
                    list
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || build_vec(n),
                |mut vec| {
                    vec.insert(0, black_box(n));
                    vec
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || build_vecdeque(n),
                |mut deque| {
                    deque.push_front(black_box(n));
                    deque
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || build_linked_list(n),
                |mut list| {
                    list.push_front(black_box(n));
                    list
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: push_back

/// Single `push_back` onto a pre-built container of `n` elements.
///
/// | Container    | Per-call complexity |
/// |:-------------|:--------------------|
/// | `SkipList`   | `$O(\log n)$`            |
/// | `Vec`        | `$O(1)$` amortised      |
/// | `VecDeque`   | `$O(1)$` amortised      |
/// | `LinkedList` | `$O(1)$`                |
fn bench_push_back(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_back");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_list(n),
                |mut list| {
                    list.push_back(black_box(n));
                    list
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || build_vec(n),
                |mut vec| {
                    vec.push(black_box(n));
                    vec
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || build_vecdeque(n),
                |mut deque| {
                    deque.push_back(black_box(n));
                    deque
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || build_linked_list(n),
                |mut list| {
                    list.push_back(black_box(n));
                    list
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: pop_front

/// Single `pop_front` on a pre-built container of `n` elements.
///
/// | Container    | Per-call complexity |
/// |:-------------|:--------------------|
/// | `SkipList`   | `$O(\log n)$`            |
/// | `Vec`        | `$O(n)$` full shift   |
/// | `VecDeque`   | `$O(1)$` amortised      |
/// | `LinkedList` | `$O(1)$`                |
fn bench_pop_front(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop_front");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_list(n),
                |mut list| {
                    black_box(list.pop_front());
                    list
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || build_vec(n),
                |mut vec| {
                    black_box(vec.remove(0));
                    vec
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || build_vecdeque(n),
                |mut deque| {
                    black_box(deque.pop_front());
                    deque
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || build_linked_list(n),
                |mut list| {
                    black_box(list.pop_front());
                    list
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: pop_back

/// Single `pop_back` on a pre-built container of `n` elements.
///
/// | Container    | Per-call complexity |
/// |:-------------|:--------------------|
/// | `SkipList`   | `$O(\log n)$`            |
/// | `Vec`        | `$O(1)$` amortised      |
/// | `VecDeque`   | `$O(1)$` amortised      |
/// | `LinkedList` | `$O(1)$`                |
fn bench_pop_back(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop_back");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_list(n),
                |mut list| {
                    black_box(list.pop_back());
                    list
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || build_vec(n),
                |mut vec| {
                    black_box(vec.pop());
                    vec
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || build_vecdeque(n),
                |mut deque| {
                    black_box(deque.pop_back());
                    deque
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || build_linked_list(n),
                |mut list| {
                    black_box(list.pop_back());
                    list
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: insert_random

/// Single insert at a cycled random index into a pre-built container of `n`
/// elements.
///
/// Indices are drawn uniformly from `0..=n` (all valid insertion positions)
/// and pre-generated into a pool of [`N_RANDOM_INDICES`] values.  Each
/// iteration advances the pool pointer by one (wrapping), so successive calls
/// target different positions.
///
/// | Container    | Per-call complexity |
/// |:-------------|:--------------------|
/// | `SkipList`   | `$O(\log n)$`            |
/// | `Vec`        | `$O(n)$` full shift   |
/// | `VecDeque`   | `$O(n)$`                |
/// | `LinkedList` | `$O(n)$` split scan   |
fn bench_insert_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_random");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let indices: Vec<usize> = {
            let mut rng = SmallRng::seed_from_u64(42);
            iter::repeat_with(|| rng.random_range(0..=n))
                .take(N_RANDOM_INDICES)
                .collect()
        };
        let counter = Cell::new(0_usize);

        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_skip_list(n), indices[i])
                },
                |(mut list, idx)| {
                    list.insert(black_box(idx), black_box(n));
                    list
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_vec(n), indices[i])
                },
                |(mut vec, idx)| {
                    vec.insert(black_box(idx), black_box(n));
                    vec
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_vecdeque(n), indices[i])
                },
                |(mut deque, idx)| {
                    deque.insert(black_box(idx), black_box(n));
                    deque
                },
                BatchSize::LargeInput,
            );
        });

        // LinkedList has no insert(idx); simulated via split_off + push_back + append.
        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_linked_list(n), indices[i])
                },
                |(mut list, idx)| {
                    let mut tail = list.split_off(black_box(idx));
                    list.push_back(black_box(n));
                    list.append(&mut tail);
                    list
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: remove_random

/// Single remove at a cycled random index from a pre-built container of `n`
/// elements.
///
/// Indices are drawn uniformly from `0..n` and pre-generated into a pool of
/// [`N_RANDOM_INDICES`] values cycled across iterations.
///
/// | Container    | Per-call complexity |
/// |:-------------|:--------------------|
/// | `SkipList`   | `$O(\log n)$`            |
/// | `Vec`        | `$O(n)$` full shift   |
/// | `VecDeque`   | `$O(n)$`                |
/// | `LinkedList` | `$O(n)$` split scan   |
fn bench_remove_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove_random");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let indices: Vec<usize> = {
            let mut rng = SmallRng::seed_from_u64(42);
            iter::repeat_with(|| rng.random_range(0..n))
                .take(N_RANDOM_INDICES)
                .collect()
        };
        let counter = Cell::new(0_usize);

        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_skip_list(n), indices[i])
                },
                |(mut list, idx)| {
                    black_box(list.remove(black_box(idx)));
                    list
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_vec(n), indices[i])
                },
                |(mut vec, idx)| {
                    black_box(vec.remove(black_box(idx)));
                    vec
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_vecdeque(n), indices[i])
                },
                |(mut deque, idx)| {
                    black_box(deque.remove(black_box(idx)));
                    deque
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    (build_linked_list(n), indices[i])
                },
                |(mut list, idx)| {
                    let mut tail = list.split_off(black_box(idx));
                    black_box(tail.pop_front());
                    list.append(&mut tail);
                    list
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: get_random

/// Single random-index lookup on a pre-built container of `n` elements.
///
/// The container is built **once per size** (not per iteration) since the
/// lookup is read-only.  Indices are drawn uniformly from `0..n` and
/// pre-generated into a pool of [`N_RANDOM_INDICES`] values cycled across
/// iterations so successive calls hit different positions.
///
/// | Container    | Per-call complexity |
/// |:-------------|:--------------------|
/// | `SkipList`   | `$O(\log n)$`            |
/// | `Vec`        | `$O(1)$`                |
/// | `VecDeque`   | `$O(1)$`                |
/// | `LinkedList` | `$O(n)$` linear scan  |
fn bench_get_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_random");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        let indices: Vec<usize> = {
            let mut rng = SmallRng::seed_from_u64(42);
            std::iter::repeat_with(|| rng.random_range(0..n))
                .take(N_RANDOM_INDICES)
                .collect()
        };
        let counter = Cell::new(0_usize);

        // Containers are built once and shared across iterations (read-only).
        let skip_list: SkipList<usize> = build_skip_list(n);
        let vec: Vec<usize> = build_vec(n);
        let vecdeque: VecDeque<usize> = build_vecdeque(n);
        let linked_list: LinkedList<usize> = build_linked_list(n);

        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    indices[i]
                },
                |idx| {
                    black_box(skip_list.get(black_box(idx)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    indices[i]
                },
                |idx| {
                    black_box(vec.get(black_box(idx)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    indices[i]
                },
                |idx| {
                    black_box(vecdeque.get(black_box(idx)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let i = counter.get();
                    counter.set((i + 1) % N_RANDOM_INDICES);
                    indices[i]
                },
                |idx| {
                    black_box(linked_list.iter().nth(black_box(idx)));
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// MARK: iter

/// Full forward iteration over a pre-built container of `n` elements.
///
/// Each iteration sums all elements in a single pass; `black_box` on the
/// accumulated sum prevents the loop from being optimised away.  The
/// container is returned from the routine so its drop time is excluded.
///
/// Throughput is `n` elements (one pass over the whole container).
fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_list(n),
                |list| {
                    for &v in &list {
                        black_box(v);
                    }
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || build_vec(n),
                |vec| {
                    for &v in &vec {
                        black_box(v);
                    }
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || build_vecdeque(n),
                |deque| {
                    for &v in &deque {
                        black_box(v);
                    }
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || build_linked_list(n),
                |list| {
                    for &v in &list {
                        black_box(v);
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: retain

/// `retain(|&x| x % 2 == 0)` on a pre-built container of `n` elements,
/// keeping roughly half.
///
/// Throughput is `n` elements examined (not just the half removed).
///
/// `LinkedList::retain` is not yet stable; the equivalent is simulated by
/// consuming the list and rebuilding it with only the matching elements.
fn bench_retain(c: &mut Criterion) {
    let mut group = c.benchmark_group("retain");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_list(n),
                |mut list| {
                    list.retain(|&x| x % 2 == 0);
                    black_box(list)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || build_vec(n),
                |mut vec| {
                    vec.retain(|&x| x % 2 == 0);
                    black_box(vec)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || build_vecdeque(n),
                |mut deque| {
                    deque.retain(|&x| x % 2 == 0);
                    black_box(deque)
                },
                BatchSize::LargeInput,
            );
        });

        // LinkedList::retain is not yet stable; simulate with drain-and-rebuild.
        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || build_linked_list(n),
                |list| {
                    let mut retained = LinkedList::new();
                    for x in list {
                        if x % 2 == 0 {
                            retained.push_back(x);
                        }
                    }
                    black_box(retained)
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: split_off

/// Single `split_off(n/2)` on a pre-built container of `n` elements.
///
/// Both halves are returned so their drop time is excluded.
/// Throughput is `n` elements (container size).
fn bench_split_off(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_off");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || build_skip_list(n),
                |mut list| {
                    let tail = list.split_off(n / 2);
                    black_box((list, tail))
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || build_vec(n),
                |mut vec| {
                    let tail = vec.split_off(n / 2);
                    black_box((vec, tail))
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || build_vecdeque(n),
                |mut deque| {
                    let tail = deque.split_off(n / 2);
                    black_box((deque, tail))
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || build_linked_list(n),
                |mut list| {
                    let tail = list.split_off(n / 2);
                    black_box((list, tail))
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// MARK: build_from_iter

/// Build a container of `n` elements from an iterator (`collect`).
///
/// Drop time is excluded by using [`Bencher::iter_with_large_drop`].
/// Throughput is `n` elements (one build per iteration).
fn bench_build_from_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_from_iter");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_with_large_drop(|| (0..n).collect::<SkipList<usize>>());
        });

        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_with_large_drop(|| (0..n).collect::<Vec<usize>>());
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_with_large_drop(|| (0..n).collect::<VecDeque<usize>>());
        });

        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_with_large_drop(|| (0..n).collect::<LinkedList<usize>>());
        });
    }

    group.finish();
}
