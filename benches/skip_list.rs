//! Benchmarks for [`SkipList<T>`] comparing insertion and removal operations
//! against equivalent operations on [`Vec`], [`VecDeque`], and [`LinkedList`].

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
    reason = "intentional midpoint indices (`n / 2`)"
)]
#![expect(
    clippy::integer_division_remainder_used,
    reason = "intentional `n / 2` and `x % 2`"
)]

use std::{
    collections::{LinkedList, VecDeque},
    hint::black_box,
};

use criterion::{
    AxisScale, BatchSize, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use rand::{RngExt as _, SeedableRng, rngs::SmallRng};
use skiplist::skip_list::SkipList;

/// Input sizes used for every benchmark group: 10⁰ … 10⁴.
///
/// Covers five orders of magnitude to expose both constant factors
/// (small `n`) and the asymptotic regime (large `n`).
const SIZES: &[usize] = &[
    1, 2, 5, //
    10, 20, 50, //
    100, 200, 500, //
    1_000, 2_000, 5_000, //
    10_000, 20_000, 50_000, //
    100_000, 200_000, 500_000, //
];

/// Benchmark building a container of `n` elements solely via `push_front` (or
/// the nearest equivalent).
///
/// Drop time is excluded from the measurement by using
/// [`Bencher::iter_with_large_drop`].
///
/// `Vec` has no native `push_front`; `Vec::insert(0, _)` shifts the entire
/// remaining slice on every call, making a full build O(n²).  It is benchmarked
/// only for n ≤ 100 000 to stay within the criterion time budget.
///
/// The container is rebuilt in the (un-timed) setup phase for each iteration;
/// only the single call is measured.  The mutated container is returned so
/// its drop time is excluded.
///
/// | Container    | Per-call complexity |
/// |:-------------|:--------------------|
/// | `SkipList`   | `$O(\log n)$`            |
/// | `Vec`        | `$O(n)$` full shift     |
/// | `VecDeque`   | `$O(1)$` amortised      |
/// | `LinkedList` | `$O(1)$`                |
fn bench_push_front(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_front");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        // ----------------------------------------------------------------
        // SkipList — O(log n) expected per push
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_with_large_drop(|| {
                let mut list = SkipList::new();
                for i in 0..n {
                    list.push_front(black_box(i));
                }
                list
            });
        });

        // ----------------------------------------------------------------
        // Vec::insert(0, _) — O(n) per call (shifts remaining slice); capped
        // at 100 000 elements to avoid O(n²) making measurement impractical
        // ----------------------------------------------------------------
        if n <= 100_000 {
            group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
                b.iter_with_large_drop(|| {
                    let mut vec: Vec<usize> = Vec::new();
                    for i in 0..n {
                        vec.insert(0, black_box(i));
                    }
                    vec
                });
            });
        }

        // ----------------------------------------------------------------
        // VecDeque — O(1) amortised; the natural O(1) prepend baseline
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_with_large_drop(|| {
                let mut deque: VecDeque<usize> = VecDeque::new();
                for i in 0..n {
                    deque.push_front(black_box(i));
                }
                deque
            });
        });

        // ----------------------------------------------------------------
        // LinkedList — O(1); pointer-based prepend, pointer-chasing cost
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_with_large_drop(|| {
                let mut list: LinkedList<usize> = LinkedList::new();
                for i in 0..n {
                    list.push_front(black_box(i));
                }
                list
            });
        });
    }

    group.finish();
}

/// Benchmark building a container of `n` elements solely via `push_back` (or
/// the nearest equivalent).
///
/// Drop time is excluded from the measurement by using
/// [`Bencher::iter_with_large_drop`].
fn bench_push_back(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_back");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        // ----------------------------------------------------------------
        // SkipList — O(log n) expected per push
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_with_large_drop(|| {
                let mut list = SkipList::new();
                for i in 0..n {
                    list.push_back(black_box(i));
                }
                list
            });
        });

        // ----------------------------------------------------------------
        // Vec::push — O(1) amortised; the natural O(1) append baseline
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_with_large_drop(|| {
                let mut vec: Vec<usize> = Vec::new();
                for i in 0..n {
                    vec.push(black_box(i));
                }
                vec
            });
        });

        // ----------------------------------------------------------------
        // VecDeque — O(1) amortised; ring-buffer append
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_with_large_drop(|| {
                let mut deque: VecDeque<usize> = VecDeque::new();
                for i in 0..n {
                    deque.push_back(black_box(i));
                }
                deque
            });
        });

        // ----------------------------------------------------------------
        // LinkedList — O(1); pointer-based append, pointer-chasing cost
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_with_large_drop(|| {
                let mut list: LinkedList<usize> = LinkedList::new();
                for i in 0..n {
                    list.push_back(black_box(i));
                }
                list
            });
        });
    }

    group.finish();
}

/// Benchmark draining a container of `n` elements via `pop_front` (or the
/// nearest equivalent).
///
/// The container is pre-filled during the setup phase so that only the
/// removal loop contributes to the measurement.  Setup time is excluded by
/// using [`Bencher::iter_batched`] with [`BatchSize::LargeInput`].
///
/// `Vec::remove(0)` shifts the entire remaining slice on every call, making
/// a full drain O(n²).  It is benchmarked only for n ≤ 10 000 to stay within
/// the criterion time budget.
fn bench_pop_front(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop_front");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        // ----------------------------------------------------------------
        // SkipList — O(log n) expected per pop
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut list = SkipList::new();
                    for i in 0..n {
                        list.push_back(black_box(i));
                    }
                    list
                },
                |mut list| {
                    for _ in 0..n {
                        black_box(list.pop_front());
                    }
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // Vec::remove(0) — O(n) per call (shifts remaining slice); capped
        // at 100 000 elements to avoid O(n²) making measurement impractical
        // ----------------------------------------------------------------
        if n <= 100_000 {
            group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
                b.iter_batched(
                    || {
                        let mut vec: Vec<usize> = Vec::with_capacity(n);
                        for i in 0..n {
                            vec.push(black_box(i));
                        }
                        vec
                    },
                    |mut vec| {
                        for _ in 0..n {
                            black_box(vec.remove(0));
                        }
                    },
                    BatchSize::LargeInput,
                );
            });
        }

        // ----------------------------------------------------------------
        // VecDeque — O(1) amortised; the natural O(1) front-removal baseline
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut deque: VecDeque<usize> = VecDeque::with_capacity(n);
                    for i in 0..n {
                        deque.push_back(black_box(i));
                    }
                    deque
                },
                |mut deque| {
                    for _ in 0..n {
                        black_box(deque.pop_front());
                    }
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // LinkedList — O(1); pointer-based removal, pointer-chasing cost
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut list: LinkedList<usize> = LinkedList::new();
                    for i in 0..n {
                        list.push_back(black_box(i));
                    }
                    list
                },
                |mut list| {
                    for _ in 0..n {
                        black_box(list.pop_front());
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Benchmark draining a container of `n` elements via `pop_back` (or the
/// nearest equivalent).
///
/// The container is pre-filled during the setup phase so that only the
/// removal loop contributes to the measurement.  Setup time is excluded by
/// using [`Bencher::iter_batched`] with [`BatchSize::LargeInput`].
///
/// Unlike `pop_front`, `Vec::pop` removes from the back in O(1) amortised
/// time (no slice shifting), so it is benchmarked across the full size range.
fn bench_pop_back(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop_back");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        // ----------------------------------------------------------------
        // SkipList — O(log n) expected per pop
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut list = SkipList::new();
                    for i in 0..n {
                        list.push_back(black_box(i));
                    }
                    list
                },
                |mut list| {
                    for _ in 0..n {
                        black_box(list.pop_back());
                    }
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // Vec::pop — O(1) amortised; no slice shift, so runs full size range
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut vec: Vec<usize> = Vec::with_capacity(n);
                    for i in 0..n {
                        vec.push(black_box(i));
                    }
                    vec
                },
                |mut vec| {
                    for _ in 0..n {
                        black_box(vec.pop());
                    }
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // VecDeque — O(1) amortised; the natural O(1) back-removal baseline
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut deque: VecDeque<usize> = VecDeque::with_capacity(n);
                    for i in 0..n {
                        deque.push_back(black_box(i));
                    }
                    deque
                },
                |mut deque| {
                    for _ in 0..n {
                        black_box(deque.pop_back());
                    }
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // LinkedList — O(1); pointer-based removal, pointer-chasing cost
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut list: LinkedList<usize> = LinkedList::new();
                    for i in 0..n {
                        list.push_back(black_box(i));
                    }
                    list
                },
                |mut list| {
                    for _ in 0..n {
                        black_box(list.pop_back());
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Benchmark inserting `n` elements at random indices.
///
/// Each benchmark iteration starts with an empty container and inserts `n`
/// values one at a time; the `i`-th insertion uses a pre-generated index drawn
/// uniformly from `0..=i` (valid for a container of `i` elements).
///
/// Insertion indices are generated once per size with a seeded [`SmallRng`] and
/// then cloned into each timed batch, so the benchmark measures only the insert
/// overhead — not RNG overhead.
///
/// `Vec::insert` and `VecDeque::insert` shift `O(n)` elements per call, making
/// a full build `O(n²)`.  Both are therefore capped at 10 000 elements to keep
/// measurement time practical.
fn bench_insert_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_random");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        // Pre-generate insertion indices outside the timed section.
        // indices[i] ∈ 0..=i so it is always a valid insertion point into
        // a container currently holding i elements.
        let indices: Vec<usize> = {
            let mut rng = SmallRng::seed_from_u64(42);
            (0..n).map(|i| rng.random_range(0..=i)).collect()
        };

        // ----------------------------------------------------------------
        // SkipList — O(log n) expected per random-index insert
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, _| {
            b.iter_batched(
                || indices.clone(),
                |idx_list| {
                    let mut list = SkipList::new();
                    for (i, &idx) in idx_list.iter().enumerate() {
                        list.insert(black_box(idx), black_box(i));
                    }
                    list
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // Vec::insert — O(n) per call; capped to avoid O(n²) at scale
        // ----------------------------------------------------------------
        if n <= 100_000 {
            group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, _| {
                b.iter_batched(
                    || indices.clone(),
                    |idx_list| {
                        let mut vec: Vec<usize> = Vec::new();
                        for (i, &idx) in idx_list.iter().enumerate() {
                            vec.insert(black_box(idx), black_box(i));
                        }
                        vec
                    },
                    BatchSize::LargeInput,
                );
            });
        }

        // ----------------------------------------------------------------
        // VecDeque::insert — O(n) per call; capped to avoid O(n²) at scale
        // ----------------------------------------------------------------
        if n <= 100_000 {
            group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, _| {
                b.iter_batched(
                    || indices.clone(),
                    |idx_list| {
                        let mut deque: VecDeque<usize> = VecDeque::new();
                        for (i, &idx) in idx_list.iter().enumerate() {
                            deque.insert(black_box(idx), black_box(i));
                        }
                        deque
                    },
                    BatchSize::LargeInput,
                );
            });
        }

        // ----------------------------------------------------------------
        // LinkedList — no insert(idx) method; simulate with split_off +
        // push_back + append.  O(n) per insertion (split_off scans to idx);
        // capped to avoid O(n²) at scale.
        // ----------------------------------------------------------------
        if n <= 20_000 {
            group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, _| {
                b.iter_batched(
                    || indices.clone(),
                    |idx_list| {
                        let mut list: LinkedList<usize> = LinkedList::new();
                        for (i, &idx) in idx_list.iter().enumerate() {
                            let mut tail = list.split_off(black_box(idx));
                            list.push_back(black_box(i));
                            list.append(&mut tail);
                        }
                        list
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }

    group.finish();
}

/// Benchmark removing a single element at a random index from a container of
/// `n` elements.
///
/// Each benchmark iteration starts with a pre-filled container and removes the
/// element at a pre-generated random index.  The container is rebuilt during
/// the setup phase so that only the single removal contributes to the
/// measurement.  Setup time is excluded by using [`Bencher::iter_batched`] with
/// [`BatchSize::LargeInput`].
///
/// The removal index is generated once per size with a seeded [`SmallRng`]
/// (drawn uniformly from `0..n`) and reused across all container variants and
/// benchmark iterations.
///
/// Each iteration performs exactly one removal, so there is no O(n²)
/// accumulation even for O(n)-per-call containers — all containers are
/// benchmarked across the full size range.
///
/// `LinkedList` has no `remove(idx)` method; it is simulated with
/// `split_off(idx) + pop_front + append`.
fn bench_remove_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove_random");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        // Throughput of 1: each iteration performs a single removal.
        group.throughput(Throughput::Elements(1));

        // Pre-generate one random removal index outside the timed section.
        let idx = {
            let mut rng = SmallRng::seed_from_u64(42);
            rng.random_range(0..n)
        };

        // ----------------------------------------------------------------
        // SkipList — O(log n) expected per removal
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut list = SkipList::new();
                    for i in 0..n {
                        list.push_back(black_box(i));
                    }
                    list
                },
                |mut list| {
                    black_box(list.remove(idx));
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // Vec::remove — O(n) per call (shifts remaining slice)
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut vec: Vec<usize> = Vec::with_capacity(n);
                    for i in 0..n {
                        vec.push(black_box(i));
                    }
                    vec
                },
                |mut vec| {
                    black_box(vec.remove(idx));
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // VecDeque::remove — O(n) per call (shifts half the ring buffer)
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut deque: VecDeque<usize> = VecDeque::with_capacity(n);
                    for i in 0..n {
                        deque.push_back(black_box(i));
                    }
                    deque
                },
                |mut deque| {
                    black_box(deque.remove(idx));
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // LinkedList — no remove(idx) method; simulate with split_off(idx) +
        // pop_front + append.  O(idx) for the split_off scan.
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || (0..n).collect::<LinkedList<usize>>(),
                |mut list| {
                    let mut tail = list.split_off(idx);
                    black_box(tail.pop_front());
                    list.append(&mut tail);
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Benchmark `n` random-index lookups on a pre-filled container of `n` elements.
///
/// Each benchmark iteration starts with a pre-built container and performs `n`
/// reads at pre-generated random indices (seeded for reproducibility).  Setup
/// time (construction + index generation) is excluded via
/// [`Bencher::iter_batched`] with [`BatchSize::LargeInput`].
///
/// `SkipList::get` is O(log n) per call; `Vec` and `VecDeque` indexing is O(1).
/// This benchmark quantifies the constant factor overhead of skip-link
/// traversal relative to direct array access.
///
/// `LinkedList` has no O(1) random-access method; `iter().nth(idx)` is used as
/// the equivalent.
///
/// Throughput is reported as `n` elements (one per lookup within the batch).
fn bench_get_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_random");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        // Pre-generate `n` random indices in `0..n` outside the timed section.
        let indices: Vec<usize> = {
            let mut rng = SmallRng::seed_from_u64(42);
            std::iter::repeat_with(|| rng.random_range(0..n))
                .take(n)
                .collect()
        };

        // ----------------------------------------------------------------
        // SkipList — O(log n) per random-index get
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut list = SkipList::new();
                    for i in 0..n {
                        list.push_back(black_box(i));
                    }
                    (list, indices.clone())
                },
                |(list, idx_list)| {
                    for &idx in &idx_list {
                        black_box(list.get(black_box(idx)));
                    }
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // Vec — O(1) per random-index get
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let vec: Vec<usize> = (0..n).collect();
                    (vec, indices.clone())
                },
                |(vec, idx_list)| {
                    for &idx in &idx_list {
                        black_box(vec.get(black_box(idx)));
                    }
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // VecDeque — O(1) per random-index get
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let deque: VecDeque<usize> = (0..n).collect();
                    (deque, indices.clone())
                },
                |(deque, idx_list)| {
                    for &idx in &idx_list {
                        black_box(deque.get(black_box(idx)));
                    }
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // LinkedList — no O(1) get; iter().nth(idx) is O(n) per lookup.
        // n × O(n) = O(n²) total; capped to avoid impractical run times.
        // ----------------------------------------------------------------
        if n <= 100_000 {
            group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
                b.iter_batched(
                    || {
                        let list: LinkedList<usize> = (0..n).collect();
                        (list, indices.clone())
                    },
                    |(list, idx_list)| {
                        for &idx in &idx_list {
                            black_box(list.iter().nth(black_box(idx)));
                        }
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }

    group.finish();
}

/// Benchmark full forward iteration over a pre-built container of `n` elements.
///
/// Each benchmark iteration starts with a pre-built container and sums all
/// elements in a single forward pass.  `black_box` on the accumulated sum
/// prevents the loop from being optimised away.  Setup time (construction) is
/// excluded via [`Bencher::iter_batched`] with [`BatchSize::LargeInput`].
///
/// `SkipList::iter` incurs O(log n) construction cost (locating front/back
/// nodes) but O(1) per step; `Vec`, `VecDeque`, and `LinkedList` are all O(1)
/// per step.  This benchmark shows the constant-factor overhead of pointer
/// chasing in the skip list relative to contiguous-memory iteration.
fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        // ----------------------------------------------------------------
        // SkipList — O(1) per step after O(1) iterator setup
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut list = SkipList::new();
                    for i in 0..n {
                        list.push_back(black_box(i));
                    }
                    list
                },
                |list| {
                    let mut sum: usize = 0;
                    for &v in &list {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum);
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // Vec — O(1) per step; cache-friendly contiguous memory
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || (0..n).collect::<Vec<usize>>(),
                |vec| {
                    let mut sum: usize = 0;
                    for &v in &vec {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum);
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // VecDeque — O(1) per step; may wrap around ring buffer
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || (0..n).collect::<VecDeque<usize>>(),
                |deque| {
                    let mut sum: usize = 0;
                    for &v in &deque {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum);
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // LinkedList — O(1) per step; pointer-chasing cost per element
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || (0..n).collect::<LinkedList<usize>>(),
                |list| {
                    let mut sum: usize = 0;
                    for &v in &list {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum);
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Benchmark `retain(|&x| x % 2 == 0)` on a pre-built container of `n`
/// elements, keeping roughly half.
///
/// Each benchmark iteration starts with a pre-built container and removes
/// all odd-indexed elements.  Setup time is excluded via
/// [`Bencher::iter_batched`] with [`BatchSize::LargeInput`].
///
/// Throughput is reported as `n` elements examined (not just the half
/// removed), matching how `Vec` reports it.
///
/// `LinkedList::retain` is not yet stable; the equivalent is simulated by
/// draining the list and rebuilding it with only the matching elements.
fn bench_retain(c: &mut Criterion) {
    let mut group = c.benchmark_group("retain");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        // ----------------------------------------------------------------
        // SkipList — O(n) retain with skip-link rebuild
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut list = SkipList::new();
                    for i in 0..n {
                        list.push_back(black_box(i));
                    }
                    list
                },
                |mut list| {
                    list.retain(|&x| x % 2 == 0);
                    black_box(list)
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // Vec — O(n) retain; in-place compaction
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || (0..n).collect::<Vec<usize>>(),
                |mut vec| {
                    vec.retain(|&x| x % 2 == 0);
                    black_box(vec)
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // VecDeque — O(n) retain; in-place compaction
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || (0..n).collect::<VecDeque<usize>>(),
                |mut deque| {
                    deque.retain(|&x| x % 2 == 0);
                    black_box(deque)
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // LinkedList — `retain` is not yet stable; simulate with a drain-
        // and-rebuild pass: O(n) time, same asymptotic cost.
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || (0..n).collect::<LinkedList<usize>>(),
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

/// Benchmark a single `split_off(n/2)` on a pre-built container of `n` elements.
///
/// Each benchmark iteration starts with a pre-built container, splits it at
/// the midpoint, and returns both halves so drop time is excluded from the
/// measurement.  Setup time is excluded via [`Bencher::iter_batched`] with
/// [`BatchSize::LargeInput`].
///
/// Throughput is reported as 1 element (one split-off operation per iteration).
fn bench_split_off(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_off");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        // Throughput of 1: each iteration performs a single split.
        group.throughput(Throughput::Elements(1));

        // ----------------------------------------------------------------
        // SkipList — O(log n + k) expected
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut list = SkipList::new();
                    for i in 0..n {
                        list.push_back(black_box(i));
                    }
                    list
                },
                |mut list| {
                    let tail = list.split_off(n / 2);
                    black_box((list, tail))
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // Vec — O(k) where k = n − at (copies the tail half)
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_batched(
                || (0..n).collect::<Vec<usize>>(),
                |mut vec| {
                    let tail = vec.split_off(n / 2);
                    black_box((vec, tail))
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // VecDeque — O(k) split
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_batched(
                || (0..n).collect::<VecDeque<usize>>(),
                |mut deque| {
                    let tail = deque.split_off(n / 2);
                    black_box((deque, tail))
                },
                BatchSize::LargeInput,
            );
        });

        // ----------------------------------------------------------------
        // LinkedList — O(at) scan to the split point; O(1) pointer relink
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_batched(
                || (0..n).collect::<LinkedList<usize>>(),
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

/// Benchmark building a container of `n` elements from an iterator
/// (`FromIterator` / `collect`).
///
/// Drop time is excluded from the measurement by using
/// [`Bencher::iter_with_large_drop`].
fn bench_build_from_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_from_iter");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in SIZES {
        group.throughput(Throughput::Elements(
            u64::try_from(n).expect("bench size fits in u64"),
        ));

        // ----------------------------------------------------------------
        // SkipList — O(n log n) via n × push_back
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("SkipList", n), &n, |b, &n| {
            b.iter_with_large_drop(|| (0..n).collect::<SkipList<usize>>());
        });

        // ----------------------------------------------------------------
        // Vec — O(n) amortised
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("Vec", n), &n, |b, &n| {
            b.iter_with_large_drop(|| (0..n).collect::<Vec<usize>>());
        });

        // ----------------------------------------------------------------
        // VecDeque — O(n) amortised
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("VecDeque", n), &n, |b, &n| {
            b.iter_with_large_drop(|| (0..n).collect::<VecDeque<usize>>());
        });

        // ----------------------------------------------------------------
        // LinkedList — O(n); pointer allocation per element
        // ----------------------------------------------------------------
        group.bench_with_input(BenchmarkId::new("LinkedList", n), &n, |b, &n| {
            b.iter_with_large_drop(|| (0..n).collect::<LinkedList<usize>>());
        });
    }

    group.finish();
}

criterion_group!(
    benches,
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
criterion_main!(benches);
