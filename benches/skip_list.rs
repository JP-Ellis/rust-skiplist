//! Benchmarks for [`SkipList<T>`] comparing `push_front` against equivalent
//! operations on [`VecDeque`] and [`LinkedList`].
//!
//! | Container    | Operation    | Asymptotic complexity |
//! | ------------ | ------------ | --------------------- |
//! | `SkipList`   | `push_front` | O(log n) expected     |
//! | `VecDeque`   | `push_front` | O(1) amortised        |
//! | `LinkedList` | `push_front` | O(1)                  |
//!
//! Run with:
//!
//! ```shell
//! cargo bench --bench skip_list
//! ```

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
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
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

criterion_group!(benches, bench_push_front);
criterion_main!(benches);
