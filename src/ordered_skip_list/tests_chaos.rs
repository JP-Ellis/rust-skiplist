//! Randomised operation sequences under comparators that answer inconsistently.
//!
//! A container must either finish an operation or panic with the documented
//! message, and it must stay structurally well formed after either outcome.
//! Dropping the list at the end of each run lets Miri report leaks and
//! use-after-free.

#![expect(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::integer_division_remainder_used,
    clippy::let_underscore_untyped,
    reason = "the harness reduces random words to small bounded values and discards results"
)]

#[cfg(feature = "cursor")]
use core::ops::Bound;
use std::panic::{AssertUnwindSafe, catch_unwind};

use rand::{RngExt as _, SeedableRng as _, rngs::SmallRng};

use crate::{
    Geometric, OrderedSkipList,
    comparator::{Comparator, ComparatorKey, OrdComparator},
    test_util::{
        Chaos, ChaosOrd, Elem, FixedLevels, Rps, STEPS, SeededLevels, assert_contract_panic,
        check_step, seed_chaos_ord, seeded_geometric, seeds,
    },
};

const INVERTED_RANGE_PANIC: &str = "range start is after range end in OrderedSkipList";

/// The number of operations `step` can pick from.
const OPS: u32 = if cfg!(feature = "cursor") { 22 } else { 18 };

// MARK: Driver

/// Runs one randomly chosen operation on `list`.
fn step<T, C, G, const N: usize>(
    list: &mut OrderedSkipList<T, N, C, G>,
    op: u32,
    k: u8,
    k2: u8,
    index: usize,
    mask: u32,
    level_seed: u32,
) where
    T: Elem + PartialEq,
    C: Comparator<T> + ComparatorKey<T, T> + Clone,
    G: SeededLevels + Clone,
{
    let pred = |v: &T| (mask >> (v.raw() % 32)) & 1 == 1;
    match op {
        0 => list.extract_if(|v| pred(v)).for_each(drop),
        1 => list.insert(T::mk(k)),
        2 => {
            let _ = list.get_or_insert(T::mk(k));
        }
        3 => {
            let _ = list.get_or_insert_with(T::mk(k), |v| v);
        }
        4 => drop(list.take(&T::mk(k))),
        5 => drop(list.take_first(&T::mk(k))),
        6 => drop(list.take_last(&T::mk(k))),
        7 => drop(list.take_fast(&T::mk(k))),
        8 => {
            if !list.is_empty() {
                let _ = list.remove(index % list.len());
            }
        }
        9 => drop(list.remove_all(&T::mk(k))),
        10 => drop(list.pop_first()),
        11 => drop(list.pop_last()),
        12 => {
            let mut half = list.split_off(&T::mk(k));
            list.append(&mut half);
        }
        13 => {
            let mut other = OrderedSkipList::with_comparator_and_level_generator(
                list.comparator().clone(),
                G::from_seed(level_seed),
            );
            other.extend([T::mk(k), T::mk(k.wrapping_add(1)), T::mk(k2)]);
            list.append(&mut other);
        }
        14 => {
            let (lo, hi) = (k.min(k2), k.max(k2));
            list.drain_range(T::mk(lo)..=T::mk(hi)).for_each(drop);
        }
        15 => list.retain(pred),
        16 => list.dedup(),
        17 => list.extend([T::mk(k), T::mk(k2), T::mk(k.wrapping_add(1))]),
        #[cfg(feature = "cursor")]
        18 => drop(
            list.lower_bound_mut(Bound::Included(&T::mk(k)))
                .insert_after(T::mk(k2)),
        ),
        #[cfg(feature = "cursor")]
        19 => drop(
            list.lower_bound_mut(Bound::Included(&T::mk(k)))
                .insert_before(T::mk(k2)),
        ),
        #[cfg(feature = "cursor")]
        20 => drop(
            list.lower_bound_mut(Bound::Included(&T::mk(k)))
                .remove_next(),
        ),
        #[cfg(feature = "cursor")]
        21 => drop(
            list.lower_bound_mut(Bound::Included(&T::mk(k)))
                .remove_prev(),
        ),
        _ => unreachable!("op is below OPS"),
    }
}

/// Applies `steps` random operations to `list`, then drops it.
///
/// A panic from the list does not end the run, so the list is exercised and
/// dropped after it has unwound.
fn run<T, C, G, const N: usize>(mut list: OrderedSkipList<T, N, C, G>, seed: u32, steps: usize)
where
    T: Elem + PartialEq,
    C: Comparator<T> + ComparatorKey<T, T> + Clone,
    G: SeededLevels + Clone,
{
    let mut rng = rng_for(seed);
    for step_index in 0..steps {
        let op = rng.random_range(0..OPS);
        let k = rng.random_range(0..8);
        let k2 = rng.random_range(0..8);
        let index = rng.random_range(0..64);
        let mask = rng.random::<u32>();
        let level_seed = rng.random::<u32>();
        let allowed: &[&str] = if op == 14 {
            &[INVERTED_RANGE_PANIC]
        } else {
            &[]
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            step(&mut list, op, k, k2, index, mask, level_seed);
        }));
        check_step(seed, step_index, op, || {
            assert_contract_panic(result, allowed);
            list.assert_well_formed();
        });
    }
}

fn rng_for(seed: u32) -> SmallRng {
    SmallRng::seed_from_u64(u64::from(seed))
}

// MARK: Tests

#[test]
fn chaos_comparator_n4() {
    for seed in seeds() {
        let list: OrderedSkipList<u8, 4, Chaos, FixedLevels> =
            OrderedSkipList::with_comparator_and_level_generator(
                Chaos::new(seed),
                FixedLevels::new(seed),
            );
        run(list, seed, STEPS);
    }
}

#[test]
fn chaos_comparator_n16() {
    for seed in seeds() {
        let list: OrderedSkipList<u8, 16, Chaos, Geometric> =
            OrderedSkipList::with_comparator_and_level_generator(
                Chaos::new(seed),
                seeded_geometric(seed),
            );
        run(list, seed, STEPS);
    }
}

#[test]
fn chaos_ord() {
    for seed in seeds() {
        seed_chaos_ord(seed);
        let list: OrderedSkipList<ChaosOrd, 16, OrdComparator, Geometric> =
            OrderedSkipList::with_level_generator(seeded_geometric(seed));
        run(list, seed, STEPS);
    }
}

#[test]
fn rps_ord() {
    for seed in seeds() {
        let list: OrderedSkipList<Rps, 16, OrdComparator, Geometric> =
            OrderedSkipList::with_level_generator(seeded_geometric(seed));
        run(list, seed, STEPS);
    }
}

// MARK: Issue 401

/// The reduced program from the issue report.
#[test]
fn issue_401_reproducer() {
    let seed: u32 = 2;
    let mut l: OrderedSkipList<u8, 4, Chaos, FixedLevels> =
        OrderedSkipList::with_comparator_and_level_generator(
            Chaos::new(seed),
            FixedLevels::new(seed),
        );
    let mut r = seed.wrapping_mul(2_654_435_761) | 1;
    let mut next = |n: u32| {
        r = r.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (r >> 16) % n
    };
    for step_index in 0..40 {
        let k = next(8) as u8;
        let op = next(7);
        let index = next(64) as usize;
        let result = catch_unwind(AssertUnwindSafe(|| match op {
            0..=2 => l.insert(k),
            3 => drop(l.take(&k)),
            4 => drop(l.pop_last()),
            5 => drop(l.pop_first()),
            _ => {
                if !l.is_empty() {
                    let _ = l.remove(index % l.len());
                }
            }
        }));
        check_step(seed, step_index, op, || {
            check_step(seed, step_index, op, || {
                assert_contract_panic(result, &[]);
                l.assert_well_formed();
            });
        });
    }
}

/// The same program through a safe `Ord` implementation and `OrdComparator`.
#[test]
fn issue_401_ord_comparator() {
    let seed_count = if cfg!(miri) { 4 } else { 200 };
    for seed in 0_u32..seed_count {
        seed_chaos_ord(seed);
        let mut l: OrderedSkipList<ChaosOrd> =
            OrderedSkipList::with_level_generator(seeded_geometric(seed));
        let mut r = seed.wrapping_mul(2_654_435_761) | 1;
        let mut next = |n: u32| {
            r = r.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (r >> 16) % n
        };
        for step_index in 0..60 {
            let k = ChaosOrd(next(8) as u8);
            let op = next(7);
            let index = next(64) as usize;
            let result = catch_unwind(AssertUnwindSafe(|| match op {
                0..=2 => l.insert(k),
                3 => drop(l.take(&k)),
                4 => drop(l.pop_last()),
                5 => drop(l.pop_first()),
                _ => {
                    if !l.is_empty() {
                        let _ = l.remove(index % l.len());
                    }
                }
            }));
            check_step(seed, step_index, op, || {
                assert_contract_panic(result, &[]);
                l.assert_well_formed();
            });
        }
    }
}
