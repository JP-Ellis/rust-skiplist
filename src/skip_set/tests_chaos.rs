//! Randomised operation sequences on [`SkipSet`] under comparators that answer
//! inconsistently.
//!
//! See the matching module for `OrderedSkipList` for the contract being checked.

#![expect(
    clippy::arithmetic_side_effects,
    clippy::integer_division_remainder_used,
    clippy::let_underscore_untyped,
    reason = "the harness reduces random words to small bounded values and discards results"
)]

#[cfg(feature = "cursor")]
use core::ops::Bound;
use std::panic::{AssertUnwindSafe, catch_unwind};

use rand::{RngExt as _, SeedableRng as _, rngs::SmallRng};

use crate::{
    Geometric, SkipSet,
    comparator::{Comparator, ComparatorKey, OrdComparator},
    level_generator::LevelGenerator,
    skip_set::Entry,
    test_util::{
        Chaos, ChaosOrd, Elem, FixedLevels, Rps, assert_contract_panic, check_step, seed_chaos_ord,
        seeded_geometric, seeds,
    },
};

/// The number of operations `step` can pick from.
const OPS: u32 = if cfg!(feature = "cursor") { 22 } else { 18 };

// MARK: Driver

/// Runs one randomly chosen operation on `set`.
fn step<T, C, G, const N: usize>(
    set: &mut SkipSet<T, N, C, G>,
    op: u32,
    k: u8,
    k2: u8,
    index: usize,
    mask: u32,
) where
    T: Elem,
    C: Comparator<T> + ComparatorKey<T, T> + Clone,
    G: LevelGenerator + Clone + Default,
{
    let pred = |v: &T| (mask >> (v.raw() % 32)) & 1 == 1;
    match op {
        0 => {
            let (lo, hi) = (k.min(k2), k.max(k2));
            set.extract_if(T::mk(lo)..=T::mk(hi), |v| pred(v))
                .for_each(drop);
        }
        1 => drop(set.insert(T::mk(k))),
        2 => {
            let _ = set.get_or_insert(T::mk(k));
        }
        3 => {
            let _ = set.get_or_insert_with(T::mk(k), |v| v);
        }
        4 => drop(set.replace(T::mk(k))),
        5 => drop(set.take(&T::mk(k))),
        6 => drop(set.remove(&T::mk(k))),
        7 => drop(set.pop_first()),
        8 => drop(set.pop_last()),
        9 => {
            let mut half = set.split_off(&T::mk(k));
            set.append(&mut half);
        }
        10 => {
            let mut half = set.split_off_index(index % (set.len() + 1));
            set.append(&mut half);
        }
        11 => {
            let mut other = SkipSet::with_comparator_and_level_generator(
                set.inner.comparator().clone(),
                G::default(),
            );
            other.extend([T::mk(k), T::mk(k.wrapping_add(1)), T::mk(k2)]);
            set.append(&mut other);
        }
        12 => set.retain(pred),
        13 => set.extend([T::mk(k), T::mk(k2), T::mk(k.wrapping_add(1))]),
        14 => {
            let _ = set.entry(T::mk(k)).or_insert();
        }
        15 => {
            if let Entry::Occupied(e) = set.entry(T::mk(k)) {
                let _ = e.remove();
            }
        }
        16 => {
            if let Entry::Vacant(e) = set.entry(T::mk(k)) {
                let _ = e.insert();
            }
        }
        17 => {
            let _ = set.entry(T::mk(k)).or_insert_with(|| T::mk(k2));
        }
        #[cfg(feature = "cursor")]
        18 => drop(
            set.lower_bound_mut(Bound::Included(&T::mk(k)))
                .insert_after(T::mk(k2)),
        ),
        #[cfg(feature = "cursor")]
        19 => drop(
            set.lower_bound_mut(Bound::Included(&T::mk(k)))
                .insert_before(T::mk(k2)),
        ),
        #[cfg(feature = "cursor")]
        20 => drop(
            set.lower_bound_mut(Bound::Included(&T::mk(k)))
                .remove_next(),
        ),
        #[cfg(feature = "cursor")]
        21 => drop(
            set.lower_bound_mut(Bound::Included(&T::mk(k)))
                .remove_prev(),
        ),
        _ => unreachable!("op is below OPS"),
    }
}

/// Applies `steps` random operations to `set`, then drops it.
///
/// A panic from the set does not end the run, so the set is exercised and
/// dropped after it has unwound.
fn run<T, C, G, const N: usize>(mut set: SkipSet<T, N, C, G>, seed: u32, steps: usize)
where
    T: Elem,
    C: Comparator<T> + ComparatorKey<T, T> + Clone,
    G: LevelGenerator + Clone + Default,
{
    let mut rng = rng_for(seed);
    for step_index in 0..steps {
        let op = rng.random_range(0..OPS);
        let k = rng.random_range(0..8);
        let k2 = rng.random_range(0..8);
        let index = rng.random_range(0..64);
        let mask = rng.random::<u32>();
        let result = catch_unwind(AssertUnwindSafe(|| step(&mut set, op, k, k2, index, mask)));
        check_step(seed, step_index, op, || {
            assert_contract_panic(result, &[]);
            set.assert_well_formed();
        });
    }
}

fn rng_for(seed: u32) -> SmallRng {
    SmallRng::seed_from_u64(u64::from(seed))
}

// MARK: Tests

#[test]
#[ignore = "issue 401: enabled once the fix lands"]
fn chaos_comparator_n4() {
    for seed in seeds() {
        let set: SkipSet<u8, 4, Chaos, FixedLevels> =
            SkipSet::with_comparator_and_level_generator(Chaos::new(seed), FixedLevels::new(seed));
        run(set, seed, 50);
    }
}

#[test]
#[ignore = "issue 401: enabled once the fix lands"]
fn chaos_comparator_n16() {
    for seed in seeds() {
        let set: SkipSet<u8, 16, Chaos, Geometric> =
            SkipSet::with_comparator_and_level_generator(Chaos::new(seed), seeded_geometric(seed));
        run(set, seed, 50);
    }
}

#[test]
#[ignore = "issue 401: enabled once the fix lands"]
fn chaos_ord() {
    for seed in seeds() {
        seed_chaos_ord(seed);
        let set: SkipSet<ChaosOrd, 16, OrdComparator, Geometric> =
            SkipSet::with_level_generator(seeded_geometric(seed));
        run(set, seed, 50);
    }
}

#[test]
#[ignore = "issue 401: enabled once the fix lands"]
fn rps_ord() {
    for seed in seeds() {
        let set: SkipSet<Rps, 16, OrdComparator, Geometric> =
            SkipSet::with_level_generator(seeded_geometric(seed));
        run(set, seed, 50);
    }
}
