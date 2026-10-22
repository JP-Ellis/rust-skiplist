//! Randomised operation sequences on [`SkipMap`] under comparators that answer
//! inconsistently.
//!
//! See the matching module for `OrderedSkipList` for the contract being checked.

#![expect(
    clippy::integer_division_remainder_used,
    clippy::let_underscore_must_use,
    clippy::let_underscore_untyped,
    reason = "the harness reduces random words to small bounded values and discards results"
)]

#[cfg(feature = "cursor")]
use core::ops::Bound;
use std::panic::{AssertUnwindSafe, catch_unwind};

use rand::{RngExt as _, SeedableRng as _, rngs::SmallRng};

use crate::{
    Geometric, SkipMap,
    comparator::{Comparator, ComparatorKey, OrdComparator},
    level_generator::LevelGenerator,
    skip_map::{Entry, OccupiedEntry},
    test_util::{
        Chaos, ChaosOrd, Elem, FixedLevels, Rps, assert_contract_panic, check_step, seed_chaos_ord,
        seeded_geometric, seeds,
    },
};

/// The number of operations `step` can pick from.
const OPS: u32 = if cfg!(feature = "cursor") { 21 } else { 17 };

// MARK: Driver

/// Runs one randomly chosen operation on `map`.
fn step<K, C, G, const N: usize>(
    map: &mut SkipMap<K, u8, N, C, G>,
    op: u32,
    k: u8,
    k2: u8,
    mask: u32,
) where
    K: Elem,
    C: Comparator<K> + ComparatorKey<K, K> + Clone,
    G: LevelGenerator + Clone + Default,
{
    let pred = |key: &K| (mask >> (key.raw() % 32)) & 1 == 1;
    match op {
        0 => map.extract_if(|key, _| pred(key)).for_each(drop),
        1 => drop(map.insert(K::mk(k), k2)),
        2 => drop(map.remove(&K::mk(k))),
        3 => drop(map.remove_entry(&K::mk(k))),
        4 => {
            if let Some(v) = map.get_mut(&K::mk(k)) {
                *v = v.wrapping_add(1);
            }
        }
        5 => {
            let _ = map.entry(K::mk(k)).or_insert(k2);
        }
        6 => {
            if let Entry::Occupied(e) = map.entry(K::mk(k)) {
                drop(e.remove_entry());
            }
        }
        7 => drop(map.pop_first()),
        8 => drop(map.pop_last()),
        9 => {
            let mut half = map.split_off(&K::mk(k));
            map.append(&mut half);
        }
        10 => {
            let mut other = SkipMap::with_comparator_and_level_generator(
                map.comparator().clone(),
                G::default(),
            );
            other.extend([(K::mk(k), 0), (K::mk(k.wrapping_add(1)), 1), (K::mk(k2), 2)]);
            map.append(&mut other);
        }
        11 => map.retain(|key, _| pred(key)),
        12 => map.extend([(K::mk(k), 0), (K::mk(k2), 1), (K::mk(k.wrapping_add(1)), 2)]),
        13 => {
            if let Entry::Occupied(e) = map.entry(K::mk(k)) {
                let _ = e.remove();
            }
        }
        14 => {
            if let Entry::Vacant(e) = map.entry(K::mk(k)) {
                let _ = e.insert(k2);
            }
        }
        15 => drop(map.first_entry().map(OccupiedEntry::remove_entry)),
        16 => drop(map.last_entry().map(OccupiedEntry::remove_entry)),
        #[cfg(feature = "cursor")]
        17 => drop(
            map.lower_bound_mut(Bound::Included(&K::mk(k)))
                .insert_after(K::mk(k2), 0),
        ),
        #[cfg(feature = "cursor")]
        18 => drop(
            map.lower_bound_mut(Bound::Included(&K::mk(k)))
                .insert_before(K::mk(k2), 0),
        ),
        #[cfg(feature = "cursor")]
        19 => drop(
            map.lower_bound_mut(Bound::Included(&K::mk(k)))
                .remove_next(),
        ),
        #[cfg(feature = "cursor")]
        20 => drop(
            map.lower_bound_mut(Bound::Included(&K::mk(k)))
                .remove_prev(),
        ),
        _ => unreachable!("op is below OPS"),
    }
}

/// Applies `steps` random operations to `map`, then drops it.
///
/// A panic from the map does not end the run, so the map is exercised and
/// dropped after it has unwound.
fn run<K, C, G, const N: usize>(mut map: SkipMap<K, u8, N, C, G>, seed: u32, steps: usize)
where
    K: Elem,
    C: Comparator<K> + ComparatorKey<K, K> + Clone,
    G: LevelGenerator + Clone + Default,
{
    let mut rng = rng_for(seed);
    for step_index in 0..steps {
        let op = rng.random_range(0..OPS);
        let k = rng.random_range(0..8);
        let k2 = rng.random_range(0..8);
        let mask = rng.random::<u32>();
        let result = catch_unwind(AssertUnwindSafe(|| step(&mut map, op, k, k2, mask)));
        check_step(seed, step_index, op, || {
            assert_contract_panic(result, &[]);
            map.assert_well_formed();
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
        let map: SkipMap<u8, u8, 4, Chaos, FixedLevels> =
            SkipMap::with_comparator_and_level_generator(Chaos::new(seed), FixedLevels::new(seed));
        run(map, seed, 50);
    }
}

#[test]
fn chaos_comparator_n16() {
    for seed in seeds() {
        let map: SkipMap<u8, u8, 16, Chaos, Geometric> =
            SkipMap::with_comparator_and_level_generator(Chaos::new(seed), seeded_geometric(seed));
        run(map, seed, 50);
    }
}

#[test]
fn chaos_ord() {
    for seed in seeds() {
        seed_chaos_ord(seed);
        let map: SkipMap<ChaosOrd, u8, 16, OrdComparator, Geometric> =
            SkipMap::with_level_generator(seeded_geometric(seed));
        run(map, seed, 50);
    }
}

#[test]
fn rps_ord() {
    for seed in seeds() {
        let map: SkipMap<Rps, u8, 16, OrdComparator, Geometric> =
            SkipMap::with_level_generator(seeded_geometric(seed));
        run(map, seed, 50);
    }
}
