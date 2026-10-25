//! Focused regression benchmarks for the library's hot paths.
//!
//! The per-type suites (`skip_list`, `ordered_skip_list`, ...) sweep sizes to
//! show asymptotic behaviour. This suite answers a narrower question: does a
//! change make the common operations faster or slower? It runs each workload
//! at a fixed size, n = 1000 (cache-resident), plus a handful at n = 100000
//! (memory-bound).
//!
//! Each workload is a steady-state step on a pre-built container: mutating
//! workloads pair an insert with the matching removal, so the container never
//! changes size and no setup runs inside the timed loop. Containers hold the
//! even keys `0, 2, ..., 2 * (n - 1)`; insertions use odd keys, so each lands
//! at a random position. The exceptions are the `list/*` workloads, which hold
//! `0..n` and address elements by index, and `osl/count_dups`, which holds each
//! value `i / DUPS` for `i` in `0..n`. Keys come from a fixed, seeded pool,
//! and every container uses a seeded level generator, so two builds of this
//! file see the same sequence of operations.
//!
//! The binary has three modes:
//!
//! - `hot [criterion args]`: run the workloads under Criterion.
//! - `hot list`: print `<workload> <n> <iters>` for every benchmark, where
//!   `iters` is the suggested step count for `opcount`.
//! - `hot opcount <workload> <n> <iters>`: build the container and run `iters`
//!   steps with no timing, for `perf stat`. The per-step cost is the difference
//!   between a run with `iters` steps and one with zero, divided by `iters`.
//!
//! Setting `HOT_OPT_IN` to a non-empty value adds the opt-in workloads to
//! `hot` and `hot list`. `hot opcount` finds them either way.

#![expect(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::integer_division,
    clippy::integer_division_remainder_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "Benchmark harness: plain index arithmetic, fixed pools, CLI output"
)]

use core::{hint::black_box, iter, time::Duration};
use std::env;

use criterion::Criterion;
use rand::{RngExt as _, SeedableRng, rngs::SmallRng, seq::SliceRandom as _};
use skiplist::{Geometric, OrderedSkipList, SkipList, SkipMap, SkipSet, skip_set};

/// Size used by every workload.
const SMALL: usize = 1_000;

/// Size for the memory-bound subset.
const LARGE: usize = 100_000;

/// Number of keys in the pool; a power of two so the cycle is a mask.
const POOL: usize = 1 << 10;

// MARK: Workloads

/// One benchmarked operation on a pre-built container of `n` elements.
trait Workload {
    /// Benchmark name, `<type>/<operation>`.
    const NAME: &'static str;

    /// Build the container.
    fn setup(n: usize) -> Self;

    /// Run one step. `key` is uniform in `0..n`.
    fn step(&mut self, key: u64);
}

/// Seeded level generator shared by all containers.
fn generator() -> Geometric {
    Geometric::new_with_seed(16, 0.5, 42).expect("valid parameters")
}

/// `SkipList` of `0..n`.
fn skip_list(n: usize) -> SkipList<u64> {
    let mut list = SkipList::with_level_generator(generator());
    for i in 0..n {
        list.push_back(i as u64);
    }
    list
}

/// `OrderedSkipList` of the even keys below `2 * n`.
fn ordered_skip_list(n: usize) -> OrderedSkipList<u64> {
    let mut list = OrderedSkipList::with_level_generator(generator());
    for i in 0..n {
        list.insert(2 * i as u64);
    }
    list
}

/// `SkipSet` of the even keys below `2 * n`.
fn skip_set(n: usize) -> SkipSet<u64> {
    let mut set = SkipSet::with_level_generator(generator());
    for i in 0..n {
        set.insert(2 * i as u64);
    }
    set
}

/// `SkipMap` from the even keys below `2 * n` to their halves.
fn skip_map(n: usize) -> SkipMap<u64, u64> {
    let mut map = SkipMap::with_level_generator(generator());
    for i in 0..n {
        map.insert(2 * i as u64, i as u64);
    }
    map
}

/// `SkipList::get` by index.
struct ListGet(SkipList<u64>);

impl Workload for ListGet {
    const NAME: &'static str = "list/get";

    fn setup(n: usize) -> Self {
        Self(skip_list(n))
    }

    fn step(&mut self, key: u64) {
        black_box(self.0.get(key as usize));
    }
}

/// `SkipList::insert` at an index, then `remove` at the same index.
struct ListInsertRemove(SkipList<u64>);

impl Workload for ListInsertRemove {
    const NAME: &'static str = "list/insert_remove";

    fn setup(n: usize) -> Self {
        Self(skip_list(n))
    }

    fn step(&mut self, key: u64) {
        self.0.insert(key as usize, key);
        black_box(self.0.remove(key as usize));
    }
}

/// `SkipList::push_back`, then `pop_front`: a queue.
struct ListQueue(SkipList<u64>);

impl Workload for ListQueue {
    const NAME: &'static str = "list/queue";

    fn setup(n: usize) -> Self {
        Self(skip_list(n))
    }

    fn step(&mut self, key: u64) {
        self.0.push_back(key);
        black_box(self.0.pop_front());
    }
}

/// `SkipList::push_back`, then `pop_back`: a stack.
struct ListStack(SkipList<u64>);

impl Workload for ListStack {
    const NAME: &'static str = "list/stack";

    fn setup(n: usize) -> Self {
        Self(skip_list(n))
    }

    fn step(&mut self, key: u64) {
        self.0.push_back(key);
        black_box(self.0.pop_back());
    }
}

/// Full `SkipList` iteration.
struct ListIter(SkipList<u64>);

impl Workload for ListIter {
    const NAME: &'static str = "list/iter";

    fn setup(n: usize) -> Self {
        Self(skip_list(n))
    }

    fn step(&mut self, _key: u64) {
        black_box(self.0.iter().fold(0_u64, |acc, v| acc.wrapping_add(*v)));
    }
}

/// `SkipList::clone`, then drop the copy.
struct ListClone(SkipList<u64>);

impl Workload for ListClone {
    const NAME: &'static str = "list/clone";

    fn setup(n: usize) -> Self {
        Self(skip_list(n))
    }

    fn step(&mut self, _key: u64) {
        black_box(self.0.clone());
    }
}

/// `SkipList::split_off` at a random index, then `append` the tail back.
struct ListSplitAppend(SkipList<u64>);

impl Workload for ListSplitAppend {
    const NAME: &'static str = "list/split_append";

    fn setup(n: usize) -> Self {
        Self(skip_list(n))
    }

    fn step(&mut self, key: u64) {
        let mut tail = self.0.split_off(key as usize);
        self.0.append(black_box(&mut tail));
    }
}

/// `OrderedSkipList::insert` of an absent key, then `take`.
struct OslInsertTake(OrderedSkipList<u64>);

impl Workload for OslInsertTake {
    const NAME: &'static str = "osl/insert_take";

    fn setup(n: usize) -> Self {
        Self(ordered_skip_list(n))
    }

    fn step(&mut self, key: u64) {
        let v = 2 * key + 1;
        self.0.insert(v);
        black_box(self.0.take(&v));
    }
}

/// `OrderedSkipList::insert` of an absent key, then `take_last`.
struct OslInsertTakeLast(OrderedSkipList<u64>);

impl Workload for OslInsertTakeLast {
    const NAME: &'static str = "osl/insert_take_last";

    fn setup(n: usize) -> Self {
        Self(ordered_skip_list(n))
    }

    fn step(&mut self, key: u64) {
        let v = 2 * key + 1;
        self.0.insert(v);
        black_box(self.0.take_last(&v));
    }
}

/// `OrderedSkipList::contains` of a present key.
struct OslContains(OrderedSkipList<u64>);

impl Workload for OslContains {
    const NAME: &'static str = "osl/contains";

    fn setup(n: usize) -> Self {
        Self(ordered_skip_list(n))
    }

    fn step(&mut self, key: u64) {
        black_box(self.0.contains(&(2 * key)));
    }
}

/// `OrderedSkipList::rank` of a present key.
struct OslRank(OrderedSkipList<u64>);

impl Workload for OslRank {
    const NAME: &'static str = "osl/rank";

    fn setup(n: usize) -> Self {
        Self(ordered_skip_list(n))
    }

    fn step(&mut self, key: u64) {
        black_box(self.0.rank(&(2 * key)));
    }
}

/// `OrderedSkipList::get_by_index`.
struct OslGetByIndex(OrderedSkipList<u64>);

impl Workload for OslGetByIndex {
    const NAME: &'static str = "osl/get_by_index";

    fn setup(n: usize) -> Self {
        Self(ordered_skip_list(n))
    }

    fn step(&mut self, key: u64) {
        black_box(self.0.get_by_index(key as usize));
    }
}

/// Number of copies of each value in `osl/count_dups`.
const DUPS: usize = 100;

/// `OrderedSkipList::count` of a value present `DUPS` times.
struct OslCountDups(OrderedSkipList<u64>);

impl Workload for OslCountDups {
    const NAME: &'static str = "osl/count_dups";

    fn setup(n: usize) -> Self {
        let mut list = OrderedSkipList::with_level_generator(generator());
        for i in 0..n {
            list.insert((i / DUPS) as u64);
        }
        Self(list)
    }

    fn step(&mut self, key: u64) {
        black_box(self.0.count(&(key / DUPS as u64)));
    }
}

/// `SkipSet::insert` of an absent key, then `remove`.
struct SetInsertRemove(SkipSet<u64>);

impl Workload for SetInsertRemove {
    const NAME: &'static str = "set/insert_remove";

    fn setup(n: usize) -> Self {
        Self(skip_set(n))
    }

    fn step(&mut self, key: u64) {
        let v = 2 * key + 1;
        black_box(self.0.insert(v));
        black_box(self.0.remove(&v));
    }
}

/// `SkipSet::insert` of a present key, which leaves the set unchanged.
struct SetInsertPresent(SkipSet<u64>);

impl Workload for SetInsertPresent {
    const NAME: &'static str = "set/insert_present";

    fn setup(n: usize) -> Self {
        Self(skip_set(n))
    }

    fn step(&mut self, key: u64) {
        black_box(self.0.insert(2 * key));
    }
}

/// `SkipSet::replace` of a present key.
struct SetReplace(SkipSet<u64>);

impl Workload for SetReplace {
    const NAME: &'static str = "set/replace";

    fn setup(n: usize) -> Self {
        Self(skip_set(n))
    }

    fn step(&mut self, key: u64) {
        black_box(self.0.replace(2 * key));
    }
}

/// `SkipSet::entry` of a present key, reading the element.
struct SetEntryHit(SkipSet<u64>);

impl Workload for SetEntryHit {
    const NAME: &'static str = "set/entry_hit";

    fn setup(n: usize) -> Self {
        Self(skip_set(n))
    }

    fn step(&mut self, key: u64) {
        match self.0.entry(2 * key) {
            skip_set::Entry::Occupied(e) => {
                black_box(e.get());
            }
            skip_set::Entry::Vacant(_) => panic!("key is present"),
        }
    }
}

/// `SkipSet::contains` of a present key.
struct SetContains(SkipSet<u64>);

impl Workload for SetContains {
    const NAME: &'static str = "set/contains";

    fn setup(n: usize) -> Self {
        Self(skip_set(n))
    }

    fn step(&mut self, key: u64) {
        black_box(self.0.contains(&(2 * key)));
    }
}

/// `SkipSet<String>::contains` of a present key, for comparator-heavy keys.
struct SetContainsString {
    /// Set of the even keys, zero-padded so string order matches numeric order.
    set: SkipSet<String>,
    /// `queries[k]` is the string for key `2 * k`.
    queries: Vec<String>,
}

impl Workload for SetContainsString {
    const NAME: &'static str = "set/contains_string";

    fn setup(n: usize) -> Self {
        let queries: Vec<String> = (0..n).map(|i| format!("key-{:08}", 2 * i)).collect();
        let mut set = SkipSet::with_level_generator(generator());
        for q in &queries {
            set.insert(q.clone());
        }
        Self { set, queries }
    }

    fn step(&mut self, key: u64) {
        black_box(self.set.contains(&self.queries[key as usize]));
    }
}

/// `SkipSet<String>::insert` of an absent key, then `remove`.
struct SetInsertRemoveString {
    /// Set of the even keys, as in `set/contains_string`.
    set: SkipSet<String>,
    /// `absent[k]` is the string for key `2 * k + 1`.
    absent: Vec<String>,
}

impl Workload for SetInsertRemoveString {
    const NAME: &'static str = "set/insert_remove_string";

    fn setup(n: usize) -> Self {
        let mut set = SkipSet::with_level_generator(generator());
        for i in 0..n {
            set.insert(format!("key-{:08}", 2 * i));
        }
        let absent = (0..n).map(|i| format!("key-{:08}", 2 * i + 1)).collect();
        Self { set, absent }
    }

    fn step(&mut self, key: u64) {
        let v = &self.absent[key as usize];
        black_box(self.set.insert(v.clone()));
        black_box(self.set.remove(v));
    }
}

/// A 264-byte value ordered by its first field, `key`.
///
/// `repr(C)` keeps `key` at offset 0, so a comparison reads the value's first
/// cache line. As the node's `Option` the value takes 272 bytes, more than
/// four cache lines.
#[repr(C)]
#[derive(Clone)]
struct Large {
    /// Sort key.
    key: u64,
    /// Filler; never compared.
    payload: [u64; 32],
}

const _: () = assert!(
    core::mem::size_of::<Large>() == 264,
    "Large must stay 264 bytes"
);

impl Large {
    /// The value with sort key `key`.
    fn new(key: u64) -> Self {
        Self {
            key,
            payload: [key; 32],
        }
    }
}

impl PartialEq for Large {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Large {}

impl PartialOrd for Large {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Large {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

/// `SkipSet<Large>` of the even keys.
fn large_set(n: usize) -> SkipSet<Large> {
    let mut set = SkipSet::with_level_generator(generator());
    for i in 0..n as u64 {
        set.insert(Large::new(2 * i));
    }
    set
}

/// `SkipSet<Large>::contains` of a present key (opt-in).
struct SetContainsLarge {
    /// Set of the even keys.
    set: SkipSet<Large>,
    /// `queries[k]` is the value with key `2 * k`.
    queries: Vec<Large>,
}

impl Workload for SetContainsLarge {
    const NAME: &'static str = "set/contains_large";

    fn setup(n: usize) -> Self {
        let queries = (0..n as u64).map(|i| Large::new(2 * i)).collect();
        Self {
            set: large_set(n),
            queries,
        }
    }

    fn step(&mut self, key: u64) {
        black_box(self.set.contains(&self.queries[key as usize]));
    }
}

/// `SkipSet<Large>::insert` of an absent key, then `remove` (opt-in).
struct SetInsertRemoveLarge {
    /// Set of the even keys.
    set: SkipSet<Large>,
    /// `absent[k]` is the value with key `2 * k + 1`.
    absent: Vec<Large>,
}

impl Workload for SetInsertRemoveLarge {
    const NAME: &'static str = "set/insert_remove_large";

    fn setup(n: usize) -> Self {
        let absent = (0..n as u64).map(|i| Large::new(2 * i + 1)).collect();
        Self {
            set: large_set(n),
            absent,
        }
    }

    fn step(&mut self, key: u64) {
        let v = &self.absent[key as usize];
        black_box(self.set.insert(v.clone()));
        black_box(self.set.remove(v));
    }
}

/// `SkipSet::range(k..)`, then the first element: a successor query.
struct SetRangeNext(SkipSet<u64>);

impl Workload for SetRangeNext {
    const NAME: &'static str = "set/range_next";

    fn setup(n: usize) -> Self {
        Self(skip_set(n))
    }

    fn step(&mut self, key: u64) {
        black_box(self.0.range(2 * key + 1..).next());
    }
}

/// `SkipSet::split_off` at a random key, then `append` the tail back.
struct SetSplitAppend(SkipSet<u64>);

impl Workload for SetSplitAppend {
    const NAME: &'static str = "set/split_append";

    fn setup(n: usize) -> Self {
        Self(skip_set(n))
    }

    fn step(&mut self, key: u64) {
        let mut tail = self.0.split_off(&(2 * key + 1));
        self.0.append(black_box(&mut tail));
    }
}

/// `SkipSet` built from `n` sorted keys, then dropped.
struct SetFromIter(usize);

impl Workload for SetFromIter {
    const NAME: &'static str = "set/from_iter";

    fn setup(n: usize) -> Self {
        Self(n)
    }

    fn step(&mut self, _key: u64) {
        let set: SkipSet<u64> = (0..self.0 as u64).collect();
        black_box(set);
    }
}

/// `SkipSet` built from `n` shuffled keys, then dropped.
struct SetFromIterShuffled(Vec<u64>);

impl Workload for SetFromIterShuffled {
    const NAME: &'static str = "set/from_iter_shuffled";

    fn setup(n: usize) -> Self {
        let mut keys: Vec<u64> = (0..n as u64).collect();
        keys.shuffle(&mut SmallRng::seed_from_u64(2));
        Self(keys)
    }

    fn step(&mut self, _key: u64) {
        let set: SkipSet<u64> = self.0.iter().copied().collect();
        black_box(set);
    }
}

/// `SkipSet::clone`, then drop the copy.
struct SetClone(SkipSet<u64>);

impl Workload for SetClone {
    const NAME: &'static str = "set/clone";

    fn setup(n: usize) -> Self {
        Self(skip_set(n))
    }

    fn step(&mut self, _key: u64) {
        black_box(self.0.clone());
    }
}

/// `SkipSet::clone`, then consume the copy with `into_iter`. Subtract
/// `set/clone` for the cost of `into_iter` alone.
struct SetIntoIter(SkipSet<u64>);

impl Workload for SetIntoIter {
    const NAME: &'static str = "set/into_iter";

    fn setup(n: usize) -> Self {
        Self(skip_set(n))
    }

    fn step(&mut self, _key: u64) {
        black_box(self.0.clone().into_iter().fold(0_u64, u64::wrapping_add));
    }
}

/// `&a | &b` for two interleaved sets of `n / 2` keys each.
struct SetBitOr(SkipSet<u64>, SkipSet<u64>);

impl Workload for SetBitOr {
    const NAME: &'static str = "set/bitor";

    fn setup(n: usize) -> Self {
        let set = skip_set(n);
        let (a, b): (Vec<u64>, Vec<u64>) = set.iter().partition(|&&v| v % 4 == 0);
        Self(a.into_iter().collect(), b.into_iter().collect())
    }

    fn step(&mut self, _key: u64) {
        black_box(&self.0 | &self.1);
    }
}

/// `SkipMap::insert` of an absent key, then `remove`.
struct MapInsertRemove(SkipMap<u64, u64>);

impl Workload for MapInsertRemove {
    const NAME: &'static str = "map/insert_remove";

    fn setup(n: usize) -> Self {
        Self(skip_map(n))
    }

    fn step(&mut self, key: u64) {
        let k = 2 * key + 1;
        black_box(self.0.insert(k, key));
        black_box(self.0.remove(&k));
    }
}

/// `SkipMap::insert` of a present key, which replaces the value.
struct MapInsertReplace(SkipMap<u64, u64>);

impl Workload for MapInsertReplace {
    const NAME: &'static str = "map/insert_replace";

    fn setup(n: usize) -> Self {
        Self(skip_map(n))
    }

    fn step(&mut self, key: u64) {
        black_box(self.0.insert(2 * key, key));
    }
}

/// `SkipMap::get` of a present key.
struct MapGet(SkipMap<u64, u64>);

impl Workload for MapGet {
    const NAME: &'static str = "map/get";

    fn setup(n: usize) -> Self {
        Self(skip_map(n))
    }

    fn step(&mut self, key: u64) {
        black_box(self.0.get(&(2 * key)));
    }
}

/// `SkipMap::get_mut` of a present key, updating the value in place.
struct MapGetMut(SkipMap<u64, u64>);

impl Workload for MapGetMut {
    const NAME: &'static str = "map/get_mut";

    fn setup(n: usize) -> Self {
        Self(skip_map(n))
    }

    fn step(&mut self, key: u64) {
        if let Some(v) = self.0.get_mut(&(2 * key)) {
            *v = v.wrapping_add(1);
        }
    }
}

/// `SkipMap::pop_last`, then re-insert the popped entry.
struct MapPopLast(SkipMap<u64, u64>);

impl Workload for MapPopLast {
    const NAME: &'static str = "map/pop_last";

    fn setup(n: usize) -> Self {
        Self(skip_map(n))
    }

    fn step(&mut self, _key: u64) {
        let (k, v) = self.0.pop_last().expect("map is not empty");
        self.0.insert(black_box(k), v);
    }
}

/// `SkipMap::entry` of a present key, updating the value in place.
struct MapEntryHit(SkipMap<u64, u64>);

impl Workload for MapEntryHit {
    const NAME: &'static str = "map/entry_hit";

    fn setup(n: usize) -> Self {
        Self(skip_map(n))
    }

    fn step(&mut self, key: u64) {
        let v = self.0.entry(2 * key).or_insert(0);
        *v = v.wrapping_add(1);
    }
}

/// `SkipMap::split_off` at a random key, then `append` the tail back.
struct MapSplitAppend(SkipMap<u64, u64>);

impl Workload for MapSplitAppend {
    const NAME: &'static str = "map/split_append";

    fn setup(n: usize) -> Self {
        Self(skip_map(n))
    }

    fn step(&mut self, key: u64) {
        let mut tail = self.0.split_off(&(2 * key + 1));
        self.0.append(black_box(&mut tail));
    }
}

/// Full `SkipMap` iteration.
struct MapIter(SkipMap<u64, u64>);

impl Workload for MapIter {
    const NAME: &'static str = "map/iter";

    fn setup(n: usize) -> Self {
        Self(skip_map(n))
    }

    fn step(&mut self, _key: u64) {
        black_box(
            self.0
                .iter()
                .fold(0_u64, |acc, (k, v)| acc.wrapping_add(*k ^ *v)),
        );
    }
}

// MARK: Registry

/// A workload and the sizes it runs at.
struct Entry {
    /// Benchmark name, `<type>/<operation>`.
    name: &'static str,
    /// Sizes to run at, each with the step count for `perf stat`.
    sizes: &'static [(usize, u64)],
    /// Criterion driver.
    bench: fn(&mut Criterion, usize),
    /// Untimed driver for `perf stat`.
    count: fn(usize, u64),
}

/// Register workload `W` at `sizes`.
const fn entry<W>(sizes: &'static [(usize, u64)]) -> Entry
where
    W: Workload,
{
    Entry {
        name: W::NAME,
        sizes,
        bench: bench::<W>,
        count: count::<W>,
    }
}

/// `perf stat` steps for an O(log n) operation.
const POINT: u64 = 1_000_000;

/// `perf stat` steps for an operation that walks the whole container.
const BULK: u64 = 10_000;

/// Reduced `perf stat` steps for workloads too slow to run `POINT` steps at
/// `LARGE`.
const SLOW: u64 = 1_000;

/// The default workloads, in run order.
const ENTRIES: &[Entry] = &[
    entry::<ListGet>(&[(SMALL, POINT), (LARGE, POINT)]),
    entry::<ListInsertRemove>(&[(SMALL, POINT)]),
    entry::<ListQueue>(&[(SMALL, POINT)]),
    entry::<ListStack>(&[(SMALL, POINT)]),
    entry::<ListIter>(&[(SMALL, BULK)]),
    entry::<ListClone>(&[(SMALL, BULK)]),
    entry::<ListSplitAppend>(&[(SMALL, BULK), (LARGE, SLOW)]),
    entry::<OslInsertTake>(&[(SMALL, POINT)]),
    entry::<OslInsertTakeLast>(&[(SMALL, POINT), (LARGE, SLOW)]),
    entry::<OslContains>(&[(SMALL, POINT), (LARGE, POINT)]),
    entry::<OslRank>(&[(SMALL, POINT)]),
    entry::<OslGetByIndex>(&[(SMALL, POINT)]),
    entry::<OslCountDups>(&[(SMALL, POINT), (LARGE, POINT)]),
    entry::<SetInsertRemove>(&[(SMALL, POINT), (LARGE, POINT)]),
    entry::<SetInsertPresent>(&[(SMALL, POINT)]),
    entry::<SetReplace>(&[(SMALL, POINT)]),
    entry::<SetEntryHit>(&[(SMALL, POINT)]),
    entry::<SetContains>(&[(SMALL, POINT)]),
    entry::<SetContainsString>(&[(SMALL, POINT)]),
    entry::<SetInsertRemoveString>(&[(SMALL, POINT)]),
    entry::<SetRangeNext>(&[(SMALL, POINT), (LARGE, SLOW)]),
    entry::<SetFromIter>(&[(SMALL, BULK)]),
    entry::<SetFromIterShuffled>(&[(SMALL, BULK)]),
    entry::<SetClone>(&[(SMALL, BULK)]),
    entry::<SetIntoIter>(&[(SMALL, BULK)]),
    entry::<SetBitOr>(&[(SMALL, BULK)]),
    entry::<SetSplitAppend>(&[(SMALL, BULK), (LARGE, SLOW)]),
    entry::<MapInsertRemove>(&[(SMALL, POINT), (LARGE, POINT)]),
    entry::<MapInsertReplace>(&[(SMALL, POINT)]),
    entry::<MapGet>(&[(SMALL, POINT), (LARGE, POINT)]),
    entry::<MapGetMut>(&[(SMALL, POINT)]),
    entry::<MapEntryHit>(&[(SMALL, POINT)]),
    entry::<MapPopLast>(&[(SMALL, POINT), (LARGE, SLOW)]),
    entry::<MapIter>(&[(SMALL, BULK)]),
    entry::<MapSplitAppend>(&[(SMALL, BULK), (LARGE, SLOW)]),
];

/// Workloads that run only when `HOT_OPT_IN` is non-empty, for questions the
/// default suite does not ask. `opcount` finds them either way.
const OPT_IN: &[Entry] = &[
    entry::<SetContainsLarge>(&[(SMALL, POINT), (LARGE, POINT)]),
    entry::<SetInsertRemoveLarge>(&[(SMALL, POINT), (LARGE, POINT)]),
];

/// The workloads to list and run: every default one, then the opt-in ones
/// when `HOT_OPT_IN` is non-empty.
fn entries() -> impl Iterator<Item = &'static Entry> {
    let opt_in: &[Entry] = if env::var_os("HOT_OPT_IN").is_some_and(|v| !v.is_empty()) {
        OPT_IN
    } else {
        &[]
    };
    ENTRIES.iter().chain(opt_in)
}

// MARK: Drivers

/// Seeded pool of keys uniform in `0..n`.
fn keys(n: usize) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(1);
    iter::repeat_with(|| rng.random_range(0..n as u64))
        .take(POOL)
        .collect()
}

/// Run `W` at size `n` under Criterion.
fn bench<W>(c: &mut Criterion, n: usize)
where
    W: Workload,
{
    let keys = keys(n);
    let mut workload = W::setup(n);
    let mut i = 0_usize;
    c.bench_function(&format!("{}/{n}", W::NAME), |b| {
        b.iter(|| {
            let key = keys[i & (POOL - 1)];
            i = i.wrapping_add(1);
            workload.step(black_box(key));
        });
    });
}

/// Run `iters` steps of `W` at size `n` with no timing.
fn count<W>(n: usize, iters: u64)
where
    W: Workload,
{
    let keys = keys(n);
    let mut workload = W::setup(n);
    for i in 0..iters {
        workload.step(black_box(keys[i as usize & (POOL - 1)]));
    }
    black_box(&mut workload);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("list") => {
            for e in entries() {
                for (n, iters) in e.sizes {
                    println!("{} {n} {iters}", e.name);
                }
            }
        }
        Some("opcount") => {
            let [_, _, name, n, iters] = args.as_slice() else {
                panic!("usage: hot opcount <workload> <n> <iters>");
            };
            let e = ENTRIES
                .iter()
                .chain(OPT_IN)
                .find(|e| e.name == name)
                .unwrap_or_else(|| panic!("unknown workload {name}"));
            (e.count)(
                n.parse().expect("n is an integer"),
                iters.parse().expect("iters is an integer"),
            );
        }
        _ => {
            let mut c = Criterion::default()
                .warm_up_time(Duration::from_secs(1))
                .measurement_time(Duration::from_secs(3))
                .without_plots()
                .configure_from_args();
            for e in entries() {
                for &(n, _) in e.sizes {
                    (e.bench)(&mut c, n);
                }
            }
            c.final_summary();
        }
    }
}
