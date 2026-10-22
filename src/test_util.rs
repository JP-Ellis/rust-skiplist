//! Test-only helpers shared across the container test modules.

#![expect(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::indexing_slicing,
    clippy::integer_division_remainder_used,
    reason = "the chaotic fixtures reduce random words to small bounded values"
)]

use core::cmp::Ordering;
use std::{cell::Cell, ops::Range, thread::Result as ThreadResult};

use crate::{
    Geometric,
    comparator::{Comparator, ComparatorKey},
    level_generator::LevelGenerator,
};

/// An element whose destructor panics when armed.
///
/// Used to check that a container stays internally consistent when an element
/// destructor unwinds while the container is midway through updating its own
/// bookkeeping.  Ordering and equality use `id` alone, so comparisons never
/// touch the armed flag and never panic.
#[derive(Debug)]
pub(crate) struct PanicOnDrop {
    id: u32,
    armed: bool,
}

impl PanicOnDrop {
    /// An element whose destructor runs normally.
    pub(crate) fn new(id: u32) -> Self {
        Self { id, armed: false }
    }

    /// An element whose destructor panics.
    pub(crate) fn armed(id: u32) -> Self {
        Self { id, armed: true }
    }

    pub(crate) fn id(&self) -> u32 {
        self.id
    }
}

impl Drop for PanicOnDrop {
    fn drop(&mut self) {
        assert!(!self.armed, "PanicOnDrop({}) destructor fired", self.id);
    }
}

impl PartialEq for PanicOnDrop {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for PanicOnDrop {}

impl PartialOrd for PanicOnDrop {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PanicOnDrop {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

/// Builds `count` elements with ids `0..count`, arming the one at `armed`.
///
/// An `armed` of `count` or above matches no id, which is how callers ask
/// for a run of elements whose destructors all pass.
pub(crate) fn bombs(count: u32, armed: u32) -> impl Iterator<Item = PanicOnDrop> {
    (0..count).map(move |i| {
        if i == armed {
            PanicOnDrop::armed(i)
        } else {
            PanicOnDrop::new(i)
        }
    })
}

// MARK: Chaotic comparators

/// The panic message a container raises when it detects a comparator that
/// answers inconsistently within one operation.
pub(crate) const TOTAL_ORDER_PANIC: &str = "comparator does not implement a total order";

/// Advances a linear congruential generator and picks an [`Ordering`] from it.
fn chaotic_ordering(state: u32) -> (u32, Ordering) {
    let next = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
    let ordering =
        [Ordering::Less, Ordering::Equal, Ordering::Greater][((next >> 16) % 3) as usize];
    (next, ordering)
}

/// A comparator whose every answer is pseudo-random.
///
/// The answer ignores both operands, so the same pair compares differently
/// from one call to the next.
#[derive(Debug, Clone)]
pub(crate) struct Chaos(Cell<u32>);

impl Chaos {
    pub(crate) fn new(seed: u32) -> Self {
        Self(Cell::new(seed))
    }
}

impl<T> Comparator<T> for Chaos {
    fn compare(&self, _a: &T, _b: &T) -> Ordering {
        let (next, ordering) = chaotic_ordering(self.0.get());
        self.0.set(next);
        ordering
    }
}

impl<T> ComparatorKey<T, T> for Chaos {
    fn compare_key(&self, stored: &T, key: &T) -> Ordering {
        self.compare(stored, key)
    }
}

/// A consistent `Ord` comparator that counts its calls in a `Cell`.
///
/// The counter lives inside the comparator, and so inside the container.
/// Every comparison therefore writes to the container's own memory through
/// a shared reference.
#[cfg(feature = "cursor")]
#[derive(Debug, Default, Clone)]
pub(crate) struct Counting(Cell<usize>);

#[cfg(feature = "cursor")]
impl Counting {
    /// Returns the number of comparisons made so far.
    pub(crate) fn calls(&self) -> usize {
        self.0.get()
    }
}

#[cfg(feature = "cursor")]
impl<T: Ord> Comparator<T> for Counting {
    fn compare(&self, a: &T, b: &T) -> Ordering {
        self.0.set(self.0.get() + 1);
        a.cmp(b)
    }
}

#[cfg(feature = "cursor")]
impl<T: Ord> ComparatorKey<T, T> for Counting {
    fn compare_key(&self, stored: &T, key: &T) -> Ordering {
        self.compare(stored, key)
    }
}

thread_local! {
    static CHAOS_ORD_STATE: Cell<u32> = const { Cell::new(0) };
}

/// Seeds the generator behind [`ChaosOrd`] for the current thread.
pub(crate) fn seed_chaos_ord(seed: u32) {
    CHAOS_ORD_STATE.with(|state| state.set(seed));
}

/// An element whose safe [`Ord`] implementation answers pseudo-randomly.
#[derive(Debug)]
pub(crate) struct ChaosOrd(pub u8);

impl PartialEq for ChaosOrd {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ChaosOrd {}

impl PartialOrd for ChaosOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ChaosOrd {
    fn cmp(&self, _other: &Self) -> Ordering {
        CHAOS_ORD_STATE.with(|state| {
            let (next, ordering) = chaotic_ordering(state.get());
            state.set(next);
            ordering
        })
    }
}

/// An element with a deterministic, non-transitive order.
///
/// Values behave like rock, paper and scissors: `x % 3` is greater than
/// `(x + 1) % 3`, so every value is greater than one value and less than
/// another.
#[derive(Debug)]
pub(crate) struct Rps(pub u8);

impl PartialEq for Rps {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Rps {}

impl PartialOrd for Rps {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rps {
    fn cmp(&self, other: &Self) -> Ordering {
        let (a, b) = (self.0 % 3, other.0 % 3);
        if a == b {
            Ordering::Equal
        } else if (a + 1) % 3 == b {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

/// A level generator with a fixed ceiling of four levels.
#[derive(Debug, Clone, Default)]
pub(crate) struct FixedLevels(u32);

impl FixedLevels {
    pub(crate) fn new(seed: u32) -> Self {
        Self(seed)
    }
}

impl LevelGenerator for FixedLevels {
    fn total(&self) -> usize {
        4
    }

    fn level(&mut self) -> usize {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 16) % 4) as usize
    }
}

// MARK: Harness support

/// An element type the chaos harnesses can build from, and read back to, a byte.
pub(crate) trait Elem {
    fn mk(x: u8) -> Self;
    fn raw(&self) -> u8;
}

impl Elem for u8 {
    fn mk(x: u8) -> Self {
        x
    }

    fn raw(&self) -> u8 {
        *self
    }
}

impl Elem for ChaosOrd {
    fn mk(x: u8) -> Self {
        Self(x)
    }

    fn raw(&self) -> u8 {
        self.0
    }
}

impl Elem for Rps {
    fn mk(x: u8) -> Self {
        Self(x)
    }

    fn raw(&self) -> u8 {
        self.0
    }
}

/// Accepts a completed operation, or a panic that is part of the contract.
///
/// A panic passes when its payload equals [`TOTAL_ORDER_PANIC`] or one of
/// `also_allowed`.  Any other payload fails the test.
pub(crate) fn assert_contract_panic(result: ThreadResult<()>, also_allowed: &[&str]) {
    let Err(payload) = result else {
        return;
    };
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>");
    assert!(
        message == TOTAL_ORDER_PANIC || also_allowed.contains(&message),
        "unexpected panic: {message}"
    );
}

/// The seeds each chaos test iterates over.
pub(crate) fn seeds() -> Range<u32> {
    if cfg!(miri) { 0..20 } else { 0..2000 }
}

/// A [`Geometric`] generator with the default shape and a fixed seed.
pub(crate) fn seeded_geometric(seed: u32) -> Geometric {
    Geometric::new_with_seed(16, 0.5, u64::from(seed))
        .expect("16 levels and q = 0.5 are valid Geometric parameters")
}

/// Runs a post-operation check, naming the seed, step and operation if it panics.
pub(crate) fn check_step(seed: u32, step: usize, op: u32, check: impl FnOnce()) {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(check)) {
        eprintln!("chaos failure: seed {seed}, step {step}, op {op}");
        std::panic::resume_unwind(payload);
    }
}
