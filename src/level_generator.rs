//! SkipLists use a probabilistic distribution of nodes over the internal
//! levels, whereby the lowest level (level 0) contains all the nodes, and each
//! level `n > 0` will contain a random subset of the nodes on level `n - 1`.
//!
//! Most commonly, a geometric distribution is used whereby the chance that a
//! node occupies level `n` is `p` times the chance of occupying level `n-1`
//! (with `0 < p < 1`).
//!
//! It is very unlikely that this will need to be changed as the default should
//! suffice, but if need be custom level generators can be implemented.

use rand::distributions::{self, Sample};

// ////////////////////////////////////////////////////////////////////////////
// Level Generator
// ////////////////////////////////////////////////////////////////////////////

/// Upon the insertion of a new node in the list, the node is replicated to high
/// levels with a certain probability as determined by a `LevelGenerator`.
pub trait LevelGenerator {
    /// The total number of levels that are assumed to exist for this level
    /// generator.
    fn total(&self) -> usize;
    /// Generate a random level for a new node in the range `[0, total)`.
    ///
    /// This must never return a level that is `>= self.total()`.
    fn random(&mut self) -> usize;
}

/// A level generator which will produce geometrically distributed numbers.
///
/// The probability of generating level `n` is `p` times the probability of
/// generating level `n-1`, with the probability truncated at the maximum number
/// of levels allowed.
pub struct GeometricalLevelGenerator {
    total: usize,
    p: f64,
    unit_range: distributions::Range<f64>,
    rng: rand::XorShiftRng, // Fast generator
}

impl GeometricalLevelGenerator {
    /// Create a new GeometricalLevelGenerator with `total` number of levels,
    /// and `p` as the probability that a given node is present in the next
    /// level.
    ///
    /// # Panics
    ///
    /// `p` must be between 0 and 1 and will panic otherwise.  Similarly,
    /// `total` must be at greater or equal to 1.
    pub fn new(total: usize, p: f64) -> Self {
        if total == 0 {
            panic!("total must be non-zero.");
        }
        if p <= 0.0 || p >= 1.0 {
            panic!("p must be in (0, 1).");
        }
        GeometricalLevelGenerator {
            total,
            p,
            unit_range: distributions::Range::new(0.0f64, 1.0),
            rng: rand::XorShiftRng::new_unseeded(),
        }
    }
}

impl LevelGenerator for GeometricalLevelGenerator {
    fn random(&mut self) -> usize {
        let mut h = 0;
        let mut x = self.p;
        let f = 1.0 - self.unit_range.sample(&mut self.rng);
        while x > f && h + 1 < self.total {
            h += 1;
            x *= self.p
        }
        h
    }

    fn total(&self) -> usize {
        self.total
    }
}
