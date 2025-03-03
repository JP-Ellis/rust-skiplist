//! Skiplists use a probabilistic distribution of nodes over the internal
//! levels, whereby the lowest level (level 0) contains all the nodes, and each
//! level $n > 0$ will contain a random subset of the nodes on level `n - 1`.
//!
//! Most commonly, a geometric distribution is used whereby the chance that a
//! node occupies level $n$ is $p$ times the chance of occupying level $n-1$
//! (with $0 < p < 1$).
//!
//! It is very unlikely that this will need to be changed as the default should
//! suffice, but if need be custom level generators can be implemented.

use anyhow::Result;
use rand::prelude::*;
use thiserror::Error;

// ////////////////////////////////////////////////////////////////////////////
// Level Generator
// ////////////////////////////////////////////////////////////////////////////

/// Upon the insertion of a new node in the list, the node is replicated to high
/// levels with a certain probability as determined by a [`LevelGenerator`].
pub trait LevelGenerator {
    /// The total number of levels that are assumed to exist.
    #[must_use]
    fn total(&self) -> usize;
    /// Generate a random level for a new node in the range `[0, total)`.
    ///
    /// This function should _never_ return a level greater or equal to
    /// [`total`][LevelGenerator::total].
    #[must_use]
    fn random(&mut self) -> usize;
}

/// A level generator using a geometric distribution.
///
/// With a geometric distribution, the probability that a node is present in
/// level $n$ is $p^n$ (with $0 < p < 1$).  The probability is truncated at the
/// maximum number of levels allowed.
#[derive(Debug)]
pub struct Geometric {
    /// The total number of levels that are assumed to exist.
    total: usize,
    /// The probability that a node is present in the next level.
    p: f64,
    /// The random number generator.
    rng: SmallRng,
}

impl Geometric {
    /// Create a new geometric level generator with `total` number of levels,
    /// and `p` as the probability that a given node is present in the next
    /// level.
    ///
    /// # Errors
    ///
    /// `p` must be between 0 and 1 and will panic otherwise.  Similarly, `max`
    /// must be at greater or equal to 1.
    #[must_use = "Level generator must be used"]
    #[inline]
    pub fn new(max: usize, p: f64) -> Result<Self, GeometricNewError> {
        if max == 0 {
            return Err(GeometricNewError::ZeroMax);
        }
        if !(0.0 < p && p < 1.0) {
            return Err(GeometricNewError::InvalidProbability);
        }
        Ok(Geometric {
            total: max,
            p,
            rng: SmallRng::from_rng(thread_rng()).map_err(|_| GeometricNewError::RngInitFailed)?,
        })
    }
}

#[derive(Error, Debug)]
/// Errors that can occur when creating a [`Geometric`] level generator.
pub enum GeometricNewError {
    /// The maximum number of levels must be non-zero.
    #[error("max must be non-zero.")]
    ZeroMax,
    /// The probability $p$ must be in the range $(0, 1)$.
    #[error("p must be in (0, 1).")]
    InvalidProbability,
    /// Failed to initialize the random number generator.
    #[error("Failed to initialize the random number generator.")]
    RngInitFailed,
}

impl LevelGenerator for Geometric {
    #[inline]
    fn random(&mut self) -> usize {
        let mut h = 0;
        let mut x = self.p;
        let f = 1.0 - self.rng.r#gen::<f64>();
        while x > f && h + 1 < self.total {
            h += 1;
            x *= self.p;
        }
        h
    }

    #[inline]
    fn total(&self) -> usize {
        self.total
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::{Geometric, LevelGenerator};

    #[test]
    fn invalid_max() -> Result<()> {
        assert_eq!(
            Geometric::new(0, 0.5).unwrap_err().to_string(),
            "max must be non-zero."
        );
        Ok(())
    }

    #[test]
    fn invalid_p_0() -> Result<()> {
        assert_eq!(
            Geometric::new(1, 0.0).unwrap_err().to_string(),
            "p must be in (0, 1)."
        );
        Ok(())
    }

    #[test]
    fn invalid_p_1() -> Result<()> {
        assert_eq!(
            Geometric::new(1, 1.0).unwrap_err().to_string(),
            "p must be in (0, 1)."
        );
        Ok(())
    }

    #[test]
    fn new() -> Result<()> {
        let generator = Geometric::new(1, 0.5)?;
        assert_eq!(generator.total(), 1);
        Ok(())
    }
}
