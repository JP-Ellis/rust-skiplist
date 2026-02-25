//! Geometric level generator.

#![expect(clippy::float_arithmetic, reason = "computing probabilities")]

use rand::prelude::*;
use thiserror::Error;

use crate::level_generator::LevelGenerator;

#[derive(Error, Debug, PartialEq, Eq)]
/// Errors that can occur when creating a [`Geometric`] level generator.
#[expect(
    clippy::module_name_repetitions,
    reason = "Using 'Error' would be too generic and may cause confusion."
)]
#[non_exhaustive]
pub enum GeometricError {
    /// The maximum number of levels must be non-zero.
    #[error("max must be non-zero.")]
    ZeroMax,
    /// The maximum number of levels must be less than `i32::MAX`.
    #[error("max must be less than i32::MAX.")]
    MaxTooLarge,
    /// The probability `$p$` must be in the range `$(0, 1)$`.
    #[error("p must be in (0, 1).")]
    InvalidProbability,
    /// Failed to initialize the random number generator.
    #[error("Failed to initialize the random number generator.")]
    RngInitFailed,
}

/// A level generator using a geometric distribution.
///
/// This distribution assumes that if a node is present at some level `$n$`,
/// then the probability that it is present at level `$n+1$` is some constant
/// `$p \in (0, 1)$`. This produces a geometric distribution, albeit truncated
/// at the maximum number of levels allowed.
#[derive(Debug)]
pub struct Geometric {
    /// The total number of levels that are assumed to exist.
    total: usize,
    /// The total number of levels as an `i32` for use in the CDF computation.
    total_i32: i32,
    /// The probability that a node is not present in the next level.
    ///
    /// While the geometric distribution is defined using the probability `$p$`,
    /// the computations needed rely on `$q = 1 - p$`.
    q: f64,
    /// The random number generator.
    rng: SmallRng,
}

impl Geometric {
    /// Creates a new geometric level generator.
    ///
    /// `total` sets the maximum number of levels (must be at least 1 and less
    /// than `i32::MAX`). `p` is the probability that an element is promoted to
    /// the next level and must be strictly between 0 and 1.
    ///
    /// # Errors
    ///
    /// Returns a [`GeometricError`] if `total` is zero, if `total` exceeds
    /// `i32::MAX`, or if `p` is not in the open interval `$(0, 1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::level_generator::LevelGenerator;
    /// use skiplist::level_generator::geometric::{Geometric, GeometricError};
    ///
    /// // Valid configuration: 16 levels, promotion probability 0.5.
    /// let mut generator = Geometric::new(16, 0.5).unwrap();
    /// assert_eq!(generator.total(), 16);
    ///
    /// // Invalid: zero levels.
    /// assert_eq!(Geometric::new(0, 0.5).unwrap_err(), GeometricError::ZeroMax);
    ///
    /// // Invalid: probability out of range.
    /// assert_eq!(Geometric::new(16, 0.0).unwrap_err(), GeometricError::InvalidProbability);
    /// assert_eq!(Geometric::new(16, 1.0).unwrap_err(), GeometricError::InvalidProbability);
    /// ```
    #[inline]
    pub fn new(total: usize, p: f64) -> Result<Self, GeometricError> {
        if total == 0 {
            return Err(GeometricError::ZeroMax);
        }
        let Ok(total_i32) = i32::try_from(total) else {
            return Err(GeometricError::MaxTooLarge);
        };
        if !(0.0 < p && p < 1.0) {
            return Err(GeometricError::InvalidProbability);
        }
        Ok(Geometric {
            total,
            total_i32,
            q: 1.0 - p,
            rng: SmallRng::from_rng(thread_rng()).map_err(|_err| GeometricError::RngInitFailed)?,
        })
    }
}

impl LevelGenerator for Geometric {
    #[inline]
    fn total(&self) -> usize {
        self.total
    }

    /// Generates a random level in the range `$[0, \text{total})$`.
    ///
    /// The level is sampled from the truncated geometric distribution configured
    /// at construction time. Lower levels are more probable; level 0 is the most
    /// common.
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "CDF domain is [0, total] so the cast is safe"
    )]
    #[expect(clippy::as_conversions, reason = "No other way to do this")]
    fn level(&mut self) -> usize {
        let u = self.rng.r#gen::<f64>();
        (1.0 + (self.q.powi(self.total_i32) - 1.0) * u)
            .log(self.q)
            .floor() as usize
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    #[cfg(not(miri))]
    use anyhow::bail;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::{Geometric, LevelGenerator};
    use crate::level_generator::geometric::GeometricError;

    #[test]
    fn invalid_max() {
        assert_eq!(Geometric::new(0, 0.5).err(), Some(GeometricError::ZeroMax));
    }

    #[test]
    fn invalid_p() {
        assert_eq!(
            Geometric::new(1, 0.0).err(),
            Some(GeometricError::InvalidProbability)
        );
        assert_eq!(
            Geometric::new(1, 1.0).err(),
            Some(GeometricError::InvalidProbability)
        );
    }

    // Miri is very slow, so we use a much smaller number of iterations, and
    // don't check for the presence of min and max level nodes.
    #[cfg(miri)]
    #[rstest]
    fn new_miri(
        #[values(1, 2, 128, 1024)] n: usize,
        #[values(0.01, 0.1, 0.5, 0.99)] p: f64,
    ) -> Result<()> {
        const MAX: usize = 10;

        let mut generator = Geometric::new(n, p)?;
        assert_eq!(generator.total(), n);
        for _ in 0..MAX {
            let level = generator.level();
            assert!((0..n).contains(&level));
        }
        Ok(())
    }

    #[cfg(not(miri))]
    #[rstest]
    fn new_small(
        #[values(1, 2, 4, 8)] n: usize,
        #[values(0.01, 0.1, 0.5, 0.8)] p: f64,
    ) -> Result<()> {
        const MAX: usize = 10_000_000;

        let mut generator = Geometric::new(n, p)?;
        assert_eq!(generator.total(), n);
        for _ in 0..1_000 {
            let level = generator.level();
            assert!((0..n).contains(&level));
        }
        // Make sure that we can produce at least one level-0 node, and one at the
        // maximum level.
        let mut found = false;
        for _ in 0..MAX {
            let level = generator.level();
            if level == 0 {
                found = true;
                break;
            }
        }
        if !found {
            bail!("Failed to generate a level-0 node.");
        }

        found = false;
        for _ in 0..MAX {
            let level = generator.level();
            if level == n.checked_sub(1).expect("n is guaranteed to be > 0") {
                found = true;
                break;
            }
        }
        if !found {
            bail!(
                "Failed to generate a level-{} node.",
                n.checked_sub(1).expect("n is guaranteed to be > 0")
            );
        }

        Ok(())
    }

    #[cfg(not(miri))]
    #[rstest]
    fn new_large(#[values(512, 1024)] n: usize, #[values(0.001, 0.01)] p: f64) -> Result<()> {
        const MAX: usize = 10_000_000;

        let mut generator = Geometric::new(n, p)?;
        assert_eq!(generator.total(), n);
        for _ in 0..1_000 {
            let level = generator.level();
            assert!((0..n).contains(&level));
        }
        // Make sure that we can produce at least one level-0 node, and one at the
        // maximum level.
        let mut found = false;
        for _ in 0..MAX {
            let level = generator.level();
            if level == 0 {
                found = true;
                break;
            }
        }
        if !found {
            bail!("Failed to generate a level-0 node.");
        }

        found = false;
        for _ in 0..MAX {
            let level = generator.level();
            if level == n.checked_sub(1).expect("n is guaranteed to be > 0") {
                found = true;
                break;
            }
        }
        if !found {
            bail!(
                "Failed to generate a level-{} node.",
                n.checked_sub(1).expect("n is guaranteed to be > 0")
            );
        }

        Ok(())
    }
}
