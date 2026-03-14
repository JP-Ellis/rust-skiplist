//! Geometric level generator.

#![expect(clippy::float_arithmetic, reason = "computing probabilities")]

use core::{error::Error, fmt};

use rand::prelude::*;

use crate::level_generator::LevelGenerator;

/// Errors that can occur when creating a [`Geometric`] level generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::module_name_repetitions,
    reason = "Using 'Error' would be too generic and may cause confusion."
)]
#[non_exhaustive]
pub enum GeometricError {
    /// The maximum number of levels must be non-zero.
    ZeroMax,
    /// The maximum number of levels must be less than `i32::MAX`.
    MaxTooLarge,
    /// The probability `$q$` must be in the range `$(0, 1)$`.
    InvalidProbability,
}

impl fmt::Display for GeometricError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMax => write!(f, "max must be non-zero."),
            Self::MaxTooLarge => write!(f, "max must be less than i32::MAX."),
            Self::InvalidProbability => write!(f, "q must be in (0, 1)."),
        }
    }
}

impl Error for GeometricError {}

/// A level generator using a geometric distribution.
///
/// Each new element is assigned a random level drawn from a truncated geometric
/// distribution. If an element exists at level `$n$`, the probability that it
/// also exists at level `$n+1$` is `$q \in (0, 1)$`. The level is capped at the
/// configured maximum.
///
/// Note that in mathematics, geometric distributions are conventionally defined
/// in terms of the success probability `$p = 1 - q$`.
///
/// Also note that for very large values of `$q$`, the assumption that `$P(k +
/// 1|k) = q$` breaks down due to the truncation at `total`.
///
/// Use [`Geometric::new`] to configure the number of levels and the promotion
/// probability, or [`Geometric::default`] for the standard 16-level, `$q =
/// 0.5$` configuration.
///
/// # Examples
///
/// ```rust
/// use skiplist::level_generator::LevelGenerator;
/// use skiplist::level_generator::geometric::Geometric;
///
/// let mut generator = Geometric::new(16, 0.5).unwrap();
/// let level = generator.level();
/// assert!(level <= generator.total());
/// ```
#[derive(Debug, Clone)]
pub struct Geometric {
    /// The total number of levels that are assumed to exist.
    total: usize,
    /// The total number of levels as an `i32` for use in the CDF computation.
    /// This needs to be 1 more than the maximum levels to ensure that it
    /// includes the full range of possible levels (0..=total).
    total_inclusive: i32,
    /// The promotion probability, i.e., the probability that a node at level
    /// `n` also appears at level `n + 1`.
    q: f64,
    /// The random number generator.
    rng: SmallRng,
}

impl Geometric {
    /// Creates a new geometric level generator.
    ///
    /// `total` sets the maximum number of levels (must be at least 1 and less
    /// than `i32::MAX`). `q` is the probability that an element is promoted to
    /// the next level and must be strictly between 0 and 1.
    ///
    /// # Errors
    ///
    /// Returns a [`GeometricError`] if `total` is zero, if `total` is too
    /// large, or if `q` is not in the open interval `$(0, 1)$`.
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
    pub fn new(total: usize, q: f64) -> Result<Self, GeometricError> {
        if total == 0 {
            return Err(GeometricError::ZeroMax);
        }
        let Some(total_inclusive) = i32::try_from(total).ok().and_then(|i| i.checked_add(1)) else {
            return Err(GeometricError::MaxTooLarge);
        };
        if !(0.0 < q && q < 1.0) {
            return Err(GeometricError::InvalidProbability);
        }
        Ok(Geometric {
            total,
            total_inclusive,
            q,
            rng: SmallRng::from_rng(&mut rand::rng()),
        })
    }
}

impl Default for Geometric {
    /// Creates a `Geometric` level generator with 16 levels and `q = 0.5`.
    ///
    /// These defaults match the standard skip list configuration and provide a
    /// good balance between memory usage and search performance for most use
    /// cases.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::level_generator::LevelGenerator;
    /// use skiplist::level_generator::geometric::Geometric;
    ///
    /// let mut generator = Geometric::default();
    /// assert_eq!(generator.total(), 16);
    /// ```
    #[inline]
    fn default() -> Self {
        #[expect(
            clippy::expect_used,
            reason = "16 levels and q = 0.5 are compile-time constants whose \
                      validity is guaranteed by the Geometric invariants"
        )]
        Geometric::new(16, 0.5)
            .expect("16 levels and q = 0.5 are always valid Geometric parameters")
    }
}

impl LevelGenerator for Geometric {
    #[inline]
    fn total(&self) -> usize {
        self.total
    }

    /// Returns the height (number of skip links) to allocate for a new node,
    /// sampled from a truncated geometric distribution.
    ///
    /// The returned value is in `$[0, \text{total}]$`.  Height `0` is the
    /// most probable outcome (the node gets no skip links and participates only
    /// in the base layer); `total` is the least probable.
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "CDF domain is [0, total] so the cast is safe after clamping"
    )]
    #[expect(clippy::as_conversions, reason = "No other way to do this")]
    fn level(&mut self) -> usize {
        // Invert the CDF of the truncated geometric distribution:
        //
        //   CDF(n) = (q^n - 1) / (q^t - 1)
        //
        // where t is the _exclusive_ upper bound (i.e., total + 1).
        //
        // Solving for n given a uniform variate u in [0, 1]:
        //
        //   n = floor( log_q( 1 + (q^t - 1) * u ) )
        //
        // where q = 1 - p and t is the total number of levels.
        let u = self.rng.random::<f64>();
        ((1.0 + (self.q.powi(self.total_inclusive) - 1.0) * u)
            .log(self.q)
            .floor() as usize)
            // When q^total underflows to 0.0 due to floating-point precision,
            // the formula can produce values > total.  This ensures that we
            // never return a level greater than total.
            .min(self.total)
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
            assert!((0..=n).contains(&level));
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
            assert!((0..=n).contains(&level));
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
            assert!((0..=n).contains(&level));
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
