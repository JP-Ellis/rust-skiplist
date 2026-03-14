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

    /// Creates a new geometric level generator with a fixed seed for
    /// reproducible output.
    ///
    /// Identical to [`Geometric::new`] except the internal RNG is seeded from
    /// `seed` instead of the thread-local RNG. Useful for deterministic tests.
    ///
    /// # Errors
    ///
    /// Returns a [`GeometricError`] if `total` is zero, if `total` is too
    /// large, or if `q` is not in the open interval `$(0, 1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::level_generator::geometric::Geometric;
    ///
    /// let mut g = Geometric::new_with_seed(16, 0.5, 42).unwrap();
    /// ```
    #[inline]
    pub fn new_with_seed(total: usize, q: f64, seed: u64) -> Result<Self, GeometricError> {
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
            rng: SmallRng::seed_from_u64(seed),
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
    use anyhow::{Result, anyhow};
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

    #[rstest]
    fn total_is_correct(
        #[values(1, 2, 4, 8, 128, 512, 1024)] n: usize,
        #[values(0.01, 0.1, 0.5, 0.99)] q: f64,
    ) -> Result<()> {
        let generator = Geometric::new_with_seed(n, q, 42)?;
        assert_eq!(generator.total(), n);
        Ok(())
    }

    /// Checks that level 0 is reachable within `MAX` attempts.
    ///
    /// P(level = 0) = 1 - q, so the expected number of draws is 1/(1-q). Even
    /// with a large q, the probability of not seeing level 0 after MAX attempts
    /// is negligible.
    #[rstest]
    fn generates_level_zero(
        #[values(1, 2, 4, 8, 128, 512, 1024)] n: usize,
        #[values(0.01, 0.1, 0.2, 0.5, 0.8, 0.99)] q: f64,
    ) -> Result<()> {
        const MAX: usize = if cfg!(miri) { 50 } else { 10_000_000 };

        let mut generator = Geometric::new_with_seed(n, q, 42)?;
        let found = (0..MAX).any(|_| {
            let level = generator.level();
            assert!(
                (0..=n).contains(&level),
                "level {level} out of range 0..={n}"
            );
            level == 0
        });

        if !cfg!(miri) {
            // Skip as miri is slow and changes are the test will fail with only
            // 50 trials.
            assert!(
                found,
                "Failed to generate a level-0 node after {MAX} attempts"
            );
        }
        Ok(())
    }

    /// Checks that the maximum level is reachable within `MAX` attempts.
    ///
    /// The probability is P(level = n) = p·q^n, so the expected number of draws
    /// is 1/(p·q^n). For small q and large n, this may require many trials,
    /// hence select values of `q` which are not _too_ small.
    #[rstest]
    fn generates_max_level_small_n(
        #[values(1, 2, 4, 8)] n: usize,
        #[values(0.2, 0.5, 0.8, 0.9, 0.99)] q: f64,
    ) -> Result<()> {
        const MAX: usize = if cfg!(miri) { 50 } else { 10_000_000 };

        let mut generator = Geometric::new_with_seed(n, q, 42)?;
        let found = (0..MAX).any(|_| {
            let level = generator.level();
            assert!(
                (0..=n).contains(&level),
                "level {level} out of range 0..={n}"
            );
            level == n
        });

        if !cfg!(miri) {
            // Skip as miri is slow and changes are the test will fail with only
            // 50 trials.
            assert!(
                found,
                "Failed to generate a level-{n} node after {MAX} attempts"
            );
        }
        Ok(())
    }

    #[rstest]
    fn generates_max_level_large_n(
        #[values(32, 64)] n: usize,
        #[values(0.99, 0.999)] q: f64,
    ) -> Result<()> {
        const MAX: usize = if cfg!(miri) { 50 } else { 10_000_000 };

        let mut generator = Geometric::new_with_seed(n, q, 42)?;
        let found = (0..MAX).any(|_| {
            let level = generator.level();
            assert!(
                (0..=n).contains(&level),
                "level {level} out of range 0..={n}"
            );
            level == n
        });

        if !cfg!(miri) {
            assert!(
                found,
                "Failed to generate a level-{n} node after {MAX} attempts"
            );
        }
        Ok(())
    }

    /// Verifies that consecutive level counts follow the expected geometric
    /// ratio.
    ///
    /// Only adjacent pairs where both counts exceed `MIN_COUNT` are checked to avoid
    /// statistical noise from low counts.
    #[rstest]
    fn distribution_ratio(
        #[values(4, 8, 16)] n: usize,
        #[values(0.1, 0.2, 0.5, 0.8, 0.9)] q: f64,
    ) -> Result<()> {
        const SAMPLES: usize = if cfg!(miri) { 50 } else { 10_000_000 };
        const MIN_COUNT: u32 = 1_000;
        const TOLERANCE: f64 = 0.05;

        let mut counts = vec![0_u32; n.strict_add(1)];
        let mut generator = Geometric::new_with_seed(n, q, 42)?;
        for _ in 0..SAMPLES {
            if let Some(count) = counts.get_mut(generator.level()) {
                *count = count.strict_add(1);
            } else {
                panic!("Generated level {} out of range 0..={n}", generator.level());
            }
        }

        if cfg!(miri) {
            return Ok(());
        }

        for k in 0..n {
            let next_k = k.strict_add(1);
            let count_k = counts
                .get(k)
                .copied()
                .ok_or_else(|| anyhow!("invalid count bin"))?;
            let count_next_k = counts
                .get(next_k)
                .copied()
                .ok_or_else(|| anyhow!("invalid count bin"))?;
            if count_k < MIN_COUNT || count_next_k < MIN_COUNT {
                // Higher levels will have even fewer samples; no point continuing.
                break;
            }

            let ratio = f64::from(count_next_k) / f64::from(count_k);
            let relative_err = (ratio - q).abs() / q;
            assert!(
                relative_err < TOLERANCE,
                "level {k}→{next_k}: count[{k}]={count_k}, count[{next_k}]={count_next_k}, \
                 ratio={ratio:.4}, expected q={q:.4} (err {:.1}%)",
                relative_err * 100.0,
            );
        }

        Ok(())
    }
}
