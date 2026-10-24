//! Geometric level generator.

#![expect(clippy::float_arithmetic, reason = "computing probabilities")]

use alloc::vec::Vec;
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
    /// The maximum number of levels exceeds [`Geometric::MAX_TOTAL`].
    MaxTooLarge,
    /// The probability `$q$` must be in the range `$(0, 1)$`.
    InvalidProbability,
    /// The platform entropy source failed while seeding the generator.
    Entropy,
}

impl fmt::Display for GeometricError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMax => write!(f, "max must be non-zero."),
            Self::MaxTooLarge => write!(f, "max must be at most 64."),
            Self::InvalidProbability => write!(f, "q must be in (0, 1)."),
            Self::Entropy => write!(f, "failed to obtain entropy."),
        }
    }
}

impl Error for GeometricError {}

/// How [`Geometric::level`] draws a sample.
#[derive(Debug, Clone)]
enum Sampler {
    /// `q == 0.5` exactly: the level is the index of the lowest set bit of a
    /// random `u64`, redrawn when zero or above `total`.
    Bisect,
    /// Any other `q`: binary search the CDF over levels `0..=total`, built at
    /// construction.
    Table(Vec<f64>),
}

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
/// Use `Geometric::new` to configure the number of levels and the promotion
/// probability, or `Geometric::default` for the standard 16-level, `$q =
/// 0.5$` configuration. Both need the `std` or `getrandom` feature;
/// `Geometric::new_with_seed` does not.
///
/// When `q` is exactly `0.5` (the default), levels are sampled by bit position.
/// For any other `q`, the cumulative distribution is tabulated once at
/// construction and each draw is a binary search of a uniform variate in that
/// table. Neither path calls `powi` or `log`, so no `libm` is needed.
///
/// The number of levels is capped at [`Geometric::MAX_TOTAL`] (64), which keeps
/// the table small. The bit sampler draws heights `0..=63` from the 64 bits of
/// its random word, so at `total = 64` it never draws height 64, whose
/// probability is about `2^-65`.
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
    /// The random number generator.
    rng: SmallRng,
    /// The sampling strategy, chosen once from `q` at construction.
    sampler: Sampler,
}

/// Builds the CDF of the truncated geometric distribution over `0..=total`.
///
/// Entry `i` is `(sum of q^j for j <= i) / (sum of q^j for j <= total)`. Prefix
/// sums of the powers avoid the cancellation in `1 - q^k` when `q` is near 1.
/// The last entry is set to exactly `1.0` so every `u < 1` lands in range.
fn cdf_table(total: usize, q: f64) -> Vec<f64> {
    let mut power = 1.0_f64;
    let mut cdf: Vec<f64> = Vec::with_capacity(total.saturating_add(1));
    let mut sum = 0.0_f64;
    for _ in 0..=total {
        sum += power;
        cdf.push(sum);
        power *= q;
    }
    for entry in &mut cdf {
        *entry /= sum;
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    cdf
}

impl Geometric {
    /// The largest supported number of levels.
    pub const MAX_TOTAL: usize = 64;

    /// Checks the parameters in the order `total == 0`, `total` too large,
    /// then `q` out of range.
    fn validate(total: usize, q: f64) -> Result<(), GeometricError> {
        (total != 0).ok_or(GeometricError::ZeroMax)?;
        (total <= Self::MAX_TOTAL).ok_or(GeometricError::MaxTooLarge)?;
        (0.0 < q && q < 1.0).ok_or(GeometricError::InvalidProbability)?;
        Ok(())
    }

    /// Validates the parameters, picks the sampler and builds the CDF table.
    fn from_rng(total: usize, q: f64, rng: SmallRng) -> Result<Self, GeometricError> {
        Self::validate(total, q)?;
        #[expect(
            clippy::float_cmp,
            reason = "exact match on the literal 0.5 selects the bit sampler"
        )]
        let sampler = if q == 0.5 {
            Sampler::Bisect
        } else {
            Sampler::Table(cdf_table(total, q))
        };
        Ok(Geometric {
            total,
            rng,
            sampler,
        })
    }

    /// Creates a new geometric level generator.
    ///
    /// `total` sets the maximum number of levels (must be at least 1 and at
    /// most [`Geometric::MAX_TOTAL`]). `q` is the probability that an element
    /// is promoted to the next level and must be strictly between 0 and 1.
    ///
    /// Available with the `std` or `getrandom` feature. Without either, use
    /// [`Geometric::new_with_seed`].
    ///
    /// # Errors
    ///
    /// Returns a [`GeometricError`] if `total` is zero, if `total` is too
    /// large, or if `q` is not in the open interval `$(0, 1)$`. Without the
    /// `std` feature, returns [`GeometricError::Entropy`] if the operating
    /// system entropy source fails.
    ///
    /// # Panics
    ///
    /// With the `std` feature, panics if the entropy source behind
    /// `rand::rng()` fails.
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
    #[cfg(any(feature = "std", feature = "getrandom"))]
    #[inline]
    pub fn new(total: usize, q: f64) -> Result<Self, GeometricError> {
        Self::validate(total, q)?;
        #[cfg(feature = "std")]
        let rng = SmallRng::from_rng(&mut rand::rng());
        #[cfg(not(feature = "std"))]
        let rng = SmallRng::try_from_rng(&mut rand::rngs::SysRng)
            .map_err(|_error| GeometricError::Entropy)?;
        Self::from_rng(total, q, rng)
    }

    /// Creates a new geometric level generator with a fixed seed for
    /// reproducible output.
    ///
    /// Identical to `Geometric::new` except the internal RNG is seeded from
    /// `seed` instead of an entropy source. Useful for deterministic tests and
    /// for `no_std` targets without the `std` or `getrandom` feature.
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
        Self::from_rng(total, q, SmallRng::seed_from_u64(seed))
    }

    /// Samples by binary search of the CDF table.
    ///
    /// Returns the first level whose cumulative probability exceeds a uniform
    /// variate in `[0, 1)`. The last entry is `1.0`, so the result is at most
    /// `total`.
    #[inline]
    fn level_table(rng: &mut SmallRng, cdf: &[f64]) -> usize {
        let u: f64 = rng.random();
        cdf.partition_point(|&c| c <= u)
    }

    /// Samples by bit position.
    ///
    /// For `q = 1/2` the truncated distribution is
    /// `P(n) = 2^-(n+1) / (1 - 2^-(total+1))` on `0..=total`.  The lowest set
    /// bit of a uniform `u64` has `P(n) = 2^-(n+1)`; redrawing when the word
    /// is zero or the bit index exceeds `total` yields the truncated law
    /// exactly for `total <= 63`.  For larger `total`, levels `64..=total`
    /// have zero probability and the total-variation gap is below `2^-64`.
    /// Expected draws per call: `1 / (1 - 2^-(total+1))`.
    #[inline]
    fn level_bisect(&mut self) -> usize {
        loop {
            let word: u64 = self.rng.next_u64();
            if let Some(bit) = word.lowest_one()
                && let Ok(level) = usize::try_from(bit)
                && level <= self.total
            {
                return level;
            }
        }
    }
}

#[cfg(any(feature = "std", feature = "getrandom"))]
impl Default for Geometric {
    /// Creates a `Geometric` level generator with 16 levels and `q = 0.5`.
    ///
    /// These defaults match the standard skip list configuration and provide a
    /// good balance between memory usage and search performance for most use
    /// cases.
    ///
    /// # Panics
    ///
    /// Panics if the entropy source fails: `rand::rng()` under `std`, or the
    /// operating system source under `getrandom` alone.
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
            .expect("16 levels and q = 0.5 are valid, and the entropy source must be available")
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
    fn level(&mut self) -> usize {
        match &self.sampler {
            Sampler::Bisect => self.level_bisect(),
            Sampler::Table(cdf) => Self::level_table(&mut self.rng, cdf),
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::indexing_slicing, reason = "test code")]

    use anyhow::{Result, anyhow};
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::{Geometric, LevelGenerator};
    use crate::level_generator::geometric::GeometricError;

    #[test]
    fn validation_order() {
        assert_eq!(Geometric::new(0, 2.0).err(), Some(GeometricError::ZeroMax));
        assert_eq!(
            Geometric::new(Geometric::MAX_TOTAL + 1, 2.0).err(),
            Some(GeometricError::MaxTooLarge)
        );
        assert_eq!(
            Geometric::new_with_seed(0, 2.0, 1).err(),
            Some(GeometricError::ZeroMax)
        );
    }

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
        #[values(1, 2, 4, 8, 32, 64)] n: usize,
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
        #[values(1, 2, 4, 8, 32, 64)] n: usize,
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

    // MARK: bisect sampler

    #[test]
    fn q_half_selects_bisect() -> Result<()> {
        let bisect = Geometric::new_with_seed(16, 0.5, 42)?;
        assert!(matches!(bisect.sampler, super::Sampler::Bisect));
        let table = Geometric::new_with_seed(16, 0.25, 42)?;
        assert!(matches!(table.sampler, super::Sampler::Table(_)));
        Ok(())
    }

    #[rstest]
    fn bisect_never_exceeds_total(#[values(1, 4, 16, 63)] n: usize) -> Result<()> {
        const SAMPLES: usize = if cfg!(miri) { 50 } else { 1_000_000 };
        let mut generator = Geometric::new_with_seed(n, 0.5, 42)?;
        for _ in 0..SAMPLES {
            let level = generator.level_bisect();
            assert!(level <= n, "level {level} exceeds total {n}");
        }
        Ok(())
    }

    /// The bisect and table samplers must produce the same truncated distribution at
    /// q = 1/2.  Compare per-level frequencies where counts are large enough
    /// for the relative error to be meaningful.
    ///
    /// Level `k` collects about `SAMPLES / 2^(k+1)` draws, so the relative
    /// standard error of the difference between two bins of `MIN_COUNT` is
    /// `sqrt(2 / MIN_COUNT)`; at 20,000 that is 1%, five times inside the
    /// tolerance, and six bins still qualify at `total = 16`.
    #[rstest]
    fn bisect_matches_table_distribution(#[values(1, 4, 16, 64)] n: usize) -> Result<()> {
        const SAMPLES: usize = if cfg!(miri) { 50 } else { 2_000_000 };
        const MIN_COUNT: u32 = 20_000;
        const TOLERANCE: f64 = 0.05;

        let mut bisect = Geometric::new_with_seed(n, 0.5, 7)?;
        let mut table = Geometric::new_with_seed(n, 0.5, 11)?;
        table.sampler = super::Sampler::Table(super::cdf_table(n, 0.5));
        let mut bisect_counts = vec![0_u32; n.strict_add(1)];
        let mut table_counts = vec![0_u32; n.strict_add(1)];
        for _ in 0..SAMPLES {
            let b = bisect.level_bisect();
            bisect_counts[b] = bisect_counts[b].strict_add(1);
            let t = table.level();
            table_counts[t] = table_counts[t].strict_add(1);
        }

        if cfg!(miri) {
            return Ok(());
        }

        for (k, (&b, &t)) in bisect_counts.iter().zip(&table_counts).enumerate() {
            if b < MIN_COUNT || t < MIN_COUNT {
                break;
            }
            let relative_err = (f64::from(b) - f64::from(t)).abs() / f64::from(t);
            assert!(
                relative_err < TOLERANCE,
                "level {k}: bisect={b}, table={t}, err {:.1}%",
                relative_err * 100.0
            );
        }
        Ok(())
    }

    // MARK: table sampler

    #[test]
    fn total_cap_boundary() -> Result<()> {
        assert_eq!(Geometric::MAX_TOTAL, 64);
        assert_eq!(Geometric::new_with_seed(64, 0.7, 0)?.total(), 64);
        assert_eq!(
            Geometric::new_with_seed(65, 0.7, 0).err(),
            Some(GeometricError::MaxTooLarge)
        );
        assert_eq!(Geometric::new(64, 0.5)?.total(), 64);
        assert_eq!(
            Geometric::new(65, 0.5).err(),
            Some(GeometricError::MaxTooLarge)
        );
        Ok(())
    }

    #[rstest]
    fn cdf_table_shape(#[values(1, 4, 16, 64)] n: usize, #[values(0.01, 0.25, 0.7, 0.999)] q: f64) {
        let cdf = super::cdf_table(n, q);
        assert_eq!(cdf.len(), n.strict_add(1));
        assert_eq!(cdf.last(), Some(&1.0));
        assert!(cdf.is_sorted());
        assert!(cdf[0] > 0.0);
    }

    #[rstest]
    fn table_extreme_q_stays_in_range(#[values(0.999_999, 1.0 - 1e-12)] q: f64) -> Result<()> {
        const SAMPLES: usize = if cfg!(miri) { 50 } else { 1_000_000 };
        let mut g = Geometric::new_with_seed(64, q, 42)?;
        let mut saw_top = false;
        for _ in 0..SAMPLES {
            let level = g.level();
            assert!(level <= 64);
            saw_top |= level == 64;
        }
        assert!(cfg!(miri) || saw_top, "top level never drawn");
        Ok(())
    }

    #[test]
    fn table_tiny_q_returns_zero() -> Result<()> {
        let mut g = Geometric::new_with_seed(64, 1e-300, 42)?;
        for _ in 0..1_000 {
            assert_eq!(g.level(), 0);
        }
        Ok(())
    }

    #[test]
    fn seed_is_reproducible() -> Result<()> {
        let mut a = Geometric::new_with_seed(16, 0.7, 9)?;
        let mut b = Geometric::new_with_seed(16, 0.7, 9)?;
        for _ in 0..1_000 {
            assert_eq!(a.level(), b.level());
        }
        Ok(())
    }
}
