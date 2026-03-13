//! Level generator for skip lists.
//!
//! Skip lists use a probabilistic distribution to determine how many levels
//! each node participates in. Nodes at higher levels span greater distances,
//! making traversal faster by skipping over large portions of the list.
//!
//! The default level generator uses a geometric distribution with a fixed
//! probability `$p$` that a node is promoted to the next level. This is
//! implemented in [`Geometric`][geometric::Geometric].
//!
//! Custom level generators can be implemented via the [`LevelGenerator`]
//! trait. In practice, the default should suffice for most use cases.

pub mod geometric;

/// Trait for generating the height of a new node inserted into a skip list.
///
/// A level generator controls how many skip-link levels (the *height*) each new
/// node receives.  Nodes with more skip links allow the list to skip over
/// larger ranges during traversal, trading memory for speed.
///
/// The returned height must always be in the range `$[0, \text{total}]$`. A
/// height of `0` means the node participates only in the base `prev`/`next`
/// linked list and has no skip links.  A height of `total` (the maximum)
/// gives the node the same number of skip links as the head sentinel.
///
/// Returning a value outside `$[0, \text{total}]$` is a logic error and may
/// cause panics or incorrect behavior.
///
/// # Examples
///
/// Implementing a trivial level generator that always returns height 0 (all
/// nodes participate only in the base layer; traversal degrades to `$O(n)$`):
///
/// ```rust
/// use skiplist::level_generator::LevelGenerator;
///
/// struct AlwaysZero {
///     max_levels: usize,
/// }
///
/// impl LevelGenerator for AlwaysZero {
///     fn total(&self) -> usize {
///         self.max_levels
///     }
///
///     fn level(&mut self) -> usize {
///         0
///     }
/// }
///
/// let mut generator = AlwaysZero { max_levels: 16 };
/// assert_eq!(generator.total(), 16);
/// assert_eq!(generator.level(), 0);
/// ```
pub trait LevelGenerator {
    /// Returns the total number of levels supported by this generator.
    ///
    /// The head sentinel is created with exactly `total` skip-link slots.
    /// All values returned by [`level`][LevelGenerator::level] will be
    /// at most this value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::level_generator::geometric::Geometric;
    /// use skiplist::level_generator::LevelGenerator;
    ///
    /// let generator = Geometric::default();
    /// assert_eq!(generator.total(), 16);
    /// ```
    #[must_use]
    fn total(&self) -> usize;

    /// Returns the number of skip links to allocate for a new node (its
    /// *height*), sampled according to the generator's distribution.
    ///
    /// The returned value must be in `$[0, \text{total}]$`:
    ///
    /// - `0` — no skip links; the node is reachable only via the base
    ///   `prev`/`next` pointers.
    /// - `k` — the node receives skip links at levels `0` through `k - 1`.
    /// - `total` — the maximum height; the node has the same number of skip
    ///   links as the head sentinel.
    ///
    /// This method must not return a value greater than
    /// [`total`][LevelGenerator::total].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::level_generator::geometric::Geometric;
    /// use skiplist::level_generator::LevelGenerator;
    ///
    /// let mut generator = Geometric::default();
    /// let height = generator.level();
    /// assert!(height <= generator.total());
    /// ```
    #[must_use]
    fn level(&mut self) -> usize;
}
