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

/// Trait for generating the level of a new node inserted into a skip list.
///
/// A level generator controls how many skip-link levels each new node
/// participates in. Higher levels allow the skip list to skip over more nodes
/// during traversal, trading memory for speed. The generator determines the
/// probabilistic balance between the two.
///
/// The returned level must always be in the range `$[0, \text{total})$`.
/// Returning a value out of that range is considered a logic error and may
/// cause panics or incorrect behavior.
///
/// # Examples
///
/// Implementing a trivial level generator that always returns level 0:
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
    /// All values returned by [`level`][LevelGenerator::level] will be
    /// strictly less than this value.
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

    /// Generate a random level for a new node in the range `$[0,
    /// \text{total})$`.
    ///
    /// This method must not return a value greater than or equal to
    /// [`total`][LevelGenerator::total].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::level_generator::geometric::Geometric;
    /// use skiplist::level_generator::LevelGenerator;
    ///
    /// let mut generator = Geometric::default();
    /// let level = generator.level();
    /// assert!(level < generator.total());
    /// ```
    #[must_use]
    fn level(&mut self) -> usize;
}
