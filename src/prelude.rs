//! Convenience re-exports for glob import.
//!
//! ```rust
//! use skiplist::prelude::*;
//! ```

#[cfg(feature = "partial-ord")]
pub use crate::PartialOrdComparator;
pub use crate::{
    Comparator, ComparatorKey, FnComparator, Geometric, LevelGenerator, OrdComparator,
    OrderedSkipList, SkipList, SkipMap, SkipSet,
};
#[cfg(feature = "cursor")]
pub use crate::{ordered_skip_list::UnorderedValueError, skip_map::UnorderedKeyError};
