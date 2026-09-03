//! Test-only helpers shared across the container test modules.

use core::cmp::Ordering;

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
