//! Operator overloads for [`SkipMap`](super::SkipMap): `Index<&K>`.

use core::ops::Index;

use crate::{comparator::Comparator, level_generator::LevelGenerator};

use super::SkipMap;

// MARK: Index<&K>

impl<K, V, C: Comparator<K>, G: LevelGenerator, const N: usize> Index<&K>
    for SkipMap<K, V, N, C, G>
{
    type Output = V;

    /// Returns a reference to the value associated with `key`.
    ///
    /// # Panics
    ///
    /// Panics if `key` is not present in the map.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_map::SkipMap;
    ///
    /// let mut map = SkipMap::<i32, &str>::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    ///
    /// assert_eq!(map[&1], "a");
    /// assert_eq!(map[&2], "b");
    /// ```
    #[expect(
        clippy::unwrap_used,
        reason = "None means the key is absent; panicking is the documented behaviour of Index"
    )]
    #[inline]
    fn index(&self, key: &K) -> &V {
        self.get(key)
            .unwrap_or_else(|| panic!("key not found in SkipMap"))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::SkipMap;

    // MARK: Index<&K>

    #[test]
    fn index_present_key() {
        let mut map = SkipMap::<i32, &str>::new();
        map.insert(1, "a");
        map.insert(2, "b");
        assert_eq!(map[&1], "a");
        assert_eq!(map[&2], "b");
    }

    #[test]
    #[should_panic(expected = "key not found")]
    fn index_absent_key_panics() {
        let map = SkipMap::<i32, &str>::new();
        let _ = map[&99];
    }

    #[test]
    fn index_returns_correct_value() {
        let mut map = SkipMap::<&str, i32>::new();
        map.insert("one", 1);
        map.insert("two", 2);
        map.insert("three", 3);
        assert_eq!(map[&"one"], 1);
        assert_eq!(map[&"two"], 2);
        assert_eq!(map[&"three"], 3);
    }
}
