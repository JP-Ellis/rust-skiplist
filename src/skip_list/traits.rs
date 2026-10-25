//! Standard trait implementations for [`SkipList`](super::SkipList):
//! `Debug`, `Clone`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash`,
//! `Extend`, `FromIterator`, and `From`.

use alloc::vec::Vec;
use core::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
};

use crate::{level_generator::LevelGenerator, node::Node, skip_list::SkipList};

// MARK: Debug

impl<T: fmt::Debug, G: LevelGenerator, const N: usize> fmt::Debug for SkipList<T, N, G> {
    /// Formats the list as a debug list, e.g. `[1, 2, 3]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let list: SkipList<i32> = [1, 2, 3].into();
    /// assert_eq!(format!("{list:?}"), "[1, 2, 3]");
    /// ```
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// MARK: Clone

impl<T: Clone, G: LevelGenerator + Clone, const N: usize> Clone for SkipList<T, N, G> {
    /// Returns a deep clone of the list.
    ///
    /// The cloned list has the same elements in the same order, with the
    /// same tower heights. The level generator is also cloned, so the clone
    /// shares the same probability distribution for future insertions but
    /// has its own independent state.
    ///
    /// This operation is `$O(n)$`.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let list: SkipList<i32> = [1, 2, 3].into();
    /// let cloned = list.clone();
    /// assert_eq!(list, cloned);
    /// ```
    #[expect(
        clippy::expect_used,
        reason = "`value()` returns None only for the head sentinel, which is never visited \
                  in the data-node walk; the expect fires only on invariant violations"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "insert_after and rebuild touch provably disjoint heap nodes; \
                  splitting across blocks would require unsafe-crossing raw-pointer variables"
    )]
    #[inline]
    fn clone(&self) -> Self {
        let mut new_list = Self::with_level_generator(self.generator.clone());
        if self.is_empty() {
            return new_list;
        }

        // Clone each node at its tower height along the base chain, then wire
        // every skip link in one pass.  A panicking `clone` leaves a valid
        // chain for `Drop`.
        //
        // SAFETY: `self.head` is the live head of this list and every
        // `src_nn` is one of its data nodes.  `new_list.head` and `prev_nn`
        // are owned by `new_list`, with no other live references.
        unsafe {
            let mut prev_nn = new_list.head;
            let mut src_opt = self.head.as_ref().next();
            while let Some(src_nn) = src_opt {
                let src_node = src_nn.as_ref();
                let val = src_node.value().expect("data node has a value").clone();
                prev_nn = Node::insert_after(prev_nn, Node::with_value(src_node.level(), val));
                src_opt = src_node.next();
            }
            Node::filter_rebuild(
                new_list.head,
                &mut new_list.len,
                &mut new_list.tail,
                |_| true,
                |_| {},
            );
        }
        new_list
    }
}

// MARK: PartialOrd / Ord

impl<T: PartialOrd, G: LevelGenerator, const N: usize> PartialOrd for SkipList<T, N, G> {
    /// Compares two lists lexicographically.
    ///
    /// Returns `None` if any pair of corresponding elements returns `None`
    /// from their own `partial_cmp`.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let a: SkipList<i32> = [1, 2, 3].into();
    /// let b: SkipList<i32> = [1, 2, 4].into();
    /// assert!(a < b);
    /// ```
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: Ord, G: LevelGenerator, const N: usize> Ord for SkipList<T, N, G> {
    /// Compares two lists lexicographically.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let a: SkipList<i32> = [1, 2, 3].into();
    /// let b: SkipList<i32> = [1, 2, 4].into();
    /// assert_eq!(a.cmp(&b), std::cmp::Ordering::Less);
    /// ```
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

// MARK: PartialEq / Eq

impl<T: PartialEq, G: LevelGenerator, G2: LevelGenerator, const N: usize>
    PartialEq<SkipList<T, N, G2>> for SkipList<T, N, G>
{
    /// Returns `true` if `self` and `other` have the same length and all
    /// corresponding elements compare equal.
    ///
    /// The level generators (`G` and `G2`) do not need to match.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let a: SkipList<i32> = [1, 2, 3].into();
    /// let b: SkipList<i32> = [1, 2, 3].into();
    /// assert_eq!(a, b);
    ///
    /// let c: SkipList<i32> = [1, 2, 4].into();
    /// assert_ne!(a, c);
    /// ```
    #[inline]
    fn eq(&self, other: &SkipList<T, N, G2>) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Eq, G: LevelGenerator, const N: usize> Eq for SkipList<T, N, G> {}

// MARK: Hash

impl<T: Hash, G: LevelGenerator, const N: usize> Hash for SkipList<T, N, G> {
    /// Hashes the length followed by each element in order.
    ///
    /// Two lists with the same elements in the same order produce the same
    /// hash value. This is consistent with the behavior of slices and
    /// `Vec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::{collections::hash_map::DefaultHasher, hash::{Hash, Hasher}};
    /// use skiplist::SkipList;
    ///
    /// let a: SkipList<i32> = [1, 2, 3].into();
    /// let b: SkipList<i32> = [1, 2, 3].into();
    ///
    /// let hash = |list: &SkipList<i32>| {
    ///     let mut h = DefaultHasher::new();
    ///     list.hash(&mut h);
    ///     h.finish()
    /// };
    ///
    /// assert_eq!(hash(&a), hash(&b));
    /// ```
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.len().hash(state);
        for item in self {
            item.hash(state);
        }
    }
}

// MARK: Extend

impl<T, G: LevelGenerator, const N: usize> Extend<T> for SkipList<T, N, G> {
    /// Appends all items from `iter` to the back of the list.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut list: SkipList<i32> = [1, 2, 3, 4, 5].into_iter().take(2).collect();
    /// list.extend([3, 4, 5]);
    /// let got: Vec<i32> = list.into_iter().collect();
    /// assert_eq!(got, [1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for item in iter {
            self.push_back(item);
        }
    }
}

impl<'a, T: Copy + 'a, G: LevelGenerator, const N: usize> Extend<&'a T> for SkipList<T, N, G> {
    /// Copies all items from `iter` and appends them to the back of the list.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut list: SkipList<i32> = [1, 2].into_iter().collect();
    /// let source = [3, 4, 5];
    /// list.extend(source.iter());
    /// let got: Vec<i32> = list.into_iter().collect();
    /// assert_eq!(got, [1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a T>,
    {
        self.extend(iter.into_iter().copied());
    }
}

// MARK: FromIterator / From

impl<T, G: LevelGenerator + Default, const N: usize> FromIterator<T> for SkipList<T, N, G> {
    /// Creates a list from an iterator by appending each item to the back.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let list: SkipList<i32> = [1, 2, 3].into_iter().collect();
    /// assert_eq!(list.iter().copied().collect::<Vec<_>>(), [1, 2, 3]);
    /// ```
    #[inline]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut list = Self::with_level_generator(G::default());
        list.extend(iter);
        list
    }
}

impl<T, G: LevelGenerator + Default, const N: usize, const M: usize> From<[T; M]>
    for SkipList<T, N, G>
{
    /// Creates a list from a fixed-size array.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let list = SkipList::<i32>::from([10, 20, 30]);
    /// assert_eq!(list.len(), 3);
    /// ```
    #[inline]
    fn from(arr: [T; M]) -> Self {
        arr.into_iter().collect()
    }
}

impl<T, G: LevelGenerator + Default, const N: usize> From<Vec<T>> for SkipList<T, N, G> {
    /// Creates a list from a `Vec`, consuming it.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let list = SkipList::<i32>::from(vec![7, 8, 9]);
    /// assert_eq!(list.len(), 3);
    /// ```
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        vec.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::{assert_eq, assert_ne};

    use super::super::SkipList;
    use crate::{level_generator::LevelGenerator, node::Node, test_util::seeded_geometric};

    // MARK: PartialOrd / Ord

    #[test]
    fn ord_equal_lists() {
        let mut a = SkipList::<i32>::new();
        let mut b = SkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.push_back(i);
            b.push_back(i);
        }
        assert_eq!(a.cmp(&b), core::cmp::Ordering::Equal);
    }

    #[test]
    fn ord_shorter_is_less() {
        let mut a = SkipList::<i32>::new();
        let mut b = SkipList::<i32>::new();
        for i in [1, 2] {
            a.push_back(i);
        }
        for i in [1, 2, 3] {
            b.push_back(i);
        }
        assert!(a < b);
    }

    #[test]
    fn ord_earlier_element_wins() {
        let mut a = SkipList::<i32>::new();
        let mut b = SkipList::<i32>::new();
        for i in [1, 3] {
            a.push_back(i);
        }
        for i in [1, 2, 99] {
            b.push_back(i);
        }
        assert!(a > b);
    }

    #[test]
    fn partial_ord_empty_equal() {
        let a = SkipList::<i32>::new();
        let b = SkipList::<i32>::new();
        assert_eq!(a.partial_cmp(&b), Some(core::cmp::Ordering::Equal));
    }

    // MARK: FromIterator / From

    #[test]
    fn from_iterator_empty() {
        let list: SkipList<i32> = core::iter::empty().collect();
        assert!(list.is_empty());
    }

    #[test]
    fn from_iterator_elements() {
        let list: SkipList<i32> = [1, 2, 3, 4, 5].into_iter().collect();
        let got: Vec<i32> = list.into_iter().collect();
        assert_eq!(got, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn from_array() {
        let list = SkipList::<i32>::from([10, 20, 30]);
        let got: Vec<i32> = list.into_iter().collect();
        assert_eq!(got, [10, 20, 30]);
    }

    #[test]
    fn from_array_empty() {
        #[expect(clippy::as_conversions, reason = "safe conversion of empty list")]
        let list = SkipList::<i32>::from([] as [i32; 0]);
        assert!(list.is_empty());
    }

    #[test]
    fn from_vec() {
        let v = vec![7, 8, 9];
        let list = SkipList::<i32>::from(v);
        let got: Vec<i32> = list.into_iter().collect();
        assert_eq!(got, [7, 8, 9]);
    }

    #[test]
    fn from_vec_empty() {
        let list = SkipList::<i32>::from(Vec::<i32>::new());
        assert!(list.is_empty());
    }

    // MARK: Extend

    #[test]
    fn extend_owned_appends_elements() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.extend([2, 3, 4]);
        let got: Vec<i32> = list.into_iter().collect();
        assert_eq!(got, [1, 2, 3, 4]);
    }

    #[test]
    fn extend_owned_empty_iter() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        #[expect(clippy::as_conversions, reason = "safe conversion of empty list")]
        list.extend([] as [i32; 0]);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn extend_refs_copies_elements() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        let source = [20, 30, 40];
        list.extend(source.iter());
        let got: Vec<i32> = list.into_iter().collect();
        assert_eq!(got, [10, 20, 30, 40]);
    }

    // MARK: Hash

    #[test]
    fn hash_equal_lists_same_hash() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut a = SkipList::<i32>::new();
        let mut b = SkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.push_back(i);
            b.push_back(i);
        }
        let hash_a = {
            let mut h = DefaultHasher::new();
            a.hash(&mut h);
            h.finish()
        };
        let hash_b = {
            let mut h = DefaultHasher::new();
            b.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn hash_different_orders_differ() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut a = SkipList::<i32>::new();
        let mut b = SkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.push_back(i);
        }
        for i in [3, 2, 1] {
            b.push_back(i);
        }
        let hash_a = {
            let mut h = DefaultHasher::new();
            a.hash(&mut h);
            h.finish()
        };
        let hash_b = {
            let mut h = DefaultHasher::new();
            b.hash(&mut h);
            h.finish()
        };
        assert_ne!(hash_a, hash_b);
    }

    // MARK: PartialEq / Eq

    #[test]
    fn eq_empty_lists() {
        let a = SkipList::<i32>::new();
        let b = SkipList::<i32>::new();
        assert_eq!(a, b);
    }

    #[test]
    fn eq_same_elements() {
        let mut a = SkipList::<i32>::new();
        let mut b = SkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.push_back(i);
            b.push_back(i);
        }
        assert_eq!(a, b);
    }

    #[test]
    fn ne_different_elements() {
        let mut a = SkipList::<i32>::new();
        let mut b = SkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.push_back(i);
        }
        for i in [1, 2, 4] {
            b.push_back(i);
        }
        assert_ne!(a, b);
    }

    #[test]
    fn ne_different_lengths() {
        let mut a = SkipList::<i32>::new();
        let mut b = SkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.push_back(i);
        }
        for i in [1, 2] {
            b.push_back(i);
        }
        assert_ne!(a, b);
    }

    // MARK: Clone

    #[test]
    fn clone_empty() {
        let list = SkipList::<i32>::new();
        let cloned = list.clone();
        assert!(cloned.is_empty());
    }

    #[test]
    fn clone_elements() {
        let mut list = SkipList::<i32>::new();
        for i in [1, 2, 3, 4, 5] {
            list.push_back(i);
        }
        let cloned = list.clone();
        let got: Vec<i32> = cloned.into_iter().collect();
        assert_eq!(got, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn clone_is_independent() {
        let mut list = SkipList::<i32>::new();
        for i in [10, 20, 30] {
            list.push_back(i);
        }
        let mut cloned = list.clone();
        cloned.push_back(40);
        assert_eq!(list.len(), 3);
        assert_eq!(cloned.len(), 4);
    }

    // Heights of every data node, in list order.
    fn heights(list: &SkipList<i32, 16, impl LevelGenerator>) -> Vec<usize> {
        let mut out = Vec::new();
        let mut node = list.head_ref().next_as_ref();
        while let Some(n) = node {
            out.push(n.level());
            node = n.next_as_ref();
        }
        out
    }

    #[test]
    fn clone_keeps_tower_heights() {
        let mut list = SkipList::<i32, 16, _>::with_level_generator(seeded_geometric(7));
        for i in 0..200 {
            list.push_back(i);
        }

        let cloned = list.clone();
        let original = heights(&list);
        assert!(
            original.iter().any(|&h| h > 1),
            "the fixture has tall towers"
        );
        assert!(original.contains(&0), "the fixture has towerless nodes");
        assert_eq!(heights(&cloned), original);
        // SAFETY: `cloned.head` is the live head of `cloned`, whose nodes are
        // all live.
        unsafe { Node::assert_well_formed(cloned.head, cloned.len, cloned.tail) };
    }

    // MARK: Debug

    #[test]
    fn debug_empty() {
        let list = SkipList::<i32>::new();
        assert_eq!(format!("{list:?}"), "[]");
    }

    #[test]
    fn debug_single() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        assert_eq!(format!("{list:?}"), "[42]");
    }

    #[test]
    fn debug_multiple() {
        let mut list = SkipList::<i32>::new();
        for i in [1, 2, 3] {
            list.push_back(i);
        }
        assert_eq!(format!("{list:?}"), "[1, 2, 3]");
    }

    #[test]
    fn debug_string_elements() {
        let mut list = SkipList::<&str>::new();
        for s in ["hello", "world"] {
            list.push_back(s);
        }
        assert_eq!(format!("{list:?}"), r#"["hello", "world"]"#);
    }
}
