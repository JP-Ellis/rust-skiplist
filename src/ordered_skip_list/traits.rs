//! Standard trait implementations for
//! [`OrderedSkipList`](super::OrderedSkipList):
//! `Debug`, `Clone`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash`,
//! `Extend`, `FromIterator`, and `From`.

use core::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    ptr::NonNull,
};

use crate::{
    comparator::Comparator, level_generator::LevelGenerator, node::Node,
    ordered_skip_list::OrderedSkipList,
};

// MARK: Debug

impl<T: fmt::Debug, C: Comparator<T>, G: LevelGenerator, const N: usize> fmt::Debug
    for OrderedSkipList<T, N, C, G>
{
    /// Formats the list as a sequence of elements in sorted order.
    ///
    /// The output uses the standard debug-list format, for example `[1, 2, 3]`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(3);
    /// list.insert(1);
    /// list.insert(2);
    /// assert_eq!(format!("{list:?}"), "[1, 2, 3]");
    /// ```
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// MARK: Clone

impl<T: Clone, C: Comparator<T> + Clone, G: LevelGenerator + Clone, const N: usize> Clone
    for OrderedSkipList<T, N, C, G>
{
    /// Returns a deep clone of the list.
    ///
    /// The cloned list contains the same elements in the same sorted order.
    /// The comparator and level generator are both cloned, so the new list
    /// behaves identically to the original.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(3);
    /// list.insert(1);
    /// list.insert(2);
    ///
    /// let cloned = list.clone();
    /// assert_eq!(list, cloned);
    ///
    /// // The two lists are independent.
    /// list.insert(4);
    /// assert_ne!(list, cloned);
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
        let max_levels = self.head_ref().level();

        // SAFETY: Box::into_raw transfers ownership; freed in Drop.
        let new_head =
            unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Node::new(max_levels)))) };
        let mut new_list = Self {
            head: new_head,
            tail: None,
            len: self.len,
            comparator: self.comparator.clone(),
            generator: self.generator.clone(),
        };

        if self.is_empty() {
            return new_list;
        }

        // Walk self's sequential next chain, cloning each node at the same
        // tower height.  insert_after wires the prev/next chain; rebuild()
        // will wire all skip links in a single subsequent pass.
        //
        // SAFETY: self.head is a valid, live head sentinel.  Every src_nn is a
        // live data node owned by self.  new_head / prev_nn are exclusively
        // owned by new_list and have no other live references.
        let tail = unsafe {
            let mut prev_nn = new_head;
            let mut src_opt = self.head.as_ref().next();

            while let Some(src_nn) = src_opt {
                let src_node = src_nn.as_ref();
                let height = src_node.level();
                let val = src_node.value().expect("data node has a value").clone();
                // insert_after requires the inserted node to be detached; Node::with_value
                // creates a detached node (prev=None, next=None, all links=None).
                prev_nn = Node::insert_after(prev_nn, Node::with_value(height, val));
                src_opt = src_node.next();
            }

            Node::rebuild(new_head)
        };
        new_list.tail = tail;

        new_list
    }
}

// MARK: PartialEq / Eq

impl<
    T: PartialEq,
    C1: Comparator<T>,
    C2: Comparator<T>,
    G1: LevelGenerator,
    G2: LevelGenerator,
    const N: usize,
> PartialEq<OrderedSkipList<T, N, C2, G2>> for OrderedSkipList<T, N, C1, G1>
{
    /// Returns `true` if `self` and `other` have the same length and all
    /// corresponding elements compare equal.
    ///
    /// The comparators (`C1` and `C2`) and level generators (`G1` and `G2`)
    /// do not need to match.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut a = OrderedSkipList::<i32>::new();
    /// let mut b = OrderedSkipList::<i32>::new();
    /// for i in [1, 2, 3] {
    ///     a.insert(i);
    ///     b.insert(i);
    /// }
    /// assert_eq!(a, b);
    ///
    /// a.insert(4);
    /// assert_ne!(a, b);
    /// ```
    #[inline]
    fn eq(&self, other: &OrderedSkipList<T, N, C2, G2>) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Eq, C: Comparator<T>, G: LevelGenerator, const N: usize> Eq
    for OrderedSkipList<T, N, C, G>
{
}

// MARK: PartialOrd / Ord

impl<T: PartialOrd, C: Comparator<T>, G: LevelGenerator, const N: usize> PartialOrd
    for OrderedSkipList<T, N, C, G>
{
    /// Compares two lists lexicographically by element value.
    ///
    /// Returns `None` if any pair of corresponding elements returns `None`
    /// from their own `partial_cmp`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let a: OrderedSkipList<i32> = [1, 2, 3].into();
    /// let b: OrderedSkipList<i32> = [1, 2, 4].into();
    /// assert!(a < b);
    /// ```
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: Ord, C: Comparator<T>, G: LevelGenerator, const N: usize> Ord
    for OrderedSkipList<T, N, C, G>
{
    /// Compares two lists lexicographically by element value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let a: OrderedSkipList<i32> = [1, 2].into();
    /// let b: OrderedSkipList<i32> = [1, 2, 3].into();
    /// assert!(a < b);
    /// ```
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

// MARK: Hash

impl<T: Hash, C: Comparator<T>, G: LevelGenerator, const N: usize> Hash
    for OrderedSkipList<T, N, C, G>
{
    /// Hashes the length followed by each element in sorted order.
    ///
    /// This matches the convention used by slices: two lists with the same
    /// elements produce the same hash value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    /// use std::collections::hash_map::DefaultHasher;
    /// use std::hash::{Hash, Hasher};
    ///
    /// let a: OrderedSkipList<i32> = [1, 2, 3].into();
    /// let b: OrderedSkipList<i32> = [1, 2, 3].into();
    ///
    /// let hash = |list: &OrderedSkipList<i32>| {
    ///     let mut h = DefaultHasher::new();
    ///     list.hash(&mut h);
    ///     h.finish()
    /// };
    ///
    /// assert_eq!(hash(&a), hash(&b));
    /// ```
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for item in self {
            item.hash(state);
        }
    }
}

// MARK: Extend

impl<T, C: Comparator<T>, G: LevelGenerator, const N: usize> Extend<T>
    for OrderedSkipList<T, N, C, G>
{
    /// Inserts all items from `iter` into their sorted positions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(1);
    /// list.extend([4, 2, 3]);
    /// assert_eq!(list.iter().copied().collect::<Vec<_>>(), [1, 2, 3, 4]);
    /// ```
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<'a, T: Copy + 'a, C: Comparator<T>, G: LevelGenerator, const N: usize> Extend<&'a T>
    for OrderedSkipList<T, N, C, G>
{
    /// Copies all items from `iter` and inserts them into their sorted
    /// positions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let mut list = OrderedSkipList::<i32>::new();
    /// list.insert(10);
    /// let source = [30, 20, 40];
    /// list.extend(source.iter());
    /// assert_eq!(list.iter().copied().collect::<Vec<_>>(), [10, 20, 30, 40]);
    /// ```
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

// MARK: FromIterator / From

impl<T, C: Comparator<T> + Default, G: LevelGenerator + Default, const N: usize> FromIterator<T>
    for OrderedSkipList<T, N, C, G>
{
    /// Creates a sorted list from an iterator.
    ///
    /// Elements are inserted in sorted order regardless of the iteration order
    /// of the source.  The list uses the default comparator and level generator
    /// for the type parameters `C` and `G`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// // Default OrdComparator ordering.
    /// let list: OrderedSkipList<i32> = [3, 1, 2].into_iter().collect();
    /// assert_eq!(list.iter().copied().collect::<Vec<_>>(), [1, 2, 3]);
    /// ```
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut list = Self::with_comparator_and_level_generator(C::default(), G::default());
        list.extend(iter);
        list
    }
}

impl<T, C: Comparator<T> + Default, G: LevelGenerator + Default, const N: usize, const M: usize>
    From<[T; M]> for OrderedSkipList<T, N, C, G>
{
    /// Creates a sorted list from a fixed-size array.
    ///
    /// Uses the default comparator and level generator for `C` and `G`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let list = OrderedSkipList::<i32>::from([30, 10, 20]);
    /// assert_eq!(list.iter().copied().collect::<Vec<_>>(), [10, 20, 30]);
    /// ```
    #[inline]
    fn from(arr: [T; M]) -> Self {
        arr.into_iter().collect()
    }
}

impl<T, C: Comparator<T> + Default, G: LevelGenerator + Default, const N: usize> From<Vec<T>>
    for OrderedSkipList<T, N, C, G>
{
    /// Creates a sorted list from a `Vec`, consuming it.
    ///
    /// Uses the default comparator and level generator for `C` and `G`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::ordered_skip_list::OrderedSkipList;
    ///
    /// let list = OrderedSkipList::<i32>::from(vec![9, 7, 8]);
    /// assert_eq!(list.iter().copied().collect::<Vec<_>>(), [7, 8, 9]);
    /// ```
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        vec.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::{assert_eq, assert_ne};

    use super::super::OrderedSkipList;
    use crate::comparator::FnComparator;

    // MARK: Debug

    #[test]
    fn debug_empty() {
        let list = OrderedSkipList::<i32>::new();
        assert_eq!(format!("{list:?}"), "[]");
    }

    #[test]
    fn debug_single() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(42);
        assert_eq!(format!("{list:?}"), "[42]");
    }

    #[test]
    fn debug_multiple() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [3, 1, 2] {
            list.insert(i);
        }
        // Stored in sorted order.
        assert_eq!(format!("{list:?}"), "[1, 2, 3]");
    }

    #[test]
    fn debug_string_elements() {
        let mut list = OrderedSkipList::<&str>::new();
        for s in ["world", "hello"] {
            list.insert(s);
        }
        assert_eq!(format!("{list:?}"), r#"["hello", "world"]"#);
    }

    // MARK: Clone

    #[test]
    fn clone_empty() {
        let list = OrderedSkipList::<i32>::new();
        let cloned = list.clone();
        assert!(cloned.is_empty());
    }

    #[test]
    fn clone_elements() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [3, 1, 4, 1, 5] {
            list.insert(i);
        }
        let cloned = list.clone();
        let got: Vec<i32> = cloned.into_iter().collect();
        assert_eq!(got, [1, 1, 3, 4, 5]);
    }

    #[test]
    fn clone_is_independent() {
        let mut list = OrderedSkipList::<i32>::new();
        for i in [10, 20, 30] {
            list.insert(i);
        }
        let mut cloned = list.clone();
        cloned.insert(40);
        assert_eq!(list.len(), 3);
        assert_eq!(cloned.len(), 4);
    }

    #[test]
    fn clone_custom_comparator() {
        let f: fn(&i32, &i32) -> core::cmp::Ordering = |a, b| b.cmp(a);
        let mut list: OrderedSkipList<i32, 16, _> =
            OrderedSkipList::with_comparator(FnComparator(f));
        for i in [1, 2, 3] {
            list.insert(i);
        }
        let cloned = list.clone();
        let got: Vec<i32> = cloned.into_iter().collect();
        // Reverse ordering: [3, 2, 1].
        assert_eq!(got, [3, 2, 1]);
    }

    // MARK: PartialEq / Eq

    #[test]
    fn eq_empty_lists() {
        let a = OrderedSkipList::<i32>::new();
        let b = OrderedSkipList::<i32>::new();
        assert_eq!(a, b);
    }

    #[test]
    fn eq_same_elements() {
        let mut a = OrderedSkipList::<i32>::new();
        let mut b = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.insert(i);
            b.insert(i);
        }
        assert_eq!(a, b);
    }

    #[test]
    fn ne_different_elements() {
        let mut a = OrderedSkipList::<i32>::new();
        let mut b = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.insert(i);
        }
        for i in [1, 2, 4] {
            b.insert(i);
        }
        assert_ne!(a, b);
    }

    #[test]
    fn ne_different_lengths() {
        let mut a = OrderedSkipList::<i32>::new();
        let mut b = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.insert(i);
        }
        for i in [1, 2] {
            b.insert(i);
        }
        assert_ne!(a, b);
    }

    // MARK: PartialOrd / Ord

    #[test]
    fn ord_equal_lists() {
        let mut a = OrderedSkipList::<i32>::new();
        let mut b = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.insert(i);
            b.insert(i);
        }
        assert_eq!(a.cmp(&b), core::cmp::Ordering::Equal);
    }

    #[test]
    fn ord_shorter_is_less() {
        let mut a = OrderedSkipList::<i32>::new();
        let mut b = OrderedSkipList::<i32>::new();
        for i in [1, 2] {
            a.insert(i);
        }
        for i in [1, 2, 3] {
            b.insert(i);
        }
        assert!(a < b);
    }

    #[test]
    fn ord_earlier_element_wins() {
        let mut a = OrderedSkipList::<i32>::new();
        let mut b = OrderedSkipList::<i32>::new();
        for i in [1, 3] {
            a.insert(i);
        }
        for i in [1, 2, 99] {
            b.insert(i);
        }
        assert!(a > b);
    }

    #[test]
    fn partial_ord_empty_equal() {
        let a = OrderedSkipList::<i32>::new();
        let b = OrderedSkipList::<i32>::new();
        assert_eq!(a.partial_cmp(&b), Some(core::cmp::Ordering::Equal));
    }

    // MARK: Hash

    #[test]
    fn hash_equal_lists_same_hash() {
        use std::{
            collections::hash_map::DefaultHasher,
            hash::{Hash, Hasher},
        };

        let mut a = OrderedSkipList::<i32>::new();
        let mut b = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.insert(i);
            b.insert(i);
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
    fn hash_different_elements_differ() {
        use std::{
            collections::hash_map::DefaultHasher,
            hash::{Hash, Hasher},
        };

        let mut a = OrderedSkipList::<i32>::new();
        let mut b = OrderedSkipList::<i32>::new();
        for i in [1, 2, 3] {
            a.insert(i);
        }
        for i in [1, 2, 4] {
            b.insert(i);
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

    // MARK: Extend

    #[test]
    fn extend_owned_inserts_sorted() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        list.extend([4, 2, 3]);
        let got: Vec<i32> = list.into_iter().collect();
        assert_eq!(got, [1, 2, 3, 4]);
    }

    #[test]
    fn extend_owned_empty_iter() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(1);
        #[expect(clippy::as_conversions, reason = "safe conversion of empty array")]
        list.extend([] as [i32; 0]);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn extend_refs_copies_elements() {
        let mut list = OrderedSkipList::<i32>::new();
        list.insert(10);
        let source = [30, 20, 40];
        list.extend(source.iter());
        let got: Vec<i32> = list.into_iter().collect();
        assert_eq!(got, [10, 20, 30, 40]);
    }

    // MARK: FromIterator / From

    #[test]
    fn from_iterator_empty() {
        let list: OrderedSkipList<i32> = core::iter::empty().collect();
        assert!(list.is_empty());
    }

    #[test]
    fn from_iterator_elements_sorted() {
        let list: OrderedSkipList<i32> = [3, 1, 4, 1, 5].into_iter().collect();
        let got: Vec<i32> = list.into_iter().collect();
        assert_eq!(got, [1, 1, 3, 4, 5]);
    }

    #[test]
    fn from_array() {
        let list = OrderedSkipList::<i32>::from([30, 10, 20]);
        let got: Vec<i32> = list.into_iter().collect();
        assert_eq!(got, [10, 20, 30]);
    }

    #[test]
    fn from_array_empty() {
        #[expect(clippy::as_conversions, reason = "safe conversion of empty array")]
        let list = OrderedSkipList::<i32>::from([] as [i32; 0]);
        assert!(list.is_empty());
    }

    #[test]
    fn from_vec() {
        let v = vec![9, 7, 8];
        let list = OrderedSkipList::<i32>::from(v);
        let got: Vec<i32> = list.into_iter().collect();
        assert_eq!(got, [7, 8, 9]);
    }

    #[test]
    fn from_vec_empty() {
        let list = OrderedSkipList::<i32>::from(Vec::<i32>::new());
        assert!(list.is_empty());
    }

    #[test]
    fn from_iterator_works_with_default_comparator() {
        // The blanket FromIterator impl covers any C: Comparator<T> + Default.
        // OrdComparator satisfies this bound; verify collect() still works.
        let list: OrderedSkipList<i32> = [3, 1, 2].into_iter().collect();
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    fn from_array_blanket_via_collect() {
        // From<[T; M]> delegates to collect(), so it also goes through the
        // blanket FromIterator impl.
        let list: OrderedSkipList<i32> = [5, 3, 4].into();
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), [3, 4, 5]);
    }

    #[test]
    fn from_vec_blanket_via_collect() {
        // Same for From<Vec<T>>.
        let list: OrderedSkipList<i32> = vec![9, 7, 8].into();
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), [7, 8, 9]);
    }
}
