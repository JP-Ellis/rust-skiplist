//! Element access methods for [`SkipList`](super::SkipList).
//!
//! Provides `get`, `get_mut`, `front`, `back`, `front_mut`, `back_mut`, and
//! the [`Index`](core::ops::Index) / [`IndexMut`](core::ops::IndexMut)
//! operators for position-based element access.

use core::ops::{Index, IndexMut};

use crate::{
    level_generator::LevelGenerator,
    node::{
        Node,
        visitor::{IndexVisitor, Visitor},
    },
    skip_list::SkipList,
};

impl<T, G: LevelGenerator, const N: usize> SkipList<T, N, G> {
    /// Returns a reference to the element at position `index`, or `None` if
    /// `index` is out of bounds.
    ///
    /// This operation is `$O(\log n)$` expected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(10);
    /// list.push_back(20);
    /// list.push_back(30);
    /// assert_eq!(list.get(0), Some(&10));
    /// assert_eq!(list.get(1), Some(&20));
    /// assert_eq!(list.get(2), Some(&30));
    /// assert_eq!(list.get(3), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        IndexVisitor::new(self.head_ref(), index.saturating_add(1))
            .traverse()
            .and_then(|node| node.value())
    }

    /// Returns a mutable reference to the element at position `index`, or
    /// `None` if `index` is out of bounds.
    ///
    /// This operation is `$O(\log n)$` expected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(10);
    /// list.push_back(20);
    /// if let Some(v) = list.get_mut(0) {
    ///     *v = 42;
    /// }
    /// assert_eq!(list.get(0), Some(&42));
    /// assert_eq!(list.get(1), Some(&20));
    /// ```
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        // SAFETY: index < self.len, so node_ptr_at returns a valid data node.
        // &mut self guarantees exclusive access; no other reference to this node
        // can exist.  The returned &mut T is bounded by &mut self's lifetime.
        unsafe { (*self.node_ptr_at(index).as_ptr()).value_mut() }
    }

    /// Returns a reference to the first element, or `None` if the list is
    /// empty.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// assert_eq!(list.front(), None);
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.front(), Some(&1));
    /// ```
    #[inline]
    #[must_use]
    pub fn front(&self) -> Option<&T> {
        self.head_ref().next_as_ref()?.value()
    }

    /// Returns a mutable reference to the first element, or `None` if the
    /// list is empty.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(10);
    /// list.push_back(20);
    /// if let Some(v) = list.front_mut() {
    ///     *v = 99;
    /// }
    /// assert_eq!(list.front(), Some(&99));
    /// assert_eq!(list.get(1), Some(&20));
    /// ```
    #[inline]
    #[must_use]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        // SAFETY: `&mut self` guarantees exclusive access; `as_ref()` on the
        // head sentinel is valid for its lifetime.  `next()` returns the
        // stored `NonNull` directly, avoiding a Frozen provenance tag under
        // Tree Borrows.  The `?` propagates `None` when the list is empty.
        let front_ptr: *mut Node<T, N> = unsafe { self.head.as_ref().next()?.as_ptr() };
        // SAFETY: `front_ptr` is a live, exclusively-owned node derived
        // immediately above; the returned `&mut T` is bounded by `&mut self`.
        unsafe { (*front_ptr).value_mut() }
    }

    /// Returns a reference to the last element, or `None` if the list is
    /// empty.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// assert_eq!(list.back(), None);
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.back(), Some(&2));
    /// ```
    #[inline]
    #[must_use]
    pub fn back(&self) -> Option<&T> {
        // SAFETY: self.tail is Some iff len > 0, an invariant maintained by all
        // mutating operations.  The pointer remains valid for the lifetime of &self.
        unsafe { self.tail?.as_ref().value() }
    }

    /// Returns a mutable reference to the last element, or `None` if the list
    /// is empty.
    ///
    /// This operation is `$O(1)$`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(10);
    /// list.push_back(20);
    /// if let Some(v) = list.back_mut() {
    ///     *v = 99;
    /// }
    /// assert_eq!(list.back(), Some(&99));
    /// assert_eq!(list.get(0), Some(&10));
    /// ```
    #[inline]
    #[must_use]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        // SAFETY: self.tail is Some iff len > 0, an invariant maintained by all
        // mutating operations.  &mut self guarantees exclusive access, so no
        // other reference to the tail node exists.  The returned &mut T is
        // bounded by &mut self's lifetime.
        unsafe { self.tail?.as_mut().value_mut() }
    }
}

// MARK: Index

impl<T, G: LevelGenerator, const N: usize> Index<usize> for SkipList<T, N, G> {
    type Output = T;

    /// Returns a reference to the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(10);
    /// list.push_back(20);
    /// list.push_back(30);
    /// assert_eq!(list[0], 10);
    /// assert_eq!(list[2], 30);
    /// ```
    #[inline]
    #[expect(
        clippy::unwrap_used,
        reason = "index < self.len was just asserted, so get() always returns Some"
    )]
    fn index(&self, index: usize) -> &T {
        assert!(
            index < self.len,
            "index out of bounds: the len is {} but the index is {index}",
            self.len
        );
        self.get(index).unwrap()
    }
}

impl<T, G: LevelGenerator, const N: usize> IndexMut<usize> for SkipList<T, N, G> {
    /// Returns a mutable reference to the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use skiplist::skip_list::SkipList;
    ///
    /// let mut list = SkipList::<i32>::new();
    /// list.push_back(10);
    /// list.push_back(20);
    /// list[0] = 99;
    /// assert_eq!(list[0], 99);
    /// assert_eq!(list[1], 20);
    /// ```
    #[inline]
    #[expect(
        clippy::unwrap_used,
        reason = "index < self.len was just asserted, so get_mut() always returns Some"
    )]
    fn index_mut(&mut self, index: usize) -> &mut T {
        assert!(
            index < self.len,
            "index out of bounds: the len is {} but the index is {index}",
            self.len
        );
        self.get_mut(index).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::super::SkipList;

    // MARK: get / get_mut

    #[test]
    fn get_from_empty() {
        let list = SkipList::<i32>::new();
        assert_eq!(list.get(0), None);
    }

    #[test]
    fn get_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.get(3), None);
        assert_eq!(list.get(100), None);
    }

    #[test]
    fn get_first() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list.get(0), Some(&10));
    }

    #[test]
    fn get_last() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list.get(2), Some(&30));
    }

    #[test]
    fn get_middle() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list.get(1), Some(&20));
    }

    #[test]
    fn get_all_elements() {
        let n = 50_usize;
        let mut list = SkipList::<usize>::new();
        for i in 0..n {
            list.push_back(i);
        }
        for i in 0..n {
            assert_eq!(list.get(i), Some(&i));
        }
    }

    #[test]
    fn get_single_element() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(42);
        assert_eq!(list.get(0), Some(&42));
        assert_eq!(list.get(1), None);
    }

    #[test]
    fn get_mut_from_empty() {
        let mut list = SkipList::<i32>::new();
        assert_eq!(list.get_mut(0), None);
    }

    #[test]
    fn get_mut_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        assert_eq!(list.get_mut(1), None);
        assert_eq!(list.get_mut(99), None);
    }

    #[test]
    fn get_mut_modify() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);

        if let Some(v) = list.get_mut(0) {
            *v = 100;
        }
        assert_eq!(list.get(0), Some(&100));
        assert_eq!(list.get(1), Some(&20));
        assert_eq!(list.get(2), Some(&30));

        if let Some(v) = list.get_mut(2) {
            *v = 300;
        }
        assert_eq!(list.get(0), Some(&100));
        assert_eq!(list.get(1), Some(&20));
        assert_eq!(list.get(2), Some(&300));

        if let Some(v) = list.get_mut(1) {
            *v = 200;
        }
        assert_eq!(list.get(0), Some(&100));
        assert_eq!(list.get(1), Some(&200));
        assert_eq!(list.get(2), Some(&300));
    }

    #[test]
    fn get_mut_all_elements() {
        let n = 30_usize;
        let mut list = SkipList::<usize>::new();
        for i in 0..n {
            list.push_back(i);
        }
        for i in 0..n {
            if let Some(v) = list.get_mut(i) {
                *v = i * 10;
            }
        }
        for i in 0..n {
            assert_eq!(list.get(i), Some(&(i * 10)));
        }
    }

    // MARK: Index / IndexMut

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing valid indexing behavior with known indices"
    )]
    fn index_basic() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list[0], 10);
        assert_eq!(list[1], 20);
        assert_eq!(list[2], 30);
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing valid mut indexing behavior with known indices"
    )]
    fn index_mut_basic() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        list[0] = 100;
        list[2] = 300;
        assert_eq!(list[0], 100);
        assert_eq!(list[1], 20);
        assert_eq!(list[2], 300);
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing out-of-bounds indexing behavior with known indices"
    )]
    #[should_panic(expected = "index out of bounds: the len is 3 but the index is 3")]
    fn index_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        _ = list[3];
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing out-of-bounds mut indexing behavior with known indices"
    )]
    #[should_panic(expected = "index out of bounds: the len is 0 but the index is 0")]
    fn index_empty() {
        let list = SkipList::<i32>::new();
        _ = list[0];
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing out-of-bounds mut indexing behavior with known indices"
    )]
    #[should_panic(expected = "index out of bounds: the len is 3 but the index is 5")]
    fn index_mut_out_of_bounds() {
        let mut list = SkipList::<i32>::with_capacity(1);
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list[5] = 99;
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "Testing valid indexing behavior after mutations with known indices"
    )]
    fn index_after_mutations() {
        let mut list = SkipList::<i32>::new();
        for i in 0..10_i32 {
            list.push_back(i);
        }
        list.remove(3); // [0,1,2,4,5,6,7,8,9]
        list.insert(3, 42); // [0,1,2,42,4,5,6,7,8,9]
        assert_eq!(list[3], 42);
        assert_eq!(list[4], 4);
    }

    // MARK: front / back

    #[test]
    fn front_empty() {
        let list = SkipList::<i32>::new();
        assert_eq!(list.front(), None);
    }

    #[test]
    fn back_empty() {
        let list = SkipList::<i32>::new();
        assert_eq!(list.back(), None);
    }

    #[test]
    fn front_single() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        assert_eq!(list.front(), Some(&42));
    }

    #[test]
    fn back_single() {
        let mut list = SkipList::<i32>::new();
        list.push_back(42);
        assert_eq!(list.back(), Some(&42));
    }

    #[test]
    fn front_and_back_are_same_for_single_element() {
        let mut list = SkipList::<i32>::new();
        list.push_back(7);
        assert_eq!(list.front(), list.back());
    }

    #[test]
    fn front_returns_first_after_push_back() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list.front(), Some(&10));
    }

    #[test]
    fn back_returns_last_after_push_back() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        assert_eq!(list.back(), Some(&30));
    }

    #[test]
    fn front_returns_first_after_push_front() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        list.push_front(99);
        assert_eq!(list.front(), Some(&99));
        assert_eq!(list.back(), Some(&10));
    }

    #[test]
    fn back_unchanged_after_push_front() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_front(0);
        // front is new element, back is still 2
        assert_eq!(list.front(), Some(&0));
        assert_eq!(list.back(), Some(&2));
    }

    #[test]
    fn front_unchanged_after_push_back() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        // front stays 1 no matter how many push_backs
        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn front_mut_modifies_first_element() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        list.push_back(20);
        *list.front_mut().expect("non-empty") = 99;
        assert_eq!(list.front(), Some(&99));
        assert_eq!(list.back(), Some(&20));
    }

    #[test]
    fn back_mut_modifies_last_element() {
        let mut list = SkipList::<i32>::new();
        list.push_back(10);
        list.push_back(20);
        *list.back_mut().expect("non-empty") = 99;
        assert_eq!(list.front(), Some(&10));
        assert_eq!(list.back(), Some(&99));
    }

    #[test]
    fn front_mut_empty_returns_none() {
        let mut list = SkipList::<i32>::new();
        assert_eq!(list.front_mut(), None);
    }

    #[test]
    fn back_mut_empty_returns_none() {
        let mut list = SkipList::<i32>::new();
        assert_eq!(list.back_mut(), None);
    }

    #[test]
    fn front_none_after_pop_front_empties_list() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.pop_front();
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn back_none_after_pop_back_empties_list() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.pop_back();
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn back_updates_after_pop_back() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.pop_back();
        assert_eq!(list.back(), Some(&2));
        list.pop_back();
        assert_eq!(list.back(), Some(&1));
    }

    #[test]
    fn front_updates_after_pop_front() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.pop_front();
        assert_eq!(list.front(), Some(&2));
        list.pop_front();
        assert_eq!(list.front(), Some(&3));
    }

    #[test]
    fn back_updates_after_insert_at_end() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.insert(2, 99); // append
        assert_eq!(list.back(), Some(&99));
    }

    #[test]
    fn back_unchanged_after_insert_in_middle() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.insert(1, 99); // middle
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn back_updates_after_remove_last() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.remove(2); // remove last
        assert_eq!(list.back(), Some(&2));
    }

    #[test]
    fn back_unchanged_after_remove_middle() {
        let mut list = SkipList::<i32>::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.remove(1); // remove middle
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn back_consistent_with_get_last() {
        let mut list = SkipList::<i32>::with_capacity(4);
        for i in 0..20_i32 {
            list.push_back(i);
        }
        assert_eq!(list.back(), list.get(list.len() - 1));
    }
}
