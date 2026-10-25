//! Fixed-capacity, per-level array with a runtime length.

use core::{
    fmt,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr, slice,
};

/// A `[T; N]` of which only the first `len` elements are in use.
///
/// Replaces a heap-free vector for per-level bookkeeping: skip-link slots
/// on a node, and precursor stacks in the mutable visitors.  Dereferences to
/// the in-use prefix, so callers index and iterate it as a slice.  Slots past
/// `len` are never initialised, so a short array costs only its prefix.
pub(crate) struct LevelArray<T, const N: usize> {
    /// Backing storage; exactly `items[..len]` is initialised.
    items: [MaybeUninit<T>; N],
    /// Number of in-use slots, always `<= N`.
    len: u8,
}

impl<T, const N: usize> LevelArray<T, N> {
    /// Builds an array whose `len` slots are `f(0)`, `f(1)`, ...
    ///
    /// # Panics
    ///
    /// Panics if `len > N`.
    #[inline]
    #[expect(
        clippy::as_conversions,
        reason = "u8::MAX widens losslessly to usize; From is not const"
    )]
    #[expect(
        clippy::expect_used,
        reason = "narrowing is guarded by the assert above"
    )]
    pub(crate) fn from_fn(len: usize, mut f: impl FnMut(usize) -> T) -> Self {
        const {
            assert!(
                N <= u8::MAX as usize,
                "LevelArray capacity must fit in a u8"
            );
        }
        assert!(len <= N, "length ({len}) exceeds capacity ({N})");
        let mut items = [const { MaybeUninit::uninit() }; N];
        // A panic in `f` leaks the slots written so far; nothing reads them.
        for (i, slot) in items.iter_mut().take(len).enumerate() {
            slot.write(f(i));
        }
        Self {
            items,
            // `len <= N <= u8::MAX` was just checked, so the narrowing cannot fail.
            len: u8::try_from(len).expect("len <= N <= u8::MAX"),
        }
    }
}

impl<T, const N: usize> Deref for LevelArray<T, N> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        let len = usize::from(self.len);
        debug_assert!(len <= N, "length ({len}) exceeds capacity ({N})");
        // SAFETY: `from_fn` initialised exactly `items[..len]`, `len <= N`,
        // and `len` is never written afterwards.
        unsafe { slice::from_raw_parts(self.items.as_ptr().cast::<T>(), len) }
    }
}

impl<T, const N: usize> DerefMut for LevelArray<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        let len = usize::from(self.len);
        debug_assert!(len <= N, "length ({len}) exceeds capacity ({N})");
        // SAFETY: as in `deref`.
        unsafe { slice::from_raw_parts_mut(self.items.as_mut_ptr().cast::<T>(), len) }
    }
}

impl<T, const N: usize> Drop for LevelArray<T, N> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: the in-use prefix is initialised and dropped only here.
        unsafe { ptr::drop_in_place(&raw mut **self) };
    }
}

impl<T: Clone, const N: usize> Clone for LevelArray<T, N> {
    #[inline]
    fn clone(&self) -> Self {
        Self::from_fn(self.len(), |i| {
            // SAFETY: `from_fn` calls this only for `i < self.len()`.
            unsafe { self.get_unchecked(i) }.clone()
        })
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for LevelArray<T, N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: PartialEq, const N: usize> PartialEq for LevelArray<T, N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Eq, const N: usize> Eq for LevelArray<T, N> {}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::LevelArray;

    #[test]
    fn from_fn_sets_len_and_items() {
        let arr: LevelArray<usize, 4> = LevelArray::from_fn(3, |i| i * 10);
        assert_eq!(arr.len(), 3);
        assert_eq!(&*arr, &[0, 10, 20]);
    }

    #[test]
    fn from_fn_zero_len_is_empty() {
        let arr: LevelArray<usize, 4> = LevelArray::from_fn(0, |i| i);
        assert!(arr.is_empty());
    }

    #[test]
    fn from_fn_full_capacity() {
        let arr: LevelArray<usize, 4> = LevelArray::from_fn(4, |i| i);
        assert_eq!(&*arr, &[0, 1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "exceeds capacity")]
    fn from_fn_over_capacity_panics() {
        let _arr: LevelArray<usize, 4> = LevelArray::from_fn(5, |i| i);
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "index 1 is within len 2")]
    fn deref_mut_writes_through() {
        let mut arr: LevelArray<usize, 4> = LevelArray::from_fn(2, |_| 0);
        arr[1] = 7;
        assert_eq!(&*arr, &[0, 7]);
    }

    #[test]
    fn debug_and_eq_use_visible_slice_only() {
        let a: LevelArray<usize, 4> = LevelArray::from_fn(2, |i| i);
        let b: LevelArray<usize, 4> = LevelArray::from_fn(2, |i| if i < 2 { i } else { 99 });
        assert_eq!(a, b);
        assert_eq!(format!("{a:?}"), "[0, 1]");
    }

    #[test]
    fn clone_preserves_len() {
        let a: LevelArray<usize, 4> = LevelArray::from_fn(3, |i| i);
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(b.len(), 3);
    }
}
