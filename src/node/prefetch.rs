//! Cache prefetch hints for traversals.

/// Hints the CPU to load the cache line holding `addr` for a read.
///
/// The hint neither faults nor changes memory, so `addr` need not be valid.
/// On targets without a hint, and under Miri, this does nothing.
///
/// # Arguments
///
/// * `addr` - An address the caller expects to read soon.
#[inline]
pub(crate) fn prefetch_read(addr: *const u8) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse", not(miri)))]
    {
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        // SAFETY: a prefetch is a hint; it never faults or writes memory.
        unsafe { _mm_prefetch::<_MM_HINT_T0>(addr.cast()) };
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    // SAFETY: `prfm` is a hint; it never faults or writes memory.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{addr}]",
            addr = in(reg) addr,
            options(nostack, readonly, preserves_flags),
        );
    }
    #[cfg(not(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(miri))))]
    let _ = addr;
}
