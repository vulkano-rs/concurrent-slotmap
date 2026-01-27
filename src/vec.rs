use core::{
    alloc::Layout,
    cell::Cell,
    fmt,
    hash::{Hash, Hasher},
    num::NonZeroUsize,
    ptr, slice,
    sync::atomic::{AtomicI32, AtomicU32, AtomicUsize, Ordering::Relaxed},
};
use std::{hash::DefaultHasher, thread};
use virtual_buffer::concurrent::vec;

/// The number of shards to use.
static SHARD_COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The thread-local shard index.
    static SHARD_INDEX: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn set_shard_count() {
    if SHARD_COUNT.load(Relaxed) == 0 {
        let num_cpus = thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(1);
        SHARD_COUNT.store(num_cpus.next_power_of_two(), Relaxed);
    }
}

#[cfg(test)]
pub(crate) fn shard_count() -> usize {
    SHARD_COUNT.load(Relaxed)
}

#[inline(never)]
pub(crate) fn set_shard_index() {
    let mut state = DefaultHasher::new();
    thread::current().id().hash(&mut state);
    let thread_id_hash = state.finish();

    let shard_count = SHARD_COUNT.load(Relaxed);

    // The index can never exceed `shard_count - 1`, which is bounded by the available parallelism.
    #[allow(clippy::cast_possible_truncation)]
    let shard_index = (thread_id_hash & (shard_count as u64 - 1)) as usize;

    SHARD_INDEX.set(shard_index);
}

pub(crate) struct RawVec<T> {
    inner: vec::RawVec<T>,
}

#[repr(transparent)]
pub(crate) struct Header {
    shards: [HeaderShard],
}

#[repr(align(128))]
pub(crate) struct HeaderShard {
    /// The list of slots which have already been dropped and are ready to be claimed by insert
    /// operations.
    pub free_list: AtomicU32,
    ///The number of occupied slots.
    pub len: AtomicI32,
}

impl<T> RawVec<T> {
    #[track_caller]
    pub unsafe fn new(max_capacity: u32) -> Self {
        let max_capacity = usize::try_from(max_capacity).unwrap();
        let shard_count = SHARD_COUNT.load(Relaxed);
        let header_layout = Layout::array::<HeaderShard>(shard_count).unwrap();

        let mut vec = RawVec {
            // SAFETY: `Slot<V>` is zeroable.
            inner: unsafe { vec::RawVec::with_header(max_capacity, header_layout) },
        };

        vec.header_mut().init();

        vec
    }

    pub unsafe fn try_new(max_capacity: u32) -> Result<Self, TryReserveError> {
        let max_capacity = usize::try_from(max_capacity).unwrap();
        let shard_count = SHARD_COUNT.load(Relaxed);
        let header_layout = Layout::array::<HeaderShard>(shard_count).unwrap();

        let mut vec = RawVec {
            // SAFETY: `Slot<V>` is zeroable.
            inner: unsafe { vec::RawVec::try_with_header(max_capacity, header_layout) }
                .map_err(TryReserveError)?,
        };

        vec.header_mut().init();

        Ok(vec)
    }

    #[inline]
    pub fn header(&self) -> &Header {
        // SAFETY: `self.inner.as_ptr()` is a valid pointer to our allocations of `Slot<V>`s.
        unsafe { &*header_ptr_from_slots(self.as_ptr().cast()) }
    }

    #[inline]
    pub fn header_mut(&mut self) -> &mut Header {
        // SAFETY: `self.inner.as_ptr()` is a valid pointer to our allocations of `Slot<V>`s.
        unsafe { &mut *header_ptr_from_slots(self.as_mut_ptr().cast()).cast_mut() }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }

    // Our capacity can never exceed `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.inner.capacity() as u32
    }

    // Our capacity can never exceed `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    pub fn capacity_mut(&mut self) -> u32 {
        self.inner.as_mut_capacity().len() as u32
    }

    #[inline]
    #[track_caller]
    pub fn push(&self) -> (u32, &T) {
        let (index, slot) = self.inner.push();

        // Our capacity can never exceed `u32::MAX`.
        #[allow(clippy::cast_possible_truncation)]
        let index = index as u32;

        (index, slot)
    }

    #[inline]
    #[track_caller]
    pub fn push_mut(&mut self) -> (u32, &mut T) {
        let (index, slot) = self.inner.push_mut();

        // Our capacity can never exceed `u32::MAX`.
        #[allow(clippy::cast_possible_truncation)]
        let index = index as u32;

        (index, slot)
    }

    #[inline]
    pub fn get(&self, index: u32) -> Option<&T> {
        let index = index as usize;

        self.inner.as_capacity().get(index)
    }

    #[inline]
    pub fn get_mut(&mut self, index: u32) -> Option<&mut T> {
        let index = index as usize;

        self.inner.as_mut_capacity().get_mut(index)
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: u32) -> &T {
        let index = index as usize;

        // SAFETY: Enforced by the caller.
        unsafe { self.inner.as_capacity().get_unchecked(index) }
    }

    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: u32) -> &mut T {
        let index = index as usize;

        // SAFETY: Enforced by the caller.
        unsafe { self.inner.as_mut_capacity().get_unchecked_mut(index) }
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.inner.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.inner.iter_mut()
    }
}

#[inline]
pub(crate) unsafe fn header_ptr_from_slots(slots: *const u8) -> *const Header {
    let shard_count = SHARD_COUNT.load(Relaxed);

    // SAFETY: The caller must ensure that `slots` is a valid pointer to the allocation of
    // `Slot<V>`s, and that allocation has the `Header` right before the start of the slots.
    let header = unsafe { slots.cast::<HeaderShard>().sub(shard_count) };

    ptr::slice_from_raw_parts(header, shard_count) as *const Header
}

impl Header {
    fn init(&mut self) {
        for shard in &mut self.shards {
            *shard.free_list.get_mut() = crate::NIL;
        }
    }

    #[inline]
    pub fn shard(&self) -> &HeaderShard {
        let shard_index = SHARD_INDEX.get();

        // SAFETY: `set_shard_index` ensures that `SHARD_INDEX` is in bounds of the shards.
        unsafe { self.shards.get_unchecked(shard_index) }
    }

    #[inline]
    pub fn shard_mut(&mut self) -> &mut HeaderShard {
        // SAFETY: There is always a non-zero number of shards.
        unsafe { self.shards.get_unchecked_mut(0) }
    }

    #[inline]
    pub fn shards(&self) -> HeaderShards<'_> {
        let shard_index = SHARD_INDEX.get();

        HeaderShards {
            shards: &self.shards,
            shard_index,
            yielded: 0,
        }
    }

    #[inline]
    pub fn shards_mut(&mut self) -> slice::IterMut<'_, HeaderShard> {
        self.shards.iter_mut()
    }

    #[inline]
    pub fn len(&self) -> u32 {
        self.shards()
            .map(|shard| shard.len.load(Relaxed))
            .sum::<i32>()
            .try_into()
            .unwrap_or(0)
    }
}

pub(crate) struct HeaderShards<'a> {
    shards: &'a [HeaderShard],
    shard_index: usize,
    yielded: usize,
}

impl<'a> Iterator for HeaderShards<'a> {
    type Item = &'a HeaderShard;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.yielded < self.shards.len() {
            let current_index = (self.shard_index + self.yielded) & (self.shards.len() - 1);
            self.yielded += 1;

            // SAFETY: `Header::shards` ensures that `current_index` starts out in bounds of the
            // shards, and we made sure that the next iteration has an index that's in bounds too.
            Some(unsafe { self.shards.get_unchecked(current_index) })
        } else {
            None
        }
    }
}

pub struct TryReserveError(virtual_buffer::vec::TryReserveError);

impl fmt::Debug for TryReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for TryReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl core::error::Error for TryReserveError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        self.0.source()
    }
}
