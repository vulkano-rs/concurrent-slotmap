use crate::OCCUPIED_BIT;
use core::{
    cell::UnsafeCell,
    cmp, fmt,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::{Deref, DerefMut},
    slice,
    sync::atomic::{
        AtomicU32, AtomicUsize,
        Ordering::{Acquire, Relaxed, Release},
    },
};
use virtual_buffer::{align_up, page_size, Allocation, Error};
use TryReserveErrorKind::{AllocError, CapacityOverflow};

pub(crate) struct Vec<T> {
    allocation: virtual_buffer::Allocation,
    max_capacity: u32,
    capacity: AtomicU32,
    reserved_len: AtomicUsize,
    marker: PhantomData<Slot<T>>,
}

impl<T> Vec<T> {
    pub fn new(max_capacity: u32) -> Self {
        handle_reserve(Self::try_new(max_capacity))
    }

    pub fn try_new(max_capacity: u32) -> Result<Self, TryReserveError> {
        assert!(mem::align_of::<Slot<T>>() <= page_size());

        let size = align_up(
            usize::try_from(max_capacity)
                .ok()
                .and_then(|cap| cap.checked_mul(mem::size_of::<Slot<T>>()))
                .ok_or(CapacityOverflow)?,
            page_size(),
        );

        #[allow(clippy::cast_possible_wrap)]
        if size > isize::MAX as usize {
            return Err(CapacityOverflow.into());
        }

        if size == 0 {
            return Ok(Self::dangling(max_capacity));
        }

        let allocation = Allocation::new(size).map_err(AllocError)?;

        Ok(Vec {
            allocation,
            max_capacity,
            capacity: AtomicU32::new(0),
            reserved_len: AtomicUsize::new(0),
            marker: PhantomData,
        })
    }

    pub const fn dangling(max_capacity: u32) -> Self {
        let allocation = Allocation::dangling(mem::align_of::<Slot<T>>());

        Vec {
            allocation,
            max_capacity,
            capacity: AtomicU32::new(0),
            reserved_len: AtomicUsize::new(0),
            marker: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &[Slot<T>] {
        let capacity = self.capacity.load(Acquire) as usize;

        // SAFETY: We know that newly-allocated pages are zeroed, so we don't need to intialize the
        // `Slot<T>` as it allows being zeroed. The `Acquire` ordering above synchronizes with the
        // `Release` ordering when setting the capacity, making sure that the reserved capacity is
        // visible here.
        unsafe { slice::from_raw_parts(self.as_ptr(), capacity) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [Slot<T>] {
        let capacity = self.capacity_mut() as usize;

        // SAFETY: We know that newly-allocated pages are zeroed, so we don't need to intialize the
        // `Slot<T>` as it allows being zeroed. The mutable reference ensures synchronization in
        // this case.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), capacity) }
    }

    pub fn as_ptr(&self) -> *const Slot<T> {
        self.allocation.ptr().cast()
    }

    pub fn as_mut_ptr(&mut self) -> *mut Slot<T> {
        self.allocation.ptr().cast()
    }

    pub fn capacity(&self) -> u32 {
        self.capacity.load(Relaxed)
    }

    pub fn capacity_mut(&mut self) -> u32 {
        *self.capacity.get_mut()
    }

    // Our capacity can never exceed `self.max_capacity`, so the index has to fit in a `u32`.
    #[allow(clippy::cast_possible_truncation)]
    pub fn push_with_tag(&self, value: T, tag: u32) -> u32 {
        // This cannot overflow because our capacity can never exceed `isize::MAX` bytes, and
        // because `self.reserve_for_push()` resets `self.reserved_len` back to `self.max_capacity`
        // if it was overshot.
        let index = self.reserved_len.fetch_add(1, Relaxed);

        if index >= self.capacity() as usize {
            self.reserve_for_push(index);
        }

        // SAFETY: We made sure that the index is in bounds above.
        let slot = unsafe { self.get_unchecked(index) };

        // SAFETY: We reserved an index by incrementing `self.reserved_len`, which means that no
        // other threads can be attempting to write to this same slot. No other threads can be
        // reading this slot either until we update the generation below.
        unsafe { slot.value.get().cast::<T>().write(value) };

        slot.generation.store(OCCUPIED_BIT | tag, Release);

        index as u32
    }

    // Our capacity can never exceed `self.max_capacity`, so the index has to fit in a `u32`.
    #[allow(clippy::cast_possible_truncation)]
    pub fn push_with_tag_mut(&mut self, value: T, tag: u32) -> u32 {
        let index = *self.reserved_len.get_mut();

        // This cannot overflow because our capacity can never exceed `isize::MAX` bytes, and
        // because `self.reserve_for_push()` resets `self.reserved_len` back to `self.max_capacity`
        // if it was overshot.
        *self.reserved_len.get_mut() += 1;

        if index >= self.capacity() as usize {
            self.reserve_for_push(index);
        }

        // SAFETY: We made sure that the index is in bounds above.
        let slot = unsafe { self.get_unchecked_mut(index) };

        slot.value.get_mut().write(value);

        *slot.generation.get_mut() = OCCUPIED_BIT | tag;

        index as u32
    }

    #[inline(never)]
    fn reserve_for_push(&self, len: usize) {
        handle_reserve(self.grow_amortized(len, 1));
    }

    // TODO: What's there to amortize over? It should be linear growth.
    fn grow_amortized(&self, len: usize, additional: usize) -> Result<(), TryReserveError> {
        debug_assert!(additional > 0);

        let required_capacity = len.checked_add(additional).ok_or(CapacityOverflow)?;

        if required_capacity > self.max_capacity as usize {
            if self.reserved_len.load(Relaxed) > self.max_capacity as usize {
                self.reserved_len.store(self.max_capacity as usize, Relaxed);
            }

            return Err(CapacityOverflow.into());
        }

        let page_capacity = u32::try_from(page_size() / mem::size_of::<Slot<T>>()).unwrap();

        // We checked that `required_capacity` doesn't exceed `self.max_capacity`, so it must fit.
        #[allow(clippy::cast_possible_truncation)]
        let new_capacity = cmp::max(self.capacity() * 2, required_capacity as u32);
        let new_capacity = cmp::max(new_capacity, page_capacity);
        let new_capacity = cmp::min(new_capacity, self.max_capacity);

        grow(
            &self.allocation,
            &self.capacity,
            new_capacity,
            mem::size_of::<Slot<T>>(),
        )
    }
}

#[inline(never)]
fn grow(
    allocation: &Allocation,
    capacity: &AtomicU32,
    new_capacity: u32,
    element_size: usize,
) -> Result<(), TryReserveError> {
    let old_capacity = capacity.load(Relaxed);

    if old_capacity >= new_capacity {
        // Another thread beat us to it.
        return Ok(());
    }

    let page_size = page_size();

    let old_size = old_capacity as usize * element_size;
    let new_size = usize::try_from(new_capacity)
        .ok()
        .and_then(|cap| cap.checked_mul(element_size))
        .ok_or(CapacityOverflow)?;

    if new_size > allocation.size() {
        return Err(CapacityOverflow.into());
    }

    let old_size = align_up(old_size, page_size);
    let new_size = align_up(new_size, page_size);
    let ptr = allocation.ptr().wrapping_add(old_size);
    let size = new_size - old_size;

    allocation.commit(ptr, size).map_err(AllocError)?;

    // The `Release` ordering synchronizes with the `Acquire` ordering in `Vec::as_slice`.
    if let Err(capacity) = capacity.compare_exchange(old_capacity, new_capacity, Release, Relaxed) {
        // We lost the race, but the winner must have updated the capacity same as we wanted to.
        assert!(capacity >= new_capacity);
    }

    Ok(())
}

#[inline]
fn handle_reserve<T>(res: Result<T, TryReserveError>) -> T {
    match res.map_err(|e| e.kind) {
        Ok(x) => x,
        Err(CapacityOverflow) => capacity_overflow(),
        Err(AllocError(err)) => handle_alloc_error(err),
    }
}

#[inline(never)]
fn capacity_overflow() -> ! {
    panic!("capacity overflow");
}

// Dear Clippy, `Error` is 4 bytes.
#[allow(clippy::needless_pass_by_value)]
#[cold]
fn handle_alloc_error(err: Error) -> ! {
    panic!("allocation failed: {err}");
}

impl<T> Deref for Vec<T> {
    type Target = [Slot<T>];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for Vec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

pub(crate) struct Slot<T> {
    pub generation: AtomicU32,
    pub next_free: AtomicU32,
    pub value: UnsafeCell<MaybeUninit<T>>,
}

// SAFETY: The user of `Slot` must ensure that access to `Slot::value` is synchronized.
unsafe impl<T: Sync> Sync for Slot<T> {}

impl<T> Slot<T> {
    #[inline(always)]
    pub unsafe fn value_unchecked(&self) -> &T {
        // SAFETY: The caller must ensure that access to the cell's inner value is synchronized.
        let value = unsafe { &*self.value.get() };

        // SAFETY: The caller must ensure that the slot has been initialized.
        unsafe { value.assume_init_ref() }
    }

    #[inline(always)]
    pub unsafe fn value_unchecked_mut(&mut self) -> &mut T {
        // SAFETY: The caller must ensure that the slot has been initialized.
        unsafe { self.value.get_mut().assume_init_mut() }
    }
}

#[derive(Debug)]
pub struct TryReserveError {
    kind: TryReserveErrorKind,
}

impl From<TryReserveErrorKind> for TryReserveError {
    #[inline]
    fn from(kind: TryReserveErrorKind) -> Self {
        TryReserveError { kind }
    }
}

#[derive(Debug)]
enum TryReserveErrorKind {
    CapacityOverflow,
    AllocError(Error),
}

impl fmt::Display for TryReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            CapacityOverflow => f.write_str(
                "memory allocation failed because the computed capacity exceeded the collection's \
                maximum",
            ),
            AllocError(_) => f.write_str(
                "memory allocation failed because the operating system returned an error",
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TryReserveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            TryReserveErrorKind::CapacityOverflow => None,
            TryReserveErrorKind::AllocError(err) => Some(err),
        }
    }
}
