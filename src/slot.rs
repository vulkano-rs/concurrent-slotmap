use crate::{Header, SlotId, NIL, OCCUPIED_TAG};
use core::{
    alloc::Layout,
    cell::UnsafeCell,
    cmp, fmt,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    panic::{RefUnwindSafe, UnwindSafe},
    ptr, slice,
    sync::atomic::{
        AtomicU32, AtomicUsize,
        Ordering::{Acquire, Relaxed, Release},
    },
};
use virtual_buffer::{align_up, page_size, Allocation, Error};
use TryReserveErrorKind::{AllocError, CapacityOverflow};

pub(crate) struct Vec<T> {
    allocation: virtual_buffer::Allocation,
    slots: *mut u8,
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
        assert!(mem::align_of::<Header>() <= page_size());
        assert!(mem::align_of::<Slot<T>>() <= page_size());

        let size = usize::try_from(max_capacity)
            .ok()
            .and_then(|cap| cap.checked_mul(mem::size_of::<Slot<T>>()))
            .ok_or(CapacityOverflow)?;

        #[allow(clippy::cast_possible_wrap)]
        if size > isize::MAX as usize {
            return Err(CapacityOverflow.into());
        }

        let slots_offset = align_up(mem::size_of::<Header>(), mem::align_of::<Slot<T>>());
        let size = slots_offset.checked_add(size).ok_or(CapacityOverflow)?;
        let size = align_up(size, page_size());
        let allocation = Allocation::new(size).map_err(AllocError)?;
        let slots = allocation.ptr().wrapping_add(slots_offset);
        #[allow(clippy::cast_possible_truncation)]
        let capacity = ((size - slots_offset) / mem::size_of::<Slot<T>>()) as u32;
        let header = slots
            .wrapping_sub(mem::size_of::<Header>())
            .cast::<Header>();

        allocation
            .commit(allocation.ptr(), size)
            .map_err(AllocError)?;

        // SAFETY: `header` is a valid pointer to the `Header` we made sure to allocate space for
        // above.
        unsafe {
            *ptr::addr_of_mut!((*header).free_list) = AtomicU32::new(NIL);
            *ptr::addr_of_mut!((*header).len) = AtomicU32::new(0);
        }

        Ok(Vec {
            allocation,
            slots,
            max_capacity,
            capacity: AtomicU32::new(capacity),
            reserved_len: AtomicUsize::new(0),
            marker: PhantomData,
        })
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const Slot<T> {
        self.slots.cast()
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Slot<T> {
        self.slots.cast()
    }

    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity.load(Relaxed)
    }

    #[inline]
    pub fn capacity_mut(&mut self) -> u32 {
        *self.capacity.get_mut()
    }

    pub fn header(&self) -> &Header {
        let header_size = mem::size_of::<Header>();

        // SAFETY: We made sure to allocate space for the `Header` and it's right before the
        // beginning of the slots.
        unsafe { &*self.slots.cast::<u8>().sub(header_size).cast::<Header>() }
    }

    pub fn header_mut(&mut self) -> &mut Header {
        let header_size = mem::size_of::<Header>();

        // SAFETY: We made sure to allocate space for the `Header` and it's right before the
        // beginning of the slots.
        unsafe { &mut *self.slots.cast::<u8>().sub(header_size).cast::<Header>() }
    }

    pub fn push_with_tag_with(&self, tag: u32, f: impl FnOnce(SlotId) -> T) -> (SlotId, &Slot<T>) {
        // This cannot overflow because our capacity can never exceed `isize::MAX` bytes, and
        // because `self.reserve_for_push()` resets `self.reserved_len` back to `self.max_capacity`
        // if it was overshot.
        let index = self.reserved_len.fetch_add(1, Relaxed);

        if index >= self.capacity() as usize {
            self.reserve_for_push(index);
        }

        // Our capacity can never exceed `self.max_capacity`, so the index has to fit in a `u32`.
        #[allow(clippy::cast_possible_truncation)]
        let index = index as u32;
        let generation = OCCUPIED_TAG | tag;

        // SAFETY: We made sure that the index is in bounds above.
        let slot = unsafe { self.get_unchecked(index) };

        // SAFETY: The state tag of the generation is `OCCUPIED_TAG`.
        let id = unsafe { SlotId::new_unchecked(index, generation) };

        // SAFETY: We reserved an index by incrementing `self.reserved_len`, which means that no
        // other threads can be attempting to write to this same slot. No other threads can be
        // reading this slot either until we update the generation below.
        unsafe { slot.value.get().cast::<T>().write(f(id)) };

        slot.generation.store(generation, Release);

        (id, slot)
    }

    pub fn push_with_tag_with_mut(&mut self, tag: u32, f: impl FnOnce(SlotId) -> T) -> SlotId {
        let index = *self.reserved_len.get_mut();

        // This cannot overflow because our capacity can never exceed `isize::MAX` bytes, and
        // because `self.reserve_for_push()` resets `self.reserved_len` back to `self.max_capacity`
        // if it was overshot.
        *self.reserved_len.get_mut() += 1;

        if index >= self.capacity() as usize {
            self.reserve_for_push(index);
        }

        // Our capacity can never exceed `self.max_capacity`, so the index has to fit in a `u32`.
        #[allow(clippy::cast_possible_truncation)]
        let index = index as u32;
        let generation = OCCUPIED_TAG | tag;

        // SAFETY: We made sure that the index is in bounds above.
        let slot = unsafe { self.get_unchecked_mut(index) };

        // SAFETY: The state tag of the generation is `OCCUPIED_TAG`.
        let id = unsafe { SlotId::new_unchecked(index, generation) };

        slot.value.get_mut().write(f(id));

        *slot.generation.get_mut() = generation;

        id
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

        // We checked that `required_capacity` doesn't exceed `self.max_capacity`, so it must fit in
        // a `u32`.
        #[allow(clippy::cast_possible_truncation)]
        let new_capacity = cmp::max(self.capacity() * 2, required_capacity as u32);
        let new_capacity = cmp::max(new_capacity, page_capacity);
        let new_capacity = cmp::min(new_capacity, self.max_capacity);

        grow(
            &self.allocation,
            &self.capacity,
            new_capacity,
            Layout::new::<Slot<T>>(),
        )
    }

    #[inline]
    pub fn get(&self, index: u32) -> Option<&Slot<T>> {
        let capacity = self.capacity.load(Acquire);

        if index >= capacity {
            return None;
        }

        // SAFETY: We checked that the index is in bounds above.
        let ptr = unsafe { self.as_ptr().add(index as usize) };

        // SAFETY: We know that newly-allocated pages are zeroed, so we don't need to intialize the
        // `Slot<T>` as it allows being zeroed. The `Acquire` ordering above synchronizes with the
        // `Release` ordering when setting the capacity, making sure that the reserved capacity is
        // visible here.
        Some(unsafe { &*ptr })
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: u32) -> &Slot<T> {
        let _capacity = self.capacity.load(Acquire);

        // SAFETY: The caller must ensure that the index is in bounds.
        let ptr = unsafe { self.as_ptr().add(index as usize) };

        // SAFETY: Same as in `get` above.
        unsafe { &*ptr }
    }

    #[inline]
    pub fn get_mut(&mut self, index: u32) -> Option<&mut Slot<T>> {
        let capacity = self.capacity_mut();

        if index >= capacity {
            return None;
        }

        // SAFETY: We checked that the index is in bounds above.
        let ptr = unsafe { self.as_mut_ptr().add(index as usize) };

        // SAFETY: We know that newly-allocated pages are zeroed, so we don't need to intialize the
        // `Slot<T>` as it allows being zeroed. The mutable reference ensures synchronization in
        // this case.
        Some(unsafe { &mut *ptr })
    }

    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: u32) -> &mut Slot<T> {
        // SAFETY: The caller must ensure that the index is in bounds.
        let ptr = unsafe { self.as_mut_ptr().add(index as usize) };

        // SAFETY: Same as in `get_mut` above.
        unsafe { &mut *ptr }
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, Slot<T>> {
        let capacity = self.capacity.load(Acquire) as usize;

        // SAFETY: We know that newly-allocated pages are zeroed, so we don't need to intialize the
        // `Slot<T>` as it allows being zeroed. The `Acquire` ordering above synchronizes with the
        // `Release` ordering when setting the capacity, making sure that the reserved capacity is
        // visible here.
        unsafe { slice::from_raw_parts(self.as_ptr(), capacity) }.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, Slot<T>> {
        let capacity = self.capacity_mut() as usize;

        // SAFETY: We know that newly-allocated pages are zeroed, so we don't need to intialize the
        // `Slot<T>` as it allows being zeroed. The mutable reference ensures synchronization in
        // this case.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), capacity) }.iter_mut()
    }
}

#[inline(never)]
fn grow(
    allocation: &Allocation,
    capacity: &AtomicU32,
    new_capacity: u32,
    element_layout: Layout,
) -> Result<(), TryReserveError> {
    let old_capacity = capacity.load(Relaxed);

    if old_capacity >= new_capacity {
        // Another thread beat us to it.
        return Ok(());
    }

    let page_size = page_size();

    let slots_offset = align_up(mem::size_of::<Header>(), element_layout.align());
    let old_size = slots_offset + old_capacity as usize * element_layout.size();
    let new_size = usize::try_from(new_capacity)
        .ok()
        .and_then(|cap| cap.checked_mul(element_layout.size()))
        .ok_or(CapacityOverflow)?;
    let new_size = slots_offset.checked_add(new_size).ok_or(CapacityOverflow)?;

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

pub struct Slot<V> {
    pub(crate) generation: AtomicU32,
    pub(crate) next_free: AtomicU32,
    pub(crate) value: UnsafeCell<MaybeUninit<V>>,
}

impl<V: UnwindSafe> UnwindSafe for Slot<V> {}
impl<V: RefUnwindSafe> RefUnwindSafe for Slot<V> {}

// SAFETY: We only ever hand out references to a slot in the presence of a `hyaline::Guard`.
unsafe impl<V: Sync> Sync for Slot<V> {}

impl<V> Slot<V> {
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation.load(Acquire)
    }

    #[inline]
    pub fn value_ptr(&self) -> *mut V {
        self.value.get().cast()
    }

    /// # Safety
    ///
    /// The value must be initialized. You can use [`generation`] to determine the state of the
    /// slot.
    ///
    /// [`generation`]: Self::generation
    #[inline(always)]
    pub unsafe fn value_unchecked(&self) -> &V {
        // SAFETY: The caller must ensure that access to the cell's inner value is synchronized.
        let value = unsafe { &*self.value.get() };

        // SAFETY: The caller must ensure that the slot has been initialized.
        unsafe { value.assume_init_ref() }
    }

    /// # Safety
    ///
    /// The value must be initialized. You can use [`generation`] to determine the state of the
    /// slot.
    ///
    /// [`generation`]: Self::generation
    #[inline(always)]
    pub unsafe fn value_unchecked_mut(&mut self) -> &mut V {
        // SAFETY: The caller must ensure that the slot has been initialized.
        unsafe { self.value.get_mut().assume_init_mut() }
    }
}

impl<V> fmt::Debug for Slot<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slot").finish_non_exhaustive()
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

impl std::error::Error for TryReserveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            TryReserveErrorKind::CapacityOverflow => None,
            TryReserveErrorKind::AllocError(err) => Some(err),
        }
    }
}
