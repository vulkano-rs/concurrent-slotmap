#![allow(unused_unsafe, clippy::inline_always, clippy::unit_arg)]
// This should be `forbid` but there's a bug in rustc:
// https://github.com/rust-lang/rust/issues/121483
#![deny(unsafe_op_in_unsafe_fn)]
// #![forbid(clippy::undocumented_unsafe_blocks)]

use core::{
    cell::UnsafeCell,
    fmt, hint,
    mem::MaybeUninit,
    num::NonZeroU32,
    ops::Deref,
    sync::atomic::{
        AtomicU32, AtomicUsize,
        Ordering::{Acquire, Relaxed, Release},
    },
};
use virtual_buffer::vec::Vec;

mod epoch;

/// The slot index used to signify the lack thereof.
const NIL: u32 = u32::MAX;

/// The bit of `SlotMap::last_collect_epoch` which signifies that a garbage collection has begun.
const COLLECTING_BIT: usize = 1 << 0;

#[derive(Debug)]
pub struct SlotMap<T> {
    slots: Vec<Slot<T>>,
    len: AtomicU32,

    /// The free-list. This is the list of slots which have already been dropped and are ready to
    /// be claimed by insert operations.
    free_list: List,

    /// Free-lists queued for inclusion in the `free_list`. Since the global epoch counter can only
    /// ever be one step apart from a local one, we only need two free-lists in the queue: one for
    /// the current global epoch and one for the previous epoch. This way a thread can always push
    /// a slot into list `(epoch / 2) % 2`, and whenever the global epoch is advanced, since we
    /// know that the lag can be at most one step, we can be certain that the list which was
    /// lagging behind before the global epoch was advanced is now safe to drop and prepend to
    /// `free_list`.
    free_list_queue: [List; 2],

    /// The epoch for which garbage was collected the last time.
    last_collect_epoch: AtomicUsize,
}

impl<T> SlotMap<T> {
    #[must_use]
    pub fn new(max_capacity: u32) -> Self {
        SlotMap {
            slots: Vec::new(max_capacity as usize),
            len: AtomicU32::new(0),
            free_list: List::new(),
            free_list_queue: [List::new(), List::new()],
            last_collect_epoch: AtomicUsize::new(0),
        }
    }

    // Our capacity can never exceed `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.slots.capacity() as u32
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len.load(Relaxed)
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn insert(&self, value: T) -> SlotId {
        let mut free_list_head = self.free_list.head.load(Acquire);

        loop {
            if free_list_head == NIL {
                break;
            }

            // SAFETY: We always push indices of existing slots into the free-lists and the slots
            // vector never shrinks, therefore the index must have staid in bounds.
            let slot = unsafe { self.slot_unchecked(free_list_head) };

            let next_free = slot.next_free.head.load(Relaxed);

            match self.free_list.head.compare_exchange_weak(
                free_list_head,
                next_free,
                Release,
                Acquire,
            ) {
                Ok(_) => {
                    // SAFETY: `SlotMap::remove` guarantees that the free-list only contains slots
                    // that are no longer read by any threads.
                    unsafe { slot.value.get().cast::<T>().write(value) };

                    let last_generation = slot.generation.fetch_add(1, Release);

                    self.len.fetch_add(1, Relaxed);

                    return SlotId::new(free_list_head, last_generation.wrapping_add(1));
                }
                Err(new_head) => free_list_head = new_head,
            }
        }

        let index = self.slots.push(Slot::new(value));

        self.len.fetch_add(1, Relaxed);

        // Our capacity can never exceed `u32::MAX`.
        #[allow(clippy::cast_possible_truncation)]
        SlotId::new(index as u32, OCCUPIED_BIT)
    }

    pub fn remove(&self, id: SlotId) -> Option<()> {
        let slot = self.slots.get(id.index as usize)?;
        let new_generation = id.generation().wrapping_add(1);

        // This works thanks to the invariant of `SlotId` that the `OCCUPIED_BIT` of its generation
        // must be set. That means that the only outcome possible in case of success here is that
        // the `OCCUPIED_BIT` is unset and the generation is advanced.
        if slot
            .generation
            .compare_exchange(id.generation(), new_generation, Relaxed, Relaxed)
            .is_err()
        {
            return None;
        }

        let guard = epoch::pin();
        let epoch = guard.epoch();
        let last_epoch = self.last_collect_epoch.load(Relaxed);

        if epoch.wrapping_sub(last_epoch) >= 3 {
            self.collect_garbage(epoch, last_epoch);
        }

        let queued_list = &self.free_list_queue[(epoch >> 1) & 1];
        let mut queued_head = queued_list.head.load(Acquire);

        loop {
            slot.next_free.head.store(queued_head, Relaxed);

            match queued_list
                .head
                .compare_exchange_weak(queued_head, id.index, Release, Acquire)
            {
                Ok(_) => break,
                Err(new_head) => queued_head = new_head,
            }
        }

        self.len.fetch_sub(1, Relaxed);

        Some(())
    }

    #[cold]
    fn collect_garbage(&self, epoch: usize, mut last_epoch: usize) {
        // If a thread ended up here, it means that the global epoch was advanced far enough to
        // allow the queued list corresponding to `epoch` to be dropped and freed. One way in which
        // that can happen is the following.
        //
        // (1) The `last_epoch` is 2 steps behind `epoch`
        //
        //     :       :
        //     | E - 4 | <- `last_epoch`
        //     |-------|
        //     | E - 2 |
        //     |-------|
        //     | E - 0 | <- `epoch`
        //     :       :
        //
        // Since the global epoch can only ever be one step ahead of any pinned local epoch, we can
        // be certain that once 2 steps have been made, no thread can be accessing any of the slots
        // removed in `last_epoch`. We would like to drop and free those slots. Since we only have
        // two lists queued for freeing, the list which contains the slots removed in `last_epoch`
        // happens to be the same list that we would be pushing to in `epoch`:
        // `self.free_list_queue[(epoch >> 1) & 1]`. We want to remove this list from the queue to
        // make sure that no more slots are pushed into it by any other threads and that our thread
        // has exclusive access to the existing slots to be able to drop them.
        //
        // Now comes the problem: if we naively swapped the list's head with `NIL` immediately and
        // proceeded to drop, we could get a classic A/B/A problem:
        // * Thread A observes `epoch - last_epoch == 4`, enters here.
        // * Thread B observes `epoch - last_epoch == 4`, enters here.
        // * Thread B swaps the list head and proceeds to push a slot into the same list.
        // * Thread A swaps the list head, drops and frees the slot that was just removed by thread
        //   B and could still be read by other threads.
        //
        // What we do instead, *before* touching the list, is to advance the `last_epoch` by a
        // "half-step", reusing the same bit used for pinning to - in this case - signify that
        // collecting is taking place.
        //
        // (2) The `last_epoch` is in an "in-between" state
        //
        //     :       :
        //     | E - 4 |
        //     |-------| <- `last_epoch`
        //     | E - 2 |
        //     |-------|
        //     | E - 0 | <- `epoch`
        //     :       :
        //
        // While the `last_epoch` is in this state, other threads **must not** touch either of the
        // queued lists until that same thread advances the `last_epoch` again, after which point
        // the other threads no longer have any reason to collect garbage and can continue with the
        // remove operation. The `last_epoch` is only in this state before and after the swap on
        // the queued list's head. After the list is successfully removed from the queue, both the
        // collecting thread and removing threads can continue concurrently.
        //
        // This means that a thread can observe the `last_epoch` in this in-between state before or
        // after entering here, and we must account for both. We ensure the former by checking that
        // the interval between the epochs is at least 3 when getting here.
        //
        // Also, what if a thread reads an old version of `last_epoch` even though it was advanced?
        // In that case, since our epochs are always increasing, the only outcome can be that the
        // older version is lesser than the current one, which would be (1) again and still satisfy
        // the same condition and make the thread end up here.
        //
        // After ending up here, one thread advancing the `last_epoch` by a half-step, that same
        // thread advances the `last_epoch` again while the other threads wait patiently.
        //
        // (3) The `last_epoch` is 1 step behind `epoch`
        //
        //     :       :
        //     | E - 4 |
        //     |-------|
        //     | E - 2 | <- `last_epoch`
        //     |-------|
        //     | E - 0 | <- `epoch`
        //     :       :
        //
        // There are also other cases instead of (1). The `last_epoch` could have been (potentially
        // severely) outdated. In this case, unlike when the `last_epoch` is 2 steps behind, we can
        // have threads that are pinned in different epochs competing for garbage collection.
        //
        // (4) The `last_epoch` is more than 2 steps behind
        //
        //     :       :
        //     | E - N | <- `last_epoch`
        //     |-------|
        //     :       :
        //     |-------|
        //     | E - 2 | <- thread B `epoch`
        //     |-------|
        //     | E - 0 | <- thread A `epoch`
        //     :       :
        //
        // Now there are two possible interleavings: either thread A or B could win the race, but
        // either way only one will be collecting at a time like in every other case (this is why
        // a thread must not touch *either* of the queued lists when the `last_epoch` is in the
        // in-between state). If thread B wins, then after it removes its list from the queue and
        // advances the `last_epoch` again, it will be 2 steps behind thread A's epoch, which is
        // going to loop back to (1) and thread A is going to remove its list as well. If thread A
        // wins then thread B's list is not going to be removed.
        //
        // Because of this case, we always advance the epoch to one step behind `epoch` after
        // removing a list from the queue, as opposed to advancing it by another half-step.

        loop {
            // Case (2): only one thread can be collecting queued free-lists at a time, so we spin.
            if last_epoch & COLLECTING_BIT != 0 {
                hint::spin_loop();
                last_epoch = self.last_collect_epoch.load(Relaxed);
                continue;
            }

            // Case (3): there is no more need for any garbage collection.
            if epoch.wrapping_sub(last_epoch) == 2 {
                return;
            }

            // Case (1) or (4): we must notify other threads that we are about to collect. This
            // results in (2) for the losing threads.
            match self.last_collect_epoch.compare_exchange_weak(
                last_epoch,
                last_epoch | COLLECTING_BIT,
                Relaxed,
                Relaxed,
            ) {
                Ok(_) => break,
                Err(new_epoch) => last_epoch = new_epoch,
            }
        }

        // We can only end up here as a result of (1) or (4) and can be certain that we have
        // exclusive access to the queue.
        let queued_list = &self.free_list_queue[(epoch >> 1) & 1];
        let queued_head = queued_list.head.swap(NIL, Acquire);

        // Notify other threads that it is safe to proceed. This turns (2) into (3) or (1) for the
        // other threads.
        self.last_collect_epoch
            .store(epoch.wrapping_sub(2), Relaxed);

        if queued_head == NIL {
            // There is no garbage.
            return;
        }

        let mut queued_tail = queued_head;
        let mut queued_tail_slot;

        // Drop the queued free-list and find the tail slot.
        loop {
            // SAFETY: We always push indices of existing slots into the free-lists and the slots
            // vector never shrinks, therefore the index must have stayed in bounds.
            queued_tail_slot = unsafe { self.slot_unchecked(queued_tail) };

            // SAFETY: Having ended up here, the global epoch must have advanced at least 2 steps
            // from the last collect, which means that no other threads can be reading this value.
            // How we can be certain that this here thread is the only one that ended up here and
            // no A/B/A occured is detailed in the wall of text above.
            unsafe { queued_tail_slot.value.get().cast::<T>().drop_in_place() };

            let next_free = queued_tail_slot.next_free.head.load(Acquire);

            if next_free == NIL {
                break;
            }

            queued_tail = next_free;
        }

        let mut free_list_head = self.free_list.head.load(Acquire);

        // Free the queued free-list by prepending it to the free-list.
        loop {
            queued_tail_slot
                .next_free
                .head
                .store(free_list_head, Relaxed);

            match self.free_list.head.compare_exchange_weak(
                free_list_head,
                queued_head,
                Release,
                Acquire,
            ) {
                Ok(_) => break,
                Err(new_head) => free_list_head = new_head,
            }
        }
    }

    #[inline]
    #[must_use]
    pub fn get(&self, id: SlotId) -> Option<Ref<'_, T>> {
        let slot = self.slots.get(id.index as usize)?;
        let generation = slot.generation.load(Acquire);

        if generation == id.generation() {
            let guard = epoch::pin();

            // SAFETY:
            // * The `Acquire` ordering when loading the slot's generation synchronizes with the
            //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value
            //   is visible here.
            // * We checked that `id.generation` matches the slot's generation, which includes the
            //   occupied bit. By `SlotId`'s invariant, it's generation's occupied bit must be set.
            //   Since the generation matched, the slot's occupied bit must be set, which makes
            //   reading the value safe as the only way the occupied bit can be set is in
            //   `SlotMap::insert` after initialization of the slot.
            Some(unsafe { Ref { slot, guard } })
        } else {
            None
        }
    }

    unsafe fn slot_unchecked(&self, index: u32) -> &Slot<T> {
        // SAFETY: The caller must ensure that the index is in bounds.
        unsafe { self.slots.get_unchecked(index as usize) }
    }
}

impl<T> Drop for SlotMap<T> {
    fn drop(&mut self) {
        if !core::mem::needs_drop::<T>() {
            return;
        }

        for list in &mut self.free_list_queue {
            while *list.head.get_mut() != NIL {
                let slot = unsafe { self.slots.get_unchecked_mut(*list.head.get_mut() as usize) };
                unsafe { slot.value.get().cast::<T>().drop_in_place() };
                *list.head.get_mut() = *slot.next_free.head.get_mut();
            }
        }

        for slot in &mut self.slots {
            if *slot.generation.get_mut() & OCCUPIED_BIT != 0 {
                unsafe { slot.value.get_mut().assume_init_drop() };
            }
        }
    }
}

const OCCUPIED_BIT: u32 = 1;

struct Slot<T> {
    generation: AtomicU32,
    next_free: List,
    value: UnsafeCell<MaybeUninit<T>>,
}

unsafe impl<T: Send> Send for Slot<T> {}
unsafe impl<T: Sync> Sync for Slot<T> {}

impl<T> Slot<T> {
    fn new(value: T) -> Self {
        Slot {
            generation: AtomicU32::new(OCCUPIED_BIT),
            next_free: List::new(),
            value: UnsafeCell::new(MaybeUninit::new(value)),
        }
    }

    unsafe fn value_unchecked(&self) -> &T {
        // SAFETY: The caller must ensure that access to the cell's inner value is synchronized.
        let value = unsafe { &*self.value.get() };

        // SAFETY: The caller must ensure that the slot has been initialized.
        unsafe { value.assume_init_ref() }
    }
}

impl<T: fmt::Debug> fmt::Debug for Slot<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _guard = epoch::pin();
        let generation = self.generation.load(Acquire);

        let mut debug = f.debug_struct("Slot");
        debug.field("generation", &generation);

        if generation & OCCUPIED_BIT != 0 {
            debug.field("value", unsafe { self.value_unchecked() });
        } else {
            debug.field("next_free", &self.next_free);
        }

        debug.finish()
    }
}

struct List {
    head: AtomicU32,
}

impl List {
    fn new() -> Self {
        List {
            head: AtomicU32::new(NIL),
        }
    }
}

impl fmt::Debug for List {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let head = self.head.load(Relaxed);

        if head == NIL {
            f.pad("NIL")
        } else {
            fmt::Debug::fmt(&head, f)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotId {
    index: u32,
    generation: NonZeroU32,
}

impl SlotId {
    const fn new(index: u32, generation: u32) -> Self {
        assert!(generation & OCCUPIED_BIT != 0);

        // SAFETY: We checked that the `OCCUPIED_BIT` is set.
        unsafe { SlotId::new_unchecked(index, generation) }
    }

    const unsafe fn new_unchecked(index: u32, generation: u32) -> Self {
        // SAFETY: The caller must ensure that the `OCCUPIED_BIT` of `generation` is set, which
        // means it must be non-zero.
        let generation = unsafe { NonZeroU32::new_unchecked(generation) };

        SlotId { index, generation }
    }

    #[inline(always)]
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    #[inline(always)]
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation.get()
    }
}

pub struct Ref<'a, T> {
    slot: &'a Slot<T>,
    #[allow(dead_code)]
    guard: epoch::Guard,
}

impl<T> Deref for Ref<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: The constructor of `Ref` must ensure that the inner value was initialized and
        // that said write was synchronized with and made visible here. As for future writes to the
        // inner value, we know that none can happen as long as our `Guard` wasn't dropped.
        unsafe { self.slot.value_unchecked() }
    }
}

impl<T: fmt::Debug> fmt::Debug for Ref<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: fmt::Display> fmt::Display for Ref<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}
