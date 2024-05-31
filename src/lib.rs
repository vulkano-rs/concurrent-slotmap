#![allow(unused_unsafe, clippy::inline_always, clippy::unit_arg)]
// This should be `forbid` but there's a bug in rustc:
// https://github.com/rust-lang/rust/issues/121483
#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(clippy::undocumented_unsafe_blocks)]

use core::{
    cell::UnsafeCell,
    fmt, hint,
    mem::MaybeUninit,
    num::NonZeroU32,
    ops::Deref,
    sync::atomic::{
        AtomicU32, AtomicU64,
        Ordering::{Acquire, Relaxed, Release},
    },
};
use std::borrow::Cow;
use virtual_buffer::vec::Vec;

pub mod epoch;

/// The slot index used to signify the lack thereof.
const NIL: u32 = u32::MAX;

#[cfg_attr(not(doc), repr(C))]
pub struct SlotMap<T> {
    slots: Vec<Slot<T>>,
    len: AtomicU32,

    _alignment1: CacheAligned,

    /// The free-list. This is the list of slots which have already been dropped and are ready to
    /// be claimed by insert operations.
    free_list: AtomicU32,

    _alignment2: CacheAligned,

    /// Free-lists queued for inclusion in the `free_list`. Since the global epoch can only be
    /// advanced if all pinned local epochs are pinned in the current global epoch, we only need
    /// two free-lists in the queue: one for the current global epoch and one for the previous
    /// epoch. This way a thread can always push a slot into list `(epoch / 2) % 2`, and whenever
    /// the global epoch is advanced, since we know that the lag can be at most one step, we can be
    /// certain that the list which was lagging behind before the global epoch was advanced is now
    /// safe to drop and prepend to the `free_list`.
    ///
    /// The atomic packs the list's head in the lower 32 bits and the epoch of the last push in the
    /// upper 32 bits. The epoch must not be updated if it would be going backwards; it's only
    /// updated when the last epoch is at least 2 steps behind the local epoch, at which point -
    /// after removing the list from the queue and updating the epoch - we can be certain that no
    /// other threads are accessing any of the slots in the list.
    free_list_queue: [AtomicU64; 2],
}

impl<T> SlotMap<T> {
    #[must_use]
    pub fn new(max_capacity: u32) -> Self {
        SlotMap {
            slots: Vec::new(max_capacity as usize),
            len: AtomicU32::new(0),
            _alignment1: CacheAligned,
            free_list: AtomicU32::new(NIL),
            _alignment2: CacheAligned,
            free_list_queue: [
                AtomicU64::new(u64::from(NIL)),
                AtomicU64::new(u64::from(NIL) | 2 << 32),
            ],
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

    pub fn insert<'a>(&'a self, value: T, _guard: Cow<'a, epoch::Guard>) -> SlotId {
        let mut free_list_head = self.free_list.load(Acquire);
        let mut backoff = Backoff::new();

        loop {
            if free_list_head == NIL {
                break;
            }

            // SAFETY: We always push indices of existing slots into the free-lists and the slots
            // vector never shrinks, therefore the index must have staid in bounds.
            let slot = unsafe { self.slot_unchecked(free_list_head) };

            let next_free = slot.next_free.load(Relaxed);

            match self
                .free_list
                .compare_exchange_weak(free_list_head, next_free, Release, Acquire)
            {
                Ok(_) => {
                    // SAFETY: `SlotMap::remove` guarantees that the free-list only contains slots
                    // that are no longer read by any threads, and we have removed the slot from
                    // the free-list such that no other threads can be writing the same slot.
                    unsafe { slot.value.get().cast::<T>().write(value) };

                    let last_generation = slot.generation.fetch_add(1, Release);

                    self.len.fetch_add(1, Relaxed);

                    return SlotId::new(free_list_head, last_generation.wrapping_add(1));
                }
                Err(new_head) => {
                    free_list_head = new_head;
                    backoff.spin();
                }
            }
        }

        let index = self.slots.push(Slot::new(value));

        self.len.fetch_add(1, Relaxed);

        // Our capacity can never exceed `u32::MAX`.
        #[allow(clippy::cast_possible_truncation)]
        SlotId::new(index as u32, OCCUPIED_BIT)
    }

    pub fn remove<'a>(&'a self, id: SlotId, guard: Cow<'a, epoch::Guard>) -> Option<Ref<'a, T>> {
        let slot = self.slots.get(id.index as usize)?;
        let new_generation = id.generation().wrapping_add(1);

        // This works thanks to the invariant of `SlotId` that the `OCCUPIED_BIT` of its generation
        // must be set. That means that the only outcome possible in case of success here is that
        // the `OCCUPIED_BIT` is unset and the generation is advanced.
        if slot
            .generation
            .compare_exchange(id.generation(), new_generation, Acquire, Relaxed)
            .is_err()
        {
            return None;
        }

        let epoch = guard.epoch();
        let queued_list = &self.free_list_queue[((epoch >> 1) & 1) as usize];
        let mut queued_state = queued_list.load(Acquire);
        let mut backoff = Backoff::new();

        loop {
            let queued_head = (queued_state & 0xFFFF_FFFF) as u32;
            let queued_epoch = (queued_state >> 32) as u32;
            let epoch_interval = epoch.wrapping_sub(queued_epoch);

            if epoch_interval == 0 {
                slot.next_free.store(queued_head, Relaxed);

                let new_state = u64::from(id.index) | u64::from(queued_epoch) << 32;

                match queued_list.compare_exchange_weak(queued_state, new_state, Release, Acquire) {
                    Ok(_) => {
                        self.len.fetch_sub(1, Relaxed);

                        // SAFETY:
                        // * The `Acquire` ordering when loading the slot's generation synchronizes
                        //   with the `Release` ordering in `SlotMap::insert`, making sure that the
                        //   newly written value is visible here.
                        // * The `compare_exchange` above succeeded, which means that the previous
                        //   generation of the slot must have matched `id.generation`. By `SlotId`'s
                        //   invariant, its generation's occupied bit must be set. Since the
                        //   generation matched, the slot's occupied bit must have been set, which
                        //   makes reading the value safe as the only way the occupied bit can be
                        //   set is in `SlotMap::insert` after initialization of the slot.
                        break Some(unsafe { Ref { slot, guard } });
                    }
                    Err(new_state) => {
                        queued_state = new_state;
                        backoff.spin();
                    }
                }
            } else {
                let local_epoch_is_behind_queue = epoch_interval & (1 << 31) != 0;

                // TODO: What's preventing this? If we pushed into the list as above in this case,
                // it could happen that:
                // * Thread A loads global epoch, preempts.
                // * Thread B advances the global epoch twice to epoch E.
                // * Thread A pins itself in epoch E - 4.
                // * Thread A removes slot S, sees the last push into the queued list was E - 4.
                // * Thread B removes slot P, sees the last push into the queued list was E - 4,
                //   drops slot S which could still be accessed by thread A.
                assert!(!local_epoch_is_behind_queue);

                debug_assert!(epoch_interval >= 4);

                slot.next_free.store(NIL, Relaxed);

                let new_state = u64::from(id.index) | u64::from(epoch) << 32;

                match queued_list.compare_exchange_weak(queued_state, new_state, Release, Acquire) {
                    Ok(_) => {
                        self.len.fetch_sub(1, Relaxed);

                        // SAFETY: Having ended up here, the global epoch must have been advanced
                        // at least 2 steps from the last push into the queued list and we removed
                        // the list from the queue, which means that no other threads can be
                        // accessing any of the slots in the list.
                        unsafe { self.collect_garbage(queued_head) };

                        // SAFETY: Same as the the `Ref` construction in the above branch.
                        break Some(unsafe { Ref { slot, guard } });
                    }
                    Err(new_state) => {
                        queued_state = new_state;
                        backoff.spin();
                    }
                }
            }
        }
    }

    #[cold]
    unsafe fn collect_garbage(&self, queued_head: u32) {
        if queued_head == NIL {
            // There is no garbage.
            return;
        }

        let mut queued_tail = queued_head;
        let mut queued_tail_slot;

        // Drop the queued free-list and find the tail slot.
        loop {
            // SAFETY: We always push indices of existing slots into the free-lists and the slots
            // vector never shrinks, therefore the index must have staid in bounds.
            queued_tail_slot = unsafe { self.slot_unchecked(queued_tail) };

            // SAFETY: The caller must ensure that we have exclusive access to this list.
            unsafe { queued_tail_slot.value.get().cast::<T>().drop_in_place() };

            let next_free = queued_tail_slot.next_free.load(Acquire);

            if next_free == NIL {
                break;
            }

            queued_tail = next_free;
        }

        let mut free_list_head = self.free_list.load(Acquire);
        let mut backoff = Backoff::new();

        // Free the queued free-list by prepending it to the free-list.
        loop {
            queued_tail_slot.next_free.store(free_list_head, Relaxed);

            match self.free_list.compare_exchange_weak(
                free_list_head,
                queued_head,
                Release,
                Acquire,
            ) {
                Ok(_) => break,
                Err(new_head) => {
                    free_list_head = new_head;
                    backoff.spin();
                }
            }
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn get<'a>(&'a self, id: SlotId, guard: Cow<'a, epoch::Guard>) -> Option<Ref<'a, T>> {
        let slot = self.slots.get(id.index as usize)?;
        let generation = slot.generation.load(Acquire);

        if generation == id.generation() {
            // SAFETY:
            // * The `Acquire` ordering when loading the slot's generation synchronizes with the
            //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value
            //   is visible here.
            // * We checked that `id.generation` matches the slot's generation, which includes the
            //   occupied bit. By `SlotId`'s invariant, its generation's occupied bit must be set.
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

// We don't want to print the `_alignment` fields.
#[allow(clippy::missing_fields_in_debug)]
impl<T: fmt::Debug> fmt::Debug for SlotMap<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct List<'a, T>(&'a SlotMap<T>, u32);

        impl<T> fmt::Debug for List<'_, T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut head = self.1;
                let mut debug = f.debug_list();

                while head != NIL {
                    debug.entry(&head);

                    // SAFETY: We always push indices of existing slots into the free-lists and the
                    // slots vector never shrinks, therefore the index must have staid in bounds.
                    let slot = unsafe { self.0.slot_unchecked(head) };

                    head = slot.next_free.load(Acquire);
                }

                debug.finish()
            }
        }

        struct QueuedList<'a, T>(&'a SlotMap<T>, u64);

        impl<T> fmt::Debug for QueuedList<'_, T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let state = self.1;
                let head = (state & 0xFFFF_FFFF) as u32;
                let epoch = (state >> 32) as u32;
                let mut debug = f.debug_struct("QueuedList");
                debug
                    .field("entries", &List(self.0, head))
                    .field("epoch", &epoch);

                debug.finish()
            }
        }

        let _guard = epoch::pin();
        let mut debug = f.debug_struct("SlotMap");
        debug
            .field("slots", &self.slots)
            .field("len", &self.len)
            .field("free_list", &List(self, self.free_list.load(Acquire)))
            .field(
                "free_list_queue",
                &[
                    QueuedList(self, self.free_list_queue[0].load(Acquire)),
                    QueuedList(self, self.free_list_queue[1].load(Acquire)),
                ],
            );

        debug.finish()
    }
}

impl<T> Drop for SlotMap<T> {
    fn drop(&mut self) {
        if !core::mem::needs_drop::<T>() {
            return;
        }

        for list in &mut self.free_list_queue {
            let mut head = (*list.get_mut() & 0xFFFF_FFFF) as u32;

            while head != NIL {
                // SAFETY: We always push indices of existing slots into the free-lists and the
                // slots vector never shrinks, therefore the index must have staid in bounds.
                let slot = unsafe { self.slots.get_unchecked_mut(head as usize) };

                // SAFETY: We can be certain that this slot has been initialized, since the only
                // way in which it could have been queued for freeing is in `SlotMap::remove` if
                // the slot was inserted before.
                unsafe { slot.value.get_mut().assume_init_drop() };

                head = *slot.next_free.get_mut();
            }
        }

        for slot in &mut self.slots {
            if *slot.generation.get_mut() & OCCUPIED_BIT != 0 {
                // SAFETY:
                // * The mutable reference makes sure that access to the slot is synchronized.
                // * We checked that the slot is occupied, which means the it must have been
                //   initialized in `SlotMap::insert`.
                unsafe { slot.value.get_mut().assume_init_drop() };
            }
        }
    }
}

const OCCUPIED_BIT: u32 = 1;

struct Slot<T> {
    generation: AtomicU32,
    next_free: AtomicU32,
    value: UnsafeCell<MaybeUninit<T>>,
}

// SAFETY: The user of `Slot` must ensure that access to `Slot::value` is synchronized.
unsafe impl<T: Sync> Sync for Slot<T> {}

impl<T> Slot<T> {
    fn new(value: T) -> Self {
        Slot {
            generation: AtomicU32::new(OCCUPIED_BIT),
            next_free: AtomicU32::new(NIL),
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
        struct Nil;

        impl fmt::Debug for Nil {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.pad("NIL")
            }
        }

        let _guard = epoch::pin();
        let generation = self.generation.load(Acquire);

        let mut debug = f.debug_struct("Slot");
        debug.field("generation", &generation);

        if generation & OCCUPIED_BIT != 0 {
            // SAFETY:
            // * The `Acquire` ordering when loading the slot's generation synchronizes with the
            //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value
            //   is visible here.
            // * We checked that the slot is occupied, which means it must have been initialized in
            //   `SlotMap::insert`.
            debug.field("value", unsafe { self.value_unchecked() });
        } else {
            let next_free = self.next_free.load(Relaxed);

            if next_free == NIL {
                debug.field("next_free", &Nil);
            } else {
                debug.field("next_free", &next_free);
            }
        }

        debug.finish()
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
    guard: Cow<'a, epoch::Guard>,
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

#[repr(align(128))]
struct CacheAligned;

const SPIN_LIMIT: u32 = 6;

struct Backoff {
    step: u32,
}

impl Backoff {
    fn new() -> Self {
        Backoff { step: 0 }
    }

    fn spin(&mut self) {
        for _ in 0..1 << self.step {
            hint::spin_loop();
        }

        if self.step <= SPIN_LIMIT {
            self.step += 1;
        }
    }
}
