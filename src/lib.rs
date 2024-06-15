#![allow(unused_unsafe, clippy::inline_always)]
// This should be `forbid` but there's a bug in rustc:
// https://github.com/rust-lang/rust/issues/121483
#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(clippy::undocumented_unsafe_blocks)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::borrow::Cow;
use core::{
    cell::UnsafeCell,
    fmt, hint,
    iter::{self, FusedIterator},
    mem::MaybeUninit,
    num::NonZeroU32,
    ops::Deref,
    slice,
    sync::atomic::{
        AtomicU32, AtomicU64,
        Ordering::{Acquire, Relaxed, Release},
    },
};
use virtual_buffer::vec::Vec;

pub mod epoch;

/// The slot index used to signify the lack thereof.
const NIL: u32 = u32::MAX;

#[cfg_attr(not(doc), repr(C))]
pub struct SlotMap<T, C: Collector<T> = DefaultCollector> {
    slots: Vec<Slot<T>>,
    len: AtomicU32,
    global: epoch::GlobalHandle,
    collector: C,

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

impl<T> SlotMap<T, DefaultCollector> {
    #[must_use]
    pub fn new(max_capacity: u32) -> Self {
        Self::with_global(max_capacity, epoch::GlobalHandle::new())
    }

    #[must_use]
    pub fn with_global(max_capacity: u32, global: epoch::GlobalHandle) -> Self {
        Self::with_global_and_collector(max_capacity, global, DefaultCollector)
    }
}

impl<T, C: Collector<T>> SlotMap<T, C> {
    #[must_use]
    pub fn with_collector(max_capacity: u32, collector: C) -> Self {
        Self::with_global_and_collector(max_capacity, epoch::GlobalHandle::new(), collector)
    }

    #[must_use]
    pub fn with_global_and_collector(
        max_capacity: u32,
        global: epoch::GlobalHandle,
        collector: C,
    ) -> Self {
        SlotMap {
            slots: Vec::new(max_capacity as usize),
            len: AtomicU32::new(0),
            global,
            collector,
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

    #[inline]
    #[must_use]
    pub fn global(&self) -> &epoch::GlobalHandle {
        &self.global
    }

    #[inline]
    #[must_use]
    pub fn collector(&self) -> &C {
        &self.collector
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` does not equal `self.global()`.
    #[inline]
    pub fn insert<'a>(&'a self, value: T, guard: impl Into<Cow<'a, epoch::Guard>>) -> SlotId {
        self.insert_inner(value, guard.into())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn insert_inner<'a>(&'a self, value: T, guard: Cow<'a, epoch::Guard>) -> SlotId {
        assert_eq!(guard.global(), &self.global);

        let mut free_list_head = self.free_list.load(Acquire);
        let mut backoff = Backoff::new();

        loop {
            if free_list_head == NIL {
                break;
            }

            // SAFETY: We always push indices of existing slots into the free-lists and the slots
            // vector never shrinks, therefore the index must have staid in bounds.
            let slot = unsafe { self.slots.get_unchecked(free_list_head as usize) };

            let next_free = slot.next_free.load(Relaxed);

            match self
                .free_list
                .compare_exchange_weak(free_list_head, next_free, Release, Acquire)
            {
                Ok(_) => {
                    // SAFETY: `SlotMap::remove[_mut]` guarantees that the free-list only contains
                    // slots that are no longer read by any threads, and we have removed the slot
                    // from the free-list such that no other threads can be writing the same slot.
                    unsafe { slot.value.get().cast::<T>().write(value) };

                    let new_generation = slot.generation.fetch_add(1, Release).wrapping_add(1);

                    self.len.fetch_add(1, Relaxed);

                    // SAFETY: `SlotMap::remove[_mut]` guarantees that a freed slot has its
                    // generation's `OCCUPIED_BIT` unset, and since we incremented the generation,
                    // the bit must have been flipped again.
                    return unsafe { SlotId::new_unchecked(free_list_head, new_generation) };
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
        // SAFETY: The `OCCUPIED_BIT` is set.
        unsafe {
            SlotId::new_unchecked(index as u32, OCCUPIED_BIT)
        }
    }

    pub fn insert_mut(&mut self, value: T) -> SlotId {
        let free_list_head = *self.free_list.get_mut();

        if free_list_head != NIL {
            // SAFETY: We always push indices of existing slots into the free-lists and the slots
            // vector never shrinks, therefore the index must have staid in bounds.
            let slot = unsafe { self.slots.get_unchecked_mut(free_list_head as usize) };

            let new_generation = slot.generation.get_mut().wrapping_add(1);

            *slot.generation.get_mut() = new_generation;

            *self.free_list.get_mut() = *slot.next_free.get_mut();

            *slot.value.get_mut() = MaybeUninit::new(value);

            *self.len.get_mut() += 1;

            // SAFETY: `SlotMap::remove[_mut]` guarantees that a freed slot has its generation's
            // `OCCUPIED_BIT` unset, and since we incremented the generation, the bit must have been
            // flipped again.
            return unsafe { SlotId::new_unchecked(free_list_head, new_generation) };
        }

        let index = self.slots.push_mut(Slot::new(value));

        *self.len.get_mut() += 1;

        // Our capacity can never exceed `u32::MAX`.
        #[allow(clippy::cast_possible_truncation)]
        // SAFETY: The `OCCUPIED_BIT` is set.
        unsafe {
            SlotId::new_unchecked(index as u32, OCCUPIED_BIT)
        }
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` does not equal `self.global()`.
    #[inline]
    pub fn remove<'a>(
        &'a self,
        id: SlotId,
        guard: impl Into<Cow<'a, epoch::Guard>>,
    ) -> Option<Ref<'a, T>> {
        self.remove_inner(id, guard.into())
    }

    fn remove_inner<'a>(&'a self, id: SlotId, guard: Cow<'a, epoch::Guard>) -> Option<Ref<'a, T>> {
        assert_eq!(guard.global(), &self.global);

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

        // SAFETY:
        // * The `Acquire` ordering when loading the slot's generation synchronizes with the
        //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value is
        //   visible here.
        // * The `compare_exchange` above succeeded, which means that the previous generation of the
        //   slot must have matched `id.generation`. By `SlotId`'s invariant, its generation's
        //   occupied bit must be set. Since the generation matched, the slot's occupied bit must
        //   have been set, which makes reading the value safe as the only way the occupied bit can
        //   be set is in `SlotMap::insert[_mut]` after initialization of the slot.
        // * We unset the slot's `OCCUPIED_BIT` such that no other threads can be attempting to push
        //   it into the free-lists.
        Some(unsafe { self.remove_inner_inner(id.index, guard) })
    }

    // Inner indeed.
    unsafe fn remove_inner_inner<'a>(
        &'a self,
        index: u32,
        guard: Cow<'a, epoch::Guard>,
    ) -> Ref<'a, T> {
        // SAFETY: The caller must ensure that `index` is in bounds.
        let slot = unsafe { self.slots.get_unchecked(index as usize) };

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

                let new_state = u64::from(index) | u64::from(queued_epoch) << 32;

                match queued_list.compare_exchange_weak(queued_state, new_state, Release, Acquire) {
                    Ok(_) => {
                        self.len.fetch_sub(1, Relaxed);

                        // SAFETY: The caller must ensure that the inner value was initialized and
                        // that said write was synchronized with and made visible here.
                        break unsafe { Ref { slot, guard } };
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

                let new_state = u64::from(index) | u64::from(epoch) << 32;

                match queued_list.compare_exchange_weak(queued_state, new_state, Release, Acquire) {
                    Ok(_) => {
                        self.len.fetch_sub(1, Relaxed);

                        // SAFETY: Having ended up here, the global epoch must have been advanced
                        // at least 2 steps from the last push into the queued list and we removed
                        // the list from the queue, which means that no other threads can be
                        // accessing any of the slots in the list.
                        unsafe { self.collect_garbage(queued_head) };

                        // SAFETY: The caller must ensure that the inner value was initialized and
                        // that said write was synchronized with and made visible here.
                        break unsafe { Ref { slot, guard } };
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
            queued_tail_slot = unsafe { self.slots.get_unchecked(queued_tail as usize) };

            let ptr = queued_tail_slot.value.get().cast::<T>();

            // SAFETY: The caller must ensure that we have exclusive access to this list.
            unsafe { self.collector.collect(ptr) };

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

    pub fn remove_mut(&mut self, id: SlotId) -> Option<T> {
        let slot = self.slots.get_mut(id.index as usize)?;
        let generation = *slot.generation.get_mut();

        if generation == id.generation() {
            *slot.generation.get_mut() = generation.wrapping_add(1);

            *slot.next_free.get_mut() = *self.free_list.get_mut();
            *self.free_list.get_mut() = id.index;

            *self.len.get_mut() -= 1;

            // SAFETY:
            // * The mutable reference makes sure that access to the slot is synchronized.
            // * We checked that `id.generation` matches the slot's generation, which includes the
            //   occupied bit. By `SlotId`'s invariant, its generation's occupied bit must be set.
            //   Since the generation matched, the slot's occupied bit must be set, which makes
            //   reading the value safe as the only way the occupied bit can be set is in
            //   `SlotMap::insert[_mut]` after initialization of the slot.
            // * We incremented the slot's generation such that its `OCCUPIED_BIT` is unset and its
            //   generation is advanced, such that future attempts to access the slot will fail.
            Some(unsafe { slot.value.get().cast::<T>().read() })
        } else {
            None
        }
    }

    #[cfg(test)]
    fn remove_index<'a>(
        &'a self,
        index: u32,
        guard: impl Into<Cow<'a, epoch::Guard>>,
    ) -> Option<Ref<'a, T>> {
        self.remove_index_inner(index, guard.into())
    }

    #[cfg(test)]
    fn remove_index_inner<'a>(
        &'a self,
        index: u32,
        guard: Cow<'a, epoch::Guard>,
    ) -> Option<Ref<'a, T>> {
        assert_eq!(guard.global(), &self.global);

        let slot = self.slots.get(index as usize)?;
        let mut generation = slot.generation.load(Relaxed);
        let mut backoff = Backoff::new();

        loop {
            if generation & OCCUPIED_BIT == 0 {
                return None;
            }

            match slot.generation.compare_exchange_weak(
                generation,
                generation.wrapping_add(1),
                Acquire,
                Relaxed,
            ) {
                Ok(_) => break,
                Err(new_generation) => {
                    generation = new_generation;
                    backoff.spin();
                }
            }
        }

        // SAFETY:
        // * The `Acquire` ordering when loading the slot's generation synchronizes with the
        //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value is
        //   visible here.
        // * The `compare_exchange_weak` above succeeded, which means that the previous generation
        //   of the slot must have had its `OCCUPIED_BIT` set, which makes reading the value safe as
        //   the only way the occupied bit can be set is in `SlotMap::insert[_mut]` after
        //   initialization of the slot.
        // * We unset the slot's `OCCUPIED_BIT` such that no other threads can be attempting to push
        //   it into the free-lists.
        Some(unsafe { self.remove_inner_inner(index, guard) })
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` does not equal `self.global()`.
    #[inline(always)]
    #[must_use]
    pub fn get<'a>(
        &'a self,
        id: SlotId,
        guard: impl Into<Cow<'a, epoch::Guard>>,
    ) -> Option<Ref<'a, T>> {
        self.get_inner(id, guard.into())
    }

    #[inline(always)]
    fn get_inner<'a>(&'a self, id: SlotId, guard: Cow<'a, epoch::Guard>) -> Option<Ref<'a, T>> {
        assert_eq!(guard.global(), &self.global);

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
            //   `SlotMap::insert[_mut]` after initialization of the slot.
            Some(unsafe { Ref { slot, guard } })
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// You must ensure that the epoch is [pinned] before you call this method and that the
    /// returned reference doesn't outlive all [`epoch::Guard`]s active on the thread, or that all
    /// accesses to `self` are externally synchronized (for example through the use of a `Mutex` or
    /// by being single-threaded).
    ///
    /// [pinned]: epoch::pin
    #[inline(always)]
    pub unsafe fn get_unprotected(&self, id: SlotId) -> Option<&T> {
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
            //   `SlotMap::insert[_mut]` after initialization of the slot.
            // * The caller must ensure that the returned reference is protected by a guard before
            //   the call and that the returned reference doesn't outlive said guard, or that
            //   synchronization is ensured externally.
            Some(unsafe { slot.value_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, id: SlotId) -> Option<&mut T> {
        let slot = self.slots.get_mut(id.index as usize)?;
        let generation = *slot.generation.get_mut();

        if generation == id.generation() {
            // SAFETY:
            // * The mutable reference makes sure that access to the slot is synchronized.
            // * We checked that `id.generation` matches the slot's generation, which includes the
            //   occupied bit. By `SlotId`'s invariant, its generation's occupied bit must be set.
            //   Since the generation matched, the slot's occupied bit must be set, which makes
            //   reading the value safe as the only way the occupied bit can be set is in
            //   `SlotMap::insert[_mut]` after initialization of the slot.
            Some(unsafe { slot.value_unchecked_mut() })
        } else {
            None
        }
    }

    #[inline]
    pub fn get_many_mut<const N: usize>(&mut self, ids: [SlotId; N]) -> Option<[&mut T; N]> {
        fn get_many_check_valid<const N: usize>(ids: &[SlotId; N], len: u32) -> bool {
            let mut valid = true;

            for (i, id) in ids.iter().enumerate() {
                valid &= id.index() < len;

                for id2 in &ids[..i] {
                    valid &= id.index() != id2.index();
                }
            }

            valid
        }

        // Our capacity can never exceed `u32::MAX`, so the length of the slots can't either.
        #[allow(clippy::cast_possible_truncation)]
        let len = self.slots.len() as u32;

        if get_many_check_valid(&ids, len) {
            // SAFETY: We checked that all indices are disjunct and in bounds of the slots vector.
            unsafe { self.get_many_unchecked_mut(ids) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_many_unchecked_mut<const N: usize>(
        &mut self,
        ids: [SlotId; N],
    ) -> Option<[&mut T; N]> {
        let slots_ptr = self.slots.as_mut_ptr();
        let mut refs = MaybeUninit::<[&mut T; N]>::uninit();
        let refs_ptr = refs.as_mut_ptr().cast::<&mut T>();

        for i in 0..N {
            // SAFETY: `i` is in bounds of the array.
            let id = unsafe { ids.get_unchecked(i) };

            // SAFETY: The caller must ensure that `ids` contains only IDs whose indices are in
            // bounds of the slots vector.
            let slot = unsafe { slots_ptr.add(id.index() as usize) };

            // SAFETY: The caller must ensure that `ids` contains only IDs with disjunct indices.
            let slot = unsafe { &mut *slot };

            let generation = *slot.generation.get_mut();

            if generation != id.generation() {
                return None;
            }

            // SAFETY:
            // * The mutable reference makes sure that access to the slot is synchronized.
            // * We checked that `id.generation` matches the slot's generation, which includes the
            //   occupied bit. By `SlotId`'s invariant, its generation's occupied bit must be set.
            //   Since the generation matched, the slot's occupied bit must be set, which makes
            //   reading the value safe as the only way the occupied bit can be set is in
            //   `SlotMap::insert[_mut]` after initialization of the slot.
            let value = unsafe { slot.value_unchecked_mut() };

            // SAFETY: `i` is in bounds of the array.
            unsafe { *refs_ptr.add(i) = value };
        }

        // SAFETY: We initialized all the elements.
        Some(unsafe { refs.assume_init() })
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` does not equal `self.global()`.
    #[inline(always)]
    pub fn index<'a>(
        &'a self,
        index: u32,
        guard: impl Into<Cow<'a, epoch::Guard>>,
    ) -> Option<Ref<'a, T>> {
        self.index_inner(index, guard.into())
    }

    #[inline(always)]
    fn index_inner<'a>(&'a self, index: u32, guard: Cow<'a, epoch::Guard>) -> Option<Ref<'a, T>> {
        assert_eq!(guard.global(), &self.global);

        let slot = self.slots.get(index as usize)?;
        let generation = slot.generation.load(Acquire);

        if generation & OCCUPIED_BIT != 0 {
            // SAFETY:
            // * The `Acquire` ordering when loading the slot's generation synchronizes with the
            //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value
            //   is visible here.
            // * We checked that the slot is occupied, which means that it must have been
            //   initialized in `SlotMap::insert[_mut]`.
            Some(unsafe { Ref { slot, guard } })
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// You must ensure that the epoch is [pinned] before you call this method and that the
    /// returned reference doesn't outlive all [`epoch::Guard`]s active on the thread, or that all
    /// accesses to `self` are externally synchronized (for example through the use of a `Mutex` or
    /// by being single-threaded).
    ///
    /// [pinned]: epoch::pin
    #[inline(always)]
    pub unsafe fn index_unprotected(&self, index: u32) -> Option<&T> {
        let slot = self.slots.get(index as usize)?;
        let generation = slot.generation.load(Acquire);

        if generation & OCCUPIED_BIT != 0 {
            // SAFETY:
            // * The `Acquire` ordering when loading the slot's generation synchronizes with the
            //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value
            //   is visible here.
            // * We checked that the slot is occupied, which means that it must have been
            //   initialized in `SlotMap::insert[_mut]`.
            // * The caller must ensure that the returned reference is protected by a guard before
            //   the call and that the returned reference doesn't outlive said guard, or that
            //   synchronization is ensured externally.
            Some(unsafe { slot.value_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn index_mut(&mut self, index: u32) -> Option<&mut T> {
        let slot = self.slots.get_mut(index as usize)?;
        let generation = *slot.generation.get_mut();

        if generation & OCCUPIED_BIT != 0 {
            // SAFETY:
            // * The mutable reference makes sure that access to the slot is synchronized.
            // * We checked that the slot is occupied, which means that it must have been
            //   initialized in `SlotMap::insert[_mut]`.
            Some(unsafe { slot.value_unchecked_mut() })
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// `index` must be in bounds of the slots vector and the slot must have been initialized and
    /// must not be free.
    ///
    /// # Panics
    ///
    /// Panics if `guard.global()` does not equal `self.global()`.
    #[inline(always)]
    pub unsafe fn index_unchecked<'a>(
        &'a self,
        index: u32,
        guard: impl Into<Cow<'a, epoch::Guard>>,
    ) -> Ref<'a, T> {
        // SAFETY: Ensured by the caller.
        unsafe { self.index_unchecked_inner(index, guard.into()) }
    }

    #[inline(always)]
    unsafe fn index_unchecked_inner<'a>(
        &'a self,
        index: u32,
        guard: Cow<'a, epoch::Guard>,
    ) -> Ref<'a, T> {
        assert_eq!(guard.global(), &self.global);

        // SAFETY: The caller must ensure that the index is in bounds.
        let slot = unsafe { self.slots.get_unchecked(index as usize) };

        let _generation = slot.generation.load(Acquire);

        // SAFETY:
        // * The `Acquire` ordering when loading the slot's generation synchronizes with the
        //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value is
        //   visible here.
        // * The caller must ensure that the slot is initialized.
        unsafe { Ref { slot, guard } }
    }

    /// # Safety
    ///
    /// * `index` must be in bounds of the slots vector and the slot must have been initialized and
    ///   must not be free.
    /// * You must ensure that the epoch is [pinned] before you call this method and that the
    ///   returned reference doesn't outlive all [`epoch::Guard`]s active on the thread, or that
    ///   all accesses to `self` are externally synchronized (for example through the use of a
    ///   `Mutex` or by being single-threaded).
    ///
    /// [pinned]: epoch::pin
    #[inline(always)]
    pub unsafe fn index_unchecked_unprotected(&self, index: u32) -> &T {
        // SAFETY: The caller must ensure that the index is in bounds.
        let slot = unsafe { self.slots.get_unchecked(index as usize) };

        let _generation = slot.generation.load(Acquire);

        // SAFETY:
        // * The `Acquire` ordering when loading the slot's generation synchronizes with the
        //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value is
        //   visible here.
        // * The caller must ensure that the slot is initialized.
        // * The caller must ensure that the returned reference is protected by a guard before the
        //   call and that the returned reference doesn't outlive said guard, or that
        //   synchronization is ensured externally.
        unsafe { slot.value_unchecked() }
    }

    /// # Safety
    ///
    /// `index` must be in bounds of the slots vector and the slot must have been initialized and
    /// must not be free.
    #[inline(always)]
    pub unsafe fn index_unchecked_mut(&mut self, index: u32) -> &mut T {
        // SAFETY: The caller must ensure that the index is in bounds.
        let slot = unsafe { self.slots.get_unchecked_mut(index as usize) };

        // SAFETY:
        // * The mutable reference makes sure that access to the slot is synchronized.
        // * The caller must ensure that the slot is initialized.
        unsafe { slot.value_unchecked_mut() }
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` does not equal `self.global()`.
    #[inline]
    pub fn iter<'a>(&'a self, guard: impl Into<Cow<'a, epoch::Guard>>) -> Iter<'a, T> {
        self.iter_inner(guard.into())
    }

    #[inline]
    fn iter_inner<'a>(&'a self, guard: Cow<'a, epoch::Guard>) -> Iter<'a, T> {
        assert_eq!(guard.global(), &self.global);

        Iter {
            slots: self.slots.iter().enumerate(),
            guard,
        }
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            slots: self.slots.iter_mut().enumerate(),
        }
    }
}

// We don't want to print the `_alignment` fields.
#[allow(clippy::missing_fields_in_debug)]
impl<T: fmt::Debug, C: Collector<T>> fmt::Debug for SlotMap<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Slots;

        impl fmt::Debug for Slots {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.pad("[..]")
            }
        }

        struct List<'a, T, C: Collector<T>>(&'a SlotMap<T, C>, u32);

        impl<T, C: Collector<T>> fmt::Debug for List<'_, T, C> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut head = self.1;
                let mut debug = f.debug_list();

                while head != NIL {
                    debug.entry(&head);

                    // SAFETY: We always push indices of existing slots into the free-lists and the
                    // slots vector never shrinks, therefore the index must have staid in bounds.
                    let slot = unsafe { self.0.slots.get_unchecked(head as usize) };

                    head = slot.next_free.load(Acquire);
                }

                debug.finish()
            }
        }

        struct QueuedList<'a, T, C: Collector<T>>(&'a SlotMap<T, C>, u64);

        impl<T, C: Collector<T>> fmt::Debug for QueuedList<'_, T, C> {
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

        let mut debug = f.debug_struct("SlotMap");
        debug
            .field("slots", &Slots)
            .field("len", &self.len)
            .field("global", &self.global)
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

impl<T, C: Collector<T>> Drop for SlotMap<T, C> {
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

                let ptr = slot.value.get_mut().as_mut_ptr();

                // SAFETY: We can be certain that this slot has been initialized, since the only
                // way in which it could have been queued for freeing is in `SlotMap::remove` if
                // the slot was inserted before.
                unsafe { self.collector.collect(ptr) };

                head = *slot.next_free.get_mut();
            }
        }

        for slot in &mut self.slots {
            if *slot.generation.get_mut() & OCCUPIED_BIT != 0 {
                let ptr = slot.value.get_mut().as_mut_ptr();

                // SAFETY:
                // * The mutable reference makes sure that access to the slot is synchronized.
                // * We checked that the slot is occupied, which means that it must have been
                //   initialized in `SlotMap::insert[_mut]`.
                unsafe { self.collector.collect(ptr) };
            }
        }
    }
}

impl<'a, T, C: Collector<T>> IntoIterator for &'a mut SlotMap<T, C> {
    type Item = (SlotId, &'a mut T);

    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
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

    unsafe fn value_unchecked_mut(&mut self) -> &mut T {
        // SAFETY: The caller must ensure that the slot has been initialized.
        unsafe { self.value.get_mut().assume_init_mut() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotId {
    index: u32,
    generation: NonZeroU32,
}

impl SlotId {
    #[cfg(test)]
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

pub struct Iter<'a, T> {
    slots: iter::Enumerate<slice::Iter<'a, Slot<T>>>,
    guard: Cow<'a, epoch::Guard>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (SlotId, Ref<'a, T>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.slots.next()?;
            let generation = slot.generation.load(Acquire);

            if generation & OCCUPIED_BIT != 0 {
                // Our capacity can never exceed `u32::MAX`.
                #[allow(clippy::cast_possible_truncation)]
                // SAFETY: We checked that the occupied bit is set.
                let id = unsafe { SlotId::new_unchecked(index as u32, generation) };

                let guard = self.guard.clone();

                // SAFETY:
                // * The `Acquire` ordering when loading the slot's generation synchronizes with the
                //   `Release` ordering in `SlotMap::insert`, making sure that the newly written
                //   value is visible here.
                // * We checked that the slot is occupied, which means that it must have been
                //   initialized in `SlotMap::insert[_mut]`.
                let r = unsafe { Ref { slot, guard } };

                break Some((id, r));
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.slots.len()))
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.slots.next_back()?;
            let generation = slot.generation.load(Acquire);

            if generation & OCCUPIED_BIT != 0 {
                // Our capacity can never exceed `u32::MAX`.
                #[allow(clippy::cast_possible_truncation)]
                // SAFETY: We checked that the occupied bit is set.
                let id = unsafe { SlotId::new_unchecked(index as u32, generation) };

                let guard = self.guard.clone();

                // SAFETY:
                // * The `Acquire` ordering when loading the slot's generation synchronizes with the
                //   `Release` ordering in `SlotMap::insert`, making sure that the newly written
                //   value is visible here.
                // * We checked that the slot is occupied, which means that it must have been
                //   initialized in `SlotMap::insert[_mut]`.
                let r = unsafe { Ref { slot, guard } };

                break Some((id, r));
            }
        }
    }
}

impl<T> FusedIterator for Iter<'_, T> {}

pub struct IterMut<'a, T> {
    slots: iter::Enumerate<slice::IterMut<'a, Slot<T>>>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (SlotId, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.slots.next()?;
            let generation = *slot.generation.get_mut();

            if generation & OCCUPIED_BIT != 0 {
                // Our capacity can never exceed `u32::MAX`.
                #[allow(clippy::cast_possible_truncation)]
                // SAFETY: We checked that the `OCCUPIED_BIT` is set.
                let id = unsafe { SlotId::new_unchecked(index as u32, generation) };

                // SAFETY: We checked that the slot is occupied, which means that it must have been
                // initialized in `SlotMap::insert[_mut]`.
                let r = unsafe { slot.value_unchecked_mut() };

                break Some((id, r));
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.slots.len()))
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.slots.next_back()?;
            let generation = *slot.generation.get_mut();

            if generation & OCCUPIED_BIT != 0 {
                // Our capacity can never exceed `u32::MAX`.
                #[allow(clippy::cast_possible_truncation)]
                // SAFETY: We checked that the `OCCUPIED_BIT` is set.
                let id = unsafe { SlotId::new_unchecked(index as u32, generation) };

                // SAFETY: We checked that the slot is occupied, which means that it must have been
                // initialized in `SlotMap::insert[_mut]`.
                let r = unsafe { slot.value_unchecked_mut() };

                break Some((id, r));
            }
        }
    }
}

impl<T> FusedIterator for IterMut<'_, T> {}

pub trait Collector<T> {
    /// # Safety
    ///
    /// This function has the same safety preconditions and semantics as [`ptr::drop_in_place`]. It
    /// must be safe to drop the value pointed to by `ptr`.
    ///
    /// [`ptr::drop_in_place`]: core::ptr::drop_in_place
    unsafe fn collect(&self, ptr: *mut T);
}

pub struct DefaultCollector;

impl<T> Collector<T> for DefaultCollector {
    #[inline(always)]
    unsafe fn collect(&self, ptr: *mut T) {
        // SAFETY: The caller must ensure that it is safe to drop the value.
        unsafe { ptr.drop_in_place() }
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

#[cfg(test)]
mod tests {
    use self::epoch::PINNINGS_BETWEEN_ADVANCE;
    use super::*;
    use std::thread;

    #[test]
    fn basic_usage1() {
        let map = SlotMap::new(10);
        let guard = &map.global().register_local().pin();

        let x = map.insert(69, guard);
        let y = map.insert(42, guard);

        assert_eq!(map.get(x, guard).as_deref(), Some(&69));
        assert_eq!(map.get(y, guard).as_deref(), Some(&42));

        map.remove(x, guard);

        let x2 = map.insert(12, guard);

        assert_eq!(map.get(x2, guard).as_deref(), Some(&12));
        assert_eq!(map.get(x, guard).as_deref(), None);

        map.remove(y, guard);
        map.remove(x2, guard);

        assert_eq!(map.get(y, guard).as_deref(), None);
        assert_eq!(map.get(x2, guard).as_deref(), None);
    }

    #[test]
    fn basic_usage2() {
        let map = SlotMap::new(10);
        let guard = &map.global().register_local().pin();

        let x = map.insert(1, guard);
        let y = map.insert(2, guard);
        let z = map.insert(3, guard);

        assert_eq!(map.get(x, guard).as_deref(), Some(&1));
        assert_eq!(map.get(y, guard).as_deref(), Some(&2));
        assert_eq!(map.get(z, guard).as_deref(), Some(&3));

        map.remove(y, guard);

        let y2 = map.insert(20, guard);

        assert_eq!(map.get(y2, guard).as_deref(), Some(&20));
        assert_eq!(map.get(y, guard).as_deref(), None);

        map.remove(x, guard);
        map.remove(z, guard);

        let x2 = map.insert(10, guard);

        assert_eq!(map.get(x2, guard).as_deref(), Some(&10));
        assert_eq!(map.get(x, guard).as_deref(), None);

        let z2 = map.insert(30, guard);

        assert_eq!(map.get(z2, guard).as_deref(), Some(&30));
        assert_eq!(map.get(x, guard).as_deref(), None);

        map.remove(x2, guard);

        assert_eq!(map.get(x2, guard).as_deref(), None);

        map.remove(y2, guard);
        map.remove(z2, guard);

        assert_eq!(map.get(y2, guard).as_deref(), None);
        assert_eq!(map.get(z2, guard).as_deref(), None);
    }

    #[test]
    fn basic_usage3() {
        let map = SlotMap::new(10);
        let guard = &map.global().register_local().pin();

        let x = map.insert(1, guard);
        let y = map.insert(2, guard);

        assert_eq!(map.get(x, guard).as_deref(), Some(&1));
        assert_eq!(map.get(y, guard).as_deref(), Some(&2));

        let z = map.insert(3, guard);

        assert_eq!(map.get(z, guard).as_deref(), Some(&3));

        map.remove(x, guard);
        map.remove(z, guard);

        let z2 = map.insert(30, guard);
        let x2 = map.insert(10, guard);

        assert_eq!(map.get(x2, guard).as_deref(), Some(&10));
        assert_eq!(map.get(z2, guard).as_deref(), Some(&30));
        assert_eq!(map.get(x, guard).as_deref(), None);
        assert_eq!(map.get(z, guard).as_deref(), None);

        map.remove(x2, guard);
        map.remove(y, guard);
        map.remove(z2, guard);

        assert_eq!(map.get(x2, guard).as_deref(), None);
        assert_eq!(map.get(y, guard).as_deref(), None);
        assert_eq!(map.get(z2, guard).as_deref(), None);
    }

    #[test]
    fn basic_usage_mut1() {
        let mut map = SlotMap::new(10);

        let x = map.insert_mut(69);
        let y = map.insert_mut(42);

        assert_eq!(map.get_mut(x), Some(&mut 69));
        assert_eq!(map.get_mut(y), Some(&mut 42));

        map.remove_mut(x);

        let x2 = map.insert_mut(12);

        assert_eq!(map.get_mut(x2), Some(&mut 12));
        assert_eq!(map.get_mut(x), None);

        map.remove_mut(y);
        map.remove_mut(x2);

        assert_eq!(map.get_mut(y), None);
        assert_eq!(map.get_mut(x2), None);
    }

    #[test]
    fn basic_usage_mut2() {
        let mut map = SlotMap::new(10);

        let x = map.insert_mut(1);
        let y = map.insert_mut(2);
        let z = map.insert_mut(3);

        assert_eq!(map.get_mut(x), Some(&mut 1));
        assert_eq!(map.get_mut(y), Some(&mut 2));
        assert_eq!(map.get_mut(z), Some(&mut 3));

        map.remove_mut(y);

        let y2 = map.insert_mut(20);

        assert_eq!(map.get_mut(y2), Some(&mut 20));
        assert_eq!(map.get_mut(y), None);

        map.remove_mut(x);
        map.remove_mut(z);

        let x2 = map.insert_mut(10);

        assert_eq!(map.get_mut(x2), Some(&mut 10));
        assert_eq!(map.get_mut(x), None);

        let z2 = map.insert_mut(30);

        assert_eq!(map.get_mut(z2), Some(&mut 30));
        assert_eq!(map.get_mut(x), None);

        map.remove_mut(x2);

        assert_eq!(map.get_mut(x2), None);

        map.remove_mut(y2);
        map.remove_mut(z2);

        assert_eq!(map.get_mut(y2), None);
        assert_eq!(map.get_mut(z2), None);
    }

    #[test]
    fn basic_usage_mut3() {
        let mut map = SlotMap::new(10);

        let x = map.insert_mut(1);
        let y = map.insert_mut(2);

        assert_eq!(map.get_mut(x), Some(&mut 1));
        assert_eq!(map.get_mut(y), Some(&mut 2));

        let z = map.insert_mut(3);

        assert_eq!(map.get_mut(z), Some(&mut 3));

        map.remove_mut(x);
        map.remove_mut(z);

        let z2 = map.insert_mut(30);
        let x2 = map.insert_mut(10);

        assert_eq!(map.get_mut(x2), Some(&mut 10));
        assert_eq!(map.get_mut(z2), Some(&mut 30));
        assert_eq!(map.get_mut(x), None);
        assert_eq!(map.get_mut(z), None);

        map.remove_mut(x2);
        map.remove_mut(y);
        map.remove_mut(z2);

        assert_eq!(map.get_mut(x2), None);
        assert_eq!(map.get_mut(y), None);
        assert_eq!(map.get_mut(z2), None);
    }

    #[test]
    fn iter1() {
        let map = SlotMap::new(10);
        let guard = &map.global().register_local().pin();

        let x = map.insert(1, guard);
        let _ = map.insert(2, guard);
        let y = map.insert(3, guard);

        let mut iter = map.iter(guard);

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(x, guard);
        map.remove(y, guard);

        let mut iter = map.iter(guard);

        assert_eq!(*iter.next().unwrap().1, 2);
        assert!(iter.next().is_none());

        map.insert(3, guard);
        map.insert(1, guard);

        let mut iter = map.iter(guard);

        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert_eq!(*iter.next().unwrap().1, 1);
        assert!(iter.next().is_none());
    }

    #[test]
    fn iter2() {
        let map = SlotMap::new(10);
        let guard = &map.global().register_local().pin();

        let x = map.insert(1, guard);
        let y = map.insert(2, guard);
        let z = map.insert(3, guard);

        map.remove(x, guard);

        let mut iter = map.iter(guard);

        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(y, guard);

        let mut iter = map.iter(guard);

        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(z, guard);

        let mut iter = map.iter(guard);

        assert!(iter.next().is_none());
    }

    #[test]
    fn iter3() {
        let map = SlotMap::new(10);
        let guard = &map.global().register_local().pin();

        let _ = map.insert(1, guard);
        let x = map.insert(2, guard);

        let mut iter = map.iter(guard);

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert!(iter.next().is_none());

        map.remove(x, guard);

        let x = map.insert(2, guard);
        let _ = map.insert(3, guard);
        let y = map.insert(4, guard);

        map.remove(y, guard);

        let mut iter = map.iter(guard);

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(x, guard);

        let mut iter = map.iter(guard);

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());
    }

    #[test]
    fn get_many_mut() {
        let mut map = SlotMap::new(3);

        let x = map.insert_mut(1);
        let y = map.insert_mut(2);
        let z = map.insert_mut(3);

        assert_eq!(map.get_many_mut([x, y]), Some([&mut 1, &mut 2]));
        assert_eq!(map.get_many_mut([y, z]), Some([&mut 2, &mut 3]));
        assert_eq!(map.get_many_mut([z, x]), Some([&mut 3, &mut 1]));

        assert_eq!(map.get_many_mut([x, y, z]), Some([&mut 1, &mut 2, &mut 3]));
        assert_eq!(map.get_many_mut([z, y, x]), Some([&mut 3, &mut 2, &mut 1]));

        assert_eq!(map.get_many_mut([x, x]), None);
        assert_eq!(map.get_many_mut([x, SlotId::new(3, OCCUPIED_BIT)]), None);

        map.remove_mut(y);

        assert_eq!(map.get_many_mut([x, z]), Some([&mut 1, &mut 3]));

        assert_eq!(map.get_many_mut([y]), None);
        assert_eq!(map.get_many_mut([x, y]), None);
        assert_eq!(map.get_many_mut([y, z]), None);

        let y = map.insert_mut(2);

        assert_eq!(map.get_many_mut([x, y, z]), Some([&mut 1, &mut 2, &mut 3]));

        map.remove_mut(x);
        map.remove_mut(z);

        assert_eq!(map.get_many_mut([y]), Some([&mut 2]));

        assert_eq!(map.get_many_mut([x]), None);
        assert_eq!(map.get_many_mut([z]), None);

        map.remove_mut(y);

        assert_eq!(map.get_many_mut([]), Some([]));
    }

    #[test]
    fn iter_mut1() {
        let mut map = SlotMap::new(10);

        let x = map.insert_mut(1);
        let _ = map.insert_mut(2);
        let y = map.insert_mut(3);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove_mut(x);
        map.remove_mut(y);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 2);
        assert!(iter.next().is_none());

        map.insert_mut(3);
        map.insert_mut(1);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());
    }

    #[test]
    fn iter_mut2() {
        let mut map = SlotMap::new(10);

        let x = map.insert_mut(1);
        let y = map.insert_mut(2);
        let z = map.insert_mut(3);

        map.remove_mut(x);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove_mut(y);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove_mut(z);

        let mut iter = map.iter_mut();

        assert!(iter.next().is_none());
    }

    #[test]
    fn iter_mut3() {
        let mut map = SlotMap::new(10);

        let _ = map.insert_mut(1);
        let x = map.insert_mut(2);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert!(iter.next().is_none());

        map.remove_mut(x);

        let x = map.insert_mut(2);
        let _ = map.insert_mut(3);
        let y = map.insert_mut(4);

        map.remove_mut(y);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove_mut(x);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());
    }

    #[test]
    fn reusing_slots_mut1() {
        let mut map = SlotMap::new(10);

        let x = map.insert_mut(0);
        let y = map.insert_mut(0);

        map.remove_mut(y);

        let y2 = map.insert_mut(0);
        assert_eq!(y2.index, y.index);
        assert_ne!(y2.generation, y.generation);

        map.remove_mut(x);

        let x2 = map.insert_mut(0);
        assert_eq!(x2.index, x.index);
        assert_ne!(x2.generation, x.generation);

        map.remove_mut(y2);
        map.remove_mut(x2);
    }

    #[test]
    fn reusing_slots_mut2() {
        let mut map = SlotMap::new(10);

        let x = map.insert_mut(0);

        map.remove_mut(x);

        let x2 = map.insert_mut(0);
        assert_eq!(x.index, x2.index);
        assert_ne!(x.generation, x2.generation);

        let y = map.insert_mut(0);
        let z = map.insert_mut(0);

        map.remove_mut(y);
        map.remove_mut(x2);

        let x3 = map.insert_mut(0);
        let y2 = map.insert_mut(0);
        assert_eq!(x3.index, x2.index);
        assert_ne!(x3.generation, x2.generation);
        assert_eq!(y2.index, y.index);
        assert_ne!(y2.generation, y.generation);

        map.remove_mut(x3);
        map.remove_mut(y2);
        map.remove_mut(z);
    }

    #[test]
    fn reusing_slots_mut3() {
        let mut map = SlotMap::new(10);

        let x = map.insert_mut(0);
        let y = map.insert_mut(0);

        map.remove_mut(x);
        map.remove_mut(y);

        let y2 = map.insert_mut(0);
        let x2 = map.insert_mut(0);
        let z = map.insert_mut(0);
        assert_eq!(x2.index, x.index);
        assert_ne!(x2.generation, x.generation);
        assert_eq!(y2.index, y.index);
        assert_ne!(y2.generation, y.generation);

        map.remove_mut(x2);
        map.remove_mut(z);
        map.remove_mut(y2);

        let y3 = map.insert_mut(0);
        let z2 = map.insert_mut(0);
        let x3 = map.insert_mut(0);
        assert_eq!(y3.index, y2.index);
        assert_ne!(y3.generation, y2.generation);
        assert_eq!(z2.index, z.index);
        assert_ne!(z2.generation, z.generation);
        assert_eq!(x3.index, x2.index);
        assert_ne!(x3.generation, x2.generation);

        map.remove_mut(x3);
        map.remove_mut(y3);
        map.remove_mut(z2);
    }

    // TODO: Testing concurrent generational collections is the most massive pain in the ass. We
    // aren't testing the actual implementations but rather ones that don't take the generation into
    // account because of that.

    #[cfg(not(miri))]
    const ITERATIONS: u32 = 1_000_000;
    #[cfg(miri)]
    const ITERATIONS: u32 = 1_000;

    #[test]
    fn multi_threaded1() {
        const THREADS: u32 = 2;

        let map = SlotMap::new(ITERATIONS);

        thread::scope(|s| {
            let inserter = || {
                let local = map.global().register_local();

                for _ in 0..ITERATIONS / THREADS {
                    map.insert(0, local.pin());
                }
            };

            for _ in 0..THREADS {
                s.spawn(inserter);
            }
        });

        assert_eq!(map.len(), ITERATIONS);

        thread::scope(|s| {
            let remover = || {
                let local = map.global().register_local();

                for index in 0..ITERATIONS {
                    let _ = map.remove(SlotId::new(index, OCCUPIED_BIT), local.pin());
                }
            };

            for _ in 0..THREADS {
                s.spawn(remover);
            }
        });

        assert_eq!(map.len(), 0);
    }

    // TODO: This test is just fundamentally broken.
    #[test]
    fn multi_threaded2() {
        const CAPACITY: u32 = PINNINGS_BETWEEN_ADVANCE as u32 * 3;

        let map = SlotMap::new(ITERATIONS / 2);

        thread::scope(|s| {
            let insert_remover = || {
                let local = map.global().register_local();

                for _ in 0..ITERATIONS / 6 {
                    let x = map.insert(0, local.pin());
                    let y = map.insert(0, local.pin());
                    map.remove(y, local.pin());
                    let z = map.insert(0, local.pin());
                    map.remove(x, local.pin());
                    map.remove(z, local.pin());
                }
            };
            let iterator = || {
                let local = map.global().register_local();

                for _ in 0..ITERATIONS / CAPACITY * 2 {
                    for index in 0..CAPACITY {
                        if let Some(value) = map.index(index, local.pin()) {
                            let _ = *value;
                        }
                    }
                }
            };

            s.spawn(iterator);
            s.spawn(iterator);
            s.spawn(iterator);
            s.spawn(insert_remover);
        });
    }

    #[test]
    fn multi_threaded3() {
        let map = SlotMap::new(ITERATIONS / 10);

        thread::scope(|s| {
            let inserter = || {
                let local = map.global().register_local();

                for i in 0..ITERATIONS {
                    if i % 10 == 0 {
                        map.insert(0, local.pin());
                    } else {
                        thread::yield_now();
                    }
                }
            };
            let remover = || {
                let local = map.global().register_local();

                for _ in 0..ITERATIONS {
                    map.remove_index(0, local.pin());
                }
            };
            let getter = || {
                let local = map.global().register_local();

                for _ in 0..ITERATIONS {
                    if let Some(value) = map.index(0, local.pin()) {
                        let _ = *value;
                    }
                }
            };

            s.spawn(getter);
            s.spawn(getter);
            s.spawn(getter);
            s.spawn(getter);
            s.spawn(remover);
            s.spawn(remover);
            s.spawn(remover);
            s.spawn(inserter);
        });
    }
}
