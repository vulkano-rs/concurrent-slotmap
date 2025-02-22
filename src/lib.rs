#![allow(unused_unsafe, clippy::inline_always)]
#![warn(rust_2018_idioms, missing_debug_implementations)]
#![forbid(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use core::{
    fmt, hint,
    iter::{self, FusedIterator},
    marker::PhantomData,
    mem::MaybeUninit,
    num::NonZeroU32,
    panic::{RefUnwindSafe, UnwindSafe},
    slice,
    sync::atomic::{
        self, AtomicU32, AtomicU64,
        Ordering::{Acquire, Relaxed, Release, SeqCst},
    },
};
use slot::{Slot, Vec};

pub mod epoch;
mod slot;

/// The slot index used to signify the lack thereof.
const NIL: u32 = u32::MAX;

/// The number of low bits that can be used for tagged generations.
const TAG_BITS: u32 = 8;

/// The mask for tagged generations.
const TAG_MASK: u32 = (1 << TAG_BITS) - 1;

/// The bit used to signify that a slot is occupied.
const OCCUPIED_BIT: u32 = 1 << TAG_BITS;

pub struct SlotMap<K, V> {
    inner: SlotMapInner<V>,
    marker: PhantomData<fn(K) -> K>,
}

#[repr(C)]
struct SlotMapInner<V> {
    slots: Vec<V>,
    len: AtomicU32,
    global: epoch::GlobalHandle,

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

impl<K, V> UnwindSafe for SlotMap<K, V> {}
impl<K, V> RefUnwindSafe for SlotMap<K, V> {}

impl<V> SlotMap<SlotId, V> {
    #[must_use]
    pub fn new(max_capacity: u32) -> Self {
        Self::with_key(max_capacity)
    }

    #[must_use]
    pub fn with_global(max_capacity: u32, global: epoch::GlobalHandle) -> Self {
        Self::with_global_and_key(max_capacity, global)
    }
}

impl<K, V> SlotMap<K, V> {
    #[must_use]
    pub fn with_key(max_capacity: u32) -> Self {
        Self::with_global_and_key(max_capacity, epoch::GlobalHandle::new())
    }

    #[must_use]
    pub fn with_global_and_key(max_capacity: u32, global: epoch::GlobalHandle) -> Self {
        SlotMap {
            inner: SlotMapInner {
                slots: Vec::new(max_capacity),
                len: AtomicU32::new(0),
                global,
                _alignment1: CacheAligned,
                free_list: AtomicU32::new(NIL),
                _alignment2: CacheAligned,
                free_list_queue: [
                    AtomicU64::new(u64::from(NIL)),
                    AtomicU64::new(u64::from(NIL) | 2 << 32),
                ],
            },
            marker: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.inner.slots.capacity()
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> u32 {
        self.inner.len.load(Relaxed)
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    #[must_use]
    pub fn global(&self) -> &epoch::GlobalHandle {
        &self.inner.global
    }
}

impl<K: Key, V> SlotMap<K, V> {
    /// # Panics
    ///
    /// Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    #[inline]
    pub fn insert<'a>(&'a self, value: V, guard: &'a epoch::Guard<'a>) -> K {
        self.insert_with_tag(value, 0, guard)
    }

    /// # Panics
    ///
    /// - Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    /// - Panics if `tag` has more than the low 8 bits set.
    #[inline]
    pub fn insert_with_tag<'a>(&'a self, value: V, tag: u32, guard: &'a epoch::Guard<'a>) -> K {
        K::from_id(self.inner.insert_with_tag(value, tag, guard))
    }

    #[inline]
    pub fn insert_mut(&mut self, value: V) -> K {
        self.insert_with_tag_mut(value, 0)
    }

    /// # Panics
    ///
    /// Panics if `tag` has more than the low 8 bits set.
    #[inline]
    pub fn insert_with_tag_mut(&mut self, value: V, tag: u32) -> K {
        K::from_id(self.inner.insert_with_tag_mut(value, tag))
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    #[inline]
    pub fn remove<'a>(&'a self, key: K, guard: &'a epoch::Guard<'a>) -> Option<&'a V> {
        self.inner.remove(key.as_id(), guard)
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    #[inline]
    pub fn try_collect(&self, guard: &epoch::Guard<'_>) {
        self.inner.try_collect(guard);
    }

    #[inline]
    pub fn remove_mut(&mut self, key: K) -> Option<V> {
        self.inner.remove_mut(key.as_id())
    }

    #[cfg(test)]
    fn remove_index<'a>(&'a self, index: u32, guard: &'a epoch::Guard<'a>) -> Option<&'a V> {
        self.inner.remove_index(index, guard)
    }

    #[inline]
    pub fn invalidate<'a>(&'a self, key: K, guard: &'a epoch::Guard<'a>) -> Option<&'a V> {
        self.inner.invalidate(key.as_id(), guard)
    }

    /// # Safety
    ///
    /// * The slot must be currently [invalidated].
    /// * At least two [epochs] must have passed since the slot was invalidated.
    /// * The slot must not have been removed already.
    ///
    /// [invalidated]: Self::invalidate
    /// [epochs]: epoch::GlobalHandle::epoch
    #[inline]
    pub unsafe fn remove_unchecked(&self, key: K) -> V {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.remove_unchecked(key.as_id()) }
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    #[inline(always)]
    #[must_use]
    pub fn get<'a>(&'a self, key: K, guard: &'a epoch::Guard<'a>) -> Option<&'a V> {
        self.inner.get(key.as_id(), guard)
    }

    #[inline(always)]
    #[must_use]
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.inner.get_mut(key.as_id())
    }

    #[inline]
    #[must_use]
    pub fn get_many_mut<const N: usize>(&mut self, keys: [K; N]) -> Option<[&mut V; N]> {
        self.inner.get_many_mut(keys.map(Key::as_id))
    }

    #[cfg(test)]
    fn index<'a>(&'a self, index: u32, guard: &'a epoch::Guard<'a>) -> Option<&'a V> {
        self.inner.index(index, guard)
    }

    /// # Safety
    ///
    /// The slot must be currently occupied.
    ///
    /// # Panics
    ///
    /// Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    #[inline(always)]
    #[must_use]
    pub unsafe fn get_unchecked<'a>(&'a self, key: K, guard: &'a epoch::Guard<'a>) -> &'a V {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.get_unchecked(key.as_id(), guard) }
    }

    /// # Safety
    ///
    /// The slot must be currently occupied.
    #[inline(always)]
    #[must_use]
    pub unsafe fn get_unchecked_mut(&mut self, key: K) -> &mut V {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.get_unchecked_mut(key.as_id()) }
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    #[inline]
    #[must_use]
    pub fn iter<'a>(&'a self, guard: &'a epoch::Guard<'a>) -> Iter<'a, K, V> {
        self.inner.check_guard(guard);

        Iter {
            slots: self.inner.slots.iter().enumerate(),
            marker: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            slots: self.inner.slots.iter_mut().enumerate(),
            marker: PhantomData,
        }
    }
}

impl<V> SlotMapInner<V> {
    fn insert_with_tag<'a>(&'a self, value: V, tag: u32, guard: &'a epoch::Guard<'a>) -> SlotId {
        self.check_guard(guard);
        assert_eq!(tag & !TAG_MASK, 0);

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
                    unsafe { slot.value.get().cast::<V>().write(value) };

                    let new_generation = slot
                        .generation
                        .fetch_add(OCCUPIED_BIT | tag, Release)
                        .wrapping_add(OCCUPIED_BIT | tag);

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

        let index = self.slots.push_with_tag(value, tag);

        self.len.fetch_add(1, Relaxed);

        // SAFETY: The `OCCUPIED_BIT` is set.
        unsafe { SlotId::new_unchecked(index, OCCUPIED_BIT | tag) }
    }

    fn insert_with_tag_mut(&mut self, value: V, tag: u32) -> SlotId {
        assert_eq!(tag & !TAG_MASK, 0);

        let free_list_head = *self.free_list.get_mut();

        if free_list_head != NIL {
            // SAFETY: We always push indices of existing slots into the free-lists and the slots
            // vector never shrinks, therefore the index must have staid in bounds.
            let slot = unsafe { self.slots.get_unchecked_mut(free_list_head as usize) };

            let new_generation = slot.generation.get_mut().wrapping_add(OCCUPIED_BIT | tag);

            *slot.generation.get_mut() = new_generation;

            *self.free_list.get_mut() = *slot.next_free.get_mut();

            *slot.value.get_mut() = MaybeUninit::new(value);

            *self.len.get_mut() += 1;

            // SAFETY: `SlotMap::remove[_mut]` guarantees that a freed slot has its generation's
            // `OCCUPIED_BIT` unset, and since we incremented the generation, the bit must have been
            // flipped again.
            return unsafe { SlotId::new_unchecked(free_list_head, new_generation) };
        }

        let index = self.slots.push_with_tag_mut(value, tag);

        *self.len.get_mut() += 1;

        // SAFETY: The `OCCUPIED_BIT` is set.
        unsafe { SlotId::new_unchecked(index, OCCUPIED_BIT | tag) }
    }

    fn remove<'a>(&'a self, id: SlotId, guard: &'a epoch::Guard<'a>) -> Option<&'a V> {
        let slot = self.invalidate(id, guard)?;

        // SAFETY: We unset the slot's `OCCUPIED_BIT` such that no other threads can be attempting
        // to push it into the free-lists.
        unsafe { self.defer_destroy(id.index) };

        Some(slot)
    }

    unsafe fn defer_destroy(&self, index: u32) {
        // SAFETY: The caller must ensure that `index` is in bounds.
        let slot = unsafe { self.slots.get_unchecked(index as usize) };

        atomic::fence(SeqCst);

        let epoch = self.global.epoch();
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
                        break;
                    }
                    Err(new_state) => {
                        queued_state = new_state;
                        backoff.spin();
                    }
                }
            } else {
                let global_epoch_is_behind_queue = epoch_interval & (1 << 31) != 0;

                debug_assert!(!global_epoch_is_behind_queue);
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
                        unsafe { self.collect_unchecked(queued_head) };

                        break;
                    }
                    Err(new_state) => {
                        queued_state = new_state;
                        backoff.spin();
                    }
                }
            }
        }
    }

    fn try_collect(&self, guard: &epoch::Guard<'_>) {
        self.check_guard(guard);

        let epoch = self.global.epoch();
        let queued_list = &self.free_list_queue[((epoch >> 1) & 1) as usize];
        let mut queued_state = queued_list.load(Acquire);
        let mut backoff = Backoff::new();

        loop {
            let queued_head = (queued_state & 0xFFFF_FFFF) as u32;
            let queued_epoch = (queued_state >> 32) as u32;
            let epoch_interval = epoch.wrapping_sub(queued_epoch);

            if epoch_interval == 0 {
                break;
            }

            let global_epoch_is_behind_queue = epoch_interval & (1 << 31) != 0;

            debug_assert!(!global_epoch_is_behind_queue);

            let new_state = u64::from(NIL) | u64::from(queued_epoch) << 32;

            match queued_list.compare_exchange_weak(queued_state, new_state, Relaxed, Acquire) {
                Ok(_) => {
                    // SAFETY: Having ended up here, the global epoch must have been advanced at
                    // least 2 steps from the last push into the queued list and we removed the
                    // list from the queue, which means that no other threads can be accessing any
                    // of the slots in the list.
                    unsafe { self.collect_unchecked(queued_head) };

                    break;
                }
                Err(new_state) => {
                    queued_state = new_state;
                    backoff.spin();
                }
            }
        }
    }

    #[cold]
    unsafe fn collect_unchecked(&self, queued_head: u32) {
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

            let ptr = queued_tail_slot.value.get().cast::<V>();

            // SAFETY: The caller must ensure that we have exclusive access to this list.
            unsafe { ptr.drop_in_place() };

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

    fn remove_mut(&mut self, id: SlotId) -> Option<V> {
        let slot = self.slots.get_mut(id.index as usize)?;
        let generation = *slot.generation.get_mut();

        if generation == id.generation() {
            *slot.generation.get_mut() = (id.generation() & !TAG_MASK).wrapping_add(OCCUPIED_BIT);

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
            Some(unsafe { slot.value.get().cast::<V>().read() })
        } else {
            None
        }
    }

    #[cfg(test)]
    fn remove_index<'a>(&'a self, index: u32, guard: &'a epoch::Guard<'a>) -> Option<&'a V> {
        self.check_guard(guard);

        let slot = self.slots.get(index as usize)?;
        let mut generation = slot.generation.load(Relaxed);
        let mut backoff = Backoff::new();

        loop {
            if generation & OCCUPIED_BIT == 0 {
                return None;
            }

            let new_generation = (generation & !TAG_MASK).wrapping_add(OCCUPIED_BIT);

            match slot.generation.compare_exchange_weak(
                generation,
                new_generation,
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

        // SAFETY: We unset the slot's `OCCUPIED_BIT` such that no other threads can be attempting
        // to push it into the free-lists.
        unsafe { self.defer_destroy(index) };

        // SAFETY:
        // * The `Acquire` ordering when loading the slot's generation synchronizes with the
        //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value is
        //   visible here.
        // * The `compare_exchange_weak` above succeeded, which means that the previous generation
        //   of the slot must have had its `OCCUPIED_BIT` set, which makes reading the value safe as
        //   the only way the occupied bit can be set is in `SlotMap::insert[_mut]` after
        //   initialization of the slot.
        Some(unsafe { slot.value_unchecked() })
    }

    fn invalidate<'a>(&'a self, id: SlotId, guard: &'a epoch::Guard<'a>) -> Option<&'a V> {
        self.check_guard(guard);

        let slot = self.slots.get(id.index as usize)?;
        let new_generation = (id.generation() & !TAG_MASK).wrapping_add(OCCUPIED_BIT);

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
        Some(unsafe { slot.value_unchecked() })
    }

    unsafe fn remove_unchecked(&self, id: SlotId) -> V {
        // SAFETY: The caller must ensure that the index is in bounds.
        let slot = unsafe { self.slots.get_unchecked(id.index as usize) };

        // SAFETY: The caller must ensure that the slot is initialized and that no other threads can
        // be accessing the slot.
        let value = unsafe { slot.value.get().cast::<V>().read() };

        let mut free_list_head = self.free_list.load(Acquire);
        let mut backoff = Backoff::new();

        loop {
            slot.next_free.store(free_list_head, Relaxed);

            match self
                .free_list
                .compare_exchange_weak(free_list_head, id.index, Release, Acquire)
            {
                Ok(_) => break value,
                Err(new_head) => {
                    free_list_head = new_head;
                    backoff.spin();
                }
            }
        }
    }

    #[inline(always)]
    fn get<'a>(&'a self, id: SlotId, guard: &'a epoch::Guard<'a>) -> Option<&'a V> {
        self.check_guard(guard);

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
            Some(unsafe { slot.value_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    fn get_mut(&mut self, id: SlotId) -> Option<&mut V> {
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
    fn get_many_mut<const N: usize>(&mut self, ids: [SlotId; N]) -> Option<[&mut V; N]> {
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

        let len = self.slots.capacity_mut();

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
    ) -> Option<[&mut V; N]> {
        let slots_ptr = self.slots.as_mut_ptr();
        let mut refs = MaybeUninit::<[&mut V; N]>::uninit();
        let refs_ptr = refs.as_mut_ptr().cast::<&mut V>();

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

    #[cfg(test)]
    fn index<'a>(&'a self, index: u32, guard: &'a epoch::Guard<'a>) -> Option<&'a V> {
        self.check_guard(guard);

        let slot = self.slots.get(index as usize)?;
        let generation = slot.generation.load(Acquire);

        if generation & OCCUPIED_BIT != 0 {
            // SAFETY:
            // * The `Acquire` ordering when loading the slot's generation synchronizes with the
            //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value
            //   is visible here.
            // * We checked that the slot is occupied, which means that it must have been
            //   initialized in `SlotMap::insert[_mut]`.
            Some(unsafe { slot.value_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked<'a>(&'a self, id: SlotId, guard: &'a epoch::Guard<'a>) -> &'a V {
        self.check_guard(guard);

        // SAFETY: The caller must ensure that the index is in bounds.
        let slot = unsafe { self.slots.get_unchecked(id.index as usize) };

        let _generation = slot.generation.load(Acquire);

        // SAFETY:
        // * The `Acquire` ordering when loading the slot's generation synchronizes with the
        //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value is
        //   visible here.
        // * The caller must ensure that the slot is initialized.
        unsafe { slot.value_unchecked() }
    }

    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, id: SlotId) -> &mut V {
        // SAFETY: The caller must ensure that the index is in bounds.
        let slot = unsafe { self.slots.get_unchecked_mut(id.index as usize) };

        // SAFETY:
        // * The mutable reference makes sure that access to the slot is synchronized.
        // * The caller must ensure that the slot is initialized.
        unsafe { slot.value_unchecked_mut() }
    }

    #[inline(always)]
    fn check_guard(&self, guard: &epoch::Guard<'_>) {
        #[inline(never)]
        fn global_mismatch() -> ! {
            panic!("`guard.global()` is `Some` but does not equal `self.global()`");
        }

        if let Some(global) = guard.global() {
            if global != &self.global {
                global_mismatch();
            }
        }
    }
}

impl<K, V: fmt::Debug> fmt::Debug for SlotMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Slots;

        impl fmt::Debug for Slots {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.pad("[..]")
            }
        }

        struct List<'a, V>(&'a SlotMapInner<V>, u32);

        impl<V> fmt::Debug for List<'_, V> {
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

        struct QueuedList<'a, V>(&'a SlotMapInner<V>, u64);

        impl<V> fmt::Debug for QueuedList<'_, V> {
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

        fn inner<V>(map: &SlotMapInner<V>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut debug = f.debug_struct("SlotMap");
            debug
                .field("slots", &Slots)
                .field("len", &map.len)
                .field("global", &map.global)
                .field("free_list", &List(map, map.free_list.load(Acquire)))
                .field(
                    "free_list_queue",
                    &[
                        QueuedList(map, map.free_list_queue[0].load(Acquire)),
                        QueuedList(map, map.free_list_queue[1].load(Acquire)),
                    ],
                );

            debug.finish()
        }

        inner(&self.inner, f)
    }
}

impl<K, V> Drop for SlotMap<K, V> {
    fn drop(&mut self) {
        fn inner<V>(map: &mut SlotMapInner<V>) {
            if !core::mem::needs_drop::<V>() {
                return;
            }

            for list in &mut map.free_list_queue {
                let mut head = (*list.get_mut() & 0xFFFF_FFFF) as u32;

                while head != NIL {
                    // SAFETY: We always push indices of existing slots into the free-lists and the
                    // slots vector never shrinks, therefore the index must have staid in bounds.
                    let slot = unsafe { map.slots.get_unchecked_mut(head as usize) };

                    let ptr = slot.value.get_mut().as_mut_ptr();

                    // SAFETY: We can be certain that this slot has been initialized, since the only
                    // way in which it could have been queued for freeing is in `SlotMap::remove` if
                    // the slot was inserted before.
                    unsafe { ptr.drop_in_place() };

                    head = *slot.next_free.get_mut();
                }
            }

            for slot in map.slots.as_mut_slice() {
                if *slot.generation.get_mut() & OCCUPIED_BIT != 0 {
                    let ptr = slot.value.get_mut().as_mut_ptr();

                    // SAFETY:
                    // * The mutable reference makes sure that access to the slot is synchronized.
                    // * We checked that the slot is occupied, which means that it must have been
                    //   initialized in `SlotMap::insert[_mut]`.
                    unsafe { ptr.drop_in_place() };
                }
            }
        }

        inner(&mut self.inner);
    }
}

impl<'a, K: Key, V> IntoIterator for &'a mut SlotMap<K, V> {
    type Item = (K, &'a mut V);

    type IntoIter = IterMut<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

pub trait Key: Sized {
    fn from_id(id: SlotId) -> Self;

    #[allow(clippy::wrong_self_convention)]
    fn as_id(self) -> SlotId;
}

impl Key for SlotId {
    #[inline(always)]
    fn from_id(id: SlotId) -> Self {
        id
    }

    #[inline(always)]
    fn as_id(self) -> SlotId {
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId {
    index: u32,
    generation: NonZeroU32,
}

impl SlotId {
    pub const INVALID: Self = SlotId {
        index: u32::MAX,
        generation: NonZeroU32::MAX,
    };

    pub const OCCUPIED_BIT: u32 = OCCUPIED_BIT;

    #[inline(always)]
    #[must_use]
    pub const fn new(index: u32, generation: u32) -> Self {
        assert!(generation & OCCUPIED_BIT != 0);

        // SAFETY: We checked that the `OCCUPIED_BIT` is set.
        unsafe { SlotId::new_unchecked(index, generation) }
    }

    /// # Safety
    ///
    /// `generation` must have its [`OCCUPIED_BIT`] set.
    ///
    /// [`OCCUPIED_BIT`]: Self::OCCUPIED_BIT
    #[inline(always)]
    #[must_use]
    pub const unsafe fn new_unchecked(index: u32, generation: u32) -> Self {
        debug_assert!(generation & OCCUPIED_BIT != 0);

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

    #[inline(always)]
    #[must_use]
    pub const fn tag(self) -> u32 {
        self.generation.get() & TAG_MASK
    }
}

#[macro_export]
macro_rules! declare_key {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident $(;)?
    ) => {
        $(#[$meta])*
        #[repr(transparent)]
        $vis struct $name($crate::SlotId);

        impl $crate::Key for $name {
            #[inline(always)]
            fn from_id(id: $crate::SlotId) -> Self {
                $name(id)
            }

            #[inline(always)]
            fn as_id(self) -> $crate::SlotId {
                self.0
            }
        }
    };
}

pub struct Iter<'a, K, V> {
    slots: iter::Enumerate<slice::Iter<'a, Slot<V>>>,
    marker: PhantomData<fn(K) -> K>,
}

impl<K, V: fmt::Debug> fmt::Debug for Iter<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Iter").finish_non_exhaustive()
    }
}

impl<'a, K: Key, V> Iterator for Iter<'a, K, V> {
    type Item = (K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.slots.next()?;
            let generation = slot.generation.load(Acquire);

            if generation & OCCUPIED_BIT != 0 {
                // Our capacity can never exceed `u32::MAX`.
                #[allow(clippy::cast_possible_truncation)]
                let index = index as u32;

                // SAFETY: We checked that the occupied bit is set.
                let id = unsafe { SlotId::new_unchecked(index, generation) };

                // SAFETY:
                // * The `Acquire` ordering when loading the slot's generation synchronizes with the
                //   `Release` ordering in `SlotMap::insert`, making sure that the newly written
                //   value is visible here.
                // * We checked that the slot is occupied, which means that it must have been
                //   initialized in `SlotMap::insert[_mut]`.
                let r = unsafe { slot.value_unchecked() };

                break Some((K::from_id(id), r));
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.slots.len()))
    }
}

impl<K: Key, V> DoubleEndedIterator for Iter<'_, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.slots.next_back()?;
            let generation = slot.generation.load(Acquire);

            if generation & OCCUPIED_BIT != 0 {
                // Our capacity can never exceed `u32::MAX`.
                #[allow(clippy::cast_possible_truncation)]
                let index = index as u32;

                // SAFETY: We checked that the occupied bit is set.
                let id = unsafe { SlotId::new_unchecked(index, generation) };

                // SAFETY:
                // * The `Acquire` ordering when loading the slot's generation synchronizes with the
                //   `Release` ordering in `SlotMap::insert`, making sure that the newly written
                //   value is visible here.
                // * We checked that the slot is occupied, which means that it must have been
                //   initialized in `SlotMap::insert[_mut]`.
                let r = unsafe { slot.value_unchecked() };

                break Some((K::from_id(id), r));
            }
        }
    }
}

impl<K: Key, V> FusedIterator for Iter<'_, K, V> {}

pub struct IterMut<'a, K, V> {
    slots: iter::Enumerate<slice::IterMut<'a, Slot<V>>>,
    marker: PhantomData<fn(K) -> K>,
}

impl<K, V: fmt::Debug> fmt::Debug for IterMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IterMut").finish_non_exhaustive()
    }
}

impl<'a, K: Key, V> Iterator for IterMut<'a, K, V> {
    type Item = (K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.slots.next()?;
            let generation = *slot.generation.get_mut();

            if generation & OCCUPIED_BIT != 0 {
                // Our capacity can never exceed `u32::MAX`.
                #[allow(clippy::cast_possible_truncation)]
                let index = index as u32;

                // SAFETY: We checked that the `OCCUPIED_BIT` is set.
                let id = unsafe { SlotId::new_unchecked(index, generation) };

                // SAFETY: We checked that the slot is occupied, which means that it must have been
                // initialized in `SlotMap::insert[_mut]`.
                let r = unsafe { slot.value_unchecked_mut() };

                break Some((K::from_id(id), r));
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.slots.len()))
    }
}

impl<K: Key, V> DoubleEndedIterator for IterMut<'_, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.slots.next_back()?;
            let generation = *slot.generation.get_mut();

            if generation & OCCUPIED_BIT != 0 {
                // Our capacity can never exceed `u32::MAX`.
                #[allow(clippy::cast_possible_truncation)]
                let index = index as u32;

                // SAFETY: We checked that the `OCCUPIED_BIT` is set.
                let id = unsafe { SlotId::new_unchecked(index, generation) };

                // SAFETY: We checked that the slot is occupied, which means that it must have been
                // initialized in `SlotMap::insert[_mut]`.
                let r = unsafe { slot.value_unchecked_mut() };

                break Some((K::from_id(id), r));
            }
        }
    }
}

impl<K: Key, V> FusedIterator for IterMut<'_, K, V> {}

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
        let guard = &map.global().register_local().into_inner().pin();

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
        let guard = &map.global().register_local().into_inner().pin();

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
        let guard = &map.global().register_local().into_inner().pin();

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
        let guard = &map.global().register_local().into_inner().pin();

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
        let guard = &map.global().register_local().into_inner().pin();

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
        let guard = &map.global().register_local().into_inner().pin();

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
    fn tagged() {
        let map = SlotMap::new(1);
        let guard = &map.global().register_local().into_inner().pin();

        let x = map.insert_with_tag(42, 1, guard);
        assert_eq!(x.generation() & TAG_MASK, 1);
        assert_eq!(map.get(x, guard).as_deref(), Some(&42));
    }

    #[test]
    fn tagged_mut() {
        let mut map = SlotMap::new(1);

        let x = map.insert_with_tag_mut(42, 1);
        assert_eq!(x.generation() & TAG_MASK, 1);
        assert_eq!(map.get_mut(x), Some(&mut 42));
    }

    // TODO: Testing concurrent generational collections is the most massive pain in the ass. We
    // aren't testing the actual implementations but rather ones that don't take the generation into
    // account because of that.

    const ITERATIONS: u32 = if cfg!(miri) { 1_000 } else { 1_000_000 };

    #[test]
    fn multi_threaded1() {
        const THREADS: u32 = 2;

        let map = SlotMap::new(ITERATIONS);

        thread::scope(|s| {
            let inserter = || {
                let local = map.global().register_local();

                for _ in 0..ITERATIONS / THREADS {
                    map.insert(0, &local.pin());
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
                    let _ = map.remove(SlotId::new(index, OCCUPIED_BIT), &local.pin());
                }
            };

            for _ in 0..THREADS {
                s.spawn(remover);
            }
        });

        assert_eq!(map.len(), 0);
    }

    #[test]
    fn multi_threaded2() {
        const CAPACITY: u32 = PINNINGS_BETWEEN_ADVANCE as u32 * 3;

        // TODO: Why does ThreadSanitizer need more than `CAPACITY` slots, but only in CI???
        let map = SlotMap::new(ITERATIONS / 2);

        thread::scope(|s| {
            let insert_remover = || {
                let local = map.global().register_local();

                for _ in 0..ITERATIONS / 6 {
                    let x = map.insert(0, &local.pin());
                    let y = map.insert(0, &local.pin());
                    map.remove(y, &local.pin());
                    let z = map.insert(0, &local.pin());
                    map.remove(x, &local.pin());
                    map.remove(z, &local.pin());
                }
            };
            let iterator = || {
                let local = map.global().register_local();

                for _ in 0..ITERATIONS / CAPACITY * 2 {
                    for index in 0..CAPACITY {
                        if let Some(value) = map.index(index, &local.pin()) {
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
                        map.insert(0, &local.pin());
                    } else {
                        thread::yield_now();
                    }
                }
            };
            let remover = || {
                let local = map.global().register_local();

                for _ in 0..ITERATIONS {
                    map.remove_index(0, &local.pin());
                }
            };
            let getter = || {
                let local = map.global().register_local();

                for _ in 0..ITERATIONS {
                    if let Some(value) = map.index(0, &local.pin()) {
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
