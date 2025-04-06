#![allow(unused_unsafe, clippy::inline_always)]
#![warn(rust_2018_idioms, missing_debug_implementations)]
#![forbid(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]

extern crate alloc;

use core::{
    cmp, fmt,
    hash::{Hash, Hasher},
    hint,
    iter::{self, FusedIterator},
    marker::PhantomData,
    mem::{self, MaybeUninit},
    num::NonZeroU32,
    slice,
    sync::atomic::{
        AtomicU32,
        Ordering::{Acquire, Relaxed, Release},
    },
};
use slot::{Slot, Vec};

pub mod hyaline;
mod slot;

/// The slot index used to signify the lack thereof.
const NIL: u32 = u32::MAX;

/// The number of low bits that can be used for user-tagged generations.
const TAG_BITS: u32 = 8;

/// The mask for user-tagged generations.
const TAG_MASK: u32 = (1 << TAG_BITS) - 1;

/// The number of bits following the user-tag bits that are used for the state of a slot.
const STATE_BITS: u32 = 2;

/// The mask for the state of a slot.
const STATE_MASK: u32 = 0b11 << TAG_BITS;

/// The state tag used to signify that the slot is vacant.
const VACANT_TAG: u32 = 0b00 << TAG_BITS;

/// The state tag used to signify that the slot is occupied.
const OCCUPIED_TAG: u32 = 0b01 << TAG_BITS;

/// The state tag used to signify that the slot has been invalidated.
const INVALIDATED_TAG: u32 = 0b10 << TAG_BITS;

/// The state tag used to signify that the invalidated slot has been reclaimed.
const RECLAIMED_TAG: u32 = 0b11 << TAG_BITS;

/// The mask for the generation.
const GENERATION_MASK: u32 = u32::MAX << (TAG_BITS + STATE_BITS);

/// A single generation.
const ONE_GENERATION: u32 = 1 << (TAG_BITS + STATE_BITS);

pub struct SlotMap<K, V> {
    inner: SlotMapInner<V>,
    marker: PhantomData<fn(K) -> K>,
}

struct SlotMapInner<V> {
    /// ```compile_fail,E0597
    /// let map = concurrent_slotmap::SlotMap::<_, &'static str>::new(1);
    /// let guard = &map.pin();
    /// let id = {
    ///     let s = "oh no".to_owned();
    ///     map.insert(&s, guard)
    /// };
    /// dbg!(map.get(id, guard));
    /// ```
    slots: Vec<V>,
    collector: hyaline::CollectorHandle,
    hot_data: *mut HotData,
}

#[repr(align(128))]
struct HotData {
    /// The list of slots which have already been dropped and are ready to be claimed by insert
    /// operations.
    free_list: AtomicU32,
    /// The number of occupied slots.
    len: AtomicU32,
}

// SAFETY: `SlotMap` is an owned collection, which makes it safe to send to another thread as long
// as the value is safe to send to another thread. The key is a phantom parameter.
unsafe impl<K, V: Send> Send for SlotMap<K, V> {}

// SAFETY: `SlotMap` allows pushing through a shared reference, which allows a shared `SlotMap` to
// be used to send values to another thread. Additionally, `SlotMap` allows getting a reference to
// any value from any thread. Therefore, it is safe to share `SlotMap` between threads as long as
// the value is both sendable and shareable. The key is a phantom parameter.
unsafe impl<K, V: Send + Sync> Sync for SlotMap<K, V> {}

impl<V> SlotMap<SlotId, V> {
    #[must_use]
    pub fn new(max_capacity: u32) -> Self {
        Self::with_key(max_capacity)
    }
}

impl<K, V> SlotMap<K, V> {
    #[must_use]
    pub fn with_key(max_capacity: u32) -> Self {
        let slots = Vec::new(max_capacity);
        let hot_data = Box::into_raw(Box::new(HotData {
            free_list: AtomicU32::new(NIL),
            len: AtomicU32::new(0),
        }));
        // SAFETY: `slots` and `hot_data` outlive any calls into the collector.
        let collector = unsafe { hyaline::CollectorHandle::new(&slots, hot_data) };

        SlotMap {
            inner: SlotMapInner {
                slots,
                collector,
                hot_data,
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
        self.inner.len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K: Key, V> SlotMap<K, V> {
    #[inline]
    pub fn pin(&self) -> hyaline::Guard<'_> {
        self.inner.pin()
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    #[inline]
    pub fn insert<'a>(&'a self, value: V, guard: &'a hyaline::Guard<'a>) -> K {
        self.insert_with_tag(value, 0, guard)
    }

    /// # Panics
    ///
    /// - Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    /// - Panics if `tag` has more than the low 8 bits set.
    #[inline]
    pub fn insert_with_tag<'a>(&'a self, value: V, tag: u32, guard: &'a hyaline::Guard<'a>) -> K {
        K::from_id(self.inner.insert_with_tag_with(tag, guard, |_| value))
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    #[inline]
    pub fn insert_with<'a>(&'a self, guard: &'a hyaline::Guard<'a>, f: impl FnOnce(K) -> V) -> K {
        self.insert_with_tag_with(0, guard, f)
    }

    /// # Panics
    ///
    /// - Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    /// - Panics if `tag` has more than the low 8 bits set.
    #[inline]
    pub fn insert_with_tag_with<'a>(
        &'a self,
        tag: u32,
        guard: &'a hyaline::Guard<'a>,
        f: impl FnOnce(K) -> V,
    ) -> K {
        let f = |id| f(K::from_id(id));

        K::from_id(self.inner.insert_with_tag_with(tag, guard, f))
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
        K::from_id(self.inner.insert_with_tag_with_mut(tag, |_| value))
    }

    #[inline]
    pub fn insert_with_mut(&mut self, f: impl FnOnce(K) -> V) -> K {
        self.insert_with_tag_with_mut(0, f)
    }

    /// # Panics
    ///
    /// Panics if `tag` has more than the low 8 bits set.
    #[inline]
    pub fn insert_with_tag_with_mut(&mut self, tag: u32, f: impl FnOnce(K) -> V) -> K {
        let f = |id| f(K::from_id(id));

        K::from_id(self.inner.insert_with_tag_with_mut(tag, f))
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    #[inline]
    pub fn remove<'a>(&'a self, key: K, guard: &'a hyaline::Guard<'a>) -> Option<&'a V> {
        self.inner.remove(key.as_id(), guard)
    }

    #[inline]
    pub fn remove_mut(&mut self, key: K) -> Option<V> {
        self.inner.remove_mut(key.as_id())
    }

    #[cfg(test)]
    fn remove_index<'a>(&'a self, index: u32, guard: &'a hyaline::Guard<'a>) -> Option<&'a V> {
        self.inner.remove_index(index, guard)
    }

    #[inline]
    pub fn invalidate<'a>(&'a self, key: K, guard: &'a hyaline::Guard<'a>) -> Option<&'a V> {
        self.inner.invalidate(key.as_id(), guard)
    }

    #[inline]
    pub fn remove_invalidated(&self, key: K) -> Option<V> {
        self.inner.remove_invalidated(key.as_id())
    }

    /// # Panics
    ///
    /// Panics if `guard.global()` is `Some` and does not equal `self.global()`.
    #[inline(always)]
    #[must_use]
    pub fn get<'a>(&'a self, key: K, guard: &'a hyaline::Guard<'a>) -> Option<&'a V> {
        self.inner.get(key.as_id(), guard)
    }

    #[inline(always)]
    #[must_use]
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.inner.get_mut(key.as_id())
    }

    #[inline]
    #[must_use]
    pub fn get_disjoint_mut<const N: usize>(&mut self, keys: [K; N]) -> Option<[&mut V; N]> {
        self.inner.get_disjoint_mut(keys.map(Key::as_id))
    }

    #[cfg(test)]
    fn index<'a>(&'a self, index: u32, guard: &'a hyaline::Guard<'a>) -> Option<&'a V> {
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
    pub unsafe fn get_unchecked<'a>(&'a self, key: K, guard: &'a hyaline::Guard<'a>) -> &'a V {
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
    pub fn iter<'a>(&'a self, guard: &'a hyaline::Guard<'a>) -> Iter<'a, K, V> {
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
    fn pin(&self) -> hyaline::Guard<'_> {
        self.collector.pin()
    }

    fn insert_with_tag_with<'a>(
        &'a self,
        tag: u32,
        guard: &'a hyaline::Guard<'a>,
        f: impl FnOnce(SlotId) -> V,
    ) -> SlotId {
        self.check_guard(guard);
        assert_eq!(tag & !TAG_MASK, 0);

        let mut free_list_head = self.free_list().load(Acquire);
        let mut backoff = Backoff::new();

        loop {
            if free_list_head == NIL {
                break;
            }

            // SAFETY: We always push indices of existing slots into the free-lists and the slots
            // vector never shrinks, therefore the index must have staid in bounds.
            let slot = unsafe { self.slots.get_unchecked(free_list_head) };

            let next_free = slot.next_free.load(Relaxed);

            match self.free_list().compare_exchange_weak(
                free_list_head,
                next_free,
                Release,
                Acquire,
            ) {
                Ok(_) => {
                    let generation = slot.generation.load(Relaxed);

                    debug_assert!(generation & STATE_MASK == VACANT_TAG);

                    let new_generation = generation | OCCUPIED_TAG | tag;

                    // SAFETY: We always push slots with their state tag set to `VACANT_TAG` into
                    // the free-list and we replaced the tag with `OCCUPIED_TAG` above.
                    let id = unsafe { SlotId::new_unchecked(free_list_head, new_generation) };

                    // SAFETY: The free-list always contains slots that are no longer read by any
                    // threads and we have removed the slot from the free-list such that no other
                    // threads can be writing the same slot.
                    unsafe { slot.value.get().cast::<V>().write(f(id)) };

                    slot.generation.store(new_generation, Release);

                    self.hot_data().len.fetch_add(1, Relaxed);

                    return id;
                }
                Err(new_head) => {
                    free_list_head = new_head;
                    backoff.spin();
                }
            }
        }

        let id = self.slots.push_with_tag_with(tag, f);

        self.hot_data().len.fetch_add(1, Relaxed);

        id
    }

    fn insert_with_tag_with_mut(&mut self, tag: u32, f: impl FnOnce(SlotId) -> V) -> SlotId {
        assert_eq!(tag & !TAG_MASK, 0);

        // SAFETY: The pointer is valid and we have exclusive access to the collection.
        let hot_data = unsafe { &mut *self.hot_data };

        let free_list_head = *hot_data.free_list.get_mut();

        if free_list_head != NIL {
            // SAFETY: We always push indices of existing slots into the free-lists and the slots
            // vector never shrinks, therefore the index must have staid in bounds.
            let slot = unsafe { self.slots.get_unchecked_mut(free_list_head) };

            *hot_data.free_list.get_mut() = *slot.next_free.get_mut();

            let generation = *slot.generation.get_mut();

            debug_assert!(generation & STATE_MASK == VACANT_TAG);

            let new_generation = generation | OCCUPIED_TAG | tag;

            // SAFETY: We always push slots with their state tag set to `VACANT_TAG` into the
            // free-list and we replaced the tag with `OCCUPIED_TAG` above.
            let id = unsafe { SlotId::new_unchecked(free_list_head, new_generation) };

            *slot.value.get_mut() = MaybeUninit::new(f(id));

            *slot.generation.get_mut() = new_generation;

            *hot_data.len.get_mut() += 1;

            return id;
        }

        let id = self.slots.push_with_tag_with_mut(tag, f);

        *hot_data.len.get_mut() += 1;

        id
    }

    fn remove<'a>(&'a self, id: SlotId, guard: &'a hyaline::Guard<'a>) -> Option<&'a V> {
        self.check_guard(guard);

        let slot = self.slots.get(id.index)?;
        let new_generation = (id.generation() & GENERATION_MASK).wrapping_add(ONE_GENERATION);

        if slot
            .generation
            .compare_exchange(id.generation(), new_generation, Acquire, Relaxed)
            .is_err()
        {
            return None;
        }

        self.hot_data().len.fetch_sub(1, Relaxed);

        // SAFETY: We set the slot's state tag to `VACANT_TAG` such that no other threads can access
        // the slot going forward.
        unsafe { guard.defer_reclaim(id.index, &self.slots) };

        // SAFETY:
        // * The `Acquire` ordering when loading the slot's generation synchronizes with the
        //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value is
        //   visible here.
        // * The `compare_exchange` above succeeded, which means that the previous state tag of the
        //   slot must have been `OCCUPIED_TAG`, which means it must have been initialized in
        //   `SlotMap::insert[_mut]`.
        Some(unsafe { slot.value_unchecked() })
    }

    fn remove_mut(&mut self, id: SlotId) -> Option<V> {
        let slot = self.slots.get_mut(id.index)?;
        let generation = *slot.generation.get_mut();

        // SAFETY: The pointer is valid and we have exclusive access to the collection.
        let hot_data = unsafe { &mut *self.hot_data };

        if generation == id.generation() {
            let new_generation = (generation & GENERATION_MASK).wrapping_add(ONE_GENERATION);
            *slot.generation.get_mut() = new_generation;

            *slot.next_free.get_mut() = *hot_data.free_list.get_mut();
            *hot_data.free_list.get_mut() = id.index;

            *hot_data.len.get_mut() -= 1;

            // SAFETY:
            // * The mutable reference makes sure that access to the slot is synchronized.
            // * We checked that `id.generation` matches the slot's generation, which means that the
            //   previous state tag of the slot must have been `OCCUPIED_TAG`, which means it must
            //   have been initialized in `SlotMap::insert[_mut]`.
            // * We set the slot's state tag to `VACANT_TAG` such that future attempts to access the
            //   slot will fail.
            Some(unsafe { slot.value.get().cast::<V>().read() })
        } else {
            None
        }
    }

    #[cfg(test)]
    fn remove_index<'a>(&'a self, index: u32, guard: &'a hyaline::Guard<'a>) -> Option<&'a V> {
        self.check_guard(guard);

        let slot = self.slots.get(index)?;
        let mut generation = slot.generation.load(Relaxed);
        let mut backoff = Backoff::new();

        loop {
            if !is_occupied(generation) {
                return None;
            }

            let new_generation = (generation & GENERATION_MASK).wrapping_add(ONE_GENERATION);

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

        self.hot_data().len.fetch_sub(1, Relaxed);

        // SAFETY: We set the slot's state tag to `VACANT_TAG` such that no other threads can access
        // the slot going forward.
        unsafe { guard.defer_reclaim(index, &self.slots) };

        // SAFETY:
        // * The `Acquire` ordering when loading the slot's generation synchronizes with the
        //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value is
        //   visible here.
        // * The `compare_exchange_weak` above succeeded, which means that the previous state tag of
        //   the slot must have been `OCCUPIED_TAG`, which means it must have been initialized in
        //   `SlotMap::insert[_mut]`.
        Some(unsafe { slot.value_unchecked() })
    }

    fn invalidate<'a>(&'a self, id: SlotId, guard: &'a hyaline::Guard<'a>) -> Option<&'a V> {
        self.check_guard(guard);

        let slot = self.slots.get(id.index)?;
        let new_generation = (id.generation() & !STATE_MASK) | INVALIDATED_TAG;

        if slot
            .generation
            .compare_exchange(id.generation(), new_generation, Acquire, Relaxed)
            .is_err()
        {
            return None;
        }

        self.hot_data().len.fetch_sub(1, Relaxed);

        // SAFETY: We set the slot's state tag to `INVALIDATED_TAG` such that no other threads can
        // access the slot going forward.
        unsafe { guard.defer_reclaim_invalidated(id.index, &self.slots) };

        // SAFETY:
        // * The `Acquire` ordering when loading the slot's generation synchronizes with the
        //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value is
        //   visible here.
        // * The `compare_exchange` above succeeded, which means that the previous state tag of the
        //   slot must have been `OCCUPIED_TAG`, which means it must have been initialized in
        //   `SlotMap::insert[_mut]`.
        Some(unsafe { slot.value_unchecked() })
    }

    fn remove_invalidated(&self, id: SlotId) -> Option<V> {
        let slot = self.slots.get(id.index)?;

        let old_generation = (id.generation() & !STATE_MASK) | RECLAIMED_TAG;
        let new_generation = (id.generation() & GENERATION_MASK).wrapping_add(ONE_GENERATION);

        if slot
            .generation
            .compare_exchange(old_generation, new_generation, Acquire, Relaxed)
            .is_err()
        {
            return None;
        }

        // SAFETY:
        // * The `Acquire` ordering when loading the slot's generation synchronizes with the
        //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value is
        //   visible here.
        // * The `compare_exchange` above succeeded, which means that the previous state tag of the
        //   slot must have been `RECLAIMED_TAG`, which means it must have been reclaimed in
        //   `reclaim_invalidated`, which means it must have been invalidated in
        //   `SlotMap::invalidate`, which means it must have been initialized in
        //   `SlotMap::insert[_mut]`.
        // * We set the slot's state tag to `VACANT_TAG` such that future attempts to access the
        //   slot will fail.
        let value = unsafe { slot.value.get().cast::<V>().read() };

        let mut free_list_head = self.free_list().load(Acquire);
        let mut backoff = Backoff::new();

        loop {
            slot.next_free.store(free_list_head, Relaxed);

            match self
                .free_list()
                .compare_exchange_weak(free_list_head, id.index, Release, Acquire)
            {
                Ok(_) => break Some(value),
                Err(new_head) => {
                    free_list_head = new_head;
                    backoff.spin();
                }
            }
        }
    }

    #[inline(always)]
    fn get<'a>(&'a self, id: SlotId, guard: &'a hyaline::Guard<'a>) -> Option<&'a V> {
        self.check_guard(guard);

        let slot = self.slots.get(id.index)?;
        let generation = slot.generation.load(Acquire);

        if generation == id.generation() {
            // SAFETY:
            // * The `Acquire` ordering when loading the slot's generation synchronizes with the
            //   `Release` ordering in `SlotMap::insert`, making sure that the newly written value
            //   is visible here.
            // * We checked that `id.generation` matches the slot's generation, which means that the
            //   state tag of the slot must have been `OCCUPIED_TAG`, which means it must have been
            //   initialized in `SlotMap::insert[_mut]`.
            Some(unsafe { slot.value_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    fn get_mut(&mut self, id: SlotId) -> Option<&mut V> {
        let slot = self.slots.get_mut(id.index)?;
        let generation = *slot.generation.get_mut();

        if generation == id.generation() {
            // SAFETY:
            // * The mutable reference makes sure that access to the slot is synchronized.
            // * We checked that `id.generation` matches the slot's generation, which means that the
            //   state tag of the slot must have been `OCCUPIED_TAG`, which means it must have been
            //   initialized in `SlotMap::insert[_mut]`.
            Some(unsafe { slot.value_unchecked_mut() })
        } else {
            None
        }
    }

    #[inline]
    fn get_disjoint_mut<const N: usize>(&mut self, ids: [SlotId; N]) -> Option<[&mut V; N]> {
        fn get_disjoint_check_valid<const N: usize>(ids: &[SlotId; N], len: u32) -> bool {
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

        if get_disjoint_check_valid(&ids, len) {
            // SAFETY: We checked that all indices are disjunct and in bounds of the slots vector.
            unsafe { self.get_disjoint_unchecked_mut(ids) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_disjoint_unchecked_mut<const N: usize>(
        &mut self,
        ids: [SlotId; N],
    ) -> Option<[&mut V; N]> {
        let mut refs = MaybeUninit::<[&mut V; N]>::uninit();
        let refs_ptr = refs.as_mut_ptr().cast::<&mut V>();

        for i in 0..N {
            // SAFETY: `i` is in bounds of the array.
            let id = unsafe { ids.get_unchecked(i) };

            // SAFETY:
            // * The caller must ensure that `ids` contains only IDs whose indices are in bounds of
            //   the slots vector.
            // * The caller must ensure that `ids` contains only IDs with disjunct indices.
            let slot = unsafe { self.slots.get_unchecked_mut(id.index()) };

            // SAFETY: We unbind the lifetime to convince the borrow checker that we don't
            // repeatedly borrow from `self.slots`. We don't for the same reasons as above.
            // `Vec::get_unchecked_mut` also only borrows the one element and not the whole `Vec`.
            let slot = unsafe { mem::transmute::<&mut Slot<V>, &mut Slot<V>>(slot) };

            let generation = *slot.generation.get_mut();

            if generation != id.generation() {
                return None;
            }

            // SAFETY:
            // * The mutable reference makes sure that access to the slot is synchronized.
            // * We checked that `id.generation` matches the slot's generation, which means that the
            //   state tag of the slot must have been `OCCUPIED_TAG`, which means it must have been
            //   initialized in `SlotMap::insert[_mut]`.
            let value = unsafe { slot.value_unchecked_mut() };

            // SAFETY: `i` is in bounds of the array.
            unsafe { *refs_ptr.add(i) = value };
        }

        // SAFETY: We initialized all the elements.
        Some(unsafe { refs.assume_init() })
    }

    #[cfg(test)]
    fn index<'a>(&'a self, index: u32, guard: &'a hyaline::Guard<'a>) -> Option<&'a V> {
        self.check_guard(guard);

        let slot = self.slots.get(index)?;
        let generation = slot.generation.load(Acquire);

        if is_occupied(generation) {
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
    unsafe fn get_unchecked<'a>(&'a self, id: SlotId, guard: &'a hyaline::Guard<'a>) -> &'a V {
        self.check_guard(guard);

        // SAFETY: The caller must ensure that the index is in bounds.
        let slot = unsafe { self.slots.get_unchecked(id.index) };

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
        let slot = unsafe { self.slots.get_unchecked_mut(id.index) };

        // SAFETY:
        // * The mutable reference makes sure that access to the slot is synchronized.
        // * The caller must ensure that the slot is initialized.
        unsafe { slot.value_unchecked_mut() }
    }

    #[inline]
    fn collector(&self) -> &hyaline::Collector {
        self.collector.collector()
    }

    fn free_list(&self) -> &AtomicU32 {
        &self.hot_data().free_list
    }

    fn len(&self) -> u32 {
        self.hot_data().len.load(Relaxed)
    }

    fn hot_data(&self) -> &HotData {
        // SAFETY: The pointer is valid.
        unsafe { &*self.hot_data }
    }

    #[inline(always)]
    fn check_guard(&self, guard: &hyaline::Guard<'_>) {
        #[inline(never)]
        fn collector_mismatch() -> ! {
            panic!("the given guard does not belong to this collection");
        }

        if guard.collector() != self.collector() {
            collector_mismatch();
        }
    }
}

unsafe fn reclaim<V>(head: u32, slots: *const Slot<V>, hot_data: &HotData) {
    let mut tail = head;
    let mut tail_slot;

    loop {
        // SAFETY: The caller must ensure that `head` denotes a valid list of slots.
        tail_slot = unsafe { &*slots.add(tail as usize) };

        let ptr = tail_slot.value.get().cast::<V>();

        // SAFETY: The caller must ensure that we have exclusive access to the slots in the list.
        unsafe { ptr.drop_in_place() };

        let next_free = tail_slot.next_free.load(Relaxed);

        if next_free == NIL {
            break;
        }

        tail = next_free;
    }

    let mut free_list_head = hot_data.free_list.load(Acquire);
    let mut backoff = Backoff::new();

    loop {
        tail_slot.next_free.store(free_list_head, Relaxed);

        match hot_data
            .free_list
            .compare_exchange_weak(free_list_head, head, Release, Acquire)
        {
            Ok(_) => break,
            Err(new_head) => {
                free_list_head = new_head;
                backoff.spin();
            }
        }
    }
}

unsafe fn reclaim_invalidated<V>(mut head: u32, slots: *const Slot<V>) {
    loop {
        // SAFETY: The caller must ensure that `head` denotes a valid list of slots.
        let slot = unsafe { &*slots.add(head as usize) };

        let generation = slot.generation.load(Relaxed);

        debug_assert!(generation & STATE_MASK == INVALIDATED_TAG);

        let new_generation = (generation & !STATE_MASK) | RECLAIMED_TAG;

        let res = slot
            .generation
            .compare_exchange(generation, new_generation, Release, Relaxed);

        debug_assert!(res.is_ok());

        let next_free = slot.next_free.load(Relaxed);

        if next_free == NIL {
            break;
        }

        head = next_free;
    }
}

impl<K: fmt::Debug + Key, V: fmt::Debug> fmt::Debug for SlotMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Slots<'a, K, V>(&'a SlotMap<K, V>);

        impl<K: fmt::Debug + Key, V: fmt::Debug> fmt::Debug for Slots<'_, K, V> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let guard = &self.0.pin();
                let mut debug = f.debug_map();

                for (k, v) in self.0.iter(guard) {
                    debug.entry(&k, v);
                }

                debug.finish()
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
                    let slot = unsafe { self.0.slots.get_unchecked(head) };

                    head = slot.next_free.load(Acquire);
                }

                debug.finish()
            }
        }

        fn inner<V>(map: &SlotMapInner<V>, mut debug: fmt::DebugStruct<'_, '_>) -> fmt::Result {
            debug
                .field("len", &map.len())
                .field("collector", &map.collector)
                .field("free_list", &List(map, map.free_list().load(Acquire)));

            debug.finish()
        }

        let mut debug = f.debug_struct("SlotMap");
        debug.field("slots", &Slots(self));

        inner(&self.inner, debug)
    }
}

impl<V> Drop for SlotMapInner<V> {
    fn drop(&mut self) {
        for slot in self.slots.iter_mut() {
            if *slot.generation.get_mut() & STATE_MASK != VACANT_TAG {
                let ptr = slot.value.get_mut().as_mut_ptr();

                // SAFETY:
                // * The mutable reference makes sure that access to the slot is synchronized.
                // * We checked that the slot is not vacant, which means that it must have been
                //   initialized in `SlotMap::insert[_mut]`.
                unsafe { ptr.drop_in_place() };
            }
        }

        // SAFETY: We allocated this pointer using `Box::new` and there can be no remaining
        // references to the allocation since we are being dropped.
        let _ = unsafe { Box::from_raw(self.hot_data) };
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

#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub struct SlotId {
    #[cfg(target_endian = "little")]
    index: u32,
    generation: NonZeroU32,
    #[cfg(target_endian = "big")]
    index: u32,
}

impl Default for SlotId {
    #[inline]
    fn default() -> Self {
        Self::INVALID
    }
}

impl SlotId {
    pub const INVALID: Self = SlotId {
        index: u32::MAX,
        generation: NonZeroU32::MAX,
    };

    pub const TAG_BITS: u32 = TAG_BITS;

    pub const TAG_MASK: u32 = TAG_MASK;

    pub const STATE_MASK: u32 = STATE_MASK;

    pub const OCCUPIED_TAG: u32 = OCCUPIED_TAG;

    #[inline(always)]
    #[must_use]
    pub const fn new(index: u32, generation: u32) -> Self {
        assert!(is_occupied(generation));

        // SAFETY: We checked that the state tag of `generation` is `OCCUPIED_TAG`.
        unsafe { SlotId::new_unchecked(index, generation) }
    }

    /// # Safety
    ///
    /// `generation`, when masked out with [`STATE_MASK`], must equal [`OCCUPIED_TAG`].
    ///
    /// [`STATE_MASK`]: Self::STATE_MASK
    /// [`OCCUPIED_TAG`]: Self::OCCUPIED_TAG
    #[inline(always)]
    #[must_use]
    pub const unsafe fn new_unchecked(index: u32, generation: u32) -> Self {
        debug_assert!(is_occupied(generation));

        // SAFETY: The caller must ensure that the state tag of `generation` is `OCCUPIED_TAG`.
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

    #[inline]
    fn as_u64(self) -> u64 {
        u64::from(self.index) | (u64::from(self.generation.get()) << 32)
    }
}

impl fmt::Debug for SlotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::INVALID {
            return f.write_str("INVALID");
        }

        let generation = self.generation() >> (TAG_BITS + STATE_BITS);
        write!(f, "{}v{}", self.index, generation)?;

        if self.generation() & TAG_MASK != 0 {
            write!(f, "t{}", self.generation() & TAG_MASK)?;
        }

        Ok(())
    }
}

impl PartialEq for SlotId {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_u64() == other.as_u64()
    }
}

impl Eq for SlotId {}

impl Hash for SlotId {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_u64().hash(state)
    }
}

impl PartialOrd for SlotId {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SlotId {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.as_u64().cmp(&other.as_u64())
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

// SAFETY: `Iter` semantically holds a reference to all values, and references are safe to send to
// another thread as long as the value is `Sync`. The key is a phantom parameter.
unsafe impl<K, V: Sync> Send for Iter<'_, K, V> {}

// SAFETY: `Iter` semantically holds a reference to all values, and references are safe to share
// among threads as long as the value is `Sync`. The key is a phantom parameter.
unsafe impl<K, V: Sync> Sync for Iter<'_, K, V> {}

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

            if is_occupied(generation) {
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

            if is_occupied(generation) {
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

// SAFETY: `IterMut` semantically holds a mutable reference to all values, and mutable references
// are safe to send to another thread as long as the value is `Send`. The key is a phantom
// parameter.
unsafe impl<K, V: Send> Send for IterMut<'_, K, V> {}

// SAFETY: `IterMut` semantically holds a mutable reference to all values, and mutable references
// are safe to share among threads as long as the value is `Sync`. The key is a phantom parameter.
unsafe impl<K, V: Sync> Sync for IterMut<'_, K, V> {}

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

            if is_occupied(generation) {
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

            if is_occupied(generation) {
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

const fn is_occupied(generation: u32) -> bool {
    generation & STATE_MASK == OCCUPIED_TAG
}

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
    use super::*;
    use std::thread;

    #[test]
    fn basic_usage1() {
        let map = SlotMap::new(10);
        let guard = &map.pin();

        let x = map.insert(69, guard);
        let y = map.insert(42, guard);

        assert_eq!(map.get(x, guard), Some(&69));
        assert_eq!(map.get(y, guard), Some(&42));

        map.remove(x, guard);

        let x2 = map.insert(12, guard);

        assert_eq!(map.get(x2, guard), Some(&12));
        assert_eq!(map.get(x, guard), None);

        map.remove(y, guard);
        map.remove(x2, guard);

        assert_eq!(map.get(y, guard), None);
        assert_eq!(map.get(x2, guard), None);
    }

    #[test]
    fn basic_usage2() {
        let map = SlotMap::new(10);
        let guard = &map.pin();

        let x = map.insert(1, guard);
        let y = map.insert(2, guard);
        let z = map.insert(3, guard);

        assert_eq!(map.get(x, guard), Some(&1));
        assert_eq!(map.get(y, guard), Some(&2));
        assert_eq!(map.get(z, guard), Some(&3));

        map.remove(y, guard);

        let y2 = map.insert(20, guard);

        assert_eq!(map.get(y2, guard), Some(&20));
        assert_eq!(map.get(y, guard), None);

        map.remove(x, guard);
        map.remove(z, guard);

        let x2 = map.insert(10, guard);

        assert_eq!(map.get(x2, guard), Some(&10));
        assert_eq!(map.get(x, guard), None);

        let z2 = map.insert(30, guard);

        assert_eq!(map.get(z2, guard), Some(&30));
        assert_eq!(map.get(x, guard), None);

        map.remove(x2, guard);

        assert_eq!(map.get(x2, guard), None);

        map.remove(y2, guard);
        map.remove(z2, guard);

        assert_eq!(map.get(y2, guard), None);
        assert_eq!(map.get(z2, guard), None);
    }

    #[test]
    fn basic_usage3() {
        let map = SlotMap::new(10);
        let guard = &map.pin();

        let x = map.insert(1, guard);
        let y = map.insert(2, guard);

        assert_eq!(map.get(x, guard), Some(&1));
        assert_eq!(map.get(y, guard), Some(&2));

        let z = map.insert(3, guard);

        assert_eq!(map.get(z, guard), Some(&3));

        map.remove(x, guard);
        map.remove(z, guard);

        let z2 = map.insert(30, guard);
        let x2 = map.insert(10, guard);

        assert_eq!(map.get(x2, guard), Some(&10));
        assert_eq!(map.get(z2, guard), Some(&30));
        assert_eq!(map.get(x, guard), None);
        assert_eq!(map.get(z, guard), None);

        map.remove(x2, guard);
        map.remove(y, guard);
        map.remove(z2, guard);

        assert_eq!(map.get(x2, guard), None);
        assert_eq!(map.get(y, guard), None);
        assert_eq!(map.get(z2, guard), None);
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
        let guard = &map.pin();

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
        let guard = &map.pin();

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
        let guard = &map.pin();

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
    fn get_disjoint_mut() {
        let mut map = SlotMap::new(3);

        let x = map.insert_mut(1);
        let y = map.insert_mut(2);
        let z = map.insert_mut(3);

        assert_eq!(map.get_disjoint_mut([x, y]), Some([&mut 1, &mut 2]));
        assert_eq!(map.get_disjoint_mut([y, z]), Some([&mut 2, &mut 3]));
        assert_eq!(map.get_disjoint_mut([z, x]), Some([&mut 3, &mut 1]));

        assert_eq!(
            map.get_disjoint_mut([x, y, z]),
            Some([&mut 1, &mut 2, &mut 3]),
        );
        assert_eq!(
            map.get_disjoint_mut([z, y, x]),
            Some([&mut 3, &mut 2, &mut 1]),
        );

        assert_eq!(map.get_disjoint_mut([x, x]), None);
        assert_eq!(
            map.get_disjoint_mut([x, SlotId::new(3, OCCUPIED_TAG)]),
            None,
        );

        map.remove_mut(y);

        assert_eq!(map.get_disjoint_mut([x, z]), Some([&mut 1, &mut 3]));

        assert_eq!(map.get_disjoint_mut([y]), None);
        assert_eq!(map.get_disjoint_mut([x, y]), None);
        assert_eq!(map.get_disjoint_mut([y, z]), None);

        let y = map.insert_mut(2);

        assert_eq!(
            map.get_disjoint_mut([x, y, z]),
            Some([&mut 1, &mut 2, &mut 3]),
        );

        map.remove_mut(x);
        map.remove_mut(z);

        assert_eq!(map.get_disjoint_mut([y]), Some([&mut 2]));

        assert_eq!(map.get_disjoint_mut([x]), None);
        assert_eq!(map.get_disjoint_mut([z]), None);

        map.remove_mut(y);

        assert_eq!(map.get_disjoint_mut([]), Some([]));
    }

    #[test]
    fn tagged() {
        let map = SlotMap::new(1);
        let guard = &map.pin();

        let x = map.insert_with_tag(42, 1, guard);
        assert_eq!(x.generation() & TAG_MASK, 1);
        assert_eq!(map.get(x, guard), Some(&42));
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
                for _ in 0..ITERATIONS / THREADS {
                    map.insert(0, &map.pin());
                }
            };

            for _ in 0..THREADS {
                s.spawn(inserter);
            }
        });

        thread::scope(|s| {
            let remover = || {
                for index in 0..ITERATIONS {
                    let _ = map.remove(SlotId::new(index, OCCUPIED_TAG), &map.pin());
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
        const CAPACITY: u32 = if cfg!(miri) { 10 } else { 1000 };

        let map = SlotMap::new(CAPACITY);

        thread::scope(|s| {
            let insert_remover = || {
                for _ in 0..ITERATIONS / 6 {
                    let x = map.insert(0, &map.pin());
                    let y = map.insert(0, &map.pin());
                    map.remove(y, &map.pin());
                    let z = map.insert(0, &map.pin());
                    map.remove(x, &map.pin());
                    map.remove(z, &map.pin());
                }
            };
            let iterator = || {
                for _ in 0..ITERATIONS / CAPACITY * 2 {
                    for index in 0..CAPACITY {
                        if let Some(value) = map.index(index, &map.pin()) {
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
                for i in 0..ITERATIONS {
                    if i % 10 == 0 {
                        map.insert(0, &map.pin());
                    } else {
                        thread::yield_now();
                    }
                }
            };
            let remover = || {
                for _ in 0..ITERATIONS {
                    map.remove_index(0, &map.pin());
                }
            };
            let getter = || {
                for _ in 0..ITERATIONS {
                    if let Some(value) = map.index(0, &map.pin()) {
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
