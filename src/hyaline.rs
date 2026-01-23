//! This module implements the [Hyaline-1 memory reclamation technique].
//!
//! [Hyaline-1 memory reclamation technique]: https://arxiv.org/pdf/1905.07903

use crate::{slot::Vec, Slot, SHARD_COUNT};
use alloc::alloc::{alloc, dealloc, handle_alloc_error, realloc, Layout};
use core::{
    cell::{Cell, UnsafeCell},
    fmt,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    panic::RefUnwindSafe,
    ptr, slice,
    sync::atomic::{
        self, AtomicPtr, AtomicUsize,
        Ordering::{Acquire, Relaxed, Release, SeqCst},
    },
};
use std::{num::NonZeroUsize, thread};
use thread_local::ThreadLocal;

// SAFETY: `usize` and `*mut Node` have the same layout.
// TODO: Replace with `ptr::without_provanance_mut` once we bump the MSRV.
#[allow(integer_to_ptr_transmutes, clippy::useless_transmute)]
const INACTIVE: *mut Node = unsafe { mem::transmute(usize::MAX) };

const MIN_RETIRED_LEN: usize = 64;

pub struct CollectorHandle {
    ptr: *mut Collector,
}

// SAFETY: `Collector` is `Send + Sync` and its lifetime is enforced with reference counting.
unsafe impl Send for CollectorHandle {}

// SAFETY: `Collector` is `Send + Sync` and its lifetime is enforced with reference counting.
unsafe impl Sync for CollectorHandle {}

impl Default for CollectorHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl CollectorHandle {
    #[must_use]
    pub fn new() -> Self {
        if SHARD_COUNT.load(Relaxed) == 0 {
            let num_cpus = thread::available_parallelism()
                .map(NonZeroUsize::get)
                .unwrap_or(1);
            SHARD_COUNT.store(num_cpus.next_power_of_two(), Relaxed);
        }

        let ptr = Box::into_raw(Box::new(Collector {
            retirement_lists: ThreadLocal::new(),
            handle_count: AtomicUsize::new(1),
        }));

        // SAFETY: We made sure that initialize the `handle_count` to `1` such that the handle's
        // drop implementation cannot drop the `Collector` while another handle still exists.
        unsafe { CollectorHandle { ptr } }
    }

    /// # Safety
    ///
    /// The returned `Guard` must not outlive any collection that uses this collector. It would
    /// result in the `Guard`'s drop implementation attempting to free slots from the already freed
    /// collection, resulting in a Use-After-Free. You must ensure that the `Guard` has its
    /// lifetime bound to the collections it protects.
    #[inline]
    #[must_use]
    pub unsafe fn pin(&self) -> Guard<'_> {
        let mut is_fresh_entry = false;

        let retirement_list = self.collector().retirement_lists.get_or(|| {
            is_fresh_entry = true;

            crate::set_shard_index();

            RetirementList {
                head: AtomicPtr::new(INACTIVE),
                // SAFETY: The collector cannot be dropped until all handles have been dropped, at
                // which point it is impossible to access this handle.
                collector: ManuallyDrop::new(unsafe { ptr::read(self) }),
                guard_count: Cell::new(0),
                batch: UnsafeCell::new(LocalBatch::new()),
            }
        });

        if is_fresh_entry {
            atomic::fence(SeqCst);
        }

        retirement_list.pin()
    }

    #[inline]
    fn collector(&self) -> &Collector {
        // SAFETY: The pointer is valid.
        unsafe { &*self.ptr }
    }
}

impl Clone for CollectorHandle {
    #[inline]
    fn clone(&self) -> Self {
        #[allow(clippy::cast_sign_loss)]
        if self.collector().handle_count.fetch_add(1, Relaxed) > isize::MAX as usize {
            std::process::abort();
        }

        // SAFETY: We incremented the `handle_count` above such that the handle's drop
        // implementation cannot drop the `Collector` while another handle still exists.
        unsafe { CollectorHandle { ptr: self.ptr } }
    }
}

impl fmt::Debug for CollectorHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CollectorHandle").finish_non_exhaustive()
    }
}

impl PartialEq for CollectorHandle {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl Eq for CollectorHandle {}

impl Drop for CollectorHandle {
    #[inline]
    fn drop(&mut self) {
        if self.collector().handle_count.fetch_sub(1, Release) == 1 {
            atomic::fence(Acquire);

            // SAFETY: The handle count has gone to zero, which means that no other threads can
            // register a new handle. The `Acquire` fence above ensures that the drop is
            // synchronized with the above decrement, such that no access to the `Collector` can be
            // ordered after the drop.
            let _ = unsafe { Box::from_raw(self.ptr) };
        }
    }
}

struct Collector {
    /// Per-thread retirement lists.
    retirement_lists: ThreadLocal<RetirementList>,
    /// The number of `CollectorHandle`s that exist.
    handle_count: AtomicUsize,
}

/// The retirement list tracking retired batches.
#[repr(align(128))]
pub(crate) struct RetirementList {
    /// The list of retired batches.
    head: AtomicPtr<Node>,
    /// The parent collector.
    collector: ManuallyDrop<CollectorHandle>,
    /// The number of `Guard`s that exist.
    guard_count: Cell<usize>,
    /// The current batch being prepared.
    batch: UnsafeCell<LocalBatch>,
}

// SAFETY: The cells are never accessed concurrently.
unsafe impl Sync for RetirementList {}

// While `RetirementList` does contain interior mutability, this is never exposed to the user and
// no invariant can be broken.
impl RefUnwindSafe for RetirementList {}

impl RetirementList {
    #[inline]
    pub(crate) fn pin(&self) -> Guard<'_> {
        let guard_count = self.guard_count.get();
        self.guard_count.set(guard_count.checked_add(1).unwrap());

        if guard_count == 0 {
            // SAFETY: The guard count was zero, which means that we couldn't have entered already.
            unsafe { self.enter() };
        }

        // SAFETY:
        // * We incremented the `guard_count` above such that the guard's drop implementation cannot
        //   unpin the participant while another guard still exists.
        // * We made sure to pin the participant if it wasn't already.
        unsafe { Guard::new(self) }
    }

    #[inline]
    unsafe fn enter(&self) {
        self.head.store(ptr::null_mut(), Relaxed);
        atomic::fence(SeqCst);
    }

    #[inline]
    unsafe fn defer_reclaim(
        &self,
        index: u32,
        slots: *const u8,
        reclaim: unsafe fn(u32, *const u8),
    ) {
        // SAFETY: The caller must ensure that this method isn't called concurrently.
        let batch = unsafe { &mut *self.batch.get() };

        // SAFETY: The caller must ensure that `index` is valid and that it is not reachable
        // anymore, that `slots` is a valid pointer to the allocation of `Slot`s, and that `reclaim`
        // is safe to call with the `index` and `slots`.
        unsafe { batch.push(index, slots, reclaim) };

        if batch.len() == MIN_RETIRED_LEN {
            // SAFETY: The caller must ensure that this method isn't called concurrently.
            unsafe { self.retire() };
        }
    }

    #[inline(never)]
    unsafe fn retire(&self) {
        // SAFETY: The caller must ensure that this method isn't called concurrently.
        let batch = unsafe { &mut *self.batch.get() };

        if batch.is_empty() {
            return;
        }

        let mut batch = mem::take(batch);
        let retired_len = batch.len();
        let mut len = 0;

        // SAFETY: `batch.len()` is the number of retired slots in the batch.
        unsafe { batch.set_retired_len(retired_len) };

        atomic::fence(SeqCst);

        for retirement_list in &self.collector.collector().retirement_lists {
            if retirement_list.head.load(Relaxed) == INACTIVE {
                continue;
            }

            if len >= retired_len {
                // SAFETY: This node is never getting retired since it is outside of `retired_len`.
                unsafe { batch.push(0, ptr::null(), |_, _| {}) };
            }

            // SAFETY: We made sure that the index is in bounds above.
            let node = unsafe { batch.as_mut_slice().get_unchecked_mut(len) };

            node.link.retirement_list = &retirement_list.head;

            len += 1;
        }

        let nodes = batch.as_mut_ptr();
        let batch = batch.into_raw();

        atomic::fence(Acquire);

        #[allow(clippy::mut_range_bound)]
        'outer: for node_index in 0..len {
            // SAFETY: The pointer is valid and the index is in bounds.
            let node = unsafe { nodes.add(node_index) };

            // SAFETY: The pointer is valid.
            unsafe { (*node).batch = batch };

            // SAFETY:
            // * The `node` pointer is valid.
            // * We pushed the `retirement_list` union variant into the batch above.
            // * The `retirement_list` pointer must have staid valid because `ThreadLocal` entries
            //   are never deinitialized.
            let list = unsafe { &*(*node).link.retirement_list };

            let mut head = list.load(Relaxed);

            loop {
                if head == INACTIVE {
                    atomic::fence(Acquire);
                    len -= 1;
                    continue 'outer;
                }

                // SAFETY: The pointer is valid.
                unsafe { (*node).link.next = head };

                match list.compare_exchange_weak(head, node, Release, Relaxed) {
                    Ok(_) => break,
                    Err(new_head) => head = new_head,
                }
            }
        }

        // SAFETY: The pointer is valid.
        if unsafe { (*batch).ref_count.fetch_add(len, Release) }.wrapping_add(len) == 0 {
            // SAFETY: We had the last reference.
            unsafe { self.reclaim(batch) };
        }
    }

    #[inline]
    unsafe fn leave(&self) {
        let head = self.head.swap(INACTIVE, Release);

        if !head.is_null() {
            // SAFETY: The caller must ensure that this method isn't called concurrently and that
            // there are no more references to any retired slots.
            unsafe { self.traverse(head) };
        }
    }

    #[cold]
    unsafe fn traverse(&self, mut head: *mut Node) {
        atomic::fence(Acquire);

        while !head.is_null() {
            // SAFETY: We always push valid pointers into the list.
            let batch = unsafe { (*head).batch };

            // SAFETY: We always push valid pointers into the list, and when doing so, we make sure
            // that the `next` union variant is initialized.
            let next = unsafe { (*head).link.next };

            // SAFETY: The caller must ensure that there are no more references to any retired
            // slots.
            let ref_count = unsafe { (*batch).ref_count.fetch_sub(1, Release) }.wrapping_sub(1);

            if ref_count == 0 {
                // SAFETY: We had the last reference.
                unsafe { self.reclaim(batch) };
            }

            head = next;
        }
    }

    unsafe fn reclaim(&self, batch: *mut Batch) {
        atomic::fence(Acquire);

        // SAFETY: The caller must ensure that we own the batch.
        let mut batch = unsafe { LocalBatch::from_raw(batch) };

        for node in batch.retired_as_mut_slice() {
            // SAFETY:
            // * We own the batch, which means that no more references to the slots can exist.
            // * We always push indices of existing vacant slots into the list.
            // * `node.slots` is the same pointer that was used when pushing the node.
            unsafe { (node.reclaim)(node.index, node.slots) };
        }
    }
}

impl Drop for Collector {
    fn drop(&mut self) {
        atomic::fence(Acquire);

        for retirement_list in &mut self.retirement_lists {
            let batch = retirement_list.batch.get_mut();

            if batch.is_empty() {
                continue;
            }

            for node in batch.retired_as_mut_slice() {
                // SAFETY:
                // * We have mutable access to the batch, which means that no more references to the
                //   slots can exist.
                // * We always push indices of existing vacant slots into the list.
                // * `node.slots` is the same pointer that was used when pushing the node.
                unsafe { (node.reclaim)(node.index, node.slots) };
            }
        }
    }
}

/// A batch of retired slots.
#[repr(C)]
struct Batch {
    /// The number of threads that can still access the slots in this batch.
    ref_count: AtomicUsize,
    /// The capacity of `nodes`.
    capacity: usize,
    /// The number of `nodes`.
    len: usize,
    /// The number of `nodes` that should be retired.
    retired_len: usize,
    /// An inline allocation of `capacity` nodes with `len` being initialized.
    nodes: [Node; 0],
}

/// A node in the retirement list.
struct Node {
    link: NodeLink,
    /// The batch that this node is a part of.
    batch: *mut Batch,
    /// The index of the retired slot.
    index: u32,
    /// A pointer to the allocation of `slot::Slot<V>`s.
    slots: *const u8,
    /// The reclamation function for the retired slot.
    reclaim: unsafe fn(u32, *const u8),
}

union NodeLink {
    /// The retirement list.
    retirement_list: *const AtomicPtr<Node>,
    /// The next node in the retirement list.
    next: *mut Node,
}

struct LocalBatch {
    ptr: *mut Batch,
}

// SAFETY: We own the batch and access to it is synchronized using mutable references.
unsafe impl Send for LocalBatch {}

// SAFETY: We own the batch and access to it is synchronized using mutable references.
unsafe impl Sync for LocalBatch {}

impl Default for LocalBatch {
    fn default() -> Self {
        LocalBatch::new()
    }
}

impl LocalBatch {
    const MIN_CAP: usize = 4;

    fn new() -> Self {
        let layout = layout_for_capacity(Self::MIN_CAP);

        // SAFETY: The layout is not zero-sized.
        let ptr = unsafe { alloc(layout) }.cast::<Batch>();

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        // SAFETY: We successfully allocated the batch.
        unsafe {
            *ptr::addr_of_mut!((*ptr).ref_count) = AtomicUsize::new(0);
            *ptr::addr_of_mut!((*ptr).capacity) = Self::MIN_CAP;
            *ptr::addr_of_mut!((*ptr).len) = 0;
            *ptr::addr_of_mut!((*ptr).retired_len) = 0;
        }

        LocalBatch { ptr }
    }

    #[inline]
    unsafe fn from_raw(ptr: *mut Batch) -> Self {
        LocalBatch { ptr }
    }

    #[inline]
    fn into_raw(self) -> *mut Batch {
        ManuallyDrop::new(self).ptr
    }

    #[inline]
    fn capacity(&self) -> usize {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).capacity }
    }

    #[inline]
    fn len(&self) -> usize {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).len }
    }

    #[inline]
    fn retired_len(&self) -> usize {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).retired_len }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Node {
        // SAFETY: The pointer is valid.
        unsafe { ptr::addr_of_mut!((*self.ptr).nodes) }.cast()
    }

    fn as_mut_slice(&mut self) -> &mut [Node] {
        // SAFETY: The pointer is valid and the caller of `LocalBatch::set_len` must ensure that the
        // length is the correct number of nodes that are initialized.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    #[inline]
    fn retired_as_mut_slice(&mut self) -> &mut [Node] {
        // SAFETY: The pointer is valid and the caller of `LocalBatch::set_retired_len` must ensure
        // that the length is the correct number of nodes that should be retired.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.retired_len()) }
    }

    #[inline]
    unsafe fn push(&mut self, index: u32, slots: *const u8, reclaim: unsafe fn(u32, *const u8)) {
        let len = self.len();

        if len == self.capacity() {
            self.grow_one();
        }

        let node = Node {
            link: NodeLink {
                retirement_list: ptr::null(),
            },
            batch: ptr::null_mut(),
            index,
            slots,
            reclaim,
        };

        // SAFETY: We made sure that the index is in bounds above.
        unsafe { self.as_mut_ptr().add(len).write(node) };

        // SAFETY: We wrote the new element above.
        unsafe { self.set_len(len + 1) };
    }

    #[inline(never)]
    fn grow_one(&mut self) {
        let capacity = self.capacity();
        let new_capacity = capacity * 2;
        let layout = layout_for_capacity(capacity);
        let new_layout = layout_for_capacity(new_capacity);

        // SAFETY:
        // * `self.ptr` was allocated via the global allocator.
        // * `layout` is the current layout.
        // * `new_layout.size()`, when rounded up to the nearest multiple of `new_layout.align()`,
        //   cannot overflow `isize` since we used `Layout` for the layout calculation.
        let new_ptr = unsafe { realloc(self.ptr.cast(), layout, new_layout.size()) };

        if new_ptr.is_null() {
            handle_alloc_error(new_layout);
        }

        self.ptr = new_ptr.cast();

        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).capacity = new_capacity };
    }

    #[inline]
    unsafe fn set_len(&mut self, len: usize) {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).len = len };
    }

    #[inline]
    unsafe fn set_retired_len(&mut self, len: usize) {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).retired_len = len };
    }
}

impl Drop for LocalBatch {
    fn drop(&mut self) {
        let layout = layout_for_capacity(self.capacity());

        // SAFETY:
        // * `self.ptr` was allocated using the global allocator.
        // * `layout` is the layout of the allocation.
        unsafe { dealloc(self.ptr.cast(), layout) };
    }
}

fn layout_for_capacity(capacity: usize) -> Layout {
    Layout::new::<Batch>()
        .extend(Layout::array::<Node>(capacity).unwrap())
        .unwrap()
        .0
}

pub struct Guard<'a> {
    retirement_list: &'a RetirementList,
    marker: PhantomData<*const ()>,
}

impl<'a> Guard<'a> {
    #[inline]
    unsafe fn new(retirement_list: &'a RetirementList) -> Self {
        Guard {
            retirement_list,
            marker: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub fn collector(&self) -> &CollectorHandle {
        &self.retirement_list.collector
    }

    #[inline]
    pub(crate) unsafe fn defer_reclaim<V>(&self, index: u32, slots: &Vec<V>) {
        let slots = slots.as_ptr().cast();
        let reclaim = transmute_reclaim_fp(crate::reclaim::<V>);

        // SAFETY:
        // * `Guard` is `!Send + !Sync`, so this cannot be called concurrently.
        // * The caller must ensure that `index` is valid and that it is not reachable anymore.
        // * `slots` is a valid pointer to the allocation of `Slot<V>`s.
        // * The caller must ensure that `reclaim` is safe to call with the `index` and `slots`.
        unsafe { self.retirement_list.defer_reclaim(index, slots, reclaim) };
    }

    #[inline]
    pub(crate) unsafe fn defer_reclaim_invalidated<V>(&self, index: u32, slots: &Vec<V>) {
        let slots = slots.as_ptr().cast();
        let reclaim = transmute_reclaim_fp(crate::reclaim_invalidated::<V>);

        // SAFETY:
        // * `Guard` is `!Send + !Sync`, so this cannot be called concurrently.
        // * The caller must ensure that `index` is valid and that it is not reachable anymore.
        // * `slots` is a valid pointer to the allocation of `Slot<V>`s.
        // * The caller must ensure that `reclaim` is safe to call with the `index` and `slots`.
        unsafe { self.retirement_list.defer_reclaim(index, slots, reclaim) }
    }

    #[inline]
    pub fn flush(&self) {
        // SAFETY: `Guard` is `!Send + !Sync`, so this cannot be called concurrently.
        unsafe { self.retirement_list.retire() };
    }
}

fn transmute_reclaim_fp<V>(fp: unsafe fn(u32, *const Slot<V>)) -> unsafe fn(u32, *const u8) {
    // SAFETY: Pointers have the same ABI for all sized pointees.
    unsafe { mem::transmute::<unsafe fn(u32, *const Slot<V>), unsafe fn(u32, *const u8)>(fp) }
}

impl fmt::Debug for Guard<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Guard").finish_non_exhaustive()
    }
}

impl Clone for Guard<'_> {
    #[inline]
    fn clone(&self) -> Self {
        let guard_count = self.retirement_list.guard_count.get();
        self.retirement_list
            .guard_count
            .set(guard_count.checked_add(1).unwrap());

        // SAFETY:
        // * We incremented the `guard_count` above, such that the guard's drop implementation
        //   cannot unpin the participant while another guard still exists.
        // * The participant is already pinned as this guard's existence is a proof of that.
        unsafe { Guard::new(self.retirement_list) }
    }
}

impl Drop for Guard<'_> {
    #[inline]
    fn drop(&mut self) {
        let guard_count = self.retirement_list.guard_count.get();
        self.retirement_list.guard_count.set(guard_count - 1);

        if guard_count == 1 {
            // SAFETY:
            // * `Guard` is `!Send + !Sync`, so this cannot be called concurrently.
            // * We are dropping the last guard, so there cannot be any more references to any
            //   retired slots.
            unsafe { self.retirement_list.leave() };
        }
    }
}
