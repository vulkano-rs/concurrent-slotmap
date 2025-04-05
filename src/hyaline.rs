//! This module implements the [Hyaline-1 memory reclamation technique].
//!
//! [Hyaline-1 memory reclamation technique]: https://arxiv.org/pdf/1905.07903

use crate::{
    slot::{Slot, Vec},
    HotData, NIL,
};
use alloc::alloc::{alloc, dealloc, handle_alloc_error, realloc, Layout};
use core::{
    cell::{Cell, UnsafeCell},
    fmt,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    panic::RefUnwindSafe,
    ptr,
    sync::atomic::{
        self, AtomicPtr, AtomicUsize,
        Ordering::{Acquire, Relaxed, Release, SeqCst},
    },
};
use thread_local::ThreadLocal;

// SAFETY: `usize` and `*mut Node` have the same layout.
// TODO: Replace with `ptr::without_provanance_mut` once we bump the MSRV.
#[allow(clippy::useless_transmute)]
const INACTIVE: *mut Node = unsafe { mem::transmute(usize::MAX) };

pub(crate) struct CollectorHandle {
    ptr: *mut Collector,
}

impl CollectorHandle {
    pub(crate) unsafe fn new<V>(slots: &Vec<V>, hot_data: *const HotData) -> Self {
        let ptr = Box::into_raw(Box::new(Collector {
            slots: slots.as_ptr().cast(),
            retirement_lists: ThreadLocal::new(),
            // SAFETY: References and pointers have the same ABI for all sized pointees.
            reclaim: unsafe {
                mem::transmute::<
                    unsafe fn(u32, *const Slot<V>, &HotData),
                    unsafe fn(u32, *const u8, *const HotData),
                >(crate::reclaim)
            },
            // SAFETY: Same as the previous.
            reclaim_invalidated: unsafe {
                mem::transmute::<unsafe fn(u32, *const Slot<V>), unsafe fn(u32, *const u8)>(
                    crate::reclaim_invalidated,
                )
            },
            hot_data,
        }));

        CollectorHandle { ptr }
    }

    #[inline]
    pub(crate) fn pin(&self) -> Guard<'_> {
        let retirement_list = self.collector().retirement_lists.get_or(|| {
            // SAFETY: `self.ptr` is obviously valid for our lifetime.
            unsafe {
                RetirementList {
                    head: AtomicPtr::new(INACTIVE),
                    collector: self.ptr,
                    guard_count: Cell::new(0),
                    batch: UnsafeCell::new(LocalBatch::new()),
                }
            }
        });

        // FIXME:
        atomic::fence(SeqCst);

        retirement_list.pin()
    }

    #[inline]
    pub(crate) fn collector(&self) -> &Collector {
        // SAFETY: The pointer is valid.
        unsafe { &*self.ptr }
    }
}

impl fmt::Debug for CollectorHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CollectorHandle").finish_non_exhaustive()
    }
}

impl Drop for CollectorHandle {
    fn drop(&mut self) {
        // SAFETY: We allocated this pointer using `Box::new` and there can be no remaining
        // references to the allocation since we are being dropped.
        let _ = unsafe { Box::from_raw(self.ptr) };
    }
}

pub(crate) struct Collector {
    /// Per-thread retirement lists.
    retirement_lists: ThreadLocal<RetirementList>,
    /// A pointer to the allocation of `slot::Slot<V>`s.
    slots: *const u8,
    /// The reclamation function for the list of retired slots.
    reclaim: unsafe fn(u32, *const u8, *const HotData),
    /// The reclamation function for the list of retired invalidated slots.
    reclaim_invalidated: unsafe fn(u32, *const u8),
    /// A pointer to the free-list of the parent data structure.
    hot_data: *const HotData,
}

impl PartialEq for Collector {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for Collector {}

/// The retirement list tracking retired batches.
#[repr(align(128))]
pub(crate) struct RetirementList {
    /// The list of retired batches.
    head: AtomicPtr<Node>,
    /// The parent collector.
    collector: *const Collector,
    /// The number of `Guard`s that exist.
    guard_count: Cell<usize>,
    /// The current batch being prepared.
    batch: UnsafeCell<LocalBatch>,
}

// SAFETY: It is safe to send `collector` to another thread because it points to a heap allocation.
unsafe impl Send for RetirementList {}

// SAFETY: It is safe to share `collector` between threads because it is only used to create shared
// references. The cells are never accessed concurrently.
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

        Guard {
            retirement_list: self,
            marker: PhantomData,
        }
    }

    #[inline]
    unsafe fn enter(&self) {
        self.head.store(ptr::null_mut(), Relaxed);
        atomic::fence(SeqCst);
    }

    #[inline]
    unsafe fn defer_reclaim<V>(&self, index: u32, slots: &Vec<V>) {
        // SAFETY: The caller must ensure that this method isn't called concurrently.
        let batch = unsafe { &mut *self.batch.get() };

        // SAFETY: The caller must ensure that `index` is valid.
        let slot = unsafe { slots.get_unchecked(index) };

        slot.next_free.store(batch.head(), Relaxed);

        // SAFETY: `index` is valid and the caller must ensure that it is not reachable anymore.
        unsafe { batch.set_head(index) };
    }

    #[inline]
    unsafe fn defer_reclaim_invalidated<V>(&self, index: u32, slots: &Vec<V>) {
        // SAFETY: The caller must ensure that this method isn't called concurrently.
        let batch = unsafe { &mut *self.batch.get() };

        // SAFETY: The caller must ensure that `index` is valid.
        let slot = unsafe { slots.get_unchecked(index) };

        slot.next_free.store(batch.invalidated_head(), Relaxed);

        // SAFETY: `index` is valid and the caller must ensure that it is not reachable anymore.
        unsafe { batch.set_invalidated_head(index) };
    }

    #[inline]
    unsafe fn leave(&self) {
        // SAFETY: The caller must ensure that this method isn't called concurrently.
        let batch_is_empty = unsafe { (*self.batch.get()).is_empty() };

        if !batch_is_empty {
            // SAFETY: The caller must ensure that this method isn't called concurrently and that
            // there are no more references to any retired slots.
            unsafe { self.retire() };
        }

        let head = self.head.swap(INACTIVE, Release);

        if !head.is_null() {
            // SAFETY: The caller must ensure that this method isn't called concurrently and that
            // there are no more references to any retired slots.
            unsafe { self.traverse(head) };
        }
    }

    #[inline(never)]
    unsafe fn retire(&self) {
        // SAFETY: The caller must ensure that this method isn't called concurrently.
        let mut batch = mem::take(unsafe { &mut *self.batch.get() });

        atomic::fence(SeqCst);

        for retirement_list in &self.collector().retirement_lists {
            if retirement_list.head.load(Relaxed) == INACTIVE {
                continue;
            }

            batch.push_node(retirement_list);
        }

        let nodes = batch.node_ptr();
        let mut len = batch.node_len();
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
        let batch = unsafe { LocalBatch::from_raw(batch) };

        let collector = self.collector();

        if batch.head() != NIL {
            // SAFETY:
            // * We own the batch, which means that no more references to the slots can exist.
            // * We always push indices of existing vacant slots into the list.
            // * `collector.slots` and `collector.hot_data` are the same pointers that were used to
            //   create the collector.
            unsafe { (collector.reclaim)(batch.head(), collector.slots, collector.hot_data) };
        }

        if batch.invalidated_head() != NIL {
            // SAFETY:
            // * We own the batch, which means that no more references to the slots can exist.
            // * We always push indices of existing invalidated slots into the list.
            // * `collector.slots` and is the same pointer that was used to create the collector.
            unsafe { (collector.reclaim_invalidated)(batch.head(), collector.slots) };
        }
    }

    fn collector(&self) -> &Collector {
        // SAFETY: The constructor of `RetirementList` must ensure the pointer is valid for our
        // lifetime.
        unsafe { &*self.collector }
    }
}

/// A batch of retired slots.
#[repr(C)]
struct Batch {
    /// The number of threads that can still access the slots in this batch.
    ref_count: AtomicUsize,
    /// The list of retired slots.
    head: u32,
    /// The list of retired invalidated slots.
    invalidated_head: u32,
    /// The capacity of `nodes`.
    node_capacity: usize,
    /// The number of `nodes`.
    node_len: usize,
    /// An inline allocation of `capacity` nodes with `len` being intialized.
    nodes: [Node; 0],
}

/// A node in the retirement list.
struct Node {
    link: NodeLink,
    /// The batch that this node is a part of.
    batch: *mut Batch,
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
            *ptr::addr_of_mut!((*ptr).head) = NIL;
            *ptr::addr_of_mut!((*ptr).invalidated_head) = NIL;
            *ptr::addr_of_mut!((*ptr).node_capacity) = Self::MIN_CAP;
            *ptr::addr_of_mut!((*ptr).node_len) = 0;
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
    fn head(&self) -> u32 {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).head }
    }

    #[inline]
    fn invalidated_head(&self) -> u32 {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).invalidated_head }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.head() == NIL && self.invalidated_head() == NIL
    }

    #[inline]
    unsafe fn set_head(&self, head: u32) {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).head = head };
    }

    #[inline]
    unsafe fn set_invalidated_head(&self, head: u32) {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).invalidated_head = head };
    }

    #[inline]
    fn node_capacity(&self) -> usize {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).node_capacity }
    }

    #[inline]
    fn node_len(&self) -> usize {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).node_len }
    }

    #[inline]
    fn node_ptr(&mut self) -> *mut Node {
        // SAFETY: The pointer is valid.
        unsafe { ptr::addr_of_mut!((*self.ptr).nodes) }.cast()
    }

    #[inline]
    fn push_node(&mut self, retirement_list: &RetirementList) {
        let len = self.node_len();

        if len == self.node_capacity() {
            self.grow_one();
        }

        let node = Node {
            link: NodeLink {
                retirement_list: &retirement_list.head,
            },
            batch: ptr::null_mut(),
        };

        // SAFETY: We made sure that the index is in bounds above.
        unsafe { self.node_ptr().add(len).write(node) };

        // SAFETY: We wrote the new element above.
        unsafe { self.set_node_len(len + 1) };
    }

    #[inline(never)]
    fn grow_one(&mut self) {
        let capacity = self.node_capacity();
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
        unsafe { (*self.ptr).node_capacity = new_capacity };
    }

    #[inline]
    unsafe fn set_node_len(&mut self, len: usize) {
        // SAFETY: The pointer is valid.
        unsafe { (*self.ptr).node_len = len };
    }
}

impl Drop for LocalBatch {
    fn drop(&mut self) {
        let layout = layout_for_capacity(self.node_capacity());

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

impl Guard<'_> {
    pub(crate) fn collector(&self) -> &Collector {
        self.retirement_list.collector()
    }

    pub(crate) unsafe fn defer_reclaim<V>(&self, index: u32, slots: &Vec<V>) {
        // SAFETY:
        // * `Guard` is `!Send + !Sync`, so this cannot be called concurrently.
        // * The caller must ensure that `index` is valid.
        // * The caller must ensure that `index` is not reachable anymore.
        unsafe { self.retirement_list.defer_reclaim(index, slots) };
    }

    pub(crate) unsafe fn defer_reclaim_invalidated<V>(&self, index: u32, slots: &Vec<V>) {
        // SAFETY:
        // * `Guard` is `!Send + !Sync`, so this cannot be called concurrently.
        // * The caller must ensure that `index` is valid.
        // * The caller must ensure that `index` is not reachable anymore.
        unsafe { self.retirement_list.defer_reclaim_invalidated(index, slots) }
    }
}

impl fmt::Debug for Guard<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Guard").finish_non_exhaustive()
    }
}

impl Drop for Guard<'_> {
    #[inline]
    fn drop(&mut self) {
        let guard_count = self.retirement_list.guard_count.get();
        self.retirement_list.guard_count.set(guard_count - 1);

        if guard_count == 1 {
            // SAFETY: We are dropping the last guard, so there cannot be any more references to any
            // retired slots.
            unsafe { self.retirement_list.leave() };
        }
    }
}
