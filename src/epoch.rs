// This module is heavily inspired by crossbeam-epoch v0.9, licensed under either of
// * Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// * MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)
// at your option.
//
// The differences include:
// * Being stripped down to only the global and local epochs and the guard.
// * Allowing retrieving the epoch a guard is pinned in.
// * The list of locals is not lock-free (as there's no useful work for threads to do concurrently
//   anyway) but instead protected by a mutex.
// * Implementing `Clone` for `Guard` to allow use inside `Cow`.
// * Other small miscellaneous changes.

use crate::{Backoff, CacheAligned};
use alloc::{borrow::Cow, boxed::Box};
use core::{
    cell::Cell,
    fmt,
    marker::PhantomData,
    ptr::NonNull,
    sync::atomic::{
        self, AtomicBool, AtomicU32, AtomicUsize,
        Ordering::{Acquire, Relaxed, Release, SeqCst},
    },
};

/// The bit of `Local::epoch` which signifies that the participant is pinned.
const PINNED_BIT: u32 = 1 << 0;

/// The number of pinnings between a participant tries to advance the global epoch.
#[cfg(not(miri))]
pub(crate) const PINNINGS_BETWEEN_ADVANCE: usize = 128;
#[cfg(miri)]
pub(crate) const PINNINGS_BETWEEN_ADVANCE: usize = 4;

/// A handle to a global epoch.
pub struct GlobalHandle {
    ptr: NonNull<Global>,
}

// SAFETY: `Global` is `Send + Sync` and its lifetime is enforced with reference counting.
unsafe impl Send for GlobalHandle {}

// SAFETY: `Global` is `Send + Sync` and its lifetime is enforced with reference counting.
unsafe impl Sync for GlobalHandle {}

impl Default for GlobalHandle {
    /// Creates a new global epoch.
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalHandle {
    /// Creates a new global epoch.
    #[must_use]
    pub fn new() -> Self {
        Global::register()
    }

    /// Registers a new local epoch in the global list of participants.
    #[inline]
    #[must_use]
    pub fn register_local(&self) -> UniqueLocalHandle {
        Local::register(self)
    }

    #[inline]
    pub(crate) fn epoch(&self) -> u32 {
        self.global().epoch.load(Relaxed)
    }

    #[inline]
    fn global(&self) -> &Global {
        // SAFETY: The constructor of `GlobalHandle` must ensure that the pointer stays valid for
        // the lifetime of the handle.
        unsafe { self.ptr.as_ref() }
    }
}

impl Clone for GlobalHandle {
    /// Creates a new handle to the same global epoch.
    #[inline]
    fn clone(&self) -> Self {
        #[allow(clippy::cast_sign_loss)]
        if self.global().handle_count.fetch_add(1, Relaxed) > isize::MAX as usize {
            abort();
        }

        // SAFETY: We incremented the `handle_count` above, such that the handle's drop
        // implementation cannot drop the `Global` while another handle still exists.
        unsafe { GlobalHandle { ptr: self.ptr } }
    }
}

impl fmt::Debug for GlobalHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlobalHandle").finish_non_exhaustive()
    }
}

impl PartialEq for GlobalHandle {
    /// Returns `true` if both handles refer to the same global epoch.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl Eq for GlobalHandle {}

impl Drop for GlobalHandle {
    /// Drops the handle.
    ///
    /// If there are no other handles to this global epoch, the global epoch will be dropped.
    #[inline]
    fn drop(&mut self) {
        if self.global().handle_count.fetch_sub(1, Release) == 1 {
            // SAFETY: The handle count has gone to zero, which means that no other threads can
            // register a new handle. `Global::unregister` ensures that the drop is synchronized
            // with the above decrement, such that no access to the `Global` can be ordered after
            // the drop.
            unsafe { Global::unregister(self.ptr) };
        }
    }
}

/// A [`LocalHandle`] that can be safely sent to other threads.
///
/// This is enforced through ownership over the local epoch and by using borrowed guards.
pub struct UniqueLocalHandle {
    inner: LocalHandle,
}

impl UniqueLocalHandle {
    /// This function behaves the same as [`LocalHandle::pin`], except that it returns a guard
    /// whose lifetime is bound to this handle.
    #[inline]
    #[must_use]
    pub fn pin(&self) -> Guard<'_> {
        self.inner.pin()
    }

    /// Returns a handle to the global epoch.
    #[inline]
    #[must_use]
    pub fn global(&self) -> &GlobalHandle {
        self.inner.global()
    }

    /// Returns the inner handle, consuming the unique wrapper.
    ///
    /// This allows you to make clones of the handle and get access to `'static` guards, however
    /// you lose the ability to send the handle to another thread in turn.
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> LocalHandle {
        self.inner
    }
}

// SAFETY: The constructor of `UniqueLocalHandle` must ensure that the handle is unique, which
// means that no other `LocalHandle`s referencing the same `Local` can exist. This, together with
// the fact that we don't allow any access to the `Local` other than borrowed, ensures that the
// `Local` cannot be accessed from more than one thread at a time.
unsafe impl Send for UniqueLocalHandle {}

impl fmt::Debug for UniqueLocalHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UniqueLocalHandle").finish_non_exhaustive()
    }
}

/// A handle to a local epoch.
pub struct LocalHandle {
    ptr: NonNull<Local>,
}

impl LocalHandle {
    /// Pins the local epoch, such that no accesses done while the returned `Guard` exists can
    /// cross an epoch boundary. It is important to pin the local epoch before doing any kind of
    /// access, such that no accesses can bleed into the previous epoch. Similarly, the pin must
    /// persist for as long as any accesses from the pinned epoch can persist.
    //
    // The `unwrap` below can't actually happen in any reasonable program.
    #[allow(clippy::missing_panics_doc)]
    #[inline]
    #[must_use]
    pub fn pin(&self) -> Guard<'static> {
        let local = self.local();
        let global = local.global();

        let guard_count = local.guard_count.get();
        local.guard_count.set(guard_count.checked_add(1).unwrap());

        if guard_count == 0 {
            let global_epoch = global.epoch.load(Relaxed);
            let new_epoch = global_epoch | PINNED_BIT;
            local.epoch.store(new_epoch, Relaxed);

            // This fence acts two-fold:
            // * It synchronizes with the `Release` store of the global epoch in
            //   `Global::try_advance`, which in turn ensures that the accesses all other
            //   participants did in a previous epoch are visible to us going forward when crossing
            //   an epoch boundary.
            // * It ensures that that no accesses we do going forward can be ordered before this
            //   point, therefore "bleeding" into the previous epoch, when crossing an epoch
            //   boundary.
            atomic::fence(SeqCst);

            local.pin_count.set(local.pin_count.get().wrapping_add(1));

            if local.pin_count.get() % PINNINGS_BETWEEN_ADVANCE == 0 {
                global.try_advance();
            }
        }

        // SAFETY:
        // * We incremented the `guard_count` above, such that the guard's drop implementation
        //   cannot unpin the participant while another guard still exists.
        // * We made sure to pin the participant if it wasn't already and made sure that accesses
        //   from this point on can't leak into the previous epoch.
        unsafe { Guard::new(self.ptr) }
    }

    /// Returns a handle to the global epoch.
    #[inline]
    #[must_use]
    pub fn global(&self) -> &GlobalHandle {
        &self.local().global
    }

    #[inline]
    fn local(&self) -> &Local {
        // SAFETY: The constructor of `LocalHandle` must ensure that the pointer stays valid for the
        // lifetime of the handle.
        unsafe { self.ptr.as_ref() }
    }
}

impl Clone for LocalHandle {
    /// Creates a new handle to the same local epoch.
    #[inline]
    fn clone(&self) -> Self {
        let local = self.local();

        local.handle_count.set(local.handle_count.get() + 1);

        // SAFETY: We incremented the `handle_count` above, such that the handle's drop
        // implementation cannot drop the `Local` while another handle still exists.
        unsafe { LocalHandle { ptr: self.ptr } }
    }
}

impl fmt::Debug for LocalHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalHandle").finish_non_exhaustive()
    }
}

impl Drop for LocalHandle {
    /// Drops the handle.
    ///
    /// If there are no other handles or guards that refer to this local epoch, it will be
    /// unregistered from the global list of participants.
    #[inline]
    fn drop(&mut self) {
        let local = self.local();

        // SAFETY: The constructor of `LocalHandle` must ensure that `handle_count` has been
        // incremented before construction of the handle.
        unsafe { local.handle_count.set(local.handle_count.get() - 1) };

        if local.handle_count.get() == 0 && local.guard_count.get() == 0 {
            // SAFETY: We checked that both the handle count and guard count went to zero, which
            // means that no other references to the `Local` can exist after this point.
            unsafe { Local::unregister(self.ptr) };
        }
    }
}

/// A guard that keeps the local epoch pinned.
pub struct Guard<'a> {
    local: NonNull<Local>,
    marker: PhantomData<&'a ()>,
}

impl Guard<'_> {
    unsafe fn new(local: NonNull<Local>) -> Self {
        Guard {
            local,
            marker: PhantomData,
        }
    }

    /// Returns a handle to the global epoch.
    #[inline]
    #[must_use]
    pub fn global(&self) -> &GlobalHandle {
        &self.local().global
    }

    /// Tries to advance the global epoch. Returns `true` if the epoch was successfully advanced.
    #[allow(clippy::must_use_candidate)]
    #[inline]
    pub fn try_advance_global(&self) -> bool {
        let local = self.local();
        // This prevents us from trying to advance the global epoch in `LocalHandle::pin`.
        local.pin_count.set(0);

        local.global().try_advance()
    }

    #[inline]
    fn local(&self) -> &Local {
        // SAFETY: The constructor of `Guard` must ensure that the pointer stays valid for the
        // lifetime of the guard.
        unsafe { self.local.as_ref() }
    }
}

impl Clone for Guard<'_> {
    /// Creates a new guard for the same local epoch.
    #[inline]
    fn clone(&self) -> Self {
        let local = self.local();

        let guard_count = local.guard_count.get();
        local.guard_count.set(guard_count.checked_add(1).unwrap());

        // SAFETY:
        // * We incremented the `guard_count` above, such that the guard's drop implementation
        //   cannot unpin the participant while another guard still exists.
        // * The participant is already pinned, as this guard's existence is a proof of that.
        unsafe { Guard::new(self.local) }
    }
}

impl fmt::Debug for Guard<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Guard").finish_non_exhaustive()
    }
}

impl Drop for Guard<'_> {
    /// Drops the guard.
    ///
    /// If there are no other guards keeping the local epoch pinned, it will be unpinned. If there
    /// are also no handles to the local epoch, it will be unregistered from the global list of
    /// participants.
    #[inline]
    fn drop(&mut self) {
        let local = self.local();

        // SAFETY: The constructor of `Guard` must ensure that the `guard_count` has been
        // incremented before construction of the guard.
        unsafe { local.guard_count.set(local.guard_count.get() - 1) };

        if local.guard_count.get() == 0 {
            // SAFETY:
            // * The `Release` ordering synchronizes with the `Acquire` fence in
            //   `Global::try_advance`, which in turn ensures that all accesses done by us until
            //   this point are visible to all other participants when crossing an epoch boundary.
            // * We checked that the guard count went to zero, which means that no other guards can
            //   exist and it is safe to unpin the participant.
            unsafe { local.epoch.store(0, Release) };

            if local.handle_count.get() == 0 {
                // SAFETY: We checked that both the handle count and guard count went to zero, which
                // means that no other references to the `Local` can exist after this point.
                unsafe { Local::unregister(self.local) };
            }
        }
    }
}

impl<'a> From<&'a Guard<'a>> for Cow<'a, Guard<'a>> {
    #[inline]
    fn from(guard: &'a Guard<'a>) -> Self {
        Cow::Borrowed(guard)
    }
}

impl<'a> From<Guard<'a>> for Cow<'_, Guard<'a>> {
    #[inline]
    fn from(guard: Guard<'a>) -> Self {
        Cow::Owned(guard)
    }
}

#[repr(C)]
struct Global {
    /// The head of the global list of `Local`s participating in the memory reclamation.
    local_list_head: Cell<Option<NonNull<Local>>>,

    /// The lock that protects the local list.
    local_list_lock: AtomicBool,

    /// The number of `GlobalHandle`s to this `Global` that exist.
    handle_count: AtomicUsize,

    _alignment: CacheAligned,

    /// The global epoch counter. This can only be advanced if all pinned local epochs are pinned
    /// in the current global epoch.
    epoch: AtomicU32,
}

// SAFETY: Access to the linked list of locals is synchronized with a mutex.
unsafe impl Sync for Global {}

impl Global {
    fn register() -> GlobalHandle {
        let global = Box::new(Global {
            local_list_head: Cell::new(None),
            local_list_lock: AtomicBool::new(false),
            handle_count: AtomicUsize::new(1),
            _alignment: CacheAligned,
            epoch: AtomicU32::new(0),
        });

        // SAFETY: `Box` is guaranteed to be non-null.
        let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(global)) };

        // SAFETY: We initialized the `handle_count` to 1, such that the handle's drop
        // implementation cannot drop the `Global` while another handle still exists.
        unsafe { GlobalHandle { ptr } }
    }

    #[inline(never)]
    unsafe fn unregister(global: NonNull<Global>) {
        // `Acquire` synchronizes with the `Release` ordering in `GlobalHandle::drop`.
        //
        // SAFETY: The caller must ensure that `global` is a valid pointer to a `Global`.
        unsafe { global.as_ref() }.handle_count.load(Acquire);

        // SAFETY: The caller must ensure that the `Global` can't be accessed after this point.
        let _ = unsafe { Box::from_raw(global.as_ptr()) };
    }

    fn lock_local_list(&self) {
        let mut backoff = Backoff::new();

        loop {
            match self
                .local_list_lock
                .compare_exchange_weak(false, true, Acquire, Relaxed)
            {
                Ok(_) => break,
                Err(_) => backoff.spin(),
            }
        }
    }

    fn try_lock_local_list(&self) -> bool {
        self.local_list_lock
            .compare_exchange(false, true, Acquire, Relaxed)
            .is_ok()
    }

    unsafe fn unlock_local_list(&self) {
        self.local_list_lock.store(false, Release);
    }

    #[inline(never)]
    fn try_advance(&self) -> bool {
        let global_epoch = self.epoch.load(Relaxed);

        // Ensure that none of the loads of the local epochs can be ordered before the load of the
        // global epoch.
        atomic::fence(SeqCst);

        if !self.try_lock_local_list() {
            // Another thread beat us to it.
            return false;
        }

        let mut head = self.local_list_head.get();

        while let Some(local) = head {
            // SAFETY: The list of locals always contains valid pointers.
            let local = unsafe { local.as_ref() };
            let local_epoch = local.epoch.load(Relaxed);

            if local_epoch & PINNED_BIT != 0 && local_epoch & !PINNED_BIT != global_epoch {
                // SAFETY: We locked the local list above.
                unsafe { self.unlock_local_list() };

                return false;
            }

            head = local.next.get();
        }

        // SAFETY: We locked the local list above.
        unsafe { self.unlock_local_list() };

        let new_epoch = global_epoch.wrapping_add(2);

        // This essentially acts as a global `AcqRel` barrier. Only after ensuring that all
        // participants are pinned in the current global epoch do we synchronize with them using
        // the `Acquire` fence, which synchronizes with every participant's `Release` store of its
        // local epoch in `Guard::drop`, ensuring that all accesses done by every participant until
        // they were unpinned are visible here. The `Release` ordering on the global epoch then
        // ensures that all participants subsequently pinned in the new epoch also see all accesses
        // of all other participants from the previous epoch.
        atomic::fence(Acquire);
        self.epoch.store(new_epoch, Release);

        true
    }
}

#[repr(C)]
struct Local {
    /// The next `Local` in the global list of participants.
    next: Cell<Option<NonNull<Self>>>,

    /// The previous `Local` in the global list of participants.
    prev: Cell<Option<NonNull<Self>>>,

    /// The local epoch counter. When this epoch is pinned, it ensures that the global epoch cannot
    /// be advanced more than one step until it is unpinned.
    epoch: AtomicU32,

    _alignment: CacheAligned,

    /// A handle to the `Global` which this `Local` is participating in.
    global: GlobalHandle,

    /// The number of `LocalHandle`s to this participant that exist.
    handle_count: Cell<usize>,

    /// The number of `Guard`s of this participant that exist.
    guard_count: Cell<usize>,

    /// The number of pinnings this participant has gone through in total.
    pin_count: Cell<usize>,
}

impl Local {
    #[inline(never)]
    fn register(global: &GlobalHandle) -> UniqueLocalHandle {
        let mut local = Box::new(Local {
            next: Cell::new(None),
            prev: Cell::new(None),
            epoch: AtomicU32::new(0),
            _alignment: CacheAligned,
            global: global.clone(),
            handle_count: Cell::new(1),
            guard_count: Cell::new(0),
            pin_count: Cell::new(0),
        });

        let global = global.global();

        global.lock_local_list();

        let head = global.local_list_head.get();
        local.next = Cell::new(head);

        // SAFETY: `Box` is guaranteed to be non-null.
        let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(local)) };

        global.local_list_head.set(Some(ptr));

        if let Some(head) = head {
            // SAFETY: The list of locals always contains valid pointers.
            unsafe { head.as_ref() }.prev.set(Some(ptr));
        }

        // SAFETY: We locked the local list above.
        unsafe { global.unlock_local_list() };

        // SAFETY: We initialized the `handle_count` to 1, such that the handle's drop
        // implementation cannot drop the `Local` while another handle still exists.
        let handle = unsafe { LocalHandle { ptr } };

        // SAFETY: We just allocated this `Local`, and this is the only existing handle to it.
        unsafe { UniqueLocalHandle { inner: handle } }
    }

    #[inline(never)]
    unsafe fn unregister(ptr: NonNull<Self>) {
        // SAFETY: The caller must ensure that `local` is in the list of locals, which means it must
        // be valid.
        let local = unsafe { ptr.as_ref() };
        let global = local.global.global();

        global.lock_local_list();

        if let Some(prev) = local.prev.get() {
            // SAFETY: The list of locals always contains valid pointers.
            unsafe { prev.as_ref() }.next.set(local.next.get());
        } else {
            global.local_list_head.set(local.next.get());
        }

        if let Some(next) = local.next.get() {
            // SAFETY: The list of locals always contains valid pointers.
            unsafe { next.as_ref() }.prev.set(local.prev.get());
        }

        // SAFETY: We locked the local list above.
        unsafe { global.unlock_local_list() };

        // SAFETY: The caller must ensure that `local` can't be accessed after this point.
        let _ = unsafe { Box::from_raw(ptr.as_ptr()) };
    }

    #[inline]
    fn global(&self) -> &Global {
        self.global.global()
    }
}

/// Polyfill for `core::intrinsics::abort`.
#[cold]
fn abort() -> ! {
    struct PanicOnDrop;

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!();
        }
    }

    let _p = PanicOnDrop;
    panic!();
}
