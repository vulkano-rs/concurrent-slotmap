use core::{
    cell::{Cell, UnsafeCell},
    fmt,
    ptr::NonNull,
    sync::atomic::{
        self, AtomicUsize,
        Ordering::{Acquire, Relaxed, Release, SeqCst},
    },
};
use std::sync::{Mutex, TryLockError};

/// The bit of `Local::epoch` which signifies that the participant is pinned.
const PINNED_BIT: usize = 1 << 0;

/// The number of pinnings between a participant tries to advance the global epoch.
const PINNINGS_BETWEEN_ADVANCE: usize = 128;

/// Pins the local epoch, such that no accesses done while the returned `Guard` exists can cross
/// into more than one global epoch advance. It is important to pin the local epoch before doing
/// any kind of access, such that no accesses can bleed into the previous epoch.
#[inline]
pub fn pin() -> Guard {
    let guard = unsafe { Guard { local: local() } };
    let local = unsafe { guard.local.as_ref() };
    let guard_count = local.guard_count.get();
    local.guard_count.set(guard_count.checked_add(1).unwrap());

    if guard_count == 0 {
        let global_epoch = global().epoch.load(Relaxed);
        let new_epoch = global_epoch | PINNED_BIT;
        local.epoch.store(new_epoch, Relaxed);

        // This fence acts two-fold:
        // * It synchronizes with the `Release` store of the global epoch in `Global::try_advance`,
        //   which in turn ensures that the accesses all other participants did in a previous epoch
        //   are visible to us going forward when crossing an epoch boundary.
        // * It ensures that that no accesses we do going forward can be ordered before this point,
        //   therefore "bleeding" into the previous epoch, when crossing an epoch boundary.
        atomic::fence(SeqCst);

        local.pin_count.set(local.pin_count.get().wrapping_add(1));

        if local.pin_count.get() % PINNINGS_BETWEEN_ADVANCE == 0 {
            global().try_advance();
        }
    }

    guard
}

#[inline]
fn global() -> &'static Global {
    static GLOBAL: Global = Global::new();

    &GLOBAL
}

#[inline]
fn local() -> NonNull<Local> {
    thread_local! {
        static LOCAL: UnsafeCell<Option<LocalHandle>> = const { UnsafeCell::new(None) };
    }

    #[cold]
    fn local_slow() -> NonNull<Local> {
        let handle = Local::register();
        let local = handle.local;

        LOCAL.with(|cell| unsafe { *cell.get() = Some(handle) });

        local
    }

    LOCAL.with(|cell| {
        if let Some(handle) = unsafe { &*cell.get() }.as_ref() {
            handle.local
        } else {
            local_slow()
        }
    })
}

#[repr(C)]
struct Global {
    /// The global list of `Local`s participating in the memory reclamation.
    local_list_head: Mutex<Option<NonNull<Local>>>,

    _alignment: CacheAligned,

    /// The global epoch counter. This can only ever be one step ahead of any pinned local epoch.
    epoch: AtomicUsize,
}

unsafe impl Send for Global {}
unsafe impl Sync for Global {}

impl Global {
    const fn new() -> Self {
        Global {
            local_list_head: Mutex::new(None),
            _alignment: CacheAligned,
            epoch: AtomicUsize::new(0),
        }
    }

    #[cold]
    fn try_advance(&self) -> usize {
        let global_epoch = self.epoch.load(Relaxed);

        // Ensure that none of the loads of the local epochs can be ordered before the load of the
        // global epoch.
        atomic::fence(SeqCst);

        let head = match self.local_list_head.try_lock() {
            Ok(guard) => guard,
            // There is no way in which a panic can happen while holding the lock.
            Err(TryLockError::Poisoned(err)) => err.into_inner(),
            Err(TryLockError::WouldBlock) => return global_epoch,
        };

        let mut head = *head;

        while let Some(local) = head {
            let local = unsafe { local.as_ref() };
            let local_epoch = local.epoch.load(Relaxed);

            if local_epoch & PINNED_BIT != 0 && local_epoch & !PINNED_BIT != global_epoch {
                return global_epoch;
            }

            head = local.next.get();
        }

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

        new_epoch
    }
}

#[repr(C)]
struct Local {
    /// The next `Local` in the global list of participants.
    next: Cell<Option<NonNull<Self>>>,

    /// The local epoch counter. When this epoch is pinned, it ensures that the global epoch cannot
    /// be advanced more than one step until it is unpinned.
    epoch: AtomicUsize,

    _alignment: CacheAligned,

    /// The number of `Guard`s that exist.
    guard_count: Cell<usize>,

    /// The number of `LocalHandle`s that exist.
    // FIXME: We don't need this.
    handle_count: Cell<usize>,

    /// The number of pinnings this participant has gone through in total.
    pin_count: Cell<usize>,
}

impl Local {
    fn register() -> LocalHandle {
        let mut local = Box::new(Local {
            next: Cell::new(None),
            epoch: AtomicUsize::new(0),
            _alignment: CacheAligned,
            guard_count: Cell::new(0),
            handle_count: Cell::new(1),
            pin_count: Cell::new(0),
        });

        let mut head = match global().local_list_head.lock() {
            Ok(guard) => guard,
            // There is no way in which a panic can happen while holding the lock.
            Err(err) => err.into_inner(),
        };

        local.next = Cell::new(*head);
        let local = unsafe { NonNull::new_unchecked(Box::into_raw(local)) };
        *head = Some(local);

        LocalHandle { local }
    }

    #[cold]
    unsafe fn unregister(local: NonNull<Self>) {
        let mut head = match global().local_list_head.lock() {
            Ok(guard) => guard,
            // There is no way in which a panic can happen while holding the lock.
            Err(err) => err.into_inner(),
        };

        if *head == Some(local) {
            *head = unsafe { local.as_ref() }.next.get();
        } else {
            let mut curr = unsafe { head.unwrap_unchecked() };

            loop {
                let next = unsafe { curr.as_ref() }.next.get();
                let next = unsafe { next.unwrap_unchecked() };

                if next == local {
                    let next = unsafe { next.as_ref() }.next.get();
                    unsafe { curr.as_ref() }.next.set(next);
                    break;
                }

                curr = next;
            }
        }

        let _ = unsafe { Box::from_raw(local.as_ptr()) };
    }
}

struct LocalHandle {
    local: NonNull<Local>,
}

impl Drop for LocalHandle {
    #[inline]
    fn drop(&mut self) {
        let local = unsafe { self.local.as_ref() };
        local.handle_count.set(local.handle_count.get() - 1);

        if local.guard_count.get() == 0 && local.handle_count.get() == 0 {
            unsafe { Local::unregister(self.local) };
        }
    }
}

pub struct Guard {
    local: NonNull<Local>,
}

impl Guard {
    #[inline]
    pub fn epoch(&self) -> usize {
        self.local().epoch.load(Relaxed) & !PINNED_BIT
    }

    #[inline]
    fn local(&self) -> &Local {
        unsafe { self.local.as_ref() }
    }
}

impl Drop for Guard {
    #[inline]
    fn drop(&mut self) {
        let local = unsafe { self.local.as_ref() };
        local.guard_count.set(local.guard_count.get() - 1);

        if local.guard_count.get() == 0 {
            // The `Release` ordering synchronizes with the `Acquire` fence in
            // `Global::try_advance`, which in turn ensures that all accesses done by us until this
            // point are visible to all other participants when crossing an epoch boundary.
            local.epoch.store(0, Release);

            if local.handle_count.get() == 0 {
                unsafe { Local::unregister(self.local) };
            }
        }
    }
}

impl fmt::Debug for Guard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Guard").finish_non_exhaustive()
    }
}

#[repr(align(128))]
struct CacheAligned;
