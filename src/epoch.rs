use crate::CacheAligned;
use core::{
    cell::Cell,
    fmt,
    ptr::NonNull,
    sync::atomic::{
        self, AtomicU32,
        Ordering::{Acquire, Relaxed, Release, SeqCst},
    },
};
use std::{
    sync::{Mutex, TryLockError},
    thread,
};

/// The bit of `Local::epoch` which signifies that the participant is pinned.
const PINNED_BIT: u32 = 1 << 0;

/// The number of pinnings between a participant tries to advance the global epoch.
const PINNINGS_BETWEEN_ADVANCE: u32 = 128;

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
        static LOCAL: Cell<Option<NonNull<Local>>> = const { Cell::new(None) };

        static THREAD_GUARD: ThreadGuard = const { ThreadGuard { local: Cell::new(None) } };
    }

    struct ThreadGuard {
        local: Cell<Option<NonNull<Local>>>,
    }

    impl Drop for ThreadGuard {
        fn drop(&mut self) {
            let local = self.local.get().unwrap();
            let guard_count = unsafe { local.as_ref() }.guard_count.get();

            if guard_count == 0 {
                let _ = LOCAL.try_with(|cell| cell.set(None));
                unsafe { Local::unregister(local) };
            } else if !thread::panicking() {
                unreachable!("storing an `epoch::Guard` inside TLS is very naughty");
            }
        }
    }

    #[cold]
    fn local_slow() -> NonNull<Local> {
        let local = Local::register();

        LOCAL.with(|cell| cell.set(Some(local)));
        THREAD_GUARD.with(|guard| guard.local.set(Some(local)));

        local
    }

    LOCAL.with(|cell| {
        if let Some(local) = cell.get() {
            local
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
    epoch: AtomicU32,
}

unsafe impl Send for Global {}
unsafe impl Sync for Global {}

impl Global {
    const fn new() -> Self {
        Global {
            local_list_head: Mutex::new(None),
            _alignment: CacheAligned,
            epoch: AtomicU32::new(0),
        }
    }

    #[cold]
    fn try_advance(&self) {
        let global_epoch = self.epoch.load(Relaxed);

        // Ensure that none of the loads of the local epochs can be ordered before the load of the
        // global epoch.
        atomic::fence(SeqCst);

        let head = match self.local_list_head.try_lock() {
            Ok(guard) => guard,
            // There is no way in which a panic can happen while holding the lock.
            Err(TryLockError::Poisoned(err)) => err.into_inner(),
            Err(TryLockError::WouldBlock) => return,
        };

        let mut head = *head;

        while let Some(local) = head {
            let local = unsafe { local.as_ref() };
            let local_epoch = local.epoch.load(Relaxed);

            if local_epoch & PINNED_BIT != 0 && local_epoch & !PINNED_BIT != global_epoch {
                return;
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
    }
}

#[repr(C)]
struct Local {
    /// The next `Local` in the global list of participants.
    next: Cell<Option<NonNull<Self>>>,

    /// The local epoch counter. When this epoch is pinned, it ensures that the global epoch cannot
    /// be advanced more than one step until it is unpinned.
    epoch: AtomicU32,

    _alignment: CacheAligned,

    /// The number of `Guard`s that exist.
    guard_count: Cell<u32>,

    /// The number of pinnings this participant has gone through in total.
    pin_count: Cell<u32>,
}

impl Local {
    fn register() -> NonNull<Self> {
        let mut local = Box::new(Local {
            next: Cell::new(None),
            epoch: AtomicU32::new(0),
            _alignment: CacheAligned,
            guard_count: Cell::new(0),
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

        local
    }

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

pub struct Guard {
    local: NonNull<Local>,
}

impl Guard {
    #[inline]
    pub fn epoch(&self) -> u32 {
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
        }
    }
}

impl fmt::Debug for Guard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Guard").finish_non_exhaustive()
    }
}
