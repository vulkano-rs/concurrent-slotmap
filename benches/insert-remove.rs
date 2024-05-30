#![feature(test)]

extern crate test;

use concurrent_slotmap::epoch;
use std::{sync::RwLock, thread};
use test::{black_box, Bencher};

const ITERATIONS: u32 = 100_000;
const THREADS: u32 = 10;

#[bench]
fn concurrent_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = concurrent_slotmap::SlotMap::new(ITERATIONS);

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in black_box(0..ITERATIONS / THREADS) {
                        let id = map.insert(black_box([0usize; 2]), &epoch::pin());
                        map.remove(black_box(id), &epoch::pin());
                    }
                });
            }
        });

        map
    });
}

#[bench]
fn rwlock_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = RwLock::new(slotmap::SlotMap::new());

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in black_box(0..ITERATIONS / THREADS) {
                        let id = map.write().unwrap().insert(black_box([0usize; 2]));
                        map.write().unwrap().remove(black_box(id));
                    }
                });
            }
        });

        map
    });
}
