#![feature(test)]

extern crate test;

use std::{sync::RwLock, thread};
use test::{black_box, Bencher};

const ITERATIONS: u32 = 100_000;
const THREADS: u32 = 10;

#[bench]
fn insert_contended_concurrent_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = concurrent_slotmap::SlotMap::new(ITERATIONS);

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in black_box(0..ITERATIONS / THREADS) {
                        map.insert(black_box([0usize; 2]), &map.pin());
                    }
                });
            }
        });

        map
    });
}

#[bench]
fn insert_contended_rwlock_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = RwLock::new(slotmap::SlotMap::new());

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in black_box(0..ITERATIONS / THREADS) {
                        map.write().unwrap().insert(black_box([0usize; 2]));
                    }
                });
            }
        });

        map
    });
}

#[bench]
fn insert_uncontended_concurrent_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = concurrent_slotmap::SlotMap::new(ITERATIONS);

        for _ in black_box(0..ITERATIONS / THREADS) {
            map.insert(black_box([0usize; 2]), &map.pin());
        }

        map
    });
}

#[bench]
fn insert_uncontended_rwlock_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = RwLock::new(slotmap::SlotMap::new());

        for _ in black_box(0..ITERATIONS / THREADS) {
            map.write().unwrap().insert(black_box([0usize; 2]));
        }

        map
    });
}
