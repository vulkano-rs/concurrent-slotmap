#![feature(test)]

extern crate test;

use std::{sync::RwLock, thread};
use test::{black_box, Bencher};

const ITERATIONS: u32 = 100_000;
const THREADS: u32 = 10;

#[bench]
fn insert_remove_contended_concurrent_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = concurrent_slotmap::SlotMap::new(ITERATIONS);

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in black_box(0..ITERATIONS / THREADS) {
                        let id = map.insert(black_box([0usize; 2]), &map.pin());
                        map.remove(black_box(id), &map.pin());
                    }
                });
            }
        });

        map
    });
}

#[bench]
fn insert_remove_contended_rwlock_slotmap(b: &mut Bencher) {
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

#[bench]
fn insert_remove_uncontended_concurrent_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = concurrent_slotmap::SlotMap::new(ITERATIONS);

        for _ in black_box(0..ITERATIONS / THREADS) {
            let id = map.insert(black_box([0usize; 2]), &map.pin());
            map.remove(black_box(id), &map.pin());
        }

        map
    });
}

#[bench]
fn insert_remove_uncontended_rwlock_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = RwLock::new(slotmap::SlotMap::new());

        for _ in black_box(0..ITERATIONS / THREADS) {
            let id = map.write().unwrap().insert(black_box([0usize; 2]));
            map.write().unwrap().remove(black_box(id));
        }

        map
    });
}
