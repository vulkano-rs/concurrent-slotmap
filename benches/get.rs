#![feature(test)]

extern crate test;

use std::{sync::RwLock, thread};
use test::{Bencher, black_box};

const ITERATIONS: u32 = 100_000;
const THREADS: u32 = 10;

#[bench]
fn get_contended_concurrent_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = concurrent_slotmap::SlotMap::new(ITERATIONS);
        let id = map.insert([0usize; 2], &map.pin());

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in black_box(0..ITERATIONS / THREADS) {
                        black_box(map.get(black_box(id), &map.pin()));
                    }
                });
            }
        });
    });
}

#[bench]
fn get_contended_rwlock_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = RwLock::new(slotmap::SlotMap::new());
        let id = map.write().unwrap().insert([0usize; 2]);

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in black_box(0..ITERATIONS / THREADS) {
                        black_box(map.read().unwrap().get(black_box(id)));
                    }
                });
            }
        });
    });
}

#[bench]
fn get_uncontended_concurrent_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = concurrent_slotmap::SlotMap::new(ITERATIONS);
        let id = map.insert([0usize; 2], &map.pin());

        for _ in black_box(0..ITERATIONS / THREADS) {
            black_box(map.get(black_box(id), &map.pin()));
        }
    });
}

#[bench]
fn get_uncontended_rwlock_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = RwLock::new(slotmap::SlotMap::new());
        let id = map.write().unwrap().insert([0usize; 2]);

        for _ in black_box(0..ITERATIONS / THREADS) {
            black_box(map.read().unwrap().get(black_box(id)));
        }
    });
}
