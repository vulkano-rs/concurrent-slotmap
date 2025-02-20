#![feature(test)]

extern crate test;

use std::{sync::RwLock, thread};
use test::{black_box, Bencher};

const ITERATIONS: u32 = 100_000;
const THREADS: u32 = 10;

#[bench]
fn concurrent_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = concurrent_slotmap::SlotMap::new(ITERATIONS);
        let id = map.insert([0usize; 2], &map.global().register_local().pin());

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    let local = map.global().register_local();

                    for _ in black_box(0..ITERATIONS / THREADS) {
                        black_box(map.get(black_box(id), &local.pin()));
                    }
                });
            }
        });
    });
}

#[bench]
fn rwlock_slotmap(b: &mut Bencher) {
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
