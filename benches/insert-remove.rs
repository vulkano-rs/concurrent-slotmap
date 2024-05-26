#![feature(test)]

extern crate test;

use std::{sync::RwLock, thread};
use test::{black_box, Bencher};

const ITERATIONS: usize = 100_000;
const THREADS: usize = 10;

#[bench]
fn concurrent_slotmap(b: &mut Bencher) {
    b.iter(|| {
        let map = concurrent_slotmap::SlotMap::new((ITERATIONS * 2) as u32);

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in 0..black_box(ITERATIONS / THREADS) {
                        let id = map.insert(black_box([0usize; 2]));
                        map.remove(id);
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
                    for _ in 0..black_box(ITERATIONS / THREADS) {
                        let id = map.write().unwrap().insert(black_box([0usize; 2]));
                        map.write().unwrap().remove(id);
                    }
                });
            }
        });

        map
    });
}
