use bellperson::bls::Fr;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fff::PrimeField;
use generic_array::typenum;
use neptune::poseidon::{HashMode, PoseidonConstants};
use neptune::*;
use rand::rngs::OsRng;
use rand::seq::SliceRandom;
use sha2::{Digest, Sha256, Sha512};

fn bench_hash<A>(c: &mut Criterion)
where
    A: Arity<Fr>,
{
    let scalars: Vec<Fr> = std::iter::repeat(())
        .take(1000)
        .enumerate()
        .map(|(i, _)| Fr::from_u64(i as u64))
        .collect();

    let mut group = c.benchmark_group(format!("hash-{}", A::to_usize() * 32));

    group.bench_with_input(
        BenchmarkId::new("Sha2 256", "Generated scalars"),
        &scalars,
        |b, s| {
            let mut h = Sha256::new();
            b.iter(|| {
                std::iter::repeat(())
                    .take(A::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        for val in scalar.into_repr().as_ref() {
                            h.update(&val.to_le_bytes());
                        }
                    });

                h.finalize_reset()
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Sha2 512", "Generated scalars"),
        &scalars,
        |b, s| {
            let mut h = Sha512::new();

            b.iter(|| {
                std::iter::repeat(())
                    .take(A::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        for val in scalar.into_repr().as_ref() {
                            h.update(&val.to_le_bytes());
                        }
                    });

                h.finalize_reset()
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Poseidon hash", "Generated scalars"),
        &scalars,
        |b, s| {
            let constants = PoseidonConstants::new_with_strength(Strength::Standard);
            let mut h = Poseidon::<Fr, A>::new(&constants);
            b.iter(|| {
                h.reset();
                std::iter::repeat(())
                    .take(A::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        h.input(*scalar).unwrap();
                    });

                h.hash_in_mode(HashMode::Correct);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Poseidon hash optimized", "Generated scalars"),
        &scalars,
        |b, s| {
            let constants = PoseidonConstants::new_with_strength(Strength::Standard);
            let mut h = Poseidon::<Fr, A>::new(&constants);
            b.iter(|| {
                h.reset();
                std::iter::repeat(())
                    .take(A::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        h.input(*scalar).unwrap();
                    });

                h.hash_in_mode(HashMode::OptimizedStatic);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "Poseidon hash optimized (strengthened)",
            "Generated scalars",
        ),
        &scalars,
        |b, s| {
            let constants = PoseidonConstants::new_with_strength(Strength::Strengthened);
            let mut h = Poseidon::<Fr, A>::new(&constants);
            b.iter(|| {
                h.reset();
                std::iter::repeat(())
                    .take(A::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        h.input(*scalar).unwrap();
                    });

                h.hash_in_mode(HashMode::OptimizedStatic);
            })
        },
    );

    group.finish();
}

criterion_group! {
    name = hash;

    config = Criterion::default();

    targets = bench_hash::<typenum::U2>, bench_hash::<typenum::U4>, bench_hash::<typenum::U8>, bench_hash::<typenum::U11>
}
criterion_main!(hash);
