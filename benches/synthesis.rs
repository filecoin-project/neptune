use crate::poseidon::{Arity, PoseidonConstants};
use bellperson::bls::{Bls12, Fr};
use bellperson::gadgets::num::AllocatedNum;
use bellperson::util_cs::bench_cs::BenchCS;
use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fff::{Field, ScalarEngine};
use generic_array::typenum;
use neptune::circuit::poseidon_hash;
use neptune::*;
use rand::thread_rng;
use std::marker::PhantomData;

struct BenchCircuit<A: Arity<Fr>> {
    n: usize,
    _a: PhantomData<A>,
}

impl<A: Arity<Fr>> Circuit<Bls12> for BenchCircuit<A> {
    fn synthesize<CS: ConstraintSystem<Bls12>>(
        self,
        mut cs: &mut CS,
    ) -> Result<(), SynthesisError> {
        let mut rng = thread_rng();
        let arity = A::to_usize();
        let constants = PoseidonConstants::<<Bls12 as ScalarEngine>::Fr, A>::new();

        for _ in 0..self.n {
            let mut i = 0;
            let mut fr_data = vec![Fr::random(&mut rng); arity];
            let data: Vec<AllocatedNum<Bls12>> = (0..arity)
                .enumerate()
                .map(|_| {
                    let fr = Fr::random(&mut rng);
                    fr_data[i] = fr;
                    i += 1;
                    AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(fr)).unwrap()
                })
                .collect::<Vec<_>>();
            let _ = poseidon_hash(&mut cs, data, &constants).expect("poseidon hashing failed");
        }
        Ok(())
    }
}

fn bench_synthesis<A>(c: &mut Criterion)
where
    A: Arity<Fr>,
{
    let mut group = c.benchmark_group(format!("synthesis-{}", A::to_usize()));

    let mut num_hashes = 1;

    for _ in 0..4 {
        group.bench_with_input(
            BenchmarkId::new(
                "Poseidon Synthesis",
                format!("arity: {}, count: {}", A::to_usize(), num_hashes),
            ),
            &num_hashes,
            |b, n| {
                b.iter(|| {
                    let mut cs = BenchCS::<Bls12>::new();
                    let circuit = BenchCircuit::<A> {
                        n: *n,
                        _a: PhantomData::<A>,
                    };
                    circuit.synthesize(&mut cs)
                })
            },
        );
        num_hashes *= 10;
    }
}

criterion_group! {
    name = synthesis;

    config = Criterion::default().sample_size(10);

    targets = bench_synthesis::<typenum::U8>
}
criterion_main!(synthesis);
