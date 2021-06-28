#![allow(dead_code)]
#![allow(unused_imports)]

pub use crate::error::Error;
pub use crate::field::PoseidonField;
pub use crate::poseidon::{Arity, Poseidon};
use crate::round_constants::generate_constants;
use crate::round_numbers::calc_round_numbers;
pub use bellperson::bls::Fr;
use generic_array::GenericArray;

#[cfg(all(feature = "futhark", feature = "opencl"))]
compile_error!("futhark and opencl features are mutually exclusive");

/// Poseidon circuit
pub mod circuit;
pub mod error;
mod matrix;
mod mds;

/// Poseidon hash
mod field;
pub mod poseidon;
mod poseidon_alt;
mod preprocessing;
mod round_constants;
mod round_numbers;

/// Hash types and domain separation tags.
pub mod hash_type;

/// Tree Builder
#[cfg(any(feature = "futhark", feature = "opencl"))]
pub mod tree_builder;

/// Column Tree Builder
#[cfg(any(feature = "futhark", feature = "opencl"))]
pub mod column_tree_builder;

#[cfg(feature = "futhark")]
pub mod triton;

/// Batch Hasher
#[cfg(any(feature = "futhark", feature = "opencl"))]
pub mod batch_hasher;

#[cfg(feature = "opencl")]
pub mod proteus;

pub(crate) const TEST_SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Strength {
    Standard,
    Strengthened,
}

pub(crate) const DEFAULT_STRENGTH: Strength = Strength::Standard;

pub trait BatchHasher<A>
where
    A: Arity<Fr>,
{
    // type State;

    fn hash(&mut self, preimages: &[GenericArray<Fr, A>]) -> Result<Vec<Fr>, Error>;

    fn hash_into_slice(
        &mut self,
        target_slice: &mut [Fr],
        preimages: &[GenericArray<Fr, A>],
    ) -> Result<(), Error> {
        assert_eq!(target_slice.len(), preimages.len());
        // FIXME: Account for max batch size.

        target_slice.copy_from_slice(self.hash(preimages)?.as_slice());
        Ok(())
    }

    /// `max_batch_size` is advisory. Implenters of `BatchHasher` should ensure that up to the returned max hashes can
    /// be safely performed on the target GPU (currently 2080Ti). The max returned should represent a safe batch size
    /// optimized for performance.
    /// `BatchHasher` users are responsible for not attempting to hash batches larger than the advised maximum.
    fn max_batch_size(&self) -> usize {
        700000
    }
}

// Returns the round numbers for a given arity `(R_F, R_P)`.
fn round_numbers_base(arity: usize) -> (usize, usize) {
    let t = arity + 1;
    calc_round_numbers(t, true)
}

// In case of newly-discovered attacks, we may need stronger security.
// This option exists so we can preemptively create circuits in order to switch
// to them quickly if needed.
//
// "A realistic alternative is to increase the number of partial rounds by 25%.
// Then it is unlikely that a new attack breaks through this number,
// but even if this happens then the complexity is almost surely above 2^64, and you will be safe."
// - D Khovratovich
fn round_numbers_strengthened(arity: usize) -> (usize, usize) {
    let (full_round, partial_rounds) = round_numbers_base(arity);

    // Increase by 25%, rounding up.
    let strengthened_partial_rounds = f64::ceil(partial_rounds as f64 * 1.25) as usize;

    (full_round, strengthened_partial_rounds)
}

pub fn round_numbers(arity: usize, strength: &Strength) -> (usize, usize) {
    match strength {
        Strength::Standard => round_numbers_base(arity),
        Strength::Strengthened => round_numbers_strengthened(arity),
    }
}

const SBOX: u8 = 1; // x^5
const FIELD: u8 = 1; // Gf(p)

fn round_constants<F: PoseidonField>(arity: usize, strength: &Strength) -> Vec<F> {
    let t = arity + 1;

    let (full_rounds, partial_rounds) = round_numbers(arity, strength);

    let r_f = full_rounds as u16;
    let r_p = partial_rounds as u16;

    let fr_num_bits = F::NUM_BITS;
    let field_size = {
        assert!(fr_num_bits <= std::u16::MAX as u32);
        // It's safe to convert to u16 for compatibility with other types.
        fr_num_bits as u16
    };

    generate_constants::<F>(FIELD, SBOX, field_size, t as u16, r_f, r_p)
}

/// Apply the quintic S-Box (s^5) to a given item
pub(crate) fn quintic_s_box<F: PoseidonField>(
    l: &mut F,
    pre_add: Option<&F>,
    post_add: Option<&F>,
) {
    if let Some(x) = pre_add {
        l.add_assign(x);
    }
    let mut tmp = *l;
    tmp.square(); // l^2
    tmp.square(); // l^4
    l.mul_assign(&tmp); // l^5
    if let Some(x) = post_add {
        l.add_assign(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strengthened_round_numbers() {
        let cases = [
            (1, 69),
            (2, 69),
            (3, 70),
            (4, 70),
            (5, 70),
            (6, 70),
            (7, 72),
            (8, 72),
            (9, 72),
            (10, 72),
            (11, 72),
            (16, 74),
            (24, 74),
            (36, 75),
            (64, 77),
        ];

        cases.iter().for_each(|(arity, expected_rounds)| {
            let (full_rounds, actual_rounds) = round_numbers_strengthened(*arity);
            assert_eq!(8, full_rounds);
            assert_eq!(
                *expected_rounds, actual_rounds,
                "wrong number of partial rounds for arity {}",
                *arity
            );
        })
    }
}
